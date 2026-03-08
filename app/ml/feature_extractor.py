import io
import os
import tempfile
import subprocess
from typing import Any

import librosa
import numpy as np
import parselmouth
import soundfile as sf


TARGET_SR = 16000


def extract_speech_features(audio_bytes: bytes) -> dict[str, Any]:
    """
    Extracts the exact feature family used by the Colab-trained model
    and also returns display-friendly frontend features.
    """
    y, sr = _load_audio(audio_bytes, target_sr=TARGET_SR)

    if y.size == 0:
        raise ValueError("Audio could not be decoded or is empty after preprocessing.")

    model_features: dict[str, float] = {}
    model_features.update(_extract_mfcc_features(y, sr, n_mfcc=13))
    model_features.update(_extract_pitch_features(y, sr))
    model_features.update(_extract_energy_features(y))
    model_features.update(_extract_spectral_features(y, sr))
    model_features.update(_extract_parselmouth_features(y, sr))
    model_features.update(_extract_duration_features(y, sr))

    display_features = {
        "pitch_mean": float(
            model_features.get("praat_pitch_mean", 0.0)
            or model_features.get("pitch_mean", 0.0)
        ),
        "jitter": float(model_features.get("jitter_local", 0.0)),
        "shimmer": float(model_features.get("shimmer_local", 0.0)),
        "lexical_diversity": _estimate_lexical_diversity(y, sr),
        "coherence": _estimate_coherence(y, sr),
        "fillers": _estimate_fillers(y, sr),
    }

    return {
        "model_features": model_features,
        "display_features": display_features,
    }


def _load_audio(audio_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load uploaded audio bytes, resample to target_sr, convert to mono, trim silence.
    Uses librosa.load natively, with a fallback to pydub if unsupported.
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    except Exception as e_librosa:
        # Fallback to ffmpeg directly for webm/mp4 and other formats
        temp_in_path = None
        temp_out_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
                tmp_in.write(audio_bytes)
                temp_in_path = tmp_in.name
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                temp_out_path = tmp_out.name

            subprocess.run([
                "ffmpeg", "-y", "-i", temp_in_path, 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", str(target_sr), "-ac", "1", 
                temp_out_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            y, sr = librosa.load(temp_out_path, sr=target_sr, mono=True)
            
        except Exception as e_ffmpeg:
            raise ValueError(f"Unsupported or corrupted audio file. Librosa error: {e_librosa}. FFmpeg fallback error: {e_ffmpeg}") from e_ffmpeg
        finally:
            if temp_in_path and os.path.exists(temp_in_path):
                try: os.remove(temp_in_path)
                except OSError: pass
            if temp_out_path and os.path.exists(temp_out_path):
                try: os.remove(temp_out_path)
                except OSError: pass

    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        return y, target_sr

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Safety normalization
    max_abs = np.max(np.abs(y)) if y.size > 0 else 0.0
    if max_abs > 0:
        y = y / max_abs

    return y, sr


def _extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict[str, float]:
    features: dict[str, float] = {}
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))
    except Exception:
        for i in range(n_mfcc):
            features[f"mfcc_{i + 1}_mean"] = 0.0
            features[f"mfcc_{i + 1}_std"] = 0.0
    return features


def _extract_pitch_features(y: np.ndarray, sr: int) -> dict[str, float]:
    try:
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]

        if len(pitch_values) == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_min": 0.0,
                "pitch_max": 0.0,
            }

        return {
            "pitch_mean": float(np.mean(pitch_values)),
            "pitch_std": float(np.std(pitch_values)),
            "pitch_min": float(np.min(pitch_values)),
            "pitch_max": float(np.max(pitch_values)),
        }
    except Exception:
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
        }


def _extract_energy_features(y: np.ndarray) -> dict[str, float]:
    try:
        rms = librosa.feature.rms(y=y)[0]
        return {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_min": float(np.min(rms)),
            "rms_max": float(np.max(rms)),
        }
    except Exception:
        return {
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "rms_min": 0.0,
            "rms_max": 0.0,
        }


def _extract_spectral_features(y: np.ndarray, sr: int) -> dict[str, float]:
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "rolloff_mean": float(np.mean(rolloff)),
            "rolloff_std": float(np.std(rolloff)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
        }
    except Exception:
        return {
            "spectral_centroid_mean": 0.0,
            "spectral_centroid_std": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "spectral_bandwidth_std": 0.0,
            "rolloff_mean": 0.0,
            "rolloff_std": 0.0,
            "zcr_mean": 0.0,
            "zcr_std": 0.0,
        }


def _extract_parselmouth_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """
    Parselmouth works best from a temp WAV generated from the decoded waveform.
    This keeps behavior consistent across uploaded formats.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        sf.write(temp_path, y, sr)
        snd = parselmouth.Sound(temp_path)

        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]

        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 500
        )

        jitter_local = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = parselmouth.praat.call(
            [snd, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )

        return {
            "praat_pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0,
            "praat_pitch_std": float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0,
            "jitter_local": _safe_float(jitter_local),
            "shimmer_local": _safe_float(shimmer_local),
        }
    except Exception:
        return {
            "praat_pitch_mean": 0.0,
            "praat_pitch_std": 0.0,
            "jitter_local": 0.0,
            "shimmer_local": 0.0,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _extract_duration_features(y: np.ndarray, sr: int) -> dict[str, float]:
    try:
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # librosa can sometimes return np.ndarray-like values
        tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) > 0 else 0.0

        return {
            "duration_sec": float(duration),
            "tempo": tempo_value,
        }
    except Exception:
        return {
            "duration_sec": 0.0,
            "tempo": 0.0,
        }


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value
    except Exception:
        return 0.0


def _estimate_lexical_diversity(y: np.ndarray, sr: int) -> float:
    """
    Frontend-only proxy metric. Not used by the model.
    """
    try:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        variation = np.std(spectral_contrast) / (np.mean(np.abs(spectral_contrast)) + 1e-10)
        return float(np.clip(variation * 0.3, 0.1, 0.95))
    except Exception:
        return 0.5


def _estimate_coherence(y: np.ndarray, sr: int) -> float:
    """
    Frontend-only proxy metric. Not used by the model.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset_env) < 2:
            return 0.5

        autocorr = np.correlate(onset_env, onset_env, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if len(autocorr) > 1 and autocorr[0] > 0:
            normalized = autocorr[1] / autocorr[0]
            return float(np.clip(normalized, 0.1, 0.95))
        return 0.5
    except Exception:
        return 0.5


def _estimate_fillers(y: np.ndarray, sr: int) -> int:
    """
    Frontend-only proxy metric. Not used by the model.
    """
    try:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold = np.mean(rms) * 0.4
        below = rms < threshold
        transitions = int(np.sum(np.diff(below.astype(int)) == -1))
        filler_estimate = max(0, int(transitions * 0.3))
        return min(filler_estimate, 30)
    except Exception:
        return 0