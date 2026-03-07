import numpy as np
import librosa
import io


def extract_speech_features(audio_bytes: bytes) -> dict:
    """
    Accepts raw audio file bytes, converts to 16kHz mono,
    extracts acoustic features for the ML model AND display features
    for the frontend AnalysisResults component.

    Returns:
        dict with keys:
            - "model_vector": np.ndarray shape (1, 32) for model prediction
            - "display_features": dict with pitch_mean, jitter, shimmer,
              lexical_diversity, coherence, fillers for frontend display
    """
    # --- 1. Load audio & convert to 16kHz mono ---
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    # --- 2. Extract MFCC features (13 coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_features = []
    for i in range(13):
        mfcc_features.append(float(np.mean(mfccs[i])))   # mfcc_{i}_mean

    # --- 3. Extract spectral features ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    centroid_mean = float(np.mean(centroid))
    bandwidth_mean = float(np.mean(bandwidth))
    rolloff_mean = float(np.mean(rolloff))
    zcr_mean = float(np.mean(zcr))
    rms_mean = float(np.mean(rms))

    spectral_features = [centroid_mean, bandwidth_mean, rolloff_mean, zcr_mean, rms_mean]

    # --- 4. Extract pitch feature ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

    # --- 5. Combine into model feature vector (19 features) ---
    feature_list = mfcc_features + spectral_features + [pitch_mean]
    model_vector = np.array(feature_list).reshape(1, -1)

    # --- 6. Compute display-only features for AnalysisResults.jsx ---
    # Jitter: measure pitch period variability (frequency perturbation)
    jitter = _compute_jitter(y, sr)

    # Shimmer: measure amplitude perturbation
    shimmer = _compute_shimmer(y, sr)

    # Linguistic proxy features (estimated from acoustic properties)
    # Since we don't have transcription, we estimate from audio characteristics
    lexical_diversity = _estimate_lexical_diversity(y, sr)
    coherence = _estimate_coherence(y, sr)
    fillers = _estimate_fillers(y, sr)

    display_features = {
        "pitch_mean": pitch_mean,
        "jitter": jitter,
        "shimmer": shimmer,
        "lexical_diversity": lexical_diversity,
        "coherence": coherence,
        "fillers": fillers,
    }

    return {
        "model_vector": model_vector,
        "display_features": display_features,
    }


def _compute_jitter(y: np.ndarray, sr: int) -> float:
    """Compute jitter (pitch period perturbation) from audio signal."""
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) < 2:
            return 0.0

        periods = [1.0 / p for p in pitch_values if p > 0]
        if len(periods) < 2:
            return 0.0

        diffs = [abs(periods[i + 1] - periods[i]) for i in range(len(periods) - 1)]
        jitter = np.mean(diffs) / np.mean(periods) if np.mean(periods) > 0 else 0.0
        return float(min(jitter, 1.0))
    except Exception:
        return 0.0


def _compute_shimmer(y: np.ndarray, sr: int) -> float:
    """Compute shimmer (amplitude perturbation) from audio signal."""
    try:
        # Use RMS energy in short frames
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_frames = rms_frames[rms_frames > 0]

        if len(rms_frames) < 2:
            return 0.0

        # Convert to dB
        rms_db = 20 * np.log10(rms_frames + 1e-10)
        diffs = [abs(rms_db[i + 1] - rms_db[i]) for i in range(len(rms_db) - 1)]
        shimmer = np.mean(diffs) / (np.mean(np.abs(rms_db)) + 1e-10)
        return float(min(shimmer, 1.0))
    except Exception:
        return 0.0


def _estimate_lexical_diversity(y: np.ndarray, sr: int) -> float:
    """
    Estimate lexical diversity proxy from spectral variation.
    Higher spectral variety ≈ richer vocabulary usage.
    """
    try:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        variation = np.std(spectral_contrast) / (np.mean(np.abs(spectral_contrast)) + 1e-10)
        # Normalize to 0-1 range
        return float(np.clip(variation * 0.3, 0.1, 0.95))
    except Exception:
        return 0.5


def _estimate_coherence(y: np.ndarray, sr: int) -> float:
    """
    Estimate speech coherence proxy from temporal regularity.
    More structured speech ≈ higher coherence.
    """
    try:
        # Use onset strength regularity as coherence proxy
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset_env) < 2:
            return 0.5

        # Autocorrelation of onset envelope — high = rhythmic/coherent
        autocorr = np.correlate(onset_env, onset_env, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if len(autocorr) > 1 and autocorr[0] > 0:
            normalized = autocorr[1] / autocorr[0]
            return float(np.clip(normalized, 0.1, 0.95))
        return 0.5
    except Exception:
        return 0.5


def _estimate_fillers(y: np.ndarray, sr: int) -> int:
    """
    Estimate filler word count from short low-energy voiced segments.
    """
    try:
        # Detect short pauses / low-energy voiced segments as filler proxies
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold = np.mean(rms) * 0.4

        # Count transitions from below to above threshold (hesitation markers)
        below = rms < threshold
        transitions = np.sum(np.diff(below.astype(int)) == -1)  # silence → speech

        # Estimate fillers as a fraction of transitions
        filler_estimate = max(0, int(transitions * 0.3))
        return min(filler_estimate, 30)
    except Exception:
        return 0
