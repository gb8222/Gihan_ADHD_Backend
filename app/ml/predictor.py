from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.ml.feature_extractor import extract_speech_features


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "adhd_rf_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "adhd_rf_scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "adhd_feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)


def predict_audio_bytes(audio_bytes: bytes) -> dict[str, Any]:
    """
    Full inference pipeline:
    audio bytes -> feature extraction -> column alignment -> scaling -> prediction
    """
    extraction_result = extract_speech_features(audio_bytes)
    model_features = extraction_result["model_features"]
    display_features = extraction_result["display_features"]

    x = _align_features(model_features)
    x_scaled = scaler.transform(x)

    prediction_class = int(model.predict(x_scaled)[0])
    adhd_probability = float(model.predict_proba(x_scaled)[0][1])

    prediction_label = "ADHD" if prediction_class == 1 else "Normal"
    classification = (
        "ADHD indicators detected"
        if prediction_class == 1
        else "No ADHD indicators detected"
    )

    return {
        "prediction_class": prediction_class,
        "prediction_label": prediction_label,
        "classification": classification,
        "adhd_probability": round(adhd_probability, 4),
        "confidence": _confidence_from_probability(adhd_probability),
        "features": display_features,
    }


def _align_features(model_features: dict[str, float]) -> pd.DataFrame:
    """
    Creates a one-row dataframe, fills missing features with 0.0,
    and reorders columns to match training exactly.
    """
    x = pd.DataFrame([model_features])

    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0.0

    x = x[feature_columns]
    return x


def _confidence_from_probability(probability: float) -> str:
    """
    Confidence based on distance from the uncertain center (0.5).
    """
    distance = abs(probability - 0.5)

    if distance >= 0.3:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"