import joblib
import numpy as np
from pathlib import Path

# --- Load model once at module import time ---
MODEL_PATH = Path(__file__).resolve().parent.parent / "speech_ml_model" / "adhd_model (3).pkl"
model = joblib.load(MODEL_PATH)


def predict_adhd(feature_vector: np.ndarray) -> dict:
    """
    Takes a feature vector (1, 32) and returns prediction result
    with label, probability, classification text, and confidence level.
    """
    # Get predicted class (0 or 1)
    prediction = int(model.predict(feature_vector)[0])

    # Get probability scores [prob_class_0, prob_class_1]
    probabilities = model.predict_proba(feature_vector)[0]
    adhd_probability = float(probabilities[1])

    # Determine classification text
    if prediction == 1:
        classification = "ADHD indicators detected"
    else:
        classification = "No ADHD indicators detected"

    # Determine confidence level
    confidence = _get_confidence(adhd_probability)

    return {
        "label": prediction,
        "probability": round(adhd_probability, 4),
        "classification": classification,
        "confidence": confidence,
    }


def _get_confidence(probability: float) -> str:
    """Derive confidence level from how far the probability is from 0.5."""
    distance = abs(probability - 0.5)
    if distance >= 0.3:
        return "High"
    elif distance >= 0.15:
        return "Medium"
    else:
        return "Low"
