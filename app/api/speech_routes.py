from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.speech_feature_extractor import extract_speech_features
from app.services.speech_predictor import predict_adhd

speech_router = APIRouter(tags=["Speech Analysis"])


@speech_router.post("/analyze")
async def analyze_speech(
    file: UploadFile = File(...),
    child_age: int = Form(default=8),
):
    """
    Upload a speech audio file and get ADHD prediction.
    Matches frontend FileUpload.jsx: POST /analyze with FormData(file, child_age).
    No authentication required — frontend does not send auth headers for this endpoint.
    """
    # --- Validate file type ---
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    filename = file.filename or ""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    # --- Read audio bytes ---
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        # --- Extract features ---
        extraction_result = extract_speech_features(audio_bytes)
        model_vector = extraction_result["model_vector"]
        display_features = extraction_result["display_features"]

        # --- Run prediction ---
        prediction = predict_adhd(model_vector)

        # --- Build response matching AnalysisResults.jsx expected shape ---
        probability = prediction["probability"]

        response = {
            "analysis": {
                "classification": prediction["classification"],
                "probability": probability,
                "confidence": prediction["confidence"],
                "transcription_available": False,
            },
            "features": {
                "pitch_mean": round(display_features["pitch_mean"], 2),
                "jitter": round(display_features["jitter"], 4),
                "shimmer": round(display_features["shimmer"], 4),
                "lexical_diversity": round(display_features["lexical_diversity"], 4),
                "coherence": round(display_features["coherence"], 4),
                "fillers": display_features["fillers"],
            },
            "transcription": None,
            "recommendations": _build_recommendations(probability, child_age),
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Speech analysis failed: {str(e)}",
        )


def _build_recommendations(probability: float, child_age: int) -> dict:
    """Build recommendation object based on ADHD probability and child age."""
    if probability >= 0.7:
        return {
            "warning": "Significant ADHD speech patterns detected",
            "note": (
                f"The analysis of this {child_age}-year-old's speech shows notable patterns "
                "commonly associated with ADHD. These include variations in pitch stability, "
                "speech rhythm, and vocal characteristics that differ from typical speech patterns."
            ),
            "advice": (
                "We strongly recommend scheduling a comprehensive evaluation with a pediatric "
                "neurologist or developmental pediatrician for a thorough clinical assessment."
            ),
        }
    elif probability >= 0.4:
        return {
            "warning": "Some ADHD-related speech patterns observed",
            "note": (
                f"The analysis of this {child_age}-year-old's speech shows some patterns "
                "that may be associated with ADHD, though the indicators are moderate. "
                "These findings warrant further monitoring."
            ),
            "advice": (
                "Consider discussing these results with your child's pediatrician during "
                "the next visit. Continued monitoring of speech and behavioral patterns is recommended."
            ),
        }
    else:
        return {
            "message": "Speech patterns appear within normal range",
            "note": (
                f"The analysis of this {child_age}-year-old's speech patterns shows characteristics "
                "that fall within the typical range for their age group. No significant ADHD-related "
                "speech markers were identified."
            ),
            "advice": (
                "Continue supporting your child's development with regular check-ups. "
                "If you have concerns about attention or behavior, consult your pediatrician."
            ),
        }
