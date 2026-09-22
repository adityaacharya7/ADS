"""
Experiment 6: Production Machine Learning Microservice API (api.py)
Framework: FastAPI + Uvicorn + Pydantic v2
Course: Applied Data Science (ADS)

Exposes high-throughput, low-latency endpoints for:
1. Real-time Customer Support Emotion & Sentiment Inference (POST /predict)
2. Vectorized Batch Inference (POST /predict/batch)
3. Container Health & Liveness Probing (GET /health)
4. Interactive OpenAPI / Swagger Documentation (GET /docs, GET /redoc)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_2.src.predict import EmotionPredictor

try:
    from experiments.experiment_4.src.experiment_4_modeling import EndToEndEmotionPipeline
    import __main__
    setattr(__main__, 'EndToEndEmotionPipeline', EndToEndEmotionPipeline)
except Exception:
    pass

# Global state
START_TIME = time.time()
_PREDICTOR: Optional[EmotionPredictor] = None
_MODEL_PATH: Optional[str] = None


def resolve_model_path() -> str:
    """Locates the champion model artifact across candidate locations."""
    candidates = [
        PROJECT_ROOT / "experiments" / "experiment_4" / "models" / "best_emotion_model_exp4.joblib",
        PROJECT_ROOT / "experiments" / "experiment_2" / "models" / "emotion_pipeline.joblib",
        PROJECT_ROOT / "models" / "emotion_pipeline.joblib"
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[1])


def get_predictor() -> EmotionPredictor:
    """Singleton getter for the inference engine."""
    global _PREDICTOR, _MODEL_PATH
    if _PREDICTOR is None:
        _MODEL_PATH = resolve_model_path()
        _PREDICTOR = EmotionPredictor(model_path=_MODEL_PATH)
    return _PREDICTOR


def compute_urgency_level(primary_emotion: str, confidence: float, compound_sentiment: float) -> str:
    """Calculates operational customer support ticket triage urgency."""
    if primary_emotion == "Anger / Frustration" or (compound_sentiment <= -0.60 and confidence >= 0.50):
        return "CRITICAL"
    elif primary_emotion in ("Anger / Frustration", "Fear / Anxiety") or compound_sentiment <= -0.30:
        return "HIGH"
    elif primary_emotion == "Disappointment / Sadness":
        return "MEDIUM"
    elif primary_emotion == "Joy / Gratitude":
        return "LOW"
    return "MEDIUM"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes ML model artifacts on startup and handles graceful shutdown."""
    global _PREDICTOR, _MODEL_PATH
    print("[*] Starting ADS Emotion API service...")
    _MODEL_PATH = resolve_model_path()
    _PREDICTOR = EmotionPredictor(model_path=_MODEL_PATH)
    print(f"[+] Champion model initialized from: {_MODEL_PATH}")
    yield
    print("[*] Shutting down ADS Emotion API service...")


# ------------------------------------------------------------------------------
# FASTAPI APPLICATION DEFINITION
# ------------------------------------------------------------------------------

app = FastAPI(
    title="ADS Customer Support Emotion & Sentiment API",
    description="Production-ready REST API for real-time customer emotion detection, sentiment scoring, and ticket urgency triage.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures endpoint execution latency and appends custom header."""
    start = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# ------------------------------------------------------------------------------
# PYDANTIC REQUEST / RESPONSE SCHEMAS
# ------------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "My package was delivered damaged and 3 weeks late! Unacceptable service!"
            }
        }
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Raw customer support tweet or message text."
    )


class PredictionResponse(BaseModel):
    status: str = Field("success", description="Response execution status")
    raw_text: str = Field(..., description="Original input text")
    primary_emotion: str = Field(..., description="Dominant predicted emotion category")
    confidence: float = Field(..., description="Confidence score for the primary emotion")
    secondary_emotion: str = Field(..., description="Secondary emotion if mixed, else None")
    is_mixed_emotion: bool = Field(..., description="True if a secondary emotion is detected")
    emotion_probabilities: Dict[str, float] = Field(..., description="Calibrated probabilities across all 5 emotion classes")
    sentiment: str = Field(..., description="Overall sentiment polarity (POSITIVE, NEGATIVE, NEUTRAL)")
    vader_compound: float = Field(..., description="Continuous sentiment compound score (-1.0 to +1.0)")
    urgency_level: str = Field(..., description="Automated support routing urgency (CRITICAL, HIGH, MEDIUM, LOW)")
    latency_ms: float = Field(..., description="End-to-end model inference latency in milliseconds")


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Thank you so much! Excellent and fast resolution to my issue! 😊",
                    "My flight got delayed 8 hours and no agent is at the desk!",
                    "Can you please check the tracking status for order #12345?"
                ]
            }
        }
    )
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Array of customer messages for vectorized batch processing."
    )


class BatchPredictionResponse(BaseModel):
    status: str = Field("success", description="Batch execution status")
    total_count: int = Field(..., description="Total records processed in the batch")
    predictions: List[PredictionResponse] = Field(..., description="List of individual inference results")
    batch_latency_ms: float = Field(..., description="Total batch processing latency in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="System operational health status")
    version: str = Field("1.0.0", description="API semantic version")
    model_loaded: bool = Field(..., description="Indicates if ML model weights are loaded in memory")
    model_path: str = Field(..., description="Filesystem location of the loaded model")
    uptime_seconds: float = Field(..., description="Service uptime duration in seconds")
    timestamp: str = Field(..., description="Current ISO-8601 UTC server timestamp")


# ------------------------------------------------------------------------------
# API ENDPOINT ROUTERS
# ------------------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint providing service overview and documentation links."""
    return {
        "service": "ADS Customer Support Emotion & Sentiment Analysis API",
        "version": "1.0.0",
        "framework": "FastAPI (ASGI)",
        "containerized": True,
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "health_check": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Production health, readiness, and liveness probe."""
    predictor = get_predictor()
    model_ok = predictor is not None and (predictor.pipeline is not None or predictor.labeler is not None)
    return HealthResponse(
        status="healthy" if model_ok else "degraded",
        version="1.0.0",
        model_loaded=model_ok,
        model_path=str(_MODEL_PATH or "In-memory rule engine"),
        uptime_seconds=round(time.time() - START_TIME, 2),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(payload: PredictionRequest):
    """
    Performs real-time emotion classification and sentiment analysis for a single customer message.
    """
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text cannot be empty or whitespace only."
        )

    t0 = time.perf_counter()
    predictor = get_predictor()
    result = predictor.predict_one(payload.text)
    latency = (time.perf_counter() - t0) * 1000.0

    primary = result["primary_emotion"]
    conf = float(result["primary_confidence"])
    compound = float(result["vader_compound"])
    urgency = compute_urgency_level(primary, conf, compound)

    return PredictionResponse(
        status="success",
        raw_text=payload.text,
        primary_emotion=primary,
        confidence=round(conf, 4),
        secondary_emotion=str(result.get("secondary_emotion", "None")),
        is_mixed_emotion=bool(result.get("is_mixed", False)),
        emotion_probabilities={k: round(v, 4) for k, v in result.get("probabilities", {}).items()},
        sentiment=str(result.get("sentiment", "NEUTRAL")),
        vader_compound=round(compound, 4),
        urgency_level=urgency,
        latency_ms=round(latency, 2)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(payload: BatchPredictionRequest):
    """
    Vectorized batch inference endpoint for high-throughput operational systems.
    """
    if not payload.texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch list must contain at least one text entry."
        )

    t0 = time.perf_counter()
    predictor = get_predictor()
    results = []

    for text in payload.texts:
        if not text.strip():
            continue
        t_sub0 = time.perf_counter()
        res = predictor.predict_one(text)
        sub_lat = (time.perf_counter() - t_sub0) * 1000.0
        primary = res["primary_emotion"]
        conf = float(res["primary_confidence"])
        compound = float(res["vader_compound"])
        urgency = compute_urgency_level(primary, conf, compound)

        results.append(PredictionResponse(
            status="success",
            raw_text=text,
            primary_emotion=primary,
            confidence=round(conf, 4),
            secondary_emotion=str(res.get("secondary_emotion", "None")),
            is_mixed_emotion=bool(res.get("is_mixed", False)),
            emotion_probabilities={k: round(v, 4) for k, v in res.get("probabilities", {}).items()},
            sentiment=str(res.get("sentiment", "NEUTRAL")),
            vader_compound=round(compound, 4),
            urgency_level=urgency,
            latency_ms=round(sub_lat, 2)
        ))

    total_batch_latency = (time.perf_counter() - t0) * 1000.0

    return BatchPredictionResponse(
        status="success",
        total_count=len(results),
        predictions=results,
        batch_latency_ms=round(total_batch_latency, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("experiments.experiment_6.src.api:app", host="127.0.0.1", port=8000, reload=True)
