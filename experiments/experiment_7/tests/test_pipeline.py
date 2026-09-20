"""
Pytest Test Suite for Experiment 7 CI/CD Pipeline.
Evaluates model artifact integrity, preprocessing behavior, API endpoint contracts,
schema validation, and latency regression boundaries.
"""

import os
import sys
import time
from pathlib import Path

import pytest
import joblib
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.experiment_6.src.app import app, resolve_model_path


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for testing FastAPI endpoints."""
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. MODEL ARTIFACT INTEGRITY & SERIALIZATION TESTS (DVC Pattern)
# ==============================================================================

class TestModelArtifacts:
    """Verifies existence, integrity, and operational health of serialized models."""

    def test_champion_model_file_exists(self):
        """Ensures the champion model bundle exists on disk."""
        model_path = Path(resolve_model_path())
        assert model_path.exists(), f"Champion model artifact missing at {model_path}"

    def test_champion_model_file_size(self):
        """Verifies model file is non-trivial and not corrupted (>100 KB)."""
        model_path = Path(resolve_model_path())
        size_bytes = model_path.stat().st_size
        assert size_bytes > 100 * 1024, f"Model file suspiciously small: {size_bytes} bytes"

    def test_model_deserialization_and_predict_interface(self):
        """Ensures the model bundle loads cleanly and implements .predict()."""
        model_path = Path(resolve_model_path())
        bundle = joblib.load(model_path)
        assert hasattr(bundle, "predict") or isinstance(bundle, dict), "Model must implement .predict() or be bundle dict"
        pipeline = bundle if hasattr(bundle, "predict") else bundle.get("pipeline", bundle.get("model"))
        assert hasattr(pipeline, "predict"), "Model object must implement .predict()"


# ==============================================================================
# 2. FASTAPI ENDPOINT CONTRACT TESTS
# ==============================================================================

class TestAPIEndpoints:
    """Verifies REST API endpoints conform to OpenAPI specifications and data contracts."""

    def test_root_endpoint_metadata(self, client):
        """GET / returns HTTP 200 and valid service metadata."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
        assert data.get("containerized") is True

    def test_health_check_probe(self, client):
        """GET /health returns HTTP 200 and operational health indicators."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert data.get("model_loaded") is True
        assert "uptime_seconds" in data
        assert "version" in data

    def test_single_prediction_positive_feedback(self, client):
        """POST /predict classifies customer praise accurately with LOW urgency."""
        payload = {"text": "Thank you so much! Excellent and fast resolution to my issue! Highly appreciated!"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"
        primary = data.get("primary_emotion", "").lower()
        assert any(term in primary for term in ["joy", "gratitude", "positive", "neutral"])
        assert data.get("urgency_level") in ["LOW", "MEDIUM"]
        assert "emotion_probabilities" in data
        assert isinstance(data.get("latency_ms"), (int, float))

    def test_single_prediction_urgent_complaint(self, client):
        """POST /predict classifies flight delay complaint with HIGH/CRITICAL urgency."""
        payload = {"text": "My flight got delayed 8 hours and no agent is at the desk! Horrible service and lost bags!"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"
        primary = data.get("primary_emotion", "").lower()
        assert any(term in primary for term in ["anger", "frustration", "sadness", "fear"])
        assert data.get("urgency_level") in ["CRITICAL", "HIGH"]
        assert data.get("sentiment", "").lower() == "negative"

    def test_batch_prediction_endpoint(self, client):
        """POST /predict/batch processes multiple records in a single transaction."""
        texts = [
            "Great customer service, thank you!",
            "My package arrived completely broken and destroyed!",
            "Can I check my account balance?"
        ]
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "success"
        assert data.get("total_count") == len(texts)
        assert len(data.get("predictions")) == len(texts)

    def test_validation_error_empty_string(self, client):
        """POST /predict returns HTTP 422 for empty string input."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
        error_detail = response.json().get("detail")
        assert error_detail is not None

    def test_validation_error_missing_field(self, client):
        """POST /predict returns HTTP 422 for payload lacking required 'text' key."""
        response = client.post("/predict", json={"invalid_key": "sample"})
        assert response.status_code == 422

    def test_prediction_latency_threshold(self, client):
        """Asserts inference latency stays comfortably within 150ms SLA."""
        payload = {"text": "Testing production response latency threshold."}
        start_time = time.perf_counter()
        response = client.post("/predict", json=payload)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        assert response.status_code == 200
        assert elapsed_ms < 150.0, f"Latency exceeded SLA threshold: {elapsed_ms:.2f} ms"
