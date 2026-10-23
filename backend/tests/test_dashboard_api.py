"""Unit tests for Dashboard and System Status APIs."""
import pytest
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.database import init_db
from backend.app.utils.sample_data import seed_sample_data_if_empty

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    seed_sample_data_if_empty()

@pytest.fixture
def client():
    return TestClient(app)

def test_dashboard_stats(client):
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "total_active_items" in data
    assert "waste_prevention_rate_pct" in data
    assert "storage_distribution" in data
    assert "freshness_distribution" in data
    assert "recent_scans" in data
    assert data["waste_prevention_rate_pct"] >= 0

def test_system_status(client):
    res = client.get("/api/system-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "ai_mode" in data
    assert data["ai_mode"] in ("model_unavailable", "pytorch_trained", "pretrained_onnx_mobilenet", "produce_onnx_v2")
    assert "model_status" in data
    assert data["model_status"] in ("loaded", "unavailable")
    assert "produce_model" in data
    assert "freshness_assessment_method" in data
    assert "model_notice" in data
    # Ensure no absolute file system path is leaked
    assert "c:\\" not in str(data).lower()
