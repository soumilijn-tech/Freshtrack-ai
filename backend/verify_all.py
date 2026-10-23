"""Comprehensive End-to-End Verification Script for FreshTrack AI.
Tests genuine MobileNetV2 ONNX produce classification, surface discoloration
heuristics, low-confidence uncertainty handling, and all backend API endpoints.
"""
import sys
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

# Ensure backend can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.database import init_db
from backend.app.utils.sample_data import seed_sample_data_if_empty
from backend.app.services.vision_service import vision_service, VisionModelService

client = TestClient(app)
init_db()
seed_sample_data_if_empty()

results = []

def run_check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        results.append((name, "PASS", None))
    except Exception as e:
        print(f"[FAIL] {name} -> {e}")
        results.append((name, "FAIL", str(e)))

# 1. Root Webpage
def check_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "<title>FreshTrack AI" in res.text
    assert "Produce Freshness Scanner" in res.text
run_check("Frontend Webpage (GET /)", check_root)

# 2. Interactive Swagger UI
def check_docs():
    res = client.get("/docs")
    assert res.status_code == 200
    assert "swagger-ui" in res.text.lower()
run_check("Interactive Swagger UI (GET /docs)", check_docs)

# 3. OpenAPI Schema
def check_openapi():
    res = client.get("/openapi.json")
    assert res.status_code == 200
    data = res.json()
    assert data["info"]["title"] == "FreshTrack AI"
    assert "/api/scan" in data["paths"]
run_check("OpenAPI Schema (GET /openapi.json)", check_openapi)

# 4. System Status API
def check_status():
    res = client.get("/api/system-status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["model_status"] == "loaded"
    assert "MobileNetV2" in data["produce_model"]
    assert "Surface Visual Discoloration" in data["freshness_assessment_method"]
    assert "c:\\" not in str(data).lower()
run_check("System Status API (GET /api/system-status)", check_status)

# 5. Model Inference: Green Lemon Photo
def check_green_lemon():
    img_path = PROJECT_ROOT / "frontend" / "assets" / "samples" / "green_lemon.jpg"
    assert img_path.exists()
    with open(img_path, "rb") as f:
        res = client.post("/api/scan", files={"file": ("test_lemon.jpg", f, "image/jpeg")})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["produce"] == "Lemon", f"Expected Lemon, got {data['produce']}"
    assert data["confidence_pct"] >= 60.0
    assert data["freshness"] == "Fresh"
    assert data["is_uncertain"] is False
    assert data["freshness_assessment_method"] == "Surface Visual Discoloration Index (Heuristic)"
run_check("Model Inference: Green Lemon (POST /api/scan)", check_green_lemon)

# 6. Model Inference: Apple Photo
def check_apple():
    img_path = PROJECT_ROOT / "frontend" / "assets" / "samples" / "fresh_apple.jpg"
    assert img_path.exists()
    with open(img_path, "rb") as f:
        res = client.post("/api/scan", files={"file": ("test_apple.jpg", f, "image/jpeg")})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["produce"] == "Apple", f"Expected Apple, got {data['produce']}"
    assert data["confidence_pct"] >= 70.0
    assert data["freshness"] == "Fresh"
run_check("Model Inference: Fresh Apple (POST /api/scan)", check_apple)

# 7. Model Inference: Spotted Banana Photo
def check_spotted_banana():
    img_path = PROJECT_ROOT / "frontend" / "assets" / "samples" / "spotted_banana.jpg"
    assert img_path.exists()
    with open(img_path, "rb") as f:
        res = client.post("/api/scan", files={"file": ("test_banana.jpg", f, "image/jpeg")})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["produce"] == "Banana", f"Expected Banana, got {data['produce']}"
    assert data["confidence_pct"] >= 75.0
    assert data["freshness"] in ("Fair / Consume Soon", "Overripe / Browning Detected")
run_check("Model Inference: Spotted Banana (POST /api/scan)", check_spotted_banana)

# 8. Model Inference: Fresh Tomato Photo
def check_tomato():
    img_path = PROJECT_ROOT / "frontend" / "assets" / "samples" / "fresh_tomato.jpg"
    assert img_path.exists()
    with open(img_path, "rb") as f:
        res = client.post("/api/scan", files={"file": ("test_tomato.jpg", f, "image/jpeg")})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["produce"] == "Tomato", f"Expected Tomato, got {data['produce']}"
run_check("Model Inference: Fresh Tomato (POST /api/scan)", check_tomato)

# 8b. Model Inference: Persimmon Photo
def check_persimmon():
    img_path = PROJECT_ROOT / "frontend" / "assets" / "samples" / "persimmon.jpg"
    assert img_path.exists()
    with open(img_path, "rb") as f:
        res = client.post("/api/scan", files={"file": ("test_persimmon.jpg", f, "image/jpeg")})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["produce"] == "Persimmon", f"Expected Persimmon, got {data['produce']}"
    assert data["confidence_pct"] >= 70.0
    assert "Countertop" in data["storage_recommendation"]["location"]
run_check("Model Inference: Persimmon (POST /api/scan)", check_persimmon)

# 9. Low-Confidence & Static Noise Handling
def check_static_noise():
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        noise_path = f.name
    try:
        Image.fromarray(arr).save(noise_path)
        with open(noise_path, "rb") as f:
            res = client.post("/api/scan", files={"file": ("noise.jpg", f, "image/jpeg")})
        assert res.status_code == 200
        data = res.json()
        assert data["is_uncertain"] is True
        assert data["warning_notice"] is not None
        assert "warning" in data["warning_notice"].lower() or "uncertainty" in data["warning_notice"].lower()
    finally:
        Path(noise_path).unlink(missing_ok=True)
run_check("Low Confidence & Ambiguity Handling (POST /api/scan with static noise)", check_static_noise)

# 10. Safe Model-Unavailable Error Handling
def check_model_unavailable():
    mock_service = VisionModelService()
    mock_service.model_status = "unavailable"
    mock_service.mode = "model_unavailable"
    mock_service.model_load_error = "Model weights not found on server."
    
    img_path = PROJECT_ROOT / "frontend" / "assets" / "samples" / "green_lemon.jpg"
    res = mock_service.predict(str(img_path))
    assert res["model_status"] == "unavailable"
    assert res["produce"] == "Model Unavailable"
    assert res["is_uncertain"] is True
    assert "AI Model Unavailable" in res["warning_notice"]
    assert "c:\\" not in res["warning_notice"].lower()
run_check("Safe Model-Unavailable Handling (no secret or path leaks)", check_model_unavailable)

# 11. Scan History API
def check_history():
    res = client.get("/api/scan/history")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
run_check("Scan History API (GET /api/scan/history)", check_history)

# 12. Kitchen Inventory API & Expiry Urgency
def check_inventory():
    res = client.get("/api/inventory")
    assert res.status_code == 200
    items = res.json()
    assert isinstance(items, list)
    assert len(items) > 0
    assert "urgency_status" in items[0]
run_check("Kitchen Inventory API (GET /api/inventory)", check_inventory)

# 13. CSV & JSON Inventory Exports
def check_exports():
    res_csv = client.get("/api/inventory/export?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers.get("content-type", "")
    assert "produce_name" in res_csv.text

    res_json = client.get("/api/inventory/export?format=json")
    assert res_json.status_code == 200
    assert isinstance(res_json.json(), list)
run_check("Inventory CSV & JSON Exports (GET /api/inventory/export?format=*)", check_exports)

# 14. Dashboard Analytics Metrics
def check_dashboard():
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    stats = res.json()
    assert "total_active_items" in stats
    assert "waste_prevention_rate_pct" in stats
    assert "freshness_distribution" in stats
    assert "storage_distribution" in stats
run_check("Dashboard Analytics API (GET /api/dashboard)", check_dashboard)

# 15. Produce Storage Encyclopedia Catalog
def check_storage():
    res = client.get("/api/storage-guides")
    assert res.status_code == 200
    guides = res.json()
    assert len(guides) >= 12
run_check("Produce Storage Catalog (GET /api/storage-guides)", check_storage)

print("\n" + "=" * 60)
passed = sum(1 for r in results if r[1] == "PASS")
failed = sum(1 for r in results if r[1] == "FAIL")
print(f"VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED")
print("=" * 60)

if failed > 0:
    sys.exit(1)
