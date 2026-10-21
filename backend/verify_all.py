"""Comprehensive verification script for FreshTrack AI."""
import urllib.request
import urllib.error
import json
import sys
from pathlib import Path

BASE = "http://127.0.0.1:8000"

results = []

def run_check(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        results.append((name, "PASS", None))
    except Exception as e:
        print(f"FAIL: {name} -> {e}")
        results.append((name, "FAIL", str(e)))

# 1. Root Webpage
def check_root():
    with urllib.request.urlopen(f"{BASE}/") as res:
        assert res.status == 200
        html = res.read().decode("utf-8")
        assert "<title>FreshTrack AI" in html
        assert "Produce Freshness Scanner" in html
run_check("Frontend Webpage (GET /)", check_root)

# 2. Swagger Docs
def check_docs():
    with urllib.request.urlopen(f"{BASE}/docs") as res:
        assert res.status == 200
        html = res.read().decode("utf-8")
        assert "swagger-ui" in html.lower()
run_check("Interactive Swagger UI (GET /docs)", check_docs)

# 3. OpenAPI Specification
def check_openapi():
    with urllib.request.urlopen(f"{BASE}/openapi.json") as res:
        assert res.status == 200
        data = json.loads(res.read())
        assert data["info"]["title"] == "FreshTrack AI"
        assert "/api/scan" in data["paths"]
run_check("OpenAPI Schema (GET /openapi.json)", check_openapi)

# 4. System Status API
def check_status():
    with urllib.request.urlopen(f"{BASE}/api/system-status") as res:
        assert res.status == 200
        data = json.loads(res.read())
        assert data["status"] == "online"
        assert data["ai_mode"] in ("demo_heuristic", "pytorch_trained", "pretrained_onnx_mobilenet")
run_check("System Status API (GET /api/system-status)", check_status)

# 5. Multipart Image Upload Scan
def check_upload():
    import httpx
    sample_img = Path("frontend/assets/samples/fresh_apple.jpg")
    assert sample_img.exists()
    with open(sample_img, "rb") as f:
        files = {"file": ("uploaded_apple.jpg", f, "image/jpeg")}
        res = httpx.post(f"{BASE}/api/scan", files=files)
        if res.status_code != 200:
            raise Exception(f"HTTP {res.status_code}: {res.text}")
        data = res.json()
        assert data["produce"] == "Apple", f"Expected Apple, got {data['produce']}"
        assert data["freshness"] == "Fresh", f"Expected Fresh, got {data['freshness']}"
        assert data["confidence_pct"] > 50
        assert "storage_recommendation" in data
        assert data["is_food_safety_guaranteed"] is False
run_check("Image Upload Scan Flow (POST /api/scan)", check_upload)

# 6. Sample Scan API
def check_sample():
    req = urllib.request.Request(f"{BASE}/api/scan/sample/spotted_banana", method="POST")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read())
        assert data["produce"] == "Banana"
        assert data["freshness"] == "Fair / Consume Soon"
run_check("Sample Scan API (POST /api/scan/sample/spotted_banana)", check_sample)

# 7. Scan History
def check_history():
    with urllib.request.urlopen(f"{BASE}/api/scan/history") as res:
        assert res.status == 200
        data = json.loads(res.read())
        assert isinstance(data, list)
        assert len(data) > 0
run_check("Scan History API (GET /api/scan/history)", check_history)

# 8. Inventory API & Expiry Urgency
def check_inventory():
    with urllib.request.urlopen(f"{BASE}/api/inventory") as res:
        assert res.status == 200
        items = json.loads(res.read())
        assert isinstance(items, list)
        assert len(items) > 0
        assert "urgency_status" in items[0]
run_check("Inventory API (GET /api/inventory)", check_inventory)

# 9. Dashboard Statistics
def check_dashboard():
    with urllib.request.urlopen(f"{BASE}/api/dashboard") as res:
        assert res.status == 200
        stats = json.loads(res.read())
        assert "total_active_items" in stats
        assert "waste_prevention_rate_pct" in stats
        assert "freshness_distribution" in stats
        assert "storage_distribution" in stats
run_check("Dashboard Analytics API (GET /api/dashboard)", check_dashboard)

# 10. Storage Guides
def check_storage():
    with urllib.request.urlopen(f"{BASE}/api/storage-guides") as res:
        assert res.status == 200
        guides = json.loads(res.read())
        assert len(guides) >= 12
run_check("Produce Storage Guides (GET /api/storage-guides)", check_storage)

print("\n" + "="*50)
print(f"SUMMARY: {sum(1 for r in results if r[1] == 'PASS')} PASSED, {sum(1 for r in results if r[1] == 'FAIL')} FAILED")
print("="*50)

if any(r[1] == "FAIL" for r in results):
    sys.exit(1)
