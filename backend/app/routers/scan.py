"""Scan and Computer Vision Inference Endpoints."""
import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.app.config import UPLOADS_DIR, SAMPLE_IMAGES_DIR
from backend.app.services.vision_service import vision_service
from backend.app.schemas.api_schemas import ScanResultResponse, ScanHistoryItem
from backend.app.database import get_db

router = APIRouter(prefix="/api", tags=["Scan & AI Inference"])

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

@router.post("/scan", response_model=ScanResultResponse)
async def upload_and_scan(file: UploadFile = File(...)):
    """Upload an image of fruit or vegetable to classify produce and estimate freshness."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Supported formats: JPG, JPEG, PNG, WEBP, BMP."
        )

    # Save to uploads directory with unique name
    unique_filename = f"scan_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
    dest_path = UPLOADS_DIR / unique_filename

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file exceeds 10MB limit.")

        with open(dest_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Run AI inference
    try:
        prediction = vision_service.predict(str(dest_path))
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

    # Record scan in database
    now = datetime.now().isoformat()
    scan_id = None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scan_history (
                produce_detected, freshness_detected, confidence_score, uncertainty_score,
                top_predictions_json, image_filename, recommended_storage, est_shelf_life_days,
                mode, scanned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction["produce"],
            prediction["freshness"],
            prediction["confidence"],
            prediction["uncertainty_score"],
            json.dumps(prediction["top_produce_predictions"]),
            unique_filename,
            prediction["storage_recommendation"]["location"],
            prediction["storage_recommendation"]["estimated_shelf_life_days"],
            prediction["mode"],
            now
        ))
        scan_id = cursor.lastrowid

    prediction["scan_id"] = scan_id
    prediction["image_url"] = f"/uploads/{unique_filename}"
    return prediction

@router.post("/scan/sample/{sample_name}", response_model=ScanResultResponse)
async def scan_sample_image(sample_name: str):
    """Run instant scan on a bundled sample test image."""
    sample_file = SAMPLE_IMAGES_DIR / f"{sample_name}.jpg"
    if not sample_file.exists():
        # Try finding by exact name
        matches = list(SAMPLE_IMAGES_DIR.glob(f"{sample_name}.*"))
        if matches:
            sample_file = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Sample image '{sample_name}' not found.")

    try:
        prediction = vision_service.predict(str(sample_file))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to scan sample image: {str(e)}")

    now = datetime.now().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scan_history (
                produce_detected, freshness_detected, confidence_score, uncertainty_score,
                top_predictions_json, image_filename, recommended_storage, est_shelf_life_days,
                mode, scanned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction["produce"],
            prediction["freshness"],
            prediction["confidence"],
            prediction["uncertainty_score"],
            json.dumps(prediction["top_produce_predictions"]),
            sample_file.name,
            prediction["storage_recommendation"]["location"],
            prediction["storage_recommendation"]["estimated_shelf_life_days"],
            prediction["mode"],
            now
        ))
        scan_id = cursor.lastrowid

    prediction["scan_id"] = scan_id
    prediction["image_url"] = f"/assets/samples/{sample_file.name}"
    return prediction

@router.get("/scan/history")
def get_scan_history(limit: int = Query(default=20, ge=1, le=100)):
    """Retrieve recent scan history records."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scan_history ORDER BY scanned_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["image_url"] = f"/uploads/{d['image_filename']}" if not d['image_filename'].startswith("sample_") else f"/assets/samples/{d['image_filename']}"
        try:
            d["top_predictions"] = json.loads(d["top_predictions_json"])
        except Exception:
            d["top_predictions"] = []
        results.append(d)
    return results

@router.delete("/scan/history")
def clear_scan_history():
    """Clear all scan history entries."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scan_history")
    return {"message": "Scan history cleared successfully"}

@router.get("/sample-images")
def list_sample_images():
    """List available pre-loaded sample images for instant demo testing."""
    samples = []
    if SAMPLE_IMAGES_DIR.exists():
        for p in SAMPLE_IMAGES_DIR.glob("*.*"):
            if p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                label = p.stem.replace("_", " ").title()
                samples.append({
                    "id": p.stem,
                    "name": label,
                    "filename": p.name,
                    "url": f"/assets/samples/{p.name}"
                })
    return samples
