"""Unit tests for Vision Model Service and Demo Mode."""
import os
import tempfile
import pytest
from PIL import Image
import numpy as np

from backend.app.services.vision_service import VisionModelService

@pytest.fixture
def test_image():
    """Create a temporary test image with red hue (simulating apple/tomato)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name
    
    # Create a 200x200 red image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = 220  # High red
    arr[:, :, 1] = 40   # Low green
    arr[:, :, 2] = 40   # Low blue
    img = Image.fromarray(arr)
    img.save(img_path)

    yield img_path

    if os.path.exists(img_path):
        os.remove(img_path)

@pytest.fixture
def spotted_image():
    """Create a temporary test image with yellow background and brown spots."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name
    
    # Yellow background
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :, 0] = 240  # Red
    arr[:, :, 1] = 220  # Green
    arr[:, :, 2] = 50   # Blue
    # Add dark brown spots
    arr[50:100, 50:100, :] = [80, 50, 20]
    img = Image.fromarray(arr)
    img.save(img_path)

    yield img_path

    if os.path.exists(img_path):
        os.remove(img_path)

def test_vision_service_initialization():
    service = VisionModelService()
    assert service.mode in ("demo_heuristic", "pytorch_trained")
    assert service.model_notice != ""

def test_vision_prediction_schema_and_disclaimer(test_image):
    service = VisionModelService()
    result = service.predict(test_image)

    # Core identification fields
    assert "produce" in result
    assert "freshness" in result
    assert "confidence" in result
    assert "confidence_pct" in result
    assert "uncertainty_score" in result
    assert "is_uncertain" in result

    # Probabilities sum sanity check
    top_p = result["top_produce_predictions"]
    assert len(top_p) == 3
    for item in top_p:
        assert 0.0 <= item["probability"] <= 1.0

    # Food safety disclosure check
    assert result["is_food_safety_guaranteed"] is False
    assert "⚠️ Visual Freshness Estimate Only" in result["safety_disclaimer"]
    assert "microorganism" in result["safety_disclaimer"].lower() or "bacteria" in result["safety_disclaimer"].lower()

    # Storage recommendation included
    storage = result["storage_recommendation"]
    assert "location" in storage
    assert "tips" in storage
    assert len(storage["tips"]) > 0

def test_spotted_image_heuristic_detection(spotted_image):
    service = VisionModelService()
    result = service.predict(spotted_image)
    
    # Should detect yellow produce and overripe/fair status
    assert result["produce"] in ("Banana", "Lemon", "Apple", "Orange")
    assert result["freshness"] in ("Fair / Consume Soon", "Overripe / Spoiled")

def test_invalid_image_raises_error():
    service = VisionModelService()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Not an image")
        bad_path = f.name

    try:
        with pytest.raises(ValueError):
            service.predict(bad_path)
    finally:
        if os.path.exists(bad_path):
            os.remove(bad_path)
