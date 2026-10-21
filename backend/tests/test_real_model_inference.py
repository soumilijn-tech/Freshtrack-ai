"""Tests for Pretrained ONNX MobileNetV2 produce classification and low-confidence warnings."""
import os
import shutil
import tempfile
import pytest
from PIL import Image
import numpy as np

from backend.app.services.vision_service import vision_service
from backend.app.config import SAMPLE_IMAGES_DIR


def test_model_is_pretrained_onnx():
    """Verify that the real pretrained ONNX MobileNetV2 model is loaded, not demo mode."""
    assert vision_service.mode == "pretrained_onnx_mobilenet"
    assert vision_service.is_trained_model_active() is True
    assert "MobileNetV2" in vision_service.model_notice


def test_green_lemon_classified_as_lemon():
    """Verify that a green lemon is classified as Lemon, solving the potato fallback bug."""
    green_lemon_sample = SAMPLE_IMAGES_DIR / "green_lemon.jpg"
    assert green_lemon_sample.exists(), "green_lemon.jpg sample image must exist"

    # Copy to a temporary file with random name to ensure classification relies on image content
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    shutil.copy(green_lemon_sample, tmp_path)

    try:
        result = vision_service.predict(tmp_path)
        # Must be Lemon, NOT Potato!
        assert result["produce"] == "Lemon"
        assert result["confidence_pct"] >= 60.0
        assert result["mode"] == "pretrained_onnx_mobilenet"
        assert result["is_uncertain"] is False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_low_confidence_triggers_warning_on_ambiguous_image():
    """Verify that ambiguous or non-food images trigger low confidence warning."""
    # Completely random static noise (non-food)
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img_path = f.name
    Image.fromarray(arr).save(img_path)

    try:
        result = vision_service.predict(img_path)
        # Ambiguous noise should be marked uncertain or have low confidence
        assert result["is_uncertain"] is True or result["confidence_pct"] < 45.0
        assert result["warning_notice"] is not None
        assert "warning" in result["warning_notice"].lower() or "ambiguity" in result["warning_notice"].lower()
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def test_clean_vs_spotted_freshness_difference():
    """Verify that spotted produce is classified as Fair or Overripe, while clean is Fresh."""
    # Clean produce image
    arr_clean = np.full((224, 224, 3), 200, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        clean_path = f.name
    Image.fromarray(arr_clean).save(clean_path)

    # Heavily spotted produce image
    arr_spotted = np.full((224, 224, 3), 200, dtype=np.uint8)
    arr_spotted[30:90, 30:90, :] = [40, 20, 10]
    arr_spotted[120:170, 100:160, :] = [30, 20, 15]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        spotted_path = f.name
    Image.fromarray(arr_spotted).save(spotted_path)

    try:
        res_clean = vision_service.predict(clean_path)
        res_spotted = vision_service.predict(spotted_path)

        assert res_clean["freshness"] == "Fresh"
        assert res_spotted["freshness"] in ("Fair / Consume Soon", "Overripe / Spoiled")
    finally:
        if os.path.exists(clean_path):
            os.remove(clean_path)
        if os.path.exists(spotted_path):
            os.remove(spotted_path)
