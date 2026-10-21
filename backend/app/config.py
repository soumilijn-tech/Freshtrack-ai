"""Configuration settings for FreshTrack AI."""
import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
MODELS_DIR = BACKEND_DIR / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SAMPLE_IMAGES_DIR = FRONTEND_DIR / "assets" / "samples"

DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "freshtrack.db"))
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "freshtrack_model.pth"))
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", str(MODELS_DIR / "mobilenetv2.onnx"))
IMAGENET_CLASSES_PATH = os.getenv("IMAGENET_CLASSES_PATH", str(MODELS_DIR / "imagenet_classes.txt"))

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# App Settings
APP_NAME = "FreshTrack AI"
APP_DESCRIPTION = "Smart Fruit & Vegetable Freshness Assistant"
APP_VERSION = "1.0.0"
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))
DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

# CORS Origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*",
]

# Supported Produce Classes
SUPPORTED_PRODUCE = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Potato",
    "Carrot",
    "Bell Pepper",
    "Broccoli",
    "Strawberry",
    "Cucumber",
    "Onion",
    "Lettuce",
    "Avocado",
    "Lemon",
]

# Freshness States
FRESHNESS_STATES = ["Fresh", "Fair / Consume Soon", "Overripe / Spoiled"]
