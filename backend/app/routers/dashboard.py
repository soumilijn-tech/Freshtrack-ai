"""Dashboard and System Status API Endpoints."""
from fastapi import APIRouter
from backend.app.schemas.api_schemas import DashboardStatsResponse
from backend.app.services.inventory_service import InventoryService
from backend.app.services.vision_service import vision_service, TORCH_AVAILABLE
from backend.app.config import APP_VERSION, APP_NAME, MODEL_PATH

router = APIRouter(prefix="/api", tags=["Dashboard & Analytics"])

@router.get("/dashboard", response_model=DashboardStatsResponse)
def get_dashboard_metrics():
    """Retrieve summarized food waste metrics, active items, and chart distributions."""
    return InventoryService.get_dashboard_stats()

@router.get("/system-status")
def get_system_status():
    """Check AI model status, environment configurations, and backend health."""
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "pytorch_installed": TORCH_AVAILABLE,
        "ai_mode": vision_service.mode,
        "model_file_path": MODEL_PATH,
        "is_custom_model_loaded": vision_service.is_trained_model_active(),
        "model_notice": vision_service.model_notice
    }
