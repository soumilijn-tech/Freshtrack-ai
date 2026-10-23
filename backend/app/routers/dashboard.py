"""Dashboard and System Status API Endpoints."""
from fastapi import APIRouter
from backend.app.schemas.api_schemas import DashboardStatsResponse
from backend.app.services.inventory_service import InventoryService
from backend.app.services.vision_service import vision_service, TORCH_AVAILABLE, ONNX_AVAILABLE
from backend.app.config import APP_VERSION, APP_NAME, MODEL_PATH, ONNX_MODEL_PATH

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
        "onnx_installed": ONNX_AVAILABLE,
        "ai_mode": vision_service.mode,
        "model_status": vision_service.model_status,
        "produce_model": vision_service.produce_model_name,
        "model_file": vision_service.model_file_display,
        "input_shape": getattr(vision_service, "input_shape", None),
        "output_shape": getattr(vision_service, "output_shape", None),
        "is_custom_model_loaded": vision_service.is_trained_model_active(),
        "freshness_assessment_method": "Surface Visual Discoloration Index (Heuristic)",
        "model_load_error": vision_service.model_load_error,
        "model_notice": vision_service.model_notice
    }
