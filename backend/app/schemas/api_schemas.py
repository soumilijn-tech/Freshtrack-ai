"""Pydantic schemas for request validation and API responses."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    label: str
    probability: float
    percentage: float

class StorageRecommendation(BaseModel):
    location: str
    ideal_temp: str
    humidity: str
    ethylene_status: str
    tips: List[str]
    estimated_shelf_life_days: int
    freeze_friendly: bool
    freezing_instructions: str

class ScanResultResponse(BaseModel):
    scan_id: Optional[int] = None
    produce: str
    category: str
    freshness: str
    confidence: float
    confidence_pct: float
    uncertainty_score: float
    is_uncertain: bool
    top_produce_predictions: List[PredictionItem]
    top_freshness_predictions: List[PredictionItem]
    storage_recommendation: StorageRecommendation
    image_url: str
    mode: str  # 'demo_heuristic' or 'pytorch_trained'
    model_notice: str
    safety_disclaimer: str
    is_food_safety_guaranteed: bool = False
    inference_time_ms: float

class ScanHistoryItem(BaseModel):
    id: int
    produce_detected: str
    freshness_detected: str
    confidence_score: float
    uncertainty_score: float
    image_filename: str
    image_url: str
    recommended_storage: str
    est_shelf_life_days: int
    mode: str
    scanned_at: str

# Inventory Schemas
class InventoryItemCreate(BaseModel):
    produce_name: str = Field(..., min_length=1, max_length=100)
    category: str = Field(default="Fruit")
    freshness_state: str = Field(default="Fresh")
    quantity: float = Field(default=1.0, gt=0)
    unit: str = Field(default="pcs")
    storage_location: str = Field(default="Refrigerator")
    purchase_date: str = Field(...)  # YYYY-MM-DD
    estimated_expiry_date: str = Field(...)  # YYYY-MM-DD
    notes: Optional[str] = Field(default="")
    image_path: Optional[str] = Field(default="")

class InventoryItemUpdate(BaseModel):
    produce_name: Optional[str] = None
    category: Optional[str] = None
    freshness_state: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    storage_location: Optional[str] = None
    purchase_date: Optional[str] = None
    estimated_expiry_date: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

class InventoryActionRequest(BaseModel):
    action: str = Field(..., pattern="^(consumed|discarded)$")
    quantity: Optional[float] = None
    estimated_value: Optional[float] = Field(default=0.0, ge=0.0)

class InventoryItemResponse(BaseModel):
    id: int
    produce_name: str
    category: str
    freshness_state: str
    quantity: float
    unit: str
    storage_location: str
    purchase_date: str
    estimated_expiry_date: str
    days_remaining: int
    urgency_status: str  # 'Fresh', 'Expiring Soon', 'Expired', 'Consumed', 'Discarded'
    status: str
    notes: str
    image_path: str
    created_at: str
    updated_at: str

# Dashboard Schemas
class DashboardStatsResponse(BaseModel):
    total_active_items: int
    fresh_count: int
    expiring_soon_count: int
    expired_count: int
    consumed_count: int
    discarded_count: int
    waste_prevention_rate_pct: float
    estimated_money_saved: float
    estimated_money_wasted: float
    storage_distribution: Dict[str, int]
    freshness_distribution: Dict[str, int]
    recent_scans: List[ScanHistoryItem]

# Storage Guide Schemas
class StorageGuideItem(BaseModel):
    name: str
    category: str
    optimal_location: str
    temperature_range: str
    humidity_recommendation: str
    ethylene_profile: str
    preservation_tips: List[str]
    shelf_life: Dict[str, str]
    can_freeze: bool
    freezing_method: str
