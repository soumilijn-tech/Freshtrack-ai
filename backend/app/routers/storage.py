"""Produce Storage Guides API Router."""
from typing import List
from fastapi import APIRouter, HTTPException
from backend.app.schemas.api_schemas import StorageGuideItem
from backend.app.services.storage_advisor import StorageAdvisor, PRODUCE_KNOWLEDGE_BASE

router = APIRouter(prefix="/api/storage-guides", tags=["Produce Storage Knowledge Base"])

@router.get("", response_model=List[StorageGuideItem])
def list_all_storage_guides():
    """Retrieve full catalog of fruits & vegetables storage guides."""
    return StorageAdvisor.get_all_guides()

@router.get("/{produce_name}", response_model=StorageGuideItem)
def get_single_storage_guide(produce_name: str):
    """Retrieve detailed storage guidance for a specific produce."""
    clean_name = produce_name.strip().title()
    data = PRODUCE_KNOWLEDGE_BASE.get(clean_name)
    if not data:
        raise HTTPException(status_code=404, detail=f"No dedicated storage guide found for '{produce_name}'.")

    return {
        "name": clean_name,
        "category": data["category"],
        "optimal_location": data["optimal_location"],
        "temperature_range": data["ideal_temp"],
        "humidity_recommendation": data["humidity"],
        "ethylene_profile": data["ethylene_status"],
        "preservation_tips": data["tips"],
        "shelf_life": data["shelf_life_table"],
        "can_freeze": data["freeze_friendly"],
        "freezing_method": data["freezing_instructions"]
    }
