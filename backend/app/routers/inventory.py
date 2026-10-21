"""Inventory API Endpoints for FreshTrack AI."""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Response
from backend.app.schemas.api_schemas import (
    InventoryItemCreate,
    InventoryItemUpdate,
    InventoryItemResponse,
    InventoryActionRequest
)
from backend.app.services.inventory_service import InventoryService

router = APIRouter(prefix="/api/inventory", tags=["Inventory Management"])

@router.get("", response_model=List[InventoryItemResponse])
def get_inventory(
    status: Optional[str] = Query(default="active", description="Filter by status: active, consumed, discarded, all"),
    location: Optional[str] = Query(default=None, description="Filter by storage location"),
    category: Optional[str] = Query(default=None, description="Filter by category: Fruit, Vegetable"),
    search: Optional[str] = Query(default=None, description="Search produce name or notes"),
    sort_by: str = Query(default="expiry", description="Sort by: expiry, purchase, name")
):
    """Retrieve filtered, sorted inventory items with computed urgency badges."""
    return InventoryService.list_items(
        status=status,
        location=location,
        category=category,
        search=search,
        sort_by=sort_by
    )

@router.get("/export")
def export_inventory(format: str = Query(default="csv", pattern="^(csv|json)$")):
    """Export the entire inventory as a downloadable CSV or JSON file."""
    content, media_type = InventoryService.export_inventory(export_format=format)
    filename = f"freshtrack_inventory.{format}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/{item_id}", response_model=InventoryItemResponse)
def get_inventory_item(item_id: int):
    """Get details of a single inventory item."""
    item = InventoryService.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Inventory item #{item_id} not found")
    return item

@router.post("", response_model=InventoryItemResponse, status_code=201)
def add_inventory_item(item: InventoryItemCreate):
    """Add a new produce item to the kitchen inventory."""
    created = InventoryService.create_item(item.model_dump())
    return created

@router.put("/{item_id}", response_model=InventoryItemResponse)
def update_inventory_item(item_id: int, updates: InventoryItemUpdate):
    """Update fields of an inventory item."""
    updated = InventoryService.update_item(item_id, updates.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Inventory item #{item_id} not found")
    return updated

@router.delete("/{item_id}")
def delete_inventory_item(item_id: int):
    """Delete an item from inventory."""
    success = InventoryService.delete_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Inventory item #{item_id} not found")
    return {"message": f"Inventory item #{item_id} deleted successfully"}

@router.post("/{item_id}/action", response_model=InventoryItemResponse)
def record_inventory_action(item_id: int, action_req: InventoryActionRequest):
    """Mark an inventory item as Consumed (saved from waste) or Discarded (wasted)."""
    result = InventoryService.log_action(
        item_id=item_id,
        action=action_req.action,
        quantity=action_req.quantity,
        estimated_value=action_req.estimated_value or 0.0
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Inventory item #{item_id} not found")
    return result
