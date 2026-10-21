"""Unit tests for Inventory API and Export."""
import pytest
from starlette.testclient import TestClient
from datetime import date, timedelta

from backend.app.main import app
from backend.app.database import init_db
from backend.app.utils.sample_data import seed_sample_data_if_empty

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    seed_sample_data_if_empty()

@pytest.fixture
def client():
    return TestClient(app)

def test_get_inventory_list(client):
    response = client.get("/api/inventory")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Verify fields
    item = data[0]
    assert "produce_name" in item
    assert "days_remaining" in item
    assert "urgency_status" in item

def test_create_and_delete_inventory_item(client):
    today = date.today()
    new_item = {
        "produce_name": "Test Avocado",
        "category": "Fruit",
        "freshness_state": "Fresh",
        "quantity": 3.0,
        "unit": "pcs",
        "storage_location": "Countertop",
        "purchase_date": today.isoformat(),
        "estimated_expiry_date": (today + timedelta(days=5)).isoformat(),
        "notes": "Testing create item"
    }

    # Create
    create_res = client.post("/api/inventory", json=new_item)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["produce_name"] == "Test Avocado"
    item_id = created_data["id"]

    # Read back
    get_res = client.get(f"/api/inventory/{item_id}")
    assert get_res.status_code == 200
    assert get_res.json()["produce_name"] == "Test Avocado"

    # Update
    update_res = client.put(f"/api/inventory/{item_id}", json={"notes": "Updated note"})
    assert update_res.status_code == 200
    assert update_res.json()["notes"] == "Updated note"

    # Action (mark as consumed)
    action_res = client.post(f"/api/inventory/{item_id}/action", json={"action": "consumed", "estimated_value": 4.50})
    assert action_res.status_code == 200
    assert action_res.json()["status"] == "consumed"

    # Delete
    del_res = client.delete(f"/api/inventory/{item_id}")
    assert del_res.status_code == 200

    # Verify deleted
    not_found_res = client.get(f"/api/inventory/{item_id}")
    assert not_found_res.status_code == 404

def test_inventory_export_csv_and_json(client):
    # CSV export
    csv_res = client.get("/api/inventory/export?format=csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers.get("content-type", "")
    assert "produce_name" in csv_res.text

    # JSON export
    json_res = client.get("/api/inventory/export?format=json")
    assert json_res.status_code == 200
    assert "application/json" in json_res.headers.get("content-type", "")
    data = json_res.json()
    assert isinstance(data, list)
