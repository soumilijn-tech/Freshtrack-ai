"""Unit tests for Storage Advisor Service."""
from backend.app.services.storage_advisor import StorageAdvisor

def test_apple_storage_recommendation():
    advice = StorageAdvisor.get_recommendation("Apple", "Fresh")
    assert "Refrigerator" in advice["location"]
    assert "Ethylene" in advice["ethylene_status"]
    assert advice["estimated_shelf_life_days"] > 10
    assert advice["freeze_friendly"] is True

def test_banana_storage_recommendation():
    advice = StorageAdvisor.get_recommendation("Banana", "Fresh")
    assert "Countertop" in advice["location"]
    assert any("crown" in tip.lower() or "stem" in tip.lower() for tip in advice["tips"])
    assert advice["estimated_shelf_life_days"] < 10

def test_potato_storage_recommendation():
    advice = StorageAdvisor.get_recommendation("Potato", "Fresh")
    assert "Pantry" in advice["location"] or "Dark" in advice["location"]
    # Should warn never to refrigerate raw potatoes
    assert any("refrigerate" in tip.lower() for tip in advice["tips"])

def test_persimmon_storage_recommendation():
    advice = StorageAdvisor.get_recommendation("Persimmon", "Fresh")
    assert "Countertop" in advice["location"]
    assert "Ethylene" in advice["ethylene_status"]
    assert any("calyx" in tip.lower() or "fuyu" in tip.lower() or "hachiya" in tip.lower() for tip in advice["tips"])
    assert advice["freeze_friendly"] is True

def test_unknown_produce_fallback():
    advice = StorageAdvisor.get_recommendation("ExoticDragonFruitXYZ", "Fresh")
    assert advice["location"] != ""
    assert len(advice["tips"]) > 0
    assert advice["estimated_shelf_life_days"] > 0

def test_get_all_guides_returns_complete_catalog():
    guides = StorageAdvisor.get_all_guides()
    assert len(guides) >= 12
    names = [g["name"] for g in guides]
    assert "Apple" in names
    assert "Banana" in names
    assert "Tomato" in names
    assert "Broccoli" in names
    assert "Carrot" in names
