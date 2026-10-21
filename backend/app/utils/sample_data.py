"""Sample data seeder for realistic demo presentation."""
from datetime import datetime, date, timedelta
import json
from backend.app.database import get_db

def seed_sample_data_if_empty():
    """Seed sample inventory items, scan records, and waste logs if database is empty."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM inventory_items")
        item_count = cursor.fetchone()["count"]

        if item_count > 0:
            return  # Already seeded

        print("[FreshTrack AI] Seeding initial sample data for portfolio demonstration...")

        today = date.today()
        now = datetime.now().isoformat()

        # 1. Sample Inventory Items
        sample_items = [
            (
                "Banana", "Fruit", "Fair / Consume Soon", 4.0, "pcs",
                "Countertop (Room Temperature)",
                (today - timedelta(days=4)).isoformat(),
                (today + timedelta(days=1)).isoformat(),
                "active", "Stem wrapped with foil; starting to develop sugar freckles.", ""
            ),
            (
                "Apple", "Fruit", "Fresh", 6.0, "pcs",
                "Refrigerator Crisper Drawer",
                (today - timedelta(days=3)).isoformat(),
                (today + timedelta(days=22)).isoformat(),
                "active", "Honeycrisp variety, stored in high humidity drawer.", ""
            ),
            (
                "Tomato", "Vegetable", "Fresh", 5.0, "pcs",
                "Countertop",
                (today - timedelta(days=2)).isoformat(),
                (today + timedelta(days=5)).isoformat(),
                "active", "Stored stem-down away from direct window sunlight.", ""
            ),
            (
                "Broccoli", "Vegetable", "Fair / Consume Soon", 1.0, "pcs",
                "Refrigerator Crisper Drawer",
                (today - timedelta(days=5)).isoformat(),
                (today + timedelta(days=1)).isoformat(),
                "active", "Floret crowns starting to soften; plan stir-fry tonight.", ""
            ),
            (
                "Carrot", "Vegetable", "Fresh", 1.0, "kg",
                "Refrigerator Crisper Drawer",
                (today - timedelta(days=4)).isoformat(),
                (today + timedelta(days=18)).isoformat(),
                "active", "Tops trimmed off; submerged in clean water container.", ""
            ),
            (
                "Strawberry", "Fruit", "Fresh", 1.0, "box",
                "Refrigerator",
                (today - timedelta(days=1)).isoformat(),
                (today + timedelta(days=3)).isoformat(),
                "active", "Lined with paper towels, unwashed until serving.", ""
            ),
            (
                "Potato", "Vegetable", "Fresh", 2.5, "kg",
                "Cool, Dark, Well-Ventilated Pantry",
                (today - timedelta(days=7)).isoformat(),
                (today + timedelta(days=28)).isoformat(),
                "active", "In burlap basket, kept away from onions.", ""
            ),
            (
                "Lettuce", "Vegetable", "Overripe / Spoiled", 1.0, "head",
                "Refrigerator",
                (today - timedelta(days=9)).isoformat(),
                (today - timedelta(days=1)).isoformat(),
                "active", "Foliage wilted and translucent. Inspect for composting.", ""
            )
        ]

        for it in sample_items:
            cursor.execute("""
                INSERT INTO inventory_items (
                    produce_name, category, freshness_state, quantity, unit,
                    storage_location, purchase_date, estimated_expiry_date,
                    status, notes, image_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*it, now, now))

        # 2. Sample Scan History
        sample_scans = [
            (
                "Apple", "Fresh", 0.94, 0.08,
                json.dumps([
                    {"label": "Apple", "probability": 0.94, "percentage": 94.0},
                    {"label": "Tomato", "probability": 0.04, "percentage": 4.0},
                    {"label": "Orange", "probability": 0.02, "percentage": 2.0}
                ]),
                "sample_apple.jpg", "Refrigerator Crisper Drawer", 25, "demo_heuristic",
                (datetime.now() - timedelta(hours=3)).isoformat()
            ),
            (
                "Banana", "Fair / Consume Soon", 0.89, 0.12,
                json.dumps([
                    {"label": "Banana", "probability": 0.89, "percentage": 89.0},
                    {"label": "Lemon", "probability": 0.07, "percentage": 7.0},
                    {"label": "Avocado", "probability": 0.04, "percentage": 4.0}
                ]),
                "sample_banana.jpg", "Countertop (Room Temperature)", 2, "demo_heuristic",
                (datetime.now() - timedelta(hours=6)).isoformat()
            ),
            (
                "Tomato", "Fresh", 0.91, 0.09,
                json.dumps([
                    {"label": "Tomato", "probability": 0.91, "percentage": 91.0},
                    {"label": "Bell Pepper", "probability": 0.06, "percentage": 6.0},
                    {"label": "Apple", "probability": 0.03, "percentage": 3.0}
                ]),
                "sample_tomato.jpg", "Countertop", 6, "demo_heuristic",
                (datetime.now() - timedelta(days=1)).isoformat()
            )
        ]

        for sc in sample_scans:
            cursor.execute("""
                INSERT INTO scan_history (
                    produce_detected, freshness_detected, confidence_score, uncertainty_score,
                    top_predictions_json, image_filename, recommended_storage, est_shelf_life_days,
                    mode, scanned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sc)

        # 3. Sample Waste Logs
        sample_logs = [
            (None, "Bell Pepper", 2.0, "pcs", "consumed", 3.00, (today - timedelta(days=3)).isoformat()),
            (None, "Cucumber", 1.0, "pcs", "consumed", 1.20, (today - timedelta(days=2)).isoformat()),
            (None, "Spinach", 1.0, "bag", "discarded", 2.50, (today - timedelta(days=4)).isoformat()),
            (None, "Orange", 3.0, "pcs", "consumed", 2.70, (today - timedelta(days=1)).isoformat()),
            (None, "Banana", 1.0, "pcs", "consumed", 0.50, today.isoformat()),
        ]

        for wl in sample_logs:
            cursor.execute("""
                INSERT INTO waste_logs (
                    item_id, produce_name, quantity, unit, action, estimated_value, logged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, wl)

        print("[FreshTrack AI] Initial sample data successfully seeded.")
