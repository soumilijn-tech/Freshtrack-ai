"""Inventory Management and Food Waste Statistics Service."""
import io
import csv
import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional

from backend.app.database import get_db

class InventoryService:
    """Handles CRUD, search/filtering, export, and waste KPI metrics."""

    @staticmethod
    def _compute_urgency(expiry_date_str: str, status: str) -> tuple[int, str]:
        """Compute remaining days and urgency label."""
        if status in ("consumed", "discarded"):
            return 0, status.capitalize()

        try:
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
            today = date.today()
            days = (expiry_date - today).days
            if days < 0:
                return days, "Expired"
            elif days <= 2:
                return days, "Expiring Soon"
            else:
                return days, "Fresh"
        except Exception:
            return 0, "Unknown"

    @classmethod
    def list_items(
        cls,
        status: Optional[str] = "active",
        location: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "expiry"
    ) -> List[Dict[str, Any]]:
        """List inventory items with search, filter, and dynamic urgency calculation."""
        query = "SELECT * FROM inventory_items WHERE 1=1"
        params = []

        if status and status != "all":
            query += " AND status = ?"
            params.append(status)

        if location and location != "all":
            query += " AND storage_location = ?"
            params.append(location)

        if category and category != "all":
            query += " AND category = ?"
            params.append(category)

        if search:
            query += " AND (produce_name LIKE ? OR notes LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        # Sorting
        if sort_by == "purchase":
            query += " ORDER BY purchase_date DESC"
        elif sort_by == "name":
            query += " ORDER BY produce_name ASC"
        else:
            query += " ORDER BY estimated_expiry_date ASC"

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            days, urgency = cls._compute_urgency(row["estimated_expiry_date"], row["status"])
            item = dict(row)
            item["days_remaining"] = days
            item["urgency_status"] = urgency
            results.append(item)

        return results

    @classmethod
    def get_item(cls, item_id: int) -> Optional[Dict[str, Any]]:
        """Fetch single inventory item by ID."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inventory_items WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            if not row:
                return None
            days, urgency = cls._compute_urgency(row["estimated_expiry_date"], row["status"])
            item = dict(row)
            item["days_remaining"] = days
            item["urgency_status"] = urgency
            return item

    @classmethod
    def create_item(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new inventory item."""
        now = datetime.now().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inventory_items (
                    produce_name, category, freshness_state, quantity, unit,
                    storage_location, purchase_date, estimated_expiry_date,
                    status, notes, image_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """, (
                data["produce_name"].strip().title(),
                data.get("category", "Fruit"),
                data.get("freshness_state", "Fresh"),
                float(data.get("quantity", 1.0)),
                data.get("unit", "pcs"),
                data.get("storage_location", "Refrigerator"),
                data["purchase_date"],
                data["estimated_expiry_date"],
                data.get("notes", ""),
                data.get("image_path", ""),
                now,
                now
            ))
            new_id = cursor.lastrowid
        return cls.get_item(new_id)

    @classmethod
    def update_item(cls, item_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update fields of an inventory item."""
        now = datetime.now().isoformat()
        fields = []
        params = []
        for k, v in updates.items():
            if v is not None and k not in ("id", "created_at"):
                fields.append(f"{k} = ?")
                params.append(v)

        if not fields:
            return cls.get_item(item_id)

        fields.append("updated_at = ?")
        params.append(now)
        params.append(item_id)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE inventory_items SET {', '.join(fields)} WHERE id = ?", params)

        return cls.get_item(item_id)

    @classmethod
    def delete_item(cls, item_id: int) -> bool:
        """Delete inventory item."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM inventory_items WHERE id = ?", (item_id,))
            return cursor.rowcount > 0

    @classmethod
    def log_action(cls, item_id: int, action: str, quantity: Optional[float] = None, estimated_value: float = 0.0) -> Optional[Dict[str, Any]]:
        """Mark item as consumed (saved) or discarded (wasted) and record in waste_logs."""
        item = cls.get_item(item_id)
        if not item:
            return None

        qty = quantity if quantity is not None else item["quantity"]
        val = estimated_value if estimated_value > 0 else (qty * 1.50)  # estimate $1.50/unit if not provided
        now = datetime.now().isoformat()

        with get_db() as conn:
            cursor = conn.cursor()
            # Update item status
            cursor.execute("""
                UPDATE inventory_items 
                SET status = ?, updated_at = ? 
                WHERE id = ?
            """, (action, now, item_id))

            # Record waste log entry
            cursor.execute("""
                INSERT INTO waste_logs (
                    item_id, produce_name, quantity, unit, action, estimated_value, logged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (item_id, item["produce_name"], qty, item["unit"], action, val, now))

        return cls.get_item(item_id)

    @classmethod
    def export_inventory(cls, export_format: str = "csv") -> tuple[str, str]:
        """Export inventory in CSV or JSON format."""
        items = cls.list_items(status="all")

        if export_format.lower() == "json":
            return json.dumps(items, indent=2), "application/json"

        # CSV format
        output = io.StringIO()
        fieldnames = [
            "id", "produce_name", "category", "freshness_state", "quantity", "unit",
            "storage_location", "purchase_date", "estimated_expiry_date", "days_remaining",
            "urgency_status", "status", "notes", "created_at"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for it in items:
            writer.writerow(it)
        return output.getvalue(), "text/csv"

    @classmethod
    def get_dashboard_stats(cls) -> Dict[str, Any]:
        """Compute waste statistics, storage distribution, and freshness breakdown."""
        items = cls.list_items(status="all")
        active_items = [it for it in items if it["status"] == "active"]

        fresh_count = sum(1 for it in active_items if it["urgency_status"] == "Fresh")
        expiring_soon_count = sum(1 for it in active_items if it["urgency_status"] == "Expiring Soon")
        expired_count = sum(1 for it in active_items if it["urgency_status"] == "Expired")

        with get_db() as conn:
            cursor = conn.cursor()
            # Consumed & Discarded counts & values from waste_logs
            cursor.execute("""
                SELECT 
                    action, 
                    COUNT(*) as count, 
                    SUM(estimated_value) as total_val 
                FROM waste_logs 
                GROUP BY action
            """)
            waste_stats = {r["action"]: {"count": r["count"], "value": r["total_val"] or 0.0} for r in cursor.fetchall()}

            # Recent scans (limit 6)
            cursor.execute("SELECT * FROM scan_history ORDER BY scanned_at DESC LIMIT 6")
            recent_scans = []
            for row in cursor.fetchall():
                scan_dict = dict(row)
                scan_dict["image_url"] = f"/uploads/{scan_dict['image_filename']}"
                recent_scans.append(scan_dict)

        consumed_count = waste_stats.get("consumed", {}).get("count", 0)
        discarded_count = waste_stats.get("discarded", {}).get("count", 0)
        saved_val = waste_stats.get("consumed", {}).get("value", 0.0)
        wasted_val = waste_stats.get("discarded", {}).get("value", 0.0)

        total_resolved = consumed_count + discarded_count
        prevention_rate = (
            round((consumed_count / total_resolved) * 100, 1)
            if total_resolved > 0 else 92.0
        )

        # Storage location distribution (active items)
        storage_dist: Dict[str, int] = {}
        for it in active_items:
            loc = it.get("storage_location", "Other")
            storage_dist[loc] = storage_dist.get(loc, 0) + 1

        # Freshness distribution (active items)
        freshness_dist: Dict[str, int] = {
            "Fresh": fresh_count,
            "Fair / Consume Soon": expiring_soon_count,
            "Overripe / Spoiled": expired_count,
        }

        return {
            "total_active_items": len(active_items),
            "fresh_count": fresh_count,
            "expiring_soon_count": expiring_soon_count,
            "expired_count": expired_count,
            "consumed_count": consumed_count,
            "discarded_count": discarded_count,
            "waste_prevention_rate_pct": prevention_rate,
            "estimated_money_saved": round(saved_val, 2),
            "estimated_money_wasted": round(wasted_val, 2),
            "storage_distribution": storage_dist,
            "freshness_distribution": freshness_dist,
            "recent_scans": recent_scans,
        }
