"""Database connection and table management using SQLite."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from backend.app.config import DATABASE_PATH

def dict_factory(cursor, row):
    """Convert SQLite row to dictionary."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@contextmanager
def get_db():
    """Context manager for SQLite database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = dict_factory
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initialize database tables and indexes."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Inventory Items Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                produce_name TEXT NOT NULL,
                category TEXT DEFAULT 'Fruit',
                freshness_state TEXT DEFAULT 'Fresh',
                quantity REAL DEFAULT 1.0,
                unit TEXT DEFAULT 'pcs',
                storage_location TEXT DEFAULT 'Refrigerator',
                purchase_date TEXT NOT NULL,
                estimated_expiry_date TEXT NOT NULL,
                status TEXT DEFAULT 'active', -- active, consumed, discarded
                notes TEXT DEFAULT '',
                image_path TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # 2. Scan History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                produce_detected TEXT NOT NULL,
                freshness_detected TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                uncertainty_score REAL NOT NULL,
                top_predictions_json TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                recommended_storage TEXT NOT NULL,
                est_shelf_life_days INTEGER NOT NULL,
                mode TEXT NOT NULL, -- 'demo_heuristic' or 'pytorch_trained'
                scanned_at TEXT NOT NULL
            );
        """)

        # 3. Waste Tracking Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS waste_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER,
                produce_name TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT NOT NULL,
                action TEXT NOT NULL, -- 'consumed' (saved) or 'discarded' (wasted)
                estimated_value REAL DEFAULT 0.0,
                logged_at TEXT NOT NULL,
                FOREIGN KEY (item_id) REFERENCES inventory_items (id) ON DELETE SET NULL
            );
        """)

        # Indexes for fast search and filtering
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inv_status ON inventory_items(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inv_expiry ON inventory_items(estimated_expiry_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_time ON scan_history(scanned_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_waste_time ON waste_logs(logged_at);")
