"""Main FastAPI Application Entrypoint for FreshTrack AI."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.app.config import (
    APP_NAME,
    APP_DESCRIPTION,
    APP_VERSION,
    ALLOWED_ORIGINS,
    UPLOADS_DIR,
    FRONTEND_DIR,
)
from backend.app.database import init_db
from backend.app.utils.sample_data import seed_sample_data_if_empty
from backend.app.routers import scan, inventory, dashboard, storage

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for database initialization and sample data seeding."""
    print(f"[FreshTrack AI] Initializing SQLite database and directories...")
    init_db()
    seed_sample_data_if_empty()
    yield
    print(f"[FreshTrack AI] Shutting down application gracefully...")

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(scan.router)
app.include_router(inventory.router)
app.include_router(dashboard.router)
app.include_router(storage.router)

# Mount Static Directories if they exist
if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

assets_dir = FRONTEND_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

css_dir = FRONTEND_DIR / "css"
if css_dir.exists():
    app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")

js_dir = FRONTEND_DIR / "js"
if js_dir.exists():
    app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

# Serve Frontend HTML
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "message": "FreshTrack AI API is running. Frontend index.html not yet placed in /frontend.",
        "docs": "/api/docs"
    })

# Fallback for client-side routing
@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("uploads/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"detail": "Not Found"}, status_code=404)
