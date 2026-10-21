# Multi-platform production Dockerfile for FreshTrack AI
# Compatible with Render, Railway, Hugging Face Spaces, and Fly.io
FROM python:3.11-slim

# Avoid writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy source code
COPY backend /app/backend
COPY frontend /app/frontend
COPY notebooks /app/notebooks
COPY .env.example /app/.env.example

# Create runtime directories
RUN mkdir -p /app/data /app/uploads /app/backend/models

# Expose port (Render/Railway use PORT env var)
ENV APP_PORT=8000
EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
