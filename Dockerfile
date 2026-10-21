# Multi-platform production Dockerfile for FreshTrack AI
# Compatible with Hugging Face Spaces, Render, and Railway
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

# Create non-root user (Hugging Face Spaces default UID is 1000)
RUN useradd -m -u 1000 user

# Copy source code
COPY --chown=user:user main.py /app/main.py
COPY --chown=user:user backend /app/backend
COPY --chown=user:user frontend /app/frontend
COPY --chown=user:user notebooks /app/notebooks
COPY --chown=user:user .env.example /app/.env.example

# Create runtime directories with appropriate permissions
RUN mkdir -p /app/data /app/uploads /app/backend/models && chown -R user:user /app

# Switch to non-root user for Hugging Face security
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI via Uvicorn on 0.0.0.0
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
