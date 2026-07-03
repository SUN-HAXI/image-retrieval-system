# ── Image Retrieval System — Dockerfile ───────────────────────────────
# Build:  docker build -t image-retrieval-system .
# Run:    docker run -p 5000:5000 -v $(pwd)/data:/app/data image-retrieval-system

FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Directories
RUN mkdir -p data/database data/uploads index

# Flask default: production-safe
ENV FLASK_DEBUG=0

EXPOSE 5000

CMD ["python", "app.py"]
