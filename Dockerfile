# ─────────────────────────────────────────────────────────────────────────────
#  Border Surveillance AI — Dockerfile
#  Builds a portable container that runs the full pipeline.
#
#  Build:
#      docker build -t border-surveillance-ai .
#
#  Run pipeline:
#      docker run --rm \
#        -v $(pwd)/data:/app/data \
#        -v $(pwd)/models:/app/models \
#        -v $(pwd)/results:/app/results \
#        -v $(pwd)/config.yaml:/app/config.yaml \
#        border-surveillance-ai \
#        python src/run_pipeline.py --video data/sample/test_video.mp4
#
#  Run dashboard:
#      docker run --rm -p 8501:8501 \
#        -v $(pwd)/alerts.db:/app/alerts.db \
#        border-surveillance-ai \
#        streamlit run dashboard/streamlit_app.py --server.port 8501
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.9-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/          src/
COPY dashboard/    dashboard/
COPY models/       models/
COPY data/sample/  data/sample/

# Copy config template (user must mount real config.yaml at runtime)
COPY config.example.yaml config.example.yaml

# Create output directories
RUN mkdir -p results/metrics results/screenshots results/charts \
             data/processed

# Expose Streamlit port
EXPOSE 8501

# Default command — override at `docker run`
CMD ["python", "src/run_pipeline.py", "--help"]
