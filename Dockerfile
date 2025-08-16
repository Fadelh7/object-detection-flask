FROM python:3.11-slim

# Minimal system libs for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

# Install CPU-only Torch/Torchvision first to avoid CUDA wheels, then the rest
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1+cpu torchvision==0.18.1+cpu && \
    pip install --no-cache-dir --prefer-binary \
      ultralytics==8.* \
      opencv-python-headless \
      pillow \
      requests \
      flask==3.* \
      gunicorn==23.*

# Copy application code
COPY app.py ./
COPY models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/
COPY images/ ./images/

# Ensure runtime dirs exist
RUN mkdir -p uploads static/outputs

# Pre-download YOLOv8n model to avoid runtime downloads
RUN python -c "import os, urllib.request; \
    model_path = 'models/yolov8n.pt'; \
    print('Checking for model files...'); \
    (not os.path.exists('models/best.pt') and not os.path.exists(model_path)) and \
    (print('Pre-downloading YOLOv8n model for faster startup...'), \
     urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt', model_path), \
     print(f'Model cached at {model_path}')) or print('Model already available')"

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "--worker-class", "sync", "--max-requests", "1000", "--max-requests-jitter", "50", "--preload", "app:app"]


