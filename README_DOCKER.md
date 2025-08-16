# Object Detection Flask App - Docker Setup

## Quick Start

Build and run the Flask object detection app with Docker:

```bash
# Build and run with docker-compose (recommended)
docker compose up --build

# Or build and run manually
docker build -t object-detection-app .
docker run -p 8080:8080 -v ${PWD}/uploads:/app/uploads -v ${PWD}/static/outputs:/app/static/outputs object-detection-app
```

## Access the App

- **Web UI**: http://localhost:8080
- **Health Check**: http://localhost:8080/health
- **API Endpoint**: POST to http://localhost:8080/predict with image file

## Docker Setup Details

### What's included:
- **Base**: `python:3.11-slim` (lightweight)
- **Dependencies**: CPU-only PyTorch + Ultralytics + OpenCV
- **Size**: ~2.5GB (much smaller than full ultralytics base image)
- **Ports**: 8080 (configurable via PORT env var)

### Environment Variables:
- `YOLO_MODEL_PATH`: Path to model weights (default: `models/best.pt`)
- `CONF_THRES`: Detection confidence threshold (default: `0.25`)
- `PORT`: Flask server port (default: `8080`)

### Volumes:
- `./uploads:/app/uploads` - Stores uploaded images
- `./static/outputs:/app/static/outputs` - Stores prediction results

## Development vs Production

The current setup uses CPU-only PyTorch for maximum compatibility and smaller image size. If you need GPU acceleration, you would need to:

1. Use a CUDA base image
2. Install GPU-enabled PyTorch
3. Add GPU runtime support to docker-compose

## Model Requirements

Ensure your YOLO model file `models/best.pt` is present before building. The app will load this model on startup.
