FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OpenCV/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default model path; override with -e YOLO_MODEL_PATH=/path/in/container.pt
ENV YOLO_MODEL_PATH=models/best.pt
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
