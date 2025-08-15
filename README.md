# Banana vs Watermelon Detector (YOLOv8 + Flask + Docker)

A minimal web app and REST API that loads a YOLOv8 model trained on
bananas/watermelons and returns a single top label ("banana" or "watermelon"),
counts per class, and an annotated preview.

## 1. Prerequisites

- Python 3.10+ (recommended 3.11)
- Your trained weights `best.pt` from Colab
- Optionally Docker

## 2. Setup (Local)

1) Place `best.pt` into `models/best.pt`.

2) Create a venv and install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run:

```bash
export YOLO_MODEL_PATH=models/best.pt
export PORT=8080
python app.py
```

Open http://localhost:8080

## 3. REST API

- POST `/predict`
  - form-data: `image` (file)
  - Response JSON:
    ```json
    {
      "label": "banana",
      "counts": {"banana": 3, "watermelon": 1},
      "annotated_image": "/static/outputs/xxxx_pred.jpg",
      "detections": [
        {
          "class_id": 0,
          "class_name": "banana",
          "confidence": 0.91,
          "box_xyxy": [x1, y1, x2, y2]
        }
      ]
    }
    ```

Example:

```bash
curl -X POST http://localhost:8080/predict \
  -F image=@/path/to/image.jpg
```

## 4. Docker

Build:

```bash
docker build -t fruit-detector .
```

Run (mount weights from host):

```bash
docker run --rm -p 8080:8080 \
  -e YOLO_MODEL_PATH=/app/models/best.pt \
  -v $(pwd)/models:/app/models \
  fruit-detector
```

Open http://localhost:8080

## 5. Deploy (Render / Railway / Fly.io / Hugging Face Spaces)

- Push this repo to GitHub (do not commit large `*.pt` files).
- In the platform:
  - Build command: `pip install -r requirements.txt`
  - Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
  - Add environment variable `YOLO_MODEL_PATH` and upload/attach your
    `best.pt` (via persistent disk/volume, or a private object store).
- Ensure port `$PORT` is used (platform usually injects it).

## 6. Notes

- Do not commit `models/best.pt` to GitHub (large files).
- App chooses a single top label by the highest-confidence detection.
- If no detections, label is `no_fruit_detected`.

## 7. License

MIT (app code). Your dataset/model may have different licenses.
