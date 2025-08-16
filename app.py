import os
import uuid
from typing import Dict, Any, List, Tuple

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)
from werkzeug.utils import secure_filename

import numpy as np

from ultralytics import YOLO


def ensure_dirs() -> None:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static/outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Place best.pt in models/ or set YOLO_MODEL_PATH."
        )
    print(f"Loading model from {model_path}...")
    
    # Load model with minimal memory usage
    import os
    os.environ['PYTORCH_JIT'] = '0'  # Disable JIT compilation to save memory
    
    model = YOLO(model_path)
    
    # Skip warming for memory efficiency on free tier
    print("Model loaded successfully (skipping warmup for memory efficiency)")
    
    return model


def top_label_from_results(res, names: Dict[int, str]) -> Tuple[str, Dict[str, int]]:
    if res.boxes is None or len(res.boxes) == 0:
        return "no_fruit_detected", {}

    boxes = res.boxes
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    confs = boxes.conf.cpu().numpy().tolist()

    # Count per class
    counts: Dict[str, int] = {}
    for c in cls:
        cname = names.get(c, str(c))
        counts[cname] = counts.get(cname, 0) + 1

    # Single decision: highest-confidence detection
    top_idx = int(np.argmax(confs))
    top_cls = cls[top_idx]
    label = names.get(top_cls, str(top_cls))
    return label, counts


def run_inference(
    model: YOLO,
    image_path: str,
    conf: float = 0.25,
    save_annotated: bool = True,
) -> Dict[str, Any]:
    # Ultra-lightweight inference settings for memory-constrained environment
    res = model.predict(
        source=image_path, 
        conf=conf, 
        verbose=False,
        device='cpu',  # Explicitly use CPU
        half=False,    # Don't use half precision on CPU
        imgsz=320,     # Smaller image size for faster processing and less memory
        max_det=10,    # Limit detections to save memory
        agnostic_nms=True,  # Faster NMS
    )[0]
    
    label, counts = top_label_from_results(res, model.names)

    annotated_rel = None
    if save_annotated:
        try:
            annotated = res.plot()  # BGR image
            out_name = f"{uuid.uuid4().hex}_pred.jpg"
            out_path = os.path.join("static", "outputs", out_name)

            # Write using cv2 without importing it globally
            import cv2

            # Optimize image compression for faster saving and smaller files
            cv2.imwrite(out_path, annotated, [
                cv2.IMWRITE_JPEG_QUALITY, 75,  # Lower quality for smaller size
                cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimize encoding
            ])
            annotated_rel = f"/static/outputs/{out_name}"
            
            # Clean up memory immediately
            del annotated
        except Exception as e:
            print(f"Failed to save annotated image: {e}")
            annotated_rel = None

    # Detections as JSON-friendly objects
    detections: List[Dict[str, Any]] = []
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        cls = boxes.cls.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
            detections.append(
                {
                    "class_id": c,
                    "class_name": model.names.get(c, str(c)),
                    "confidence": float(p),
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    # Clean up memory
    del res
    
    return {
        "label": label,  # "banana" or "watermelon" (or "no_fruit_detected")
        "counts": counts,
        "annotated_image": annotated_rel,
        "detections": detections,
    }


def create_app() -> Flask:
    ensure_dirs()
    model_path = os.environ.get("YOLO_MODEL_PATH", "models/best.pt")
    
    # If YOLO_MODEL_PATH is a URL, download it into models/ so the loader can use a local file.
    def _ensure_local_model(path_or_url: str) -> str:
        if not path_or_url:
            return path_or_url
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            # Download to models/<basename>
            import urllib.request
            from urllib.parse import urlparse

            parsed = urlparse(path_or_url)
            fname = os.path.basename(parsed.path) or "best.pt"
            dest = os.path.join("models", fname)
            if not os.path.exists(dest):
                print(f"Downloading model from {path_or_url} to {dest}...")
                try:
                    urllib.request.urlretrieve(path_or_url, dest)
                except Exception as e:
                    raise RuntimeError(f"Failed to download model from {path_or_url}: {e}")
            else:
                print(f"Model already exists at {dest}, skipping download.")
            return dest
        return path_or_url

    # If the model file doesn't exist, use YOLO's built-in model download
    def _get_default_model(path: str) -> str:
        if os.path.exists(path):
            return path
        
        # If custom model doesn't exist, fall back to YOLOv8n
        print(f"Model file {path} not found. Using YOLOv8n as fallback...")
        fallback_path = "models/yolov8n.pt"
        
        # Check if we already have it cached
        if os.path.exists(fallback_path):
            print(f"Using cached YOLOv8n model at {fallback_path}")
            return fallback_path
        
        # Download YOLOv8n directly to our models directory to avoid repeated downloads
        print("Downloading YOLOv8n model...")
        try:
            import urllib.request
            yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
            print(f"Downloading from {yolo_url}...")
            urllib.request.urlretrieve(yolo_url, fallback_path)
            print(f"Downloaded and cached model to {fallback_path}")
        except Exception as e:
            print(f"Failed to download model directly: {e}")
            # Fallback to ultralytics auto-download
            print("Falling back to ultralytics auto-download...")
            return 'yolov8n.pt'
        
        return fallback_path

    model_path = _ensure_local_model(model_path)
    model_path = _get_default_model(model_path)
    conf = float(os.environ.get("CONF_THRES", "0.25"))

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # Reduce to 8MB for memory efficiency
    
    # Store model path but don't load until first request (lazy loading)
    app.model_path = model_path
    app.model = None
    app.conf_thres = conf
    
    def get_model():
        """Lazy load the model on first request"""
        if app.model is None:
            print("First request - loading model...")
            app.model = load_model(app.model_path)
            print("Model loaded successfully!")
        return app.model

    @app.route('/images/<path:filename>')
    def images(filename):
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
        return send_from_directory(images_dir, filename)

    @app.route("/health", methods=["GET"])
    def health() -> Tuple[str, int]:
        return "ok", 200

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            image_file = request.files.get("image")
            image_url = request.form.get("image_url", "").strip()

            if not image_file and not image_url:
                return render_template("index.html", error="Please upload an image or provide a URL.")


            if image_file and image_file.filename != "":
                fname = secure_filename(image_file.filename)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    return render_template("index.html", error="Unsupported file type. Use JPG/PNG/BMP/WEBP.")
                save_name = f"{uuid.uuid4().hex}{ext}"
                save_path = os.path.join("uploads", save_name)
                image_file.save(save_path)
                image_path = save_path
            elif image_url:
                if image_url.startswith("data:image/"):
                    import re, base64
                    match = re.match(r"data:image/(\w+);base64,(.+)", image_url)
                    if not match:
                        return render_template("index.html", error="Invalid base64 image data.")
                    ext = match.group(1)
                    b64data = match.group(2)
                    try:
                        img_bytes = base64.b64decode(b64data)
                        ext_map = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "bmp": ".bmp", "webp": ".webp"}
                        ext = ext_map.get(ext.lower(), ".jpg")
                        save_name = f"{uuid.uuid4().hex}{ext}"
                        save_path = os.path.join("uploads", save_name)
                        with open(save_path, "wb") as f:
                            f.write(img_bytes)
                        image_path = save_path
                    except Exception as e:
                        return render_template("index.html", error=f"Failed to decode base64 image: {e}")
                elif image_url.startswith("/images/"):
                    # Local static image, just use the path directly
                    image_path = os.path.join("images", os.path.basename(image_url))
                else:
                    import requests
                    try:
                        resp = requests.get(image_url, timeout=10)
                        resp.raise_for_status()
                        ext = os.path.splitext(image_url)[1].lower()
                        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                            return render_template("index.html", error="Unsupported file type in URL. Use JPG/PNG/BMP/WEBP.")
                        save_name = f"{uuid.uuid4().hex}{ext}"
                        save_path = os.path.join("uploads", save_name)
                        with open(save_path, "wb") as f:
                            f.write(resp.content)
                        image_path = save_path
                    except Exception as e:
                        return render_template("index.html", error=f"Failed to fetch image from URL: {e}")
            else:
                return render_template("index.html", error="No image provided.")

            result = run_inference(
                model=get_model(),
                image_path=image_path,
                conf=app.conf_thres,
                save_annotated=True,
            )
            return render_template("index.html", result=result)

        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No 'image' file in form-data."}), 400

        f = request.files["image"]
        if f.filename == "":
            return jsonify({"error": "No file selected."}), 400

        fname = secure_filename(f.filename)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            return jsonify({"error": "Unsupported file type."}), 415

        save_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join("uploads", save_name)
        f.save(save_path)

        result = run_inference(
            model=get_model(),
            image_path=save_path,
            conf=app.conf_thres,
            save_annotated=True,
        )
        return jsonify(result), 200

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
