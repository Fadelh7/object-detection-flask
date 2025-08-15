import sys
import json
from ultralytics import YOLO

def predict(model_path, image_path, conf=0.25):
    model = YOLO(model_path)
    names = model.names
    res = model.predict(source=image_path, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return {"label": "no_fruit_detected", "counts": {}, "detections": []}
    boxes = res.boxes
    cls = boxes.cls.cpu().numpy().astype(int).tolist()
    confs = boxes.conf.cpu().numpy().tolist()
    xyxys = boxes.xyxy.cpu().numpy().astype(int).tolist() # absolute pixel coordinates

    # Count occurrences of each class
    class_counts = {}
    for class_id in cls:
        class_name = names.get(class_id, f"unknown_{class_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Format detections
    detections = []
    for i in range(len(cls)):
        detections.append({
            "class_id": cls[i],
            "class_name": names.get(cls[i], f"unknown_{cls[i]}"),
            "confidence": confs[i],
            "bbox": xyxys[i]
        })

    # Determine dominant label
    dominant_label = "multiple_fruits"
    if len(class_counts) == 1:
        dominant_label = list(class_counts.keys())[0]
    elif len(class_counts) == 0:
        dominant_label = "no_fruit_detected"

    return {
        "label": dominant_label,
        "counts": class_counts,
        "detections": detections
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Example usage (you can modify this for your specific needs)
    results = predict(model_path, image_path)
    print(json.dumps(results, indent=2))
