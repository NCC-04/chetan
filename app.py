from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io, base64
import cv2
from collections import Counter
import os

app = Flask(__name__)
os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", "/tmp/ultralytics")
# Load YOLO model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s.pt")
model = YOLO(MODEL_PATH)  # Pretrained YOLOv8s model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # Open image and convert to RGB
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((640, 640))  # Resize for YOLO input

    # Run YOLO prediction
    results = model.predict(image, verbose=False)

    # Annotate image in color
    annotated = results[0].plot()  # returns BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(annotated_rgb)

    # Convert annotated image to base64
    buffer = io.BytesIO()
    im_pil.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Collect detected labels
    all_labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
    counts = Counter(all_labels)
    text = ", ".join([f"{v} {k}{'s' if v>1 else ''}" for k,v in counts.items()]) or "No objects detected"

    return jsonify({"image": img_str, "text": text})

# Run Flask on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
