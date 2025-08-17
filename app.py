import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from collections import Counter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure YOLO config directory works on Render
os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# Use lightweight YOLO model for faster CPU inference
MODEL_PATH = "yolov8n.pt"  # use nano model
model = YOLO(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Open image with PIL
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((640, 640))  # resize for YOLO input

        # Run YOLO prediction
        results = model.predict(image, verbose=False)

        # Annotated image
        annotated = results[0].plot()  # numpy array (BGR)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(annotated_rgb)

        # Convert to base64 string
        buffer = io.BytesIO()
        im_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Get detected labels
        all_labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
        counts = Counter(all_labels)
        text = ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]) or "No objects detected"

        return jsonify({"image": img_str, "text": text})

    except Exception as e:
        # This helps debug in Render logs
        print("Error during prediction:", e, flush=True)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
