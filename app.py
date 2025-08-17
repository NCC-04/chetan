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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure YOLO config directory works on Render
os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# Auto-download nano model (fastest for CPU)
model = YOLO("yolov8n.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Open + resize
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((640, 640))

        # Run YOLO
        results = model.predict(image, verbose=False)

        # Annotated image
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()
        im_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Labels
        all_labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
        counts = Counter(all_labels)
        text = ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]) or "No objects detected"

        return jsonify({"image": img_str, "text": text})

    except Exception as e:
        print("🔥 ERROR:", e, flush=True)  # This shows in Render logs
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
