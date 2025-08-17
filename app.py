from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import cv2
import os, uuid
from collections import Counter

app = Flask(_name_, static_folder="static")

# Load YOLO model
MODEL_PATH = os.path.join(os.path.dirname(_file_), "yolov8s.pt")
model = YOLO(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # Open image
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((640, 640))

    # Run YOLO prediction
    results = model.predict(image, verbose=False)

    # Annotate image
    annotated = results[0].plot()  # numpy array (BGR)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(annotated_rgb)

    # Save annotated image to static folder
    filename = f"result_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.static_folder, filename)
    im_pil.save(filepath, format="JPEG", quality=70)

    # Collect labels
    all_labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
    counts = Counter(all_labels)
    text = ", ".join([f"{v} {k}{'s' if v>1 else ''}" for k,v in counts.items()]) or "No objects detected"

    # Return static image URL
    return jsonify({"image_url": f"/static/{filename}", "text": text})

# Run Flask on Render
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
