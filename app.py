from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io, base64, cv2, os
from collections import Counter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Max upload size: 5MB

# Load YOLO model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s.pt")
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
        # Open image and resize
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((640, 640))

        # Run YOLO prediction
        results = model.predict(image, verbose=False)

        # Annotated image
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(annotated_rgb)

        # Convert to base64
        buffer = io.BytesIO()
        im_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Collect labels
        all_labels = [model.names[int(box.cls[0])] for box in results[0].boxes]
        counts = Counter(all_labels)
        text = ", ".join([f"{v} {k}{'s' if v>1 else ''}" for k, v in counts.items()]) or "No objects detected"

        return jsonify({"image": img_str, "text": text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Render entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
