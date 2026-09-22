
from pathlib import Path
import json
import uuid

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_pneumonia.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
UPLOAD_DIR = BASE_DIR / "uploads"

IMG_SIZE = (224, 224)
PRIMARY_THRESHOLD = 0.50
MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_DIR.mkdir(exist_ok=True)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"0": "NORMAL", "1": "PNEUMONIA"}


CLASS_NAMES = load_class_names()


# Load once when the server starts.
# This avoids loading the ~model on every request.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Copy best_pneumonia.keras into the model/ directory."
    )

model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(file_path):
    image = Image.open(file_path).convert("RGB")
    display_image = image.copy()

    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    return array, display_image


def predict_image(file_path, threshold=PRIMARY_THRESHOLD):
    array, _ = prepare_image(file_path)

    probability = float(
        model.predict(array, verbose=0).ravel()[0]
    )

    prediction = (
        CLASS_NAMES["1"]
        if probability >= threshold
        else CLASS_NAMES["0"]
    )

    confidence = (
        probability
        if prediction == CLASS_NAMES["1"]
        else 1.0 - probability
    )

    return {
        "prediction": prediction,
        "pneumonia_probability": probability,
        "normal_probability": 1.0 - probability,
        "confidence": confidence,
        "threshold": threshold,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics={
            "accuracy": "87.98%",
            "sensitivity": "96.41%",
            "specificity": "73.93%",
            "f1": "90.93%",
            "roc_auc": "95.52%",
            "pr_auc": "97.08%",
        },
    )


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No X-ray image was uploaded."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Please select an X-ray image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Use PNG, JPG, JPEG or WEBP."
        }), 400

    # A unique filename prevents collisions and avoids trusting user paths.
    extension = Path(secure_filename(file.filename)).suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / filename

    try:
        file.save(file_path)
        result = predict_image(file_path)

        # The frontend needs the uploaded image for preview only.
        result["image_url"] = f"/uploads/{filename}"

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": f"Could not process this image: {exc}"
        }), 500


@app.get("/uploads/<filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({
        "error": "File is too large. Maximum size is 10 MB."
    }), 413


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "EfficientNetB0",
        "input_size": "224x224",
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
