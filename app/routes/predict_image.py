from flask import Blueprint, request, jsonify
import numpy as np, io
from PIL import Image
from tensorflow.keras.preprocessing import image as kimage

from ..services.model_registry import get_cnn_model, get_treatment_map_lower
from ..utils.labels import CLASS_LABELS

image_bp = Blueprint("image", __name__)

def _norm(key: str) -> str:
    # normalize so it matches keys in your bundle (lowercase is usually enough)
    return key.lower().strip()

@image_bp.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))

    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, 0)

    model = get_cnn_model()
    preds = model.predict(arr)[0]

    best_idx = int(np.argmax(preds))
    best_class = CLASS_LABELS[best_idx]
    best_prob = float(preds[best_idx])

    # treatment lookup
    tmap = get_treatment_map_lower()
    best_treatment = tmap.get(_norm(best_class))  # dict or None

    # top-others (include treatment too)
    sorted_idx = np.argsort(preds)[::-1]
    alternatives = []
    for i in sorted_idx:
        if i == best_idx:
            continue
        cls = CLASS_LABELS[i]
        alternatives.append({
            "class": cls,
            "prob": float(preds[i]),
            "treatment": tmap.get(_norm(cls))  # may be None if not found
        })

    return jsonify({
        "prediction": best_class,
        "confidence": best_prob,
        "treatment": best_treatment,
        "alternatives": alternatives
    })
