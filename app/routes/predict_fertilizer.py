from flask import Blueprint, request, jsonify
import pandas as pd

from ..services.model_registry import (
    get_label_encoders, get_fertilizer_bundle, get_treatment_map_lower
)

fert_bp = Blueprint("fertilizer", __name__)

def _get_treatment(key: str):
    return get_treatment_map_lower().get(key.lower())

@fert_bp.route("/predict", methods=["POST"])
def predict_fertilizer_and_treatment():
    data = request.get_json(silent=True) or {}
    required = ["Nitrogen","Phosphorus","Potassium","pH","Rainfall",
                "Temperature","Soil_color","Crop","Disease"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing keys: {', '.join(missing)}"}), 400

    le_soil_color, le_crop = get_label_encoders()
    fert_model, _ = get_fertilizer_bundle()

    try:
        new_df = pd.DataFrame({
            "Nitrogen":     [data["Nitrogen"]],
            "Phosphorus":   [data["Phosphorus"]],
            "Potassium":    [data["Potassium"]],
            "pH":           [data["pH"]],
            "Rainfall":     [data["Rainfall"]],
            "Temperature":  [data["Temperature"]],
            "Soil_color":   [le_soil_color.transform([data["Soil_color"]])[0]],
            "Crop":         [le_crop.transform([data["Crop"]])[0]],
        })

        pred_fert = fert_model.predict(new_df)[0]
        treatment  = _get_treatment(data["Disease"])

        return jsonify({
            "predicted_fertilizer": str(pred_fert),
            "treatment_suggestion": treatment
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
