from functools import lru_cache
from tensorflow.keras.models import load_model
import joblib

@lru_cache(maxsize=1)
def get_cnn_model():
    # compile=False to speed up load; you only need predict
    return load_model("models/mobilenetv2.keras", compile=False)

@lru_cache(maxsize=1)
def get_label_encoders():
    enc = joblib.load("models/label_encoder.pkl")
    # Expecting {'le_soil_color': ..., 'le_crop': ...}
    return enc["le_soil_color"], enc["le_crop"]

@lru_cache(maxsize=1)
def get_fertilizer_bundle():
    bundle = joblib.load("models/model_and_disease.pkl")
    # Expecting {'model': ..., 'crop_disease_list': ...}
    return bundle["model"], bundle["crop_disease_list"]

def get_treatment_map_lower():
    _, crop_disease_list = get_fertilizer_bundle()
    return {k.lower(): v for k, v in dict(crop_disease_list).items()}
