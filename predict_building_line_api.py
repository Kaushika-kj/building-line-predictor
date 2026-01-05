# ============================================
# app.py
# REST API for building line prediction
# ============================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------
# Load models and scaler
# --------------------------------------------
model_weights = load_model("building_line_weight_model.keras")
model_building_line = load_model("building_line_final_model.keras")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------
# Define input data model
# --------------------------------------------
class BuildingLineInput(BaseModel):
    road_width: float
    traffic_volume: float
    functional_mix: float
    building_density: float
    connectivity: float

class ManualWeights(BaseModel):
    road_width: float
    traffic_volume: float
    functional_mix: float
    building_density: float
    connectivity: float

class BuildingLineWithWeightInput(BaseModel):
    road_width: float
    traffic_volume: float
    functional_mix: float
    building_density: float
    connectivity: float
    weights: ManualWeights


# --------------------------------------------
# FastAPI instance
# --------------------------------------------
app = FastAPI(title="Building Line Prediction API")

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, allows all origins. Replace "*" with your domain in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# --------------------------------------------
# Prediction function
# --------------------------------------------
def predict_building_line(road_width, traffic_volume, functional_mix, building_density, connectivity):
    X_new = np.array([[road_width, traffic_volume, functional_mix, building_density, connectivity]])
    X_scaled = scaler.transform(X_new)

    # Predict weights
    weights_pred = model_weights.predict(X_scaled)[0]
    weight_names = ['road_width_weight', 'traffic_volume_weight',
                    'functional_mix_weight', 'building_density_weight', 'connectivity_weight']
    weights = dict(zip(weight_names, weights_pred))

    # Normalize weights
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    # Weighted overlay
    weighted_overlay = (
            weights["road_width_weight"] * road_width +
            weights["traffic_volume_weight"] * traffic_volume +
            weights["functional_mix_weight"] * functional_mix +
            weights["building_density_weight"] * building_density +
            weights["connectivity_weight"] * connectivity
    )

    # Final AI-adjusted building line
    X_combined = np.concatenate([X_scaled, np.array([list(weights.values())])], axis=1)
    building_line_final = model_building_line.predict(X_combined)[0][0]

    # Convert numpy types to native Python types
    weights = {k: float(v) for k, v in weights.items()}
    weighted_overlay = float(weighted_overlay)
    building_line_final = float(building_line_final)

    return {
        "weights": weights,
        "weighted_overlay": weighted_overlay,
        "building_line_final": building_line_final
    }

# --------------------------------------------
# Prediction function
# --------------------------------------------
def predict_building_line_with_weight(
        road_width, traffic_volume, functional_mix,
        building_density, connectivity, weights
):
    X_new = np.array([[road_width, traffic_volume,
                       functional_mix, building_density, connectivity]])
    X_scaled = scaler.transform(X_new)

    # Normalize manual weights
    weight_dict = {
        "road_width_weight": weights.road_width,
        "traffic_volume_weight": weights.traffic_volume,
        "functional_mix_weight": weights.functional_mix,
        "building_density_weight": weights.building_density,
        "connectivity_weight": weights.connectivity,
    }

    total = sum(weight_dict.values())
    for k in weight_dict:
        weight_dict[k] /= total

    # Weighted overlay
    weighted_overlay = (
            weight_dict["road_width_weight"] * road_width +
            weight_dict["traffic_volume_weight"] * traffic_volume +
            weight_dict["functional_mix_weight"] * functional_mix +
            weight_dict["building_density_weight"] * building_density +
            weight_dict["connectivity_weight"] * connectivity
    )

    # AI-adjusted final prediction
    X_combined = np.concatenate(
        [X_scaled, np.array([list(weight_dict.values())])],
        axis=1
    )

    building_line_final = model_building_line.predict(X_combined)[0][0]

    return {
        "weights": {k: float(v) for k, v in weight_dict.items()},
        "weighted_overlay": float(weighted_overlay),
        "building_line_final": float(building_line_final)
    }


# --------------------------------------------
# API endpoint
# --------------------------------------------
@app.post("/predict")
def predict(input_data: BuildingLineInput):
    result = predict_building_line(
        input_data.road_width,
        input_data.traffic_volume,
        input_data.functional_mix,
        input_data.building_density,
        input_data.connectivity
    )
    return result

@app.post("/predictWithWeight")
def predictWithWeight(input_data: BuildingLineWithWeightInput):
    return predict_building_line_with_weight(
        input_data.road_width,
        input_data.traffic_volume,
        input_data.functional_mix,
        input_data.building_density,
        input_data.connectivity,
        input_data.weights
    )

# --------------------------------------------
# Run with:
# uvicorn app:app --reload
# --------------------------------------------
