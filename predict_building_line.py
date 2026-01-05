# ============================================
# predict_building_line.py
# Use the trained AI models to predict
#  - four weights
#  - final building line (m)
# ============================================

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------------------
# Load trained models and scaler
# --------------------------------------------
# Make sure these files exist in the same folder:
#  - building_line_weight_model.h5
#  - building_line_final_model.h5
#  - scaler.pkl
model_weights = load_model("building_line_weight_model.keras")
model_building_line = load_model("building_line_final_model.keras")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------
# Function to predict weights and building line
# --------------------------------------------
def predict_building_line(road_width, traffic_volume, building_density, functional_mix, connectivity):
    # Create input array
    X_new = np.array([[road_width, traffic_volume, building_density, functional_mix, connectivity]])

    # Scale inputs
    X_scaled = scaler.transform(X_new)

    # Predict weights
    weights_pred = model_weights.predict(X_scaled)[0]
    weight_names = ['road_width_weight', 'traffic_volume_weight',
                     'functional_mix_weight', 'building_density_weight', 'connectivity_weight']
    weights = dict(zip(weight_names, weights_pred))

    # Normalize weights so they sum to 1
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    # Weighted overlay building line estimation
    weighted_overlay = (
            weights["road_width_weight"] * road_width +
            weights["traffic_volume_weight"] * traffic_volume +
            weights["functional_mix_weight"] * building_density +
            weights["building_density_weight"] * functional_mix +
            weights["connectivity_weight"] * connectivity
    )

    # Final building line (AI correction model)
    X_combined = np.concatenate([X_scaled, np.array([list(weights.values())])], axis=1)
    building_line_final = model_building_line.predict(X_combined)[0][0]

    # Output results
    print("\nPredicted Weights:")
    for name, val in weights.items():
        print(f"  {name}: {val:.3f}")

    print(f"\nWeighted overlay building line (raw): {weighted_overlay:.2f} m")
    print(f"Final AI-adjusted building line: {building_line_final:.2f} m")

    return weights, weighted_overlay, building_line_final

# --------------------------------------------
# Example usage
# --------------------------------------------
if __name__ == "__main__":
    # Example input (replace with actual values)
    road_width = 19.6
    traffic_volume = 26087
    building_density = 7
    functional_mix = 1
    connectivity = 5

    predict_building_line(road_width, traffic_volume, building_density, functional_mix, connectivity)
