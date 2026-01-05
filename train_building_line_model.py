import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# -------------------------------------------------
# 1. Load dataset and trained weight model
# -------------------------------------------------
data = pd.read_csv("Train-Dataset-CMC-with-Weights.csv")

# Load your previously trained weight model
weight_model = load_model("building_line_weight_model.keras")

# Define input (X) and output (y)
X = data[['road_width', 'traffic_volume', 'functional_mix', 'building_density', 'connectivity']]
y_building_line = data[['building_line']]

# Scale input features (same as before)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 2. Predict weights using the existing model
# -------------------------------------------------
predicted_weights = weight_model.predict(X_scaled)
predicted_weights_df = pd.DataFrame(predicted_weights,
                                    columns=['road_width', 'traffic_volume', 'functional_mix',
                                             'building_density', 'connectivity'])

# Combine original features + predicted weights
X_combined = np.concatenate([X_scaled, predicted_weights], axis=1)

# -------------------------------------------------
# 3. Train building line model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_building_line, test_size=0.2, random_state=42)

model_building_line = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # 8 inputs
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output: building_line
])

model_building_line.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nTraining building line model...")
history = model_building_line.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate
loss, mae = model_building_line.evaluate(X_test, y_test, verbose=0)
print(f"Building Line Model MAE: {mae:.4f}")

# -------------------------------------------------
# 4. Save models and scaler
# -------------------------------------------------
model_building_line.save("building_line_final_model.keras")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Saved model as 'building_line_final_model.h5' and scaler as 'scaler.pkl'")
