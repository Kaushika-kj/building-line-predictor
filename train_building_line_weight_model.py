import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
data = pd.read_csv("Train-Dataset-CMC-with-Weights.csv")

# Define input (X) and output (y)
X = data[['road_width', 'traffic_volume', 'functional_mix', 'building_density', 'connectivity']]
y = data[['road_width_weight', 'traffic_volume_weight',
          'functional_mix_weight', 'building_density_weight', 'connectivity_weight']]

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(5, activation='linear')  # 4 outputs for 4 weights
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate model
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test MAE: {mae:.4f}")

# Predict
predictions = model.predict(X_test_scaled)
pred_df = pd.DataFrame(predictions, columns=y.columns)
print(pred_df.head())

model.save("building_line_weight_model.keras")

model.save("building_line_weight_model.h5")
