Urban Building Line Predictor
This project is a Full-Stack Machine Learning application designed to predict the Building Line (Setback) for urban planning. It uses a two-step AI approach: predicting importance weights for urban features and then calculating a final AI-adjusted building line.

🚀 Features
Dual Neural Network Models: One model predicts the optimal weights for urban factors, and a second model provides an AI-corrected final prediction.

FastAPI Backend: A robust REST API to handle prediction requests.

Interactive Web UI: Two front-end interfaces:

index.html: Standard prediction based on model-derived weights.


generate.html: Scenario testing where users can manually assign weights.

📁 Project Structure
app.py: The FastAPI server script.

train_building_line_model.py: Script to train the final correction model.

train_building_line_weight_model.py: Script to train the feature weighting model.

Train-Dataset-CMC-with-Weights.csv: The dataset containing road names, traffic volumes, and building lines.

models/: Contains the saved .keras models and the scaler.pkl file.

🛠️ Installation & Setup
1. Install Dependencies
Ensure you have Python installed, then run:

Bash

pip install -r requirements.txt
2. Start the API Server
Run the FastAPI backend using Uvicorn:

Bash

uvicorn app:app --reload
3. Launch the UI
Simply open index.html in any web browser to start predicting.

📊 How it Works
The model takes 5 key urban inputs:

Road Width (m)

Traffic Volume

Functional Mix (Live, Work, Visit, etc.)

Building Density (High, Moderate, Low)

Connectivity

It first calculates the Weighted Overlay and then applies an AI-adjusted correction to ensure accuracy based on historical urban data.