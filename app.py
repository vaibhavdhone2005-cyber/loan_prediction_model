import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Trained XGBoost Loan Model
MODEL_PATH = "loan_predication_model.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model file not found or corrupted."}), 500

    try:
        # Extract inputs from form
        data = request.form

        # Categorical features conversion
        # Education: 1 for Graduate, 0 for Not Graduate
        education = int(data.get("education", 1))
        # Self Employed: 1 for Yes, 0 for No
        self_employed = int(data.get("self_employed", 0))

        # Numeric features extraction
        income_annum = float(data.get("income_annum", 0))
        loan_amount = float(data.get("loan_amount", 0))
        loan_term = float(data.get("loan_term", 0))
        cibil_score = float(data.get("cibil_score", 0))
        residential_assets_value = float(data.get("residential_assets_value", 0))
        commercial_assets_value = float(data.get("commercial_assets_value", 0))
        luxury_assets_value = float(data.get("luxury_assets_value", 0))
        bank_asset_value = float(data.get("bank_asset_value", 0))

        # Construct input array in exact feature order expected by XGBoost model
        feature_names = [
            "education", "self_employed", "income_annum", "loan_amount",
            "loan_term", "cibil_score", "residential_assets_value",
            "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
        ]

        input_df = pd.DataFrame([[
            education, self_employed, income_annum, loan_amount,
            loan_term, cibil_score, residential_assets_value,
            commercial_assets_value, luxury_assets_value, bank_asset_value
        ]], columns=feature_names)

        # Prediction logic
        prediction = model.predict(input_df)[0]
        
        # Get probability score if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1] * 100
        else:
            probability = 100.0 if prediction == 1 else 0.0

        # Model output logic (Assuming 1 = Approved, 0 = Rejected)
        is_approved = bool(prediction == 1)
        
        return jsonify({
            "status": "success",
            "approved": is_approved,
            "probability": round(probability, 2),
            "cibil_score": cibil_score
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
