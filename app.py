import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "loan_prediction_model.pkl"
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    
    if request.method == "POST":
        try:
            # Extract inputs matching the exact feature order of your XGBoost model:
            # [education, self_employed, income_annum, loan_amount, loan_term,
            #  cibil_score, residential_assets_value, commercial_assets_value,
            #  luxury_assets_value, bank_asset_value]
            
            education = int(request.form.get("education", 0))
            self_employed = int(request.form.get("self_employed", 0))
            income_annum = float(request.form.get("income_annum", 0))
            loan_amount = float(request.form.get("loan_amount", 0))
            loan_term = float(request.form.get("loan_term", 0))
            cibil_score = float(request.form.get("cibil_score", 0))
            residential_assets = float(request.form.get("residential_assets_value", 0))
            commercial_assets = float(request.form.get("commercial_assets_value", 0))
            luxury_assets = float(request.form.get("luxury_assets_value", 0))
            bank_asset = float(request.form.get("bank_asset_value", 0))

            features = np.array([[
                education, self_employed, income_annum, loan_amount,
                loan_term, cibil_score, residential_assets,
                commercial_assets, luxury_assets, bank_asset
            ]])

            # Predict probability if model supports it, else raw prediction
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features)[0][1]
            else:
                prob = float(model.predict(features)[0])

            status = "APPROVED" if prob >= 0.5 else "REJECTED"
            confidence = round(prob * 100, 1) if status == "APPROVED" else round((1 - prob) * 100, 1)

            prediction_result = {
                "status": status,
                "confidence": confidence,
                "probability": round(prob * 100, 1)
            }

        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template("index.html", result=prediction_result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
