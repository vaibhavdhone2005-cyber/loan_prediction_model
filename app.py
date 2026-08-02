import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "loan_prediction_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please place the model file in the same directory.")

# Embedded HTML/CSS UI for self-contained, lightweight deployment
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Loan Approval Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1E3A8A;
            --primary-hover: #1D4ED8;
            --bg-main: #F8FAFC;
            --surface: #FFFFFF;
            --text-main: #0F172A;
            --text-muted: #64748B;
            --border: #E2E8F0;
            --success: #10B981;
            --danger: #EF4444;
            --accent: #3B82F6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-main);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Executive Header */
        header {
            background-color: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-logo {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.1rem;
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.025em;
        }

        .badge {
            background-color: #EFF6FF;
            color: var(--accent);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            border: 1px solid #BFDBFE;
        }

        /* Main Container Layout */
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            width: 100%;
            display: grid;
            grid-template-columns: 1.8fr 1.2fr;
            gap: 2rem;
            flex: 1;
        }

        @media (max-width: 992px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        /* Card Styles */
        .card {
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        .card-header {
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }

        .card-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .card-subtitle {
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* Form Controls */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        @media (max-width: 600px) {
            .form-group.full-width {
                grid-column: span 1;
            }
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-main);
        }

        input, select {
            padding: 0.65rem 0.85rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.9rem;
            color: var(--text-main);
            background-color: #FAFAFA;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            background-color: var(--surface);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }

        .btn-submit {
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            padding: 0.85rem 1.5rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            width: 100%;
            font-size: 0.95rem;
            margin-top: 1.5rem;
            transition: background-color 0.2s ease, transform 0.1s ease;
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        /* Result Section UI */
        .result-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            height: 100%;
            min-height: 300px;
        }

        .placeholder-state {
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.75rem;
        }

        .placeholder-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background-color: #F1F5F9;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
        }

        .result-box {
            display: none;
            width: 100%;
            animation: fadeIn 0.4s ease-in-out forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.25rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 1.5rem;
            letter-spacing: 0.025em;
        }

        .status-badge.approved {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }

        .status-badge.rejected {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }

        .metric-card {
            background-color: #F8FAFC;
            border-radius: 8px;
            padding: 1.25rem;
            border: 1px solid var(--border);
            margin-bottom: 1rem;
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-main);
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            background-color: var(--surface);
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-logo">AI</div>
            <div class="brand-title">FinRisk Analytics</div>
        </div>
        <div class="badge">XGBoost Powered Model</div>
    </header>

    <main class="container">
        <!-- Input Form Section -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Applicant Assessment Form</h2>
                <p class="card-subtitle">Input financial metrics to run decision logic.</p>
            </div>

            <form id="loanForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="education">Education Level</label>
                        <select id="education" name="education" required>
                            <option value="1">Graduate</option>
                            <option value="0">Not Graduate</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="self_employed">Employment Type</label>
                        <select id="self_employed" name="self_employed" required>
                            <option value="0">Salaried / Corporate</option>
                            <option value="1">Self Employed</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="income_annum">Annual Income ($)</label>
                        <input type="number" id="income_annum" name="income_annum" value="6500000" placeholder="e.g. 5000000" required>
                    </div>

                    <div class="form-group">
                        <label for="loan_amount">Requested Loan ($)</label>
                        <input type="number" id="loan_amount" name="loan_amount" value="15000000" placeholder="e.g. 12000000" required>
                    </div>

                    <div class="form-group">
                        <label for="loan_term">Loan Tenure (Years)</label>
                        <input type="number" id="loan_term" name="loan_term" value="12" placeholder="e.g. 10" required>
                    </div>

                    <div class="form-group">
                        <label for="cibil_score">CIBIL Score (300-900)</label>
                        <input type="number" id="cibil_score" name="cibil_score" value="750" min="300" max="900" placeholder="e.g. 750" required>
                    </div>

                    <div class="form-group">
                        <label for="residential_assets_value">Residential Assets ($)</label>
                        <input type="number" id="residential_assets_value" name="residential_assets_value" value="4000000" required>
                    </div>

                    <div class="form-group">
                        <label for="commercial_assets_value">Commercial Assets ($)</label>
                        <input type="number" id="commercial_assets_value" name="commercial_assets_value" value="2500000" required>
                    </div>

                    <div class="form-group">
                        <label for="luxury_assets_value">Luxury Assets ($)</label>
                        <input type="number" id="luxury_assets_value" name="luxury_assets_value" value="8000000" required>
                    </div>

                    <div class="form-group">
                        <label for="bank_asset_value">Bank Assets ($)</label>
                        <input type="number" id="bank_asset_value" name="bank_asset_value" value="5000000" required>
                    </div>
                </div>

                <button type="submit" class="btn-submit">Evaluate Application</button>
            </form>
        </div>

        <!-- Result Section -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Risk Analysis Result</h2>
                <p class="card-subtitle">Real-time model prediction outcome.</p>
            </div>

            <div class="result-container">
                <div id="placeholder" class="placeholder-state">
                    <div class="placeholder-icon">
                        <svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                            <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                        </svg>
                    </div>
                    <p>Enter details and click "Evaluate Application" to assess eligibility.</p>
                </div>

                <div id="resultBox" class="result-box">
                    <div id="statusBadge" class="status-badge">
                        <span id="statusText">APPROVED</span>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Approval Confidence Rate</div>
                        <div class="metric-value" id="probabilityVal">0%</div>
                    </div>

                    <p id="resultDescription" style="font-size: 0.875rem; color: var(--text-muted); line-height: 1.5;">
                        The model predicted approval based on high creditworthiness and solid asset-to-loan ratio coverage.
                    </p>
                </div>
            </div>
        </div>
    </main>

    <footer>
        &copy; FinRisk Analytics Engine. Enterprise Decision Platform.
    </footer>

    <script>
        document.getElementById('loanForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.status === 'success') {
                    document.getElementById('placeholder').style.display = 'none';
                    const resultBox = document.getElementById('resultBox');
                    const statusBadge = document.getElementById('statusBadge');
                    const statusText = document.getElementById('statusText');
                    const probabilityVal = document.getElementById('probabilityVal');
                    const resultDescription = document.getElementById('resultDescription');

                    resultBox.style.display = 'block';
                    probabilityVal.innerText = (result.probability * 100).toFixed(1) + '%';

                    if (result.prediction === 1) {
                        statusBadge.className = 'status-badge approved';
                        statusText.innerText = '✓ LOAN APPROVED';
                        resultDescription.innerText = 'Applicant satisfies the credit risk threshold based on the asset base, CIBIL score, and debt service ratio.';
                    } else {
                        statusBadge.className = 'status-badge rejected';
                        statusText.innerText = '✕ LOAN REJECTED';
                        resultDescription.innerText = 'Applicant exhibits high default risk due to insufficient asset coverage relative to the requested loan size or credit history.';
                    }
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (err) {
                alert('An error occurred while connecting to the server.');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model file not found or failed to load.'})

    try:
        data = request.get_json()
        
        # Expected feature order based on model specifications
        feature_order = [
            'education', 'self_employed', 'income_annum', 'loan_amount',
            'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ]

        features = [data.get(feat, 0) for feat in feature_order]
        input_data = np.array([features])

        # Model Inference
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction])

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability': confidence
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
