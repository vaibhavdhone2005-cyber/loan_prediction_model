import streamlit as st
import numpy as np
import pickle
import xgboost as xgb

# Set Page Config for Business Presentation
st.set_page_config(
    page_title="Executive Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Executive UI
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 25px;
    }
    .header-title {
        color: #00F2FE;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .header-subtitle {
        color: #E0E0E0;
        font-size: 1.1rem;
    }

    /* Cards Layout */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 10px;
    }

    /* Predict Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
    }
    
    /* Input Box Labels */
    label {
        color: #4FACFE !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Function
@st.cache_resource
def load_model():
    with open('loan_predication_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'loan_predication_model.pkl' is in the root directory.")

# UI Header
st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 Smart Loan Risk Evaluator</div>
    <div class="header-subtitle">Enterprise AI-Driven Assessment Engine</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.image("https://img.icons8.com/isometric-headers/100/4a90e2/bank.png", width=80)
st.sidebar.title("Navigation & Overview")
st.sidebar.info("""
**System Highlights:**
- **Model:** XGBoost Classifier
- **Inference Time:** Real-time (<50ms)
- **Features Analyzed:** 10 Economic Indicators
""")

# Form Layout
st.subheader("📋 Borrower Profile & Financial Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Details")
    education = st.selectbox("Education Level", options=["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Employment Type", options=["Salaried", "Self Employed"])
    income_annum = st.number_input("Annual Income ($)", min_value=0, value=50000, step=5000)
    cibil_score = st.slider("CIBIL Credit Score", min_value=300, max_value=900, value=750)

with col2:
    st.markdown("### 💵 Loan & Asset Valuation")
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=150000, step=10000)
    loan_term = st.number_input("Loan Duration (Years)", min_value=1, max_value=30, value=10)
    residential_assets = st.number_input("Residential Asset Value ($)", min_value=0, value=100000, step=5000)
    commercial_assets = st.number_input("Commercial Asset Value ($)", min_value=0, value=50000, step=5000)
    luxury_assets = st.number_input("Luxury Asset Value ($)", min_value=0, value=20000, step=5000)
    bank_asset = st.number_input("Bank Liquid Deposits ($)", min_value=0, value=30000, step=5000)

# Preprocessing Inputs based on XGBoost feature order from pickle
edu_val = 1 if education == "Graduate" else 0
emp_val = 1 if self_employed == "Self Employed" else 0

input_features = np.array([[
    edu_val,
    emp_val,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_asset
]])

st.markdown("---")

# Predict Trigger Button
if st.button("🚀 Evaluate Loan Application"):
    prediction = model.predict(input_features)[0]
    
    # Calculate probability if available
    try:
        proba = model.predict_proba(input_features)[0]
        approval_prob = proba[1] if len(proba) > 1 else proba[0]
    except Exception:
        approval_prob = None

    # Result Section
    st.markdown("---")
    st.markdown("## 📊 Evaluation Results")
    
    res_col1, res_col2 = st.columns([1.5, 1])
    
    with res_col1:
        if prediction == 1 or prediction == " Approved":
            st.success("### ✅ LOAN APPROVED")
            st.write("The applicant satisfies the creditworthiness parameters based on assets, CIBIL score, and debt-to-income ratio.")
        else:
            st.error("### ❌ LOAN REJECTED")
            st.write("The applicant presents a higher default risk under current financial parameters.")

    with res_col2:
        if approval_prob is not None:
            st.metric(label="Approval Probability", value=f"{approval_prob*100:.1f}%")
            st.progress(float(approval_prob))
