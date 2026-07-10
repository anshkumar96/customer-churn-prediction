"""
app.py
-------
Interactive Streamlit dashboard for the Customer Churn Prediction project.

Two tabs:
  1. Predict — a form where a user enters a customer's attributes and
     gets a churn probability + risk label, powered by the same
     preprocessing pipeline used in training (no train/serve skew).
  2. Model Insights — shows the saved model comparison table, ROC
     curves, confusion matrix, and feature-importance chart produced
     by train_models.py.

Run locally:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

MODEL_PATH = "models/best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def prepare_input(data: dict) -> pd.DataFrame:
    row = pd.DataFrame([data])
    row["avg_monthly_spend"] = row["TotalCharges"] / row["tenure"].replace(0, 1)
    return row


st.title("📉 Customer Churn Prediction Dashboard")
st.caption(
    "End-to-end ML project: synthetic telecom data → feature engineering → "
    "model selection (Logistic Regression / Random Forest / XGBoost) → "
    "hyperparameter tuning → deployment."
)

tab_predict, tab_insights = st.tabs(["🔮 Predict Churn", "📊 Model Insights"])

with tab_predict:
    st.subheader("Enter customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=float(monthly_charges * max(tenure, 1))
    )

    if st.button("Predict Churn Risk", type="primary"):
        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        try:
            model = load_model()
            row = prepare_input(customer)
            proba = model.predict_proba(row)[0, 1]

            risk = "🔴 High" if proba >= 0.6 else ("🟡 Medium" if proba >= 0.3 else "🟢 Low")
            st.metric("Churn Probability", f"{proba*100:.1f}%")
            st.subheader(f"Risk Level: {risk}")
            st.progress(min(float(proba), 1.0))
        except FileNotFoundError:
            st.error("Model not found. Run `python src/train_models.py` first to train and save a model.")

with tab_insights:
    st.subheader("Model Comparison")
    try:
        comparison = pd.read_csv("reports/model_comparison.csv")
        st.dataframe(comparison, use_container_width=True)
    except FileNotFoundError:
        st.info("Run `python src/train_models.py` to generate the model comparison report.")

    col_a, col_b = st.columns(2)
    with col_a:
        try:
            st.image(Image.open("reports/roc_curve_comparison.png"), caption="ROC Curve Comparison")
        except FileNotFoundError:
            pass
    with col_b:
        try:
            st.image(Image.open("reports/feature_importance.png"), caption="Top Feature Importances")
        except FileNotFoundError:
            pass
