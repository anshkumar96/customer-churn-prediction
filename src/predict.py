"""
predict.py
-----------
Small demo of loading the persisted pipeline and scoring a brand-new
customer record. This is the same logic app.py uses under the hood.

Run:
    python src/predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"


def predict_single(customer: dict) -> dict:
    pipeline = joblib.load(MODEL_PATH)
    row = pd.DataFrame([customer])
    row["avg_monthly_spend"] = row["TotalCharges"] / row["tenure"].replace(0, 1)

    proba = pipeline.predict_proba(row)[0, 1]
    label = "Yes" if proba >= 0.5 else "No"
    return {"churn_prediction": label, "churn_probability": round(float(proba), 4)}


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 286.5,
    }
    result = predict_single(sample_customer)
    print("Sample customer prediction:", result)
