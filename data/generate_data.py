"""
generate_data.py
-----------------
Generates a realistic, synthetic telecom customer-churn dataset.

Why synthetic data: it lets the whole pipeline run end-to-end, offline,
and reproducibly (fixed random seed) without depending on any external
download. The feature set and churn-logic mirror the well-known
IBM Telco Customer Churn dataset schema, so anyone familiar with that
domain will recognize the columns immediately.

Run:
    python data/generate_data.py
Produces:
    data/customer_churn.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 3000

rng = np.random.default_rng(RANDOM_SEED)


def generate_dataset(n=N_CUSTOMERS):
    customer_id = [f"CUST-{10000 + i}" for i in range(n)]

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, size=n)  # months, 0-72

    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def dependent_internet_feature(p_yes):
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes]),
        )

    online_security = dependent_internet_feature(0.29)
    online_backup = dependent_internet_feature(0.35)
    device_protection = dependent_internet_feature(0.34)
    tech_support = dependent_internet_feature(0.29)
    streaming_tv = dependent_internet_feature(0.38)
    streaming_movies = dependent_internet_feature(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # --- Charges: correlated with services chosen ---
    base_charge = rng.normal(20, 3, size=n)
    internet_charge = np.select(
        [internet_service == "DSL", internet_service == "Fiber optic", internet_service == "No"],
        [rng.normal(25, 4, size=n), rng.normal(45, 6, size=n), 0],
    )
    addon_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    addon_charge = addon_count * rng.normal(5, 1, size=n)

    monthly_charges = np.clip(base_charge + internet_charge + addon_charge, 18, 120).round(2)
    total_charges = np.clip(monthly_charges * tenure + rng.normal(0, 20, size=n), 0, None).round(2)

    # --- Churn probability logic (the "ground truth" signal) ---
    logit = (
        -1.8
        + 1.6 * (contract == "Month-to-month")
        + 0.5 * (contract == "One year")
        - 0.03 * tenure
        + 0.02 * (monthly_charges - 60)
        + 0.5 * (internet_service == "Fiber optic")
        - 0.5 * (online_security == "Yes")
        - 0.5 * (tech_support == "Yes")
        + 0.3 * (senior_citizen == 1)
        + 0.25 * (paperless_billing == "Yes")
        + 0.35 * (payment_method == "Electronic check")
        - 0.3 * (partner == "Yes")
        - 0.2 * (dependents == "Yes")
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob_churn).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/customer_churn.csv", index=False)
    print(f"Saved data/customer_churn.csv with shape {df.shape}")
    print(df["Churn"].value_counts(normalize=True))
