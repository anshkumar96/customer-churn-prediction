"""
preprocessing.py
-----------------
Cleaning + feature engineering for the churn dataset.
Exposes `load_and_clean()` and `build_preprocessor()` used by train_models.py
and app.py so the exact same transformation is applied at train and
inference time (no train/serve skew).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges can arrive as blank strings for tenure==0 customers in the
    # real-world Telco dataset — coerce and impute defensively even though
    # our synthetic generator already emits numeric values.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    # Feature engineering: a couple of derived signals that tend to help
    # tree-based models separate churners from non-churners.
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_target_split(df: pd.DataFrame):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y
