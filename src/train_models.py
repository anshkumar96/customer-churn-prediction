"""
train_models.py
-----------------
End-to-end training script:
  1. Load + clean data, engineer features
  2. Train/test split (stratified, since churn is imbalanced ~17/83)
  3. Train 3 candidate models inside a single sklearn Pipeline
     (preprocessing + classifier) so there is zero train/serve skew:
        - Logistic Regression   (fast, interpretable baseline)
        - Random Forest          (non-linear, feature interactions)
        - Gradient Boosting      (xgboost if installed, else sklearn's
                                  GradientBoostingClassifier as a drop-in
                                  fallback so the script runs anywhere)
  4. Hyperparameter tuning via GridSearchCV (5-fold, scoring = ROC-AUC,
     the right metric for an imbalanced binary target)
  5. Evaluate every model on the held-out test set, save a comparison
     table + confusion matrices + ROC curves + feature importances
  6. Persist the best pipeline to models/best_model.pkl for app.py

Run:
    python src/train_models.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, get_feature_target_split, load_and_clean

warnings.filterwarnings("ignore")

DATA_PATH = "data/customer_churn.csv"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_candidate_models():
    """Returns {name: (estimator, param_grid)} — param grid keys are
    prefixed with 'clf__' since the estimator sits inside a Pipeline."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [6, 10, None],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        )
    else:
        models["Gradient Boosting"] = (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [150, 300],
                "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1],
            },
        )
    return models


def evaluate(name, pipeline, X_test, y_test, results, roc_ax):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    results.append(metrics)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    # ROC curve — plotted on the shared axis passed in
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=roc_ax)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


def main():
    print("Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    candidates = get_candidate_models()
    results = []
    fitted_pipelines = {}

    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))

    for name, (estimator, param_grid) in candidates.items():
        print(f"\nTuning {name} ...")
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("clf", estimator)]
        )
        grid = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")

        fitted_pipelines[name] = best_pipeline
        evaluate(name, best_pipeline, X_test, y_test, results, ax_roc)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_title("ROC Curve Comparison")
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(REPORTS_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close(fig_roc)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_name]
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")
    with open(MODELS_DIR / "best_model_info.json", "w") as f:
        json.dump(
            {"best_model": best_name, "metrics": results_df.iloc[0].to_dict()}, f, indent=2
        )
    print(f"\nBest model: {best_name} -> saved to models/best_model.pkl")

    # Feature importance — always shown from the Random Forest run (tree-based,
    # trained regardless of which model ends up "best") so the dashboard
    # always has an interpretability chart to display.
    importance_source = best_name if best_name in ("Random Forest", "XGBoost", "Gradient Boosting") else "Random Forest"
    imp_pipeline = fitted_pipelines[importance_source]
    clf = imp_pipeline.named_steps["clf"]
    preproc = imp_pipeline.named_steps["preprocessor"]
    feature_names = preproc.get_feature_names_out()
    importances = clf.feature_importances_
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#1F3864")
    ax.set_title(f"Top 15 Feature Importances — {importance_source}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved feature_importance.png")


if __name__ == "__main__":
    main()
