# 📉 Customer Churn Prediction & Analytics Dashboard

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn, built with a full pipeline — data generation, feature engineering, model selection with hyperparameter tuning, evaluation, and an interactive Streamlit deployment.

## Problem Statement

Customer churn (customers leaving a subscription service) directly hits recurring revenue. Acquiring a new customer costs far more than retaining an existing one, so being able to flag **at-risk customers early** lets a business intervene (discounts, support outreach, contract upgrades) before losing them.

This project builds a binary classifier that outputs a **churn probability** for each customer, using demographic, account, and service-usage attributes.

## Dataset

The dataset schema mirrors the well-known **IBM Telco Customer Churn** dataset (21 columns: demographics, account info, services subscribed, charges, and the `Churn` target). To keep the project fully self-contained and reproducible without external downloads, `data/generate_data.py` synthesizes 3,000 customer records with a realistic churn-probability model — churn risk increases with month-to-month contracts, high monthly charges, low tenure, fiber-optic internet, and electronic-check payment, and decreases with add-on services like online security/tech support, mirroring patterns well documented in real churn analyses. Swap in the real IBM dataset CSV (same column names) and the rest of the pipeline works unchanged.

- **Rows:** 3,000 customers
- **Target:** `Churn` (Yes/No) — ~17% positive class (realistic imbalance)
- **Features:** 19 (demographics, contract/billing details, subscribed services, charges)

## Tech Stack

| Layer | Tools |
|---|---|
| Data & features | Python, Pandas, NumPy |
| Modeling | scikit-learn (Logistic Regression, Random Forest), XGBoost (falls back to Gradient Boosting if unavailable) |
| Tuning & evaluation | GridSearchCV (5-fold, ROC-AUC scoring), classification report, ROC curves, confusion matrices |
| Deployment | Streamlit (interactive prediction dashboard) |
| Visualization | Matplotlib |

## Project Structure

```
churn-prediction/
├── data/
│   ├── generate_data.py      # synthetic dataset generator
│   └── customer_churn.csv    # generated dataset (created after running the script)
├── src/
│   ├── preprocessing.py      # cleaning, feature engineering, sklearn ColumnTransformer
│   ├── train_models.py       # trains + tunes 3 models, saves best pipeline + reports
│   └── predict.py            # CLI demo: load saved model, score a new customer
├── models/
│   └── best_model.pkl        # persisted best pipeline (created after training)
├── reports/                  # confusion matrices, ROC curves, feature importance, metrics table
├── app.py                    # Streamlit dashboard (predict + model insights)
├── requirements.txt
└── README.md
```

## Methodology

1. **Data cleaning & feature engineering** (`src/preprocessing.py`) — handles missing/blank `TotalCharges`, engineers an `avg_monthly_spend` signal, and wraps numeric scaling + one-hot encoding in a single `ColumnTransformer` so training and inference use *identical* transformations.
2. **Model selection** (`src/train_models.py`) — trains three candidate algorithms inside one `Pipeline` (preprocessing + classifier), each tuned with `GridSearchCV` (5-fold cross-validation, optimizing ROC-AUC since the target is imbalanced). Class imbalance is handled via `class_weight="balanced"` for Logistic Regression and Random Forest.
3. **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC on a held-out 20% test set; confusion matrices and ROC curves saved to `reports/`.
4. **Deployment** — the best pipeline (by ROC-AUC) is persisted with `joblib` and served through a Streamlit app with a live prediction form and a model-insights tab.

## Results

*(from the bundled synthetic dataset — regenerate to get fresh numbers, or swap in the real Telco dataset)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.69 | 0.32 | 0.73 | 0.45 | **0.77** |
| Gradient Boosting | 0.83 | 0.47 | 0.08 | 0.13 | 0.75 |
| Random Forest | 0.72 | 0.31 | 0.54 | 0.40 | 0.74 |

Logistic Regression wins on ROC-AUC and recall — important here, since missing an actual churner (false negative) is costlier than a false alarm for a retention team. See `reports/roc_curve_comparison.png` and `reports/feature_importance.png` for visuals.

## How to Run

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd churn-prediction
pip install -r requirements.txt

# 2. Generate the dataset
python data/generate_data.py

# 3. Train and evaluate all models (saves the best one to models/)
python src/train_models.py

# 4. Try a quick CLI prediction
python src/predict.py

# 5. Launch the interactive dashboard
streamlit run app.py
```

## Future Improvements

- Swap synthetic data for the real IBM Telco dataset for a production-realistic benchmark
- Add SHAP-based explainability to the dashboard so each prediction shows *why* (per-feature contribution)
- Handle imbalance with SMOTE (`imbalanced-learn`) as an alternative to class weighting
- Track experiments with MLflow and containerize the app with Docker for deployment

## Author

Ansh Kumar — B.Tech CSE (Data Science), Galgotias University
[LinkedIn](https://www.linkedin.com/in/ansh-kumar-614aa7243) · [GitHub](https://github.com/anshkumar96) · [LeetCode](https://leetcode.com/u/ansh_kumar96/)
