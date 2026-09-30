# SecureTxn AI — Real-Time Fraud Detection

A client-ready Streamlit application for detecting fraudulent transactions
in real time, built on a Gradient Boosting model trained on a synthetic
Indian transaction dataset, with SHAP-based explainability.

## Features

- **Live Monitor** — simulates a real-time transaction stream, scores each
  transaction instantly, flags Low/Medium/High risk, and raises alerts for
  high-risk transactions. Includes a "Simulate next transaction" button and
  an "Auto-stream" toggle for continuous live scoring.
- **Check a Transaction** — a form where a client/agent enters a single
  transaction's details and gets an instant fraud probability, a clear
  verdict (Approve / Step-up verification / Hold), and a plain-language
  breakdown of which factors drove the score (SHAP).
- **Analytics** — portfolio-level dashboards: fraud rate by hour, by
  merchant category, by payment mode, amount distributions, and global
  feature importance.

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python generate_data.py

# 2. Train the model (creates model.joblib, bg_sample.joblib, etc.)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Creates `transactions.csv`, a 25,000-row synthetic Indian transaction dataset with realistic fraud patterns (odd hours, distance from home, new devices, velocity, etc.) |
| `train_model.py` | Trains a Gradient Boosting classifier in a scikit-learn pipeline (scaling + one-hot encoding) and saves it, along with a SHAP background sample |
| `app.py` | The Streamlit application (3 pages: Live Monitor, Check a Transaction, Analytics) |
| `requirements.txt` | Python dependencies |

## Model performance (on held-out 20% test set)

- ROC-AUC: **0.970**
- Precision (fraud class): **0.83**
- Recall (fraud class): **0.61**

## Swapping in real data

Replace `transactions.csv` with your real transaction log using the same
column names (see `generate_data.py` for the schema), then re-run
`train_model.py`. No changes to `app.py` are needed as long as the column
names match.

## Notes for going to production

This app simulates "real time" by generating a new transaction on each
click / auto-refresh cycle — ideal for demos and client walkthroughs. For
a production deployment, replace `random_transaction()` in `app.py` with a
call to your live transaction stream (e.g. a Kafka consumer, a webhook, or
a polling call to your payments API) and keep the scoring/UI logic as is.
