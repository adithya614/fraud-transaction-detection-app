"""
Trains a Gradient Boosting fraud classifier on transactions.csv and saves
the fitted pipeline + a background sample for SHAP to model.joblib / bg_sample.joblib.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("transactions.csv")

NUM_FEATURES = ["amount", "hour_of_day", "distance_from_home_km",
                 "account_age_days", "txns_last_1h", "avg_txn_amount_30d",
                 "is_new_device", "is_new_merchant"]
CAT_FEATURES = ["merchant_category", "payment_mode", "device_type"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "is_fraud"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", clf),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

joblib.dump(pipeline, "model.joblib")
joblib.dump(X_train.sample(100, random_state=42), "bg_sample.joblib")
joblib.dump(FEATURES, "features.joblib")
joblib.dump(NUM_FEATURES, "num_features.joblib")
joblib.dump(CAT_FEATURES, "cat_features.joblib")
print("Saved model.joblib, bg_sample.joblib, features.joblib")
