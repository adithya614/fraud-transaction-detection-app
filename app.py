import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# --------------------------------------------------------------------------
# Page config & style
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SecureTxn AI — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.big-title {font-size: 2rem; font-weight: 700; margin-bottom: 0;}
.subtitle {color: #6b7280; margin-top: 0;}
.risk-high {background-color:#fee2e2; color:#991b1b; padding:4px 10px;
    border-radius:999px; font-weight:600; font-size:0.85rem;}
.risk-medium {background-color:#fef3c7; color:#92400e; padding:4px 10px;
    border-radius:999px; font-weight:600; font-size:0.85rem;}
.risk-low {background-color:#dcfce7; color:#166534; padding:4px 10px;
    border-radius:999px; font-weight:600; font-size:0.85rem;}
.kpi-card {background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px;
    padding:16px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Load artifacts (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    bg_sample = joblib.load("bg_sample.joblib")
    features = joblib.load("features.joblib")
    num_features = joblib.load("num_features.joblib")
    cat_features = joblib.load("cat_features.joblib")
    return model, bg_sample, features, num_features, cat_features


@st.cache_resource
def load_history():
    return pd.read_csv("transactions.csv")


@st.cache_resource
def make_explainer(_model, _bg_sample):
    preproc = _model.named_steps["preprocessor"]
    clf = _model.named_steps["classifier"]
    bg_transformed = preproc.transform(_bg_sample)
    return shap.TreeExplainer(clf, bg_transformed)


model, bg_sample, FEATURES, NUM_FEATURES, CAT_FEATURES = load_artifacts()
history_df = load_history()
explainer = make_explainer(model, bg_sample)

CITIES = sorted(history_df["home_city"].unique())
MERCHANT_CATS = sorted(history_df["merchant_category"].unique())
PAYMENT_MODES = sorted(history_df["payment_mode"].unique())
DEVICES = sorted(history_df["device_type"].unique())


def risk_bucket(prob):
    if prob >= 0.7:
        return "High", "risk-high"
    elif prob >= 0.3:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


def score_transaction(row_dict):
    row = pd.DataFrame([row_dict])[FEATURES]
    prob = model.predict_proba(row)[0, 1]
    preproc = model.named_steps["preprocessor"]
    transformed = preproc.transform(row)
    shap_values = explainer.shap_values(transformed)
    cat_encoder = preproc.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CAT_FEATURES))
    all_names = NUM_FEATURES + cat_names
    contrib = pd.Series(shap_values[0], index=all_names).sort_values(
        key=abs, ascending=False
    )
    return prob, contrib


def random_transaction(fraud_bias=False):
    is_fraud_like = fraud_bias and np.random.rand() < 0.5
    hour = np.random.choice([0, 1, 2, 3, 23]) if is_fraud_like else np.random.randint(0, 24)
    amount = np.random.gamma(2.2, 1200)
    if is_fraud_like:
        amount *= np.random.uniform(3, 10)
    amount = round(min(amount, 500000), 2)
    home = np.random.choice(CITIES)
    dist = 0.0
    txn_city = home
    if is_fraud_like and np.random.rand() < 0.5:
        txn_city = np.random.choice([c for c in CITIES if c != home])
        dist = round(np.random.uniform(50, 2000), 1)
    return {
        "amount": amount,
        "hour_of_day": int(hour),
        "distance_from_home_km": dist,
        "account_age_days": int(np.random.uniform(0, 20)) if is_fraud_like else int(np.random.exponential(600)),
        "txns_last_1h": int(np.random.randint(3, 10)) if is_fraud_like and np.random.rand() < 0.5 else int(np.random.poisson(0.3)),
        "avg_txn_amount_30d": round(max(amount * np.random.uniform(0.2, 1.0), 100), 2),
        "is_new_device": 1 if (is_fraud_like and np.random.rand() < 0.45) else int(np.random.rand() < 0.08),
        "is_new_merchant": 1 if (is_fraud_like and np.random.rand() < 0.5) else int(np.random.rand() < 0.3),
        "merchant_category": np.random.choice(MERCHANT_CATS),
        "payment_mode": np.random.choice(PAYMENT_MODES),
        "device_type": np.random.choice(DEVICES),
        "home_city": home,
        "transaction_city": txn_city,
    }


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "feed" not in st.session_state:
    st.session_state.feed = []
if "live_on" not in st.session_state:
    st.session_state.live_on = False

# --------------------------------------------------------------------------
# Header + nav
# --------------------------------------------------------------------------
col_logo, col_nav = st.columns([2, 3])
with col_logo:
    st.markdown('<p class="big-title">🛡️ SecureTxn AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time fraud detection for payment transactions</p>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["📡 Live Monitor", "🔍 Check a Transaction", "📊 Analytics"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: Gradient Boosting · Trained on Indian transaction data\n\n"
    "Explainability: SHAP"
)

# --------------------------------------------------------------------------
# PAGE 1 — Live Monitor
# --------------------------------------------------------------------------
if page == "📡 Live Monitor":
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        start = st.button("▶ Simulate next transaction", use_container_width=True)
    with c2:
        st.session_state.live_on = st.toggle("Auto-stream", value=st.session_state.live_on)
    with c3:
        bias = st.slider("Suspicious-activity likelihood (demo control)", 0, 100, 15) / 100

    if start:
        txn = random_transaction(fraud_bias=(np.random.rand() < bias))
        prob, contrib = score_transaction(txn)
        label, css = risk_bucket(prob)
        txn.update({
            "time": datetime.now().strftime("%H:%M:%S"),
            "fraud_probability": round(prob * 100, 1),
            "risk": label,
        })
        st.session_state.feed.insert(0, txn)
        st.session_state.feed = st.session_state.feed[:200]

    feed = st.session_state.feed

    k1, k2, k3, k4 = st.columns(4)
    total = len(feed)
    flagged = sum(1 for t in feed if t["risk"] in ("High", "Medium"))
    high = sum(1 for t in feed if t["risk"] == "High")
    avg_prob = round(np.mean([t["fraud_probability"] for t in feed]), 1) if feed else 0
    for col, label, val in zip(
        [k1, k2, k3, k4],
        ["Transactions scored", "Flagged (med+high)", "High risk", "Avg fraud score"],
        [total, flagged, high, f"{avg_prob}%"],
    ):
        col.markdown(f'<div class="kpi-card"><div style="font-size:1.6rem;font-weight:700;">{val}</div>'
                      f'<div style="color:#6b7280;font-size:0.85rem;">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("### Live transaction feed")
    if not feed:
        st.info("Click **Simulate next transaction** or turn on **Auto-stream** to see live scoring.")
    else:
        show_cols = ["time", "amount", "merchant_category", "payment_mode",
                     "transaction_city", "fraud_probability", "risk"]
        disp = pd.DataFrame(feed)[show_cols].rename(columns={
            "time": "Time", "amount": "Amount (₹)", "merchant_category": "Category",
            "payment_mode": "Mode", "transaction_city": "City",
            "fraud_probability": "Fraud Score (%)", "risk": "Risk",
        })

        def highlight(row):
            color = {"High": "#fee2e2", "Medium": "#fef3c7", "Low": "#dcfce7"}[row["Risk"]]
            return [f"background-color: {color}"] * len(row)

        st.dataframe(disp.style.apply(highlight, axis=1), use_container_width=True, height=420)

        high_risk_recent = [t for t in feed[:5] if t["risk"] == "High"]
        for t in high_risk_recent:
            st.error(
                f"⚠️ High-risk transaction at {t['time']}: ₹{t['amount']:,.0f} via "
                f"{t['payment_mode']} in {t['transaction_city']} — {t['fraud_probability']}% fraud score"
            )

        csv = pd.DataFrame(feed).to_csv(index=False).encode()
        st.download_button("⬇ Download feed as CSV", csv, "transaction_feed.csv", "text/csv")

    if st.session_state.live_on:
        time.sleep(2)
        txn = random_transaction(fraud_bias=(np.random.rand() < bias))
        prob, contrib = score_transaction(txn)
        label, css = risk_bucket(prob)
        txn.update({
            "time": datetime.now().strftime("%H:%M:%S"),
            "fraud_probability": round(prob * 100, 1),
            "risk": label,
        })
        st.session_state.feed.insert(0, txn)
        st.session_state.feed = st.session_state.feed[:200]
        st.rerun()

# --------------------------------------------------------------------------
# PAGE 2 — Manual Check
# --------------------------------------------------------------------------
elif page == "🔍 Check a Transaction":
    st.markdown("### Check a single transaction")
    st.caption("Enter transaction details below to get an instant fraud risk assessment.")

    with st.form("manual_check"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Transaction amount (₹)", min_value=1.0, value=2500.0, step=100.0)
            hour = st.slider("Hour of day", 0, 23, 14)
            home_city = st.selectbox("Customer's home city", CITIES)
        with c2:
            txn_city = st.selectbox("Transaction city", CITIES, index=CITIES.index(home_city))
            merchant_cat = st.selectbox("Merchant category", MERCHANT_CATS)
            payment_mode = st.selectbox("Payment mode", PAYMENT_MODES)
        with c3:
            device = st.selectbox("Device type", DEVICES)
            account_age = st.number_input("Account age (days)", min_value=0, value=400)
            avg_amount = st.number_input("Customer's avg. transaction (₹, last 30 days)", min_value=1.0, value=1800.0)

        c4, c5, c6 = st.columns(3)
        with c4:
            txns_1h = st.number_input("Transactions by this customer in last 1 hour", min_value=0, value=0)
        with c5:
            new_device = st.checkbox("First time on this device?")
        with c6:
            new_merchant = st.checkbox("First transaction with this merchant?")

        submitted = st.form_submit_button("🔎 Assess risk", use_container_width=True)

    if submitted:
        dist = 0.0 if txn_city == home_city else np.random.uniform(50, 2000)
        row = {
            "amount": amount, "hour_of_day": hour, "distance_from_home_km": round(dist, 1),
            "account_age_days": account_age, "txns_last_1h": txns_1h,
            "avg_txn_amount_30d": avg_amount, "is_new_device": int(new_device),
            "is_new_merchant": int(new_merchant), "merchant_category": merchant_cat,
            "payment_mode": payment_mode, "device_type": device,
            "home_city": home_city, "transaction_city": txn_city,
        }
        prob, contrib = score_transaction(row)
        label, css = risk_bucket(prob)

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"#### Verdict: <span class='{css}'>{label} risk</span>", unsafe_allow_html=True)
            st.metric("Fraud probability", f"{prob*100:.1f}%")
            if label == "High":
                st.error("Recommend: **Hold / manually verify** this transaction before approving.")
            elif label == "Medium":
                st.warning("Recommend: **Step-up verification** (OTP / call-back) before approving.")
            else:
                st.success("Recommend: **Approve** — low fraud indicators.")

        with r2:
            top = contrib.head(6)
            readable = {
                "amount": "Transaction amount", "hour_of_day": "Hour of day",
                "distance_from_home_km": "Distance from home", "account_age_days": "Account age",
                "txns_last_1h": "Transactions in last hour", "avg_txn_amount_30d": "Avg. 30-day amount",
                "is_new_device": "New device", "is_new_merchant": "New merchant",
            }
            labels = [readable.get(n, n) for n in top.index]
            fig = go.Figure(go.Bar(
                x=top.values, y=labels, orientation="h",
                marker_color=["#dc2626" if v > 0 else "#16a34a" for v in top.values],
            ))
            fig.update_layout(
                title="What drove this score (red = raises risk, green = lowers risk)",
                height=320, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Impact on fraud score",
            )
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 3 — Analytics
# --------------------------------------------------------------------------
else:
    st.markdown("### Portfolio analytics (historical training data)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total transactions", f"{len(history_df):,}")
    k2.metric("Fraud rate", f"{history_df['is_fraud'].mean()*100:.2f}%")
    k3.metric("Avg. transaction", f"₹{history_df['amount'].mean():,.0f}")
    k4.metric("Model ROC-AUC", "0.970")

    c1, c2 = st.columns(2)
    with c1:
        by_hour = history_df.groupby("hour_of_day")["is_fraud"].mean().reset_index()
        fig1 = px.bar(by_hour, x="hour_of_day", y="is_fraud",
                       title="Fraud rate by hour of day",
                       labels={"hour_of_day": "Hour", "is_fraud": "Fraud rate"})
        fig1.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        by_cat = history_df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(by_cat, x="is_fraud", y="merchant_category", orientation="h",
                       title="Fraud rate by merchant category",
                       labels={"is_fraud": "Fraud rate", "merchant_category": ""})
        fig2.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(history_df, x="amount", color="is_fraud", nbins=60,
                             barmode="overlay", title="Amount distribution: fraud vs. legit",
                             labels={"amount": "Amount (₹)", "is_fraud": "Fraud"})
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        by_mode = history_df.groupby("payment_mode")["is_fraud"].mean().reset_index()
        fig4 = px.pie(by_mode, names="payment_mode", values="is_fraud",
                       title="Share of fraud by payment mode")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Global feature importance")
    clf = model.named_steps["classifier"]
    preproc = model.named_steps["preprocessor"]
    cat_names = list(preproc.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES))
    all_names = NUM_FEATURES + cat_names
    imp = pd.Series(clf.feature_importances_, index=all_names).sort_values(ascending=False).head(12)
    fig5 = px.bar(imp[::-1], orientation="h", labels={"value": "Importance", "index": ""})
    fig5.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig5, use_container_width=True)
