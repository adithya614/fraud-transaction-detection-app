"""
Generates a synthetic Indian financial transaction dataset with realistic
fraud patterns for training the fraud detection model.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 25000
FRAUD_RATE = 0.03

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Coimbatore", "Erode",
          "Hyderabad", "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
MERCHANT_CATS = ["Grocery", "Electronics", "Fuel", "Travel", "Food Delivery",
                  "Utility Bill", "Online Shopping", "ATM Withdrawal",
                  "Jewellery", "Pharmacy", "Entertainment", "Wallet Transfer"]
PAYMENT_MODES = ["UPI", "Debit Card", "Credit Card", "Net Banking", "Wallet"]
DEVICES = ["Android", "iOS", "Web"]

rows = []
for i in range(N):
    is_fraud = 1 if np.random.rand() < FRAUD_RATE else 0

    hour = np.random.randint(0, 24)
    if is_fraud and np.random.rand() < 0.55:
        hour = np.random.choice([0, 1, 2, 3, 4, 23])

    amount = np.random.gamma(2.2, 1200)
    if is_fraud and np.random.rand() < 0.5:
        amount *= np.random.uniform(3, 12)
    amount = round(min(amount, 500000), 2)

    home_city = np.random.choice(CITIES)
    txn_city = home_city if np.random.rand() > 0.15 else np.random.choice(CITIES)
    if is_fraud and np.random.rand() < 0.4:
        txn_city = np.random.choice([c for c in CITIES if c != home_city])

    distance_km = 0 if txn_city == home_city else np.random.uniform(50, 2000)

    merchant_cat = np.random.choice(MERCHANT_CATS)
    if is_fraud and np.random.rand() < 0.35:
        merchant_cat = np.random.choice(["ATM Withdrawal", "Wallet Transfer",
                                          "Online Shopping", "Jewellery"])

    payment_mode = np.random.choice(PAYMENT_MODES)
    device = np.random.choice(DEVICES)

    account_age_days = int(np.random.exponential(600))
    if is_fraud and np.random.rand() < 0.4:
        account_age_days = int(np.random.uniform(0, 20))

    txns_last_1h = np.random.poisson(0.3)
    if is_fraud and np.random.rand() < 0.5:
        txns_last_1h += np.random.randint(3, 10)

    avg_txn_amount_30d = round(max(amount * np.random.uniform(0.2, 1.0), 100), 2)
    is_new_device = 1 if (np.random.rand() < 0.08 or (is_fraud and np.random.rand() < 0.45)) else 0
    is_new_merchant = 1 if (np.random.rand() < 0.3 or (is_fraud and np.random.rand() < 0.5)) else 0

    rows.append({
        "transaction_id": f"TXN{100000 + i}",
        "amount": amount,
        "hour_of_day": hour,
        "home_city": home_city,
        "transaction_city": txn_city,
        "distance_from_home_km": round(distance_km, 1),
        "merchant_category": merchant_cat,
        "payment_mode": payment_mode,
        "device_type": device,
        "account_age_days": account_age_days,
        "txns_last_1h": txns_last_1h,
        "avg_txn_amount_30d": avg_txn_amount_30d,
        "is_new_device": is_new_device,
        "is_new_merchant": is_new_merchant,
        "is_fraud": is_fraud,
    })

df = pd.DataFrame(rows)
df.to_csv("transactions.csv", index=False)
print(f"Generated {len(df)} rows, fraud rate = {df['is_fraud'].mean():.3%}")
print(df.head())
