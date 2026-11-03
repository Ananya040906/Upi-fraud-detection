# UPI Fraud Detection

Real-time anomaly detection for UPI and credit/debit card transactions. It combines **behavioural rules** (impossible travel, sudden high-value spikes, transaction bursts) with an **Isolation Forest** model, and ships with a **Streamlit** app for live simulation and stream replay.

> **Live demo:** https://upi-fraud-detection-kjg6uibmrxpueearqjr7au.streamlit.app/

---

## ✨ Features

- **Real-time scoring** – each transaction is scored the moment it arrives, with a 0–100 risk score and human-readable reasons.
- **Per-user behavioural profile** – running average and spread of amounts, last known location and time, and a 1-hour transaction count.
- **Rule engine**
  -**Impossible travel** – implied speed above 900 km/h across more than 100 km
  -**Sudden high value** – amount at least 6× the user's average and 4+ standard deviations above it
  -**Velocity burst** – 4 or more transactions within one hour
- **Isolation Forest** – an unsupervised model that catches subtler outliers the rules miss; no fraud labels needed for training.
- **Profile protection** – flagged transactions do not update the user's profile, so fraud can't gradually look "normal".
- **Streamlit app** – a live transaction simulator and a stream replay with map, flagged table, precision and recall.
- **Synthetic data generator** – creates realistic users and injects three fraud patterns for testing.

---

## 🧠 How it works

```
Transaction ──► Feature engineering ──► Rules ───────┐
 (user, time,    (distance, speed,                   ├──► Flag + risk score + reasons
  amount, lat,    amount ratio, txn/hour,            │
  lon)            hour, log amount)  ──► Isolation ──┘
                                         Forest
                         ▲
                 Per-user state (updated only for non-flagged transactions)
```

Training and live scoring use the **same feature code**, so there is no train/serve mismatch.

| Feature | Meaning |
|---|---|
| `log_amount` | log(1 + amount) |
| `amount_ratio` | amount ÷ user's historical average |
| `distance_km` | haversine distance from the user's previous transaction |
| `speed_kmph` | implied travel speed since the previous transaction |
| `txn_1h` | transactions by the user in the last hour |
| `hour` | hour of day |

---

## 📁 Project structure

```
upi-fraud-detection/
├── app.py                 # Streamlit app
├── train.py               # Train, evaluate and save the detector
├── requirements.txt
├── src/
│   ├── detector.py        # FraudDetector, features, rules, Isolation Forest
│   └── data_gen.py        # Synthetic data with injected fraud
├── tests/
│   └── test_detector.py   # Unit tests
└── models/                # Saved model (created by train.py, git-ignored)
```

---

## 🚀 Quick start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Ananya040906/upi-fraud-detection.git
cd upi-fraud-detection

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
python train.py                    # trains and saves models/detector.joblib
python -m pytest                   # run tests
streamlit run app.py               # opens http://localhost:8501
```

If `models/detector.joblib` is missing, the app trains it automatically on first launch.

---

## 🧪 Try it in the app

**Simulator tab**

1. Select `user_000` (home city: Bengaluru), amount ₹1,000, city Bengaluru → ✅ normal.
2. Submit ₹1,000 in **Delhi** with a 5-minute gap → *Impossible travel*.
3. Submit ₹5,00,000 → *Sudden high value*.

**Stream replay tab** – generates a fresh stream with injected fraud and shows flagged transactions on a map, with precision and recall.

---

## 🧩 Use the detector in your own code

```python
import joblib, pandas as pd

detector = joblib.load("models/detector.joblib")

result = detector.process({
    "user_id": "user_000",
    "timestamp": pd.Timestamp("2026-03-01 12:00"),
    "amount": 1500.0,
    "lat": 12.97,
    "lon": 77.59,
})
print(result)
# {'risk_score': 12.4, 'flagged': False, 'reasons': [], 'features': {...}}
```

---

## ⚙️ Configuration

Edit `RULES` in `src/detector.py`:

```python
RULES = {
    "max_speed_kmph": 900,
    "min_distance_km": 100,
    "amount_ratio": 6.0,
    "amount_zscore": 4.0,
    "max_txn_per_hour": 4,
}
```

The Isolation Forest flag threshold is the `threshold` argument of `FraudDetector` (default `70`). Lower values flag more transactions (higher recall, lower precision).

---

## 📊 Results

On a held-out **synthetic** stream with injected fraud, the detector reached roughly **0.9 precision and 0.8 recall**. These numbers come from generated data and are only a sanity check, not a claim about real-world accuracy. Evaluate on real, labelled data (with a time-based split) before relying on it.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repo, branch `main`, main file `app.py`.
4. Click **Deploy**.

---

## 🗺️ Roadmap

- [ ] FastAPI `POST /score` endpoint
- [ ] Redis-backed user state for multi-instance deployments
- [ ] Support real datasets (CSV loader and column mapping)
- [ ] Merchant and device-based features
- [ ] Model monitoring and drift detection

---

## ⚠️ Disclaimer

This is an educational/demo project using synthetic data. It is not production-ready fraud prevention and should not be used as the sole control for real financial decisions.

## 📄 License

MIT – add a `LICENSE` file if you want to publish under this license.