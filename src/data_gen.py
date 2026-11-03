"""Synthetic UPI/card transactions with injected fraud patterns."""
import numpy as np
import pandas as pd
from .detector import haversine

CITIES = {"Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
          "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36), "Hyderabad": (17.38, 78.48),
          "Pune": (18.52, 73.86), "Jaipur": (26.91, 75.79)}
NAMES = list(CITIES)


def generate(n_users=200, days=30, per_day=3, seed=42, start="2026-01-01", fraud_rate=0.02):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    rows = []
    for i in range(n_users):
        home = NAMES[i % len(NAMES)]
        base = np.random.default_rng(1000 + i).uniform(300, 3000)
        n = rng.poisson(days * per_day)
        for s in np.sort(rng.uniform(0, days * 86400, n)):
            rows.append(dict(user_id=f"user_{i:03d}", timestamp=start + pd.Timedelta(seconds=float(s)),
                             amount=round(float(rng.lognormal(np.log(base), 0.5)), 2),
                             city=home, lat=CITIES[home][0] + rng.normal(0, .02),
                             lon=CITIES[home][1] + rng.normal(0, .02),
                             channel=rng.choice(["UPI", "CARD"]), is_fraud=0))
    df = pd.DataFrame(rows)
    fraud = []
    for j, ix in enumerate(rng.choice(len(df), max(3, int(fraud_rate * len(df))), replace=False)):
        r = df.iloc[ix].to_dict()
        r["is_fraud"] = 1
        kind = j % 3
        if kind == 0:      # sudden high value
            r["timestamp"] += pd.Timedelta(minutes=30)
            r["amount"] = round(r["amount"] * rng.uniform(12, 25), 2)
            fraud.append(r)
        elif kind == 1:    # rapid geographic shift
            far = [c for c in NAMES if haversine(*CITIES[r["city"]], *CITIES[c]) > 800]
            c = str(rng.choice(far))
            r.update(city=c, lat=CITIES[c][0], lon=CITIES[c][1],
                     timestamp=r["timestamp"] + pd.Timedelta(minutes=int(rng.integers(5, 15))))
            fraud.append(r)
        else:              # velocity burst
            for k in range(1, 8):
                fraud.append({**r, "timestamp": r["timestamp"] + pd.Timedelta(seconds=40 * k),
                              "amount": round(float(rng.uniform(100, 900)), 2)})
    df = pd.concat([df, pd.DataFrame(fraud)])
    return df.sort_values("timestamp").reset_index(drop=True)
