"""Real-time fraud detector: per-user behaviour state + rules + Isolation Forest."""
from collections import defaultdict, deque
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["log_amount", "amount_ratio", "distance_km", "speed_kmph", "txn_1h", "hour"]

RULES = {
    "max_speed_kmph": 900,      # faster than a commercial flight => impossible travel
    "min_distance_km": 100,
    "amount_ratio": 6.0,        # amount vs user's average
    "amount_zscore": 4.0,
    "max_txn_per_hour": 4,
}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


class UserState:
    """Rolling profile of one user (Welford running mean/std, last location, recent times)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.last = None            # (lat, lon, timestamp)
        self.times = deque()

    @property
    def std(self):
        return sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


class FraudDetector:
    def __init__(self, threshold=70):
        self.states = defaultdict(UserState)
        self.model = IsolationForest(n_estimators=100, random_state=42)
        self.threshold = threshold
        self.lo = self.hi = None

    # ---------- feature engineering (identical for training and live scoring) ----------
    def _features(self, txn, st):
        ts = pd.Timestamp(txn["timestamp"])
        while st.times and (ts - st.times[0]).total_seconds() > 3600:
            st.times.popleft()
        amt = float(txn["amount"])
        ratio = amt / st.mean if st.n >= 3 and st.mean > 0 else 1.0
        z = (amt - st.mean) / st.std if st.n >= 5 and st.std > 0 else 0.0
        dist = speed = minutes = 0.0
        if st.last:
            dist = haversine(st.last[0], st.last[1], txn["lat"], txn["lon"])
            minutes = max((ts - st.last[2]).total_seconds() / 60, 1.0)
            speed = min(dist / (minutes / 60), 2000)
        return {"log_amount": np.log1p(amt), "amount_ratio": min(ratio, 100), "distance_km": dist,
                "speed_kmph": speed, "txn_1h": len(st.times) + 1, "hour": ts.hour,
                "_z": z, "_minutes": minutes}

    def _update(self, st, txn, flagged):
        ts = pd.Timestamp(txn["timestamp"])
        st.times.append(ts)
        if flagged:                 # don't let suspicious activity pollute the user's profile
            return
        amt = float(txn["amount"])
        st.n += 1
        d = amt - st.mean
        st.mean += d / st.n
        st.m2 += d * (amt - st.mean)
        st.last = (txn["lat"], txn["lon"], ts)

    # ---------- training ----------
    def fit(self, df):
        df = df.sort_values("timestamp")
        rows = []
        for t in df.to_dict("records"):
            st = self.states[t["user_id"]]
            f = self._features(t, st)
            rows.append([f[c] for c in FEATURES])
            self._update(st, t, False)
        X = np.array(rows)
        self.model.fit(X)
        raw = -self.model.score_samples(X)
        self.lo, self.hi = np.percentile(raw, 50), np.percentile(raw, 99.5)
        return self

    # ---------- real-time scoring ----------
    def _rules(self, f):
        r = []
        if f["speed_kmph"] > RULES["max_speed_kmph"] and f["distance_km"] > RULES["min_distance_km"]:
            r.append(f"Impossible travel: {f['distance_km']:.0f} km in {f['_minutes']:.0f} min")
        if f["amount_ratio"] >= RULES["amount_ratio"] and f["_z"] >= RULES["amount_zscore"]:
            r.append(f"Sudden high value: {f['amount_ratio']:.1f}x user's average")
        if f["txn_1h"] >= RULES["max_txn_per_hour"]:
            r.append(f"Velocity: {f['txn_1h']} transactions in 1 hour")
        return r

    def process(self, txn, update=True):
        """txn: dict(user_id, timestamp, amount, lat, lon). Returns risk score, flag and reasons."""
        st = self.states[txn["user_id"]]
        f = self._features(txn, st)
        raw = -self.model.score_samples(np.array([[f[c] for c in FEATURES]]))[0]
        risk = float(np.clip((raw - self.lo) / (self.hi - self.lo), 0, 1) * 100)
        reasons = self._rules(f)
        if risk >= self.threshold and not reasons:
            reasons.append("Statistical anomaly (Isolation Forest)")
        flagged = bool(reasons)
        if update:
            self._update(st, txn, flagged)
        return {"risk_score": round(risk, 1), "flagged": flagged, "reasons": reasons,
                "features": {k: round(float(v), 2) for k, v in f.items() if not k.startswith("_")}}
