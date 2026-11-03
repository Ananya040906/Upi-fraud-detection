import os
import joblib
from src.data_gen import generate
from src.detector import FraudDetector


def evaluate(det, df):
    flags = [det.process(t)["flagged"] for t in df.to_dict("records")]
    y, p = df["is_fraud"].values, __import__("numpy").array(flags)
    tp = int((y & p).sum()); fp = int(((1 - y) & p).sum()); fn = int((y & ~p).sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def main():
    det = FraudDetector().fit(generate(seed=42, start="2026-01-01"))
    prec, rec = evaluate(det, generate(seed=7, start="2026-02-05", n_users=40, days=10))
    print(f"Hold-out stream -> precision {prec:.2f}, recall {rec:.2f}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(det, "models/detector.joblib")
    return det


if __name__ == "__main__":
    main()
