import pandas as pd
from src.data_gen import generate
from src.detector import FraudDetector

det = FraudDetector().fit(generate(n_users=24, days=15, seed=1))
T0 = pd.Timestamp("2026-03-01 12:00")
BLR = (12.97, 77.59)
DEL = (28.61, 77.21)


def txn(amount, loc, minutes=0):
    return dict(user_id="user_000", timestamp=T0 + pd.Timedelta(minutes=minutes),
                amount=amount, lat=loc[0], lon=loc[1])


def test_normal_not_flagged():
    assert not det.process(txn(900, BLR))["flagged"]


def test_impossible_travel():
    det.process(txn(900, BLR, 100))
    res = det.process(txn(900, DEL, 105))
    assert res["flagged"] and any("travel" in r for r in res["reasons"])


def test_high_value():
    res = det.process(txn(500000, BLR, 400))
    assert res["flagged"]
