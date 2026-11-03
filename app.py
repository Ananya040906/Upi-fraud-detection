import copy
import os

import joblib
import pandas as pd
import streamlit as st

from src.data_gen import CITIES, generate

st.set_page_config(page_title="UPI / Card Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_detector():
    if not os.path.exists("models/detector.joblib"):
        from train import main
        main()
    return joblib.load("models/detector.joblib")


base = load_detector()
st.title("UPI / Credit Card Fraud Detection")
st.caption("Isolation Forest + behavioural rules (impossible travel, sudden high value, velocity)")
tab1, tab2 = st.tabs(["Live transaction simulator", "Stream replay"])

with tab1:
    if "det" not in st.session_state:
        st.session_state.det = copy.deepcopy(base)
        st.session_state.clock = pd.Timestamp("2026-03-01 12:00")
        st.session_state.log = []
    c1, c2 = st.columns([1, 2])
    with c1:
        user = st.selectbox("User", [f"user_{i:03d}" for i in range(16)])
        amount = st.number_input("Amount (₹)", 1.0, 1_000_000.0, 1500.0, step=100.0)
        city = st.selectbox("Transaction city", list(CITIES))
        gap = st.number_input("Minutes since previous action", 1, 1440, 5)
        if st.button("Submit transaction", type="primary"):
            st.session_state.clock += pd.Timedelta(minutes=int(gap))
            t = dict(user_id=user, timestamp=st.session_state.clock, amount=amount,
                     lat=CITIES[city][0], lon=CITIES[city][1])
            res = st.session_state.det.process(t)
            st.session_state.log.insert(0, {"time": t["timestamp"], "user": user, "city": city,
                                            "amount": amount, "risk": res["risk_score"],
                                            "flagged": res["flagged"], "reasons": "; ".join(res["reasons"])})
            st.metric("Risk score", res["risk_score"])
            (st.error if res["flagged"] else st.success)(
                "🚨 FLAGGED: " + "; ".join(res["reasons"]) if res["flagged"] else "✅ Looks normal")
        if st.button("Reset session"):
            for k in ("det", "clock", "log"):
                st.session_state.pop(k, None)
            st.rerun()
    with c2:
        st.write("Try: user_000 (Bengaluru) → ₹1,000 in Bengaluru, then ₹1,000 in Delhi 5 min later.")
        st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True)

with tab2:
    n_users = st.slider("Users", 50, 200, 100)
    if st.button("Generate & replay stream"):
        det = copy.deepcopy(base)
        df = generate(n_users=n_users, days=10, seed=99, start="2026-02-10")
        out = [det.process(t) for t in df.to_dict("records")]
        df["risk"] = [o["risk_score"] for o in out]
        df["flagged"] = [o["flagged"] for o in out]
        df["reasons"] = ["; ".join(o["reasons"]) for o in out]
        tp = int(((df.flagged) & (df.is_fraud == 1)).sum())
        m = st.columns(4)
        m[0].metric("Transactions", len(df))
        m[1].metric("Flagged", int(df.flagged.sum()))
        m[2].metric("Precision", f"{tp / max(df.flagged.sum(), 1):.2f}")
        m[3].metric("Recall", f"{tp / max(df.is_fraud.sum(), 1):.2f}")
        st.map(df[df.flagged][["lat", "lon"]])
        st.dataframe(df[df.flagged][["timestamp", "user_id", "city", "amount", "risk", "reasons", "is_fraud"]],
                     use_container_width=True)
