# UPI / Credit Card Fraud Detection

Real-time anomaly detection for UPI and card transactions.

- **Per-user behavioural state**: running mean/std of amounts, last location, 1-hour velocity
- **Rules**: impossible travel, sudden high value, velocity burst
- **ML**: Isolation Forest risk score (0-100) for subtle outliers
- **UI**: Streamlit simulator and stream replay

## Run
```bash
pip install -r requirements.txt
python train.py
pytest
streamlit run app.py
```
