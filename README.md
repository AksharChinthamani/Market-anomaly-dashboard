# Market-anomaly-dashboard

Live market anomaly detection and volatility forecasting using Binance market
data, a scikit-learn Isolation Forest, and an XGBoost model.

## Run locally

From this directory, install the dependencies and start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

The dashboard endpoint is available at `http://localhost:8000/`. The app reads
the trained model files from `models/`; the included model files are tracked
with this project.
