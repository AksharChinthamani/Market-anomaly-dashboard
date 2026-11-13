OVERVIEW
This project real-time financial dashboard that ingests high-frequency tick data from the Binance cryptocurrency exchange. It uses an ensemble of five Machine Learning models to evaluate market conditions instantly, detecting anomalies and forecasting volatility with quantified confidence intervals.

MACHINE LEARNING ARCHITECTURE
The system relies on five separate models running in parallel

Standard Scaler - Pre-processing
Normalizes the real-time feature vectors (returns, volume Z-scores) so they match the exact distribution the models were trained on, preventing train/serve skew.

Isolation Forest - Anomaly Detection
An unsupervised model that learns the shape of normal market behavior. It flags multivariate anomalies (such as a massive volume spike with flat price) without needing labeled data.

XGBoost Median - Volatility Forecast
A supervised regression model that predicts the expected magnitude of volatility over the next 10 minutes. It forecasts the standard deviation of returns, not the price direction.

XGBoost Lower Quantile - 10th Percentile
A quantile regression model trained to predict the absolute best-case scenario for future volatility.

XGBoost Upper Quantile - 90th Percentile
A quantile regression model trained to predict the absolute worst-case scenario for future volatility.

How Confidence is Calculated
The backend compares the spread between the Lower Quantile and Upper Quantile predictions. If the spread is tight, the dashboard shows high confidence (green). If the spread is wide (meaning the market is unpredictable), the dashboard drops the confidence score (red). This ensures the user is warned when the model is uncertain.

SYSTEM ARCHITECTURE
The system is built as a continuous pipeline:

Data Ingestion: An asynchronous WebSocket connection to the Binance public stream.
Feature Engineering: Rolling mathematical calculations (returns, volatility, volume Z-score) are shared strictly between offline training and live inference.
ML Inference: Live predictions are generated using the pre-trained joblib models.
Backend: A FastAPI server handles the inference loop and broadcasts the results to connected browsers via WebSockets.
Frontend: The user interface uses TradingView Lightweight Charts for high-performance, synchronized rendering of the live price and predicted volatility charts.

TECH STACK

Programming Language: Python (version 3.14)
Machine Learning: Scikit-Learn, XGBoost, Pandas, Numpy, Joblib
Web Framework: FastAPI, Uvicorn, WebSockets
Frontend: HTML, JavaScript, TradingView Lightweight Charts

HOW TO RUN LOCALLY

Activate the virtual environment:
.venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt

Train the models (this downloads 30 days of historical data and saves all 5 models to the models folder):
python train_models.py

Start the FastAPI server:
uvicorn app:app

Open your web browser and navigate to:
http://localhost:8000
