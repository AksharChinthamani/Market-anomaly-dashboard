MARKET ANOMALY AND VOLATILITY DASHBOARD

This project is a live, real-time financial dashboard that ingests high-frequency tick data from the Binance cryptocurrency exchange.
It uses two parallel Machine Learning models to evaluate market conditions instantly:

1. Isolation Forest: An unsupervised model that flags multivariate anomalies like unusual combinations of price and volume.
2. XGBoost: A supervised model that forecasts the expected magnitude of volatility over the next 10 minutes.
The dashboard compares these ML predictions against a naive threshold baseline to visually demonstrate the value of machine learning in financial surveillance.

ARCHITECTURE AND DATA FLOW
The system is built as a continuous pipeline:
- Data Ingestion: An asynchronous WebSocket connection to the Binance public stream.
- Feature Engineering: Rolling mathematical calculations (returns, volatility, volume Z-score) are shared strictly between offline training and live inference to prevent train/serve skew.
- ML Inference: Live predictions are generated using pre-trained joblib models.
- Backend: A FastAPI server handles the inference loop and broadcasts the results to connected browsers via WebSockets.
- Frontend: The user interface uses TradingView Lightweight Charts for high-performance, synchronized rendering of the live price and predicted volatility charts.

TECH STACK
- Programming Language: Python (version 3.14)
- Machine Learning: Scikit-Learn (Isolation Forest), XGBoost (Volatility Regressor), Pandas
- Web Framework: FastAPI, Uvicorn
- Frontend: HTML, JavaScript, TradingView Lightweight Charts

HOW TO RUN LOCALLY
1. Activate the virtual environment: 
   .venv\Scripts\activate
2. Install the required dependencies: 
   pip install -r requirements.txt
3. Train the models (this downloads 30 days of historical data and saves the models to the /models folder): 
   python train_models.py
4. Start the FastAPI server: 
   uvicorn app:app
5. Open your web browser and navigate to: 
   http://localhost:8000
