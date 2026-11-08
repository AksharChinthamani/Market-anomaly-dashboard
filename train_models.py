"""
Offline training script.

1. Downloads 30 days of 1-minute historical K-lines.
2. Engineers features using features.py.
3. Trains Isolation Forest (Anomaly).
4. Trains XGBoost (Volatility forecast).
5. Saves models to the /models directory.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from features import compute_features, FEATURE_COLUMNS

# --- Config ---
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
DAYS = 30
MODEL_DIR = "models"

# The horizon for our volatility forecast (10 minutes)
FORECAST_HORIZON = 10 

os.makedirs(MODEL_DIR, exist_ok=True)

def download_historical_data(days: int) -> pd.DataFrame:
    """
    Fetch K-lines from Binance REST API.
    Binance limits to 1000 candles per request, so we paginate backward.
    """
    print(f"Downloading {days} days of {SYMBOL} {INTERVAL} data...")
    end_time = int(time.time() * 1000)
    all_dfs = []
    
    for _ in range(days):
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}&endTime={end_time}"
        resp = requests.get(url).json()
        
        if not isinstance(resp, list) or len(resp) == 0:
            break
            
        df = pd.DataFrame(resp, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
        ])
        
        all_dfs.append(df)
        # Set the next end_time to just before the oldest candle we just got
        end_time = int(df["open_time"].iloc[0]) - 1
        time.sleep(0.5) # Be polite to the API
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df["price"] = final_df["close"].astype(float)
    final_df["volume"] = final_df["volume"].astype(float)
    final_df["open_time"] = pd.to_datetime(final_df["open_time"], unit="ms")
    
    final_df = final_df.sort_values("open_time").reset_index(drop=True)
    print(f"Downloaded {len(final_df)} rows.")
    return final_df

def prepare_training_data(df: pd.DataFrame):
    """Engineer features and create labels."""
    # 1. Compute features
    df = compute_features(df)
    
    # 2. Create the target for XGBoost: 
    # Future 10-minute volatility (std dev of the NEXT 10 returns)
    # We use shift(-10) to look into the future for the TARGET only.
    df["target_vol_10"] = df["return_1"].rolling(window=FORECAST_HORIZON).std().shift(-FORECAST_HORIZON)
    
    # 3. Drop NaNs (caused by rolling windows and future shifts)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_vol_10"])
    
    X = df[FEATURE_COLUMNS]
    y_vol = df["target_vol_10"]
    
    return X, y_vol

def main():
    # 1. Get Data
    df = download_historical_data(DAYS)
    X, y_vol = prepare_training_data(df)
    
    print(f"Training data ready: {len(X)} samples.")
    
    # Scale features for Isolation Forest (XGBoost doesn't strictly need it, but we'll use one scaler for both)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- TRAIN ISOLATION FOREST ---
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=0.02, # Assume 2% of data is anomalous
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    
    # --- TRAIN XGBOOST ---
    print("Training XGBoost Volatility Forecaster...")
    # Use a time-series split (last 20% as validation)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_vol[:split_idx], y_vol[split_idx:]
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Quick evaluation
    from sklearn.metrics import mean_absolute_error
    preds = xgb_model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"XGBoost Validation MAE (Volatility): {mae:.6f}")
    
    # --- SAVE EVERYTHING ---
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(iso_forest, f"{MODEL_DIR}/iso_forest.joblib")
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgb_model.joblib")
    
    print(f"Models successfully saved to /{MODEL_DIR}")

if __name__ == "__main__":
    main()