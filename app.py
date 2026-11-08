"""
FastAPI Backend for Live ML Inference.

This file ties everything together:
1. Loads the trained models.
2. Runs a background task to listen to Binance.
3. Computes features on every tick.
4. Runs live inference (Anomaly + Volatility).
5. Broadcasts the result to connected browsers via WebSocket.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import websockets

import config
from ingestion import TickBuffer, parse_ticker_message, build_stream_url
from features import compute_features, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Load Models ──────────────────────────────────────────────────────
logger.info("Loading models...")
scaler = joblib.load("models/scaler.joblib")
iso_forest = joblib.load("models/iso_forest.joblib")
xgb_model = joblib.load("models/xgb_model.joblib")
logger.info("Models loaded successfully.")

# ── WebSocket Connection Manager ─────────────────────────────────────
class ConnectionManager:
    """Keeps track of all connected browsers so we can broadcast to them."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Browser might have closed abruptly; ignore and clean up later
                pass

manager = ConnectionManager()
buffer = TickBuffer(size=config.BUFFER_SIZE)

# ── The Live ML Loop ─────────────────────────────────────────────────
async def binance_stream_and_inference():
    """
    Connects to Binance, updates the buffer, runs ML, and broadcasts.
    """
    url = build_stream_url(config.SYMBOLS, config.STREAM_TYPE)
    
    while True:
        try:
            logger.info(f"Connecting to {url}...")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logger.info("Connected to Binance.")
                
                async for raw_msg in ws:
                    tick = parse_ticker_message(raw_msg)
                    if not tick:
                        continue
                    
                    buffer.add(tick)
                    
                    # We need at least 20 ticks to compute our rolling features
                    if len(buffer.get(tick.symbol)) < 20:
                        continue
                        
                    # 1. Convert the recent buffer into a DataFrame
                    ticks = list(buffer.get(tick.symbol))
                    df = pd.DataFrame([
                        {"price": t.price, "volume": t.volume} for t in ticks
                    ])
                    
                    # 2. Compute features
                    df_feat = compute_features(df)
                    
                    # 3. Get the very last row (the current moment)
                    # Fill any NaN values just in case (though the length check prevents this)
                    current_features = df_feat[FEATURE_COLUMNS].iloc[[-1]].fillna(0)
                    
                    # 4. Scale and Predict
                    X_scaled = scaler.transform(current_features)
                    
                    # Isolation Forest outputs 1 (normal) or -1 (anomaly)
                    iso_pred = iso_forest.predict(X_scaled)[0]
                    is_anomaly = bool(iso_pred == -1)
                    
                    # XGBoost outputs the predicted volatility
                    vol_pred = float(xgb_model.predict(X_scaled)[0])
                    
                    # Naive Baseline (3% move in the last 1 minute)
                    # We just use return_1 for this simple baseline
                    return_1 = float(current_features["return_1"].iloc[0])
                    naive_alert = abs(return_1) > 0.03
                    
                    # 5. Assemble the payload
                    payload = {
                        "ts": tick.event_time_ms,
                        "symbol": tick.symbol.upper(),
                        "price": tick.price,
                        "anomaly": is_anomaly,
                        "vol_forecast": round(vol_pred, 6),
                        "naive_alert": bool(naive_alert)
                    }
                    
                    # 6. Broadcast to all connected browsers
                    if manager.active_connections:
                        await manager.broadcast(json.dumps(payload))
                        
        except Exception as e:
            logger.error(f"Stream error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

# ── FastAPI Lifespan (Startup/Shutdown) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch the Binance task in the background
    task = asyncio.create_task(binance_stream_and_inference())
    yield
    # Shutdown: cancel the task
    task.cancel()

app = FastAPI(lifespan=lifespan)

# ── Routes ───────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect the browser to send us messages, 
            # but we need to await to keep the connection open.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def get():
    # For Week 3, we just serve a raw JSON viewer to prove it works.
    # In Week 4, we will replace this with the actual dashboard HTML.
    return HTMLResponse("""
    <html>
        <body style="background: #111; color: #0f0; font-family: monospace; padding: 20px;">
            <h2>Live ML Stream (Raw JSON)</h2>
            <div id="output" style="font-size: 14px; line-height: 1.5;"></div>
            <script>
                const ws = new WebSocket("ws://localhost:8000/ws");
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    const el = document.createElement('div');
                    el.innerText = `Price: ${data.price} | Anomaly: ${data.anomaly} | Vol: ${data.vol_forecast}`;
                    document.getElementById('output').prepend(el);
                };
            </script>
        </body>
    </html>
    """)