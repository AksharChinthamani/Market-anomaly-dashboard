"""
FastAPI Backend & Dashboard for Live ML Inference.

Features:
1. Live data ingestion from Binance.
2. Live ML inference (Isolation Forest + XGBoost).
3. Dual synchronized charts (Price + Volatility).
4. Explainable AI alerts (shows exact feature values causing the anomaly).
"""

import asyncio
import json
import logging
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
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()
buffer = TickBuffer(size=config.BUFFER_SIZE)


# ── The Live ML Loop ─────────────────────────────────────────────────
async def binance_stream_and_inference():
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
                    
                    if len(buffer.get(tick.symbol)) < 20:
                        continue
                        
                    ticks = list(buffer.get(tick.symbol))
                    df = pd.DataFrame([
                        {"price": t.price, "volume": t.volume} for t in ticks
                    ])
                    
                    df_feat = compute_features(df)
                    current_features = df_feat[FEATURE_COLUMNS].iloc[[-1]].fillna(0)
                    
                    X_scaled = scaler.transform(current_features)
                    
                    iso_pred = iso_forest.predict(X_scaled)[0]
                    is_anomaly = bool(iso_pred == -1)
                    
                    vol_pred = float(xgb_model.predict(X_scaled)[0])
                    
                    return_1 = float(current_features["return_1"].iloc[0])
                    volume_z = float(current_features["volume_zscore"].iloc[0])
                    naive_alert = abs(return_1) > 0.03
                    
                    # ── UPGRADE 1: Explainable Payload ──
                    payload = {
                        "ts": tick.event_time_ms,
                        "symbol": tick.symbol.upper(),
                        "price": tick.price,
                        "anomaly": is_anomaly,
                        "vol_forecast": round(vol_pred, 6),
                        "naive_alert": bool(naive_alert),
                        "volume_z": round(volume_z, 2),
                        "return_1": round(return_1 * 100, 4) # Convert to percentage
                    }
                    
                    if manager.active_connections:
                        await manager.broadcast(json.dumps(payload))
                        
        except Exception as e:
            logger.error(f"Stream error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


# ── FastAPI Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(binance_stream_and_inference())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/")
async def get():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live Market Anomaly Dashboard</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        
        /* ── UPGRADE 2: Dual Chart Layout ── */
        #charts-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #30363d;
        }
        #price-chart { flex: 3; }
        #vol-chart { flex: 1; border-top: 1px solid #30363d; }

        #sidebar {
            width: 350px;
            display: flex;
            flex-direction: column;
            background-color: #161b22;
        }
        .panel {
            padding: 20px;
            border-bottom: 1px solid #30363d;
        }
        h2 { margin-top: 0; font-size: 16px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
        .stat { font-size: 24px; font-weight: bold; margin: 5px 0; }
        .stat-label { font-size: 12px; color: #8b949e; }
        .green { color: #3fb950; }
        .red { color: #f85149; }
        #alert-feed {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            list-style: none;
            margin: 0;
        }
        .alert-item {
            background-color: rgba(248, 81, 73, 0.1);
            border-left: 3px solid #f85149;
            padding: 10px;
            margin-bottom: 10px;
            font-size: 13px;
            border-radius: 0 4px 4px 0;
        }
        .alert-item .time { color: #8b949e; font-size: 11px; display: block; margin-bottom: 4px; }
        .alert-item .price { font-weight: bold; color: #c9d1d9; }
        .alert-item .reason { color: #e3b341; font-size: 12px; margin-top: 5px; display: block; }
        
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            text-align: center;
        }
        .comparison-box {
            background: #0d1117;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }
    </style>
</head>
<body>

    <div id="charts-wrapper">
        <div id="price-chart"></div>
        <div id="vol-chart"></div>
    </div>

    <div id="sidebar">
        <div class="panel">
            <h2>Live Market Data</h2>
            <div class="stat-label">Current Price</div>
            <div class="stat" id="current-price">--</div>
            <div class="stat-label" style="margin-top: 15px;">Predicted Volatility (10m)</div>
            <div class="stat" id="vol-forecast">--</div>
        </div>

        <div class="panel">
            <h2>Alert Comparison</h2>
            <div class="comparison-grid">
                <div class="comparison-box">
                    <div class="stat-label">ML Alerts</div>
                    <div class="stat red" id="ml-count">0</div>
                </div>
                <div class="comparison-box">
                    <div class="stat-label">Naive Alerts</div>
                    <div class="stat" style="color: #e3b341;" id="naive-count">0</div>
                </div>
            </div>
        </div>

        <div class="panel" style="border-bottom: none; padding-bottom: 10px;">
            <h2>Anomaly Feed</h2>
        </div>
        <ul id="alert-feed"></ul>
    </div>

    <script>
        // --- 1. Setup the Price Chart ---
        const priceChartElement = document.getElementById('price-chart');
        const priceChart = LightweightCharts.createChart(priceChartElement, {
            layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
            grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
            timeScale: { visible: false }, // Hide time axis on top chart
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        });

        const priceSeries = priceChart.addSeries(LightweightCharts.LineSeries, {
            color: '#58a6ff',
            lineWidth: 2,
        });

        // --- 2. Setup the Volatility Chart ---
        const volChartElement = document.getElementById('vol-chart');
        const volChart = LightweightCharts.createChart(volChartElement, {
            layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
            grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
            timeScale: { timeVisible: true, secondsVisible: true },
            rightPriceScale: { borderVisible: false },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        });

        const volSeries = volChart.addSeries(LightweightCharts.HistogramSeries, {
            color: '#e3b341', // Yellow bars
        });

        // --- 3. Sync the Charts ---
        priceChart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
            if (logicalRange !== null) volChart.timeScale().setVisibleLogicalRange(logicalRange);
        });
        volChart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
            if (logicalRange !== null) priceChart.timeScale().setVisibleLogicalRange(logicalRange);
        });

        // Handle window resize
        new ResizeObserver(entries => {
            if (entries.length === 0 || entries[0].target !== document.getElementById('charts-wrapper')) { return; }
            const newRect = entries[0].contentRect;
            priceChart.applyOptions({ width: newRect.width });
            volChart.applyOptions({ width: newRect.width });
        }).observe(document.getElementById('charts-wrapper'));

        // --- 4. Setup Data & State ---
        let mlAlertCount = 0;
        let naiveAlertCount = 0;
        let markers = [];
        let lastChartTime = 0;

        // --- 5. Connect WebSocket ---
        const ws = new WebSocket("ws://localhost:8000/ws");

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const time = Math.floor(data.ts / 1000);

            // Update Charts ONLY if time has strictly increased
            if (time > lastChartTime) {
                priceSeries.update({ time: time, value: data.price });
                // Scale volatility by 10000 so the bars are visually readable
                volSeries.update({ time: time, value: data.vol_forecast * 10000 });
                lastChartTime = time;
            }
            
            // Update Sidebar Stats
            document.getElementById('current-price').innerText = `$${data.price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('vol-forecast').innerText = `${(data.vol_forecast * 100).toFixed(3)}%`;
            
            const volEl = document.getElementById('vol-forecast');
            volEl.className = data.vol_forecast > 0.002 ? 'stat red' : 'stat green';

            // Handle ML Anomaly (Explainable AI)
            if (data.anomaly) {
                mlAlertCount++;
                document.getElementById('ml-count').innerText = mlAlertCount;

                markers.push({
                    time: time,
                    position: 'aboveBar',
                    color: '#f85149',
                    shape: 'circle',
                    text: 'Anomaly'
                });
                priceSeries.setMarkers(markers);

                const feed = document.getElementById('alert-feed');
                const item = document.createElement('li');
                item.className = 'alert-item';
                const dateStr = new Date(data.ts).toLocaleTimeString();
                
                // UPGRADE 1: Show exactly why the anomaly fired
                item.innerHTML = `
                    <span class="time">${dateStr}</span>
                    <span class="price">$${data.price.toLocaleString()}</span><br>
                    <span class="reason">
                        <strong>Features:</strong> Vol Z-Score: ${data.volume_z} | 1-Tick Return: ${data.return_1}%
                    </span>
                `;
                feed.prepend(item);

                if (feed.children.length > 50) {
                    feed.removeChild(feed.lastChild);
                }
            }

            // Handle Naive Alert
            if (data.naive_alert) {
                naiveAlertCount++;
                document.getElementById('naive-count').innerText = naiveAlertCount;
            }
        };
    </script>
</body>
</html>
""")