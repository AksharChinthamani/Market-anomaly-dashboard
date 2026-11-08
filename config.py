"""
Central configuration for the market anomaly dashboard.

Rule of thumb: anything you might want to *change* lives here.
Anything you would never change lives in the code that uses it.
"""

# ── Binance endpoints ──────────────────────────────────────────────
# Binance offers two WebSocket URL roots. Both work; :9443 is the
# "classic" root and is what most tutorials use. The /stream suffix
# enables the *combined* stream endpoint, which lets us subscribe to
# multiple symbols in one connection.
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"

# ── What we monitor ────────────────────────────────────────────────
# Lowercase is required by Binance stream naming (btcusdt, not BTCUSDT).
# Start with one symbol. Add ethusdt, bnbusdt later if you want.
SYMBOLS = ["btcusdt"]

# Binance offers several stream types per symbol:
#   @ticker  -> rolling 24h ticker, ~1 update/sec, has price + volume
#   @trade   -> every individual trade (irregular intervals, noisy)
#   @kline_1m -> candle updates for the 1-minute timeframe
#
# We use @ticker because:
#   - It gives a regular ~1Hz sample (great for rolling volatility)
#   - It carries volume in one message (avoids a second subscription)
#   - It is the *live* counterpart of the historical K-lines we'll
#     download in Week 2 — same fields, so features will match.
STREAM_TYPE = "ticker"

# ── Rolling buffer ─────────────────────────────────────────────────
# How many recent ticks to retain in memory per symbol.
# 500 ticks @ ~1/sec ≈ ~8 minutes of context — enough for the
# 20-tick rolling features we'll build in Week 2 with headroom.
BUFFER_SIZE = 500

# ── Reconnection (exponential backoff) ─────────────────────────────
# If the connection drops, we retry with growing delay so we don't
# hammer Binance. Reset to INITIAL on every successful reconnect.
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 30
MAX_RECONNECT_ATTEMPTS = 10

# ── Logging noise control ──────────────────────────────────────────
# Print a heartbeat line every N ticks. Don't print every tick —
# your terminal will drown and you won't be able to read logs.
PRINT_EVERY_N_TICKS = 5