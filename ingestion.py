"""
Binance WebSocket ingestion layer.

This module is responsible for ONE thing:
    Connect to Binance, receive ticks, store them in a rolling buffer.

It deliberately contains no ML and no feature engineering. Those
belong in later modules. Keeping ingestion clean means when
something breaks, you know whether it's a network issue (here)
or a model issue (elsewhere).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque

import websockets
from websockets.exceptions import ConnectionClosed

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Tick:
    """
    One parsed market data point.

    Why a dataclass? It gives us typed, named fields without boilerplate.
    Why not a dict? Because `tick.price` is checked by the IDE and
    typos are caught at import time; `tick["prise"]` silently fails.
    """
    symbol: str
    price: float          # last traded price ('c' in Binance ticker)
    volume: float         # 24h traded volume in base asset ('v')
    event_time_ms: int    # Binance event time, ms since epoch ('E')
    received_at: float    # our local clock when we parsed it

    @property
    def event_time_iso(self) -> str:
        """Human-readable UTC timestamp for logs and the alert feed."""
        return datetime.fromtimestamp(
            self.event_time_ms / 1000, tz=timezone.utc
        ).isoformat()


# ─────────────────────────────────────────────────────────────────────
# Rolling buffer
# ─────────────────────────────────────────────────────────────────────
class TickBuffer:
    """
    Fixed-size ring buffer that keeps the most recent N ticks per symbol.

    Why collections.deque(maxlen=N)?
      - Append is O(1): when the buffer is full, the oldest tick
        is dropped automatically. No shifting, no allocation.
      - Memory is bounded: a firehose of ticks can't grow the process.
      - It is thread-safe for append/pop on CPython, which matters
        later when FastAPI's event loop and the ingestion task share it.

    Why per-symbol dict inside?
      So adding ETH later is a config change, not a code change.
    """

    def __init__(self, size: int):
        self.size = size
        self._data: dict[str, Deque[Tick]] = {}

    def add(self, tick: Tick) -> None:
        if tick.symbol not in self._data:
            self._data[tick.symbol] = deque(maxlen=self.size)
        self._data[tick.symbol].append(tick)

    def get(self, symbol: str) -> Deque[Tick]:
        """Read-only view of the buffer for a symbol (empty if none)."""
        return self._data.get(symbol, deque())

    def latest(self, symbol: str) -> Tick | None:
        """Most recent tick for a symbol, or None."""
        buf = self._data.get(symbol)
        return buf[-1] if buf else None

    def __len__(self) -> int:
        return sum(len(b) for b in self._data.values())


# ─────────────────────────────────────────────────────────────────────
# Message parsing
# ─────────────────────────────────────────────────────────────────────
def parse_ticker_message(raw: str) -> Tick | None:
    """
    Parse a raw Binance 'ticker' message into a Tick, or None on failure.

    Binance combined-stream payload wraps the real data:
        {"stream":"btcusdt@ticker", "data": { ...ticker fields... }}
    Single-stream payload (if you connect to /ws/btcusdt@ticker) is
    just the inner dict. We handle both for robustness.

    Relevant ticker fields we use:
      's'  symbol                  e.g. "BTCUSDT"
      'c'  last traded price       e.g. "68012.40"   (string!)
      'v'  24h traded volume       e.g. "15234.12"   (string!)
      'E'  event time (ms)         e.g. 1730000000000

    Why wrap in try/except?
      Binance occasionally sends test messages or malformed frames
      during maintenance. One bad message should not kill the stream.
      Log, skip, continue.
    """
    try:
        msg = json.loads(raw)
        payload = msg.get("data", msg)
        return Tick(
            symbol=payload["s"].lower(),
            price=float(payload["c"]),
            volume=float(payload["v"]),
            event_time_ms=int(payload["E"]),
            received_at=time.time(),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse ticker message: %s | raw=%s",
            exc, raw[:200],
        )
        return None


# ─────────────────────────────────────────────────────────────────────
# Stream URL builder
# ─────────────────────────────────────────────────────────────────────
def build_stream_url(symbols: list[str], stream_type: str) -> str:
    """
    Build a Binance combined-stream URL.

    Example output for ["btcusdt", "ethusdt"] and "ticker":
      wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker

    Why combined endpoint? One TCP connection, many symbols. Cheaper
    and simpler than one socket per symbol.
    """
    streams = "/".join(f"{s}@{stream_type}" for s in symbols)
    return f"{config.BINANCE_WS_URL}?streams={streams}"


# ─────────────────────────────────────────────────────────────────────
# Main ingestion loop with reconnection
# ─────────────────────────────────────────────────────────────────────
async def ingest_forever(buffer: TickBuffer) -> None:
    """
    Connect to Binance, stream ticks into the buffer, reconnect on drop.

    Reconnection strategy — exponential backoff:
      1s -> 2s -> 4s -> 8s -> 16s -> 30s (capped)
    with a counter reset to 0 after every *successful* connection.

    Why exponential backoff? Binance imposes IP rate limits. If we
    retried every second during an outage, we'd burn the limit and
    get locked out. Backoff is polite and correct.
    """
    url = build_stream_url(config.SYMBOLS, config.STREAM_TYPE)
    backoff = config.INITIAL_BACKOFF_SECONDS
    attempts = 0
    tick_counter = 0

    while attempts < config.MAX_RECONNECT_ATTEMPTS:
        try:
            logger.info("Connecting to %s ...", url)
            async with websockets.connect(
                url,
                ping_interval=20,   # send WS ping every 20s
                ping_timeout=10,    # treat no-pong-for-10s as dead
            ) as ws:
                logger.info("Connected. Streaming ticks...")
                backoff = config.INITIAL_BACKOFF_SECONDS
                attempts = 0

                async for raw in ws:
                    tick = parse_ticker_message(raw)
                    if tick is None:
                        continue

                    buffer.add(tick)
                    tick_counter += 1

                    if tick_counter % config.PRINT_EVERY_N_TICKS == 0:
                        latest = buffer.latest(tick.symbol)
                        logger.info(
                            "[%d ticks] %s price=%.2f vol=%.0f buffer=%d",
                            tick_counter,
                            latest.symbol.upper(),
                            latest.price,
                            latest.volume,
                            len(buffer),
                        )

        except (ConnectionClosed, OSError) as exc:
            attempts += 1
            logger.warning(
                "Disconnected (%s). Retrying in %ss (attempt %d/%d)...",
                exc, backoff, attempts, config.MAX_RECONNECT_ATTEMPTS,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, config.MAX_BACKOFF_SECONDS)

    logger.error("Max reconnect attempts reached. Giving up.")


# ─────────────────────────────────────────────────────────────────────
# Standalone entrypoint (Ctrl+C to stop)
# ─────────────────────────────────────────────────────────────────────
async def main() -> None:
    buffer = TickBuffer(size=config.BUFFER_SIZE)
    await ingest_forever(buffer)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")