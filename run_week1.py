"""
Week 1 verification runner.

Runs ingestion for N seconds, then prints a summary of what we caught.
Use this to confirm the pipeline is healthy before moving to Week 2.

Why a separate runner? Because `ingestion.py` is a *library*. It
should be importable and testable without side effects. The runner
is the *application* that wires the library to a temporary run.
"""

from __future__ import annotations

import asyncio
import logging
import statistics

import config
from ingestion import TickBuffer, ingest_forever

RUN_SECONDS = 60


async def run_for(seconds: int) -> None:
    buffer = TickBuffer(size=config.BUFFER_SIZE)

    # Start ingestion as a background task
    task = asyncio.create_task(ingest_forever(buffer))

    try:
        await asyncio.sleep(seconds)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"WEEK 1 VERIFICATION SUMMARY  (ran for {seconds}s)")
    print("=" * 62)

    for symbol in config.SYMBOLS:
        ticks = buffer.get(symbol)
        if not ticks:
            print(f"\n{symbol.upper()}: NO TICKS CAPTUREDE")
            continue

        prices = [t.price for t in ticks]
        print(f"\n{symbol.upper()}:")
        print(f"  Ticks captured : {len(ticks)}")
        print(f"  First price    : {prices[0]:,.2f}")
        print(f"  Last price     : {prices[-1]:,.2f}")
        print(f"  Min / Max      : {min(prices):,.2f} / {max(prices):,.2f}")
        if len(prices) > 1:
            print(f"  Price stdev    : {statistics.stdev(prices):,.4f}")
        print(f"  First event    : {ticks[0].event_time_iso}")
        print(f"  Last event     : {ticks[-1].event_time_iso}")

    print("\n" + "=" * 62)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        asyncio.run(run_for(RUN_SECONDS))
    except KeyboardInterrupt:
        print("\nInterrupted before summary.")


if __name__ == "__main__":
    main()