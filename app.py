"""
Hugging Face Spaces entry point.

Initializes the SQLite database and starts the FastAPI dashboard on port 7860.
Runs in demo/paper mode — no live broker or API keys required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

from dashboard.api import app, set_database
from db.database import Database


def _bootstrap_demo_data(db: Database) -> None:
    """Seed the dashboard with some demo signals/trades for first-time visitors."""
    from db import queries

    # Only seed if empty
    existing = db.execute("SELECT COUNT(*) AS n FROM trades")
    if existing and existing[0]["n"] > 0:
        return

    queries.update_bot_state(db, "us", capital=100_000, peak_capital=100_000)
    queries.update_bot_state(db, "india", capital=1_000_000, peak_capital=1_000_000)

    for asset, direction, entry, exit_price, pnl in [
        ("AAPL", "long", 180.0, 185.5, 55.0),
        ("MSFT", "long", 410.0, 418.0, 80.0),
        ("TSLA", "short", 240.0, 232.0, 80.0),
        ("NVDA", "long", 880.0, 895.0, 150.0),
    ]:
        tid = queries.insert_trade(
            db, market="us", asset=asset, direction=direction,
            entry_price=entry, quantity=10,
            entry_time="2026-04-17T09:30:00", strategy="momentum",
        )
        queries.close_trade(
            db, tid, exit_price=exit_price,
            exit_time="2026-04-17T15:30:00",
            pnl=pnl, pnl_pct=pnl / (entry * 10),
            net_pnl=pnl - 2.0, fees=2.0,
        )

    for asset, score, headline in [
        ("AAPL", 0.72, "Apple beats Q2 earnings expectations"),
        ("TSLA", -0.35, "Tesla faces production delays in Shanghai"),
        ("NVDA", 0.85, "NVIDIA unveils next-gen AI accelerator"),
    ]:
        queries.insert_sentiment(
            db, market="us", asset=asset, score=score,
            headline=headline, source="demo",
        )


def main() -> None:
    db_path = os.environ.get("DB_PATH", "/tmp/trading.db")
    db = Database(db_path)
    db.migrate()
    _bootstrap_demo_data(db)
    set_database(db)

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
