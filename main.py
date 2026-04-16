"""
AI Algo Trading System — Entry Point

Starts the trading bot: initializes config, database, logging, brokers,
and kicks off the LangGraph trading agent loop.

Usage:
    python main.py                 # run with defaults
    python main.py --log-level DEBUG  # verbose logging
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

import structlog

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import get_config
from db.database import Database
from logs import setup_logging
from notifications.email_alert import EmailAlert
from notifications.sms_alert import SMSAlert

logger = structlog.get_logger("main")

# Global shutdown flag
_shutdown = asyncio.Event()


def _handle_signal(signum: int, frame) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    logger.info("shutdown.signal_received", signal=signum)
    _shutdown.set()


async def run() -> None:
    """Main async entry point."""
    # Parse args
    parser = argparse.ArgumentParser(description="AI Algo Trading Bot")
    parser.add_argument(
        "--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)"
    )
    args = parser.parse_args()

    # Load config
    config = get_config()
    log_level = args.log_level or config.raw.get("system", {}).get("log_level", "INFO")

    # Initialize logging
    setup_logging(log_level=log_level)
    logger.info("bot.starting", version=config.raw.get("system", {}).get("version", "unknown"))

    # Initialize database
    db = Database(
        db_path=config.database.full_path,
        journal_mode=config.database.journal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    )
    db.initialize()
    db.log_audit("bot_started", "main", {"log_level": log_level})

    # Initialize notification channels
    email = EmailAlert()
    sms = SMSAlert()

    logger.info(
        "notifications.status",
        email_enabled=email.enabled,
        sms_enabled=sms.enabled,
    )

    # Send startup notification
    await email.send_bot_status("started")
    await sms.send_bot_status("started")

    logger.info(
        "bot.ready",
        us_enabled=config.us_market.enabled,
        us_mode=config.us_market.mode,
        india_enabled=config.india_market.enabled,
        india_mode=config.india_market.mode,
    )

    # ---- Main loop placeholder (LangGraph agent goes here in Phase 4) ----
    # For now, keep the process alive and respond to shutdown signals
    logger.info("bot.running", message="Waiting for trading agent (Phase 4)")

    try:
        while not _shutdown.is_set():
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

    # Graceful shutdown
    logger.info("bot.shutting_down")
    db.log_audit("bot_stopped", "main")
    await email.send_bot_status("stopped")
    await sms.send_bot_status("stopped")
    logger.info("bot.stopped")


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nShutdown requested. Goodbye.")


if __name__ == "__main__":
    main()
