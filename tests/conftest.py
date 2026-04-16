"""
Shared test fixtures for all test modules.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary database path for tests."""
    return tmp_path / "test_trading.db"


@pytest.fixture
def test_db(tmp_db_path):
    """Provide an initialized test database."""
    from db.database import Database
    db = Database(db_path=tmp_db_path)
    db.initialize()
    return db


@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config files for testing."""
    import yaml

    config = {
        "system": {"name": "test", "version": "0.0.1", "log_level": "DEBUG"},
        "markets": {
            "us": {
                "enabled": True,
                "mode": "paper",
                "broker": "alpaca",
                "currency": "USD",
                "exchange": "NYSE/NASDAQ",
                "watchlist": ["AAPL", "MSFT"],
                "trading_hours": {
                    "open": "09:30",
                    "close": "16:00",
                    "timezone": "US/Eastern",
                    "no_trade_open_minutes": 15,
                    "no_trade_close_minutes": 15,
                },
                "fees": {
                    "commission": 0.0,
                    "sec_fee_per_dollar": 0.0000278,
                    "taf_fee_per_share": 0.000166,
                    "slippage_pct": 0.001,
                    "spread_pct": 0.0005,
                },
            },
            "india": {
                "enabled": True,
                "mode": "paper",
                "broker": "zerodha",
                "currency": "INR",
                "exchange": "NSE/BSE",
                "watchlist": ["RELIANCE", "TCS"],
                "trading_hours": {
                    "open": "09:15",
                    "close": "15:30",
                    "timezone": "Asia/Kolkata",
                    "no_trade_open_minutes": 15,
                    "no_trade_close_minutes": 15,
                },
                "fees": {
                    "brokerage_per_trade": 20,
                    "stt_delivery_pct": 0.001,
                    "stt_intraday_sell_pct": 0.00025,
                    "transaction_charges_pct": 0.0000345,
                    "sebi_charges_pct": 0.000001,
                    "gst_pct": 0.18,
                    "stamp_duty_buy_pct": 0.00015,
                    "slippage_pct": 0.001,
                    "spread_pct": 0.0005,
                },
            },
        },
        "risk": {
            "max_risk_per_trade_pct": 0.01,
            "max_open_positions_per_market": 3,
            "max_trades_per_market_per_day": 10,
            "max_daily_drawdown_pct": 0.03,
            "max_weekly_drawdown_pct": 0.08,
            "stop_loss_atr_multiplier": 2.0,
            "take_profit_rr_ratio": 2.0,
            "trailing_stop_activation_rr": 1.5,
            "trailing_stop_atr_multiplier": 1.5,
        },
        "psychology": {
            "revenge_trade_lookback": 2,
            "revenge_trade_cooldown_minutes": 30,
            "min_signal_confidence": 0.65,
            "sentiment_override_threshold": -0.7,
            "no_averaging_down": True,
        },
        "graduation": {
            "min_trades": 100,
            "min_days": 30,
            "min_win_rate": 0.70,
            "min_sharpe": 1.0,
            "max_drawdown": 0.10,
            "max_single_day_loss_pct": 0.03,
            "win_rate_ci_lower_bound": 0.65,
            "require_human_confirmation": True,
        },
        "database": {
            "type": "sqlite",
            "path": "db/test_trading.db",
            "journal_mode": "WAL",
            "busy_timeout_ms": 5000,
        },
    }

    holidays = {
        "nyse": {
            "name": "NYSE",
            "timezone": "US/Eastern",
            "2026": [
                {"date": "2026-01-01", "name": "New Year"},
                {"date": "2026-12-25", "name": "Christmas"},
            ],
        },
        "nse": {
            "name": "NSE",
            "timezone": "Asia/Kolkata",
            "2026": [
                {"date": "2026-01-26", "name": "Republic Day"},
                {"date": "2026-08-15", "name": "Independence Day"},
            ],
        },
    }

    config_path = tmp_path / "config.yaml"
    holidays_path = tmp_path / "holidays.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f)
    with open(holidays_path, "w") as f:
        yaml.dump(holidays, f)

    return config_path, holidays_path
