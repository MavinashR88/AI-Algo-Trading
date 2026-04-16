"""Tests for config loading and validation."""

from __future__ import annotations

from datetime import date, time

import pytest
import yaml

from config.loader import Config, load_config


class TestConfigLoader:
    """Test config.yaml loading and parsing."""

    def test_load_from_temp_files(self, config_dir):
        """Config loads correctly from valid YAML files."""
        config_path, holidays_path = config_dir
        config = load_config(config_path, holidays_path)

        assert isinstance(config, Config)
        assert config.us_market.enabled is True
        assert config.india_market.enabled is True

    def test_us_market_config(self, config_dir):
        """US market config has correct values."""
        config = load_config(*config_dir)

        assert config.us_market.broker == "alpaca"
        assert config.us_market.currency == "USD"
        assert config.us_market.mode == "paper"
        assert "AAPL" in config.us_market.watchlist
        assert config.us_market.trading_hours.open == time(9, 30)
        assert config.us_market.trading_hours.close == time(16, 0)
        assert config.us_market.trading_hours.no_trade_open_minutes == 15
        assert config.us_market.fees.commission == 0.0
        assert config.us_market.fees.slippage_pct == 0.001

    def test_india_market_config(self, config_dir):
        """India market config has correct values."""
        config = load_config(*config_dir)

        assert config.india_market.broker == "zerodha"
        assert config.india_market.currency == "INR"
        assert "RELIANCE" in config.india_market.watchlist
        assert config.india_market.trading_hours.open == time(9, 15)
        assert config.india_market.fees.brokerage_per_trade == 20
        assert config.india_market.fees.gst_pct == 0.18

    def test_risk_config(self, config_dir):
        """Risk parameters load correctly."""
        config = load_config(*config_dir)

        assert config.risk.max_risk_per_trade_pct == 0.01
        assert config.risk.max_open_positions_per_market == 3
        assert config.risk.max_daily_drawdown_pct == 0.03
        assert config.risk.max_weekly_drawdown_pct == 0.08
        assert config.risk.take_profit_rr_ratio == 2.0

    def test_psychology_config(self, config_dir):
        """Psychology guard settings load correctly."""
        config = load_config(*config_dir)

        assert config.psychology.revenge_trade_lookback == 2
        assert config.psychology.revenge_trade_cooldown_minutes == 30
        assert config.psychology.min_signal_confidence == 0.65
        assert config.psychology.sentiment_override_threshold == -0.7
        assert config.psychology.no_averaging_down is True

    def test_graduation_config(self, config_dir):
        """Graduation criteria load correctly."""
        config = load_config(*config_dir)

        assert config.graduation.min_trades == 100
        assert config.graduation.min_days == 30
        assert config.graduation.min_win_rate == 0.70
        assert config.graduation.require_human_confirmation is True

    def test_holidays_parsed(self, config_dir):
        """Holiday calendars parse into date objects."""
        config = load_config(*config_dir)

        assert len(config.us_market.holidays) >= 1
        assert len(config.india_market.holidays) >= 1
        assert all(isinstance(d, date) for d in config.us_market.holidays)
        assert date(2026, 1, 1) in config.us_market.holidays
        assert date(2026, 1, 26) in config.india_market.holidays

    def test_missing_section_raises(self, tmp_path):
        """Missing required section raises ValueError."""
        bad_config = {"markets": {}, "risk": {}}  # missing psychology, graduation, etc.
        config_path = tmp_path / "bad.yaml"
        holidays_path = tmp_path / "holidays.yaml"

        with open(config_path, "w") as f:
            yaml.dump(bad_config, f)
        with open(holidays_path, "w") as f:
            yaml.dump({"nyse": {}, "nse": {}}, f)

        with pytest.raises(ValueError, match="Missing required config section"):
            load_config(config_path, holidays_path)

    def test_raw_dict_available(self, config_dir):
        """Raw config dict is accessible for strategy params."""
        config = load_config(*config_dir)
        assert isinstance(config.raw, dict)
        assert "markets" in config.raw


class TestConfigFromProjectFiles:
    """Test loading the actual project config files."""

    def test_load_project_config(self):
        """The real config.yaml + holidays.yaml load without errors."""
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
        holidays_path = project_root / "config" / "holidays.yaml"

        config = load_config(config_path, holidays_path)

        assert config.us_market.enabled is True
        assert config.india_market.enabled is True
        assert len(config.us_market.watchlist) >= 5
        assert len(config.india_market.watchlist) >= 5
        assert config.risk.max_daily_drawdown_pct > 0
        assert config.graduation.min_win_rate >= 0.70
