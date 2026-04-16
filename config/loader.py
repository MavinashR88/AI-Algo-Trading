"""
Configuration loader with validation.
Loads config.yaml + holidays.yaml, merges with environment overrides,
and provides a typed Config object for the rest of the system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import yaml
from zoneinfo import ZoneInfo

_CONFIG_DIR = Path(__file__).parent
_ROOT_DIR = _CONFIG_DIR.parent

# ---------------------------------------------------------------------------
# Typed config sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeeConfig:
    slippage_pct: float
    spread_pct: float
    # US-specific
    commission: float = 0.0
    sec_fee_per_dollar: float = 0.0
    taf_fee_per_share: float = 0.0
    # India-specific
    brokerage_per_trade: float = 0.0
    stt_delivery_pct: float = 0.0
    stt_intraday_sell_pct: float = 0.0
    transaction_charges_pct: float = 0.0
    sebi_charges_pct: float = 0.0
    gst_pct: float = 0.0
    stamp_duty_buy_pct: float = 0.0


@dataclass(frozen=True)
class TradingHours:
    open: time
    close: time
    timezone: ZoneInfo
    no_trade_open_minutes: int
    no_trade_close_minutes: int


@dataclass(frozen=True)
class MarketConfig:
    enabled: bool
    mode: str  # "paper" | "live"
    broker: str
    currency: str
    exchange: str
    watchlist: list[str]
    trading_hours: TradingHours
    fees: FeeConfig
    holidays: list[date] = field(default_factory=list)


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade_pct: float
    max_open_positions_per_market: int
    max_trades_per_market_per_day: int
    max_daily_drawdown_pct: float
    max_weekly_drawdown_pct: float
    stop_loss_atr_multiplier: float
    take_profit_rr_ratio: float
    trailing_stop_activation_rr: float
    trailing_stop_atr_multiplier: float


@dataclass(frozen=True)
class PsychologyConfig:
    revenge_trade_lookback: int
    revenge_trade_cooldown_minutes: int
    min_signal_confidence: float
    sentiment_override_threshold: float
    no_averaging_down: bool


@dataclass(frozen=True)
class GraduationConfig:
    min_trades: int
    min_days: int
    min_win_rate: float
    min_sharpe: float
    max_drawdown: float
    max_single_day_loss_pct: float
    win_rate_ci_lower_bound: float
    require_human_confirmation: bool


@dataclass(frozen=True)
class DatabaseConfig:
    type: str
    path: str
    journal_mode: str
    busy_timeout_ms: int

    @property
    def full_path(self) -> Path:
        return _ROOT_DIR / self.path


@dataclass(frozen=True)
class Config:
    """Top-level typed configuration object."""
    us_market: MarketConfig
    india_market: MarketConfig
    risk: RiskConfig
    psychology: PsychologyConfig
    graduation: GraduationConfig
    database: DatabaseConfig
    raw: dict[str, Any]  # full raw YAML for strategy params, etc.


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> time:
    return datetime.strptime(t, "%H:%M").time()


def _parse_holidays(holidays_data: dict, exchange_key: str) -> list[date]:
    """Extract holiday dates for current + next year."""
    exchange = holidays_data.get(exchange_key, {})
    dates: list[date] = []
    current_year = date.today().year
    for year_key in [str(current_year), str(current_year + 1)]:
        year_holidays = exchange.get(year_key, [])
        for h in year_holidays:
            dates.append(date.fromisoformat(h["date"]))
    return dates


def _build_trading_hours(raw: dict) -> TradingHours:
    return TradingHours(
        open=_parse_time(raw["open"]),
        close=_parse_time(raw["close"]),
        timezone=ZoneInfo(raw["timezone"]),
        no_trade_open_minutes=raw["no_trade_open_minutes"],
        no_trade_close_minutes=raw["no_trade_close_minutes"],
    )


def _build_fee_config(raw: dict) -> FeeConfig:
    return FeeConfig(**{k: v for k, v in raw.items()})


def _build_market_config(
    raw_market: dict, holidays: list[date]
) -> MarketConfig:
    return MarketConfig(
        enabled=raw_market["enabled"],
        mode=raw_market["mode"],
        broker=raw_market["broker"],
        currency=raw_market["currency"],
        exchange=raw_market["exchange"],
        watchlist=raw_market["watchlist"],
        trading_hours=_build_trading_hours(raw_market["trading_hours"]),
        fees=_build_fee_config(raw_market["fees"]),
        holidays=holidays,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cached_config: Config | None = None


def load_config(
    config_path: Path | None = None,
    holidays_path: Path | None = None,
) -> Config:
    """Load and validate configuration from YAML files."""
    config_path = config_path or (_CONFIG_DIR / "config.yaml")
    holidays_path = holidays_path or (_CONFIG_DIR / "holidays.yaml")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    with open(holidays_path, "r") as f:
        holidays_raw = yaml.safe_load(f)

    # Validate required top-level keys
    for key in ["markets", "risk", "psychology", "graduation", "database"]:
        if key not in raw:
            raise ValueError(f"Missing required config section: {key}")

    us_holidays = _parse_holidays(holidays_raw, "nyse")
    india_holidays = _parse_holidays(holidays_raw, "nse")

    us_market = _build_market_config(raw["markets"]["us"], us_holidays)
    india_market = _build_market_config(raw["markets"]["india"], india_holidays)

    risk = RiskConfig(**raw["risk"])
    psychology = PsychologyConfig(**raw["psychology"])
    graduation = GraduationConfig(**raw["graduation"])
    database = DatabaseConfig(**raw["database"])

    return Config(
        us_market=us_market,
        india_market=india_market,
        risk=risk,
        psychology=psychology,
        graduation=graduation,
        database=database,
        raw=raw,
    )


def get_config(force_reload: bool = False) -> Config:
    """Get cached config singleton. Thread-safe for reads after first load."""
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = load_config()
    return _cached_config
