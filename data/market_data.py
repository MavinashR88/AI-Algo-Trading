"""
Market data fetcher using yfinance.

Supports both US (NYSE/NASDAQ) and Indian (NSE/BSE) markets.
- Historical OHLCV data at multiple timeframes
- Bad data detection (gaps, spikes, zero volume)
- SQLite caching via market_data_cache table
- India symbols auto-suffixed with .NS for yfinance
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from db.database import Database

logger = structlog.get_logger(__name__)

# India NSE suffix for yfinance
_NSE_SUFFIX = ".NS"

# Timeframe mapping: our config names → yfinance intervals
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "60m",   # yfinance doesn't support 4h — we'll resample from 1h
    "1d": "1d",
    "1wk": "1wk",
}

# Max lookback per interval (yfinance limits)
MAX_LOOKBACK = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": 10000,
    "1wk": 10000,
}


@dataclass
class OHLCV:
    """Single candle."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BadDataReport:
    """Report of data quality issues."""
    asset: str
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


class MarketDataFetcher:
    """Fetches and caches market data for US and India."""

    def __init__(self, db: Database | None = None):
        self.db = db

    def _resolve_symbol(self, asset: str, market: str) -> str:
        """Convert our asset name to yfinance ticker."""
        if market == "india" and not asset.endswith(_NSE_SUFFIX):
            return f"{asset}{_NSE_SUFFIX}"
        return asset

    async def get_historical(
        self,
        asset: str,
        market: str,
        timeframe: str = "1d",
        lookback_days: int | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            asset: Ticker symbol (e.g. "AAPL", "RELIANCE").
            market: "us" or "india".
            timeframe: Candle interval (1m, 5m, 15m, 1h, 4h, 1d).
            lookback_days: Number of days to look back from today.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, indexed by datetime.
        """
        symbol = self._resolve_symbol(asset, market)
        yf_interval = TIMEFRAME_MAP.get(timeframe, "1d")
        need_resample = timeframe == "4h"

        if need_resample:
            yf_interval = "1h"

        # Determine date range
        if lookback_days and not start:
            max_days = MAX_LOOKBACK.get(yf_interval, 365)
            lookback_days = min(lookback_days, max_days)
            start_date = datetime.utcnow() - timedelta(days=lookback_days)
            start = start_date.strftime("%Y-%m-%d")

        logger.info(
            "market_data.fetch_historical",
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            interval=yf_interval,
            start=start,
            end=end,
        )

        # Run yfinance in thread pool (it's blocking I/O)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: self._fetch_yfinance(symbol, yf_interval, start, end),
        )

        if df.empty:
            logger.warning("market_data.empty_result", symbol=symbol)
            return df

        # Resample 1h → 4h if needed
        if need_resample and not df.empty:
            df = self._resample_to_4h(df)

        # Cache to database
        if self.db is not None:
            await self._cache_to_db(df, asset, market, timeframe)

        return df

    def _fetch_yfinance(
        self,
        symbol: str,
        interval: str,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """Synchronous yfinance download."""
        try:
            ticker = yf.Ticker(symbol)
            kwargs: dict[str, Any] = {"interval": interval}
            if start:
                kwargs["start"] = start
            if end:
                kwargs["end"] = end
            if not start and not end:
                kwargs["period"] = "1y"

            df = ticker.history(**kwargs)

            if df.empty:
                return df

            # Normalize column names
            df.columns = [c.title() for c in df.columns]
            # Keep only OHLCV
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            return df[keep].copy()

        except Exception as e:
            logger.error("market_data.yfinance_error", symbol=symbol, error=str(e))
            return pd.DataFrame()

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to 4-hour candles."""
        return df.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    async def get_latest_bars(
        self,
        asset: str,
        market: str,
        timeframe: str = "15m",
        count: int = 100,
    ) -> pd.DataFrame:
        """Get the most recent N bars for an asset.

        For intraday timeframes, fetches enough lookback to get `count` bars.
        """
        # Estimate days needed
        if timeframe in ("1m", "5m", "15m", "30m"):
            # ~6.5 trading hours per day for US, ~6.25 for India
            bars_per_day = {"1m": 390, "5m": 78, "15m": 26, "30m": 13}
            bpd = bars_per_day.get(timeframe, 26)
            lookback_days = max(3, int(count / bpd * 1.5))
        elif timeframe == "1h":
            lookback_days = max(5, int(count / 7 * 1.5))
        elif timeframe == "4h":
            lookback_days = max(10, int(count / 2 * 1.5))
        else:
            lookback_days = max(count, 30)

        df = await self.get_historical(
            asset, market, timeframe=timeframe, lookback_days=lookback_days,
        )
        return df.tail(count) if not df.empty else df

    async def get_quote(self, asset: str, market: str) -> dict[str, float] | None:
        """Get current price info for an asset.

        Returns dict with keys: last, open, high, low, close, volume, prev_close.
        """
        symbol = self._resolve_symbol(asset, market)

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: self._get_ticker_info(symbol))

        if not info:
            return None
        return info

    def _get_ticker_info(self, symbol: str) -> dict[str, float] | None:
        """Synchronous quote fetch."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or "regularMarketPrice" not in info:
                # Fallback: get last bar from history
                hist = ticker.history(period="1d")
                if hist.empty:
                    return None
                last_row = hist.iloc[-1]
                return {
                    "last": float(last_row["Close"]),
                    "open": float(last_row["Open"]),
                    "high": float(last_row["High"]),
                    "low": float(last_row["Low"]),
                    "close": float(last_row["Close"]),
                    "volume": float(last_row["Volume"]),
                }

            return {
                "last": float(info.get("regularMarketPrice", 0)),
                "open": float(info.get("regularMarketOpen", 0)),
                "high": float(info.get("regularMarketDayHigh", 0)),
                "low": float(info.get("regularMarketDayLow", 0)),
                "close": float(info.get("regularMarketPreviousClose", 0)),
                "volume": float(info.get("regularMarketVolume", 0)),
                "prev_close": float(info.get("regularMarketPreviousClose", 0)),
            }
        except Exception as e:
            logger.error("market_data.quote_error", symbol=symbol, error=str(e))
            return None

    def detect_bad_data(self, df: pd.DataFrame, asset: str) -> BadDataReport:
        """Detect data quality issues in OHLCV data.

        Checks:
        - Missing values (NaN rows)
        - Zero or negative prices
        - Zero volume days (possible gap)
        - Extreme price spikes (>20% single-bar move)
        - OHLC consistency (high >= low, high >= open/close)
        """
        report = BadDataReport(asset=asset)

        if df.empty:
            report.issues.append("Empty dataset")
            return report

        # 1. Missing values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            report.issues.append(f"Found {nan_count} NaN values")

        # 2. Zero or negative prices
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        for col in price_cols:
            bad = (df[col] <= 0).sum()
            if bad > 0:
                report.issues.append(f"{bad} rows with {col} <= 0")

        # 3. Zero volume
        if "Volume" in df.columns:
            zero_vol = (df["Volume"] == 0).sum()
            if zero_vol > 0:
                pct = zero_vol / len(df) * 100
                if pct > 10:  # Only flag if >10% of bars have zero volume
                    report.issues.append(
                        f"{zero_vol} bars ({pct:.1f}%) with zero volume"
                    )

        # 4. Extreme price spikes (>20% single bar)
        if "Close" in df.columns and len(df) > 1:
            returns = df["Close"].pct_change().abs()
            spikes = (returns > 0.20).sum()
            if spikes > 0:
                report.issues.append(f"{spikes} bars with >20% price change")

        # 5. OHLC consistency
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            bad_high = (df["High"] < df["Low"]).sum()
            if bad_high > 0:
                report.issues.append(f"{bad_high} bars where High < Low")

            bad_open_high = (df["High"] < df["Open"]).sum()
            if bad_open_high > 0:
                report.issues.append(f"{bad_open_high} bars where High < Open")

            bad_close_high = (df["High"] < df["Close"]).sum()
            if bad_close_high > 0:
                report.issues.append(f"{bad_close_high} bars where High < Close")

        if report.is_clean:
            logger.info("market_data.data_quality_ok", asset=asset, rows=len(df))
        else:
            logger.warning(
                "market_data.data_quality_issues",
                asset=asset,
                issues=report.issues,
            )

        return report

    async def _cache_to_db(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str,
        timeframe: str,
    ) -> None:
        """Cache OHLCV data to market_data_cache table."""
        if self.db is None or df.empty:
            return

        rows = []
        for ts, row in df.iterrows():
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append((
                market, asset, timeframe, ts_str,
                float(row["Open"]), float(row["High"]),
                float(row["Low"]), float(row["Close"]),
                float(row["Volume"]),
            ))

        try:
            self.db.execute_many(
                """INSERT OR IGNORE INTO market_data_cache
                (market, asset, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            logger.info(
                "market_data.cached",
                asset=asset,
                market=market,
                timeframe=timeframe,
                rows=len(rows),
            )
        except Exception as e:
            logger.error("market_data.cache_error", asset=asset, error=str(e))

    async def get_cached_data(
        self,
        asset: str,
        market: str,
        timeframe: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Load cached OHLCV data from database."""
        if self.db is None:
            return pd.DataFrame()

        rows = self.db.execute(
            """SELECT timestamp, open, high, low, close, volume
            FROM market_data_cache
            WHERE market = ? AND asset = ? AND timeframe = ?
            ORDER BY timestamp DESC LIMIT ?""",
            (market, asset, timeframe, limit),
        )

        if not rows:
            return pd.DataFrame()

        data = [
            {
                "timestamp": r["timestamp"],
                "Open": r["open"],
                "High": r["high"],
                "Low": r["low"],
                "Close": r["close"],
                "Volume": r["volume"],
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    async def get_multi_asset_data(
        self,
        assets: list[str],
        market: str,
        timeframe: str = "1d",
        lookback_days: int = 60,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple assets concurrently."""
        tasks = [
            self.get_historical(asset, market, timeframe, lookback_days=lookback_days)
            for asset in assets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for asset, result in zip(assets, results):
            if isinstance(result, Exception):
                logger.error(
                    "market_data.multi_fetch_error",
                    asset=asset,
                    error=str(result),
                )
                data[asset] = pd.DataFrame()
            else:
                data[asset] = result
        return data
