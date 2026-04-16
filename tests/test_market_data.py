"""Tests for market data fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.market_data import BadDataReport, MarketDataFetcher, OHLCV


class TestSymbolResolution:
    """Test ticker symbol resolution for different markets."""

    def test_us_symbol_unchanged(self):
        fetcher = MarketDataFetcher()
        assert fetcher._resolve_symbol("AAPL", "us") == "AAPL"

    def test_india_symbol_gets_ns_suffix(self):
        fetcher = MarketDataFetcher()
        assert fetcher._resolve_symbol("RELIANCE", "india") == "RELIANCE.NS"

    def test_india_symbol_no_double_suffix(self):
        fetcher = MarketDataFetcher()
        assert fetcher._resolve_symbol("RELIANCE.NS", "india") == "RELIANCE.NS"


class TestBadDataDetection:
    """Test data quality checks."""

    def setup_method(self):
        self.fetcher = MarketDataFetcher()

    def _make_df(self, rows: int = 50, **overrides) -> pd.DataFrame:
        """Create a clean OHLCV DataFrame for testing."""
        dates = pd.date_range("2025-01-01", periods=rows, freq="D")
        base_price = 100.0
        data = {
            "Open": np.random.uniform(base_price * 0.99, base_price * 1.01, rows),
            "High": np.random.uniform(base_price * 1.01, base_price * 1.03, rows),
            "Low": np.random.uniform(base_price * 0.97, base_price * 0.99, rows),
            "Close": np.random.uniform(base_price * 0.99, base_price * 1.01, rows),
            "Volume": np.random.uniform(1e6, 5e6, rows),
        }
        # Ensure High is max and Low is min for each row
        df = pd.DataFrame(data, index=dates)
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1) * 1.001
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1) * 0.999
        return df

    def test_clean_data_no_issues(self):
        """Clean data should produce no issues."""
        df = self._make_df()
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert report.is_clean

    def test_empty_dataframe(self):
        """Empty DataFrame should be flagged."""
        df = pd.DataFrame()
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert not report.is_clean
        assert "Empty dataset" in report.issues[0]

    def test_nan_values_detected(self):
        """NaN values should be flagged."""
        df = self._make_df()
        df.iloc[5, 0] = np.nan  # Open
        df.iloc[10, 3] = np.nan  # Close
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert any("NaN" in issue for issue in report.issues)

    def test_negative_prices_detected(self):
        """Negative prices should be flagged."""
        df = self._make_df()
        df.iloc[3, 0] = -5.0  # Negative Open
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert any("<= 0" in issue for issue in report.issues)

    def test_zero_prices_detected(self):
        """Zero prices should be flagged."""
        df = self._make_df()
        df.iloc[3, 3] = 0.0  # Zero Close
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert any("<= 0" in issue for issue in report.issues)

    def test_extreme_spike_detected(self):
        """Price spikes >20% should be flagged."""
        df = self._make_df()
        df.iloc[10, 3] = df.iloc[9, 3] * 1.5  # 50% spike in Close
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert any("20% price change" in issue for issue in report.issues)

    def test_high_less_than_low_detected(self):
        """High < Low is invalid OHLC data."""
        df = self._make_df()
        df.iloc[5, 1] = df.iloc[5, 2] - 1.0  # High < Low
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert any("High < Low" in issue for issue in report.issues)

    def test_zero_volume_low_percentage_ok(self):
        """A few zero volume bars (<10%) should not be flagged."""
        df = self._make_df(rows=100)
        df.iloc[5, 4] = 0.0  # Just 1% zero volume
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert not any("zero volume" in issue for issue in report.issues)

    def test_zero_volume_high_percentage_flagged(self):
        """Many zero volume bars (>10%) should be flagged."""
        df = self._make_df(rows=50)
        df.iloc[:10, 4] = 0.0  # 20% zero volume
        report = self.fetcher.detect_bad_data(df, "TEST")
        assert any("zero volume" in issue for issue in report.issues)


class TestBadDataReport:
    """Test BadDataReport dataclass."""

    def test_clean_report(self):
        report = BadDataReport(asset="TEST")
        assert report.is_clean
        assert report.asset == "TEST"

    def test_report_with_issues(self):
        report = BadDataReport(asset="TEST", issues=["Issue 1", "Issue 2"])
        assert not report.is_clean
        assert len(report.issues) == 2


class TestMarketDataCaching:
    """Test database caching of OHLCV data."""

    @pytest.mark.asyncio
    async def test_cache_to_db(self, test_db):
        """Data gets cached to market_data_cache table."""
        fetcher = MarketDataFetcher(db=test_db)

        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
        }, index=dates)

        await fetcher._cache_to_db(df, "AAPL", "us", "1d")

        rows = test_db.execute(
            "SELECT * FROM market_data_cache WHERE asset = 'AAPL'"
        )
        assert len(rows) == 5
        assert rows[0]["market"] == "us"
        assert rows[0]["timeframe"] == "1d"

    @pytest.mark.asyncio
    async def test_cache_idempotent(self, test_db):
        """Caching same data twice doesn't duplicate (INSERT OR IGNORE)."""
        fetcher = MarketDataFetcher(db=test_db)

        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1e6, 1.1e6, 1.2e6],
        }, index=dates)

        await fetcher._cache_to_db(df, "MSFT", "us", "1d")
        await fetcher._cache_to_db(df, "MSFT", "us", "1d")

        rows = test_db.execute(
            "SELECT * FROM market_data_cache WHERE asset = 'MSFT'"
        )
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_get_cached_data(self, test_db):
        """Cached data can be retrieved as a DataFrame."""
        fetcher = MarketDataFetcher(db=test_db)

        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1e6, 1.1e6, 1.2e6],
        }, index=dates)

        await fetcher._cache_to_db(df, "GOOGL", "us", "1d")

        cached = await fetcher.get_cached_data("GOOGL", "us", "1d")
        assert len(cached) == 3
        assert "Open" in cached.columns
        assert "Close" in cached.columns

    @pytest.mark.asyncio
    async def test_empty_cache_returns_empty_df(self, test_db):
        """Querying empty cache returns empty DataFrame."""
        fetcher = MarketDataFetcher(db=test_db)
        cached = await fetcher.get_cached_data("NONEXISTENT", "us", "1d")
        assert cached.empty

    @pytest.mark.asyncio
    async def test_no_db_cache_returns_empty(self):
        """Without DB, cached data returns empty DataFrame."""
        fetcher = MarketDataFetcher(db=None)
        cached = await fetcher.get_cached_data("AAPL", "us", "1d")
        assert cached.empty


class TestResample:
    """Test 1h to 4h resampling."""

    def test_resample_to_4h(self):
        fetcher = MarketDataFetcher()
        # Start at 08:00 so 8 bars fit exactly into two 4h buckets
        dates = pd.date_range("2025-01-02 08:00", periods=8, freq="1h")
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104, 105, 106, 107],
            "High": [110, 111, 112, 113, 114, 115, 116, 117],
            "Low": [90, 91, 92, 93, 94, 95, 96, 97],
            "Close": [105, 106, 107, 108, 109, 110, 111, 112],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        }, index=dates)

        resampled = fetcher._resample_to_4h(df)
        assert len(resampled) == 2
        # First 4h candle (08:00-11:00): Open is first, High is max, Low is min
        assert resampled.iloc[0]["Open"] == 100
        assert resampled.iloc[0]["High"] == 113
        assert resampled.iloc[0]["Low"] == 90
        assert resampled.iloc[0]["Close"] == 108
        assert resampled.iloc[0]["Volume"] == 4600  # sum of first 4
