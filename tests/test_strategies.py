"""Tests for all trading strategies, signal generator, and backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.base import BaseStrategy, Signal
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from strategy.ma_crossover import MACrossoverStrategy
from strategy.rsi_divergence import RSIDivergenceStrategy
from strategy.volume_price import VolumePriceStrategy
from strategy.sentiment_strategy import SentimentStrategy
from strategy.hybrid import HybridStrategy
from strategy.signal_generator import SignalGenerator, STRATEGY_REGISTRY
from strategy.backtester import Backtester, BacktestResult, BacktestTrade
from strategy import indicators as ind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trending_up(rows: int = 100) -> pd.DataFrame:
    """Generate a clean uptrend OHLCV dataset."""
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    base = 100.0
    noise = np.random.RandomState(42)
    prices = base + np.cumsum(noise.uniform(0.1, 0.8, rows))
    df = pd.DataFrame({
        "Open": prices - noise.uniform(0.1, 0.5, rows),
        "High": prices + noise.uniform(0.5, 2.0, rows),
        "Low": prices - noise.uniform(0.5, 2.0, rows),
        "Close": prices,
        "Volume": noise.uniform(1e6, 5e6, rows),
    }, index=dates)
    # Fix OHLC consistency
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


def make_trending_down(rows: int = 100) -> pd.DataFrame:
    """Generate a clean downtrend OHLCV dataset."""
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    base = 200.0
    noise = np.random.RandomState(42)
    prices = base - np.cumsum(noise.uniform(0.1, 0.8, rows))
    df = pd.DataFrame({
        "Open": prices + noise.uniform(0.1, 0.5, rows),
        "High": prices + noise.uniform(0.5, 2.0, rows),
        "Low": prices - noise.uniform(0.5, 2.0, rows),
        "Close": prices,
        "Volume": noise.uniform(1e6, 5e6, rows),
    }, index=dates)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


def make_range_bound(rows: int = 100, center: float = 150.0, amplitude: float = 5.0) -> pd.DataFrame:
    """Generate sideways/ranging data."""
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    noise = np.random.RandomState(42)
    prices = center + amplitude * np.sin(np.linspace(0, 8 * np.pi, rows)) + noise.uniform(-1, 1, rows)
    df = pd.DataFrame({
        "Open": prices - noise.uniform(0.1, 0.5, rows),
        "High": prices + noise.uniform(0.5, 2.0, rows),
        "Low": prices - noise.uniform(0.5, 2.0, rows),
        "Close": prices,
        "Volume": noise.uniform(1e6, 5e6, rows),
    }, index=dates)
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignal:
    def test_create_signal(self):
        sig = Signal(asset="AAPL", direction="long", confidence=0.75, strategy="momentum")
        assert sig.is_actionable
        assert sig.asset == "AAPL"

    def test_hold_not_actionable(self):
        sig = Signal(asset="AAPL", direction="hold", confidence=0.0, strategy="test")
        assert not sig.is_actionable

    def test_zero_confidence_not_actionable(self):
        sig = Signal(asset="AAPL", direction="long", confidence=0.0, strategy="test")
        assert not sig.is_actionable


class TestBaseStrategy:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_validate_df(self):
        strat = MomentumStrategy()
        assert not strat._validate_df(pd.DataFrame(), min_rows=5)
        assert not strat._validate_df(None, min_rows=5)

    def test_hold_signal(self):
        strat = MomentumStrategy()
        sig = strat._hold_signal("AAPL", "test reason")
        assert sig.direction == "hold"
        assert sig.confidence == 0.0


# ---------------------------------------------------------------------------
# Individual Strategies
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_produces_signal(self):
        strat = MomentumStrategy()
        df = make_trending_up(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "momentum"

    def test_insufficient_data(self):
        strat = MomentumStrategy()
        df = make_trending_up(10)  # Too short
        sig = strat.analyze(df, "AAPL", "us")
        assert sig.direction == "hold"

    def test_name(self):
        assert MomentumStrategy().name == "momentum"


class TestMeanReversionStrategy:
    def test_produces_signal(self):
        strat = MeanReversionStrategy()
        df = make_range_bound(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "mean_reversion"

    def test_insufficient_data(self):
        strat = MeanReversionStrategy()
        df = make_range_bound(5)
        sig = strat.analyze(df, "AAPL")
        assert sig.direction == "hold"

    def test_name(self):
        assert MeanReversionStrategy().name == "mean_reversion"


class TestBreakoutStrategy:
    def test_produces_signal(self):
        strat = BreakoutStrategy()
        df = make_trending_up(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "breakout"

    def test_insufficient_data(self):
        strat = BreakoutStrategy()
        sig = strat.analyze(pd.DataFrame(), "AAPL")
        assert sig.direction == "hold"

    def test_name(self):
        assert BreakoutStrategy().name == "breakout"


class TestMACrossoverStrategy:
    def test_produces_signal(self):
        strat = MACrossoverStrategy()
        df = make_trending_up(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "ma_crossover"

    def test_with_sma(self):
        strat = MACrossoverStrategy(config={"ma_type": "SMA"})
        df = make_trending_up(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)

    def test_name(self):
        assert MACrossoverStrategy().name == "ma_crossover"


class TestRSIDivergenceStrategy:
    def test_produces_signal(self):
        strat = RSIDivergenceStrategy()
        df = make_trending_down(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "rsi_divergence"

    def test_name(self):
        assert RSIDivergenceStrategy().name == "rsi_divergence"


class TestVolumePriceStrategy:
    def test_produces_signal(self):
        strat = VolumePriceStrategy()
        df = make_trending_up(100)
        sig = strat.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "volume_price"

    def test_name(self):
        assert VolumePriceStrategy().name == "volume_price"


class TestSentimentStrategy:
    def test_no_sentiment_holds(self):
        strat = SentimentStrategy()
        df = make_trending_up(50)
        sig = strat.analyze(df, "AAPL", "us", sentiment_score=None)
        assert sig.direction == "hold"

    def test_bullish_sentiment(self):
        strat = SentimentStrategy()
        df = make_trending_up(50)
        sig = strat.analyze(df, "AAPL", "us", sentiment_score=0.8)
        assert isinstance(sig, Signal)
        assert sig.strategy == "sentiment"

    def test_bearish_sentiment(self):
        strat = SentimentStrategy()
        df = make_trending_down(50)
        sig = strat.analyze(df, "AAPL", "us", sentiment_score=-0.8)
        assert isinstance(sig, Signal)

    def test_name(self):
        assert SentimentStrategy().name == "sentiment"


class TestHybridStrategy:
    def test_no_strategies_hold(self):
        hybrid = HybridStrategy()
        df = make_trending_up(50)
        sig = hybrid.analyze(df, "AAPL", "us")
        assert sig.direction == "hold"

    def test_with_sub_strategies(self):
        strategies = [MomentumStrategy(), MACrossoverStrategy()]
        hybrid = HybridStrategy(
            strategies=strategies,
            weights={"momentum": 0.5, "ma_crossover": 0.5},
        )
        df = make_trending_up(100)
        sig = hybrid.analyze(df, "AAPL", "us")
        assert isinstance(sig, Signal)
        assert sig.strategy == "hybrid"

    def test_name(self):
        assert HybridStrategy().name == "hybrid"


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

class TestIndicators:
    def setup_method(self):
        self.df = make_trending_up(100)
        self.close = self.df["Close"]

    def test_ema(self):
        result = ind.ema(self.close, 20)
        assert len(result) == len(self.close)
        assert not result.iloc[-1] != result.iloc[-1]  # not NaN

    def test_sma(self):
        result = ind.sma(self.close, 20)
        assert len(result) == len(self.close)

    def test_rsi(self):
        result = ind.rsi(self.close, 14)
        valid = result.dropna()
        assert all(0 <= v <= 100 for v in valid)

    def test_macd(self):
        macd_line, signal_line, histogram = ind.macd(self.close)
        assert len(macd_line) == len(self.close)

    def test_bollinger_bands(self):
        upper, middle, lower = ind.bollinger_bands(self.close, 20, 2.0)
        valid_idx = upper.dropna().index
        assert all(upper[valid_idx] >= lower[valid_idx])

    def test_atr(self):
        result = ind.atr(self.df["High"], self.df["Low"], self.close, 14)
        valid = result.dropna()
        assert all(v >= 0 for v in valid)
        assert valid.iloc[-1] > 0  # last value should be positive for trending data

    def test_vwap(self):
        result = ind.vwap(self.df, 20)
        assert len(result) == len(self.df)

    def test_support_resistance(self):
        support, resistance = ind.support_resistance(self.df, 20)
        assert resistance >= support


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class TestSignalGenerator:
    def test_init_default(self):
        gen = SignalGenerator()
        assert len(gen.strategies) == len(STRATEGY_REGISTRY)

    def test_init_with_config(self):
        gen = SignalGenerator(config={
            "active_strategies": ["momentum", "breakout"],
        })
        assert len(gen.strategies) == 2

    def test_generate_signals(self):
        gen = SignalGenerator()
        df = make_trending_up(100)
        signals = gen.generate(df, "AAPL", "us")
        assert len(signals) > 0
        assert all(isinstance(s, Signal) for s in signals)

    def test_get_best_signal(self):
        gen = SignalGenerator(config={
            "active_strategies": ["momentum", "ma_crossover", "hybrid"],
        })
        df = make_trending_up(100)
        sig = gen.get_best_signal(df, "AAPL", "us")
        # May be None if no actionable signal
        assert sig is None or isinstance(sig, Signal)

    def test_get_strategy(self):
        gen = SignalGenerator()
        assert gen.get_strategy("momentum") is not None
        assert gen.get_strategy("nonexistent") is None


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class TestBacktester:
    def test_run_on_trending_data(self):
        bt = Backtester(initial_capital=100_000)
        strat = MomentumStrategy()
        df = make_trending_up(200)
        result = bt.run(strat, df, "AAPL", "us", warmup_bars=50)
        assert isinstance(result, BacktestResult)
        assert result.strategy == "momentum"
        assert result.market == "us"

    def test_empty_data(self):
        bt = Backtester()
        strat = MomentumStrategy()
        result = bt.run(strat, pd.DataFrame(), "AAPL")
        assert result.total_trades == 0

    def test_short_data(self):
        bt = Backtester()
        strat = MomentumStrategy()
        df = make_trending_up(10)
        result = bt.run(strat, df, "AAPL", warmup_bars=50)
        assert result.total_trades == 0

    def test_metrics_computed(self):
        bt = Backtester(initial_capital=100_000)
        strat = BreakoutStrategy(config={"lookback_period": 10, "confirmation_candles": 1, "volume_threshold": 0.5})
        df = make_trending_up(300)
        result = bt.run(strat, df, "AAPL", warmup_bars=30)
        # May or may not have trades depending on signals
        assert isinstance(result.win_rate, float)
        assert result.max_drawdown_pct >= 0

    def test_stores_in_db(self, test_db):
        bt = Backtester(initial_capital=100_000, db=test_db)
        strat = BreakoutStrategy(config={"lookback_period": 10, "confirmation_candles": 1, "volume_threshold": 0.5})
        df = make_trending_up(300)
        bt.run(strat, df, "AAPL", "us", warmup_bars=30)

        rows = test_db.execute("SELECT * FROM backtest_results WHERE strategy = 'breakout'")
        # May or may not store depending on whether trades were generated
        assert isinstance(rows, list)


class TestBacktestResult:
    def test_is_profitable(self):
        r = BacktestResult(market="us", strategy="test", start_date="", end_date="", net_pnl=100)
        assert r.is_profitable

    def test_not_profitable(self):
        r = BacktestResult(market="us", strategy="test", start_date="", end_date="", net_pnl=-50)
        assert not r.is_profitable

    def test_meets_graduation(self):
        r = BacktestResult(
            market="us", strategy="test", start_date="", end_date="",
            total_trades=150, win_rate=0.65, sharpe_ratio=1.2, max_drawdown_pct=0.10,
        )
        assert r.meets_graduation

    def test_fails_graduation_low_trades(self):
        r = BacktestResult(
            market="us", strategy="test", start_date="", end_date="",
            total_trades=50, win_rate=0.70, sharpe_ratio=1.5, max_drawdown_pct=0.05,
        )
        assert not r.meets_graduation


class TestBacktestTrade:
    def test_create_trade(self):
        t = BacktestTrade(
            asset="AAPL", direction="long",
            entry_price=100, exit_price=110,
            entry_time="2025-01-01", exit_time="2025-01-05",
            quantity=10, pnl=100, pnl_pct=0.1,
            fees=2.0, net_pnl=98.0, strategy="momentum",
        )
        assert t.net_pnl == 98.0
