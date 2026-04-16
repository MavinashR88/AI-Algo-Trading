"""
Technical indicator calculations.

Thin wrappers around the `ta` library for the indicators our strategies need.
Returns NaN-aware Series that strategies can use directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return ta.trend.ema_indicator(series, window=period)


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return ta.trend.sma_indicator(series, window=period)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    return ta.momentum.rsi(series, window=period)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    macd_obj = ta.trend.MACD(series, window_slow=slow, window_fast=fast, window_sign=signal)
    return macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, middle, lower."""
    bb = ta.volatility.BollingerBands(series, window=period, window_dev=std_dev)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    return ta.volatility.average_true_range(high, low, close, window=period)


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Weighted Average Price (rolling approximation)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap_values = (typical_price * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    return vwap_values


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index."""
    return ta.trend.adx(high, low, close, window=period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return ta.volume.on_balance_volume(close, volume)


def stochastic_rsi(
    series: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI: %K and %D lines."""
    stoch = ta.momentum.StochRSIIndicator(
        series, window=period, smooth1=smooth_k, smooth2=smooth_d,
    )
    return stoch.stochrsi_k(), stoch.stochrsi_d()


def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume moving average."""
    return volume.rolling(window=period).mean()


def price_rate_of_change(series: pd.Series, period: int = 12) -> pd.Series:
    """Price Rate of Change (ROC)."""
    return ta.momentum.roc(series, window=period)


def support_resistance(
    df: pd.DataFrame,
    lookback: int = 20,
) -> tuple[float, float]:
    """Simple support and resistance from recent high/low."""
    recent = df.tail(lookback)
    return float(recent["Low"].min()), float(recent["High"].max())
