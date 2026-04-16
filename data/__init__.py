"""Data layer: market data, news, and sentiment analysis."""

from data.market_data import MarketDataFetcher
from data.news_fetcher import NewsFetcher
from data.sentiment import SentimentAnalyzer

__all__ = ["MarketDataFetcher", "NewsFetcher", "SentimentAnalyzer"]
