"""Tests for news fetcher module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.news_fetcher import NewsFetcher, NewsFetchResult, NewsItem


class TestNewsItem:
    """Test NewsItem dataclass."""

    def test_create_news_item(self):
        item = NewsItem(
            title="Apple beats earnings",
            source="newsapi",
            url="https://example.com/article",
            published_at="2026-04-16T10:00:00",
            description="Apple reported strong Q2 results",
            asset="AAPL",
            market="us",
        )
        assert item.title == "Apple beats earnings"
        assert item.source == "newsapi"
        assert item.asset == "AAPL"

    def test_minimal_news_item(self):
        item = NewsItem(title="Test", source="test")
        assert item.url == ""
        assert item.asset == ""


class TestNewsFetchResult:
    """Test NewsFetchResult dataclass."""

    def test_empty_result(self):
        result = NewsFetchResult()
        assert result.count == 0
        assert not result.has_errors

    def test_result_with_items(self):
        result = NewsFetchResult(
            items=[
                NewsItem(title="News 1", source="test"),
                NewsItem(title="News 2", source="test"),
            ]
        )
        assert result.count == 2

    def test_result_with_errors(self):
        result = NewsFetchResult(errors=["API error"])
        assert result.has_errors
        assert result.count == 0


class TestNewsFetcherInit:
    """Test NewsFetcher initialization."""

    def test_init_without_key(self, monkeypatch):
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        fetcher = NewsFetcher()
        assert fetcher.newsapi_key == ""

    def test_init_with_env_key(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key-123")
        fetcher = NewsFetcher()
        assert fetcher.newsapi_key == "test-key-123"

    def test_init_with_explicit_key(self):
        fetcher = NewsFetcher(newsapi_key="explicit-key")
        assert fetcher.newsapi_key == "explicit-key"


class TestSearchQuery:
    """Test search query building."""

    def test_us_known_ticker(self):
        fetcher = NewsFetcher()
        query = fetcher._build_search_query("AAPL", "us")
        assert "Apple" in query
        assert "stock" in query

    def test_india_known_ticker(self):
        fetcher = NewsFetcher()
        query = fetcher._build_search_query("RELIANCE", "india")
        assert "Reliance" in query

    def test_unknown_ticker(self):
        fetcher = NewsFetcher()
        query = fetcher._build_search_query("UNKNOWN_TICKER", "us")
        assert "UNKNOWN_TICKER" in query
        assert "stock market" in query


class TestYahooTimestamp:
    """Test Yahoo timestamp formatting."""

    def test_valid_timestamp(self):
        fetcher = NewsFetcher()
        result = fetcher._format_yahoo_timestamp(1713264000)
        assert "2024" in result

    def test_zero_timestamp(self):
        fetcher = NewsFetcher()
        result = fetcher._format_yahoo_timestamp(0)
        assert result == ""

    def test_none_timestamp(self):
        fetcher = NewsFetcher()
        result = fetcher._format_yahoo_timestamp(None)
        assert result == ""


class TestYahooNewsFetch:
    """Test Yahoo Finance news fetching (mocked)."""

    def test_get_yahoo_news_returns_list(self):
        fetcher = NewsFetcher()
        with patch("data.news_fetcher.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.news = [
                {"title": "Test News", "link": "https://example.com", "providerPublishTime": 1713264000},
            ]
            mock_ticker.return_value = mock_instance
            result = fetcher._get_yahoo_news("AAPL")
            assert len(result) == 1
            assert result[0]["title"] == "Test News"

    def test_get_yahoo_news_empty(self):
        fetcher = NewsFetcher()
        with patch("data.news_fetcher.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.news = []
            mock_ticker.return_value = mock_instance
            result = fetcher._get_yahoo_news("AAPL")
            assert result == []

    def test_get_yahoo_news_exception(self):
        fetcher = NewsFetcher()
        with patch("data.news_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")
            result = fetcher._get_yahoo_news("AAPL")
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_yahoo_success(self):
        fetcher = NewsFetcher()
        with patch.object(fetcher, "_get_yahoo_news") as mock_news:
            mock_news.return_value = [
                {"title": "AAPL Up", "link": "https://example.com", "providerPublishTime": 1713264000},
                {"title": "AAPL Down", "link": "https://example.com", "providerPublishTime": 1713260000},
            ]
            result = await fetcher.fetch_yahoo("AAPL", "us")
            assert result.count == 2
            assert result.items[0].source == "yahoo_finance"
            assert result.items[0].asset == "AAPL"

    @pytest.mark.asyncio
    async def test_fetch_yahoo_error(self):
        fetcher = NewsFetcher()
        with patch.object(fetcher, "_get_yahoo_news") as mock_news:
            mock_news.side_effect = Exception("Failed")
            result = await fetcher.fetch_yahoo("AAPL", "us")
            assert result.has_errors
            assert result.count == 0


class TestNewsAPIFetch:
    """Test NewsAPI fetching."""

    @pytest.mark.asyncio
    async def test_newsapi_no_key(self, monkeypatch):
        """Without API key, returns error."""
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        fetcher = NewsFetcher(newsapi_key="")
        result = await fetcher.fetch_newsapi("AAPL", "us")
        assert result.has_errors
        assert "not configured" in result.errors[0]

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Close without opening is safe."""
        fetcher = NewsFetcher()
        await fetcher.close()  # Should not raise


class TestFetchAll:
    """Test the aggregate fetch_all method."""

    @pytest.mark.asyncio
    async def test_fetch_all_deduplicates(self):
        """Duplicate headlines across sources are removed."""
        fetcher = NewsFetcher(newsapi_key="")

        google_result = NewsFetchResult(items=[
            NewsItem(title="Apple Earnings Beat", source="google_news", asset="AAPL", market="us"),
            NewsItem(title="Apple New Product", source="google_news", asset="AAPL", market="us"),
        ])
        yahoo_result = NewsFetchResult(items=[
            NewsItem(title="Apple Earnings Beat", source="yahoo_finance", asset="AAPL", market="us"),
            NewsItem(title="Apple Stock Rises", source="yahoo_finance", asset="AAPL", market="us"),
        ])

        with patch.object(fetcher, "fetch_google_rss", return_value=google_result), \
             patch.object(fetcher, "fetch_yahoo", return_value=yahoo_result):
            result = await fetcher.fetch_all("AAPL", "us")
            # "Apple Earnings Beat" should appear only once
            titles = [item.title for item in result.items]
            assert titles.count("Apple Earnings Beat") == 1
            assert result.count == 3  # 2 from google + 1 unique from yahoo

    @pytest.mark.asyncio
    async def test_fetch_all_handles_exceptions(self):
        """Exceptions from sources are captured as errors."""
        fetcher = NewsFetcher(newsapi_key="")

        with patch.object(fetcher, "fetch_google_rss", side_effect=Exception("RSS failed")), \
             patch.object(fetcher, "fetch_yahoo", return_value=NewsFetchResult()):
            result = await fetcher.fetch_all("AAPL", "us")
            assert result.has_errors
            assert "RSS failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_fetch_all_max_results(self):
        """Results are trimmed to max_results."""
        fetcher = NewsFetcher(newsapi_key="")

        many_items = NewsFetchResult(items=[
            NewsItem(title=f"News {i}", source="google_news", asset="AAPL", market="us")
            for i in range(100)
        ])

        with patch.object(fetcher, "fetch_google_rss", return_value=many_items), \
             patch.object(fetcher, "fetch_yahoo", return_value=NewsFetchResult()):
            result = await fetcher.fetch_all("AAPL", "us", max_results=10)
            assert result.count == 10
