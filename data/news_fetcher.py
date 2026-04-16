"""
News aggregator for trading sentiment analysis.

Sources:
- NewsAPI (newsapi.org) — structured JSON API
- Google News RSS — free, no key required
- Yahoo Finance news — via yfinance

Returns normalized NewsItem objects ready for sentiment scoring.
"""

from __future__ import annotations

import asyncio
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote_plus

import aiohttp
import structlog
import yfinance as yf

logger = structlog.get_logger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


@dataclass
class NewsItem:
    """Normalized news article."""
    title: str
    source: str          # "newsapi", "google_news", "yahoo_finance"
    url: str = ""
    published_at: str = ""
    description: str = ""
    asset: str = ""      # ticker this relates to
    market: str = ""     # "us" or "india"


@dataclass
class NewsFetchResult:
    """Result of a news fetch operation."""
    items: list[NewsItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class NewsFetcher:
    """Aggregates news from multiple sources."""

    def __init__(self, newsapi_key: str | None = None):
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY", "")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_all(
        self,
        asset: str,
        market: str = "us",
        max_results: int = 50,
    ) -> NewsFetchResult:
        """Fetch news from all available sources for an asset.

        Args:
            asset: Ticker symbol (e.g. "AAPL", "RELIANCE").
            market: "us" or "india".
            max_results: Max total headlines to return.

        Returns:
            NewsFetchResult with deduplicated NewsItems.
        """
        tasks = [
            self.fetch_google_rss(asset, market),
            self.fetch_yahoo(asset, market),
        ]

        # Only use NewsAPI if key is available
        if self.newsapi_key:
            tasks.append(self.fetch_newsapi(asset, market))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined = NewsFetchResult()
        seen_titles: set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                combined.errors.append(str(result))
                continue
            if isinstance(result, NewsFetchResult):
                combined.errors.extend(result.errors)
                for item in result.items:
                    # Deduplicate by title (normalized)
                    key = item.title.strip().lower()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        combined.items.append(item)

        # Trim to max
        combined.items = combined.items[:max_results]

        logger.info(
            "news.fetch_all_complete",
            asset=asset,
            market=market,
            count=combined.count,
            errors=len(combined.errors),
        )
        return combined

    async def fetch_newsapi(
        self,
        asset: str,
        market: str = "us",
        max_results: int = 20,
    ) -> NewsFetchResult:
        """Fetch from NewsAPI.org.

        Requires NEWSAPI_KEY environment variable.
        Free tier: 100 requests/day, 1-month history.
        """
        result = NewsFetchResult()

        if not self.newsapi_key:
            result.errors.append("NewsAPI key not configured")
            return result

        # Build query based on market
        query = self._build_search_query(asset, market)
        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": str(max_results),
            "language": "en",
            "apiKey": self.newsapi_key,
        }

        try:
            session = await self._get_session()
            async with session.get(
                f"{NEWSAPI_BASE}/everything", params=params,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.errors.append(
                        f"NewsAPI returned {resp.status}: {body[:200]}"
                    )
                    return result

                data = await resp.json()

            for article in data.get("articles", []):
                result.items.append(NewsItem(
                    title=article.get("title", ""),
                    source="newsapi",
                    url=article.get("url", ""),
                    published_at=article.get("publishedAt", ""),
                    description=article.get("description", "") or "",
                    asset=asset,
                    market=market,
                ))

            logger.info(
                "news.newsapi_fetched",
                asset=asset,
                count=len(result.items),
            )

        except asyncio.TimeoutError:
            result.errors.append("NewsAPI request timed out")
        except Exception as e:
            result.errors.append(f"NewsAPI error: {e}")
            logger.error("news.newsapi_error", asset=asset, error=str(e))

        return result

    async def fetch_google_rss(
        self,
        asset: str,
        market: str = "us",
        max_results: int = 20,
    ) -> NewsFetchResult:
        """Fetch from Google News RSS feed (no API key required)."""
        result = NewsFetchResult()

        query = self._build_search_query(asset, market)
        url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
        if market == "india":
            url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en&gl=IN&ceid=IN:en"

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    result.errors.append(f"Google News RSS returned {resp.status}")
                    return result

                text = await resp.text()

            # Parse RSS XML
            root = ET.fromstring(text)
            channel = root.find("channel")
            if channel is None:
                result.errors.append("No channel in Google News RSS")
                return result

            for item in channel.findall("item")[:max_results]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                description = item.findtext("description", "")

                result.items.append(NewsItem(
                    title=title,
                    source="google_news",
                    url=link,
                    published_at=pub_date,
                    description=description or "",
                    asset=asset,
                    market=market,
                ))

            logger.info(
                "news.google_rss_fetched",
                asset=asset,
                count=len(result.items),
            )

        except asyncio.TimeoutError:
            result.errors.append("Google News RSS request timed out")
        except ET.ParseError as e:
            result.errors.append(f"Google News RSS parse error: {e}")
        except Exception as e:
            result.errors.append(f"Google News RSS error: {e}")
            logger.error("news.google_rss_error", asset=asset, error=str(e))

        return result

    async def fetch_yahoo(
        self,
        asset: str,
        market: str = "us",
        max_results: int = 10,
    ) -> NewsFetchResult:
        """Fetch news from Yahoo Finance via yfinance."""
        result = NewsFetchResult()

        symbol = asset
        if market == "india" and not asset.endswith(".NS"):
            symbol = f"{asset}.NS"

        try:
            loop = asyncio.get_event_loop()
            news_items = await loop.run_in_executor(
                None, lambda: self._get_yahoo_news(symbol),
            )

            for item in news_items[:max_results]:
                result.items.append(NewsItem(
                    title=item.get("title", ""),
                    source="yahoo_finance",
                    url=item.get("link", ""),
                    published_at=self._format_yahoo_timestamp(
                        item.get("providerPublishTime", 0)
                    ),
                    description="",
                    asset=asset,
                    market=market,
                ))

            logger.info(
                "news.yahoo_fetched",
                asset=asset,
                count=len(result.items),
            )

        except Exception as e:
            result.errors.append(f"Yahoo Finance news error: {e}")
            logger.error("news.yahoo_error", asset=asset, error=str(e))

        return result

    def _get_yahoo_news(self, symbol: str) -> list[dict[str, Any]]:
        """Synchronous Yahoo Finance news fetch."""
        try:
            ticker = yf.Ticker(symbol)
            return list(ticker.news or [])
        except Exception:
            return []

    def _build_search_query(self, asset: str, market: str) -> str:
        """Build a search query string for an asset."""
        # Map well-known tickers to company names for better search results
        company_names = {
            # US
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google Alphabet",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "META": "Meta Facebook",
            "AMD": "AMD",
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            # India
            "RELIANCE": "Reliance Industries",
            "TCS": "TCS Tata Consultancy",
            "INFY": "Infosys",
            "HDFCBANK": "HDFC Bank",
            "ICICIBANK": "ICICI Bank",
            "SBIN": "State Bank India SBI",
            "BHARTIARTL": "Bharti Airtel",
            "ITC": "ITC Limited",
            "KOTAKBANK": "Kotak Mahindra Bank",
            "LT": "Larsen Toubro L&T",
        }

        name = company_names.get(asset, asset)
        return f"{name} stock market"

    def _format_yahoo_timestamp(self, ts: int | float) -> str:
        """Convert Unix timestamp to ISO string."""
        if not ts:
            return ""
        try:
            return datetime.utcfromtimestamp(ts).isoformat()
        except (ValueError, OSError):
            return ""

    async def fetch_market_news(
        self,
        assets: list[str],
        market: str = "us",
        max_per_asset: int = 20,
    ) -> dict[str, NewsFetchResult]:
        """Fetch news for multiple assets concurrently."""
        tasks = [
            self.fetch_all(asset, market, max_results=max_per_asset)
            for asset in assets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, NewsFetchResult] = {}
        for asset, result in zip(assets, results):
            if isinstance(result, Exception):
                output[asset] = NewsFetchResult(errors=[str(result)])
            else:
                output[asset] = result

        return output
