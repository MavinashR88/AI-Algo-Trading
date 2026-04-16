"""Tests for sentiment analysis engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from data.news_fetcher import NewsItem
from data.sentiment import SentimentAnalyzer, SentimentResult, SentimentScore


class TestSentimentScore:
    """Test SentimentScore dataclass."""

    def test_create_score(self):
        score = SentimentScore(
            headline="Apple beats earnings",
            score=0.7,
            reasoning="Strong Q2 results",
            source="newsapi",
            asset="AAPL",
            market="us",
        )
        assert score.score == 0.7
        assert score.headline == "Apple beats earnings"

    def test_minimal_score(self):
        score = SentimentScore(headline="Test", score=0.0, reasoning="Neutral")
        assert score.source == ""
        assert score.asset == ""


class TestSentimentResult:
    """Test SentimentResult dataclass."""

    def test_empty_result(self):
        result = SentimentResult()
        assert result.avg_score == 0.0
        assert result.is_neutral

    def test_bullish_result(self):
        result = SentimentResult(avg_score=0.5)
        assert result.is_bullish
        assert not result.is_bearish
        assert not result.is_neutral

    def test_bearish_result(self):
        result = SentimentResult(avg_score=-0.5)
        assert result.is_bearish
        assert not result.is_bullish
        assert not result.is_neutral

    def test_neutral_boundaries(self):
        assert SentimentResult(avg_score=0.3).is_neutral
        assert SentimentResult(avg_score=-0.3).is_neutral
        assert not SentimentResult(avg_score=0.31).is_neutral
        assert not SentimentResult(avg_score=-0.31).is_neutral


class TestSentimentAnalyzerInit:
    """Test SentimentAnalyzer initialization."""

    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        analyzer = SentimentAnalyzer(api_key="")
        assert analyzer.enabled is False

    def test_enabled_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        analyzer = SentimentAnalyzer()
        assert analyzer.enabled is True

    def test_explicit_key(self):
        analyzer = SentimentAnalyzer(api_key="explicit-key")
        assert analyzer.enabled is True
        assert analyzer.api_key == "explicit-key"

    def test_default_model(self):
        analyzer = SentimentAnalyzer(api_key="key")
        assert "claude" in analyzer.model.lower()

    def test_custom_model(self):
        analyzer = SentimentAnalyzer(api_key="key", model="claude-haiku-4-5-20251001")
        assert analyzer.model == "claude-haiku-4-5-20251001"


class TestResponseParsing:
    """Test Claude response parsing."""

    def setup_method(self):
        self.analyzer = SentimentAnalyzer(api_key="test")

    def test_parse_clean_json(self):
        raw = json.dumps([
            {"index": 0, "score": 0.7, "reasoning": "Positive earnings"},
            {"index": 1, "score": -0.3, "reasoning": "Lawsuit filed"},
        ])
        scores = self.analyzer._parse_response(
            raw,
            headlines=["AAPL beats", "AAPL sued"],
            sources=["newsapi", "google_news"],
            asset="AAPL",
            market="us",
        )
        assert len(scores) == 2
        assert scores[0].score == 0.7
        assert scores[0].headline == "AAPL beats"
        assert scores[0].source == "newsapi"
        assert scores[1].score == -0.3

    def test_parse_with_markdown_fences(self):
        raw = """```json
[
  {"index": 0, "score": 0.5, "reasoning": "Good news"}
]
```"""
        scores = self.analyzer._parse_response(
            raw, headlines=["Test"], sources=["test"], asset="AAPL", market="us",
        )
        assert len(scores) == 1
        assert scores[0].score == 0.5

    def test_parse_with_surrounding_text(self):
        raw = """Here are the results:
[{"index": 0, "score": -0.2, "reasoning": "Mixed signals"}]
Hope this helps!"""
        scores = self.analyzer._parse_response(
            raw, headlines=["Test"], sources=["test"], asset="AAPL", market="us",
        )
        assert len(scores) == 1
        assert scores[0].score == -0.2

    def test_score_clamping(self):
        """Scores outside [-1, 1] are clamped."""
        raw = json.dumps([
            {"index": 0, "score": 1.5, "reasoning": "Way too positive"},
            {"index": 1, "score": -2.0, "reasoning": "Way too negative"},
        ])
        scores = self.analyzer._parse_response(
            raw, headlines=["A", "B"], sources=["", ""], asset="X", market="us",
        )
        assert scores[0].score == 1.0
        assert scores[1].score == -1.0

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty list."""
        scores = self.analyzer._parse_response(
            "This is not JSON at all",
            headlines=["Test"],
            sources=["test"],
            asset="AAPL",
            market="us",
        )
        assert scores == []

    def test_parse_empty_response(self):
        scores = self.analyzer._parse_response(
            "", headlines=[], sources=[], asset="AAPL", market="us",
        )
        assert scores == []


class TestAnalyzeHeadlines:
    """Test headline analysis flow (mocked Claude)."""

    @pytest.mark.asyncio
    async def test_disabled_returns_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        analyzer = SentimentAnalyzer(api_key="")
        result = await analyzer.analyze_headlines(["Test headline"], asset="AAPL")
        assert result.errors
        assert "not configured" in result.errors[0]

    @pytest.mark.asyncio
    async def test_empty_headlines(self):
        analyzer = SentimentAnalyzer(api_key="test-key")
        result = await analyzer.analyze_headlines([], asset="AAPL")
        assert result.scores == []
        assert result.avg_score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_with_mocked_claude(self):
        """Full analysis flow with mocked Claude response."""
        analyzer = SentimentAnalyzer(api_key="test-key")

        mock_response = json.dumps([
            {"index": 0, "score": 0.6, "reasoning": "Earnings beat"},
            {"index": 1, "score": -0.4, "reasoning": "Lawsuit concern"},
        ])

        with patch.object(analyzer, "_call_claude", return_value=mock_response):
            result = await analyzer.analyze_headlines(
                ["Apple beats earnings", "Apple faces lawsuit"],
                asset="AAPL",
                market="us",
            )
            assert len(result.scores) == 2
            assert result.avg_score == pytest.approx(0.1, abs=0.01)
            assert result.model_used == analyzer.model

    @pytest.mark.asyncio
    async def test_analyze_news_items(self):
        """Analysis works with NewsItem objects."""
        analyzer = SentimentAnalyzer(api_key="test-key")

        items = [
            NewsItem(title="Strong Q2 Report", source="newsapi", asset="AAPL", market="us"),
            NewsItem(title="Product Launch", source="google_news", asset="AAPL", market="us"),
        ]

        mock_response = json.dumps([
            {"index": 0, "score": 0.7, "reasoning": "Strong results"},
            {"index": 1, "score": 0.5, "reasoning": "New product positive"},
        ])

        with patch.object(analyzer, "_call_claude", return_value=mock_response):
            result = await analyzer.analyze_headlines(items, asset="AAPL", market="us")
            assert len(result.scores) == 2
            assert result.scores[0].source == "newsapi"
            assert result.scores[1].source == "google_news"

    @pytest.mark.asyncio
    async def test_analyze_stores_in_db(self, test_db):
        """Sentiment scores are stored in database."""
        analyzer = SentimentAnalyzer(api_key="test-key", db=test_db)

        mock_response = json.dumps([
            {"index": 0, "score": 0.8, "reasoning": "Great news"},
        ])

        with patch.object(analyzer, "_call_claude", return_value=mock_response):
            await analyzer.analyze_headlines(
                ["Apple soars"], asset="AAPL", market="us",
            )

        # Verify stored in DB
        from db import queries
        latest = queries.get_latest_sentiment(test_db, "us", "AAPL")
        assert latest is not None
        assert latest["score"] == 0.8
        assert latest["asset"] == "AAPL"

    @pytest.mark.asyncio
    async def test_analyze_handles_api_error(self):
        """API errors are captured gracefully."""
        analyzer = SentimentAnalyzer(api_key="test-key")

        with patch.object(analyzer, "_call_claude", side_effect=Exception("API timeout")):
            result = await analyzer.analyze_headlines(
                ["Test headline"], asset="AAPL", market="us",
            )
            assert result.errors
            assert "API timeout" in result.errors[0]
            assert result.scores == []


class TestScoreSingle:
    """Test single headline scoring."""

    @pytest.mark.asyncio
    async def test_score_single(self):
        analyzer = SentimentAnalyzer(api_key="test-key")

        mock_response = json.dumps([
            {"index": 0, "score": 0.9, "reasoning": "Very bullish"},
        ])

        with patch.object(analyzer, "_call_claude", return_value=mock_response):
            score = await analyzer.score_single(
                "Apple reports record revenue", asset="AAPL", market="us",
            )
            assert score is not None
            assert score.score == 0.9

    @pytest.mark.asyncio
    async def test_score_single_disabled(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        analyzer = SentimentAnalyzer(api_key="")
        score = await analyzer.score_single("Test", asset="AAPL")
        assert score is None


class TestDBIntegration:
    """Test database integration methods."""

    @pytest.mark.asyncio
    async def test_get_market_sentiment(self, test_db):
        """Get average sentiment from DB."""
        from db import queries

        queries.insert_sentiment(test_db, market="us", asset="AAPL", score=0.5,
                                 headline="Good", source="test")
        queries.insert_sentiment(test_db, market="us", asset="AAPL", score=0.7,
                                 headline="Great", source="test")

        analyzer = SentimentAnalyzer(db=test_db, api_key="test")
        avg = await analyzer.get_market_sentiment("AAPL", "us")
        assert avg is not None
        assert avg == pytest.approx(0.6, abs=0.01)

    @pytest.mark.asyncio
    async def test_get_latest_score(self, test_db):
        """Get latest sentiment score from DB."""
        from db import queries

        queries.insert_sentiment(test_db, market="us", asset="MSFT", score=0.3,
                                 headline="OK", source="test")
        queries.insert_sentiment(test_db, market="us", asset="MSFT", score=0.8,
                                 headline="Great", source="test")

        analyzer = SentimentAnalyzer(db=test_db, api_key="test")
        latest = await analyzer.get_latest_score("MSFT", "us")
        assert latest is not None
        assert latest["score"] == 0.8

    @pytest.mark.asyncio
    async def test_no_db_returns_none(self):
        """Without DB, methods return None."""
        analyzer = SentimentAnalyzer(api_key="test")
        assert await analyzer.get_market_sentiment("AAPL") is None
        assert await analyzer.get_latest_score("AAPL") is None
