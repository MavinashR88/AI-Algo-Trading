"""
AI-powered sentiment analysis engine using Claude.

Scores news headlines/descriptions on a -1.0 to +1.0 scale:
  -1.0 = extremely bearish
   0.0 = neutral
  +1.0 = extremely bullish

Batches headlines for efficient API usage. Results are stored
in the sentiment_scores table via db.queries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import structlog

from data.news_fetcher import NewsItem
from db.database import Database
from db import queries

logger = structlog.get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"

SENTIMENT_PROMPT = """You are a financial sentiment analyst. Score each headline on a scale from -1.0 to +1.0:

- **-1.0**: Extremely bearish (bankruptcy, fraud, massive losses, regulatory crackdown)
- **-0.5**: Moderately bearish (earnings miss, downgrade, lawsuit, management exodus)
- **0.0**: Neutral (routine announcement, no clear market impact)
- **+0.5**: Moderately bullish (earnings beat, upgrade, new partnership, expansion)
- **+1.0**: Extremely bullish (breakthrough product, massive contract win, transformative acquisition)

Consider the asset ticker when scoring — the same headline may be bullish for one stock and bearish for a competitor.

Return ONLY a JSON array of objects. Each object must have:
- "index": the headline number (starting from 0)
- "score": float between -1.0 and +1.0
- "reasoning": one short sentence explaining the score

Example response:
[
  {"index": 0, "score": 0.6, "reasoning": "Earnings beat expectations with strong guidance."},
  {"index": 1, "score": -0.3, "reasoning": "Supply chain disruption may impact margins."}
]

IMPORTANT: Return ONLY the JSON array, no markdown, no explanation outside the array."""


@dataclass
class SentimentScore:
    """Sentiment score for a single headline."""
    headline: str
    score: float
    reasoning: str
    source: str = ""
    asset: str = ""
    market: str = ""


@dataclass
class SentimentResult:
    """Batch sentiment analysis result."""
    scores: list[SentimentScore] = field(default_factory=list)
    avg_score: float = 0.0
    model_used: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.avg_score > 0.3

    @property
    def is_bearish(self) -> bool:
        return self.avg_score < -0.3

    @property
    def is_neutral(self) -> bool:
        return -0.3 <= self.avg_score <= 0.3


class SentimentAnalyzer:
    """Claude-powered sentiment analysis engine."""

    def __init__(
        self,
        db: Database | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.db = db
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("SENTIMENT_MODEL", DEFAULT_MODEL)
        self._client: Any = None

    @property
    def enabled(self) -> bool:
        """Sentiment analysis requires an Anthropic API key."""
        return bool(self.api_key)

    def _get_client(self) -> Any:
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    async def analyze_headlines(
        self,
        headlines: list[NewsItem] | list[str],
        asset: str = "",
        market: str = "",
    ) -> SentimentResult:
        """Score a batch of headlines using Claude.

        Args:
            headlines: List of NewsItem objects or plain strings.
            asset: Ticker symbol for context.
            market: "us" or "india".

        Returns:
            SentimentResult with individual scores and average.
        """
        result = SentimentResult(model_used=self.model)

        if not self.enabled:
            result.errors.append("Anthropic API key not configured")
            return result

        if not headlines:
            return result

        # Normalize to strings
        headline_texts = []
        sources = []
        for h in headlines:
            if isinstance(h, NewsItem):
                headline_texts.append(h.title)
                sources.append(h.source)
            else:
                headline_texts.append(str(h))
                sources.append("")

        # Build the prompt
        numbered = "\n".join(
            f"{i}. [{asset}] {text}" for i, text in enumerate(headline_texts)
        )
        user_message = f"Asset: {asset} ({market.upper()} market)\n\nHeadlines:\n{numbered}"

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(
                None,
                lambda: self._call_claude(user_message),
            )

            scores = self._parse_response(raw_response, headline_texts, sources, asset, market)
            result.scores = scores

            if scores:
                result.avg_score = sum(s.score for s in scores) / len(scores)

            # Store in database
            if self.db is not None:
                for score in scores:
                    queries.insert_sentiment(
                        self.db,
                        market=market or "us",
                        asset=asset,
                        score=score.score,
                        headline=score.headline,
                        source=score.source,
                        model_used=self.model,
                        raw_response=raw_response[:500],
                    )

            logger.info(
                "sentiment.analyzed",
                asset=asset,
                market=market,
                count=len(scores),
                avg_score=round(result.avg_score, 3),
            )

        except Exception as e:
            result.errors.append(f"Sentiment analysis error: {e}")
            logger.error("sentiment.analysis_error", asset=asset, error=str(e))

        return result

    def _call_claude(self, user_message: str) -> str:
        """Synchronous Claude API call."""
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": f"{SENTIMENT_PROMPT}\n\n{user_message}"},
            ],
        )
        return response.content[0].text

    def _parse_response(
        self,
        raw: str,
        headlines: list[str],
        sources: list[str],
        asset: str,
        market: str,
    ) -> list[SentimentScore]:
        """Parse Claude's JSON response into SentimentScore objects."""
        scores = []

        # Strip any markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.error("sentiment.parse_failed", raw=text[:200])
                    return scores
            else:
                logger.error("sentiment.no_json_found", raw=text[:200])
                return scores

        for item in data:
            idx = item.get("index", -1)
            score_val = item.get("score", 0.0)
            reasoning = item.get("reasoning", "")

            # Clamp score to [-1, 1]
            score_val = max(-1.0, min(1.0, float(score_val)))

            headline = headlines[idx] if 0 <= idx < len(headlines) else ""
            source = sources[idx] if 0 <= idx < len(sources) else ""

            scores.append(SentimentScore(
                headline=headline,
                score=score_val,
                reasoning=reasoning,
                source=source,
                asset=asset,
                market=market,
            ))

        return scores

    async def score_single(
        self,
        headline: str,
        asset: str = "",
        market: str = "",
    ) -> SentimentScore | None:
        """Score a single headline. Convenience wrapper."""
        result = await self.analyze_headlines([headline], asset=asset, market=market)
        return result.scores[0] if result.scores else None

    async def get_market_sentiment(
        self,
        asset: str,
        market: str = "us",
        hours: int = 4,
    ) -> float | None:
        """Get average sentiment for an asset over recent hours from DB."""
        if self.db is None:
            return None
        return queries.get_asset_sentiment_avg(self.db, market, asset, hours=hours)

    async def get_latest_score(
        self,
        asset: str,
        market: str = "us",
    ) -> dict | None:
        """Get the most recent sentiment score from DB."""
        if self.db is None:
            return None
        return queries.get_latest_sentiment(self.db, market, asset)
