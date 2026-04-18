"""
Hugging Face Spaces entry point — LIVE trading bot + dashboard.

Runs the full trading pipeline:
  1. Alpaca paper trading (US market)
  2. yfinance market data
  3. NewsAPI + Google RSS + Yahoo news
  4. VADER sentiment analysis (free, no API key)
  5. 8 trading strategies + hybrid ensemble
  6. Risk management + psychology guard
  7. Real-time FastAPI dashboard on port 7860
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import structlog
import uvicorn

from dashboard.api import app, set_database, broadcast
from db.database import Database
from db import queries
from data.market_data import MarketDataFetcher
from data.news_fetcher import NewsFetcher, NewsItem
from data.sentiment import SentimentScore, SentimentResult
from strategy.signal_generator import SignalGenerator
from strategy import indicators as ind
from risk.risk_manager import RiskManager
from agent.psychology_guard import PsychologyGuard

logger = structlog.get_logger("app")

US_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ", "AMD"]

RISK_CONFIG = {
    "max_risk_per_trade_pct": 0.01,
    "max_open_positions_per_market": 3,
    "max_trades_per_market_per_day": 10,
    "max_daily_drawdown_pct": 0.03,
    "max_weekly_drawdown_pct": 0.08,
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_rr_ratio": 2.0,
    "trailing_stop_activation_rr": 1.5,
    "trailing_stop_atr_multiplier": 1.5,
}

PSYCH_CONFIG = {
    "revenge_trade_lookback": 2,
    "revenge_trade_cooldown_minutes": 30,
    "min_signal_confidence": 0.65,
    "sentiment_override_threshold": -0.7,
    "no_averaging_down": True,
}

SCAN_INTERVAL_SECONDS = 300  # 5 minutes
NEWS_INTERVAL_SECONDS = 900  # 15 minutes


# ---------------------------------------------------------------------------
# VADER Sentiment (free, local, no API key)
# ---------------------------------------------------------------------------

class VaderSentiment:
    """VADER-based sentiment scoring — free alternative to Claude API."""

    def __init__(self, db: Database | None = None):
        self.db = db
        self._vader = None

    def _get_vader(self):
        if self._vader is None:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    async def analyze_headlines(
        self,
        headlines: list[NewsItem] | list[str],
        asset: str = "",
        market: str = "us",
    ) -> SentimentResult:
        vader = self._get_vader()
        scores: list[SentimentScore] = []

        for h in headlines:
            if isinstance(h, NewsItem):
                text, source = h.title, h.source
            else:
                text, source = str(h), ""

            vs = vader.polarity_scores(text)
            score_val = max(-1.0, min(1.0, vs["compound"]))

            scores.append(SentimentScore(
                headline=text,
                score=score_val,
                reasoning=f"pos={vs['pos']:.2f} neg={vs['neg']:.2f} neu={vs['neu']:.2f}",
                source=source,
                asset=asset,
                market=market,
            ))

            if self.db:
                try:
                    queries.insert_sentiment(
                        self.db, market=market, asset=asset,
                        score=score_val, headline=text, source=source,
                    )
                except Exception:
                    pass

        avg = sum(s.score for s in scores) / len(scores) if scores else 0.0
        return SentimentResult(scores=scores, avg_score=avg, model_used="vader")


# ---------------------------------------------------------------------------
# Trading Loop
# ---------------------------------------------------------------------------

async def trading_loop(db: Database) -> None:
    """Main trading loop — connects to Alpaca and trades the US watchlist."""

    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")

    if not alpaca_key or not alpaca_secret:
        logger.warning("trading_loop.no_alpaca_keys", msg="Alpaca keys not set — running dashboard only")
        await broadcast("log", {"message": "No Alpaca API keys — dashboard-only mode"})
        return

    from brokers.alpaca_broker import AlpacaBroker
    from brokers.base import OrderSide

    broker = AlpacaBroker(paper=True)
    data_fetcher = MarketDataFetcher(db=db)
    news_fetcher = NewsFetcher(newsapi_key=os.environ.get("NEWSAPI_KEY", ""))
    sentiment = VaderSentiment(db=db)
    signal_gen = SignalGenerator()
    risk_mgr = RiskManager(RISK_CONFIG, db=db)
    psych_guard = PsychologyGuard(PSYCH_CONFIG, db=db)

    # Connect to Alpaca
    try:
        await broker.connect()
        account = await broker.get_account()
        logger.info("trading_loop.connected", buying_power=account.buying_power)
        await broadcast("log", {"message": f"Alpaca connected — ${account.portfolio_value:,.2f} portfolio"})

        queries.update_bot_state(db, "us", capital=account.portfolio_value, peak_capital=account.portfolio_value)
    except Exception as e:
        logger.error("trading_loop.connect_failed", error=str(e))
        await broadcast("log", {"message": f"Alpaca connection failed: {e}"})
        return

    # Sentiment cache: {asset: (score, timestamp)}
    sentiment_cache: dict[str, tuple[float, datetime]] = {}
    news_rotation_idx = 0

    while True:
        try:
            market_open = False
            try:
                market_open = await broker.is_market_open()
            except Exception as e:
                logger.error("trading_loop.market_check_error", error=str(e))

            if market_open:
                await broadcast("log", {"message": "Market is OPEN — scanning..."})

                # Refresh account
                try:
                    account = await broker.get_account()
                    capital = account.portfolio_value
                    queries.update_bot_state(db, "us", capital=capital)
                except Exception as e:
                    logger.error("trading_loop.account_error", error=str(e))
                    capital = 100_000

                # Fetch news for 2-3 assets per cycle (rotate to conserve API limits)
                news_batch_size = 3
                news_assets = US_WATCHLIST[news_rotation_idx:news_rotation_idx + news_batch_size]
                news_rotation_idx = (news_rotation_idx + news_batch_size) % len(US_WATCHLIST)

                for asset in news_assets:
                    try:
                        news_result = await news_fetcher.fetch_all(asset, "us", max_results=10)
                        if news_result.items:
                            sent_result = await sentiment.analyze_headlines(
                                news_result.items, asset=asset, market="us",
                            )
                            sentiment_cache[asset] = (sent_result.avg_score, datetime.now(timezone.utc))
                            await broadcast("log", {
                                "message": f"Sentiment {asset}: {sent_result.avg_score:+.2f} ({len(news_result.items)} headlines)"
                            })
                    except Exception as e:
                        logger.error("trading_loop.news_error", asset=asset, error=str(e))

                # Scan each asset for signals
                for asset in US_WATCHLIST:
                    try:
                        df = await data_fetcher.get_latest_bars(asset, "us", "15m", 100)
                        if df is None or df.empty or len(df) < 30:
                            continue

                        # Get cached sentiment
                        sent_score = None
                        if asset in sentiment_cache:
                            score, ts = sentiment_cache[asset]
                            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
                            if age_hours < 2:
                                sent_score = score

                        # Generate best signal
                        signal = signal_gen.get_best_signal(df, asset, "us", sentiment_score=sent_score)

                        if signal and signal.is_actionable:
                            entry_price = float(df["Close"].iloc[-1])
                            atr_series = ind.atr(df["High"], df["Low"], df["Close"], 14)
                            atr_val = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else entry_price * 0.02

                            # Risk check
                            assessment = risk_mgr.assess_trade(
                                market="us", asset=asset, direction=signal.direction,
                                entry_price=entry_price, current_atr=atr_val, capital=capital,
                            )

                            if assessment.approved:
                                # Psychology check
                                psych_check = psych_guard.check(
                                    market="us", asset=asset, direction=signal.direction,
                                    confidence=signal.confidence, sentiment_score=sent_score,
                                )

                                if psych_check.approved:
                                    # EXECUTE TRADE
                                    side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
                                    qty = max(1, int(assessment.position_size))

                                    try:
                                        order = await broker.place_order(
                                            asset=asset, side=side, quantity=qty,
                                        )

                                        # Record signal
                                        signal_id = queries.insert_signal(
                                            db, market="us", asset=asset,
                                            direction=signal.direction,
                                            confidence=signal.confidence,
                                            timeframe="15m", strategy=signal.strategy,
                                            sentiment_score=sent_score,
                                            reasoning=signal.reasoning or "",
                                        )
                                        queries.mark_signal_acted(db, signal_id)

                                        # Record trade
                                        fill_price = order.filled_avg_price or entry_price
                                        trade_id = queries.insert_trade(
                                            db, market="us", asset=asset,
                                            direction=signal.direction,
                                            entry_price=fill_price, quantity=qty,
                                            entry_time=datetime.now(timezone.utc).isoformat(),
                                            strategy=signal.strategy,
                                            stop_loss=assessment.stop_loss,
                                            take_profit=assessment.take_profit,
                                            signal_id=signal_id,
                                            confidence=signal.confidence,
                                            order_id=str(order.order_id),
                                        )

                                        msg = (
                                            f"TRADE: {signal.direction.upper()} {qty} {asset} "
                                            f"@ ${fill_price:.2f} | strategy={signal.strategy} "
                                            f"| conf={signal.confidence:.2f}"
                                        )
                                        logger.info("trading_loop.trade_executed", asset=asset,
                                                     direction=signal.direction, qty=qty, price=fill_price)
                                        await broadcast("trade", {
                                            "asset": asset, "direction": signal.direction,
                                            "price": fill_price, "quantity": qty,
                                            "strategy": signal.strategy, "message": msg,
                                        })
                                        await broadcast("log", {"message": msg})

                                    except Exception as e:
                                        logger.error("trading_loop.order_error", asset=asset, error=str(e))
                                        await broadcast("log", {"message": f"Order error {asset}: {e}"})
                                else:
                                    await broadcast("log", {
                                        "message": f"PSYCH BLOCKED {asset}: {psych_check.reason}"
                                    })
                            else:
                                if assessment.reason:
                                    await broadcast("log", {
                                        "message": f"RISK BLOCKED {asset}: {assessment.reason}"
                                    })

                    except Exception as e:
                        logger.error("trading_loop.scan_error", asset=asset, error=str(e))

                # Check open positions for exits
                await check_positions(broker, db, risk_mgr)

                # Update drawdown
                try:
                    risk_mgr.update_drawdown("us", capital)
                except Exception:
                    pass

            else:
                await broadcast("log", {"message": "Market CLOSED — waiting..."})

        except Exception as e:
            logger.error("trading_loop.cycle_error", error=str(e))
            await broadcast("log", {"message": f"Loop error: {e}"})

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def check_positions(broker, db: Database, risk_mgr: RiskManager) -> None:
    """Check open Alpaca positions against stop-loss and take-profit levels."""
    try:
        positions = await broker.get_positions()
        open_trades = queries.get_open_trades(db, "us")

        for pos in positions:
            # Find matching DB trade
            matching = [t for t in open_trades if t["asset"] == pos.asset]
            if not matching:
                continue

            trade = matching[0]
            stop_loss = trade.get("stop_loss")
            take_profit = trade.get("take_profit")
            direction = trade.get("direction", "long")
            entry_price = trade.get("entry_price", pos.avg_entry_price)

            should_close = False
            close_reason = ""

            if direction == "long":
                if stop_loss and pos.current_price <= stop_loss:
                    should_close = True
                    close_reason = f"Stop loss hit: ${pos.current_price:.2f} <= ${stop_loss:.2f}"
                elif take_profit and pos.current_price >= take_profit:
                    should_close = True
                    close_reason = f"Take profit hit: ${pos.current_price:.2f} >= ${take_profit:.2f}"
            else:
                if stop_loss and pos.current_price >= stop_loss:
                    should_close = True
                    close_reason = f"Stop loss hit: ${pos.current_price:.2f} >= ${stop_loss:.2f}"
                elif take_profit and pos.current_price <= take_profit:
                    should_close = True
                    close_reason = f"Take profit hit: ${pos.current_price:.2f} <= ${take_profit:.2f}"

            if should_close:
                try:
                    await broker.close_position(pos.asset)

                    pnl = pos.unrealized_pnl
                    pnl_pct = pos.unrealized_pnl_pct
                    fees = abs(pnl) * 0.001

                    queries.close_trade(
                        db, trade["id"],
                        exit_price=pos.current_price,
                        exit_time=datetime.now(timezone.utc).isoformat(),
                        pnl=pnl, pnl_pct=pnl_pct,
                        net_pnl=pnl - fees, fees=fees,
                    )

                    from agent.psychology_guard import PsychologyGuard
                    psych = PsychologyGuard(PSYCH_CONFIG, db=db)
                    psych.record_trade_result("us", is_win=(pnl > 0))

                    msg = f"CLOSED {pos.asset}: {close_reason} | P&L: ${pnl:+,.2f}"
                    logger.info("trading_loop.position_closed", asset=pos.asset, pnl=pnl)
                    await broadcast("trade", {"asset": pos.asset, "pnl": pnl, "message": msg})
                    await broadcast("log", {"message": msg})

                except Exception as e:
                    logger.error("trading_loop.close_error", asset=pos.asset, error=str(e))

    except Exception as e:
        logger.error("trading_loop.positions_error", error=str(e))


# ---------------------------------------------------------------------------
# App Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """Launch the trading loop as a background task when FastAPI starts."""
    db_path = os.environ.get("DB_PATH", "/tmp/trading.db")
    db = Database(db_path)
    db.initialize()

    queries.update_bot_state(db, "us", capital=100_000, peak_capital=100_000)
    queries.update_bot_state(db, "india", capital=0, peak_capital=0)

    set_database(db)

    await broadcast("log", {"message": "AI Algo Trading Bot starting..."})

    asyncio.create_task(trading_loop(db))


def main() -> None:
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
