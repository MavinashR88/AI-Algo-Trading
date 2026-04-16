"""
Parameterized query helpers for common operations.

Every function accepts a Database instance and uses parameterized queries
to prevent SQL injection. No raw string interpolation anywhere.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from db.database import Database


# =========================================================================
# Trades
# =========================================================================

def insert_trade(
    db: Database,
    *,
    market: str,
    asset: str,
    direction: str,
    entry_price: float,
    quantity: float,
    entry_time: str,
    strategy: str,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    signal_id: int | None = None,
    confidence: float | None = None,
    sentiment_score: float | None = None,
    reasoning: str | None = None,
    order_id: str | None = None,
    mode: str = "paper",
    fees: float = 0.0,
) -> int:
    return db.execute_insert(
        """INSERT INTO trades
        (market, asset, direction, entry_price, quantity, entry_time,
         strategy, stop_loss, take_profit, signal_id, confidence,
         sentiment_score, reasoning, order_id, mode, fees)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (market, asset, direction, entry_price, quantity, entry_time,
         strategy, stop_loss, take_profit, signal_id, confidence,
         sentiment_score, reasoning, order_id, mode, fees),
    )


def close_trade(
    db: Database,
    trade_id: int,
    *,
    exit_price: float,
    exit_time: str,
    pnl: float,
    pnl_pct: float,
    net_pnl: float,
    fees: float,
    trailing_stop: float | None = None,
) -> None:
    db.execute(
        """UPDATE trades
        SET exit_price = ?, exit_time = ?, status = 'closed',
            pnl = ?, pnl_pct = ?, net_pnl = ?, fees = ?,
            trailing_stop = ?
        WHERE id = ?""",
        (exit_price, exit_time, pnl, pnl_pct, net_pnl, fees,
         trailing_stop, trade_id),
    )


def get_open_trades(db: Database, market: str | None = None) -> list[dict]:
    if market:
        rows = db.execute(
            "SELECT * FROM trades WHERE status = 'open' AND market = ?",
            (market,),
        )
    else:
        rows = db.execute("SELECT * FROM trades WHERE status = 'open'")
    return [dict(r) for r in rows]


def get_recent_trades(db: Database, market: str, limit: int = 10) -> list[dict]:
    rows = db.execute(
        "SELECT * FROM trades WHERE market = ? ORDER BY created_at DESC LIMIT ?",
        (market, limit),
    )
    return [dict(r) for r in rows]


def get_trades_for_date(db: Database, market: str, date_str: str) -> list[dict]:
    rows = db.execute(
        """SELECT * FROM trades
        WHERE market = ? AND date(entry_time) = ?
        ORDER BY entry_time""",
        (market, date_str),
    )
    return [dict(r) for r in rows]


def count_trades_today(db: Database, market: str) -> int:
    row = db.execute_one(
        """SELECT COUNT(*) as cnt FROM trades
        WHERE market = ? AND date(entry_time) = date('now')""",
        (market,),
    )
    return row["cnt"] if row else 0


# =========================================================================
# Signals
# =========================================================================

def insert_signal(
    db: Database,
    *,
    market: str,
    asset: str,
    direction: str,
    confidence: float,
    timeframe: str,
    strategy: str,
    sentiment_score: float | None = None,
    reasoning: str | None = None,
    generated_at: str | None = None,
) -> int:
    generated_at = generated_at or datetime.utcnow().isoformat()
    return db.execute_insert(
        """INSERT INTO signals
        (market, asset, direction, confidence, timeframe, strategy,
         sentiment_score, reasoning, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (market, asset, direction, confidence, timeframe, strategy,
         sentiment_score, reasoning, generated_at),
    )


def mark_signal_acted(db: Database, signal_id: int) -> None:
    db.execute(
        "UPDATE signals SET acted_on = 1 WHERE id = ?",
        (signal_id,),
    )


def mark_signal_rejected(db: Database, signal_id: int, reason: str) -> None:
    db.execute(
        "UPDATE signals SET acted_on = 0, reject_reason = ? WHERE id = ?",
        (reason, signal_id),
    )


# =========================================================================
# Sentiment
# =========================================================================

def insert_sentiment(
    db: Database,
    *,
    market: str,
    asset: str,
    score: float,
    headline: str | None = None,
    source: str | None = None,
    url: str | None = None,
    model_used: str | None = None,
    raw_response: str | None = None,
    scored_at: str | None = None,
) -> int:
    scored_at = scored_at or datetime.utcnow().isoformat()
    return db.execute_insert(
        """INSERT INTO sentiment_scores
        (market, asset, score, headline, source, url, model_used,
         raw_response, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (market, asset, score, headline, source, url, model_used,
         raw_response, scored_at),
    )


def get_latest_sentiment(db: Database, market: str, asset: str) -> dict | None:
    row = db.execute_one(
        """SELECT * FROM sentiment_scores
        WHERE market = ? AND asset = ?
        ORDER BY scored_at DESC LIMIT 1""",
        (market, asset),
    )
    return dict(row) if row else None


def get_asset_sentiment_avg(
    db: Database, market: str, asset: str, hours: int = 4
) -> float | None:
    row = db.execute_one(
        """SELECT AVG(score) as avg_score FROM sentiment_scores
        WHERE market = ? AND asset = ?
        AND scored_at >= datetime('now', ? || ' hours')""",
        (market, asset, f"-{hours}"),
    )
    return row["avg_score"] if row and row["avg_score"] is not None else None


# =========================================================================
# Bot State
# =========================================================================

def get_bot_state(db: Database, market: str) -> dict | None:
    row = db.execute_one(
        "SELECT * FROM bot_state WHERE market = ?", (market,)
    )
    return dict(row) if row else None


def update_bot_state(db: Database, market: str, **kwargs: Any) -> None:
    if not kwargs:
        return
    set_clauses = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [market]
    db.execute(
        f"UPDATE bot_state SET {set_clauses}, updated_at = datetime('now') WHERE market = ?",
        tuple(values),
    )


# =========================================================================
# Daily Summaries
# =========================================================================

def upsert_daily_summary(
    db: Database,
    *,
    market: str,
    summary_date: str,
    starting_capital: float,
    ending_capital: float,
    pnl: float,
    pnl_pct: float,
    trades_taken: int,
    winning_trades: int,
    losing_trades: int,
    win_rate: float | None,
    max_drawdown_pct: float | None,
    best_trade_pnl: float | None,
    worst_trade_pnl: float | None,
    fees_total: float,
    mode: str,
) -> int:
    return db.execute_insert(
        """INSERT INTO daily_summaries
        (market, summary_date, starting_capital, ending_capital, pnl, pnl_pct,
         trades_taken, winning_trades, losing_trades, win_rate, max_drawdown_pct,
         best_trade_pnl, worst_trade_pnl, fees_total, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market, summary_date) DO UPDATE SET
            ending_capital = excluded.ending_capital,
            pnl = excluded.pnl,
            pnl_pct = excluded.pnl_pct,
            trades_taken = excluded.trades_taken,
            winning_trades = excluded.winning_trades,
            losing_trades = excluded.losing_trades,
            win_rate = excluded.win_rate,
            max_drawdown_pct = excluded.max_drawdown_pct,
            best_trade_pnl = excluded.best_trade_pnl,
            worst_trade_pnl = excluded.worst_trade_pnl,
            fees_total = excluded.fees_total""",
        (market, summary_date, starting_capital, ending_capital, pnl, pnl_pct,
         trades_taken, winning_trades, losing_trades, win_rate, max_drawdown_pct,
         best_trade_pnl, worst_trade_pnl, fees_total, mode),
    )


# =========================================================================
# Error Logs
# =========================================================================

def insert_error_log(
    db: Database,
    *,
    level: str,
    module: str,
    message: str,
    traceback: str | None = None,
    context: dict | None = None,
) -> int:
    return db.execute_insert(
        """INSERT INTO error_logs (level, module, message, traceback, context)
        VALUES (?, ?, ?, ?, ?)""",
        (level, module, message, traceback, json.dumps(context) if context else None),
    )


# =========================================================================
# Backtest Results
# =========================================================================

def insert_backtest_result(
    db: Database,
    *,
    market: str,
    strategy: str,
    start_date: str,
    end_date: str,
    total_trades: int,
    winning_trades: int,
    losing_trades: int,
    win_rate: float,
    sharpe_ratio: float | None,
    max_drawdown_pct: float | None,
    profit_factor: float | None,
    avg_trade_duration: str | None,
    max_consecutive_losses: int | None,
    total_pnl: float,
    total_fees: float,
    net_pnl: float,
    parameters: dict | None = None,
) -> int:
    return db.execute_insert(
        """INSERT INTO backtest_results
        (market, strategy, start_date, end_date, total_trades,
         winning_trades, losing_trades, win_rate, sharpe_ratio,
         max_drawdown_pct, profit_factor, avg_trade_duration,
         max_consecutive_losses, total_pnl, total_fees, net_pnl, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (market, strategy, start_date, end_date, total_trades,
         winning_trades, losing_trades, win_rate, sharpe_ratio,
         max_drawdown_pct, profit_factor, avg_trade_duration,
         max_consecutive_losses, total_pnl, total_fees, net_pnl,
         json.dumps(parameters) if parameters else None),
    )
