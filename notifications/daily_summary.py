"""
Daily summary generator.

Computes end-of-day performance metrics and stores them in
daily_summaries table. Triggers email/SMS notifications.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from db.database import Database
from db import queries

logger = structlog.get_logger(__name__)


class DailySummaryGenerator:
    """Generates end-of-day trading summaries for each market."""

    def __init__(self, db: Database):
        self.db = db

    def generate(self, market: str, date_str: str | None = None) -> dict[str, Any]:
        """Generate daily summary for a market.

        Args:
            market: "us" or "india".
            date_str: Date string (YYYY-MM-DD), defaults to today.

        Returns:
            Summary dict with all metrics.
        """
        date_str = date_str or datetime.utcnow().strftime("%Y-%m-%d")

        # Get all trades for the day
        trades = queries.get_trades_for_date(self.db, market, date_str)
        closed_trades = [t for t in trades if t.get("status") == "closed"]

        # Get bot state for capital info
        state = queries.get_bot_state(self.db, market)
        current_capital = state["capital"] if state else 0.0

        # Compute metrics
        trades_taken = len(trades)
        winning = [t for t in closed_trades if (t.get("net_pnl") or t.get("pnl") or 0) > 0]
        losing = [t for t in closed_trades if (t.get("net_pnl") or t.get("pnl") or 0) <= 0]

        total_pnl = sum(t.get("pnl", 0) or 0 for t in closed_trades)
        total_fees = sum(t.get("fees", 0) or 0 for t in closed_trades)
        net_pnl = sum(t.get("net_pnl", 0) or 0 for t in closed_trades)

        starting_capital = current_capital - net_pnl
        ending_capital = current_capital
        pnl_pct = net_pnl / starting_capital if starting_capital > 0 else 0

        win_rate = len(winning) / len(closed_trades) if closed_trades else None

        best_pnl = max((t.get("net_pnl") or t.get("pnl") or 0 for t in closed_trades), default=None)
        worst_pnl = min((t.get("net_pnl") or t.get("pnl") or 0 for t in closed_trades), default=None)

        # Max drawdown for the day
        max_dd = state.get("current_drawdown_pct", 0) if state else 0

        summary = {
            "market": market,
            "date": date_str,
            "trades_taken": trades_taken,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "starting_capital": round(starting_capital, 2),
            "ending_capital": round(ending_capital, 2),
            "best_trade_pnl": round(best_pnl, 2) if best_pnl is not None else None,
            "worst_trade_pnl": round(worst_pnl, 2) if worst_pnl is not None else None,
            "max_drawdown_pct": round(max_dd, 4),
            "mode": state["mode"] if state else "paper",
        }

        # Store in DB
        queries.upsert_daily_summary(
            self.db,
            market=market,
            summary_date=date_str,
            starting_capital=summary["starting_capital"],
            ending_capital=summary["ending_capital"],
            pnl=summary["net_pnl"],
            pnl_pct=summary["pnl_pct"],
            trades_taken=summary["trades_taken"],
            winning_trades=summary["winning_trades"],
            losing_trades=summary["losing_trades"],
            win_rate=summary["win_rate"],
            max_drawdown_pct=summary["max_drawdown_pct"],
            best_trade_pnl=summary["best_trade_pnl"],
            worst_trade_pnl=summary["worst_trade_pnl"],
            fees_total=summary["total_fees"],
            mode=summary["mode"],
        )

        # Update bot state stats
        all_closed = queries.get_recent_trades(self.db, market, limit=1000)
        all_closed = [t for t in all_closed if t.get("status") == "closed"]
        total_wins = sum(1 for t in all_closed if (t.get("net_pnl") or 0) > 0)
        total_all = len(all_closed)
        overall_win_rate = total_wins / total_all if total_all > 0 else 0

        queries.update_bot_state(
            self.db, market,
            total_trades=total_all,
            winning_trades=total_wins,
            losing_trades=total_all - total_wins,
            win_rate=round(overall_win_rate, 4),
        )

        logger.info(
            "daily_summary.generated",
            market=market,
            date=date_str,
            trades=trades_taken,
            net_pnl=summary["net_pnl"],
        )

        return summary

    def format_text(self, summary: dict[str, Any]) -> str:
        """Format summary as plain text for SMS."""
        s = summary
        lines = [
            f"Daily Summary: {s['market'].upper()} ({s['date']})",
            f"Trades: {s['trades_taken']} | W:{s['winning_trades']} L:{s['losing_trades']}",
            f"Win Rate: {s['win_rate']:.0%}" if s["win_rate"] is not None else "Win Rate: N/A",
            f"P&L: {s['net_pnl']:+.2f} ({s['pnl_pct']:+.2%})",
            f"Capital: {s['ending_capital']:,.2f}",
            f"Best: {s['best_trade_pnl']:+.2f}" if s["best_trade_pnl"] is not None else "",
            f"Worst: {s['worst_trade_pnl']:+.2f}" if s["worst_trade_pnl"] is not None else "",
        ]
        return "\n".join(line for line in lines if line)

    def format_html(self, summary: dict[str, Any]) -> str:
        """Format summary as HTML for email."""
        s = summary
        pnl_color = "#3fb950" if s["net_pnl"] >= 0 else "#f85149"
        wr = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else "N/A"

        return f"""
        <div style="font-family: monospace; max-width: 600px;">
            <h2 style="color: #58a6ff;">Daily Summary: {s['market'].upper()}</h2>
            <p style="color: #8b949e;">{s['date']} | Mode: {s['mode']}</p>
            <table style="width:100%; border-collapse:collapse;">
                <tr><td style="padding:8px;">Trades</td><td style="padding:8px;font-weight:bold;">{s['trades_taken']}</td></tr>
                <tr><td style="padding:8px;">Win/Loss</td><td style="padding:8px;">{s['winning_trades']}W / {s['losing_trades']}L</td></tr>
                <tr><td style="padding:8px;">Win Rate</td><td style="padding:8px;">{wr}</td></tr>
                <tr><td style="padding:8px;">Net P&L</td><td style="padding:8px;color:{pnl_color};font-weight:bold;">{s['net_pnl']:+.2f} ({s['pnl_pct']:+.2%})</td></tr>
                <tr><td style="padding:8px;">Capital</td><td style="padding:8px;">{s['ending_capital']:,.2f}</td></tr>
                <tr><td style="padding:8px;">Max Drawdown</td><td style="padding:8px;">{s['max_drawdown_pct']:.2%}</td></tr>
            </table>
        </div>
        """
