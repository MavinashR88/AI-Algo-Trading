"""
Tax export — generates trade reports for tax filing.

Supports:
- CSV export of all trades for a tax year
- Wash sale detection (US market)
- Short-term vs long-term capital gains classification
- Summary by asset with total P&L
- India STT/charges breakdown
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

from db.database import Database
from db import queries

logger = structlog.get_logger(__name__)


@dataclass
class TaxTrade:
    """Trade formatted for tax reporting."""
    asset: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    holding_period: str  # "short_term" or "long_term"
    is_wash_sale: bool
    strategy: str
    market: str
    mode: str


@dataclass
class TaxSummary:
    """Summary of tax-relevant trades."""
    market: str
    year: int
    total_trades: int = 0
    total_gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_net_pnl: float = 0.0
    short_term_gains: float = 0.0
    long_term_gains: float = 0.0
    wash_sale_adjustments: float = 0.0
    trades: list[TaxTrade] = field(default_factory=list)

    @property
    def taxable_income(self) -> float:
        return self.total_net_pnl - self.wash_sale_adjustments


class TaxExporter:
    """Generates tax reports from trade history."""

    # US: short-term < 1 year, long-term >= 1 year
    # India: short-term < 1 year for equity delivery
    SHORT_TERM_DAYS = 365

    def __init__(self, db: Database):
        self.db = db

    def generate_report(
        self,
        market: str,
        year: int,
        mode: str | None = None,
    ) -> TaxSummary:
        """Generate tax report for a market and year.

        Args:
            market: "us" or "india".
            year: Tax year (e.g. 2026).
            mode: Filter by "paper" or "live", or None for both.

        Returns:
            TaxSummary with all trades and metrics.
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Get all closed trades for the year
        all_trades = self.db.execute(
            """SELECT * FROM trades
            WHERE market = ? AND status = 'closed'
            AND exit_time >= ? AND exit_time <= ?
            ORDER BY exit_time""",
            (market, start_date, end_date + "T23:59:59"),
        )

        if mode:
            all_trades = [t for t in all_trades if t["mode"] == mode]

        summary = TaxSummary(market=market, year=year)
        recent_losses: dict[str, list[dict]] = {}  # for wash sale detection

        for trade in all_trades:
            trade = dict(trade)
            tax_trade = self._process_trade(trade, recent_losses)
            summary.trades.append(tax_trade)
            summary.total_trades += 1
            summary.total_gross_pnl += tax_trade.gross_pnl
            summary.total_fees += tax_trade.fees
            summary.total_net_pnl += tax_trade.net_pnl

            if tax_trade.holding_period == "short_term":
                summary.short_term_gains += tax_trade.net_pnl
            else:
                summary.long_term_gains += tax_trade.net_pnl

            if tax_trade.is_wash_sale:
                summary.wash_sale_adjustments += abs(tax_trade.net_pnl)

        logger.info(
            "tax_export.report_generated",
            market=market,
            year=year,
            trades=summary.total_trades,
            net_pnl=round(summary.total_net_pnl, 2),
        )

        return summary

    def _process_trade(
        self,
        trade: dict,
        recent_losses: dict[str, list[dict]],
    ) -> TaxTrade:
        """Convert a DB trade record to a TaxTrade."""
        entry_date = trade.get("entry_time", "")[:10]
        exit_date = trade.get("exit_time", "")[:10]

        # Holding period
        try:
            entry_dt = datetime.fromisoformat(trade["entry_time"])
            exit_dt = datetime.fromisoformat(trade["exit_time"])
            days_held = (exit_dt - entry_dt).days
            holding_period = "long_term" if days_held >= self.SHORT_TERM_DAYS else "short_term"
        except (ValueError, TypeError):
            holding_period = "short_term"
            days_held = 0

        gross_pnl = trade.get("pnl", 0) or 0
        fees = trade.get("fees", 0) or 0
        net_pnl = trade.get("net_pnl", 0) or 0

        # Wash sale detection (US only): loss on asset, then buy same asset within 30 days
        is_wash = False
        asset = trade["asset"]
        market = trade["market"]

        if market == "us" and net_pnl < 0:
            # Record this loss
            if asset not in recent_losses:
                recent_losses[asset] = []
            recent_losses[asset].append(trade)

        if market == "us" and net_pnl < 0:
            # Check if we bought the same asset within 30 days before or after
            try:
                exit_dt = datetime.fromisoformat(trade["exit_time"])
                window_start = exit_dt - timedelta(days=30)
                window_end = exit_dt + timedelta(days=30)

                # Check for repurchase in the window
                repurchases = self.db.execute(
                    """SELECT id FROM trades
                    WHERE asset = ? AND market = ? AND direction = 'long'
                    AND entry_time >= ? AND entry_time <= ?
                    AND id != ?""",
                    (asset, market, window_start.isoformat(), window_end.isoformat(), trade["id"]),
                )
                is_wash = len(repurchases) > 0
            except (ValueError, TypeError):
                pass

        return TaxTrade(
            asset=asset,
            direction=trade.get("direction", ""),
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=trade.get("entry_price", 0) or 0,
            exit_price=trade.get("exit_price", 0) or 0,
            quantity=trade.get("quantity", 0) or 0,
            gross_pnl=round(gross_pnl, 2),
            fees=round(fees, 2),
            net_pnl=round(net_pnl, 2),
            holding_period=holding_period,
            is_wash_sale=is_wash,
            strategy=trade.get("strategy", ""),
            market=market,
            mode=trade.get("mode", "paper"),
        )

    def export_csv(self, summary: TaxSummary) -> str:
        """Export trades as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Asset", "Direction", "Entry Date", "Exit Date",
            "Entry Price", "Exit Price", "Quantity",
            "Gross P&L", "Fees", "Net P&L",
            "Holding Period", "Wash Sale", "Strategy", "Mode",
        ])

        for t in summary.trades:
            writer.writerow([
                t.asset, t.direction, t.entry_date, t.exit_date,
                f"{t.entry_price:.4f}", f"{t.exit_price:.4f}", f"{t.quantity:.2f}",
                f"{t.gross_pnl:.2f}", f"{t.fees:.2f}", f"{t.net_pnl:.2f}",
                t.holding_period, "Yes" if t.is_wash_sale else "No",
                t.strategy, t.mode,
            ])

        # Summary rows
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        writer.writerow(["Total Trades", summary.total_trades])
        writer.writerow(["Gross P&L", f"{summary.total_gross_pnl:.2f}"])
        writer.writerow(["Total Fees", f"{summary.total_fees:.2f}"])
        writer.writerow(["Net P&L", f"{summary.total_net_pnl:.2f}"])
        writer.writerow(["Short-term Gains", f"{summary.short_term_gains:.2f}"])
        writer.writerow(["Long-term Gains", f"{summary.long_term_gains:.2f}"])
        if summary.market == "us":
            writer.writerow(["Wash Sale Adjustments", f"{summary.wash_sale_adjustments:.2f}"])
            writer.writerow(["Taxable Income", f"{summary.taxable_income:.2f}"])

        return output.getvalue()

    def export_asset_summary(self, summary: TaxSummary) -> list[dict[str, Any]]:
        """Summarize P&L by asset."""
        by_asset: dict[str, dict[str, Any]] = {}

        for t in summary.trades:
            if t.asset not in by_asset:
                by_asset[t.asset] = {
                    "asset": t.asset,
                    "trades": 0,
                    "gross_pnl": 0.0,
                    "fees": 0.0,
                    "net_pnl": 0.0,
                }
            by_asset[t.asset]["trades"] += 1
            by_asset[t.asset]["gross_pnl"] += t.gross_pnl
            by_asset[t.asset]["fees"] += t.fees
            by_asset[t.asset]["net_pnl"] += t.net_pnl

        return sorted(by_asset.values(), key=lambda x: x["net_pnl"], reverse=True)
