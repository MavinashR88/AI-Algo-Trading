"""Tests for daily summary and tax export modules."""

from __future__ import annotations

from datetime import datetime

import pytest

from db.database import Database
from db import queries
from notifications.daily_summary import DailySummaryGenerator
from notifications.tax_export import TaxExporter, TaxSummary, TaxTrade


# ---------------------------------------------------------------------------
# Daily Summary
# ---------------------------------------------------------------------------

class TestDailySummaryGenerator:
    def test_generate_empty_day(self, test_db: Database):
        gen = DailySummaryGenerator(test_db)
        summary = gen.generate("us", "2026-04-16")
        assert summary["trades_taken"] == 0
        assert summary["net_pnl"] == 0
        assert summary["market"] == "us"

    def test_generate_with_trades(self, test_db: Database):
        # Insert and close a trade
        tid = queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=150, quantity=10, entry_time="2026-04-16T10:00:00",
            strategy="momentum",
        )
        queries.close_trade(
            test_db, tid, exit_price=155, exit_time="2026-04-16T14:00:00",
            pnl=50, pnl_pct=0.033, net_pnl=49, fees=1,
        )
        queries.update_bot_state(test_db, "us", capital=100_049)

        gen = DailySummaryGenerator(test_db)
        summary = gen.generate("us", "2026-04-16")
        assert summary["trades_taken"] == 1
        assert summary["winning_trades"] == 1
        assert summary["net_pnl"] == 49.0

    def test_stores_in_db(self, test_db: Database):
        gen = DailySummaryGenerator(test_db)
        gen.generate("us", "2026-04-16")

        rows = test_db.execute(
            "SELECT * FROM daily_summaries WHERE market = 'us' AND summary_date = '2026-04-16'"
        )
        assert len(rows) == 1

    def test_upsert_updates(self, test_db: Database):
        gen = DailySummaryGenerator(test_db)
        gen.generate("us", "2026-04-16")
        gen.generate("us", "2026-04-16")  # Second call should update

        rows = test_db.execute(
            "SELECT * FROM daily_summaries WHERE market = 'us' AND summary_date = '2026-04-16'"
        )
        assert len(rows) == 1  # Not duplicated

    def test_format_text(self, test_db: Database):
        gen = DailySummaryGenerator(test_db)
        summary = gen.generate("us", "2026-04-16")
        text = gen.format_text(summary)
        assert "Daily Summary" in text
        assert "US" in text

    def test_format_html(self, test_db: Database):
        gen = DailySummaryGenerator(test_db)
        summary = gen.generate("us", "2026-04-16")
        html = gen.format_html(summary)
        assert "<h2" in html
        assert "US" in html


# ---------------------------------------------------------------------------
# Tax Export
# ---------------------------------------------------------------------------

class TestTaxExporter:
    def _create_closed_trade(self, db, market="us", asset="AAPL",
                              entry_price=150, exit_price=155,
                              pnl=50, net_pnl=49, fees=1,
                              entry_time="2026-04-16T10:00:00",
                              exit_time="2026-04-16T14:00:00"):
        tid = queries.insert_trade(
            db, market=market, asset=asset, direction="long",
            entry_price=entry_price, quantity=10,
            entry_time=entry_time, strategy="test",
        )
        queries.close_trade(
            db, tid, exit_price=exit_price,
            exit_time=exit_time,
            pnl=pnl, pnl_pct=pnl/entry_price/10,
            net_pnl=net_pnl, fees=fees,
        )
        return tid

    def test_empty_report(self, test_db: Database):
        exporter = TaxExporter(test_db)
        summary = exporter.generate_report("us", 2026)
        assert summary.total_trades == 0
        assert summary.total_net_pnl == 0.0

    def test_report_with_trades(self, test_db: Database):
        self._create_closed_trade(test_db)
        self._create_closed_trade(test_db, asset="MSFT", pnl=-20, net_pnl=-21, fees=1, exit_price=395)

        exporter = TaxExporter(test_db)
        summary = exporter.generate_report("us", 2026)
        assert summary.total_trades == 2
        assert summary.total_net_pnl == 28.0  # 49 + (-21)

    def test_short_term_classification(self, test_db: Database):
        self._create_closed_trade(test_db)
        exporter = TaxExporter(test_db)
        summary = exporter.generate_report("us", 2026)
        assert summary.trades[0].holding_period == "short_term"
        assert summary.short_term_gains == 49.0

    def test_csv_export(self, test_db: Database):
        self._create_closed_trade(test_db)
        exporter = TaxExporter(test_db)
        summary = exporter.generate_report("us", 2026)
        csv_str = exporter.export_csv(summary)
        assert "AAPL" in csv_str
        assert "SUMMARY" in csv_str
        assert "Net P&L" in csv_str

    def test_asset_summary(self, test_db: Database):
        self._create_closed_trade(test_db, asset="AAPL", net_pnl=49)
        self._create_closed_trade(test_db, asset="AAPL", net_pnl=30, exit_price=153,
                                   pnl=30, entry_time="2026-04-17T10:00:00",
                                   exit_time="2026-04-17T14:00:00")
        self._create_closed_trade(test_db, asset="MSFT", net_pnl=-10, exit_price=390,
                                   pnl=-10, entry_time="2026-04-17T10:00:00",
                                   exit_time="2026-04-17T14:00:00")

        exporter = TaxExporter(test_db)
        summary = exporter.generate_report("us", 2026)
        by_asset = exporter.export_asset_summary(summary)
        assert len(by_asset) == 2
        # AAPL should be first (highest P&L)
        assert by_asset[0]["asset"] == "AAPL"
        assert by_asset[0]["trades"] == 2

    def test_market_filter(self, test_db: Database):
        self._create_closed_trade(test_db, market="us")
        self._create_closed_trade(test_db, market="india", asset="RELIANCE",
                                   entry_price=2500, exit_price=2550, pnl=500,
                                   net_pnl=480, fees=20)

        exporter = TaxExporter(test_db)
        us_report = exporter.generate_report("us", 2026)
        india_report = exporter.generate_report("india", 2026)
        assert us_report.total_trades == 1
        assert india_report.total_trades == 1


class TestTaxSummary:
    def test_taxable_income(self):
        s = TaxSummary(market="us", year=2026, total_net_pnl=1000, wash_sale_adjustments=100)
        assert s.taxable_income == 900

    def test_no_wash_sales(self):
        s = TaxSummary(market="us", year=2026, total_net_pnl=1000)
        assert s.taxable_income == 1000


class TestTaxTrade:
    def test_create(self):
        t = TaxTrade(
            asset="AAPL", direction="long",
            entry_date="2026-04-16", exit_date="2026-04-16",
            entry_price=150, exit_price=155, quantity=10,
            gross_pnl=50, fees=1, net_pnl=49,
            holding_period="short_term", is_wash_sale=False,
            strategy="momentum", market="us", mode="paper",
        )
        assert t.net_pnl == 49
        assert not t.is_wash_sale
