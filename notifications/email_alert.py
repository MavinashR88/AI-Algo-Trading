"""
Email alerts via Gmail SMTP.

Sends HTML-formatted emails for all trading events.
All credentials come from environment variables.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import structlog

logger = structlog.get_logger(__name__)


class EmailAlert:
    """Gmail SMTP email sender."""

    def __init__(self) -> None:
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender = os.environ.get("EMAIL_SENDER", "")
        self.password = os.environ.get("EMAIL_PASSWORD", "")  # App password
        self.recipients = [
            r.strip()
            for r in os.environ.get("EMAIL_RECIPIENTS", "").split(",")
            if r.strip()
        ]
        self._enabled = bool(self.sender and self.password and self.recipients)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send(self, subject: str, body_html: str) -> bool:
        """Send an HTML email. Returns True on success."""
        if not self._enabled:
            logger.warning("email.disabled", reason="missing credentials")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[AI-Trading] {subject}"
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)

        # Plain text fallback
        plain = body_html.replace("<br>", "\n").replace("</tr>", "\n")
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            logger.info("email.sent", subject=subject, recipients=len(self.recipients))
            return True
        except Exception as e:
            logger.error("email.failed", subject=subject, error=str(e))
            return False

    # ------------------------------------------------------------------
    # Pre-built email templates
    # ------------------------------------------------------------------

    async def send_trade_alert(
        self,
        asset: str,
        direction: str,
        price: float,
        quantity: float,
        pnl: float | None,
        market: str,
        currency: str,
    ) -> bool:
        symbol = "$" if currency == "USD" else "₹"
        pnl_str = f"{symbol}{pnl:+,.2f}" if pnl is not None else "N/A"
        html = f"""
        <html><body>
        <h2>Trade Executed</h2>
        <table border="1" cellpadding="8" style="border-collapse:collapse;">
        <tr><td><b>Asset</b></td><td>{asset}</td></tr>
        <tr><td><b>Direction</b></td><td>{direction.upper()}</td></tr>
        <tr><td><b>Price</b></td><td>{symbol}{price:,.2f}</td></tr>
        <tr><td><b>Quantity</b></td><td>{quantity}</td></tr>
        <tr><td><b>P&L</b></td><td>{pnl_str}</td></tr>
        <tr><td><b>Market</b></td><td>{market.upper()}</td></tr>
        </table>
        </body></html>
        """
        return await self.send(f"Trade: {direction.upper()} {asset} @ {symbol}{price:,.2f}", html)

    async def send_drawdown_warning(self, market: str, current_pct: float, limit_pct: float) -> bool:
        html = f"""
        <html><body>
        <h2 style="color:orange;">Drawdown Warning</h2>
        <p>Market: <b>{market.upper()}</b></p>
        <p>Current drawdown: <b>{current_pct:.1%}</b> (limit: {limit_pct:.1%})</p>
        <p>Bot will halt if limit is breached.</p>
        </body></html>
        """
        return await self.send(f"Drawdown Warning: {market.upper()} at {current_pct:.1%}", html)

    async def send_halt_alert(self, market: str, reason: str) -> bool:
        html = f"""
        <html><body>
        <h2 style="color:red;">Bot Halted</h2>
        <p>Market: <b>{market.upper()}</b></p>
        <p>Reason: <b>{reason}</b></p>
        <p>Bot will not execute trades until the halt condition is cleared.</p>
        </body></html>
        """
        return await self.send(f"HALT: {market.upper()} - {reason}", html)

    async def send_graduation_alert(self, market: str, stats: dict) -> bool:
        html = f"""
        <html><body>
        <h2 style="color:green;">Graduation Criteria Met!</h2>
        <p>Market: <b>{market.upper()}</b> is ready to go live.</p>
        <table border="1" cellpadding="8" style="border-collapse:collapse;">
        <tr><td><b>Total Trades</b></td><td>{stats.get('total_trades', 'N/A')}</td></tr>
        <tr><td><b>Win Rate</b></td><td>{stats.get('win_rate', 0):.1%}</td></tr>
        <tr><td><b>Sharpe Ratio</b></td><td>{stats.get('sharpe', 'N/A')}</td></tr>
        <tr><td><b>Max Drawdown</b></td><td>{stats.get('max_drawdown', 0):.1%}</td></tr>
        <tr><td><b>Days Traded</b></td><td>{stats.get('days', 'N/A')}</td></tr>
        </table>
        <p><b>Human confirmation required before switching to live mode.</b></p>
        </body></html>
        """
        return await self.send(f"GRADUATION READY: {market.upper()}", html)

    async def send_daily_summary(
        self,
        market: str,
        summary: dict,
        currency: str,
    ) -> bool:
        symbol = "$" if currency == "USD" else "₹"
        html = f"""
        <html><body>
        <h2>Daily Summary — {market.upper()}</h2>
        <table border="1" cellpadding="8" style="border-collapse:collapse;">
        <tr><td><b>Date</b></td><td>{summary.get('date', 'N/A')}</td></tr>
        <tr><td><b>P&L</b></td><td>{symbol}{summary.get('pnl', 0):+,.2f}</td></tr>
        <tr><td><b>Trades</b></td><td>{summary.get('trades', 0)}</td></tr>
        <tr><td><b>Win Rate</b></td><td>{summary.get('win_rate', 0):.1%}</td></tr>
        <tr><td><b>Drawdown</b></td><td>{summary.get('drawdown', 0):.1%}</td></tr>
        <tr><td><b>Portfolio Value</b></td><td>{symbol}{summary.get('portfolio', 0):,.2f}</td></tr>
        </table>
        </body></html>
        """
        return await self.send(f"Daily Summary: {market.upper()}", html)

    async def send_error_alert(self, module: str, error: str) -> bool:
        html = f"""
        <html><body>
        <h2 style="color:red;">Error Occurred</h2>
        <p>Module: <b>{module}</b></p>
        <p>Error: <code>{error}</code></p>
        </body></html>
        """
        return await self.send(f"ERROR in {module}", html)

    async def send_bot_status(self, status: str) -> bool:
        html = f"""
        <html><body>
        <h2>Bot Status: {status.upper()}</h2>
        </body></html>
        """
        return await self.send(f"Bot {status.upper()}", html)
