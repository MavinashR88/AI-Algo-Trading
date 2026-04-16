"""
SMS alerts via Twilio.

Keeps messages short and actionable. All credentials from env vars.
"""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)


class SMSAlert:
    """Twilio SMS sender."""

    def __init__(self) -> None:
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER", "")
        self.to_number = os.environ.get("TWILIO_TO_NUMBER", "")
        self._enabled = bool(
            self.account_sid and self.auth_token
            and self.from_number and self.to_number
        )
        self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_client(self):
        if self._client is None and self._enabled:
            from twilio.rest import Client
            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    async def send(self, message: str) -> bool:
        """Send an SMS. Returns True on success."""
        if not self._enabled:
            logger.warning("sms.disabled", reason="missing credentials")
            return False

        # Twilio SMS max is 1600 chars, truncate if needed
        if len(message) > 1500:
            message = message[:1497] + "..."

        try:
            client = self._get_client()
            client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number,
            )
            logger.info("sms.sent", length=len(message))
            return True
        except Exception as e:
            logger.error("sms.failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Pre-built SMS templates (short format)
    # ------------------------------------------------------------------

    async def send_trade_alert(
        self,
        asset: str,
        direction: str,
        price: float,
        pnl: float | None,
        win_rate: float,
        currency: str,
    ) -> bool:
        sym = "$" if currency == "USD" else "INR "
        pnl_str = f"{sym}{pnl:+,.2f}" if pnl is not None else "Open"
        msg = (
            f"Trade: {direction.upper()} {asset} {sym}{price:,.2f} | "
            f"P&L: {pnl_str} | WR: {win_rate:.0%}"
        )
        return await self.send(msg)

    async def send_halt_alert(self, market: str, reason: str) -> bool:
        return await self.send(f"HALT {market.upper()}: {reason}")

    async def send_graduation_alert(self, market: str) -> bool:
        return await self.send(
            f"GRADUATION: {market.upper()} ready for live! Confirm in dashboard."
        )

    async def send_error_alert(self, module: str, error: str) -> bool:
        return await self.send(f"ERR [{module}]: {error[:100]}")

    async def send_daily_summary(
        self,
        market: str,
        pnl: float,
        trades: int,
        win_rate: float,
        currency: str,
    ) -> bool:
        sym = "$" if currency == "USD" else "INR "
        return await self.send(
            f"Daily {market.upper()}: P&L {sym}{pnl:+,.2f} | "
            f"{trades} trades | WR: {win_rate:.0%}"
        )

    async def send_bot_status(self, status: str) -> bool:
        return await self.send(f"Bot {status.upper()}")
