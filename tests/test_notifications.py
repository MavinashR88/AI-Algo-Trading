"""Tests for notification modules (email + SMS)."""

from __future__ import annotations

import pytest

from notifications.email_alert import EmailAlert
from notifications.sms_alert import SMSAlert


class TestEmailAlert:
    """Test email alert module."""

    def test_disabled_without_credentials(self, monkeypatch):
        """Email is disabled when credentials are missing."""
        monkeypatch.delenv("EMAIL_SENDER", raising=False)
        monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
        monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)
        alert = EmailAlert()
        assert alert.enabled is False

    def test_enabled_with_credentials(self, monkeypatch):
        """Email is enabled when all credentials are set."""
        monkeypatch.setenv("EMAIL_SENDER", "test@gmail.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "testpassword")
        monkeypatch.setenv("EMAIL_RECIPIENTS", "r1@test.com,r2@test.com")
        alert = EmailAlert()
        assert alert.enabled is True
        assert len(alert.recipients) == 2

    @pytest.mark.asyncio
    async def test_send_returns_false_when_disabled(self, monkeypatch):
        """Sending when disabled returns False without error."""
        monkeypatch.delenv("EMAIL_SENDER", raising=False)
        monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
        alert = EmailAlert()
        result = await alert.send("Test Subject", "<p>Test</p>")
        assert result is False


class TestSMSAlert:
    """Test SMS alert module."""

    def test_disabled_without_credentials(self, monkeypatch):
        """SMS is disabled when credentials are missing."""
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TWILIO_FROM_NUMBER", raising=False)
        monkeypatch.delenv("TWILIO_TO_NUMBER", raising=False)
        alert = SMSAlert()
        assert alert.enabled is False

    def test_enabled_with_credentials(self, monkeypatch):
        """SMS is enabled when all credentials are set."""
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "testtoken")
        monkeypatch.setenv("TWILIO_FROM_NUMBER", "+1234567890")
        monkeypatch.setenv("TWILIO_TO_NUMBER", "+0987654321")
        alert = SMSAlert()
        assert alert.enabled is True

    @pytest.mark.asyncio
    async def test_send_returns_false_when_disabled(self, monkeypatch):
        """Sending when disabled returns False without error."""
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        alert = SMSAlert()
        result = await alert.send("Test message")
        assert result is False
