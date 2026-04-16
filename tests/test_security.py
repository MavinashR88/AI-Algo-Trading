"""Tests for security hardening: encryption, validation, rate limiting."""

from __future__ import annotations

import time

import pytest

from config.encryption import (
    generate_key,
    encrypt_value,
    decrypt_value,
    is_encrypted,
    encrypt_config,
    decrypt_config,
    rotate_key,
)
from config.security import (
    validate_market,
    validate_direction,
    validate_ticker,
    validate_quantity,
    validate_price,
    validate_trade_params,
    sanitize_key,
    hash_key,
    validate_api_key_format,
    RateLimiter,
)


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

class TestEncryption:
    @pytest.fixture
    def key(self):
        return generate_key()

    def test_generate_key(self):
        key = generate_key()
        assert len(key) > 20
        assert isinstance(key, str)

    def test_encrypt_decrypt_roundtrip(self, key):
        original = "my-secret-api-key-12345"
        encrypted = encrypt_value(original, key)
        assert encrypted.startswith("ENC:")
        assert original not in encrypted
        decrypted = decrypt_value(encrypted, key)
        assert decrypted == original

    def test_is_encrypted(self):
        assert is_encrypted("ENC:somedata")
        assert not is_encrypted("plain text")
        assert not is_encrypted("")

    def test_decrypt_plain_text_passthrough(self, key):
        result = decrypt_value("not encrypted", key)
        assert result == "not encrypted"

    def test_different_keys_fail(self):
        key1 = generate_key()
        key2 = generate_key()
        encrypted = encrypt_value("secret", key1)
        with pytest.raises(Exception):
            decrypt_value(encrypted, key2)


class TestConfigEncryption:
    @pytest.fixture
    def key(self):
        return generate_key()

    def test_encrypt_config(self, key):
        config = {
            "name": "test",
            "api_key": "sk-123456",
            "database": {
                "host": "localhost",
                "password": "db-pass",
            },
        }
        encrypted = encrypt_config(config, key)
        assert is_encrypted(encrypted["api_key"])
        assert is_encrypted(encrypted["database"]["password"])
        assert encrypted["name"] == "test"  # Not encrypted
        assert encrypted["database"]["host"] == "localhost"  # Not encrypted

    def test_decrypt_config(self, key):
        config = {
            "api_key": "sk-123456",
            "nested": {"password": "secret"},
        }
        encrypted = encrypt_config(config, key)
        decrypted = decrypt_config(encrypted, key)
        assert decrypted["api_key"] == "sk-123456"
        assert decrypted["nested"]["password"] == "secret"

    def test_rotate_key(self, key):
        new_key = generate_key()
        config = {"api_key": encrypt_value("my-key", key)}
        rotated = rotate_key(config, key, new_key)
        assert is_encrypted(rotated["api_key"])
        # Decrypt with new key
        result = decrypt_value(rotated["api_key"], new_key)
        assert result == "my-key"


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_markets(self):
        assert validate_market("us")
        assert validate_market("india")
        assert not validate_market("uk")
        assert not validate_market("")

    def test_valid_directions(self):
        assert validate_direction("long")
        assert validate_direction("short")
        assert not validate_direction("up")

    def test_valid_tickers(self):
        assert validate_ticker("AAPL")
        assert validate_ticker("RELIANCE")
        assert validate_ticker("RELIANCE.NS")
        assert validate_ticker("SPY")
        assert not validate_ticker("")
        assert not validate_ticker("aapl")  # lowercase
        assert not validate_ticker("A" * 25)  # too long
        assert not validate_ticker("AA PL")  # space

    def test_valid_quantity(self):
        assert validate_quantity(1)
        assert validate_quantity(100.5)
        assert not validate_quantity(0)
        assert not validate_quantity(-1)
        assert not validate_quantity(2_000_000)

    def test_valid_price(self):
        assert validate_price(150.0)
        assert validate_price(0.01)
        assert not validate_price(0)
        assert not validate_price(-10)

    def test_validate_trade_params(self):
        valid, _ = validate_trade_params("us", "AAPL", "long", 10, 150.0)
        assert valid

        valid, err = validate_trade_params("uk", "AAPL", "long", 10, 150.0)
        assert not valid
        assert "market" in err.lower()

        valid, err = validate_trade_params("us", "aapl", "long", 10, 150.0)
        assert not valid
        assert "ticker" in err.lower()


# ---------------------------------------------------------------------------
# API Key Sanitization
# ---------------------------------------------------------------------------

class TestApiKeySanitization:
    def test_sanitize_key(self):
        assert sanitize_key("sk-ant-api03-abcdef12345") == "***2345"
        assert sanitize_key("short") == "***"
        assert sanitize_key("") == "***"

    def test_hash_key(self):
        h = hash_key("my-key")
        assert len(h) == 16
        assert hash_key("my-key") == h  # deterministic

    def test_validate_api_key_format(self):
        assert validate_api_key_format("sk-ant-abc123", "anthropic")
        assert not validate_api_key_format("bad-key", "anthropic")
        assert validate_api_key_format("AC" + "a" * 32, "twilio_sid")
        assert validate_api_key_format("somethinglong", "unknown")  # generic
        assert not validate_api_key_format("", "any")


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter()
        rl.configure("test", max_calls=3, window_seconds=60)
        assert rl.call_if_allowed("test")
        assert rl.call_if_allowed("test")
        assert rl.call_if_allowed("test")
        assert not rl.call_if_allowed("test")  # 4th call blocked

    def test_unknown_service_allowed(self):
        rl = RateLimiter()
        assert rl.call_if_allowed("unknown_service")

    def test_remaining_calls(self):
        rl = RateLimiter()
        rl.configure("test", max_calls=5, window_seconds=60)
        assert rl.remaining("test") == 5
        rl.record("test")
        rl.record("test")
        assert rl.remaining("test") == 3

    def test_check_without_record(self):
        rl = RateLimiter()
        rl.configure("test", max_calls=2, window_seconds=60)
        assert rl.check("test")  # Just check, don't record
        assert rl.check("test")  # Still allowed
        rl.record("test")
        rl.record("test")
        assert not rl.check("test")  # Now blocked

    def test_default_limits_exist(self):
        rl = RateLimiter()
        assert rl.remaining("newsapi") > 0
        assert rl.remaining("anthropic") > 0
        assert rl.remaining("yfinance") > 0
