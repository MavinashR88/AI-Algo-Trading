"""
Configuration encryption utilities.

Encrypts sensitive config values (API keys, passwords) at rest.
Uses Fernet symmetric encryption from the cryptography library.

Usage:
    # Generate a key (once, store in env var)
    key = generate_key()

    # Encrypt a value
    encrypted = encrypt_value("my-api-key", key)

    # Decrypt a value
    original = decrypt_value(encrypted, key)

    # Encrypt/decrypt entire config sections
    encrypted_config = encrypt_config(config_dict, key, fields=["api_key", "password"])
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _get_fernet():
    """Lazy import Fernet to avoid hard dependency."""
    try:
        from cryptography.fernet import Fernet
        return Fernet
    except ImportError:
        raise RuntimeError(
            "cryptography package required for encryption. "
            "Install: pip install cryptography"
        )


def generate_key() -> str:
    """Generate a new Fernet encryption key.

    Returns:
        Base64-encoded key string. Store this securely (e.g., env var).
    """
    Fernet = _get_fernet()
    return Fernet.generate_key().decode()


def encrypt_value(value: str, key: str) -> str:
    """Encrypt a string value.

    Args:
        value: Plaintext string to encrypt.
        key: Fernet key (from generate_key() or env var).

    Returns:
        Base64-encoded encrypted string prefixed with 'ENC:'.
    """
    Fernet = _get_fernet()
    f = Fernet(key.encode() if isinstance(key, str) else key)
    encrypted = f.encrypt(value.encode())
    return f"ENC:{encrypted.decode()}"


def decrypt_value(encrypted: str, key: str) -> str:
    """Decrypt an encrypted value.

    Args:
        encrypted: Encrypted string (with 'ENC:' prefix).
        key: Same Fernet key used for encryption.

    Returns:
        Original plaintext string.
    """
    if not encrypted.startswith("ENC:"):
        return encrypted  # Not encrypted, return as-is

    Fernet = _get_fernet()
    f = Fernet(key.encode() if isinstance(key, str) else key)
    token = encrypted[4:]  # Strip 'ENC:' prefix
    return f.decrypt(token.encode()).decode()


def is_encrypted(value: str) -> bool:
    """Check if a value is encrypted (has ENC: prefix)."""
    return isinstance(value, str) and value.startswith("ENC:")


def encrypt_config(
    config: dict[str, Any],
    key: str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Encrypt sensitive fields in a config dict.

    Args:
        config: Configuration dictionary.
        key: Fernet encryption key.
        fields: List of field names to encrypt. If None, encrypts common sensitive fields.

    Returns:
        New dict with specified fields encrypted.
    """
    if fields is None:
        fields = [
            "api_key", "secret_key", "password", "token",
            "auth_token", "access_token", "api_secret",
        ]

    result = {}
    for k, v in config.items():
        if isinstance(v, dict):
            result[k] = encrypt_config(v, key, fields)
        elif isinstance(v, str) and any(f in k.lower() for f in fields) and not is_encrypted(v):
            result[k] = encrypt_value(v, key)
        else:
            result[k] = v

    return result


def decrypt_config(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Decrypt all encrypted values in a config dict.

    Recursively traverses the dict and decrypts any value with 'ENC:' prefix.
    """
    result = {}
    for k, v in config.items():
        if isinstance(v, dict):
            result[k] = decrypt_config(v, key)
        elif isinstance(v, str) and is_encrypted(v):
            try:
                result[k] = decrypt_value(v, key)
            except Exception as e:
                logger.error("encryption.decrypt_failed", field=k, error=str(e))
                result[k] = v  # Return encrypted value on failure
        else:
            result[k] = v

    return result


def get_encryption_key() -> str | None:
    """Get encryption key from environment variable.

    Returns:
        Key string, or None if not configured.
    """
    return os.environ.get("CONFIG_ENCRYPTION_KEY")


def rotate_key(
    config: dict[str, Any],
    old_key: str,
    new_key: str,
) -> dict[str, Any]:
    """Re-encrypt config with a new key.

    Args:
        config: Config with values encrypted using old_key.
        old_key: Current encryption key.
        new_key: New encryption key.

    Returns:
        Config with values re-encrypted using new_key.
    """
    decrypted = decrypt_config(config, old_key)

    # Find which fields were encrypted and re-encrypt them
    result = {}
    for k, v in config.items():
        if isinstance(v, dict):
            result[k] = rotate_key(v, old_key, new_key)
        elif isinstance(v, str) and is_encrypted(v):
            plaintext = decrypted[k]
            result[k] = encrypt_value(plaintext, new_key)
        else:
            result[k] = v

    return result
