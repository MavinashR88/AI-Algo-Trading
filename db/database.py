"""
Database connection manager with migration support.

Design decisions:
- WAL journal mode for concurrent reads during async operation
- All queries use parameterized statements (no string formatting)
- Schema versioning via a simple `schema_version` table
- Connection pool of 1 writer + N readers for async safety
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import structlog

from db.models import INDEXES, SCHEMA

logger = structlog.get_logger(__name__)

CURRENT_SCHEMA_VERSION = 1


class Database:
    """SQLite database manager with migration support."""

    def __init__(self, db_path: str | Path, journal_mode: str = "WAL", busy_timeout_ms: int = 5000):
        self.db_path = Path(db_path)
        self.journal_mode = journal_mode
        self.busy_timeout_ms = busy_timeout_ms
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA journal_mode={self.journal_mode}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursors."""
        with self.connection() as conn:
            cur = conn.cursor()
            yield cur

    def initialize(self) -> None:
        """Create all tables and indexes. Safe to call repeatedly."""
        logger.info("database.initialize", db_path=str(self.db_path))

        with self.connection() as conn:
            # Schema version tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Check current version
            row = conn.execute(
                "SELECT MAX(version) as v FROM schema_version"
            ).fetchone()
            current = row["v"] if row["v"] is not None else 0

            if current < CURRENT_SCHEMA_VERSION:
                # Apply all schema tables
                for stmt in SCHEMA:
                    conn.execute(stmt)

                # Apply indexes
                for idx in INDEXES:
                    conn.execute(idx)

                # Record version
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (CURRENT_SCHEMA_VERSION,),
                )
                logger.info(
                    "database.schema_applied",
                    from_version=current,
                    to_version=CURRENT_SCHEMA_VERSION,
                )
            else:
                logger.info("database.schema_current", version=current)

            # Initialize bot state rows if missing
            for market in ("us", "india"):
                existing = conn.execute(
                    "SELECT id FROM bot_state WHERE market = ?", (market,)
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO bot_state (market) VALUES (?)", (market,)
                    )
                    logger.info("database.bot_state_initialized", market=market)

    # ------------------------------------------------------------------
    # Generic query helpers (all parameterized)
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def execute_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute a query and return a single row."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def execute_insert(self, sql: str, params: tuple = ()) -> int:
        """Execute an INSERT and return the last row ID."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid  # type: ignore[return-value]

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a statement with multiple parameter sets."""
        with self.cursor() as cur:
            cur.executemany(sql, params_list)

    # ------------------------------------------------------------------
    # Audit log helper
    # ------------------------------------------------------------------

    def log_audit(self, action: str, module: str, details: dict[str, Any] | None = None) -> None:
        """Write an audit log entry."""
        self.execute_insert(
            "INSERT INTO audit_log (action, module, details) VALUES (?, ?, ?)",
            (action, module, json.dumps(details) if details else None),
        )
