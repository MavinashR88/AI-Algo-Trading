"""
Database schema definitions.

All tables use TEXT for dates/timestamps (ISO 8601) for SQLite compatibility
while keeping the schema portable to PostgreSQL (swap TEXT → TIMESTAMPTZ).

Every table has:
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - created_at TEXT DEFAULT (datetime('now'))
"""

# Each string is a CREATE TABLE IF NOT EXISTS statement.
# Order matters for foreign key references.

SCHEMA: list[str] = [
    # -----------------------------------------------------------------
    # Bot state — one row per market, tracks mode + performance
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS bot_state (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        market       TEXT NOT NULL UNIQUE,          -- 'us' or 'india'
        mode         TEXT NOT NULL DEFAULT 'paper',  -- 'paper' or 'live'
        capital      REAL NOT NULL DEFAULT 0.0,
        win_rate     REAL NOT NULL DEFAULT 0.0,
        total_trades INTEGER NOT NULL DEFAULT 0,
        winning_trades INTEGER NOT NULL DEFAULT 0,
        losing_trades  INTEGER NOT NULL DEFAULT 0,
        current_drawdown_pct REAL NOT NULL DEFAULT 0.0,
        peak_capital REAL NOT NULL DEFAULT 0.0,
        sharpe_ratio REAL NOT NULL DEFAULT 0.0,
        paper_start_date TEXT,
        consecutive_losses INTEGER NOT NULL DEFAULT 0,
        daily_trade_count  INTEGER NOT NULL DEFAULT 0,
        daily_trade_date   TEXT,
        weekly_drawdown_pct REAL NOT NULL DEFAULT 0.0,
        week_start_capital  REAL NOT NULL DEFAULT 0.0,
        halted       INTEGER NOT NULL DEFAULT 0,     -- 1 if halted
        halt_reason  TEXT,
        updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # -----------------------------------------------------------------
    # Trades — every executed trade, full audit trail
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        market          TEXT NOT NULL,               -- 'us' or 'india'
        asset           TEXT NOT NULL,
        direction       TEXT NOT NULL,               -- 'long' or 'short'
        entry_price     REAL NOT NULL,
        exit_price      REAL,
        quantity         REAL NOT NULL,
        entry_time      TEXT NOT NULL,
        exit_time       TEXT,
        status          TEXT NOT NULL DEFAULT 'open', -- 'open','closed','cancelled'
        pnl             REAL,
        pnl_pct         REAL,
        fees            REAL NOT NULL DEFAULT 0.0,
        net_pnl         REAL,
        stop_loss       REAL,
        take_profit     REAL,
        trailing_stop   REAL,
        strategy        TEXT NOT NULL,
        signal_id       INTEGER,
        confidence      REAL,
        sentiment_score REAL,
        reasoning       TEXT,                         -- plain English
        order_id        TEXT,                         -- broker order ID
        fill_type       TEXT DEFAULT 'full',          -- 'full' or 'partial'
        mode            TEXT NOT NULL DEFAULT 'paper', -- trade executed in paper/live
        is_wash_sale    INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # -----------------------------------------------------------------
    # Signals — every signal generated, whether acted on or not
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        market          TEXT NOT NULL,
        asset           TEXT NOT NULL,
        direction       TEXT NOT NULL,           -- 'long' or 'short'
        confidence      REAL NOT NULL,
        timeframe       TEXT NOT NULL,
        strategy        TEXT NOT NULL,
        sentiment_score REAL,
        reasoning       TEXT,
        acted_on        INTEGER NOT NULL DEFAULT 0, -- 1 if trade executed
        reject_reason   TEXT,                       -- why not acted on
        generated_at    TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # -----------------------------------------------------------------
    # Sentiment scores — per asset, per news item
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sentiment_scores (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        market      TEXT NOT NULL,
        asset       TEXT NOT NULL,
        score       REAL NOT NULL,              -- -1.0 to +1.0
        headline    TEXT,
        source      TEXT,
        url         TEXT,
        model_used  TEXT,
        raw_response TEXT,
        scored_at   TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # -----------------------------------------------------------------
    # Market data cache — OHLCV candles
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS market_data_cache (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        market     TEXT NOT NULL,
        asset      TEXT NOT NULL,
        timeframe  TEXT NOT NULL,
        timestamp  TEXT NOT NULL,
        open       REAL NOT NULL,
        high       REAL NOT NULL,
        low        REAL NOT NULL,
        close      REAL NOT NULL,
        volume     REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(market, asset, timeframe, timestamp)
    )
    """,

    # -----------------------------------------------------------------
    # Error logs — persistent error tracking
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS error_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        level       TEXT NOT NULL,              -- 'WARNING','ERROR','CRITICAL'
        module      TEXT NOT NULL,
        message     TEXT NOT NULL,
        traceback   TEXT,
        context     TEXT,                       -- JSON blob with extra context
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # -----------------------------------------------------------------
    # Daily summaries — one row per market per day
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS daily_summaries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        market          TEXT NOT NULL,
        summary_date    TEXT NOT NULL,
        starting_capital REAL NOT NULL,
        ending_capital   REAL NOT NULL,
        pnl             REAL NOT NULL,
        pnl_pct         REAL NOT NULL,
        trades_taken    INTEGER NOT NULL DEFAULT 0,
        winning_trades  INTEGER NOT NULL DEFAULT 0,
        losing_trades   INTEGER NOT NULL DEFAULT 0,
        win_rate        REAL,
        max_drawdown_pct REAL,
        best_trade_pnl  REAL,
        worst_trade_pnl REAL,
        fees_total      REAL NOT NULL DEFAULT 0.0,
        mode            TEXT NOT NULL DEFAULT 'paper',
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(market, summary_date)
    )
    """,

    # -----------------------------------------------------------------
    # Backtest results — per strategy per market
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        market              TEXT NOT NULL,
        strategy            TEXT NOT NULL,
        start_date          TEXT NOT NULL,
        end_date            TEXT NOT NULL,
        total_trades        INTEGER NOT NULL,
        winning_trades      INTEGER NOT NULL,
        losing_trades       INTEGER NOT NULL,
        win_rate            REAL NOT NULL,
        sharpe_ratio        REAL,
        max_drawdown_pct    REAL,
        profit_factor       REAL,
        avg_trade_duration  TEXT,
        max_consecutive_losses INTEGER,
        total_pnl           REAL NOT NULL,
        total_fees          REAL NOT NULL,
        net_pnl             REAL NOT NULL,
        parameters          TEXT,              -- JSON blob of strategy params
        created_at          TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # -----------------------------------------------------------------
    # Audit log — every action the bot takes
    # -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        action     TEXT NOT NULL,
        module     TEXT NOT NULL,
        details    TEXT,                       -- JSON blob
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
]

# Indexes for common queries
INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market)",
    "CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset)",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)",
    "CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market)",
    "CREATE INDEX IF NOT EXISTS idx_signals_asset ON signals(asset)",
    "CREATE INDEX IF NOT EXISTS idx_signals_generated ON signals(generated_at)",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_market_asset ON sentiment_scores(market, asset)",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_scored ON sentiment_scores(scored_at)",
    "CREATE INDEX IF NOT EXISTS idx_market_data_lookup ON market_data_cache(market, asset, timeframe)",
    "CREATE INDEX IF NOT EXISTS idx_error_logs_level ON error_logs(level)",
    "CREATE INDEX IF NOT EXISTS idx_daily_summaries_date ON daily_summaries(summary_date)",
    "CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action)",
    "CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at)",
]
