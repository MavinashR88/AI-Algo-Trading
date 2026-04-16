"""
FastAPI dashboard — REST API + WebSocket for real-time trading dashboard.

Endpoints:
  GET  /api/status          — bot status for both markets
  GET  /api/trades           — recent trades
  GET  /api/trades/open      — open positions
  GET  /api/signals          — recent signals
  GET  /api/sentiment/{asset}— latest sentiment
  GET  /api/performance      — daily summaries
  GET  /api/backtest         — backtest results
  GET  /api/config           — current config (sanitized)
  POST /api/halt             — emergency halt
  POST /api/resume           — resume trading
  WS   /ws                   — real-time updates
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import structlog

from db.database import Database
from db import queries

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="AI Algo Trading Dashboard",
    version="1.0.0",
    description="Real-time trading bot monitoring and control",
)

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global references — set during app startup
_db: Database | None = None
_ws_clients: list[WebSocket] = []


def set_database(db: Database) -> None:
    """Inject database dependency."""
    global _db
    _db = db


def _get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


# -------------------------------------------------------------------------
# WebSocket manager
# -------------------------------------------------------------------------

async def broadcast(event: str, data: dict[str, Any]) -> None:
    """Send event to all connected WebSocket clients."""
    global _ws_clients
    message = json.dumps({"event": event, "data": data, "timestamp": datetime.utcnow().isoformat()})
    still_connected = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
            still_connected.append(ws)
        except Exception:
            pass
    _ws_clients = still_connected


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    global _ws_clients
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("dashboard.ws_connected", clients=len(_ws_clients))
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"event": "ack", "data": data}))
    except WebSocketDisconnect:
        _ws_clients = [c for c in _ws_clients if c is not websocket]
        logger.info("dashboard.ws_disconnected", clients=len(_ws_clients))


# -------------------------------------------------------------------------
# REST Endpoints
# -------------------------------------------------------------------------

@app.get("/")
async def root():
    """Dashboard root — serves simple status page."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/api/status")
async def get_status():
    """Get bot status for both markets."""
    db = _get_db()
    us_state = queries.get_bot_state(db, "us")
    india_state = queries.get_bot_state(db, "india")
    return {
        "us": dict(us_state) if us_state else None,
        "india": dict(india_state) if india_state else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/trades")
async def get_trades(
    market: str = Query("us", pattern="^(us|india)$"),
    limit: int = Query(20, ge=1, le=200),
):
    """Get recent trades."""
    db = _get_db()
    trades = queries.get_recent_trades(db, market, limit=limit)
    return {"trades": trades, "count": len(trades)}


@app.get("/api/trades/open")
async def get_open_trades(
    market: str | None = Query(None, pattern="^(us|india)$"),
):
    """Get all open positions."""
    db = _get_db()
    trades = queries.get_open_trades(db, market)
    return {"trades": trades, "count": len(trades)}


@app.get("/api/signals")
async def get_signals(
    market: str = Query("us", pattern="^(us|india)$"),
    limit: int = Query(20, ge=1, le=200),
):
    """Get recent signals."""
    db = _get_db()
    rows = db.execute(
        """SELECT * FROM signals WHERE market = ?
        ORDER BY generated_at DESC LIMIT ?""",
        (market, limit),
    )
    return {"signals": [dict(r) for r in rows], "count": len(rows)}


@app.get("/api/sentiment/{asset}")
async def get_sentiment(
    asset: str,
    market: str = Query("us", pattern="^(us|india)$"),
):
    """Get latest sentiment for an asset."""
    db = _get_db()
    latest = queries.get_latest_sentiment(db, market, asset)
    avg = queries.get_asset_sentiment_avg(db, market, asset, hours=4)
    return {
        "asset": asset,
        "market": market,
        "latest": latest,
        "avg_4h": avg,
    }


@app.get("/api/performance")
async def get_performance(
    market: str = Query("us", pattern="^(us|india)$"),
    limit: int = Query(30, ge=1, le=365),
):
    """Get daily performance summaries."""
    db = _get_db()
    rows = db.execute(
        """SELECT * FROM daily_summaries WHERE market = ?
        ORDER BY summary_date DESC LIMIT ?""",
        (market, limit),
    )
    return {"summaries": [dict(r) for r in rows], "count": len(rows)}


@app.get("/api/backtest")
async def get_backtest_results(
    market: str | None = Query(None, pattern="^(us|india)$"),
    strategy: str | None = None,
    limit: int = Query(20, ge=1, le=100),
):
    """Get backtest results."""
    db = _get_db()
    sql = "SELECT * FROM backtest_results"
    params: list = []
    conditions = []

    if market:
        conditions.append("market = ?")
        params.append(market)
    if strategy:
        conditions.append("strategy = ?")
        params.append(strategy)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, tuple(params))
    return {"results": [dict(r) for r in rows], "count": len(rows)}


@app.get("/api/errors")
async def get_errors(limit: int = Query(20, ge=1, le=100)):
    """Get recent error logs."""
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM error_logs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return {"errors": [dict(r) for r in rows], "count": len(rows)}


@app.post("/api/halt")
async def halt_market(market: str = Query("us", pattern="^(us|india)$"), reason: str = "Manual halt"):
    """Emergency halt for a market."""
    db = _get_db()
    queries.update_bot_state(db, market, halted=1, halt_reason=reason)
    db.log_audit("manual_halt", "dashboard", {"market": market, "reason": reason})
    await broadcast("halt", {"market": market, "reason": reason})
    return {"status": "halted", "market": market, "reason": reason}


@app.post("/api/resume")
async def resume_market(market: str = Query("us", pattern="^(us|india)$")):
    """Resume trading for a market."""
    db = _get_db()
    queries.update_bot_state(db, market, halted=0, halt_reason=None)
    db.log_audit("manual_resume", "dashboard", {"market": market})
    await broadcast("resume", {"market": market})
    return {"status": "resumed", "market": market}


@app.get("/api/config")
async def get_config():
    """Get sanitized config (no secrets)."""
    try:
        from config.loader import get_config
        config = get_config()
        return {
            "us_market": {
                "enabled": config.us_market.enabled,
                "mode": config.us_market.mode,
                "broker": config.us_market.broker,
                "watchlist": config.us_market.watchlist,
            },
            "india_market": {
                "enabled": config.india_market.enabled,
                "mode": config.india_market.mode,
                "broker": config.india_market.broker,
                "watchlist": config.india_market.watchlist,
            },
            "risk": {
                "max_risk_per_trade_pct": config.risk.max_risk_per_trade_pct,
                "max_open_positions": config.risk.max_open_positions_per_market,
                "max_daily_drawdown_pct": config.risk.max_daily_drawdown_pct,
            },
        }
    except Exception:
        return {"error": "Config not loaded"}


# -------------------------------------------------------------------------
# Embedded Dashboard HTML
# -------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Algo Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
               background: #0d1117; color: #c9d1d9; }
        .header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d;
                  display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 20px; color: #58a6ff; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-dot.active { background: #3fb950; }
        .status-dot.halted { background: #f85149; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card h2 { font-size: 14px; color: #8b949e; text-transform: uppercase; margin-bottom: 12px; }
        .metric { font-size: 28px; font-weight: bold; }
        .metric.green { color: #3fb950; }
        .metric.red { color: #f85149; }
        .metric.neutral { color: #c9d1d9; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }
        th { color: #8b949e; font-weight: 600; }
        .btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer;
               font-size: 13px; font-weight: 600; }
        .btn-danger { background: #da3633; color: white; }
        .btn-success { background: #238636; color: white; }
        .btn:hover { opacity: 0.85; }
        #log { background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
               padding: 12px; font-family: monospace; font-size: 12px;
               max-height: 300px; overflow-y: auto; margin-top: 16px; }
        .log-entry { padding: 2px 0; color: #8b949e; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Algo Trading Dashboard</h1>
        <div>
            <span class="status-dot active" id="ws-status"></span>
            <span id="ws-label">Connecting...</span>
        </div>
    </div>
    <div class="container">
        <div class="grid" id="markets"></div>
        <div class="card" style="margin-top: 16px;">
            <h2>Recent Trades</h2>
            <table>
                <thead><tr><th>Time</th><th>Market</th><th>Asset</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Strategy</th></tr></thead>
                <tbody id="trades-body"></tbody>
            </table>
        </div>
        <div class="card" style="margin-top: 16px;">
            <h2>Controls</h2>
            <button class="btn btn-danger" onclick="halt('us')">Halt US</button>
            <button class="btn btn-danger" onclick="halt('india')">Halt India</button>
            <button class="btn btn-success" onclick="resume('us')">Resume US</button>
            <button class="btn btn-success" onclick="resume('india')">Resume India</button>
        </div>
        <div class="card" style="margin-top: 16px;">
            <h2>Live Log</h2>
            <div id="log"></div>
        </div>
    </div>
    <script>
        let ws;
        function connectWS() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);
            ws.onopen = () => {
                document.getElementById('ws-status').className = 'status-dot active';
                document.getElementById('ws-label').textContent = 'Connected';
            };
            ws.onclose = () => {
                document.getElementById('ws-status').className = 'status-dot halted';
                document.getElementById('ws-label').textContent = 'Disconnected';
                setTimeout(connectWS, 3000);
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                addLog(`[${msg.event}] ${JSON.stringify(msg.data)}`);
                if (msg.event === 'halt' || msg.event === 'resume') refresh();
            };
        }
        function addLog(text) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `${new Date().toLocaleTimeString()} ${text}`;
            log.prepend(entry);
            if (log.children.length > 100) log.lastChild.remove();
        }
        async function refresh() {
            try {
                const status = await (await fetch('/api/status')).json();
                const markets = document.getElementById('markets');
                markets.innerHTML = '';
                for (const [name, data] of Object.entries(status)) {
                    if (name === 'timestamp' || !data) continue;
                    const halted = data.halted ? 'halted' : 'active';
                    markets.innerHTML += `
                        <div class="card">
                            <h2><span class="status-dot ${halted}"></span>${name.toUpperCase()} Market</h2>
                            <div class="metric ${data.capital > 0 ? 'green' : 'neutral'}">
                                ${data.mode === 'paper' ? 'PAPER' : 'LIVE'} — ${data.capital.toLocaleString()}
                            </div>
                            <p style="margin-top:8px;color:#8b949e;">
                                Trades: ${data.total_trades} | Win: ${(data.win_rate * 100).toFixed(1)}% |
                                DD: ${(data.current_drawdown_pct * 100).toFixed(1)}%
                            </p>
                        </div>`;
                }
                const trades = await (await fetch('/api/trades?limit=10&market=us')).json();
                const tbody = document.getElementById('trades-body');
                tbody.innerHTML = '';
                (trades.trades || []).forEach(t => {
                    const pnl = t.net_pnl || t.pnl || 0;
                    const cls = pnl > 0 ? 'green' : pnl < 0 ? 'red' : 'neutral';
                    tbody.innerHTML += `<tr>
                        <td>${t.entry_time || ''}</td><td>${t.market}</td><td>${t.asset}</td>
                        <td>${t.direction}</td><td>${t.entry_price}</td><td>${t.exit_price || '-'}</td>
                        <td class="${cls}">${pnl.toFixed(2)}</td><td>${t.strategy}</td></tr>`;
                });
            } catch(e) { addLog('Refresh error: ' + e.message); }
        }
        async function halt(market) { await fetch(`/api/halt?market=${market}`, {method:'POST'}); refresh(); }
        async function resume(market) { await fetch(`/api/resume?market=${market}`, {method:'POST'}); refresh(); }
        connectWS();
        refresh();
        setInterval(refresh, 10000);
    </script>
</body>
</html>"""
