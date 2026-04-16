"""FastAPI dashboard with real-time WebSocket updates."""

from dashboard.api import app, set_database, broadcast

__all__ = ["app", "set_database", "broadcast"]
