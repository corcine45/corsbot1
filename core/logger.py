"""
Structured JSON logging for Corsbot.

Every log record is emitted as a single JSON line with consistent fields:
  timestamp, level, logger, message, + any extra kwargs passed to log calls.

Usage:
    from core.logger import get_logger
    log = get_logger(__name__)

    log.info("response_generated", user_id=123, latency_ms=240, tokens=180)
    log.warning("rate_limited", user_id=123, guild_id=456, wait="30s")
    log.error("agent_failed", user_id=123, error=str(e))

On Railway, filter by field:
    grep '"event":"response_generated"' logs
    grep '"user_id":123' logs
    grep '"level":"error"' logs
"""

import json
import logging
import os
import sys
import time
from typing import Any


# ────────────────────────────────────────────────────────────────────────────────
# JSON FORMATTER
# ────────────────────────────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    # Fields from LogRecord we don't want to re-emit as extras
    _SKIP = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        # Base fields always present
        doc: dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname.lower(),
            "logger":  record.name,
            "event":   record.getMessage(),
        }

        # Exception info
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)

        # Any extra= kwargs passed to the log call
        for key, val in record.__dict__.items():
            if key not in self._SKIP:
                doc[key] = val

        return json.dumps(doc, default=str, ensure_ascii=False)


# ────────────────────────────────────────────────────────────────────────────────
# BOUND LOGGER WRAPPER
# ────────────────────────────────────────────────────────────────────────────────

class BoundLogger:
    """
    Thin wrapper around stdlib Logger that accepts keyword args as structured fields.

    log.info("msg", user_id=1, latency_ms=200)
    → {"event": "msg", "user_id": 1, "latency_ms": 200, ...}
    """

    def __init__(self, logger: logging.Logger):
        self._log = logger

    def _emit(self, level: int, msg: str, **kwargs):
        if self._log.isEnabledFor(level):
            self._log.log(level, msg, extra=kwargs)

    def debug(self, msg: str, **kwargs):   self._emit(logging.DEBUG,    msg, **kwargs)
    def info(self, msg: str, **kwargs):    self._emit(logging.INFO,     msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._emit(logging.WARNING,  msg, **kwargs)
    def error(self, msg: str, **kwargs):   self._emit(logging.ERROR,    msg, **kwargs)
    def exception(self, msg: str, **kwargs):
        kwargs["exc_info"] = True
        self._emit(logging.ERROR, msg, **kwargs)

    # Allow passing through to underlying logger for compatibility
    def isEnabledFor(self, level: int) -> bool:
        return self._log.isEnabledFor(level)


# ────────────────────────────────────────────────────────────────────────────────
# SETUP
# ────────────────────────────────────────────────────────────────────────────────

_configured = False

def configure_logging(level: str = "INFO", json_logs: bool = True):
    """
    Call once at startup (done automatically by get_logger on first call).
    json_logs=False falls back to human-readable format for local dev.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root.level)

    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("discord", "discord.http", "discord.gateway", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> BoundLogger:
    """Get a structured logger. Configures logging on first call."""
    # Auto-configure: JSON in production (Railway sets LOG_LEVEL or just use JSON always),
    # plain text if LOG_FORMAT=text is set locally
    if not _configured:
        level = os.getenv("LOG_LEVEL", "INFO")
        json_logs = os.getenv("LOG_FORMAT", "json").lower() != "text"
        configure_logging(level=level, json_logs=json_logs)

    return BoundLogger(logging.getLogger(name))
