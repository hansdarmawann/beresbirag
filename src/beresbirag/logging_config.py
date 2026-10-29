"""Structured text and JSON logging configuration for beresbirag."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_STANDARD_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def _extract_extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    req_id = request_id_var.get()
    if req_id is not None:
        extras["request_id"] = req_id
    for key, value in record.__dict__.items():
        if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
            extras[key] = value
    return extras


class JsonLogFormatter(logging.Formatter):
    """Single-line JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extract_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextLogFormatter(logging.Formatter):
    """Human-readable text log formatter that appends structured extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = record.getMessage()
        extras = _extract_extra_fields(record)
        extra_str = " ".join(f"{k}={v}" for k, v in extras.items() if f"{k}=" not in msg)
        base = f"{ts} {record.levelname} [{record.name}] {msg}"
        if extra_str:
            base = f"{base} | {extra_str}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(
    level: str = "INFO",
    fmt: Literal["text", "json"] = "text",
) -> None:
    """Configure the ``beresbirag`` logger hierarchy with text or single-line JSON output."""
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    formatter: logging.Formatter = (
        JsonLogFormatter() if str(fmt).lower() == "json" else TextLogFormatter()
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(formatter)

    pkg_logger = logging.getLogger("beresbirag")
    pkg_logger.setLevel(numeric_level)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = True

