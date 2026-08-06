"""Structured logging via structlog.

Provides a single `get_logger()` factory. Secrets are never logged:
values passed with keys from `SENSITIVE_KEYS` are redacted.
"""

import logging
import sys

import structlog

SENSITIVE_KEYS = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "authorization",
        "api_key",
        "apikey",
        "key",
    }
)


def _redact_event_dict(logger: str, method_name: str, event_dict: dict) -> dict:
    for key in list(event_dict):
        lower = key.lower()
        if any(s in lower for s in ("token", "secret", "password", "key", "auth")):
            event_dict[key] = "***"
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _redact_event_dict,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or "atlas")
