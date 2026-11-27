import contextvars
import logging
import re
from typing import Any

import structlog

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")

_SECRET_PATTERN = re.compile(r"(AIza[0-9A-Za-z\-_]{20,}|sk-[0-9A-Za-z\-_]{20,}|Bearer\s+[0-9A-Za-z\-_.]+)")


def _redact_secrets(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = _SECRET_PATTERN.sub("[REDACTED]", value)
    event_dict["correlation_id"] = correlation_id_var.get()
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            _redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
