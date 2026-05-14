"""
Structured JSON logging setup.

Call setup_logging() once at application startup (in api/main.py).
All modules then use standard logging.getLogger(__name__) — this module
attaches the JSON formatter globally so every logger inherits it.

Log entries look like:
  {
    "ts": "2024-11-01T12:00:00.123Z",
    "level": "INFO",
    "logger": "agent.agent_loop",
    "msg": "agent_step",
    "run_id": "abc123",
    "iteration": 2,
    "action": "search"
  }
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Any extra fields passed via logger.info("msg", extra={...}) are
    merged into the top-level JSON object.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra fields the caller passed
        skip = {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName",
        }
        for key, val in record.__dict__.items():
            if key not in skip:
                base[key] = val

        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        return json.dumps(base, default=str)


def setup_logging(config: Optional[dict] = None) -> None:
    """
    Configure root logger with JSON formatting.
    Writes to both stdout and a rotating file under logs/.
    Call once at startup.
    """
    cfg = config or load_config()
    log_cfg = cfg["logging"]

    level = getattr(logging, log_cfg["level"].upper(), logging.INFO)
    log_dir = Path(log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()

    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    # rotating file handler (10 MB per file, keep 5)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "agent.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "logging_initialized",
        extra={"level": log_cfg["level"], "log_dir": str(log_dir)},
    )