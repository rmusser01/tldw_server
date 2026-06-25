"""JSON log formatter for structured log shipping.

Provides a loguru-compatible formatter that emits one JSON object per line,
suitable for ingestion by Loki (via Promtail), ELK (via Filebeat), or any
JSONL-capable log aggregation pipeline.
"""
from __future__ import annotations

import json
from datetime import timezone
from typing import Any


def json_log_format(record: dict[str, Any]) -> str:
    """Format a loguru record as a JSON line for Loki/ELK ingestion.

    Parameters
    ----------
    record:
        The loguru log record dict.

    Returns
    -------
    str
        A single JSON line (newline-terminated).
    """
    timestamp = record["time"]
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    log_entry: dict[str, Any] = {
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    exc = record.get("exception")
    if exc is not None:
        log_entry["exception"] = str(exc)
    # Include extra fields attached via logger.bind()
    extra = record.get("extra", {})
    if extra:
        log_entry["extra"] = {k: str(v) for k, v in extra.items()}
    return json.dumps(log_entry, default=str) + "\n"
