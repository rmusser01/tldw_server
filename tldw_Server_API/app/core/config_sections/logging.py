from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike


@dataclass(frozen=True)
class LoggingConfig:
    log_level: str
    log_file: str
    log_metrics_file: str
    backup_count: int
    system_log_file_path: str
    system_log_file_max_entries: int


def load_logging_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> LoggingConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ
    g = lambda key, default: str(
        env_map.get(f"LOG_{key.upper()}", "")
        or config_parser.get("Logging", key, fallback=default)
    ).strip() or default

    return LoggingConfig(
        log_level=env_map.get("LOG_LEVEL", "") or g("log_level", "INFO"),
        log_file=g("log_file", "./Logs/tldw_app_logs.json"),
        log_metrics_file=g("log_metrics_file", "./Logs/tldw_metrics_logs.json"),
        backup_count=int(g("backup_count", "5")),
        system_log_file_path=g("system_log_file_path", "Databases/system_logs.jsonl"),
        system_log_file_max_entries=int(g("system_log_file_max_entries", "5000")),
    )
