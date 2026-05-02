from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .types import ConfigParserLike

_SYSTEM_LOG_ENV_KEYS = {
    "system_log_file_path": ("SYSTEM_LOG_FILE_PATH", "LOG_SYSTEM_LOG_FILE_PATH"),
    "system_log_file_max_entries": ("SYSTEM_LOG_FILE_MAX_ENTRIES", "LOG_SYSTEM_LOG_FILE_MAX_ENTRIES"),
}


@dataclass(frozen=True)
class LoggingConfig:
    log_level: str
    log_file: str
    log_metrics_file: str
    backup_count: int
    system_log_file_path: str
    system_log_file_max_entries: int


def _env_keys_for(option: str) -> tuple[str, ...]:
    """Return canonical and legacy environment keys for a logging option."""
    if option in _SYSTEM_LOG_ENV_KEYS:
        return _SYSTEM_LOG_ENV_KEYS[option]

    env_key = option.upper()
    if env_key.startswith("LOG_"):
        return (env_key, f"LOG_{env_key}")
    return (f"LOG_{env_key}",)


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    option: str,
    default: str,
) -> str:
    """Return the first non-empty env override, parser value, or default."""
    for env_key in _env_keys_for(option):
        raw = env_map.get(env_key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()

    raw = config_parser.get("Logging", option, fallback=default)
    text = str(raw).strip()
    return text or default


def _parse_int(raw: object, default: int) -> int:
    """Parse an integer value, returning the default for malformed input."""
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def load_logging_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> LoggingConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    return LoggingConfig(
        log_level=_get_raw(config_parser, env_map, "log_level", "INFO"),
        log_file=_get_raw(config_parser, env_map, "log_file", "./Logs/tldw_app_logs.json"),
        log_metrics_file=_get_raw(config_parser, env_map, "log_metrics_file", "./Logs/tldw_metrics_logs.json"),
        backup_count=_parse_int(_get_raw(config_parser, env_map, "backup_count", "5"), 5),
        system_log_file_path=_get_raw(config_parser, env_map, "system_log_file_path", "Databases/system_logs.jsonl"),
        system_log_file_max_entries=_parse_int(
            _get_raw(config_parser, env_map, "system_log_file_max_entries", "5000"),
            5000,
        ),
    )
