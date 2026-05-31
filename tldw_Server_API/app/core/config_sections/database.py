"""Typed loader for database-related configuration settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.testing import is_truthy

from .types import ConfigParserLike


@dataclass(frozen=True)
class DatabaseConfig:
    """Database, PostgreSQL pool, Chroma, and prompts database settings."""

    type: str
    sqlite_path: str
    sqlite_wal_mode: bool
    sqlite_foreign_keys: bool
    backup_path: str
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_sslmode: str
    pg_pool_size: int
    pg_max_overflow: int
    pg_pool_timeout: float
    chroma_db_path: str
    prompts_db_path: str


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    option: str,
    default: str,
) -> str:
    """Return an environment override, parser value, or default for a database option."""
    raw = env_map.get(f"DB_{option.upper()}")
    if raw is None or str(raw).strip() == "":
        raw = config_parser.get("Database", option, fallback=default)
    text = str(raw).strip()
    return text or default


def _parse_bool(raw: object, default: bool) -> bool:
    """Parse a boolean value using the project truthy set and conservative fallback."""
    text = str(raw).strip().lower()
    if not text:
        return default
    if is_truthy(text):
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(raw: object, default: int) -> int:
    """Parse an integer value, returning the default for malformed input."""
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _parse_float(raw: object, default: float) -> float:
    """Parse a float value, returning the default for malformed input."""
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def load_database_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> DatabaseConfig:
    """Load typed database settings with environment variables taking precedence."""
    env_map: Mapping[str, str] = env if env is not None else os.environ

    return DatabaseConfig(
        type=_get_raw(config_parser, env_map, "type", "sqlite"),
        sqlite_path=_get_raw(config_parser, env_map, "sqlite_path", "Databases/server_media_summary.db"),
        sqlite_wal_mode=_parse_bool(_get_raw(config_parser, env_map, "sqlite_wal_mode", "true"), True),
        sqlite_foreign_keys=_parse_bool(_get_raw(config_parser, env_map, "sqlite_foreign_keys", "true"), True),
        backup_path=_get_raw(config_parser, env_map, "backup_path", "./tldw_DB_Backups/"),
        pg_host=_get_raw(config_parser, env_map, "pg_host", "localhost"),
        pg_port=_parse_int(_get_raw(config_parser, env_map, "pg_port", "5432"), 5432),
        pg_database=_get_raw(config_parser, env_map, "pg_database", "tldw_content"),
        pg_user=_get_raw(config_parser, env_map, "pg_user", "tldw_user"),
        pg_sslmode=_get_raw(config_parser, env_map, "pg_sslmode", "prefer"),
        pg_pool_size=_parse_int(_get_raw(config_parser, env_map, "pg_pool_size", "20"), 20),
        pg_max_overflow=_parse_int(_get_raw(config_parser, env_map, "pg_max_overflow", "40"), 40),
        pg_pool_timeout=_parse_float(_get_raw(config_parser, env_map, "pg_pool_timeout", "30.0"), 30.0),
        chroma_db_path=_get_raw(config_parser, env_map, "chroma_db_path", "Databases/chroma_db"),
        prompts_db_path=_get_raw(config_parser, env_map, "prompts_db_path", "Databases/prompts.db"),
    )
