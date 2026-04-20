from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike


@dataclass(frozen=True)
class DatabaseConfig:
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


def load_database_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> DatabaseConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ
    g = lambda key, default: str(
        env_map.get(f"DB_{key.upper()}", "")
        or config_parser.get("Database", key, fallback=default)
    ).strip() or default

    return DatabaseConfig(
        type=g("type", "sqlite"),
        sqlite_path=g("sqlite_path", "Databases/server_media_summary.db"),
        sqlite_wal_mode=g("sqlite_wal_mode", "true").lower() in ("true", "1", "yes"),
        sqlite_foreign_keys=g("sqlite_foreign_keys", "true").lower() in ("true", "1", "yes"),
        backup_path=g("backup_path", "./tldw_DB_Backups/"),
        pg_host=g("pg_host", "localhost"),
        pg_port=int(g("pg_port", "5432")),
        pg_database=g("pg_database", "tldw_content"),
        pg_user=g("pg_user", "tldw_user"),
        pg_sslmode=g("pg_sslmode", "prefer"),
        pg_pool_size=int(g("pg_pool_size", "20")),
        pg_max_overflow=int(g("pg_max_overflow", "40")),
        pg_pool_timeout=float(g("pg_pool_timeout", "30.0")),
        chroma_db_path=g("chroma_db_path", "Databases/chroma_db"),
        prompts_db_path=g("prompts_db_path", "Databases/prompts.db"),
    )
