"""Persistent voice registry backing store with lightweight schema migrations."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    resolve_trusted_database_path,
)
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)

_SCHEMA_VERSION = 1


def _normalize_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return datetime.utcnow().isoformat()


def _normalize_voice_record(record: Mapping[str, Any]) -> dict[str, Any]:
    voice_id = str(record.get("voice_id") or "").strip()
    if not voice_id:
        raise ValueError("voice_id is required")

    name = str(record.get("name") or voice_id).strip() or voice_id
    file_path = str(record.get("file_path") or "").strip()
    if not file_path:
        raise ValueError("file_path is required")

    fmt = str(record.get("format") or "").strip().lower() or "wav"
    provider = str(record.get("provider") or "").strip().lower() or "vibevoice"

    description_raw = record.get("description")
    description = None if description_raw is None else str(description_raw)

    duration_raw = record.get("duration")
    duration = float(duration_raw) if duration_raw is not None else 0.0

    sample_rate_raw = record.get("sample_rate")
    sample_rate = int(sample_rate_raw) if sample_rate_raw is not None else None

    size_raw = record.get("size_bytes")
    size_bytes = int(size_raw) if size_raw is not None else 0

    created_at = _normalize_timestamp(record.get("created_at"))
    file_hash = str(record.get("file_hash") or "")

    return {
        "voice_id": voice_id,
        "name": name,
        "description": description,
        "file_path": file_path,
        "format": fmt,
        "duration": duration,
        "sample_rate": sample_rate,
        "size_bytes": size_bytes,
        "provider": provider,
        "created_at": created_at,
        "file_hash": file_hash,
    }


class VoiceRegistryDB:
    """SQLite-backed registry for user voice records."""

    def __init__(self, db_path: str | Path):
        resolved_db_path = resolve_trusted_database_path(db_path, label="voice registry db")
        user_db_base = DatabasePaths.get_user_db_base_dir().resolve(strict=False)
        try:
            resolved_db_path.relative_to(user_db_base)
        except ValueError as exc:
            raise ValueError(
                f"Voice registry DB path {resolved_db_path!r} escapes user DB base directory {user_db_base!r}"
            ) from exc

        self.db_path = resolved_db_path
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        return conn

    def _initialize_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS voice_registry_schema_version ("
                    "id INTEGER PRIMARY KEY CHECK (id = 1), "
                    "version INTEGER NOT NULL)"
                )
                row = conn.execute(
                    "SELECT version FROM voice_registry_schema_version WHERE id = 1"
                ).fetchone()
                current_version = 0
                if row is None:
                    conn.execute(
                        "INSERT INTO voice_registry_schema_version (id, version) VALUES (1, 0)"
                    )
                else:
                    current_version = int(row["version"] or 0)

                if current_version < 1:
                    self._apply_migration_v1(conn)
                    conn.execute(
                        "UPDATE voice_registry_schema_version SET version = ? WHERE id = 1",
                        (_SCHEMA_VERSION,),
                    )
                elif current_version > _SCHEMA_VERSION:
                    logger.warning(
                        "Voice registry DB schema version {} is newer than supported {} for {}",
                        current_version,
                        _SCHEMA_VERSION,
                        self.db_path,
                    )

                self._ensure_compat_columns(conn)

    @staticmethod
    def _apply_migration_v1(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_registry (
                user_id INTEGER NOT NULL,
                voice_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                file_path TEXT NOT NULL,
                format TEXT NOT NULL,
                duration REAL NOT NULL DEFAULT 0,
                sample_rate INTEGER,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_hash TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, voice_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_registry_user_id ON voice_registry(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_registry_user_path ON voice_registry(user_id, file_path)"
        )

    @staticmethod
    def _ensure_compat_columns(conn: sqlite3.Connection) -> None:
        expected_columns = {
            "description": "TEXT",
            "sample_rate": "INTEGER",
            "file_hash": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
        }
        rows = conn.execute("PRAGMA table_info(voice_registry)").fetchall()
        existing = {str(row["name"]) for row in rows}
        for column, ddl in expected_columns.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE voice_registry ADD COLUMN {column} {ddl}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_registry_user_id ON voice_registry(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_registry_user_path ON voice_registry(user_id, file_path)"
        )

    @staticmethod
    def _upsert_stmt() -> str:
        return (
            "INSERT INTO voice_registry ("
            "user_id, voice_id, name, description, file_path, format, duration, sample_rate, "
            "size_bytes, provider, created_at, file_hash, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(user_id, voice_id) DO UPDATE SET "
            "name = excluded.name, "
            "description = excluded.description, "
            "file_path = excluded.file_path, "
            "format = excluded.format, "
            "duration = excluded.duration, "
            "sample_rate = excluded.sample_rate, "
            "size_bytes = excluded.size_bytes, "
            "provider = excluded.provider, "
            "created_at = excluded.created_at, "
            "file_hash = excluded.file_hash, "
            "updated_at = CURRENT_TIMESTAMP"
        )

    @staticmethod
    def _params(user_id: int, record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(user_id),
            record["voice_id"],
            record["name"],
            record.get("description"),
            record["file_path"],
            record["format"],
            float(record.get("duration") or 0.0),
            record.get("sample_rate"),
            int(record.get("size_bytes") or 0),
            record["provider"],
            record["created_at"],
            record.get("file_hash") or "",
        )

    def upsert_voice(self, user_id: int, record: Mapping[str, Any]) -> None:
        payload = _normalize_voice_record(record)
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._upsert_stmt(), self._params(user_id, payload))

    def replace_user_voices(self, user_id: int, records: Sequence[Mapping[str, Any]]) -> None:
        # Deduplicate by voice_id while preserving latest values.
        dedup: dict[str, dict[str, Any]] = {}
        for record in records:
            normalized = _normalize_voice_record(record)
            dedup[normalized["voice_id"]] = normalized
        normalized_records = list(dedup.values())

        with self._lock:
            with self._connect() as conn:
                begin_immediate_if_needed(conn)
                for record in normalized_records:
                    conn.execute(self._upsert_stmt(), self._params(user_id, record))

                voice_ids = [record["voice_id"] for record in normalized_records]
                if voice_ids:
                    placeholders = ",".join("?" for _ in voice_ids)
                    voice_ids_clause = f"({placeholders})"
                    delete_voice_sql_template = (
                        "DELETE FROM voice_registry WHERE user_id = ? AND voice_id NOT IN {voice_ids_clause}"
                    )
                    delete_voice_sql = delete_voice_sql_template.format_map(locals())  # nosec B608
                    conn.execute(
                        delete_voice_sql,
                        [int(user_id), *voice_ids],
                    )
                else:
                    conn.execute(
                        "DELETE FROM voice_registry WHERE user_id = ?",
                        (int(user_id),),
                    )
                conn.commit()

    def get_voice(self, user_id: int, voice_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT voice_id, name, description, file_path, format, duration, sample_rate, "
                    "size_bytes, provider, created_at, file_hash "
                    "FROM voice_registry WHERE user_id = ? AND voice_id = ?",
                    (int(user_id), str(voice_id)),
                ).fetchone()
                if row is None:
                    return None
                return dict(row)

    def list_voices(self, user_id: int) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT voice_id, name, description, file_path, format, duration, sample_rate, "
                    "size_bytes, provider, created_at, file_hash "
                    "FROM voice_registry WHERE user_id = ? ORDER BY created_at DESC, voice_id ASC",
                    (int(user_id),),
                ).fetchall()
                return [dict(row) for row in rows]

    def delete_voice(self, user_id: int, voice_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM voice_registry WHERE user_id = ? AND voice_id = ?",
                    (int(user_id), str(voice_id)),
                )
                return bool(cursor.rowcount and cursor.rowcount > 0)

    def clear_user_voices(self, user_id: int) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM voice_registry WHERE user_id = ?",
                    (int(user_id),),
                )
