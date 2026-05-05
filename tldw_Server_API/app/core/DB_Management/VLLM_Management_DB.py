"""SQLite-backed storage for managed vLLM instances.

The managed vLLM domain owns lifecycle and routing behavior; this module owns
the SQLite driver usage and schema details.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.VLLM_Management.models import (
    VLLMInstanceCreate,
    VLLMInstanceRecord,
    utc_now_iso,
)

_SCHEMA_VERSION = 2
_DEFAULT_INSTANCE_KEY = "default_instance_id"
_ALLOWED_UPDATE_COLUMNS = {
    "name",
    "execution_mode",
    "transport_config_json",
    "launch_spec_json",
    "routing_policy_json",
    "declared_capabilities_json",
    "desired_state",
    "observed_state",
    "probed_capabilities_json",
    "effective_capabilities_json",
    "last_known_base_url",
    "last_error",
    "executor_handle_json",
    "updated_at",
}


class SqliteVLLMInstanceRepository:
    """Persist managed vLLM instance specs and routing metadata in SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _initialize_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vllm_instance_schema_version (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        version INTEGER NOT NULL
                    )
                    """
                )
                version_row = conn.execute(
                    "SELECT version FROM vllm_instance_schema_version WHERE id = 1"
                ).fetchone()
                if version_row is None:
                    conn.execute(
                        "INSERT INTO vllm_instance_schema_version (id, version) VALUES (1, ?)",
                        (_SCHEMA_VERSION,),
                    )
                elif int(version_row["version"] or 0) < _SCHEMA_VERSION:
                    conn.execute(
                        "UPDATE vllm_instance_schema_version SET version = ? WHERE id = 1",
                        (_SCHEMA_VERSION,),
                    )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vllm_instances (
                        instance_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        execution_mode TEXT NOT NULL,
                        transport_config_json TEXT NOT NULL,
                        launch_spec_json TEXT NOT NULL,
                        routing_policy_json TEXT NOT NULL,
                        declared_capabilities_json TEXT NOT NULL,
                        desired_state TEXT NOT NULL,
                        observed_state TEXT NOT NULL,
                        probed_capabilities_json TEXT NOT NULL DEFAULT '{}',
                        effective_capabilities_json TEXT NOT NULL DEFAULT '{}',
                        last_known_base_url TEXT,
                        last_error TEXT,
                        executor_handle_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                self._ensure_instance_columns(conn)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vllm_instance_settings (
                        setting_key TEXT PRIMARY KEY,
                        setting_value TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_vllm_instances_created_at "
                    "ON vllm_instances(created_at)"
                )

    @staticmethod
    def _ensure_instance_columns(conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(vllm_instances)").fetchall()
        }
        if "probed_capabilities_json" not in columns:
            conn.execute(
                "ALTER TABLE vllm_instances "
                "ADD COLUMN probed_capabilities_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "effective_capabilities_json" not in columns:
            conn.execute(
                "ALTER TABLE vllm_instances "
                "ADD COLUMN effective_capabilities_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "last_known_base_url" not in columns:
            conn.execute(
                "ALTER TABLE vllm_instances "
                "ADD COLUMN last_known_base_url TEXT"
            )
        if "last_error" not in columns:
            conn.execute(
                "ALTER TABLE vllm_instances "
                "ADD COLUMN last_error TEXT"
            )
        if "executor_handle_json" not in columns:
            conn.execute(
                "ALTER TABLE vllm_instances "
                "ADD COLUMN executor_handle_json TEXT NOT NULL DEFAULT '{}'"
            )

    @staticmethod
    def _dump_json(value: dict[str, Any]) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _load_json(value: str) -> dict[str, Any]:
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return loaded
        raise ValueError("Expected JSON object payload for persisted vLLM instance field")

    @classmethod
    def _row_to_record(cls, row: sqlite3.Row) -> VLLMInstanceRecord:
        row_keys = set(row.keys())
        return VLLMInstanceRecord(
            instance_id=str(row["instance_id"]),
            name=str(row["name"]),
            execution_mode=str(row["execution_mode"]),
            transport_config=cls._load_json(str(row["transport_config_json"])),
            launch_spec=cls._load_json(str(row["launch_spec_json"])),
            routing_policy=cls._load_json(str(row["routing_policy_json"])),
            declared_capabilities=cls._load_json(str(row["declared_capabilities_json"])),
            desired_state=str(row["desired_state"]),
            observed_state=str(row["observed_state"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            probed_capabilities=(
                cls._load_json(str(row["probed_capabilities_json"]))
                if "probed_capabilities_json" in row_keys
                else {}
            ),
            effective_capabilities=(
                cls._load_json(str(row["effective_capabilities_json"]))
                if "effective_capabilities_json" in row_keys
                else {}
            ),
            last_known_base_url=(
                None
                if "last_known_base_url" not in row_keys or row["last_known_base_url"] is None
                else str(row["last_known_base_url"])
            ),
            last_error=(
                None
                if "last_error" not in row_keys or row["last_error"] is None
                else str(row["last_error"])
            ),
            executor_handle=(
                cls._load_json(str(row["executor_handle_json"]))
                if "executor_handle_json" in row_keys
                else {}
            ),
        )

    def create_instance(self, payload: VLLMInstanceCreate) -> VLLMInstanceRecord:
        instance_id = uuid.uuid4().hex
        now = utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO vllm_instances (
                        instance_id,
                        name,
                        execution_mode,
                        transport_config_json,
                        launch_spec_json,
                        routing_policy_json,
                        declared_capabilities_json,
                        desired_state,
                        observed_state,
                        probed_capabilities_json,
                        effective_capabilities_json,
                        last_known_base_url,
                        last_error,
                        executor_handle_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        instance_id,
                        payload.name,
                        payload.execution_mode,
                        self._dump_json(payload.transport_config),
                        self._dump_json(payload.launch_spec),
                        self._dump_json(payload.routing_policy),
                        self._dump_json(payload.declared_capabilities),
                        "stopped",
                        "stopped",
                        self._dump_json({}),
                        self._dump_json({}),
                        None,
                        None,
                        self._dump_json({}),
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM vllm_instances WHERE instance_id = ?",
                    (instance_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to persist managed vLLM instance")
                return self._row_to_record(row)

    def get_instance(self, instance_id: str) -> VLLMInstanceRecord | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM vllm_instances WHERE instance_id = ?",
                    (instance_id,),
                ).fetchone()
                return None if row is None else self._row_to_record(row)

    def list_instances(self) -> list[VLLMInstanceRecord]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM vllm_instances ORDER BY created_at ASC, instance_id ASC"
                ).fetchall()
                return [self._row_to_record(row) for row in rows]

    def update_instance(self, instance_id: str, patch: dict[str, Any]) -> VLLMInstanceRecord:
        updates: list[tuple[str, Any]] = []
        if "name" in patch and patch["name"] is not None:
            updates.append(("name", str(patch["name"])))
        if "execution_mode" in patch and patch["execution_mode"] is not None:
            updates.append(("execution_mode", str(patch["execution_mode"])))
        if "transport_config" in patch:
            updates.append(("transport_config_json", self._dump_json(dict(patch["transport_config"] or {}))))
        if "launch_spec" in patch:
            updates.append(("launch_spec_json", self._dump_json(dict(patch["launch_spec"] or {}))))
        if "routing_policy" in patch:
            updates.append(("routing_policy_json", self._dump_json(dict(patch["routing_policy"] or {}))))
        if "declared_capabilities" in patch:
            updates.append(
                ("declared_capabilities_json", self._dump_json(dict(patch["declared_capabilities"] or {})))
            )
        return self._apply_updates(instance_id, updates)

    def update_instance_runtime(self, instance_id: str, patch: dict[str, Any]) -> VLLMInstanceRecord:
        updates: list[tuple[str, Any]] = []
        if "desired_state" in patch and patch["desired_state"] is not None:
            updates.append(("desired_state", str(patch["desired_state"])))
        if "observed_state" in patch and patch["observed_state"] is not None:
            updates.append(("observed_state", str(patch["observed_state"])))
        if "probed_capabilities" in patch:
            updates.append(
                ("probed_capabilities_json", self._dump_json(dict(patch["probed_capabilities"] or {})))
            )
        if "effective_capabilities" in patch:
            updates.append(
                ("effective_capabilities_json", self._dump_json(dict(patch["effective_capabilities"] or {})))
            )
        if "last_known_base_url" in patch:
            updates.append(("last_known_base_url", patch["last_known_base_url"]))
        if "last_error" in patch:
            updates.append(("last_error", patch["last_error"]))
        if "executor_handle" in patch:
            updates.append(("executor_handle_json", self._dump_json(dict(patch["executor_handle"] or {}))))
        return self._apply_updates(instance_id, updates)

    def _apply_updates(self, instance_id: str, updates: list[tuple[str, Any]]) -> VLLMInstanceRecord:
        if not updates:
            record = self.get_instance(instance_id)
            if record is None:
                raise ValueError(f"Unknown managed vLLM instance: {instance_id}")
            return record

        normalized_updates = list(updates)
        normalized_updates.append(("updated_at", utc_now_iso()))
        invalid_columns = [column for column, _ in normalized_updates if column not in _ALLOWED_UPDATE_COLUMNS]
        if invalid_columns:
            raise ValueError(f"Unsupported managed vLLM update columns: {invalid_columns}")
        assignments = ", ".join(f"{column} = ?" for column, _ in normalized_updates)
        params = [value for _, value in normalized_updates]
        params.append(instance_id)
        query = f"UPDATE vllm_instances SET {assignments} WHERE instance_id = ?"  # nosec B608

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    query,
                    params,
                )
                if cursor.rowcount == 0:
                    raise ValueError(f"Unknown managed vLLM instance: {instance_id}")
                row = conn.execute(
                    "SELECT * FROM vllm_instances WHERE instance_id = ?",
                    (instance_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Failed to fetch updated managed vLLM instance")
                return self._row_to_record(row)

    def delete_instance(self, instance_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM vllm_instances WHERE instance_id = ?",
                    (instance_id,),
                )
                deleted = cursor.rowcount > 0
                if deleted:
                    conn.execute(
                        "UPDATE vllm_instance_settings SET setting_value = NULL "
                        "WHERE setting_key = ? AND setting_value = ?",
                        (_DEFAULT_INSTANCE_KEY, instance_id),
                    )
                return deleted

    def set_default_instance(self, instance_id: str | None) -> None:
        with self._lock:
            with self._connect() as conn:
                if instance_id is not None:
                    exists = conn.execute(
                        "SELECT 1 FROM vllm_instances WHERE instance_id = ?",
                        (instance_id,),
                    ).fetchone()
                    if exists is None:
                        raise ValueError(f"Unknown managed vLLM instance: {instance_id}")
                conn.execute(
                    """
                    INSERT INTO vllm_instance_settings (setting_key, setting_value)
                    VALUES (?, ?)
                    ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
                    """,
                    (_DEFAULT_INSTANCE_KEY, instance_id),
                )

    def get_default_instance_id(self) -> str | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT setting_value FROM vllm_instance_settings WHERE setting_key = ?",
                    (_DEFAULT_INSTANCE_KEY,),
                ).fetchone()
                if row is None:
                    return None
                value = row["setting_value"]
                return None if value is None else str(value)
