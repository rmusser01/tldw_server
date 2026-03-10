"""SQLite-backed storage for managed vLLM instances."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .models import VLLMInstanceCreate, VLLMInstanceRecord, utc_now_iso

_SCHEMA_VERSION = 1
_DEFAULT_INSTANCE_KEY = "default_instance_id"


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
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
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
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
