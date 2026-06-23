"""MeetingsDatabase: per-user storage for meeting sessions, templates, and artifacts."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)

_SESSION_STATUSES = {"scheduled", "live", "processing", "completed", "failed"}
_SOURCE_TYPES = {"upload", "stream", "import"}
_TEMPLATE_SCOPES = {"builtin", "org", "team", "personal"}
_ARTIFACT_KINDS = {
    "transcript",
    "summary",
    "action_items",
    "decisions",
    "risks",
    "speaker_stats",
    "sentiment",
}
_DISPATCH_STATUSES = {"queued", "processing", "retrying", "delivered", "failed"}


class MeetingsDatabaseError(Exception):
    """Base exception for meetings database errors."""


class SchemaError(MeetingsDatabaseError):
    """Raised when schema initialization fails."""


class InputError(ValueError):
    """Raised for invalid input arguments."""


class MeetingsDatabase:
    """Lightweight per-user meetings persistence over SQLite."""

    _schema_init_paths: ClassVar[set[str]] = set()

    def __init__(self, db_path: str | Path, client_id: str, user_id: int | str) -> None:
        if not client_id:
            raise InputError("client_id is required")
        self.client_id = str(client_id)
        self.user_id = self._normalize_user_id(user_id)

        if isinstance(db_path, Path):
            self.db_path = db_path.resolve() if str(db_path) != ":memory:" else Path(":memory:")
            self._db_path_str = str(self.db_path)
        else:
            self._db_path_str = str(db_path)
            self.db_path = (
                Path(self._db_path_str).resolve()
                if self._db_path_str != ":memory:"
                else Path(":memory:")
            )

        self._local = threading.local()
        self.ensure_schema()

    @classmethod
    def for_user(cls, user_id: int | str) -> MeetingsDatabase:
        db_path = DatabasePaths.get_media_db_path(user_id)
        return cls(db_path=db_path, client_id=f"meetings-{user_id}", user_id=user_id)

    @staticmethod
    def _normalize_user_id(user_id: int | str | None) -> str:
        raw = str(user_id).strip() if user_id is not None else ""
        if not raw:
            raise InputError("user_id is required")
        return raw

    def _resolve_user_id(self, user_id: int | str | None) -> str:
        if user_id is None:
            return self.user_id
        return self._normalize_user_id(user_id)

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            configure_sqlite_connection(conn)
            self._local.connection = conn
        return conn

    def close_connection(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        conn = self.get_connection()
        started = begin_immediate_if_needed(conn)
        try:
            yield conn
            if started:
                conn.commit()
        except Exception:
            if started:
                conn.rollback()
            raise

    def ensure_schema(self) -> None:
        if self._db_path_str in self._schema_init_paths:
            return
        conn = self.get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meeting_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    meeting_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    source_type TEXT NOT NULL DEFAULT 'upload',
                    language TEXT,
                    template_id TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_sessions_user_created
                    ON meeting_sessions(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_meeting_sessions_user_status
                    ON meeting_sessions(user_id, status);

                CREATE TABLE IF NOT EXISTS meeting_templates (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'personal',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    schema_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_templates_user_scope
                    ON meeting_templates(user_id, scope);

                CREATE TABLE IF NOT EXISTS meeting_artifacts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    format TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES meeting_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_artifacts_session
                    ON meeting_artifacts(user_id, session_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS meeting_integration_dispatch (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    integration_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT,
                    response_json TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES meeting_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_dispatch_user_session
                    ON meeting_integration_dispatch(user_id, session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_meeting_dispatch_due
                    ON meeting_integration_dispatch(user_id, status, next_attempt_at, updated_at);

                CREATE TABLE IF NOT EXISTS meeting_event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES meeting_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_event_user_session
                    ON meeting_event_log(user_id, session_id, created_at DESC);
                """
            )
            self._run_schema_migrations(conn)
            conn.commit()
            self._schema_init_paths.add(self._db_path_str)
        except sqlite3.Error as exc:
            conn.rollback()
            raise SchemaError(f"Failed to initialize Meetings DB schema: {exc}") from exc

    def _run_schema_migrations(self, conn: sqlite3.Connection) -> None:
        cols = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(meeting_integration_dispatch)").fetchall()
        }
        if "next_attempt_at" not in cols:
            conn.execute("ALTER TABLE meeting_integration_dispatch ADD COLUMN next_attempt_at TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meeting_dispatch_due
            ON meeting_integration_dispatch(user_id, status, next_attempt_at, updated_at)
            """
        )
        if not self._has_session_cascade_fk(conn, "meeting_integration_dispatch", "session_id"):
            self._rebuild_integration_dispatch_with_session_fk(conn)
        if not self._has_session_cascade_fk(conn, "meeting_event_log", "session_id"):
            self._rebuild_event_log_with_session_fk(conn)

    @staticmethod
    def _has_session_cascade_fk(
        conn: sqlite3.Connection,
        table_name: str,
        from_column: str,
    ) -> bool:
        try:
            fk_rows = conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()
        except sqlite3.Error:
            return False
        for row in fk_rows:
            table = str(row["table"])
            source_column = str(row["from"])
            on_delete = str(row["on_delete"]).upper()
            if table == "meeting_sessions" and source_column == from_column and on_delete == "CASCADE":
                return True
        return False

    def _rebuild_integration_dispatch_with_session_fk(self, conn: sqlite3.Connection) -> None:
        conn.execute("ALTER TABLE meeting_integration_dispatch RENAME TO meeting_integration_dispatch_legacy")
        conn.execute(
            """
            CREATE TABLE meeting_integration_dispatch (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                integration_type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                response_json TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES meeting_sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            INSERT INTO meeting_integration_dispatch (
                id, user_id, session_id, integration_type, status, payload_json, response_json,
                attempts, next_attempt_at, last_error, created_at, updated_at
            )
            SELECT
                id, user_id, session_id, integration_type, status, payload_json, response_json,
                attempts, next_attempt_at, last_error, created_at, updated_at
            FROM meeting_integration_dispatch_legacy legacy
            WHERE EXISTS (
                SELECT 1 FROM meeting_sessions sessions WHERE sessions.id = legacy.session_id
            )
            """
        )
        conn.execute("DROP TABLE meeting_integration_dispatch_legacy")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meeting_dispatch_user_session
            ON meeting_integration_dispatch(user_id, session_id, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meeting_dispatch_due
            ON meeting_integration_dispatch(user_id, status, next_attempt_at, updated_at)
            """
        )

    def _rebuild_event_log_with_session_fk(self, conn: sqlite3.Connection) -> None:
        conn.execute("ALTER TABLE meeting_event_log RENAME TO meeting_event_log_legacy")
        conn.execute(
            """
            CREATE TABLE meeting_event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES meeting_sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            INSERT INTO meeting_event_log (id, user_id, session_id, event_type, payload_json, created_at)
            SELECT id, user_id, session_id, event_type, payload_json, created_at
            FROM meeting_event_log_legacy legacy
            WHERE EXISTS (
                SELECT 1 FROM meeting_sessions sessions WHERE sessions.id = legacy.session_id
            )
            """
        )
        conn.execute("DROP TABLE meeting_event_log_legacy")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meeting_event_user_session
            ON meeting_event_log(user_id, session_id, created_at DESC)
            """
        )

    @staticmethod
    def _loads_maybe_json(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        if text[0] not in {"{", "["}:
            return value
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value

    @classmethod
    def _row_to_dict(cls, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("metadata_json", "schema_json", "payload_json", "response_json"):
            if key in data:
                data[key] = cls._loads_maybe_json(data[key])
        return data

    @staticmethod
    def _validate_status(status: str) -> str:
        normalized = str(status).strip().lower()
        if normalized not in _SESSION_STATUSES:
            raise InputError(f"Invalid meeting session status: {status}")
        return normalized

    @staticmethod
    def _validate_source_type(source_type: str) -> str:
        normalized = str(source_type).strip().lower()
        if normalized not in _SOURCE_TYPES:
            raise InputError(f"Invalid meeting source_type: {source_type}")
        return normalized

    @staticmethod
    def _validate_template_scope(scope: str) -> str:
        normalized = str(scope).strip().lower()
        if normalized not in _TEMPLATE_SCOPES:
            raise InputError(f"Invalid meeting template scope: {scope}")
        return normalized

    @staticmethod
    def _validate_artifact_kind(kind: str) -> str:
        normalized = str(kind).strip().lower()
        if normalized not in _ARTIFACT_KINDS:
            raise InputError(f"Invalid meeting artifact kind: {kind}")
        return normalized

    @staticmethod
    def _validate_dispatch_status(status: str) -> str:
        normalized = str(status).strip().lower()
        if normalized not in _DISPATCH_STATUSES:
            raise InputError(f"Invalid meeting dispatch status: {status}")
        return normalized

    @staticmethod
    def _clamp_limit(limit: int) -> int:
        return max(1, min(int(limit), 200))

    @staticmethod
    def _clamp_offset(offset: int) -> int:
        return max(0, int(offset))

    def create_session(
        self,
        *,
        title: str,
        meeting_type: str,
        source_type: str = "upload",
        language: str | None = None,
        template_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "scheduled",
        user_id: int | str | None = None,
    ) -> str:
        clean_title = str(title).strip()
        if not clean_title:
            raise InputError("title is required")
        clean_meeting_type = str(meeting_type).strip()
        if not clean_meeting_type:
            raise InputError("meeting_type is required")

        resolved_user_id = self._resolve_user_id(user_id)
        normalized_status = self._validate_status(status)
        normalized_source_type = self._validate_source_type(source_type)
        now = self._utcnow_iso()
        session_id = f"sess_{uuid.uuid4().hex}"
        metadata_json = json.dumps(metadata or {})

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO meeting_sessions (
                    id, user_id, title, meeting_type, status, source_type, language, template_id,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    resolved_user_id,
                    clean_title,
                    clean_meeting_type,
                    normalized_status,
                    normalized_source_type,
                    language,
                    template_id,
                    metadata_json,
                    now,
                    now,
                ),
            )
        return session_id

    def get_session(self, session_id: str, user_id: int | str | None = None) -> dict[str, Any] | None:
        resolved_user_id = self._resolve_user_id(user_id)
        row = self.get_connection().execute(
            """
            SELECT id, user_id, title, meeting_type, status, source_type, language, template_id,
                   metadata_json, created_at, updated_at
            FROM meeting_sessions
            WHERE id = ? AND user_id = ?
            """,
            (str(session_id), resolved_user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_sessions(
        self,
        *,
        user_id: int | str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        query = (
            """
            SELECT id, user_id, title, meeting_type, status, source_type, language, template_id,
                   metadata_json, created_at, updated_at
            FROM meeting_sessions
            WHERE user_id = ?
            """
        )
        params: list[Any] = [resolved_user_id]
        if status:
            query += " AND status = ?"
            params.append(self._validate_status(status))
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.append(self._clamp_limit(limit))
        params.append(self._clamp_offset(offset))

        rows = self.get_connection().execute(query, tuple(params)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_session_status(
        self,
        *,
        session_id: str,
        status: str,
        user_id: int | str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        resolved_user_id = self._resolve_user_id(user_id)
        normalized_status = self._validate_status(status)
        normalized_expected_status = (
            self._validate_status(expected_status)
            if expected_status is not None
            else None
        )
        now = self._utcnow_iso()
        with self.transaction() as conn:
            if normalized_expected_status is not None:
                cursor = conn.execute(
                    """
                    UPDATE meeting_sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ? AND user_id = ? AND status = ?
                    """,
                    (
                        normalized_status,
                        now,
                        str(session_id),
                        resolved_user_id,
                        normalized_expected_status,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE meeting_sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (normalized_status, now, str(session_id), resolved_user_id),
                )
            return int(cursor.rowcount or 0) > 0

    def create_template(
        self,
        *,
        name: str,
        schema_json: dict[str, Any],
        scope: str = "personal",
        enabled: bool = True,
        is_default: bool = False,
        user_id: int | str | None = None,
    ) -> str:
        clean_name = str(name).strip()
        if not clean_name:
            raise InputError("template name is required")
        resolved_user_id = self._resolve_user_id(user_id)
        normalized_scope = self._validate_template_scope(scope)
        now = self._utcnow_iso()
        template_id = f"tmpl_{uuid.uuid4().hex}"

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO meeting_templates (
                    id, user_id, name, scope, enabled, is_default, version, schema_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    template_id,
                    resolved_user_id,
                    clean_name,
                    normalized_scope,
                    1 if enabled else 0,
                    1 if is_default else 0,
                    json.dumps(schema_json),
                    now,
                    now,
                ),
            )
        return template_id

    def get_template(self, template_id: str, user_id: int | str | None = None) -> dict[str, Any] | None:
        resolved_user_id = self._resolve_user_id(user_id)
        row = self.get_connection().execute(
            """
            SELECT id, user_id, name, scope, enabled, is_default, version, schema_json, created_at, updated_at
            FROM meeting_templates
            WHERE id = ? AND user_id = ?
            """,
            (str(template_id), resolved_user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_templates(
        self,
        *,
        user_id: int | str | None = None,
        scope: str | None = None,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        query = (
            """
            SELECT id, user_id, name, scope, enabled, is_default, version, schema_json, created_at, updated_at
            FROM meeting_templates
            WHERE user_id = ?
            """
        )
        params: list[Any] = [resolved_user_id]
        if scope:
            query += " AND scope = ?"
            params.append(self._validate_template_scope(scope))
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY created_at DESC"
        rows = self.get_connection().execute(query, tuple(params)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def create_artifact(
        self,
        *,
        session_id: str,
        kind: str,
        format: str,
        payload_json: dict[str, Any],
        version: int = 1,
        user_id: int | str | None = None,
    ) -> str:
        resolved_user_id = self._resolve_user_id(user_id)
        if self.get_session(session_id=str(session_id), user_id=resolved_user_id) is None:
            raise KeyError(f"meeting session not found: {session_id}")

        clean_format = str(format).strip()
        if not clean_format:
            raise InputError("artifact format is required")
        normalized_kind = self._validate_artifact_kind(kind)
        artifact_id = f"art_{uuid.uuid4().hex}"
        now = self._utcnow_iso()

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO meeting_artifacts (
                    id, user_id, session_id, kind, format, payload_json, version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    resolved_user_id,
                    str(session_id),
                    normalized_kind,
                    clean_format,
                    json.dumps(payload_json),
                    max(1, int(version)),
                    now,
                ),
            )
        return artifact_id

    def replace_artifacts(
        self,
        *,
        session_id: str,
        artifacts: list[dict[str, Any]],
        user_id: int | str | None = None,
    ) -> list[str]:
        resolved_user_id = self._resolve_user_id(user_id)
        prepared: list[tuple[str, str, str, str, int, str]] = []

        for artifact in artifacts:
            normalized_kind = self._validate_artifact_kind(str(artifact.get("kind") or ""))
            clean_format = str(artifact.get("format") or "").strip()
            if not clean_format:
                raise InputError("artifact format is required")
            payload_json = artifact.get("payload_json")
            if not isinstance(payload_json, dict):
                raise InputError("artifact payload_json must be an object")
            version = max(1, int(artifact.get("version") or 1))
            prepared.append(
                (
                    f"art_{uuid.uuid4().hex}",
                    str(session_id),
                    normalized_kind,
                    clean_format,
                    version,
                    json.dumps(payload_json),
                )
            )

        now = self._utcnow_iso()
        with self.transaction() as conn:
            session_row = conn.execute(
                """
                SELECT 1
                FROM meeting_sessions
                WHERE id = ? AND user_id = ?
                """,
                (str(session_id), resolved_user_id),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"meeting session not found: {session_id}")

            for _artifact_id, prepared_session_id, kind, _format, version, _payload in prepared:
                conn.execute(
                    """
                    DELETE FROM meeting_artifacts
                    WHERE user_id = ? AND session_id = ? AND kind = ? AND version = ?
                    """,
                    (resolved_user_id, prepared_session_id, kind, version),
                )

            for artifact_id, prepared_session_id, kind, format_, version, payload in prepared:
                conn.execute(
                    """
                    INSERT INTO meeting_artifacts (
                        id, user_id, session_id, kind, format, payload_json, version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        resolved_user_id,
                        prepared_session_id,
                        kind,
                        format_,
                        payload,
                        version,
                        now,
                    ),
                )

        return [artifact_id for artifact_id, *_rest in prepared]

    def get_artifact(self, artifact_id: str, user_id: int | str | None = None) -> dict[str, Any] | None:
        resolved_user_id = self._resolve_user_id(user_id)
        row = self.get_connection().execute(
            """
            SELECT id, user_id, session_id, kind, format, payload_json, version, created_at
            FROM meeting_artifacts
            WHERE id = ? AND user_id = ?
            """,
            (str(artifact_id), resolved_user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_artifacts(self, *, session_id: str, user_id: int | str | None = None) -> list[dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        rows = self.get_connection().execute(
            """
            SELECT id, user_id, session_id, kind, format, payload_json, version, created_at
            FROM meeting_artifacts
            WHERE session_id = ? AND user_id = ?
            ORDER BY created_at DESC
            """,
            (str(session_id), resolved_user_id),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def append_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload_json: dict[str, Any] | None = None,
        user_id: int | str | None = None,
    ) -> int:
        resolved_user_id = self._resolve_user_id(user_id)
        clean_event_type = str(event_type).strip()
        if not clean_event_type:
            raise InputError("event_type is required")
        now = self._utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO meeting_event_log (user_id, session_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_user_id,
                    str(session_id),
                    clean_event_type,
                    json.dumps(payload_json) if payload_json is not None else None,
                    now,
                ),
            )
            return int(cursor.lastrowid or 0)

    def list_events(
        self,
        *,
        session_id: str,
        user_id: int | str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        rows = self.get_connection().execute(
            """
            SELECT id, user_id, session_id, event_type, payload_json, created_at
            FROM meeting_event_log
            WHERE session_id = ? AND user_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (str(session_id), resolved_user_id, self._clamp_limit(limit)),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def record_integration_dispatch(
        self,
        *,
        session_id: str,
        integration_type: str,
        status: str,
        payload_json: dict[str, Any] | None = None,
        response_json: dict[str, Any] | None = None,
        attempts: int = 0,
        next_attempt_at: str | None = None,
        last_error: str | None = None,
        user_id: int | str | None = None,
    ) -> int:
        resolved_user_id = self._resolve_user_id(user_id)
        clean_integration_type = str(integration_type).strip().lower()
        clean_status = self._validate_dispatch_status(status)
        if not clean_integration_type:
            raise InputError("integration_type is required")
        now = self._utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO meeting_integration_dispatch (
                    user_id, session_id, integration_type, status, payload_json, response_json,
                    attempts, next_attempt_at, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_user_id,
                    str(session_id),
                    clean_integration_type,
                    clean_status,
                    json.dumps(payload_json) if payload_json is not None else None,
                    json.dumps(response_json) if response_json is not None else None,
                    max(0, int(attempts)),
                    str(next_attempt_at).strip() if next_attempt_at else None,
                    last_error,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid or 0)

    def get_integration_dispatch(
        self,
        *,
        dispatch_id: int,
        user_id: int | str | None = None,
    ) -> dict[str, Any] | None:
        resolved_user_id = self._resolve_user_id(user_id)
        row = self.get_connection().execute(
            """
            SELECT id, user_id, session_id, integration_type, status, payload_json, response_json,
                   attempts, next_attempt_at, last_error, created_at, updated_at
            FROM meeting_integration_dispatch
            WHERE id = ? AND user_id = ?
            """,
            (int(dispatch_id), resolved_user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_integration_dispatches(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        user_id: int | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        query = (
            """
            SELECT id, user_id, session_id, integration_type, status, payload_json, response_json,
                   attempts, next_attempt_at, last_error, created_at, updated_at
            FROM meeting_integration_dispatch
            WHERE user_id = ?
            """
        )
        params: list[Any] = [resolved_user_id]
        if session_id:
            query += " AND session_id = ?"
            params.append(str(session_id))
        if status:
            query += " AND status = ?"
            params.append(self._validate_dispatch_status(status))
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.append(self._clamp_limit(limit))
        params.append(self._clamp_offset(offset))

        rows = self.get_connection().execute(query, tuple(params)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def claim_due_integration_dispatches(
        self,
        *,
        limit: int = 25,
        max_attempts: int = 8,
        user_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        now = self._utcnow_iso()
        safe_limit = self._clamp_limit(limit)
        safe_max_attempts = max(1, int(max_attempts))
        claimable_states = ("queued", "retrying")

        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, session_id, integration_type, status, payload_json, response_json,
                       attempts, next_attempt_at, last_error, created_at, updated_at
                FROM meeting_integration_dispatch
                WHERE user_id = ?
                  AND status IN (?, ?)
                  AND attempts < ?
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY id ASC
                LIMIT ?
                """,
                (
                    resolved_user_id,
                    claimable_states[0],
                    claimable_states[1],
                    safe_max_attempts,
                    now,
                    safe_limit,
                ),
            ).fetchall()

            claimed: list[dict[str, Any]] = []
            for row in rows:
                dispatch_id = int(row["id"])
                cursor = conn.execute(
                    """
                    UPDATE meeting_integration_dispatch
                    SET status = 'processing', updated_at = ?
                    WHERE id = ? AND user_id = ? AND status IN (?, ?)
                    """,
                    (now, dispatch_id, resolved_user_id, claimable_states[0], claimable_states[1]),
                )
                if int(cursor.rowcount or 0) <= 0:
                    continue
                as_dict = self._row_to_dict(row)
                as_dict["status"] = "processing"
                as_dict["updated_at"] = now
                claimed.append(as_dict)
            return claimed

    def update_integration_dispatch(
        self,
        *,
        dispatch_id: int,
        status: str,
        attempts: int,
        next_attempt_at: str | None = None,
        last_error: str | None = None,
        response_json: dict[str, Any] | None = None,
        user_id: int | str | None = None,
    ) -> bool:
        resolved_user_id = self._resolve_user_id(user_id)
        clean_status = self._validate_dispatch_status(status)
        now = self._utcnow_iso()

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE meeting_integration_dispatch
                SET status = ?, attempts = ?, next_attempt_at = ?, last_error = ?, response_json = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    clean_status,
                    max(0, int(attempts)),
                    str(next_attempt_at).strip() if next_attempt_at else None,
                    last_error,
                    json.dumps(response_json) if response_json is not None else None,
                    now,
                    int(dispatch_id),
                    resolved_user_id,
                ),
            )
            return int(cursor.rowcount or 0) > 0
