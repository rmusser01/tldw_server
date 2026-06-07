"""Persistence for deep research sessions, checkpoints, and artifacts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_UNSET = object()
from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


@dataclass(frozen=True)
class ResearchSessionRow:
    id: str
    owner_user_id: str
    status: str
    phase: str
    query: str
    source_policy: str
    autonomy_mode: str
    limits_json: dict[str, Any]
    provider_overrides_json: dict[str, Any]
    follow_up_json: dict[str, Any]
    control_state: str
    progress_percent: float | None
    progress_message: str | None
    active_job_id: str | None
    latest_checkpoint_id: str | None
    created_at: str
    updated_at: str
    completed_at: str | None
    chat_id: str | None = None


@dataclass(frozen=True)
class ResearchCheckpointRow:
    id: str
    session_id: str
    checkpoint_type: str
    status: str
    resolution: str | None
    proposed_payload: dict[str, Any]
    user_patch_payload: dict[str, Any]
    created_at: str
    resolved_at: str | None


@dataclass(frozen=True)
class ResearchArtifactRow:
    id: str
    session_id: str
    artifact_name: str
    artifact_version: int
    storage_path: str
    content_type: str
    byte_size: int
    checksum: str
    phase: str
    job_id: str | None
    created_at: str


@dataclass(frozen=True)
class ResearchRunEventRow:
    id: int
    session_id: str
    owner_user_id: str
    event_type: str
    event_payload: dict[str, Any]
    phase: str | None
    job_id: str | None
    created_at: str


@dataclass(frozen=True)
class ResearchChatHandoffRow:
    session_id: str
    owner_user_id: str
    chat_id: str
    launch_message_id: str | None
    handoff_status: str
    delivered_chat_message_id: str | None
    delivered_notification_id: int | None
    last_error: str | None
    created_at: str
    updated_at: str
    delivered_at: str | None


@dataclass(frozen=True)
class ResearchChatLinkedRunRow:
    run_id: str
    query: str
    status: str
    phase: str
    control_state: str
    latest_checkpoint_id: str | None
    updated_at: str


class ResearchSessionsDB:
    """SQLite-backed storage for research sessions and related metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    query TEXT NOT NULL,
                    source_policy TEXT NOT NULL,
                    autonomy_mode TEXT NOT NULL,
                    limits_json TEXT NOT NULL DEFAULT '{}',
                    provider_overrides_json TEXT NOT NULL DEFAULT '{}',
                    follow_up_json TEXT NOT NULL DEFAULT '{}',
                    control_state TEXT NOT NULL DEFAULT 'running',
                    progress_percent REAL,
                    progress_message TEXT,
                    active_job_id TEXT,
                    latest_checkpoint_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS research_checkpoints (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolution TEXT,
                    proposed_payload TEXT NOT NULL DEFAULT '{}',
                    user_patch_payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)
                );

                CREATE TABLE IF NOT EXISTS research_artifacts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    artifact_version INTEGER NOT NULL,
                    storage_path TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    job_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)
                );

                CREATE TABLE IF NOT EXISTS research_run_events (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_payload_json TEXT NOT NULL,
                    phase TEXT,
                    job_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)
                );

                CREATE TABLE IF NOT EXISTS research_chat_handoffs (
                    session_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    launch_message_id TEXT,
                    handoff_status TEXT NOT NULL,
                    delivered_chat_message_id TEXT,
                    delivered_notification_id INTEGER,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    delivered_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES research_sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_research_sessions_owner
                    ON research_sessions(owner_user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_checkpoints_session
                    ON research_checkpoints(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_research_artifacts_session
                    ON research_artifacts(session_id, artifact_name, artifact_version DESC);
                CREATE INDEX IF NOT EXISTS idx_research_run_events_owner_session
                    ON research_run_events(owner_user_id, session_id, id ASC);
                CREATE INDEX IF NOT EXISTS idx_research_chat_handoffs_owner_chat
                    ON research_chat_handoffs(owner_user_id, chat_id, handoff_status);
                """
            )
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info('research_sessions')").fetchall()}
            if "provider_overrides_json" not in columns:
                conn.execute(
                    "ALTER TABLE research_sessions ADD COLUMN provider_overrides_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "follow_up_json" not in columns:
                conn.execute(
                    "ALTER TABLE research_sessions ADD COLUMN follow_up_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "control_state" not in columns:
                conn.execute(
                    "ALTER TABLE research_sessions ADD COLUMN control_state TEXT NOT NULL DEFAULT 'running'"
                )
            if "progress_percent" not in columns:
                conn.execute("ALTER TABLE research_sessions ADD COLUMN progress_percent REAL")
            if "progress_message" not in columns:
                conn.execute("ALTER TABLE research_sessions ADD COLUMN progress_message TEXT")

    @staticmethod
    def _session_from_row(row: sqlite3.Row | None) -> ResearchSessionRow | None:
        if row is None:
            return None
        keys = set(row.keys())
        return ResearchSessionRow(
            id=str(row["id"]),
            owner_user_id=str(row["owner_user_id"]),
            status=str(row["status"]),
            phase=str(row["phase"]),
            query=str(row["query"]),
            source_policy=str(row["source_policy"]),
            autonomy_mode=str(row["autonomy_mode"]),
            limits_json=_parse_json_dict(row["limits_json"]),
            provider_overrides_json=(
                _parse_json_dict(row["provider_overrides_json"])
                if "provider_overrides_json" in keys
                else {}
            ),
            follow_up_json=(
                _parse_json_dict(row["follow_up_json"])
                if "follow_up_json" in keys
                else {}
            ),
            control_state=(
                str(row["control_state"])
                if "control_state" in keys and row["control_state"] is not None
                else "running"
            ),
            progress_percent=(
                float(row["progress_percent"])
                if "progress_percent" in keys and row["progress_percent"] is not None
                else None
            ),
            progress_message=(
                str(row["progress_message"])
                if "progress_message" in keys and row["progress_message"] is not None
                else None
            ),
            active_job_id=str(row["active_job_id"]) if row["active_job_id"] else None,
            latest_checkpoint_id=str(row["latest_checkpoint_id"]) if row["latest_checkpoint_id"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            completed_at=str(row["completed_at"]) if row["completed_at"] else None,
        )

    @staticmethod
    def _checkpoint_from_row(row: sqlite3.Row | None) -> ResearchCheckpointRow | None:
        if row is None:
            return None
        return ResearchCheckpointRow(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            checkpoint_type=str(row["checkpoint_type"]),
            status=str(row["status"]),
            resolution=str(row["resolution"]) if row["resolution"] else None,
            proposed_payload=_parse_json_dict(row["proposed_payload"]),
            user_patch_payload=_parse_json_dict(row["user_patch_payload"]),
            created_at=str(row["created_at"]),
            resolved_at=str(row["resolved_at"]) if row["resolved_at"] else None,
        )

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row | None) -> ResearchArtifactRow | None:
        if row is None:
            return None
        return ResearchArtifactRow(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            artifact_name=str(row["artifact_name"]),
            artifact_version=int(row["artifact_version"]),
            storage_path=str(row["storage_path"]),
            content_type=str(row["content_type"]),
            byte_size=int(row["byte_size"]),
            checksum=str(row["checksum"]),
            phase=str(row["phase"]),
            job_id=str(row["job_id"]) if row["job_id"] else None,
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _run_event_from_row(row: sqlite3.Row | None) -> ResearchRunEventRow | None:
        if row is None:
            return None
        return ResearchRunEventRow(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            owner_user_id=str(row["owner_user_id"]),
            event_type=str(row["event_type"]),
            event_payload=_parse_json_dict(row["event_payload_json"]),
            phase=str(row["phase"]) if row["phase"] else None,
            job_id=str(row["job_id"]) if row["job_id"] else None,
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _chat_handoff_from_row(row: sqlite3.Row | None) -> ResearchChatHandoffRow | None:
        if row is None:
            return None
        return ResearchChatHandoffRow(
            session_id=str(row["session_id"]),
            owner_user_id=str(row["owner_user_id"]),
            chat_id=str(row["chat_id"]),
            launch_message_id=str(row["launch_message_id"]) if row["launch_message_id"] else None,
            handoff_status=str(row["handoff_status"]),
            delivered_chat_message_id=(
                str(row["delivered_chat_message_id"]) if row["delivered_chat_message_id"] else None
            ),
            delivered_notification_id=(
                int(row["delivered_notification_id"])
                if row["delivered_notification_id"] is not None
                else None
            ),
            last_error=str(row["last_error"]) if row["last_error"] else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            delivered_at=str(row["delivered_at"]) if row["delivered_at"] else None,
        )

    @staticmethod
    def _chat_linked_run_from_row(row: sqlite3.Row | None) -> ResearchChatLinkedRunRow | None:
        if row is None:
            return None
        return ResearchChatLinkedRunRow(
            run_id=str(row["run_id"]),
            query=str(row["query"]),
            status=str(row["status"]),
            phase=str(row["phase"]),
            control_state=str(row["control_state"]),
            latest_checkpoint_id=str(row["latest_checkpoint_id"]) if row["latest_checkpoint_id"] else None,
            updated_at=str(row["updated_at"]),
        )

    def create_session(
        self,
        *,
        owner_user_id: str,
        query: str,
        source_policy: str,
        autonomy_mode: str,
        limits_json: dict[str, Any],
        provider_overrides_json: dict[str, Any] | None = None,
        follow_up_json: dict[str, Any] | None = None,
        status: str = "queued",
        phase: str = "drafting_plan",
    ) -> ResearchSessionRow:
        session_id = f"rs_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        payload = json.dumps(limits_json or {}, sort_keys=True)
        provider_payload = json.dumps(provider_overrides_json or {}, sort_keys=True)
        follow_up_payload = json.dumps(follow_up_json or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_sessions (
                    id, owner_user_id, status, phase, query, source_policy,
                    autonomy_mode, limits_json, provider_overrides_json, follow_up_json,
                    control_state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(owner_user_id),
                    status,
                    phase,
                    query,
                    source_policy,
                    autonomy_mode,
                    payload,
                    provider_payload,
                    follow_up_payload,
                    "running",
                    now,
                    now,
                ),
            )
        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError("failed_to_create_research_session")
        return session

    def create_chat_handoff(
        self,
        *,
        session_id: str,
        owner_user_id: str,
        chat_id: str,
        launch_message_id: str | None = None,
    ) -> ResearchChatHandoffRow:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_chat_handoffs (
                    session_id, owner_user_id, chat_id, launch_message_id,
                    handoff_status, delivered_chat_message_id, delivered_notification_id,
                    last_error, created_at, updated_at, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(owner_user_id),
                    chat_id,
                    launch_message_id,
                    "pending",
                    None,
                    None,
                    None,
                    now,
                    now,
                    None,
                ),
            )
        handoff = self.get_chat_handoff(session_id)
        if handoff is None:
            raise RuntimeError("failed_to_create_research_chat_handoff")
        return handoff

    def get_chat_handoff(self, session_id: str) -> ResearchChatHandoffRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_chat_handoffs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._chat_handoff_from_row(row)

    def list_chat_linked_runs(
        self,
        *,
        owner_user_id: str,
        chat_id: str,
        terminal_limit: int = 10,
    ) -> list[ResearchChatLinkedRunRow]:
        bounded_terminal_limit = max(0, int(terminal_limit))
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH linked_runs AS (
                    SELECT
                        s.id AS run_id,
                        s.query,
                        s.status,
                        s.phase,
                        s.control_state,
                        s.latest_checkpoint_id,
                        s.updated_at,
                        CASE
                            WHEN s.status IN ('completed', 'failed', 'cancelled') THEN 1
                            ELSE 0
                        END AS is_terminal,
                        CASE
                            WHEN s.status IN ('completed', 'failed', 'cancelled') THEN
                                ROW_NUMBER() OVER (
                                    PARTITION BY CASE
                                        WHEN s.status IN ('completed', 'failed', 'cancelled') THEN 1
                                        ELSE 0
                                    END
                                    ORDER BY s.updated_at DESC, s.id DESC
                                )
                            ELSE 0
                        END AS terminal_rank
                    FROM research_chat_handoffs AS h
                    INNER JOIN research_sessions AS s
                        ON s.id = h.session_id
                    WHERE h.owner_user_id = ?
                      AND h.chat_id = ?
                      AND s.owner_user_id = ?
                )
                SELECT
                    run_id,
                    query,
                    status,
                    phase,
                    control_state,
                    latest_checkpoint_id,
                    updated_at
                FROM linked_runs
                WHERE is_terminal = 0 OR terminal_rank <= ?
                ORDER BY is_terminal ASC, updated_at DESC, run_id DESC
                """,
                (str(owner_user_id), str(chat_id), str(owner_user_id), bounded_terminal_limit),
            ).fetchall()
        return [run for row in rows if (run := self._chat_linked_run_from_row(row)) is not None]

    def mark_chat_handoff_chat_inserted(
        self,
        session_id: str,
        *,
        delivered_chat_message_id: str,
    ) -> ResearchChatHandoffRow:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_chat_handoffs
                SET handoff_status = ?, delivered_chat_message_id = ?, last_error = NULL,
                    updated_at = ?, delivered_at = ?
                WHERE session_id = ?
                """,
                (
                    "chat_inserted",
                    str(delivered_chat_message_id),
                    now,
                    now,
                    session_id,
                ),
            )
        handoff = self.get_chat_handoff(session_id)
        if handoff is None:
            raise KeyError(session_id)
        return handoff

    def mark_chat_handoff_notification_only(
        self,
        session_id: str,
        *,
        delivered_notification_id: int,
    ) -> ResearchChatHandoffRow:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_chat_handoffs
                SET handoff_status = ?, delivered_notification_id = ?, last_error = NULL,
                    updated_at = ?, delivered_at = ?
                WHERE session_id = ?
                """,
                (
                    "notification_only",
                    int(delivered_notification_id),
                    now,
                    now,
                    session_id,
                ),
            )
        handoff = self.get_chat_handoff(session_id)
        if handoff is None:
            raise KeyError(session_id)
        return handoff

    def mark_chat_handoff_failed(self, session_id: str, *, last_error: str) -> ResearchChatHandoffRow:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_chat_handoffs
                SET handoff_status = ?, last_error = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    "failed",
                    str(last_error),
                    now,
                    session_id,
                ),
            )
        handoff = self.get_chat_handoff(session_id)
        if handoff is None:
            raise KeyError(session_id)
        return handoff

    def get_session(self, session_id: str) -> ResearchSessionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._session_from_row(row)

    def list_sessions(
        self,
        owner_user_id: str,
        *,
        limit: int = 25,
        offset: int = 0,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[ResearchSessionRow]:
        bounded_limit = max(1, int(limit))
        bounded_offset = max(0, int(offset))
        query = """
                SELECT * FROM research_sessions
                WHERE owner_user_id = ?
                """
        params: list[Any] = [str(owner_user_id)]
        if session_id:
            query += " AND id = ?"
            params.append(str(session_id))
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += """
                ORDER BY created_at DESC, rowid DESC
                LIMIT ? OFFSET ?
                """
        params.extend([bounded_limit, bounded_offset])
        with self._connect() as conn:
            rows = conn.execute(
                query,
                tuple(params),
            ).fetchall()
        return [session for row in rows if (session := self._session_from_row(row)) is not None]

    def delete_session_cascade(self, session_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM research_chat_handoffs WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM research_run_events WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM research_artifacts WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM research_checkpoints WHERE session_id = ?",
                (session_id,),
            )
            cursor = conn.execute(
                "DELETE FROM research_sessions WHERE id = ?",
                (session_id,),
            )
        return bool(cursor.rowcount)

    def update_phase(
        self,
        session_id: str,
        *,
        phase: str,
        status: str,
        completed_at: str | None = None,
        control_state: str | None = None,
        active_job_id: str | None | object = _UNSET,
    ) -> ResearchSessionRow:
        now = _utc_now()
        params: tuple[Any, ...]
        sql: str
        if active_job_id is _UNSET and control_state is None:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, now, session_id)
        elif active_job_id is _UNSET:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, control_state = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, control_state, now, session_id)
        elif control_state is None:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, active_job_id, now, session_id)
        else:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, control_state = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, control_state, active_job_id, now, session_id)
        return self._execute_session_update(session_id=session_id, sql=sql, params=params)

    def update_phase_with_event(
        self,
        session_id: str,
        *,
        phase: str,
        status: str,
        owner_user_id: str,
        event_type: str,
        event_payload: dict[str, Any],
        event_phase: str | None = None,
        event_job_id: str | None = None,
        completed_at: str | None = None,
        control_state: str | None = None,
        active_job_id: str | None | object = _UNSET,
    ) -> tuple[ResearchSessionRow, ResearchRunEventRow]:
        now = _utc_now()
        params: tuple[Any, ...]
        sql: str
        if active_job_id is _UNSET and control_state is None:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, now, session_id)
        elif active_job_id is _UNSET:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, control_state = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, control_state, now, session_id)
        elif control_state is None:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, active_job_id, now, session_id)
        else:
            sql = """
                UPDATE research_sessions
                SET phase = ?, status = ?, completed_at = ?, control_state = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """
            params = (phase, status, completed_at, control_state, active_job_id, now, session_id)
        payload_json = json.dumps(event_payload or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(sql, params)
            event_id = self._record_run_event_with_conn(
                conn,
                owner_user_id=str(owner_user_id),
                session_id=session_id,
                event_type=event_type,
                event_payload_json=payload_json,
                phase=event_phase,
                job_id=event_job_id,
                created_at=now,
            )
        session = self.get_session(session_id)
        event = self.get_run_event(event_id)
        if session is None:
            raise KeyError(session_id)
        if event is None:
            raise RuntimeError("failed_to_record_research_run_event")
        return session, event

    def attach_active_job(self, session_id: str, job_id: str | None) -> ResearchSessionRow:
        now = _utc_now()
        return self._execute_session_update(
            session_id=session_id,
            sql="""
                UPDATE research_sessions
                SET active_job_id = ?, updated_at = ?
                WHERE id = ?
            """,
            params=(job_id, now, session_id),
        )

    def update_control_state(
        self,
        session_id: str,
        *,
        control_state: str,
        active_job_id: str | None | object = _UNSET,
    ) -> ResearchSessionRow:
        now = _utc_now()
        if active_job_id is _UNSET:
            return self._execute_session_update(
                session_id=session_id,
                sql="""
                    UPDATE research_sessions
                    SET control_state = ?, updated_at = ?
                    WHERE id = ?
                """,
                params=(control_state, now, session_id),
            )
        return self._execute_session_update(
            session_id=session_id,
            sql="""
                UPDATE research_sessions
                SET control_state = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """,
            params=(control_state, active_job_id, now, session_id),
        )

    def update_control_state_with_event(
        self,
        session_id: str,
        *,
        control_state: str,
        owner_user_id: str,
        event_type: str,
        event_payload: dict[str, Any],
        event_phase: str | None = None,
        event_job_id: str | None = None,
        active_job_id: str | None | object = _UNSET,
    ) -> tuple[ResearchSessionRow, ResearchRunEventRow]:
        now = _utc_now()
        if active_job_id is _UNSET:
            sql = """
                UPDATE research_sessions
                SET control_state = ?, updated_at = ?
                WHERE id = ?
            """
            params: tuple[Any, ...] = (control_state, now, session_id)
        else:
            sql = """
                UPDATE research_sessions
                SET control_state = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """
            params = (control_state, active_job_id, now, session_id)
        payload_json = json.dumps(event_payload or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(sql, params)
            event_id = self._record_run_event_with_conn(
                conn,
                owner_user_id=str(owner_user_id),
                session_id=session_id,
                event_type=event_type,
                event_payload_json=payload_json,
                phase=event_phase,
                job_id=event_job_id,
                created_at=now,
            )
        session = self.get_session(session_id)
        event = self.get_run_event(event_id)
        if session is None:
            raise KeyError(session_id)
        if event is None:
            raise RuntimeError("failed_to_record_research_run_event")
        return session, event

    def update_progress(
        self,
        session_id: str,
        *,
        progress_percent: float | None,
        progress_message: str | None,
    ) -> ResearchSessionRow:
        now = _utc_now()
        return self._execute_session_update(
            session_id=session_id,
            sql="""
                UPDATE research_sessions
                SET progress_percent = ?, progress_message = ?, updated_at = ?
                WHERE id = ?
            """,
            params=(progress_percent, progress_message, now, session_id),
        )

    def update_progress_with_event(
        self,
        session_id: str,
        *,
        progress_percent: float | None,
        progress_message: str | None,
        owner_user_id: str,
        event_type: str,
        event_payload: dict[str, Any],
        event_phase: str | None = None,
        event_job_id: str | None = None,
    ) -> tuple[ResearchSessionRow, ResearchRunEventRow]:
        now = _utc_now()
        payload_json = json.dumps(event_payload or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_sessions
                SET progress_percent = ?, progress_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (progress_percent, progress_message, now, session_id),
            )
            event_id = self._record_run_event_with_conn(
                conn,
                owner_user_id=str(owner_user_id),
                session_id=session_id,
                event_type=event_type,
                event_payload_json=payload_json,
                phase=event_phase,
                job_id=event_job_id,
                created_at=now,
            )
        session = self.get_session(session_id)
        event = self.get_run_event(event_id)
        if session is None:
            raise KeyError(session_id)
        if event is None:
            raise RuntimeError("failed_to_record_research_run_event")
        return session, event

    def update_status(
        self,
        session_id: str,
        *,
        status: str,
        control_state: str | None = None,
        active_job_id: str | None | object = _UNSET,
        completed_at: str | None = None,
    ) -> ResearchSessionRow:
        now = _utc_now()
        sql, params = self._build_update_status_statement(
            session_id=session_id,
            status=status,
            control_state=control_state,
            active_job_id=active_job_id,
            completed_at=completed_at,
            updated_at=now,
        )
        return self._execute_session_update(session_id=session_id, sql=sql, params=params)

    @staticmethod
    def _build_update_status_statement(
        *,
        session_id: str,
        status: str,
        control_state: str | None = None,
        active_job_id: str | None | object = _UNSET,
        completed_at: str | None = None,
        updated_at: str,
    ) -> tuple[str, tuple[Any, ...]]:
        if active_job_id is _UNSET and control_state is None:
            return (
                """
                    UPDATE research_sessions
                    SET status = ?, completed_at = ?, updated_at = ?
                    WHERE id = ?
                """,
                (status, completed_at, updated_at, session_id),
            )
        if active_job_id is _UNSET:
            return (
                """
                    UPDATE research_sessions
                    SET status = ?, completed_at = ?, control_state = ?, updated_at = ?
                    WHERE id = ?
                """,
                (status, completed_at, control_state, updated_at, session_id),
            )
        if control_state is None:
            return (
                """
                    UPDATE research_sessions
                    SET status = ?, completed_at = ?, active_job_id = ?, updated_at = ?
                    WHERE id = ?
                """,
                (status, completed_at, active_job_id, updated_at, session_id),
            )
        return (
            """
                UPDATE research_sessions
                SET status = ?, completed_at = ?, control_state = ?, active_job_id = ?, updated_at = ?
                WHERE id = ?
            """,
            (status, completed_at, control_state, active_job_id, updated_at, session_id),
        )

    def _execute_session_update(
        self,
        *,
        session_id: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> ResearchSessionRow:
        with self._connect() as conn:
            conn.execute(sql, params)
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def create_checkpoint(
        self,
        *,
        session_id: str,
        checkpoint_type: str,
        proposed_payload: dict[str, Any],
        status: str = "pending",
    ) -> ResearchCheckpointRow:
        checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        proposed_json = json.dumps(proposed_payload or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_checkpoints (
                    id, session_id, checkpoint_type, status, proposed_payload,
                    user_patch_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    session_id,
                    checkpoint_type,
                    status,
                    proposed_json,
                    "{}",
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE research_sessions
                SET latest_checkpoint_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (checkpoint_id, now, session_id),
            )
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise RuntimeError("failed_to_create_research_checkpoint")
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> ResearchCheckpointRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_checkpoints WHERE id = ?",
                (checkpoint_id,),
            ).fetchone()
        return self._checkpoint_from_row(row)

    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        *,
        resolution: str,
        user_patch_payload: dict[str, Any] | None = None,
    ) -> ResearchCheckpointRow:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_checkpoints
                SET status = ?, resolution = ?, user_patch_payload = ?, resolved_at = ?
                WHERE id = ?
                """,
                (
                    "resolved",
                    resolution,
                    json.dumps(user_patch_payload or {}, sort_keys=True),
                    now,
                    checkpoint_id,
                ),
            )
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise KeyError(checkpoint_id)
        return checkpoint

    def record_artifact(
        self,
        *,
        session_id: str,
        artifact_name: str,
        artifact_version: int,
        storage_path: str,
        content_type: str,
        byte_size: int,
        checksum: str,
        phase: str,
        job_id: str | None = None,
    ) -> ResearchArtifactRow:
        artifact_id = f"ra_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        with self._connect() as conn:
            self._record_artifact_with_conn(
                conn,
                artifact_id=artifact_id,
                session_id=session_id,
                artifact_name=artifact_name,
                artifact_version=artifact_version,
                storage_path=storage_path,
                content_type=content_type,
                byte_size=byte_size,
                checksum=checksum,
                phase=phase,
                job_id=job_id,
                created_at=now,
            )
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise RuntimeError("failed_to_record_research_artifact")
        return artifact

    def _record_artifact_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        artifact_id: str,
        session_id: str,
        artifact_name: str,
        artifact_version: int,
        storage_path: str,
        content_type: str,
        byte_size: int,
        checksum: str,
        phase: str,
        job_id: str | None,
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO research_artifacts (
                id, session_id, artifact_name, artifact_version, storage_path,
                content_type, byte_size, checksum, phase, job_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                session_id,
                artifact_name,
                int(artifact_version),
                storage_path,
                content_type,
                int(byte_size),
                checksum,
                phase,
                job_id,
                created_at,
            ),
        )

    def record_artifact_with_event(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        artifact_name: str,
        artifact_version: int,
        storage_path: str,
        content_type: str,
        byte_size: int,
        checksum: str,
        phase: str,
        job_id: str | None,
        event_type: str,
        event_payload: dict[str, Any],
    ) -> tuple[ResearchArtifactRow, ResearchRunEventRow]:
        artifact_id = f"ra_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        payload_json = json.dumps(event_payload or {}, sort_keys=True)
        with self._connect() as conn:
            self._record_artifact_with_conn(
                conn,
                artifact_id=artifact_id,
                session_id=session_id,
                artifact_name=artifact_name,
                artifact_version=artifact_version,
                storage_path=storage_path,
                content_type=content_type,
                byte_size=byte_size,
                checksum=checksum,
                phase=phase,
                job_id=job_id,
                created_at=now,
            )
            event_id = self._record_run_event_with_conn(
                conn,
                owner_user_id=str(owner_user_id),
                session_id=session_id,
                event_type=event_type,
                event_payload_json=payload_json,
                phase=phase,
                job_id=job_id,
                created_at=now,
            )
        artifact = self.get_artifact(artifact_id)
        event = self.get_run_event(event_id)
        if artifact is None:
            raise RuntimeError("failed_to_record_research_artifact")
        if event is None:
            raise RuntimeError("failed_to_record_research_run_event")
        return artifact, event

    def record_run_event(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        event_type: str,
        event_payload: dict[str, Any],
        phase: str | None = None,
        job_id: str | None = None,
    ) -> ResearchRunEventRow:
        now = _utc_now()
        payload_json = json.dumps(event_payload or {}, sort_keys=True)
        with self._connect() as conn:
            event_id = self._record_run_event_with_conn(
                conn,
                owner_user_id=str(owner_user_id),
                session_id=session_id,
                event_type=event_type,
                event_payload_json=payload_json,
                phase=phase,
                job_id=job_id,
                created_at=now,
            )
        event = self.get_run_event(event_id)
        if event is None:
            raise RuntimeError("failed_to_record_research_run_event")
        return event

    def _record_run_event_with_conn(
        self,
        conn: sqlite3.Connection,
        *,
        owner_user_id: str,
        session_id: str,
        event_type: str,
        event_payload_json: str,
        phase: str | None,
        job_id: str | None,
        created_at: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO research_run_events (
                session_id, owner_user_id, event_type, event_payload_json,
                phase, job_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                owner_user_id,
                event_type,
                event_payload_json,
                phase,
                job_id,
                created_at,
            ),
        )
        return int(cursor.lastrowid)

    def update_status_with_event(
        self,
        session_id: str,
        *,
        status: str,
        owner_user_id: str,
        event_type: str,
        event_payload: dict[str, Any],
        phase: str | None = None,
        job_id: str | None = None,
        control_state: str | None = None,
        active_job_id: str | None | object = _UNSET,
        completed_at: str | None = None,
    ) -> tuple[ResearchSessionRow, ResearchRunEventRow]:
        now = _utc_now()
        sql, params = self._build_update_status_statement(
            session_id=session_id,
            status=status,
            control_state=control_state,
            active_job_id=active_job_id,
            completed_at=completed_at,
            updated_at=now,
        )
        payload_json = json.dumps(event_payload or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(sql, params)
            event_id = self._record_run_event_with_conn(
                conn,
                owner_user_id=str(owner_user_id),
                session_id=session_id,
                event_type=event_type,
                event_payload_json=payload_json,
                phase=phase,
                job_id=job_id,
                created_at=now,
            )
        session = self.get_session(session_id)
        event = self.get_run_event(event_id)
        if session is None:
            raise KeyError(session_id)
        if event is None:
            raise RuntimeError("failed_to_record_research_run_event")
        return session, event

    def get_run_event(self, event_id: int) -> ResearchRunEventRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_run_events WHERE id = ?",
                (int(event_id),),
            ).fetchone()
        return self._run_event_from_row(row)

    def list_run_events_after(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        after_id: int,
        limit: int | None = None,
    ) -> list[ResearchRunEventRow]:
        sql = """
            SELECT * FROM research_run_events
            WHERE owner_user_id = ? AND session_id = ? AND id > ?
            ORDER BY id ASC
        """
        params: tuple[Any, ...] = (str(owner_user_id), session_id, int(after_id))
        if limit is not None:
            sql += " LIMIT ?"
            params = params + (int(limit),)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [event for row in rows if (event := self._run_event_from_row(row)) is not None]

    def get_latest_run_event(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        event_type: str,
    ) -> ResearchRunEventRow | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM research_run_events
                WHERE owner_user_id = ? AND session_id = ? AND event_type = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(owner_user_id), session_id, event_type),
            ).fetchone()
        return self._run_event_from_row(row)

    def get_latest_run_event_id(
        self,
        *,
        owner_user_id: str,
        session_id: str,
    ) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM research_run_events
                WHERE owner_user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(owner_user_id), session_id),
            ).fetchone()
        if row is None:
            return 0
        return int(row["id"])

    def get_artifact(self, artifact_id: str) -> ResearchArtifactRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()
        return self._artifact_from_row(row)

    def list_artifacts(self, session_id: str) -> list[ResearchArtifactRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM research_artifacts
                WHERE session_id = ?
                ORDER BY artifact_name ASC, artifact_version DESC, created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [artifact for row in rows if (artifact := self._artifact_from_row(row)) is not None]


__all__ = [
    "ResearchArtifactRow",
    "ResearchChatLinkedRunRow",
    "ResearchCheckpointRow",
    "ResearchRunEventRow",
    "ResearchSessionRow",
    "ResearchSessionsDB",
]
