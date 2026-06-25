from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, cast

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RPG.constants import (
    RPG_EVENT_SCHEMA_VERSION,
    RPG_REDUCER_VERSION,
    RPG_SNAPSHOT_SCHEMA_VERSION,
    RPG_SOURCE_TYPES,
)
from tldw_Server_API.app.core.RPG.errors import RPGConflictError, RPGNotFoundError, RPGValidationError
from tldw_Server_API.app.core.RPG.models import (
    RPGCampaign,
    RPGSession,
    RPGSessionEvent,
    RPGSnapshotRecord,
    RPGSnapshotState,
    RPGSourceType,
)


@dataclass(frozen=True, slots=True)
class CommitEventsResult:
    events: list[RPGSessionEvent]
    replayed: bool


class RPGRepository:
    """Persistence adapter for RPG campaign/session state in a user's ChaChaNotes DB."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> "RPGRepository":
        repo = cls(db)
        repo.ensure_schema()
        return repo

    def ensure_schema(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS rpg_campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                default_adapter_key TEXT NOT NULL,
                default_adapter_version TEXT NOT NULL,
                settings_json TEXT NOT NULL DEFAULT '{}',
                linked_rules_pack_refs_json TEXT NOT NULL DEFAULT '[]',
                version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rpg_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL REFERENCES rpg_campaigns(id) ON DELETE CASCADE,
                owner_user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                adapter_key TEXT NOT NULL,
                adapter_version TEXT NOT NULL,
                authority_settings_json TEXT NOT NULL DEFAULT '{}',
                linked_chat_id INTEGER,
                active_rules_pack_refs_json TEXT NOT NULL DEFAULT '[]',
                current_snapshot_version INTEGER NOT NULL DEFAULT 0,
                last_event_sequence INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rpg_session_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES rpg_sessions(id) ON DELETE CASCADE,
                owner_user_id INTEGER NOT NULL,
                base_event_sequence INTEGER NOT NULL,
                base_snapshot_version INTEGER NOT NULL,
                proposed_events_json TEXT NOT NULL,
                patch_json TEXT,
                rationale TEXT,
                confidence REAL,
                source_type TEXT NOT NULL,
                source_actor_id TEXT,
                model_metadata_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                review_notes TEXT,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                rejected_at TEXT,
                CHECK(source_type IN ('user', 'system', 'mcp', 'model', 'import')),
                CHECK(status IN ('pending', 'applied', 'rejected', 'expired', 'conflicted'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rpg_session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES rpg_sessions(id) ON DELETE CASCADE,
                owner_user_id INTEGER NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_payload_json TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_actor_id TEXT,
                source_label TEXT,
                operation_id INTEGER,
                event_schema_version TEXT NOT NULL,
                adapter_key TEXT NOT NULL,
                adapter_version TEXT NOT NULL,
                proposal_id INTEGER REFERENCES rpg_session_proposals(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                UNIQUE(owner_user_id, session_id, sequence_number),
                CHECK(source_type IN ('user', 'system', 'mcp', 'model', 'import'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rpg_idempotency_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_user_id INTEGER NOT NULL,
                session_id INTEGER REFERENCES rpg_sessions(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL,
                operation_scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_payload_hash TEXT NOT NULL,
                event_ids_json TEXT NOT NULL DEFAULT '[]',
                response_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(owner_user_id, source_type, operation_scope, idempotency_key),
                CHECK(source_type IN ('user', 'system', 'mcp', 'model', 'import'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rpg_session_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES rpg_sessions(id) ON DELETE CASCADE,
                owner_user_id INTEGER NOT NULL,
                snapshot_version INTEGER NOT NULL,
                last_event_sequence INTEGER NOT NULL,
                reducer_version TEXT NOT NULL,
                snapshot_schema_version TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(owner_user_id, session_id, snapshot_version)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_rpg_sessions_campaign ON rpg_sessions(owner_user_id, campaign_id)",
            "CREATE INDEX IF NOT EXISTS idx_rpg_events_session ON rpg_session_events(owner_user_id, session_id, sequence_number)",
            "CREATE INDEX IF NOT EXISTS idx_rpg_snapshots_latest ON rpg_session_snapshots(owner_user_id, session_id, snapshot_version DESC)",
        )
        with self.db.transaction() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            for statement in statements:
                conn.execute(statement)
        logger.debug("RPG repository schema ensured")

    def create_campaign(
        self,
        owner_user_id: int,
        title: str,
        description: str | None,
        default_adapter_key: str,
        default_adapter_version: str,
        settings: dict[str, Any],
        linked_rules_pack_refs: list[dict[str, Any]],
        idempotency_key: str,
        request_payload_hash: str,
        source_type: str,
    ) -> RPGCampaign:
        self._validate_source_type(source_type)
        operation_scope = "campaigns"
        now = self._now()
        with self.db.transaction() as conn:
            replay = self._find_idempotency_record(
                conn,
                owner_user_id=owner_user_id,
                session_id=None,
                source_type=source_type,
                operation_scope=operation_scope,
                idempotency_key=idempotency_key,
            )
            if replay is not None:
                self._ensure_replay_hash(replay, request_payload_hash)
                campaign_id = int(self._from_json(self._row_value(replay, "response_json"))["campaign_id"])
                return self._get_campaign_with_conn(conn, owner_user_id, campaign_id)

            cursor = conn.execute(
                """
                INSERT INTO rpg_campaigns (
                    owner_user_id, title, description, default_adapter_key, default_adapter_version,
                    settings_json, linked_rules_pack_refs_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    title,
                    description,
                    default_adapter_key,
                    default_adapter_version,
                    self._to_json(settings),
                    self._to_json(linked_rules_pack_refs),
                    now,
                    now,
                ),
            )
            campaign_id = int(cursor.lastrowid)
            self._insert_idempotency_record(
                conn,
                owner_user_id=owner_user_id,
                session_id=None,
                source_type=source_type,
                operation_scope=operation_scope,
                idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash,
                event_ids=[],
                response={"campaign_id": campaign_id},
                created_at=now,
            )
            return self._get_campaign_with_conn(conn, owner_user_id, campaign_id)

    def create_session(
        self,
        owner_user_id: int,
        campaign_id: int,
        title: str,
        adapter_key: str,
        adapter_version: str,
        authority_settings: dict[str, Any],
        linked_chat_id: int | None,
        active_rules_pack_refs: list[dict[str, Any]],
        idempotency_key: str,
        request_payload_hash: str,
        source_type: str,
    ) -> RPGSession:
        self._validate_source_type(source_type)
        operation_scope = f"campaign:{campaign_id}:sessions"
        now = self._now()
        with self.db.transaction() as conn:
            replay = self._find_idempotency_record(
                conn,
                owner_user_id=owner_user_id,
                session_id=None,
                source_type=source_type,
                operation_scope=operation_scope,
                idempotency_key=idempotency_key,
            )
            if replay is not None:
                self._ensure_replay_hash(replay, request_payload_hash)
                session_id = int(self._from_json(self._row_value(replay, "response_json"))["session_id"])
                return self._get_session_with_conn(conn, owner_user_id, session_id)

            self._get_campaign_with_conn(conn, owner_user_id, campaign_id)
            cursor = conn.execute(
                """
                INSERT INTO rpg_sessions (
                    campaign_id, owner_user_id, title, adapter_key, adapter_version,
                    authority_settings_json, linked_chat_id, active_rules_pack_refs_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    owner_user_id,
                    title,
                    adapter_key,
                    adapter_version,
                    self._to_json(authority_settings),
                    linked_chat_id,
                    self._to_json(active_rules_pack_refs),
                    now,
                    now,
                ),
            )
            session_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO rpg_session_snapshots (
                    session_id, owner_user_id, snapshot_version, last_event_sequence,
                    reducer_version, snapshot_schema_version, snapshot_json, diagnostics_json, created_at
                ) VALUES (?, ?, 0, 0, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    RPG_REDUCER_VERSION,
                    RPG_SNAPSHOT_SCHEMA_VERSION,
                    self._to_json(asdict(RPGSnapshotState())),
                    self._to_json({"initial": True}),
                    now,
                ),
            )
            self._insert_idempotency_record(
                conn,
                owner_user_id=owner_user_id,
                session_id=session_id,
                source_type=source_type,
                operation_scope=operation_scope,
                idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash,
                event_ids=[],
                response={"session_id": session_id},
                created_at=now,
            )
            return self._get_session_with_conn(conn, owner_user_id, session_id)

    def get_session(self, owner_user_id: int, session_id: int) -> RPGSession:
        with self.db.transaction() as conn:
            return self._get_session_with_conn(conn, owner_user_id, session_id)

    def get_event(self, owner_user_id: int, event_id: int) -> RPGSessionEvent:
        with self.db.transaction() as conn:
            return self._get_event_with_conn(conn, owner_user_id, event_id)

    def get_latest_snapshot(self, owner_user_id: int, session_id: int) -> RPGSnapshotRecord:
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM rpg_session_snapshots
                WHERE owner_user_id = ? AND session_id = ?
                ORDER BY snapshot_version DESC
                LIMIT 1
                """,
                (owner_user_id, session_id),
            ).fetchone()
            if row is None:
                raise RPGNotFoundError("rpg_snapshot_not_found")
            return self._snapshot_from_row(row)

    def commit_events_and_snapshot(
        self,
        owner_user_id: int,
        session_id: int,
        expected_last_event_sequence: int,
        base_snapshot_version: int,
        events: list[dict[str, Any]],
        snapshot: dict[str, Any],
        diagnostics: dict[str, Any],
        idempotency_key: str,
        request_payload_hash: str,
        adapter_key: str,
        adapter_version: str,
        proposal_id: int | None,
        proposal_review_notes: str | None = None,
    ) -> CommitEventsResult:
        if not events:
            raise RPGConflictError("events_required")
        source_type = self._event_source_type(events)
        operation_scope = f"session:{session_id}:events"
        now = self._now()
        try:
            with self.db.transaction() as conn:
                replay = self._find_idempotency_record(
                    conn,
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    source_type=source_type,
                    operation_scope=operation_scope,
                    idempotency_key=idempotency_key,
                )
                if replay is not None:
                    self._ensure_replay_hash(replay, request_payload_hash)
                    event_ids = self._from_json(self._row_value(replay, "event_ids_json"))
                    return CommitEventsResult(
                        events=self._events_by_ids(conn, owner_user_id, [int(event_id) for event_id in event_ids]),
                        replayed=True,
                    )

                session = self._get_session_with_conn(conn, owner_user_id, session_id)
                if session.last_event_sequence != expected_last_event_sequence:
                    raise RPGConflictError("stale_event_sequence")
                if session.current_snapshot_version != base_snapshot_version:
                    raise RPGConflictError("stale_snapshot_version")

                event_ids: list[int] = []
                for offset, event in enumerate(events):
                    event_source_type = event["source_type"]
                    self._validate_source_type(event_source_type)
                    cursor = conn.execute(
                        """
                        INSERT INTO rpg_session_events (
                            session_id, owner_user_id, sequence_number, event_type, event_payload_json,
                            source_type, source_actor_id, source_label, operation_id,
                            event_schema_version, adapter_key, adapter_version, proposal_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            owner_user_id,
                            expected_last_event_sequence + offset + 1,
                            event["event_type"],
                            self._to_json(event["event_payload"]),
                            event_source_type,
                            event.get("source_actor_id"),
                            event.get("source_label"),
                            event.get("event_schema_version", RPG_EVENT_SCHEMA_VERSION),
                            adapter_key,
                            adapter_version,
                            proposal_id,
                            now,
                        ),
                    )
                    event_ids.append(int(cursor.lastrowid))

                next_sequence = expected_last_event_sequence + len(events)
                next_snapshot_version = base_snapshot_version + 1
                conn.execute(
                    """
                    INSERT INTO rpg_session_snapshots (
                        session_id, owner_user_id, snapshot_version, last_event_sequence,
                        reducer_version, snapshot_schema_version, snapshot_json, diagnostics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        owner_user_id,
                        next_snapshot_version,
                        next_sequence,
                        RPG_REDUCER_VERSION,
                        RPG_SNAPSHOT_SCHEMA_VERSION,
                        self._to_json(snapshot),
                        self._to_json(diagnostics),
                        now,
                    ),
                )
                update_cursor = conn.execute(
                    """
                    UPDATE rpg_sessions
                    SET last_event_sequence = ?, current_snapshot_version = ?,
                        version = version + 1, updated_at = ?
                    WHERE id = ? AND owner_user_id = ?
                      AND last_event_sequence = ? AND current_snapshot_version = ?
                    """,
                    (
                        next_sequence,
                        next_snapshot_version,
                        now,
                        session_id,
                        owner_user_id,
                        expected_last_event_sequence,
                        base_snapshot_version,
                    ),
                )
                if update_cursor.rowcount != 1:
                    raise RPGConflictError("stale_session_cursor")
                if proposal_id is not None:
                    proposal_cursor = conn.execute(
                        """
                        UPDATE rpg_session_proposals
                        SET status = 'applied', applied_at = ?, review_notes = COALESCE(?, review_notes)
                        WHERE id = ? AND owner_user_id = ? AND session_id = ? AND status = 'pending'
                        """,
                        (now, proposal_review_notes, proposal_id, owner_user_id, session_id),
                    )
                    if proposal_cursor.rowcount != 1:
                        raise RPGConflictError("proposal_not_pending")

                operation_id = self._insert_idempotency_record(
                    conn,
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    source_type=source_type,
                    operation_scope=operation_scope,
                    idempotency_key=idempotency_key,
                    request_payload_hash=request_payload_hash,
                    event_ids=event_ids,
                    response={"event_ids": event_ids},
                    created_at=now,
                )
                self._set_event_operation_id(conn, operation_id, event_ids)
                return CommitEventsResult(
                    events=self._events_by_ids(conn, owner_user_id, event_ids),
                    replayed=False,
                )
        except sqlite3.IntegrityError as exc:
            raise RPGConflictError("rpg_event_append_conflict") from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _from_json(value: str | None, default: Any = None) -> Any:
        if value is None:
            return default
        return json.loads(value)

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @staticmethod
    def _row_value(row: Any, key: str) -> Any:
        if isinstance(row, dict):
            return row[key]
        return row[key]

    @classmethod
    def _campaign_from_row(cls, row: Any) -> RPGCampaign:
        return RPGCampaign(
            id=int(cls._row_value(row, "id")),
            owner_user_id=int(cls._row_value(row, "owner_user_id")),
            title=str(cls._row_value(row, "title")),
            description=cls._row_value(row, "description"),
            default_adapter_key=str(cls._row_value(row, "default_adapter_key")),
            default_adapter_version=str(cls._row_value(row, "default_adapter_version")),
            settings=cls._from_json(cls._row_value(row, "settings_json"), {}),
            linked_rules_pack_refs=cls._from_json(cls._row_value(row, "linked_rules_pack_refs_json"), []),
            version=int(cls._row_value(row, "version")),
            status=str(cls._row_value(row, "status")),
            created_at=cls._parse_datetime(str(cls._row_value(row, "created_at"))),
            updated_at=cls._parse_datetime(str(cls._row_value(row, "updated_at"))),
        )

    @classmethod
    def _session_from_row(cls, row: Any) -> RPGSession:
        return RPGSession(
            id=int(cls._row_value(row, "id")),
            campaign_id=int(cls._row_value(row, "campaign_id")),
            owner_user_id=int(cls._row_value(row, "owner_user_id")),
            title=str(cls._row_value(row, "title")),
            status=str(cls._row_value(row, "status")),
            adapter_key=str(cls._row_value(row, "adapter_key")),
            adapter_version=str(cls._row_value(row, "adapter_version")),
            authority_settings=cls._from_json(cls._row_value(row, "authority_settings_json"), {}),
            linked_chat_id=cls._row_value(row, "linked_chat_id"),
            active_rules_pack_refs=cls._from_json(cls._row_value(row, "active_rules_pack_refs_json"), []),
            current_snapshot_version=int(cls._row_value(row, "current_snapshot_version")),
            last_event_sequence=int(cls._row_value(row, "last_event_sequence")),
            version=int(cls._row_value(row, "version")),
            created_at=cls._parse_datetime(str(cls._row_value(row, "created_at"))),
            updated_at=cls._parse_datetime(str(cls._row_value(row, "updated_at"))),
        )

    @classmethod
    def _event_from_row(cls, row: Any) -> RPGSessionEvent:
        return RPGSessionEvent(
            id=int(cls._row_value(row, "id")),
            session_id=int(cls._row_value(row, "session_id")),
            owner_user_id=int(cls._row_value(row, "owner_user_id")),
            sequence_number=int(cls._row_value(row, "sequence_number")),
            event_type=str(cls._row_value(row, "event_type")),
            event_payload=cls._from_json(cls._row_value(row, "event_payload_json"), {}),
            source_type=cast(RPGSourceType, cls._row_value(row, "source_type")),
            source_actor_id=cls._row_value(row, "source_actor_id"),
            source_label=cls._row_value(row, "source_label"),
            operation_id=cls._row_value(row, "operation_id"),
            event_schema_version=str(cls._row_value(row, "event_schema_version")),
            adapter_key=str(cls._row_value(row, "adapter_key")),
            adapter_version=str(cls._row_value(row, "adapter_version")),
            proposal_id=cls._row_value(row, "proposal_id"),
            created_at=cls._parse_datetime(str(cls._row_value(row, "created_at"))),
        )

    @classmethod
    def _snapshot_from_row(cls, row: Any) -> RPGSnapshotRecord:
        return RPGSnapshotRecord(
            id=int(cls._row_value(row, "id")),
            session_id=int(cls._row_value(row, "session_id")),
            owner_user_id=int(cls._row_value(row, "owner_user_id")),
            snapshot_version=int(cls._row_value(row, "snapshot_version")),
            last_event_sequence=int(cls._row_value(row, "last_event_sequence")),
            reducer_version=str(cls._row_value(row, "reducer_version")),
            snapshot_schema_version=str(cls._row_value(row, "snapshot_schema_version")),
            snapshot_json=cls._from_json(cls._row_value(row, "snapshot_json"), {}),
            diagnostics_json=cls._from_json(cls._row_value(row, "diagnostics_json"), {}),
            created_at=cls._parse_datetime(str(cls._row_value(row, "created_at"))),
        )

    def _get_campaign_with_conn(self, conn: Any, owner_user_id: int, campaign_id: int) -> RPGCampaign:
        row = conn.execute(
            """
            SELECT *
            FROM rpg_campaigns
            WHERE id = ? AND owner_user_id = ?
            """,
            (campaign_id, owner_user_id),
        ).fetchone()
        if row is None:
            raise RPGNotFoundError("rpg_campaign_not_found")
        return self._campaign_from_row(row)

    def _get_session_with_conn(self, conn: Any, owner_user_id: int, session_id: int) -> RPGSession:
        row = conn.execute(
            """
            SELECT *
            FROM rpg_sessions
            WHERE id = ? AND owner_user_id = ?
            """,
            (session_id, owner_user_id),
        ).fetchone()
        if row is None:
            raise RPGNotFoundError("rpg_session_not_found")
        return self._session_from_row(row)

    def _get_event_with_conn(self, conn: Any, owner_user_id: int, event_id: int) -> RPGSessionEvent:
        row = conn.execute(
            """
            SELECT *
            FROM rpg_session_events
            WHERE id = ? AND owner_user_id = ?
            """,
            (event_id, owner_user_id),
        ).fetchone()
        if row is None:
            raise RPGNotFoundError("rpg_event_not_found")
        return self._event_from_row(row)

    def _events_by_ids(self, conn: Any, owner_user_id: int, event_ids: list[int]) -> list[RPGSessionEvent]:
        return [self._get_event_with_conn(conn, owner_user_id, event_id) for event_id in event_ids]

    def _find_idempotency_record(
        self,
        conn: Any,
        *,
        owner_user_id: int,
        session_id: int | None,
        source_type: str,
        operation_scope: str,
        idempotency_key: str,
    ) -> Any | None:
        if session_id is None:
            return conn.execute(
                """
                SELECT *
                FROM rpg_idempotency_records
                WHERE owner_user_id = ?
                  AND source_type = ?
                  AND operation_scope = ?
                  AND idempotency_key = ?
                """,
                (owner_user_id, source_type, operation_scope, idempotency_key),
            ).fetchone()
        return conn.execute(
            """
            SELECT *
            FROM rpg_idempotency_records
            WHERE owner_user_id = ?
              AND source_type = ?
              AND operation_scope = ?
              AND idempotency_key = ?
              AND session_id = ?
            """,
            (owner_user_id, source_type, operation_scope, idempotency_key, session_id),
        ).fetchone()

    def _insert_idempotency_record(
        self,
        conn: Any,
        *,
        owner_user_id: int,
        session_id: int | None,
        source_type: str,
        operation_scope: str,
        idempotency_key: str,
        request_payload_hash: str,
        event_ids: list[int],
        response: dict[str, Any],
        created_at: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO rpg_idempotency_records (
                owner_user_id, session_id, source_type, operation_scope, idempotency_key,
                request_payload_hash, event_ids_json, response_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_user_id,
                session_id,
                source_type,
                operation_scope,
                idempotency_key,
                request_payload_hash,
                self._to_json(event_ids),
                self._to_json(response),
                created_at,
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _ensure_replay_hash(row: Any, request_payload_hash: str) -> None:
        row_hash = row["request_payload_hash"] if isinstance(row, dict) else row["request_payload_hash"]
        if row_hash != request_payload_hash:
            raise RPGConflictError("idempotency_key_conflict")

    def _set_event_operation_id(self, conn: Any, operation_id: int, event_ids: list[int]) -> None:
        for event_id in event_ids:
            conn.execute(
                "UPDATE rpg_session_events SET operation_id = ? WHERE id = ?",
                (operation_id, event_id),
            )

    @staticmethod
    def _validate_source_type(source_type: str) -> None:
        if source_type not in RPG_SOURCE_TYPES:
            raise RPGValidationError("invalid_source_type")

    def _event_source_type(self, events: list[dict[str, Any]]) -> str:
        source_type = str(events[0]["source_type"])
        self._validate_source_type(source_type)
        for event in events[1:]:
            if event["source_type"] != source_type:
                raise RPGConflictError("mixed_event_sources")
        return source_type
