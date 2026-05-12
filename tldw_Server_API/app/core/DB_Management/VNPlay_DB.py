"""VN Play runtime storage for per-user ChaChaNotes databases."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

STORY_BRANCH_LABEL_MAX_LENGTH = 160


VN_PLAY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vn_play_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    primary_character_id INTEGER NOT NULL,
    additional_character_ids_json TEXT NOT NULL DEFAULT '[]',
    linked_chat_id TEXT,
    vn_asset_pack_id INTEGER NOT NULL,
    asset_manifest_version TEXT,
    source_world_book_ids_json TEXT NOT NULL DEFAULT '[]',
    content_rating TEXT NOT NULL DEFAULT 'general',
    trust_level TEXT NOT NULL DEFAULT 'local',
    linked_chat_mode TEXT NOT NULL DEFAULT 'read_only_context',
    seed TEXT,
    settings_json TEXT NOT NULL DEFAULT '{}',
    script_id INTEGER,
    script_version_id INTEGER,
    script_manifest_snapshot_id INTEGER,
    script_policy_snapshot_id INTEGER,
    script_generation_profile_snapshot_id INTEGER,
    script_position_json TEXT NOT NULL DEFAULT '{}',
    scene_version INTEGER NOT NULL DEFAULT 0,
    active_turn_request_id INTEGER,
    active_session_action_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vn_play_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    sequence_number INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_payload_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'runtime',
    model_provider TEXT,
    model_name TEXT,
    branch_node_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS vn_play_turn_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_payload_hash TEXT NOT NULL,
    base_scene_version INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    input_event_id INTEGER REFERENCES vn_play_events(id),
    turn_started_event_id INTEGER REFERENCES vn_play_events(id),
    turn_completed_event_id INTEGER REFERENCES vn_play_events(id),
    response_payload_json TEXT,
    error_json TEXT,
    lease_owner TEXT,
    locked_until DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS vn_play_session_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    response_payload_json TEXT,
    error_json TEXT,
    lease_owner TEXT,
    locked_until DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS vn_play_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    script_id INTEGER,
    script_version_id INTEGER,
    generation_point_key TEXT NOT NULL,
    opcode_id TEXT,
    opcode_label TEXT,
    opcode_index INTEGER,
    output_schema TEXT NOT NULL,
    generation_profile_key TEXT NOT NULL,
    generation_profile_snapshot_id INTEGER NOT NULL,
    active_revision_id INTEGER,
    latest_request_id INTEGER,
    status TEXT NOT NULL DEFAULT 'not_started',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, generation_point_key)
);

CREATE TABLE IF NOT EXISTS vn_play_generation_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES vn_play_generations(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    script_id INTEGER,
    script_version_id INTEGER,
    generation_point_key TEXT NOT NULL,
    generation_profile_key TEXT NOT NULL,
    generation_profile_snapshot_id INTEGER NOT NULL,
    request_kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_confirmation',
    create_action_id INTEGER,
    execute_action_id INTEGER,
    cancel_action_id INTEGER,
    client_scene_version INTEGER NOT NULL DEFAULT 0,
    opcode_snapshot_json TEXT NOT NULL DEFAULT '{}',
    prompt_fingerprint TEXT,
    checkpoint_id_before INTEGER,
    provider_call_started_at DATETIME,
    provider_call_completed_at DATETIME,
    lease_expires_at DATETIME,
    public_error_code TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_play_generation_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    generation_id INTEGER REFERENCES vn_play_generations(id) ON DELETE CASCADE,
    generation_request_id INTEGER REFERENCES vn_play_generation_requests(id) ON DELETE CASCADE,
    generation_revision_id INTEGER REFERENCES vn_play_generation_revisions(id) ON DELETE SET NULL,
    action_kind TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    completed_action_response_json TEXT,
    public_error_code TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, idempotency_key),
    UNIQUE(
        owner_user_id,
        session_id,
        action_kind,
        generation_request_id,
        idempotency_key
    )
);

CREATE TABLE IF NOT EXISTS vn_play_generation_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES vn_play_generations(id) ON DELETE CASCADE,
    generation_request_id INTEGER NOT NULL REFERENCES vn_play_generation_requests(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    generation_point_key TEXT NOT NULL,
    generation_profile_key TEXT NOT NULL,
    generation_profile_snapshot_id INTEGER NOT NULL,
    revision_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    output_schema TEXT NOT NULL,
    public_output_json TEXT NOT NULL DEFAULT '{}',
    applied_visuals_json TEXT NOT NULL DEFAULT '[]',
    rejected_visuals_json TEXT NOT NULL DEFAULT '[]',
    public_error_code TEXT,
    raw_output_debug_json TEXT,
    parser_diagnostics_json TEXT NOT NULL DEFAULT '{}',
    moderation_diagnostics_json TEXT NOT NULL DEFAULT '{}',
    model_metadata_json TEXT NOT NULL DEFAULT '{}',
    usage_metadata_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'model',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, generation_id, revision_number)
);

CREATE TABLE IF NOT EXISTS vn_play_scene_state (
    session_id INTEGER PRIMARY KEY REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    last_event_id INTEGER REFERENCES vn_play_events(id),
    current_background_item_id INTEGER,
    current_depth_item_id INTEGER,
    active_sprite_items_json TEXT NOT NULL DEFAULT '[]',
    location_key TEXT,
    mood TEXT,
    time_of_day TEXT,
    weather TEXT,
    active_branch_node_id INTEGER,
    visible_choices_json TEXT NOT NULL DEFAULT '[]',
    transcript_cursor INTEGER,
    scene_version INTEGER NOT NULL DEFAULT 0,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_play_branches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    parent_event_id INTEGER REFERENCES vn_play_events(id),
    branch_label TEXT,
    branch_path_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_play_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    event_id INTEGER REFERENCES vn_play_events(id),
    scene_version INTEGER NOT NULL,
    scene_state_snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_play_save_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES vn_play_sessions(id) ON DELETE CASCADE,
    owner_user_id INTEGER NOT NULL,
    slot_key TEXT NOT NULL,
    title TEXT NOT NULL,
    checkpoint_id INTEGER NOT NULL REFERENCES vn_play_checkpoints(id) ON DELETE RESTRICT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    deleted BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, session_id, slot_key)
);

CREATE INDEX IF NOT EXISTS idx_vn_play_sessions_owner_user_id
    ON vn_play_sessions(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_sessions_owner_status
    ON vn_play_sessions(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_vn_play_sessions_pack_id
    ON vn_play_sessions(vn_asset_pack_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_events_session_sequence
    ON vn_play_events(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_vn_play_events_session_branch_sequence
    ON vn_play_events(session_id, branch_node_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_vn_play_events_owner_user_id
    ON vn_play_events(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_turn_requests_session
    ON vn_play_turn_requests(session_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_turn_requests_owner_status
    ON vn_play_turn_requests(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_vn_play_session_actions_session
    ON vn_play_session_actions(session_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_session_actions_owner_status
    ON vn_play_session_actions(owner_user_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vn_play_generations_owner_session_point_unique
    ON vn_play_generations(owner_user_id, session_id, generation_point_key);
CREATE INDEX IF NOT EXISTS idx_vn_play_generations_session
    ON vn_play_generations(session_id, owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_generations_owner_status
    ON vn_play_generations(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_vn_play_generation_requests_generation
    ON vn_play_generation_requests(owner_user_id, session_id, generation_id, id);
CREATE INDEX IF NOT EXISTS idx_vn_play_generation_requests_owner_status
    ON vn_play_generation_requests(owner_user_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vn_play_generation_actions_owner_session_key_unique
    ON vn_play_generation_actions(owner_user_id, session_id, idempotency_key);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vn_play_generation_actions_request_key_unique
    ON vn_play_generation_actions(
        owner_user_id,
        session_id,
        action_kind,
        generation_request_id,
        idempotency_key
    );
CREATE INDEX IF NOT EXISTS idx_vn_play_generation_actions_session
    ON vn_play_generation_actions(owner_user_id, session_id, idempotency_key);
CREATE INDEX IF NOT EXISTS idx_vn_play_generation_actions_request
    ON vn_play_generation_actions(owner_user_id, session_id, generation_request_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_generation_revisions_history
    ON vn_play_generation_revisions(owner_user_id, session_id, generation_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_vn_play_generation_revisions_request
    ON vn_play_generation_revisions(generation_request_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_scene_state_owner_user_id
    ON vn_play_scene_state(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_branches_session
    ON vn_play_branches(session_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_checkpoints_session
    ON vn_play_checkpoints(session_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_save_slots_session
    ON vn_play_save_slots(session_id, deleted);
"""

VN_PLAY_SCHEMA_STATEMENTS = tuple(
    statement.strip()
    for statement in VN_PLAY_SCHEMA_SQL.split(";")
    if statement.strip()
)


def ensure_vn_play_tables(db: CharactersRAGDB) -> None:
    """Create VN Play runtime tables in the provided ChaChaNotes database."""
    _require_sqlite_chacha_db(db)
    with db.transaction() as conn:
        for statement in VN_PLAY_SCHEMA_STATEMENTS:
            if _is_index_statement(statement):
                continue
            conn.execute(statement)
        _ensure_vn_play_session_columns(conn)
        _ensure_vn_play_generation_columns(conn)
        for statement in VN_PLAY_SCHEMA_STATEMENTS:
            if _is_index_statement(statement):
                conn.execute(statement)


class VNPlayRepository:
    """Repository for VN Play sessions and events in a user's ChaChaNotes DB."""

    def __init__(self, db: CharactersRAGDB):
        _require_sqlite_chacha_db(db)
        self.db = db
        self._schema_initialized = False

    @classmethod
    def initialized(cls, db: CharactersRAGDB) -> VNPlayRepository:
        repo = cls(db)
        repo.initialize_schema()
        return repo

    def initialize_schema(self) -> None:
        ensure_vn_play_tables(self.db)
        self._schema_initialized = True

    def create_session(
        self,
        *,
        owner_user_id: int,
        mode: str,
        title: str,
        primary_character_id: int,
        vn_asset_pack_id: int,
        additional_character_ids: Sequence[int] | None = None,
        linked_chat_id: str | None = None,
        asset_manifest_version: str | None = None,
        source_world_book_ids: Sequence[int] | None = None,
        content_rating: str = "general",
        trust_level: str = "local",
        linked_chat_mode: str = "read_only_context",
        seed: str | None = None,
        settings: Mapping[str, Any] | None = None,
        script_id: int | None = None,
        script_version_id: int | None = None,
        script_manifest_snapshot_id: int | None = None,
        script_policy_snapshot_id: int | None = None,
        script_generation_profile_snapshot_id: int | None = None,
        script_position: Mapping[str, Any] | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_play_sessions (
                    owner_user_id,
                    mode,
                    title,
                    status,
                    primary_character_id,
                    additional_character_ids_json,
                    linked_chat_id,
                    vn_asset_pack_id,
                    asset_manifest_version,
                    source_world_book_ids_json,
                    content_rating,
                    trust_level,
                    linked_chat_mode,
                    seed,
                    settings_json,
                    script_id,
                    script_version_id,
                    script_manifest_snapshot_id,
                    script_policy_snapshot_id,
                    script_generation_profile_snapshot_id,
                    script_position_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    mode,
                    title,
                    status,
                    primary_character_id,
                    _json_dump(list(additional_character_ids or [])),
                    linked_chat_id,
                    vn_asset_pack_id,
                    asset_manifest_version,
                    _json_dump(list(source_world_book_ids or [])),
                    content_rating,
                    trust_level,
                    linked_chat_mode,
                    seed,
                    _json_dump(dict(settings or {})),
                    script_id,
                    script_version_id,
                    script_manifest_snapshot_id,
                    script_policy_snapshot_id,
                    script_generation_profile_snapshot_id,
                    _json_dump(dict(script_position or {})),
                ),
            )
            session_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO vn_play_scene_state (
                    session_id,
                    owner_user_id
                )
                VALUES (?, ?)
                """,
                (session_id, owner_user_id),
            )

        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError("created_session_not_found")
        return session

    def get_session(
        self,
        session_id: int,
        *,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        if owner_user_id is None and include_deleted:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_play_sessions WHERE id = ?",
                (session_id,),
            )
        elif owner_user_id is None:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_play_sessions WHERE id = ? AND deleted = 0",
                (session_id,),
            )
        elif include_deleted:
            cursor = self.db.execute_query(
                "SELECT * FROM vn_play_sessions WHERE id = ? AND owner_user_id = ?",
                (session_id, owner_user_id),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE id = ? AND owner_user_id = ? AND deleted = 0
                """,
                (session_id, owner_user_id),
            )
        row = cursor.fetchone()
        return _decode_session(row) if row is not None else None

    def list_sessions(
        self,
        *,
        owner_user_id: int,
        include_deleted: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        if include_deleted and limit is None:
            cursor = self.db.execute_query(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE owner_user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (owner_user_id,),
            )
        elif include_deleted:
            cursor = self.db.execute_query(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE owner_user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (owner_user_id, limit, offset),
            )
        elif limit is None:
            cursor = self.db.execute_query(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE owner_user_id = ? AND deleted = 0
                ORDER BY updated_at DESC, id DESC
                """,
                (owner_user_id,),
            )
        else:
            cursor = self.db.execute_query(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE owner_user_id = ? AND deleted = 0
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (owner_user_id, limit, offset),
            )
        return [_decode_session(row) for row in cursor.fetchall()]

    def update_session(
        self,
        session_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        update_values = _mapped_update_values(
            fields,
            _SESSION_UPDATE_COLUMNS,
            json_fields={
                "additional_character_ids",
                "source_world_book_ids",
                "settings",
                "script_position",
            },
        )
        if not update_values:
            return self.get_session(session_id, owner_user_id=owner_user_id)

        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _SESSION_UPDATE_STATEMENTS[field_name]
                conn.execute(
                    statement,
                    (value, session_id, owner_user_id, owner_user_id),
                )
        return self.get_session(session_id, owner_user_id=owner_user_id)

    def try_acquire_turn_lock(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        turn_request_id: int,
        expected_scene_version: int,
    ) -> bool:
        """Attach an active turn to a session if its scene version is still current."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE vn_play_sessions
                SET active_turn_request_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND owner_user_id = ?
                  AND deleted = 0
                  AND active_turn_request_id IS NULL
                  AND active_session_action_id IS NULL
                  AND scene_version = ?
                """,
                (
                    turn_request_id,
                    session_id,
                    owner_user_id,
                    expected_scene_version,
                ),
            )
            return cursor.rowcount == 1

    def try_acquire_session_action_lock(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_id: int,
        expected_scene_version: int,
    ) -> bool:
        """Attach an active session action if no turn or restore action is running."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE vn_play_sessions
                SET active_session_action_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND owner_user_id = ?
                  AND deleted = 0
                  AND active_turn_request_id IS NULL
                  AND active_session_action_id IS NULL
                  AND scene_version = ?
                  AND EXISTS (
                      SELECT 1
                      FROM vn_play_session_actions AS action
                      WHERE action.id = ?
                        AND action.session_id = vn_play_sessions.id
                        AND action.owner_user_id = vn_play_sessions.owner_user_id
                        AND action.status IN ('pending', 'abandoned')
                  )
                """,
                (
                    action_id,
                    session_id,
                    owner_user_id,
                    expected_scene_version,
                    action_id,
                ),
            )
            return cursor.rowcount == 1

    def clear_session_action_lock(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_id: int | None = None,
    ) -> None:
        """Clear the active session action marker, optionally guarded by action id."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_play_sessions
                SET active_session_action_id = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND owner_user_id = ?
                  AND (? IS NULL OR active_session_action_id = ?)
                """,
                (session_id, owner_user_id, action_id, action_id),
            )

    def append_event(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        event_type: str,
        event_payload: Mapping[str, Any] | None = None,
        source: str = "runtime",
        model_provider: str | None = None,
        model_name: str | None = None,
        branch_node_id: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            event_id = _insert_event(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                event_type=event_type,
                event_payload=event_payload,
                source=source,
                model_provider=model_provider,
                model_name=model_name,
                branch_node_id=branch_node_id,
            )

        event = self.get_event(event_id)
        if event is None:
            raise RuntimeError("created_event_not_found")
        return event

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_play_events WHERE id = ?",
            (event_id,),
        )
        row = cursor.fetchone()
        return _decode_event(row) if row is not None else None

    def list_events(
        self,
        session_id: int,
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_events
            WHERE session_id = ?
              AND (? IS NULL OR sequence_number > ?)
            ORDER BY sequence_number ASC
            LIMIT COALESCE(?, -1)
            """,
            (session_id, after_sequence, after_sequence, limit),
        )
        return [_decode_event(row) for row in cursor.fetchall()]

    def can_filter_branch_events_by_tags(self, session_id: int) -> bool:
        """Return true when explicit branch tags can satisfy branch event filtering."""
        self._ensure_schema_initialized()
        first_tagged_cursor = self.db.execute_query(
            """
            SELECT MIN(sequence_number) AS first_tagged_sequence
            FROM vn_play_events
            WHERE session_id = ?
              AND branch_node_id IS NOT NULL
            """,
            (session_id,),
        )
        first_tagged = first_tagged_cursor.fetchone()
        first_tagged_sequence = (
            first_tagged["first_tagged_sequence"]
            if first_tagged is not None
            else None
        )
        if first_tagged_sequence is None:
            return False

        untagged_cursor = self.db.execute_query(
            """
            SELECT 1
            FROM vn_play_events
            WHERE session_id = ?
              AND branch_node_id IS NULL
              AND sequence_number > ?
            LIMIT 1
            """,
            (session_id, first_tagged_sequence),
        )
        return untagged_cursor.fetchone() is None

    def list_events_for_branch_nodes(
        self,
        session_id: int,
        branch_node_ids: Sequence[int],
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List events for explicit branch ids using SQL-level filtering."""
        self._ensure_schema_initialized()
        normalized_branch_ids = sorted({int(branch_id) for branch_id in branch_node_ids})
        if not normalized_branch_ids:
            return []

        events: list[dict[str, Any]] = []
        for branch_node_id in normalized_branch_ids:
            cursor = self.db.execute_query(
                """
                SELECT *
                FROM vn_play_events
                WHERE session_id = ?
                  AND branch_node_id = ?
                  AND (? IS NULL OR sequence_number > ?)
                ORDER BY sequence_number ASC
                LIMIT COALESCE(?, -1)
                """,
                (
                    session_id,
                    branch_node_id,
                    after_sequence,
                    after_sequence,
                    limit,
                ),
            )
            events.extend(_decode_event(row) for row in cursor.fetchall())

        events.sort(key=lambda event: int(event["sequence_number"]))
        if limit is None:
            return events
        return events[: max(0, limit)]

    def create_turn_request(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        idempotency_key: str,
        request_payload_hash: str,
        base_scene_version: int,
        status: str = "pending",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        existing = self.get_turn_request_by_key(
            session_id=session_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if existing["request_payload_hash"] != request_payload_hash:
                raise ValueError("idempotency_key_conflict")
            return existing

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_play_turn_requests (
                    session_id,
                    owner_user_id,
                    idempotency_key,
                    request_payload_hash,
                    base_scene_version,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    idempotency_key,
                    request_payload_hash,
                    base_scene_version,
                    status,
                ),
            )
            turn_request_id = int(cursor.lastrowid)

        turn_request = self.get_turn_request(turn_request_id)
        if turn_request is None:
            raise RuntimeError("created_turn_request_not_found")
        return turn_request

    def create_session_action(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_type: str,
        idempotency_key: str,
        request_payload_hash: str,
        status: str = "pending",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        existing = self.get_session_action_by_key(
            session_id=session_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if existing["request_payload_hash"] != request_payload_hash:
                raise ValueError("idempotency_key_conflict")
            return existing

        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vn_play_session_actions (
                        session_id,
                        owner_user_id,
                        action_type,
                        idempotency_key,
                        request_payload_hash,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        owner_user_id,
                        action_type,
                        idempotency_key,
                        request_payload_hash,
                        status,
                    ),
                )
                action_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            return self._recover_session_action_insert_conflict(
                session_id=session_id,
                owner_user_id=owner_user_id,
                idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash,
                exc=exc,
            )

        session_action = self.get_session_action(action_id)
        if session_action is None:
            raise RuntimeError("created_session_action_not_found")
        return session_action

    def _recover_session_action_insert_conflict(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        idempotency_key: str,
        request_payload_hash: str,
        exc: sqlite3.IntegrityError,
    ) -> dict[str, Any]:
        existing = self.get_session_action_by_key(
            session_id=session_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is None:
            raise exc
        if existing["request_payload_hash"] != request_payload_hash:
            raise ValueError("idempotency_key_conflict") from exc
        return existing

    def get_session_action(
        self,
        action_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_session_actions
            WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
            """,
            (action_id, owner_user_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_session_action(row) if row is not None else None

    def get_session_action_by_key(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_session_actions
            WHERE session_id = ? AND owner_user_id = ? AND idempotency_key = ?
            """,
            (session_id, owner_user_id, idempotency_key),
        )
        row = cursor.fetchone()
        return _decode_session_action(row) if row is not None else None

    def latest_active_session_action(
        self,
        *,
        session_id: int,
        owner_user_id: int,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT action.*
            FROM vn_play_session_actions AS action
            JOIN vn_play_sessions AS session
              ON session.active_session_action_id = action.id
             AND session.id = action.session_id
             AND session.owner_user_id = action.owner_user_id
            WHERE session.id = ?
              AND session.owner_user_id = ?
              AND session.deleted = 0
            ORDER BY action.updated_at DESC, action.id DESC
            LIMIT 1
            """,
            (session_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_session_action(row) if row is not None else None

    def update_session_action(
        self,
        action_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        update_values = _mapped_update_values(
            fields,
            _SESSION_ACTION_UPDATE_COLUMNS,
            json_fields={"response_payload", "error"},
        )
        if not update_values:
            return self.get_session_action(action_id, owner_user_id=owner_user_id)

        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _SESSION_ACTION_UPDATE_STATEMENTS[field_name]
                conn.execute(
                    statement,
                    (value, action_id, owner_user_id, owner_user_id),
                )
        return self.get_session_action(action_id, owner_user_id=owner_user_id)

    def mark_session_action_terminal(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_id: int,
        status: str,
        error: Mapping[str, Any] | None = None,
        response_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Mark an action terminal and clear its matching active session lock atomically."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE vn_play_session_actions
                SET status = ?,
                    response_payload_json = ?,
                    error_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND session_id = ?
                  AND owner_user_id = ?
                """,
                (
                    status,
                    _json_dump(dict(response_payload)) if response_payload is not None else None,
                    _json_dump(dict(error)) if error is not None else None,
                    action_id,
                    session_id,
                    owner_user_id,
                ),
            )
            if cursor.rowcount != 1:
                return None
            conn.execute(
                """
                UPDATE vn_play_sessions
                SET active_session_action_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND owner_user_id = ?
                  AND active_session_action_id = ?
                """,
                (session_id, owner_user_id, action_id),
            )
            row = conn.execute(
                """
                SELECT *
                FROM vn_play_session_actions
                WHERE id = ? AND owner_user_id = ?
                """,
                (action_id, owner_user_id),
            ).fetchone()
        return _decode_session_action(row) if row is not None else None

    def commit_session_restore_action(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_id: int,
        event_payload: Mapping[str, Any],
        scene_state: Mapping[str, Any],
        scene_version: int,
        response_payload_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        branch_node_id: int | None = None,
        script_position: Mapping[str, Any] | None = None,
        active_generation_revisions: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist a restore event, scene state, session state, and action response."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            active_session = conn.execute(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE id = ?
                  AND owner_user_id = ?
                  AND deleted = 0
                  AND active_session_action_id = ?
                """,
                (session_id, owner_user_id, action_id),
            ).fetchone()
            if active_session is None:
                raise RuntimeError("session_action_lock_not_active")

            event_id = _insert_event(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                event_type="session_restored",
                event_payload=event_payload,
                source="runtime",
                branch_node_id=branch_node_id,
            )
            conn.execute(
                """
                INSERT INTO vn_play_scene_state (
                    session_id,
                    owner_user_id,
                    last_event_id,
                    current_background_item_id,
                    current_depth_item_id,
                    active_sprite_items_json,
                    location_key,
                    mood,
                    time_of_day,
                    weather,
                    active_branch_node_id,
                    visible_choices_json,
                    transcript_cursor,
                    scene_version,
                    warnings_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    owner_user_id = excluded.owner_user_id,
                    last_event_id = excluded.last_event_id,
                    current_background_item_id = excluded.current_background_item_id,
                    current_depth_item_id = excluded.current_depth_item_id,
                    active_sprite_items_json = excluded.active_sprite_items_json,
                    location_key = excluded.location_key,
                    mood = excluded.mood,
                    time_of_day = excluded.time_of_day,
                    weather = excluded.weather,
                    active_branch_node_id = excluded.active_branch_node_id,
                    visible_choices_json = excluded.visible_choices_json,
                    transcript_cursor = excluded.transcript_cursor,
                    scene_version = excluded.scene_version,
                    warnings_json = excluded.warnings_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    owner_user_id,
                    event_id,
                    scene_state.get("current_background_item_id"),
                    scene_state.get("current_depth_item_id"),
                    _json_dump(list(scene_state.get("active_sprite_items") or [])),
                    scene_state.get("location_key"),
                    scene_state.get("mood"),
                    scene_state.get("time_of_day"),
                    scene_state.get("weather"),
                    scene_state.get("active_branch_node_id"),
                    _json_dump(list(scene_state.get("visible_choices") or [])),
                    scene_state.get("transcript_cursor"),
                    scene_version,
                    _json_dump(list(scene_state.get("warnings") or [])),
                ),
            )
            session_cursor = conn.execute(
                """
                UPDATE vn_play_sessions
                SET scene_version = ?,
                    script_position_json = CASE
                        WHEN ? IS NULL THEN script_position_json
                        ELSE ?
                    END,
                    active_session_action_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND owner_user_id = ?
                  AND active_session_action_id = ?
                """,
                (
                    scene_version,
                    (
                        None
                        if script_position is None
                        else _json_dump(dict(script_position))
                    ),
                    (
                        None
                        if script_position is None
                        else _json_dump(dict(script_position))
                    ),
                    session_id,
                    owner_user_id,
                    action_id,
                ),
            )
            if session_cursor.rowcount != 1:
                raise RuntimeError("session_action_lock_not_active")
            if active_generation_revisions is not None:
                _apply_active_generation_revision_map(
                    conn,
                    session_id=session_id,
                    owner_user_id=owner_user_id,
                    active_generation_revisions=active_generation_revisions,
                )

            restore_event_row = conn.execute(
                "SELECT * FROM vn_play_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            session_row = conn.execute(
                "SELECT * FROM vn_play_sessions WHERE id = ? AND owner_user_id = ?",
                (session_id, owner_user_id),
            ).fetchone()
            scene_state_row = conn.execute(
                """
                SELECT *
                FROM vn_play_scene_state
                WHERE session_id = ? AND owner_user_id = ?
                """,
                (session_id, owner_user_id),
            ).fetchone()
            event_rows = conn.execute(
                """
                SELECT *
                FROM vn_play_events
                WHERE session_id = ?
                ORDER BY sequence_number ASC
                """,
                (session_id,),
            ).fetchall()
            branch_rows = conn.execute(
                """
                SELECT *
                FROM vn_play_branches
                WHERE session_id = ? AND owner_user_id = ?
                ORDER BY id ASC
                """,
                (session_id, owner_user_id),
            ).fetchall()
            if restore_event_row is None or session_row is None or scene_state_row is None:
                raise RuntimeError("session_restore_action_state_not_found")

            response_payload = dict(
                response_payload_factory(
                    {
                        "restore_event": _decode_event(restore_event_row),
                        "session": _decode_session(session_row),
                        "scene_state": _decode_scene_state(scene_state_row),
                        "events": [_decode_event(row) for row in event_rows],
                        "branches": [_decode_branch(row) for row in branch_rows],
                    }
                )
            )
            action_cursor = conn.execute(
                """
                UPDATE vn_play_session_actions
                SET status = ?,
                    response_payload_json = ?,
                    error_json = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND session_id = ?
                  AND owner_user_id = ?
                """,
                (
                    "completed",
                    _json_dump(response_payload),
                    action_id,
                    session_id,
                    owner_user_id,
                ),
            )
            if action_cursor.rowcount != 1:
                raise RuntimeError("session_action_not_found")
            return response_payload

    def commit_save_slot_create_action(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_id: int,
        slot_key: str,
        title: str,
        metadata: Mapping[str, Any],
        event_id: int | None,
        scene_version: int,
        scene_state_snapshot: Mapping[str, Any],
        response_payload_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Atomically create a checkpoint, save-slot pointer, and action response."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            active_session = conn.execute(
                """
                SELECT *
                FROM vn_play_sessions
                WHERE id = ?
                  AND owner_user_id = ?
                  AND deleted = 0
                  AND active_session_action_id = ?
                """,
                (session_id, owner_user_id, action_id),
            ).fetchone()
            if active_session is None:
                raise RuntimeError("session_action_lock_not_active")

            checkpoint_cursor = conn.execute(
                """
                INSERT INTO vn_play_checkpoints (
                    session_id,
                    owner_user_id,
                    label,
                    event_id,
                    scene_version,
                    scene_state_snapshot_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    title,
                    event_id,
                    scene_version,
                    _json_dump(dict(scene_state_snapshot)),
                ),
            )
            checkpoint_id = int(checkpoint_cursor.lastrowid)
            _insert_event(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                event_type="session_checkpoint_created",
                event_payload={
                    "checkpoint_id": checkpoint_id,
                    "label": title,
                    "scene_version": scene_version,
                },
                source="runtime",
            )
            conn.execute(
                """
                INSERT INTO vn_play_save_slots (
                    session_id,
                    owner_user_id,
                    slot_key,
                    title,
                    checkpoint_id,
                    metadata_json,
                    deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(owner_user_id, session_id, slot_key) DO UPDATE SET
                    title = excluded.title,
                    checkpoint_id = excluded.checkpoint_id,
                    metadata_json = excluded.metadata_json,
                    deleted = 0,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    owner_user_id,
                    slot_key,
                    title,
                    checkpoint_id,
                    _json_dump(dict(metadata or {})),
                ),
            )
            save_slot_row = conn.execute(
                """
                SELECT *
                FROM vn_play_save_slots
                WHERE session_id = ? AND owner_user_id = ? AND slot_key = ?
                """,
                (session_id, owner_user_id, slot_key),
            ).fetchone()
            if save_slot_row is None:
                raise RuntimeError("save_slot_not_found")
            response_payload = dict(response_payload_factory(_decode_save_slot(save_slot_row)))
            action_cursor = conn.execute(
                """
                UPDATE vn_play_session_actions
                SET status = ?,
                    response_payload_json = ?,
                    error_json = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND session_id = ?
                  AND owner_user_id = ?
                """,
                (
                    "completed",
                    _json_dump(response_payload),
                    action_id,
                    session_id,
                    owner_user_id,
                ),
            )
            if action_cursor.rowcount != 1:
                raise RuntimeError("session_action_not_found")
            session_cursor = conn.execute(
                """
                UPDATE vn_play_sessions
                SET active_session_action_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND owner_user_id = ?
                  AND active_session_action_id = ?
                """,
                (session_id, owner_user_id, action_id),
            )
            if session_cursor.rowcount != 1:
                raise RuntimeError("session_action_lock_not_active")
            return response_payload

    def get_turn_request(self, turn_request_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_play_turn_requests WHERE id = ?",
            (turn_request_id,),
        )
        row = cursor.fetchone()
        return _decode_turn_request(row) if row is not None else None

    def get_turn_request_by_key(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_turn_requests
            WHERE session_id = ? AND owner_user_id = ? AND idempotency_key = ?
            """,
            (session_id, owner_user_id, idempotency_key),
        )
        row = cursor.fetchone()
        return _decode_turn_request(row) if row is not None else None

    def latest_retryable_turn_request(
        self,
        *,
        session_id: int,
        owner_user_id: int,
    ) -> dict[str, Any] | None:
        """Return the newest input-bearing turn request only if it is retryable."""
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_turn_requests
            WHERE session_id = ?
              AND owner_user_id = ?
              AND input_event_id IS NOT NULL
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (session_id, owner_user_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        turn_request = _decode_turn_request(row)
        if turn_request["status"] not in {
            "model_failed",
            "parse_failed",
            "abandoned",
        }:
            return None
        return turn_request

    def update_turn_request(
        self,
        turn_request_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        update_values = _mapped_update_values(
            fields,
            _TURN_REQUEST_UPDATE_COLUMNS,
            json_fields={"response_payload", "error"},
        )
        if not update_values:
            return self.get_turn_request(turn_request_id)

        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _TURN_REQUEST_UPDATE_STATEMENTS[field_name]
                conn.execute(
                    statement,
                    (value, turn_request_id, owner_user_id, owner_user_id),
                )
        return self.get_turn_request(turn_request_id)

    def get_or_create_generation(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        generation_point_key: str,
        output_schema: str,
        generation_profile_key: str,
        generation_profile_snapshot_id: int,
        script_id: int | None = None,
        script_version_id: int | None = None,
        opcode_id: str | None = None,
        opcode_label: str | None = None,
        opcode_index: int | None = None,
        status: str = "not_started",
    ) -> dict[str, Any]:
        """Create or replay a session-scoped generation point."""
        self._ensure_schema_initialized()
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            raise ValueError("session_not_found")
        existing = self.get_generation_by_point(
            session_id=session_id,
            owner_user_id=owner_user_id,
            generation_point_key=generation_point_key,
        )
        generation_fields = {
            "script_id": script_id if script_id is not None else session.get("script_id"),
            "script_version_id": (
                script_version_id
                if script_version_id is not None
                else session.get("script_version_id")
            ),
            "generation_point_key": generation_point_key,
            "opcode_id": opcode_id,
            "opcode_label": opcode_label,
            "opcode_index": opcode_index,
            "output_schema": output_schema,
            "generation_profile_key": generation_profile_key,
            "generation_profile_snapshot_id": generation_profile_snapshot_id,
        }
        if existing is not None:
            for field_name, expected_value in generation_fields.items():
                if expected_value is None and field_name in {
                    "opcode_id",
                    "opcode_label",
                    "opcode_index",
                }:
                    continue
                if existing.get(field_name) != expected_value:
                    raise ValueError("generation_point_conflict")
            return existing

        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vn_play_generations (
                        session_id,
                        owner_user_id,
                        script_id,
                        script_version_id,
                        generation_point_key,
                        opcode_id,
                        opcode_label,
                        opcode_index,
                        output_schema,
                        generation_profile_key,
                        generation_profile_snapshot_id,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        owner_user_id,
                        generation_fields["script_id"],
                        generation_fields["script_version_id"],
                        generation_point_key,
                        opcode_id,
                        opcode_label,
                        opcode_index,
                        output_schema,
                        generation_profile_key,
                        generation_profile_snapshot_id,
                        status,
                    ),
                )
                generation_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            existing = self.get_generation_by_point(
                session_id=session_id,
                owner_user_id=owner_user_id,
                generation_point_key=generation_point_key,
            )
            if existing is None:
                raise exc
            for field_name, expected_value in generation_fields.items():
                if existing.get(field_name) != expected_value:
                    raise ValueError("generation_point_conflict") from exc
            return existing

        generation = self.get_generation(generation_id, owner_user_id=owner_user_id)
        if generation is None:
            raise RuntimeError("created_generation_not_found")
        return generation

    def get_generation(
        self,
        generation_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generations
            WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
            """,
            (generation_id, owner_user_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_generation(row) if row is not None else None

    def list_generations(
        self,
        session_id: int,
        *,
        owner_user_id: int,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generations
            WHERE session_id = ?
              AND owner_user_id = ?
            ORDER BY id ASC
            """,
            (session_id, owner_user_id),
        )
        return [_decode_generation(row) for row in cursor.fetchall()]

    def get_generation_by_point(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        generation_point_key: str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generations
            WHERE session_id = ?
              AND owner_user_id = ?
              AND generation_point_key = ?
            """,
            (session_id, owner_user_id, generation_point_key),
        )
        row = cursor.fetchone()
        return _decode_generation(row) if row is not None else None

    def update_generation(
        self,
        generation_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        update_values = _mapped_update_values(
            fields,
            {
                "active_revision_id": "active_revision_id",
                "latest_request_id": "latest_request_id",
                "status": "status",
            },
            json_fields=set(),
        )
        if not update_values:
            return self.get_generation(generation_id, owner_user_id=owner_user_id)
        with self.db.transaction() as conn:
            for field_name, value in update_values:
                column_name = {
                    "active_revision_id": "active_revision_id",
                    "latest_request_id": "latest_request_id",
                    "status": "status",
                }[field_name]
                conn.execute(
                    f"""
                    UPDATE vn_play_generations
                    SET {column_name} = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
                    """,  # nosec B608 - column name is from a fixed local map.
                    (value, generation_id, owner_user_id, owner_user_id),
                )
        return self.get_generation(generation_id, owner_user_id=owner_user_id)

    def create_generation_request(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        generation_id: int,
        request_kind: str,
        client_scene_version: int,
        status: str = "pending_confirmation",
        opcode_snapshot: Mapping[str, Any] | None = None,
        prompt_fingerprint: str | None = None,
        checkpoint_id_before: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        generation = self.get_generation(generation_id, owner_user_id=owner_user_id)
        if generation is None or int(generation["session_id"]) != int(session_id):
            raise ValueError("generation_not_found")
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_play_generation_requests (
                    generation_id,
                    session_id,
                    owner_user_id,
                    script_id,
                    script_version_id,
                    generation_point_key,
                    generation_profile_key,
                    generation_profile_snapshot_id,
                    request_kind,
                    status,
                    client_scene_version,
                    opcode_snapshot_json,
                    prompt_fingerprint,
                    checkpoint_id_before
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    session_id,
                    owner_user_id,
                    generation.get("script_id"),
                    generation.get("script_version_id"),
                    generation["generation_point_key"],
                    generation["generation_profile_key"],
                    generation["generation_profile_snapshot_id"],
                    request_kind,
                    status,
                    client_scene_version,
                    _json_dump(dict(opcode_snapshot or {})),
                    prompt_fingerprint,
                    checkpoint_id_before,
                ),
            )
            request_id = int(cursor.lastrowid)
            conn.execute(
                """
                UPDATE vn_play_generations
                SET latest_request_id = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                (request_id, status, generation_id, owner_user_id),
            )
        request = self.get_generation_request(request_id, owner_user_id=owner_user_id)
        if request is None:
            raise RuntimeError("created_generation_request_not_found")
        return request

    def get_generation_request(
        self,
        generation_request_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generation_requests
            WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
            """,
            (generation_request_id, owner_user_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_generation_request(row) if row is not None else None

    def update_generation_request(
        self,
        generation_request_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current = self.get_generation_request(
            generation_request_id,
            owner_user_id=owner_user_id,
        )
        if current is None:
            return None
        self._validate_generation_request_action_links(current, fields)
        update_values = _mapped_update_values(
            fields,
            _GENERATION_REQUEST_UPDATE_COLUMNS,
            json_fields={"opcode_snapshot"},
        )
        if not update_values:
            return current
        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _GENERATION_REQUEST_UPDATE_STATEMENTS[field_name]
                conn.execute(
                    statement,
                    (value, generation_request_id, owner_user_id, owner_user_id),
                )
            if "status" in fields:
                conn.execute(
                    """
                    UPDATE vn_play_generations
                    SET status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                      AND owner_user_id = ?
                      AND latest_request_id = ?
                    """,
                    (
                        fields["status"],
                        int(current["generation_id"]),
                        int(current["owner_user_id"]),
                        generation_request_id,
                    ),
                )
        return self.get_generation_request(
            generation_request_id,
            owner_user_id=owner_user_id,
        )

    def _validate_generation_request_action_links(
        self,
        request: Mapping[str, Any],
        fields: Mapping[str, Any],
    ) -> None:
        for field_name in ("create_action_id", "execute_action_id", "cancel_action_id"):
            if field_name not in fields or fields[field_name] is None:
                continue
            action = self.get_generation_action(
                int(fields[field_name]),
                owner_user_id=int(request["owner_user_id"]),
            )
            if action is None:
                raise ValueError("generation_action_not_found")
            if int(action["session_id"]) != int(request["session_id"]):
                raise ValueError("generation_action_mismatch")
            if (
                action.get("generation_id") is not None
                and int(action["generation_id"]) != int(request["generation_id"])
            ):
                raise ValueError("generation_action_mismatch")
            if (
                action.get("generation_request_id") is not None
                and int(action["generation_request_id"]) != int(request["id"])
            ):
                raise ValueError("generation_action_mismatch")

    def create_generation_action(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        action_kind: str,
        idempotency_key: str,
        request_payload_hash: str,
        generation_id: int | None = None,
        generation_request_id: int | None = None,
        generation_revision_id: int | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        existing = self.get_generation_action_by_key(
            session_id=session_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if not _generation_action_matches(
                existing,
                action_kind=action_kind,
                request_payload_hash=request_payload_hash,
                generation_id=generation_id,
                generation_request_id=generation_request_id,
                generation_revision_id=generation_revision_id,
            ):
                raise ValueError("idempotency_key_conflict")
            return existing
        if generation_id is not None:
            generation = self.get_generation(generation_id, owner_user_id=owner_user_id)
            if generation is None or int(generation["session_id"]) != int(session_id):
                raise ValueError("generation_not_found")
        if generation_request_id is not None:
            request = self.get_generation_request(
                generation_request_id,
                owner_user_id=owner_user_id,
            )
            if request is None or int(request["session_id"]) != int(session_id):
                raise ValueError("generation_request_not_found")
            if generation_id is not None and int(request["generation_id"]) != int(generation_id):
                raise ValueError("generation_request_mismatch")
        if generation_revision_id is not None:
            revision = self.get_generation_revision(
                generation_revision_id,
                owner_user_id=owner_user_id,
            )
            if revision is None or int(revision["session_id"]) != int(session_id):
                raise ValueError("generation_revision_not_found")
            if generation_id is not None and int(revision["generation_id"]) != int(generation_id):
                raise ValueError("generation_revision_mismatch")
            if (
                generation_request_id is not None
                and int(revision["generation_request_id"]) != int(generation_request_id)
            ):
                raise ValueError("generation_revision_mismatch")

        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO vn_play_generation_actions (
                        session_id,
                        owner_user_id,
                        generation_id,
                        generation_request_id,
                        generation_revision_id,
                        action_kind,
                        idempotency_key,
                        request_payload_hash,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        owner_user_id,
                        generation_id,
                        generation_request_id,
                        generation_revision_id,
                        action_kind,
                        idempotency_key,
                        request_payload_hash,
                        status,
                    ),
                )
                action_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            return self._recover_generation_action_insert_conflict(
                session_id=session_id,
                owner_user_id=owner_user_id,
                idempotency_key=idempotency_key,
                action_kind=action_kind,
                request_payload_hash=request_payload_hash,
                exc=exc,
            )

        action = self.get_generation_action(action_id, owner_user_id=owner_user_id)
        if action is None:
            raise RuntimeError("created_generation_action_not_found")
        return action

    def _recover_generation_action_insert_conflict(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        idempotency_key: str,
        action_kind: str,
        request_payload_hash: str,
        exc: sqlite3.IntegrityError,
    ) -> dict[str, Any]:
        existing = self.get_generation_action_by_key(
            session_id=session_id,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is None:
            raise exc
        if (
            existing["request_payload_hash"] != request_payload_hash
            or existing["action_kind"] != action_kind
        ):
            raise ValueError("idempotency_key_conflict") from exc
        return existing

    def get_generation_action(
        self,
        action_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generation_actions
            WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
            """,
            (action_id, owner_user_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_generation_action(row) if row is not None else None

    def get_generation_action_by_key(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generation_actions
            WHERE session_id = ?
              AND owner_user_id = ?
              AND idempotency_key = ?
            """,
            (session_id, owner_user_id, idempotency_key),
        )
        row = cursor.fetchone()
        return _decode_generation_action(row) if row is not None else None

    def update_generation_action(
        self,
        action_id: int,
        fields: Mapping[str, Any],
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current = self.get_generation_action(action_id, owner_user_id=owner_user_id)
        if current is None:
            return None
        self._validate_generation_action_relation_update(current, fields)
        update_values = _mapped_update_values(
            fields,
            _GENERATION_ACTION_UPDATE_COLUMNS,
            json_fields={"completed_action_response"},
        )
        if not update_values:
            return current
        with self.db.transaction() as conn:
            for field_name, value in update_values:
                statement = _GENERATION_ACTION_UPDATE_STATEMENTS[field_name]
                conn.execute(
                    statement,
                    (value, action_id, owner_user_id, owner_user_id),
                )
        return self.get_generation_action(action_id, owner_user_id=owner_user_id)

    def _validate_generation_action_relation_update(
        self,
        action: Mapping[str, Any],
        fields: Mapping[str, Any],
    ) -> None:
        owner_user_id = int(action["owner_user_id"])
        session_id = int(action["session_id"])
        generation_id = fields.get("generation_id", action.get("generation_id"))
        generation_request_id = fields.get(
            "generation_request_id",
            action.get("generation_request_id"),
        )
        generation_revision_id = fields.get(
            "generation_revision_id",
            action.get("generation_revision_id"),
        )
        if generation_id is not None:
            generation = self.get_generation(int(generation_id), owner_user_id=owner_user_id)
            if generation is None or int(generation["session_id"]) != session_id:
                raise ValueError("generation_not_found")
        if generation_request_id is not None:
            request = self.get_generation_request(
                int(generation_request_id),
                owner_user_id=owner_user_id,
            )
            if request is None or int(request["session_id"]) != session_id:
                raise ValueError("generation_request_not_found")
            if generation_id is not None and int(request["generation_id"]) != int(generation_id):
                raise ValueError("generation_request_mismatch")
        if generation_revision_id is not None:
            revision = self.get_generation_revision(
                int(generation_revision_id),
                owner_user_id=owner_user_id,
            )
            if revision is None or int(revision["session_id"]) != session_id:
                raise ValueError("generation_revision_not_found")
            if generation_id is not None and int(revision["generation_id"]) != int(generation_id):
                raise ValueError("generation_revision_mismatch")
            if (
                generation_request_id is not None
                and int(revision["generation_request_id"]) != int(generation_request_id)
            ):
                raise ValueError("generation_revision_mismatch")

    def create_generation_revision(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        generation_id: int,
        generation_request_id: int,
        status: str,
        output_schema: str,
        public_output: Mapping[str, Any] | None = None,
        applied_visuals: Sequence[Mapping[str, Any]] | None = None,
        rejected_visuals: Sequence[Mapping[str, Any]] | None = None,
        public_error_code: str | None = None,
        raw_output_debug: Mapping[str, Any] | None = None,
        parser_diagnostics: Mapping[str, Any] | None = None,
        moderation_diagnostics: Mapping[str, Any] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
        usage_metadata: Mapping[str, Any] | None = None,
        source: str = "model",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        generation = self.get_generation(generation_id, owner_user_id=owner_user_id)
        if generation is None or int(generation["session_id"]) != int(session_id):
            raise ValueError("generation_not_found")
        request = self.get_generation_request(
            generation_request_id,
            owner_user_id=owner_user_id,
        )
        if request is None or int(request["generation_id"]) != int(generation_id):
            raise ValueError("generation_request_not_found")
        with self.db.transaction() as conn:
            revision_cursor = conn.execute(
                """
                SELECT COALESCE(MAX(revision_number), 0) + 1 AS next_revision_number
                FROM vn_play_generation_revisions
                WHERE generation_id = ? AND owner_user_id = ?
                """,
                (generation_id, owner_user_id),
            )
            revision_number = int(revision_cursor.fetchone()["next_revision_number"])
            cursor = conn.execute(
                """
                INSERT INTO vn_play_generation_revisions (
                    generation_id,
                    generation_request_id,
                    session_id,
                    owner_user_id,
                    generation_point_key,
                    generation_profile_key,
                    generation_profile_snapshot_id,
                    revision_number,
                    status,
                    output_schema,
                    public_output_json,
                    applied_visuals_json,
                    rejected_visuals_json,
                    public_error_code,
                    raw_output_debug_json,
                    parser_diagnostics_json,
                    moderation_diagnostics_json,
                    model_metadata_json,
                    usage_metadata_json,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    generation_request_id,
                    session_id,
                    owner_user_id,
                    generation["generation_point_key"],
                    generation["generation_profile_key"],
                    generation["generation_profile_snapshot_id"],
                    revision_number,
                    status,
                    output_schema,
                    _json_dump(dict(public_output or {})),
                    _json_dump(list(applied_visuals or [])),
                    _json_dump(list(rejected_visuals or [])),
                    public_error_code,
                    (
                        None
                        if raw_output_debug is None
                        else _json_dump(dict(raw_output_debug))
                    ),
                    _json_dump(dict(parser_diagnostics or {})),
                    _json_dump(dict(moderation_diagnostics or {})),
                    _json_dump(dict(model_metadata or {})),
                    _json_dump(dict(usage_metadata or {})),
                    source,
                ),
            )
            revision_id = int(cursor.lastrowid)
        revision = self.get_generation_revision(revision_id, owner_user_id=owner_user_id)
        if revision is None:
            raise RuntimeError("created_generation_revision_not_found")
        return revision

    def get_generation_revision(
        self,
        revision_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generation_revisions
            WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
            """,
            (revision_id, owner_user_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_generation_revision(row) if row is not None else None

    def update_generation_revision_diagnostics(
        self,
        revision_id: int,
        *,
        raw_output_debug: Mapping[str, Any] | None = None,
        parser_diagnostics: Mapping[str, Any] | None = None,
        moderation_diagnostics: Mapping[str, Any] | None = None,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Update diagnostic fields for a generation revision."""
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_play_generation_revisions
                SET raw_output_debug_json = ?,
                    parser_diagnostics_json = ?,
                    moderation_diagnostics_json = ?
                WHERE id = ? AND (? IS NULL OR owner_user_id = ?)
                """,
                (
                    None if raw_output_debug is None else _json_dump(dict(raw_output_debug)),
                    _json_dump(dict(parser_diagnostics or {})),
                    _json_dump(dict(moderation_diagnostics or {})),
                    revision_id,
                    owner_user_id,
                    owner_user_id,
                ),
            )
        return self.get_generation_revision(revision_id, owner_user_id=owner_user_id)

    def list_generation_revisions(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        generation_id: int,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_generation_revisions
            WHERE session_id = ?
              AND owner_user_id = ?
              AND generation_id = ?
            ORDER BY id DESC
            LIMIT COALESCE(?, -1) OFFSET ?
            """,
            (session_id, owner_user_id, generation_id, limit, offset),
        )
        return [_decode_generation_revision(row) for row in cursor.fetchall()]

    def set_active_generation_revision(
        self,
        *,
        generation_id: int,
        owner_user_id: int,
        revision_id: int,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            generation_row = conn.execute(
                """
                SELECT *
                FROM vn_play_generations
                WHERE id = ? AND owner_user_id = ?
                """,
                (generation_id, owner_user_id),
            ).fetchone()
            if generation_row is None:
                raise ValueError("generation_not_found")
            revision_row = conn.execute(
                """
                SELECT *
                FROM vn_play_generation_revisions
                WHERE id = ? AND owner_user_id = ?
                """,
                (revision_id, owner_user_id),
            ).fetchone()
            if revision_row is None:
                raise ValueError("generation_revision_not_found")
            if int(revision_row["generation_id"]) != int(generation_id):
                raise ValueError("active_revision_generation_mismatch")
            if revision_row["status"] != "succeeded":
                raise ValueError("active_revision_not_succeeded")
            conn.execute(
                """
                UPDATE vn_play_generations
                SET active_revision_id = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                (revision_id, "completed", generation_id, owner_user_id),
            )
            updated_row = conn.execute(
                """
                SELECT *
                FROM vn_play_generations
                WHERE id = ? AND owner_user_id = ?
                """,
                (generation_id, owner_user_id),
            ).fetchone()
        if updated_row is None:
            raise RuntimeError("generation_not_found")
        return _decode_generation(updated_row)

    def record_story_choice_selection(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        turn_request_id: int,
        client_scene_version: int,
        selected_choice: Mapping[str, Any],
        parent_event_id: int | None,
        branch_label: str | None,
        branch_path: Sequence[Any] | None,
        expected_scene_last_event_id: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        self._ensure_schema_initialized()
        choice = dict(selected_choice)
        with self.db.transaction() as conn:
            turn_row = conn.execute(
                """
                SELECT id
                FROM vn_play_turn_requests
                WHERE id = ?
                  AND session_id = ?
                  AND owner_user_id = ?
                  AND status = ?
                  AND turn_started_event_id IS NULL
                  AND input_event_id IS NULL
                  AND base_scene_version = ?
                """,
                (
                    turn_request_id,
                    session_id,
                    owner_user_id,
                    "pending",
                    client_scene_version,
                ),
            ).fetchone()
            if turn_row is None:
                raise RuntimeError("turn_request_not_pending")

            current_scene_state = conn.execute(
                """
                SELECT last_event_id, visible_choices_json
                FROM vn_play_scene_state
                WHERE session_id = ? AND owner_user_id = ?
                """,
                (session_id, owner_user_id),
            ).fetchone()
            if current_scene_state is None:
                raise RuntimeError("choice_not_visible")
            if expected_scene_last_event_id is not None:
                current_last_event_id = current_scene_state["last_event_id"]
                if (
                    current_last_event_id is None
                    or int(current_last_event_id) != expected_scene_last_event_id
                ):
                    raise RuntimeError("scene_state_moved")
            visible_choices = _json_loads(
                current_scene_state["visible_choices_json"],
                [],
            )
            if not _choice_id_is_visible(visible_choices, choice.get("id")):
                raise RuntimeError("choice_not_visible")
            bounded_branch_label = _bounded_branch_label(branch_label)
            bounded_branch_path = _bounded_branch_path(branch_path)

            branch_cursor = conn.execute(
                """
                INSERT INTO vn_play_branches (
                    session_id,
                    owner_user_id,
                    parent_event_id,
                    branch_label,
                    branch_path_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    parent_event_id,
                    bounded_branch_label,
                    _json_dump(bounded_branch_path),
                ),
            )
            branch_id = int(branch_cursor.lastrowid)
            turn_started_event_id = _insert_event(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                event_type="turn_started",
                event_payload={
                    "turn_request_id": turn_request_id,
                    "scene_version": client_scene_version,
                },
                source="runtime",
            )
            choice_selected_event_id = _insert_event(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                event_type="choice_selected",
                event_payload={
                    "schema_version": 1,
                    "turn_request_id": turn_request_id,
                    "choice_id": choice.get("id"),
                    "choice": choice,
                    "branch_node_id": branch_id,
                    "scene_version": client_scene_version,
                },
                source="user",
                branch_node_id=branch_id,
            )
            turn_cursor = conn.execute(
                """
                UPDATE vn_play_turn_requests
                SET status = ?,
                    turn_started_event_id = ?,
                    input_event_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND session_id = ?
                  AND owner_user_id = ?
                  AND status = ?
                  AND turn_started_event_id IS NULL
                  AND input_event_id IS NULL
                  AND base_scene_version = ?
                """,
                (
                    "model_calling",
                    turn_started_event_id,
                    choice_selected_event_id,
                    turn_request_id,
                    session_id,
                    owner_user_id,
                    "pending",
                    client_scene_version,
                ),
            )
            if turn_cursor.rowcount != 1:
                raise RuntimeError("turn_request_not_pending")
            conn.execute(
                """
                INSERT INTO vn_play_scene_state (
                    session_id,
                    owner_user_id,
                    last_event_id,
                    active_branch_node_id,
                    visible_choices_json,
                    scene_version
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    owner_user_id = excluded.owner_user_id,
                    last_event_id = excluded.last_event_id,
                    active_branch_node_id = excluded.active_branch_node_id,
                    visible_choices_json = excluded.visible_choices_json,
                    scene_version = excluded.scene_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    owner_user_id,
                    choice_selected_event_id,
                    branch_id,
                    _json_dump([]),
                    client_scene_version,
                ),
            )
            branch_row = conn.execute(
                "SELECT * FROM vn_play_branches WHERE id = ?",
                (branch_id,),
            ).fetchone()
            turn_started_row = conn.execute(
                "SELECT * FROM vn_play_events WHERE id = ?",
                (turn_started_event_id,),
            ).fetchone()
            choice_selected_row = conn.execute(
                "SELECT * FROM vn_play_events WHERE id = ?",
                (choice_selected_event_id,),
            ).fetchone()
            scene_state_row = conn.execute(
                """
                SELECT *
                FROM vn_play_scene_state
                WHERE session_id = ? AND owner_user_id = ?
                """,
                (session_id, owner_user_id),
            ).fetchone()

        if (
            branch_row is None
            or turn_started_row is None
            or choice_selected_row is None
            or scene_state_row is None
        ):
            raise RuntimeError("story_choice_selection_not_found")
        return {
            "branch": _decode_branch(branch_row),
            "turn_started": _decode_event(turn_started_row),
            "choice_selected": _decode_event(choice_selected_row),
            "scene_state": _decode_scene_state(scene_state_row),
        }

    def set_scene_state(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        last_event_id: int | None = None,
        current_background_item_id: int | None = None,
        current_depth_item_id: int | None = None,
        active_sprite_items: Sequence[Mapping[str, Any]] | None = None,
        location_key: str | None = None,
        mood: str | None = None,
        time_of_day: str | None = None,
        weather: str | None = None,
        active_branch_node_id: int | None = None,
        visible_choices: Sequence[Mapping[str, Any]] | None = None,
        transcript_cursor: int | None = None,
        scene_version: int = 0,
        warnings: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO vn_play_scene_state (
                    session_id,
                    owner_user_id,
                    last_event_id,
                    current_background_item_id,
                    current_depth_item_id,
                    active_sprite_items_json,
                    location_key,
                    mood,
                    time_of_day,
                    weather,
                    active_branch_node_id,
                    visible_choices_json,
                    transcript_cursor,
                    scene_version,
                    warnings_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    owner_user_id = excluded.owner_user_id,
                    last_event_id = excluded.last_event_id,
                    current_background_item_id = excluded.current_background_item_id,
                    current_depth_item_id = excluded.current_depth_item_id,
                    active_sprite_items_json = excluded.active_sprite_items_json,
                    location_key = excluded.location_key,
                    mood = excluded.mood,
                    time_of_day = excluded.time_of_day,
                    weather = excluded.weather,
                    active_branch_node_id = excluded.active_branch_node_id,
                    visible_choices_json = excluded.visible_choices_json,
                    transcript_cursor = excluded.transcript_cursor,
                    scene_version = excluded.scene_version,
                    warnings_json = excluded.warnings_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    owner_user_id,
                    last_event_id,
                    current_background_item_id,
                    current_depth_item_id,
                    _json_dump(list(active_sprite_items or [])),
                    location_key,
                    mood,
                    time_of_day,
                    weather,
                    active_branch_node_id,
                    _json_dump(list(visible_choices or [])),
                    transcript_cursor,
                    scene_version,
                    _json_dump(list(warnings or [])),
                ),
            )
            conn.execute(
                """
                UPDATE vn_play_sessions
                SET scene_version = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                (scene_version, session_id, owner_user_id),
            )

        state = self.get_scene_state(session_id)
        if state is None:
            raise RuntimeError("scene_state_not_found")
        return state

    def get_scene_state(
        self,
        session_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_scene_state
            WHERE session_id = ?
              AND (? IS NULL OR owner_user_id = ?)
            """,
            (session_id, owner_user_id, owner_user_id),
        )
        row = cursor.fetchone()
        return _decode_scene_state(row) if row is not None else None

    def create_branch(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        parent_event_id: int | None = None,
        branch_label: str | None = None,
        branch_path: Sequence[Any] | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_play_branches (
                    session_id,
                    owner_user_id,
                    parent_event_id,
                    branch_label,
                    branch_path_json,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    parent_event_id,
                    branch_label,
                    _json_dump(list(branch_path or [])),
                    status,
                ),
            )
            branch_id = int(cursor.lastrowid)

        branch = self.get_branch(branch_id)
        if branch is None:
            raise RuntimeError("created_branch_not_found")
        return branch

    def get_branch(self, branch_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_play_branches WHERE id = ?",
            (branch_id,),
        )
        row = cursor.fetchone()
        return _decode_branch(row) if row is not None else None

    def list_branches(
        self,
        session_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_branches
            WHERE session_id = ?
              AND (? IS NULL OR owner_user_id = ?)
            ORDER BY id ASC
            """,
            (session_id, owner_user_id, owner_user_id),
        )
        return [_decode_branch(row) for row in cursor.fetchall()]

    def create_checkpoint(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        label: str,
        event_id: int | None,
        scene_version: int,
        scene_state_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO vn_play_checkpoints (
                    session_id,
                    owner_user_id,
                    label,
                    event_id,
                    scene_version,
                    scene_state_snapshot_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_user_id,
                    label,
                    event_id,
                    scene_version,
                    _json_dump(dict(scene_state_snapshot)),
                ),
            )
            checkpoint_id = int(cursor.lastrowid)

        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise RuntimeError("created_checkpoint_not_found")
        return checkpoint

    def get_checkpoint(self, checkpoint_id: int) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            "SELECT * FROM vn_play_checkpoints WHERE id = ?",
            (checkpoint_id,),
        )
        row = cursor.fetchone()
        return _decode_checkpoint(row) if row is not None else None

    def find_asset_cleanup_blockers(
        self,
        *,
        owner_user_id: int,
        asset_pack_id: int,
        generated_file_ids: set[int],
        item_ids_by_file_id: Mapping[int, int],
    ) -> dict[int, list[dict[str, str]]]:
        """Find generated files referenced by VN play sessions and checkpoints."""
        self._ensure_schema_initialized()
        if not generated_file_ids:
            return {}
        item_to_file_id = {
            int(item_id): int(file_id)
            for file_id, item_id in item_ids_by_file_id.items()
        }
        session_rows = self.db.execute_query(
            """
            SELECT id
            FROM vn_play_sessions
            WHERE owner_user_id = ?
              AND vn_asset_pack_id = ?
              AND deleted = 0
            """,
            (owner_user_id, asset_pack_id),
        ).fetchall()
        blockers: dict[int, list[dict[str, str]]] = {}
        for session_row in session_rows:
            session_id = int(session_row["id"])
            event_rows = self.db.execute_query(
                """
                SELECT id, event_payload_json
                FROM vn_play_events
                WHERE owner_user_id = ?
                  AND session_id = ?
                """,
                (owner_user_id, session_id),
            ).fetchall()
            for row in event_rows:
                _add_cleanup_blockers_from_payload(
                    blockers,
                    payload=_json_loads(row["event_payload_json"], {}),
                    generated_file_ids=generated_file_ids,
                    item_to_file_id=item_to_file_id,
                    source_type="event",
                    source_id=int(row["id"]),
                )

            checkpoint_rows = self.db.execute_query(
                """
                SELECT id, scene_state_snapshot_json
                FROM vn_play_checkpoints
                WHERE owner_user_id = ?
                  AND session_id = ?
                """,
                (owner_user_id, session_id),
            ).fetchall()
            for row in checkpoint_rows:
                _add_cleanup_blockers_from_payload(
                    blockers,
                    payload=_json_loads(row["scene_state_snapshot_json"], {}),
                    generated_file_ids=generated_file_ids,
                    item_to_file_id=item_to_file_id,
                    source_type="checkpoint",
                    source_id=int(row["id"]),
                )

            scene_row = self.db.execute_query(
                """
                SELECT id,
                       current_background_item_id,
                       current_depth_item_id,
                       active_sprite_items_json
                FROM vn_play_scene_state
                WHERE owner_user_id = ?
                  AND session_id = ?
                """,
                (owner_user_id, session_id),
            ).fetchone()
            if scene_row is not None:
                _add_cleanup_blockers_from_payload(
                    blockers,
                    payload={
                        "current_background_item_id": scene_row["current_background_item_id"],
                        "current_depth_item_id": scene_row["current_depth_item_id"],
                        "active_sprite_items": _json_loads(
                            scene_row["active_sprite_items_json"],
                            [],
                        ),
                    },
                    generated_file_ids=generated_file_ids,
                    item_to_file_id=item_to_file_id,
                    source_type="scene_state",
                    source_id=int(scene_row["id"]),
                )
        return blockers

    def list_checkpoints(
        self,
        session_id: int,
        *,
        owner_user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_checkpoints
            WHERE session_id = ?
              AND (? IS NULL OR owner_user_id = ?)
            ORDER BY id DESC
            """,
            (session_id, owner_user_id, owner_user_id),
        )
        return [_decode_checkpoint(row) for row in cursor.fetchall()]

    def upsert_save_slot(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        slot_key: str,
        title: str,
        checkpoint_id: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_schema_initialized()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO vn_play_save_slots (
                    session_id,
                    owner_user_id,
                    slot_key,
                    title,
                    checkpoint_id,
                    metadata_json,
                    deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(owner_user_id, session_id, slot_key) DO UPDATE SET
                    title = excluded.title,
                    checkpoint_id = excluded.checkpoint_id,
                    metadata_json = excluded.metadata_json,
                    deleted = 0,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    owner_user_id,
                    slot_key,
                    title,
                    checkpoint_id,
                    _json_dump(dict(metadata or {})),
                ),
            )
            row = conn.execute(
                """
                SELECT *
                FROM vn_play_save_slots
                WHERE session_id = ? AND owner_user_id = ? AND slot_key = ?
                """,
                (session_id, owner_user_id, slot_key),
            ).fetchone()
        if row is None:
            raise RuntimeError("save_slot_not_found")
        return _decode_save_slot(row)

    def get_save_slot(
        self,
        save_slot_id: int,
        *,
        session_id: int | None = None,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_save_slots
            WHERE id = ?
              AND (? IS NULL OR session_id = ?)
              AND (? IS NULL OR owner_user_id = ?)
              AND (? OR deleted = 0)
            """,
            (
                save_slot_id,
                session_id,
                session_id,
                owner_user_id,
                owner_user_id,
                include_deleted,
            ),
        )
        row = cursor.fetchone()
        return _decode_save_slot(row) if row is not None else None

    def list_save_slots(
        self,
        session_id: int,
        *,
        owner_user_id: int | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        self._ensure_schema_initialized()
        cursor = self.db.execute_query(
            """
            SELECT *
            FROM vn_play_save_slots
            WHERE session_id = ?
              AND (? IS NULL OR owner_user_id = ?)
              AND (? OR deleted = 0)
            ORDER BY updated_at DESC, id DESC
            """,
            (session_id, owner_user_id, owner_user_id, include_deleted),
        )
        return [_decode_save_slot(row) for row in cursor.fetchall()]

    def update_save_slot(
        self,
        save_slot_id: int,
        *,
        session_id: int,
        owner_user_id: int,
        title: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        deleted: bool | None = None,
    ) -> dict[str, Any] | None:
        self._ensure_schema_initialized()
        current = self.get_save_slot(
            save_slot_id,
            session_id=session_id,
            owner_user_id=owner_user_id,
            include_deleted=True,
        )
        if current is None:
            return None
        next_title = str(title) if title is not None else str(current["title"])
        next_metadata = dict(metadata) if metadata is not None else dict(current["metadata"])
        next_deleted = bool(deleted) if deleted is not None else bool(current["deleted"])
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE vn_play_save_slots
                SET title = ?,
                    metadata_json = ?,
                    deleted = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND session_id = ?
                  AND owner_user_id = ?
                """,
                (
                    next_title,
                    _json_dump(next_metadata),
                    int(next_deleted),
                    save_slot_id,
                    session_id,
                    owner_user_id,
                ),
            )
        return self.get_save_slot(
            save_slot_id,
            session_id=session_id,
            owner_user_id=owner_user_id,
            include_deleted=True,
        )

    def _ensure_schema_initialized(self) -> None:
        if self._schema_initialized:
            return
        self.initialize_schema()


_SESSION_UPDATE_COLUMNS = {
    "mode": "mode",
    "title": "title",
    "status": "status",
    "primary_character_id": "primary_character_id",
    "additional_character_ids": "additional_character_ids_json",
    "linked_chat_id": "linked_chat_id",
    "vn_asset_pack_id": "vn_asset_pack_id",
    "asset_manifest_version": "asset_manifest_version",
    "source_world_book_ids": "source_world_book_ids_json",
    "content_rating": "content_rating",
    "trust_level": "trust_level",
    "linked_chat_mode": "linked_chat_mode",
    "seed": "seed",
    "settings": "settings_json",
    "script_id": "script_id",
    "script_version_id": "script_version_id",
    "script_manifest_snapshot_id": "script_manifest_snapshot_id",
    "script_policy_snapshot_id": "script_policy_snapshot_id",
    "script_generation_profile_snapshot_id": "script_generation_profile_snapshot_id",
    "script_position": "script_position_json",
    "scene_version": "scene_version",
    "active_turn_request_id": "active_turn_request_id",
    "active_session_action_id": "active_session_action_id",
    "deleted": "deleted",
}

_TURN_REQUEST_UPDATE_COLUMNS = {
    "request_payload_hash": "request_payload_hash",
    "base_scene_version": "base_scene_version",
    "status": "status",
    "input_event_id": "input_event_id",
    "turn_started_event_id": "turn_started_event_id",
    "turn_completed_event_id": "turn_completed_event_id",
    "response_payload": "response_payload_json",
    "error": "error_json",
    "lease_owner": "lease_owner",
    "locked_until": "locked_until",
}

_SESSION_ACTION_UPDATE_COLUMNS = {
    "status": "status",
    "response_payload": "response_payload_json",
    "error": "error_json",
    "lease_owner": "lease_owner",
    "locked_until": "locked_until",
}

_GENERATION_REQUEST_UPDATE_COLUMNS = {
    "status": "status",
    "create_action_id": "create_action_id",
    "execute_action_id": "execute_action_id",
    "cancel_action_id": "cancel_action_id",
    "client_scene_version": "client_scene_version",
    "opcode_snapshot": "opcode_snapshot_json",
    "prompt_fingerprint": "prompt_fingerprint",
    "checkpoint_id_before": "checkpoint_id_before",
    "provider_call_started_at": "provider_call_started_at",
    "provider_call_completed_at": "provider_call_completed_at",
    "lease_expires_at": "lease_expires_at",
    "public_error_code": "public_error_code",
}

_GENERATION_ACTION_UPDATE_COLUMNS = {
    "generation_request_id": "generation_request_id",
    "generation_revision_id": "generation_revision_id",
    "status": "status",
    "completed_action_response": "completed_action_response_json",
    "public_error_code": "public_error_code",
}

_SESSION_UPDATE_STATEMENTS = {
    "mode": (
        "UPDATE vn_play_sessions SET mode = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "title": (
        "UPDATE vn_play_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "status": (
        "UPDATE vn_play_sessions SET status = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "primary_character_id": (
        "UPDATE vn_play_sessions SET primary_character_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "additional_character_ids": (
        "UPDATE vn_play_sessions SET additional_character_ids_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "linked_chat_id": (
        "UPDATE vn_play_sessions SET linked_chat_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "vn_asset_pack_id": (
        "UPDATE vn_play_sessions SET vn_asset_pack_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "asset_manifest_version": (
        "UPDATE vn_play_sessions SET asset_manifest_version = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "source_world_book_ids": (
        "UPDATE vn_play_sessions SET source_world_book_ids_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "content_rating": (
        "UPDATE vn_play_sessions SET content_rating = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "trust_level": (
        "UPDATE vn_play_sessions SET trust_level = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "linked_chat_mode": (
        "UPDATE vn_play_sessions SET linked_chat_mode = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "seed": (
        "UPDATE vn_play_sessions SET seed = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "settings": (
        "UPDATE vn_play_sessions SET settings_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "script_id": (
        "UPDATE vn_play_sessions SET script_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "script_version_id": (
        "UPDATE vn_play_sessions SET script_version_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "script_manifest_snapshot_id": (
        "UPDATE vn_play_sessions SET script_manifest_snapshot_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "script_policy_snapshot_id": (
        "UPDATE vn_play_sessions SET script_policy_snapshot_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "script_generation_profile_snapshot_id": (
        "UPDATE vn_play_sessions SET script_generation_profile_snapshot_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "script_position": (
        "UPDATE vn_play_sessions SET script_position_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "scene_version": (
        "UPDATE vn_play_sessions SET scene_version = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "active_turn_request_id": (
        "UPDATE vn_play_sessions SET active_turn_request_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "active_session_action_id": (
        "UPDATE vn_play_sessions SET active_session_action_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "deleted": (
        "UPDATE vn_play_sessions SET deleted = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
}

_TURN_REQUEST_UPDATE_STATEMENTS = {
    "request_payload_hash": (
        "UPDATE vn_play_turn_requests SET request_payload_hash = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "base_scene_version": (
        "UPDATE vn_play_turn_requests SET base_scene_version = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "status": (
        "UPDATE vn_play_turn_requests SET status = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "input_event_id": (
        "UPDATE vn_play_turn_requests SET input_event_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "turn_started_event_id": (
        "UPDATE vn_play_turn_requests SET turn_started_event_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "turn_completed_event_id": (
        "UPDATE vn_play_turn_requests SET turn_completed_event_id = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "response_payload": (
        "UPDATE vn_play_turn_requests SET response_payload_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "error": (
        "UPDATE vn_play_turn_requests SET error_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "lease_owner": (
        "UPDATE vn_play_turn_requests SET lease_owner = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "locked_until": (
        "UPDATE vn_play_turn_requests SET locked_until = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
}

_SESSION_ACTION_UPDATE_STATEMENTS = {
    "status": (
        "UPDATE vn_play_session_actions SET status = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "response_payload": (
        "UPDATE vn_play_session_actions SET response_payload_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "error": (
        "UPDATE vn_play_session_actions SET error_json = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "lease_owner": (
        "UPDATE vn_play_session_actions SET lease_owner = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "locked_until": (
        "UPDATE vn_play_session_actions SET locked_until = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
}

_GENERATION_REQUEST_UPDATE_STATEMENTS = {
    field_name: (
        f"UPDATE vn_play_generation_requests SET {column_name} = ?, "  # nosec B608
        "updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    )
    for field_name, column_name in _GENERATION_REQUEST_UPDATE_COLUMNS.items()
}

_GENERATION_ACTION_UPDATE_STATEMENTS = {
    field_name: (
        f"UPDATE vn_play_generation_actions SET {column_name} = ?, "  # nosec B608
        "updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    )
    for field_name, column_name in _GENERATION_ACTION_UPDATE_COLUMNS.items()
}


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    if getattr(db, "backend_type", None) != BackendType.SQLITE:
        raise NotImplementedError(
            "VN Play metadata currently supports SQLite ChaChaNotes databases only."
        )


def _is_index_statement(statement: str) -> bool:
    normalized = statement.upper()
    return normalized.startswith("CREATE INDEX") or normalized.startswith(
        "CREATE UNIQUE INDEX"
    )


def _ensure_vn_play_session_columns(conn: Any) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(vn_play_sessions)").fetchall()
    }
    if "active_session_action_id" not in existing_columns:
        conn.execute("ALTER TABLE vn_play_sessions ADD COLUMN active_session_action_id INTEGER")
    column_defaults = {
        "script_id": "INTEGER",
        "script_version_id": "INTEGER",
        "script_manifest_snapshot_id": "INTEGER",
        "script_policy_snapshot_id": "INTEGER",
        "script_generation_profile_snapshot_id": "INTEGER",
        "script_position_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for column_name, column_definition in column_defaults.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE vn_play_sessions ADD COLUMN {column_name} {column_definition}"  # nosec B608
            )


def _ensure_vn_play_generation_columns(conn: Any) -> None:
    generation_columns = _table_column_names(conn, "vn_play_generations")
    generation_defaults = {
        "script_id": "INTEGER",
        "script_version_id": "INTEGER",
        "generation_point_key": "TEXT NOT NULL DEFAULT ''",
        "opcode_id": "TEXT",
        "opcode_label": "TEXT",
        "opcode_index": "INTEGER",
        "output_schema": "TEXT NOT NULL DEFAULT 'narrative_dialogue'",
        "generation_profile_key": "TEXT NOT NULL DEFAULT 'default'",
        "generation_profile_snapshot_id": "INTEGER NOT NULL DEFAULT 0",
        "active_revision_id": "INTEGER",
        "latest_request_id": "INTEGER",
        "status": "TEXT NOT NULL DEFAULT 'not_started'",
        "created_at": "DATETIME",
        "updated_at": "DATETIME",
    }
    for column_name, column_definition in generation_defaults.items():
        if column_name not in generation_columns:
            conn.execute(
                f"ALTER TABLE vn_play_generations ADD COLUMN {column_name} {column_definition}"  # nosec B608
            )

    request_columns = _table_column_names(conn, "vn_play_generation_requests")
    request_defaults = {
        "script_id": "INTEGER",
        "script_version_id": "INTEGER",
        "generation_point_key": "TEXT NOT NULL DEFAULT ''",
        "generation_profile_key": "TEXT NOT NULL DEFAULT 'default'",
        "generation_profile_snapshot_id": "INTEGER NOT NULL DEFAULT 0",
        "request_kind": "TEXT NOT NULL DEFAULT 'automatic'",
        "status": "TEXT NOT NULL DEFAULT 'pending_confirmation'",
        "create_action_id": "INTEGER",
        "execute_action_id": "INTEGER",
        "cancel_action_id": "INTEGER",
        "client_scene_version": "INTEGER NOT NULL DEFAULT 0",
        "opcode_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
        "prompt_fingerprint": "TEXT",
        "checkpoint_id_before": "INTEGER",
        "provider_call_started_at": "DATETIME",
        "provider_call_completed_at": "DATETIME",
        "lease_expires_at": "DATETIME",
        "public_error_code": "TEXT",
        "created_at": "DATETIME",
        "updated_at": "DATETIME",
    }
    for column_name, column_definition in request_defaults.items():
        if column_name not in request_columns:
            conn.execute(
                f"ALTER TABLE vn_play_generation_requests ADD COLUMN {column_name} {column_definition}"  # nosec B608
            )

    action_columns = _table_column_names(conn, "vn_play_generation_actions")
    action_defaults = {
        "generation_id": "INTEGER",
        "generation_request_id": "INTEGER",
        "generation_revision_id": "INTEGER",
        "action_kind": "TEXT NOT NULL DEFAULT 'execute'",
        "idempotency_key": "TEXT NOT NULL DEFAULT ''",
        "request_payload_hash": "TEXT NOT NULL DEFAULT ''",
        "status": "TEXT NOT NULL DEFAULT 'pending'",
        "completed_action_response_json": "TEXT",
        "public_error_code": "TEXT",
        "created_at": "DATETIME",
        "updated_at": "DATETIME",
    }
    for column_name, column_definition in action_defaults.items():
        if column_name not in action_columns:
            conn.execute(
                f"ALTER TABLE vn_play_generation_actions ADD COLUMN {column_name} {column_definition}"  # nosec B608
            )

    revision_columns = _table_column_names(conn, "vn_play_generation_revisions")
    revision_defaults = {
        "generation_point_key": "TEXT NOT NULL DEFAULT ''",
        "generation_profile_key": "TEXT NOT NULL DEFAULT 'default'",
        "generation_profile_snapshot_id": "INTEGER NOT NULL DEFAULT 0",
        "revision_number": "INTEGER NOT NULL DEFAULT 1",
        "status": "TEXT NOT NULL DEFAULT 'succeeded'",
        "output_schema": "TEXT NOT NULL DEFAULT 'narrative_dialogue'",
        "public_output_json": "TEXT NOT NULL DEFAULT '{}'",
        "applied_visuals_json": "TEXT NOT NULL DEFAULT '[]'",
        "rejected_visuals_json": "TEXT NOT NULL DEFAULT '[]'",
        "public_error_code": "TEXT",
        "raw_output_debug_json": "TEXT",
        "parser_diagnostics_json": "TEXT NOT NULL DEFAULT '{}'",
        "moderation_diagnostics_json": "TEXT NOT NULL DEFAULT '{}'",
        "model_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "usage_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "source": "TEXT NOT NULL DEFAULT 'model'",
        "created_at": "DATETIME",
    }
    for column_name, column_definition in revision_defaults.items():
        if column_name not in revision_columns:
            conn.execute(
                f"ALTER TABLE vn_play_generation_revisions ADD COLUMN {column_name} {column_definition}"  # nosec B608
            )


def _table_column_names(conn: Any, table_name: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}  # nosec B608


def _insert_event(
    conn: Any,
    *,
    session_id: int,
    owner_user_id: int,
    event_type: str,
    event_payload: Mapping[str, Any] | None = None,
    source: str = "runtime",
    model_provider: str | None = None,
    model_name: str | None = None,
    branch_node_id: int | None = None,
) -> int:
    sequence_cursor = conn.execute(
        """
        SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next_sequence
        FROM vn_play_events
        WHERE session_id = ?
        """,
        (session_id,),
    )
    sequence_number = int(sequence_cursor.fetchone()["next_sequence"])
    cursor = conn.execute(
        """
        INSERT INTO vn_play_events (
            session_id,
            owner_user_id,
            sequence_number,
            event_type,
            event_payload_json,
            source,
            model_provider,
            model_name,
            branch_node_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            owner_user_id,
            sequence_number,
            event_type,
            _json_dump(dict(event_payload or {})),
            source,
            model_provider,
            model_name,
            branch_node_id,
        ),
    )
    return int(cursor.lastrowid)


def _apply_active_generation_revision_map(
    conn: Any,
    *,
    session_id: int,
    owner_user_id: int,
    active_generation_revisions: Mapping[str, Any],
) -> None:
    normalized_map = {
        str(point_key): (None if revision_id is None else int(revision_id))
        for point_key, revision_id in active_generation_revisions.items()
        if str(point_key)
    }
    generation_rows = conn.execute(
        """
        SELECT *
        FROM vn_play_generations
        WHERE session_id = ? AND owner_user_id = ?
        """,
        (session_id, owner_user_id),
    ).fetchall()
    existing_point_keys = {
        str(generation_row["generation_point_key"])
        for generation_row in generation_rows
    }
    unknown_point_keys = set(normalized_map) - existing_point_keys
    if unknown_point_keys:
        raise ValueError("generation_point_not_found")
    for generation_row in generation_rows:
        generation_id = int(generation_row["id"])
        point_key = str(generation_row["generation_point_key"])
        revision_id = normalized_map.get(point_key)
        if revision_id is None:
            conn.execute(
                """
                UPDATE vn_play_generations
                SET active_revision_id = NULL,
                    latest_request_id = NULL,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND owner_user_id = ?
                """,
                ("not_started", generation_id, owner_user_id),
            )
            continue
        revision_row = conn.execute(
            """
            SELECT id, generation_id, generation_request_id, session_id, owner_user_id, status
            FROM vn_play_generation_revisions
            WHERE id = ?
              AND generation_id = ?
              AND session_id = ?
              AND owner_user_id = ?
            """,
            (revision_id, generation_id, session_id, owner_user_id),
        ).fetchone()
        if revision_row is None:
            raise ValueError("generation_revision_not_found")
        if revision_row["status"] != "succeeded":
            raise ValueError("active_revision_not_succeeded")
        conn.execute(
            """
            UPDATE vn_play_generations
            SET active_revision_id = ?,
                latest_request_id = ?,
                status = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND owner_user_id = ?
            """,
            (
                revision_id,
                int(revision_row["generation_request_id"]),
                "completed",
                generation_id,
                owner_user_id,
            ),
        )


def _mapped_update_values(
    fields: Mapping[str, Any],
    column_map: Mapping[str, str],
    *,
    json_fields: set[str],
) -> list[tuple[str, Any]]:
    update_values: list[tuple[str, Any]] = []
    for field_name, raw_value in fields.items():
        column_name = column_map.get(field_name)
        if column_name is None:
            continue
        value = _json_dump(raw_value) if field_name in json_fields else raw_value
        update_values.append((field_name, value))
    return update_values


def _decode_session(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["additional_character_ids"] = _json_loads(data.pop("additional_character_ids_json"), [])
    data["source_world_book_ids"] = _json_loads(data.pop("source_world_book_ids_json"), [])
    data["settings"] = _json_loads(data.pop("settings_json"), {})
    data["script_position"] = _json_loads(data.pop("script_position_json", None), {})
    return data


def _decode_event(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["event_payload"] = _json_loads(data.pop("event_payload_json"), {})
    return data


def _decode_turn_request(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["response_payload"] = _json_loads(data.pop("response_payload_json"), None)
    data["error"] = _json_loads(data.pop("error_json"), None)
    return data


def _decode_session_action(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["response_payload"] = _json_loads(data.pop("response_payload_json"), None)
    data["error"] = _json_loads(data.pop("error_json"), None)
    return data


def _generation_action_matches(
    action: Mapping[str, Any],
    *,
    action_kind: str,
    request_payload_hash: str,
    generation_id: int | None,
    generation_request_id: int | None,
    generation_revision_id: int | None,
) -> bool:
    if action["request_payload_hash"] != request_payload_hash:
        return False
    if action["action_kind"] != action_kind:
        return False
    expected_links = {
        "generation_id": generation_id,
        "generation_request_id": generation_request_id,
        "generation_revision_id": generation_revision_id,
    }
    for field_name, expected_value in expected_links.items():
        if expected_value is not None and action.get(field_name) != expected_value:
            return False
    return True


def _decode_generation(row: Any) -> dict[str, Any]:
    return dict(row)


def _decode_generation_request(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["opcode_snapshot"] = _json_loads(data.pop("opcode_snapshot_json"), {})
    return data


def _decode_generation_action(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["completed_action_response"] = _json_loads(
        data.pop("completed_action_response_json"),
        None,
    )
    return data


def _decode_generation_revision(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["public_output"] = _json_loads(data.pop("public_output_json"), {})
    data["applied_visuals"] = _json_loads(data.pop("applied_visuals_json"), [])
    data["rejected_visuals"] = _json_loads(data.pop("rejected_visuals_json"), [])
    data["raw_output_debug"] = _json_loads(data.pop("raw_output_debug_json"), None)
    data["parser_diagnostics"] = _json_loads(data.pop("parser_diagnostics_json"), {})
    data["moderation_diagnostics"] = _json_loads(
        data.pop("moderation_diagnostics_json"),
        {},
    )
    data["model_metadata"] = _json_loads(data.pop("model_metadata_json"), {})
    data["usage_metadata"] = _json_loads(data.pop("usage_metadata_json"), {})
    return data


def _decode_scene_state(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["active_sprite_items"] = _json_loads(data.pop("active_sprite_items_json"), [])
    data["visible_choices"] = _json_loads(data.pop("visible_choices_json"), [])
    data["warnings"] = _json_loads(data.pop("warnings_json"), [])
    return data


def _decode_branch(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["branch_path"] = _json_loads(data.pop("branch_path_json"), [])
    return data


def _bounded_branch_label(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)[:STORY_BRANCH_LABEL_MAX_LENGTH]


def _bounded_branch_path(value: Sequence[Any] | None) -> list[Any]:
    bounded_path: list[Any] = []
    for item in list(value or []):
        if not isinstance(item, Mapping):
            bounded_path.append(item)
            continue
        next_item = dict(item)
        choice_text = next_item.get("choice_text")
        if choice_text is not None:
            next_item["choice_text"] = str(choice_text)[:STORY_BRANCH_LABEL_MAX_LENGTH]
        bounded_path.append(next_item)
    return bounded_path


def _decode_checkpoint(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["scene_state_snapshot"] = _json_loads(data.pop("scene_state_snapshot_json"), {})
    return data


def _decode_save_slot(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["metadata"] = _json_loads(data.pop("metadata_json"), {})
    data["deleted"] = bool(data.get("deleted"))
    return data


def _collect_int_values(value: Any, keys: set[str]) -> set[int]:
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in keys:
                try:
                    found.add(int(nested))
                except (TypeError, ValueError):
                    pass
            found.update(_collect_int_values(nested, keys))
    elif isinstance(value, list):
        for nested in value:
            found.update(_collect_int_values(nested, keys))
    return found


def _add_cleanup_blockers_from_payload(
    blockers: dict[int, list[dict[str, str]]],
    *,
    payload: Any,
    generated_file_ids: set[int],
    item_to_file_id: Mapping[int, int],
    source_type: str,
    source_id: int,
) -> None:
    referenced_file_ids = _collect_int_values(payload, {"generated_file_id", "file_id"})
    referenced_item_ids = _collect_int_values(
        payload,
        {"item_id", "current_background_item_id", "current_depth_item_id"},
    )
    for item_id in referenced_item_ids:
        file_id = item_to_file_id.get(item_id)
        if file_id is not None:
            referenced_file_ids.add(file_id)
    for file_id in generated_file_ids.intersection(referenced_file_ids):
        blockers.setdefault(file_id, []).append(
            {
                "code": f"vn_play_{source_type}",
                "message": f"File is referenced by VN play {source_type} {source_id}.",
            }
        )


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _choice_id_is_visible(choices: Any, choice_id: Any) -> bool:
    if choice_id is None:
        return False
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return False
    selected_id = str(choice_id)
    return any(
        isinstance(choice, Mapping) and str(choice.get("id")) == selected_id
        for choice in choices
    )
