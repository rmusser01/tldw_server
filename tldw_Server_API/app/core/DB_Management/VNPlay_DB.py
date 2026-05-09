"""VN Play runtime storage for per-user ChaChaNotes databases."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


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
    scene_version INTEGER NOT NULL DEFAULT 0,
    active_turn_request_id INTEGER,
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

CREATE INDEX IF NOT EXISTS idx_vn_play_sessions_owner_user_id
    ON vn_play_sessions(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_sessions_owner_status
    ON vn_play_sessions(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_vn_play_sessions_pack_id
    ON vn_play_sessions(vn_asset_pack_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_events_session_sequence
    ON vn_play_events(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_vn_play_events_owner_user_id
    ON vn_play_events(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_turn_requests_session
    ON vn_play_turn_requests(session_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_turn_requests_owner_status
    ON vn_play_turn_requests(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_vn_play_scene_state_owner_user_id
    ON vn_play_scene_state(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_branches_session
    ON vn_play_branches(session_id);
CREATE INDEX IF NOT EXISTS idx_vn_play_checkpoints_session
    ON vn_play_checkpoints(session_id);
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
                    settings_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def record_story_choice_selection(
        self,
        *,
        session_id: int,
        owner_user_id: int,
        turn_request_id: int,
        client_scene_version: int,
        selected_choice: Mapping[str, Any],
        parent_event_id: int | None,
        expected_scene_last_event_id: int | None = None,
        branch_label: str | None,
        branch_path: Sequence[Any] | None,
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
                    branch_label,
                    _json_dump(list(branch_path or [])),
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
    "scene_version": "scene_version",
    "active_turn_request_id": "active_turn_request_id",
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
    "scene_version": (
        "UPDATE vn_play_sessions SET scene_version = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND (? IS NULL OR owner_user_id = ?)"
    ),
    "active_turn_request_id": (
        "UPDATE vn_play_sessions SET active_turn_request_id = ?, updated_at = CURRENT_TIMESTAMP "
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


def _require_sqlite_chacha_db(db: CharactersRAGDB) -> None:
    if getattr(db, "backend_type", None) != BackendType.SQLITE:
        raise NotImplementedError(
            "VN Play metadata currently supports SQLite ChaChaNotes databases only."
        )


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


def _decode_checkpoint(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["scene_state_snapshot"] = _json_loads(data.pop("scene_state_snapshot_json"), {})
    return data


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
