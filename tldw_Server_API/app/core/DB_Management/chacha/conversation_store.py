from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    FTSQueryTranslator,
    InputError,
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    logger,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class ConversationStore:
    """Focused persistence seam for conversation lifecycle and settings behavior."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _ensure_conversation_settings_table(self) -> None:
        """Ensure the conversation_settings table exists for the active backend."""
        if self._db.backend_type == BackendType.SQLITE:
            self._db.execute_query(
                """
                CREATE TABLE IF NOT EXISTS conversation_settings(
                  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                  settings_json TEXT NOT NULL,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """,
                script=False,
                commit=True,
            )
            return

        if self._db.backend_type == BackendType.POSTGRESQL:
            self._db.backend.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_settings(
                  conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                  settings_json TEXT NOT NULL,
                  last_modified TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
            return

        raise NotImplementedError(
            "conversation_settings table creation not supported for backend "
            f"{self._db.backend_type.value}"
        )

    def upsert_conversation_settings(self, conversation_id: str, settings: dict[str, Any]) -> bool:
        """Upsert per-conversation settings JSON without changing the caller-visible shape."""
        try:
            self._ensure_conversation_settings_table()
            payload = json.dumps(settings)
            if self._db.backend_type == BackendType.SQLITE:
                query = (
                    "INSERT INTO conversation_settings(conversation_id, settings_json, last_modified) "
                    "VALUES (?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(conversation_id) DO UPDATE SET settings_json=excluded.settings_json, "
                    "last_modified=CURRENT_TIMESTAMP"
                )
                self._db.execute_query(query, (conversation_id, payload), commit=True)
                self._db.execute_query(
                    "UPDATE conversations SET version = version + 1, last_modified = CURRENT_TIMESTAMP "
                    "WHERE id = ? AND deleted = 0",
                    (conversation_id,),
                    commit=True,
                )
                return True

            upsert = (
                "INSERT INTO conversation_settings(conversation_id, settings_json, last_modified) "
                "VALUES (%s, %s, NOW()) "
                "ON CONFLICT (conversation_id) DO UPDATE SET settings_json = EXCLUDED.settings_json, "
                "last_modified = NOW()"
            )
            self._db.backend.execute(upsert, (conversation_id, payload))
            self._db.backend.execute(
                "UPDATE conversations SET version = version + 1, last_modified = NOW() "
                "WHERE id = %s AND deleted = 0",
                (conversation_id,),
            )
            return True
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"upsert_conversation_settings failed for {conversation_id}: {exc}")
            return False

    def get_conversation_settings(self, conversation_id: str) -> dict[str, Any] | None:
        """Fetch wrapped settings for a conversation if present."""
        try:
            self._ensure_conversation_settings_table()
            if self._db.backend_type == BackendType.SQLITE:
                cursor = self._db.execute_query(
                    "SELECT settings_json, last_modified FROM conversation_settings WHERE conversation_id = ?",
                    (conversation_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                settings_json, last_modified = row
            else:
                result = self._db.backend.execute(
                    "SELECT settings_json, last_modified FROM conversation_settings WHERE conversation_id = %s",
                    (conversation_id,),
                )
                row = result.fetchone()
                if not row:
                    return None
                settings_json, last_modified = row

            settings = json.loads(settings_json) if settings_json else {}
            return {"settings": settings, "last_modified": last_modified}
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"get_conversation_settings failed for {conversation_id}: {exc}")
            return None

    def _normalize_conversation_state(self, state: str | None) -> str:
        if state is None:
            return self._db._DEFAULT_CONVERSATION_STATE
        if not isinstance(state, str):
            raise InputError(f"Conversation state must be a string. Got: {state!r}")  # noqa: TRY003
        normalized = state.strip().lower()
        if not normalized:
            raise InputError("Conversation state cannot be empty.")  # noqa: TRY003
        if normalized not in self._db._ALLOWED_CONVERSATION_STATES:
            raise InputError(
                f"Invalid conversation state '{state}'. Allowed: {', '.join(self._db._ALLOWED_CONVERSATION_STATES)}"
            )  # noqa: TRY003
        return normalized

    def _normalize_conversation_character_scope(self, character_scope: str | None) -> str:
        if character_scope is None:
            return "all"
        if not isinstance(character_scope, str):
            raise InputError(
                f"Conversation character scope must be a string. Got: {character_scope!r}"
            )  # noqa: TRY003
        normalized = character_scope.strip().lower()
        if not normalized:
            raise InputError("Conversation character scope cannot be empty.")  # noqa: TRY003
        if normalized not in self._db._ALLOWED_CONVERSATION_CHARACTER_SCOPES:
            raise InputError(
                "Invalid conversation character scope "
                f"'{character_scope}'. Allowed: {', '.join(self._db._ALLOWED_CONVERSATION_CHARACTER_SCOPES)}"
            )  # noqa: TRY003
        return normalized

    def _conversation_character_scope_clause(
        self,
        character_scope: str | None,
        *,
        column: str = "character_id",
    ) -> str | None:
        normalized = self._normalize_conversation_character_scope(character_scope)
        if normalized == "all":
            return None
        if normalized == "character":
            return f"{column} IS NOT NULL"
        return f"{column} IS NULL"

    def _conversation_deleted_scope_clause(
        self,
        *,
        include_deleted: bool = False,
        deleted_only: bool = False,
        column: str = "deleted",
        true_literal: str = "1",
        false_literal: str = "0",
    ) -> str | None:
        if deleted_only:
            return f"{column} = {true_literal}"
        if include_deleted:
            return None
        return f"{column} = {false_literal}"

    def _normalize_scope(
        self,
        scope_type: str | None,
        workspace_id: str | None,
    ) -> tuple[str, str | None]:
        if scope_type is None:
            scope_type = "global"
        scope_type = scope_type.strip().lower()
        if scope_type not in self._db._ALLOWED_SCOPE_TYPES:
            raise InputError(
                f"Invalid scope_type '{scope_type}'. Allowed: {', '.join(self._db._ALLOWED_SCOPE_TYPES)}"
            )  # noqa: TRY003
        if scope_type == "workspace":
            if not workspace_id:
                raise InputError("workspace_id is required when scope_type is 'workspace'.")  # noqa: TRY003
        else:
            workspace_id = None
        return scope_type, workspace_id

    def _normalize_conversation_assistant_identity(
        self,
        *,
        character_id: Any,
        assistant_kind: Any,
        assistant_id: Any,
        persona_memory_mode: Any,
    ) -> tuple[str, str, int | None, str | None]:
        normalized_kind = self._db._normalize_nullable_text(assistant_kind)
        normalized_assistant_id = self._db._normalize_nullable_text(assistant_id)
        normalized_memory_mode = self._db._normalize_nullable_text(persona_memory_mode)

        if normalized_kind is None:
            normalized_kind = "character" if character_id is not None else None
        if normalized_kind is None:
            if character_id is None and normalized_assistant_id is None and normalized_memory_mode is None:
                raise InputError("Required field 'character_id' is missing")  # noqa: TRY003
            raise InputError(
                "Conversation requires either 'character_id' or assistant identity fields."
            )  # noqa: TRY003

        normalized_kind = normalized_kind.strip().lower()
        if normalized_kind not in self._db._ALLOWED_CONVERSATION_ASSISTANT_KINDS:
            raise InputError(
                f"Invalid assistant_kind '{normalized_kind}'. Allowed: {', '.join(self._db._ALLOWED_CONVERSATION_ASSISTANT_KINDS)}"
            )  # noqa: TRY003

        if normalized_kind == "character":
            if character_id is None:
                if normalized_assistant_id is None:
                    raise InputError(
                        "Character conversations require 'character_id' or a numeric 'assistant_id'."
                    )  # noqa: TRY003
                try:
                    normalized_character_id = int(normalized_assistant_id)
                except (TypeError, ValueError) as exc:
                    raise InputError(
                        f"Character assistant_id must be numeric. Got: {normalized_assistant_id}"
                    ) from exc  # noqa: TRY003
            else:
                try:
                    normalized_character_id = int(character_id)
                except (TypeError, ValueError) as exc:
                    raise InputError(f"character_id must be numeric. Got: {character_id}") from exc  # noqa: TRY003

            if normalized_memory_mode is not None:
                raise InputError("persona_memory_mode is only valid for persona-backed conversations.")  # noqa: TRY003
            return "character", str(normalized_character_id), normalized_character_id, None

        if normalized_assistant_id is None:
            raise InputError("Persona conversations require a non-empty 'assistant_id'.")  # noqa: TRY003

        if normalized_memory_mode is not None:
            normalized_memory_mode = normalized_memory_mode.strip().lower()
            if normalized_memory_mode not in self._db._ALLOWED_PERSONA_MEMORY_MODES:
                raise InputError(
                    f"Invalid persona_memory_mode '{normalized_memory_mode}'. Allowed: {', '.join(self._db._ALLOWED_PERSONA_MEMORY_MODES)}"
                )  # noqa: TRY003

        return "persona", normalized_assistant_id, None, normalized_memory_mode

    def add_conversation(self, conv_data: dict[str, Any]) -> str | None:
        conv_id = conv_data.get("id") or self._db._generate_uuid()
        root_id = conv_data.get("root_id") or conv_id

        client_id = conv_data.get("client_id") or self._db.client_id
        if not client_id:
            raise InputError(
                "Client ID is required for conversation (either in conv_data or DB instance)."
            )  # noqa: TRY003

        state = self._normalize_conversation_state(conv_data.get("state"))
        topic_label = self._db._normalize_nullable_text(conv_data.get("topic_label"))
        cluster_id = self._db._normalize_nullable_text(conv_data.get("cluster_id"))
        source = self._db._normalize_nullable_text(conv_data.get("source"))
        external_ref = self._db._normalize_nullable_text(conv_data.get("external_ref"))
        assistant_kind, assistant_id, character_id, persona_memory_mode = (
            self._normalize_conversation_assistant_identity(
                character_id=conv_data.get("character_id"),
                assistant_kind=conv_data.get("assistant_kind"),
                assistant_id=conv_data.get("assistant_id"),
                persona_memory_mode=conv_data.get("persona_memory_mode"),
            )
        )
        scope_type, workspace_id = self._normalize_scope(
            conv_data.get("scope_type"),
            conv_data.get("workspace_id"),
        )

        now = self._db._get_current_utc_timestamp_iso()
        query = """
                INSERT INTO conversations (id, root_id, forked_from_message_id, parent_conversation_id, \
                                           character_id, assistant_kind, assistant_id, persona_memory_mode, \
                                           title, state, topic_label, cluster_id, source, external_ref, rating, \
                                           created_at, last_modified, client_id, version, deleted, \
                                           scope_type, workspace_id) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                """
        if self._db.backend_type == BackendType.POSTGRESQL:
            params = (
                conv_id,
                root_id,
                conv_data.get("forked_from_message_id"),
                conv_data.get("parent_conversation_id"),
                character_id,
                assistant_kind,
                assistant_id,
                persona_memory_mode,
                conv_data.get("title"),
                state,
                topic_label,
                cluster_id,
                source,
                external_ref,
                conv_data.get("rating"),
                now,
                now,
                client_id,
                1,
                False,
                scope_type,
                workspace_id,
            )
        else:
            params = (
                conv_id,
                root_id,
                conv_data.get("forked_from_message_id"),
                conv_data.get("parent_conversation_id"),
                character_id,
                assistant_kind,
                assistant_id,
                persona_memory_mode,
                conv_data.get("title"),
                state,
                topic_label,
                cluster_id,
                source,
                external_ref,
                conv_data.get("rating"),
                now,
                now,
                client_id,
                1,
                0,
                scope_type,
                workspace_id,
            )
        try:
            with self._db.transaction() as conn:
                conn.execute(query, params)
            logger.info(f"Added conversation ID: {conv_id}.")
            return conv_id
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: conversations.id" in str(exc):
                raise ConflictError(
                    f"Conversation with ID '{conv_id}' already exists.",
                    entity="conversations",
                    entity_id=conv_id,
                ) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Database integrity error adding conversation: {exc}") from exc  # noqa: TRY003
        except CharactersRAGDBError:
            raise
        return None

    def get_conversation_by_id(
        self,
        conversation_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM conversations WHERE id = ?" if include_deleted else (
            "SELECT * FROM conversations WHERE id = ? AND deleted = 0"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError as exc:
            logger.error(f"Database error fetching conversation ID {conversation_id}: {exc}")
            raise

    def get_conversations_for_character(
        self,
        character_id: int,
        limit: int = 50,
        offset: int = 0,
        client_id: str | None = None,
    ) -> list[dict[str, Any]]:
        client_filter = self._db.client_id if client_id is None else client_id
        query = "SELECT * FROM conversations WHERE character_id = ? AND deleted = 0"
        params: list[Any] = [character_id]
        if client_filter is not None:
            query += " AND client_id = ?"
            params.append(client_filter)
        query += " ORDER BY last_modified DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        try:
            cursor = self._db.execute_query(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as exc:
            logger.error(f"Database error fetching conversations for character ID {character_id}: {exc}")
            raise

    def count_conversations_for_user(
        self,
        client_id: str,
        include_deleted: bool = False,
        deleted_only: bool = False,
        character_scope: str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> int:
        if deleted_only:
            deleted_clause = "deleted = 1"
        elif include_deleted:
            deleted_clause = "1 = 1"
        else:
            deleted_clause = "deleted = 0"
        clauses = ["client_id = ?", deleted_clause]
        params: list[Any] = [client_id]
        character_scope_clause = self._conversation_character_scope_clause(character_scope)
        if character_scope_clause:
            clauses.append(character_scope_clause)
        normalized_workspace_id = self._db._normalize_nullable_text(workspace_id)
        normalized_scope, normalized_workspace_id = self._normalize_scope(scope_type, normalized_workspace_id)
        clauses.append("scope_type = ?")
        params.append(normalized_scope)
        if normalized_workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(normalized_workspace_id)
        query = f"SELECT COUNT(*) as cnt FROM conversations WHERE {' AND '.join(clauses)}"  # nosec B608
        try:
            cursor = self._db.execute_query(query, tuple(params))
            row = cursor.fetchone()
            return int(row[0] if row else 0)
        except CharactersRAGDBError as exc:
            logger.error(f"Database error counting conversations for client_id {client_id}: {exc}")
            raise

    def get_conversations_for_user(
        self,
        client_id: str,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
        deleted_only: bool = False,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        character_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        if deleted_only:
            deleted_clause = "deleted = 1"
        elif include_deleted:
            deleted_clause = "1 = 1"
        else:
            deleted_clause = "deleted = 0"

        clauses = ["client_id = ?", deleted_clause]
        params: list[Any] = [client_id]
        character_scope_clause = self._conversation_character_scope_clause(character_scope)
        if character_scope_clause:
            clauses.append(character_scope_clause)
        normalized_workspace_id = self._db._normalize_nullable_text(workspace_id)
        normalized_scope, normalized_workspace_id = self._normalize_scope(scope_type, normalized_workspace_id)
        clauses.append("scope_type = ?")
        params.append(normalized_scope)
        if normalized_workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(normalized_workspace_id)
        query = (
            "SELECT * FROM conversations "  # nosec B608
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY last_modified DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        try:
            cursor = self._db.execute_query(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as exc:
            logger.error(f"Database error listing conversations for client_id {client_id}: {exc}")
            raise

    def count_conversations_for_user_by_character(
        self,
        client_id: str,
        character_id: int,
        include_deleted: bool = False,
        deleted_only: bool = False,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> int:
        if deleted_only:
            deleted_clause = "deleted = 1"
        elif include_deleted:
            deleted_clause = "1 = 1"
        else:
            deleted_clause = "deleted = 0"
        normalized_workspace_id = self._db._normalize_nullable_text(workspace_id)
        normalized_scope, normalized_workspace_id = self._normalize_scope(scope_type, normalized_workspace_id)
        query = (
            f"SELECT COUNT(1) FROM conversations WHERE client_id = ? AND character_id = ? AND {deleted_clause} "  # nosec B608
            "AND scope_type = ?"
        )
        params: list[Any] = [client_id, character_id, normalized_scope]
        if normalized_workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(normalized_workspace_id)
        try:
            cursor = self._db.execute_query(query, tuple(params))
            row = cursor.fetchone()
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as exc:
            logger.error(
                f"Database error counting conversations for client_id {client_id} and character_id {character_id}: {exc}"
            )
            raise

    def get_conversation_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        query = "SELECT * FROM conversation_clusters WHERE cluster_id = ?"
        try:
            cursor = self._db.execute_query(query, (cluster_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row) if isinstance(row, dict) else {
                "cluster_id": row[0],
                "title": row[1],
                "centroid": row[2],
                "size": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            }
        except CharactersRAGDBError as exc:
            logger.error("Failed to fetch conversation cluster {}: {}", cluster_id, exc)
            raise

    def get_conversations_for_user_and_character(
        self,
        client_id: str,
        character_id: int,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
        deleted_only: bool = False,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if deleted_only:
            deleted_clause = "deleted = 1"
        elif include_deleted:
            deleted_clause = "1 = 1"
        else:
            deleted_clause = "deleted = 0"

        normalized_workspace_id = self._db._normalize_nullable_text(workspace_id)
        normalized_scope, normalized_workspace_id = self._normalize_scope(scope_type, normalized_workspace_id)
        query = (
            "SELECT * FROM conversations "  # nosec B608
            f"WHERE client_id = ? AND character_id = ? AND {deleted_clause} "
            "AND scope_type = ? "
        )
        params: list[Any] = [client_id, character_id, normalized_scope]
        if normalized_workspace_id is not None:
            query += "AND workspace_id = ? "
            params.append(normalized_workspace_id)
        query += "ORDER BY last_modified DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        try:
            cursor = self._db.execute_query(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as exc:
            logger.error(
                f"Database error listing conversations for client_id {client_id} and character_id {character_id}: {exc}"
            )
            raise

    def update_conversation(self, conversation_id: str, update_data: dict[str, Any], expected_version: int) -> bool | None:
        logger.debug(
            "Starting update_conversation for ID {}, expected_version {} (FTS handled by DB triggers)",
            conversation_id,
            expected_version,
        )

        if "rating" in update_data and update_data["rating"] is not None and not (1 <= update_data["rating"] <= 5):
            raise InputError(f"Rating must be between 1 and 5. Got: {update_data['rating']}")  # noqa: TRY003

        if "state" in update_data:
            state_val = update_data.get("state")
            if state_val is None:
                raise InputError("Conversation state cannot be empty.")  # noqa: TRY003
            update_data["state"] = self._normalize_conversation_state(state_val)

        if "topic_label_source" in update_data:
            source_val = update_data.get("topic_label_source")
            if source_val is None:
                update_data["topic_label_source"] = None
            else:
                normalized_source = str(source_val).strip().lower()
                if normalized_source not in {"manual", "auto"}:
                    raise InputError("topic_label_source must be 'manual' or 'auto'.")  # noqa: TRY003
                update_data["topic_label_source"] = normalized_source

        if "topic_last_tagged_at" in update_data:
            tag_val = update_data.get("topic_last_tagged_at")
            if isinstance(tag_val, datetime):
                if tag_val.tzinfo is None:
                    tag_val = tag_val.replace(tzinfo=timezone.utc)
                tag_val = tag_val.astimezone(timezone.utc).isoformat()
            update_data["topic_last_tagged_at"] = tag_val

        now = self._db._get_current_utc_timestamp_iso()

        try:
            with self._db.transaction() as conn:
                current_state = conn.execute(
                    """
                    SELECT rowid, title, version, deleted, character_id, assistant_kind, assistant_id, persona_memory_mode
                    FROM conversations
                    WHERE id = ?
                    """,
                    (conversation_id,),
                ).fetchone()

                if not current_state:
                    raise ConflictError(
                        f"Conversation ID {conversation_id} not found for update.",
                        entity="conversations",
                        entity_id=conversation_id,
                    )  # noqa: TRY003
                if current_state["deleted"]:
                    raise ConflictError(
                        f"Conversation ID {conversation_id} is deleted, cannot update.",
                        entity="conversations",
                        entity_id=conversation_id,
                    )  # noqa: TRY003

                current_db_version = current_state["version"]
                current_title = current_state["title"]
                assistant_update_requested = any(
                    field in update_data
                    for field in ("assistant_kind", "assistant_id", "character_id", "persona_memory_mode")
                )
                normalized_assistant_kind = current_state["assistant_kind"]
                normalized_assistant_id = current_state["assistant_id"]
                normalized_character_id = current_state["character_id"]
                normalized_persona_memory_mode = current_state["persona_memory_mode"]

                if assistant_update_requested:
                    (
                        normalized_assistant_kind,
                        normalized_assistant_id,
                        normalized_character_id,
                        normalized_persona_memory_mode,
                    ) = self._normalize_conversation_assistant_identity(
                        character_id=update_data.get("character_id", current_state["character_id"]),
                        assistant_kind=update_data.get("assistant_kind", current_state["assistant_kind"]),
                        assistant_id=update_data.get("assistant_id", current_state["assistant_id"]),
                        persona_memory_mode=update_data.get(
                            "persona_memory_mode",
                            current_state["persona_memory_mode"],
                        ),
                    )

                if current_db_version != expected_version:
                    raise ConflictError(
                        "Conversation ID {} update failed: version mismatch (db has {}, client expected {}).".format(
                            conversation_id,
                            current_db_version,
                            expected_version,
                        ),
                        entity="conversations",
                        entity_id=conversation_id,
                    )  # noqa: TRY003

                fields_to_update_sql: list[str] = []
                params_for_set_clause: list[Any] = []
                title_changed_flag = False

                if "title" in update_data:
                    fields_to_update_sql.append("title = ?")
                    params_for_set_clause.append(update_data["title"])
                    if update_data["title"] != current_title:
                        title_changed_flag = True
                if "rating" in update_data:
                    fields_to_update_sql.append("rating = ?")
                    params_for_set_clause.append(update_data["rating"])
                if "state" in update_data:
                    fields_to_update_sql.append("state = ?")
                    params_for_set_clause.append(update_data["state"])
                if "topic_label" in update_data:
                    fields_to_update_sql.append("topic_label = ?")
                    params_for_set_clause.append(self._db._normalize_nullable_text(update_data.get("topic_label")))
                if "topic_label_source" in update_data:
                    fields_to_update_sql.append("topic_label_source = ?")
                    params_for_set_clause.append(update_data.get("topic_label_source"))
                if "topic_last_tagged_at" in update_data:
                    fields_to_update_sql.append("topic_last_tagged_at = ?")
                    params_for_set_clause.append(update_data.get("topic_last_tagged_at"))
                if "topic_last_tagged_message_id" in update_data:
                    fields_to_update_sql.append("topic_last_tagged_message_id = ?")
                    params_for_set_clause.append(
                        self._db._normalize_nullable_text(update_data.get("topic_last_tagged_message_id"))
                    )
                if "cluster_id" in update_data:
                    fields_to_update_sql.append("cluster_id = ?")
                    params_for_set_clause.append(self._db._normalize_nullable_text(update_data.get("cluster_id")))
                if "source" in update_data:
                    fields_to_update_sql.append("source = ?")
                    params_for_set_clause.append(self._db._normalize_nullable_text(update_data.get("source")))
                if "external_ref" in update_data:
                    fields_to_update_sql.append("external_ref = ?")
                    params_for_set_clause.append(self._db._normalize_nullable_text(update_data.get("external_ref")))

                if assistant_update_requested:
                    fields_to_update_sql.extend(
                        [
                            "character_id = ?",
                            "assistant_kind = ?",
                            "assistant_id = ?",
                            "persona_memory_mode = ?",
                        ]
                    )
                    params_for_set_clause.extend(
                        [
                            normalized_character_id,
                            normalized_assistant_kind,
                            normalized_assistant_id,
                            normalized_persona_memory_mode,
                        ]
                    )

                next_version_val = expected_version + 1
                if not fields_to_update_sql:
                    main_update_query = (
                        "UPDATE conversations SET last_modified = ?, version = ? "
                        "WHERE id = ? AND version = ? AND deleted = 0"
                    )
                    main_update_params = (now, next_version_val, conversation_id, expected_version)
                else:
                    fields_to_update_sql.extend(["last_modified = ?", "version = ?"])
                    final_set_values = params_for_set_clause[:] + [now, next_version_val]
                    main_update_query = (
                        f"UPDATE conversations SET {', '.join(fields_to_update_sql)} "  # nosec B608
                        "WHERE id = ? AND version = ? AND deleted = 0"
                    )
                    main_update_params = tuple(final_set_values + [conversation_id, expected_version])

                cursor_main = conn.execute(main_update_query, main_update_params)
                if cursor_main.rowcount == 0:
                    final_state = conn.execute(
                        "SELECT version, deleted FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone()
                    msg = (
                        f"Main update for conversation ID {conversation_id} (expected v{expected_version}) affected 0 rows."
                    )
                    if not final_state:
                        msg = (
                            f"Conversation ID {conversation_id} disappeared before update completion "
                            f"(expected v{expected_version})."
                        )
                    elif final_state["deleted"]:
                        msg = (
                            f"Conversation ID {conversation_id} was soft-deleted concurrently "
                            f"(expected v{expected_version} for update)."
                        )
                    elif final_state["version"] != expected_version:
                        msg = (
                            f"Conversation ID {conversation_id} version changed to {final_state['version']} "
                            f"concurrently (expected v{expected_version} for update)."
                        )
                    raise ConflictError(msg, entity="conversations", entity_id=conversation_id)

                logger.info(
                    "Updated conversation ID {} from version {} to version {} (FTS handled by DB triggers). "
                    "Title changed: {}",
                    conversation_id,
                    expected_version,
                    next_version_val,
                    title_changed_flag,
                )
                return True

        except sqlite3.IntegrityError as exc:
            raise CharactersRAGDBError(f"Database integrity error during update_conversation: {exc}") from exc  # noqa: TRY003
        except sqlite3.DatabaseError as exc:
            logger.critical(
                "DATABASE ERROR during update_conversation (FTS handled by DB triggers): {}",
                exc,
            )
            raise CharactersRAGDBError(f"Database error during update_conversation: {exc}") from exc  # noqa: TRY003
        except ConflictError:
            raise
        except InputError:
            raise
        except CharactersRAGDBError:
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(
                f"Unexpected Python error in update_conversation for ID {conversation_id}: {exc}",
                exc_info=True,
            )
            raise CharactersRAGDBError(f"Unexpected error during update_conversation: {exc}") from exc  # noqa: TRY003

    def soft_delete_conversation(self, conversation_id: str, expected_version: int) -> bool | None:
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1
        query = (
            "UPDATE conversations SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 0"
        )
        params = (now, next_version_val, self._db.client_id, conversation_id, expected_version)

        try:
            with self._db.transaction() as conn:
                try:
                    current_db_version = self._db._get_current_db_version(
                        conn,
                        "conversations",
                        "id",
                        conversation_id,
                    )
                except ConflictError:
                    record_status = conn.execute(
                        "SELECT deleted, version FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone()
                    if record_status and record_status["deleted"]:
                        logger.info(f"Conversation ID {conversation_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise

                if current_db_version != expected_version:
                    raise ConflictError(
                        (
                            f"Soft delete for Conversation ID {conversation_id} failed: "
                            f"version mismatch (db has {current_db_version}, client expected {expected_version})."
                        ),
                        entity="conversations",
                        entity_id=conversation_id,
                    )

                cursor = conn.execute(query, params)
                if cursor.rowcount == 0:
                    final_state = conn.execute(
                        "SELECT version, deleted FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone()
                    msg = (
                        f"Soft delete for conversation ID {conversation_id} "
                        f"(expected v{expected_version}) affected 0 rows."
                    )
                    if not final_state:
                        msg = f"Conversation ID {conversation_id} disappeared."
                    elif final_state["deleted"]:
                        logger.info(f"Conversation ID {conversation_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state["version"] != expected_version:
                        msg = (
                            f"Conversation ID {conversation_id} version changed to {final_state['version']} "
                            "concurrently."
                        )
                    raise ConflictError(msg, entity="conversations", entity_id=conversation_id)

                logger.info(
                    f"Soft-deleted conversation ID {conversation_id} (was v{expected_version}), "
                    f"new version {next_version_val}."
                )
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError:
            raise

    def restore_conversation(self, conversation_id: str, expected_version: int) -> bool | None:
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1
        query = (
            "UPDATE conversations "
            "SET deleted = 0, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 1"
        )
        params = (now, next_version_val, self._db.client_id, conversation_id, expected_version)

        try:
            with self._db.transaction() as conn:
                record_status = conn.execute(
                    "SELECT deleted, version FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if not record_status:
                    raise ConflictError(
                        f"Conversation ID {conversation_id} not found.",
                        entity="conversations",
                        entity_id=conversation_id,
                    )
                if not record_status["deleted"]:
                    logger.info(
                        f"Conversation ID {conversation_id} already active. Restore successful (idempotent)."
                    )
                    return True

                current_db_version = record_status["version"]
                if current_db_version != expected_version:
                    raise ConflictError(
                        (
                            f"Restore for Conversation ID {conversation_id} failed: "
                            f"version mismatch (db has {current_db_version}, client expected {expected_version})."
                        ),
                        entity="conversations",
                        entity_id=conversation_id,
                    )

                cursor = conn.execute(query, params)
                if cursor.rowcount == 0:
                    final_state = conn.execute(
                        "SELECT version, deleted FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone()
                    msg = (
                        f"Restore for conversation ID {conversation_id} "
                        f"(expected v{expected_version}) affected 0 rows."
                    )
                    if not final_state:
                        msg = f"Conversation ID {conversation_id} disappeared."
                    elif not final_state["deleted"]:
                        logger.info(f"Conversation ID {conversation_id} was restored concurrently. Success.")
                        return True
                    elif final_state["version"] != expected_version:
                        msg = (
                            f"Conversation ID {conversation_id} version changed to "
                            f"{final_state['version']} concurrently."
                        )
                    raise ConflictError(msg, entity="conversations", entity_id=conversation_id)

                logger.info(
                    f"Restored conversation ID {conversation_id} (was v{expected_version}), "
                    f"new version {next_version_val}."
                )
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError:
            raise

    def hard_delete_conversation(self, conversation_id: str) -> bool:
        try:
            with self._db.transaction() as conn:
                rowcount = conn.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).rowcount
                return bool(rowcount and rowcount > 0)
        except CharactersRAGDBError as exc:
            logger.error(f"Database error hard-deleting conversation ID {conversation_id}: {exc}", exc_info=True)
            raise

    def search_conversations_by_title(
        self,
        title_query: str,
        character_id: int | None = None,
        character_scope: str | None = None,
        limit: int = 10,
        offset: int = 0,
        client_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not title_query.strip():
            logger.warning("Empty title_query provided for conversation search. Returning empty list.")
            return []

        client_filter = self._db.client_id if client_id is None else client_id
        normalized_character_scope = self._normalize_conversation_character_scope(character_scope)

        if self._db.backend_type == BackendType.POSTGRESQL:
            tsquery = FTSQueryTranslator.normalize_query(title_query, "postgresql")
            if not tsquery:
                logger.debug("Conversation title query normalized to empty tsquery for input '{}'", title_query)
                return []

            base_query = """
                SELECT c.*, ts_rank(c.conversations_fts_tsv, to_tsquery('english', ?)) AS bm25_raw
                FROM conversations c
                WHERE c.deleted = FALSE
                  AND c.conversations_fts_tsv @@ to_tsquery('english', ?)
            """
            params_list: list[Any] = [tsquery, tsquery]
            filters: list[str] = []
            if character_id is not None:
                filters.append("c.character_id = ?")
                params_list.append(character_id)
            elif normalized_character_scope != "all":
                filters.append(
                    self._conversation_character_scope_clause(
                        normalized_character_scope,
                        column="c.character_id",
                    )
                )
            if client_filter is not None:
                filters.append("c.client_id = ?")
                params_list.append(client_filter)
            if filters:
                base_query += " AND " + " AND ".join(filters)

            cursor = self._db.execute_query(base_query, tuple(params_list))
            rows = [dict(row) for row in cursor.fetchall()]
            if not rows:
                return []
            max_score = max([row.get("bm25_raw", 0) or 0 for row in rows]) or 0
            for row in rows:
                raw = row.get("bm25_raw", 0) or 0
                row["bm25_norm"] = (raw / max_score) if max_score else 0
            rows.sort(
                key=lambda row: (-(row.get("bm25_norm") or 0), str(row.get("last_modified") or ""), row.get("id") or ""),
                reverse=False,
            )
            return rows[offset: offset + limit]

        filters: list[str] = ["conversations_fts MATCH ?", "c.deleted = 0"]
        params_filters: list[Any] = [title_query]
        if character_id is not None:
            filters.append("c.character_id = ?")
            params_filters.append(character_id)
        elif normalized_character_scope != "all":
            filters.append(
                self._conversation_character_scope_clause(
                    normalized_character_scope,
                    column="c.character_id",
                )
            )
        if client_filter is not None:
            filters.append("c.client_id = ?")
            params_filters.append(client_filter)

        where_clause = " AND ".join(filters)
        select_query = """
            SELECT c.*, bm25(conversations_fts) AS bm25_raw
            FROM conversations_fts
            JOIN conversations c ON conversations_fts.rowid = c.rowid
            WHERE {where_clause}
        """.format_map(locals())  # nosec B608

        cursor = self._db.execute_query(select_query, tuple(params_filters))
        rows = [dict(row) for row in cursor.fetchall()]
        if not rows:
            return []
        max_bm25 = max([-1 * (row.get("bm25_raw", 0) or 0) for row in rows]) or 0
        for row in rows:
            raw = -1 * (row.get("bm25_raw", 0) or 0)
            row["bm25_norm"] = (raw / max_bm25) if max_bm25 else 0

        rows.sort(
            key=lambda row: (-(row.get("bm25_norm") or 0), str(row.get("last_modified") or ""), row.get("id") or ""),
            reverse=False,
        )
        return rows[offset: offset + limit]

    def _normalize_conversation_search_order(self, order_by: str | None) -> str:
        if order_by is None:
            return "recency"
        if not isinstance(order_by, str):
            raise InputError(f"Conversation search order must be a string. Got: {order_by!r}")  # noqa: TRY003
        normalized = order_by.strip().lower()
        if not normalized:
            raise InputError("Conversation search order cannot be empty.")  # noqa: TRY003
        if normalized not in self._db._ALLOWED_CONVERSATION_SEARCH_ORDER:
            raise InputError(
                "Invalid conversation search order "
                f"'{order_by}'. Allowed: {', '.join(self._db._ALLOWED_CONVERSATION_SEARCH_ORDER)}"
            )  # noqa: TRY003
        return normalized

    def _build_conversation_search_filters(
        self,
        *,
        alias: str,
        client_filter: str | None,
        include_deleted: bool,
        deleted_only: bool,
        character_id: int | None,
        character_scope: str | None,
        scope_type: str | None,
        workspace_id: str | None,
        state: str | None,
        topic_label: str | None,
        topic_prefix: bool,
        cluster_id: str | None,
        keywords: list[str] | None,
        start_date: str | None,
        end_date: str | None,
        date_expr: str,
        keyword_table: str,
        keyword_deleted_literal: str,
        deleted_true_literal: str,
        deleted_false_literal: str,
    ) -> tuple[list[str], list[Any]]:
        normalized_character_scope = self._normalize_conversation_character_scope(character_scope)
        filters: list[str] = []
        params: list[Any] = []

        deleted_clause = self._conversation_deleted_scope_clause(
            include_deleted=include_deleted,
            deleted_only=deleted_only,
            column=f"{alias}.deleted",
            true_literal=deleted_true_literal,
            false_literal=deleted_false_literal,
        )
        if deleted_clause:
            filters.append(deleted_clause)

        if character_id is not None:
            filters.append(f"{alias}.character_id = ?")
            params.append(character_id)
        elif normalized_character_scope != "all":
            filters.append(
                self._conversation_character_scope_clause(
                    normalized_character_scope,
                    column=f"{alias}.character_id",
                )
            )

        normalized_workspace_id = self._db._normalize_nullable_text(workspace_id)
        normalized_scope, normalized_workspace_id = self._normalize_scope(scope_type, normalized_workspace_id)
        filters.append(f"{alias}.scope_type = ?")
        params.append(normalized_scope)
        if normalized_workspace_id is not None:
            filters.append(f"{alias}.workspace_id = ?")
            params.append(normalized_workspace_id)

        if client_filter is not None:
            filters.append(f"{alias}.client_id = ?")
            params.append(client_filter)

        if state is not None:
            filters.append(f"{alias}.state = ?")
            params.append(self._normalize_conversation_state(state))

        if topic_label:
            normalized_topic = topic_label.rstrip("*").strip().lower()
            if normalized_topic:
                if topic_prefix:
                    filters.append(f"LOWER({alias}.topic_label) LIKE ?")
                    params.append(f"{normalized_topic}%")
                else:
                    filters.append(f"LOWER({alias}.topic_label) = ?")
                    params.append(normalized_topic)

        if cluster_id:
            filters.append(f"{alias}.cluster_id = ?")
            params.append(cluster_id)

        if start_date:
            filters.append(f"{date_expr} >= ?")
            params.append(start_date)
        if end_date:
            filters.append(f"{date_expr} <= ?")
            params.append(end_date)
        if keywords:
            for keyword in keywords:
                filters.append(
                    f"EXISTS (SELECT 1 FROM conversation_keywords ck "  # nosec B608
                    f"JOIN {keyword_table} k ON k.id = ck.keyword_id "
                    f"WHERE ck.conversation_id = {alias}.id AND k.deleted = {keyword_deleted_literal} "
                    "AND LOWER(k.keyword) = ?)"
                )
                params.append(keyword.lower())
        return filters, params

    def _conversation_deleted_text_search_clause(
        self,
        *,
        alias: str,
        query: str,
    ) -> tuple[str, list[str]]:
        normalized_query = query.strip().lower()
        like_pattern = f"%{normalized_query}%"
        clause = (
            f"(LOWER(COALESCE({alias}.title, '')) LIKE ? "
            f"OR LOWER(COALESCE({alias}.topic_label, '')) LIKE ? "
            f"OR LOWER(COALESCE({alias}.state, '')) LIKE ?)"
        )
        return clause, [like_pattern, like_pattern, like_pattern]

    def search_conversations(
        self,
        query: str | None,
        *,
        client_id: str | None = None,
        include_deleted: bool = False,
        deleted_only: bool = False,
        character_id: int | None = None,
        character_scope: str | None = None,
        state: str | None = None,
        topic_label: str | None = None,
        topic_prefix: bool = False,
        cluster_id: str | None = None,
        keywords: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        date_field: str = "last_modified",
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        client_filter = self._db.client_id if client_id is None else client_id
        safe_query = (query or "").strip() or None

        if date_field not in {"last_modified", "created_at"}:
            raise InputError("date_field must be 'last_modified' or 'created_at'")  # noqa: TRY003

        keyword_table = self._db._map_table_for_backend("keywords")
        use_deleted_text_search = safe_query is not None and (include_deleted or deleted_only)

        if self._db.backend_type == BackendType.POSTGRESQL:
            date_expr = "c.created_at" if date_field == "created_at" else "COALESCE(c.last_modified, c.created_at)"
            base_query = "SELECT c.*, 0.0 AS bm25_raw FROM conversations c WHERE TRUE"
            params: list[Any] = []
            if safe_query:
                if use_deleted_text_search:
                    text_clause, text_params = self._conversation_deleted_text_search_clause(alias="c", query=safe_query)
                    base_query += f" AND {text_clause}"
                    params.extend(text_params)
                else:
                    tsquery = FTSQueryTranslator.normalize_query(safe_query, "postgresql")
                    if not tsquery:
                        return []
                    base_query = (
                        "SELECT c.*, ts_rank(c.conversations_fts_tsv, to_tsquery('english', ?)) AS bm25_raw "
                        "FROM conversations c "
                        "WHERE c.conversations_fts_tsv @@ to_tsquery('english', ?)"
                    )
                    params.extend([tsquery, tsquery])

            filters, filter_params = self._build_conversation_search_filters(
                alias="c",
                client_filter=client_filter,
                include_deleted=include_deleted or deleted_only,
                deleted_only=deleted_only,
                character_id=character_id,
                character_scope=character_scope,
                state=state,
                topic_label=topic_label,
                topic_prefix=topic_prefix,
                cluster_id=cluster_id,
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                date_expr=date_expr,
                keyword_table=keyword_table,
                keyword_deleted_literal="FALSE",
                deleted_true_literal="TRUE",
                deleted_false_literal="FALSE",
                scope_type=scope_type,
                workspace_id=workspace_id,
            )
            params.extend(filter_params)
            if filters:
                base_query += " AND " + " AND ".join(filters)
            cursor = self._db.execute_query(base_query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

        date_expr = "c.created_at" if date_field == "created_at" else "COALESCE(NULLIF(c.last_modified,''), c.created_at)"
        params: list[Any] = []
        filters: list[str] = []

        if safe_query:
            if use_deleted_text_search:
                text_clause, text_params = self._conversation_deleted_text_search_clause(alias="c", query=safe_query)
                filters.append(text_clause)
                params.extend(text_params)
                base_query = "SELECT c.*, 0.0 AS bm25_raw FROM conversations c WHERE 1 = 1"
            else:
                filters.append("conversations_fts MATCH ?")
                params.append(safe_query)
                base_query = (
                    "SELECT c.*, (bm25(conversations_fts) * -1) AS bm25_raw "
                    "FROM conversations_fts JOIN conversations c ON conversations_fts.rowid = c.rowid "
                    "WHERE 1 = 1"
                )
        else:
            base_query = "SELECT c.*, 0.0 AS bm25_raw FROM conversations c WHERE 1 = 1"

        extra_filters, extra_params = self._build_conversation_search_filters(
            alias="c",
            client_filter=client_filter,
            include_deleted=include_deleted or deleted_only,
            deleted_only=deleted_only,
            character_id=character_id,
            character_scope=character_scope,
            state=state,
            topic_label=topic_label,
            topic_prefix=topic_prefix,
            cluster_id=cluster_id,
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            date_expr=date_expr,
            keyword_table=keyword_table,
            keyword_deleted_literal="0",
            deleted_true_literal="1",
            deleted_false_literal="0",
            scope_type=scope_type,
            workspace_id=workspace_id,
        )
        filters.extend(extra_filters)
        params.extend(extra_params)
        if filters:
            base_query += " AND " + " AND ".join(filters)

        cursor = self._db.execute_query(base_query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def search_conversations_page(self, query: str | None, **kwargs: Any) -> tuple[list[dict[str, Any]], int, float]:
        return self._db._search_conversations_page_impl(query, **kwargs)
