from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def normalize_character_tags_for_operation(
    tags_value: Any,
    *,
    folder_tag_prefix: str,
) -> list[str]:
    """Normalize character tags while preserving single-folder semantics."""
    if tags_value is None:
        return []

    raw_tags: list[Any]
    if isinstance(tags_value, (list, set, tuple)):
        raw_tags = list(tags_value)
    elif isinstance(tags_value, str):
        if not tags_value.strip():
            return []
        try:
            parsed = json.loads(tags_value)
            if isinstance(parsed, list):
                raw_tags = parsed
            else:
                raw_tags = [tags_value]
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_tags = [tags_value]
    else:
        raw_tags = [tags_value]

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        tag_str = str(tag)
        if not tag_str.strip() or tag_str in seen:
            continue
        seen.add(tag_str)
        normalized.append(tag_str)

    folder_tag: str | None = None
    non_folder_tags: list[str] = []
    for tag in normalized:
        if tag.startswith(folder_tag_prefix):
            folder_tag = tag
            continue
        non_folder_tags.append(tag)
    if folder_tag:
        non_folder_tags.append(folder_tag)
    return non_folder_tags


class CharacterStore:
    """Focused character-card lifecycle store used by CharactersRAGDB."""

    def __init__(self, db: CharactersRAGDB):
        self.db = db

    def ensure_character_tables_ready(self) -> None:
        if not hasattr(self.db, "_schema_lock"):
            self.db._schema_lock = threading.RLock()
        with self.db._schema_lock:
            try:
                self.db.execute_query("SELECT 1 FROM character_cards LIMIT 1")
                return
            except self.db_characters_error() as exc:
                msg = str(exc).lower()
                missing_markers = (
                    "no such table",
                    "does not exist",
                    "missing relation",
                    "undefined table",
                )
                if "character_cards" not in msg or not any(marker in msg for marker in missing_markers):
                    raise
                logger.warning(
                    "Detected missing character_cards table for {}; re-initializing schema.",
                    self.db.db_path_str,
                )

            self.db.close_connection()
            try:
                self.db._initialize_schema()
            except (self.db_schema_error(), self.db_characters_error()):
                raise

            try:
                self.db.execute_query("SELECT 1 FROM character_cards LIMIT 1")
            except self.db_characters_error() as exc:
                logger.error(
                    "Failed to verify character_cards table after schema re-initialization for {}: {}",
                    self.db.db_path_str,
                    exc,
                )
                raise self.db_schema_error()(  # noqa: TRY003
                    f"Schema re-initialization completed but character_cards table is still missing for {self.db.db_path_str}: {exc}"
                ) from exc

    def add_character_card(self, card_data: dict[str, Any]) -> int | None:
        required_fields = ["name"]
        for field in required_fields:
            if field not in card_data or not card_data[field]:
                raise self.db_input_error()(f"Required field '{field}' is missing or empty.")  # noqa: TRY003

        now = self.db._get_current_utc_timestamp_iso()

        def get_json_field_as_string(field_value: Any) -> str | None:
            if isinstance(field_value, str):
                return field_value
            return self.db._ensure_json_string(field_value)

        alt_greetings_json = get_json_field_as_string(card_data.get("alternate_greetings"))
        tags_field_value = card_data.get("tags")
        if tags_field_value is not None:
            if isinstance(tags_field_value, str):
                raw_tags_value = tags_field_value
                stripped_tags_value = raw_tags_value.strip()
                if not stripped_tags_value:
                    tags_field_value = []
                else:
                    try:
                        parsed_tags = json.loads(stripped_tags_value)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        tags_field_value = raw_tags_value
                    else:
                        if isinstance(parsed_tags, list):
                            tags_field_value = self._normalize_character_tags_for_operation(parsed_tags)
                        else:
                            tags_field_value = raw_tags_value
            else:
                tags_field_value = self._normalize_character_tags_for_operation(tags_field_value)
        tags_json = get_json_field_as_string(tags_field_value)
        extensions_json = get_json_field_as_string(card_data.get("extensions"))

        base_query = """
            INSERT INTO character_cards (
                name, description, personality, scenario, image, post_history_instructions,
                first_message, message_example, creator_notes, system_prompt,
                alternate_greetings, tags, creator, character_version, extensions,
                created_at, last_modified, client_id, version, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        client_id = card_data.get("client_id") or self.db.client_id
        params = (
            card_data["name"], card_data.get("description"), card_data.get("personality"),
            card_data.get("scenario"), card_data.get("image"), card_data.get("post_history_instructions"),
            card_data.get("first_message"), card_data.get("message_example"), card_data.get("creator_notes"),
            card_data.get("system_prompt"), alt_greetings_json, tags_json,
            card_data.get("creator"), card_data.get("character_version"), extensions_json,
            now, now, client_id,
        )
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                if self.db.backend_type == BackendType.POSTGRESQL:
                    query = base_query + " RETURNING id"
                    exec_params = params + (1, False)
                    prepared_query, prepared_params = self.db._prepare_backend_statement(query, exec_params)
                    cursor.execute(prepared_query, prepared_params)
                    row = cursor.fetchone()
                    char_id = row["id"] if row else None
                else:
                    exec_params = params + (1, 0)
                    cursor.execute(base_query, exec_params)
                    char_id = cursor.lastrowid
                logger.info("Added character card '{}' with ID: {}.", card_data["name"], char_id)
                return char_id
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: character_cards.name" in str(exc):
                logger.warning("Character card with name '{}' already exists.", card_data["name"])
                raise self.db_conflict_error()(  # noqa: TRY003
                    f"Character card with name '{card_data['name']}' already exists.",
                    entity="character_cards",
                    entity_id=card_data["name"],
                ) from exc
            raise self.db_characters_error()(f"Database integrity error adding character card: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self.db._is_unique_violation(exc):
                logger.warning(
                    "Character card with name '{}' already exists (backend {}).",
                    card_data["name"],
                    self.db.backend_type.value,
                )
                raise self.db_conflict_error()(  # noqa: TRY003
                    f"Character card with name '{card_data['name']}' already exists.",
                    entity="character_cards",
                    entity_id=card_data["name"],
                ) from exc
            raise self.db_characters_error()(f"Database integrity error adding character card: {exc}") from exc  # noqa: TRY003
        except self.db_characters_error() as exc:
            logger.error("Database error adding character card '{}': {}", card_data.get("name"), exc)
            raise
        return None

    def get_character_card_by_id(self, character_id: int) -> dict[str, Any] | None:
        query = "SELECT * FROM character_cards WHERE id = ? AND deleted = 0"
        try:
            cursor = self.db.execute_query(query, (character_id,))
            row = cursor.fetchone()
            return self.db._deserialize_row_fields(row, self.db._CHARACTER_CARD_JSON_FIELDS)
        except self.db_characters_error() as exc:
            logger.error("Database error fetching character card ID {}: {}", character_id, exc)
            raise

    def get_character_card_by_name(self, name: str) -> dict[str, Any] | None:
        query = "SELECT * FROM character_cards WHERE name = ? AND deleted = 0"
        try:
            cursor = self.db.execute_query(query, (name,))
            row = cursor.fetchone()
            return self.db._deserialize_row_fields(row, self.db._CHARACTER_CARD_JSON_FIELDS)
        except self.db_characters_error() as exc:
            if self.db._is_missing_character_table_error(exc):
                logger.warning(
                    "Detected missing character_cards table while fetching by name; attempting schema recovery."
                )
                try:
                    self.ensure_character_tables_ready()
                    cursor = self.db.execute_query(query, (name,))
                    row = cursor.fetchone()
                    return self.db._deserialize_row_fields(row, self.db._CHARACTER_CARD_JSON_FIELDS)
                except (self.db_characters_error(), self.db_schema_error()):
                    logger.error(
                        "Schema recovery failed while fetching character card by name '{}'.",
                        name,
                        exc_info=True,
                    )
                    raise
            logger.error("Database error fetching character card by name '{}': {}", name, exc)
            raise

    def list_character_cards(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        query = "SELECT * FROM character_cards WHERE deleted = 0 ORDER BY name LIMIT ? OFFSET ?"
        try:
            cursor = self.db.execute_query(query, (limit, offset))
            rows = cursor.fetchall()
            return [
                self.db._deserialize_row_fields(row, self.db._CHARACTER_CARD_JSON_FIELDS)
                for row in rows
                if row
            ]
        except self.db_characters_error() as exc:
            logger.error("Database error listing character cards: {}", exc)
            raise

    def query_character_cards(
        self,
        *,
        query: str | None = None,
        tags: list[str] | None = None,
        match_all_tags: bool = False,
        creator: str | None = None,
        has_conversations: bool | None = None,
        favorite_only: bool = False,
        created_from: str | None = None,
        created_to: str | None = None,
        updated_from: str | None = None,
        updated_to: str | None = None,
        include_deleted: bool = False,
        deleted_only: bool = False,
        sort_by: str = "name",
        sort_order: str = "asc",
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        normalized_limit = max(1, int(limit))
        normalized_offset = max(0, int(offset))
        normalized_query = (query or "").strip().lower()
        normalized_creator = (creator or "").strip().lower()
        normalized_tags = [str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()]
        deleted_false = "FALSE" if self.db.backend_type == BackendType.POSTGRESQL else "0"
        deleted_true = "TRUE" if self.db.backend_type == BackendType.POSTGRESQL else "1"
        updated_expr = "COALESCE(cc.last_modified, cc.created_at)"
        conversation_count_expr = (
            "SELECT COUNT(1) FROM conversations conv "  # nosec B608
            f"WHERE conv.deleted = {deleted_false} AND conv.character_id = cc.id"
        )
        last_used_expr = (
            "COALESCE(("  # nosec B608
            "SELECT MAX(conv.last_modified) FROM conversations conv "
            f"WHERE conv.deleted = {deleted_false} AND conv.character_id = cc.id"
            "), cc.created_at)"
        )

        filters: list[str] = []
        params: list[Any] = []

        if normalized_query:
            like_value = f"%{normalized_query}%"
            filters.append(
                "("
                "LOWER(COALESCE(cc.name, '')) LIKE ? OR "
                "LOWER(COALESCE(cc.description, '')) LIKE ? OR "
                "LOWER(COALESCE(cc.personality, '')) LIKE ? OR "
                "LOWER(COALESCE(cc.scenario, '')) LIKE ? OR "
                "LOWER(COALESCE(cc.system_prompt, '')) LIKE ?"
                ")"
            )
            params.extend([like_value, like_value, like_value, like_value, like_value])

        if normalized_tags:
            tag_clauses: list[str] = []
            for tag in normalized_tags:
                if self.db.backend_type == BackendType.SQLITE:
                    tag_clauses.append(
                        "("
                        "(json_valid(cc.tags) AND EXISTS ("
                        "SELECT 1 FROM json_each(cc.tags) je "
                        "WHERE LOWER(TRIM(COALESCE(je.value, ''))) = ?"
                        ")) "
                        "OR LOWER(COALESCE(cc.tags, '')) LIKE ?"
                        ")"
                    )
                    params.append(tag)
                    params.append(f'%"{tag}"%')
                else:
                    tag_clauses.append("LOWER(COALESCE(cc.tags, '')) LIKE ?")
                    params.append(f'%"{tag}"%')
            joiner = " AND " if match_all_tags else " OR "
            filters.append("(" + joiner.join(tag_clauses) + ")")

        if normalized_creator:
            filters.append("LOWER(COALESCE(cc.creator, '')) = ?")
            params.append(normalized_creator)

        if has_conversations is True:
            filters.append(
                "EXISTS ("  # nosec B608
                "SELECT 1 FROM conversations conv "
                f"WHERE conv.deleted = {deleted_false} AND conv.character_id = cc.id"
                ")"
            )
        elif has_conversations is False:
            filters.append(
                "NOT EXISTS ("  # nosec B608
                "SELECT 1 FROM conversations conv "
                f"WHERE conv.deleted = {deleted_false} AND conv.character_id = cc.id"
                ")"
            )

        if favorite_only:
            if self.db.backend_type == BackendType.SQLITE:
                filters.append(
                    "("
                    "json_valid(cc.extensions) AND "
                    "LOWER(COALESCE("
                    "CAST(json_extract(cc.extensions, '$.tldw.favorite') AS TEXT), "
                    "CAST(json_extract(cc.extensions, '$.favorite') AS TEXT), "
                    "'false'"
                    ")) IN ('1', 'true')"
                    ")"
                )
            else:
                filters.append(
                    "("
                    "LOWER(COALESCE("
                    "cc.extensions::jsonb #>> '{tldw,favorite}', "
                    "cc.extensions::jsonb ->> 'favorite', "
                    "'false'"
                    ")) IN ('1', 'true')"
                    ")"
                )

        if created_from:
            filters.append("cc.created_at >= ?")
            params.append(created_from)
        if created_to:
            filters.append("cc.created_at <= ?")
            params.append(created_to)
        if updated_from:
            filters.append(f"{updated_expr} >= ?")
            params.append(updated_from)
        if updated_to:
            filters.append(f"{updated_expr} <= ?")
            params.append(updated_to)

        sort_key_map: dict[str, str] = {
            "name": "LOWER(COALESCE(cc.name, ''))",
            "creator": "LOWER(COALESCE(cc.creator, ''))",
            "created_at": "cc.created_at",
            "updated_at": updated_expr,
            "last_used_at": last_used_expr,
            "conversation_count": f"({conversation_count_expr})",
        }
        normalized_sort_by = sort_by if sort_by in sort_key_map else "name"
        normalized_sort_order = "DESC" if str(sort_order).lower() == "desc" else "ASC"
        sort_expr = sort_key_map[normalized_sort_by]

        if deleted_only:
            deleted_filter = f"cc.deleted = {deleted_true}"
        elif include_deleted:
            deleted_filter = "1=1"
        else:
            deleted_filter = f"cc.deleted = {deleted_false}"

        base_query = f"FROM character_cards cc WHERE {deleted_filter}"
        if filters:
            base_query += " AND " + " AND ".join(filters)

        total_query = f"SELECT COUNT(1) AS total {base_query}"
        data_query = (
            f"SELECT cc.* {base_query} "
            f"ORDER BY {sort_expr} {normalized_sort_order}, cc.id {normalized_sort_order} "
            "LIMIT ? OFFSET ?"
        )

        try:
            total_cursor = self.db.execute_query(total_query, tuple(params))
            total_row = total_cursor.fetchone()
            if total_row is None:
                total = 0
            elif isinstance(total_row, dict):
                total = int(total_row.get("total", 0))
            else:
                try:
                    total = int(total_row["total"])
                except self.db_noncritical_exceptions():
                    total = int(total_row[0]) if len(total_row) > 0 else 0

            data_params = list(params)
            data_params.extend([normalized_limit, normalized_offset])
            data_cursor = self.db.execute_query(data_query, tuple(data_params))
            rows = data_cursor.fetchall()
            items = [
                self.db._deserialize_row_fields(row, self.db._CHARACTER_CARD_JSON_FIELDS)
                for row in rows
                if row
            ]
            return items, total
        except self.db_characters_error() as exc:
            logger.error("Database error querying character cards: {}", exc)
            raise

    def _normalize_character_tags_for_operation(self, tags_value: Any) -> list[str]:
        return normalize_character_tags_for_operation(
            tags_value,
            folder_tag_prefix=self.db._CHARACTER_FOLDER_TAG_PREFIX,
        )

    @staticmethod
    def _apply_character_tag_operation_to_list(
        tags: list[str],
        operation: str,
        source_tag: str,
        target_tag: str | None,
    ) -> list[str]:
        seen: set[str] = set()
        next_tags: list[str] = []

        for tag in tags:
            if operation == "delete" and tag == source_tag:
                continue
            candidate = target_tag if operation in {"rename", "merge"} and tag == source_tag else tag
            candidate = str(candidate or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            next_tags.append(candidate)

        return next_tags

    def manage_character_tags(
        self,
        *,
        operation: str,
        source_tag: str,
        target_tag: str | None = None,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        normalized_operation = str(operation or "").strip().lower()
        if normalized_operation not in {"rename", "merge", "delete"}:
            raise self.db_input_error()(
                f"Unsupported tag operation '{operation}'. Expected rename, merge, or delete."
            )

        normalized_source = str(source_tag or "").strip()
        normalized_target = str(target_tag or "").strip() if target_tag is not None else None

        if not normalized_source:
            raise self.db_input_error()("source_tag is required for tag operations")

        if normalized_operation in {"rename", "merge"} and not normalized_target:
            raise self.db_input_error()("target_tag is required for rename and merge operations")

        normalized_limit = max(1, int(limit))
        all_cards: list[dict[str, Any]] = []
        offset = 0
        batch_size = min(500, normalized_limit)
        while len(all_cards) < normalized_limit:
            batch = self.list_character_cards(limit=batch_size, offset=offset)
            if not batch:
                break
            all_cards.extend(batch)
            if len(batch) < batch_size:
                break
            offset += len(batch)
            remaining = normalized_limit - len(all_cards)
            if remaining <= 0:
                break
            batch_size = min(500, remaining)

        matched_count = 0
        updated_character_ids: list[int] = []
        failed_character_ids: list[int] = []

        for card in all_cards:
            card_id_raw = card.get("id")
            card_version_raw = card.get("version")

            try:
                card_id = int(card_id_raw)
                card_version = int(card_version_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping character tag operation for record with invalid id/version: id={}, version={}",
                    card_id_raw,
                    card_version_raw,
                )
                continue

            existing_tags = self._normalize_character_tags_for_operation(card.get("tags"))
            if normalized_source not in existing_tags:
                continue

            matched_count += 1
            next_tags = self._apply_character_tag_operation_to_list(
                existing_tags,
                normalized_operation,
                normalized_source,
                normalized_target,
            )

            if next_tags == existing_tags:
                continue

            try:
                self.update_character_card(
                    card_id,
                    {"tags": next_tags},
                    expected_version=card_version,
                )
                updated_character_ids.append(card_id)
            except (self.db_conflict_error(), self.db_input_error(), self.db_characters_error()) as exc:
                logger.warning(
                    "Failed to apply '{}' tag operation for character {}: {}",
                    normalized_operation,
                    card_id,
                    exc,
                )
                failed_character_ids.append(card_id)

        return {
            "operation": normalized_operation,
            "source_tag": normalized_source,
            "target_tag": normalized_target if normalized_operation != "delete" else None,
            "matched_count": matched_count,
            "updated_count": len(updated_character_ids),
            "failed_count": len(failed_character_ids),
            "updated_character_ids": updated_character_ids,
            "failed_character_ids": failed_character_ids,
        }

    def update_character_card(
        self,
        character_id: int,
        card_data: dict[str, Any],
        expected_version: int,
    ) -> bool | None:
        logger.debug(
            "Starting update_character_card for ID {} expected_version {} (SINGLE UPDATE STRATEGY)",
            character_id,
            expected_version,
        )

        if not card_data:
            logger.info("No data provided in card_data for character card update ID {}. No-op.", character_id)
            return True

        now = self.db._get_current_utc_timestamp_iso()

        try:
            with self.db.transaction() as conn:
                logger.debug("Transaction started. Connection object: {}", id(conn))

                current_db_version_initial_check = self.db._get_current_db_version(
                    conn,
                    "character_cards",
                    "id",
                    character_id,
                )
                logger.debug(
                    "Initial DB version: {}, Client expected: {}",
                    current_db_version_initial_check,
                    expected_version,
                )

                if current_db_version_initial_check != expected_version:
                    raise self.db_conflict_error()(  # noqa: TRY003, TRY301
                        f"Update failed: version mismatch (db has {current_db_version_initial_check}, client expected {expected_version}) for character_cards ID {character_id}.",
                        entity="character_cards",
                        entity_id=character_id,
                    )

                set_clauses_sql: list[str] = []
                params_for_set_clause: list[Any] = []
                fields_updated_log: list[str] = []

                updatable_direct_fields = [
                    "name", "description", "personality", "scenario", "image",
                    "post_history_instructions", "first_message", "message_example",
                    "creator_notes", "system_prompt", "creator", "character_version",
                ]

                for key, value in card_data.items():
                    if key in self.db._CHARACTER_CARD_JSON_FIELDS:
                        set_clauses_sql.append(f"{key} = ?")
                        normalized_value = value
                        if key == "tags" and value is not None:
                            normalized_value = self._normalize_character_tags_for_operation(value)
                        if isinstance(normalized_value, str):
                            params_for_set_clause.append(normalized_value)
                        else:
                            params_for_set_clause.append(self.db._ensure_json_string(normalized_value))
                        fields_updated_log.append(key)
                    elif key in updatable_direct_fields:
                        set_clauses_sql.append(f"{key} = ?")
                        params_for_set_clause.append(value)
                        fields_updated_log.append(key)
                    elif key not in ["id", "created_at", "last_modified", "version", "client_id", "deleted"]:
                        logger.warning(
                            "Skipping unknown or non-updatable field '{}' in update_character_card payload.",
                            key,
                        )

                next_version_val = expected_version + 1

                set_clauses_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])
                params_for_set_clause.extend([now, next_version_val, self.db.client_id])

                final_update_query = (
                    f"UPDATE character_cards SET {', '.join(set_clauses_sql)} "  # nosec B608
                    "WHERE id = ? AND version = ? AND deleted = 0"
                )
                where_params = [character_id, expected_version]
                final_params = tuple(params_for_set_clause + where_params)

                logger.debug("Executing SINGLE character update query: {}", final_update_query)
                logger.debug("Params: {}", final_params)

                cursor = conn.execute(final_update_query, final_params)
                logger.debug("Character Update executed, rowcount: {}", cursor.rowcount)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = f"Update for character_cards ID {character_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Character card ID {character_id} disappeared before update completion (expected v{expected_version})."
                    elif final_state["deleted"]:
                        msg = f"Character card ID {character_id} was soft-deleted concurrently (expected v{expected_version} for update)."
                    elif final_state["version"] != expected_version:
                        msg = f"Character card ID {character_id} version changed to {final_state['version']} concurrently (expected v{expected_version} for update's WHERE clause)."
                    else:
                        msg = f"Update for character card ID {character_id} (expected v{expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise self.db_conflict_error()(msg, entity="character_cards", entity_id=character_id)  # noqa: TRY301

                log_msg_fields_updated = (
                    f"Fields from payload processed: {fields_updated_log if fields_updated_log else 'None'}."
                )
                logger.info(
                    "Updated character card ID {} (SINGLE UPDATE) from client-expected version {} to final DB version {}. {}",
                    character_id,
                    expected_version,
                    next_version_val,
                    log_msg_fields_updated,
                )
                return True

        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: character_cards.name" in str(exc):
                updated_name = card_data.get("name", "[name not in update_data]")
                logger.warning(
                    "Update for character card ID {} failed: name '{}' already exists.",
                    character_id,
                    updated_name,
                )
                raise self.db_conflict_error()(  # noqa: TRY003
                    f"Cannot update character card ID {character_id}: name '{updated_name}' already exists.",
                    entity="character_cards",
                    entity_id=updated_name,
                ) from exc
            logger.critical(
                "DATABASE IntegrityError during update_character_card (SINGLE UPDATE STRATEGY) for ID {}: {}",
                character_id,
                exc,
                exc_info=True,
            )
            raise self.db_characters_error()(f"Database integrity error during single update: {exc}") from exc  # noqa: TRY003
        except sqlite3.DatabaseError as exc:
            logger.critical(
                "DATABASE ERROR during update_character_card (SINGLE UPDATE STRATEGY) for ID {}: {}",
                character_id,
                exc,
                exc_info=True,
            )
            raise self.db_characters_error()(f"Database error during single update: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self.db._is_unique_violation(exc):
                updated_name = card_data.get("name", "[name not in update_data]")
                logger.warning(
                    "Update for character card ID {} failed on backend {}: name '{}' already exists.",
                    character_id,
                    self.db.backend_type.value,
                    updated_name,
                )
                raise self.db_conflict_error()(  # noqa: TRY003
                    f"Cannot update character card ID {character_id}: name '{updated_name}' already exists.",
                    entity="character_cards",
                    entity_id=updated_name,
                ) from exc
            logger.critical(
                "Backend error during update_character_card (SINGLE UPDATE STRATEGY) for ID {}: {}",
                character_id,
                exc,
                exc_info=True,
            )
            raise self.db_characters_error()(f"Database error during single update: {exc}") from exc  # noqa: TRY003
        except self.db_conflict_error():
            logger.warning("ConflictError during update_character_card for ID {}.", character_id, exc_info=False)
            raise
        except self.db_input_error():
            logger.warning("InputError during update_character_card for ID {}.", character_id, exc_info=False)
            raise
        except self.db_noncritical_exceptions() as exc:
            logger.error(
                "Unexpected Python error in update_character_card (SINGLE UPDATE STRATEGY) for ID {}: {}",
                character_id,
                exc,
                exc_info=True,
            )
            raise self.db_characters_error()(f"Unexpected error updating character card: {exc}") from exc  # noqa: TRY003

    def soft_delete_character_card(self, character_id: int, expected_version: int) -> bool | None:
        now = self.db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = (
            "UPDATE character_cards SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 0"
        )
        params = (now, next_version_val, self.db.client_id, character_id, expected_version)

        try:
            with self.db.transaction() as conn:
                try:
                    current_db_version = self.db._get_current_db_version(conn, "character_cards", "id", character_id)
                except self.db_conflict_error():
                    check_status_cursor = conn.execute(
                        "SELECT deleted, version FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status["deleted"]:
                        logger.info(
                            "Character card ID {} already soft-deleted. Soft delete successful (idempotent).",
                            character_id,
                        )
                        return True
                    raise

                if current_db_version != expected_version:
                    raise self.db_conflict_error()(  # noqa: TRY003, TRY301
                        f"Soft delete for Character ID {character_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="character_cards",
                        entity_id=character_id,
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = f"Soft delete for Character ID {character_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Character card ID {character_id} disappeared before soft delete (expected active version {expected_version})."
                    elif final_state["deleted"]:
                        logger.info(
                            "Character card ID {} was soft-deleted concurrently to version {}. Soft delete successful.",
                            character_id,
                            final_state["version"],
                        )
                        return True
                    elif final_state["version"] != expected_version:
                        msg = f"Soft delete for Character ID {character_id} failed: version changed to {final_state['version']} concurrently (expected {expected_version})."
                    else:
                        msg = f"Soft delete for Character ID {character_id} (expected version {expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise self.db_conflict_error()(msg, entity="character_cards", entity_id=character_id)  # noqa: TRY301

                logger.info(
                    "Soft-deleted character card ID {} (was version {}), new version {}.",
                    character_id,
                    expected_version,
                    next_version_val,
                )
                return True
        except self.db_conflict_error():
            raise
        except BackendDatabaseError as exc:
            logger.error(
                "Backend error soft-deleting character card ID {} (expected v{}): {}",
                character_id,
                expected_version,
                exc,
            )
            raise self.db_characters_error()(f"Backend error during soft delete: {exc}") from exc  # noqa: TRY003
        except self.db_characters_error() as exc:
            logger.error(
                "Database error soft-deleting character card ID {} (expected v{}): {}",
                character_id,
                expected_version,
                exc,
                exc_info=True,
            )
            raise

    def restore_character_card(
        self,
        character_id: int,
        expected_version: int,
        *,
        retention_days: int | None = None,
    ) -> bool | None:
        now = self.db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = (
            "UPDATE character_cards SET deleted = 0, last_modified = ?, version = ?, client_id = ? "
            "WHERE id = ? AND version = ? AND deleted = 1"
        )
        params = (now, next_version_val, self.db.client_id, character_id, expected_version)

        try:
            with self.db.transaction() as conn:
                check_cursor = conn.execute(
                    "SELECT deleted, version, last_modified FROM character_cards WHERE id = ?",
                    (character_id,),
                )
                record_status = check_cursor.fetchone()

                if not record_status:
                    raise self.db_conflict_error()(  # noqa: TRY003, TRY301
                        f"Character card ID {character_id} not found.",
                        entity="character_cards",
                        entity_id=character_id,
                    )

                if not record_status["deleted"]:
                    raise self.db_conflict_error()(  # noqa: TRY003, TRY301
                        f"Character card ID {character_id} is already active; restore cannot succeed.",
                        entity="character_cards",
                        entity_id=character_id,
                    )

                current_db_version = record_status["version"]
                if current_db_version != expected_version:
                    raise self.db_conflict_error()(  # noqa: TRY003, TRY301
                        f"Restore for Character ID {character_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="character_cards",
                        entity_id=character_id,
                    )

                if retention_days is not None:
                    if retention_days < 0:
                        raise self.db_input_error()(  # noqa: TRY301
                            f"Invalid retention_days value {retention_days}. Must be >= 0."
                        )

                    deleted_at_raw = record_status["last_modified"]
                    deleted_at_dt: datetime | None = None

                    if isinstance(deleted_at_raw, datetime):
                        deleted_at_dt = deleted_at_raw
                    elif isinstance(deleted_at_raw, str):
                        normalized = deleted_at_raw.strip()
                        if normalized.endswith("Z"):
                            normalized = f"{normalized[:-1]}+00:00"
                        try:
                            deleted_at_dt = datetime.fromisoformat(normalized)
                        except ValueError:
                            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                                try:
                                    deleted_at_dt = datetime.strptime(normalized, fmt)
                                    break
                                except ValueError:
                                    continue

                    if deleted_at_dt is None:
                        raise self.db_characters_error()(  # noqa: TRY301
                            f"Cannot evaluate restore retention window for character {character_id}: "
                            f"invalid deleted timestamp {deleted_at_raw!r}."
                        )

                    if deleted_at_dt.tzinfo is None:
                        deleted_at_dt = deleted_at_dt.replace(tzinfo=timezone.utc)
                    else:
                        deleted_at_dt = deleted_at_dt.astimezone(timezone.utc)

                    restore_expires_at_dt = deleted_at_dt + timedelta(days=retention_days)
                    now_utc = datetime.now(timezone.utc)
                    if now_utc > restore_expires_at_dt:
                        deleted_at_iso = deleted_at_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                        restore_expires_at_iso = (
                            restore_expires_at_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                        )
                        raise self.db_restore_window_expired_error()(  # noqa: TRY301
                            character_id=character_id,
                            retention_days=retention_days,
                            deleted_at_iso=deleted_at_iso,
                            restore_expires_at_iso=restore_expires_at_iso,
                        )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = f"Restore for Character ID {character_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Character card ID {character_id} disappeared before restore (expected deleted version {expected_version})."
                    elif not final_state["deleted"]:
                        msg = (
                            f"Character card ID {character_id} is already active; "
                            f"restore cannot succeed (concurrent restore detected, current version {final_state['version']})."
                        )
                    elif final_state["version"] != expected_version:
                        msg = f"Restore for Character ID {character_id} failed: version changed to {final_state['version']} concurrently (expected {expected_version})."
                    else:
                        msg = f"Restore for Character ID {character_id} (expected version {expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise self.db_conflict_error()(msg, entity="character_cards", entity_id=character_id)  # noqa: TRY301

                logger.info(
                    "Restored character card ID {} (was version {}), new version {}.",
                    character_id,
                    expected_version,
                    next_version_val,
                )
                return True
        except self.db_conflict_error():
            raise
        except BackendDatabaseError as exc:
            logger.error(
                "Backend error restoring character card ID {} (expected v{}): {}",
                character_id,
                expected_version,
                exc,
            )
            raise self.db_characters_error()(f"Backend error during restore: {exc}") from exc  # noqa: TRY003
        except self.db_characters_error() as exc:
            logger.error(
                "Database error restoring character card ID {} (expected v{}): {}",
                character_id,
                expected_version,
                exc,
                exc_info=True,
            )
            raise

    @staticmethod
    def db_characters_error():
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        return CharactersRAGDBError

    @staticmethod
    def db_schema_error():
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

        return SchemaError

    @staticmethod
    def db_input_error():
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

        return InputError

    @staticmethod
    def db_conflict_error():
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

        return ConflictError

    @staticmethod
    def db_restore_window_expired_error():
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import RestoreWindowExpiredError

        return RestoreWindowExpiredError

    @staticmethod
    def db_noncritical_exceptions():
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import _CHACHA_NONCRITICAL_EXCEPTIONS

        return _CHACHA_NONCRITICAL_EXCEPTIONS
