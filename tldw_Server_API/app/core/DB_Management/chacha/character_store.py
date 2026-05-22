"""Character card persistence operations extracted from ``ChaChaNotes_DB``.

This store owns character-card CRUD, search, tag normalization, and lifecycle
updates while delegating connection, transaction, serialization, and backend
adapter behavior to the parent ``CharactersRAGDB`` instance.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    FTSQueryTranslator,
    InputError,
    RestoreWindowExpiredError,
    SchemaError,
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    logger,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.chacha import exemplar_normalization

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


# ---------------------------------------------------------------------------
# Constants duplicated from CharactersRAGDB so the store is self-contained
# for tag-normalisation logic that uses classmethod references.
# ---------------------------------------------------------------------------
_CHARACTER_FOLDER_TAG_PREFIX: str = "__tldw_folder_id:"


class CharacterStore:
    """Focused persistence seam for character card CRUD operations."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _deleted_literal(self, deleted: bool) -> str:
        """Return a backend-safe SQL literal for soft-delete predicates."""
        if self._db.backend_type == BackendType.POSTGRESQL:
            return "TRUE" if deleted else "FALSE"
        return "1" if deleted else "0"

    def _deleted_value(self, deleted: bool) -> bool | int:
        """Return the backend-native value for a soft-delete flag."""
        return deleted if self._db.backend_type == BackendType.POSTGRESQL else int(deleted)

    # ------------------------------------------------------------------
    # Character card creation
    # ------------------------------------------------------------------

    def add_character_card(self, card_data: dict[str, Any]) -> int | None:
        """
        Adds a new character card to the database.

        The ``client_id`` for the new record is taken from the ``CharactersRAGDB``
        instance.  ``version`` defaults to 1.  ``created_at`` and ``last_modified``
        are set to the current UTC time.  Fields like ``alternate_greetings``,
        ``tags``, and ``extensions`` (from ``_CHARACTER_CARD_JSON_FIELDS``) are
        stored as JSON strings.

        FTS updates (``character_cards_fts``) and ``sync_log`` entries for
        creations are handled automatically by SQL triggers.

        Args:
            card_data: A dictionary containing the character card data.
                       Required fields: 'name'.
                       Optional fields include: 'description', 'personality',
                       'scenario', 'image', 'post_history_instructions',
                       'first_message', 'message_example', 'creator_notes',
                       'system_prompt', 'alternate_greetings' (list/set/JSON str),
                       'tags' (list/set/JSON str), 'creator', 'character_version',
                       'extensions' (dict/JSON str).

        Returns:
            The integer ID of the newly created character card.

        Raises:
            InputError: If required fields (e.g., 'name') are missing or empty.
            ConflictError: If a character card with the same 'name' already exists.
            CharactersRAGDBError: For other database-related errors during insertion.
        """
        required_fields = ['name']
        for field in required_fields:
            if field not in card_data or not card_data[field]:
                raise InputError(f"Required field '{field}' is missing or empty.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()

        # Ensure JSON fields are strings or None
        def get_json_field_as_string(field_value: Any) -> str | None:
            if isinstance(field_value, str):
                # Assume it's already a JSON string if it's a string
                return field_value
            return self._db._ensure_json_string(field_value)

        alt_greetings_json = get_json_field_as_string(card_data.get('alternate_greetings'))
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
                        # Preserve legacy behavior for invalid JSON tag strings.
                        tags_field_value = raw_tags_value
                    else:
                        if isinstance(parsed_tags, list):
                            tags_field_value = self._normalize_character_tags_for_operation(parsed_tags)
                        else:
                            tags_field_value = raw_tags_value
            else:
                tags_field_value = self._normalize_character_tags_for_operation(tags_field_value)
        tags_json = get_json_field_as_string(tags_field_value)
        extensions_json = get_json_field_as_string(card_data.get('extensions'))

        base_query = """
            INSERT INTO character_cards (
                name, description, personality, scenario, image, post_history_instructions,
                first_message, message_example, creator_notes, system_prompt,
                alternate_greetings, tags, creator, character_version, extensions,
                created_at, last_modified, client_id, version, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            card_data['name'], card_data.get('description'), card_data.get('personality'),
            card_data.get('scenario'), card_data.get('image'), card_data.get('post_history_instructions'),
            card_data.get('first_message'), card_data.get('message_example'), card_data.get('creator_notes'),
            card_data.get('system_prompt'), alt_greetings_json, tags_json,
            card_data.get('creator'), card_data.get('character_version'), extensions_json,
            now, now, self._db.client_id,  # created_at, last_modified, client_id
        )
        try:
            with self._db.transaction() as conn:
                cursor = conn.cursor()
                if self._db.backend_type == BackendType.POSTGRESQL:
                    query = base_query + " RETURNING id"
                    exec_params = params + (1, False)
                    prepared_query, prepared_params = self._db._prepare_backend_statement(query, exec_params)
                    cursor.execute(prepared_query, prepared_params)
                    row = cursor.fetchone()
                    char_id = row['id'] if row else None
                else:
                    exec_params = params + (1, 0)
                    cursor.execute(base_query, exec_params)
                    char_id = cursor.lastrowid
                logger.info(f"Added character card '{card_data['name']}' with ID: {char_id}.")
                return char_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: character_cards.name" in str(e):
                logger.warning(f"Character card with name '{card_data['name']}' already exists.")
                raise ConflictError(  # noqa: TRY003
                    f"Character card with name '{card_data['name']}' already exists.",
                    entity="character_cards", entity_id=card_data['name'],
                ) from e
            raise CharactersRAGDBError(f"Database integrity error adding character card: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            if self._db._is_unique_violation(e):
                logger.warning(
                    "Character card with name '{}' already exists (backend {}).",
                    card_data['name'],
                    self._db.backend_type.value,
                )
                raise ConflictError(  # noqa: TRY003
                    f"Character card with name '{card_data['name']}' already exists.",
                    entity="character_cards",
                    entity_id=card_data['name'],
                ) from e
            raise CharactersRAGDBError(f"Database integrity error adding character card: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding character card '{card_data.get('name')}': {e}")
            raise
        return None  # Should not be reached

    # ------------------------------------------------------------------
    # Character card retrieval
    # ------------------------------------------------------------------

    def get_character_card_by_id(
        self,
        character_id: int,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """
        Retrieve a specific character card by its ID.

        Only non-deleted cards are returned.  JSON fields (alternate_greetings,
        tags, extensions as defined in ``_CHARACTER_CARD_JSON_FIELDS``)
        are deserialized from strings to Python objects.

        Args:
            character_id: The integer ID of the character card.

        Returns:
            A dictionary containing the character card data if found and not
            deleted, otherwise None.

        Raises:
            CharactersRAGDBError: For database errors during fetching.
        """
        query = "SELECT * FROM character_cards WHERE id = ?"
        params: tuple[Any, ...]
        if include_deleted:
            params = (character_id,)
        else:
            query += f" AND deleted = {self._deleted_literal(False)}"
            params = (character_id,)
        try:
            cursor = self._db.execute_query(query, params)
            row = cursor.fetchone()
            return self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching character card ID {character_id}: {e}")
            raise

    def get_character_cards_by_ids(
        self,
        character_ids: list[int],
        include_deleted: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """
        Retrieve multiple character cards by ID in one query.

        Args:
            character_ids: Candidate character card IDs. Duplicates and invalid
                non-positive IDs are ignored.
            include_deleted: Whether soft-deleted character cards should be
                included in the result.

        Returns:
            Mapping of character card ID to deserialized character card data.

        Raises:
            CharactersRAGDBError: For database errors during fetching.
        """
        normalized_ids: list[int] = []
        seen: set[int] = set()
        for character_id in character_ids:
            try:
                normalized_id = int(character_id)
            except (TypeError, ValueError):
                continue
            if normalized_id <= 0 or normalized_id in seen:
                continue
            normalized_ids.append(normalized_id)
            seen.add(normalized_id)

        if not normalized_ids:
            return {}

        placeholders = ", ".join("?" for _ in normalized_ids)
        query = f"SELECT * FROM character_cards WHERE id IN ({placeholders})"  # nosec B608
        params: list[Any] = list(normalized_ids)
        if not include_deleted:
            query += f" AND deleted = {self._deleted_literal(False)}"

        try:
            cursor = self._db.execute_query(query, tuple(params))
            characters: dict[int, dict[str, Any]] = {}
            for row in cursor.fetchall():
                character = self._db._deserialize_row_fields(
                    row,
                    self._db._CHARACTER_CARD_JSON_FIELDS,
                )
                if not character:
                    continue
                try:
                    result_id = int(character.get("id"))
                except (TypeError, ValueError):
                    continue
                characters[result_id] = character
            return characters
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching character card IDs {normalized_ids}: {e}")
            raise

    def get_character_card_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Retrieve a specific character card by its unique name.

        Only non-deleted cards are returned.  JSON fields (see
        ``_CHARACTER_CARD_JSON_FIELDS``) are deserialized.  Name comparison
        is case-sensitive as per default SQLite behavior because the schema
        column "name" does not specify ``COLLATE NOCASE``.

        Args:
            name: The unique name of the character card.

        Returns:
            A dictionary containing character card data if found and not
            deleted, otherwise None.

        Raises:
            CharactersRAGDBError: For database errors during fetching.
        """
        query = "SELECT * FROM character_cards WHERE name = ? AND deleted = ?"
        try:
            cursor = self._db.execute_query(query, (name, self._deleted_value(False)))
            row = cursor.fetchone()
            return self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
        except CharactersRAGDBError as e:
            if self._db._is_missing_character_table_error(e):
                logger.warning(
                    "Detected missing character_cards table while fetching by name; attempting schema recovery."
                )
                try:
                    self._db.ensure_character_tables_ready()
                    cursor = self._db.execute_query(query, (name, self._deleted_value(False)))
                    row = cursor.fetchone()
                    return self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
                except (CharactersRAGDBError, SchemaError):
                    logger.error(
                        "Schema recovery failed while fetching character card by name '{}'.",
                        name,
                        exc_info=True,
                    )
                    raise
            logger.error(f"Database error fetching character card by name '{name}': {e}")
            raise

    # ------------------------------------------------------------------
    # Character card listing / querying
    # ------------------------------------------------------------------

    def list_character_cards(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """
        Lists character cards, ordered by name.

        Only non-deleted cards are returned.  JSON fields (see
        ``_CHARACTER_CARD_JSON_FIELDS``) are deserialized.

        Args:
            limit: The maximum number of cards to return.  Defaults to 100.
            offset: The number of cards to skip before starting to return.
                    Defaults to 0.

        Returns:
            A list of dictionaries, each representing a character card.
            The list may be empty if no cards are found.

        Raises:
            CharactersRAGDBError: For database errors during listing.
        """
        query = "SELECT * FROM character_cards WHERE deleted = ? ORDER BY name LIMIT ? OFFSET ?"
        try:
            cursor = self._db.execute_query(query, (self._deleted_value(False), limit, offset))
            rows = cursor.fetchall()
            return [self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS) for row in rows if row]
        except CharactersRAGDBError as e:
            logger.error(f"Database error listing character cards: {e}")
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
        """
        Query character cards with server-side filtering, sorting, and pagination.

        Returns:
            tuple of (items, total_count)
        """
        normalized_limit = max(1, int(limit))
        normalized_offset = max(0, int(offset))
        normalized_query = (query or "").strip().lower()
        normalized_creator = (creator or "").strip().lower()
        normalized_tags = [
            str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()
        ]
        deleted_false = "FALSE" if self._db.backend_type == BackendType.POSTGRESQL else "0"
        deleted_true = "TRUE" if self._db.backend_type == BackendType.POSTGRESQL else "1"
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
                if self._db.backend_type == BackendType.SQLITE:
                    tag_clauses.append(
                        "("
                        "(json_valid(cc.tags) AND EXISTS ("
                        "SELECT 1 FROM json_each(cc.tags) je "
                        "WHERE LOWER(TRIM(COALESCE(je.value, ''))) = ?"
                        ")) "
                        "OR LOWER(TRIM(COALESCE(cc.tags, ''))) = ? "
                        "OR LOWER(COALESCE(cc.tags, '')) LIKE ?"
                        ")"
                    )
                    params.append(tag)
                    params.append(tag)
                    params.append(f'%"{tag}"%')
                else:
                    tag_clauses.append(
                        "("
                        "LOWER(TRIM(COALESCE(cc.tags, ''))) = ? OR "
                        "LOWER(COALESCE(cc.tags, '')) LIKE ?"
                        ")"
                    )
                    params.append(tag)
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
            if self._db.backend_type == BackendType.SQLITE:
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
            total_cursor = self._db.execute_query(total_query, tuple(params))
            total_row = total_cursor.fetchone()
            if total_row is None:
                total = 0
            elif isinstance(total_row, dict):
                total = int(total_row.get("total", 0))
            else:
                try:
                    total = int(total_row["total"])  # sqlite Row / adapter row
                except _CHACHA_NONCRITICAL_EXCEPTIONS:
                    total = int(total_row[0]) if len(total_row) > 0 else 0

            data_params = list(params)
            data_params.extend([normalized_limit, normalized_offset])
            data_cursor = self._db.execute_query(data_query, tuple(data_params))
            rows = data_cursor.fetchall()
            items = [
                self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
                for row in rows
                if row
            ]
            return items, total
        except CharactersRAGDBError as e:
            logger.error(f"Database error querying character cards: {e}")
            raise

    def query_character_setup_options(
        self,
        *,
        query: str | None = None,
        include_deleted: bool = False,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Query lightweight character selector rows for setup screens.

        The returned rows intentionally omit image BLOBs and large prompt fields.
        ``has_image`` is computed in SQL so callers can render image affordances
        without materializing the stored image bytes.
        """
        normalized_limit = max(1, int(limit))
        normalized_offset = max(0, int(offset))
        normalized_query = (query or "").strip().lower()
        params: list[Any] = []
        filters: list[str] = []
        deleted_false = "FALSE" if self._db.backend_type == BackendType.POSTGRESQL else "0"

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

        deleted_filter = "1=1" if include_deleted else f"cc.deleted = {deleted_false}"
        base_query = f"FROM character_cards cc WHERE {deleted_filter}"  # nosec B608
        if filters:
            base_query += " AND " + " AND ".join(filters)

        total_query = f"SELECT COUNT(1) AS total {base_query}"  # nosec B608
        data_query = (
            "SELECT "
            "cc.id, cc.name, cc.description, cc.tags, cc.extensions, cc.deleted, "
            "CASE WHEN cc.image IS NOT NULL THEN 1 ELSE 0 END AS has_image "
            f"{base_query} "
            "ORDER BY LOWER(COALESCE(cc.name, '')) ASC, cc.id ASC "
            "LIMIT ? OFFSET ?"
        )

        try:
            total_cursor = self._db.execute_query(total_query, tuple(params))
            total_row = total_cursor.fetchone()
            if total_row is None:
                total = 0
            elif isinstance(total_row, dict):
                total = int(total_row.get("total", 0))
            else:
                try:
                    total = int(total_row["total"])  # sqlite Row / adapter row
                except _CHACHA_NONCRITICAL_EXCEPTIONS:
                    total = int(total_row[0]) if len(total_row) > 0 else 0

            data_params = [*params, normalized_limit, normalized_offset]
            data_cursor = self._db.execute_query(data_query, tuple(data_params))
            rows = data_cursor.fetchall()
            return [
                self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
                for row in rows
                if row
            ], total
        except CharactersRAGDBError as e:
            logger.error(f"Database error querying character setup options: {e}")
            raise

    def get_character_setup_option_by_id(
        self,
        character_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """
        Retrieve one lightweight character selector row by ID.

        This mirrors ``get_character_card_by_id`` ownership/deleted semantics
        while omitting image bytes and prompt-heavy columns for setup-option
        callers that only need selector-safe metadata.
        """
        query = (
            "SELECT "
            "id, name, description, tags, extensions, deleted, "
            "CASE WHEN image IS NOT NULL THEN 1 ELSE 0 END AS has_image "
            "FROM character_cards WHERE id = ?"
        )
        params: tuple[Any, ...] = (character_id,)
        if not include_deleted:
            query += f" AND deleted = {self._deleted_literal(False)}"
        try:
            cursor = self._db.execute_query(query, params)
            row = cursor.fetchone()
            return self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching character setup option ID {character_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # Tag normalisation helpers
    # ------------------------------------------------------------------

    def _get_raw_character_tags(self, character_id: int) -> Any:
        """Return the stored tags value without JSON deserialization."""
        cursor = self._db.execute_query(
            "SELECT tags FROM character_cards WHERE id = ?",
            (character_id,),
        )
        row = cursor.fetchone()
        return dict(row).get("tags") if row else None

    @staticmethod
    def _normalize_character_tags_for_operation(tags_value: Any) -> list[str]:
        """Normalize tags and enforce a single reserved character-folder token."""
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
            if tag is None:
                continue
            tag_str = str(tag).strip()
            if not tag_str or tag_str in seen:
                continue
            seen.add(tag_str)
            normalized.append(tag_str)

        # Enforce single-folder assignment semantics for reserved folder tokens.
        # If multiple folder tokens are provided, keep only the most recent one.
        folder_tag: str | None = None
        non_folder_tags: list[str] = []
        for tag in normalized:
            if tag.startswith(_CHARACTER_FOLDER_TAG_PREFIX):
                folder_tag = tag
                continue
            non_folder_tags.append(tag)
        if folder_tag:
            non_folder_tags.append(folder_tag)
        return non_folder_tags

    @staticmethod
    def _apply_character_tag_operation_to_list(
        tags: list[str],
        operation: str,
        source_tag: str,
        target_tag: str | None,
    ) -> list[str]:
        """Apply a rename/merge/delete operation to a normalized tag list."""
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

    # ------------------------------------------------------------------
    # Bulk tag management
    # ------------------------------------------------------------------

    def manage_character_tags(
        self,
        *,
        operation: str,
        source_tag: str,
        target_tag: str | None = None,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        """
        Apply rename/merge/delete tag operations across character cards.

        Returns:
            Summary dictionary with matched/updated/failed counts and affected IDs.
        """
        normalized_operation = str(operation or "").strip().lower()
        if normalized_operation not in {"rename", "merge", "delete"}:
            raise InputError(
                f"Unsupported tag operation '{operation}'. Expected rename, merge, or delete."
            )

        normalized_source = str(source_tag or "").strip()
        normalized_target = str(target_tag or "").strip() if target_tag is not None else None

        if not normalized_source:
            raise InputError("source_tag is required for tag operations")  # noqa: TRY003

        if normalized_operation in {"rename", "merge"} and not normalized_target:
            raise InputError("target_tag is required for rename and merge operations")  # noqa: TRY003

        normalized_limit = max(1, int(limit))
        candidate_cards, _ = self.query_character_cards(
            tags=[normalized_source],
            include_deleted=False,
            limit=normalized_limit,
        )

        matched_count = 0
        updated_character_ids: list[int] = []
        failed_character_ids: list[int] = []

        with self._db.transaction():
            for card in candidate_cards:
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

                tags_value = card.get("tags")
                if tags_value is None:
                    tags_value = self._get_raw_character_tags(card_id)
                existing_tags = self._normalize_character_tags_for_operation(tags_value)
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
                except (ConflictError, InputError, CharactersRAGDBError) as exc:
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

    # ------------------------------------------------------------------
    # Character card update
    # ------------------------------------------------------------------

    def update_character_card(self, character_id: int, card_data: dict[str, Any], expected_version: int) -> bool | None:
        """Update character card with optimistic locking."""
        logger.debug(
            f"Starting update_character_card for ID {character_id}, expected_version {expected_version} (SINGLE UPDATE STRATEGY)")

        # If card_data is empty, treat as a no-op as per original behavior.
        # No version check, no transaction, no version bump.
        if not card_data:
            logger.info(f"No data provided in card_data for character card update ID {character_id}. No-op.")
            return True

        now = self._db._get_current_utc_timestamp_iso()

        try:
            with self._db.transaction() as conn:
                logger.debug(f"Transaction started. Connection object: {id(conn)}")

                # Initial version check. This also confirms the record exists and is not deleted.
                current_db_version_initial_check = self._db._get_current_db_version(
                    conn, "character_cards", "id", character_id,
                )
                logger.debug(
                    f"Initial DB version: {current_db_version_initial_check}, Client expected: {expected_version}")

                if current_db_version_initial_check != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Update failed: version mismatch (db has {current_db_version_initial_check}, "
                        f"client expected {expected_version}) for character_cards ID {character_id}.",
                        entity="character_cards", entity_id=character_id,
                    )

                set_clauses_sql: list[str] = []
                params_for_set_clause: list[Any] = []
                fields_updated_log: list[str] = []  # For logging which fields from payload were processed

                # Define fields that can be directly updated and JSON fields
                updatable_direct_fields = [
                    "name", "description", "personality", "scenario", "image",
                    "post_history_instructions", "first_message", "message_example",
                    "creator_notes", "system_prompt", "creator", "character_version",
                ]

                for key, value in card_data.items():
                    if key in self._db._CHARACTER_CARD_JSON_FIELDS:
                        set_clauses_sql.append(f"{key} = ?")
                        normalized_value = value
                        if key == "tags" and value is not None:
                            normalized_value = self._normalize_character_tags_for_operation(value)
                        # Check if value is already a JSON string
                        if isinstance(normalized_value, str):
                            # Assume it's already a JSON string if it's a string
                            params_for_set_clause.append(normalized_value)
                        else:
                            params_for_set_clause.append(self._db._ensure_json_string(normalized_value))
                        fields_updated_log.append(key)
                    elif key in updatable_direct_fields:
                        set_clauses_sql.append(f"{key} = ?")
                        params_for_set_clause.append(value)
                        fields_updated_log.append(key)
                    elif key not in ['id', 'created_at', 'last_modified', 'version', 'client_id', 'deleted']:
                        # Log if a key in card_data is not recognized as updatable, but don't error.
                        logger.warning(
                            f"Skipping unknown or non-updatable field '{key}' in update_character_card payload.")

                next_version_val = expected_version + 1

                # Add metadata fields to be updated
                set_clauses_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])
                params_for_set_clause.extend([now, next_version_val, self._db.client_id])

                # Construct the final query
                final_update_query = (
                    f"UPDATE character_cards SET {', '.join(set_clauses_sql)} "  # nosec B608
                    f"WHERE id = ? AND version = ? AND deleted = {self._deleted_literal(False)}"
                )

                # WHERE clause parameters
                where_params: list[Any] = [character_id, expected_version]
                final_params = tuple(params_for_set_clause + where_params)

                logger.debug("Executing SINGLE character update query: {}", final_update_query)
                logger.debug("Character update parameter count: {}", len(final_params))

                cursor = conn.execute(final_update_query, final_params)
                logger.debug(f"Character Update executed, rowcount: {cursor.rowcount}")

                if cursor.rowcount == 0:
                    # Re-check the record's state to provide a more specific error.
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = f"Update for character_cards ID {character_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = (
                            f"Character card ID {character_id} disappeared before update "
                            f"completion (expected v{expected_version})."
                        )
                    elif final_state['deleted']:
                        msg = (
                            f"Character card ID {character_id} was soft-deleted concurrently "
                            f"(expected v{expected_version} for update)."
                        )
                    elif final_state['version'] != expected_version:
                        msg = (
                            f"Character card ID {character_id} version changed to "
                            f"{final_state['version']} concurrently "
                            f"(expected v{expected_version} for update's WHERE clause)."
                        )
                    else:
                        msg = (
                            f"Update for character card ID {character_id} "
                            f"(expected v{expected_version}) affected 0 rows for an unknown "
                            "reason after passing initial checks."
                        )
                    raise ConflictError(msg, entity="character_cards", entity_id=character_id)  # noqa: TRY301

                log_msg_fields_updated = (
                    f"Fields from payload processed: "
                    f"{fields_updated_log if fields_updated_log else 'None'}."
                )
                logger.info(
                    f"Updated character card ID {character_id} (SINGLE UPDATE) from "
                    f"client-expected version {expected_version} to final DB version "
                    f"{next_version_val}. {log_msg_fields_updated}"
                )
                return True

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: character_cards.name" in str(e):
                updated_name = card_data.get("name", "[name not in update_data]")
                logger.warning(
                    f"Update for character card ID {character_id} failed: name '{updated_name}' already exists."
                )
                raise ConflictError(  # noqa: TRY003
                    f"Cannot update character card ID {character_id}: name '{updated_name}' already exists.",
                    entity="character_cards", entity_id=updated_name,
                ) from e
            logger.critical(
                f"DATABASE IntegrityError during update_character_card (SINGLE UPDATE STRATEGY) "
                f"for ID {character_id}: {e}",
                exc_info=True,
            )
            raise CharactersRAGDBError(f"Database integrity error during single update: {e}") from e  # noqa: TRY003
        except sqlite3.DatabaseError as e:
            logger.critical(
                f"DATABASE ERROR during update_character_card (SINGLE UPDATE STRATEGY) "
                f"for ID {character_id}: {e}",
                exc_info=True,
            )
            raise CharactersRAGDBError(f"Database error during single update: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            if self._db._is_unique_violation(e):
                updated_name = card_data.get("name", "[name not in update_data]")
                logger.warning(
                    "Update for character card ID {} failed on backend {}: name '{}' already exists.",
                    character_id,
                    self._db.backend_type.value,
                    updated_name,
                )
                raise ConflictError(  # noqa: TRY003
                    f"Cannot update character card ID {character_id}: name '{updated_name}' already exists.",
                    entity="character_cards",
                    entity_id=updated_name,
                ) from e
            logger.critical(
                'Backend error during update_character_card (SINGLE UPDATE STRATEGY) for ID {}: {}',
                character_id,
                e,
                exc_info=True,
            )
            raise CharactersRAGDBError(f"Database error during single update: {e}") from e  # noqa: TRY003
        except ConflictError:
            logger.warning(
                f"ConflictError during update_character_card for ID {character_id}.",
                exc_info=False,
            )
            raise
        except InputError:
            logger.warning(
                f"InputError during update_character_card for ID {character_id}.",
                exc_info=False,
            )
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.error(
                f"Unexpected Python error in update_character_card (SINGLE UPDATE STRATEGY) "
                f"for ID {character_id}: {e}",
                exc_info=True,
            )
            raise CharactersRAGDBError(f"Unexpected error updating character card: {e}") from e  # noqa: TRY003

    # ------------------------------------------------------------------
    # Soft delete / restore
    # ------------------------------------------------------------------

    def soft_delete_character_card(self, character_id: int, expected_version: int) -> bool | None:
        """
        Soft-deletes a character card using optimistic locking.

        Sets the ``deleted`` flag to 1, updates ``last_modified``, increments
        ``version``, and sets ``client_id``.  The operation succeeds only if
        ``expected_version`` matches the current database version and the card
        is not already deleted.

        If the card is already soft-deleted (idempotency check), the method
        considers this a success and returns True.

        FTS updates (removal from ``character_cards_fts``) and ``sync_log``
        entries for deletions are handled by SQL triggers.

        Args:
            character_id: The ID of the character card to soft-delete.
            expected_version: The version number the client expects the record
                              to have.

        Returns:
            True if the soft-delete was successful or if the card was already
            soft-deleted.

        Raises:
            ConflictError: If the card is not found (and not already deleted),
                           or if ``expected_version`` does not match, or if a
                           concurrent modification prevents the update.
            CharactersRAGDBError: For other database-related errors.
        """
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = (
            "UPDATE character_cards SET deleted = ?, last_modified = ?, version = ?, "
            "client_id = ? WHERE id = ? AND version = ? AND deleted = ?"
        )
        params = (
            self._deleted_value(True),
            now,
            next_version_val,
            self._db.client_id,
            character_id,
            expected_version,
            self._deleted_value(False),
        )

        try:
            with self._db.transaction() as conn:
                try:
                    current_db_version = self._db._get_current_db_version(
                        conn, "character_cards", "id", character_id,
                    )
                    # If here, record is active.
                except ConflictError:
                    # Check if ConflictError was because it's ALREADY soft-deleted.
                    check_status_cursor = conn.execute(
                        "SELECT deleted, version FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(
                            f"Character card ID {character_id} already soft-deleted. "
                            "Soft delete successful (idempotent)."
                        )
                        return True
                    # If not found, or some other conflict, re-raise.
                    raise

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Soft delete for Character ID {character_id} failed: version mismatch "
                        f"(db has {current_db_version}, client expected {expected_version}).",
                        entity="character_cards", entity_id=character_id,
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    # Race condition: Record changed between pre-check and UPDATE.
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = (
                        f"Soft delete for Character ID {character_id} "
                        f"(expected v{expected_version}) affected 0 rows."
                    )
                    if not final_state:
                        msg = (
                            f"Character card ID {character_id} disappeared before soft delete "
                            f"(expected active version {expected_version})."
                        )
                    elif final_state['deleted']:
                        logger.info(
                            f"Character card ID {character_id} was soft-deleted concurrently "
                            f"to version {final_state['version']}. Soft delete successful."
                        )
                        return True
                    elif final_state['version'] != expected_version:
                        msg = (
                            f"Soft delete for Character ID {character_id} failed: version "
                            f"changed to {final_state['version']} concurrently "
                            f"(expected {expected_version})."
                        )
                    else:
                        msg = (
                            f"Soft delete for Character ID {character_id} "
                            f"(expected version {expected_version}) affected 0 rows for an "
                            "unknown reason after passing initial checks."
                        )
                    raise ConflictError(msg, entity="character_cards", entity_id=character_id)  # noqa: TRY301

                logger.info(
                    f"Soft-deleted character card ID {character_id} "
                    f"(was version {expected_version}), new version {next_version_val}."
                )
                return True
        except ConflictError:
            raise
        except BackendDatabaseError as e:
            logger.error(
                'Backend error soft-deleting character card ID {} (expected v{}): {}',
                character_id,
                expected_version,
                e,
            )
            raise CharactersRAGDBError(f"Backend error during soft delete: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError as e:
            logger.error(
                f"Database error soft-deleting character card ID {character_id} "
                f"(expected v{expected_version}): {e}",
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
        """
        Restores a soft-deleted character card using optimistic locking.

        Sets the ``deleted`` flag to 0, updates ``last_modified``, increments
        ``version``, and sets ``client_id``.  The operation succeeds only if
        ``expected_version`` matches the current database version and the card
        is currently deleted.

        If the card is already active (not deleted), the method raises a
        ``ConflictError``.

        Args:
            character_id: The ID of the character card to restore.
            expected_version: The version number the client expects the record
                              to have.
            retention_days: If provided, the restore will be rejected when the
                            soft-delete timestamp is older than this many days.

        Returns:
            True if the restore was successful.

        Raises:
            ConflictError: If the card is not found, is already active, or if
                           ``expected_version`` does not match, or if a
                           concurrent modification prevents the update.
            RestoreWindowExpiredError: If ``retention_days`` is set and the
                                      soft-delete timestamp exceeds the window.
            CharactersRAGDBError: For other database-related errors.
        """
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = (
            "UPDATE character_cards SET deleted = ?, last_modified = ?, version = ?, "
            "client_id = ? WHERE id = ? AND version = ? AND deleted = ?"
        )
        params = (
            self._deleted_value(False),
            now,
            next_version_val,
            self._db.client_id,
            character_id,
            expected_version,
            self._deleted_value(True),
        )

        try:
            with self._db.transaction() as conn:
                # First check if record exists at all
                check_cursor = conn.execute(
                    "SELECT deleted, version, last_modified FROM character_cards WHERE id = ?",
                    (character_id,),
                )
                record_status = check_cursor.fetchone()

                if not record_status:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Character card ID {character_id} not found.",
                        entity="character_cards", entity_id=character_id,
                    )

                # Restoring an active character is a conflict, not a no-op.
                if not record_status['deleted']:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Character card ID {character_id} is already active; restore cannot succeed.",
                        entity="character_cards",
                        entity_id=character_id,
                    )

                # Check version matches
                current_db_version = record_status['version']
                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Restore for Character ID {character_id} failed: version mismatch "
                        f"(db has {current_db_version}, client expected {expected_version}).",
                        entity="character_cards", entity_id=character_id,
                    )

                if retention_days is not None:
                    if retention_days < 0:
                        raise InputError(  # noqa: TRY301
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
                                    deleted_at_dt = datetime.strptime(normalized, fmt)  # noqa: DTZ007
                                    break
                                except ValueError:
                                    continue

                    if deleted_at_dt is None:
                        raise CharactersRAGDBError(  # noqa: TRY301
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
                        deleted_at_iso = (
                            deleted_at_dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                        )
                        restore_expires_at_iso = (
                            restore_expires_at_dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                        )
                        raise RestoreWindowExpiredError(  # noqa: TRY301
                            character_id=character_id,
                            retention_days=retention_days,
                            deleted_at_iso=deleted_at_iso,
                            restore_expires_at_iso=restore_expires_at_iso,
                        )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    # Race condition: Record changed between pre-check and UPDATE.
                    check_again_cursor = conn.execute(
                        "SELECT version, deleted FROM character_cards WHERE id = ?",
                        (character_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = (
                        f"Restore for Character ID {character_id} "
                        f"(expected v{expected_version}) affected 0 rows."
                    )
                    if not final_state:
                        msg = (
                            f"Character card ID {character_id} disappeared before restore "
                            f"(expected deleted version {expected_version})."
                        )
                    elif not final_state['deleted']:
                        msg = (
                            f"Character card ID {character_id} is already active; "
                            f"restore cannot succeed (concurrent restore detected, "
                            f"current version {final_state['version']})."
                        )
                    elif final_state['version'] != expected_version:
                        msg = (
                            f"Restore for Character ID {character_id} failed: version "
                            f"changed to {final_state['version']} concurrently "
                            f"(expected {expected_version})."
                        )
                    else:
                        msg = (
                            f"Restore for Character ID {character_id} "
                            f"(expected version {expected_version}) affected 0 rows for an "
                            "unknown reason after passing initial checks."
                        )
                    raise ConflictError(msg, entity="character_cards", entity_id=character_id)  # noqa: TRY301

                logger.info(
                    f"Restored character card ID {character_id} "
                    f"(was version {expected_version}), new version {next_version_val}."
                )
                return True
        except ConflictError:
            raise
        except BackendDatabaseError as e:
            logger.error(
                'Backend error restoring character card ID {} (expected v{}): {}',
                character_id,
                expected_version,
                e,
            )
            raise CharactersRAGDBError(f"Backend error during restore: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError as e:
            logger.error(
                f"Database error restoring character card ID {character_id} "
                f"(expected v{expected_version}): {e}",
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Full-text and tag search
    # ------------------------------------------------------------------

    def search_character_cards(self, search_term: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Searches character cards using Full-Text Search (FTS).

        The search is performed on the ``character_cards_fts`` table, matching
        against 'name', 'description', 'personality', 'scenario', and
        'system_prompt' fields.  Returns full card details for matching,
        non-deleted cards, ordered by relevance (rank).  JSON fields (see
        ``_CHARACTER_CARD_JSON_FIELDS``) in the results are deserialized.

        Args:
            search_term: The term(s) to search for.  Supports FTS query syntax.
            limit: The maximum number of results to return.  Defaults to 10.

        Returns:
            A list of dictionaries, each representing a matching character card.

        Raises:
            CharactersRAGDBError: For database errors during the search.
        """
        if not search_term.strip():
            logger.warning("Empty character card search term provided; returning no results.")
            return []

        if self._db.backend_type == BackendType.POSTGRESQL:
            tsquery = FTSQueryTranslator.normalize_query(search_term, 'postgresql')
            if not tsquery:
                logger.debug("FTS normalization produced empty tsquery for input '{}'", search_term)
                return []

            query = """
                SELECT cc.*, ts_rank(cc.character_cards_fts_tsv, to_tsquery('english', ?)) AS rank
                FROM character_cards cc
                WHERE cc.deleted = FALSE
                  AND cc.character_cards_fts_tsv @@ to_tsquery('english', ?)
                ORDER BY rank DESC, cc.last_modified DESC
                LIMIT ?
            """
            try:
                cursor = self._db.execute_query(query, (tsquery, tsquery, limit))
                rows = cursor.fetchall()
                return [
                    self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
                    for row in rows
                    if row
                ]
            except CharactersRAGDBError as exc:
                logger.error(
                    "PostgreSQL FTS search failed for character cards term '{}': {}",
                    search_term,
                    exc,
                )
                raise

        # Escape embedded quotes to avoid breaking the literal phrase wrapper
        safe_literal = search_term.replace('"', '""')
        safe_search_term = f'"{safe_literal}"' if '"' in search_term else safe_literal
        deleted_false = self._deleted_literal(False)
        query = """
                SELECT cc.*
                FROM character_cards_fts, character_cards cc
                WHERE character_cards_fts.rowid = cc.id
                  AND character_cards_fts MATCH ?
                  AND cc.deleted = {deleted_false}
                ORDER BY cc.last_modified DESC
                LIMIT ?
                """.format_map(locals())  # nosec B608
        try:
            cursor = self._db.execute_query(query, (safe_search_term, limit))
            rows = cursor.fetchall()
            return [
                self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
                for row in rows
                if row
            ]
        except CharactersRAGDBError as e:
            logger.error("Error searching character cards for '{}': {}", safe_search_term, e)
            raise

    def search_character_cards_by_tags(self, tag_keywords: list[str], limit: int = 10) -> list[dict[str, Any]]:
        """
        Search character cards efficiently by their tags using database-level filtering.

        Uses SQLite JSON functions when available, or falls back to a normalized
        tag approach if JSON functions are not supported.

        Args:
            tag_keywords: List of tag strings to search for (case-insensitive).
            limit: Maximum number of results to return.  Defaults to 10.

        Returns:
            List of character card dictionaries that contain any of the
            specified tags.  Results are ordered by name and limited to
            non-deleted cards.

        Raises:
            CharactersRAGDBError: For database errors during the search.
            InputError: If tag_keywords is empty or contains invalid values.
        """
        if not tag_keywords:
            raise InputError("tag_keywords cannot be empty")  # noqa: TRY003

        # Normalize tag keywords for case-insensitive matching
        normalized_tags = [tag.lower().strip() for tag in tag_keywords if tag.strip()]
        if not normalized_tags:
            raise InputError("No valid tag keywords provided after normalization")  # noqa: TRY003

        logger.debug(f"Searching character cards by tags: {normalized_tags}")

        # Check if SQLite supports JSON functions
        if self._check_json_support():
            try:
                return self._search_cards_by_tags_json(normalized_tags, limit)
            except CharactersRAGDBError as exc:
                logger.warning(
                    "SQLite JSON tag search failed; falling back to Python tag filtering: {}",
                    exc,
                )

        # Fallback to loading and filtering in Python (original approach but optimized).
        logger.warning("SQLite JSON functions not available, using fallback tag search method")
        return self._search_cards_by_tags_fallback(normalized_tags, limit)

    # ------------------------------------------------------------------
    # Private helpers for tag search
    # ------------------------------------------------------------------

    def _check_json_support(self) -> bool:
        """
        Check if the current SQLite version supports JSON functions.

        Returns:
            True if JSON functions are available, False otherwise.
        """
        if self._db.backend_type != BackendType.SQLITE:
            return False

        try:
            cursor = self._db.execute_query("SELECT json('{}') as test")
            cursor.fetchone()
            return True  # noqa: TRY300
        except (sqlite3.OperationalError, CharactersRAGDBError):
            return False

    def _search_cards_by_tags_json(self, normalized_tags: list[str], limit: int) -> list[dict[str, Any]]:
        """
        Search character cards by tags using SQLite JSON functions.

        This is the optimal approach for SQLite versions that support JSON functions.
        """
        try:
            # Build query with JSON_EACH to extract and check tags
            placeholders = ','.join('?' for _ in normalized_tags)
            fallback_like_clauses = " OR ".join("LOWER(COALESCE(cc.tags, '')) LIKE ?" for _ in normalized_tags)
            deleted_false = self._deleted_literal(False)
            query = """
                SELECT DISTINCT cc.*
                FROM character_cards cc
                WHERE cc.deleted = {deleted_false}
                  AND cc.tags IS NOT NULL
                  AND cc.tags != 'null'
                  AND (
                      (
                          json_valid(cc.tags)
                          AND EXISTS (
                              SELECT 1
                              FROM json_each(cc.tags) je
                              WHERE lower(trim(je.value)) IN ({placeholders})
                          )
                      )
                      OR {fallback_like_clauses}
                  )
                ORDER BY cc.name
                LIMIT ?
            """.format_map(locals())  # nosec B608

            params: list[Any] = normalized_tags + [f"%{tag}%" for tag in normalized_tags] + [limit]
            cursor = self._db.execute_query(query, params)
            rows = cursor.fetchall()

            result = [
                self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)
                for row in rows
                if row
            ]
            logger.debug(f"Found {len(result)} character cards matching tags using JSON functions")
            return result  # noqa: TRY300

        except CharactersRAGDBError as e:
            logger.error(f"Database error in JSON-based tag search: {e}")
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            # JSON function might have failed, log and re-raise as database error
            logger.error(f"Unexpected error in JSON-based tag search: {e}")
            raise CharactersRAGDBError(f"JSON tag search failed: {e}") from e  # noqa: TRY003

    def _search_cards_by_tags_fallback(self, normalized_tags: list[str], limit: int) -> list[dict[str, Any]]:
        """
        Fallback tag search that loads cards and filters in Python.

        Used when SQLite doesn't support JSON functions, but optimized to only
        load necessary data and exit early when limit is reached.
        """
        try:
            # Use a reasonable batch size to avoid loading everything at once
            batch_size = min(1000, limit * 10)  # Load 10x limit as heuristic
            offset = 0
            results: list[dict[str, Any]] = []
            normalized_tags_set = set(normalized_tags)

            while len(results) < limit:
                # Load cards in batches
                query = "SELECT * FROM character_cards WHERE deleted = ? ORDER BY name LIMIT ? OFFSET ?"
                cursor = self._db.execute_query(query, (self._deleted_value(False), batch_size, offset))
                batch_rows = cursor.fetchall()

                if not batch_rows:
                    break  # No more cards to process

                # Process this batch
                for row in batch_rows:
                    if len(results) >= limit:
                        break

                    card = self._db._deserialize_row_fields(row, self._db._CHARACTER_CARD_JSON_FIELDS)

                    # Check if card has matching tags
                    tags_data = card.get('tags') if card else None
                    if tags_data:
                        try:
                            # Handle both cases: already deserialized list or JSON string
                            if isinstance(tags_data, list):
                                tags_list = tags_data
                            elif isinstance(tags_data, str):
                                tags_list = json.loads(tags_data)
                            else:
                                tags_list = []

                            if isinstance(tags_list, list):
                                card_tags_normalized = {str(tag).lower().strip() for tag in tags_list}
                                # Check for intersection with our target tags
                                if not card_tags_normalized.isdisjoint(normalized_tags_set):
                                    results.append(card)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Invalid JSON in tags for character card ID {card.get('id') if card else '?'}: "
                                f"{tags_data}"
                            )
                            continue

                offset += batch_size

                # If we got fewer rows than batch_size, we've reached the end
                if len(batch_rows) < batch_size:
                    break

            logger.debug(f"Found {len(results)} character cards matching tags using fallback method")
            return results  # noqa: TRY300

        except CharactersRAGDBError as e:
            logger.error(f"Database error in fallback tag search: {e}")
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Unexpected error in fallback tag search: {e}")
            raise CharactersRAGDBError(f"Fallback tag search failed: {e}") from e  # noqa: TRY003

    # --- Character Exemplar Methods ---
    @staticmethod
    def _estimate_text_token_count(text: str) -> int:
        """Estimate token count from text with a lightweight fallback heuristic."""
        if not text:
            return 1
        return max(1, len(text.split()))

    def _normalize_exemplar_enum(
        self,
        value: Any,
        *,
        allowed: tuple[str, ...],
        field_name: str,
        default: str,
    ) -> str:
        """Normalize and validate enum-like exemplar fields."""
        return exemplar_normalization.normalize_exemplar_enum(
            value,
            allowed=allowed,
            field_name=field_name,
            default=default,
        )

    def _normalize_exemplar_string_list(self, value: Any, field_name: str) -> list[str]:
        """Normalize list-like exemplar metadata to a string list."""
        return exemplar_normalization.normalize_exemplar_string_list(value, field_name)

    def _normalize_character_exemplar_row(self, row: sqlite3.Row | dict[str, Any] | None) -> dict[str, Any] | None:
        """Deserialize exemplar JSON fields and normalize bool-like values."""
        if not row:
            return None
        item = self._db._deserialize_row_fields(row, self._db._CHARACTER_EXEMPLAR_JSON_FIELDS)
        if not item:
            return None
        if 'rights_public_figure' in item:
            item['rights_public_figure'] = bool(item['rights_public_figure'])
        if 'is_deleted' in item:
            item['is_deleted'] = bool(item['is_deleted'])
        return item

    def add_character_exemplar(self, character_id: int, exemplar_data: dict[str, Any]) -> dict[str, Any]:
        """Create a character-scoped exemplar."""
        if not self.get_character_card_by_id(character_id):
            raise InputError(f"Character ID {character_id} not found.")  # noqa: TRY003

        text = self._db._normalize_nullable_text(exemplar_data.get('text'))
        if not text:
            raise InputError("Exemplar text is required.")  # noqa: TRY003

        source_type = self._normalize_exemplar_enum(
            exemplar_data.get('source_type'),
            allowed=self._db._ALLOWED_EXEMPLAR_SOURCE_TYPES,
            field_name='source_type',
            default='other',
        )
        novelty_hint = self._normalize_exemplar_enum(
            exemplar_data.get('novelty_hint'),
            allowed=self._db._ALLOWED_EXEMPLAR_NOVELTY_HINTS,
            field_name='novelty_hint',
            default='unknown',
        )
        emotion = self._normalize_exemplar_enum(
            exemplar_data.get('emotion'),
            allowed=self._db._ALLOWED_EXEMPLAR_EMOTIONS,
            field_name='emotion',
            default='other',
        )
        scenario = self._normalize_exemplar_enum(
            exemplar_data.get('scenario'),
            allowed=self._db._ALLOWED_EXEMPLAR_SCENARIOS,
            field_name='scenario',
            default='other',
        )

        rhetorical = self._normalize_exemplar_string_list(exemplar_data.get('rhetorical'), 'rhetorical')
        safety_allowed = self._normalize_exemplar_string_list(exemplar_data.get('safety_allowed'), 'safety_allowed')
        safety_blocked = self._normalize_exemplar_string_list(exemplar_data.get('safety_blocked'), 'safety_blocked')

        requested_length = exemplar_data.get('length_tokens')
        if requested_length is None:
            length_tokens = self._estimate_text_token_count(text)
        else:
            try:
                length_tokens = int(requested_length)
            except (TypeError, ValueError) as exc:
                raise InputError("length_tokens must be an integer >= 1.") from exc  # noqa: TRY003
            if length_tokens < 1:
                raise InputError("length_tokens must be >= 1.")  # noqa: TRY003

        exemplar_id = self._db._normalize_nullable_text(exemplar_data.get('id')) or self._db._generate_uuid()
        now = self._db._get_current_utc_timestamp_iso()

        query = """
            INSERT INTO character_exemplars (
                id, character_id, text, source_type, source_url_or_id, source_date,
                novelty_hint, emotion, scenario, rhetorical, register, safety_allowed,
                safety_blocked, rights_public_figure, rights_notes, length_tokens,
                created_at, updated_at, is_deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        if self._db.backend_type == BackendType.POSTGRESQL:
            rights_public_figure = bool(exemplar_data.get('rights_public_figure', True))
        else:
            rights_public_figure = 1 if exemplar_data.get('rights_public_figure', True) else 0
        is_deleted = self._deleted_value(False)

        params = (
            exemplar_id,
            character_id,
            text,
            source_type,
            self._db._normalize_nullable_text(exemplar_data.get('source_url_or_id')),
            self._db._normalize_nullable_text(exemplar_data.get('source_date')),
            novelty_hint,
            emotion,
            scenario,
            self._db._ensure_json_string(rhetorical) or "[]",
            self._db._normalize_nullable_text(exemplar_data.get('register')),
            self._db._ensure_json_string(safety_allowed) or "[]",
            self._db._ensure_json_string(safety_blocked) or "[]",
            rights_public_figure,
            self._db._normalize_nullable_text(exemplar_data.get('rights_notes')),
            length_tokens,
            now,
            now,
            is_deleted,
        )

        try:
            with self._db.transaction() as conn:
                prepared_query, prepared_params = self._db._prepare_backend_statement(query, params)
                conn.execute(prepared_query, prepared_params)
        except sqlite3.IntegrityError as exc:
            msg = str(exc).lower()
            if "unique constraint failed: character_exemplars.id" in msg:
                raise ConflictError(  # noqa: TRY003
                    f"Character exemplar with ID '{exemplar_id}' already exists.",
                    entity="character_exemplars",
                    entity_id=exemplar_id,
                ) from exc
            raise CharactersRAGDBError(f"Database integrity error adding character exemplar: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._db._is_unique_violation(exc):
                raise ConflictError(  # noqa: TRY003
                    f"Character exemplar with ID '{exemplar_id}' already exists.",
                    entity="character_exemplars",
                    entity_id=exemplar_id,
                ) from exc
            raise CharactersRAGDBError(f"Backend error adding character exemplar: {exc}") from exc  # noqa: TRY003

        created = self.get_character_exemplar_by_id(character_id, exemplar_id)
        if not created:
            raise CharactersRAGDBError("Created character exemplar could not be retrieved.")  # noqa: TRY003
        return created

    def get_character_exemplar_by_id(
        self,
        character_id: int,
        exemplar_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a character exemplar by ID."""
        query = """
            SELECT *
            FROM character_exemplars
            WHERE id = ? AND character_id = ?
        """
        params: list[Any] = [exemplar_id, character_id]
        if not include_deleted:
            query += " AND is_deleted = ?"
            params.append(self._deleted_value(False))
        query += " LIMIT 1"
        try:
            cursor = self._db.execute_query(query, tuple(params))
            row = cursor.fetchone()
            return self._normalize_character_exemplar_row(row)
        except CharactersRAGDBError:
            raise

    def list_character_exemplars(self, character_id: int, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List non-deleted exemplars for a character."""
        query = """
            SELECT *
            FROM character_exemplars
            WHERE character_id = ? AND is_deleted = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        try:
            cursor = self._db.execute_query(
                query,
                (character_id, self._deleted_value(False), limit, offset),
            )
            rows = cursor.fetchall()
            return [self._normalize_character_exemplar_row(row) for row in rows if row]
        except CharactersRAGDBError:
            raise

    def update_character_exemplar(
        self,
        character_id: int,
        exemplar_id: str,
        update_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update mutable fields for a character exemplar."""
        existing = self.get_character_exemplar_by_id(character_id, exemplar_id)
        if not existing:
            return None

        if not update_data:
            return existing

        set_clauses: list[str] = []
        params: list[Any] = []

        if 'text' in update_data:
            text = self._db._normalize_nullable_text(update_data.get('text'))
            if not text:
                raise InputError("Exemplar text cannot be empty.")  # noqa: TRY003
            set_clauses.append("text = ?")
            params.append(text)
            if 'length_tokens' not in update_data:
                set_clauses.append("length_tokens = ?")
                params.append(self._estimate_text_token_count(text))

        if 'source_type' in update_data:
            source_type = self._normalize_exemplar_enum(
                update_data.get('source_type'),
                allowed=self._db._ALLOWED_EXEMPLAR_SOURCE_TYPES,
                field_name='source_type',
                default='other',
            )
            set_clauses.append("source_type = ?")
            params.append(source_type)

        if 'source_url_or_id' in update_data:
            set_clauses.append("source_url_or_id = ?")
            params.append(self._db._normalize_nullable_text(update_data.get('source_url_or_id')))

        if 'source_date' in update_data:
            set_clauses.append("source_date = ?")
            params.append(self._db._normalize_nullable_text(update_data.get('source_date')))

        if 'novelty_hint' in update_data:
            novelty_hint = self._normalize_exemplar_enum(
                update_data.get('novelty_hint'),
                allowed=self._db._ALLOWED_EXEMPLAR_NOVELTY_HINTS,
                field_name='novelty_hint',
                default='unknown',
            )
            set_clauses.append("novelty_hint = ?")
            params.append(novelty_hint)

        if 'emotion' in update_data:
            emotion = self._normalize_exemplar_enum(
                update_data.get('emotion'),
                allowed=self._db._ALLOWED_EXEMPLAR_EMOTIONS,
                field_name='emotion',
                default='other',
            )
            set_clauses.append("emotion = ?")
            params.append(emotion)

        if 'scenario' in update_data:
            scenario = self._normalize_exemplar_enum(
                update_data.get('scenario'),
                allowed=self._db._ALLOWED_EXEMPLAR_SCENARIOS,
                field_name='scenario',
                default='other',
            )
            set_clauses.append("scenario = ?")
            params.append(scenario)

        if 'rhetorical' in update_data:
            rhetorical = self._normalize_exemplar_string_list(update_data.get('rhetorical'), 'rhetorical')
            set_clauses.append("rhetorical = ?")
            params.append(self._db._ensure_json_string(rhetorical) or "[]")

        if 'register' in update_data:
            set_clauses.append("register = ?")
            params.append(self._db._normalize_nullable_text(update_data.get('register')))

        if 'safety_allowed' in update_data:
            safety_allowed = self._normalize_exemplar_string_list(update_data.get('safety_allowed'), 'safety_allowed')
            set_clauses.append("safety_allowed = ?")
            params.append(self._db._ensure_json_string(safety_allowed) or "[]")

        if 'safety_blocked' in update_data:
            safety_blocked = self._normalize_exemplar_string_list(update_data.get('safety_blocked'), 'safety_blocked')
            set_clauses.append("safety_blocked = ?")
            params.append(self._db._ensure_json_string(safety_blocked) or "[]")

        if 'rights_public_figure' in update_data:
            set_clauses.append("rights_public_figure = ?")
            if self._db.backend_type == BackendType.POSTGRESQL:
                params.append(bool(update_data.get('rights_public_figure')))
            else:
                params.append(1 if update_data.get('rights_public_figure') else 0)

        if 'rights_notes' in update_data:
            set_clauses.append("rights_notes = ?")
            params.append(self._db._normalize_nullable_text(update_data.get('rights_notes')))

        if 'length_tokens' in update_data:
            try:
                length_tokens = int(update_data.get('length_tokens'))
            except (TypeError, ValueError) as exc:
                raise InputError("length_tokens must be an integer >= 1.") from exc  # noqa: TRY003
            if length_tokens < 1:
                raise InputError("length_tokens must be >= 1.")  # noqa: TRY003
            set_clauses.append("length_tokens = ?")
            params.append(length_tokens)

        if not set_clauses:
            return existing

        set_clauses.append("updated_at = ?")
        params.append(self._db._get_current_utc_timestamp_iso())

        set_clause_sql = ", ".join(set_clauses)
        query = (
            "UPDATE character_exemplars "
            + "SET "
            + set_clause_sql
            + " WHERE id = ? AND character_id = ? AND is_deleted = ?"
        )  # nosec B608
        params.extend(
            [
                exemplar_id,
                character_id,
                self._deleted_value(False),
            ]
        )
        try:
            with self._db.transaction() as conn:
                prepared_query, prepared_params = self._db._prepare_backend_statement(query, tuple(params))
                cursor = conn.cursor()
                cursor.execute(prepared_query, prepared_params)
                if cursor.rowcount == 0:
                    return None
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Error updating character exemplar '{exemplar_id}': {exc}") from exc  # noqa: TRY003

        return self.get_character_exemplar_by_id(character_id, exemplar_id)

    def soft_delete_character_exemplar(self, character_id: int, exemplar_id: str) -> bool:
        """Soft-delete a character exemplar (idempotent)."""
        query = """
            UPDATE character_exemplars
            SET is_deleted = ?, updated_at = ?
            WHERE id = ? AND character_id = ? AND is_deleted = ?
        """
        now = self._db._get_current_utc_timestamp_iso()
        set_deleted = self._deleted_value(True)
        active_flag = self._deleted_value(False)

        try:
            with self._db.transaction() as conn:
                prepared_query, prepared_params = self._db._prepare_backend_statement(
                    query,
                    (set_deleted, now, exemplar_id, character_id, active_flag),
                )
                cursor = conn.cursor()
                cursor.execute(prepared_query, prepared_params)
                if cursor.rowcount > 0:
                    return True

                check_query, check_params = self._db._prepare_backend_statement(
                    "SELECT is_deleted FROM character_exemplars WHERE id = ? AND character_id = ?",
                    (exemplar_id, character_id),
                )
                check_row = conn.execute(check_query, check_params).fetchone()
                return bool(check_row and bool(check_row['is_deleted']))  # noqa: TRY300
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Error deleting character exemplar '{exemplar_id}': {exc}") from exc  # noqa: TRY003

    def search_character_exemplars(
        self,
        character_id: int,
        *,
        query: str | None = None,
        emotion: str | None = None,
        scenario: str | None = None,
        rhetorical: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Search exemplars for a character via FTS + structured filters."""
        emotion_filter = None
        if emotion is not None:
            emotion_filter = self._normalize_exemplar_enum(
                emotion,
                allowed=self._db._ALLOWED_EXEMPLAR_EMOTIONS,
                field_name='emotion',
                default='other',
            )

        scenario_filter = None
        if scenario is not None:
            scenario_filter = self._normalize_exemplar_enum(
                scenario,
                allowed=self._db._ALLOWED_EXEMPLAR_SCENARIOS,
                field_name='scenario',
                default='other',
            )

        rhetorical_filter = {
            item.strip().lower()
            for item in (rhetorical or [])
            if isinstance(item, str) and item.strip()
        }

        filter_params: list[Any] = [
            character_id,
            self._deleted_value(False),
            emotion_filter,
            emotion_filter,
            scenario_filter,
            scenario_filter,
        ]
        normalized_query = (query or "").strip()

        if normalized_query and self._db.backend_type == BackendType.POSTGRESQL:
            tsquery = FTSQueryTranslator.normalize_query(normalized_query, 'postgresql')
            if tsquery:
                sql = """
                    SELECT ce.*, ts_rank(ce.character_exemplars_fts_tsv, to_tsquery('english', ?)) AS rank
                    FROM character_exemplars ce
                    WHERE ce.character_id = ?
                      AND ce.is_deleted = ?
                      AND (? IS NULL OR ce.emotion = ?)
                      AND (? IS NULL OR ce.scenario = ?)
                      AND ce.character_exemplars_fts_tsv @@ to_tsquery('english', ?)
                    ORDER BY rank DESC, ce.updated_at DESC
                """
                query_params = [tsquery] + filter_params + [tsquery]
            else:
                sql = """
                    SELECT ce.*
                    FROM character_exemplars ce
                    WHERE ce.character_id = ?
                      AND ce.is_deleted = ?
                      AND (? IS NULL OR ce.emotion = ?)
                      AND (? IS NULL OR ce.scenario = ?)
                      AND ce.text ILIKE ?
                    ORDER BY ce.updated_at DESC
                """
                query_params = filter_params + [f"%{normalized_query}%"]
        elif normalized_query:
            safe_literal = normalized_query.replace('"', '""')
            safe_fts = f'"{safe_literal}"'
            sql = """
                SELECT ce.*, bm25(character_exemplars_fts) AS bm25_score
                FROM character_exemplars_fts
                JOIN character_exemplars ce ON character_exemplars_fts.rowid = ce.rowid
                WHERE character_exemplars_fts MATCH ?
                  AND ce.character_id = ?
                  AND ce.is_deleted = ?
                  AND (? IS NULL OR ce.emotion = ?)
                  AND (? IS NULL OR ce.scenario = ?)
                ORDER BY bm25_score, ce.updated_at DESC
            """
            query_params = [safe_fts] + filter_params
        else:
            sql = """
                SELECT ce.*
                FROM character_exemplars ce
                WHERE ce.character_id = ?
                  AND ce.is_deleted = ?
                  AND (? IS NULL OR ce.emotion = ?)
                  AND (? IS NULL OR ce.scenario = ?)
                ORDER BY ce.updated_at DESC
            """
            query_params = filter_params

        try:
            cursor = self._db.execute_query(sql, tuple(query_params))
            rows = cursor.fetchall()
        except CharactersRAGDBError as exc:
            logger.error(
                "Error searching character exemplars for character_id={} query='{}': {}",
                character_id,
                normalized_query,
                exc,
            )
            raise

        results = [self._normalize_character_exemplar_row(row) for row in rows if row]

        if rhetorical_filter:
            filtered_results: list[dict[str, Any]] = []
            for item in results:
                values = item.get('rhetorical') or []
                if isinstance(values, str):
                    try:
                        values = json.loads(values)
                    except json.JSONDecodeError:
                        values = []
                normalized_values = {
                    str(entry).strip().lower()
                    for entry in values
                    if str(entry).strip()
                }
                if rhetorical_filter.issubset(normalized_values):
                    filtered_results.append(item)
            results = filtered_results

        total = len(results)
        paginated = results[offset:offset + limit]
        return paginated, total
