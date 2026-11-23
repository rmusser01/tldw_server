"""Flashcard domain store: decks, templates, flashcards, review sessions, study packs and suggestions
(TASK-13950.22 PR J2)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    _FLASHCARD_REVIEW_MODES,
    _FLASHCARD_REVIEW_SESSION_JSON_FIELDS,
    _FLASHCARD_REVIEW_SESSION_STATUSES,
    _FLASHCARD_REVIEW_SESSION_TIMEOUT,
    _FLASHCARD_SQLITE_FTS_COLUMN_ALIASES,
    _FLASHCARD_TEMPLATE_FIELD_NAMES,
    _FLASHCARD_TEMPLATE_TOKEN_RE,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    _build_scheduler_next_interval_previews,
    _coerce_scheduler_type,
    _normalize_deck_share_role,
    _normalize_deck_visibility,
    _normalize_scheduler_settings_envelope,
    _simulate_scheduler_review_transition,
)
from tldw_Server_API.app.core.Flashcards.asset_refs import (
    extract_flashcard_asset_uuids,
    sanitize_flashcard_text_for_search,
)
from tldw_Server_API.app.core.Flashcards.scheduler_sm2 import (
    MATURE_INTERVAL_DAYS,
    coerce_queue_state,
    scheduler_settings_to_json,
    to_iso_z,
)

_STUDY_PACK_JSON_FIELDS = ['source_bundle_json', 'generation_options_json']

_SUGGESTION_SNAPSHOT_JSON_FIELDS = ('payload_json', 'user_selection_json')

_SAFE_SUGGESTION_PAYLOAD_EXACT_KEYS = frozenset(
    {
        "summary",
        "topics",
        "counts",
        "flags",
        "score",
        "selected",
        "display_label",
        "label",
        "topic_key",
        "normalization_version",
        "evidence_reasons",
        "source_id",
        "source_type",
        "source_ref",
        "ref",
        "refs",
        "status",
        "available",
        "enabled",
        "count",
        "source_count",
        "total",
        "id",
        "ids",
        "type",
        "types",
        "service",
        "services",
    }
)

_SAFE_SUGGESTION_PAYLOAD_SUFFIXES = (
    "_id",
    "_ids",
    "_type",
    "_types",
    "_label",
    "_labels",
    "_ref",
    "_refs",
    "_count",
    "_counts",
    "_flag",
    "_flags",
    "_score",
    "_status",
)

_SAFE_SUGGESTION_PAYLOAD_MAX_TEXT_LENGTH = 256

_SAFE_SUGGESTION_EVIDENCE_REASONS = frozenset(
    {
        "missed_question",
        "source_citation",
        "tag_match",
        "derived_label",
    }
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext


class FlashcardStore:
    def __init__(self, ctx: ChaChaContext) -> None:
        self._ctx = ctx

    def _build_flashcard_search_shadow_fields(
        self,
        *,
        front: str | None,
        back: str | None,
        notes: str | None,
    ) -> tuple[str, str, str]:
        """Forward to ``ChaChaContext.build_flashcard_search_shadow_fields``."""
        return self._ctx.build_flashcard_search_shadow_fields(front=front, back=back, notes=notes)

    # ==========================
    # Flashcards & Decks (V5)
    # ==========================
    def _flashcard_owner_filter(self, alias: str = "") -> tuple[str, tuple[str, ...]]:
        """Scope shared PostgreSQL resources; SQLite client IDs identify devices."""
        if self._ctx.backend_type != BackendType.POSTGRESQL:
            return "", ()
        columns = {"": "client_id", "f": "f.client_id", "d": "d.client_id", "fa": "fa.client_id", "fr": "fr.client_id", "kw": "kw.client_id"}
        return f" AND {columns[alias]} = ?", (self._ctx.client_id,)

    def _validate_flashcard_deck_locked(self, conn: Any, deck_id: Any) -> None:
        """Require a live owned deck when a PostgreSQL card references one."""
        if self._ctx.backend_type != BackendType.POSTGRESQL or deck_id is None:
            return
        row = conn.execute(
            "SELECT id FROM decks WHERE id = ? AND client_id = ? AND deleted = FALSE FOR SHARE",
            (deck_id, self._ctx.client_id),
        ).fetchone()
        if not row:
            raise InputError("Deck not found")  # noqa: TRY003

    def _validate_flashcard_workspace_locked(self, conn: Any, workspace_id: str | None) -> None:
        """Validate the owner at the deck boundary without broadening workspace APIs."""
        if self._ctx.backend_type != BackendType.POSTGRESQL or workspace_id is None:
            return
        row = conn.execute(
            "SELECT id FROM workspaces WHERE id = ? AND client_id = ? AND deleted = FALSE FOR SHARE",
            (workspace_id, self._ctx.client_id),
        ).fetchone()
        if not row:
            raise InputError("Workspace not found")  # noqa: TRY003

    @staticmethod
    def _normalize_deck_parent_id(parent_deck_id: Any) -> int | None:
        if parent_deck_id is None:
            return None
        if isinstance(parent_deck_id, bool):
            raise InputError("parent_deck_id must be a positive integer")  # noqa: TRY003
        try:
            normalized = int(parent_deck_id)
        except (TypeError, ValueError) as exc:
            raise InputError("parent_deck_id must be a positive integer") from exc  # noqa: TRY003
        if normalized <= 0:
            raise InputError("parent_deck_id must be a positive integer")  # noqa: TRY003
        return normalized

    def _validate_deck_parent_locked(
        self,
        conn: Any,
        *,
        deck_id: int | None,
        parent_deck_id: Any,
    ) -> int | None:
        parent_id = self._normalize_deck_parent_id(parent_deck_id)
        if parent_id is None:
            return None

        target_deck_id = int(deck_id) if deck_id is not None else None
        if target_deck_id is not None and parent_id == target_deck_id:
            raise InputError("Deck cannot be its own parent")  # noqa: TRY003

        owner_params: tuple[str, ...]
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            parent_lookup_query = (
                "SELECT id, parent_deck_id FROM decks WHERE id = ? AND deleted = 0 AND client_id = ? FOR UPDATE"
            )
            ancestor_lookup_query = "SELECT parent_deck_id FROM decks WHERE id = ? AND client_id = ? FOR UPDATE"
            owner_params = (self._ctx.client_id,)
        else:
            parent_lookup_query = "SELECT id, parent_deck_id FROM decks WHERE id = ? AND deleted = 0"
            ancestor_lookup_query = "SELECT parent_deck_id FROM decks WHERE id = ?"
            owner_params = ()
        parent_row = self._ctx.db.coerce_mapping_row(
            conn.execute(
                parent_lookup_query,
                (parent_id, *owner_params),
            ).fetchone()
        )
        if not parent_row:
            raise InputError("Parent deck not found")  # noqa: TRY003

        if target_deck_id is None and self._ctx.backend_type != BackendType.POSTGRESQL:
            return parent_id

        seen: set[int] = set()
        current_parent_id: int | None = parent_id
        while current_parent_id is not None:
            if current_parent_id == target_deck_id or current_parent_id in seen:
                raise InputError("Deck parent would create a cycle")  # noqa: TRY003
            seen.add(current_parent_id)
            row = self._ctx.db.coerce_mapping_row(
                conn.execute(
                    ancestor_lookup_query,
                    (current_parent_id, *owner_params),
                ).fetchone()
            )
            if not row:
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    raise InputError("Parent deck ancestry is not available to this owner")  # noqa: TRY003
                return parent_id
            raw_parent_id = row.get("parent_deck_id")
            current_parent_id = None if raw_parent_id is None else self._normalize_deck_parent_id(raw_parent_id)

        return parent_id

    def add_deck(
        self,
        name: str,
        description: str | None = None,
        scheduler_settings: Mapping[str, Any] | str | None = None,
        *,
        scheduler_type: str = "sm2_plus",
        workspace_id: str | None = None,
        parent_deck_id: Any = ...,
        visibility: str | None = None,
        review_prompt_side: str | None = None,
    ) -> int:
        """Create a deck and return its id."""
        now = self._ctx.utc_now_iso()
        scheduler_type = _coerce_scheduler_type(scheduler_type)
        normalized_visibility = None if visibility is None else _normalize_deck_visibility(visibility)
        visibility_for_insert = normalized_visibility or "private"
        normalized_prompt_side = (
            None
            if review_prompt_side is None
            else "back" if str(review_prompt_side).strip().lower() == "back" else "front"
        )
        prompt_side_for_insert = normalized_prompt_side or "front"
        scheduler_settings_json = scheduler_settings_to_json(scheduler_settings)
        owner_filter, owner_params = self._flashcard_owner_filter()
        try:
            with self._ctx.transaction() as conn:
                self._validate_flashcard_workspace_locked(conn, workspace_id)
                parent_for_insert = (
                    None
                    if parent_deck_id is ...
                    else self._validate_deck_parent_locked(
                        conn,
                        deck_id=None,
                        parent_deck_id=parent_deck_id,
                    )
                )
                # Undelete if a deck with the same name exists but is deleted
                deleted_row = conn.execute(
                    "SELECT id, version FROM decks WHERE name = ? AND deleted = 1" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (name, *owner_params),
                ).fetchone()
                if deleted_row:
                    deck_id, current_version = int(deleted_row["id"]), int(deleted_row["version"])
                    if parent_deck_id is not ...:
                        self._validate_deck_parent_locked(
                            conn,
                            deck_id=deck_id,
                            parent_deck_id=parent_for_insert,
                        )
                    next_version = current_version + 1
                    deleted_value = False if self._ctx.backend_type == BackendType.POSTGRESQL else 0
                    set_parts = [
                        "deleted = ?",
                        "last_modified = ?",
                        "version = ?",
                        "client_id = ?",
                        "description = COALESCE(?, description)",
                        "workspace_id = ?",
                        "visibility = COALESCE(?, visibility)",
                        "review_prompt_side = COALESCE(?, review_prompt_side)",
                        "scheduler_type = COALESCE(?, scheduler_type)",
                        "scheduler_settings_json = COALESCE(?, scheduler_settings_json)",
                    ]
                    params: list[Any] = [
                        deleted_value,
                        now,
                        next_version,
                        self._ctx.client_id,
                        description,
                        workspace_id,
                        normalized_visibility,
                        normalized_prompt_side,
                        scheduler_type,
                        scheduler_settings_json,
                    ]
                    if parent_deck_id is not ...:
                        set_parts.append("parent_deck_id = ?")
                        params.append(parent_for_insert)
                    params.extend([deck_id, current_version, *owner_params])
                    rc = conn.execute(
                        f"UPDATE decks SET {', '.join(set_parts)} WHERE id = ? AND version = ?{owner_filter}",  # nosec B608
                        tuple(params),
                    ).rowcount
                    if rc == 0:
                        raise ConflictError(  # noqa: TRY003
                            "Failed to undelete deck due to version mismatch",
                            entity="decks",
                            identifier=name,
                        )
                    return deck_id
                insert_sql = (
                    "INSERT INTO decks("
                    "name, description, parent_deck_id, workspace_id, visibility, review_prompt_side, scheduler_settings_json, scheduler_type, created_at, last_modified, client_id, version, deleted"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                insert_params = (
                    name,
                    description,
                    parent_for_insert,
                    workspace_id,
                    visibility_for_insert,
                    prompt_side_for_insert,
                    scheduler_settings_json,
                    scheduler_type,
                    now,
                    now,
                    self._ctx.client_id,
                    1,
                    False,
                )

                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_sql + " RETURNING id", insert_params)
                    row = cursor.fetchone()
                    inserted_deck_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_sql, insert_params)
                    inserted_deck_id = cursor.lastrowid

                if inserted_deck_id is None:
                    raise CharactersRAGDBError("Failed to determine deck ID after insert")  # noqa: TRY003
                inserted_deck_id = int(inserted_deck_id)
                return inserted_deck_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: decks.name" in str(e):
                raise ConflictError("Deck name already exists", entity="decks", identifier=name) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to create deck: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            if self._ctx.is_unique_violation(e):
                raise ConflictError("Deck name already exists", entity="decks", identifier=name) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to create deck: {e}") from e  # noqa: TRY003
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to create deck: {e}") from e  # noqa: TRY003

    def list_decks(
        self,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
        shared_with_user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if shared_with_user_id is not None:
            query = (
                "SELECT d.id, d.name, d.description, d.parent_deck_id, d.workspace_id, d.visibility, d.review_prompt_side, "
                "d.scheduler_settings_json, d.scheduler_type, d.created_at, d.last_modified, "
                "d.deleted, d.client_id, d.version FROM decks d "
                "JOIN deck_shares ds ON ds.deck_id = d.id "
            )
            conditions = ["ds.user_id = ?"]
            params_list: list[Any] = [int(shared_with_user_id)]
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                conditions.append("d.client_id = ?")
                params_list.append(self._ctx.client_id)
            if not include_deleted:
                conditions.append("d.deleted = 0")
            if workspace_id is not None:
                if include_workspace_items:
                    conditions.append("(d.workspace_id = ? OR d.workspace_id IS NULL)")
                else:
                    conditions.append("d.workspace_id = ?")
                params_list.append(workspace_id)
            query += "WHERE " + " AND ".join(conditions) + " ORDER BY d.name LIMIT ? OFFSET ?"
            params_list.extend([limit, offset])
            try:
                cursor = self._ctx.execute_query(query, tuple(params_list), read_only=True)
                return [dict(row) for row in cursor.fetchall()]
            except CharactersRAGDBError:  # noqa: TRY203
                raise

        select_sql = (
            "SELECT id, name, description, parent_deck_id, workspace_id, visibility, review_prompt_side, scheduler_settings_json, scheduler_type, created_at, last_modified, "
            "deleted, client_id, version FROM decks "
        )
        owner_filter, owner_params = self._flashcard_owner_filter()
        query = select_sql + "WHERE 1 = 1" + owner_filter
        params: list[str | int] = list(owner_params)
        if not include_deleted:
            query += " AND deleted = 0"
        if workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        elif not include_workspace_items:
            query += " AND workspace_id IS NULL"
        query += " ORDER BY name LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        try:
            cursor = self._ctx.execute_query(query, tuple(params), read_only=True)
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def get_deck(self, deck_id: int) -> dict[str, Any] | None:
        """Fetch a single deck row by id."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        query = (
            "SELECT id, name, description, parent_deck_id, workspace_id, visibility, review_prompt_side, scheduler_settings_json, scheduler_type, created_at, last_modified, deleted, client_id, version "
            "FROM decks WHERE id = ?" + owner_filter  # nosec B608 -- Fixed owner clauses; all data values are bound.
        )
        try:
            cursor = self._ctx.execute_query(query, (deck_id, *owner_params), read_only=True)
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def get_deck_by_name(self, name: str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        """Fetch one deck row by name in the selected owner's repository."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        if include_deleted:
            query = (
                "SELECT id, name, description, parent_deck_id, workspace_id, review_prompt_side, scheduler_settings_json, scheduler_type, created_at, "
                "last_modified, deleted, client_id, version, visibility FROM decks WHERE name = ? "
                + owner_filter + " ORDER BY id LIMIT 1"  # nosec B608 -- Fixed owner clauses; all data values are bound.
            )
        else:
            query = (
                "SELECT id, name, description, parent_deck_id, workspace_id, review_prompt_side, scheduler_settings_json, scheduler_type, created_at, "
                "last_modified, deleted, client_id, version, visibility FROM decks WHERE name = ? AND deleted = 0 "
                + owner_filter + " ORDER BY id LIMIT 1"  # nosec B608 -- Fixed owner clauses; all data values are bound.
            )
        try:
            cursor = self._ctx.execute_query(query, (name, *owner_params), read_only=True)
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def upsert_deck_share(
        self,
        deck_id: int,
        *,
        user_id: int,
        role: str = "viewer",
        shared_by: int,
    ) -> dict[str, Any]:
        """Create or update a per-user share record for a deck owned by this DB."""
        target_user_id = int(user_id)
        sharing_user_id = int(shared_by)
        if target_user_id <= 0:
            raise InputError("user_id must be a positive integer")  # noqa: TRY003
        if sharing_user_id <= 0:
            raise InputError("shared_by must be a positive integer")  # noqa: TRY003
        if target_user_id == sharing_user_id:
            raise InputError("Cannot share a deck with its owner")  # noqa: TRY003

        normalized_role = _normalize_deck_share_role(role)
        now = self._ctx.utc_now_iso()
        owner_filter, owner_params = self._flashcard_owner_filter()
        lock_clause = " FOR UPDATE" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        try:
            with self._ctx.transaction() as conn:
                deck = conn.execute(
                    "SELECT id FROM decks WHERE id = ? AND deleted = 0" + owner_filter + lock_clause,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (int(deck_id), *owner_params),
                ).fetchone()
                if deck is None:
                    raise ConflictError("Deck not found", entity="decks", identifier=deck_id)  # noqa: TRY003

                insert_sql = """
                    INSERT INTO deck_shares(
                        deck_id, user_id, role, shared_by, shared_at, last_modified, client_id, version
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(deck_id, user_id) DO UPDATE SET
                        role = excluded.role,
                        shared_by = excluded.shared_by,
                        last_modified = excluded.last_modified,
                        client_id = excluded.client_id,
                        version = deck_shares.version + 1
                """
                params = (
                    int(deck_id),
                    target_user_id,
                    normalized_role,
                    sharing_user_id,
                    now,
                    now,
                    self._ctx.client_id,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    row = conn.execute(
                        insert_sql
                        + " RETURNING deck_id, user_id, role, shared_by, shared_at, last_modified, client_id, version",
                        params,
                    ).fetchone()
                    return dict(row)

                conn.execute(insert_sql, params)
                row = conn.execute(
                    """
                    SELECT deck_id, user_id, role, shared_by, shared_at, last_modified, client_id, version
                      FROM deck_shares
                     WHERE deck_id = ? AND user_id = ?
                    """,
                    (int(deck_id), target_user_id),
                ).fetchone()
                if row is None:
                    raise CharactersRAGDBError("Failed to read deck share after upsert")  # noqa: TRY003
                return dict(row)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to share deck: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to share deck: {exc}") from exc  # noqa: TRY003

    def list_deck_shares(self, deck_id: int) -> list[dict[str, Any]]:
        """List all per-user share records for a deck owned by this DB."""
        owner_filter = ""
        owner_params: tuple[str, ...] = ()
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            owner_filter = " AND EXISTS (SELECT 1 FROM decks d WHERE d.id = deck_shares.deck_id AND d.client_id = ?)"
            owner_params = (self._ctx.client_id,)
        try:
            cursor = self._ctx.execute_query(
                f"""
                SELECT deck_id, user_id, role, shared_by, shared_at, last_modified, client_id, version
                  FROM deck_shares
                 WHERE deck_id = ?{owner_filter}
                 ORDER BY user_id
                """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                (int(deck_id), *owner_params),
                read_only=True,
            )
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def get_deck_share(self, deck_id: int, *, user_id: int) -> dict[str, Any] | None:
        """Fetch one per-user share record for a deck owned by this DB."""
        owner_filter = ""
        owner_params: tuple[str, ...] = ()
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            owner_filter = " AND EXISTS (SELECT 1 FROM decks d WHERE d.id = deck_shares.deck_id AND d.client_id = ?)"
            owner_params = (self._ctx.client_id,)
        try:
            cursor = self._ctx.execute_query(
                f"""
                SELECT deck_id, user_id, role, shared_by, shared_at, last_modified, client_id, version
                  FROM deck_shares
                 WHERE deck_id = ? AND user_id = ?{owner_filter}
                """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                (int(deck_id), int(user_id), *owner_params),
                read_only=True,
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def delete_deck_share(self, deck_id: int, *, user_id: int) -> bool:
        """Remove one per-user share record from a deck owned by this DB."""
        owner_filter = ""
        owner_params: tuple[str, ...] = ()
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            owner_filter = " AND EXISTS (SELECT 1 FROM decks d WHERE d.id = deck_shares.deck_id AND d.client_id = ?)"
            owner_params = (self._ctx.client_id,)
        try:
            with self._ctx.transaction() as conn:
                rowcount = conn.execute(
                    "DELETE FROM deck_shares WHERE deck_id = ? AND user_id = ?" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (int(deck_id), int(user_id), *owner_params),
                ).rowcount
                return rowcount > 0
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to remove deck share: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to remove deck share: {exc}") from exc  # noqa: TRY003

    def update_deck(
        self,
        deck_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        review_prompt_side: str | None = None,
        visibility: str | None = None,
        scheduler_settings: Mapping[str, Any] | str | None = None,
        scheduler_type: str | None = None,
        workspace_id: Any = ...,
        parent_deck_id: Any = ...,
        expected_version: int | None = None,
    ) -> bool:
        """Update mutable deck fields with optimistic locking."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        set_parts: list[str] = []
        params: list[Any] = []
        if name is not None:
            set_parts.append("name = ?")
            params.append(name)
        if description is not None:
            set_parts.append("description = ?")
            params.append(description)
        if review_prompt_side is not None:
            set_parts.append("review_prompt_side = ?")
            params.append("back" if str(review_prompt_side).strip().lower() == "back" else "front")
        if visibility is not None:
            set_parts.append("visibility = ?")
            params.append(_normalize_deck_visibility(visibility))
        if scheduler_settings is not None:
            set_parts.append("scheduler_settings_json = ?")
            params.append(scheduler_settings_to_json(scheduler_settings))
        if scheduler_type is not None:
            set_parts.append("scheduler_type = ?")
            params.append(_coerce_scheduler_type(scheduler_type))
        if workspace_id is not ...:
            set_parts.append("workspace_id = ?")
            params.append(workspace_id)
        if parent_deck_id is not ...:
            set_parts.append("parent_deck_id = ?")
            params.append(self._normalize_deck_parent_id(parent_deck_id))
        if not set_parts:
            if expected_version is None and self._ctx.backend_type != BackendType.POSTGRESQL:
                return True
            deck = self.get_deck(deck_id)
            if not deck:
                raise ConflictError("Deck not found", entity="decks", identifier=deck_id)  # noqa: TRY003
            if expected_version is not None and int(deck["version"]) != expected_version:
                raise ConflictError("Version mismatch updating deck", entity="decks", identifier=deck_id)  # noqa: TRY003
            return True

        now = self._ctx.utc_now_iso()
        set_parts.extend(["last_modified = ?", "version = version + 1", "client_id = ?"])
        params.extend([now, self._ctx.client_id])

        try:
            with self._ctx.transaction() as conn:
                row = conn.execute(
                    "SELECT version FROM decks WHERE id = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (deck_id, *owner_params),
                ).fetchone()
                if not row:
                    raise ConflictError("Deck not found", entity="decks", identifier=deck_id)  # noqa: TRY003
                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError("Version mismatch updating deck", entity="decks", identifier=deck_id)  # noqa: TRY003
                if workspace_id is not ...:
                    self._validate_flashcard_workspace_locked(conn, workspace_id)
                if parent_deck_id is not ...:
                    params[set_parts.index("parent_deck_id = ?")] = self._validate_deck_parent_locked(
                        conn,
                        deck_id=int(deck_id),
                        parent_deck_id=parent_deck_id,
                    )

                params_final = params + [deck_id, *owner_params]
                query = f"UPDATE decks SET {', '.join(set_parts)} WHERE id = ? AND deleted = 0{owner_filter}"  # nosec B608
                rc = conn.execute(query, tuple(params_final)).rowcount
                return rc > 0
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: decks.name" in str(exc):
                raise ConflictError("Deck name already exists", entity="decks", identifier=name or deck_id) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to update deck: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to update deck: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError("Deck name already exists", entity="decks", identifier=name or deck_id) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to update deck: {exc}") from exc  # noqa: TRY003

    def _flashcard_template_default_placeholders(self, raw_definitions: Any) -> list[dict[str, Any]]:
        if raw_definitions in (None, ""):
            return []
        if not isinstance(raw_definitions, list):
            raise InputError("placeholder_definitions must be a list")  # noqa: TRY003
        normalized: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for raw_definition in raw_definitions:
            if not isinstance(raw_definition, Mapping):
                raise InputError("placeholder_definitions items must be objects")  # noqa: TRY003
            key = str(raw_definition.get("key") or "").strip()
            label = str(raw_definition.get("label") or "").strip()
            help_text = self._ctx.normalize_nullable_text(raw_definition.get("help_text"))
            default_value = self._ctx.normalize_nullable_text(raw_definition.get("default_value"))
            required = bool(raw_definition.get("required", False))
            targets_raw = raw_definition.get("targets") or []
            if not key:
                raise InputError("Placeholder definitions require a non-empty key")  # noqa: TRY003
            if key in seen_keys:
                raise InputError(f"Duplicate placeholder definition: {key}")  # noqa: TRY003
            if not label:
                raise InputError(f"Placeholder definition '{key}' requires a non-empty label")  # noqa: TRY003
            if not isinstance(targets_raw, list) or not targets_raw:
                raise InputError(f"Placeholder definition '{key}' requires at least one target")  # noqa: TRY003
            targets: list[str] = []
            for target in targets_raw:
                target_name = str(target or "").strip()
                if target_name not in _FLASHCARD_TEMPLATE_FIELD_NAMES:
                    raise InputError(
                        f"Placeholder definition '{key}' has unsupported target '{target_name}'"
                    )  # noqa: TRY003
                if target_name not in targets:
                    targets.append(target_name)
            seen_keys.add(key)
            normalized.append(
                {
                    "key": key,
                    "label": label,
                    "help_text": help_text,
                    "default_value": default_value,
                    "required": required,
                    "targets": targets,
                }
            )
        return normalized

    def _flashcard_template_tokens(self, text: Any) -> set[str]:
        raw = str(text or "")
        return {match.group(1) for match in _FLASHCARD_TEMPLATE_TOKEN_RE.finditer(raw)}

    def _flashcard_template_deleted_value(self, deleted: bool) -> bool | int:
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            return bool(deleted)
        return 1 if deleted else 0

    def _validate_flashcard_template_contract(
        self,
        *,
        model_type: str,
        front_template: Any,
        back_template: Any,
        notes_template: Any,
        extra_template: Any,
        placeholder_definitions: Any,
    ) -> tuple[str, str, str | None, str | None, str | None, list[dict[str, Any]]]:
        normalized_model_type = str(model_type or "").strip().lower()
        if normalized_model_type not in ("basic", "basic_reverse", "cloze"):
            raise InputError("model_type must be one of 'basic', 'basic_reverse', or 'cloze'")  # noqa: TRY003

        normalized_front = str(front_template or "").strip()
        if not normalized_front:
            raise InputError("front_template is required")  # noqa: TRY003

        normalized_back = self._ctx.normalize_nullable_text(back_template)
        if normalized_model_type in ("basic", "basic_reverse") and not normalized_back:
            raise InputError("back_template is required for basic and basic_reverse templates")  # noqa: TRY003

        normalized_notes = self._ctx.normalize_nullable_text(notes_template)
        normalized_extra = self._ctx.normalize_nullable_text(extra_template)
        normalized_definitions = self._flashcard_template_default_placeholders(placeholder_definitions)
        definitions_by_key = {definition["key"]: definition for definition in normalized_definitions}

        fields_by_name = {
            "front_template": normalized_front,
            "back_template": normalized_back or "",
            "notes_template": normalized_notes or "",
            "extra_template": normalized_extra or "",
        }
        for field_name, text in fields_by_name.items():
            tokens = self._flashcard_template_tokens(text)
            if not tokens:
                continue
            missing = sorted(token for token in tokens if token not in definitions_by_key)
            if missing:
                raise InputError(
                    f"{field_name} references undefined placeholder(s): {', '.join(missing)}"
                )  # noqa: TRY003

        for definition in normalized_definitions:
            targets = definition["targets"]
            if not any(definition["key"] in self._flashcard_template_tokens(fields_by_name[target]) for target in targets):
                raise InputError(
                    f"Placeholder definition '{definition['key']}' must be referenced in at least one targeted field"
                )  # noqa: TRY003

        return (
            normalized_model_type,
            normalized_front,
            normalized_back,
            normalized_notes,
            normalized_extra,
            normalized_definitions,
        )

    def _serialize_flashcard_template_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        raw_definitions = item.get("placeholder_definitions_json")
        if isinstance(raw_definitions, str):
            try:
                parsed = json.loads(raw_definitions)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "_serialize_flashcard_template_row failed to parse placeholder_definitions_json "
                    "for flashcard template {}: {}. Raw value: {}",
                    item.get("id"),
                    exc,
                    raw_definitions,
                )
                parsed = []
        else:
            parsed = raw_definitions
        if not isinstance(parsed, list):
            parsed = []
        item["placeholder_definitions"] = self._flashcard_template_default_placeholders(parsed)
        item.pop("placeholder_definitions_json", None)
        return item

    def add_flashcard_template(
        self,
        *,
        name: str,
        model_type: str,
        front_template: str,
        back_template: str | None = None,
        notes_template: str | None = None,
        extra_template: str | None = None,
        placeholder_definitions: list[dict[str, Any]] | None = None,
    ) -> int:
        """Create or undelete a flashcard template and return its integer id."""
        template_name = str(name or "").strip()
        if not template_name:
            raise InputError("name is required")  # noqa: TRY003

        (
            normalized_model_type,
            normalized_front,
            normalized_back,
            normalized_notes,
            normalized_extra,
            normalized_definitions,
        ) = self._validate_flashcard_template_contract(
            model_type=model_type,
            front_template=front_template,
            back_template=back_template,
            notes_template=notes_template,
            extra_template=extra_template,
            placeholder_definitions=placeholder_definitions,
        )

        now = self._ctx.utc_now_iso()
        definitions_json = json.dumps(normalized_definitions)
        deleted_value = self._flashcard_template_deleted_value(True)
        try:
            with self._ctx.transaction() as conn:
                deleted_row = conn.execute(
                    """
                    SELECT id, version
                      FROM flashcard_templates
                     WHERE name = ? AND deleted = ?
                     ORDER BY id DESC
                     LIMIT 1
                    """,
                    (template_name, deleted_value),
                ).fetchone()
                if deleted_row:
                    template_id = int(deleted_row[0])
                    current_version = int(deleted_row[1])
                    next_version = current_version + 1
                    rc = conn.execute(
                        """
                        UPDATE flashcard_templates
                           SET deleted = ?,
                               name = ?,
                               model_type = ?,
                               front_template = ?,
                               back_template = ?,
                               notes_template = ?,
                               extra_template = ?,
                               placeholder_definitions_json = ?,
                               last_modified = ?,
                               version = ?,
                               client_id = ?
                         WHERE id = ? AND version = ?
                        """,
                        (
                            self._flashcard_template_deleted_value(False),
                            template_name,
                            normalized_model_type,
                            normalized_front,
                            normalized_back,
                            normalized_notes,
                            normalized_extra,
                            definitions_json,
                            now,
                            next_version,
                            self._ctx.client_id,
                            template_id,
                            current_version,
                        ),
                    ).rowcount
                    if rc == 0:
                        raise ConflictError(
                            "Failed to undelete flashcard template due to version mismatch",
                            entity="flashcard_templates",
                            identifier=template_name,
                        )  # noqa: TRY003
                    return template_id

                insert_sql = """
                    INSERT INTO flashcard_templates(
                        name, model_type, front_template, back_template, notes_template, extra_template,
                        placeholder_definitions_json, created_at, last_modified, deleted, client_id, version
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    template_name,
                    normalized_model_type,
                    normalized_front,
                    normalized_back,
                    normalized_notes,
                    normalized_extra,
                    definitions_json,
                    now,
                    now,
                    self._flashcard_template_deleted_value(False),
                    self._ctx.client_id,
                    1,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_sql + " RETURNING id", params)
                    row = cursor.fetchone()
                    inserted_template_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_sql, params)
                    inserted_template_id = cursor.lastrowid
                if inserted_template_id is None:
                    raise CharactersRAGDBError("Failed to determine flashcard template ID after insert")  # noqa: TRY003
                inserted_template_id = int(inserted_template_id)
                return inserted_template_id
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: flashcard_templates.name" in str(exc):
                raise ConflictError(
                    "Flashcard template name already exists",
                    entity="flashcard_templates",
                    identifier=template_name,
                ) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to create flashcard template: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError(
                    "Flashcard template name already exists",
                    entity="flashcard_templates",
                    identifier=template_name,
                ) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to create flashcard template: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to create flashcard template: {exc}") from exc  # noqa: TRY003

    def count_flashcard_templates(self) -> int:
        cursor = self._ctx.execute_query(
            "SELECT COUNT(*) AS cnt FROM flashcard_templates WHERE deleted = ?",
            (self._flashcard_template_deleted_value(False),),
            read_only=True,
        )
        row = cursor.fetchone()
        return int(row["cnt"]) if row else 0

    def list_flashcard_templates(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        query = (
            "SELECT id, name, model_type, front_template, back_template, notes_template, extra_template, "
            "placeholder_definitions_json, created_at, last_modified, deleted, client_id, version "
            "FROM flashcard_templates WHERE deleted = ? ORDER BY name ASC LIMIT ? OFFSET ?"
        )
        cursor = self._ctx.execute_query(
            query,
            (self._flashcard_template_deleted_value(False), limit, offset),
            read_only=True,
        )
        return [self._serialize_flashcard_template_row(dict(row)) for row in cursor.fetchall()]

    def get_flashcard_template(self, template_id: int, *, include_deleted: bool = False) -> dict[str, Any] | None:
        query = (
            "SELECT id, name, model_type, front_template, back_template, notes_template, extra_template, "
            "placeholder_definitions_json, created_at, last_modified, deleted, client_id, version "
            "FROM flashcard_templates WHERE id = ?"
        )
        params: list[Any] = [template_id]
        if not include_deleted:
            query += " AND deleted = ?"
            params.append(self._flashcard_template_deleted_value(False))
        cursor = self._ctx.execute_query(query, tuple(params), read_only=True)
        row = cursor.fetchone()
        return self._serialize_flashcard_template_row(dict(row)) if row else None

    def update_flashcard_template(
        self,
        template_id: int,
        updates: dict[str, Any],
        expected_version: int | None = None,
    ) -> bool:
        current = self.get_flashcard_template(template_id)
        if not current:
            return False
        if not updates:
            if expected_version is not None and int(current["version"]) != expected_version:
                raise ConflictError(
                    "Version mismatch updating flashcard template",
                    entity="flashcard_templates",
                    identifier=template_id,
                )  # noqa: TRY003
            return True

        merged = {
            "name": current["name"],
            "model_type": current["model_type"],
            "front_template": current["front_template"],
            "back_template": current.get("back_template"),
            "notes_template": current.get("notes_template"),
            "extra_template": current.get("extra_template"),
            "placeholder_definitions": current.get("placeholder_definitions") or [],
        }
        for key in ("name", "model_type", "front_template", "back_template", "notes_template", "extra_template", "placeholder_definitions"):
            if key in updates:
                merged[key] = updates[key]

        (
            normalized_model_type,
            normalized_front,
            normalized_back,
            normalized_notes,
            normalized_extra,
            normalized_definitions,
        ) = self._validate_flashcard_template_contract(
            model_type=merged["model_type"],
            front_template=merged["front_template"],
            back_template=merged["back_template"],
            notes_template=merged["notes_template"],
            extra_template=merged["extra_template"],
            placeholder_definitions=merged["placeholder_definitions"],
        )
        template_name = str(merged["name"] or "").strip()
        if not template_name:
            raise InputError("name is required")  # noqa: TRY003

        now = self._ctx.utc_now_iso()
        definitions_json = json.dumps(normalized_definitions)
        set_parts = [
            "name = ?",
            "model_type = ?",
            "front_template = ?",
            "back_template = ?",
            "notes_template = ?",
            "extra_template = ?",
            "placeholder_definitions_json = ?",
            "last_modified = ?",
            "version = version + 1",
            "client_id = ?",
        ]
        params: list[Any] = [
            template_name,
            normalized_model_type,
            normalized_front,
            normalized_back,
            normalized_notes,
            normalized_extra,
            definitions_json,
            now,
            self._ctx.client_id,
        ]
        active_value = self._flashcard_template_deleted_value(False)
        try:
            with self._ctx.transaction() as conn:
                row = conn.execute(
                    "SELECT version FROM flashcard_templates WHERE id = ? AND deleted = ?",
                    (template_id, active_value),
                ).fetchone()
                if not row:
                    return False
                current_version = int(row[0])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch updating flashcard template",
                        entity="flashcard_templates",
                        identifier=template_id,
                    )  # noqa: TRY003

                if expected_version is not None:
                    params.extend([template_id, expected_version, active_value])
                    query = f"UPDATE flashcard_templates SET {', '.join(set_parts)} WHERE id = ? AND version = ? AND deleted = ?"  # nosec B608
                else:
                    params.extend([template_id, active_value])
                    query = f"UPDATE flashcard_templates SET {', '.join(set_parts)} WHERE id = ? AND deleted = ?"  # nosec B608
                rc = conn.execute(query, tuple(params)).rowcount
                return rc > 0
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: flashcard_templates.name" in str(exc):
                raise ConflictError(
                    "Flashcard template name already exists",
                    entity="flashcard_templates",
                    identifier=template_name,
                ) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to update flashcard template: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError(
                    "Flashcard template name already exists",
                    entity="flashcard_templates",
                    identifier=template_name,
                ) from exc  # noqa: TRY003
            raise CharactersRAGDBError(f"Failed to update flashcard template: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to update flashcard template: {exc}") from exc  # noqa: TRY003

    def soft_delete_flashcard_template(self, template_id: int, expected_version: int) -> bool:
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                row = conn.execute(
                    "SELECT version, deleted FROM flashcard_templates WHERE id = ?",
                    (template_id,),
                ).fetchone()
                if not row:
                    return False
                current_version = int(row[0])
                deleted = bool(row[1])
                if deleted:
                    return False
                if current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch deleting flashcard template",
                        entity="flashcard_templates",
                        identifier=template_id,
                    )  # noqa: TRY003
                rc = conn.execute(
                    """
                    UPDATE flashcard_templates
                       SET deleted = ?,
                           last_modified = ?,
                           version = ?,
                           client_id = ?
                     WHERE id = ? AND version = ? AND deleted = ?
                    """,
                    (
                        self._flashcard_template_deleted_value(True),
                        now,
                        expected_version + 1,
                        self._ctx.client_id,
                        template_id,
                        expected_version,
                        self._flashcard_template_deleted_value(False),
                    ),
                ).rowcount
                if rc == 0:
                    raise ConflictError(
                        "Version mismatch deleting flashcard template",
                        entity="flashcard_templates",
                        identifier=template_id,
                    )  # noqa: TRY003
                return rc > 0
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to delete flashcard template: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to delete flashcard template: {exc}") from exc  # noqa: TRY003

    def _get_flashcard_id_from_uuid(self, conn: sqlite3.Connection, card_uuid: str) -> int | None:
        row = conn.execute("SELECT id FROM flashcards WHERE uuid = ? AND deleted = 0", (card_uuid,)).fetchone()
        return int(row[0]) if row else None

    def add_flashcard(self, card_data: dict[str, Any]) -> str:
        """
        Create a flashcard. card_data keys: deck_id (optional), front, back, notes?, is_cloze?, tags_json?,
        source_ref_type?, source_ref_id?
        Returns uuid.
        """
        now = self._ctx.utc_now_iso()
        uuid_val = self._ctx.new_uuid()
        deck_id = card_data.get('deck_id')
        front = card_data['front']
        back = card_data['back']
        notes = card_data.get('notes')
        extra = card_data.get('extra')
        front_search, back_search, notes_search = self._build_flashcard_search_shadow_fields(
            front=front,
            back=back,
            notes=notes,
        )
        is_cloze = 1 if card_data.get('is_cloze') else 0
        tags_json = card_data.get('tags_json')
        source_ref_type = card_data.get('source_ref_type', 'manual')
        source_ref_id = card_data.get('source_ref_id')
        conversation_id = self._ctx.normalize_nullable_text(card_data.get('conversation_id'))
        message_id = self._ctx.normalize_nullable_text(card_data.get('message_id'))
        ef = max(1.3, float(card_data.get('ef', 2.5) or 2.5))
        interval_days = max(0, int(card_data.get('interval_days', 0) or 0))
        repetitions = max(0, int(card_data.get('repetitions', 0) or 0))
        lapses = max(0, int(card_data.get('lapses', 0) or 0))
        due_at = self._ctx.normalize_nullable_text(card_data.get('due_at'))
        last_reviewed_at = self._ctx.normalize_nullable_text(card_data.get('last_reviewed_at'))
        queue_state = coerce_queue_state(
            {
                "queue_state": card_data.get("queue_state"),
                "last_reviewed_at": last_reviewed_at,
                "repetitions": repetitions,
                "lapses": lapses,
                "interval_days": interval_days,
                "due_at": due_at,
            }
        )
        step_index = card_data.get("step_index")
        step_index = int(step_index) if step_index is not None else None
        suspended_reason = self._ctx.normalize_nullable_text(card_data.get("suspended_reason"))
        if queue_state != "suspended":
            suspended_reason = None
        if due_at is None and queue_state in ("learning", "relearning", "review"):
            due_at = now
        model_type = card_data.get('model_type')
        reverse_flag = card_data.get('reverse')
        if not model_type:
            model_type = 'cloze' if is_cloze else 'basic_reverse' if reverse_flag else 'basic'
        if model_type not in ('basic', 'basic_reverse', 'cloze'):
            raise InputError("Invalid model_type; must be 'basic','basic_reverse','cloze'")  # noqa: TRY003
        # If reverse is not explicitly set, derive from model_type
        if reverse_flag is None:
            reverse_flag = 1 if model_type == 'basic_reverse' else 0
        try:
            with self._ctx.transaction() as conn:
                self._validate_flashcard_deck_locked(conn, deck_id)
                insert_sql = (
                    """
                    INSERT INTO flashcards(
                        uuid, deck_id, front, back, notes, extra, front_search, back_search, notes_search, is_cloze,
                        tags_json, source_ref_type, source_ref_id, conversation_id, message_id, ef, interval_days,
                        repetitions, lapses, due_at, last_reviewed_at, created_at, last_modified,
                        deleted, client_id, version, model_type, reverse, queue_state, step_index, suspended_reason
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                )

                params = (
                    uuid_val,
                    deck_id,
                    front,
                    back,
                    notes,
                    extra,
                    front_search,
                    back_search,
                    notes_search,
                    bool(is_cloze),
                    tags_json,
                    source_ref_type,
                    source_ref_id,
                    conversation_id,
                    message_id,
                    ef,
                    interval_days,
                    repetitions,
                    lapses,
                    due_at,
                    last_reviewed_at,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                    model_type,
                    bool(reverse_flag),
                    queue_state,
                    step_index,
                    suspended_reason,
                )

                conn.execute(insert_sql, params)
                return uuid_val
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to add flashcard: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to add flashcard: {exc}") from exc  # noqa: TRY003

    def add_flashcards_bulk(self, cards: list[dict[str, Any]]) -> list[str]:
        """
        Bulk create flashcards; returns list of uuids in the same order.

        Supports newer fields: extra, model_type, reverse. If model_type is not
        provided, it will be inferred from is_cloze/reverse similar to add_flashcard().
        """
        uuids: list[str] = []
        try:
            with self._ctx.transaction() as conn:
                insert_sql = (
                    """
                    INSERT INTO flashcards(
                        uuid, deck_id, front, back, notes, extra, front_search, back_search, notes_search, is_cloze,
                        tags_json, source_ref_type, source_ref_id, conversation_id, message_id, ef, interval_days,
                        repetitions, lapses, due_at, last_reviewed_at, created_at, last_modified,
                        deleted, client_id, version, model_type, reverse, queue_state, step_index, suspended_reason
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                )

                params_list: list[tuple] = []
                for card_data in cards:
                    uuid_val = self._ctx.new_uuid()
                    uuids.append(uuid_val)
                    now = self._ctx.utc_now_iso()

                    deck_id = card_data.get('deck_id')
                    self._validate_flashcard_deck_locked(conn, deck_id)
                    front = card_data['front']
                    back = card_data['back']
                    notes = card_data.get('notes')
                    extra = card_data.get('extra')
                    front_search, back_search, notes_search = self._build_flashcard_search_shadow_fields(
                        front=front,
                        back=back,
                        notes=notes,
                    )
                    is_cloze = bool(card_data.get('is_cloze'))
                    tags_json = card_data.get('tags_json')
                    source_ref_type = card_data.get('source_ref_type', 'manual')
                    source_ref_id = card_data.get('source_ref_id')
                    conversation_id = self._ctx.normalize_nullable_text(card_data.get('conversation_id'))
                    message_id = self._ctx.normalize_nullable_text(card_data.get('message_id'))
                    ef = max(1.3, float(card_data.get('ef', 2.5) or 2.5))
                    interval_days = max(0, int(card_data.get('interval_days', 0) or 0))
                    repetitions = max(0, int(card_data.get('repetitions', 0) or 0))
                    lapses = max(0, int(card_data.get('lapses', 0) or 0))
                    due_at = self._ctx.normalize_nullable_text(card_data.get('due_at'))
                    last_reviewed_at = self._ctx.normalize_nullable_text(card_data.get('last_reviewed_at'))
                    queue_state = coerce_queue_state(
                        {
                            "queue_state": card_data.get("queue_state"),
                            "last_reviewed_at": last_reviewed_at,
                            "repetitions": repetitions,
                            "lapses": lapses,
                            "interval_days": interval_days,
                            "due_at": due_at,
                        }
                    )
                    step_index = card_data.get("step_index")
                    step_index = int(step_index) if step_index is not None else None
                    suspended_reason = self._ctx.normalize_nullable_text(card_data.get("suspended_reason"))
                    if queue_state != "suspended":
                        suspended_reason = None
                    if due_at is None and queue_state in ("learning", "relearning", "review"):
                        due_at = now
                    model_type = card_data.get('model_type')
                    reverse_flag = card_data.get('reverse')
                    if not model_type:
                        model_type = 'cloze' if is_cloze else 'basic_reverse' if reverse_flag else 'basic'
                    if model_type not in ('basic', 'basic_reverse', 'cloze'):
                        raise InputError("Invalid model_type; must be 'basic','basic_reverse','cloze'")  # noqa: TRY003
                    if reverse_flag is None:
                        reverse_flag = (model_type == 'basic_reverse')

                    params_list.append(
                        (
                            uuid_val,
                            deck_id,
                            front,
                            back,
                            notes,
                            extra,
                            front_search,
                            back_search,
                            notes_search,
                            is_cloze,
                            tags_json,
                            source_ref_type,
                            source_ref_id,
                            conversation_id,
                            message_id,
                            ef,
                            interval_days,
                            repetitions,
                            lapses,
                            due_at,
                            last_reviewed_at,
                            now,
                            now,
                            False,
                            self._ctx.client_id,
                            1,
                            model_type,
                            bool(reverse_flag),
                            queue_state,
                            step_index,
                            suspended_reason,
                        )
                    )

                # Batch insert for efficiency (supports both SQLite and Postgres backends)
                self._ctx.execute_many(insert_sql, params_list, commit=False)
            return uuids  # noqa: TRY300
        except KeyError as e:
            raise InputError(f"Missing required field in bulk flashcard: {e}") from e  # noqa: TRY003
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to add flashcards in bulk: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to add flashcards in bulk: {exc}") from exc  # noqa: TRY003

    def _flashcard_visibility_filter(
        self,
        *,
        deck_id: int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
        deck_alias: str = "d",
        deck_id_column: str = "f.deck_id",
    ) -> tuple[str, tuple[Any, ...], bool]:
        """Return the visibility predicate and whether a deck join is required."""
        if workspace_id is not None:
            if deck_id is not None:
                return f"{deck_id_column} = ? AND {deck_alias}.workspace_id = ?", (deck_id, workspace_id), True
            return f"{deck_alias}.workspace_id = ?", (workspace_id,), True
        if deck_id is not None:
            return f"{deck_id_column} = ?", (deck_id,), False
        if not include_workspace_items:
            return f"{deck_alias}.workspace_id IS NULL", (), True
        return "", (), False

    @staticmethod
    def _normalize_flashcard_sqlite_query(q: str) -> str:
        return FTSQueryTranslator.normalize_query(
            q,
            'sqlite',
            sqlite_column_aliases=_FLASHCARD_SQLITE_FTS_COLUMN_ALIASES,
        )

    def list_flashcards(self,
                        deck_id: int | None = None,
                        workspace_id: str | None = None,
                        include_workspace_items: bool = False,
                        tag: str | None = None,
                        due_status: str = 'all',
                        q: str | None = None,
                        include_deleted: bool = False,
                        limit: int = 100,
                        offset: int = 0,
                        order_by: str = 'due_at') -> list[dict[str, Any]]:
        """List flashcards with filters. due_status in {'new','learning','due','all'}."""
        where_clauses = ["1=1"]
        params: list[Any] = []
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            where_clauses.append("f.client_id = ?")
            params.append(self._ctx.client_id)
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        keyword_table = self._ctx.map_table("keywords")
        if not include_deleted:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                where_clauses.append("f.deleted = FALSE")
            else:
                where_clauses.append("f.deleted = 0")
        visibility_clause, visibility_params, _ = self._flashcard_visibility_filter(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
        if visibility_clause:
            where_clauses.append(visibility_clause)
            params.extend(visibility_params)
        # due filter
        now_iso = self._ctx.utc_now_iso()
        if due_status == 'new':
            where_clauses.append("f.queue_state = ?")
            params.append("new")
        elif due_status == 'learning':
            where_clauses.append("f.queue_state IN ('learning', 'relearning') AND f.due_at IS NOT NULL AND f.due_at <= ?")
            params.append(now_iso)
        elif due_status == 'due':
            where_clauses.append("f.queue_state = 'review' AND f.due_at IS NOT NULL AND f.due_at <= ?")
            params.append(now_iso)

        # tag filter (single tag)
        join_tag = ""
        if tag:
            join_tag = f"JOIN flashcard_keywords fk ON fk.card_id = f.id JOIN {keyword_table} kw ON kw.id = fk.keyword_id"
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                join_tag += " AND kw.client_id = f.client_id"
            where_clauses.append(self._ctx.unique_text_match_sql("keywords", "kw.keyword"))
            params.append(tag)

        # FTS filter
        fts_filter = ""
        if q:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                tsquery = FTSQueryTranslator.normalize_query(q, 'postgresql')
                if not tsquery:
                    logger.debug("Flashcard query normalized to empty tsquery for input '{}'", q)
                    return []
                fts_filter = "AND f.flashcards_fts_tsv @@ to_tsquery('english', ?)"
                params.append(tsquery)
            else:
                # Normalize query for SQLite FTS5 (quotes/operators)
                norm_q = self._normalize_flashcard_sqlite_query(q)
                if not norm_q:
                    logger.debug("Flashcard query normalized to empty sqlite tsquery for input '{}'", q)
                    return []
                fts_filter = "AND f.rowid IN (SELECT rowid FROM flashcards_fts WHERE flashcards_fts MATCH ?)"
                params.append(norm_q)

        # order by
        order_sql = "ORDER BY f.due_at, f.created_at DESC" if order_by == 'due_at' else "ORDER BY f.created_at DESC"

        where_sql = " AND ".join(where_clauses)
        query = """
            SELECT f.uuid, f.deck_id, d.name AS deck_name, d.workspace_id AS workspace_id, f.front, f.back, f.notes, f.extra, f.is_cloze, f.tags_json,
                   f.source_ref_type, f.source_ref_id, f.conversation_id, f.message_id,
                   f.ef, f.interval_days, f.repetitions, f.lapses, f.due_at, f.last_reviewed_at,
                   f.queue_state, f.step_index, f.suspended_reason,
                   f.created_at, f.last_modified, f.deleted, f.client_id, f.version, f.model_type, f.reverse
            FROM flashcards f
            LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}
            {join_tag}
            WHERE {where_sql} {fts_filter}
            {order_sql}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params.extend([limit, offset])
        try:
            cursor = self._ctx.execute_query(query, tuple(params), read_only=True)
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def count_flashcards(self,
                         deck_id: int | None = None,
                         workspace_id: str | None = None,
                         include_workspace_items: bool = False,
                         tag: str | None = None,
                         due_status: str = 'all',
                         q: str | None = None,
                         include_deleted: bool = False) -> int:
        """Count flashcards matching filters. Mirrors list_flashcards filters."""
        where_clauses = ["1=1"]
        params: list[Any] = []
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            where_clauses.append("f.client_id = ?")
            params.append(self._ctx.client_id)
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        keyword_table = self._ctx.map_table("keywords")
        if not include_deleted:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                where_clauses.append("f.deleted = FALSE")
            else:
                where_clauses.append("f.deleted = 0")
        visibility_clause, visibility_params, needs_deck_join = self._flashcard_visibility_filter(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
        visibility_join = f" LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}" if needs_deck_join else ""
        if visibility_clause:
            where_clauses.append(visibility_clause)
            params.extend(visibility_params)

        now_iso = self._ctx.utc_now_iso()
        if due_status == 'new':
            where_clauses.append("f.queue_state = ?")
            params.append("new")
        elif due_status == 'learning':
            where_clauses.append("f.queue_state IN ('learning', 'relearning') AND f.due_at IS NOT NULL AND f.due_at <= ?")
            params.append(now_iso)
        elif due_status == 'due':
            where_clauses.append("f.queue_state = 'review' AND f.due_at IS NOT NULL AND f.due_at <= ?")
            params.append(now_iso)

        join_tag = ""
        if tag:
            join_tag = f"JOIN flashcard_keywords fk ON fk.card_id = f.id JOIN {keyword_table} kw ON kw.id = fk.keyword_id"
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                join_tag += " AND kw.client_id = f.client_id"
            where_clauses.append(self._ctx.unique_text_match_sql("keywords", "kw.keyword"))
            params.append(tag)

        fts_filter = ""
        if q:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                tsquery = FTSQueryTranslator.normalize_query(q, 'postgresql')
                if not tsquery:
                    return 0
                fts_filter = "AND f.flashcards_fts_tsv @@ to_tsquery('english', ?)"
                params.append(tsquery)
            else:
                norm_q = self._normalize_flashcard_sqlite_query(q)
                if not norm_q:
                    return 0
                fts_filter = "AND f.rowid IN (SELECT rowid FROM flashcards_fts WHERE flashcards_fts MATCH ?)"
                params.append(norm_q)

        where_sql = " AND ".join(where_clauses)
        # DISTINCT avoids double-counting when tag join introduces duplicates
        query = """
            SELECT COUNT(DISTINCT f.id) AS cnt
              FROM flashcards f{visibility_join}
              {join_tag}
             WHERE {where_sql} {fts_filter}
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._ctx.execute_query(query, tuple(params), read_only=True)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def list_flashcard_tag_suggestions(
        self,
        q: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List global flashcard tag suggestions with per-tag usage counts."""
        if limit <= 0:
            return []

        keyword_table = self._ctx.map_table("keywords")
        normalized_q = (q or "").strip()
        where_clauses: list[str] = []
        params: list[Any] = []

        if self._ctx.backend_type == BackendType.POSTGRESQL:
            where_clauses.extend(
                [
                    "f.deleted = FALSE",
                    "kw.deleted = FALSE",
                    "(f.deck_id IS NULL OR (d.id IS NOT NULL AND d.deleted = FALSE))",
                ]
            )
        else:
            where_clauses.extend(
                [
                    "f.deleted = 0",
                    "kw.deleted = 0",
                    "(f.deck_id IS NULL OR (d.id IS NOT NULL AND d.deleted = 0))",
                ]
            )

        if self._ctx.backend_type == BackendType.POSTGRESQL:
            where_clauses.extend(["f.client_id = ?", "kw.client_id = f.client_id", "(d.id IS NULL OR d.client_id = f.client_id)"])
            params.append(self._ctx.client_id)

        if normalized_q:
            where_clauses.append("LOWER(kw.keyword) LIKE ?")
            params.append(f"%{normalized_q.lower()}%")

        order_keyword = self._ctx.case_insensitive_order_expression("kw.keyword")
        where_sql = " AND ".join(where_clauses)
        query = """
            SELECT
                kw.keyword AS tag,
                COUNT(DISTINCT f.id) AS usage_count
            FROM flashcards f
            JOIN flashcard_keywords fk ON fk.card_id = f.id
            JOIN {keyword_table} kw ON kw.id = fk.keyword_id
            LEFT JOIN decks d ON d.id = f.deck_id
            WHERE {where_sql}
            GROUP BY kw.keyword
            ORDER BY usage_count DESC, {order_keyword}
            LIMIT ?
        """.format_map(locals())  # nosec B608
        params.append(limit)

        try:
            cursor = self._ctx.execute_query(query, tuple(params), read_only=True)
            return [
                {"tag": str(row["tag"]), "count": int(row["usage_count"] or 0)}
                for row in cursor.fetchall()
            ]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def get_flashcards_by_uuids(self, uuids: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple flashcards by UUIDs in one query. Order is not guaranteed; caller can reorder."""
        if not uuids:
            return []
        placeholders = ",".join(["?"] * len(uuids))
        deleted_clause = "f.deleted = FALSE" if self._ctx.backend_type == BackendType.POSTGRESQL else "f.deleted = 0"
        owner_filter, owner_params = self._flashcard_owner_filter("f")
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        query = f"""
            SELECT f.uuid, f.deck_id, d.name AS deck_name, d.workspace_id AS workspace_id, f.front, f.back, f.notes, f.extra, f.is_cloze, f.tags_json,
                   f.source_ref_type, f.source_ref_id, f.conversation_id, f.message_id,
                   f.ef, f.interval_days, f.repetitions, f.lapses, f.due_at, f.last_reviewed_at,
                   f.queue_state, f.step_index, f.suspended_reason,
                   f.created_at, f.last_modified, f.deleted, f.client_id, f.version, f.model_type, f.reverse
              FROM flashcards f
              LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}
             WHERE f.uuid IN ({placeholders}) AND {deleted_clause}{owner_filter}
        """  # nosec B608
        try:
            cursor = self._ctx.execute_query(query, (*uuids, *owner_params), read_only=True)
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def _srs_sm2_update(self, ef: float, interval_days: int, repetitions: int, lapses: int, rating: int) -> dict[str, Any]:
        """
        Apply SM-2 style scheduling update and return new values.
        rating: 0-5 (Anki scale). q<3 counts as lapse.
        """
        q = max(0, min(5, int(rating)))
        was_lapse = q < 3
        if was_lapse:
            lapses += 1
            repetitions = 0
            interval_days = 1
            # EF unchanged for lapse (keep, clamp)
        else:
            # Ease factor update
            ef = ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
            ef = max(1.3, ef)
            repetitions += 1
            if repetitions == 1:
                interval_days = 1
            elif repetitions == 2:
                interval_days = 6
            else:
                interval_days = int(round(interval_days * ef)) if interval_days > 0 else 6
        return {
            'ef': float(ef),
            'interval_days': int(interval_days),
            'repetitions': int(repetitions),
            'lapses': int(lapses),
            'was_lapse': was_lapse,
        }

    def _normalize_flashcard_review_mode(self, review_mode: str) -> str:
        normalized = str(review_mode or "").strip().lower()
        if normalized not in _FLASHCARD_REVIEW_MODES:
            raise InputError("review_mode must be one of: due, cram")  # noqa: TRY003
        return normalized

    def _normalize_flashcard_review_session_status(self, status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized not in _FLASHCARD_REVIEW_SESSION_STATUSES:
            raise InputError("status must be one of: active, completed, abandoned")  # noqa: TRY003
        return normalized

    @staticmethod
    def _normalize_flashcard_review_session_source_bundle(source_bundle_value: Any) -> list[dict[str, Any]]:
        if isinstance(source_bundle_value, dict) and isinstance(source_bundle_value.get("items"), list):
            source_bundle_value = source_bundle_value.get("items")

        if isinstance(source_bundle_value, list):
            return [dict(entry) for entry in source_bundle_value if isinstance(entry, dict)]
        if isinstance(source_bundle_value, dict):
            return [dict(source_bundle_value)]
        return []

    def _deserialize_flashcard_review_session_row(self, row: Any) -> dict[str, Any] | None:
        item = self._ctx.deserialize_row_fields(row, _FLASHCARD_REVIEW_SESSION_JSON_FIELDS)
        if not item:
            return None

        if item.get("cards_reviewed") is not None:
            item["cards_reviewed"] = int(item["cards_reviewed"])
        if item.get("correct_count") is not None:
            item["correct_count"] = int(item["correct_count"])
        if item.get("study_pack_id") is not None:
            item["study_pack_id"] = int(item["study_pack_id"])
        item["deck_name_snapshot"] = self._ctx.normalize_nullable_text(item.get("deck_name_snapshot"))

        normalized_bundle = self._normalize_flashcard_review_session_source_bundle(item.get("source_bundle_json"))
        item["source_bundle_json"] = normalized_bundle or None
        item["source_bundle"] = normalized_bundle
        return item

    def _get_flashcard_review_session_deck_name_snapshot_for_conn(
        self,
        conn: Any,
        deck_id: int | None,
    ) -> str | None:
        """Return the current deck name to snapshot when a review session starts."""
        if deck_id is None:
            return None
        owner_filter, owner_params = self._flashcard_owner_filter()
        row = conn.execute("SELECT name FROM decks WHERE id = ?" + owner_filter, (int(deck_id), *owner_params)).fetchone()  # nosec B608 -- Fixed owner clauses; all data values are bound.
        if not row:
            return None
        try:
            raw_name = row["name"]
        except (KeyError, IndexError, TypeError):
            raw_name = row[0]
        return self._ctx.normalize_nullable_text(raw_name)

    def _get_flashcard_review_session_row_for_conn(self, conn: Any, session_id: int) -> dict[str, Any] | None:
        owner_filter, owner_params = self._flashcard_owner_filter()
        row = conn.execute(
            f"""
            SELECT id, deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                   started_at, last_activity_at, completed_at, client_id,
                   cards_reviewed, correct_count, source_bundle_json, study_pack_id
              FROM flashcard_review_sessions
             WHERE id = ?{owner_filter}
            """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
            (int(session_id), *owner_params),
        ).fetchone()
        return self._deserialize_flashcard_review_session_row(row) if row else None

    def _get_flashcard_review_session_reconstructed_aggregates(self, conn: Any, session_id: int) -> tuple[int, int]:
        owner_filter, owner_params = self._flashcard_owner_filter()
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS cards_reviewed,
                   COALESCE(SUM(CASE WHEN NOT was_lapse AND rating <> 0 THEN 1 ELSE 0 END), 0) AS correct_count
              FROM flashcard_reviews
             WHERE review_session_id = ?{owner_filter}
            """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
            (int(session_id), *owner_params),
        ).fetchone()
        if not row:
            return 0, 0
        return int(row["cards_reviewed"] or 0), int(row["correct_count"] or 0)

    def _repair_flashcard_review_session_aggregates_if_unchanged(
        self,
        conn: Any,
        session_id: int,
        *,
        expected_cards_reviewed: int | None,
        expected_correct_count: int | None,
        repaired_cards_reviewed: int,
        repaired_correct_count: int,
    ) -> bool:
        owner_filter, owner_params = self._flashcard_owner_filter()
        cursor = conn.execute(
            f"""
            UPDATE flashcard_review_sessions
               SET cards_reviewed = ?,
                   correct_count = ?
             WHERE id = ?{owner_filter}
               AND ((cards_reviewed IS NULL AND ? IS NULL) OR cards_reviewed = ?)
               AND ((correct_count IS NULL AND ? IS NULL) OR correct_count = ?)
            """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
            (
                repaired_cards_reviewed,
                repaired_correct_count,
                int(session_id),
                *owner_params,
                expected_cards_reviewed,
                expected_cards_reviewed,
                expected_correct_count,
                expected_correct_count,
            ),
        )
        return max(0, int(getattr(cursor, "rowcount", 0) or 0)) > 0

    def abandon_stale_flashcard_review_sessions(self, *, scope_key: str | None = None) -> int:
        """Mark active review sessions as abandoned when inactive for over 30 minutes."""
        cutoff = to_iso_z(datetime.now(timezone.utc) - _FLASHCARD_REVIEW_SESSION_TIMEOUT) or self._ctx.utc_now_iso()
        query = (
            """
            UPDATE flashcard_review_sessions
               SET status = 'abandoned'
             WHERE status = 'active'
               AND last_activity_at < ?
            """
        )
        owner_filter, owner_params = self._flashcard_owner_filter()
        query += owner_filter
        params: list[Any] = [cutoff, *owner_params]
        if scope_key:
            query += " AND scope_key = ?"
            params.append(scope_key)
        with self._ctx.transaction() as conn:
            cursor = conn.execute(query, tuple(params))
            return max(0, int(getattr(cursor, "rowcount", 0) or 0))

    def list_flashcard_review_sessions(
        self,
        *,
        deck_id: int | None = None,
        scope_key: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List persisted review sessions ordered from newest to oldest."""
        self.abandon_stale_flashcard_review_sessions(scope_key=scope_key)
        query_parts = [
            """
            SELECT id, deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                   started_at, last_activity_at, completed_at, client_id,
                   cards_reviewed, correct_count, source_bundle_json, study_pack_id
              FROM flashcard_review_sessions
             WHERE 1 = 1
            """
        ]
        owner_filter, owner_params = self._flashcard_owner_filter()
        query_parts.append(owner_filter)
        params: list[Any] = list(owner_params)
        if deck_id is not None:
            query_parts.append("AND deck_id = ?")
            params.append(int(deck_id))
        if scope_key:
            query_parts.append("AND scope_key = ?")
            params.append(scope_key)
        if status:
            query_parts.append("AND status = ?")
            params.append(self._normalize_flashcard_review_session_status(status))
        query_parts.append("ORDER BY last_activity_at DESC, started_at DESC, id DESC LIMIT ?")
        params.append(max(1, int(limit)))
        cursor = self._ctx.execute_query(" ".join(query_parts), tuple(params), read_only=True)
        return [
            session
            for row in cursor.fetchall()
            if (session := self._deserialize_flashcard_review_session_row(row)) is not None
        ]

    def get_or_create_flashcard_review_session(
        self,
        *,
        deck_id: int | None,
        review_mode: str,
        tag_filter: str | None,
        scope_key: str,
    ) -> dict[str, Any]:
        """Return the authoritative active session for a scope or create a new one."""
        normalized_scope_key = str(scope_key or "").strip()
        if not normalized_scope_key:
            raise InputError("scope_key is required")  # noqa: TRY003
        normalized_mode = self._normalize_flashcard_review_mode(review_mode)
        normalized_tag_filter = self._ctx.normalize_nullable_text(tag_filter)
        self.abandon_stale_flashcard_review_sessions(scope_key=normalized_scope_key)
        owner_filter, owner_params = self._flashcard_owner_filter()
        now = self._ctx.utc_now_iso()

        try:
            with self._ctx.transaction() as conn:
                self._validate_flashcard_deck_locked(conn, deck_id)
                rows = conn.execute(
                    f"""
                    SELECT id, deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                           started_at, last_activity_at, completed_at, client_id,
                           cards_reviewed, correct_count, source_bundle_json, study_pack_id
                      FROM flashcard_review_sessions
                     WHERE scope_key = ? AND status = 'active'{owner_filter}
                     ORDER BY last_activity_at DESC, started_at DESC, id DESC
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (normalized_scope_key, *owner_params),
                ).fetchall()
                if rows:
                    authoritative = self._deserialize_flashcard_review_session_row(rows[0]) or {}
                    if len(rows) > 1:
                        duplicate_ids = [int(row["id"]) for row in rows[1:]]
                        placeholders = ", ".join("?" for _ in duplicate_ids)
                        conn.execute(
                            f"UPDATE flashcard_review_sessions SET status = 'abandoned' WHERE id IN ({placeholders}){owner_filter}",  # nosec B608
                            (*duplicate_ids, *owner_params),
                        )
                    conn.execute(
                        f"""
                        UPDATE flashcard_review_sessions
                           SET last_activity_at = ?
                        WHERE id = ?{owner_filter}
                        """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (now, int(authoritative["id"]), *owner_params),
                    )
                    authoritative["last_activity_at"] = now
                    return authoritative

                deck_name_snapshot = self._get_flashcard_review_session_deck_name_snapshot_for_conn(
                    conn,
                    deck_id,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(
                        """
                        INSERT INTO flashcard_review_sessions(
                            deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                            started_at, last_activity_at, completed_at, cards_reviewed,
                            correct_count, source_bundle_json, study_pack_id, client_id
                        )
                        VALUES(?, ?, ?, ?, ?, 'active', ?, ?, NULL, 0, 0, NULL, NULL, ?)
                        RETURNING id
                        """,
                        (
                            int(deck_id) if deck_id is not None else None,
                            deck_name_snapshot,
                            normalized_mode,
                            normalized_tag_filter,
                            normalized_scope_key,
                            now,
                            now,
                            self._ctx.client_id,
                        ),
                    )
                    row = cursor.fetchone()
                    session_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(
                        """
                        INSERT INTO flashcard_review_sessions(
                            deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                            started_at, last_activity_at, completed_at, cards_reviewed,
                            correct_count, source_bundle_json, study_pack_id, client_id
                        )
                        VALUES(?, ?, ?, ?, ?, 'active', ?, ?, NULL, 0, 0, NULL, NULL, ?)
                        """,
                        (
                            int(deck_id) if deck_id is not None else None,
                            deck_name_snapshot,
                            normalized_mode,
                            normalized_tag_filter,
                            normalized_scope_key,
                            now,
                            now,
                            self._ctx.client_id,
                        ),
                    )
                    session_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
                if session_id is None:
                    raise CharactersRAGDBError("Failed to determine flashcard review session ID after insert")  # noqa: TRY003
                row = conn.execute(
                    f"""
                    SELECT id, deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                           started_at, last_activity_at, completed_at, client_id,
                           cards_reviewed, correct_count, source_bundle_json, study_pack_id
                      FROM flashcard_review_sessions
                     WHERE id = ?{owner_filter}
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (session_id, *owner_params),
                ).fetchone()
                return self._deserialize_flashcard_review_session_row(row) or {}
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to get or create flashcard review session: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to get or create flashcard review session: {exc}") from exc  # noqa: TRY003

    def get_flashcard_review_session(self, session_id: int) -> dict[str, Any] | None:
        """Fetch a single flashcard review session by id, or None if not found."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        cursor = self._ctx.execute_query(
            f"""
            SELECT id, deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                   started_at, last_activity_at, completed_at, client_id,
                   cards_reviewed, correct_count, source_bundle_json, study_pack_id
              FROM flashcard_review_sessions
             WHERE id = ?{owner_filter}
            """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
            (int(session_id), *owner_params),
            read_only=True,
        )
        row = cursor.fetchone()
        return self._deserialize_flashcard_review_session_row(row) if row else None

    def mark_flashcard_review_session_completed(self, session_id: int) -> dict[str, Any]:
        """Mark a persisted review session as completed."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        now = self._ctx.utc_now_iso()
        try:
            cursor = self._ctx.execute_query(
                f"""
                UPDATE flashcard_review_sessions
                   SET status = 'completed',
                       last_activity_at = ?,
                       completed_at = ?
                 WHERE id = ?{owner_filter}
                """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                (now, now, int(session_id), *owner_params),
                commit=True,
            )
            if getattr(cursor, "rowcount", 0) == 0:
                raise ConflictError("Flashcard review session not found", entity="flashcard_review_sessions", entity_id=session_id)  # noqa: TRY003
            row = self._ctx.execute_query(
                f"""
                SELECT id, deck_id, deck_name_snapshot, review_mode, tag_filter, scope_key, status,
                       started_at, last_activity_at, completed_at, client_id,
                       cards_reviewed, correct_count, source_bundle_json, study_pack_id
                  FROM flashcard_review_sessions
                 WHERE id = ?{owner_filter}
                """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                (int(session_id), *owner_params),
                read_only=True,
            ).fetchone()
            return self._deserialize_flashcard_review_session_row(row) or {}
        except CharactersRAGDBError:
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            raise CharactersRAGDBError(f"Failed to mark flashcard review session completed: {exc}") from exc  # noqa: TRY003

    def get_flashcard_review_session_rollup(
        self,
        session_id: int,
        *,
        repair_session_aggregates: bool = True,
    ) -> dict[str, Any] | None:
        """Return a merged session view that trusts review rows when cached aggregates drift."""
        try:
            with self._ctx.transaction() as conn:
                session = self._get_flashcard_review_session_row_for_conn(conn, int(session_id))
                if not session:
                    return None

                reconstructed_cards_reviewed, reconstructed_correct_count = (
                    self._get_flashcard_review_session_reconstructed_aggregates(conn, int(session_id))
                )
                session_cards_reviewed = session.get("cards_reviewed")
                session_correct_count = session.get("correct_count")
                session_aggregates_match = (
                    session_cards_reviewed is not None
                    and session_correct_count is not None
                    and session_cards_reviewed >= 0
                    and session_correct_count >= 0
                    and session_correct_count <= session_cards_reviewed
                    and session_cards_reviewed == reconstructed_cards_reviewed
                    and session_correct_count == reconstructed_correct_count
                )

                if session_aggregates_match:
                    session["aggregate_source"] = "session"
                    return session

                if repair_session_aggregates:
                    repaired = self._repair_flashcard_review_session_aggregates_if_unchanged(
                        conn,
                        int(session_id),
                        expected_cards_reviewed=session_cards_reviewed,
                        expected_correct_count=session_correct_count,
                        repaired_cards_reviewed=reconstructed_cards_reviewed,
                        repaired_correct_count=reconstructed_correct_count,
                    )
                    if repaired:
                        session["cards_reviewed"] = reconstructed_cards_reviewed
                        session["correct_count"] = reconstructed_correct_count
                        session["aggregate_source"] = "reconstructed"
                        return session

                    session = self._get_flashcard_review_session_row_for_conn(conn, int(session_id))
                    if not session:
                        return None
                    reconstructed_cards_reviewed, reconstructed_correct_count = (
                        self._get_flashcard_review_session_reconstructed_aggregates(conn, int(session_id))
                    )
                    session_cards_reviewed = session.get("cards_reviewed")
                    session_correct_count = session.get("correct_count")
                    session_aggregates_match = (
                        session_cards_reviewed is not None
                        and session_correct_count is not None
                        and session_cards_reviewed >= 0
                        and session_correct_count >= 0
                        and session_correct_count <= session_cards_reviewed
                        and session_cards_reviewed == reconstructed_cards_reviewed
                        and session_correct_count == reconstructed_correct_count
                    )
                    if session_aggregates_match:
                        session["aggregate_source"] = "session"
                        return session

                session["cards_reviewed"] = reconstructed_cards_reviewed
                session["correct_count"] = reconstructed_correct_count
                session["aggregate_source"] = "reconstructed"
                return session
        except CharactersRAGDBError:
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            raise CharactersRAGDBError(f"Failed to fetch flashcard review session rollup: {exc}") from exc  # noqa: TRY003

    def get_flashcard_reviewed_cards(self, session_id: int) -> list[dict[str, Any]]:
        """Return reviewed flashcard metadata for a session in review order."""
        owner_filter, owner_params = self._flashcard_owner_filter("f")
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            owner_filter += " AND fr.client_id = f.client_id AND EXISTS (SELECT 1 FROM flashcard_review_sessions s WHERE s.id = fr.review_session_id AND s.client_id = f.client_id)"
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        cursor = self._ctx.execute_query(
            f"""
            SELECT f.uuid,
                   MAX(f.tags_json) AS tags_json,
                   MAX(f.source_ref_type) AS source_ref_type,
                   MAX(f.source_ref_id) AS source_ref_id,
                   MAX(d.name) AS deck_name
              FROM flashcard_reviews fr
              JOIN flashcards f ON f.id = fr.card_id
              LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}
             WHERE fr.review_session_id = ?{owner_filter}
             GROUP BY f.uuid
             ORDER BY MIN(fr.id)
            """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
            (int(session_id), *owner_params),
            read_only=True,
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_flashcard_review(self, card_uuid: str) -> dict[str, Any] | None:
        """Return the most recent review row for a flashcard, including nullable session linkage."""
        owner_filter, owner_params = self._flashcard_owner_filter("f")
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            owner_filter += " AND fr.client_id = f.client_id"
        cursor = self._ctx.execute_query(
            f"""
            SELECT fr.id, fr.card_id, fr.reviewed_at, fr.rating, fr.answer_time_ms,
                   fr.scheduled_interval_days, fr.new_ef, fr.new_repetitions, fr.was_lapse,
                   fr.client_id, fr.scheduler_type, fr.previous_queue_state, fr.next_queue_state,
                   fr.previous_due_at, fr.next_due_at, fr.review_session_id
              FROM flashcard_reviews fr
              JOIN flashcards f ON f.id = fr.card_id
             WHERE f.uuid = ?{owner_filter}
             ORDER BY fr.id DESC
             LIMIT 1
            """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
            (card_uuid, *owner_params),
            read_only=True,
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def review_flashcard(
        self,
        card_uuid: str,
        rating: int,
        answer_time_ms: int | None = None,
        review_session_id: int | None = None,
        *,
        review_mode: str | None = None,
        review_deck_id: int | None = None,
        review_tag_filter: str | None = None,
    ) -> dict[str, Any]:
        """Review a card within a validated session and update its schedule atomically.

        An explicit review_mode selects the caller's queue scope. A null
        review_deck_id then means all decks; omitted context keeps the legacy
        per-card deck scope. Acknowledged session IDs must still be active and
        match that explicit context before any review or scheduling write.
        """
        owner_filter, owner_params = self._flashcard_owner_filter()
        card_owner_filter, _ = self._flashcard_owner_filter("f")
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        now_dt = datetime.now(timezone.utc)
        now = to_iso_z(now_dt) or self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                card = conn.execute(
                    f"""
                    SELECT f.id, f.uuid, f.deck_id, f.ef, f.interval_days, f.repetitions, f.lapses,
                           f.due_at, f.last_reviewed_at, f.queue_state, f.step_index, f.suspended_reason,
                           f.scheduler_state_json,
                           COALESCE(d.scheduler_type, 'sm2_plus') AS scheduler_type,
                           COALESCE(d.scheduler_settings_json, ?) AS scheduler_settings_json
                      FROM flashcards f
                      LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}
                     WHERE f.uuid = ? AND f.deleted = 0{card_owner_filter}
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (scheduler_settings_to_json(None), card_uuid, *owner_params),
                ).fetchone()
                if not card:
                    raise ConflictError(
                        "Flashcard not found", entity="flashcards", identifier=card_uuid
                    )  # noqa: TRY003
                card_id = int(card["id"])
                card_deck_id = int(card["deck_id"]) if card["deck_id"] is not None else None
                explicit_scope = review_mode is not None
                scope_mode = self._normalize_flashcard_review_mode(review_mode if review_mode is not None else "due")
                scope_deck_id = review_deck_id if explicit_scope else card_deck_id
                scope_tag = self._ctx.normalize_nullable_text(review_tag_filter) if explicit_scope else None
                scope_key = (
                    f"{scope_mode}:deck:{scope_deck_id}" if scope_deck_id is not None else f"{scope_mode}:global"
                )
                if scope_tag:
                    scope_key += f":tag:{scope_tag}"
                if explicit_scope and scope_deck_id is not None and scope_deck_id != card_deck_id:
                    raise InputError("Review context must include the flashcard deck")  # noqa: TRY003
                if scope_tag:
                    tag_owner_filter, tag_owner_params = self._flashcard_owner_filter("kw")
                    keyword_table = self._ctx.map_table("keywords")
                    tag_match_sql = self._ctx.unique_text_match_sql("keywords", "kw.keyword")
                    matching_tag = conn.execute(
                        f"""
                        SELECT 1 FROM flashcard_keywords fk
                        JOIN {keyword_table} kw ON kw.id = fk.keyword_id
                        WHERE fk.card_id = ? AND {tag_match_sql}{tag_owner_filter}
                        """,  # nosec B608 -- keyword table comes from the backend mapping
                        (card_id, scope_tag, *tag_owner_params),
                    ).fetchone()
                    if not matching_tag:
                        raise InputError("Flashcard does not match the review tag filter")  # noqa: TRY003
                resolved_review_session_id = int(review_session_id) if review_session_id is not None else None
                if resolved_review_session_id is None:
                    auto_session = self.get_or_create_flashcard_review_session(
                        deck_id=scope_deck_id,
                        review_mode=scope_mode,
                        tag_filter=scope_tag,
                        scope_key=scope_key,
                    )
                    resolved_review_session_id = int(auto_session["id"])
                if resolved_review_session_id is not None:
                    session_row = conn.execute(
                        f"""
                        SELECT id, deck_id, status, review_mode, tag_filter, scope_key,
                               CASE WHEN last_activity_at < ? THEN 1 ELSE 0 END AS is_stale
                          FROM flashcard_review_sessions
                         WHERE id = ?{owner_filter}
                        """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (to_iso_z(now_dt - _FLASHCARD_REVIEW_SESSION_TIMEOUT), resolved_review_session_id, *owner_params),
                    ).fetchone()
                    if not session_row:
                        raise ConflictError(
                            "Flashcard review session not found",
                            entity="flashcard_review_sessions",
                            identifier=resolved_review_session_id,
                        )
                    session_deck_id = int(session_row["deck_id"]) if session_row["deck_id"] is not None else None
                    if explicit_scope and (
                        session_deck_id != scope_deck_id
                        or session_row["review_mode"] != scope_mode
                        or self._ctx.normalize_nullable_text(session_row["tag_filter"]) != scope_tag
                        or session_row["scope_key"] != scope_key
                    ):
                        raise InputError("review_session_id must match the requested review context")  # noqa: TRY003
                    if not explicit_scope and session_deck_id != card_deck_id:
                        raise InputError("review_session_id must match the flashcard deck scope")  # noqa: TRY003
                    if str(session_row["status"] or "").strip().lower() != "active" or session_row["is_stale"]:
                        raise ConflictError(
                            "Flashcard review session is not active",
                            entity="flashcard_review_sessions",
                            identifier=resolved_review_session_id,
                        )
                previous_queue_state = coerce_queue_state(dict(card))
                previous_due_at = self._ctx.normalize_nullable_text(card["due_at"])
                scheduler_type = _coerce_scheduler_type(card["scheduler_type"])
                scheduler_settings = _normalize_scheduler_settings_envelope(card["scheduler_settings_json"])
                upd = _simulate_scheduler_review_transition(
                    dict(card),
                    scheduler_type=scheduler_type,
                    scheduler_settings_envelope=scheduler_settings,
                    rating=int(rating),
                    now=now_dt,
                )

                conn.execute(
                    f"""
                    UPDATE flashcards
                       SET ef = ?, interval_days = ?, repetitions = ?, lapses = ?,
                           due_at = ?, last_reviewed_at = ?, queue_state = ?, step_index = ?,
                           suspended_reason = ?, scheduler_state_json = ?, last_modified = ?, version = version + 1, client_id = ?
                     WHERE id = ? AND deleted = 0{owner_filter}
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (
                        upd["ef"],
                        upd["interval_days"],
                        upd["repetitions"],
                        upd["lapses"],
                        upd["due_at"],
                        upd["last_reviewed_at"],
                        upd["queue_state"],
                        upd["step_index"],
                        upd["suspended_reason"],
                        upd.get("scheduler_state_json", "{}"),
                        now,
                        self._ctx.client_id,
                        card_id,
                        *owner_params,
                    )
                )
                conn.execute(
                    """
                    INSERT INTO flashcard_reviews(
                        card_id, reviewed_at, rating, answer_time_ms, scheduled_interval_days,
                        new_ef, new_repetitions, was_lapse, client_id, scheduler_type,
                        previous_queue_state, next_queue_state, previous_due_at, next_due_at, review_session_id
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        now,
                        int(rating),
                        answer_time_ms,
                        upd["interval_days"] if upd["queue_state"] == "review" else None,
                        upd["ef"],
                        upd["repetitions"],
                        bool(upd["was_lapse"]),
                        self._ctx.client_id,
                        scheduler_type,
                        previous_queue_state,
                        upd["queue_state"],
                        previous_due_at,
                        upd["due_at"],
                        resolved_review_session_id,
                    )
                )
                if resolved_review_session_id is not None:
                    # Again during learning is incorrect even before a mature lapse.
                    correct_increment = int(not upd["was_lapse"] and int(rating) != 0)
                    conn.execute(
                        f"""
                        UPDATE flashcard_review_sessions
                           SET last_activity_at = ?,
                               cards_reviewed = COALESCE(cards_reviewed, 0) + 1,
                               correct_count = COALESCE(correct_count, 0) + ?
                         WHERE id = ?{owner_filter}
                        """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (now, correct_increment, resolved_review_session_id, *owner_params),
                    )
                updated = conn.execute(
                    f"""
                    SELECT uuid, ef, interval_days, repetitions, lapses, due_at, last_reviewed_at,
                           last_modified, version, queue_state, step_index, suspended_reason, scheduler_state_json
                      FROM flashcards
                     WHERE id = ?{owner_filter}
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (card_id, *owner_params)
                ).fetchone()
                updated_payload = dict(updated)
                updated_payload["scheduler_type"] = scheduler_type
                updated_payload["next_intervals"] = _build_scheduler_next_interval_previews(
                    updated_payload,
                    scheduler_type=scheduler_type,
                    scheduler_settings_envelope=scheduler_settings,
                    now=now_dt,
                )
                updated_payload["review_session_id"] = resolved_review_session_id
                return updated_payload
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to review flashcard: {e}") from e  # noqa: TRY003

    def get_flashcard_analytics_summary(
        self,
        deck_id: int | None = None,
        workspace_id: str | None = None,
        include_workspace_items: bool = False,
    ) -> dict[str, Any]:
        """Return review analytics summary and per-deck progress counts."""
        now_dt = datetime.now(timezone.utc)
        today_start_dt = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_start_dt = today_start_dt + timedelta(days=1)
        now_iso = now_dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        today_start_iso = today_start_dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        tomorrow_start_iso = tomorrow_start_dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        normalized_deck_id = int(deck_id) if deck_id is not None else None
        visibility_clause, visibility_params, needs_deck_join = self._flashcard_visibility_filter(
            deck_id=normalized_deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        visibility_join = f" LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}" if needs_deck_join else ""
        visibility_suffix = f" AND {visibility_clause}" if visibility_clause else ""
        owner_filter, owner_params = self._flashcard_owner_filter("f")
        visibility_suffix += owner_filter
        visibility_params = (*visibility_params, *owner_params)
        metrics_join = visibility_join or f" LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}"
        metrics_suffix = f"{visibility_suffix} AND (d.id IS NULL OR d.deleted = 0)"
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            metrics_suffix += " AND fr.client_id = f.client_id"
        deck_rows_clause, deck_rows_params, _ = self._flashcard_visibility_filter(
            deck_id=normalized_deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            deck_alias="d",
            deck_id_column="d.id",
        )
        deck_rows_suffix = f" AND {deck_rows_clause}" if deck_rows_clause else ""
        deck_owner_filter, deck_owner_params = self._flashcard_owner_filter("d")
        deck_rows_suffix += deck_owner_filter
        deck_rows_params = (*deck_rows_params, *deck_owner_params)

        try:
            # Trust stored scheduler outcomes, including historical rows. Hard is
            # successful recall; ratings alone cannot identify mature lapses.
            daily_row = self._ctx.execute_query(
                """
                SELECT
                    COUNT(*) AS reviewed_today,
                    SUM(CASE WHEN fr.was_lapse THEN 1 ELSE 0 END) AS lapses_today,
                    AVG(answer_time_ms) AS avg_answer_time_ms_today
                FROM flashcard_reviews fr
                JOIN flashcards f ON f.id = fr.card_id AND f.deleted = 0
                {metrics_join}
                WHERE fr.reviewed_at >= ? AND fr.reviewed_at < ?{metrics_suffix}
                """.format_map(locals()),  # nosec B608
                (today_start_iso, tomorrow_start_iso, *visibility_params),
                read_only=True,
            ).fetchone()

            reviewed_today = int((daily_row["reviewed_today"] if daily_row else 0) or 0)
            lapses_today = int((daily_row["lapses_today"] if daily_row else 0) or 0)
            avg_answer_time_ms_today = (
                float(daily_row["avg_answer_time_ms_today"])
                if daily_row and daily_row["avg_answer_time_ms_today"] is not None
                else None
            )
            retention_rate_today = None
            lapse_rate_today = None
            if reviewed_today > 0:
                lapse_rate_today = (lapses_today / reviewed_today) * 100.0
                retention_rate_today = 100.0 - lapse_rate_today

            # UTC day streak: count consecutive days ending at today with >=1 review
            review_day_sql = (
                "(fr.reviewed_at AT TIME ZONE 'UTC')::date"
                if self._ctx.backend_type == BackendType.POSTGRESQL
                else "substr(fr.reviewed_at, 1, 10)"
            )
            day_rows = self._ctx.execute_query(
                """
                SELECT DISTINCT {review_day_sql} AS review_day
                FROM flashcard_reviews fr
                JOIN flashcards f ON f.id = fr.card_id AND f.deleted = 0
                {metrics_join}
                WHERE 1 = 1{metrics_suffix}
                ORDER BY review_day DESC
                LIMIT 400
                """.format_map(locals()),  # nosec B608
                visibility_params,
                read_only=True,
            ).fetchall()
            reviewed_days = {
                str(row["review_day"])
                for row in day_rows
                if row and row["review_day"] is not None
            }
            streak = 0
            cursor_day = today_start_dt.date()
            while cursor_day.isoformat() in reviewed_days:
                streak += 1
                cursor_day = cursor_day - timedelta(days=1)

            # Per-deck progress counts
            deck_rows = self._ctx.execute_query(
                """
                SELECT
                    d.id AS deck_id,
                    d.name AS deck_name,
                    COUNT(f.id) AS total_count,
                    SUM(CASE WHEN f.queue_state = 'new' THEN 1 ELSE 0 END) AS new_count,
                    SUM(CASE WHEN f.queue_state IN ('learning', 'relearning') THEN 1 ELSE 0 END) AS learning_count,
                    SUM(CASE WHEN f.queue_state IN ('learning', 'relearning', 'review') AND f.due_at IS NOT NULL AND f.due_at <= ? THEN 1 ELSE 0 END) AS due_count,
                    SUM(CASE WHEN f.queue_state = 'review' AND f.interval_days >= ? THEN 1 ELSE 0 END) AS mature_count
                FROM decks d
                LEFT JOIN flashcards f
                    ON f.deck_id = d.id
                   AND f.deleted = 0{deck_owner_join}
                WHERE d.deleted = 0{deck_rows_suffix}
                GROUP BY d.id, d.name
                ORDER BY d.name ASC
                """.format_map(locals()),  # nosec B608
                (now_iso, MATURE_INTERVAL_DAYS, *deck_rows_params),
                read_only=True,
            ).fetchall()

            decks = []
            for row in deck_rows:
                decks.append(
                    {
                        "deck_id": int(row["deck_id"]),
                        "deck_name": str(row["deck_name"] or f"Deck {row['deck_id']}"),
                        "total": int(row["total_count"] or 0),
                        "new": int(row["new_count"] or 0),
                        "learning": int(row["learning_count"] or 0),
                        "due": int(row["due_count"] or 0),
                        "mature": int(row["mature_count"] or 0),
                    }
                )

            return {
                "reviewed_today": reviewed_today,
                "retention_rate_today": retention_rate_today,
                "lapse_rate_today": lapse_rate_today,
                "avg_answer_time_ms_today": avg_answer_time_ms_today,
                "study_streak_days": streak,
                "generated_at": now_iso,
                "decks": decks,
            }
        except CharactersRAGDBError:  # noqa: TRY203
            raise
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            raise CharactersRAGDBError(f"Failed to build flashcard analytics summary: {e}") from e  # noqa: TRY003

    def export_flashcards_csv(self,
                              deck_id: int | None = None,
                              workspace_id: str | None = None,
                              include_workspace_items: bool = False,
                              tag: str | None = None,
                              q: str | None = None,
                              *,
                              delimiter: str = '\t',
                              include_header: bool = False,
                              extended_header: bool = False) -> bytes:
        """Export flashcards to delimited text. Columns: Deck, Front, Back, Tags, Notes.
        Args:
            delimiter: field separator (default tab)
            include_header: include a header row
        """
        rows = self.list_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            tag=tag,
            q=q,
            due_status='all',
            include_deleted=False,
            limit=100000,
            offset=0,
        )
        output_lines: list[str] = []
        if include_header:
            base_cols = ["Deck", "Front", "Back", "Tags", "Notes"]
            if extended_header:
                base_cols += ["Extra", "Reverse"]
            output_lines.append(delimiter.join(base_cols))
        for r in rows:
            deck_name = r.get('deck_name') or ''
            front = r.get('front') or ''
            back = r.get('back') or ''
            tags = ''
            if r.get('tags_json'):
                try:
                    tags_list = json.loads(r['tags_json'])
                    if isinstance(tags_list, list):
                        tags = " ".join(str(t) for t in tags_list)
                except _CHACHA_NONCRITICAL_EXCEPTIONS:
                    tags = ''
            notes = r.get('notes') or ''
            extra = r.get('extra') or ''
            reverse = 'true' if bool(r.get('reverse')) else 'false'
            # Escape delimiter/tabs/newlines minimally
            def esc(s: str) -> str:
                return s.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace(delimiter, ' ').strip()
            row = [esc(deck_name), esc(front), esc(back), esc(tags), esc(notes)]
            if extended_header:
                row += [esc(extra), reverse]
            output_lines.append(delimiter.join(row))
        csv_bytes = ("\n".join(output_lines) + ("\n" if output_lines else "")).encode('utf-8')
        return csv_bytes

    def get_flashcard(self, card_uuid: str) -> dict[str, Any] | None:
        """Fetch a single flashcard by uuid (active only)."""
        owner_filter, owner_params = self._flashcard_owner_filter("f")
        deck_owner_join = " AND d.client_id = f.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        query = f"""
            SELECT f.uuid, f.deck_id, d.name AS deck_name, d.workspace_id AS workspace_id, f.front, f.back, f.notes, f.extra, f.is_cloze, f.tags_json,
                   f.source_ref_type, f.source_ref_id, f.conversation_id, f.message_id,
                   f.ef, f.interval_days, f.repetitions, f.lapses, f.due_at, f.last_reviewed_at,
                   f.queue_state, f.step_index, f.suspended_reason, f.scheduler_state_json, d.scheduler_type,
                   f.created_at, f.last_modified, f.deleted, f.client_id, f.version, f.model_type, f.reverse
              FROM flashcards f
              LEFT JOIN decks d ON d.id = f.deck_id{deck_owner_join}
             WHERE f.uuid = ? AND f.deleted = 0{owner_filter}
        """  # nosec B608 -- Fixed owner clauses; all data values are bound.
        try:
            cur = self._ctx.execute_query(query, (card_uuid, *owner_params), read_only=True)
            row = cur.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def add_flashcard_asset(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        original_filename: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        """Persist a managed flashcard image asset and return its UUID."""
        if not image_bytes:
            raise InputError("Flashcard asset image_bytes must not be empty")  # noqa: TRY003

        asset_uuid = self._ctx.new_uuid()
        now = self._ctx.utc_now_iso()
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        try:
            with self._ctx.transaction() as conn:
                insert_sql = (
                    """
                    INSERT INTO flashcard_assets(
                        uuid, card_id, mime_type, original_filename, byte_size, sha256,
                        image_data, width, height, created_at, last_modified, deleted, client_id, version
                    )
                    VALUES(?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                )
                params = (
                    asset_uuid,
                    mime_type,
                    original_filename,
                    len(image_bytes),
                    sha256,
                    image_bytes,
                    width,
                    height,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
                conn.execute(insert_sql, params)
            return asset_uuid
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to add flashcard asset: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to add flashcard asset: {exc}") from exc  # noqa: TRY003

    def get_flashcard_asset(self, asset_uuid: str) -> dict[str, Any] | None:
        """Fetch flashcard asset metadata by UUID."""
        owner_filter, owner_params = self._flashcard_owner_filter("fa")
        card_owner_join = " AND f.client_id = fa.client_id" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        query = f"""
            SELECT fa.uuid, fa.mime_type, fa.original_filename, fa.byte_size, fa.sha256,
                   fa.width, fa.height, fa.created_at, fa.last_modified, fa.deleted,
                   fa.client_id, fa.version, f.uuid AS card_uuid
              FROM flashcard_assets fa
              LEFT JOIN flashcards f ON f.id = fa.card_id{card_owner_join}
             WHERE fa.uuid = ? AND fa.deleted = 0{owner_filter}
        """  # nosec B608 -- Fixed owner clauses; all data values are bound.
        try:
            cur = self._ctx.execute_query(query, (asset_uuid, *owner_params), read_only=True)
            row = cur.fetchone()
            return dict(row) if row else None
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def get_flashcard_asset_content(self, asset_uuid: str) -> bytes | None:
        """Fetch raw bytes for a flashcard asset by UUID."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        query = "SELECT image_data FROM flashcard_assets WHERE uuid = ? AND deleted = 0" + owner_filter  # nosec B608 -- Fixed owner clauses; all data values are bound.
        try:
            cur = self._ctx.execute_query(query, (asset_uuid, *owner_params), read_only=True)
            row = cur.fetchone()
        except CharactersRAGDBError:  # noqa: TRY203
            raise
        if not row:
            return None
        blob = row["image_data"]
        if isinstance(blob, memoryview):
            return blob.tobytes()
        return bytes(blob) if blob is not None else None

    def reconcile_flashcard_asset_refs(
        self,
        card_uuid: str,
        *,
        front: str | None,
        back: str | None,
        extra: str | None,
        notes: str | None,
    ) -> list[str]:
        """Attach referenced assets to a card and detach removed assets."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        referenced_asset_uuids: list[str] = []
        for text in (front, back, extra, notes):
            for asset_uuid in extract_flashcard_asset_uuids(text):
                if asset_uuid not in referenced_asset_uuids:
                    referenced_asset_uuids.append(asset_uuid)

        try:
            with self._ctx.transaction() as conn:
                card_row = conn.execute(
                    "SELECT id FROM flashcards WHERE uuid = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (card_uuid, *owner_params),
                ).fetchone()
                if not card_row:
                    raise CharactersRAGDBError("Flashcard not found or deleted")  # noqa: TRY003
                card_id = int(card_row["id"])

                existing_asset_rows = conn.execute(
                    "SELECT id, uuid, card_id FROM flashcard_assets WHERE deleted = 0" + owner_filter, owner_params  # nosec B608 -- Fixed owner clauses; all data values are bound.
                ).fetchall()
                assets_by_uuid = {str(row["uuid"]): row for row in existing_asset_rows}
                for asset_uuid in referenced_asset_uuids:
                    asset_row = assets_by_uuid.get(asset_uuid)
                    if not asset_row:
                        raise InputError(f"Flashcard asset not found: {asset_uuid}")  # noqa: TRY003
                    attached_card_id = asset_row["card_id"]
                    if attached_card_id is not None and int(attached_card_id) != card_id:
                        raise ConflictError(
                            "Flashcard asset is attached to a different card",
                            entity="flashcard_assets",
                            identifier=asset_uuid,
                        )

                now = self._ctx.utc_now_iso()
                for asset_uuid in referenced_asset_uuids:
                    conn.execute(
                        f"""
                        UPDATE flashcard_assets
                           SET card_id = ?, last_modified = ?, version = version + 1, client_id = ?
                         WHERE uuid = ? AND deleted = 0{owner_filter}
                        """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (card_id, now, self._ctx.client_id, asset_uuid, *owner_params),
                    )

                existing_attached_rows = conn.execute(
                    "SELECT uuid FROM flashcard_assets WHERE card_id = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (card_id, *owner_params),
                ).fetchall()
                referenced_set = set(referenced_asset_uuids)
                for row in existing_attached_rows:
                    existing_uuid = str(row["uuid"])
                    if existing_uuid in referenced_set:
                        continue
                    conn.execute(
                        f"""
                        UPDATE flashcard_assets
                           SET card_id = NULL, last_modified = ?, version = version + 1, client_id = ?
                         WHERE uuid = ? AND deleted = 0{owner_filter}
                        """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (now, self._ctx.client_id, existing_uuid, *owner_params),
                    )
            return referenced_asset_uuids
        except (BackendDatabaseError, sqlite3.Error) as exc:
            raise CharactersRAGDBError(f"Failed to reconcile flashcard asset refs: {exc}") from exc  # noqa: TRY003

    def cleanup_stale_flashcard_assets(self, *, older_than: timedelta | None = None) -> int:
        """Soft-delete unattached flashcard assets older than the provided age."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        age = older_than or timedelta(hours=24)
        cutoff = (datetime.now(timezone.utc) - age).strftime("%Y-%m-%dT%H:%M:%SZ")
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    f"""
                    UPDATE flashcard_assets
                       SET deleted = 1,
                           last_modified = ?,
                           version = version + 1,
                           client_id = ?
                     WHERE card_id IS NULL
                       AND deleted = 0
                       AND created_at <= ?{owner_filter}
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (now, self._ctx.client_id, cutoff, *owner_params),
                )
                return int(cursor.rowcount or 0)
        except (BackendDatabaseError, sqlite3.Error) as exc:
            raise CharactersRAGDBError(f"Failed to cleanup stale flashcard assets: {exc}") from exc  # noqa: TRY003

    def update_flashcard(
        self,
        card_uuid: str,
        updates: dict[str, Any],
        expected_version: int | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """
        Update mutable fields: deck_id, front, back, notes, is_cloze, tags_json.
        If expected_version is provided, enforce optimistic locking.
        """
        owner_filter, owner_params = self._flashcard_owner_filter()
        allowed = {"deck_id", "front", "back", "notes", "extra", "is_cloze", "tags_json", "model_type", "reverse"}
        set_parts = []
        params: list[Any] = []
        for k, v in updates.items():
            if k in allowed:
                set_parts.append(f"{k} = ?")
                params.append(v)
                if k == "front":
                    set_parts.append("front_search = ?")
                    params.append(sanitize_flashcard_text_for_search(v))
                elif k == "back":
                    set_parts.append("back_search = ?")
                    params.append(sanitize_flashcard_text_for_search(v))
                elif k == "notes":
                    set_parts.append("notes_search = ?")
                    params.append(sanitize_flashcard_text_for_search(v))
        norm_tags: list[str] | None = None
        if tags is not None:
            norm_tags = [t.strip() for t in tags if t and t.strip()]
            set_parts.append("tags_json = ?")
            params.append(json.dumps(norm_tags))
        if not set_parts:
            if expected_version is None and self._ctx.backend_type != BackendType.POSTGRESQL:
                return True
            try:
                with self._ctx.transaction() as conn:
                    row = conn.execute(
                        "SELECT version FROM flashcards WHERE uuid = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (card_uuid, *owner_params),
                    ).fetchone()
                    if not row:
                        raise CharactersRAGDBError("Flashcard not found or deleted")  # noqa: TRY003
                    current_version = int(row["version"])
                    if expected_version is not None and current_version != expected_version:
                        raise ConflictError(  # noqa: TRY003
                            "Version mismatch updating flashcard",
                            entity="flashcards",
                            identifier=card_uuid,
                        )
            except sqlite3.Error as e:
                raise CharactersRAGDBError(f"Failed to update flashcard: {e}") from e  # noqa: TRY003
            return True
        now = self._ctx.utc_now_iso()
        set_parts.extend(["last_modified = ?", "version = version + 1", "client_id = ?"])
        params.extend([now, self._ctx.client_id])

        try:
            with self._ctx.transaction() as conn:
                # get id and (optionally) version
                row = conn.execute("SELECT id, version FROM flashcards WHERE uuid = ? AND deleted = 0" + owner_filter, (card_uuid, *owner_params)).fetchone()  # nosec B608 -- Fixed owner clauses; all data values are bound.
                if not row:
                    raise CharactersRAGDBError("Flashcard not found or deleted")  # noqa: TRY003
                card_id, current_version = int(row["id"]), int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError("Version mismatch updating flashcard", entity="flashcards", identifier=card_uuid)  # noqa: TRY003
                if "deck_id" in updates:
                    self._validate_flashcard_deck_locked(conn, updates["deck_id"])
                if norm_tags is not None:
                    self._sync_flashcard_keyword_links(conn, card_id, norm_tags)
                params_final = params + [card_id, *owner_params]
                query = f"UPDATE flashcards SET {', '.join(set_parts)} WHERE id = ? AND deleted = 0{owner_filter}"  # nosec B608
                rc = conn.execute(query, tuple(params_final)).rowcount
                return rc > 0
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to update flashcard: {e}") from e  # noqa: TRY003

    def soft_delete_flashcard(self, card_uuid: str, expected_version: int) -> bool:
        """Soft delete a flashcard with optimistic locking."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                row = conn.execute("SELECT id, version, deleted FROM flashcards WHERE uuid = ?" + owner_filter, (card_uuid, *owner_params)).fetchone()  # nosec B608 -- Fixed owner clauses; all data values are bound.
                if not row:
                    raise ConflictError("Flashcard not found", entity="flashcards", identifier=card_uuid)  # noqa: TRY003
                card_id, cur_ver, deleted = int(row["id"]), int(row["version"]), int(row["deleted"])
                if deleted:
                    return True
                if cur_ver != expected_version:
                    raise ConflictError("Version mismatch deleting flashcard", entity="flashcards", identifier=card_uuid)  # noqa: TRY003
                rc = conn.execute(
                    "UPDATE flashcards SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (now, expected_version + 1, self._ctx.client_id, card_id, *owner_params)
                ).rowcount
                return rc > 0
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to delete flashcard: {e}") from e  # noqa: TRY003

    def reset_flashcard_scheduling(self, card_uuid: str, expected_version: int | None = None) -> bool:
        """Reset scheduling fields to new-card defaults with optimistic locking."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                row = conn.execute(
                    "SELECT id, version FROM flashcards WHERE uuid = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (card_uuid, *owner_params),
                ).fetchone()
                if not row:
                    raise ConflictError("Flashcard not found", entity="flashcards", identifier=card_uuid)  # noqa: TRY003
                card_id, current_version = int(row["id"]), int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError(
                        "Version mismatch resetting flashcard scheduling",
                        entity="flashcards",
                        identifier=card_uuid,
                    )  # noqa: TRY003
                rc = conn.execute(
                    f"""
                    UPDATE flashcards
                       SET ef = 2.5,
                           interval_days = 0,
                           repetitions = 0,
                           lapses = 0,
                           queue_state = 'new',
                           step_index = NULL,
                           suspended_reason = NULL,
                           scheduler_state_json = '{{}}',
                           due_at = ?,
                           last_reviewed_at = NULL,
                           last_modified = ?,
                           version = version + 1,
                           client_id = ?
                     WHERE id = ? AND deleted = 0{owner_filter}
                    """,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (now, now, self._ctx.client_id, card_id, *owner_params),
                ).rowcount
                return rc > 0
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to reset flashcard scheduling: {e}") from e  # noqa: TRY003

    def get_keywords_for_flashcard(self, card_uuid: str) -> list[dict[str, Any]]:
        """Return keywords linked to a flashcard."""
        owner_filter, owner_params = self._flashcard_owner_filter("f")
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            owner_filter += " AND kw.client_id = f.client_id"
        keyword_table = self._ctx.map_table("keywords")
        order_clause = self._ctx.case_insensitive_order_clause("kw.keyword")
        query = """
            SELECT kw.*
              FROM flashcards f
              JOIN flashcard_keywords fk ON fk.card_id = f.id
              JOIN {keyword_table} kw ON kw.id = fk.keyword_id
             WHERE f.uuid = ? AND f.deleted = 0 AND kw.deleted = 0{owner_filter}
             {order_clause}
        """.format_map(locals())  # nosec B608
        try:
            cur = self._ctx.execute_query(query, (card_uuid, *owner_params), read_only=True)
            return [dict(r) for r in cur.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def set_flashcard_tags(self, card_uuid: str, tags: list[str]) -> bool:
        """
        Replace flashcard tags (keywords) with provided list.
        Ensures keywords exist, links missing, removes extra.
        Also updates flashcards.tags_json accordingly.
        """
        owner_filter, owner_params = self._flashcard_owner_filter()
        norm_tags = [t.strip() for t in tags if t and t.strip()]
        try:
            with self._ctx.transaction() as conn:
                row = conn.execute("SELECT id FROM flashcards WHERE uuid = ? AND deleted = 0" + owner_filter, (card_uuid, *owner_params)).fetchone()  # nosec B608 -- Fixed owner clauses; all data values are bound.
                if not row:
                    raise CharactersRAGDBError("Flashcard not found or deleted")  # noqa: TRY003
                card_id = int(row["id"])
                self._sync_flashcard_keyword_links(conn, card_id, norm_tags)
                # update tags_json mirror
                conn.execute(
                    "UPDATE flashcards SET tags_json = ?, last_modified = ?, version = version + 1, client_id = ? WHERE id = ?" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                    (json.dumps(norm_tags), self._ctx.utc_now_iso(), self._ctx.client_id, card_id, *owner_params),
                )
                return True
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to set flashcard tags: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to set flashcard tags: {exc}") from exc  # noqa: TRY003

    def _sync_flashcard_keyword_links(self, conn: sqlite3.Connection, card_id: int, tags: list[str]) -> None:
        # current keyword ids
        cur_kw_ids = {
            r["keyword_id"] for r in conn.execute(
                "SELECT keyword_id FROM flashcard_keywords WHERE card_id = ?", (card_id,)
            ).fetchall()
        }
        # ensure keywords exist and collect ids
        desired_kw_ids = set()
        keyword_id: int | None
        for t in tags:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                keyword_id = self._flashcard_owned_keyword_id(conn, t)
            else:
                kw = self._ctx.db.get_keyword_by_text(t)
                keyword_id = self._ctx.db.add_keyword(t) if not kw else kw["id"]
            if keyword_id is not None:
                desired_kw_ids.add(int(keyword_id))
        # link missing
        insert_ts = self._ctx.utc_now_iso()
        for kid in desired_kw_ids - cur_kw_ids:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                insert_query = (
                    "INSERT INTO flashcard_keywords(card_id, keyword_id, created_at) "
                    "VALUES(?, ?, ?) ON CONFLICT (card_id, keyword_id) DO NOTHING"
                )
            else:
                insert_query = (
                    "INSERT OR IGNORE INTO flashcard_keywords(card_id, keyword_id, created_at) VALUES(?, ?, ?)"
                )
            conn.execute(insert_query, (card_id, int(kid), insert_ts))
        # unlink extras
        for kid in cur_kw_ids - desired_kw_ids:
            conn.execute("DELETE FROM flashcard_keywords WHERE card_id = ? AND keyword_id = ?", (card_id, int(kid)))

    def _flashcard_owned_keyword_id(self, conn: Any, text: str) -> int:
        """Resolve an owner-local PostgreSQL tag without changing generic keyword APIs."""
        from tldw_Server_API.app.core.Sync_Contracts.notes_organization import new_organization_sync_id

        now = self._ctx.utc_now_iso()
        conn.execute(
            "INSERT INTO chacha_keywords(sync_id, keyword, created_at, last_modified, deleted, client_id, version) "
            "VALUES (?, ?, ?, ?, FALSE, ?, 1) "
            "ON CONFLICT (client_id, LOWER(keyword)) DO NOTHING",
            (new_organization_sync_id(), text, now, now, self._ctx.client_id),
        )
        row = conn.execute(
            "SELECT id, deleted FROM chacha_keywords WHERE client_id = ? AND LOWER(keyword) = LOWER(?) FOR UPDATE",
            (self._ctx.client_id, text),
        ).fetchone()
        if not row:
            raise CharactersRAGDBError("Flashcard keyword could not be resolved")  # noqa: TRY003
        if row["deleted"]:
            conn.execute(
                "UPDATE chacha_keywords SET deleted = FALSE, merged_into_sync_id = NULL, last_modified = ?, version = version + 1 "
                "WHERE id = ? AND client_id = ? AND deleted = TRUE",
                (now, row["id"], self._ctx.client_id),
            )
        return int(row["id"])

    # ==========================
    # Study Packs
    # ==========================
    def create_study_pack(
        self,
        *,
        title: str,
        workspace_id: str | None,
        deck_id: int | None,
        source_bundle_json: dict[str, Any] | str,
        generation_options_json: dict[str, Any] | str | None,
        status: str = "active",
        superseded_by_pack_id: int | None = None,
    ) -> int:
        """Create a study pack row and return its integer id."""
        if status not in ("active", "superseded"):
            raise InputError("Invalid study pack status; must be 'active' or 'superseded'")  # noqa: TRY003

        source_bundle_candidate: Any = source_bundle_json
        if isinstance(source_bundle_json, str):
            with contextlib.suppress(json.JSONDecodeError):
                source_bundle_candidate = json.loads(source_bundle_json)
        if isinstance(source_bundle_candidate, dict):
            source_items = source_bundle_candidate.get("items")
            if isinstance(source_items, list):
                for source_item in source_items:
                    if not isinstance(source_item, dict):
                        continue
                    if self._ctx.normalize_nullable_text(source_item.get("source_id")) is None:
                        raise InputError("Study pack source_id is required")  # noqa: TRY003

        now = self._ctx.utc_now_iso()
        source_bundle = self._ctx.db.ensure_json_string_from_mixed(source_bundle_json)
        generation_options = self._ctx.db.ensure_json_string_from_mixed(generation_options_json)

        try:
            with self._ctx.transaction() as conn:
                self._validate_flashcard_deck_locked(conn, deck_id)
                self._ctx.require_owner_row(conn, "study_packs", superseded_by_pack_id, self._ctx.client_id)
                insert_sql = (
                    "INSERT INTO study_packs("
                    "workspace_id, title, deck_id, source_bundle_json, generation_options_json, "
                    "status, superseded_by_pack_id, created_at, last_modified, deleted, client_id, version"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                params = (
                    workspace_id,
                    title,
                    deck_id,
                    source_bundle,
                    generation_options,
                    status,
                    superseded_by_pack_id,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )

                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_sql + " RETURNING id", params)
                    row = cursor.fetchone()
                    pack_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_sql, params)
                    pack_id = cursor.lastrowid

                if pack_id is None:
                    raise CharactersRAGDBError("Failed to determine study pack ID after insert")  # noqa: TRY003
                pack_id = int(pack_id)
                return pack_id
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to create study pack: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to create study pack: {exc}") from exc  # noqa: TRY003

    @classmethod
    def _is_safe_suggestion_payload_key(cls, key: str) -> bool:
        normalized = str(key or "").strip().lower()
        return normalized in _SAFE_SUGGESTION_PAYLOAD_EXACT_KEYS or normalized.endswith(
            _SAFE_SUGGESTION_PAYLOAD_SUFFIXES
        )

    @classmethod
    def _sanitize_suggestion_evidence_reasons(cls, value: Any) -> list[str] | None:
        """Persist only structured evidence reason codes from the approved set."""
        if value is None:
            return None

        if isinstance(value, str):
            raw_items: list[Any] = [value]
        elif isinstance(value, list | tuple | set):
            raw_items = list(value)
        else:
            return None

        sanitized: list[str] = []
        for item in raw_items:
            reason = str(item or "").strip().lower()
            if not reason or reason not in _SAFE_SUGGESTION_EVIDENCE_REASONS:
                continue
            if reason not in sanitized:
                sanitized.append(reason)
        return sanitized or None

    @classmethod
    def _sanitize_suggestion_payload(cls, payload: Any, *, parent_key: str | None = None) -> Any:
        """Persist only safe labels/refs/flags/counts by default."""
        if isinstance(payload, dict):
            sanitized: dict[str, Any] = {}
            for raw_key, value in payload.items():
                key_text = str(raw_key)
                if not cls._is_safe_suggestion_payload_key(key_text):
                    continue
                if key_text.strip().lower() == "evidence_reasons":
                    cleaned_value = cls._sanitize_suggestion_evidence_reasons(value)
                else:
                    cleaned_value = cls._sanitize_suggestion_payload(value, parent_key=key_text)
                if cleaned_value in (None, {}, []):
                    continue
                sanitized[key_text] = cleaned_value
            return sanitized

        if isinstance(payload, list | tuple):
            sanitized_items = [
                cls._sanitize_suggestion_payload(item, parent_key=parent_key)
                for item in payload
            ]
            return [item for item in sanitized_items if item not in (None, {}, [])]

        if isinstance(payload, bool | int | float):
            return payload

        if isinstance(payload, str):
            if not parent_key or not cls._is_safe_suggestion_payload_key(parent_key):
                return None
            text = payload.strip()
            if not text or len(text) > _SAFE_SUGGESTION_PAYLOAD_MAX_TEXT_LENGTH:
                return None
            return text

        return None

    def create_suggestion_snapshot(
        self,
        *,
        service: str,
        activity_type: str,
        anchor_type: str,
        anchor_id: int,
        suggestion_type: str,
        payload_json: dict[str, Any] | list[Any] | str,
        status: str = "active",
        user_selection_json: dict[str, Any] | list[Any] | str | None = None,
        refreshed_from_snapshot_id: int | None = None,
    ) -> int:
        """Create a suggestion snapshot row and return its integer id."""
        if status not in ("active", "superseded"):
            raise InputError("Invalid suggestion snapshot status; must be 'active' or 'superseded'")  # noqa: TRY003

        sanitized_payload: Any = payload_json
        if isinstance(payload_json, str):
            with contextlib.suppress(json.JSONDecodeError):
                sanitized_payload = json.loads(payload_json)
        sanitized_payload = self._sanitize_suggestion_payload(sanitized_payload)
        payload_value = self._ctx.db.ensure_json_string_from_mixed(sanitized_payload)
        if payload_value is None:
            raise InputError("Suggestion snapshot payload_json is required")  # noqa: TRY003

        now = self._ctx.utc_now_iso()

        try:
            with self._ctx.transaction() as conn:
                if isinstance(user_selection_json, str):
                    try:
                        parsed_user_selection = json.loads(user_selection_json)
                    except json.JSONDecodeError as exc:
                        raise InputError("Suggestion snapshot user_selection_json must be valid JSON") from exc  # noqa: TRY003
                    user_selection_value = self._ctx.db.ensure_json_string_from_mixed(parsed_user_selection)
                else:
                    user_selection_value = self._ctx.db.ensure_json_string_from_mixed(user_selection_json)

                if refreshed_from_snapshot_id is not None:
                    owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
                    parent_row = conn.execute(
                        """
                        SELECT service, activity_type, anchor_type, anchor_id, suggestion_type, user_selection_json
                          FROM suggestion_snapshots
                         WHERE id = ? AND deleted = 0
                        """ + owner_clause,  # nosec B608 -- Fixed owner clause; values are bound.
                        (refreshed_from_snapshot_id, *owner_params),
                    ).fetchone()
                    if not parent_row:
                        raise ConflictError(
                            "Suggestion snapshot refresh source not found",
                            entity="suggestion_snapshots",
                            identifier=refreshed_from_snapshot_id,
                        )
                    if (
                        str(parent_row["service"]) != str(service)
                        or str(parent_row["activity_type"]) != str(activity_type)
                        or str(parent_row["anchor_type"]) != str(anchor_type)
                        or int(parent_row["anchor_id"]) != int(anchor_id)
                        or str(parent_row["suggestion_type"]) != str(suggestion_type)
                    ):
                        raise InputError("refreshed_from_snapshot_id must reference the same suggestion lineage")  # noqa: TRY003
                    if user_selection_value is None:
                        user_selection_value = parent_row["user_selection_json"]
                insert_sql = (
                    "INSERT INTO suggestion_snapshots("
                    "service, activity_type, anchor_type, anchor_id, suggestion_type, status, "
                    "payload_json, user_selection_json, refreshed_from_snapshot_id, "
                    "created_at, last_modified, deleted, client_id, version"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                params = (
                    service,
                    activity_type,
                    anchor_type,
                    int(anchor_id),
                    suggestion_type,
                    status,
                    payload_value,
                    user_selection_value,
                    refreshed_from_snapshot_id,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_sql + " RETURNING id", params)
                    row = cursor.fetchone()
                    snapshot_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_sql, params)
                    snapshot_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
                if snapshot_id is None:
                    raise CharactersRAGDBError("Failed to determine suggestion snapshot ID after insert")  # noqa: TRY003
                return snapshot_id
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to create suggestion snapshot: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to create suggestion snapshot: {exc}") from exc  # noqa: TRY003

    def get_suggestion_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        """Fetch a single active suggestion snapshot by id."""
        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
        query = """
            SELECT id, service, activity_type, anchor_type, anchor_id, suggestion_type, status,
                   payload_json, user_selection_json, refreshed_from_snapshot_id,
                   created_at, last_modified, deleted, client_id, version
              FROM suggestion_snapshots
             WHERE id = ? AND deleted = 0
        """
        cursor = self._ctx.execute_query(query + owner_clause, (snapshot_id, *owner_params))
        row = cursor.fetchone()
        return self._ctx.deserialize_row_fields(row, _SUGGESTION_SNAPSHOT_JSON_FIELDS)

    def list_suggestion_snapshots_for_anchor(
        self,
        anchor_type: str,
        anchor_id: int,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List snapshots for a concrete anchor in newest-first order."""
        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
        deleted_clause = "1=1" if include_deleted else "deleted = 0"
        query = f"""
            SELECT id, service, activity_type, anchor_type, anchor_id, suggestion_type, status,
                   payload_json, user_selection_json, refreshed_from_snapshot_id,
                   created_at, last_modified, deleted, client_id, version
              FROM suggestion_snapshots
             WHERE anchor_type = ? AND anchor_id = ? AND {deleted_clause}{owner_clause}
             ORDER BY id DESC
        """  # nosec B608
        cursor = self._ctx.execute_query(query, (anchor_type, anchor_id, *owner_params))
        return [
            item
            for row in cursor.fetchall()
            if (item := self._ctx.deserialize_row_fields(row, _SUGGESTION_SNAPSHOT_JSON_FIELDS)) is not None
        ]

    def _suggestion_link_owner_filter(self) -> tuple[str, tuple[str, ...]]:
        """Require both the PostgreSQL link and its live snapshot to belong to the caller."""
        if self._ctx.backend_type != BackendType.POSTGRESQL:
            return "", ()
        return (
            " AND client_id = ? AND EXISTS ("
            "SELECT 1 FROM suggestion_snapshots snapshot "
            "WHERE snapshot.id = suggestion_generation_links.snapshot_id "
            "AND snapshot.deleted = FALSE AND snapshot.client_id = ?)",
            (self._ctx.client_id, self._ctx.client_id),
        )

    def _validate_suggestion_snapshot_locked(self, conn: Any, snapshot_id: int) -> None:
        """Require an owned live PostgreSQL snapshot before creating action links."""
        if self._ctx.backend_type != BackendType.POSTGRESQL:
            return
        row = conn.execute(
            "SELECT id FROM suggestion_snapshots WHERE id = ? AND client_id = ? AND deleted = FALSE FOR SHARE",
            (snapshot_id, self._ctx.client_id),
        ).fetchone()
        if row is None:
            raise InputError("Suggestion snapshot not found")  # noqa: TRY003

    def create_suggestion_generation_link(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        target_id: str,
        selection_fingerprint: str,
    ) -> int:
        """Persist one durable link per concrete snapshot/action result."""
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                self._validate_suggestion_snapshot_locked(conn, snapshot_id)
                insert_sql = (
                    "INSERT INTO suggestion_generation_links("
                    "snapshot_id, target_service, target_type, target_id, selection_fingerprint, "
                    "created_at, last_modified, deleted, client_id, version"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                params = (
                    snapshot_id,
                    target_service,
                    target_type,
                    target_id,
                    selection_fingerprint,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_sql + " RETURNING id", params)
                    row = cursor.fetchone()
                    link_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_sql, params)
                    link_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
                if link_id is None:
                    raise CharactersRAGDBError("Failed to determine suggestion generation link ID after insert")  # noqa: TRY003
                return link_id
        except sqlite3.IntegrityError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError(
                    "Suggestion generation link already exists",
                    entity="suggestion_generation_links",
                    identifier=f"{snapshot_id}:{target_service}:{target_type}:{selection_fingerprint}",
                ) from exc
            raise CharactersRAGDBError(f"Failed to create suggestion generation link: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError(
                    "Suggestion generation link already exists",
                    entity="suggestion_generation_links",
                    identifier=f"{snapshot_id}:{target_service}:{target_type}:{selection_fingerprint}",
                ) from exc
            raise CharactersRAGDBError(f"Failed to create suggestion generation link: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to create suggestion generation link: {exc}") from exc  # noqa: TRY003

    def find_suggestion_generation_link(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        target_id: str,
        selection_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Find the durable link row for a concrete snapshot/action result."""
        owner_clause, owner_params = self._suggestion_link_owner_filter()
        query = f"""
            SELECT id, snapshot_id, target_service, target_type, target_id, selection_fingerprint,
                   created_at, last_modified, deleted, client_id, version
              FROM suggestion_generation_links
             WHERE snapshot_id = ?
               AND target_service = ?
               AND target_type = ?
               AND target_id = ?
               AND selection_fingerprint = ?
               AND deleted = 0{owner_clause}
             ORDER BY id DESC
             LIMIT 1
        """  # nosec B608 -- Fixed owner clause; values are bound.
        cursor = self._ctx.execute_query(
            query,
            (snapshot_id, target_service, target_type, target_id, selection_fingerprint, *owner_params),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def replace_suggestion_generation_link(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        target_id: str,
        selection_fingerprint: str,
    ) -> int:
        """Upsert a direct generation link without accumulating multiple active rows."""
        owner_clause, owner_params = self._suggestion_link_owner_filter()
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                self._validate_suggestion_snapshot_locked(conn, snapshot_id)
                rows = conn.execute(
                    f"""
                    SELECT id
                      FROM suggestion_generation_links
                     WHERE snapshot_id = ?
                       AND target_service = ?
                       AND target_type = ?
                       AND selection_fingerprint = ?
                       AND deleted = 0{owner_clause}
                     ORDER BY id DESC
                    """,  # nosec B608 -- Fixed owner clause; values are bound.
                    (
                        int(snapshot_id),
                        target_service,
                        target_type,
                        selection_fingerprint,
                        *owner_params,
                    ),
                ).fetchall()
                if rows:
                    keeper_id = int(rows[0]["id"])
                    updated = conn.execute(
                        """
                        UPDATE suggestion_generation_links
                           SET target_id = ?, last_modified = ?, version = version + 1, client_id = ?
                         WHERE id = ? AND deleted = 0
                        """ + owner_clause,  # nosec B608 -- Fixed owner clause; values are bound.
                        (
                            str(target_id),
                            now,
                            self._ctx.client_id,
                            keeper_id,
                            *owner_params,
                        ),
                    ).rowcount
                    stale_ids = [int(row["id"]) for row in rows[1:]]
                    for stale_id in stale_ids:
                        conn.execute(
                            """
                            UPDATE suggestion_generation_links
                               SET deleted = ?, last_modified = ?, version = version + 1, client_id = ?
                             WHERE id = ? AND deleted = 0
                            """ + owner_clause,  # nosec B608 -- Fixed owner clause; values are bound.
                            (True, now, self._ctx.client_id, stale_id, *owner_params),
                        )
                    if updated <= 0:
                        raise CharactersRAGDBError("Failed to update existing suggestion generation link")  # noqa: TRY003
                    return keeper_id

                insert_sql = (
                    "INSERT INTO suggestion_generation_links("
                    "snapshot_id, target_service, target_type, target_id, selection_fingerprint, "
                    "created_at, last_modified, deleted, client_id, version"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                params = (
                    int(snapshot_id),
                    target_service,
                    target_type,
                    str(target_id),
                    selection_fingerprint,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_sql + " RETURNING id", params)
                    row = cursor.fetchone()
                    link_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_sql, params)
                    link_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
                if link_id is None:
                    raise CharactersRAGDBError("Failed to determine suggestion generation link ID after insert")  # noqa: TRY003
                return link_id
        except sqlite3.IntegrityError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError(
                    "Suggestion generation link already exists",
                    entity="suggestion_generation_links",
                    identifier=f"{snapshot_id}:{target_service}:{target_type}:{selection_fingerprint}",
                ) from exc
            raise CharactersRAGDBError(f"Failed to replace suggestion generation link: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._ctx.is_unique_violation(exc):
                raise ConflictError(
                    "Suggestion generation link already exists",
                    entity="suggestion_generation_links",
                    identifier=f"{snapshot_id}:{target_service}:{target_type}:{selection_fingerprint}",
                ) from exc
            raise CharactersRAGDBError(f"Failed to replace suggestion generation link: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to replace suggestion generation link: {exc}") from exc  # noqa: TRY003

    def find_suggestion_generation_link_by_fingerprint(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        selection_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Find an existing generation link by fingerprint without requiring the target id."""
        owner_clause, owner_params = self._suggestion_link_owner_filter()
        query = f"""
            SELECT id, snapshot_id, target_service, target_type, target_id, selection_fingerprint,
                   created_at, last_modified, deleted, client_id, version
              FROM suggestion_generation_links
             WHERE snapshot_id = ?
               AND target_service = ?
               AND target_type = ?
               AND selection_fingerprint = ?
               AND deleted = 0{owner_clause}
             ORDER BY id DESC
             LIMIT 1
        """  # nosec B608 -- Fixed owner clause; values are bound.
        cursor = self._ctx.execute_query(
            query,
            (int(snapshot_id), target_service, target_type, selection_fingerprint, *owner_params),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def finalize_suggestion_generation_link(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        selection_fingerprint: str,
        final_target_id: str,
    ) -> int:
        """Replace an in-progress reservation target id with the real durable target id.

        Returns the number of rows updated (0 means the reservation was not found).
        """
        owner_clause, owner_params = self._suggestion_link_owner_filter()
        now = self._ctx.utc_now_iso()
        pending_target_id = f"pending:{selection_fingerprint}"
        try:
            with self._ctx.transaction() as conn:
                updated = conn.execute(
                    """
                    UPDATE suggestion_generation_links
                       SET target_id = ?, last_modified = ?, version = version + 1, client_id = ?
                     WHERE snapshot_id = ?
                       AND target_service = ?
                       AND target_type = ?
                       AND target_id = ?
                       AND selection_fingerprint = ?
                       AND deleted = 0
                    """ + owner_clause,  # nosec B608 -- Fixed owner clause; values are bound.
                    (
                        str(final_target_id),
                        now,
                        self._ctx.client_id,
                        int(snapshot_id),
                        target_service,
                        target_type,
                        pending_target_id,
                        selection_fingerprint,
                        *owner_params,
                    ),
                ).rowcount
            return int(updated)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to finalize suggestion generation link: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to finalize suggestion generation link: {exc}") from exc  # noqa: TRY003

    def soft_delete_suggestion_generation_link(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        selection_fingerprint: str,
    ) -> int:
        """Soft-delete active generation links matching a snapshot fingerprint."""
        owner_clause, owner_params = self._suggestion_link_owner_filter()
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                updated = conn.execute(
                    """
                    UPDATE suggestion_generation_links
                       SET deleted = ?, last_modified = ?, version = version + 1, client_id = ?
                     WHERE snapshot_id = ?
                       AND target_service = ?
                       AND target_type = ?
                       AND selection_fingerprint = ?
                       AND deleted = 0
                    """ + owner_clause,  # nosec B608 -- Fixed owner clause; values are bound.
                    (
                        True,
                        now,
                        self._ctx.client_id,
                        int(snapshot_id),
                        target_service,
                        target_type,
                        selection_fingerprint,
                        *owner_params,
                    ),
                ).rowcount
            return int(updated)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to soft-delete suggestion generation link: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to soft-delete suggestion generation link: {exc}") from exc  # noqa: TRY003

    def release_suggestion_generation_link_reservation(
        self,
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        selection_fingerprint: str,
    ) -> None:
        """Soft-delete an in-progress reservation after generation failure."""
        owner_clause, owner_params = self._suggestion_link_owner_filter()
        now = self._ctx.utc_now_iso()
        pending_target_id = f"pending:{selection_fingerprint}"
        try:
            with self._ctx.transaction() as conn:
                conn.execute(
                    """
                    UPDATE suggestion_generation_links
                       SET deleted = ?, last_modified = ?, version = version + 1, client_id = ?
                     WHERE snapshot_id = ?
                       AND target_service = ?
                       AND target_type = ?
                       AND target_id = ?
                       AND selection_fingerprint = ?
                       AND deleted = 0
                    """ + owner_clause,  # nosec B608 -- Fixed owner clause; values are bound.
                    (
                        True,
                        now,
                        self._ctx.client_id,
                        int(snapshot_id),
                        target_service,
                        target_type,
                        pending_target_id,
                        selection_fingerprint,
                        *owner_params,
                    ),
                )
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to release suggestion generation link reservation: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to release suggestion generation link reservation: {exc}") from exc  # noqa: TRY003

    def soft_delete_deck_by_id(self, deck_id: int, expected_version: int | None = None) -> None:
        """Best-effort soft-delete of a deck by id, optionally with optimistic locking."""
        owner_filter, owner_params = self._flashcard_owner_filter()
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                params: list[Any] = [
                    True,
                    now,
                    self._ctx.client_id,
                    int(deck_id),
                ]
                version_clause = ""
                if expected_version is not None:
                    version_clause = " AND version = ?"
                    params.append(int(expected_version))
                params.extend(owner_params)
                rowcount = conn.execute(
                    f"""
                    UPDATE decks
                       SET deleted = ?, last_modified = ?, version = version + 1, client_id = ?
                     WHERE id = ? AND deleted = 0{version_clause}{owner_filter}
                    """,  # nosec B608
                    tuple(params),
                ).rowcount
                if rowcount == 0 and expected_version is not None:
                    existing = conn.execute(
                        "SELECT id FROM decks WHERE id = ? AND deleted = 0" + owner_filter,  # nosec B608 -- Fixed owner clauses; all data values are bound.
                        (int(deck_id), *owner_params),
                    ).fetchone()
                    if existing is None:
                        raise ConflictError("Deck not found", entity="decks", identifier=deck_id)  # noqa: TRY003
                    raise ConflictError("Version mismatch deleting deck", entity="decks", identifier=deck_id)  # noqa: TRY003
                if rowcount > 0:
                    conn.execute("DELETE FROM deck_shares WHERE deck_id = ?", (int(deck_id),))
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to soft-delete deck: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to soft-delete deck: {exc}") from exc  # noqa: TRY003

    def get_study_pack(self, pack_id: int) -> dict[str, Any] | None:
        """Fetch a single active study pack by id."""
        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
        query = """
            SELECT id, workspace_id, title, deck_id, source_bundle_json, generation_options_json,
                   status, superseded_by_pack_id, created_at, last_modified, deleted, client_id, version
              FROM study_packs
             WHERE id = ? AND deleted = 0
        """
        try:
            cursor = self._ctx.execute_query(query + owner_clause, (pack_id, *owner_params))
            row = cursor.fetchone()
            return self._ctx.deserialize_row_fields(row, _STUDY_PACK_JSON_FIELDS)
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def add_study_pack_cards(self, study_pack_id: int, flashcard_uuids: list[str]) -> int:
        """Attach flashcards to a study pack by UUID without relying on deck membership."""
        if not flashcard_uuids:
            return 0

        now = self._ctx.utc_now_iso()
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            insert_sql = (
                "INSERT INTO study_pack_cards("
                "study_pack_id, flashcard_uuid, created_at, last_modified, deleted, client_id, version"
                ") VALUES(?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING"
            )
        else:
            insert_sql = (
                "INSERT OR IGNORE INTO study_pack_cards("
                "study_pack_id, flashcard_uuid, created_at, last_modified, deleted, client_id, version"
                ") VALUES(?, ?, ?, ?, ?, ?, ?)"
            )

        params = [
            (study_pack_id, card_uuid, now, now, False, self._ctx.client_id, 1)
            for card_uuid in flashcard_uuids
        ]
        try:
            with self._ctx.transaction() as conn:
                self._ctx.require_owner_row(conn, "study_packs", study_pack_id, self._ctx.client_id)
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    for card_uuid in sorted(set(flashcard_uuids)):
                        self._ctx.require_owner_row(conn, "flashcards", card_uuid, self._ctx.client_id)
                before_row = conn.execute(
                    "SELECT COUNT(*) FROM study_pack_cards WHERE study_pack_id = ? AND deleted = 0",
                    (study_pack_id,),
                ).fetchone()
                before_count = int(before_row["count"] if self._ctx.backend_type == BackendType.POSTGRESQL else before_row[0]) if before_row else 0
                self._ctx.execute_many(insert_sql, params, commit=False)
                after_row = conn.execute(
                    "SELECT COUNT(*) FROM study_pack_cards WHERE study_pack_id = ? AND deleted = 0",
                    (study_pack_id,),
                ).fetchone()
                after_count = int(after_row["count"] if self._ctx.backend_type == BackendType.POSTGRESQL else after_row[0]) if after_row else before_count
            return max(0, after_count - before_count)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to add study pack cards: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to add study pack cards: {exc}") from exc  # noqa: TRY003

    def list_study_pack_cards(self, study_pack_id: int, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        """List study pack membership rows in insertion order."""
        deleted_clause = "1=1" if include_deleted else "spc.deleted = 0"
        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id, "spc")
        if owner_params:
            owner_clause += (
                " AND f.client_id = ? AND EXISTS (SELECT 1 FROM study_packs sp "
                "WHERE sp.id = spc.study_pack_id AND sp.client_id = ?)"
            )
            owner_params += (self._ctx.client_id, self._ctx.client_id)
        query = f"""
            SELECT spc.id, spc.study_pack_id, spc.flashcard_uuid, spc.created_at, spc.last_modified,
                   spc.deleted, spc.client_id, spc.version, f.deck_id
              FROM study_pack_cards spc
              LEFT JOIN flashcards f ON f.uuid = spc.flashcard_uuid
             WHERE spc.study_pack_id = ? AND {deleted_clause}{owner_clause}
             ORDER BY spc.id
        """  # nosec B608
        try:
            cursor = self._ctx.execute_query(query, (study_pack_id, *owner_params))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def add_flashcard_citations(self, flashcard_uuid: str, citations: list[dict[str, Any]]) -> int:
        """Insert provenance citations for a flashcard."""
        if not citations:
            return 0

        now = self._ctx.utc_now_iso()
        params: list[tuple[Any, ...]] = []
        for citation in citations:
            source_type = str(citation.get("source_type") or "").strip().lower()
            if source_type not in ("note", "media", "message"):
                raise InputError("Invalid flashcard citation source_type")  # noqa: TRY003
            normalized_source_id = self._ctx.normalize_nullable_text(citation.get("source_id"))
            if normalized_source_id is None:
                raise InputError("Flashcard citation source_id is required")  # noqa: TRY003
            normalized_locator = self._ctx.db.ensure_json_string_from_mixed(citation.get("locator"))
            if isinstance(normalized_locator, str):
                normalized_locator = normalized_locator.strip() or None
            try:
                ordinal = int(citation.get("ordinal", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise InputError("Flashcard citation ordinal must be an integer >= 0") from exc  # noqa: TRY003
            if ordinal < 0:
                raise InputError("Flashcard citation ordinal must be >= 0")  # noqa: TRY003
            params.append(
                (
                    flashcard_uuid,
                    source_type,
                    normalized_source_id,
                    self._ctx.normalize_nullable_text(citation.get("citation_text")),
                    normalized_locator,
                    ordinal,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
            )

        try:
            with self._ctx.transaction() as conn:
                self._ctx.require_owner_row(conn, "flashcards", flashcard_uuid, self._ctx.client_id)
                self._ctx.execute_many(
                    """
                    INSERT INTO flashcard_citations(
                        flashcard_uuid, source_type, source_id, citation_text, locator, ordinal,
                        created_at, last_modified, deleted, client_id, version
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                    commit=False,
                )
            return len(params)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to add flashcard citations: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to add flashcard citations: {exc}") from exc  # noqa: TRY003

    def replace_flashcard_citations(self, flashcard_uuid: str, citations: list[dict[str, Any]]) -> int:
        """Replace the active citation set for a flashcard with the provided rows."""
        now = self._ctx.utc_now_iso()
        params: list[tuple[Any, ...]] = []
        for citation in citations:
            source_type = str(citation.get("source_type") or "").strip().lower()
            if source_type not in ("note", "media", "message"):
                raise InputError("Invalid flashcard citation source_type")  # noqa: TRY003
            normalized_source_id = self._ctx.normalize_nullable_text(citation.get("source_id"))
            if normalized_source_id is None:
                raise InputError("Flashcard citation source_id is required")  # noqa: TRY003
            normalized_locator = self._ctx.db.ensure_json_string_from_mixed(citation.get("locator"))
            if isinstance(normalized_locator, str):
                normalized_locator = normalized_locator.strip() or None
            try:
                ordinal = int(citation.get("ordinal", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise InputError("Flashcard citation ordinal must be an integer >= 0") from exc  # noqa: TRY003
            if ordinal < 0:
                raise InputError("Flashcard citation ordinal must be >= 0")  # noqa: TRY003
            params.append(
                (
                    flashcard_uuid,
                    source_type,
                    normalized_source_id,
                    self._ctx.normalize_nullable_text(citation.get("citation_text")),
                    normalized_locator,
                    ordinal,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
            )

        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
        try:
            with self._ctx.transaction() as conn:
                self._ctx.require_owner_row(conn, "flashcards", flashcard_uuid, self._ctx.client_id)
                conn.execute(
                    f"""
                    UPDATE flashcard_citations
                       SET deleted = 1, last_modified = ?, version = version + 1, client_id = ?
                     WHERE flashcard_uuid = ? AND deleted = 0{owner_clause}
                    """,  # nosec B608 - fixed owner predicate; values remain bound.
                    (now, self._ctx.client_id, flashcard_uuid, *owner_params),
                )
                if params:
                    self._ctx.execute_many(
                        """
                        INSERT INTO flashcard_citations(
                            flashcard_uuid, source_type, source_id, citation_text, locator, ordinal,
                            created_at, last_modified, deleted, client_id, version
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        params,
                        commit=False,
                    )
            return len(params)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to replace flashcard citations: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to replace flashcard citations: {exc}") from exc  # noqa: TRY003

    def replace_flashcard_citations_and_source_reference_summary(
        self,
        flashcard_uuid: str,
        citations: list[dict[str, Any]],
        *,
        source_ref_type: str | None,
        source_ref_id: str | None,
    ) -> int:
        """Atomically replace citations and mirror the primary legacy source reference."""
        normalized_source_ref_id = self._ctx.normalize_nullable_text(source_ref_id)
        normalized_source_ref_type = str(source_ref_type or "manual").strip().lower() or "manual"
        if normalized_source_ref_type not in ("note", "media", "message", "manual"):
            raise InputError("Invalid flashcard source_ref_type")  # noqa: TRY003
        if normalized_source_ref_id is None:
            normalized_source_ref_type = "manual"

        now = self._ctx.utc_now_iso()
        params: list[tuple[Any, ...]] = []
        for citation in citations:
            source_type = str(citation.get("source_type") or "").strip().lower()
            if source_type not in ("note", "media", "message"):
                raise InputError("Invalid flashcard citation source_type")  # noqa: TRY003
            normalized_citation_source_id = self._ctx.normalize_nullable_text(citation.get("source_id"))
            if normalized_citation_source_id is None:
                raise InputError("Flashcard citation source_id is required")  # noqa: TRY003
            normalized_locator = self._ctx.db.ensure_json_string_from_mixed(citation.get("locator"))
            if isinstance(normalized_locator, str):
                normalized_locator = normalized_locator.strip() or None
            try:
                ordinal = int(citation.get("ordinal", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise InputError("Flashcard citation ordinal must be an integer >= 0") from exc  # noqa: TRY003
            if ordinal < 0:
                raise InputError("Flashcard citation ordinal must be >= 0")  # noqa: TRY003
            params.append(
                (
                    flashcard_uuid,
                    source_type,
                    normalized_citation_source_id,
                    self._ctx.normalize_nullable_text(citation.get("citation_text")),
                    normalized_locator,
                    ordinal,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
            )

        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
        try:
            with self._ctx.transaction() as conn:
                self._ctx.require_owner_row(conn, "flashcards", flashcard_uuid, self._ctx.client_id)
                row = conn.execute(
                    "SELECT id FROM flashcards WHERE uuid = ? AND deleted = 0",
                    (flashcard_uuid,),
                ).fetchone()
                if not row:
                    raise ConflictError("Flashcard not found", entity="flashcards", identifier=flashcard_uuid)  # noqa: TRY003
                conn.execute(
                    """
                    UPDATE flashcards
                       SET source_ref_type = ?, source_ref_id = ?, last_modified = ?,
                           version = version + 1, client_id = ?
                     WHERE id = ? AND deleted = 0
                    """,
                    (
                        normalized_source_ref_type,
                        normalized_source_ref_id,
                        now,
                        self._ctx.client_id,
                        int(row["id"]),
                    ),
                )
                conn.execute(
                    f"""
                    UPDATE flashcard_citations
                       SET deleted = 1, last_modified = ?, version = version + 1, client_id = ?
                     WHERE flashcard_uuid = ? AND deleted = 0{owner_clause}
                    """,  # nosec B608 - fixed owner predicate; values remain bound.
                    (now, self._ctx.client_id, flashcard_uuid, *owner_params),
                )
                if params:
                    self._ctx.execute_many(
                        """
                        INSERT INTO flashcard_citations(
                            flashcard_uuid, source_type, source_id, citation_text, locator, ordinal,
                            created_at, last_modified, deleted, client_id, version
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        params,
                        commit=False,
                    )
            return len(params)
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(
                f"Failed to replace flashcard citations and source summary: {exc}"
            ) from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(
                f"Failed to replace flashcard citations and source summary: {exc}"
            ) from exc  # noqa: TRY003

    def list_flashcard_citations(self, flashcard_uuid: str, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        """List citations for a flashcard ordered by ordinal."""
        deleted_clause = "1=1" if include_deleted else "deleted = 0"
        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id)
        if owner_params:
            owner_clause += (
                " AND EXISTS (SELECT 1 FROM flashcards f "
                "WHERE f.uuid = flashcard_citations.flashcard_uuid AND f.client_id = ?)"
            )
            owner_params += (self._ctx.client_id,)
        query = f"""
            SELECT id, flashcard_uuid, source_type, source_id, citation_text, locator, ordinal,
                   created_at, last_modified, deleted, client_id, version
              FROM flashcard_citations
             WHERE flashcard_uuid = ? AND {deleted_clause}{owner_clause}
             ORDER BY ordinal, id
        """  # nosec B608
        try:
            cursor = self._ctx.execute_query(query, (flashcard_uuid, *owner_params), read_only=True)
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def set_flashcard_source_reference_summary(
        self,
        flashcard_uuid: str,
        *,
        source_ref_type: str | None,
        source_ref_id: str | None,
    ) -> bool:
        """Persist the legacy flashcard source summary from the primary citation only."""
        normalized_source_ref_id = self._ctx.normalize_nullable_text(source_ref_id)
        normalized_source_ref_type = str(source_ref_type or "manual").strip().lower() or "manual"
        if normalized_source_ref_type not in ("note", "media", "message", "manual"):
            raise InputError("Invalid flashcard source_ref_type")  # noqa: TRY003
        if normalized_source_ref_id is None:
            normalized_source_ref_type = "manual"

        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                self._ctx.require_owner_row(conn, "flashcards", flashcard_uuid, self._ctx.client_id)
                row = conn.execute(
                    "SELECT id FROM flashcards WHERE uuid = ? AND deleted = 0",
                    (flashcard_uuid,),
                ).fetchone()
                if not row:
                    raise ConflictError("Flashcard not found", entity="flashcards", identifier=flashcard_uuid)  # noqa: TRY003
                cursor = conn.execute(
                    """
                    UPDATE flashcards
                       SET source_ref_type = ?, source_ref_id = ?, last_modified = ?,
                           version = version + 1, client_id = ?
                     WHERE id = ? AND deleted = 0
                    """,
                    (
                        normalized_source_ref_type,
                        normalized_source_ref_id,
                        now,
                        self._ctx.client_id,
                        int(row["id"]),
                    ),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to update flashcard source summary: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to update flashcard source summary: {exc}") from exc  # noqa: TRY003

    def get_study_pack_for_flashcard(self, flashcard_uuid: str) -> dict[str, Any] | None:
        """Return the first active study pack containing the flashcard."""
        owner_clause, owner_params = self._ctx.owner_filter(self._ctx.client_id, "sp")
        if owner_params:
            owner_clause += (
                " AND spc.client_id = ? AND EXISTS (SELECT 1 FROM flashcards f "
                "WHERE f.uuid = spc.flashcard_uuid AND f.client_id = ?)"
            )
            owner_params += (self._ctx.client_id, self._ctx.client_id)
        query = f"""
            SELECT sp.id, sp.workspace_id, sp.title, sp.deck_id, sp.source_bundle_json, sp.generation_options_json,
                   sp.status, sp.superseded_by_pack_id, sp.created_at, sp.last_modified, sp.deleted, sp.client_id, sp.version
              FROM study_pack_cards spc
              JOIN study_packs sp ON sp.id = spc.study_pack_id
             WHERE spc.flashcard_uuid = ? AND spc.deleted = 0 AND sp.deleted = 0{owner_clause}
             ORDER BY spc.id ASC, sp.id ASC
             LIMIT 1
        """  # nosec B608 - fixed owner predicate; values remain bound.
        try:
            cursor = self._ctx.execute_query(query, (flashcard_uuid, *owner_params), read_only=True)
            row = cursor.fetchone()
            return self._ctx.deserialize_row_fields(row, _STUDY_PACK_JSON_FIELDS)
        except CharactersRAGDBError:  # noqa: TRY203
            raise

    def soft_delete_study_pack(self, pack_id: int, expected_version: int | None = None) -> bool:
        """Soft delete a study pack with optional optimistic locking."""
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                self._ctx.require_owner_row(conn, "study_packs", pack_id, self._ctx.client_id, include_deleted=True)
                row = conn.execute("SELECT version, deleted FROM study_packs WHERE id = ?", (pack_id,)).fetchone()
                if not row:
                    raise ConflictError("Study pack not found", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                current_version = int(row["version"])
                deleted = bool(row["deleted"])
                if deleted:
                    return True
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError("Version mismatch deleting study pack", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                update_query = """
                    UPDATE study_packs
                       SET deleted = 1, last_modified = ?, version = version + 1, client_id = ?
                     WHERE id = ? AND deleted = 0
                """
                update_params: list[Any] = [now, self._ctx.client_id, pack_id]
                if expected_version is not None:
                    update_query += " AND version = ?"
                    update_params.append(expected_version)
                cursor = conn.execute(update_query, tuple(update_params))
                if cursor.rowcount == 0:
                    changed_row = conn.execute(
                        "SELECT version, deleted FROM study_packs WHERE id = ?",
                        (pack_id,),
                    ).fetchone()
                    if not changed_row:
                        raise ConflictError("Study pack not found", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                    if bool(changed_row["deleted"]):
                        return True
                    if expected_version is not None and int(changed_row["version"]) != expected_version:
                        raise ConflictError("Version mismatch deleting study pack", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                    raise ConflictError(
                        "Concurrent update deleting study pack",
                        entity="study_packs",
                        identifier=pack_id,
                    )
                return True
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Failed to delete study pack: {exc}") from exc  # noqa: TRY003

    def supersede_study_pack(
        self,
        pack_id: int,
        *,
        superseded_by_pack_id: int,
        expected_version: int | None = None,
    ) -> bool:
        """Mark a study pack superseded by another pack."""
        if superseded_by_pack_id == pack_id:
            raise InputError("Study pack cannot self-supersede")  # noqa: TRY003
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    locked_rows: dict[int, Any] = {}
                    for locked_pack_id in sorted((pack_id, superseded_by_pack_id)):
                        locked_row = conn.execute(
                            "SELECT id, version, deleted FROM study_packs WHERE id = ? AND client_id = ? FOR UPDATE",
                            (locked_pack_id, self._ctx.client_id),
                        ).fetchone()
                        if locked_row:
                            locked_rows[int(locked_row["id"])] = locked_row
                    row = locked_rows.get(pack_id)
                    replacement_row = locked_rows.get(superseded_by_pack_id)
                else:
                    row = conn.execute("SELECT version, deleted FROM study_packs WHERE id = ?", (pack_id,)).fetchone()
                    replacement_row = conn.execute(
                        "SELECT id, deleted FROM study_packs WHERE id = ?",
                        (superseded_by_pack_id,),
                    ).fetchone()

                if not row or bool(row["deleted"]):
                    raise ConflictError("Study pack not found", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                current_version = int(row["version"])
                if expected_version is not None and current_version != expected_version:
                    raise ConflictError("Version mismatch updating study pack", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                if not replacement_row:
                    raise ConflictError(
                        "Study pack replacement pack not found",
                        entity="study_packs",
                        identifier=superseded_by_pack_id,
                    )
                if bool(replacement_row["deleted"]):
                    raise ConflictError(
                        "Study pack replacement pack is deleted",
                        entity="study_packs",
                        identifier=superseded_by_pack_id,
                    )
                update_query = """
                    UPDATE study_packs
                       SET status = 'superseded',
                           superseded_by_pack_id = ?,
                           last_modified = ?,
                           version = version + 1,
                           client_id = ?
                     WHERE id = ? AND deleted = 0
                """
                update_params: list[Any] = [superseded_by_pack_id, now, self._ctx.client_id, pack_id]
                if expected_version is not None:
                    update_query += " AND version = ?"
                    update_params.append(expected_version)
                cursor = conn.execute(update_query, tuple(update_params))
                if cursor.rowcount == 0:
                    changed_row = conn.execute(
                        "SELECT version, deleted FROM study_packs WHERE id = ?",
                        (pack_id,),
                    ).fetchone()
                    if not changed_row:
                        raise ConflictError("Study pack not found", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                    if bool(changed_row["deleted"]):
                        raise ConflictError("Study pack not found", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                    if expected_version is not None and int(changed_row["version"]) != expected_version:
                        raise ConflictError("Version mismatch updating study pack", entity="study_packs", identifier=pack_id)  # noqa: TRY003
                    raise ConflictError(
                        "Concurrent update superseding study pack",
                        entity="study_packs",
                        identifier=pack_id,
                    )
                return True
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Failed to supersede study pack: {exc}") from exc  # noqa: TRY003
