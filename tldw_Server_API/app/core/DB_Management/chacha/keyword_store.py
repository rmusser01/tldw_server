from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    FTSQueryTranslator,
    InputError,
    logger,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import (
    NotesOrganizationValidationError,
    new_organization_sync_id,
    validate_resource_sync_id,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class KeywordStore:
    """Focused persistence seam for keyword and keyword-collection CRUD."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _deleted_literal(self, deleted: bool) -> str:
        """Return a backend-safe SQL literal for soft-delete predicates."""
        if self._db.backend_type == BackendType.POSTGRESQL:
            return "TRUE" if deleted else "FALSE"
        return "1" if deleted else "0"

    # ------------------------------------------------------------------
    # Keyword CRUD
    # ------------------------------------------------------------------

    def add_keyword(self, keyword_text: str) -> int | None:
        """
        Adds a new keyword or undeletes an existing soft-deleted one.

        Keyword text is stripped of leading/trailing whitespace.
        Uniqueness is case-insensitive due to `COLLATE NOCASE` on the `keyword` column (schema).
        FTS and sync_log entries are handled by SQL triggers.

        Args:
            keyword_text: The text of the keyword. Cannot be empty or whitespace only.

        Returns:
            The integer ID of the keyword.

        Raises:
            InputError: If `keyword_text` is empty or effectively empty after stripping.
            ConflictError: If an active keyword with the same text already exists, or if undelete fails.
            CharactersRAGDBError: For other database errors.
        """
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword text cannot be empty.")  # noqa: TRY003
        return self._db._add_generic_item(
            "keywords",
            "keyword",
            {"sync_id": new_organization_sync_id(), "merged_into_sync_id": None},
            keyword_text.strip(),
            {"sync_id": "sync_id", "merged_into_sync_id": "merged_into_sync_id"},
            owner_client_id=self._db.client_id,
        )

    def resolve_merge_survivor(
        self, sync_id: str, *, conn: Any | None = None, for_update: bool = False,
    ) -> dict[str, Any] | None:
        """Resolve current local identity; locked resolution belongs to the caller transaction.

        PostgreSQL applies the selected owner at every hop. SQLite retains its
        per-file/device-label contract. Invalid, deleted and cyclic chains have
        no survivor. A changed chain after lock acquisition is retryable.
        """
        if for_update and conn is None:
            raise InputError("Locked keyword resolution requires a caller transaction")  # noqa: TRY003
        table = self._db._map_table_for_backend("keywords")
        owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id)
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        current = sync_id
        while len(chain) < 100:
            try:
                validate_resource_sync_id(current)
            except NotesOrganizationValidationError:
                return None
            if current in seen:
                return None
            seen.add(current)
            query = f"SELECT * FROM {table} WHERE sync_id = ?{owner_clause}"  # nosec B608
            params = (current,) + owner_params
            cursor = conn.execute(query, params) if conn is not None else self._db.execute_query(query, params, read_only=True)
            row = cursor.fetchone()
            if row is None:
                return None
            item = dict(row)
            chain.append(item)
            if not item["deleted"]:
                break
            current = item["merged_into_sync_id"]
        else:
            return None

        if for_update:
            ids = sorted(item["id"] for item in chain)
            placeholders = ", ".join("?" for _ in ids)
            lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
            locked = conn.execute(
                f"SELECT * FROM {table} WHERE id IN ({placeholders}){owner_clause} ORDER BY id{lock}",  # nosec B608
                tuple(ids) + owner_params,
            ).fetchall()
            if [dict(row) for row in locked] != sorted(chain, key=lambda item: item["id"]):
                raise ConflictError("Keyword merge chain changed during resolution", entity="keywords", entity_id=sync_id)  # noqa: TRY003
        return chain[-1]

    def get_keyword_by_id(self, keyword_id: int) -> dict[str, Any] | None:
        """
        Retrieves a keyword by its integer ID. Returns active (non-deleted) keywords only.

        Args:
            keyword_id: The ID of the keyword.

        Returns:
            Keyword data as a dictionary, or None if not found/deleted.
        """
        return self._db._get_generic_item_by_id("keywords", keyword_id, owner_client_id=self._db.client_id)

    def get_keyword_by_text(self, keyword_text: str) -> dict[str, Any] | None:
        """
        Retrieves a keyword by its text (case-insensitive due to schema).
        Returns active (non-deleted) keywords only.

        Args:
            keyword_text: The text of the keyword (stripped before query).

        Returns:
            Keyword data as a dictionary, or None if not found/deleted.
        """
        return self._db._get_generic_item_by_unique_text("keywords", "keyword", keyword_text.strip(), owner_client_id=self._db.client_id)

    def list_keywords(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """
        Lists active keywords, ordered by text (case-insensitively).

        Args:
            limit: Max number of keywords.
            offset: Number to skip.

        Returns:
            A list of keyword dictionaries.
        """
        return self._db._list_generic_items("keywords", "keyword COLLATE NOCASE", limit, offset, owner_client_id=self._db.client_id)

    def count_keywords(self) -> int:
        """Return count of active (non-deleted) keywords."""
        keyword_table = self._db._map_table_for_backend("keywords")
        query = (
            f"SELECT COUNT(*) AS cnt FROM {keyword_table} "  # nosec B608
            f"WHERE deleted = {self._deleted_literal(False)}"
        )
        owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id)
        query += owner_clause
        try:
            cursor = self._db.execute_query(query, owner_params)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0
        except CharactersRAGDBError as exc:
            logger.error(f"Error counting keywords: {exc}")
            raise

    def soft_delete_keyword(self, keyword_id: int, expected_version: int) -> bool:
        """
        Soft-deletes a keyword using optimistic locking.

        Sets `deleted = 1`, updates metadata. Succeeds if `expected_version` matches
        and record is active. Idempotent if already deleted.
        FTS and sync_log handled by triggers.

        Args:
            keyword_id: The ID of the keyword to soft-delete.
            expected_version: The version number the client expects the record to have.

        Returns:
            True if successful or already deleted.

        Raises:
            ConflictError: If not found (not deleted), or active with version mismatch.
            CharactersRAGDBError: For other database errors.
        """
        return self._db._soft_delete_generic_item(
            table_name="keywords",
            item_id=keyword_id,
            expected_version=expected_version,
            pk_col_name="id",  # Explicitly pass, though "id" is default
            owner_client_id=self._db.client_id,
        )

    def rename_keyword(self, keyword_id: int, new_keyword_text: str, expected_version: int) -> dict[str, Any]:
        """
        Rename an active keyword using optimistic locking.

        Raises:
            InputError: If new keyword text is empty.
            ConflictError: If versions conflict or the destination text already exists.
            CharactersRAGDBError: On storage failures.
        """
        normalized_text = str(new_keyword_text or "").strip()
        if not normalized_text:
            raise InputError("Keyword text cannot be empty.")  # noqa: TRY003

        keyword_table = self._db._map_table_for_backend("keywords")
        owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id)
        now_iso = self._db._get_current_utc_timestamp_iso()
        next_version = expected_version + 1

        try:
            with self._db.transaction() as conn:
                self._db._require_selected_owner_row(conn, "keywords", keyword_id, self._db.client_id)
                current_version = self._db._get_current_db_version(conn, keyword_table, "id", keyword_id)
                if current_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Keyword {keyword_id} version mismatch (db={current_version}, expected={expected_version}).",
                        entity="keywords",
                        entity_id=keyword_id,
                    )

                duplicate = conn.execute(
                    f"SELECT id FROM {keyword_table} WHERE deleted = {self._deleted_literal(False)} AND keyword = ? AND id <> ?{owner_clause} LIMIT 1",  # nosec B608
                    (normalized_text, keyword_id) + owner_params,
                ).fetchone()
                if duplicate:
                    raise ConflictError(  # noqa: TRY003
                        f"Keyword '{normalized_text}' already exists.",
                        entity="keywords",
                        entity_id=normalized_text,
                    )

                cursor = conn.execute(
                    (
                        f"UPDATE {keyword_table} "  # nosec B608
                        f"SET keyword = ?, last_modified = ?, version = ?, client_id = ? "
                        f"WHERE id = ? AND version = ? AND deleted = {self._deleted_literal(False)}{owner_clause}"
                    ),
                    (
                        normalized_text,
                        now_iso,
                        next_version,
                        self._db.client_id,
                        keyword_id,
                        expected_version,
                    ) + owner_params,
                )
                if cursor.rowcount == 0:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Keyword {keyword_id} update affected 0 rows.",
                        entity="keywords",
                        entity_id=keyword_id,
                    )

                refreshed = conn.execute(
                    f"SELECT * FROM {keyword_table} WHERE id = ? AND deleted = {self._deleted_literal(False)}{owner_clause}",  # nosec B608
                    (keyword_id,) + owner_params,
                ).fetchone()
                if not refreshed:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Keyword {keyword_id} not found after rename.",
                        entity="keywords",
                        entity_id=keyword_id,
                    )
                return dict(refreshed)
        except sqlite3.IntegrityError as exc:
            if "unique constraint failed" in str(exc).lower():
                raise ConflictError(  # noqa: TRY003
                    f"Keyword '{normalized_text}' already exists.",
                    entity="keywords",
                    entity_id=normalized_text,
                ) from exc
            raise CharactersRAGDBError(f"Failed to rename keyword {keyword_id}: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            if self._db._is_unique_violation(exc):
                raise ConflictError(  # noqa: TRY003
                    f"Keyword '{normalized_text}' already exists.",
                    entity="keywords",
                    entity_id=normalized_text,
                ) from exc
            raise CharactersRAGDBError(f"Failed to rename keyword {keyword_id}: {exc}") from exc  # noqa: TRY003

    def _merge_keyword_links_for_table(
        self,
        conn: Any,
        *,
        link_table: str,
        entity_column: str,
        source_keyword_id: int,
        target_keyword_id: int,
        now_iso: str,
    ) -> int:
        """Move link-table references from source keyword to target keyword."""
        owner_clause, owner_params = self._db._selected_keyword_link_filter(link_table, "src", owner_client_id=self._db.client_id)
        if self._db.backend_type == BackendType.POSTGRESQL:
            insert_sql = (
                f"INSERT INTO {link_table} ({entity_column}, keyword_id, created_at) "  # nosec B608
                f"SELECT src.{entity_column}, ?, ? "
                f"FROM {link_table} src "
                f"WHERE src.keyword_id = ?{owner_clause} "
                f"ON CONFLICT ({entity_column}, keyword_id) DO NOTHING"
            )
        else:
            insert_sql = (
                f"INSERT OR IGNORE INTO {link_table} ({entity_column}, keyword_id, created_at) "  # nosec B608
                f"SELECT src.{entity_column}, ?, ? "
                f"FROM {link_table} src "
                f"WHERE src.keyword_id = ?"
            )

        insert_cursor = conn.execute(
            insert_sql,
            (target_keyword_id, now_iso, source_keyword_id) + owner_params,
        )
        inserted_count = int(insert_cursor.rowcount or 0)
        if inserted_count < 0:
            inserted_count = 0

        if owner_params:
            conn.execute(
                f"DELETE FROM {link_table} src WHERE keyword_id = ?{owner_clause}",  # nosec B608
                (source_keyword_id,) + owner_params,
            )
        else:
            conn.execute(
                f"DELETE FROM {link_table} WHERE keyword_id = ?",  # nosec B608
                (source_keyword_id,),
            )
        return inserted_count

    def synchronized_merge_snapshot(
        self,
        *,
        source_keyword_id: int,
        target_keyword_id: int,
        expected_source_version: int,
        expected_target_version: int | None = None,
    ) -> dict[str, Any]:
        """Read one stable merge plan without changing keyword projection state."""

        if source_keyword_id == target_keyword_id:
            raise InputError("Source and target keyword IDs must differ.")
        keyword_table = self._db._map_table_for_backend("keywords")
        collection_table = self._db._map_table_for_backend("keyword_collections")
        owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id)
        active = self._deleted_literal(False)
        with self._db.transaction() as conn:
            source = conn.execute(
                f"SELECT * FROM {keyword_table} WHERE id = ? AND deleted = {active}{owner_clause}",  # nosec B608
                (source_keyword_id,) + owner_params,
            ).fetchone()
            target = conn.execute(
                f"SELECT * FROM {keyword_table} WHERE id = ? AND deleted = {active}{owner_clause}",  # nosec B608
                (target_keyword_id,) + owner_params,
            ).fetchone()
            if not source or int(source["version"]) != expected_source_version:
                raise ConflictError(
                    "Source keyword version changed before merge",
                    entity="keywords",
                    entity_id=source_keyword_id,
                )
            if not target or (
                expected_target_version is not None
                and int(target["version"]) != expected_target_version
            ):
                raise ConflictError(
                    "Target keyword version changed before merge",
                    entity="keywords",
                    entity_id=target_keyword_id,
                )

            note_scope, note_params = self._db._selected_keyword_link_filter("note_keywords", "links", owner_client_id=self._db.client_id)
            conversation_scope, conversation_params = self._db._selected_keyword_link_filter("conversation_keywords", "links", owner_client_id=self._db.client_id)
            collection_scope, collection_params = self._db._selected_keyword_link_filter("collection_keywords", "links", owner_client_id=self._db.client_id)
            flashcard_scope, flashcard_params = self._db._selected_keyword_link_filter("flashcard_keywords", "links", owner_client_id=self._db.client_id)
            target_notes = {
                str(row["note_id"])
                for row in conn.execute(
                    f"SELECT note_id FROM note_keywords links WHERE keyword_id = ?{note_scope}",  # nosec B608 - fixed owner SQL fragments; values stay bound.
                    (target_keyword_id,) + note_params,
                ).fetchall()
            }
            target_conversations = {
                str(row["conversation_id"])
                for row in conn.execute(
                    "SELECT conversation_id FROM conversation_keywords links "
                    f"WHERE keyword_id = ?{conversation_scope}",  # nosec B608 - fixed owner SQL fragments; values stay bound.
                    (target_keyword_id,) + conversation_params,
                ).fetchall()
            }
            target_collections = {
                str(row["sync_id"])
                for row in conn.execute(
                    f"SELECT c.sync_id FROM collection_keywords links "  # nosec B608
                    f"JOIN {collection_table} c ON c.id = links.collection_id "
                    f"WHERE links.keyword_id = ?{collection_scope}",
                    (target_keyword_id,) + collection_params,
                ).fetchall()
            }
            relationships: list[dict[str, object]] = []
            for row in conn.execute(
                f"SELECT note_id FROM note_keywords links WHERE keyword_id = ?{note_scope} ORDER BY note_id",  # nosec B608 - fixed owner SQL fragments; values stay bound.
                (source_keyword_id,) + note_params,
            ).fetchall():
                note_id = str(row["note_id"])
                relationships.append(
                    {
                        "domain": "notes.keyword_link",
                        "members": {
                            "subject_type": "note",
                            "subject_id": note_id,
                        },
                        "target_present": note_id in target_notes,
                    }
                )
            for row in conn.execute(
                "SELECT conversation_id FROM conversation_keywords links "
                f"WHERE keyword_id = ?{conversation_scope} ORDER BY conversation_id",  # nosec B608 - fixed owner SQL fragments; values stay bound.
                (source_keyword_id,) + conversation_params,
            ).fetchall():
                conversation_id = str(row["conversation_id"])
                relationships.append(
                    {
                        "domain": "notes.keyword_link",
                        "members": {
                            "subject_type": "conversation",
                            "subject_id": conversation_id,
                        },
                        "target_present": conversation_id in target_conversations,
                    }
                )
            for row in conn.execute(
                f"SELECT c.sync_id FROM collection_keywords links "  # nosec B608
                f"JOIN {collection_table} c ON c.id = links.collection_id "
                f"WHERE links.keyword_id = ?{collection_scope} ORDER BY c.sync_id",
                (source_keyword_id,) + collection_params,
            ).fetchall():
                collection_sync_id = str(row["sync_id"])
                relationships.append(
                    {
                        "domain": "notes.keyword_collection_link",
                        "members": {"collection_sync_id": collection_sync_id},
                        "target_present": collection_sync_id in target_collections,
                    }
                )
            flashcard_dependency = conn.execute(
                f"SELECT 1 FROM flashcard_keywords links WHERE keyword_id = ?{flashcard_scope} LIMIT 1",  # nosec B608 - fixed owner SQL fragments; values stay bound.
                (source_keyword_id,) + flashcard_params,
            ).fetchone()
            return {
                "source": dict(source),
                "target": dict(target),
                "relationships": tuple(relationships),
                "has_unsynchronized_dependency": bool(flashcard_dependency),
            }

    @staticmethod
    def _relationship_set_hash(relationships: list[tuple[str, str]]) -> str:
        encoded = json.dumps(
            sorted(relationships),
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def empty_synchronized_relationship_set_hash(cls) -> str:
        """Return the opaque token for a keyword with no dependent relationships."""

        return cls._relationship_set_hash([])

    def synchronized_relationship_set_hash(
        self,
        keyword_id: int,
        *,
        conn: Any | None = None,
    ) -> str:
        """Hash all synchronized and unsynchronized keyword relationships."""

        def _locked(connection: Any) -> str:
            relationships: list[tuple[str, str]] = []
            for tag, table, column in (
                ("note", "note_keywords", "note_id"),
                ("conversation", "conversation_keywords", "conversation_id"),
                ("collection", "collection_keywords", "collection_id"),
                ("flashcard", "flashcard_keywords", "card_id"),
            ):
                rows = connection.execute(
                    f"SELECT {column} FROM {table} WHERE keyword_id = ?",  # nosec B608
                    (keyword_id,),
                ).fetchall()
                relationships.extend((tag, str(row[column])) for row in rows)
            return self._relationship_set_hash(relationships)

        if conn is not None:
            return _locked(conn)
        with self._db.transaction() as transaction:
            return _locked(transaction)

    def merge_keywords(
        self,
        *,
        source_keyword_id: int,
        target_keyword_id: int,
        expected_source_version: int,
        expected_target_version: int | None = None,
    ) -> dict[str, Any]:
        """
        Merge one keyword into another and soft-delete the source keyword atomically.

        Link tables migrated:
        - note_keywords
        - conversation_keywords
        - collection_keywords
        - flashcard_keywords
        """
        if source_keyword_id == target_keyword_id:
            raise InputError("Source and target keyword IDs must differ.")  # noqa: TRY003

        keyword_table = self._db._map_table_for_backend("keywords")
        owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id)
        now_iso = self._db._get_current_utc_timestamp_iso()
        source_next_version = expected_source_version + 1

        try:
            with self._db.transaction() as conn:
                for keyword_id in sorted((source_keyword_id, target_keyword_id)):
                    self._db._require_selected_owner_row(conn, "keywords", keyword_id, self._db.client_id)
                source_version = self._db._get_current_db_version(
                    conn, keyword_table, "id", source_keyword_id
                )
                if source_version != expected_source_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        (
                            f"Source keyword {source_keyword_id} version mismatch "
                            f"(db={source_version}, expected={expected_source_version})."
                        ),
                        entity="keywords",
                        entity_id=source_keyword_id,
                    )

                target_version = self._db._get_current_db_version(
                    conn, keyword_table, "id", target_keyword_id
                )
                if (
                    expected_target_version is not None
                    and target_version != expected_target_version
                ):
                    raise ConflictError(  # noqa: TRY003, TRY301
                        (
                            f"Target keyword {target_keyword_id} version mismatch "
                            f"(db={target_version}, expected={expected_target_version})."
                        ),
                        entity="keywords",
                        entity_id=target_keyword_id,
                    )

                merged_note_links = self._merge_keyword_links_for_table(
                    conn,
                    link_table="note_keywords",
                    entity_column="note_id",
                    source_keyword_id=source_keyword_id,
                    target_keyword_id=target_keyword_id,
                    now_iso=now_iso,
                )
                merged_conversation_links = self._merge_keyword_links_for_table(
                    conn,
                    link_table="conversation_keywords",
                    entity_column="conversation_id",
                    source_keyword_id=source_keyword_id,
                    target_keyword_id=target_keyword_id,
                    now_iso=now_iso,
                )
                merged_collection_links = self._merge_keyword_links_for_table(
                    conn,
                    link_table="collection_keywords",
                    entity_column="collection_id",
                    source_keyword_id=source_keyword_id,
                    target_keyword_id=target_keyword_id,
                    now_iso=now_iso,
                )
                merged_flashcard_links = self._merge_keyword_links_for_table(
                    conn,
                    link_table="flashcard_keywords",
                    entity_column="card_id",
                    source_keyword_id=source_keyword_id,
                    target_keyword_id=target_keyword_id,
                    now_iso=now_iso,
                )

                target_row = conn.execute(
                    f"SELECT sync_id FROM {keyword_table} WHERE id = ?{owner_clause}",  # nosec B608
                    (target_keyword_id,) + owner_params,
                ).fetchone()
                target_sync_id = validate_resource_sync_id(target_row["sync_id"])
                soft_delete_cursor = conn.execute(
                    (
                        f"UPDATE {keyword_table} "  # nosec B608
                        f"SET merged_into_sync_id = ?, deleted = {self._deleted_literal(True)}, last_modified = ?, version = ?, client_id = ? "
                        f"WHERE id = ? AND version = ? AND deleted = {self._deleted_literal(False)}{owner_clause}"
                    ),
                    (
                        target_sync_id,
                        now_iso,
                        source_next_version,
                        self._db.client_id,
                        source_keyword_id,
                        expected_source_version,
                    ) + owner_params,
                )
                if soft_delete_cursor.rowcount == 0:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        (
                            f"Source keyword {source_keyword_id} changed during merge "
                            f"(expected version {expected_source_version})."
                        ),
                        entity="keywords",
                        entity_id=source_keyword_id,
                    )

                return {
                    "source_keyword_id": source_keyword_id,
                    "target_keyword_id": target_keyword_id,
                    "source_deleted_version": source_next_version,
                    "target_version": target_version,
                    "merged_note_links": merged_note_links,
                    "merged_conversation_links": merged_conversation_links,
                    "merged_collection_links": merged_collection_links,
                    "merged_flashcard_links": merged_flashcard_links,
                }
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(
                f"Failed to merge keyword {source_keyword_id} into {target_keyword_id}: {exc}"
            ) from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(
                f"Failed to merge keyword {source_keyword_id} into {target_keyword_id}: {exc}"
            ) from exc  # noqa: TRY003

    def search_keywords(self, search_term: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Searches keywords by text using FTS.

        Matches against the 'keyword' field in `keywords_fts`.
        Returns active keywords, ordered by relevance.

        Args:
            search_term: FTS query string for keyword text.
            limit: Max number of results.

        Returns:
            A list of matching keyword dictionaries.
        """
        if search_term is None:
            raise InputError("Search term cannot be empty.")  # noqa: TRY003
        search_term = search_term.strip()
        if not search_term:
            raise InputError("Search term cannot be empty.")  # noqa: TRY003
        if '"' in search_term or "'" in search_term:
            raise InputError("Search term contains unsupported characters.")  # noqa: TRY003
        # SQLite FTS prefix queries treat punctuation such as "-" as operators.
        # Keep raw FTS only for plain word tokens and fall back to LIKE/ILIKE
        # for literal punctuation searches like "C++" or "foo-bar".
        is_simple_token = re.fullmatch(r"\w+", search_term) is not None
        if self._db.backend_type == BackendType.POSTGRESQL:
            owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id, "k")
            if is_simple_token:
                tsquery = FTSQueryTranslator.normalize_query(search_term, 'postgresql')
                if tsquery:
                    source_table = self._db._map_table_for_backend("keywords")
                    fts_column = "keywords_fts_tsv"
                    query = """
                        SELECT k.*, ts_rank(k.{fts_column}, to_tsquery('english', ?)) AS rank
                        FROM {source_table} k
                        WHERE k.deleted = FALSE{owner_clause}
                          AND k.{fts_column} @@ to_tsquery('english', ?)
                        ORDER BY rank DESC, k.last_modified DESC
                        LIMIT ?
                    """.format_map(locals())  # nosec B608
                    try:
                        cursor = self._db.execute_query(query, (tsquery,) + owner_params + (tsquery, limit))
                        return [dict(row) for row in cursor.fetchall()]
                    except CharactersRAGDBError as exc:
                        logger.error("PostgreSQL FTS search failed for keywords term '{}': {}", search_term, exc)
                        raise
                logger.debug("Keyword search term normalized to empty tsquery for input '{}'", search_term)
                return []

            # Non-simple tokens (e.g., "C++") skip FTS and fall back to ILIKE.
            source_table = self._db._map_table_for_backend("keywords")
            escaped = search_term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query = """
                SELECT k.*
                FROM {source_table} k
                WHERE k.deleted = FALSE{owner_clause}
                  AND k.keyword ILIKE (?) ESCAPE '\\'
                ORDER BY k.last_modified DESC
                LIMIT ?
            """.format_map(locals())  # nosec B608
            cursor = self._db.execute_query(query, owner_params + (f"%{escaped}%", limit))
            return [dict(row) for row in cursor.fetchall()]

        # SQLite: use FTS prefix for simple tokens.
        if is_simple_token:
            # Support prefix/substring search expectations in tests by using prefix match
            # e.g., 'fru' should match 'fruit'. FTS5 uses '*' for prefix queries.
            fts_query = f"{search_term}*"
            try:
                return self._db._search_generic_items_fts("keywords_fts", "keywords", "keyword", fts_query, limit)
            except CharactersRAGDBError as exc:
                msg = str(exc).lower()
                if "fts" in msg or "match" in msg or "syntax" in msg:
                    raise InputError("Search term contains unsupported characters.") from exc  # noqa: TRY003
                raise

        keyword_table = self._db._map_table_for_backend("keywords")
        escaped = search_term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        deleted_false = self._deleted_literal(False)
        query = """
            SELECT k.*
            FROM {keyword_table} k
            WHERE k.deleted = {deleted_false}
              AND k.keyword LIKE ? ESCAPE '\\'
            ORDER BY k.last_modified DESC
            LIMIT ?
        """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (f"%{escaped}%", limit))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Keyword Collection CRUD
    # ------------------------------------------------------------------

    def add_keyword_collection(self, name: str, parent_id: int | None = None) -> int | None:
        """
        Adds a new keyword collection or undeletes an existing one.

        Collection name is stripped. Uniqueness is case-insensitive (`COLLATE NOCASE` in schema).
        FTS and sync_log handled by triggers.

        Args:
            name: The name of the collection. Cannot be empty or whitespace only.
            parent_id: Optional integer ID of a parent collection for hierarchy.

        Returns:
            The integer ID of the collection.

        Raises:
            InputError: If `name` is empty.
            ConflictError: If an active collection with the same name exists, or undelete fails.
            CharactersRAGDBError: For other DB errors.
        """
        if not name or not name.strip():
            raise InputError("Collection name cannot be empty.")  # noqa: TRY003
        with self._db.transaction() as conn:
            self._db._require_selected_owner_row(conn, "keyword_collections", parent_id, self._db.client_id)
            return self._db._add_generic_item(
                "keyword_collections",
                "name",
                {"parent_id": parent_id, "sync_id": new_organization_sync_id()},
                name.strip(),
                {"parent_id": "parent_id", "sync_id": "sync_id"},
                owner_client_id=self._db.client_id,
            )

    def get_keyword_collection_by_id(self, collection_id: int) -> dict[str, Any] | None:
        """
        Retrieves a keyword collection by ID. Active collections only.

        Args:
            collection_id: ID of the collection.

        Returns:
            Collection data as dictionary, or None.
        """
        return self._db._get_generic_item_by_id("keyword_collections", collection_id, owner_client_id=self._db.client_id)

    def get_keyword_collection_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Retrieves a keyword collection by name (case-insensitive). Active collections only.

        Args:
            name: Name of the collection (stripped).

        Returns:
            Collection data as dictionary, or None.
        """
        return self._db._get_generic_item_by_unique_text("keyword_collections", "name", name.strip(), owner_client_id=self._db.client_id)

    def list_keyword_collections(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """
        Lists active keyword collections, ordered by name (case-insensitively).

        Args:
            limit: Max number of collections.
            offset: Number to skip.

        Returns:
            A list of collection dictionaries.
        """
        return self._db._list_generic_items("keyword_collections", "name COLLATE NOCASE", limit, offset, owner_client_id=self._db.client_id)

    def count_keyword_collections(self) -> int:
        """Return count of active (non-deleted) keyword collections."""
        collections_table = self._db._map_table_for_backend("keyword_collections")
        query = (
            f"SELECT COUNT(*) AS cnt FROM {collections_table} "  # nosec B608
            f"WHERE deleted = {self._deleted_literal(False)}"
        )
        owner_clause, owner_params = self._db._selected_owner_filter(self._db.client_id)
        query += owner_clause
        try:
            cursor = self._db.execute_query(query, owner_params, read_only=True)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0
        except CharactersRAGDBError as exc:
            logger.error(f"Error counting keyword collections: {exc}")
            raise

    def update_keyword_collection(self, collection_id: int, update_data: dict[str, Any], expected_version: int) -> bool:
        """
        Updates a keyword collection with optimistic locking.

        Args:
            collection_id: The ID of the keyword collection to update.
            update_data: A dictionary containing the fields to update (e.g., 'name', 'parent_id').
            expected_version: The version number the client expects the record to have.

        Returns:
            True if the update was successful.

        Raises:
            InputError: If no update data is provided.
            ConflictError: If the record is not found, already soft-deleted,
                           or if the expected_version does not match the current database version.
            CharactersRAGDBError: For other database-related errors.
        """
        # pk_col_name for 'keyword_collections' is 'id' (default in _update_generic_item)
        # item_id is int.
        with self._db.transaction() as conn:
            self._db._require_selected_owner_row(conn, "keyword_collections", update_data.get("parent_id"), self._db.client_id)
            return self._db._update_generic_item(
                table_name="keyword_collections",
                item_id=collection_id,
                update_data=update_data,
                expected_version=expected_version,
                allowed_fields=['name', 'parent_id'],
                pk_col_name="id",  # Explicitly pass, though "id" is default
                unique_col_name_in_data='name',  # For handling unique constraint on name if it's updated
                owner_client_id=self._db.client_id,
            )

    def soft_delete_keyword_collection(self, collection_id: int, expected_version: int) -> bool:
        """
        Soft-deletes a keyword collection with optimistic locking.

        Args:
            collection_id: The ID of the keyword collection to soft-delete.
            expected_version: The version number the client expects the record to have.

        Returns:
            True if the soft-delete was successful or if the collection was already soft-deleted.

        Raises:
            ConflictError: If the record is not found, or if (it's active and)
                           the expected_version does not match the current database version.
            CharactersRAGDBError: For other database-related errors.
        """
        # pk_col_name for 'keyword_collections' is 'id' (default in _soft_delete_generic_item)
        # item_id is int.
        return self._db._soft_delete_generic_item(
            table_name="keyword_collections",
            item_id=collection_id,
            expected_version=expected_version,
            pk_col_name="id",  # Explicitly pass, though "id" is default
            owner_client_id=self._db.client_id,
        )

    def search_keyword_collections(self, search_term: str, limit: int = 10) -> list[dict[str, Any]]:
        safe_literal = search_term.replace('"', '""')
        safe_search_term = f'"{safe_literal}"'
        return self._db._search_generic_items_fts("keyword_collections_fts", "keyword_collections", "name",
                                                  safe_search_term, limit, owner_client_id=self._db.client_id)

    # ------------------------------------------------------------------
    # Conversation <-> Keyword links
    # ------------------------------------------------------------------

    def link_conversation_to_keyword(self, conversation_id: str, keyword_id: int) -> bool:
        return self._db._manage_link("conversation_keywords", "conversation_id", conversation_id, "keyword_id",
                                     keyword_id, "link", owner_client_id=self._db.client_id)

    def unlink_conversation_from_keyword(self, conversation_id: str, keyword_id: int) -> bool:
        return self._db._manage_link("conversation_keywords", "conversation_id", conversation_id, "keyword_id",
                                     keyword_id, "unlink", owner_client_id=self._db.client_id)

    def get_keywords_for_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        owner_clause, owner_params = self._db._selected_keyword_link_filter("conversation_keywords", "ck", owner_client_id=self._db.client_id)
        keyword_table = self._db._map_table_for_backend("keywords")
        order_clause = self._db._case_insensitive_order_clause("k.keyword")
        deleted_false = self._deleted_literal(False)
        query = """
                SELECT k.* \
                FROM {keyword_table} k \
                         JOIN conversation_keywords ck ON k.id = ck.keyword_id
                WHERE ck.conversation_id = ? \
                  AND k.deleted = {deleted_false}{owner_clause} \
                {order_clause}
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (conversation_id,) + owner_params)
        return [dict(row) for row in cursor.fetchall()]

    def count_conversation_keyword_links(self) -> int:
        owner_clause, owner_params = self._db._selected_keyword_link_filter("conversation_keywords", "ck", owner_client_id=self._db.client_id)
        cursor = self._db.execute_query(f"SELECT COUNT(*) AS total FROM conversation_keywords ck WHERE 1 = 1{owner_clause}", owner_params)  # nosec B608
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(row["total"] if isinstance(row, dict) else row[0])

    def list_conversation_keyword_links(self, limit: int, offset: int) -> list[dict[str, Any]]:
        """Conversation-keyword links, owner-scoped exactly as count_conversation_keyword_links."""
        owner_clause, owner_params = self._db._selected_keyword_link_filter("conversation_keywords", "ck", owner_client_id=self._db.client_id)
        cursor = self._db.execute_query(
            "SELECT ck.conversation_id, ck.keyword_id FROM conversation_keywords ck"  # nosec B608 - fixed fragments
            f" WHERE 1 = 1{owner_clause} ORDER BY ck.conversation_id ASC, ck.keyword_id ASC LIMIT ? OFFSET ?",
            (*owner_params, limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_keywords_for_conversations(self, conversation_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Fetch keywords for multiple conversations in a single query."""
        if not conversation_ids:
            return {}
        owner_clause, owner_params = self._db._selected_keyword_link_filter("conversation_keywords", "ck", owner_client_id=self._db.client_id)
        keyword_table = self._db._map_table_for_backend("keywords")
        placeholders = ",".join(["?"] * len(conversation_ids))
        order_expr = self._db._case_insensitive_order_expression("k.keyword")
        deleted_false = self._deleted_literal(False)
        query = """
                SELECT ck.conversation_id as conversation_id, k.* \
                FROM {keyword_table} k \
                         JOIN conversation_keywords ck ON k.id = ck.keyword_id
                WHERE ck.conversation_id IN ({placeholders}) \
                  AND k.deleted = {deleted_false}{owner_clause} \
                ORDER BY ck.conversation_id, {order_expr}
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, tuple(conversation_ids) + owner_params)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        result: dict[str, list[dict[str, Any]]] = {cid: [] for cid in conversation_ids}
        for row in rows:
            record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
            conv_id = record.pop("conversation_id", None)
            if conv_id is None:
                continue
            result.setdefault(str(conv_id), []).append(record)
        return result

    def get_conversations_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        deleted_false = self._deleted_literal(False)
        owner_clause, owner_params = self._db._selected_keyword_link_filter("conversation_keywords", "ck", owner_client_id=self._db.client_id)
        query = """
                SELECT c.* \
                FROM conversations c \
                         JOIN conversation_keywords ck ON c.id = ck.conversation_id
                WHERE ck.keyword_id = ? \
                  AND c.deleted = {deleted_false}{owner_clause}
                ORDER BY c.last_modified DESC LIMIT ? \
                OFFSET ? \
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (keyword_id,) + owner_params + (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Collection <-> Keyword links
    # ------------------------------------------------------------------

    def link_collection_to_keyword(self, collection_id: int, keyword_id: int) -> bool:
        return self._db._manage_link("collection_keywords", "collection_id", collection_id, "keyword_id", keyword_id,
                                     "link", owner_client_id=self._db.client_id)

    def unlink_collection_from_keyword(self, collection_id: int, keyword_id: int) -> bool:
        return self._db._manage_link("collection_keywords", "collection_id", collection_id, "keyword_id", keyword_id,
                                     "unlink", owner_client_id=self._db.client_id)

    def unlink_collection_to_keyword(
        self,
        collection_id: int,
        keyword_id: int,
    ) -> bool:  # pragma: no cover - compat alias
        """Backward-compatible alias for the extracted facade delegation typo."""
        return self.unlink_collection_from_keyword(collection_id, keyword_id)

    def get_keywords_for_collection(self, collection_id: int) -> list[dict[str, Any]]:
        owner_clause, owner_params = self._db._selected_keyword_link_filter("collection_keywords", "ck", owner_client_id=self._db.client_id)
        keyword_table = self._db._map_table_for_backend("keywords")
        order_clause = self._db._case_insensitive_order_clause("k.keyword")
        deleted_false = self._deleted_literal(False)
        query = """
                SELECT k.* \
                FROM {keyword_table} k \
                         JOIN collection_keywords ck ON k.id = ck.keyword_id
                WHERE ck.collection_id = ? \
                  AND k.deleted = {deleted_false}{owner_clause} \
                {order_clause}
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (collection_id,) + owner_params, read_only=True)
        return [dict(row) for row in cursor.fetchall()]

    def count_collection_keyword_links(self) -> int:
        owner_clause, owner_params = self._db._selected_keyword_link_filter("collection_keywords", "ck", owner_client_id=self._db.client_id)
        cursor = self._db.execute_query(f"SELECT COUNT(*) AS total FROM collection_keywords ck WHERE 1 = 1{owner_clause}", owner_params)  # nosec B608
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(row["total"] if isinstance(row, dict) else row[0])

    def list_collection_keyword_links(self, limit: int, offset: int) -> list[dict[str, Any]]:
        """Collection-keyword links, owner-scoped exactly as count_collection_keyword_links."""
        owner_clause, owner_params = self._db._selected_keyword_link_filter("collection_keywords", "ck", owner_client_id=self._db.client_id)
        cursor = self._db.execute_query(
            "SELECT ck.collection_id, ck.keyword_id FROM collection_keywords ck"  # nosec B608 - fixed fragments
            f" WHERE 1 = 1{owner_clause} ORDER BY ck.collection_id ASC, ck.keyword_id ASC LIMIT ? OFFSET ?",
            (*owner_params, limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_collections_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        order_clause = self._db._case_insensitive_order_clause("kc.name")
        deleted_false = self._deleted_literal(False)
        owner_clause, owner_params = self._db._selected_keyword_link_filter("collection_keywords", "ck", owner_client_id=self._db.client_id)
        query = """
                SELECT kc.* \
                FROM keyword_collections kc \
                         JOIN collection_keywords ck ON kc.id = ck.collection_id
                WHERE ck.keyword_id = ? \
                  AND kc.deleted = {deleted_false}{owner_clause}
                {order_clause} LIMIT ? \
                OFFSET ? \
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (keyword_id,) + owner_params + (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Note <-> Keyword links
    # ------------------------------------------------------------------

    def link_note_to_keyword(self, note_id: str, keyword_id: int) -> bool:  # note_id is str
        return self._db._manage_link("note_keywords", "note_id", note_id, "keyword_id", keyword_id, "link", owner_client_id=self._db.client_id)

    def unlink_note_from_keyword(self, note_id: str, keyword_id: int) -> bool:  # note_id is str
        return self._db._manage_link("note_keywords", "note_id", note_id, "keyword_id", keyword_id, "unlink", owner_client_id=self._db.client_id)

    def get_keywords_for_note(self, note_id: str) -> list[dict[str, Any]]:  # note_id is str
        owner_clause, owner_params = self._db._selected_keyword_link_filter("note_keywords", "nk", owner_client_id=self._db.client_id)
        keyword_table = self._db._map_table_for_backend("keywords")
        order_clause = self._db._case_insensitive_order_clause("k.keyword")
        deleted_false = self._deleted_literal(False)
        query = """
                SELECT k.* \
                FROM {keyword_table} k \
                         JOIN note_keywords nk ON k.id = nk.keyword_id
                WHERE nk.note_id = ? \
                  AND k.deleted = {deleted_false}{owner_clause} \
                {order_clause}
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (note_id,) + owner_params)
        return [dict(row) for row in cursor.fetchall()]

    def get_keywords_for_notes(self, note_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return keywords for multiple notes as a map of note_id -> keywords list."""
        if not note_ids:
            return {}
        owner_clause, owner_params = self._db._selected_keyword_link_filter("note_keywords", "nk", owner_client_id=self._db.client_id)
        keyword_table = self._db._map_table_for_backend("keywords")
        order_clause = self._db._case_insensitive_order_clause("k.keyword")
        deleted_false = self._deleted_literal(False)
        out: dict[str, list[dict[str, Any]]] = {nid: [] for nid in note_ids}
        # SQLite has a default variable cap of 999; keep a buffer to be safe.
        max_vars = 900
        for start in range(0, len(note_ids), max_vars):
            batch = note_ids[start:start + max_vars]
            placeholders = ",".join(["?"] * len(batch))
            query = """
                    SELECT nk.note_id AS note_id, k.* \
                    FROM {keyword_table} k \
                             JOIN note_keywords nk ON k.id = nk.keyword_id
                    WHERE nk.note_id IN ({placeholders}) \
                      AND k.deleted = {deleted_false}{owner_clause} \
                    {order_clause}
                    """.format_map(locals())  # nosec B608
            cursor = self._db.execute_query(query, tuple(batch) + owner_params)
            rows = cursor.fetchall()
            for row in rows:
                record = dict(row)
                note_id_val = record.pop("note_id", None)
                if not note_id_val:
                    continue
                out.setdefault(note_id_val, []).append(record)
        return out

    def get_notes_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        deleted_false = self._deleted_literal(False)
        owner_clause, owner_params = self._db._selected_keyword_link_filter("note_keywords", "nk", owner_client_id=self._db.client_id)
        query = """
                SELECT n.* \
                FROM notes n \
                         JOIN note_keywords nk ON n.id = nk.note_id
                WHERE nk.keyword_id = ? \
                  AND n.deleted = {deleted_false}{owner_clause}
                ORDER BY n.last_modified DESC LIMIT ? \
                OFFSET ? \
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (keyword_id,) + owner_params + (limit, offset))
        return [dict(row) for row in cursor.fetchall()]
