from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    FTSQueryTranslator,
    InputError,
    _SUPPORTED_NOTE_STUDIO_HANDWRITING_MODES,
    _SUPPORTED_NOTE_STUDIO_TEMPLATE_TYPES,
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    logger,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class NoteStore:
    """Focused persistence seam for note CRUD operations."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _deleted_value(self, deleted: bool) -> bool | int:
        """Return the backend-native value for a soft-delete flag."""
        return deleted if self._db.backend_type == BackendType.POSTGRESQL else int(deleted)

    # ------------------------------------------------------------------
    # Note creation
    # ------------------------------------------------------------------

    def add_note(
        self,
        title: str,
        content: str,
        note_id: str | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> str | None:
        if not title or not title.strip():
            raise InputError("Note title cannot be empty.")  # noqa: TRY003
        if content is None:  # Allow empty string for content
            raise InputError("Note content cannot be None.")  # noqa: TRY003

        final_note_id = note_id or self._db._generate_uuid()
        now = self._db._get_current_utc_timestamp_iso()
        client_id_to_use = self._db.client_id  # Notes use the instance's client_id directly
        normalized_conversation_id = self._db._normalize_nullable_text(conversation_id)
        normalized_message_id = self._db._normalize_nullable_text(message_id)

        query = """
            INSERT INTO notes (id, title, content, last_modified, client_id, version, deleted, created_at, conversation_id, message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        if self._db.backend_type == BackendType.POSTGRESQL:
            params = (
                final_note_id, title.strip(), content, now, client_id_to_use, 1, False, now,
                normalized_conversation_id, normalized_message_id
            )
        else:
            params = (
                final_note_id, title.strip(), content, now, client_id_to_use, 1, 0, now,
                normalized_conversation_id, normalized_message_id
            )

        try:
            def _execute(transaction_conn: sqlite3.Connection | BackendConnectionWrapper) -> str:
                transaction_conn.execute(query, params)
                logger.info(f"Added note '{title.strip()}' with ID: {final_note_id}.")
                return final_note_id

            if conn is None:
                with self._db.transaction() as transaction_conn:
                    return _execute(transaction_conn)
            return _execute(conn)
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "foreign key constraint failed" in msg:
                raise ConflictError("Conversation or message not found.", entity="notes", entity_id=final_note_id) from e  # noqa: TRY003
            if "unique constraint failed: notes.id" in msg:
                raise ConflictError(f"Note with ID '{final_note_id}' already exists.", entity="notes", entity_id=final_note_id) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Database integrity error adding note: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            msg = str(e).lower()
            if "foreign key" in msg:
                raise ConflictError("Conversation or message not found.", entity="notes", entity_id=final_note_id) from e  # noqa: TRY003
            if "duplicate key" in msg or "unique constraint" in msg:
                raise ConflictError(f"Note with ID '{final_note_id}' already exists.", entity="notes", entity_id=final_note_id) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Backend error adding note: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding note '{title.strip()}': {e}")
            raise

    def upsert_note_from_sync(
        self,
        *,
        note_id: str,
        title: str,
        content: str,
        conversation_id: str | None,
        message_id: str | None,
        sync_client_id: str,
        object_revision: int,
        object_hash: str,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> bool:
        """Create or update a note projection from an accepted Sync v2 envelope."""

        del object_hash
        normalized_note_id = str(note_id).strip()
        normalized_title = title.strip() if isinstance(title, str) else ""
        if not normalized_note_id:
            raise InputError("note_id cannot be empty.")  # noqa: TRY003
        if not normalized_title:
            raise InputError("Note title cannot be empty.")  # noqa: TRY003
        if content is None:
            raise InputError("Note content cannot be None.")  # noqa: TRY003
        if object_revision < 1:
            raise InputError("object_revision must be greater than zero.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        normalized_conversation_id = self._db._normalize_nullable_text(conversation_id)
        normalized_message_id = self._db._normalize_nullable_text(message_id)
        query = """
            INSERT INTO notes (
                id, title, content, last_modified, client_id, version, deleted,
                created_at, conversation_id, message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                last_modified = excluded.last_modified,
                client_id = excluded.client_id,
                version = excluded.version,
                deleted = excluded.deleted,
                conversation_id = excluded.conversation_id,
                message_id = excluded.message_id
        """
        params = (
            normalized_note_id,
            normalized_title,
            content,
            now,
            sync_client_id,
            object_revision,
            self._deleted_value(False),
            now,
            normalized_conversation_id,
            normalized_message_id,
        )

        try:
            def _execute(transaction_conn: sqlite3.Connection | BackendConnectionWrapper) -> bool:
                transaction_conn.execute(query, params)
                logger.info("Upserted note projection from Sync v2 for ID: {}.", normalized_note_id)
                return True

            if conn is None:
                with self._db.transaction() as transaction_conn:
                    return _execute(transaction_conn)
            return _execute(conn)
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "foreign key constraint failed" in msg:
                raise ConflictError("Conversation or message not found.", entity="notes", entity_id=normalized_note_id) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Database integrity error upserting synced note: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            msg = str(e).lower()
            if "foreign key" in msg:
                raise ConflictError("Conversation or message not found.", entity="notes", entity_id=normalized_note_id) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Backend error upserting synced note: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError:
            logger.error("Database error upserting synced note ID {}.", normalized_note_id, exc_info=True)
            raise

    def tombstone_note_from_sync(
        self,
        *,
        note_id: str,
        sync_client_id: str,
        object_revision: int,
        object_hash: str,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> bool:
        """Soft-delete a note projection from an accepted Sync v2 tombstone."""

        del object_hash
        normalized_note_id = str(note_id).strip()
        if not normalized_note_id:
            raise InputError("note_id cannot be empty.")  # noqa: TRY003
        if object_revision < 1:
            raise InputError("object_revision must be greater than zero.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        query = """
            UPDATE notes
               SET deleted = ?,
                   last_modified = ?,
                   version = ?,
                   client_id = ?
             WHERE id = ?
        """
        params = (
            self._deleted_value(True),
            now,
            object_revision,
            sync_client_id,
            normalized_note_id,
        )

        try:
            def _execute(transaction_conn: sqlite3.Connection | BackendConnectionWrapper) -> bool:
                cursor = transaction_conn.execute(query, params)
                if cursor.rowcount == 0:
                    raise ConflictError(  # noqa: TRY003
                        "Note not found for Sync v2 tombstone.",
                        entity="notes",
                        entity_id=normalized_note_id,
                    )
                self._db._invalidate_note_clipper_sidecars(normalized_note_id, conn=transaction_conn, deleted=True)
                logger.info("Soft-deleted note projection from Sync v2 for ID: {}.", normalized_note_id)
                return True

            if conn is None:
                with self._db.transaction() as transaction_conn:
                    return _execute(transaction_conn)
            return _execute(conn)
        except ConflictError:
            raise
        except CharactersRAGDBError:
            logger.error("Database error tombstoning synced note ID {}.", normalized_note_id, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Note retrieval
    # ------------------------------------------------------------------

    def get_note_by_id(
        self,
        note_id: str,
        include_deleted: bool = False,
        include_studio_summary: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM notes WHERE id = ?"
        params: list[Any] = [note_id]
        if not include_deleted:
            query += " AND deleted = ?"
            params.append(False if self._db.backend_type == BackendType.POSTGRESQL else 0)
        cursor = self._db.execute_query(query, tuple(params))
        row = cursor.fetchone()
        note = dict(row) if row else None
        if note and include_studio_summary:
            studio_document = self._db.get_note_studio_document(note_id)
            if studio_document:
                note["studio"] = self._build_note_studio_summary(studio_document)
        return note

    @staticmethod
    def _serialize_note_studio_json_field(value: dict[str, Any] | None, field_name: str, *, required: bool) -> str | None:
        if value is None:
            if required:
                raise InputError(f"{field_name} cannot be None.")  # noqa: TRY003
            return None
        if not isinstance(value, dict):
            raise InputError(f"{field_name} must be a JSON object.")  # noqa: TRY003
        try:
            return json.dumps(value)
        except TypeError as exc:
            raise InputError(f"{field_name} must be JSON serializable.") from exc  # noqa: TRY003

    @staticmethod
    def _build_note_studio_summary(document: dict[str, Any]) -> dict[str, Any]:
        return {
            "note_id": document["note_id"],
            "template_type": document["template_type"],
            "handwriting_mode": document["handwriting_mode"],
            "source_note_id": document.get("source_note_id"),
            "excerpt_hash": document.get("excerpt_hash"),
            "companion_content_hash": document.get("companion_content_hash"),
            "render_version": document.get("render_version", 1),
        }

    def _fetch_note_studio_document_row(
        self,
        note_id: str,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM note_studio_documents WHERE note_id = ?"
        if conn is None:
            cursor = self._db.execute_query(query, (note_id,))
        else:
            cursor = conn.execute(query, (note_id,))
        row = cursor.fetchone()
        return self._db._deserialize_row_fields(row, ["payload_json", "diagram_manifest_json"]) if row else None

    def get_note_studio_document(self, note_id: str) -> dict[str, Any] | None:
        return self._fetch_note_studio_document_row(note_id)

    def _write_note_studio_document(
        self,
        *,
        note_id: str,
        payload_json: dict[str, Any],
        template_type: str,
        handwriting_mode: str,
        source_note_id: str | None,
        excerpt_snapshot: str | None,
        excerpt_hash: str | None,
        diagram_manifest_json: dict[str, Any] | None,
        companion_content_hash: str | None,
        render_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper | None,
        upsert: bool,
    ) -> dict[str, Any]:
        normalized_note_id = str(note_id).strip()
        if not normalized_note_id:
            raise InputError("note_id cannot be empty.")  # noqa: TRY003
        if template_type not in _SUPPORTED_NOTE_STUDIO_TEMPLATE_TYPES:
            raise InputError(
                f"template_type must be one of {sorted(_SUPPORTED_NOTE_STUDIO_TEMPLATE_TYPES)}."
            )  # noqa: TRY003
        if handwriting_mode not in _SUPPORTED_NOTE_STUDIO_HANDWRITING_MODES:
            raise InputError(
                f"handwriting_mode must be one of {sorted(_SUPPORTED_NOTE_STUDIO_HANDWRITING_MODES)}."
            )  # noqa: TRY003
        if not isinstance(render_version, int) or render_version < 1:
            raise InputError("render_version must be an integer >= 1.")  # noqa: TRY003

        payload_json_str = self._serialize_note_studio_json_field(payload_json, "payload_json", required=True)
        diagram_manifest_json_str = self._serialize_note_studio_json_field(
            diagram_manifest_json,
            "diagram_manifest_json",
            required=False,
        )
        now = self._db._get_current_utc_timestamp_iso()

        if upsert:
            query = (
                "INSERT INTO note_studio_documents ("
                "note_id, payload_json, template_type, handwriting_mode, source_note_id, "
                "excerpt_snapshot, excerpt_hash, diagram_manifest_json, companion_content_hash, "
                "render_version, created_at, last_modified"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(note_id) DO UPDATE SET "
                "payload_json = excluded.payload_json, "
                "template_type = excluded.template_type, "
                "handwriting_mode = excluded.handwriting_mode, "
                "source_note_id = excluded.source_note_id, "
                "excerpt_snapshot = excluded.excerpt_snapshot, "
                "excerpt_hash = excluded.excerpt_hash, "
                "diagram_manifest_json = excluded.diagram_manifest_json, "
                "companion_content_hash = excluded.companion_content_hash, "
                "render_version = excluded.render_version, "
                "last_modified = excluded.last_modified"
            )
        else:
            query = (
                "INSERT INTO note_studio_documents ("
                "note_id, payload_json, template_type, handwriting_mode, source_note_id, "
                "excerpt_snapshot, excerpt_hash, diagram_manifest_json, companion_content_hash, "
                "render_version, created_at, last_modified"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )

        params = (
            normalized_note_id,
            payload_json_str,
            template_type,
            handwriting_mode,
            source_note_id,
            excerpt_snapshot,
            excerpt_hash,
            diagram_manifest_json_str,
            companion_content_hash,
            render_version,
            now,
            now,
        )

        def _execute(inner_conn: sqlite3.Connection | BackendConnectionWrapper) -> dict[str, Any]:
            prepared_query, prepared_params = self._db._prepare_backend_statement(query, params)
            inner_conn.execute(prepared_query, prepared_params or ())
            document = self._fetch_note_studio_document_row(normalized_note_id, conn=inner_conn)
            if not document:
                raise CharactersRAGDBError(f"Failed to read note studio document for note ID '{normalized_note_id}'.")
            return document

        try:
            if conn is None:
                with self._db.transaction() as transaction_conn:
                    return _execute(transaction_conn)
            return _execute(conn)
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "foreign key" in msg:
                raise ConflictError("Note not found.", entity="notes", entity_id=normalized_note_id) from e  # noqa: TRY003
            if "unique constraint" in msg:
                raise ConflictError(  # noqa: TRY003
                    f"Note studio document for note ID '{normalized_note_id}' already exists.",
                    entity="note_studio_documents",
                    entity_id=normalized_note_id,
                ) from e
            raise CharactersRAGDBError(f"Database integrity error writing note studio document: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            msg = str(e).lower()
            if "foreign key" in msg:
                raise ConflictError("Note not found.", entity="notes", entity_id=normalized_note_id) from e  # noqa: TRY003
            if "duplicate key" in msg or "unique constraint" in msg:
                raise ConflictError(  # noqa: TRY003
                    f"Note studio document for note ID '{normalized_note_id}' already exists.",
                    entity="note_studio_documents",
                    entity_id=normalized_note_id,
                ) from e
            raise CharactersRAGDBError(f"Backend error writing note studio document: {e}") from e  # noqa: TRY003

    def create_note_studio_document(
        self,
        *,
        note_id: str,
        payload_json: dict[str, Any],
        template_type: str,
        handwriting_mode: str,
        source_note_id: str | None = None,
        excerpt_snapshot: str | None = None,
        excerpt_hash: str | None = None,
        diagram_manifest_json: dict[str, Any] | None = None,
        companion_content_hash: str | None = None,
        render_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> dict[str, Any]:
        return self._write_note_studio_document(
            note_id=note_id,
            payload_json=payload_json,
            template_type=template_type,
            handwriting_mode=handwriting_mode,
            source_note_id=source_note_id,
            excerpt_snapshot=excerpt_snapshot,
            excerpt_hash=excerpt_hash,
            diagram_manifest_json=diagram_manifest_json,
            companion_content_hash=companion_content_hash,
            render_version=render_version,
            conn=conn,
            upsert=False,
        )

    def upsert_note_studio_document(
        self,
        *,
        note_id: str,
        payload_json: dict[str, Any],
        template_type: str,
        handwriting_mode: str,
        source_note_id: str | None = None,
        excerpt_snapshot: str | None = None,
        excerpt_hash: str | None = None,
        diagram_manifest_json: dict[str, Any] | None = None,
        companion_content_hash: str | None = None,
        render_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> dict[str, Any]:
        return self._write_note_studio_document(
            note_id=note_id,
            payload_json=payload_json,
            template_type=template_type,
            handwriting_mode=handwriting_mode,
            source_note_id=source_note_id,
            excerpt_snapshot=excerpt_snapshot,
            excerpt_hash=excerpt_hash,
            diagram_manifest_json=diagram_manifest_json,
            companion_content_hash=companion_content_hash,
            render_version=render_version,
            conn=conn,
            upsert=True,
        )

    def list_notes(
        self,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
        only_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List notes ordered by most recently modified.

        By default this returns only active notes (``deleted = 0``).
        Set ``only_deleted=True`` to list trash items, or ``include_deleted=True``
        to list both active and deleted notes.
        """
        where_clause = ""
        params: list[Any] = []
        if only_deleted:
            where_clause = " WHERE deleted = ?"
            params.append(True if self._db.backend_type == BackendType.POSTGRESQL else 1)
        elif not include_deleted:
            where_clause = " WHERE deleted = ?"
            params.append(False if self._db.backend_type == BackendType.POSTGRESQL else 0)

        query = (
            f"SELECT * FROM notes{where_clause} "  # nosec B608
            "ORDER BY last_modified DESC "
            "LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        cursor = self._db.execute_query(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def list_deleted_notes(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List only soft-deleted notes (trash)."""
        return self.list_notes(limit=limit, offset=offset, only_deleted=True)

    def get_notes_batch(self, note_ids: list[str], include_deleted: bool = True) -> list[dict[str, Any]]:
        """Return note rows for the given IDs. Batches in groups of 900."""
        if not note_ids:
            return []
        results: list[dict[str, Any]] = []
        for batch in self._db._chunk_list(note_ids, self._db._SQLITE_PARAM_LIMIT):
            ph = ",".join(["?"] * len(batch))
            params: list[Any] = list(batch)
            deleted_clause = ""
            if not include_deleted:
                deleted_clause = " AND deleted = ?"
                params.append(self._deleted_value(False))
            query = (
                f"SELECT id, title, content, created_at, last_modified, deleted, conversation_id "  # nosec B608
                f"FROM notes WHERE id IN ({ph}){deleted_clause}"
            )
            cur = self._db.execute_query(query, tuple(params))
            for row in cur.fetchall():
                r = dict(row) if hasattr(row, "keys") else {
                    "id": row[0], "title": row[1], "content": row[2],
                    "created_at": row[3], "last_modified": row[4],
                    "deleted": row[5], "conversation_id": row[6],
                }
                results.append(r)
        return results

    def get_all_note_ids_for_graph(self, include_deleted: bool = True, limit: int = 500) -> list[str]:
        """Return note IDs ordered by last_modified DESC, id ASC. For seedless graph."""
        params: list[Any] = [limit]
        deleted_clause = ""
        if not include_deleted:
            deleted_clause = " WHERE deleted = ?"
            params.insert(0, self._deleted_value(False))
        query = f"SELECT id FROM notes{deleted_clause} ORDER BY last_modified DESC, id ASC LIMIT ?"  # nosec B608
        cur = self._db.execute_query(query, tuple(params))
        return [row[0] for row in cur.fetchall()]

    def get_note_tag_edges(self, note_ids: list[str]) -> list[dict[str, Any]]:
        """Return (note_id, keyword_id, keyword) for notes with active keywords."""
        if not note_ids:
            return []
        results: list[dict[str, Any]] = []
        for batch in self._db._chunk_list(note_ids, self._db._SQLITE_PARAM_LIMIT):
            ph = ",".join(["?"] * len(batch))
            query = (
                f"SELECT nk.note_id, k.id AS keyword_id, k.keyword "  # nosec B608
                f"FROM note_keywords nk "
                f"JOIN keywords k ON k.id = nk.keyword_id "
                f"WHERE nk.note_id IN ({ph}) AND k.deleted = ?"
            )
            cur = self._db.execute_query(query, tuple([*batch, self._deleted_value(False)]))
            for row in cur.fetchall():
                r = dict(row) if hasattr(row, "keys") else {
                    "note_id": row[0], "keyword_id": row[1], "keyword": row[2],
                }
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Note counting
    # ------------------------------------------------------------------

    def count_notes(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        """Count notes by deletion scope.

        Defaults to active-note count only.
        """
        where_clause = ""
        params: list[Any] = []
        if only_deleted:
            where_clause = " WHERE deleted = ?"
            params.append(True if self._db.backend_type == BackendType.POSTGRESQL else 1)
        elif not include_deleted:
            where_clause = " WHERE deleted = ?"
            params.append(False if self._db.backend_type == BackendType.POSTGRESQL else 0)

        query = f"SELECT COUNT(*) AS cnt FROM notes{where_clause}"  # nosec B608
        try:
            cursor = self._db.execute_query(query, tuple(params) if params else None)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0
        except CharactersRAGDBError as exc:
            logger.error(f"Error counting notes: {exc}")
            raise

    def count_deleted_notes(self) -> int:
        """Return count of soft-deleted notes."""
        return self.count_notes(only_deleted=True)

    def count_user_notes(self, include_deleted: bool = True) -> int:
        """Count total notes for seedless query gate."""
        if include_deleted:
            query = "SELECT COUNT(*) FROM notes"
            params: tuple[Any, ...] | None = None
        else:
            query = "SELECT COUNT(*) FROM notes WHERE deleted = ?"
            params = (self._deleted_value(False),)
        cur = self._db.execute_query(query, params)
        return cur.fetchone()[0]

    def count_notes_per_tag(self) -> dict[int, int]:
        """Return {keyword_id: note_count} for popularity cutoff."""
        query = (
            "SELECT nk.keyword_id, COUNT(DISTINCT nk.note_id) AS cnt "
            "FROM note_keywords nk "
            "JOIN notes n ON n.id = nk.note_id AND n.deleted = ? "
            "JOIN keywords k ON k.id = nk.keyword_id AND k.deleted = ? "
            "GROUP BY nk.keyword_id"
        )
        cur = self._db.execute_query(query, (self._deleted_value(False), self._deleted_value(False)))
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_note_source_info(self, note_ids: list[str]) -> list[dict[str, Any]]:
        """Return source info for notes that have a conversation with source set."""
        if not note_ids:
            return []
        results: list[dict[str, Any]] = []
        for batch in self._db._chunk_list(note_ids, self._db._SQLITE_PARAM_LIMIT):
            ph = ",".join(["?"] * len(batch))
            query = (
                f"SELECT n.id AS note_id, c.id AS conversation_id, c.source, c.external_ref "  # nosec B608
                f"FROM notes n "
                f"JOIN conversations c ON c.id = n.conversation_id "
                f"WHERE n.id IN ({ph}) AND c.source IS NOT NULL"
            )
            cur = self._db.execute_query(query, tuple(batch))
            for row in cur.fetchall():
                r = dict(row) if hasattr(row, "keys") else {
                    "note_id": row[0], "conversation_id": row[1],
                    "source": row[2], "external_ref": row[3],
                }
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Note mutation
    # ------------------------------------------------------------------

    def update_note(
        self,
        note_id: str,
        update_data: dict[str, Any],
        expected_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> bool | None:
        if not update_data:
            raise InputError("No data provided for note update.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []

        allowed_to_update = ['title', 'content', 'conversation_id', 'message_id']
        for key, value in update_data.items():
            if key in allowed_to_update:
                if key == 'title' and isinstance(value, str):
                    fields_to_update_sql.append(f"{key} = ?")
                    params_for_set_clause.append(value.strip())
                elif key in ('conversation_id', 'message_id'):
                    fields_to_update_sql.append(f"{key} = ?")
                    params_for_set_clause.append(self._db._normalize_nullable_text(value))
                else:
                    fields_to_update_sql.append(f"{key} = ?")
                    params_for_set_clause.append(value)
            elif key not in ['id', 'created_at', 'last_modified', 'version', 'client_id', 'deleted']:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in note ID {note_id}, skipping.")

        if not fields_to_update_sql:
            logger.info(f"No updatable fields provided for note ID {note_id}.")
            return True

        next_version_val = expected_version + 1
        fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])

        all_set_values = params_for_set_clause[:]
        all_set_values.extend([now, next_version_val, self._db.client_id])

        where_values = [note_id, expected_version]
        final_params_for_execute = tuple(all_set_values + where_values)

        query = f"UPDATE notes SET {', '.join(fields_to_update_sql)} WHERE id = ? AND version = ? AND deleted = 0"  # nosec B608

        try:
            def _execute(transaction_conn: sqlite3.Connection | BackendConnectionWrapper) -> bool:
                current_db_version = self._db._get_current_db_version(transaction_conn, "notes", "id", note_id)

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Note ID {note_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="notes", entity_id=note_id
                    )

                cursor = transaction_conn.execute(query, final_params_for_execute)

                if cursor.rowcount == 0:
                    check_again_cursor = transaction_conn.execute("SELECT version, deleted FROM notes WHERE id = ?", (note_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Note ID {note_id} disappeared."
                    elif final_state['deleted']:
                        msg = f"Note ID {note_id} was soft-deleted concurrently."
                    elif final_state['version'] != expected_version:
                        msg = f"Note ID {note_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Update for note ID {note_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="notes", entity_id=note_id)  # noqa: TRY301

                logger.info(f"Updated note ID {note_id} from version {expected_version} to version {next_version_val}.")
                return True

            if conn is None:
                with self._db.transaction() as transaction_conn:
                    return _execute(transaction_conn)
            return _execute(conn)
        # No specific UNIQUE constraint on notes.title or notes.content in the schema, so sqlite3.IntegrityError less likely for these fields.
        except ConflictError:
            raise
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "foreign key constraint failed" in msg:
                raise ConflictError("Conversation or message not found.", entity="notes", entity_id=note_id) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Database integrity error updating note: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            msg = str(e).lower()
            if "foreign key" in msg:
                raise ConflictError("Conversation or message not found.", entity="notes", entity_id=note_id) from e  # noqa: TRY003
            raise CharactersRAGDBError(f"Backend error updating note: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError as e:  # Catches sqlite3.Error
            logger.error(f"Database error updating note ID {note_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Note deletion and restoration
    # ------------------------------------------------------------------

    def soft_delete_note(self, note_id: str, expected_version: int) -> bool | None:
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE notes SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
        params = (now, next_version_val, self._db.client_id, note_id, expected_version)

        try:
            with self._db.transaction() as conn:
                try:
                    current_db_version = self._db._get_current_db_version(conn, "notes", "id", note_id)
                except ConflictError:
                    check_status_cursor = conn.execute("SELECT deleted, version FROM notes WHERE id = ?", (note_id,))
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(f"Note ID {note_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Soft delete for Note ID {note_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="notes", entity_id=note_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = conn.execute("SELECT version, deleted FROM notes WHERE id = ?", (note_id,))
                    final_state = check_again_cursor.fetchone()
                    if not final_state:
                        msg = f"Note ID {note_id} disappeared."
                    elif final_state['deleted']:
                        logger.info(f"Note ID {note_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Note ID {note_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Soft delete for note ID {note_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="notes", entity_id=note_id)  # noqa: TRY301

                self._db._invalidate_note_clipper_sidecars(note_id, conn=conn, deleted=True)
                logger.info(
                    f"Soft-deleted note ID {note_id} (was v{expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error soft-deleting note ID {note_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    def delete_note(self, note_id: str, expected_version: int | None = None, hard_delete: bool = False) -> bool:
        """Soft or hard delete a note."""
        now = self._db._get_current_utc_timestamp_iso()
        try:
            with self._db.transaction() as conn:
                row = conn.execute("SELECT id, version, deleted FROM notes WHERE id = ?", (note_id,)).fetchone()
                if not row:
                    return False
                cur_ver = int(row["version"])
                deleted = bool(row["deleted"])
                if hard_delete:
                    self._db._delete_note_clipper_sidecars(note_id, conn=conn)
                    conn.execute("DELETE FROM note_studio_documents WHERE note_id = ?", (note_id,))
                    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                    return True
                if deleted:
                    return True
                if expected_version is not None and cur_ver != expected_version:
                    raise ConflictError("Version mismatch deleting note", entity="notes", identifier=note_id)  # noqa: TRY003
                deleted_val = True if self._db.backend_type == BackendType.POSTGRESQL else 1
                rc = conn.execute(
                    "UPDATE notes SET deleted = ?, last_modified = ?, version = ?, client_id = ? "
                    "WHERE id = ? AND deleted = 0",
                    (deleted_val, now, cur_ver + 1, self._db.client_id, note_id),
                ).rowcount
                if rc > 0:
                    self._db._invalidate_note_clipper_sidecars(note_id, conn=conn, deleted=True)
                return rc > 0
        except BackendDatabaseError as e:
            raise CharactersRAGDBError(f"Failed to delete note: {e}") from e  # noqa: TRY003
        except sqlite3.Error as e:
            raise CharactersRAGDBError(f"Failed to delete note: {e}") from e  # noqa: TRY003

    def restore_note(self, note_id: str, expected_version: int) -> bool | None:
        """
        Restores a soft-deleted note using optimistic locking.

        Sets the ``deleted`` flag to 0, updates ``last_modified``, increments ``version``,
        and sets ``client_id``. The operation succeeds only if ``expected_version`` matches
        the current database version and the note is currently deleted.

        If the note is already active (not deleted), the method considers
        this a success and returns True (idempotency).

        Args:
            note_id: The ID of the note to restore.
            expected_version: The version number the client expects the record to have.

        Returns:
            True if the restore was successful or if the note was already active.

        Raises:
            ConflictError: If the note is not found, or if ``expected_version`` does
                           not match, or if a concurrent modification prevents the update.
            CharactersRAGDBError: For other database-related errors.
        """
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE notes SET deleted = 0, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 1"
        params = (now, next_version_val, self._db.client_id, note_id, expected_version)

        try:
            with self._db.transaction() as conn:
                # First check if record exists at all
                check_cursor = conn.execute("SELECT deleted, version FROM notes WHERE id = ?", (note_id,))
                record_status = check_cursor.fetchone()

                if not record_status:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Note ID {note_id} not found.",
                        entity="notes", entity_id=note_id
                    )

                # If already active, return success (idempotent)
                if not record_status['deleted']:
                    logger.info(f"Note ID {note_id} already active. Restore successful (idempotent).")
                    return True

                # Check version matches
                current_db_version = record_status['version']
                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Restore for Note ID {note_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="notes", entity_id=note_id
                    )

                cursor = conn.execute(query, params)

                if cursor.rowcount == 0:
                    # Race condition: Record changed between pre-check and UPDATE.
                    check_again_cursor = conn.execute("SELECT version, deleted FROM notes WHERE id = ?", (note_id,))
                    final_state = check_again_cursor.fetchone()
                    msg = f"Restore for Note ID {note_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Note ID {note_id} disappeared before restore (expected deleted version {expected_version})."
                    elif not final_state['deleted']:
                        # If it got restored by another process. Consider this success.
                        logger.info(
                            f"Note ID {note_id} was restored concurrently to version {final_state['version']}. Restore successful.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Restore for Note ID {note_id} failed: version changed to {final_state['version']} concurrently (expected {expected_version})."
                    else:
                        msg = f"Restore for Note ID {note_id} (expected version {expected_version}) affected 0 rows for an unknown reason after passing initial checks."
                    raise ConflictError(msg, entity="notes", entity_id=note_id)  # noqa: TRY301

                self._db._invalidate_note_clipper_sidecars(note_id, conn=conn, deleted=False)
                logger.info(
                    f"Restored note ID {note_id} (was version {expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except BackendDatabaseError as e:
            logger.error(
                'Backend error restoring note ID {} (expected v{}): {}',
                note_id,
                expected_version,
                e,
            )
            raise CharactersRAGDBError(f"Backend error during restore: {e}") from e  # noqa: TRY003
        except CharactersRAGDBError as e:
            logger.error(
                f"Database error restoring note ID {note_id} (expected v{expected_version}): {e}",
                exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Note search
    # ------------------------------------------------------------------

    def search_notes(self, search_term: str, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        """Searches notes_fts (title and content) with optional pagination."""
        # Debug: Log FTS table state to help diagnose E2E test failures
        # Only run for SQLite; PostgreSQL uses tsvector columns, not a notes_fts table.
        if self._db.backend_type == BackendType.SQLITE:
            try:
                fts_count = self._db.execute_query("SELECT COUNT(*) as cnt FROM notes_fts").fetchone()
                notes_count = self._db.execute_query("SELECT COUNT(*) as cnt FROM notes WHERE deleted = 0").fetchone()
                logger.debug(
                    f"search_notes: term='{search_term[:50] if search_term else ''}' notes_fts_count={fts_count['cnt'] if fts_count else 0} "
                    f"notes_count={notes_count['cnt'] if notes_count else 0}"
                )
            except _CHACHA_NONCRITICAL_EXCEPTIONS as diag_err:
                logger.debug(f"search_notes diagnostic query failed: {diag_err}")

        if self._db.backend_type == BackendType.POSTGRESQL:
            if not search_term or not str(search_term).strip():
                logger.debug("Empty notes search term; returning no results.")
                return []
            tsquery = FTSQueryTranslator.normalize_query(search_term, 'postgresql')
            fallback_query = """
                SELECT n.*
                FROM notes n
                WHERE n.deleted = FALSE
                  AND (n.title ILIKE ? OR n.content ILIKE ?)
                ORDER BY n.last_modified DESC
                LIMIT ? OFFSET ?
            """
            fallback_params = (f"%{search_term}%", f"%{search_term}%", limit, offset)
            if not tsquery:
                logger.debug("Notes search term normalized to empty tsquery for input '{}'", search_term)
                cursor = self._db.execute_query(fallback_query, fallback_params)
                return [dict(row) for row in cursor.fetchall()]

            query = """
                SELECT n.*, ts_rank(n.notes_fts_tsv, to_tsquery('english', ?)) AS rank
                FROM notes n
                WHERE n.deleted = FALSE
                  AND n.notes_fts_tsv @@ to_tsquery('english', ?)
                ORDER BY rank DESC, n.last_modified DESC
                LIMIT ? OFFSET ?
            """
            try:
                cursor = self._db.execute_query(query, (tsquery, tsquery, limit, offset))
                rows = cursor.fetchall()
                if rows:
                    return [dict(row) for row in rows]
                # Fallback: if FTS vectors are missing/stale, use ILIKE search.
                cursor = self._db.execute_query(fallback_query, fallback_params)
                return [dict(row) for row in cursor.fetchall()]
            except CharactersRAGDBError as exc:
                logger.error("PostgreSQL FTS search failed for notes term '{}': {}", search_term, exc)
                raise

        safe_literal = search_term.replace('"', '""')
        safe_search_term = f'"{safe_literal}"'

        query = """
                SELECT main.*, bm25(notes_fts) AS bm25_score
                FROM notes_fts
                JOIN notes AS main ON notes_fts.rowid = main.rowid
                WHERE notes_fts MATCH ?
                  AND main.deleted = 0
                ORDER BY bm25_score, main.last_modified DESC
                LIMIT ? OFFSET ?
                """
        try:
            cursor = self._db.execute_query(query, (safe_search_term, limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        except CharactersRAGDBError as e:
            logger.error(f"Error searching notes for '{search_term}': {e}")
            raise

    def search_notes_with_keywords(
        self,
        search_term: str | None,
        keyword_tokens: list[str],
        limit: int = 10,
        offset: int = 0
    ) -> list[dict[str, Any]]:
        """Search notes with an optional FTS query and keyword-token filter."""
        tokens = [t.strip().lower() for t in keyword_tokens if isinstance(t, str) and t.strip()]
        if not tokens:
            if not search_term or not str(search_term).strip():
                return []
            return self.search_notes(search_term=str(search_term), limit=limit, offset=offset)

        keyword_table = self._db._map_table_for_backend("keywords")
        like_clause = " OR ".join(["LOWER(k.keyword) LIKE ?"] * len(tokens))
        like_params = [f"%{t}%" for t in tokens]

        if self._db.backend_type == BackendType.POSTGRESQL:
            if search_term and str(search_term).strip():
                tsquery = FTSQueryTranslator.normalize_query(str(search_term), 'postgresql')
                if not tsquery:
                    logger.debug("Notes search term normalized to empty tsquery for input '{}'", search_term)
                    return []
                query = """
                    SELECT DISTINCT n.*, ts_rank(n.notes_fts_tsv, to_tsquery('english', ?)) AS rank
                    FROM notes n
                    JOIN note_keywords nk ON n.id = nk.note_id
                    JOIN {keyword_table} k ON k.id = nk.keyword_id
                    WHERE n.deleted = FALSE
                      AND k.deleted = FALSE
                      AND n.notes_fts_tsv @@ to_tsquery('english', ?)
                      AND ({like_clause})
                    ORDER BY rank DESC, n.last_modified DESC
                    LIMIT ? OFFSET ?
                """.format_map(locals())  # nosec B608
                params = (tsquery, tsquery, *like_params, limit, offset)
            else:
                query = """
                    SELECT DISTINCT n.*
                    FROM notes n
                    JOIN note_keywords nk ON n.id = nk.note_id
                    JOIN {keyword_table} k ON k.id = nk.keyword_id
                    WHERE n.deleted = FALSE
                      AND k.deleted = FALSE
                      AND ({like_clause})
                    ORDER BY n.last_modified DESC
                    LIMIT ? OFFSET ?
                """.format_map(locals())  # nosec B608
                params = (*like_params, limit, offset)
            cursor = self._db.execute_query(query, params)
            return [dict(row) for row in cursor.fetchall()]

        if search_term and str(search_term).strip():
            safe_literal = str(search_term).replace('"', '""')
            safe_search_term = f'"{safe_literal}"'
            query = """
                    SELECT DISTINCT n.*
                    FROM notes_fts
                    JOIN notes AS n ON notes_fts.rowid = n.rowid
                    JOIN note_keywords nk ON n.id = nk.note_id
                    JOIN {keyword_table} k ON k.id = nk.keyword_id
                    WHERE notes_fts MATCH ?
                      AND n.deleted = 0
                      AND k.deleted = 0
                      AND ({like_clause})
                    ORDER BY bm25(notes_fts), n.last_modified DESC
                    LIMIT ? OFFSET ?
                    """.format_map(locals())  # nosec B608
            params = (safe_search_term, *like_params, limit, offset)
        else:
            query = """
                    SELECT DISTINCT n.*
                    FROM notes n
                    JOIN note_keywords nk ON n.id = nk.note_id
                    JOIN {keyword_table} k ON k.id = nk.keyword_id
                    WHERE n.deleted = 0
                      AND k.deleted = 0
                      AND ({like_clause})
                    ORDER BY n.last_modified DESC
                    LIMIT ? OFFSET ?
                    """.format_map(locals())  # nosec B608
            params = (*like_params, limit, offset)

        cursor = self._db.execute_query(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Note <-> Keyword links
    # ------------------------------------------------------------------

    def link_note_to_keyword(self, note_id: str, keyword_id: int) -> bool:  # note_id is str
        return self._db._manage_link("note_keywords", "note_id", note_id, "keyword_id", keyword_id, "link")

    def unlink_note_from_keyword(self, note_id: str, keyword_id: int) -> bool:  # note_id is str
        return self._db._manage_link("note_keywords", "note_id", note_id, "keyword_id", keyword_id, "unlink")

    def unlink_note_to_keyword(self, note_id: str, keyword_id: int) -> bool:  # pragma: no cover - compat alias
        """Backward-compatible alias for the extracted facade delegation typo."""
        return self.unlink_note_from_keyword(note_id, keyword_id)

    def get_keywords_for_note(self, note_id: str) -> list[dict[str, Any]]:  # note_id is str
        keyword_table = self._db._map_table_for_backend("keywords")
        order_clause = self._db._case_insensitive_order_clause("k.keyword")
        query = """
                SELECT k.* \
                FROM {keyword_table} k \
                         JOIN note_keywords nk ON k.id = nk.keyword_id
                WHERE nk.note_id = ? \
                  AND k.deleted = 0 \
                {order_clause}
                """.format_map(locals())  # nosec B608
        cursor = self._db.execute_query(query, (note_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_keywords_for_notes(self, note_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return keywords for multiple notes as a map of note_id -> keywords list."""
        if not note_ids:
            return {}
        keyword_table = self._db._map_table_for_backend("keywords")
        order_clause = self._db._case_insensitive_order_clause("k.keyword")
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
                      AND k.deleted = 0 \
                    {order_clause}
                    """.format_map(locals())  # nosec B608
            cursor = self._db.execute_query(query, tuple(batch))
            rows = cursor.fetchall()
            for row in rows:
                record = dict(row)
                note_id = record.pop("note_id", None)
                if not note_id:
                    continue
                out.setdefault(note_id, []).append(record)
        return out

    def get_notes_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        query = """
                SELECT n.* \
                FROM notes n \
                         JOIN note_keywords nk ON n.id = nk.note_id
                WHERE nk.keyword_id = ? \
                  AND n.deleted = 0
                ORDER BY n.last_modified DESC LIMIT ? \
                OFFSET ? \
                """
        cursor = self._db.execute_query(query, (keyword_id, limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    def get_note_counts_for_keywords(self, keyword_ids: list[int] | None = None) -> dict[int, int]:
        """Return active-note usage counts keyed by keyword ID."""
        normalized_ids: list[int] = []
        if keyword_ids:
            seen = set()
            for raw in keyword_ids:
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                if value <= 0 or value in seen:
                    continue
                seen.add(value)
                normalized_ids.append(value)

        keyword_table = self._db._map_table_for_backend("keywords")
        notes_table = self._db._map_table_for_backend("notes")
        keyword_filter = ""
        params: list[Any] = []
        if normalized_ids:
            placeholders = ", ".join(["?"] * len(normalized_ids))
            keyword_filter = f" AND nk.keyword_id IN ({placeholders})"
            params.extend(normalized_ids)

        deleted_note_value = "FALSE" if self._db.backend_type == BackendType.POSTGRESQL else "0"
        deleted_keyword_value = "FALSE" if self._db.backend_type == BackendType.POSTGRESQL else "0"

        query = """
            SELECT nk.keyword_id AS keyword_id, COUNT(DISTINCT nk.note_id) AS note_count
            FROM note_keywords nk
            JOIN {notes_table} n ON n.id = nk.note_id
            JOIN {keyword_table} k ON k.id = nk.keyword_id
            WHERE n.deleted = {deleted_note_value}
              AND k.deleted = {deleted_keyword_value}
              {keyword_filter}
            GROUP BY nk.keyword_id
        """.format_map(locals())  # nosec B608

        cursor = self._db.execute_query(query, tuple(params) if params else None)
        out: dict[int, int] = {}
        for row in cursor.fetchall():
            try:
                keyword_id_val = int(row["keyword_id"])
                out[keyword_id_val] = int(row["note_count"] or 0)
            except (TypeError, ValueError):
                continue
        return out
