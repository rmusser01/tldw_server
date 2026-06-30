"""Repository for document workspace storage."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

_UNSET = object()


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def _normalize_reference_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("raw_text", ""))
        else:
            text = str(item)
        cleaned = text.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


class DocumentWorkspaceRepository:
    """Storage access for reading progress, annotations, and parsed reference cache."""

    def __init__(self, db: Any) -> None:
        self.db = db

    @classmethod
    def from_media_db(cls, db: Any) -> "DocumentWorkspaceRepository":
        """Create a document workspace repository for a Media DB session."""
        return cls(db)

    def _is_postgres(self) -> bool:
        return getattr(self.db, "backend_type", None) == BackendType.POSTGRESQL

    def get_reading_progress(self, *, media_id: int, user_id: str) -> dict[str, Any] | None:
        """Return one reading-progress row for a media item and user."""
        query = """
        SELECT media_id, user_id, current_page, total_pages, zoom_level, view_mode, cfi, percentage, last_read_at
        FROM document_reading_progress
        WHERE media_id = ? AND user_id = ?
        """
        with self.db.transaction() as conn:
            return self.db._fetchone_with_connection(conn, query, (media_id, user_id))

    def upsert_reading_progress(
        self,
        *,
        media_id: int,
        user_id: str,
        current_page: int,
        total_pages: int,
        zoom_level: int,
        view_mode: str,
        cfi: str | None,
        percentage: float | None,
        last_read_at: str,
    ) -> dict[str, Any]:
        """Insert or replace reading progress and return the saved row."""
        if self._is_postgres():
            upsert_sql = """
            INSERT INTO document_reading_progress
            (media_id, user_id, current_page, total_pages, zoom_level, view_mode, cfi, percentage, last_read_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (media_id, user_id) DO UPDATE SET
                current_page = EXCLUDED.current_page,
                total_pages = EXCLUDED.total_pages,
                zoom_level = EXCLUDED.zoom_level,
                view_mode = EXCLUDED.view_mode,
                cfi = EXCLUDED.cfi,
                percentage = EXCLUDED.percentage,
                last_read_at = EXCLUDED.last_read_at
            """
        else:
            upsert_sql = """
            INSERT OR REPLACE INTO document_reading_progress
            (media_id, user_id, current_page, total_pages, zoom_level, view_mode, cfi, percentage, last_read_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        with self.db.transaction() as conn:
            self.db._execute_with_connection(
                conn,
                upsert_sql,
                (
                    media_id,
                    user_id,
                    current_page,
                    total_pages,
                    zoom_level,
                    view_mode,
                    cfi,
                    percentage,
                    last_read_at,
                ),
            )
        saved = self.get_reading_progress(media_id=media_id, user_id=user_id)
        if saved is None:
            raise RuntimeError("Reading progress upsert did not produce a row")
        return saved

    def delete_reading_progress(self, *, media_id: int, user_id: str) -> bool:
        """Delete reading progress for a media item and user."""
        delete_sql = """
        DELETE FROM document_reading_progress
        WHERE media_id = ? AND user_id = ?
        """
        with self.db.transaction() as conn:
            cursor = self.db._execute_with_connection(conn, delete_sql, (media_id, user_id))
            return cursor.rowcount > 0

    def list_annotations(self, *, media_id: int, user_id: str) -> list[dict[str, Any]]:
        """Return active annotations for a media item and user."""
        query = """
        SELECT id, location, text, color, note, annotation_type, chapter_title, percentage, created_at, updated_at
        FROM document_annotations
        WHERE media_id = ? AND user_id = ? AND deleted = 0
        ORDER BY created_at DESC, id ASC
        """
        with self.db.transaction() as conn:
            return self.db._fetchall_with_connection(conn, query, (media_id, user_id))

    def create_annotation(
        self,
        *,
        annotation_id: str,
        media_id: int,
        user_id: str,
        location: str,
        text: str,
        color: str,
        note: str | None,
        annotation_type: str,
        chapter_title: str | None,
        percentage: float | None,
        created_at: str,
        updated_at: str,
    ) -> dict[str, Any]:
        """Insert an annotation row and return it."""
        insert_sql = """
        INSERT INTO document_annotations
        (id, media_id, user_id, location, text, color, note, annotation_type, chapter_title, percentage, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            self.db._execute_with_connection(
                conn,
                insert_sql,
                (
                    annotation_id,
                    media_id,
                    user_id,
                    location,
                    text,
                    color,
                    note,
                    annotation_type,
                    chapter_title,
                    percentage,
                    created_at,
                    updated_at,
                ),
            )
        row = self.get_annotation(annotation_id=annotation_id, media_id=media_id, user_id=user_id)
        if row is None:
            raise RuntimeError("Annotation insert did not produce a row")
        return row

    def get_annotation(self, *, annotation_id: str, media_id: int, user_id: str) -> dict[str, Any] | None:
        """Return one active annotation row."""
        query = """
        SELECT id, location, text, color, note, annotation_type, chapter_title, percentage, created_at, updated_at
        FROM document_annotations
        WHERE id = ? AND media_id = ? AND user_id = ? AND deleted = 0
        """
        with self.db.transaction() as conn:
            return self.db._fetchone_with_connection(conn, query, (annotation_id, media_id, user_id))

    def update_annotation(
        self,
        *,
        annotation_id: str,
        media_id: int,
        user_id: str,
        text: Any = _UNSET,
        color: Any = _UNSET,
        note: Any = _UNSET,
        updated_at: str,
    ) -> dict[str, Any] | None:
        """Update mutable annotation fields and return the active row."""
        existing = self.get_annotation(annotation_id=annotation_id, media_id=media_id, user_id=user_id)
        if existing is None:
            return None

        updates: list[str] = []
        params: list[Any] = []
        if text is not _UNSET:
            updates.append("text = ?")
            params.append(text)
        if color is not _UNSET:
            updates.append("color = ?")
            params.append(color)
        if note is not _UNSET:
            updates.append("note = ?")
            params.append(note)

        if not updates:
            return existing

        updates.append("updated_at = ?")
        params.append(updated_at)
        params.extend([annotation_id, media_id, user_id])
        update_sql = f"""
        UPDATE document_annotations
        SET {", ".join(updates)}
        WHERE id = ? AND media_id = ? AND user_id = ? AND deleted = 0
        """  # nosec B608
        with self.db.transaction() as conn:
            self.db._execute_with_connection(conn, update_sql, tuple(params))
        return self.get_annotation(annotation_id=annotation_id, media_id=media_id, user_id=user_id)

    def _list_annotations_by_ids(
        self,
        *,
        media_id: int,
        user_id: str,
        annotation_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Return active annotation rows in the caller-provided id order."""
        if not annotation_ids:
            return []

        placeholders = ", ".join("?" for _ in annotation_ids)
        # Dynamic placeholder count is generated here; values remain bound parameters.
        query = f"""
        SELECT id, location, text, color, note, annotation_type, chapter_title, percentage, created_at, updated_at
        FROM document_annotations
        WHERE media_id = ? AND user_id = ? AND deleted = 0 AND id IN ({placeholders})
        """  # nosec B608
        with self.db.transaction() as conn:
            rows = self.db._fetchall_with_connection(
                conn,
                query,
                tuple([media_id, user_id, *annotation_ids]),
            )
        rows_by_id = {normalized["id"]: normalized for normalized in (_row_to_dict(row) for row in rows)}
        return [
            rows_by_id[annotation_id]
            for annotation_id in annotation_ids
            if annotation_id in rows_by_id
        ]

    def sync_annotations(
        self,
        *,
        media_id: int,
        user_id: str,
        annotation_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert a batch of annotation rows and return the saved rows."""
        insert_sql = """
        INSERT INTO document_annotations
        (id, media_id, user_id, location, text, color, note, annotation_type, chapter_title, percentage, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            for row in annotation_rows:
                self.db._execute_with_connection(
                    conn,
                    insert_sql,
                    (
                        row["id"],
                        media_id,
                        user_id,
                        row["location"],
                        row["text"],
                        row["color"],
                        row.get("note"),
                        row["annotation_type"],
                        row.get("chapter_title"),
                        row.get("percentage"),
                        row["created_at"],
                        row["updated_at"],
                    ),
                )
        return self._list_annotations_by_ids(
            media_id=media_id,
            user_id=user_id,
            annotation_ids=[str(item["id"]) for item in annotation_rows],
        )

    def soft_delete_annotation(
        self,
        *,
        annotation_id: str,
        media_id: int,
        user_id: str,
        updated_at: str,
    ) -> bool:
        """Mark an annotation deleted."""
        delete_sql = """
        UPDATE document_annotations
        SET deleted = 1, updated_at = ?
        WHERE id = ? AND media_id = ? AND user_id = ? AND deleted = 0
        """
        with self.db.transaction() as conn:
            cursor = self.db._execute_with_connection(
                conn,
                delete_sql,
                (updated_at, annotation_id, media_id, user_id),
            )
            return cursor.rowcount > 0

    def get_parsed_references_cache(
        self,
        *,
        media_id: int,
        user_id: str,
        parser_version: str,
        content_hash: str,
    ) -> tuple[list[str], int] | None:
        """Return parsed-reference cache payload for a content hash."""
        query = """
        SELECT references_json, total_detected
        FROM document_parsed_references_cache
        WHERE media_id = ? AND user_id = ? AND parser_version = ? AND content_hash = ?
        LIMIT 1
        """
        with self.db.transaction() as conn:
            row = self.db._fetchone_with_connection(
                conn,
                query,
                (media_id, user_id, parser_version, content_hash),
            )
        if not row:
            return None
        payload = row.get("references_json")
        if not isinstance(payload, str) or not payload:
            return None
        parsed_refs = _normalize_reference_list(json.loads(payload))
        total_detected = int(row.get("total_detected") or len(parsed_refs))
        if not parsed_refs and total_detected <= 0:
            return None
        return parsed_refs, max(total_detected, len(parsed_refs))

    def upsert_parsed_references_cache(
        self,
        *,
        media_id: int,
        user_id: str,
        parser_version: str,
        content_hash: str,
        references: list[str],
        total_detected: int,
        updated_at: str,
    ) -> None:
        """Replace parsed-reference cache rows for the parser version."""
        delete_sql = """
        DELETE FROM document_parsed_references_cache
        WHERE media_id = ? AND user_id = ? AND parser_version = ?
        """
        insert_sql = """
        INSERT INTO document_parsed_references_cache
        (media_id, user_id, parser_version, content_hash, references_json, total_detected, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        references_json = json.dumps(references, ensure_ascii=False)
        with self.db.transaction() as conn:
            self.db._execute_with_connection(
                conn,
                delete_sql,
                (media_id, user_id, parser_version),
            )
            self.db._execute_with_connection(
                conn,
                insert_sql,
                (
                    media_id,
                    user_id,
                    parser_version,
                    content_hash,
                    references_json,
                    int(total_detected),
                    updated_at,
                ),
            )


__all__ = ["DocumentWorkspaceRepository"]
