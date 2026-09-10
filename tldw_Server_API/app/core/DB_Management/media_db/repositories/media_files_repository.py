from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import uuid
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike


class MediaFilesRepository:
    """Repository for MediaFiles rows."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> "MediaFilesRepository":
        return cls(session=db)

    def insert(
        self,
        media_id: int,
        file_type: str,
        storage_path: str,
        *,
        original_filename: str | None = None,
        file_size: int | None = None,
        mime_type: str | None = None,
        checksum: str | None = None,
    ) -> str:
        db = self.session
        new_uuid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        data: dict[str, Any] = {
            "media_id": media_id,
            "file_type": file_type,
            "storage_path": storage_path,
            "original_filename": original_filename,
            "file_size": file_size,
            "mime_type": mime_type,
            "checksum": checksum,
            "uuid": new_uuid,
            "created_at": now,
            "last_modified": now,
            "version": 1,
            "client_id": db.client_id,
            "deleted": 0,
            "prev_version": None,
            "merge_parent_uuid": None,
        }
        placeholders = ", ".join([f":{key}" for key in data])
        columns = ", ".join(data.keys())
        sql = f"INSERT INTO MediaFiles ({columns}) VALUES ({placeholders})"  # nosec B608
        try:
            with db.transaction() as conn:
                db._execute_with_connection(conn, sql, data)
                with suppress(Exception):
                    db._log_sync_event(
                        conn,
                        "MediaFiles",
                        new_uuid,
                        "create",
                        1,
                        {
                            "media_id": media_id,
                            "file_type": file_type,
                            "storage_path": storage_path,
                        },
                    )
        except Exception as exc:
            raise DatabaseError(f"Failed to insert MediaFile: {exc}") from exc  # noqa: TRY003
        return new_uuid

    def get_for_media(
        self,
        media_id: int,
        file_type: str = "original",
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        db = self.session
        conn = db.get_connection()
        clauses: list[str] = ["media_id = :media_id", "file_type = :file_type"]
        params: dict[str, Any] = {"media_id": media_id, "file_type": file_type}
        if not include_deleted:
            clauses.append("deleted = 0")
        where_sql = " AND ".join(clauses)
        # Reuploads keep distinct blobs; serve the latest successfully registered file.
        sql = f"SELECT * FROM MediaFiles WHERE {where_sql} ORDER BY id DESC LIMIT 1"  # nosec B608
        try:
            rows = db._fetchall_with_connection(conn, sql, params)
            return rows[0] if rows else None
        except Exception as exc:
            raise DatabaseError(f"Failed to get MediaFile for media_id={media_id}: {exc}") from exc  # noqa: TRY003

    def list_for_media(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        db = self.session
        conn = db.get_connection()
        clauses: list[str] = ["media_id = :media_id"]
        params: dict[str, Any] = {"media_id": media_id}
        if not include_deleted:
            clauses.append("deleted = 0")
        where_sql = " AND ".join(clauses)
        sql = f"SELECT * FROM MediaFiles WHERE {where_sql} ORDER BY file_type, id"  # nosec B608
        try:
            return db._fetchall_with_connection(conn, sql, params)
        except Exception as exc:
            raise DatabaseError(f"Failed to list MediaFiles for media_id={media_id}: {exc}") from exc  # noqa: TRY003

    def has_original_file(self, media_id: int) -> bool:
        return self.get_for_media(media_id, "original", include_deleted=False) is not None

    def soft_delete(self, file_id: int) -> None:
        db = self.session
        try:
            with db.transaction() as conn:
                rows = db._fetchall_with_connection(
                    conn,
                    "SELECT uuid, version FROM MediaFiles WHERE id = :id",
                    {"id": file_id},
                )
                if not rows:
                    return
                row = rows[0]
                file_uuid = row.get("uuid")
                current_version = int(row.get("version") or 1)
                new_version = current_version + 1
                now = datetime.now(timezone.utc).isoformat()

                db._execute_with_connection(
                    conn,
                    "UPDATE MediaFiles SET deleted = 1, version = :version, last_modified = :last_modified WHERE id = :id",
                    {"id": file_id, "version": new_version, "last_modified": now},
                )
                with suppress(Exception):
                    db._log_sync_event(
                        conn,
                        "MediaFiles",
                        file_uuid,
                        "delete",
                        new_version,
                        {"file_id": file_id},
                    )
        except Exception as exc:
            raise DatabaseError(f"Failed to soft-delete MediaFile id={file_id}: {exc}") from exc  # noqa: TRY003

    def soft_delete_for_media(
        self,
        media_id: int,
        *,
        hard_delete: bool = False,
    ) -> None:
        db = self.session
        try:
            with db.transaction() as conn:
                rows = db._fetchall_with_connection(
                    conn,
                    "SELECT id, uuid, version, deleted FROM MediaFiles WHERE media_id = :media_id",
                    {"media_id": media_id},
                )
                if hard_delete:
                    if not rows:
                        return
                    db._execute_with_connection(
                        conn,
                        "DELETE FROM MediaFiles WHERE media_id = :media_id",
                        {"media_id": media_id},
                    )
                    for row in rows:
                        file_uuid = row.get("uuid")
                        current_version = int(row.get("version") or 1)
                        with suppress(Exception):
                            db._log_sync_event(
                                conn,
                                "MediaFiles",
                                file_uuid,
                                "delete",
                                current_version + 1,
                                {"media_id": media_id, "hard_delete": True},
                            )
                    return

                now = datetime.now(timezone.utc).isoformat()
                for row in rows:
                    if int(row.get("deleted") or 0) == 1:
                        continue
                    file_uuid = row.get("uuid")
                    current_version = int(row.get("version") or 1)
                    new_version = current_version + 1
                    db._execute_with_connection(
                        conn,
                        "UPDATE MediaFiles SET deleted = 1, version = :version, last_modified = :last_modified WHERE uuid = :uuid",
                        {"uuid": file_uuid, "version": new_version, "last_modified": now},
                    )
                    with suppress(Exception):
                        db._log_sync_event(
                            conn,
                            "MediaFiles",
                            file_uuid,
                            "delete",
                            new_version,
                            {"media_id": media_id},
                        )
        except Exception as exc:
            raise DatabaseError(f"Failed to delete MediaFiles for media_id={media_id}: {exc}") from exc  # noqa: TRY003
