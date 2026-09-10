"""Package-owned media lifecycle helpers for the Media DB runtime."""

from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def soft_delete_media(self: Any, media_id: int, cascade: bool = True) -> bool:
    """Soft delete a media row and optionally cascade to linked child rows."""
    current_time = self._get_current_utc_timestamp_str()
    client_id = self.client_id
    logger.info(
        f"Attempting soft delete for Media ID: {media_id} [Client: {client_id}, Cascade: {cascade}]"
    )

    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0 "
                "AND system_operation_id IS NULL",
                (media_id,),
            )
            if not media_info:
                logger.warning(
                    f"Cannot soft delete: Media ID {media_id} not found or already deleted."
                )
                return False

            media_uuid, current_media_version = media_info["uuid"], media_info["version"]
            new_media_version = current_media_version + 1

            update_cursor = self._execute_with_connection(
                conn,
                "UPDATE Media SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND system_operation_id IS NULL",
                (current_time, new_media_version, client_id, media_id, current_media_version),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError(entity="Media", identifier=media_id)  # noqa: TRY301

            delete_payload = {
                "uuid": media_uuid,
                "last_modified": current_time,
                "version": new_media_version,
                "client_id": client_id,
                "deleted": 1,
            }
            self._log_sync_event(conn, "Media", media_uuid, "delete", new_media_version, delete_payload)
            self._delete_fts_media(conn, media_id)

            if cascade:
                logger.info(f"Performing explicit cascade delete for Media ID: {media_id}")
                keywords_to_unlink = self._fetchall_with_connection(
                    conn,
                    "SELECT mk.keyword_id AS keyword_id, k.uuid AS keyword_uuid FROM MediaKeywords mk "
                    "JOIN Keywords k ON mk.keyword_id = k.id "
                    "WHERE mk.media_id = ? AND k.deleted = 0",
                    (media_id,),
                )
                if keywords_to_unlink:
                    keyword_ids = [keyword["keyword_id"] for keyword in keywords_to_unlink]
                    placeholders = ",".join("?" * len(keyword_ids))
                    params = (media_id, *keyword_ids)
                    self._execute_with_connection(
                        conn,
                        f"DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id IN ({placeholders})",  # nosec B608
                        params,
                    )
                    unlink_version = 1
                    for kw_link in keywords_to_unlink:
                        link_uuid = f"{media_uuid}_{kw_link['keyword_uuid']}"
                        unlink_payload = {
                            "media_uuid": media_uuid,
                            "keyword_uuid": kw_link["keyword_uuid"],
                        }
                        self._log_sync_event(
                            conn,
                            "MediaKeywords",
                            link_uuid,
                            "unlink",
                            unlink_version,
                            unlink_payload,
                        )

                child_tables = [
                    (
                        "Transcripts",
                        "media_id",
                        "uuid",
                        "UPDATE Transcripts SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0",
                    ),
                    (
                        "MediaChunks",
                        "media_id",
                        "uuid",
                        "UPDATE MediaChunks SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0",
                    ),
                    (
                        "UnvectorizedMediaChunks",
                        "media_id",
                        "uuid",
                        "UPDATE UnvectorizedMediaChunks SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0",
                    ),
                    (
                        "DocumentVersions",
                        "media_id",
                        "uuid",
                        "UPDATE DocumentVersions SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0",
                    ),
                ]
                for table, fk_col, uuid_col, update_sql in child_tables:
                    children = self._fetchall_with_connection(
                        conn,
                        f"SELECT id, {uuid_col} AS uuid, version FROM {table} WHERE {fk_col} = ? AND deleted = 0",  # nosec B608
                        (media_id,),
                    )
                    if not children:
                        continue

                    processed_children_count = 0
                    for child in children:
                        child_id = child["id"]
                        child_uuid = child["uuid"]
                        child_current_version = child["version"]
                        child_new_version = child_current_version + 1
                        params = (current_time, child_new_version, client_id, child_id, child_current_version)
                        child_cursor = self._execute_with_connection(conn, update_sql, params)
                        if child_cursor.rowcount == 1:
                            processed_children_count += 1
                            child_delete_payload = {
                                "uuid": child_uuid,
                                "media_uuid": media_uuid,
                                "last_modified": current_time,
                                "version": child_new_version,
                                "client_id": client_id,
                                "deleted": 1,
                            }
                            self._log_sync_event(
                                conn,
                                table,
                                child_uuid,
                                "delete",
                                child_new_version,
                                child_delete_payload,
                            )
                        else:
                            logger.warning(f"Conflict/error cascade deleting {table} ID {child_id}")
                    logger.debug(
                        f"Cascade deleted {processed_children_count}/{len(children)} records in {table}."
                    )

        logger.info(f"Soft delete successful for Media ID: {media_id}.")
        try:
            from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
                invalidate_intra_doc_vectors,
            )

            invalidate_intra_doc_vectors(str(media_id))
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            pass
    except (ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error soft deleting media ID {media_id}: {exc}", exc_info=True)
        if isinstance(exc, (ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed to soft delete media: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error soft deleting media ID {media_id}: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected error during soft delete: {exc}") from exc  # noqa: TRY003
    else:
        return True


def share_media(
    self: Any,
    media_id: int,
    visibility: str,
    *,
    org_id: int | None = None,
    team_id: int | None = None,
) -> bool:
    """Update a media item visibility scope."""
    valid_visibilities = ("personal", "team", "org")
    if visibility not in valid_visibilities:
        raise InputError(f"Invalid visibility '{visibility}'. Must be one of: {valid_visibilities}")  # noqa: TRY003

    if visibility == "team" and team_id is None:
        raise InputError("team_id is required for 'team' visibility")  # noqa: TRY003
    if visibility == "org" and org_id is None:
        raise InputError("org_id is required for 'org' visibility")  # noqa: TRY003

    now = self._get_current_utc_timestamp_str()

    try:
        with self.transaction() as conn:
            row = self._fetchone_with_connection(
                conn,
                "SELECT id, uuid, version, visibility, org_id, team_id FROM Media "
                "WHERE id = ? AND deleted = 0 AND system_operation_id IS NULL",
                (media_id,),
            )
            if not row:
                raise InputError(f"Media ID {media_id} not found or deleted")  # noqa: TRY003, TRY301

            media_uuid = row["uuid"]
            current_version = row["version"]
            new_version = current_version + 1

            new_org_id = org_id if visibility in ("team", "org") else None
            new_team_id = team_id if visibility == "team" else None

            update_sql = """
                UPDATE Media
                SET visibility = ?, org_id = ?, team_id = ?, version = ?, last_modified = ?, client_id = ?
                WHERE id = ? AND version = ? AND system_operation_id IS NULL
            """
            cursor = self._execute_with_connection(
                conn,
                update_sql,
                (visibility, new_org_id, new_team_id, new_version, now, self.client_id, media_id, current_version),
            )
            if cursor.rowcount == 0:
                raise ConflictError(
                    f"Concurrent modification detected for media ID {media_id}",
                    entity="Media",
                    identifier=media_id,
                )  # noqa: TRY003, TRY301

            payload = {
                "visibility": visibility,
                "org_id": new_org_id,
                "team_id": new_team_id,
                "version": new_version,
                "last_modified": now,
            }
            self._log_sync_event(conn, "Media", media_uuid, "update", new_version, payload)

            logger.info(f"Shared media ID {media_id} with visibility '{visibility}'")
            return True
    except (InputError, ConflictError, DatabaseError) as exc:
        logger.error(f"Error sharing media ID {media_id}: {exc}", exc_info=True)
        raise
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error sharing media ID {media_id}: {exc}", exc_info=True)
        raise DatabaseError(f"Failed to share media: {exc}") from exc  # noqa: TRY003


def unshare_media(self: Any, media_id: int) -> bool:
    """Restore a media item to personal visibility."""
    return self.share_media(media_id, visibility="personal")


def get_media_visibility(self: Any, media_id: int) -> dict[str, Any] | None:
    """Return the current visibility state for a media row."""
    cursor = self.execute_query(
        "SELECT visibility, org_id, team_id, owner_user_id, client_id "
        "FROM Media WHERE id = ? AND deleted = 0 AND system_operation_id IS NULL",
        (media_id,),
    )
    row = cursor.fetchone() if cursor else None
    if not row:
        return None

    return {
        "visibility": row.get("visibility", "personal"),
        "org_id": row.get("org_id"),
        "team_id": row.get("team_id"),
        "owner_user_id": row.get("owner_user_id"),
        "client_id": row.get("client_id"),
    }


def mark_as_trash(self: Any, media_id: int) -> bool:
    """Mark a media row as trash without soft-deleting it."""
    current_time = self._get_current_utc_timestamp_str()
    client_id = self.client_id
    logger.debug(f"Marking media {media_id} as trash.")
    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version, is_trash FROM Media WHERE id = ? AND deleted = 0 "
                "AND system_operation_id IS NULL",
                (media_id,),
            )
            if not media_info:
                logger.warning(f"Cannot trash: Media {media_id} not found/deleted.")
                return False
            if media_info["is_trash"]:
                logger.warning(f"Media {media_id} already in trash.")
                return False

            media_uuid, current_version = media_info["uuid"], media_info["version"]
            new_version = current_version + 1
            update_cursor = self._execute_with_connection(
                conn,
                "UPDATE Media SET is_trash=1, trash_date=?, last_modified=?, version=?, "
                "client_id=? WHERE id=? AND version=? AND system_operation_id IS NULL",
                (current_time, current_time, new_version, client_id, media_id, current_version),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            sync_payload = self._fetchone_with_connection(
                conn,
                "SELECT * FROM Media WHERE id = ? AND system_operation_id IS NULL",
                (media_id,),
            ) or {}
            self._log_sync_event(conn, "Media", media_uuid, "update", new_version, sync_payload)
            logger.info(f"Media {media_id} marked as trash. New ver: {new_version}")
            return True
    except (ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error marking media {media_id} as trash: {exc}", exc_info=True)
        if isinstance(exc, (ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed mark as trash: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error marking media {media_id} trash: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected mark trash error: {exc}") from exc  # noqa: TRY003


def restore_from_trash(self: Any, media_id: int) -> bool:
    """Restore a media row from trash without affecting FTS."""
    current_time = self._get_current_utc_timestamp_str()
    client_id = self.client_id
    logger.debug(f"Restoring media {media_id} from trash.")
    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version, is_trash FROM Media WHERE id = ? AND deleted = 0 "
                "AND system_operation_id IS NULL",
                (media_id,),
            )
            if not media_info:
                logger.warning(f"Cannot restore: Media {media_id} not found/deleted.")
                return False
            if not media_info["is_trash"]:
                logger.warning(f"Cannot restore: Media {media_id} not in trash.")
                return False

            media_uuid, current_version = media_info["uuid"], media_info["version"]
            new_version = current_version + 1
            update_cursor = self._execute_with_connection(
                conn,
                "UPDATE Media SET is_trash=0, trash_date=NULL, last_modified=?, version=?, "
                "client_id=? WHERE id=? AND version=? AND system_operation_id IS NULL",
                (current_time, new_version, client_id, media_id, current_version),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            sync_payload = self._fetchone_with_connection(
                conn,
                "SELECT * FROM Media WHERE id = ? AND system_operation_id IS NULL",
                (media_id,),
            ) or {}
            self._log_sync_event(conn, "Media", media_uuid, "update", new_version, sync_payload)
            logger.info(f"Media {media_id} restored from trash. New ver: {new_version}")
            return True
    except (ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error restoring media {media_id} trash: {exc}", exc_info=True)
        if isinstance(exc, (ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed restore trash: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error restoring media {media_id} trash: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected restore trash error: {exc}") from exc  # noqa: TRY003


__all__ = [
    "get_media_visibility",
    "mark_as_trash",
    "restore_from_trash",
    "share_media",
    "soft_delete_media",
    "unshare_media",
]
