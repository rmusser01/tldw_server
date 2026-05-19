"""Repository for generated files tracking in AuthNZ.

This module provides CRUD operations for the generated_files table,
which tracks user-generated content files (TTS audio, images, voice clones,
mindmaps, spreadsheets).
"""
from __future__ import annotations

import contextlib
import json
import uuid as uuid_module
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

# File categories (used for file_category field)
FILE_CATEGORY_TTS_AUDIO = "tts_audio"
FILE_CATEGORY_STT_AUDIO = "stt_audio"
FILE_CATEGORY_IMAGE = "image"
FILE_CATEGORY_VOICE_CLONE = "voice_clone"
FILE_CATEGORY_MINDMAP = "mindmap"
FILE_CATEGORY_SPREADSHEET = "spreadsheet"

VALID_FILE_CATEGORIES = {
    FILE_CATEGORY_TTS_AUDIO,
    FILE_CATEGORY_STT_AUDIO,
    FILE_CATEGORY_IMAGE,
    FILE_CATEGORY_VOICE_CLONE,
    FILE_CATEGORY_MINDMAP,
    FILE_CATEGORY_SPREADSHEET,
}

# Source features (used for source_feature field)
SOURCE_FEATURE_TTS = "tts"
SOURCE_FEATURE_STT = "stt"
SOURCE_FEATURE_IMAGE_GEN = "image_gen"
SOURCE_FEATURE_VOICE_STUDIO = "voice_studio"
SOURCE_FEATURE_MINDMAP = "mindmap"
SOURCE_FEATURE_DATA_TABLES = "data_tables"
SOURCE_FEATURE_EXPORT = "export"
SOURCE_FEATURE_VN_ASSETS = "vn_assets"

VALID_SOURCE_FEATURES = {
    SOURCE_FEATURE_TTS,
    SOURCE_FEATURE_STT,
    SOURCE_FEATURE_IMAGE_GEN,
    SOURCE_FEATURE_VOICE_STUDIO,
    SOURCE_FEATURE_MINDMAP,
    SOURCE_FEATURE_DATA_TABLES,
    SOURCE_FEATURE_EXPORT,
    SOURCE_FEATURE_VN_ASSETS,
}

# Retention policies
RETENTION_POLICY_USER_DEFAULT = "user_default"
RETENTION_POLICY_PERMANENT = "permanent"
RETENTION_POLICY_TRANSIENT = "transient"
RETENTION_POLICY_CUSTOM = "custom"


@dataclass
class AuthnzGeneratedFilesRepo:
    """
    Repository for AuthNZ generated_files records.

    Provides CRUD operations for tracking user-generated content files
    with support for both SQLite and PostgreSQL backends.
    """

    db_pool: DatabasePool

    def _is_postgres(self) -> bool:
        """Detect whether the current AuthNZ backend is PostgreSQL."""
        return getattr(self.db_pool, "pool", None) is not None

    @staticmethod
    def _normalize_record(row: Any) -> dict[str, Any]:
        """
        Normalize backend-specific row types to a consistent dict with
        JSON-friendly types.
        """
        if row is None:
            return {}
        try:
            record = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else {}
        except Exception:
            record = {}

        # Type conversions
        if "id" in record and record["id"] is not None:
            with contextlib.suppress(Exception):
                record["id"] = int(record["id"])

        if "user_id" in record and record["user_id"] is not None:
            with contextlib.suppress(Exception):
                record["user_id"] = int(record["user_id"])

        for field in ("org_id", "team_id", "file_size_bytes"):
            if field in record and record[field] is not None:
                with contextlib.suppress(Exception):
                    record[field] = int(record[field])

        for field in ("is_transient", "is_deleted"):
            if field in record:
                with contextlib.suppress(Exception):
                    record[field] = bool(record[field])

        # Parse JSON tags field
        if "tags" in record and record["tags"]:
            try:
                if isinstance(record["tags"], str):
                    record["tags"] = json.loads(record["tags"])
            except json.JSONDecodeError:
                record["tags"] = []

        return record

    async def create_file(
        self,
        *,
        user_id: int,
        filename: str,
        storage_path: str,
        file_category: str,
        source_feature: str,
        file_size_bytes: int = 0,
        org_id: int | None = None,
        team_id: int | None = None,
        original_filename: str | None = None,
        mime_type: str | None = None,
        checksum: str | None = None,
        source_ref: str | None = None,
        folder_tag: str | None = None,
        tags: list[str] | None = None,
        is_transient: bool = False,
        expires_at: datetime | None = None,
        retention_policy: str = RETENTION_POLICY_USER_DEFAULT,
    ) -> dict[str, Any]:
        """
        Create a new generated file record.

        Args:
            user_id: Owner user ID
            filename: Stored filename
            storage_path: Relative path to file in storage
            file_category: Category (tts_audio, image, voice_clone, mindmap, spreadsheet)
            source_feature: Feature that generated the file (tts, image_gen, etc.)
            file_size_bytes: File size in bytes
            org_id: Optional organization ID
            team_id: Optional team ID
            original_filename: Original filename from user
            mime_type: MIME type
            checksum: SHA-256 checksum
            source_ref: Reference to source entity (e.g., media_id)
            folder_tag: Virtual folder tag
            tags: Additional tags (JSON array)
            is_transient: Whether file is transient
            expires_at: Expiration timestamp
            retention_policy: Retention policy

        Returns:
            Created file record as dict
        """
        file_uuid = str(uuid_module.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags) if tags else None

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    # PostgreSQL path
                    row = await conn.fetchrow(
                        """
                        INSERT INTO generated_files (
                            uuid, user_id, org_id, team_id, filename, original_filename,
                            storage_path, mime_type, file_size_bytes, checksum,
                            file_category, source_feature, source_ref, folder_tag, tags,
                            is_transient, expires_at, retention_policy,
                            is_deleted, created_at, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18,
                            FALSE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                        )
                        RETURNING *
                        """,
                        file_uuid, user_id, org_id, team_id, filename, original_filename,
                        storage_path, mime_type, file_size_bytes, checksum,
                        file_category, source_feature, source_ref, folder_tag, tags_json,
                        is_transient, expires_at, retention_policy,
                    )
                    return self._normalize_record(row)
                else:
                    # SQLite path
                    cursor = await conn.execute(
                        """
                        INSERT INTO generated_files (
                            uuid, user_id, org_id, team_id, filename, original_filename,
                            storage_path, mime_type, file_size_bytes, checksum,
                            file_category, source_feature, source_ref, folder_tag, tags,
                            is_transient, expires_at, retention_policy,
                            is_deleted, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            file_uuid, user_id, org_id, team_id, filename, original_filename,
                            storage_path, mime_type, file_size_bytes, checksum,
                            file_category, source_feature, source_ref, folder_tag, tags_json,
                            1 if is_transient else 0, expires_at.isoformat() if expires_at else None,
                            retention_policy, now_iso, now_iso,
                        ),
                    )
                    file_id = getattr(cursor, "lastrowid", None)
                    if file_id is None:
                        raise RuntimeError("Failed to obtain file id")

                    # Fetch the created record
                    cursor = await conn.execute(
                        "SELECT * FROM generated_files WHERE id = ?",
                        (file_id,),
                    )
                    row = await cursor.fetchone()
                    if row:
                        cols = [desc[0] for desc in cursor.description]
                        return self._normalize_record(dict(zip(cols, row)))
                    return {"id": file_id, "uuid": file_uuid}

        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.create_file failed: {exc}")
            raise

    async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
        """Fetch a file record by ID."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    row = await conn.fetchrow(
                        "SELECT * FROM generated_files WHERE id = $1",
                        file_id,
                    )
                    return self._normalize_record(row) if row else None
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM generated_files WHERE id = ?",
                        (file_id,),
                    )
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    cols = [desc[0] for desc in cursor.description]
                    return self._normalize_record(dict(zip(cols, row)))
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.get_file_by_id failed: {exc}")
            raise

    async def get_files_by_ids(self, file_ids: list[int]) -> list[dict[str, Any]]:
        """Fetch generated file records for a bounded list of IDs."""
        if not file_ids:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    rows = await conn.fetch(
                        "SELECT * FROM generated_files WHERE id = ANY($1)",
                        file_ids,
                    )
                    return [self._normalize_record(row) for row in rows]

                placeholders = ",".join("?" for _ in file_ids)
                id_list_clause = f"({placeholders})"
                select_sql_template = "SELECT * FROM generated_files WHERE id IN {id_list_clause}"
                select_sql = select_sql_template.format_map(locals())  # nosec B608
                cursor = await conn.execute(select_sql, tuple(file_ids))
                rows = await cursor.fetchall()
                cols = [desc[0] for desc in cursor.description] if cursor.description else []
                return [self._normalize_record(dict(zip(cols, row))) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.get_files_by_ids failed: {exc}")
            raise

    async def get_file_by_uuid(self, file_uuid: str) -> dict[str, Any] | None:
        """Fetch a file record by UUID."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    row = await conn.fetchrow(
                        "SELECT * FROM generated_files WHERE uuid = $1",
                        file_uuid,
                    )
                    return self._normalize_record(row) if row else None
                else:
                    cursor = await conn.execute(
                        "SELECT * FROM generated_files WHERE uuid = ?",
                        (file_uuid,),
                    )
                    row = await cursor.fetchone()
                    if not row:
                        return None
                    cols = [desc[0] for desc in cursor.description]
                    return self._normalize_record(dict(zip(cols, row)))
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.get_file_by_uuid failed: {exc}")
            raise

    async def list_files(
        self,
        *,
        user_id: int,
        offset: int = 0,
        limit: int = 50,
        file_category: str | None = None,
        source_feature: str | None = None,
        folder_tag: str | None = None,
        include_deleted: bool = False,
        search: str | None = None,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List files for a user with filtering and pagination.

        Returns:
            Tuple of (list of file records, total count)
        """
        try:
            async with self.db_pool.acquire() as conn:
                is_pg = self._is_postgres()

                # Build WHERE clause
                conditions = ["user_id = " + ("$1" if is_pg else "?")]
                params: list[Any] = [user_id]
                param_idx = 1

                if not include_deleted:
                    conditions.append("is_deleted = " + ("FALSE" if is_pg else "0"))

                if file_category:
                    param_idx += 1
                    conditions.append("file_category = " + (f"${param_idx}" if is_pg else "?"))
                    params.append(file_category)

                if source_feature:
                    param_idx += 1
                    conditions.append("source_feature = " + (f"${param_idx}" if is_pg else "?"))
                    params.append(source_feature)

                if folder_tag:
                    param_idx += 1
                    conditions.append("folder_tag = " + (f"${param_idx}" if is_pg else "?"))
                    params.append(folder_tag)

                if org_id:
                    param_idx += 1
                    conditions.append("org_id = " + (f"${param_idx}" if is_pg else "?"))
                    params.append(org_id)

                if team_id:
                    param_idx += 1
                    conditions.append("team_id = " + (f"${param_idx}" if is_pg else "?"))
                    params.append(team_id)

                if search:
                    param_idx += 1
                    search_pattern = f"%{search}%"
                    if is_pg:
                        conditions.append(f"(filename ILIKE ${param_idx} OR original_filename ILIKE ${param_idx})")
                    else:
                        conditions.append("(filename LIKE ? OR original_filename LIKE ?)")
                        params.append(search_pattern)  # For first LIKE
                    params.append(search_pattern)

                where_clause = " AND ".join(conditions)

                # Get count
                count_sql_template = "SELECT COUNT(*) FROM generated_files WHERE {where_clause}"
                count_sql = count_sql_template.format_map(locals())  # nosec B608
                if is_pg:
                    total = await conn.fetchval(count_sql, *params)
                else:
                    cursor = await conn.execute(count_sql, tuple(params))
                    row = await cursor.fetchone()
                    total = row[0] if row else 0

                # Get paginated results
                if is_pg:
                    param_idx += 1
                    offset_param = f"${param_idx}"
                    param_idx += 1
                    limit_param = f"${param_idx}"
                    select_sql_template = """
                        SELECT * FROM generated_files
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        OFFSET {offset_param} LIMIT {limit_param}
                    """
                    select_sql = select_sql_template.format_map(locals())  # nosec B608
                    rows = await conn.fetch(select_sql, *params, offset, limit)
                    files = [self._normalize_record(row) for row in rows]
                else:
                    select_sql_template = """
                        SELECT * FROM generated_files
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """
                    select_sql = select_sql_template.format_map(locals())  # nosec B608
                    cursor = await conn.execute(select_sql, (*params, limit, offset))
                    rows = await cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description] if cursor.description else []
                    files = [self._normalize_record(dict(zip(cols, row))) for row in rows]

                return files, int(total)

        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.list_files failed: {exc}")
            raise

    async def update_file(
        self,
        file_id: int,
        *,
        folder_tag: str | None = None,
        tags: list[str] | None = None,
        retention_policy: str | None = None,
        expires_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Update file metadata."""
        updates = []
        params: list[Any] = []
        param_idx = 0

        if folder_tag is not None:
            param_idx += 1
            updates.append("folder_tag = " + (f"${param_idx}" if self._is_postgres() else "?"))
            params.append(folder_tag if folder_tag else None)

        if tags is not None:
            param_idx += 1
            updates.append("tags = " + (f"${param_idx}" if self._is_postgres() else "?"))
            params.append(json.dumps(tags))

        if retention_policy is not None:
            param_idx += 1
            updates.append("retention_policy = " + (f"${param_idx}" if self._is_postgres() else "?"))
            params.append(retention_policy)

        if expires_at is not None:
            param_idx += 1
            updates.append("expires_at = " + (f"${param_idx}" if self._is_postgres() else "?"))
            params.append(expires_at.isoformat() if expires_at else None)

        if not updates:
            return await self.get_file_by_id(file_id)

        # Always update updated_at
        if self._is_postgres():
            updates.append("updated_at = CURRENT_TIMESTAMP")
        else:
            param_idx += 1
            updates.append("updated_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

        # Add file_id to params
        param_idx += 1
        params.append(file_id)

        try:
            async with self.db_pool.transaction() as conn:
                file_id_param = f"${param_idx}" if self._is_postgres() else "?"
                set_clause_sql = ", ".join(updates)
                update_sql_template = """
                    UPDATE generated_files
                    SET {set_clause_sql}
                    WHERE id = {file_id_param}
                """
                update_sql = update_sql_template.format_map(locals())  # nosec B608

                if self._is_postgres():
                    await conn.execute(update_sql, *params)
                else:
                    await conn.execute(update_sql, tuple(params))

            return await self.get_file_by_id(file_id)

        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.update_file failed: {exc}")
            raise

    async def soft_delete_file(self, file_id: int) -> bool:
        """Soft delete a file (mark as deleted)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        """
                        UPDATE generated_files
                        SET is_deleted = TRUE, deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1 AND is_deleted = FALSE
                        """,
                        file_id,
                    )
                    return "UPDATE 1" in str(result)
                else:
                    cursor = await conn.execute(
                        """
                        UPDATE generated_files
                        SET is_deleted = 1, deleted_at = ?, updated_at = ?
                        WHERE id = ? AND is_deleted = 0
                        """,
                        (now_iso, now_iso, file_id),
                    )
                    return cursor.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.soft_delete_file failed: {exc}")
            raise

    async def restore_file(self, file_id: int) -> bool:
        """Restore a soft-deleted file."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        """
                        UPDATE generated_files
                        SET is_deleted = FALSE, deleted_at = NULL, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1 AND is_deleted = TRUE
                        """,
                        file_id,
                    )
                    return "UPDATE 1" in str(result)
                else:
                    cursor = await conn.execute(
                        """
                        UPDATE generated_files
                        SET is_deleted = 0, deleted_at = NULL, updated_at = ?
                        WHERE id = ? AND is_deleted = 1
                        """,
                        (now_iso, file_id),
                    )
                    return cursor.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.restore_file failed: {exc}")
            raise

    async def hard_delete_file(self, file_id: int) -> bool:
        """Permanently delete a file record."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        "DELETE FROM generated_files WHERE id = $1",
                        file_id,
                    )
                    return "DELETE 1" in str(result)
                else:
                    cursor = await conn.execute(
                        "DELETE FROM generated_files WHERE id = ?",
                        (file_id,),
                    )
                    return cursor.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.hard_delete_file failed: {exc}")
            raise

    async def bulk_soft_delete(self, file_ids: list[int]) -> int:
        """Soft delete multiple files. Returns count of deleted files."""
        if not file_ids:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        """
                        UPDATE generated_files
                        SET is_deleted = TRUE, deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ANY($1) AND is_deleted = FALSE
                        """,
                        file_ids,
                    )
                    # Parse "UPDATE N" to get count
                    parts = str(result).split()
                    return int(parts[1]) if len(parts) > 1 else 0
                else:
                    placeholders = ",".join("?" for _ in file_ids)
                    id_list_clause = f"({placeholders})"
                    update_sql_template = """
                        UPDATE generated_files
                        SET is_deleted = 1, deleted_at = ?, updated_at = ?
                        WHERE id IN {id_list_clause} AND is_deleted = 0
                        """
                    update_sql = update_sql_template.format_map(locals())  # nosec B608
                    cursor = await conn.execute(
                        update_sql,
                        (now_iso, now_iso, *file_ids),
                    )
                    return cursor.rowcount
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.bulk_soft_delete failed: {exc}")
            raise

    async def bulk_move_to_folder(self, file_ids: list[int], folder_tag: str | None) -> int:
        """Move multiple files to a folder (tag). Returns count of updated files."""
        if not file_ids:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    result = await conn.execute(
                        """
                        UPDATE generated_files
                        SET folder_tag = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ANY($2) AND is_deleted = FALSE
                        """,
                        folder_tag,
                        file_ids,
                    )
                    parts = str(result).split()
                    return int(parts[1]) if len(parts) > 1 else 0
                else:
                    placeholders = ",".join("?" for _ in file_ids)
                    id_list_clause = f"({placeholders})"
                    update_sql_template = """
                        UPDATE generated_files
                        SET folder_tag = ?, updated_at = ?
                        WHERE id IN {id_list_clause} AND is_deleted = 0
                        """
                    update_sql = update_sql_template.format_map(locals())  # nosec B608
                    cursor = await conn.execute(
                        update_sql,
                        (folder_tag, now_iso, *file_ids),
                    )
                    return cursor.rowcount
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.bulk_move_to_folder failed: {exc}")
            raise

    async def list_folders(self, user_id: int) -> list[dict[str, Any]]:
        """List unique folder tags for a user with file counts."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    rows = await conn.fetch(
                        """
                        SELECT folder_tag, COUNT(*) as file_count, SUM(file_size_bytes) as total_bytes
                        FROM generated_files
                        WHERE user_id = $1 AND is_deleted = FALSE AND folder_tag IS NOT NULL
                        GROUP BY folder_tag
                        ORDER BY folder_tag
                        """,
                        user_id,
                    )
                    return [
                        {
                            "folder_tag": row["folder_tag"],
                            "file_count": int(row["file_count"]),
                            "total_bytes": int(row["total_bytes"] or 0),
                        }
                        for row in rows
                    ]
                else:
                    cursor = await conn.execute(
                        """
                        SELECT folder_tag, COUNT(*) as file_count, SUM(file_size_bytes) as total_bytes
                        FROM generated_files
                        WHERE user_id = ? AND is_deleted = 0 AND folder_tag IS NOT NULL
                        GROUP BY folder_tag
                        ORDER BY folder_tag
                        """,
                        (user_id,),
                    )
                    rows = await cursor.fetchall()
                    return [
                        {
                            "folder_tag": row[0],
                            "file_count": int(row[1]),
                            "total_bytes": int(row[2] or 0),
                        }
                        for row in rows
                    ]
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.list_folders failed: {exc}")
            raise

    async def list_trashed_files(
        self,
        user_id: int,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """List soft-deleted files for a user."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    total = await conn.fetchval(
                        """
                        SELECT COUNT(*) FROM generated_files
                        WHERE user_id = $1 AND is_deleted = TRUE
                        """,
                        user_id,
                    )
                    rows = await conn.fetch(
                        """
                        SELECT * FROM generated_files
                        WHERE user_id = $1 AND is_deleted = TRUE
                        ORDER BY deleted_at DESC
                        OFFSET $2 LIMIT $3
                        """,
                        user_id, offset, limit,
                    )
                    files = [self._normalize_record(row) for row in rows]
                else:
                    cursor = await conn.execute(
                        """
                        SELECT COUNT(*) FROM generated_files
                        WHERE user_id = ? AND is_deleted = 1
                        """,
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    total = row[0] if row else 0

                    cursor = await conn.execute(
                        """
                        SELECT * FROM generated_files
                        WHERE user_id = ? AND is_deleted = 1
                        ORDER BY deleted_at DESC
                        LIMIT ? OFFSET ?
                        """,
                        (user_id, limit, offset),
                    )
                    rows = await cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description] if cursor.description else []
                    files = [self._normalize_record(dict(zip(cols, row))) for row in rows]

                return files, int(total)

        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.list_trashed_files failed: {exc}")
            raise

    async def get_user_storage_usage(self, user_id: int) -> dict[str, Any]:
        """Get storage usage statistics for a user."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    # Total usage
                    total_bytes = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(file_size_bytes), 0) FROM generated_files
                        WHERE user_id = $1 AND is_deleted = FALSE
                        """,
                        user_id,
                    )

                    # By category
                    category_rows = await conn.fetch(
                        """
                        SELECT file_category, COUNT(*) as file_count, COALESCE(SUM(file_size_bytes), 0) as total_bytes
                        FROM generated_files
                        WHERE user_id = $1 AND is_deleted = FALSE
                        GROUP BY file_category
                        """,
                        user_id,
                    )
                    by_category = {
                        row["file_category"]: {
                            "file_count": int(row["file_count"]),
                            "total_bytes": int(row["total_bytes"]),
                        }
                        for row in category_rows
                    }

                    # Trash usage
                    trash_bytes = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(file_size_bytes), 0) FROM generated_files
                        WHERE user_id = $1 AND is_deleted = TRUE
                        """,
                        user_id,
                    )
                else:
                    cursor = await conn.execute(
                        """
                        SELECT COALESCE(SUM(file_size_bytes), 0) FROM generated_files
                        WHERE user_id = ? AND is_deleted = 0
                        """,
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    total_bytes = row[0] if row else 0

                    cursor = await conn.execute(
                        """
                        SELECT file_category, COUNT(*) as file_count, COALESCE(SUM(file_size_bytes), 0) as total_bytes
                        FROM generated_files
                        WHERE user_id = ? AND is_deleted = 0
                        GROUP BY file_category
                        """,
                        (user_id,),
                    )
                    category_rows = await cursor.fetchall()
                    by_category = {
                        row[0]: {
                            "file_count": int(row[1]),
                            "total_bytes": int(row[2]),
                        }
                        for row in category_rows
                    }

                    cursor = await conn.execute(
                        """
                        SELECT COALESCE(SUM(file_size_bytes), 0) FROM generated_files
                        WHERE user_id = ? AND is_deleted = 1
                        """,
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    trash_bytes = row[0] if row else 0

                return {
                    "total_bytes": int(total_bytes),
                    "total_mb": round(int(total_bytes) / (1024 * 1024), 2),
                    "by_category": by_category,
                    "trash_bytes": int(trash_bytes),
                    "trash_mb": round(int(trash_bytes) / (1024 * 1024), 2),
                }

        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.get_user_storage_usage failed: {exc}")
            raise

    async def get_expired_files(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get files that have passed their expiration date."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    rows = await conn.fetch(
                        """
                        SELECT * FROM generated_files
                        WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                          AND is_deleted = FALSE
                        ORDER BY expires_at
                        LIMIT $1
                        """,
                        limit,
                    )
                    return [self._normalize_record(row) for row in rows]
                else:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    cursor = await conn.execute(
                        """
                        SELECT * FROM generated_files
                        WHERE expires_at IS NOT NULL AND expires_at < ?
                          AND is_deleted = 0
                        ORDER BY expires_at
                        LIMIT ?
                        """,
                        (now_iso, limit),
                    )
                    rows = await cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description] if cursor.description else []
                    return [self._normalize_record(dict(zip(cols, row))) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.get_expired_files failed: {exc}")
            raise

    async def get_old_trashed_files(self, days_old: int = 30, limit: int = 100) -> list[dict[str, Any]]:
        """Get files that have been in trash for more than specified days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_old)

        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    # Use interval arithmetic in SQL to avoid timezone-aware Python
                    # datetime binds against TIMESTAMP (without timezone) columns.
                    rows = await conn.fetch(
                        """
                        SELECT * FROM generated_files
                        WHERE is_deleted = TRUE
                          AND deleted_at < (CURRENT_TIMESTAMP - ($1::int * INTERVAL '1 day'))
                        ORDER BY deleted_at
                        LIMIT $2
                        """,
                        days_old,
                        limit,
                    )
                    return [self._normalize_record(row) for row in rows]
                else:
                    cursor = await conn.execute(
                        """
                        SELECT * FROM generated_files
                        WHERE is_deleted = 1
                          AND deleted_at < ?
                        ORDER BY deleted_at
                        LIMIT ?
                        """,
                        (cutoff.isoformat(), limit),
                    )
                    rows = await cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description] if cursor.description else []
                    return [self._normalize_record(dict(zip(cols, row))) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.get_old_trashed_files failed: {exc}")
            raise

    async def update_accessed_at(self, file_id: int) -> None:
        """Update the accessed_at timestamp for a file."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres():
                    await conn.execute(
                        "UPDATE generated_files SET accessed_at = CURRENT_TIMESTAMP WHERE id = $1",
                        file_id,
                    )
                else:
                    await conn.execute(
                        "UPDATE generated_files SET accessed_at = ? WHERE id = ?",
                        (now_iso, file_id),
                    )
        except Exception as exc:
            logger.debug(f"AuthnzGeneratedFilesRepo.update_accessed_at failed: {exc}")

    async def list_least_accessed(
        self,
        user_id: int,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List files by least recent access time (for cleanup suggestions).

        Useful for users approaching quota limits who need to free space.
        Files that have never been accessed are sorted by created_at instead.
        """
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    rows = await conn.fetch(
                        """
                        SELECT * FROM generated_files
                        WHERE user_id = $1 AND is_deleted = FALSE
                        ORDER BY COALESCE(accessed_at, created_at) ASC
                        LIMIT $2
                        """,
                        user_id, limit,
                    )
                    return [self._normalize_record(row) for row in rows]
                else:
                    cursor = await conn.execute(
                        """
                        SELECT * FROM generated_files
                        WHERE user_id = ? AND is_deleted = 0
                        ORDER BY COALESCE(accessed_at, created_at) ASC
                        LIMIT ?
                        """,
                        (user_id, limit),
                    )
                    rows = await cursor.fetchall()
                    cols = [desc[0] for desc in cursor.description] if cursor.description else []
                    return [self._normalize_record(dict(zip(cols, row))) for row in rows]
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.list_least_accessed failed: {exc}")
            raise

    async def count_least_accessed(self, user_id: int) -> int:
        """Count current cleanup candidates for least-accessed pagination metadata."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres():
                    return int(
                        await conn.fetchval(
                            """
                            SELECT COUNT(*) FROM generated_files
                            WHERE user_id = $1 AND is_deleted = FALSE
                            """,
                            user_id,
                        )
                        or 0
                    )

                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM generated_files
                    WHERE user_id = ? AND is_deleted = 0
                    """,
                    (user_id,),
                )
                row = await cursor.fetchone()
                return int(row[0] if row else 0)
        except Exception as exc:
            logger.error(f"AuthnzGeneratedFilesRepo.count_least_accessed failed: {exc}")
            raise
