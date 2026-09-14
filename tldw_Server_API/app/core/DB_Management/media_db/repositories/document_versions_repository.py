from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike


class DocumentVersionsRepository:
    """Repository for document version persistence and lookup."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> DocumentVersionsRepository:
        return cls(session=db)

    def create(
        self,
        media_id: int,
        content: str,
        prompt: str | None = None,
        analysis_content: str | None = None,
        safe_metadata: str | None = None,
    ) -> dict[str, Any]:
        if content is None:
            raise InputError("Content is required for a document version.")  # noqa: TRY003

        db = self.session
        current_time = db._get_current_utc_timestamp_str()
        client_id = db.client_id
        new_uuid = db._generate_uuid()
        new_version = 1

        try:
            with db.transaction() as conn:
                def _exec(query: str, params: tuple | list | dict | None = None):
                    return db._execute_with_connection(conn, query, params)

                def _fetchone(query: str, params: tuple | list | dict | None = None):
                    return db._fetchone_with_connection(conn, query, params)

                media_info = _fetchone(
                    "SELECT uuid FROM Media WHERE id = ? AND deleted = 0 "
                    "AND system_operation_id IS NULL",
                    (media_id,),
                )
                if not media_info:
                    raise InputError(f"Parent Media ID {media_id} not found or deleted.")  # noqa: TRY003, TRY301
                media_uuid = media_info["uuid"]

                next_version_row = _fetchone(
                    "SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version FROM DocumentVersions WHERE media_id = ?",
                    (media_id,),
                )
                local_version_number = next_version_row["next_version"] if next_version_row else 1
                logger.debug(
                    "Creating document version {} for media ID {}, UUID {}",
                    local_version_number,
                    media_id,
                    new_uuid,
                )

                insert_data = {
                    "media_id": media_id,
                    "version_number": local_version_number,
                    "content": content,
                    "prompt": prompt,
                    "analysis_content": analysis_content,
                    "safe_metadata": safe_metadata,
                    "created_at": current_time,
                    "uuid": new_uuid,
                    "last_modified": current_time,
                    "version": new_version,
                    "client_id": client_id,
                    "deleted": 0,
                    "media_uuid": media_uuid,
                }
                try:
                    insert_sql = """
                        INSERT INTO DocumentVersions (
                            media_id, version_number, content, prompt, analysis_content, safe_metadata, created_at,
                            uuid, last_modified, version, client_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    insert_params = (
                        insert_data["media_id"],
                        insert_data["version_number"],
                        insert_data["content"],
                        insert_data["prompt"],
                        insert_data["analysis_content"],
                        insert_data["safe_metadata"],
                        insert_data["created_at"],
                        insert_data["uuid"],
                        insert_data["last_modified"],
                        insert_data["version"],
                        insert_data["client_id"],
                    )
                    if db.backend_type.name == "POSTGRESQL":
                        insert_sql += " RETURNING id"
                    insert_cursor = _exec(insert_sql, insert_params)
                except DatabaseError as exc:
                    message = str(exc).lower()
                    if "safe_metadata" in message and "no column" in message:
                        insert_sql = """
                            INSERT INTO DocumentVersions (
                                media_id, version_number, content, prompt, analysis_content, created_at,
                                uuid, last_modified, version, client_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                        insert_params = (
                            insert_data["media_id"],
                            insert_data["version_number"],
                            insert_data["content"],
                            insert_data["prompt"],
                            insert_data["analysis_content"],
                            insert_data["created_at"],
                            insert_data["uuid"],
                            insert_data["last_modified"],
                            insert_data["version"],
                            insert_data["client_id"],
                        )
                        insert_data["safe_metadata"] = None
                        if db.backend_type.name == "POSTGRESQL":
                            insert_sql += " RETURNING id"
                        insert_cursor = _exec(insert_sql, insert_params)
                    else:
                        raise

                if db.backend_type.name == "POSTGRESQL":
                    inserted_row = insert_cursor.fetchone()
                    version_id = inserted_row["id"] if inserted_row else None
                else:
                    version_id = insert_cursor.lastrowid
                if not version_id:
                    raise DatabaseError("Failed to get last row ID for new document version.")  # noqa: TRY003, TRY301

                try:
                    if insert_data.get("safe_metadata"):
                        import json as _json

                        try:
                            safe_metadata_data = (
                                _json.loads(insert_data["safe_metadata"])
                                if isinstance(insert_data["safe_metadata"], str)
                                else insert_data["safe_metadata"]
                            )
                        except Exception:
                            safe_metadata_data = None
                        if isinstance(safe_metadata_data, dict):
                            doi = safe_metadata_data.get("doi") or safe_metadata_data.get("DOI")
                            pmid = safe_metadata_data.get("pmid") or safe_metadata_data.get("PMID")
                            pmcid = safe_metadata_data.get("pmcid") or safe_metadata_data.get("PMCID")
                            arxiv_id = (
                                safe_metadata_data.get("arxiv_id")
                                or safe_metadata_data.get("arxiv")
                                or safe_metadata_data.get("ArXiv")
                            )
                            s2id = safe_metadata_data.get("s2_paper_id") or safe_metadata_data.get("paperId")
                            try:
                                if db.backend_type.name == "POSTGRESQL":
                                    ident_sql = (
                                        "INSERT INTO DocumentVersionIdentifiers (dv_id, doi, pmid, pmcid, arxiv_id, s2_paper_id) "
                                        "VALUES (?, ?, ?, ?, ?, ?) "
                                        "ON CONFLICT (dv_id) DO UPDATE SET "
                                        "doi = EXCLUDED.doi, pmid = EXCLUDED.pmid, pmcid = EXCLUDED.pmcid, "
                                        "arxiv_id = EXCLUDED.arxiv_id, s2_paper_id = EXCLUDED.s2_paper_id"
                                    )
                                else:
                                    ident_sql = (
                                        "INSERT OR REPLACE INTO DocumentVersionIdentifiers (dv_id, doi, pmid, pmcid, arxiv_id, s2_paper_id) "
                                        "VALUES (?, ?, ?, ?, ?, ?)"
                                    )
                                _exec(ident_sql, (version_id, doi, pmid, pmcid, arxiv_id, s2id))
                            except DatabaseError:
                                pass
                except Exception as exc:
                    logger.warning(
                        "Could not populate identifiers for version_id={}: {}",
                        version_id,
                        exc,
                    )

                db._log_sync_event(conn, "DocumentVersions", new_uuid, "create", new_version, insert_data)
        except (InputError, DatabaseError, sqlite3.Error) as exc:
            if "foreign key constraint failed" in str(exc).lower():
                logger.exception(
                    "Failed create document version: Media ID {} not found.",
                    media_id,
                    exc_info=False,
                )
                raise InputError(f"Cannot create document version: Media ID {media_id} not found.") from exc  # noqa: TRY003
            logger.error("DB error creating document version for media {}: {}", media_id, exc, exc_info=True)
            if isinstance(exc, (InputError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed create document version: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error creating document version for media {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error creating document version: {exc}") from exc  # noqa: TRY003
        else:
            return {
                "id": version_id,
                "uuid": new_uuid,
                "media_id": media_id,
                "version_number": local_version_number,
            }

    def get(
        self,
        media_id: int,
        version_number: int | None = None,
        include_content: bool = True,
    ) -> dict[str, Any] | None:
        db = self.session
        if not isinstance(media_id, int):
            raise TypeError("media_id must be an integer.")  # noqa: TRY003
        if version_number is not None and (not isinstance(version_number, int) or version_number < 1):
            raise ValueError("Version number must be a positive integer.")  # noqa: TRY003

        log_msg = f"Getting {'latest' if version_number is None else f'version {version_number}'} for media_id={media_id}"
        logger.debug("{} (active only) from DB: {}", log_msg, db.db_path_str)
        try:
            select_cols = [
                "dv.id",
                "dv.uuid",
                "dv.media_id",
                "dv.version_number",
                "dv.created_at",
                "dv.prompt",
                "dv.analysis_content",
                "dv.safe_metadata",
                "dv.last_modified",
                "dv.version",
                "dv.client_id",
                "dv.deleted",
            ]
            if include_content:
                select_cols.append("dv.content")
            select_clause = ", ".join(select_cols)
            params: list[Any] = [media_id]
            query_base = (
                "FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id "
                "WHERE dv.media_id = ? AND dv.deleted = 0 AND m.deleted = 0 "
                "AND m.system_operation_id IS NULL"
            )
            order_limit = ""
            if version_number is None:
                order_limit = "ORDER BY dv.version_number DESC LIMIT 1"
            else:
                query_base += " AND dv.version_number = ?"
                params.append(version_number)
            final_query = f"SELECT {select_clause} {query_base} {order_limit}"
            cursor = db.execute_query(final_query, tuple(params))
            result = cursor.fetchone()
            if not result:
                logger.warning(
                    "Active doc version {} not found for active media {}",
                    version_number or "latest",
                    media_id,
                )
                return None
            return dict(result)
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Error retrieving {} from DB '{}': {}", log_msg, db.db_path_str, exc, exc_info=True)
            raise DatabaseError(f"DB error retrieving version: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error retrieving {} from DB '{}': {}",
                log_msg,
                db.db_path_str,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error retrieving version: {exc}") from exc  # noqa: TRY003

    def list(
        self,
        media_id: int,
        *,
        include_content: bool = False,
        include_deleted: bool = False,
        limit: int | None = None,
        offset: int | None = 0,
    ) -> list[dict[str, Any]]:
        db = self.session
        if not isinstance(media_id, int):
            raise TypeError("media_id must be an integer.")  # noqa: TRY003
        if not isinstance(include_content, bool):
            raise TypeError("include_content must be a boolean.")  # noqa: TRY003
        if not isinstance(include_deleted, bool):
            raise TypeError("include_deleted must be a boolean.")  # noqa: TRY003
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("Limit must be a positive integer.")  # noqa: TRY003
        if offset is not None and (not isinstance(offset, int) or offset < 0):
            raise ValueError("Offset must be a non-negative integer.")  # noqa: TRY003

        logger.debug(
            "Listing {} document versions for media_id={} from DB: {}",
            "all" if include_deleted else "active",
            media_id,
            db.db_path_str,
        )
        try:
            select_cols_list = [
                "dv.id",
                "dv.uuid",
                "dv.media_id",
                "dv.version_number",
                "dv.created_at",
                "dv.prompt",
                "dv.analysis_content",
                "dv.safe_metadata",
                "dv.last_modified",
                "dv.version",
                "dv.client_id",
                "dv.deleted",
            ]
            if include_content:
                select_cols_list.append("dv.content")
            select_clause = ", ".join(select_cols_list)

            params: list[Any] = [media_id]
            where_conditions = [
                "dv.media_id = ?",
                "m.deleted = 0",
                "m.system_operation_id IS NULL",
            ]
            if not include_deleted:
                where_conditions.append("dv.deleted = 0")

            limit_offset_clause = ""
            if limit is not None:
                limit_offset_clause += " LIMIT ?"
                params.append(limit)
                if offset is not None and offset > 0:
                    limit_offset_clause += " OFFSET ?"
                    params.append(offset)

            query_parts = [
                f"SELECT {select_clause}",
                "FROM DocumentVersions dv",
                "JOIN Media m ON dv.media_id = m.id",
                f"WHERE {' AND '.join(where_conditions)}",
                "ORDER BY dv.version_number DESC",
            ]
            if limit_offset_clause:
                query_parts.append(limit_offset_clause)
            final_query = " ".join(query_parts)

            cursor = db.execute_query(final_query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            logger.error(
                "SQLite error retrieving versions for media_id {} from {}: {}",
                media_id,
                db.db_path_str,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to retrieve document versions: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error retrieving versions for media_id {} from {}: {}",
                media_id,
                db.db_path_str,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"An unexpected error occurred: {exc}") from exc  # noqa: TRY003

    def soft_delete(self, version_uuid: str) -> bool:
        if not version_uuid:
            raise InputError("Version UUID required.")  # noqa: TRY003

        db = self.session
        current_time = db._get_current_utc_timestamp_str()
        client_id = db.client_id
        logger.debug("Attempting soft delete DocVersion UUID: {}", version_uuid)
        try:
            with db.transaction() as conn:
                version_info = db._fetchone_with_connection(
                    conn,
                    "SELECT dv.id, dv.media_id, dv.version, m.uuid as media_uuid "
                    "FROM DocumentVersions dv "
                    "JOIN Media m ON dv.media_id = m.id "
                    "WHERE dv.uuid = ? AND dv.deleted = 0 "
                    "AND m.system_operation_id IS NULL",
                    (version_uuid,),
                )
                if not version_info:
                    logger.warning("DocVersion UUID {} not found/deleted.", version_uuid)
                    return False
                version_id = version_info["id"]
                media_id = version_info["media_id"]
                current_sync_version = version_info["version"]
                media_uuid = version_info["media_uuid"]
                new_sync_version = current_sync_version + 1

                active_row = db._fetchone_with_connection(
                    conn,
                    "SELECT COUNT(*) AS active_count FROM DocumentVersions WHERE media_id = ? AND deleted = 0",
                    (media_id,),
                )
                active_count = int((active_row or {}).get("active_count", 0))
                if active_count <= 1:
                    logger.warning("Cannot delete DocVersion UUID {} - last active.", version_uuid)
                    return False

                update_cursor = db._execute_with_connection(
                    conn,
                    "UPDATE DocumentVersions SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                    (current_time, new_sync_version, client_id, version_id, current_sync_version),
                )
                if getattr(update_cursor, "rowcount", 0) == 0:
                    raise ConflictError("DocumentVersions", version_id)  # noqa: TRY301

                delete_payload = {
                    "uuid": version_uuid,
                    "media_uuid": media_uuid,
                    "last_modified": current_time,
                    "version": new_sync_version,
                    "client_id": client_id,
                    "deleted": 1,
                }
                db._log_sync_event(
                    conn,
                    "DocumentVersions",
                    version_uuid,
                    "delete",
                    new_sync_version,
                    delete_payload,
                )
                logger.info(
                    "Soft deleted DocVersion UUID {}. New ver: {}",
                    version_uuid,
                    new_sync_version,
                )
                return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
            logger.error("Error soft delete DocVersion UUID {}: {}", version_uuid, exc, exc_info=True)
            if isinstance(exc, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed soft delete doc version: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected soft delete DocVersion error UUID {}: {}",
                version_uuid,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected version soft delete error: {exc}") from exc  # noqa: TRY003
