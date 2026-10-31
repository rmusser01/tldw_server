from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike
from tldw_Server_API.app.core.Ingestion_Media_Processing.logging_safety import exception_type_for_log
from tldw_Server_API.app.core.Utils.keyword_identity import (
    dedupe_keywords,
    normalize_keyword_identity,
    rank_identity_matches,
)


class KeywordsRepository:
    """Repository for keyword rows and media-keyword links."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> KeywordsRepository:
        return cls(session=db)

    def fetch_for_media(self, media_id: int) -> list[str]:
        order_expr = self.session._keyword_order_expression("k.keyword")
        query = (
            f"SELECT k.keyword FROM Keywords k "  # nosec B608
            "JOIN MediaKeywords mk ON k.id = mk.keyword_id "
            "JOIN Media m ON mk.media_id = m.id "
            "WHERE mk.media_id = ? AND k.deleted = ? AND m.deleted = ? "
            "AND m.system_operation_id IS NULL "
            f"ORDER BY {order_expr}"
        )
        try:
            cursor = self.session.execute_query(query, (media_id, False, False))
            return [row["keyword"] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Keyword lookup failed (error_type={})", exception_type_for_log(exc))
            raise DatabaseError(f"Failed fetch keywords {media_id}") from exc  # noqa: TRY003

    def match_identities(
        self,
        conn: Any | None,
        keywords: Iterable[object],
        *,
        include_deleted: bool = True,
        all_matches: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Map each requested keyword's ``normalize_keyword_identity`` to its stored rows.

        Rows come active first, then oldest. One SQL lookup asks the NOCASE column
        (SQLite) or ``LOWER(keyword)`` (Postgres, indexed) for the requested texts and
        their identities, which finds every ASCII row: an ASCII row's casefold is its
        lowercase. Only rows with a non-ASCII character can still match by identity
        (ß/SS, É/é on SQLite, NFC forms), so the fallback reads just those, and only when
        an identity has no active hit or ``all_matches`` wants every legacy duplicate
        (TASK-13950.5). On Postgres an identity still without an active row takes the
        active row the unique ``LOWER(keyword)`` index holds for it: LOWER() folds some
        letters the identity keeps apart (``LOWER('İ')`` is ``i``; the identity keeps the
        dot), and the index treats them as one keyword. ``conn=None`` reads through
        ``execute_query``.
        """
        db = self.session
        terms = dedupe_keywords(keywords)
        wanted = {normalize_keyword_identity(term) for term in terms}
        if not wanted:
            return {}
        # A DB-like without backend_type is SQLite (see api._supports_keyword_delete_repository).
        postgres = getattr(db, "backend_type", None) == BackendType.POSTGRESQL
        values = [*terms, *(wanted - set(terms))]
        placeholders = ", ".join(["LOWER(?)" if postgres else "?"] * len(values))
        active = "" if include_deleted else (" AND deleted = FALSE" if postgres else " AND deleted = 0")
        select = "SELECT id, uuid, keyword, deleted, version FROM Keywords WHERE "
        match = f"{'LOWER(keyword)' if postgres else 'keyword'} IN ({placeholders})"
        rows = self._fetch_rows(conn, f"{select}{match}{active}", tuple(values))  # nosec B608
        groups = rank_identity_matches(wanted, rows)
        if all_matches or any(identity not in groups or groups[identity][0]["deleted"] for identity in wanted):
            # ponytail: reads every non-ASCII keyword row (a partial-index scan on Postgres, a table
            # scan on SQLite); add a stored identity column if non-ASCII vocabularies get large.
            non_ascii = "keyword ~ '[^[:ascii:]]'" if postgres else "keyword GLOB '*[^ -~]*'"
            rows += self._fetch_rows(conn, f"{select}{non_ascii}{active}")  # nosec B608
            groups = rank_identity_matches(wanted, rows)
        for term in terms if postgres else []:
            identity = normalize_keyword_identity(term)
            if identity not in groups or groups[identity][0]["deleted"]:
                holder = self.active_lower_match(conn, term)
                if holder:
                    groups[identity] = [holder, *groups.get(identity, [])]
        return groups

    def find(self, conn: Any | None, keyword: str, *, include_deleted: bool = True) -> dict[str, Any] | None:
        """Return the stored row for ``keyword``'s identity: active first, then oldest."""
        rows = self.match_identities(conn, [keyword], include_deleted=include_deleted).get(
            normalize_keyword_identity(keyword)
        )
        return rows[0] if rows else None

    def filter_clauses(
        self,
        must_have: Iterable[object] | None,
        must_not_have: Iterable[object] | None,
        media_column: str = "m.id",
    ) -> tuple[list[str], list[Any]]:
        """Return SQL predicates (and params) on ``media_column`` for keyword filters.

        Filters match by keyword identity over every active variant; a must-have keyword
        that does not exist matches nothing.
        """
        required = dedupe_keywords(must_have or [])
        excluded = dedupe_keywords(must_not_have or [])
        found = self.match_identities(None, [*required, *excluded], include_deleted=False, all_matches=True)
        clauses: list[str] = []
        params: list[Any] = []
        link = f"SELECT 1 FROM MediaKeywords mk WHERE mk.media_id = {media_column} AND mk.keyword_id IN"
        for keyword in required:
            ids = [row["id"] for row in found.get(normalize_keyword_identity(keyword), [])]
            if not ids:
                return ["1 = 0"], []
            clauses.append(f"EXISTS ({link} ({', '.join('?' * len(ids))}))")
            params.extend(ids)
        excluded_ids = [row["id"] for keyword in excluded for row in found.get(normalize_keyword_identity(keyword), [])]
        if excluded_ids:
            clauses.append(f"NOT EXISTS ({link} ({', '.join('?' * len(excluded_ids))}))")
            params.extend(excluded_ids)
        return clauses, params

    def _fetch_rows(self, conn: Any | None, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if conn is None:
            return [dict(row) for row in self.session.execute_query(query, params).fetchall()]
        return list(self.session._fetchall_with_connection(conn, query, params))

    def add(self, keyword: str, conn: Any | None = None) -> tuple[int | None, str | None]:
        """Return the keyword matching ``keyword``'s identity, restoring or creating it.

        A new keyword keeps the caller's display casing; an existing one keeps its own.
        """
        if not keyword or not keyword.strip():
            raise InputError("Keyword cannot be empty.")  # noqa: TRY003

        db = self.session
        keyword = keyword.strip()
        current_time = db._get_current_utc_timestamp_str()
        client_id = db.client_id
        postgres = db.backend_type == BackendType.POSTGRESQL

        try:
            if conn is not None:
                existing = self.find(conn, keyword)

                if existing:
                    kw_id = existing["id"]
                    kw_uuid = existing["uuid"]
                    is_deleted = existing["deleted"]
                    current_version = existing["version"]
                    if is_deleted:
                        new_version = current_version + 1
                        logger.info("Keyword restored (keyword_id={}, version={})", kw_id, new_version)
                        update_cursor = db._execute_with_connection(
                            conn,
                            "UPDATE Keywords SET deleted=0, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                            (current_time, new_version, client_id, kw_id, current_version),
                        )
                        if update_cursor.rowcount == 0:
                            raise ConflictError("Keywords", kw_id)  # noqa: TRY301

                        payload_data = db._fetchone_with_connection(
                            conn,
                            "SELECT * FROM Keywords WHERE id=?",
                            (kw_id,),
                        ) or {}
                        db._log_sync_event(conn, "Keywords", kw_uuid, "update", new_version, payload_data)
                        db._update_fts_keyword(conn, kw_id, existing["keyword"])
                        return kw_id, kw_uuid

                    logger.debug("Keyword '{}' already active.", keyword)
                    return kw_id, kw_uuid

                new_uuid = db._generate_uuid()
                new_version = 1
                logger.info("Adding keyword")
                insert_sql = (
                    "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id) "
                    "VALUES (?, ?, ?, ?, ?)"
                )
                if postgres:
                    # A concurrent add or a LOWER()-only fold may already hold the slot.
                    insert_sql += " ON CONFLICT DO NOTHING RETURNING id"

                insert_cursor = db._execute_with_connection(
                    conn,
                    insert_sql,
                    (keyword, new_uuid, current_time, new_version, client_id),
                )

                if postgres:
                    inserted_row = insert_cursor.fetchone()
                    if not inserted_row:
                        holder = self.active_lower_match(conn, keyword)
                        if not holder:
                            raise DatabaseError("Keyword insert conflicted with a row it cannot reuse.")  # noqa: TRY003, TRY301
                        return holder["id"], holder["uuid"]
                    kw_id = inserted_row["id"]
                else:
                    kw_id = insert_cursor.lastrowid

                if not kw_id:
                    raise DatabaseError("Failed to get last row ID for new keyword.")  # noqa: TRY003, TRY301

                payload_data = db._fetchone_with_connection(
                    conn,
                    "SELECT * FROM Keywords WHERE id=?",
                    (kw_id,),
                ) or {}
                db._log_sync_event(conn, "Keywords", new_uuid, "create", new_version, payload_data)
                db._update_fts_keyword(conn, kw_id, keyword)
                return kw_id, new_uuid

            with db.transaction() as tx_conn:
                return self.add(keyword, conn=tx_conn)
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
            logger.error("Keyword persistence failed (error_type={})", exception_type_for_log(exc))
            if isinstance(exc, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to add/update keyword: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error("Keyword persistence failed unexpectedly (error_type={})", exception_type_for_log(exc))
            raise DatabaseError(f"Unexpected error adding/updating keyword: {exc}") from exc  # noqa: TRY003

    def active_lower_match(self, conn: Any | None, keyword: str) -> dict[str, Any] | None:
        """Postgres: the active row holding ``keyword``'s slot in the unique LOWER(keyword) index."""
        rows = self._fetch_rows(
            conn,
            "SELECT id, uuid, keyword, deleted, version FROM Keywords "
            "WHERE LOWER(keyword) = LOWER(?) AND deleted = FALSE",
            (keyword,),
        )
        return rows[0] if rows else None

    def replace_keywords(
        self,
        media_id: int,
        keywords: list[str],
        conn: Any | None = None,
    ) -> bool:
        db = self.session
        valid_keywords = dedupe_keywords(keywords)
        if conn is None:
            with db.transaction() as tx_conn:
                return self.replace_keywords(media_id, keywords, conn=tx_conn)
        connection = conn

        try:
            media_info = db._fetchone_with_connection(
                connection,
                "SELECT uuid FROM Media WHERE id = ? AND deleted = 0 "
                "AND system_operation_id IS NULL",
                (media_id,),
            )
            if not media_info:
                raise InputError(f"Cannot update keywords: Media ID {media_id} not found or deleted.")  # noqa: TRY003, TRY301
            media_uuid = media_info["uuid"]

            current_rows = db._fetchall_with_connection(
                connection,
                "SELECT mk.keyword_id, k.uuid AS keyword_uuid "
                "FROM MediaKeywords mk JOIN Keywords k ON k.id = mk.keyword_id "
                "WHERE mk.media_id = ? AND k.deleted = 0",
                (media_id,),
            )
            current_links = {row["keyword_id"]: row["keyword_uuid"] for row in current_rows}
            current_keyword_ids = set(current_links.keys())

            target_keyword_data: dict[int, str] = {}
            for kw_text in valid_keywords:
                kw_id, kw_uuid = self.add(kw_text, conn=connection)
                if kw_id and kw_uuid:
                    target_keyword_data[kw_id] = kw_uuid
                    continue
                raise DatabaseError(f"Failed get/add keyword '{kw_text}'")  # noqa: TRY003, TRY301

            target_keyword_ids = set(target_keyword_data.keys())
            ids_to_add = target_keyword_ids - current_keyword_ids
            ids_to_remove = current_keyword_ids - target_keyword_ids
            link_sync_version = 1

            if ids_to_remove:
                remove_placeholders = ",".join("?" * len(ids_to_remove))
                params = (media_id, *list(ids_to_remove))
                db._execute_with_connection(
                    connection,
                    f"DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id IN ({remove_placeholders})",  # nosec B608
                    params,
                )
                for removed_id in ids_to_remove:
                    keyword_uuid = current_links.get(removed_id)
                    if keyword_uuid:
                        link_uuid = f"{media_uuid}_{keyword_uuid}"
                        payload = {"media_uuid": media_uuid, "keyword_uuid": keyword_uuid}
                        db._log_sync_event(
                            connection,
                            "MediaKeywords",
                            link_uuid,
                            "unlink",
                            link_sync_version,
                            payload,
                        )

            if ids_to_add:
                insert_params = [(media_id, keyword_id) for keyword_id in ids_to_add]
                insert_sql = "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)"
                if db.backend_type == BackendType.POSTGRESQL:
                    insert_sql += " ON CONFLICT DO NOTHING"
                db._executemany_with_connection(connection, insert_sql, insert_params)
                for added_id in ids_to_add:
                    keyword_uuid = target_keyword_data.get(added_id)
                    if keyword_uuid:
                        link_uuid = f"{media_uuid}_{keyword_uuid}"
                        payload = {"media_uuid": media_uuid, "keyword_uuid": keyword_uuid}
                        db._log_sync_event(
                            connection,
                            "MediaKeywords",
                            link_uuid,
                            "link",
                            link_sync_version,
                            payload,
                        )

            if ids_to_add or ids_to_remove:
                logger.debug(
                    "Keywords updated media {}. Added: {}, Removed: {}.",
                    media_id,
                    len(ids_to_add),
                    len(ids_to_remove),
                )
            else:
                logger.debug("No keyword changes media {}.", media_id)
            return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
            logger.error("Media keyword update failed (error_type={})", exception_type_for_log(exc))
            if isinstance(exc, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError(f"Keyword update failed: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error("Media keyword update failed unexpectedly (error_type={})", exception_type_for_log(exc))
            raise DatabaseError(f"Unexpected keyword update error: {exc}") from exc  # noqa: TRY003

    def soft_delete(self, keyword: str) -> bool:
        if not keyword or not keyword.strip():
            raise InputError("Keyword cannot be empty.")  # noqa: TRY003

        db = self.session
        current_time = db._get_current_utc_timestamp_str()
        client_id = db.client_id

        try:
            with db.transaction() as conn:
                keyword_info = self.find(conn, keyword, include_deleted=False)
                if not keyword_info:
                    logger.warning("Keyword deletion skipped: keyword unavailable")
                    return False

                keyword_id = keyword_info["id"]
                keyword_uuid = keyword_info["uuid"]
                current_version = keyword_info["version"]
                new_version = current_version + 1

                logger.info("Soft deleting keyword (keyword_id={}, version={})", keyword_id, new_version)
                update_cursor = db._execute_with_connection(
                    conn,
                    "UPDATE Keywords SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                    (current_time, new_version, client_id, keyword_id, current_version),
                )
                if getattr(update_cursor, "rowcount", 0) == 0:
                    raise ConflictError("Keywords", keyword_id)  # noqa: TRY301

                delete_payload = {
                    "uuid": keyword_uuid,
                    "last_modified": current_time,
                    "version": new_version,
                    "client_id": client_id,
                    "deleted": 1,
                }
                db._log_sync_event(conn, "Keywords", keyword_uuid, "delete", new_version, delete_payload)
                db._delete_fts_keyword(conn, keyword_id)

                linked_cursor = db._execute_with_connection(
                    conn,
                    "SELECT mk.media_id, m.uuid AS media_uuid FROM MediaKeywords mk "
                    "JOIN Media m ON mk.media_id = m.id WHERE mk.keyword_id = ? "
                    "AND m.deleted = 0 AND m.system_operation_id IS NULL",
                    (keyword_id,),
                )
                media_to_unlink = linked_cursor.fetchall()
                if media_to_unlink:
                    media_mappings: list[dict[str, Any]] = []
                    for record in media_to_unlink:
                        if isinstance(record, dict):
                            media_mappings.append(record)
                            continue
                        try:
                            media_mappings.append(dict(record))
                        except Exception:
                            try:
                                media_id_val = record[0]
                            except Exception:
                                media_id_val = None
                            media_uuid_val = None
                            with suppress(Exception):
                                media_uuid_val = record[1]
                            media_mappings.append(
                                {
                                    "media_id": media_id_val,
                                    "media_uuid": media_uuid_val,
                                }
                            )

                    media_ids = [
                        mapping["media_id"]
                        for mapping in media_mappings
                        if mapping.get("media_id") is not None
                    ]
                    if media_ids:
                        placeholders = ",".join("?" for _ in media_ids)
                        delete_cursor = db._execute_with_connection(
                            conn,
                            f"DELETE FROM MediaKeywords WHERE keyword_id = ? AND media_id IN ({placeholders})",  # nosec B608
                            (keyword_id, *media_ids),
                        )
                        deleted_link_count = getattr(delete_cursor, "rowcount", 0) or 0
                        unlink_version = 1
                        for mapping in media_mappings:
                            media_uuid_val = mapping.get("media_uuid")
                            link_uuid = f"{media_uuid_val}_{keyword_uuid}"
                            unlink_payload = {
                                "media_uuid": media_uuid_val,
                                "keyword_uuid": keyword_uuid,
                            }
                            db._log_sync_event(
                                conn,
                                "MediaKeywords",
                                link_uuid,
                                "unlink",
                                unlink_version,
                                unlink_payload,
                            )
                        logger.info("Keyword links removed (count={})", deleted_link_count)
                return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
            logger.error("Keyword soft deletion failed (error_type={})", exception_type_for_log(exc))
            if isinstance(exc, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed soft delete keyword: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error("Keyword soft deletion failed unexpectedly (error_type={})", exception_type_for_log(exc))
            raise DatabaseError(f"Unexpected soft delete keyword error: {exc}") from exc  # noqa: TRY003
