from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
    normalize_media_dedupe_url,
)
from tldw_Server_API.app.core.DB_Management.media_db.email_identity import email_identity_url
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.collections import (
    load_collections_database_cls,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike
from tldw_Server_API.app.core.Ingestion_Media_Processing.Email.email_ingestion_metrics import (
    record_email_dedupe,
    track_email_persistence,
)


def _load_legacy_media_support():
    return (
        load_collections_database_cls(),
        MEDIA_NONCRITICAL_EXCEPTIONS,
        media_dedupe_url_candidates,
        normalize_media_dedupe_url,
    )


class MediaRepository:
    """Caller-facing seam for media persistence while internals migrate out of the shim."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> MediaRepository:
        return cls(session=db)

    @track_email_persistence
    def add_media_with_keywords(
        self,
        *,
        url: str | None = None,
        title: str | None = None,
        media_type: str | None = None,
        content: str | None = None,
        keywords: list[str] | None = None,
        prompt: str | None = None,
        analysis_content: str | None = None,
        safe_metadata: str | None = None,
        source_hash: str | None = None,
        transcription_model: str | None = None,
        author: str | None = None,
        ingestion_date: str | None = None,
        overwrite: bool = False,
        chunk_options: dict[str, Any] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        visibility: str | None = None,
        owner_user_id: int | None = None,
    ) -> tuple[int | None, str | None, str]:
        """Add or update media while the legacy API migrates behind repository seams.

        This intentionally centralizes the transitional compatibility logic in one
        place while callers migrate off the legacy shim and onto repository
        helpers.
        """
        db = self.session
        (
            _collections_db_cls,
            noncritical_exceptions,
            media_dedupe_url_candidates,
            normalize_media_dedupe_url,
        ) = _load_legacy_media_support()

        if content is None:
            raise InputError("Content cannot be None.")  # noqa: TRY003

        title = title or "Untitled"
        media_type = media_type or "unknown"
        keywords_norm = [k.strip().lower() for k in keywords or [] if k and k.strip()]

        valid_visibilities = ("personal", "team", "org")
        requested_visibility = visibility if visibility else None
        if requested_visibility is not None and requested_visibility not in valid_visibilities:
            raise InputError(  # noqa: TRY003
                f"Invalid visibility '{requested_visibility}'. Must be one of: {valid_visibilities}"
            )

        now = db._get_current_utc_timestamp_str()
        ingestion_date = ingestion_date or now
        client_id = db.client_id

        derived_owner_user_id = None
        if client_id is not None:
            try:
                derived_owner_user_id = int(client_id)
            except (TypeError, ValueError):
                derived_owner_user_id = None

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        source_hash_norm = None
        if source_hash is not None:
            source_hash_str = str(source_hash).strip()
            source_hash_norm = source_hash_str if source_hash_str else None
        owner_lookup_value = str(owner_user_id).strip() if owner_user_id is not None else None
        raw_url_input = str(url).strip() if url is not None else ""
        if raw_url_input:
            dedupe_url_candidates = media_dedupe_url_candidates(raw_url_input)
            url = normalize_media_dedupe_url(raw_url_input) or raw_url_input
        else:
            url = f"local://{media_type}/{content_hash}"
            dedupe_url_candidates = (url,)

        email_metadata: dict[str, Any] | None = None
        email_owner_lookup = str(owner_user_id if owner_user_id is not None else client_id)
        email_tenant = email_owner_lookup
        if media_type == "email" and owner_user_id is None:
            email_tenant = db._resolve_email_tenant_id()
        original_dedupe_candidates = dedupe_url_candidates
        if media_type == "email":
            try:
                decoded_metadata = json.loads(safe_metadata) if safe_metadata else {}
            except (TypeError, ValueError) as exc:
                raise InputError("Invalid email metadata JSON") from exc  # noqa: TRY003
            if not isinstance(decoded_metadata, dict):
                raise InputError("Email metadata must be an object")  # noqa: TRY003
            url, email_metadata = email_identity_url(
                metadata=decoded_metadata, source_url=raw_url_input, content_hash=content_hash, tenant_id=email_tenant
            )
            safe_metadata = json.dumps(email_metadata, ensure_ascii=False)
            dedupe_url_candidates = (url,)

        final_chunk_status = "completed" if chunks is not None else "pending"

        logger.info("Media persistence started")
        try:
            from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import (
                get_topic_monitoring_service,
            )

            mon = get_topic_monitoring_service()
            uid = str(client_id) if client_id is not None else None
            if title:
                mon.schedule_evaluate_and_alert(
                    user_id=uid,
                    text=title,
                    source="ingestion.media",
                    scope_type="user",
                    scope_id=uid,
                )
            if content:
                mon.schedule_evaluate_and_alert(
                    user_id=uid,
                    text=content,
                    source="ingestion.media",
                    scope_type="user",
                    scope_id=uid,
                )
            if analysis_content:
                mon.schedule_evaluate_and_alert(
                    user_id=uid,
                    text=analysis_content,
                    source="ingestion.media",
                    scope_type="user",
                    scope_id=uid,
                )
        except noncritical_exceptions as exc:
            logger.bind(error_type=type(exc).__name__[:80]).warning("Topic monitoring unavailable during media ingestion")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.bind(error_type=type(exc).__name__[:80]).warning("Topic monitoring failed during media ingestion")

        def _media_payload(
            uuid_: str,
            version_: int,
            *,
            chunk_status: str,
            source_hash: str | None,
            org_id: int | None,
            team_id: int | None,
            visibility: str,
            owner_user_id: int | None,
        ) -> dict[str, Any]:
            bool_false = False if db.backend_type == BackendType.POSTGRESQL else 0
            return {
                "url": url,
                "title": title,
                "type": media_type,
                "content": content,
                "author": author,
                "ingestion_date": ingestion_date,
                "transcription_model": transcription_model,
                "content_hash": content_hash,
                "source_hash": source_hash,
                "is_trash": bool_false,
                "trash_date": None,
                "chunking_status": chunk_status,
                "vector_processing": 0,
                "uuid": uuid_,
                "last_modified": now,
                "version": version_,
                "org_id": org_id,
                "team_id": team_id,
                "visibility": visibility,
                "owner_user_id": owner_user_id,
                "client_id": client_id,
                "deleted": bool_false,
            }

        def _persist_chunks(cnx: sqlite3.Connection, media_id: int) -> None:
            if chunks is None:
                return

            if overwrite:
                db._execute_with_connection(
                    cnx,
                    "DELETE FROM UnvectorizedMediaChunks WHERE media_id = ?",
                    (media_id,),
                )

            if not chunks:
                return

            created = db._get_current_utc_timestamp_str()
            for idx, ch in enumerate(chunks):
                if not isinstance(ch, dict) or ch.get("text") is None:
                    logger.warning("Skipping invalid chunk index {} for media_id {}", idx, media_id)
                    continue

                chunk_uuid = db._generate_uuid()
                bool_false = False if db.backend_type == BackendType.POSTGRESQL else 0
                db._execute_with_connection(
                    cnx,
                    """INSERT INTO UnvectorizedMediaChunks (media_id, chunk_text, chunk_index, start_char, end_char,
                                                            chunk_type, creation_date, last_modified_orig, is_processed,
                                                            metadata, uuid, last_modified, version, client_id, deleted,
                                                            prev_version, merge_parent_uuid)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        media_id,
                        ch["text"],
                        idx,
                        ch.get("start_char"),
                        ch.get("end_char"),
                        ch.get("chunk_type"),
                        created,
                        created,
                        False,
                        json.dumps(ch.get("metadata")) if isinstance(ch.get("metadata"), dict) else None,
                        chunk_uuid,
                        created,
                        1,
                        client_id,
                        bool_false,
                        None,
                        None,
                    ),
                )
                db._log_sync_event(
                    cnx,
                    "UnvectorizedMediaChunks",
                    chunk_uuid,
                    "create",
                    1,
                    {
                        **ch,
                        "media_id": media_id,
                        "uuid": chunk_uuid,
                        "chunk_index": idx,
                        "creation_date": created,
                        "last_modified": created,
                        "version": 1,
                        "client_id": client_id,
                        "deleted": bool_false,
                    },
                )

        try:
            with db.transaction() as conn:
                def _exec(query: str, params: tuple | list | dict | None = None):
                    return db._execute_with_connection(conn, query, params)

                def _fetchone(query: str, params: tuple | list | dict | None = None):
                    return db._fetchone_with_connection(conn, query, params)

                def _fetch_existing_by_url(select_columns: str):
                    if email_metadata is not None:
                        selected = ", ".join(f"m.{column.strip()}" for column in select_columns.split(","))
                        email = email_metadata.get("email")
                        email = email if isinstance(email, dict) else {}
                        # Reuse normalized identities created before identity URLs,
                        # including the RFC secondary collision key for providers.
                        identity_fields = (
                            ("source_message_id", email.get("source_message_id") or email.get("id") or email_metadata.get("source_message_id")),
                            ("message_id", email.get("message_id") or email_metadata.get("message_id")),
                        )
                        if db.backend_type == BackendType.POSTGRESQL:
                            # Fixed column/identity branches preserve URL, provider
                            # and RFC precedence in one bound database round trip.
                            branches = [
                                f"(SELECT {selected}, 0 AS identity_priority FROM Media m WHERE m.url = ? "  # nosec B608
                                "AND m.type = 'email' AND m.deleted = 0 AND m.system_operation_id IS NULL "
                                "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ? LIMIT 1)"
                            ]
                            identity_params = [url, email_owner_lookup]
                            for priority, (field, value) in enumerate(identity_fields, start=1):
                                if not value:
                                    continue
                                branches.append(
                                    f"(SELECT {selected}, {priority} AS identity_priority FROM Media m "  # nosec B608
                                    "JOIN email_messages e ON e.media_id = m.id "
                                    "JOIN email_sources s ON s.id = e.source_id AND s.tenant_id = e.tenant_id "
                                    "WHERE e.tenant_id = ? AND s.provider = ? AND s.source_key = ? "
                                    f"AND e.{field} = ? AND m.type = 'email' AND m.deleted = 0 "
                                    "AND m.system_operation_id IS NULL "
                                    "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ? LIMIT 1)"
                                )
                                identity_params.extend(
                                    (email_tenant, email_metadata["email_source_provider"], email_metadata["source_key"],
                                     str(value).strip(), email_owner_lookup)
                                )
                            row = _fetchone(
                                f"SELECT {select_columns} FROM (" + " UNION ALL ".join(branches)  # nosec B608
                                + ") AS identity_candidates ORDER BY identity_priority LIMIT 1",
                                tuple(identity_params),
                            )
                            if row:
                                return row
                        else:
                            row = _fetchone(
                                f"SELECT {selected} FROM Media m WHERE m.url = ? "  # nosec B608
                                "AND m.type = 'email' AND m.deleted = 0 AND m.system_operation_id IS NULL "
                                "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ? LIMIT 1",
                                (url, email_owner_lookup),
                            )
                            if row:
                                return row
                            for field, value in identity_fields:
                                if not value:
                                    continue
                                row = _fetchone(
                                    f"SELECT {selected} FROM Media m "  # nosec B608
                                    "JOIN email_messages e ON e.media_id = m.id "
                                    "JOIN email_sources s ON s.id = e.source_id AND s.tenant_id = e.tenant_id "
                                    "WHERE e.tenant_id = ? AND s.provider = ? AND s.source_key = ? "
                                    f"AND e.{field} = ? AND m.type = 'email' AND m.deleted = 0 "
                                    "AND m.system_operation_id IS NULL "
                                    "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ? LIMIT 1",
                                    (email_tenant, email_metadata["email_source_provider"], email_metadata["source_key"],
                                     str(value).strip(), email_owner_lookup),
                                )
                                if row:
                                    return row
                        # Legacy-only imports may have no normalized row. Inspect
                        # only the original URL's candidates, never a mailbox scan.
                        for old_url in original_dedupe_candidates:
                            candidates = db._fetchall_with_connection(
                                conn,
                                f"SELECT {selected}, m.content_hash AS identity_content_hash, "  # nosec B608
                                "m.url AS identity_source_url, (SELECT dv.safe_metadata FROM DocumentVersions dv "
                                "WHERE dv.media_id = m.id AND dv.deleted = 0 "
                                "ORDER BY dv.version_number DESC LIMIT 1) AS identity_metadata "
                                "FROM Media m WHERE m.url = ? AND m.type = 'email' AND m.deleted = 0 "
                                "AND m.system_operation_id IS NULL "
                                "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?",
                                (old_url, email_owner_lookup),
                            )
                            for candidate in candidates:
                                try:
                                    old_metadata = json.loads(candidate["identity_metadata"] or "{}")
                                except (TypeError, ValueError):
                                    continue
                                if not isinstance(old_metadata, dict):
                                    continue
                                old_identity, _ = email_identity_url(
                                    metadata=old_metadata, source_url=candidate["identity_source_url"],
                                    content_hash=candidate["identity_content_hash"], tenant_id=email_tenant,
                                )
                                if old_identity == url:
                                    return candidate
                        return None
                    if len(dedupe_url_candidates) == 1:
                        return _fetchone(
                            f"SELECT {select_columns} "  # nosec B608
                            "FROM Media WHERE url = ? AND deleted = 0 "
                            "AND system_operation_id IS NULL LIMIT 1",
                            (dedupe_url_candidates[0],),
                        )
                    placeholders = ", ".join(["?"] * len(dedupe_url_candidates))
                    return _fetchone(
                        f"SELECT {select_columns} "  # nosec B608
                        f"FROM Media WHERE url IN ({placeholders}) AND deleted = 0 "
                        "AND system_operation_id IS NULL "
                        "ORDER BY last_modified DESC, id DESC LIMIT 1",
                        tuple(dedupe_url_candidates),
                    )

                row = _fetch_existing_by_url(
                    "id, uuid, version, url, content_hash, source_hash, visibility, owner_user_id, org_id, team_id"
                )

                if not row and media_type != "email":
                    if owner_lookup_value:
                        row = _fetchone(
                            "SELECT id, uuid, version, url, content_hash, source_hash, visibility, owner_user_id, org_id, team_id "
                            "FROM Media WHERE content_hash = ? AND deleted = 0 AND type != 'email' "
                            "AND system_operation_id IS NULL "
                            "AND COALESCE(CAST(owner_user_id AS TEXT), client_id) = ? LIMIT 1",
                            (content_hash, owner_lookup_value),
                        )
                    else:
                        row = _fetchone(
                            "SELECT id, uuid, version, url, content_hash, source_hash, visibility, owner_user_id, org_id, team_id "
                            "FROM Media WHERE content_hash = ? AND deleted = 0 AND type != 'email' "
                            "AND system_operation_id IS NULL LIMIT 1",
                            (content_hash,),
                        )

                if row:
                    if media_type == "email":
                        record_email_dedupe(backend=db.backend_type)
                    media_id = row["id"]
                    media_uuid = row["uuid"]
                    current_ver = row["version"]
                    existing_url = row["url"]
                    existing_hash = row["content_hash"]
                    existing_source_hash = row.get("source_hash")
                    existing_visibility = row.get("visibility") or "personal"
                    existing_owner_user_id = row.get("owner_user_id")
                    existing_org_id = row.get("org_id")
                    existing_team_id = row.get("team_id")

                    if overwrite:
                        if content_hash == existing_hash:
                            logger.info(
                                f"Media content for ID {media_id} is identical. Updating metadata/chunks only."
                            )

                            def _update_latest_document_version_metadata() -> bool:
                                latest_version = _fetchone(
                                    "SELECT id, uuid, version FROM DocumentVersions "
                                    "WHERE media_id = ? AND deleted = 0 "
                                    "ORDER BY version_number DESC LIMIT 1",
                                    (media_id,),
                                )
                                if not latest_version:
                                    return False

                                update_fields: list[str] = []
                                update_params: list[Any] = []
                                payload_updates: dict[str, Any] = {"last_modified": now}

                                if prompt is not None:
                                    update_fields.append("prompt = ?")
                                    update_params.append(prompt)
                                    payload_updates["prompt"] = prompt
                                if analysis_content is not None:
                                    update_fields.append("analysis_content = ?")
                                    update_params.append(analysis_content)
                                    payload_updates["analysis_content"] = analysis_content
                                if safe_metadata is not None:
                                    update_fields.append("safe_metadata = ?")
                                    update_params.append(safe_metadata)
                                    payload_updates["safe_metadata"] = safe_metadata

                                if not update_fields:
                                    return False

                                current_doc_version = int(latest_version.get("version") or 0)
                                new_doc_version = current_doc_version + 1
                                update_fields.extend(
                                    ["last_modified = ?", "version = ?", "client_id = ?"]
                                )
                                update_params.extend([now, new_doc_version, client_id])
                                update_sql = (
                                    f"UPDATE DocumentVersions SET {', '.join(update_fields)} "  # nosec B608
                                    "WHERE id = ? AND version = ? "
                                    "AND EXISTS (SELECT 1 FROM Media m "
                                    "WHERE m.id = DocumentVersions.media_id "
                                    "AND m.system_operation_id IS NULL)"
                                )
                                update_params.extend([latest_version["id"], current_doc_version])
                                update_cursor = _exec(update_sql, tuple(update_params))
                                if update_cursor.rowcount == 0:
                                    raise ConflictError(  # noqa: TRY003, TRY301
                                        f"DocumentVersions (identical content metadata update for media id={media_id})",
                                        media_id,
                                    )

                                db._log_sync_event(
                                    conn,
                                    "DocumentVersions",
                                    latest_version["uuid"],
                                    "update",
                                    new_doc_version,
                                    payload_updates,
                                )
                                return True

                            _update_latest_document_version_metadata()
                            db.update_keywords_for_media(media_id, keywords_norm, conn=conn)
                            _persist_chunks(conn, media_id)

                            source_hash_update_needed = (
                                source_hash_norm is not None and source_hash_norm != existing_source_hash
                            )
                            chunk_status_update_needed = chunks is not None
                            if chunk_status_update_needed or source_hash_update_needed:
                                logger.info(
                                    f"Updating media metadata for identical content id={media_id}."
                                )
                                new_ver = current_ver + 1
                                update_fields = []
                                update_params: list[Any] = []
                                payload_updates: dict[str, Any] = {"last_modified": now}
                                if chunk_status_update_needed:
                                    update_fields.append("chunking_status = ?")
                                    update_params.append("completed")
                                    payload_updates["chunking_status"] = "completed"
                                if source_hash_update_needed:
                                    update_fields.append("source_hash = ?")
                                    update_params.append(source_hash_norm)
                                    payload_updates["source_hash"] = source_hash_norm
                                update_fields.append("version = ?")
                                update_params.append(new_ver)
                                update_fields.append("last_modified = ?")
                                update_params.append(now)
                                update_sql = (
                                    f"UPDATE Media SET {', '.join(update_fields)} "  # nosec B608
                                    "WHERE id = ? AND version = ?"
                                )
                                update_params.extend([media_id, current_ver])
                                update_cursor = _exec(update_sql, tuple(update_params))
                                if update_cursor.rowcount == 0:
                                    raise ConflictError(  # noqa: TRY003, TRY301
                                        f"Media (updating metadata for identical content id={media_id})",
                                        media_id,
                                    )

                                db._log_sync_event(
                                    conn,
                                    "Media",
                                    media_uuid,
                                    "update",
                                    new_ver,
                                    payload_updates,
                                )

                            return media_id, media_uuid, f"Media '{title}' is already up-to-date."

                        new_ver = current_ver + 1
                        effective_visibility = (
                            requested_visibility if requested_visibility is not None else existing_visibility
                        )
                        effective_owner_user_id = (
                            owner_user_id if owner_user_id is not None else existing_owner_user_id
                        )
                        effective_source_hash = (
                            source_hash_norm if source_hash_norm is not None else existing_source_hash
                        )
                        payload = _media_payload(
                            media_uuid,
                            new_ver,
                            chunk_status=final_chunk_status,
                            source_hash=effective_source_hash,
                            org_id=existing_org_id,
                            team_id=existing_team_id,
                            visibility=effective_visibility,
                            owner_user_id=effective_owner_user_id,
                        )
                        update_sql = """
                            UPDATE Media
                               SET url = ?, title = ?, type = ?, content = ?, author = ?,
                                   ingestion_date = ?, transcription_model = ?,
                                   content_hash = ?, source_hash = ?, is_trash = ?, trash_date = ?,
                                   chunking_status = ?, vector_processing = ?,
                                   last_modified = ?, version = ?, org_id = ?, team_id = ?,
                                   visibility = ?, owner_user_id = ?, client_id = ?, deleted = ?
                               WHERE id = ? AND version = ?
                                 AND system_operation_id IS NULL
                        """
                        update_params = (
                            payload["url"],
                            payload["title"],
                            payload["type"],
                            payload["content"],
                            payload["author"],
                            payload["ingestion_date"],
                            payload["transcription_model"],
                            payload["content_hash"],
                            payload["source_hash"],
                            payload["is_trash"],
                            payload["trash_date"],
                            payload["chunking_status"],
                            payload["vector_processing"],
                            payload["last_modified"],
                            payload["version"],
                            payload["org_id"],
                            payload["team_id"],
                            payload["visibility"],
                            payload["owner_user_id"],
                            payload["client_id"],
                            payload["deleted"],
                            media_id,
                            current_ver,
                        )
                        update_cursor = _exec(update_sql, update_params)
                        if update_cursor.rowcount == 0:
                            raise ConflictError(  # noqa: TRY003, TRY301
                                f"Media (full update id={media_id})",
                                media_id,
                            )

                        db._log_sync_event(conn, "Media", media_uuid, "update", new_ver, payload)
                        db._update_fts_media(conn, media_id, payload["title"], payload["content"])
                        db.update_keywords_for_media(media_id, keywords_norm, conn=conn)
                        db.create_document_version(
                            media_id=media_id,
                            content=content,
                            prompt=prompt,
                            analysis_content=analysis_content,
                            safe_metadata=safe_metadata,
                        )
                        _persist_chunks(conn, media_id)
                        # This table is part of the Media schema. Reuse the write
                        # connection: initializing Collections here can deadlock
                        # SQLite through a second connection, including in nested
                        # transactions. Highlight state must roll back with content.
                        _exec(
                            "UPDATE reading_highlights SET state = 'stale' "
                            "WHERE user_id = ? AND item_id = ? "
                            "AND content_hash_ref IS NOT NULL AND content_hash_ref <> ?",
                            (str(client_id), media_id, content_hash),
                        )
                        try:
                            from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
                                invalidate_intra_doc_vectors,
                            )

                            invalidate_intra_doc_vectors(str(media_id))
                        except noncritical_exceptions:
                            pass
                        return media_id, media_uuid, f"Media '{title}' updated to new version."

                    is_canonicalisation = (
                        existing_url.startswith("local://")
                        and not url.startswith("local://")
                        and content_hash == existing_hash
                    )
                    if is_canonicalisation:
                        logger.info(f"Canonicalizing URL for media_id {media_id} to {url}")
                        new_ver = current_ver + 1
                        canon_cursor = _exec(
                            "UPDATE Media SET url = ?, last_modified = ?, version = ?, "
                            "client_id = ? WHERE id = ? AND version = ? "
                            "AND system_operation_id IS NULL",
                            (url, now, new_ver, client_id, media_id, current_ver),
                        )
                        if canon_cursor.rowcount == 0:
                            raise ConflictError(  # noqa: TRY003, TRY301
                                f"Media (canonicalization id={media_id})",
                                media_id,
                            )

                        db._log_sync_event(
                            conn,
                            "Media",
                            media_uuid,
                            "update",
                            new_ver,
                            {"url": url, "last_modified": now},
                        )
                        return media_id, media_uuid, f"Media '{title}' URL canonicalized."

                    try:
                        new_ver = current_ver + 1
                        touch_cursor = _exec(
                            "UPDATE Media SET last_modified = ?, version = ?, client_id = ? "
                            "WHERE id = ? AND version = ? AND system_operation_id IS NULL",
                            (now, new_ver, client_id, media_id, current_ver),
                        )
                        if touch_cursor.rowcount == 1:
                            db._log_sync_event(
                                conn,
                                "Media",
                                media_uuid,
                                "update",
                                new_ver,
                                {"last_modified": now, "touched": True},
                            )
                        else:
                            logger.debug(
                                f"No rows updated when touching media {media_id}; possible version change."
                            )
                    except noncritical_exceptions as touch_error:
                        logger.debug(f"Non-fatal: failed to touch media {media_id}: {touch_error}")

                    return media_id, media_uuid, f"Media '{title}' already exists. Overwrite not enabled."

                with db._media_insert_lock:
                    recheck_row = _fetch_existing_by_url("id, uuid, version")
                    if recheck_row:
                        if media_type == "email":
                            record_email_dedupe(backend=db.backend_type)
                        media_id = recheck_row["id"]
                        media_uuid = recheck_row["uuid"]
                        if not overwrite:
                            return (
                                media_id,
                                media_uuid,
                                f"Media '{title}' already exists (concurrent insert). Overwrite not enabled.",
                            )
                        return media_id, media_uuid, f"Media '{title}' already exists (handled concurrent insert)."

                    media_uuid = db._generate_uuid()
                    scope_org_id, scope_team_id = db._resolve_scope_ids()
                    payload = _media_payload(
                        media_uuid,
                        1,
                        chunk_status=final_chunk_status,
                        source_hash=source_hash_norm,
                        org_id=scope_org_id,
                        team_id=scope_team_id,
                        visibility=requested_visibility or "personal",
                        owner_user_id=owner_user_id
                        if owner_user_id is not None
                        else derived_owner_user_id,
                    )
                    insert_sql = """
                        INSERT INTO Media (url, title, type, content, author, ingestion_date,
                                           transcription_model, content_hash, source_hash, is_trash, trash_date,
                                           chunking_status, vector_processing, uuid, last_modified,
                                           version, org_id, team_id, visibility, owner_user_id,
                                           client_id, deleted)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    insert_params = (
                        payload["url"],
                        payload["title"],
                        payload["type"],
                        payload["content"],
                        payload["author"],
                        payload["ingestion_date"],
                        payload["transcription_model"],
                        payload["content_hash"],
                        payload["source_hash"],
                        payload["is_trash"],
                        payload["trash_date"],
                        payload["chunking_status"],
                        payload["vector_processing"],
                        payload["uuid"],
                        payload["last_modified"],
                        payload["version"],
                        payload["org_id"],
                        payload["team_id"],
                        payload["visibility"],
                        payload["owner_user_id"],
                        payload["client_id"],
                        payload["deleted"],
                    )
                    if db.backend_type == BackendType.POSTGRESQL:
                        insert_sql += " RETURNING id"

                    insert_cursor = _exec(insert_sql, insert_params)
                    if db.backend_type == BackendType.POSTGRESQL:
                        inserted_row = insert_cursor.fetchone()
                        media_id = inserted_row["id"] if inserted_row else None
                    else:
                        media_id = insert_cursor.lastrowid
                    if not media_id:
                        raise DatabaseError("Failed to obtain new media ID.")  # noqa: TRY003

                db._log_sync_event(conn, "Media", media_uuid, "create", 1, payload)
                db._update_fts_media(conn, media_id, payload["title"], payload["content"])
                # A newly inserted row cannot have keyword links yet. Keep
                # replacements and nonempty creation on the existing path.
                if db.backend_type != BackendType.POSTGRESQL or keywords_norm:
                    db.update_keywords_for_media(media_id, keywords_norm, conn=conn)
                db.create_document_version(
                    media_id=media_id,
                    content=content,
                    prompt=prompt,
                    analysis_content=analysis_content,
                    safe_metadata=safe_metadata,
                )
                _persist_chunks(conn, media_id)

                try:
                    from tldw_Server_API.app.core.config import rag_enable_structure_index

                    enable_si = rag_enable_structure_index()
                except noncritical_exceptions:
                    enable_si = True
                if enable_si and chunks:
                    try:
                        def _normalize_path(path_value: Any) -> list[str]:
                            if path_value is None:
                                return []
                            if isinstance(path_value, str):
                                raw = path_value.strip()
                                if not raw:
                                    return []
                                if " > " in raw:
                                    parts = [p.strip() for p in raw.split(" > ")]
                                elif "/" in raw:
                                    parts = [p.strip() for p in raw.split("/")]
                                else:
                                    parts = [raw]
                            elif isinstance(path_value, (list, tuple)):
                                parts = [str(p).strip() for p in path_value if str(p).strip()]
                            else:
                                parts = [str(path_value).strip()]
                            return [p for p in parts if p]

                        def _path_key(parts: list[str]) -> str:
                            return " / ".join(parts)

                        def _row_get(row: Any, key: str, default: Any = None) -> Any:
                            if isinstance(row, dict):
                                return row.get(key, default)
                            try:
                                return row[key]
                            except noncritical_exceptions:
                                return default

                        sections_agg: dict[str, dict[str, Any]] = {}
                        order = 0
                        for ch in chunks:
                            md = ch.get("metadata") or {}
                            start = ch.get("start_char")
                            end = ch.get("end_char")
                            if start is None or end is None:
                                continue
                            path_parts = _normalize_path(
                                md.get("section_path") or md.get("ancestry_titles")
                            )
                            if not path_parts:
                                continue

                            for depth in range(1, len(path_parts) + 1):
                                current_parts = path_parts[:depth]
                                key = _path_key(current_parts)
                                rec = sections_agg.get(key)
                                if rec is None:
                                    parent_parts = current_parts[:-1]
                                    sections_agg[key] = {
                                        "kind": "section",
                                        "level": depth,
                                        "title": str(current_parts[-1]),
                                        "start_char": int(start),
                                        "end_char": int(end),
                                        "order_index": order,
                                        "path": key,
                                        "parent_path": _path_key(parent_parts) if parent_parts else None,
                                    }
                                    order += 1
                                else:
                                    rec["start_char"] = min(int(start), int(rec["start_char"]))
                                    rec["end_char"] = max(int(end), int(rec["end_char"]))

                        section_ids_by_range: list[dict[str, Any]] = []
                        if sections_agg:
                            section_records = sorted(
                                sections_agg.values(),
                                key=lambda r: (
                                    int(r.get("level") or 0),
                                    int(r.get("order_index") or 0),
                                ),
                            )
                            db._write_structure_index_records(conn, media_id, section_records)

                            bool_false = False if db.backend_type == BackendType.POSTGRESQL else 0
                            try:
                                cur = db._execute_with_connection(
                                    conn,
                                    "SELECT id, start_char, end_char, path FROM DocumentStructureIndex "
                                    "WHERE media_id = ? AND deleted = ? AND kind IN ('section','header') "
                                    "ORDER BY COALESCE(level, 0) DESC, start_char DESC",
                                    (media_id, bool_false),
                                )
                                fetched = cur.fetchall() or []
                                section_ids_by_range = [
                                    {
                                        "id": int(_row_get(row, "id", 0) or 0),
                                        "start_char": int(_row_get(row, "start_char", 0) or 0),
                                        "end_char": int(_row_get(row, "end_char", 0) or 0),
                                        "path": str(_row_get(row, "path", "") or ""),
                                    }
                                    for row in fetched
                                    if _row_get(row, "id") is not None
                                ]
                            except noncritical_exceptions:
                                section_ids_by_range = []

                            try:
                                now_ts = db._get_current_utc_timestamp_str()
                                path_to_id = {
                                    str(r.get("path") or ""): int(r.get("id"))
                                    for r in section_ids_by_range
                                    if r.get("path")
                                }
                                for rec in section_records:
                                    parent_path = rec.get("parent_path")
                                    if not parent_path:
                                        continue
                                    child_id = path_to_id.get(str(rec.get("path") or ""))
                                    parent_id = path_to_id.get(str(parent_path))
                                    if not child_id or not parent_id or child_id == parent_id:
                                        continue
                                    db._execute_with_connection(
                                        conn,
                                        "UPDATE DocumentStructureIndex "
                                        "SET parent_id = ?, last_modified = ?, client_id = ? "
                                        "WHERE id = ?",
                                        (int(parent_id), now_ts, client_id, int(child_id)),
                                    )
                            except noncritical_exceptions as parent_error:
                                logger.warning(
                                    "Media structure population failed: parent-link; error_type={}", type(parent_error).__name__[:80]
                                )

                        try:
                            created = db._get_current_utc_timestamp_str()
                            bool_false = False if db.backend_type == BackendType.POSTGRESQL else 0
                            order = 0
                            for ch in chunks:
                                if not isinstance(ch, dict):
                                    continue
                                ctype = (ch.get("chunk_type") or "").lower()
                                if ctype not in ("text", "list", "code"):
                                    continue
                                s = ch.get("start_char")
                                e = ch.get("end_char")
                                if s is None or e is None:
                                    continue
                                parent_id = None
                                try:
                                    for row in section_ids_by_range:
                                        if int(row.get("start_char") or 0) <= int(s) < int(
                                            row.get("end_char") or 0
                                        ):
                                            parent_id = int(row.get("id"))
                                            break
                                except noncritical_exceptions:
                                    parent_id = None
                                if parent_id is None:
                                    try:
                                        cur2 = db._execute_with_connection(
                                            conn,
                                            "SELECT id FROM DocumentStructureIndex "
                                            "WHERE media_id = ? AND deleted = ? "
                                            "AND kind IN ('section','header') "
                                            "AND start_char <= ? AND end_char > ? "
                                            "ORDER BY COALESCE(level, 0) DESC, start_char DESC LIMIT 1",
                                            (media_id, bool_false, int(s), int(s)),
                                        )
                                        row2 = cur2.fetchone()
                                        parent_id = int(_row_get(row2, "id", 0) or 0) if row2 else None
                                    except noncritical_exceptions:
                                        parent_id = None

                                md = ch.get("metadata") or {}
                                paragraph_path_parts = _normalize_path(
                                    md.get("section_path") or md.get("ancestry_titles")
                                )
                                paragraph_path = (
                                    _path_key(paragraph_path_parts) if paragraph_path_parts else None
                                )
                                db._execute_with_connection(
                                    conn,
                                    """
                                    INSERT INTO DocumentStructureIndex (
                                        media_id, parent_id, kind, level, title, start_char, end_char,
                                        order_index, path, created_at, last_modified, version, client_id, deleted
                                    ) VALUES (?, ?, 'paragraph', NULL, NULL, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                                    """,
                                    (
                                        media_id,
                                        parent_id,
                                        int(s),
                                        int(e),
                                        order,
                                        paragraph_path,
                                        created,
                                        created,
                                        client_id,
                                        bool_false,
                                    ),
                                )
                                order += 1
                        except noncritical_exceptions as paragraph_error:
                            logger.warning(
                                "Media structure population failed: paragraph; error_type={}", type(paragraph_error).__name__[:80]
                            )
                    except noncritical_exceptions as structure_error:
                        logger.warning(
                            "Media structure population failed: structure; error_type={}", type(structure_error).__name__[:80]
                        )
                if chunk_options:
                    logger.info("Unused chunk options supplied")

                return media_id, media_uuid, f"Media '{title}' added."

        except (InputError, ConflictError, sqlite3.IntegrityError) as exc:
            logger.bind(error_type=type(exc).__name__[:80]).error("Media transaction failed; rolling back")
            raise

    def add_text_media(
        self,
        *,
        url: str | None = None,
        title: str | None = None,
        media_type: str | None = None,
        content: str | None = None,
        keywords: list[str] | None = None,
        prompt: str | None = None,
        analysis_content: str | None = None,
        safe_metadata: str | None = None,
        source_hash: str | None = None,
        transcription_model: str | None = None,
        author: str | None = None,
        ingestion_date: str | None = None,
        overwrite: bool = False,
        chunk_options: dict[str, Any] | None = None,
        chunks: list[dict[str, Any]] | None = None,
        visibility: str | None = None,
        owner_user_id: int | None = None,
    ) -> tuple[int | None, str | None, str]:
        return self.add_media_with_keywords(
            url=url,
            title=title,
            media_type=media_type,
            content=content,
            keywords=keywords,
            prompt=prompt,
            analysis_content=analysis_content,
            safe_metadata=safe_metadata,
            source_hash=source_hash,
            transcription_model=transcription_model,
            author=author,
            ingestion_date=ingestion_date,
            overwrite=overwrite,
            chunk_options=chunk_options,
            chunks=chunks,
            visibility=visibility,
            owner_user_id=owner_user_id,
        )
