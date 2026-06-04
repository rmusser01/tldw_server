"""FTS maintenance helpers for the package-native Media DB runtime."""

from __future__ import annotations

import os
from loguru import logger as logging
import sqlite3
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _update_fts_media(
    self: Any,
    conn: sqlite3.Connection,
    media_id: int,
    title: str,
    content: str | None,
    *,
    old_title: str | None = None,
    old_content: str | None = None,
):
    if self.backend_type == BackendType.SQLITE:
        content = content or ""
        try:
            from tldw_Server_API.app.core.RAG.rag_service.synonyms_registry import (
                get_corpus_synonyms,  # type: ignore
            )

            corpus = os.getenv("DEFAULT_FTS_CORPUS", "").strip() or None
            syn_map = get_corpus_synonyms(corpus)
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            syn_map = {}

        expanded_terms: list[str] = []
        if syn_map:
            try:
                import re as _re

                tokens = {t for t in _re.split(r"\W+", f"{title} {content}".lower()) if t}
                for token in tokens:
                    aliases = syn_map.get(token)
                    if aliases:
                        expanded_terms.extend(alias for alias in aliases if alias)
                max_terms = int(os.getenv("FTS_SYNONYM_EXPANSION_LIMIT", "200") or 200)
                if len(expanded_terms) > max_terms:
                    expanded_terms = expanded_terms[:max_terms]
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                expanded_terms = []

        expansion_suffix = (" " + " ".join(expanded_terms)) if expanded_terms else ""
        try:
            if old_title is not None or old_content is not None:
                conn.execute(
                    "INSERT INTO media_fts (media_fts, rowid, title, content) VALUES ('delete', ?, ?, ?)",
                    (media_id, old_title or "", old_content or ""),
                )
            conn.execute(
                "INSERT OR REPLACE INTO media_fts (rowid, title, content) VALUES (?, ?, ?)",
                (media_id, title, f"{content}{expansion_suffix}"),
            )
            logging.debug("Updated SQLite FTS entry for Media ID {}", media_id)
        except sqlite3.Error as exc:
            logging.error(
                "Failed to update media_fts for Media ID {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to update FTS for Media ID {media_id}: {exc}") from exc  # noqa: TRY003
        return

    if self.backend_type == BackendType.POSTGRESQL:
        try:
            from tldw_Server_API.app.core.testing import env_flag_enabled

            enable_synonyms = env_flag_enabled("PG_FTS_ENABLE_SYNONYMS")
            if enable_synonyms:
                try:
                    ensure_fn = getattr(self.backend, "ensure_synonyms_support", None)
                    if callable(ensure_fn):
                        ensure_fn(connection=conn)
                except _MEDIA_NONCRITICAL_EXCEPTIONS as syn_err:
                    logging.debug(f"Synonyms support ensure failed (non-fatal): {syn_err}")
            if enable_synonyms:
                sql = (
                    "UPDATE media SET media_fts_tsv = CASE "
                    "WHEN deleted IS FALSE THEN "
                    " setweight(to_tsvector('english', synonyms_expand(coalesce(title, ''))),'A') || "
                    " setweight(to_tsvector('english', synonyms_expand(coalesce(content, ''))),'C') "
                    "ELSE NULL END WHERE id = ?"
                )
            else:
                sql = (
                    "UPDATE media SET media_fts_tsv = CASE "
                    "WHEN deleted IS FALSE THEN "
                    " setweight(to_tsvector('english', coalesce(title, '')),'A') || "
                    " setweight(to_tsvector('english', coalesce(content, '')),'C') "
                    "ELSE NULL END WHERE id = ?"
                )
            self._execute_with_connection(conn, sql, (media_id,))
            logging.debug("Updated PostgreSQL FTS vector for Media ID {}", media_id)
        except DatabaseError as exc:
            logging.error(
                "Failed to update PostgreSQL FTS for Media ID {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise
        return


def _delete_fts_media(
    self: Any,
    conn: sqlite3.Connection,
    media_id: int,
):
    if self.backend_type == BackendType.SQLITE:
        try:
            conn.execute("DELETE FROM media_fts WHERE rowid = ?", (media_id,))
            logging.debug("Deleted SQLite FTS entry for Media ID {}", media_id)
        except sqlite3.Error as exc:
            logging.error(
                "Failed to delete from media_fts for Media ID {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to delete FTS for Media ID {media_id}: {exc}") from exc  # noqa: TRY003
        return

    if self.backend_type == BackendType.POSTGRESQL:
        try:
            self._execute_with_connection(
                conn,
                "UPDATE media SET media_fts_tsv = NULL WHERE id = ?",
                (media_id,),
            )
            logging.debug("Cleared PostgreSQL FTS vector for Media ID {}", media_id)
        except DatabaseError as exc:
            logging.error(
                "Failed to clear PostgreSQL FTS for Media ID {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise
        return


def _update_fts_keyword(
    self: Any,
    conn: sqlite3.Connection,
    keyword_id: int,
    keyword: str,
):
    if self.backend_type == BackendType.SQLITE:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO keyword_fts (rowid, keyword) VALUES (?, ?)",
                (keyword_id, keyword),
            )
            logging.debug("Updated SQLite FTS entry for Keyword ID {}", keyword_id)
        except sqlite3.Error as exc:
            logging.error(
                "Failed to update keyword_fts for Keyword ID {}: {}",
                keyword_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to update FTS for Keyword ID {keyword_id}: {exc}") from exc  # noqa: TRY003
        return

    if self.backend_type == BackendType.POSTGRESQL:
        try:
            self._execute_with_connection(
                conn,
                (
                    "UPDATE keywords SET keyword_fts_tsv = CASE "
                    "WHEN deleted IS FALSE THEN to_tsvector('english', coalesce(keyword, '')) "
                    "ELSE NULL END WHERE id = ?"
                ),
                (keyword_id,),
            )
            logging.debug("Updated PostgreSQL FTS vector for Keyword ID {}", keyword_id)
        except DatabaseError as exc:
            logging.error(
                "Failed to update PostgreSQL FTS for Keyword ID {}: {}",
                keyword_id,
                exc,
                exc_info=True,
            )
            raise
        return


def _delete_fts_keyword(
    self: Any,
    conn: sqlite3.Connection,
    keyword_id: int,
):
    if self.backend_type == BackendType.SQLITE:
        try:
            conn.execute("DELETE FROM keyword_fts WHERE rowid = ?", (keyword_id,))
            logging.debug("Deleted SQLite FTS entry for Keyword ID {}", keyword_id)
        except sqlite3.Error as exc:
            logging.error(
                "Failed to delete from keyword_fts for Keyword ID {}: {}",
                keyword_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to delete FTS for Keyword ID {keyword_id}: {exc}") from exc  # noqa: TRY003
        return

    if self.backend_type == BackendType.POSTGRESQL:
        try:
            self._execute_with_connection(
                conn,
                "UPDATE keywords SET keyword_fts_tsv = NULL WHERE id = ?",
                (keyword_id,),
            )
            logging.debug("Cleared PostgreSQL FTS vector for Keyword ID {}", keyword_id)
        except DatabaseError as exc:
            logging.error(
                "Failed to clear PostgreSQL FTS for Keyword ID {}: {}",
                keyword_id,
                exc,
                exc_info=True,
            )
            raise
        return


def sync_refresh_fts_for_entity(
    self: Any,
    conn,
    *,
    entity: str,
    entity_uuid: str,
    operation: str,
    payload: dict[str, Any] | None = None,
) -> None:
    payload = payload or {}

    if entity == "Media":
        row = self._fetchone_with_connection(
            conn,
            "SELECT id, title, content, deleted FROM Media WHERE uuid = ?",
            (entity_uuid,),
        )
        if not row:
            logging.warning(
                "sync_refresh_fts_for_entity: Media row not found for uuid={} operation={}",
                entity_uuid,
                operation,
            )
            return

        media_id = int(row["id"])
        if operation == "delete" or bool(row.get("deleted", 0)):
            self._delete_fts_media(conn, media_id)
            return

        if operation in {"create", "update"}:
            if operation == "update" and not any(
                key in payload for key in ("title", "content", "deleted")
            ):
                return
            self._update_fts_media(
                conn,
                media_id,
                str(row.get("title") or ""),
                row.get("content"),
            )
        return

    if entity == "Keywords":
        row = self._fetchone_with_connection(
            conn,
            "SELECT id, keyword, deleted FROM Keywords WHERE uuid = ?",
            (entity_uuid,),
        )
        if not row:
            logging.warning(
                "sync_refresh_fts_for_entity: Keyword row not found for uuid={} operation={}",
                entity_uuid,
                operation,
            )
            return

        keyword_id = int(row["id"])
        if operation == "delete" or bool(row.get("deleted", 0)):
            self._delete_fts_keyword(conn, keyword_id)
            return

        if operation in {"create", "update"}:
            if operation == "update" and not any(
                key in payload for key in ("keyword", "deleted")
            ):
                return
            self._update_fts_keyword(conn, keyword_id, str(row.get("keyword") or ""))
        return


__all__ = [
    "_delete_fts_keyword",
    "_delete_fts_media",
    "_update_fts_keyword",
    "_update_fts_media",
    "sync_refresh_fts_for_entity",
]
