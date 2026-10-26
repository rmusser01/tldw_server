"""Package-owned keyword access helpers for Media DB."""

from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.repositories.keywords_repository import (
    KeywordsRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.Utils.keyword_identity import dedupe_keywords, normalize_keyword_identity

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def add_keyword(self, keyword: str, conn: Any | None = None) -> tuple[int | None, str | None]:
    return KeywordsRepository.from_legacy_db(self).add(keyword, conn=conn)


def fetch_media_for_keywords(
    self,
    keywords: list[str],
    include_trash: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Group active media by requested keyword, keyed by ``normalize_keyword_identity``.

    Keywords resolve in one batch through ``KeywordsRepository.match_identities``, so a
    request in any case (including ß/SS and accented capitals) finds every stored variant.
    """
    if not isinstance(keywords, list):
        raise TypeError("Input 'keywords' must be a list of strings.")  # noqa: TRY003

    requested = dedupe_keywords(keywords)
    if not requested:
        logger.debug("fetch_media_for_keywords: no valid keywords after stripping.")
        return {}

    results_by_keyword: dict[str, list[dict[str, Any]]] = {
        normalize_keyword_identity(keyword): [] for keyword in requested
    }

    media_conditions = ["m.deleted = ?", "m.system_operation_id IS NULL"]
    media_params: list[Any] = [False]
    if not include_trash:
        media_conditions.append("m.is_trash = ?")
        media_params.append(False)
    media_where_clause = " AND ".join(media_conditions)

    media_fields = (
        "m.id AS media_id, m.uuid AS media_uuid, m.title AS media_title, "
        "m.type AS media_type, m.url AS media_url, m.content_hash AS media_content_hash, "
        "m.last_modified AS media_last_modified, m.ingestion_date AS media_ingestion_date, "
        "m.author AS media_author"
    )

    try:
        conn = self.get_connection()
        matches = KeywordsRepository.from_legacy_db(self).match_identities(
            conn, requested, include_deleted=False, all_matches=True
        )
        identity_by_id = {int(row["id"]): identity for identity, rows in matches.items() for row in rows}
        if not identity_by_id:
            return results_by_keyword

        placeholders = ",".join("?" * len(identity_by_id))
        order_expr = self._keyword_order_expression("k.keyword")
        query = """
            SELECT
                k.id AS keyword_id,
                {media_fields}
            FROM Keywords k
            JOIN MediaKeywords mk ON k.id = mk.keyword_id
            JOIN Media m ON mk.media_id = m.id
            WHERE {media_where_clause}
              AND k.id IN ({placeholders})
            ORDER BY {order_expr}, m.last_modified DESC, m.id DESC
        """.format_map(locals())  # nosec B608
        params = tuple(media_params + list(identity_by_id))

        for row in self._fetchall_with_connection(conn, query, params):
            results_by_keyword[identity_by_id[int(row["keyword_id"])]].append(
                {
                    "id": row["media_id"],
                    "uuid": row["media_uuid"],
                    "title": row["media_title"],
                    "type": row["media_type"],
                    "url": row["media_url"],
                    "content_hash": row["media_content_hash"],
                    "last_modified": row["media_last_modified"],
                    "ingestion_date": row["media_ingestion_date"],
                    "author": row["media_author"],
                }
            )
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            "Unexpected error fetching media for keywords from DB {}: {}",
            self.db_path_str,
            exc,
            exc_info=True,
        )
        raise DatabaseError(
            f"An unexpected error occurred while fetching media for keywords: {exc}"
        ) from exc  # noqa: TRY003
    else:
        return results_by_keyword


__all__ = [
    "add_keyword",
    "fetch_media_for_keywords",
]
