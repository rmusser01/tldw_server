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

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def add_keyword(self, keyword: str, conn: Any | None = None) -> tuple[int | None, str | None]:
    return KeywordsRepository.from_legacy_db(self).add(keyword, conn=conn)


def fetch_media_for_keywords(
    self,
    keywords: list[str],
    include_trash: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(keywords, list):
        raise TypeError("Input 'keywords' must be a list of strings.")  # noqa: TRY003

    if not keywords:
        logger.debug("fetch_media_for_keywords called with an empty list of keywords.")
        return {}

    potential_keywords = [keyword.strip().lower() for keyword in keywords if keyword and keyword.strip()]
    if not potential_keywords:
        logger.debug("fetch_media_for_keywords: no valid keywords after initial cleaning and stripping.")
        return {}

    unique_clean_keywords = sorted(set(potential_keywords))
    if not unique_clean_keywords:
        logger.debug("fetch_media_for_keywords: no unique valid keywords remain.")
        return {}

    placeholders = ",".join("?" * len(unique_clean_keywords))
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

    order_expr = self._keyword_order_expression("k.keyword")
    query = """
        SELECT
            k.keyword AS keyword_text,
            {media_fields}
        FROM Keywords k
        JOIN MediaKeywords mk ON k.id = mk.keyword_id
        JOIN Media m ON mk.media_id = m.id
        WHERE {media_where_clause}
          AND k.keyword IN ({placeholders})
          AND k.deleted = ?
        ORDER BY {order_expr}, m.last_modified DESC, m.id DESC
    """.format_map(locals())  # nosec B608

    params = tuple(media_params + unique_clean_keywords + [False])

    logger.debug(
        "Executing fetch_media_for_keywords query for keywords: {}, include_trash: {}",
        unique_clean_keywords,
        include_trash,
    )

    results_by_keyword: dict[str, list[dict[str, Any]]] = {keyword: [] for keyword in unique_clean_keywords}

    try:
        conn = self.get_connection()
        rows = self._fetchall_with_connection(conn, query, params)

        for row in rows:
            db_keyword = row["keyword_text"]
            media_item = {
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

            if db_keyword in results_by_keyword:
                results_by_keyword[db_keyword].append(media_item)
            else:
                logger.error(
                    "Data consistency alert in fetch_media_for_keywords: "
                    "Keyword '{}' from DB results was not in the expected set of "
                    "unique_clean_keywords: {}. This may indicate a mismatch in case "
                    "handling or normalization.",
                    db_keyword,
                    unique_clean_keywords,
                )
                results_by_keyword[db_keyword] = [media_item]

        num_keywords_with_media = len([keyword for keyword, media_items in results_by_keyword.items() if media_items])
        total_media_items_found = sum(len(media_items) for media_items in results_by_keyword.values())
        logger.info(
            "Fetched media for keywords. Queried unique keywords: {}. Keywords with media found: {}. "
            "Total media items grouped: {}",
            len(unique_clean_keywords),
            num_keywords_with_media,
            total_media_items_found,
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
