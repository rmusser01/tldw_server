"""Chapter membership queries for the flashcard browser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def list_flashcard_chapter_rows(db: CharactersRAGDB, deck_id: int) -> list[dict[str, Any]]:
    """Read live cards and saved chapter locators from one owner-scoped deck."""
    postgres = db.backend_type == BackendType.POSTGRESQL
    citation_scope = "AND c.client_id = f.client_id" if postgres else ""
    card_scope = "AND f.client_id = ?" if postgres else ""
    params: tuple[str | int, ...] = ('%"study_pack_chapter"%', deck_id)
    if postgres:
        params += (db.client_id,)
    query = f"""
        SELECT f.uuid, c.source_id, c.locator
          FROM flashcards f
          LEFT JOIN flashcard_citations c
            ON c.flashcard_uuid = f.uuid
           {citation_scope}
           AND c.deleted = FALSE
           AND c.source_type = 'media'
           AND c.locator LIKE ?
         WHERE f.deck_id = ? AND f.deleted = FALSE {card_scope}
         ORDER BY f.id, c.ordinal, c.id
    """  # nosec B608 - Fixed owner predicates; data values remain bound.
    cursor = db.execute_query(query, params, read_only=True)
    return [dict(row) for row in cursor.fetchall()]
