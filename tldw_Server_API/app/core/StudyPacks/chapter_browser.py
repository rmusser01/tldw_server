"""Browse a deck by the canonical chapters attached to its generated cards."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.chacha.flashcard_chapter_queries import list_flashcard_chapter_rows

from .types import CHAPTER_ID_PATTERN, CHAPTER_LOCATOR_KEY

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _chapter_from_row(row: dict[str, Any]) -> tuple[str, str, int, str, str] | None:
    source_id = row.get("source_id")
    locator = row.get("locator")
    if not isinstance(source_id, str) or not isinstance(locator, str):
        return None
    try:
        parsed = json.loads(locator)
    except (ValueError, RecursionError):
        return None
    chapter = parsed.get(CHAPTER_LOCATOR_KEY) if isinstance(parsed, dict) else None
    if not isinstance(chapter, dict):
        return None
    index = chapter.get("section_index")
    revision = chapter.get("section_revision")
    title = chapter.get("title")
    source_title = chapter.get("source_title")
    if (
        type(index) is not int
        or not isinstance(revision, str)
        or not isinstance(title, str)
        or not isinstance(source_title, str)
    ):
        return None
    identifier = f"{source_id}:{index}:{revision}"
    if re.fullmatch(CHAPTER_ID_PATTERN, identifier) is None:
        return None
    return identifier, source_id, index, title, source_title


def browse_deck_cards(
    db: CharactersRAGDB,
    deck_id: int,
    *,
    chapter_id: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Count active deck cards and load only the requested page of card bodies."""
    memberships: dict[str, str | None] = {}
    chapters: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in list_flashcard_chapter_rows(db, deck_id):
        uuid = row["uuid"]
        if memberships.setdefault(uuid, None) is not None:
            continue
        chapter = _chapter_from_row(row)
        if chapter is None:
            continue

        identifier, source_id, index, title, source_title = chapter
        entry = chapters.setdefault((int(source_id), index, identifier), {
            "id": identifier,
            "title": title,
            "source_id": source_id,
            "source_title": source_title,
            "card_count": 0,
        })
        entry["card_count"] += 1
        memberships[uuid] = identifier

    matching_ids = [uuid for uuid, membership in memberships.items() if chapter_id is None or membership == chapter_id]
    window = matching_ids[offset:offset + limit]
    cards = {row["uuid"]: row for row in db.get_flashcards_by_uuids(window) if row["deck_id"] == deck_id}
    return {
        "chapters": [chapters[key] for key in sorted(chapters, key=lambda key: key[:2])],
        "all_cards_count": len(memberships),
        "items": [cards[uuid] for uuid in window if uuid in cards],
        "total": len(matching_ids),
    }
