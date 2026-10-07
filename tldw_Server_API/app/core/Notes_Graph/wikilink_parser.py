"""Compatibility wrapper for the neutral Notes wikilink parser."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_TARGETS,
    WIKILINK_PARSER_VERSION,
    WikilinkProjection,
    parse_wikilinks,
)


@dataclass(frozen=True, slots=True)
class WikilinkRef:
    """A resolved wikilink target (lowercase-normalized UUID)."""

    target_note_id: str


def extract_wikilinks(content: str) -> list[WikilinkRef]:
    """Return deduplicated ``[[id:<UUID>]]`` refs in order of first occurrence.

    ``[[Title]]`` links are not returned: a title resolves to a note only
    against the owner's notes, which ``NoteGraphProjectionStore`` does when it
    projects edges (see ``tldw_Server_API.app.core.Notes.wikilinks``).
    """
    return [
        WikilinkRef(target_note_id=target)
        for target in parse_wikilinks(content).target_note_ids
    ]


__all__ = [
    "MAX_WIKILINK_TARGETS",
    "WIKILINK_PARSER_VERSION",
    "WikilinkProjection",
    "WikilinkRef",
    "extract_wikilinks",
    "parse_wikilinks",
]
