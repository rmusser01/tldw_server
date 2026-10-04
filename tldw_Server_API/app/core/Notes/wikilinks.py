"""Deterministic parsing for local note wikilink projections.

Two link forms are supported (UX review decision D2, NE-02 / #3110):

* ``[[id:<UUID>]]`` links one note by its immutable id. A malformed id is
  ignored; the ``id:`` prefix is reserved and never falls back to a title.
* ``[[Title]]`` links a note by title. Titles may contain single ``[`` or
  ``]`` characters (``[[[Draft] Proposal]]``) but not ``[[`` or a newline.

This module only parses. Resolving a title to a note needs the owner's notes,
so it happens in the persistence layer
(``NoteGraphProjectionStore``), which applies :func:`select_wikilink_title_target`.

Title resolution rules
----------------------
* Titles match after trimming, collapsing whitespace, and lower-casing
  (:func:`normalize_wikilink_title`), the same rule the WebUI uses.
* Only live notes of the same owner are candidates, never the linking note.
* Ambiguous titles resolve deterministically: an exact (case-sensitive) title
  match wins, then the oldest note, then the lowest note id. "Oldest wins"
  keeps existing links stable when a duplicate title is created later; link a
  specific duplicate with ``[[id:<UUID>]]``.
* A title with no live candidate is unresolved: it creates no edge, and the
  WebUI offers to create the note.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

WIKILINK_PARSER_VERSION = 2
MAX_WIKILINK_TARGETS = 1_024
MAX_WIKILINK_TITLE_LENGTH = 1_024

# ``[[`` + content without a newline or a nested ``[[`` + the first ``]]`` that
# is not followed by another ``]``. The WebUI tokenizer uses the same pattern.
_WIKILINK_TOKEN_RE = re.compile(r"\[\[((?:(?!\[\[)[^\n])+?)\]\](?!\])")
_CANONICAL_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_ID_PREFIX = "id:"
_TITLE_REFERENCE_PREFIX = "title:"


@dataclass(frozen=True, slots=True)
class WikilinkProjection:
    """One bounded, first-occurrence-ordered parser result.

    ``target_note_ids`` holds canonical ``[[id:UUID]]`` targets.
    ``target_titles`` holds unresolved ``[[Title]]`` targets (trimmed, with
    whitespace collapsed), deduplicated by normalized title.
    """

    target_note_ids: tuple[str, ...]
    truncated: bool
    parser_version: int = WIKILINK_PARSER_VERSION
    target_titles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WikilinkTitleCandidate:
    """One live note that may answer a ``[[Title]]`` link."""

    note_id: str
    title: str
    created_at: Any = None


def collapse_wikilink_title(title: str | None) -> str:
    """Trim a title and collapse internal whitespace runs to one space."""

    return " ".join(str(title or "").split())


def normalize_wikilink_title(title: str | None) -> str:
    """Return the case-insensitive comparison key for a link or note title."""

    return collapse_wikilink_title(title).lower()


def wikilink_title_reference_key(title: str | None) -> str:
    """Return the fixed-size projection key that records a ``[[Title]]`` reference.

    The key is stored beside resolved edges so a later title change can find
    the notes whose links must be re-resolved. It is never a valid UUID, so it
    can't collide with a canonical note-id target.
    """

    digest = hashlib.sha256(normalize_wikilink_title(title).encode("utf-8")).hexdigest()
    return f"{_TITLE_REFERENCE_PREFIX}{digest}"


def is_wikilink_title_reference_key(value: str | None) -> bool:
    """Report whether a projected target is a title reference key, not a note id."""

    return isinstance(value, str) and value.startswith(_TITLE_REFERENCE_PREFIX)


def select_wikilink_title_target(
    link_title: str,
    candidates: Iterable[WikilinkTitleCandidate],
) -> str | None:
    """Pick the note a ``[[Title]]`` link resolves to, or ``None`` when unresolved.

    Candidates are filtered to normalized-title matches. An exact
    (case-sensitive, whitespace-collapsed) title match wins, then the oldest
    ``created_at``, then the lowest note id.
    """

    key = normalize_wikilink_title(link_title)
    if not key:
        return None
    matches = [candidate for candidate in candidates if normalize_wikilink_title(candidate.title) == key]
    if not matches:
        return None
    exact_title = collapse_wikilink_title(link_title)
    exact = [candidate for candidate in matches if collapse_wikilink_title(candidate.title) == exact_title]
    pool = exact or matches
    return min(pool, key=lambda candidate: (str(candidate.created_at or ""), str(candidate.note_id))).note_id


def parse_wikilinks(
    content: str,
    *,
    source_note_id: str | None = None,
    max_targets: int = MAX_WIKILINK_TARGETS,
) -> WikilinkProjection:
    """Parse id and title link targets without resolving titles against product rows.

    ``max_targets`` bounds id and title targets together, in first-occurrence
    order. Self links by id are omitted here; self links by title are omitted
    during resolution.
    """

    if max_targets < 1:
        raise ValueError("max_targets must be at least one")
    normalized_source = _normalized_uuid(source_note_id) if source_note_id else None
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    targets: list[str] = []
    titles: list[str] = []
    truncated = False
    for match in _WIKILINK_TOKEN_RE.finditer(content or ""):
        inner = match.group(1).strip()
        if inner.startswith(_ID_PREFIX):
            target = canonical_wikilink_note_id(inner[len(_ID_PREFIX):])
            if target is None or target == normalized_source or target in seen_ids:
                continue
            seen_ids.add(target)
            if len(targets) + len(titles) == max_targets:
                truncated = True
                continue
            targets.append(target)
            continue
        title = collapse_wikilink_title(inner)
        title_key = title.lower()
        if not title or len(title) > MAX_WIKILINK_TITLE_LENGTH or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        if len(targets) + len(titles) == max_targets:
            truncated = True
            continue
        titles.append(title)
    return WikilinkProjection(tuple(targets), truncated, target_titles=tuple(titles))


def canonical_wikilink_note_id(value: str | None) -> str | None:
    """Return the lower-case UUID an ``[[id:...]]`` link names, or ``None`` if malformed."""

    raw = str(value or "").strip()
    return _normalized_uuid(raw) if _CANONICAL_UUID_RE.fullmatch(raw) else None


def _normalized_uuid(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


__all__ = [
    "MAX_WIKILINK_TARGETS",
    "MAX_WIKILINK_TITLE_LENGTH",
    "WIKILINK_PARSER_VERSION",
    "WikilinkProjection",
    "WikilinkTitleCandidate",
    "canonical_wikilink_note_id",
    "collapse_wikilink_title",
    "is_wikilink_title_reference_key",
    "normalize_wikilink_title",
    "parse_wikilinks",
    "select_wikilink_title_target",
    "wikilink_title_reference_key",
]
