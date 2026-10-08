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

Rewriting links after a rename
------------------------------
Links follow titles, so renaming a note leaves ``[[Old title]]`` links
unresolved. :func:`rewrite_wikilink_title_tokens` replaces exactly the tokens
this parser reads as a link to the old title (:func:`iter_wikilink_tokens`),
and :func:`restore_wikilink_tokens` puts the original tokens back. There is no
alias form (``[[Title|label]]`` is the title ``Title|label``) and code spans
are not special: a link inside code is a link here, so it is rewritten too.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

WIKILINK_PARSER_VERSION = 2
MAX_WIKILINK_TARGETS = 1_024
MAX_WIKILINK_TITLE_LENGTH = 1_024
# Bounds for the rename rewrite and its undo (``wikilink_rename.py``).
# Notes one rewrite or undo request may name; callers page through more.
MAX_WIKILINK_RENAME_NOTES = 200
# Links to one title that one note may have rewritten, and the longest token
# text undo carries back. A note beyond either is reported, not rewritten.
MAX_WIKILINK_REPLACEMENTS_PER_NOTE = 1_000
MAX_WIKILINK_TOKEN_TEXT_LENGTH = 4_096

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


@dataclass(frozen=True, slots=True)
class WikilinkToken:
    """One ``[[...]]`` token, classified the way the projection reads it.

    ``note_id`` is set for a valid ``[[id:UUID]]`` link and ``title`` (trimmed,
    whitespace collapsed) for a ``[[Title]]`` link. Both are ``None`` for a
    token that is not a link: a malformed id, a blank or an over-long title.
    ``index`` counts every token in the content, link or not.
    """

    index: int
    start: int
    end: int
    raw: str
    note_id: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class WikilinkTokenReplacement:
    """One rewritten token: its ordinal in the content and its original text."""

    token_index: int
    original: str


class WikilinkRewriteUnsafeError(ValueError):
    """Rewriting a link in place would change which link the text holds."""


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
    for token in iter_wikilink_tokens(content):
        if token.note_id is not None:
            target = token.note_id
            if target == normalized_source or target in seen_ids:
                continue
            seen_ids.add(target)
            if len(targets) + len(titles) == max_targets:
                truncated = True
                continue
            targets.append(target)
            continue
        if token.title is None:
            continue
        title_key = token.title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        if len(targets) + len(titles) == max_targets:
            truncated = True
            continue
        titles.append(token.title)
    return WikilinkProjection(tuple(targets), truncated, target_titles=tuple(titles))


def iter_wikilink_tokens(content: str | None) -> Iterator[WikilinkToken]:
    """Yield every ``[[...]]`` token in ``content``, in order.

    This is the one place that decides what a link is. ``parse_wikilinks`` and
    the rename rewrite both read tokens through it, so they can't disagree.
    """

    for index, match in enumerate(_WIKILINK_TOKEN_RE.finditer(content or "")):
        inner = match.group(1).strip()
        note_id: str | None = None
        title: str | None = None
        if inner.startswith(_ID_PREFIX):
            # The prefix is reserved: a malformed id is not a link at all.
            note_id = canonical_wikilink_note_id(inner[len(_ID_PREFIX):])
        else:
            collapsed = collapse_wikilink_title(inner)
            if collapsed and len(collapsed) <= MAX_WIKILINK_TITLE_LENGTH:
                title = collapsed
        yield WikilinkToken(index, match.start(), match.end(), match.group(0), note_id, title)


def wikilink_title_link_text(title: str | None) -> str | None:
    """Return ``[[Title]]`` when the parser reads it back as a link to ``title``.

    ``None`` means no title link can name this title: it is blank, too long,
    starts with the reserved ``id:`` prefix, or contains ``[[`` or ``]]``.
    """

    collapsed = collapse_wikilink_title(title)
    if not collapsed:
        return None
    text = f"[[{collapsed}]]"
    token = _only_wikilink_token(text)
    return text if token is not None and token.title == collapsed else None


def wikilink_id_link_text(note_id: str | None) -> str | None:
    """Return ``[[id:<uuid>]]`` for a canonical note id, else ``None``."""

    canonical = canonical_wikilink_note_id(note_id)
    return None if canonical is None else f"[[{_ID_PREFIX}{canonical}]]"


def is_single_wikilink(text: str | None) -> bool:
    """Report whether ``text`` is exactly one ``[[Title]]`` or ``[[id:UUID]]`` link."""

    return _only_wikilink_token(text) is not None


def rewrite_wikilink_title_tokens(
    content: str,
    *,
    old_title: str,
    replacement: str,
) -> tuple[str, tuple[WikilinkTokenReplacement, ...]]:
    """Replace every ``[[old_title]]`` link in ``content`` with ``replacement``.

    Only whole tokens the parser reads as a link to ``old_title`` (ignoring
    case and extra whitespace) are replaced; ``[[id:UUID]]`` links, similar
    titles and all other text are returned byte for byte. ``replacement`` must
    itself be exactly one link. The returned replacements are what
    :func:`restore_wikilink_tokens` needs to undo the rewrite, and the result
    is only returned if that undo is exact.

    Raises:
        WikilinkRewriteUnsafeError: a replacement would join neighbouring
            brackets and read as a different link. This needs a link to a
            title that starts with ``[`` directly after another ``[``
            (``[[[[Old]]``); such text can't be rewritten in place.
    """

    if not is_single_wikilink(replacement):
        raise ValueError("replacement must be exactly one [[Title]] or [[id:UUID]] link")
    source = content or ""
    old_key = normalize_wikilink_title(old_title)
    if not old_key:
        return source, ()
    pieces: list[str] = []
    replaced: list[WikilinkTokenReplacement] = []
    cursor = 0
    for token in iter_wikilink_tokens(source):
        if token.title is None or token.title.lower() != old_key:
            continue
        pieces.extend((source[cursor:token.start], replacement))
        cursor = token.end
        replaced.append(WikilinkTokenReplacement(token.index, token.raw))
    if not replaced:
        return source, ()
    pieces.append(source[cursor:])
    rewritten = "".join(pieces)
    replacements = tuple(replaced)
    # Reading the result back must find the replacement at the same tokens.
    if restore_wikilink_tokens(rewritten, replacements, old_title=old_title, replacement=replacement) != source:
        raise WikilinkRewriteUnsafeError(
            "a link to this title can't be rewritten in place: the new link would join neighbouring brackets"
        )
    return rewritten, replacements


def restore_wikilink_tokens(
    content: str,
    replacements: Sequence[WikilinkTokenReplacement],
    *,
    old_title: str,
    replacement: str,
) -> str | None:
    """Undo :func:`rewrite_wikilink_title_tokens`, or return ``None`` if it can't be exact.

    Each named token must still be ``replacement``, and each original must be
    one ``[[old_title]]`` link, so this can only turn the rewritten links back
    into links to the old title.
    """

    source = content or ""
    old_key = normalize_wikilink_title(old_title)
    originals = {item.token_index: item.original for item in replacements}
    if not old_key or not originals or len(originals) != len(replacements):
        return None
    if not is_single_wikilink(replacement):
        return None
    for original_text in originals.values():
        original_token = _only_wikilink_token(original_text)
        if (
            original_token is None
            or original_token.title is None
            or original_token.title.lower() != old_key
        ):
            return None
    pieces: list[str] = []
    cursor = 0
    restored = 0
    for token in iter_wikilink_tokens(source):
        original = originals.get(token.index)
        if original is None:
            continue
        if token.raw != replacement:
            return None
        pieces.extend((source[cursor:token.start], original))
        cursor = token.end
        restored += 1
    if restored != len(originals):
        return None
    pieces.append(source[cursor:])
    return "".join(pieces)


def _only_wikilink_token(text: str | None) -> WikilinkToken | None:
    """Return the link ``text`` is, when it is exactly one link and nothing else."""

    tokens = list(iter_wikilink_tokens(text))
    if len(tokens) != 1:
        return None
    token = tokens[0]
    if token.start != 0 or token.end != len(text or ""):
        return None
    return token if token.note_id is not None or token.title is not None else None


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
    "MAX_WIKILINK_RENAME_NOTES",
    "MAX_WIKILINK_REPLACEMENTS_PER_NOTE",
    "MAX_WIKILINK_TARGETS",
    "MAX_WIKILINK_TITLE_LENGTH",
    "MAX_WIKILINK_TOKEN_TEXT_LENGTH",
    "WIKILINK_PARSER_VERSION",
    "WikilinkProjection",
    "WikilinkRewriteUnsafeError",
    "WikilinkTitleCandidate",
    "WikilinkToken",
    "WikilinkTokenReplacement",
    "canonical_wikilink_note_id",
    "collapse_wikilink_title",
    "is_single_wikilink",
    "is_wikilink_title_reference_key",
    "iter_wikilink_tokens",
    "normalize_wikilink_title",
    "parse_wikilinks",
    "restore_wikilink_tokens",
    "rewrite_wikilink_title_tokens",
    "select_wikilink_title_target",
    "wikilink_id_link_text",
    "wikilink_title_link_text",
    "wikilink_title_reference_key",
]
