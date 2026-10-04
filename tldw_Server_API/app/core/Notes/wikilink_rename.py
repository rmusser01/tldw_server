"""Update ``[[Old title]]`` links after a note is renamed, and undo that update.

Links follow titles (``wikilinks.py``), so a rename leaves links to the old
title unresolved. Nothing here runs on its own: the WebUI asks how many notes
link to the old title (``NoteGraphProjectionStore.list_title_referrers``) and
offers the rewrite, and the user can undo it.

Rules
-----
* Only whole links to the old title change, matched the way the parser
  matches them. See :func:`rewrite_wikilink_title_tokens`.
* Only links the rename broke change. If another live note still has the old
  title, the link resolves to that note and is left alone
  (``skipped_resolved``).
* Each note is saved on its own under optimistic locking. A note edited since
  the caller read its version is skipped (``skipped_conflict``), never
  overwritten, and a failed save leaves that note's text whole.
* The new link is ``[[New title]]``. When another live note shares the new
  title, that link could resolve to the other note, so the renamed note is
  linked by id instead: ``[[id:<UUID>]]``
  (:func:`plan_renamed_note_link`).
* Undo holds no server state. The rewrite returns each replaced token's
  ordinal and original text, and :func:`restore_title_links` puts exactly
  those tokens back if the note has not changed since.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
from tldw_Server_API.app.core.Notes.wikilinks import (
    MAX_WIKILINK_RENAME_NOTES,
    MAX_WIKILINK_REPLACEMENTS_PER_NOTE,
    MAX_WIKILINK_TOKEN_TEXT_LENGTH,
    WikilinkTokenReplacement,
    restore_wikilink_tokens,
    rewrite_wikilink_title_tokens,
    wikilink_id_link_text,
    wikilink_title_link_text,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

WikilinkRenameStatus = Literal[
    "updated",
    "restored",
    "skipped_conflict",
    "skipped_no_match",
    "skipped_not_found",
    "skipped_resolved",
    "failed",
]
# Saves one note's new content at the version just read, and returns the saved row.
SaveNoteContent = Callable[[dict[str, Any], str], dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class RenamedNoteLink:
    """How links to a renamed note are written."""

    note_id: str
    new_title: str
    # "title" for ``[[New title]]``, "id" for ``[[id:UUID]]``, None when neither can name the note.
    link_form: Literal["title", "id"] | None
    replacement: str | None
    # Another live note has the same title, ignoring case and extra whitespace.
    new_title_shared: bool


@dataclass(frozen=True, slots=True)
class WikilinkRewriteTarget:
    note_id: str
    expected_version: int


@dataclass(frozen=True, slots=True)
class WikilinkRestoreTarget:
    note_id: str
    expected_version: int
    replacements: tuple[WikilinkTokenReplacement, ...]


@dataclass(frozen=True, slots=True)
class WikilinkNoteResult:
    """What happened to one note. ``version`` is its version afterwards."""

    note_id: str
    status: WikilinkRenameStatus
    title: str | None = None
    version: int | None = None
    replacements: tuple[WikilinkTokenReplacement, ...] = ()


def plan_renamed_note_link(db: CharactersRAGDB, note_id: str) -> RenamedNoteLink | None:
    """Choose the link text for a renamed note, or ``None`` if it is not a live note.

    A title link is used when it is unambiguous. When another live note shares
    the title, or no title link can name it, the note is linked by id. A note
    whose id is not a UUID has no id link: it keeps the title link (flagged as
    shared), or has no link form at all.
    """

    note = db.get_note_by_id(note_id=note_id)
    if note is None:
        return None
    new_title = str(note.get("title") or "")
    others = set(db.note_graph_projection_store.list_live_note_ids_titled(new_title)) - {str(note["id"])}
    title_link = wikilink_title_link_text(new_title)
    id_link = wikilink_id_link_text(str(note["id"]))
    if title_link is not None and (not others or id_link is None):
        return RenamedNoteLink(str(note["id"]), new_title, "title", title_link, bool(others))
    if id_link is not None:
        return RenamedNoteLink(str(note["id"]), new_title, "id", id_link, bool(others))
    return RenamedNoteLink(str(note["id"]), new_title, None, None, bool(others))


def rewrite_title_links(
    db: CharactersRAGDB,
    *,
    old_title: str,
    replacement: str,
    targets: Sequence[WikilinkRewriteTarget],
    save: SaveNoteContent | None = None,
) -> list[WikilinkNoteResult]:
    """Rewrite ``[[old_title]]`` links to ``replacement`` in each target note.

    Results come back in request order, one per distinct note.
    """

    _require_bounded(targets)
    save_content = save or _default_save(db)
    answering = set(db.note_graph_projection_store.list_live_note_ids_titled(old_title))
    results: list[WikilinkNoteResult] = []
    for target in _distinct(targets):
        note = db.get_note_by_id(note_id=target.note_id)
        if note is None:
            results.append(WikilinkNoteResult(target.note_id, "skipped_not_found"))
            continue
        if int(note["version"]) != target.expected_version:
            results.append(_result(note, "skipped_conflict"))
            continue
        if answering - {str(note["id"])}:
            # Another live note still has the old title, so this link is not broken.
            results.append(_result(note, "skipped_resolved"))
            continue
        content = str(note.get("content") or "")
        new_content, replaced = rewrite_wikilink_title_tokens(
            content, old_title=old_title, replacement=replacement
        )
        if not replaced:
            results.append(_result(note, "skipped_no_match"))
            continue
        if not _undoable(content, new_content, replaced, old_title=old_title, replacement=replacement):
            logger.warning("Wikilink rename skipped note {}: undo could not restore it exactly", note["id"])
            results.append(_result(note, "failed"))
            continue
        results.append(_save(db, save_content, note, new_content, "updated", replaced))
    return results


def restore_title_links(
    db: CharactersRAGDB,
    *,
    old_title: str,
    replacement: str,
    targets: Sequence[WikilinkRestoreTarget],
    save: SaveNoteContent | None = None,
) -> list[WikilinkNoteResult]:
    """Undo :func:`rewrite_title_links` for each target note, restoring its previous text."""

    _require_bounded(targets)
    save_content = save or _default_save(db)
    results: list[WikilinkNoteResult] = []
    for target in _distinct(targets):
        note = db.get_note_by_id(note_id=target.note_id)
        if note is None:
            results.append(WikilinkNoteResult(target.note_id, "skipped_not_found"))
            continue
        if int(note["version"]) != target.expected_version:
            results.append(_result(note, "skipped_conflict"))
            continue
        restored = restore_wikilink_tokens(
            str(note.get("content") or ""),
            target.replacements,
            old_title=old_title,
            replacement=replacement,
        )
        if restored is None:
            results.append(_result(note, "skipped_no_match"))
            continue
        results.append(_save(db, save_content, note, restored, "restored", ()))
    return results


def _undoable(
    content: str,
    new_content: str,
    replaced: tuple[WikilinkTokenReplacement, ...],
    *,
    old_title: str,
    replacement: str,
) -> bool:
    """Report whether the undo request can carry this rewrite and restore the text exactly."""

    if len(replaced) > MAX_WIKILINK_REPLACEMENTS_PER_NOTE:
        return False
    if any(len(item.original) > MAX_WIKILINK_TOKEN_TEXT_LENGTH for item in replaced):
        return False
    restored = restore_wikilink_tokens(new_content, replaced, old_title=old_title, replacement=replacement)
    return restored == content


def _require_bounded(targets: Sequence[object]) -> None:
    if len(targets) > MAX_WIKILINK_RENAME_NOTES:
        raise ValueError(f"at most {MAX_WIKILINK_RENAME_NOTES} notes can be changed per request")


def _distinct(targets: Sequence[Any]) -> list[Any]:
    """Keep the first target for each note id, in request order."""

    seen: set[str] = set()
    kept: list[Any] = []
    for target in targets:
        if target.note_id not in seen:
            seen.add(target.note_id)
            kept.append(target)
    return kept


def _result(
    note: dict[str, Any],
    status: WikilinkRenameStatus,
    replacements: tuple[WikilinkTokenReplacement, ...] = (),
) -> WikilinkNoteResult:
    return WikilinkNoteResult(
        str(note["id"]),
        status,
        title=str(note.get("title") or ""),
        version=int(note["version"]),
        replacements=replacements,
    )


def _default_save(db: CharactersRAGDB) -> SaveNoteContent:
    def save(note: dict[str, Any], content: str) -> dict[str, Any] | None:
        db.update_note(
            note_id=str(note["id"]),
            update_data={"content": content},
            expected_version=int(note["version"]),
        )
        return db.get_note_by_id(note_id=str(note["id"]))

    return save


def _save(
    db: CharactersRAGDB,
    save_content: SaveNoteContent,
    note: dict[str, Any],
    content: str,
    done: WikilinkRenameStatus,
    replacements: tuple[WikilinkTokenReplacement, ...],
) -> WikilinkNoteResult:
    """Save one note and report it. The save is one transaction: all of the note or none."""

    try:
        saved = save_content(note, content)
    except ConflictError:
        # Edited between the read above and the write: report the note as it is now.
        current = db.get_note_by_id(note_id=str(note["id"]))
        if current is None:
            return WikilinkNoteResult(str(note["id"]), "skipped_not_found")
        return _result(current, "skipped_conflict")
    except Exception as exc:  # noqa: BLE001 - one note's failure must not stop the batch.
        logger.warning("Wikilink rename could not save note {}: {}", note["id"], exc)
        return _result(note, "failed")
    return _result(saved or note, done, replacements)


__all__ = [
    "RenamedNoteLink",
    "SaveNoteContent",
    "WikilinkNoteResult",
    "WikilinkRenameStatus",
    "WikilinkRestoreTarget",
    "WikilinkRewriteTarget",
    "plan_renamed_note_link",
    "restore_title_links",
    "rewrite_title_links",
]
