"""Offer to update ``[[Old title]]`` links after a rename (#3110, owner decision).

Links follow titles, so a rename leaves links to the old title unresolved.
Nothing is rewritten silently: the count, the rewrite and its undo are explicit
calls, each note is updated under optimistic locking, and undo restores the
previous text exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Notes import wikilink_rename
from tldw_Server_API.app.core.Notes.wikilink_rename import (
    WikilinkRestoreTarget,
    WikilinkRewriteTarget,
    plan_renamed_note_link,
    restore_title_links,
    rewrite_title_links,
)

pytestmark = pytest.mark.unit

RENAMED_ID = "10000000-0000-4000-8000-000000000001"
LINKER_A_ID = "20000000-0000-4000-8000-000000000002"
LINKER_B_ID = "30000000-0000-4000-8000-000000000003"
OTHER_ID = "40000000-0000-4000-8000-000000000004"
TRASHED_ID = "50000000-0000-4000-8000-000000000005"
DUPLICATE_ID = "60000000-0000-4000-8000-000000000006"

LINKER_A_TEXT = "Intro [[Old title]] and plain Old title."
LINKER_B_TEXT = "Twice: [[old  TITLE]] then [[ Old title ]], not [[Old title 2]]."
NEW_LINK = "[[New title]]"


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "wikilink-rename.db"), client_id="owner-1")
    try:
        yield database
    finally:
        database.close_connection()


def _note(db: CharactersRAGDB, note_id: str) -> dict:
    note = db.get_note_by_id(note_id, include_deleted=True)
    assert note is not None
    return note


def _version(db: CharactersRAGDB, note_id: str) -> int:
    return int(_note(db, note_id)["version"])


def _content(db: CharactersRAGDB, note_id: str) -> str:
    return str(_note(db, note_id)["content"])


def _seed_renamed_library(db: CharactersRAGDB) -> None:
    """A note renamed from "Old title" to "New title", and the notes around it."""

    db.add_note("Old title", "Self link [[Old title]].", note_id=RENAMED_ID)
    db.add_note("Linker A", LINKER_A_TEXT, note_id=LINKER_A_ID)
    db.add_note("Linker B", LINKER_B_TEXT, note_id=LINKER_B_ID)
    db.add_note("Unrelated", "Links [[Something else]] and [[Old title 2]].", note_id=OTHER_ID)
    db.add_note("Trashed linker", "Gone [[Old title]].", note_id=TRASHED_ID)
    db.soft_delete_note(TRASHED_ID, expected_version=1)
    db.update_note(RENAMED_ID, {"title": "New title"}, expected_version=1)


def _targets(db: CharactersRAGDB, *note_ids: str) -> list[WikilinkRewriteTarget]:
    return [WikilinkRewriteTarget(note_id, _version(db, note_id)) for note_id in note_ids]


def _restore_targets(results) -> list[WikilinkRestoreTarget]:
    return [
        WikilinkRestoreTarget(result.note_id, int(result.version), result.replacements)
        for result in results
        if result.status == "updated"
    ]


def _statuses(results) -> dict[str, str]:
    return {result.note_id: result.status for result in results}


# --- count -----------------------------------------------------------------


def test_referrers_counts_live_notes_linking_to_the_old_title(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    page = db.note_graph_projection_store.list_title_referrers(
        "old   TITLE", exclude_note_id=RENAMED_ID, unresolved_only=True
    )

    assert page.total == 2
    assert [(note.note_id, note.title, note.version) for note in page.notes] == [
        (LINKER_A_ID, "Linker A", 1),
        (LINKER_B_ID, "Linker B", 1),
    ]


def test_referrers_include_the_renamed_note_unless_it_is_excluded(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    page = db.note_graph_projection_store.list_title_referrers("Old title")

    assert [note.note_id for note in page.notes] == [RENAMED_ID, LINKER_A_ID, LINKER_B_ID]


def test_referrers_exclude_trashed_notes_until_they_are_restored(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    store = db.note_graph_projection_store

    assert TRASHED_ID not in [note.note_id for note in store.list_title_referrers("Old title").notes]

    db.restore_note(TRASHED_ID, expected_version=_version(db, TRASHED_ID))

    assert TRASHED_ID in [note.note_id for note in store.list_title_referrers("Old title").notes]


def test_referrers_are_paged_by_note_id(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    store = db.note_graph_projection_store

    first = store.list_title_referrers("Old title", exclude_note_id=RENAMED_ID, limit=1)
    second = store.list_title_referrers(
        "Old title", exclude_note_id=RENAMED_ID, after_note_id=first.notes[-1].note_id, limit=1
    )

    assert (first.total, [note.note_id for note in first.notes]) == (2, [LINKER_A_ID])
    assert (second.total, [note.note_id for note in second.notes]) == (2, [LINKER_B_ID])
    with pytest.raises(ValueError, match="limit"):
        store.list_title_referrers("Old title", limit=0)


def test_no_title_and_no_links_count_nothing(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    store = db.note_graph_projection_store

    assert store.list_title_referrers("   ").total == 0
    assert store.list_title_referrers("Never linked").total == 0


def test_unresolved_only_skips_links_another_note_still_answers(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    store = db.note_graph_projection_store
    # A second note still has the old title, so [[Old title]] resolves to it:
    # the rename broke nothing, and those links are not offered for rewriting.
    db.add_note("Old title", "I also link [[Old title]].", note_id=DUPLICATE_ID)

    resolved = store.list_title_referrers("Old title", exclude_note_id=RENAMED_ID, unresolved_only=True)
    every = store.list_title_referrers("Old title", exclude_note_id=RENAMED_ID)

    # A note never answers its own link, so only the duplicate's link is broken.
    assert [note.note_id for note in resolved.notes] == [DUPLICATE_ID]
    assert resolved.total == 1
    assert [note.note_id for note in every.notes] == [LINKER_A_ID, LINKER_B_ID, DUPLICATE_ID]


# --- how the renamed note is linked ------------------------------------------


def test_a_unique_new_title_is_linked_by_title(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    plan = plan_renamed_note_link(db, RENAMED_ID)

    assert plan is not None
    assert (plan.new_title, plan.link_form, plan.replacement, plan.new_title_shared) == (
        "New title",
        "title",
        NEW_LINK,
        False,
    )


def test_a_shared_new_title_is_linked_by_id(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    db.add_note("new   TITLE", "an older duplicate", note_id=DUPLICATE_ID)

    plan = plan_renamed_note_link(db, RENAMED_ID)

    assert plan is not None
    assert (plan.link_form, plan.replacement, plan.new_title_shared) == (
        "id",
        f"[[id:{RENAMED_ID}]]",
        True,
    )


def test_a_trashed_duplicate_does_not_make_the_new_title_shared(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    db.add_note("New title", "trashed duplicate", note_id=DUPLICATE_ID)
    db.soft_delete_note(DUPLICATE_ID, expected_version=1)

    plan = plan_renamed_note_link(db, RENAMED_ID)

    assert plan is not None
    assert (plan.link_form, plan.new_title_shared) == ("title", False)


def test_a_title_no_link_can_name_is_linked_by_id(db: CharactersRAGDB) -> None:
    db.add_note("Old title", "body", note_id=RENAMED_ID)
    db.update_note(RENAMED_ID, {"title": "id:reserved prefix"}, expected_version=1)

    plan = plan_renamed_note_link(db, RENAMED_ID)

    assert plan is not None
    assert (plan.link_form, plan.replacement) == ("id", f"[[id:{RENAMED_ID}]]")


def test_a_note_without_a_uuid_id_falls_back_to_its_title(db: CharactersRAGDB) -> None:
    db.add_note("New title", "imported with a non-UUID id", note_id="legacy-note-1")
    db.add_note("New title", "duplicate", note_id=DUPLICATE_ID)

    shared = plan_renamed_note_link(db, "legacy-note-1")
    db.update_note("legacy-note-1", {"title": "a]]b"}, expected_version=1)
    unlinkable = plan_renamed_note_link(db, "legacy-note-1")

    assert shared is not None and unlinkable is not None
    # No [[id:...]] form exists for this id: keep the title link and flag it as shared.
    assert (shared.link_form, shared.replacement, shared.new_title_shared) == ("title", NEW_LINK, True)
    assert (unlinkable.link_form, unlinkable.replacement) == (None, None)


def test_a_missing_or_trashed_renamed_note_has_no_plan(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    assert plan_renamed_note_link(db, TRASHED_ID) is None
    assert plan_renamed_note_link(db, "70000000-0000-4000-8000-000000000007") is None


# --- rewrite ---------------------------------------------------------------


def test_rewrite_updates_only_exact_links_and_restores_the_backlink(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    store = db.note_graph_projection_store
    assert RENAMED_ID not in store.list_live_outgoing(LINKER_A_ID)

    results = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_A_ID, LINKER_B_ID),
    )

    assert _statuses(results) == {LINKER_A_ID: "updated", LINKER_B_ID: "updated"}
    assert _content(db, LINKER_A_ID) == "Intro [[New title]] and plain Old title."
    assert _content(db, LINKER_B_ID) == "Twice: [[New title]] then [[New title]], not [[Old title 2]]."
    assert [(result.title, result.version) for result in results] == [("Linker A", 2), ("Linker B", 2)]
    assert [[item.original for item in result.replacements] for result in results] == [
        ["[[Old title]]"],
        ["[[old  TITLE]]", "[[ Old title ]]"],
    ]
    # The links resolve again, so the renamed note regains its backlinks.
    assert RENAMED_ID in store.list_live_outgoing(LINKER_A_ID)
    assert RENAMED_ID in store.list_live_outgoing(LINKER_B_ID)
    assert store.list_title_referrers("Old title", exclude_note_id=RENAMED_ID).total == 0


def test_rewrite_leaves_untargeted_notes_alone(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    rewrite_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=_targets(db, LINKER_A_ID)
    )

    assert _content(db, LINKER_B_ID) == LINKER_B_TEXT
    assert _content(db, RENAMED_ID) == "Self link [[Old title]]."
    assert _content(db, TRASHED_ID) == "Gone [[Old title]]."
    assert _version(db, LINKER_B_ID) == 1


def test_a_note_edited_since_the_count_is_skipped_not_overwritten(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    targets = _targets(db, LINKER_A_ID, LINKER_B_ID)
    edited = "Edited after the count, still [[Old title]]."
    db.update_note(LINKER_A_ID, {"content": edited}, expected_version=1)

    results = rewrite_title_links(db, old_title="Old title", replacement=NEW_LINK, targets=targets)

    assert _statuses(results) == {LINKER_A_ID: "skipped_conflict", LINKER_B_ID: "updated"}
    assert _content(db, LINKER_A_ID) == edited
    assert _version(db, LINKER_A_ID) == 2
    conflict = results[0]
    assert (conflict.title, conflict.version, conflict.replacements) == ("Linker A", 2, ())


def test_a_note_without_the_link_is_reported_and_untouched(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    results = rewrite_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=_targets(db, OTHER_ID)
    )

    assert _statuses(results) == {OTHER_ID: "skipped_no_match"}
    assert _version(db, OTHER_ID) == 1


def test_trashed_and_unknown_notes_are_not_found(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    unknown = "70000000-0000-4000-8000-000000000007"

    results = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=[
            WikilinkRewriteTarget(TRASHED_ID, _version(db, TRASHED_ID)),
            WikilinkRewriteTarget(unknown, 1),
        ],
    )

    assert _statuses(results) == {TRASHED_ID: "skipped_not_found", unknown: "skipped_not_found"}
    assert _content(db, TRASHED_ID) == "Gone [[Old title]]."


def test_links_another_note_still_answers_are_not_rewritten(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    db.add_note("Old title", "I also link [[Old title]].", note_id=DUPLICATE_ID)

    results = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_A_ID, DUPLICATE_ID),
    )

    # Linker A's link resolves to the duplicate. The duplicate never answers its own link.
    assert _statuses(results) == {LINKER_A_ID: "skipped_resolved", DUPLICATE_ID: "updated"}
    assert _content(db, LINKER_A_ID) == LINKER_A_TEXT
    assert _content(db, DUPLICATE_ID) == "I also link [[New title]]."


def test_duplicate_targets_are_processed_once(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    target = WikilinkRewriteTarget(LINKER_A_ID, 1)

    results = rewrite_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=[target, target]
    )

    assert [result.status for result in results] == ["updated"]
    assert _version(db, LINKER_A_ID) == 2


def test_the_batch_is_bounded(db: CharactersRAGDB) -> None:
    targets = [
        WikilinkRewriteTarget(f"note-{index}", 1)
        for index in range(wikilink_rename.MAX_WIKILINK_RENAME_NOTES + 1)
    ]

    with pytest.raises(ValueError, match="at most"):
        rewrite_title_links(db, old_title="Old title", replacement=NEW_LINK, targets=targets)


def test_a_failed_save_leaves_that_note_whole_and_the_rest_updated(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)

    def save(note: dict, content: str) -> dict:
        if note["id"] == LINKER_B_ID:
            raise CharactersRAGDBError("disk full")
        db.update_note(note["id"], {"content": content}, expected_version=int(note["version"]))
        return db.get_note_by_id(note["id"])

    results = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_B_ID, LINKER_A_ID),
        save=save,
    )

    assert _statuses(results) == {LINKER_B_ID: "failed", LINKER_A_ID: "updated"}
    # Both of Linker B's links are intact: a note is rewritten whole or not at all.
    assert _content(db, LINKER_B_ID) == LINKER_B_TEXT
    assert _version(db, LINKER_B_ID) == 1


def test_an_edit_racing_the_save_is_a_conflict(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    targets = _targets(db, LINKER_A_ID)
    raced = "A concurrent edit landed first: [[Old title]]."

    def save(note: dict, content: str) -> dict:
        # Another writer commits between this note's read and its write.
        db.update_note(note["id"], {"content": raced}, expected_version=int(note["version"]))
        db.update_note(note["id"], {"content": content}, expected_version=int(note["version"]))
        raise AssertionError("the stale write must not succeed")

    results = rewrite_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=targets, save=save
    )

    assert _statuses(results) == {LINKER_A_ID: "skipped_conflict"}
    assert _content(db, LINKER_A_ID) == raced
    assert results[0].version == 2


def test_a_note_with_more_links_than_undo_can_carry_is_not_rewritten(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_renamed_library(db)
    monkeypatch.setattr(wikilink_rename, "MAX_WIKILINK_REPLACEMENTS_PER_NOTE", 1)

    results = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_A_ID, LINKER_B_ID),
    )

    assert _statuses(results) == {LINKER_A_ID: "updated", LINKER_B_ID: "failed"}
    assert _content(db, LINKER_B_ID) == LINKER_B_TEXT


def test_a_link_too_long_for_undo_to_carry_is_not_rewritten(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    padded = "[[Old" + " " * (wikilink_rename.MAX_WIKILINK_TOKEN_TEXT_LENGTH + 1) + "title]]"
    db.update_note(LINKER_A_ID, {"content": padded}, expected_version=1)

    results = rewrite_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=_targets(db, LINKER_A_ID)
    )

    assert _statuses(results) == {LINKER_A_ID: "failed"}
    assert _content(db, LINKER_A_ID) == padded
    assert _version(db, LINKER_A_ID) == 2


def test_a_shared_new_title_rewrites_links_to_the_renamed_notes_id(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    # An older note already has the new title: [[New title]] would resolve to it.
    db.add_note("New title", "the other New title", note_id=DUPLICATE_ID)
    db.execute_query(
        "UPDATE notes SET created_at = ? WHERE id = ?", ("2001-01-01T00:00:00.000Z", DUPLICATE_ID)
    )
    plan = plan_renamed_note_link(db, RENAMED_ID)
    assert plan is not None and plan.replacement is not None

    results = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=plan.replacement,
        targets=_targets(db, LINKER_A_ID),
    )

    assert _statuses(results) == {LINKER_A_ID: "updated"}
    assert _content(db, LINKER_A_ID) == f"Intro [[id:{RENAMED_ID}]] and plain Old title."
    outgoing = db.note_graph_projection_store.list_live_outgoing(LINKER_A_ID)
    assert RENAMED_ID in outgoing
    assert DUPLICATE_ID not in outgoing


# --- undo ------------------------------------------------------------------


def test_undo_restores_the_previous_text_exactly(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    rewritten = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_A_ID, LINKER_B_ID),
    )

    restored = restore_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=_restore_targets(rewritten)
    )

    assert _statuses(restored) == {LINKER_A_ID: "restored", LINKER_B_ID: "restored"}
    assert _content(db, LINKER_A_ID) == LINKER_A_TEXT
    assert _content(db, LINKER_B_ID) == LINKER_B_TEXT
    assert [result.version for result in restored] == [3, 3]
    assert db.note_graph_projection_store.list_title_referrers(
        "Old title", exclude_note_id=RENAMED_ID
    ).total == 2


def test_undo_restores_an_id_link_rewrite(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    id_link = f"[[id:{RENAMED_ID}]]"
    rewritten = rewrite_title_links(
        db, old_title="Old title", replacement=id_link, targets=_targets(db, LINKER_B_ID)
    )

    restored = restore_title_links(
        db, old_title="Old title", replacement=id_link, targets=_restore_targets(rewritten)
    )

    assert _statuses(restored) == {LINKER_B_ID: "restored"}
    assert _content(db, LINKER_B_ID) == LINKER_B_TEXT


def test_undo_skips_a_note_edited_since_the_rewrite(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    rewritten = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_A_ID, LINKER_B_ID),
    )
    edited = "Edited after the rewrite: [[New title]]!"
    db.update_note(LINKER_A_ID, {"content": edited}, expected_version=2)

    restored = restore_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=_restore_targets(rewritten)
    )

    assert _statuses(restored) == {LINKER_A_ID: "skipped_conflict", LINKER_B_ID: "restored"}
    assert _content(db, LINKER_A_ID) == edited
    assert _content(db, LINKER_B_ID) == LINKER_B_TEXT


def test_undo_skips_trashed_notes_and_text_that_does_not_match(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    rewritten = rewrite_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_targets(db, LINKER_A_ID, LINKER_B_ID),
    )
    targets = _restore_targets(rewritten)
    db.soft_delete_note(LINKER_A_ID, expected_version=2)
    # Undo data that does not describe this note's text restores nothing.
    wrong_link = restore_title_links(
        db, old_title="Old title", replacement="[[Not what was written]]", targets=targets
    )

    assert _statuses(wrong_link) == {LINKER_A_ID: "skipped_not_found", LINKER_B_ID: "skipped_no_match"}
    assert _content(db, LINKER_B_ID) == "Twice: [[New title]] then [[New title]], not [[Old title 2]]."
    assert _version(db, LINKER_B_ID) == 2


def test_a_conflict_error_from_a_restore_save_is_a_conflict(db: CharactersRAGDB) -> None:
    _seed_renamed_library(db)
    rewritten = rewrite_title_links(
        db, old_title="Old title", replacement=NEW_LINK, targets=_targets(db, LINKER_A_ID)
    )

    def save(note: dict, content: str) -> dict:
        raise ConflictError("version changed", entity="notes", entity_id=note["id"])

    restored = restore_title_links(
        db,
        old_title="Old title",
        replacement=NEW_LINK,
        targets=_restore_targets(rewritten),
        save=save,
    )

    assert _statuses(restored) == {LINKER_A_ID: "skipped_conflict"}
    assert _content(db, LINKER_A_ID) == "Intro [[New title]] and plain Old title."
