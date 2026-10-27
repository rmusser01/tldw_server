"""Owner-scoped ``[[Title]]`` resolution in the Notes graph projection.

NE-02 (#3110), decision D2: notes link by ``[[Title]]`` or ``[[id:UUID]]``.
Titles resolve to the owner's live notes when edges are projected, and links
are re-resolved when a note is created, renamed, deleted or restored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import (
    note_graph_projection_store as store_module,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_projection_store import (
    NoteGraphProjectionStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, CharactersRAGDB
from tldw_Server_API.app.core.Notes.wikilinks import (
    is_wikilink_title_reference_key,
)
from tldw_Server_API.app.core.Notes_Graph.projection_service import (
    NoteGraphProjectionService,
)

pytestmark = pytest.mark.unit

SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
OTHER_ID = "33333333-3333-4333-8333-333333333333"
SECOND_SOURCE_ID = "44444444-4444-4444-8444-444444444444"
OLDER_ID = "99999999-9999-4999-8999-999999999999"
NEWER_ID = "00000000-0000-4000-8000-000000000000"


@pytest.fixture()
def graph_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(tmp_path / "wikilink-titles.db"), client_id="owner-1")
    try:
        yield db
    finally:
        db.close_connection()


def _set_created_at(db: CharactersRAGDB, note_id: str, created_at: str) -> None:
    db.execute_query("UPDATE notes SET created_at = ? WHERE id = ?", (created_at, note_id))
    # The direct write marks the note dirty; project it like the maintenance worker would.
    NoteGraphProjectionService(db).process_dirty()


def _version(db: CharactersRAGDB, note_id: str) -> int:
    note = db.get_note_by_id(note_id, include_deleted=True)
    assert note is not None
    return int(note["version"])


def _outgoing(db: CharactersRAGDB, note_id: str) -> tuple[str, ...]:
    return db.note_graph_projection_store.list_outgoing(note_id)


def _add_older_and_newer_duplicates(db: CharactersRAGDB, title: str = "Shared") -> None:
    # The newer note has the lower id, so "oldest wins" is not an id tiebreak.
    db.add_note(title, "older", note_id=OLDER_ID)
    db.add_note(title, "newer", note_id=NEWER_ID)
    _set_created_at(db, OLDER_ID, "2026-01-01T00:00:00.000Z")
    _set_created_at(db, NEWER_ID, "2026-02-01T00:00:00.000Z")


def test_title_link_resolves_case_and_whitespace_insensitively(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Target  Note", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", "See [[ target   NOTE ]].", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)
    assert graph_db.note_graph_projection_store.list_live_outgoing(SOURCE_ID) == (TARGET_ID,)


def test_title_with_brackets_resolves(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("[Draft] Proposal", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", "Read [[[Draft] Proposal]].", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)


def test_title_with_like_wildcards_matches_only_the_literal_title(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("100% done_ok", "plain", note_id=TARGET_ID)
    graph_db.add_note("1000 doneXok", "plain", note_id=OTHER_ID)
    graph_db.add_note("Source", "[[100% done_ok]]", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)


def test_unresolved_title_creates_no_edge_until_the_note_is_created(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Source", "Plan: [[Future Note]]", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == ()

    # The WebUI's "create note" action for an unresolved link ends in add_note.
    graph_db.add_note("Future Note", "now it exists", note_id=TARGET_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)
    assert graph_db.note_graph_projection_store.count_dirty() == 0


def test_ambiguous_title_resolves_to_the_oldest_note(graph_db: CharactersRAGDB) -> None:
    _add_older_and_newer_duplicates(graph_db)
    graph_db.add_note("Source", "[[Shared]]", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (OLDER_ID,)


def test_creating_a_duplicate_title_does_not_steal_existing_links(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Shared", "first", note_id=OLDER_ID)
    _set_created_at(graph_db, OLDER_ID, "2026-01-01T00:00:00.000Z")
    graph_db.add_note("Source", "[[Shared]]", note_id=SOURCE_ID)

    graph_db.add_note("Shared", "duplicate", note_id=NEWER_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (OLDER_ID,)


def test_exact_case_title_match_wins_over_an_older_case_variant(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("shared", "lower case", note_id=OLDER_ID)
    _set_created_at(graph_db, OLDER_ID, "2026-01-01T00:00:00.000Z")
    graph_db.add_note("Source", "[[Shared]]", note_id=SOURCE_ID)
    assert _outgoing(graph_db, SOURCE_ID) == (OLDER_ID,)

    graph_db.add_note("Shared", "exact case", note_id=NEWER_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (NEWER_ID,)


def test_self_title_link_skips_the_linking_note(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Weekly sync", "Previous: [[Weekly sync]]", note_id=SOURCE_ID)
    assert _outgoing(graph_db, SOURCE_ID) == ()

    graph_db.add_note("Weekly sync", "an earlier sync", note_id=OTHER_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (OTHER_ID,)
    assert _outgoing(graph_db, OTHER_ID) == ()


def test_id_and_title_links_to_the_same_note_project_one_edge(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Target Note", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", f"[[id:{TARGET_ID}]] and [[Target Note]]", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)


def test_id_link_still_targets_its_note_regardless_of_title(graph_db: CharactersRAGDB) -> None:
    _add_older_and_newer_duplicates(graph_db)
    graph_db.add_note("Source", f"[[id:{NEWER_ID}]]", note_id=SOURCE_ID)

    assert _outgoing(graph_db, SOURCE_ID) == (NEWER_ID,)


def test_title_reference_keys_never_surface_as_edges(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Target Note", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", "[[Target Note]] [[Missing]]", note_id=SOURCE_ID)
    store = graph_db.note_graph_projection_store

    edges = store.list_live_edges_for_notes([SOURCE_ID, TARGET_ID])

    assert [(edge.source_note_id, edge.target_note_id) for edge in edges] == [(SOURCE_ID, TARGET_ID)]
    assert not any(is_wikilink_title_reference_key(target) for target in _outgoing(graph_db, SOURCE_ID))
    assert store.list_orphan_note_ids(after_note_id=None, limit=10) == ()


def test_renaming_a_note_moves_title_links(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Alpha", "plain", note_id=TARGET_ID)
    graph_db.add_note("Links alpha", "[[Alpha]]", note_id=SOURCE_ID)
    graph_db.add_note("Links beta", "[[Beta]]", note_id=SECOND_SOURCE_ID)
    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)
    assert _outgoing(graph_db, SECOND_SOURCE_ID) == ()

    graph_db.update_note(TARGET_ID, {"title": "Beta"}, expected_version=_version(graph_db, TARGET_ID))

    # Links are by title: the old title is now unresolved, the new title resolves.
    assert _outgoing(graph_db, SOURCE_ID) == ()
    assert _outgoing(graph_db, SECOND_SOURCE_ID) == (TARGET_ID,)
    assert graph_db.note_graph_projection_store.count_dirty() == 0


def test_soft_delete_and_restore_reresolve_title_links(graph_db: CharactersRAGDB) -> None:
    _add_older_and_newer_duplicates(graph_db)
    graph_db.add_note("Source", "[[Shared]]", note_id=SOURCE_ID)
    assert _outgoing(graph_db, SOURCE_ID) == (OLDER_ID,)

    graph_db.soft_delete_note(OLDER_ID, expected_version=_version(graph_db, OLDER_ID))
    assert _outgoing(graph_db, SOURCE_ID) == (NEWER_ID,)

    graph_db.restore_note(OLDER_ID, expected_version=_version(graph_db, OLDER_ID))
    assert _outgoing(graph_db, SOURCE_ID) == (OLDER_ID,)
    assert graph_db.note_graph_projection_store.count_dirty() == 0


def test_hard_delete_reresolves_title_links(graph_db: CharactersRAGDB) -> None:
    _add_older_and_newer_duplicates(graph_db)
    graph_db.add_note("Source", "[[Shared]]", note_id=SOURCE_ID)

    assert graph_db.delete_note(OLDER_ID, hard_delete=True) is True

    assert _outgoing(graph_db, SOURCE_ID) == (NEWER_ID,)


def test_last_matching_note_deleted_leaves_the_link_unresolved(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Target Note", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", "[[Target Note]]", note_id=SOURCE_ID)

    graph_db.delete_note(TARGET_ID, expected_version=_version(graph_db, TARGET_ID))

    assert _outgoing(graph_db, SOURCE_ID) == ()


def test_sync_rename_reresolves_title_links(graph_db: CharactersRAGDB) -> None:
    graph_db.add_note("Alpha", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", "[[Beta]]", note_id=SOURCE_ID)

    graph_db.upsert_note_from_sync(
        note_id=TARGET_ID,
        title="Beta",
        content="plain",
        conversation_id=None,
        message_id=None,
        sync_client_id=graph_db.client_id,
        object_revision=5,
        object_hash="sha256:test",
    )

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)


def test_projection_rebuild_resolves_title_links_written_outside_note_store(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Target Note", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    # A direct write (for example, content projected before the parser upgrade)
    # only marks the note dirty; the maintenance pass projects it.
    graph_db.execute_query(
        "UPDATE notes SET content = ?, version = version + 1 WHERE id = ?",
        ("Now links [[Target Note]]", SOURCE_ID),
    )
    assert graph_db.note_graph_projection_store.count_dirty() == 1

    NoteGraphProjectionService(graph_db).process_dirty()

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)
    assert graph_db.note_graph_projection_store.count_dirty() == 0


def test_resolve_wikilink_titles_reports_targets_and_candidate_counts(graph_db: CharactersRAGDB) -> None:
    _add_older_and_newer_duplicates(graph_db)
    graph_db.add_note("Target Note", "plain", note_id=TARGET_ID)

    resolved = graph_db.note_graph_projection_store.resolve_wikilink_titles(
        ["shared", "Target  Note", "Missing", ""],
        exclude_note_id=TARGET_ID,
    )

    assert resolved["shared"].note_id == OLDER_ID
    assert resolved["shared"].note_title == "Shared"
    assert resolved["shared"].candidate_count == 2
    # The excluded (linking) note never answers its own title link.
    assert resolved["Target  Note"].note_id is None
    assert resolved["Target  Note"].candidate_count == 0
    assert resolved["Missing"].note_id is None
    assert "" not in resolved


def test_rename_beyond_the_inline_bound_queues_remaining_referrers(
    graph_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "MAX_INLINE_TITLE_REFERRER_REFRESH", 1)
    graph_db.add_note("Alpha", "plain", note_id=TARGET_ID)
    graph_db.add_note("First", "[[Beta]]", note_id=SOURCE_ID)
    graph_db.add_note("Second", "[[Beta]]", note_id=SECOND_SOURCE_ID)
    store = graph_db.note_graph_projection_store

    graph_db.update_note(TARGET_ID, {"title": "Beta"}, expected_version=_version(graph_db, TARGET_ID))

    assert _outgoing(graph_db, SOURCE_ID) == (TARGET_ID,)
    assert _outgoing(graph_db, SECOND_SOURCE_ID) == ()
    assert store.count_dirty() == 1

    NoteGraphProjectionService(graph_db).process_dirty()

    assert _outgoing(graph_db, SECOND_SOURCE_ID) == (TARGET_ID,)
    assert store.count_dirty() == 0


def test_postgres_title_lookup_is_owner_scoped_and_live_only() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        @staticmethod
        def fetchall() -> list[dict[str, object]]:
            return [{"id": TARGET_ID, "title": "Target Note", "created_at": "2026-01-01T00:00:00Z"}]

    class _Connection:
        @staticmethod
        def execute(query: str, params: tuple[object, ...]) -> _Cursor:
            calls.append((query, params))
            return _Cursor()

    class _DB:
        backend_type = BackendType.POSTGRESQL
        client_id = "owner-1"

    resolved = NoteGraphProjectionStore(_DB()).resolve_wikilink_titles(  # type: ignore[arg-type]
        ["target note"],
        exclude_note_id=SOURCE_ID,
        conn=_Connection(),
    )

    assert resolved["target note"].note_id == TARGET_ID
    query, params = calls[0]
    assert "deleted = ?" in query
    assert "client_id = ?" in query
    assert "id <> ?" in query
    assert params[0] is False
    assert params[1] == "%target%note%"
    assert params[-2:] == ("owner-1", SOURCE_ID)
