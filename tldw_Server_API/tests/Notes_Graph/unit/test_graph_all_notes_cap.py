"""The All-notes graph holds up to 500 notes (NL-11, decision 15, #3101).

The cap was 100, and every response clamped it to that response's own
``max_nodes``, so a focused graph that asked for 120 nodes reported a cap of
at most 120. One setting now governs the All-notes graph:
``NOTES_GRAPH_ALL_NOTES_NOTE_CAP`` (500 by default). A graph with no focus note
(All notes, or a tag or source filter) may hold every note under that cap
plus its tags and sources; a focused graph keeps its own limits.
"""

import uuid

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_graph import NoteGraphRequest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.Notes_Graph import graph_service as graph_service_module
from tldw_Server_API.app.core.Notes_Graph.graph_service import NoteGraphService
from tldw_Server_API.tests.Notes_Graph.unit.test_graph_service import _mock_db, _note

pytestmark = pytest.mark.unit

_LIBRARY_REQUEST_MAX_NODES = 1_000


def _library(count: int) -> tuple[list[str], list[dict]]:
    ids = sorted(str(uuid.uuid4()) for _ in range(count))
    return ids, [_note(note_id) for note_id in ids]


@pytest.fixture(autouse=True)
def _default_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NOTES_GRAPH_ALL_NOTES_NOTE_CAP",
        "NOTES_GRAPH_MAX_NODES",
        "NOTES_GRAPH_MAX_EDGES",
        "NOTES_GRAPH_MAX_DEGREE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_all_notes_cap_defaults_to_500() -> None:
    assert graph_service_module.ALL_NOTES_NOTE_CAP() == 500


def test_all_notes_graph_shows_every_note_of_a_500_note_library() -> None:
    ids, notes = _library(500)
    db = _mock_db(notes=notes, note_count=500, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(
        NoteGraphRequest(radius=1, max_nodes=_LIBRARY_REQUEST_MAX_NODES)
    )

    assert len([node for node in graph.nodes if node.type == "note"]) == 500
    assert graph.truncated is False
    assert graph.all_notes_note_cap == 500
    assert graph.all_notes_eligible is True


def test_focused_graph_reports_the_all_notes_cap_not_its_own_node_limit() -> None:
    ids, notes = _library(300)
    db = _mock_db(notes=notes, note_count=300, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(
        NoteGraphRequest(center_note_id=ids[0], radius=1, max_nodes=120)
    )

    assert graph.limits.max_nodes == 120
    assert graph.all_notes_note_cap == 500
    assert graph.all_notes_eligible is True


def test_all_notes_graph_is_refused_above_the_cap() -> None:
    ids, notes = _library(501)
    db = _mock_db(notes=notes, note_count=501, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    with pytest.raises(InputError):
        service.generate_graph(
            NoteGraphRequest(radius=1, max_nodes=_LIBRARY_REQUEST_MAX_NODES)
        )


def test_the_cap_setting_moves_both_eligibility_and_the_seedless_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTES_GRAPH_ALL_NOTES_NOTE_CAP", "40")
    ids, notes = _library(41)
    db = _mock_db(notes=notes, note_count=41, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    focused = service.generate_graph(NoteGraphRequest(center_note_id=ids[0], radius=1))
    assert focused.all_notes_note_cap == 40
    assert focused.all_notes_eligible is False
    with pytest.raises(InputError):
        service.generate_graph(
            NoteGraphRequest(radius=1, max_nodes=_LIBRARY_REQUEST_MAX_NODES)
        )


def test_a_library_graph_leaves_room_for_tags_and_sources() -> None:
    service = NoteGraphService(user_id="u1", db=_mock_db(note_count=0))

    library_nodes, _, _, _ = service._resolve_effective_limits(
        NoteGraphRequest(radius=1, max_nodes=_LIBRARY_REQUEST_MAX_NODES)
    )
    focused_nodes, _, _, _ = service._resolve_effective_limits(
        NoteGraphRequest(
            center_note_id=str(uuid.uuid4()),
            radius=1,
            max_nodes=_LIBRARY_REQUEST_MAX_NODES,
        )
    )

    assert library_nodes == 500 + graph_service_module._TAG_CAP + graph_service_module._SOURCE_CAP
    assert focused_nodes == graph_service_module.MAX_NODES()


def test_a_tag_filtered_library_graph_holds_more_than_the_focused_note_cap() -> None:
    ids, notes = _library(400)
    db = _mock_db(notes=notes, note_count=2_000, all_ids=ids, tag_seed_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(
        NoteGraphRequest(radius=1, tag="reading", max_nodes=_LIBRARY_REQUEST_MAX_NODES)
    )

    assert len([node for node in graph.nodes if node.type == "note"]) == 400
    assert graph.all_notes_eligible is False
