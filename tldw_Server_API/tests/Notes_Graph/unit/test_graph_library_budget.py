"""Library graphs (no focus note): what they hold and what bounds them (NL-11 review, #3101).

* A tag or source filter shows exactly the notes with that tag or source.
  It used to expand one hop from them, so 50 tagged notes each linking 3
  untagged notes showed 177 notes, and the neighbours could push tagged
  notes out of the cut.
* A library graph gets an edge budget to match its node budget (4 links per
  node), not the focused graph's 1,200.
* An operator who sets NOTES_GRAPH_MAX_NODES (or _MAX_EDGES) lower than the
  library budget keeps that limit; the All-notes cap shrinks to fit and the
  response reports it.
"""

import uuid

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_graph import EdgeType, NoteGraphRequest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.Notes_Graph import graph_service as graph_service_module
from tldw_Server_API.app.core.Notes_Graph.graph_service import NoteGraphService
from tldw_Server_API.tests.Notes_Graph.unit.test_graph_service import (
    _manual_edge,
    _mock_db,
    _note,
)

pytestmark = pytest.mark.unit

_LIBRARY = 1_000
_LIBRARY_EDGES = 4_000
_BUDGET = 500 + graph_service_module._TAG_CAP + graph_service_module._SOURCE_CAP


@pytest.fixture(autouse=True)
def _default_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NOTES_GRAPH_ALL_NOTES_NOTE_CAP",
        "NOTES_GRAPH_MAX_NODES",
        "NOTES_GRAPH_MAX_EDGES",
        "NOTES_GRAPH_MAX_DEGREE",
    ):
        monkeypatch.delenv(name, raising=False)


def _ids(count: int) -> list[str]:
    return sorted(str(uuid.uuid4()) for _ in range(count))


def _note_at(note_id: str, modified: str) -> dict:
    return {**_note(note_id), "last_modified": modified}


def _tagged_library(tagged_count: int, links_each: int):
    """Tagged notes (older) each manually linked to their own untagged notes (newer)."""
    tagged = _ids(tagged_count)
    untagged: list[str] = []
    edges = []
    for note_id in tagged:
        for _ in range(links_each):
            neighbour = str(uuid.uuid4())
            untagged.append(neighbour)
            edges.append(_manual_edge(note_id, neighbour))
    notes = [_note_at(note_id, "2025-01-01T00:00:00") for note_id in tagged]
    notes += [_note_at(note_id, "2026-06-01T00:00:00") for note_id in untagged]
    return tagged, untagged, notes, edges


def _note_ids(graph) -> set[str]:
    return {node.id for node in graph.nodes if node.type == "note"}


def test_a_tag_filter_shows_exactly_the_tagged_notes() -> None:
    tagged, _, notes, edges = _tagged_library(50, 3)
    db = _mock_db(notes=notes, edges=edges, note_count=len(notes), tag_seed_ids=tagged)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(
        NoteGraphRequest(radius=1, tag="reading", max_nodes=_LIBRARY, max_edges=_LIBRARY_EDGES)
    )

    assert _note_ids(graph) == set(tagged)


def test_tagged_notes_survive_the_cut_ahead_of_newer_neighbours() -> None:
    tagged, untagged, notes, edges = _tagged_library(4, 2)
    db = _mock_db(notes=notes, edges=edges, note_count=len(notes), tag_seed_ids=tagged)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(NoteGraphRequest(radius=1, tag="reading", max_nodes=4))

    assert _note_ids(graph) == set(tagged)
    assert not _note_ids(graph) & set(untagged)


def test_a_tag_filter_keeps_links_among_the_tagged_notes() -> None:
    tagged = _ids(3)
    outsider = str(uuid.uuid4())
    edges = [_manual_edge(tagged[0], tagged[1]), _manual_edge(tagged[1], outsider)]
    notes = [_note(note_id) for note_id in [*tagged, outsider]]
    db = _mock_db(notes=notes, edges=edges, note_count=4, tag_seed_ids=tagged)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(
        NoteGraphRequest(radius=1, tag="reading", edge_types=[EdgeType.manual], max_nodes=_LIBRARY)
    )

    assert _note_ids(graph) == set(tagged)
    assert {(edge.source, edge.target) for edge in graph.edges} == {(tagged[0], tagged[1])}


def test_a_source_filter_shows_exactly_the_notes_from_that_source() -> None:
    sourced, _, notes, edges = _tagged_library(5, 2)
    db = _mock_db(notes=notes, edges=edges, note_count=len(notes), source_seed_ids=sourced)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(NoteGraphRequest(radius=1, source="web", max_nodes=_LIBRARY))

    assert _note_ids(graph) == set(sourced)


def test_a_library_graph_gets_four_links_per_node() -> None:
    service = NoteGraphService(user_id="u1", db=_mock_db(note_count=0))

    nodes, edges, _, _ = service._resolve_effective_limits(
        NoteGraphRequest(radius=1, max_nodes=_LIBRARY, max_edges=_LIBRARY_EDGES)
    )
    _, focused_edges, _, _ = service._resolve_effective_limits(
        NoteGraphRequest(
            center_note_id=str(uuid.uuid4()),
            radius=1,
            max_nodes=_LIBRARY,
            max_edges=_LIBRARY_EDGES,
        )
    )

    assert nodes == _BUDGET
    assert edges == 4 * _BUDGET
    assert edges <= _LIBRARY_EDGES
    assert focused_edges == graph_service_module.MAX_EDGES()


def test_an_operator_max_nodes_below_the_budget_bounds_the_library_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTES_GRAPH_MAX_NODES", "300")
    monkeypatch.setenv("NOTES_GRAPH_MAX_EDGES", "1000")
    ids = _ids(10)
    db = _mock_db(notes=[_note(note_id) for note_id in ids], note_count=10, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    graph = service.generate_graph(
        NoteGraphRequest(radius=1, max_nodes=_LIBRARY, max_edges=_LIBRARY_EDGES)
    )

    assert graph.limits.max_nodes == 300
    assert graph.limits.max_edges == 1000
    assert graph.all_notes_note_cap == 300


def test_an_operator_max_nodes_above_the_budget_raises_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTES_GRAPH_MAX_NODES", "800")
    service = NoteGraphService(user_id="u1", db=_mock_db(note_count=0))

    nodes, _, _, _ = service._resolve_effective_limits(
        NoteGraphRequest(radius=1, max_nodes=_LIBRARY)
    )

    assert nodes == 800


def test_a_library_over_the_cap_is_refused_naming_the_cap() -> None:
    ids = _ids(501)
    db = _mock_db(notes=[_note(note_id) for note_id in ids], note_count=501, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    with pytest.raises(InputError, match="All-notes cap of 500"):
        service.generate_graph(NoteGraphRequest(radius=1, max_nodes=_LIBRARY))


def test_a_request_too_small_for_the_library_says_so() -> None:
    ids = _ids(120)
    db = _mock_db(notes=[_note(note_id) for note_id in ids], note_count=120, all_ids=ids)
    service = NoteGraphService(user_id="u1", db=db)

    with pytest.raises(InputError, match="max_nodes=50 is below this library's 120 notes"):
        service.generate_graph(NoteGraphRequest(radius=1, max_nodes=50))
