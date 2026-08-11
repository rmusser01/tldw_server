from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_graph import EdgeType, NoteGraphRequest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.graph_cache import GraphCache
from tldw_Server_API.app.core.Notes_Graph.graph_service import (
    GraphProjectionNotReadyError,
    NoteGraphService,
)

pytestmark = pytest.mark.integration

SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
OTHER_ID = "33333333-3333-4333-8333-333333333333"
ORPHAN_ID = "44444444-4444-4444-8444-444444444444"


@pytest.fixture()
def graph_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(str(tmp_path / "graph-lifecycle.db"), client_id="owner-1")


def _service(graph_db: CharactersRAGDB, *, cache: GraphCache | None = None) -> NoteGraphService:
    return NoteGraphService(
        user_id="owner-1",
        dataset_id="dataset-default",
        db=graph_db,
        cache=cache,
    )


def test_manual_link_is_hidden_while_endpoint_is_trashed_and_reappears_on_restore(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    graph_db.add_note("Target", "plain", note_id=TARGET_ID)
    edge = graph_db.create_manual_note_edge(
        user_id="owner-1",
        from_note_id=SOURCE_ID,
        to_note_id=TARGET_ID,
        directed=False,
        weight=1.0,
        created_by="owner-1",
    )
    edge_id = str(edge["edge_id"])
    original = graph_db.notes_link_store.get(edge_id)
    assert original is not None

    request = NoteGraphRequest(
        center_note_id=SOURCE_ID,
        edge_types=[EdgeType.manual],
    )
    live = _service(graph_db).generate_graph(request)
    assert {node.id for node in live.nodes} == {SOURCE_ID, TARGET_ID}
    assert [item.type for item in live.edges] == [EdgeType.manual]

    assert graph_db.soft_delete_note(TARGET_ID, expected_version=1)
    hidden = _service(graph_db).generate_graph(request)
    assert {node.id for node in hidden.nodes} == {SOURCE_ID}
    assert hidden.edges == []
    assert graph_db.notes_link_store.get(edge_id) == original

    assert graph_db.restore_note(TARGET_ID, expected_version=2)
    restored = _service(graph_db).generate_graph(request)
    assert {node.id for node in restored.nodes} == {SOURCE_ID, TARGET_ID}
    assert [item.type for item in restored.edges] == [EdgeType.manual]
    assert graph_db.notes_link_store.get(edge_id) == original


def test_wikilinks_and_backlinks_read_only_the_persisted_projection(
    graph_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_db.add_note("Target", "plain", note_id=TARGET_ID)
    graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)

    def fail_request_time_parse(*_args, **_kwargs):
        raise AssertionError("graph reads must not parse note content")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notes.wikilinks.parse_wikilinks",
        fail_request_time_parse,
    )
    wikilink = _service(graph_db).generate_graph(
        NoteGraphRequest(center_note_id=SOURCE_ID, edge_types=[EdgeType.wikilink])
    )
    assert [(edge.source, edge.target, edge.type) for edge in wikilink.edges] == [
        (SOURCE_ID, TARGET_ID, EdgeType.wikilink)
    ]

    backlink = _service(graph_db).generate_graph(
        NoteGraphRequest(center_note_id=TARGET_ID, edge_types=[EdgeType.backlink])
    )
    assert [(edge.source, edge.target, edge.type) for edge in backlink.edges] == [
        (TARGET_ID, SOURCE_ID, EdgeType.backlink)
    ]


def test_only_manual_graph_reads_remain_available_during_projection_rebuild(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    graph_db.add_note("Target", "plain", note_id=TARGET_ID)
    graph_db.create_manual_note_edge(
        user_id="owner-1",
        from_note_id=SOURCE_ID,
        to_note_id=TARGET_ID,
        directed=False,
        weight=1.0,
        created_by="owner-1",
    )
    graph_db.execute_query("UPDATE note_graph_projection_state SET rebuild_state = 'pending' WHERE singleton_id = 1")

    manual = _service(graph_db).generate_graph(NoteGraphRequest(center_note_id=SOURCE_ID, edge_types=[EdgeType.manual]))
    assert len(manual.edges) == 1

    with pytest.raises(GraphProjectionNotReadyError):
        _service(graph_db).generate_graph(NoteGraphRequest(center_note_id=SOURCE_ID, edge_types=[EdgeType.wikilink]))
    with pytest.raises(GraphProjectionNotReadyError):
        _service(graph_db).list_orphans(limit=50, cursor=None)


def test_graph_cache_is_invalidated_by_owner_revision_without_explicit_clear(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    service = _service(graph_db, cache=cache)
    request = NoteGraphRequest(center_note_id=SOURCE_ID, edge_types=[EdgeType.manual])
    first = service.generate_graph(request)
    assert first.nodes[0].label == "Source"

    graph_db.update_note(SOURCE_ID, {"title": "Changed"}, expected_version=1)
    second = service.generate_graph(request)
    assert second.nodes[0].label == "Changed"
    assert cache.stats()["misses"] == 2


def test_source_membership_change_advances_revision_and_invalidates_cache(
    graph_db: CharactersRAGDB,
) -> None:
    character_id = graph_db.add_character_card({"name": "Graph source"})
    conversation_id = graph_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Source conversation",
            "source": "youtube",
            "external_ref": "old",
        }
    )
    note_id = graph_db.add_note(
        "Sourced note",
        "plain",
        conversation_id=conversation_id,
    )
    cache = GraphCache(ttl_seconds=60, max_keys=10)
    service = _service(graph_db, cache=cache)
    request = NoteGraphRequest(
        center_note_id=note_id,
        edge_types=[EdgeType.source_membership],
    )

    first = service.generate_graph(request)
    assert {node.id for node in first.nodes if node.type == "source"} == {"source:youtube:old"}
    assert graph_db.update_conversation(
        conversation_id,
        {"source": "web", "external_ref": "new"},
        expected_version=1,
    )
    second = service.generate_graph(request)

    assert {node.id for node in second.nodes if node.type == "source"} == {"source:web:new"}
    assert cache.stats()["misses"] == 2


def test_orphans_ignore_tags_and_sources_but_respect_live_manual_and_wikilink_edges(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)
    graph_db.add_note("Target", "plain", note_id=TARGET_ID)
    graph_db.add_note("Other", "plain", note_id=OTHER_ID)
    graph_db.add_note("Orphan", "plain", note_id=ORPHAN_ID)
    keyword_id = graph_db.add_keyword("tagged")
    graph_db.link_note_to_keyword(ORPHAN_ID, keyword_id)
    graph_db.create_manual_note_edge(
        user_id="owner-1",
        from_note_id=TARGET_ID,
        to_note_id=OTHER_ID,
        directed=False,
        weight=1.0,
        created_by="owner-1",
    )

    assert graph_db.note_graph_projection_store.list_orphan_note_ids(
        after_note_id=None,
        limit=50,
    ) == (ORPHAN_ID,)


def test_unresolved_wikilink_target_is_not_visible_and_does_not_hide_orphan(
    graph_db: CharactersRAGDB,
) -> None:
    missing_id = "99999999-9999-4999-8999-999999999999"
    graph_db.add_note("Source", f"[[id:{missing_id}]]", note_id=SOURCE_ID)

    graph = _service(graph_db).generate_graph(
        NoteGraphRequest(center_note_id=SOURCE_ID, edge_types=[EdgeType.wikilink])
    )

    assert {node.id for node in graph.nodes} == {SOURCE_ID}
    assert graph.edges == []
    assert graph_db.note_graph_projection_store.list_orphan_note_ids(
        after_note_id=None,
        limit=50,
    ) == (SOURCE_ID,)


def test_tag_and_source_compatibility_nodes_remain_visible(
    graph_db: CharactersRAGDB,
) -> None:
    character_id = graph_db.add_character_card({"name": "Compatibility source"})
    conversation_id = graph_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Compatibility conversation",
            "source": "article",
            "external_ref": "compatibility",
        }
    )
    note_id = graph_db.add_note(
        "Compatibility note",
        "plain",
        conversation_id=conversation_id,
    )
    keyword_id = graph_db.add_keyword("compatibility-tag")
    graph_db.link_note_to_keyword(note_id, keyword_id)

    graph = _service(graph_db).generate_graph(
        NoteGraphRequest(
            center_note_id=note_id,
            edge_types=[EdgeType.tag_membership, EdgeType.source_membership],
        )
    )

    assert {node.type for node in graph.nodes} == {"note", "tag", "source"}
    assert {edge.type for edge in graph.edges} == {
        EdgeType.tag_membership,
        EdgeType.source_membership,
    }
