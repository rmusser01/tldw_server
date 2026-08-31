"""Tests for NoteGraphService."""

import base64
import hashlib
import json
import uuid
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    NoteGraphRequest,
    TimeRange,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_projection_store import (
    ProjectionStatus,
    WikilinkProjectionEdge,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
from tldw_Server_API.app.core.Notes.wikilinks import parse_wikilinks
from tldw_Server_API.app.core.Notes_Graph import graph_service as graph_service_module
from tldw_Server_API.app.core.Notes_Graph.graph_cache import GraphCache
from tldw_Server_API.app.core.Notes_Graph.graph_service import (
    NoteGraphService,
    _decode_cursor,
    _encode_cursor,
)

pytestmark = pytest.mark.unit


def _uid():
    return str(uuid.uuid4())


def _mock_db(notes=None, edges=None, tag_edges=None, tag_counts=None,
             source_info=None, note_count=0, all_ids=None,
             tag_seed_ids=None, source_seed_ids=None,
             projection_state="ready", projection_dirty=0):
    """Build a MagicMock CharactersRAGDB with graph helpers."""
    db = MagicMock()
    _notes = notes or []
    _note_map = {n["id"]: n for n in _notes}

    class _ProjectionStore:
        def get_projection_status(self):
            return ProjectionStatus(1, projection_state, None)

        def get_revision(self):
            return 0

        def count_dirty(self):
            return projection_dirty

        def list_live_edges_for_notes(self, note_ids):
            wanted_ids = set(note_ids)
            projected = []
            for source_id, row in _note_map.items():
                if row.get("deleted"):
                    continue
                for target_id in parse_wikilinks(
                    str(row.get("content") or ""),
                    source_note_id=source_id,
                ).target_note_ids:
                    target = _note_map.get(target_id)
                    if target is None or target.get("deleted"):
                        continue
                    if source_id in wanted_ids or target_id in wanted_ids:
                        projected.append(WikilinkProjectionEdge(source_id, target_id))
            return tuple(projected)

    def get_notes_batch(ids, include_deleted=True):
        return [_note_map[i] for i in ids if i in _note_map]

    def get_manual_edges_for_notes(user_id, note_ids):
        if edges is None:
            return []
        return [e for e in edges if e["from_note_id"] in note_ids or e["to_note_id"] in note_ids]

    def get_all_note_ids_for_graph(include_deleted=True, limit=500):
        return (all_ids or list(_note_map))[:limit]

    def get_note_tag_edges(note_ids):
        if tag_edges is None:
            return []
        return [t for t in tag_edges if t["note_id"] in note_ids]

    def count_notes_per_tag():
        return tag_counts or {}

    def get_note_source_info(note_ids):
        if source_info is None:
            return []
        return [s for s in source_info if s["note_id"] in note_ids]

    def count_user_notes(include_deleted=True):
        return note_count

    def get_note_ids_by_tag_for_graph(tag, include_deleted=True, limit=500):
        if tag_seed_ids is not None:
            return list(tag_seed_ids)[:limit]
        if tag_edges is None:
            return []
        tagged = [
            row["note_id"]
            for row in tag_edges
            if str(row.get("keyword", "")).lower() == str(tag).lower()
        ]
        return tagged[:limit]

    def get_note_ids_by_source_for_graph(source, include_deleted=True, limit=500):
        if source_seed_ids is not None:
            return list(source_seed_ids)[:limit]
        if source_info is None:
            return []
        source_text = str(source)
        if source_text.startswith("source:"):
            source_text = source_text[len("source:"):]
        parts = source_text.split(":", 1)
        wanted_source = parts[0]
        wanted_ref = parts[1] if len(parts) == 2 else None
        sourced = [
            row["note_id"]
            for row in source_info
            if row.get("source") == wanted_source
            and (wanted_ref is None or row.get("external_ref") == wanted_ref)
        ]
        return sourced[:limit]

    db.get_notes_batch = MagicMock(side_effect=get_notes_batch)
    db.get_manual_edges_for_notes = MagicMock(side_effect=get_manual_edges_for_notes)
    db.get_all_note_ids_for_graph = MagicMock(side_effect=get_all_note_ids_for_graph)
    db.get_note_tag_edges = MagicMock(side_effect=get_note_tag_edges)
    db.count_notes_per_tag = MagicMock(side_effect=count_notes_per_tag)
    db.get_note_source_info = MagicMock(side_effect=get_note_source_info)
    db.count_user_notes = MagicMock(side_effect=count_user_notes)
    db.get_note_ids_by_tag_for_graph = MagicMock(side_effect=get_note_ids_by_tag_for_graph)
    db.get_note_ids_by_source_for_graph = MagicMock(side_effect=get_note_ids_by_source_for_graph)
    db.note_graph_projection_store = _ProjectionStore()
    return db


def _note(nid, title="N", content="body", deleted=0, conv_id=None):
    return {
        "id": nid, "title": title, "content": content,
        "created_at": "2025-01-01T00:00:00", "last_modified": "2025-06-01T00:00:00",
        "deleted": deleted, "conversation_id": conv_id,
    }


def _manual_edge(from_id, to_id, user_id="u1", directed=False, weight=1.0):
    return {
        "edge_id": _uid(), "user_id": user_id,
        "from_note_id": from_id, "to_note_id": to_id,
        "type": "manual", "directed": int(directed), "weight": weight,
        "created_at": "2025-01-01T00:00:00", "created_by": "test", "metadata": None,
    }


class TestEgoGraphRadius1:
    def test_center_with_3_neighbors(self):
        center = _uid()
        n1, n2, n3 = _uid(), _uid(), _uid()
        notes = [_note(center), _note(n1), _note(n2), _note(n3)]
        edges = [
            _manual_edge(center, n1),
            _manual_edge(center, n2),
            _manual_edge(center, n3),
        ]
        db = _mock_db(notes=notes, edges=edges, note_count=4)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(center_note_id=center, radius=1)
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert center in note_ids
        assert {n1, n2, n3} <= note_ids
        assert len(resp.edges) == 3


class TestEgoGraphRadius2:
    def test_two_layers(self):
        c = _uid()
        n1 = _uid()
        n2 = _uid()  # 2nd-hop neighbor
        notes = [_note(c), _note(n1), _note(n2)]
        edges_r1 = [_manual_edge(c, n1)]
        edges_r2 = [_manual_edge(n1, n2)]
        all_edges = edges_r1 + edges_r2
        db = _mock_db(notes=notes, edges=all_edges, note_count=3)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(center_note_id=c, radius=2)
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert {c, n1, n2} == note_ids


class TestSeedlessSmallCollection:
    def test_full_graph(self):
        ids = [_uid() for _ in range(5)]
        notes = [_note(i) for i in ids]
        db = _mock_db(notes=notes, note_count=5, all_ids=ids)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1)
        resp = svc.generate_graph(req)
        assert len([n for n in resp.nodes if n.type == "note"]) == 5


class TestSeedlessLargeRejected:
    def test_422_error(self):
        db = _mock_db(note_count=500)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1)
        with pytest.raises(InputError):
            svc.generate_graph(req)


def test_projection_query_node_limit_is_never_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_service_module, "MAX_NODES", lambda: 1_200)
    db = _mock_db(note_count=0)
    service = NoteGraphService(user_id="u1", db=db, allow_heavy_limits=True)

    max_nodes, _, _, _ = service._resolve_effective_limits(
        NoteGraphRequest(radius=1, allow_heavy=True, max_nodes=2_000)
    )

    assert max_nodes == 1_000


class TestTagFilterSeeds:
    def test_correct_seed_set(self):
        n1, n2, n3 = _uid(), _uid(), _uid()
        notes = [_note(n1), _note(n2), _note(n3)]
        tag_edges = [
            {"note_id": n1, "keyword_id": 1, "keyword": "ml"},
            {"note_id": n2, "keyword_id": 1, "keyword": "ml"},
        ]
        db = _mock_db(notes=notes, tag_edges=tag_edges, note_count=3, all_ids=[n1, n2, n3])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(tag="ml", radius=1)
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert n1 in note_ids
        assert n2 in note_ids

    def test_uses_direct_tag_seed_lookup_beyond_recent_window(self):
        matching = _uid()
        recent_unrelated = [_uid(), _uid()]
        notes = [_note(matching, title="Older tagged note")]
        db = _mock_db(
            notes=notes,
            note_count=500,
            all_ids=recent_unrelated,
            tag_seed_ids=[matching],
        )
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(tag="ml", radius=1)

        resp = svc.generate_graph(req)

        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert matching in note_ids
        db.get_note_ids_by_tag_for_graph.assert_called_once_with(
            "ml", include_deleted=False, limit=300,
        )


class TestSourceFilterSeeds:
    def test_uses_direct_source_seed_lookup_beyond_recent_window(self):
        matching = _uid()
        recent_unrelated = [_uid(), _uid()]
        notes = [_note(matching, title="Older sourced note")]
        source_info = [
            {
                "note_id": matching,
                "conversation_id": _uid(),
                "source": "youtube",
                "external_ref": "abc123",
            }
        ]
        db = _mock_db(
            notes=notes,
            source_info=source_info,
            note_count=500,
            all_ids=recent_unrelated,
            source_seed_ids=[matching],
        )
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(source="source:youtube:abc123", radius=1)

        resp = svc.generate_graph(req)

        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert matching in note_ids
        db.get_note_ids_by_source_for_graph.assert_called_once_with(
            "source:youtube:abc123", include_deleted=False, limit=300,
        )


class TestWikilinkEdges:
    def test_wikilink_in_content(self):
        n1 = _uid()
        n2 = _uid()
        notes = [
            _note(n1, content=f"See [[id:{n2}]] for details"),
            _note(n2),
        ]
        db = _mock_db(notes=notes, note_count=2, all_ids=[n1, n2])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.wikilink])
        resp = svc.generate_graph(req)
        wl = [e for e in resp.edges if e.type == EdgeType.wikilink]
        assert len(wl) == 1
        assert wl[0].source == n1
        assert wl[0].target == n2
        assert wl[0].directed is True


class TestBacklinkEdges:
    def test_reverse_of_wikilinks(self):
        n1 = _uid()
        n2 = _uid()
        notes = [
            _note(n1, content=f"Ref [[id:{n2}]]"),
            _note(n2),
        ]
        db = _mock_db(notes=notes, note_count=2, all_ids=[n1, n2])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.backlink])
        resp = svc.generate_graph(req)
        bl = [e for e in resp.edges if e.type == EdgeType.backlink]
        assert len(bl) == 1
        assert bl[0].source == n2
        assert bl[0].target == n1


class TestTagMembershipEdges:
    def test_from_note_keywords(self):
        n1 = _uid()
        notes = [_note(n1)]
        tag_edges = [{"note_id": n1, "keyword_id": 10, "keyword": "ai"}]
        db = _mock_db(notes=notes, tag_edges=tag_edges, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.tag_membership])
        resp = svc.generate_graph(req)
        tm = [e for e in resp.edges if e.type == EdgeType.tag_membership]
        assert len(tm) == 1
        tag_n = [n for n in resp.nodes if n.type == "tag"]
        assert len(tag_n) == 1
        assert tag_n[0].label == "ai"


class TestPopularTagCutoff:
    def test_excluded(self):
        n1 = _uid()
        notes = [_note(n1)]
        tag_edges = [{"note_id": n1, "keyword_id": 5, "keyword": "popular"}]
        # 30 out of 100 notes = 30% > 15% cutoff, and 30 >= 25 absolute min
        tag_counts = {5: 30}
        db = _mock_db(
            notes=notes, tag_edges=tag_edges, tag_counts=tag_counts,
            note_count=100, all_ids=[n1],
        )
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.tag_membership])
        resp = svc.generate_graph(req)
        tm = [e for e in resp.edges if e.type == EdgeType.tag_membership]
        assert len(tm) == 0

    def test_not_excluded_below_absolute(self):
        n1 = _uid()
        notes = [_note(n1)]
        tag_edges = [{"note_id": n1, "keyword_id": 5, "keyword": "semi"}]
        # 20% > 15% cutoff, but 20 < 25 absolute min → NOT excluded
        tag_counts = {5: 20}
        db = _mock_db(
            notes=notes, tag_edges=tag_edges, tag_counts=tag_counts,
            note_count=100, all_ids=[n1],
        )
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.tag_membership])
        resp = svc.generate_graph(req)
        tm = [e for e in resp.edges if e.type == EdgeType.tag_membership]
        assert len(tm) == 1


class TestSourceMembershipEdges:
    def test_from_conversations(self):
        n1 = _uid()
        conv_id = _uid()
        notes = [_note(n1, conv_id=conv_id)]
        source_info = [
            {"note_id": n1, "conversation_id": conv_id, "source": "youtube", "external_ref": "abc123"},
        ]
        db = _mock_db(notes=notes, source_info=source_info, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.source_membership])
        resp = svc.generate_graph(req)
        sm = [e for e in resp.edges if e.type == EdgeType.source_membership]
        assert len(sm) == 1
        src_n = [n for n in resp.nodes if n.type == "source"]
        assert len(src_n) == 1
        assert src_n[0].id == "source:youtube:abc123"


class TestMaxDegreeEnforced:
    def test_excess_neighbors_trimmed(self):
        center = _uid()
        neighbors = [_uid() for _ in range(10)]
        notes = [_note(center)] + [_note(n) for n in neighbors]
        edges = [_manual_edge(center, n) for n in neighbors]
        db = _mock_db(notes=notes, edges=edges, note_count=11)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(center_note_id=center, radius=1, max_degree=3)
        resp = svc.generate_graph(req)
        # center + 3 max neighbors
        note_count = len([n for n in resp.nodes if n.type == "note"])
        assert note_count <= 4

    def test_dense_neighborhood_keeps_query_count_radius_bounded(self):
        center = _uid()
        neighbors = [_uid() for _ in range(500)]
        notes = [_note(center)] + [_note(note_id) for note_id in neighbors]
        edges = [_manual_edge(center, note_id) for note_id in neighbors]
        db = _mock_db(notes=notes, edges=edges, note_count=len(notes))

        response = NoteGraphService(user_id="u1", db=db).generate_graph(
            NoteGraphRequest(center_note_id=center, radius=1, max_degree=7)
        )

        assert len([node for node in response.nodes if node.type == "note"]) == 8
        assert response.truncated is True
        assert "max_degree" in response.truncated_by
        assert db.get_manual_edges_for_notes.call_count == 1
        assert db.get_notes_batch.call_count == 1


class TestLimitHardCaps:
    def test_non_heavy_request_limits_are_clamped_to_configured_caps(self, monkeypatch):
        monkeypatch.setenv("NOTES_GRAPH_MAX_NODES", "3")
        monkeypatch.setenv("NOTES_GRAPH_MAX_EDGES", "4")
        monkeypatch.setenv("NOTES_GRAPH_MAX_DEGREE", "2")
        n1 = _uid()
        db = _mock_db(notes=[_note(n1)], note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(
            radius=1,
            max_nodes=999,
            max_edges=999,
            max_degree=999,
        )

        resp = svc.generate_graph(req)

        assert resp.limits.max_nodes == 3
        assert resp.limits.max_edges == 4
        assert resp.limits.max_degree == 2


class TestMaxNodesTruncation:
    def test_truncated_true(self):
        center = _uid()
        neighbors = [_uid() for _ in range(10)]
        notes = [_note(center)] + [_note(n) for n in neighbors]
        edges = [_manual_edge(center, n) for n in neighbors]
        db = _mock_db(notes=notes, edges=edges, note_count=11)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(center_note_id=center, radius=1, max_nodes=5)
        resp = svc.generate_graph(req)
        assert resp.truncated is True
        assert "max_nodes" in resp.truncated_by


class TestRadius2Caps:
    def test_stricter_limits(self):
        center = _uid()
        notes = [_note(center)]
        db = _mock_db(notes=notes, note_count=1)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(center_note_id=center, radius=2, max_nodes=300)
        resp = svc.generate_graph(req)
        assert resp.limits.max_nodes == 200
        assert resp.limits.max_edges == 800
        assert resp.limits.max_degree == 20
        assert resp.radius_cap_applied is True


class TestCursorPagination:
    def test_cursor_advances_within_large_neighbor_list(self):
        center = _uid()
        neighbors = sorted(_uid() for _ in range(5))
        notes = [_note(center)] + [_note(n) for n in neighbors]
        edges = [_manual_edge(center, n) for n in neighbors]
        db = _mock_db(notes=notes, edges=edges, note_count=6)
        svc = NoteGraphService(user_id="u1", db=db)

        first = svc.generate_graph(NoteGraphRequest(center_note_id=center, radius=1, max_nodes=3))
        second = svc.generate_graph(
            NoteGraphRequest(center_note_id=center, radius=1, max_nodes=3, cursor=first.cursor)
        )

        first_neighbors = {n.id for n in first.nodes if n.type == "note"} - {center}
        second_neighbors = {n.id for n in second.nodes if n.type == "note"} - {center}
        assert first.has_more is True
        assert second_neighbors
        assert first_neighbors.isdisjoint(second_neighbors)

    def test_malformed_cursor_shape_raises_input_error(self):
        raw = base64.urlsafe_b64encode(json.dumps({"layer": 0}).encode()).decode()
        n1 = _uid()
        db = _mock_db(notes=[_note(n1)], note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)

        with pytest.raises(InputError, match="Invalid graph cursor"):
            svc.generate_graph(NoteGraphRequest(center_note_id=n1, radius=1, cursor=raw))


class TestEdgeTypeFilter:
    def test_only_requested_types(self):
        n1 = _uid()
        n2 = _uid()
        notes = [
            _note(n1, content=f"[[id:{n2}]]"),
            _note(n2),
        ]
        edges = [_manual_edge(n1, n2)]
        tag_edges = [{"note_id": n1, "keyword_id": 1, "keyword": "test"}]
        db = _mock_db(notes=notes, edges=edges, tag_edges=tag_edges, note_count=2, all_ids=[n1, n2])
        svc = NoteGraphService(user_id="u1", db=db)
        # Only request manual edges
        req = NoteGraphRequest(radius=1, edge_types=[EdgeType.manual])
        resp = svc.generate_graph(req)
        types = {e.type for e in resp.edges}
        assert EdgeType.wikilink not in types
        assert EdgeType.tag_membership not in types

    def test_omitted_edges_keep_legacy_projection_readiness_behavior(self):
        note_id = _uid()
        db = _mock_db(
            notes=[_note(note_id)],
            note_count=1,
            all_ids=[note_id],
            projection_state="rebuilding",
        )
        service = NoteGraphService(user_id="u1", db=db)

        with pytest.raises(graph_service_module.GraphProjectionNotReadyError):
            service.generate_graph(NoteGraphRequest(center_note_id=note_id))

        semantic_only = service.generate_graph(
            NoteGraphRequest(
                center_note_id=note_id,
                edge_types=[EdgeType.semantic],
            )
        )
        assert [node.id for node in semantic_only.nodes] == [note_id]
        assert semantic_only.edges == []


class TestDeletedNotesFlagged:
    def test_soft_deleted_in_graph(self):
        n1 = _uid()
        notes = [_note(n1, deleted=1)]
        db = _mock_db(notes=notes, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1)
        resp = svc.generate_graph(req)
        note_nodes = [n for n in resp.nodes if n.type == "note"]
        assert len(note_nodes) == 1
        assert note_nodes[0].deleted is True


class TestCacheHit:
    def test_second_call_uses_cache(self):
        n1 = _uid()
        notes = [_note(n1)]
        db = _mock_db(notes=notes, note_count=1, all_ids=[n1])
        db.note_semantic_store = MagicMock()
        cache = GraphCache(ttl_seconds=60, max_keys=100)
        svc = NoteGraphService(user_id="u1", db=db, cache=cache)
        req = NoteGraphRequest(radius=1)
        resp1 = svc.generate_graph(req)
        # Reset call count
        db.get_notes_batch.reset_mock()
        resp2 = svc.generate_graph(req)
        # Second call should not hit DB
        db.get_notes_batch.assert_not_called()
        db.note_semantic_store.assert_not_called()
        assert resp1 == resp2

    def test_omitted_edge_cursor_keeps_the_pre_semantic_request_hash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("NOTES_GRAPH_MAX_NODES", "300")
        monkeypatch.setenv("NOTES_GRAPH_MAX_EDGES", "1200")
        monkeypatch.setenv("NOTES_GRAPH_MAX_DEGREE", "40")
        center = _uid()
        neighbors = sorted(_uid() for _ in range(3))
        service = NoteGraphService(
            user_id="u1",
            db=_mock_db(
                notes=[_note(center), *(_note(note_id) for note_id in neighbors)],
                edges=[_manual_edge(center, note_id) for note_id in neighbors],
                note_count=4,
            ),
        )
        response = service.generate_graph(
            NoteGraphRequest(center_note_id=center, max_nodes=2)
        )
        legacy_query = {
            "center": center,
            "radius": 1,
            "edge_types": None,
            "tag": None,
            "source": None,
            "time_range": None,
            "time_range_field": "updated_at",
            "max_nodes": 2,
            "max_edges": 1_200,
            "max_degree": 40,
            "allow_heavy": False,
        }
        expected_hash = hashlib.sha256(
            json.dumps(legacy_query, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        assert response.cursor is not None
        assert _decode_cursor(response.cursor)["request"] == expected_hash


class TestSemanticCandidateGeneration:
    def test_candidate_generation_preserves_public_graph_and_exposes_bounded_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("NOTES_GRAPH_MAX_DEGREE", "100")
        center = _uid()
        neighbors = sorted(_uid() for _ in range(60))
        notes = [_note(center), *(_note(note_id) for note_id in neighbors)]
        edges = [_manual_edge(center, note_id) for note_id in neighbors]
        request = NoteGraphRequest(
            center_note_id=center,
            edge_types=[EdgeType.manual, EdgeType.semantic],
            max_nodes=2,
            max_edges=1,
            max_degree=100,
        )

        service = NoteGraphService(
            user_id="u1",
            db=_mock_db(notes=notes, edges=edges, note_count=len(notes)),
        )
        ordinary = service.generate_graph(request)
        candidates = service.generate_semantic_candidates(
            request,
            additional_nodes=10_000,
            additional_edges=10_000,
        )

        assert isinstance(
            candidates,
            graph_service_module.SemanticGraphCandidateResult,
        )
        assert candidates.public_graph == ordinary
        assert ordinary.limits.max_nodes == 2
        assert ordinary.limits.max_edges == 1
        assert ordinary.all_notes_note_cap == 2
        assert ordinary.all_notes_eligible is False
        assert len([node for node in ordinary.nodes if node.type == "note"]) <= 2
        assert len(ordinary.edges) <= 1
        assert candidates.candidate_limits.max_nodes == 52
        assert candidates.candidate_limits.max_edges == 51
        assert len(
            [node for node in candidates.candidate_nodes if node.type == "note"]
        ) == 52
        assert len(candidates.candidate_edges) == 51

    def test_public_and_candidate_graphs_use_distinct_cache_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("NOTES_GRAPH_MAX_DEGREE", "100")
        center = _uid()
        neighbors = sorted(_uid() for _ in range(6))
        cache = GraphCache(ttl_seconds=60, max_keys=10)
        service = NoteGraphService(
            user_id="u1",
            db=_mock_db(
                notes=[_note(center), *(_note(note_id) for note_id in neighbors)],
                edges=[_manual_edge(center, note_id) for note_id in neighbors],
                note_count=7,
            ),
            cache=cache,
        )
        request = NoteGraphRequest(
            center_note_id=center,
            edge_types=[EdgeType.manual, EdgeType.semantic],
            max_nodes=2,
            max_edges=1,
            max_degree=100,
        )

        result = service.generate_semantic_candidates(
            request,
            additional_nodes=3,
            additional_edges=3,
        )

        assert result.public_graph.limits.max_nodes == 2
        assert result.candidate_limits.max_nodes == 5
        assert len(result.public_graph.nodes) == 2
        assert len(result.candidate_nodes) == 5
        assert cache.stats()["size"] == 2
        assert service.generate_graph(request) == result.public_graph

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("additional_nodes", -1),
            ("additional_nodes", True),
            ("additional_nodes", 1.5),
            ("additional_edges", -1),
            ("additional_edges", True),
            ("additional_edges", 1.5),
        ],
    )
    def test_candidate_generation_rejects_invalid_allowances(
        self,
        field: str,
        value: object,
    ):
        service = NoteGraphService(user_id="u1", db=_mock_db())
        kwargs: dict[str, object] = {
            "additional_nodes": 1,
            "additional_edges": 1,
            field: value,
        }

        with pytest.raises(InputError, match="allowance"):
            service.generate_semantic_candidates(
                NoteGraphRequest(
                    center_note_id=_uid(),
                    edge_types=[EdgeType.semantic],
                ),
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("graph_request", "message"),
        [
            (
                NoteGraphRequest(
                    center_note_id=_uid(),
                    edge_types=[EdgeType.manual],
                ),
                "semantic",
            ),
            (NoteGraphRequest(edge_types=[EdgeType.semantic]), "center"),
            (
                NoteGraphRequest(
                    center_note_id=_uid(),
                    edge_types=[EdgeType.semantic],
                    cursor="later-page",
                ),
                "first page",
            ),
        ],
    )
    def test_candidate_generation_requires_semantic_focus_and_first_page(
        self,
        graph_request: NoteGraphRequest,
        message: str,
    ):
        service = NoteGraphService(user_id="u1", db=_mock_db())

        with pytest.raises(InputError, match=message):
            service.generate_semantic_candidates(
                graph_request,
                additional_nodes=1,
                additional_edges=1,
            )

    def test_candidate_generation_keeps_public_cursor_on_public_traversal(self):
        center = _uid()
        neighbors = sorted(_uid() for _ in range(5))
        notes = [_note(center), *(_note(note_id) for note_id in neighbors)]
        edges = [_manual_edge(center, note_id) for note_id in neighbors]
        request = NoteGraphRequest(
            center_note_id=center,
            edge_types=[EdgeType.manual, EdgeType.semantic],
            max_nodes=2,
            max_edges=10,
        )
        service = NoteGraphService(
            user_id="u1",
            db=_mock_db(notes=notes, edges=edges, note_count=len(notes)),
        )

        expected_public = service.generate_graph(request)
        candidates = service.generate_semantic_candidates(
            request,
            additional_nodes=3,
            additional_edges=3,
        )
        assert candidates.public_graph.cursor == expected_public.cursor

        continuation = service.generate_graph(
            request.model_copy(update={"cursor": candidates.public_graph.cursor})
        )
        continuation_note_ids = {
            node.id for node in continuation.nodes if node.type == "note"
        }

        assert continuation_note_ids == {center, neighbors[1]}

    def test_candidate_generation_rejects_revision_change_between_passes(self):
        center = _uid()
        neighbor = _uid()
        db = _mock_db(
            notes=[_note(center), _note(neighbor)],
            edges=[_manual_edge(center, neighbor)],
            note_count=2,
        )
        db.note_graph_projection_store.get_revision = MagicMock(
            side_effect=[7, 7, 8, 8]
        )
        service = NoteGraphService(user_id="u1", db=db)
        request = NoteGraphRequest(
            center_note_id=center,
            edge_types=[EdgeType.manual, EdgeType.semantic],
            max_nodes=1,
        )

        with pytest.raises(
            graph_service_module.GraphProjectionNotReadyError,
            match="changed",
        ):
            service.generate_semantic_candidates(
                request,
                additional_nodes=1,
                additional_edges=1,
            )


class TestCursorRoundtrip:
    def test_encode_decode(self):
        encoded = _encode_cursor(1, 42, "abc-123")
        decoded = _decode_cursor(encoded)
        assert decoded["layer"] == 1
        assert decoded["pos"] == 42
        assert decoded["last_id"] == "abc-123"

    def test_revision_bound_cursor_rejects_dataset_revision_parser_and_query_mismatch(self):
        encoded = _encode_cursor(
            1,
            2,
            "abc-123",
            dataset_hash="dataset-hash",
            graph_revision=7,
            parser_version=2,
            request_hash="request-hash",
        )
        assert _decode_cursor(
            encoded,
            expected_dataset_hash="dataset-hash",
            expected_graph_revision=7,
            expected_parser_version=2,
            expected_request_hash="request-hash",
        )["last_id"] == "abc-123"

        for field, value in (
            ("expected_dataset_hash", "other"),
            ("expected_graph_revision", 8),
            ("expected_parser_version", 3),
            ("expected_request_hash", "other"),
        ):
            expected = {
                "expected_dataset_hash": "dataset-hash",
                "expected_graph_revision": 7,
                "expected_parser_version": 2,
                "expected_request_hash": "request-hash",
            }
            expected[field] = value
            with pytest.raises(InputError, match="stale or mismatched"):
                _decode_cursor(encoded, **expected)

    def test_cursor_size_limits_fail_closed(self):
        with pytest.raises(InputError, match="too large"):
            _decode_cursor("a" * 8193)

        oversized_json = {"layer": 0, "pos": 0, "last_id": "x" * 4100}
        encoded = base64.urlsafe_b64encode(json.dumps(oversized_json).encode()).decode()
        with pytest.raises(InputError, match="too large"):
            _decode_cursor(encoded)

    def test_semantic_binding_mismatch_fails_closed_and_matching_cursor_continues(self):
        center = _uid()
        neighbors = sorted(_uid() for _ in range(3))
        db = _mock_db(
            notes=[_note(center), *(_note(note_id) for note_id in neighbors)],
            edges=[_manual_edge(center, note_id) for note_id in neighbors],
            note_count=4,
        )
        db.note_semantic_store = MagicMock()
        service = NoteGraphService(user_id="u1", db=db)
        request = NoteGraphRequest(
            center_note_id=center,
            edge_types=[EdgeType.manual, EdgeType.semantic],
            semantic_top_k=2,
            semantic_threshold=0.75,
            max_nodes=2,
        )
        first = service.generate_graph(request)
        bound_cursor = graph_service_module.bind_semantic_cursor(
            first.cursor,
            semantic_binding="semantic-binding-a",
        )
        assert _decode_cursor(
            bound_cursor,
            expected_semantic_binding="semantic-binding-a",
        )["semantic"] == "semantic-binding-a"

        with pytest.raises(InputError, match="stale or mismatched"):
            _decode_cursor(
                bound_cursor,
                expected_semantic_binding="semantic-binding-b",
            )

        continuation = service.generate_graph(
            request.model_copy(update={"cursor": bound_cursor})
        )

        assert continuation.nodes
        db.note_semantic_store.assert_not_called()

    def test_semantic_binding_rejects_rebinding(self):
        ordinary = _encode_cursor(
            1,
            2,
            "abc-123",
            dataset_hash="dataset-hash",
            graph_revision=7,
            parser_version=2,
            request_hash="request-hash",
        )
        bound = graph_service_module.bind_semantic_cursor(
            ordinary,
            semantic_binding="semantic-binding-a",
        )

        with pytest.raises(InputError, match="already.*semantic"):
            graph_service_module.bind_semantic_cursor(
                bound,
                semantic_binding="semantic-binding-b",
            )


class TestDeterministicOrdering:
    def test_same_query_same_result(self):
        n1 = _uid()
        n2 = _uid()
        notes = [_note(n1), _note(n2)]
        edges = [_manual_edge(n1, n2)]
        db = _mock_db(notes=notes, edges=edges, note_count=2, all_ids=[n1, n2])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1)
        r1 = svc.generate_graph(req)
        r2 = svc.generate_graph(req)
        assert [n.id for n in r1.nodes] == [n.id for n in r2.nodes]
        assert [e.id for e in r1.edges] == [e.id for e in r2.edges]

    def test_note_nodes_are_ordered_by_updated_at_desc_then_id(self):
        ids = [_uid() for _ in range(5)]
        notes = []
        for index, nid in enumerate(ids):
            notes.append({
                "id": nid,
                "title": f"N{index}",
                "content": "body",
                "created_at": "2025-01-01T00:00:00",
                "last_modified": f"2025-06-0{index + 1}T00:00:00",
                "deleted": 0,
                "conversation_id": None,
            })
        expected = [n["id"] for n in sorted(
            notes,
            key=lambda row: (row["last_modified"], row["id"]),
            reverse=True,
        )]
        db = _mock_db(notes=notes, note_count=len(notes), all_ids=ids)
        svc = NoteGraphService(user_id="u1", db=db)

        resp = svc.generate_graph(NoteGraphRequest(radius=1, edge_types=[EdgeType.manual]))

        assert [n.id for n in resp.nodes if n.type == "note"] == expected

    def test_tag_and_source_edge_order_does_not_depend_on_db_row_order(self):
        n1, n2 = sorted([_uid(), _uid()])
        notes = [_note(n1), _note(n2)]
        tag_edges = [
            {"note_id": n2, "keyword_id": 2, "keyword": "zeta"},
            {"note_id": n1, "keyword_id": 1, "keyword": "alpha"},
        ]
        source_info = [
            {"note_id": n2, "conversation_id": _uid(), "source": "youtube", "external_ref": "b"},
            {"note_id": n1, "conversation_id": _uid(), "source": "article", "external_ref": "a"},
        ]
        db_unsorted = _mock_db(
            notes=notes,
            tag_edges=tag_edges,
            source_info=source_info,
            note_count=2,
            all_ids=[n2, n1],
        )
        db_reversed = _mock_db(
            notes=notes,
            tag_edges=list(reversed(tag_edges)),
            source_info=list(reversed(source_info)),
            note_count=2,
            all_ids=[n2, n1],
        )
        req = NoteGraphRequest(
            radius=1,
            edge_types=[EdgeType.tag_membership, EdgeType.source_membership],
            max_edges=3,
        )

        first = NoteGraphService(user_id="u1", db=db_unsorted).generate_graph(req)
        second = NoteGraphService(user_id="u1", db=db_reversed).generate_graph(req)

        assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


class TestEmptyGraph:
    def test_no_notes(self):
        db = _mock_db(note_count=0, all_ids=[])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1)
        resp = svc.generate_graph(req)
        assert resp.nodes == []
        assert resp.edges == []
        assert resp.truncated is False


class TestPruningOrder:
    def test_tag_source_before_wikilinks_before_manual(self):
        """When max_edges is hit, tag/source edges are pruned before wikilinks, manual last."""
        n1, n2 = _uid(), _uid()
        notes = [
            _note(n1, content=f"[[id:{n2}]]"),
            _note(n2),
        ]
        manual_edges = [_manual_edge(n1, n2)]
        tag_edges = [
            {"note_id": n1, "keyword_id": 1, "keyword": "t1"},
            {"note_id": n2, "keyword_id": 2, "keyword": "t2"},
        ]
        db = _mock_db(
            notes=notes, edges=manual_edges, tag_edges=tag_edges,
            note_count=2, all_ids=[n1, n2],
        )
        svc = NoteGraphService(user_id="u1", db=db)
        # Request all types but restrict edges to 2
        req = NoteGraphRequest(
            radius=1,
            edge_types=[EdgeType.manual, EdgeType.wikilink, EdgeType.tag_membership],
            max_edges=2,
        )
        resp = svc.generate_graph(req)
        types = [e.type for e in resp.edges]
        # Manual should survive; tag_membership should be pruned first
        assert EdgeType.manual in types


class TestTimeRangeFilter:
    def test_excludes_old_notes(self):
        """Notes before start are excluded from the graph."""
        n1 = _uid()
        n2 = _uid()
        notes = [
            {
                "id": n1, "title": "Old", "content": "old",
                "created_at": "2024-01-01T00:00:00", "last_modified": "2024-01-01T00:00:00",
                "deleted": 0, "conversation_id": None,
            },
            {
                "id": n2, "title": "New", "content": "new",
                "created_at": "2025-06-01T00:00:00", "last_modified": "2025-06-01T00:00:00",
                "deleted": 0, "conversation_id": None,
            },
        ]
        db = _mock_db(notes=notes, note_count=2, all_ids=[n1, n2])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(
            radius=1,
            time_range=TimeRange(start="2025-01-01T00:00:00"),
        )
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert n2 in note_ids
        assert n1 not in note_ids

    def test_includes_boundary_note(self):
        """Notes at exact start are included (inclusive)."""
        n1 = _uid()
        notes = [
            {
                "id": n1, "title": "Boundary", "content": "body",
                "created_at": "2025-03-15T12:00:00", "last_modified": "2025-03-15T12:00:00",
                "deleted": 0, "conversation_id": None,
            },
        ]
        db = _mock_db(notes=notes, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(
            radius=1,
            time_range=TimeRange(start="2025-03-15T12:00:00"),
        )
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert n1 in note_ids

    def test_created_at_field_default(self):
        """Default time_range_field is created_at, which maps to the created_at DB column."""
        n1 = _uid()
        notes = [
            {
                "id": n1, "title": "X", "content": "x",
                "created_at": "2024-06-01T00:00:00", "last_modified": "2025-06-01T00:00:00",
                "deleted": 0, "conversation_id": None,
            },
        ]
        db = _mock_db(notes=notes, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        # Filter by created_at — note was created in 2024, filter requires 2025+
        req = NoteGraphRequest(
            radius=1,
            time_range=TimeRange(start="2025-01-01T00:00:00"),
            time_range_field="created_at",
        )
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert n1 not in note_ids

    def test_updated_at_maps_to_last_modified(self):
        """time_range_field=updated_at maps to the last_modified DB column."""
        n1 = _uid()
        notes = [
            {
                "id": n1, "title": "X", "content": "x",
                "created_at": "2024-01-01T00:00:00", "last_modified": "2025-06-01T00:00:00",
                "deleted": 0, "conversation_id": None,
            },
        ]
        db = _mock_db(notes=notes, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        # Filter by updated_at — last_modified is 2025-06, filter requires 2025-01+
        req = NoteGraphRequest(
            radius=1,
            time_range=TimeRange(start="2025-01-01T00:00:00"),
            time_range_field="updated_at",
        )
        resp = svc.generate_graph(req)
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert n1 in note_ids

    def test_timezone_aware_range_compares_by_instant(self):
        n1 = _uid()
        notes = [
            {
                "id": n1, "title": "Before absolute start", "content": "x",
                "created_at": "2026-01-01T07:30:00+00:00",
                "last_modified": "2026-01-01T07:30:00+00:00",
                "deleted": 0, "conversation_id": None,
            },
        ]
        db = _mock_db(notes=notes, note_count=1, all_ids=[n1])
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(
            radius=1,
            time_range=TimeRange(start="2026-01-01T00:00:00-08:00"),
            time_range_field="updated_at",
        )

        resp = svc.generate_graph(req)

        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert n1 not in note_ids


class TestAllowHeavy:
    def test_allow_heavy_without_elevated_permission_still_rejects_large_seedless_graph(self):
        """The endpoint grants allow_heavy_limits only after a permission check."""
        ids = [_uid() for _ in range(10)]
        notes = [_note(i) for i in ids]
        db = _mock_db(notes=notes, note_count=500, all_ids=ids)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(radius=1, allow_heavy=True)

        with pytest.raises(InputError):
            svc.generate_graph(req)

    def test_allow_heavy_returns_graph_when_elevated_limits_are_authorized(self):
        """When allow_heavy is authorized and note count exceeds max_nodes, return capped graph."""
        ids = [_uid() for _ in range(10)]
        notes = [_note(i) for i in ids]
        db = _mock_db(notes=notes, note_count=500, all_ids=ids)
        svc = NoteGraphService(user_id="u1", db=db, allow_heavy_limits=True)
        req = NoteGraphRequest(radius=1, allow_heavy=True)
        resp = svc.generate_graph(req)
        # Should not raise; returns notes up to max_nodes
        assert len([n for n in resp.nodes if n.type == "note"]) > 0


class TestCenterNoteNotFound:
    def test_raises_input_error(self):
        """When center_note_id doesn't exist, raise InputError."""
        bogus = _uid()
        db = _mock_db(notes=[], note_count=0)
        svc = NoteGraphService(user_id="u1", db=db)
        req = NoteGraphRequest(center_note_id=bogus, radius=1)
        with pytest.raises(InputError, match="not found"):
            svc.generate_graph(req)


class TestCursorResumeForRadius2:
    def test_resume_rejects_cursor_position_that_no_longer_matches_node(self):
        center = "00000000-0000-4000-8000-000000000000"
        neighbors = [
            f"00000000-0000-4000-8000-{index:012d}"
            for index in range(1, 4)
        ]
        service = NoteGraphService(
            user_id="u1",
            db=_mock_db(
                notes=[_note(center), *(_note(note_id) for note_id in neighbors)],
                edges=[_manual_edge(center, note_id) for note_id in neighbors],
                note_count=4,
            ),
        )
        first = service.generate_graph(
            NoteGraphRequest(center_note_id=center, radius=2, max_nodes=2)
        )
        payload = _decode_cursor(first.cursor)
        payload["last_id"] = neighbors[-1]
        tampered = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode()

        with pytest.raises(InputError, match="stale or mismatched"):
            service.generate_graph(
                NoteGraphRequest(
                    center_note_id=center,
                    radius=2,
                    max_nodes=2,
                    cursor=tampered,
                )
            )

    def test_resume_from_second_layer_reconstructs_prior_frontier(self):
        center = "00000000-0000-4000-8000-000000000000"
        first_hop = [
            f"00000000-0000-4000-8000-{index:012d}"
            for index in range(1, 5)
        ]
        second_hop = [
            "00000000-0000-4000-8000-000000000005",
            "00000000-0000-4000-8000-000000000006",
        ]
        notes = [
            _note(center),
            *(_note(note_id) for note_id in first_hop),
            *(_note(note_id) for note_id in second_hop),
        ]
        edges = [
            *(_manual_edge(center, note_id) for note_id in first_hop),
            _manual_edge(first_hop[2], second_hop[0]),
            _manual_edge(first_hop[3], second_hop[1]),
        ]
        service = NoteGraphService(
            user_id="u1",
            db=_mock_db(notes=notes, edges=edges, note_count=len(notes)),
        )

        first = service.generate_graph(
            NoteGraphRequest(center_note_id=center, radius=2, max_nodes=3)
        )
        second = service.generate_graph(
            NoteGraphRequest(
                center_note_id=center,
                radius=2,
                max_nodes=3,
                cursor=first.cursor,
            )
        )
        assert second.cursor is not None
        assert _decode_cursor(second.cursor)["layer"] == 1

        third = service.generate_graph(
            NoteGraphRequest(
                center_note_id=center,
                radius=2,
                max_nodes=3,
                cursor=second.cursor,
            )
        )

        third_note_ids = {node.id for node in third.nodes if node.type == "note"}
        assert set(second_hop) <= third_note_ids

    def test_revision_bound_cursor_is_accepted_for_radius_two(self):
        c = _uid()
        neighbors = [_uid() for _ in range(3)]
        notes = [_note(c), *(_note(note_id) for note_id in neighbors)]
        edges = [_manual_edge(c, note_id) for note_id in neighbors]
        db = _mock_db(notes=notes, edges=edges, note_count=4)
        svc = NoteGraphService(user_id="u1", db=db)
        first = svc.generate_graph(
            NoteGraphRequest(center_note_id=c, radius=2, max_nodes=2)
        )
        assert first.cursor is not None

        resp = svc.generate_graph(
            NoteGraphRequest(
                center_note_id=c,
                radius=2,
                max_nodes=2,
                cursor=first.cursor,
            )
        )
        note_ids = {n.id for n in resp.nodes if n.type == "note"}
        assert c in note_ids
