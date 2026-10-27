"""Property and example tests for deterministic semantic graph composition."""

from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphEdge,
    GraphLimits,
    GraphNode,
    NoteGraphResponse,
    SemanticEdgeEvidence,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_projector import (
    compose_semantic_graph,
)

pytestmark = pytest.mark.property


def _node(note_id: str, *, node_type: str = "note") -> GraphNode:
    return GraphNode(id=note_id, type=node_type, label=note_id, degree=0)


def _semantic_edge(target: str, *, score: float = 0.9) -> GraphEdge:
    return GraphEdge(
        id=f"semantic:{target}",
        source="focus",
        target=target,
        type=EdgeType.semantic,
        directed=False,
        weight=score,
        evidence=SemanticEdgeEvidence(
            similarity=score,
            qualitative_band="very_high" if score >= 0.9 else "high",
            source_note_id="focus",
            target_note_id=target,
            source_content_version=1,
            target_content_version=1,
            generation_id="generation-a",
            semantic_index_revision=2,
            configuration_revision=3,
            normalization_version="notes-semantic-normalization-v1",
            chunker_version="notes-semantic-chunker-v1",
            provider_label="Provider",
            model_label="Model",
        ),
    )


def _response(max_nodes: int, max_edges: int, max_degree: int) -> NoteGraphResponse:
    return NoteGraphResponse(
        nodes=[_node("focus")],
        edges=[],
        limits=GraphLimits(
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_degree=max_degree,
        ),
        active_note_count=20,
        all_notes_note_cap=max_nodes,
        all_notes_eligible=False,
    )


def test_manual_supersedes_semantic_while_wikilink_can_coexist() -> None:
    public = _response(10, 10, 10)
    candidate_nodes = (_node("focus"), _node("target"))
    candidate_edges = (
        GraphEdge(
            id="manual",
            source="target",
            target="focus",
            type=EdgeType.manual,
            directed=False,
        ),
        GraphEdge(
            id="wikilink",
            source="focus",
            target="target",
            type=EdgeType.wikilink,
            directed=True,
        ),
    )

    result = compose_semantic_graph(
        public_graph=public,
        candidate_nodes=candidate_nodes,
        candidate_edges=candidate_edges,
        semantic_nodes=(_node("target"),),
        semantic_edges=(_semantic_edge("target"),),
        focus_note_id="focus",
    )

    assert [edge.type for edge in result.edges] == [
        EdgeType.manual,
        EdgeType.wikilink,
    ]


def test_semantic_displaces_only_membership_and_unused_allowance_returns() -> None:
    public = _response(4, 3, 4)
    candidate_nodes = (
        _node("focus"),
        _node("manual-target"),
        _node("semantic-target"),
        _node("tag:a", node_type="tag"),
        _node("tag:b", node_type="tag"),
    )
    candidate_edges = (
        GraphEdge(
            id="manual",
            source="focus",
            target="manual-target",
            type=EdgeType.manual,
            directed=False,
        ),
        GraphEdge(
            id="tag-a",
            source="focus",
            target="tag:a",
            type=EdgeType.tag_membership,
            directed=False,
        ),
        GraphEdge(
            id="tag-b",
            source="manual-target",
            target="tag:b",
            type=EdgeType.tag_membership,
            directed=False,
        ),
    )

    with_semantic = compose_semantic_graph(
        public_graph=public,
        candidate_nodes=candidate_nodes,
        candidate_edges=candidate_edges,
        semantic_nodes=(_node("semantic-target"),),
        semantic_edges=(_semantic_edge("semantic-target"),),
        focus_note_id="focus",
    )
    without_semantic = compose_semantic_graph(
        public_graph=public,
        candidate_nodes=candidate_nodes,
        candidate_edges=candidate_edges,
        semantic_nodes=(),
        semantic_edges=(),
        focus_note_id="focus",
    )

    assert {edge.type for edge in with_semantic.edges} == {
        EdgeType.manual,
        EdgeType.semantic,
        EdgeType.tag_membership,
    }
    assert len(without_semantic.edges) == 3
    assert all(edge.type is not EdgeType.semantic for edge in without_semantic.edges)


def test_rejected_semantic_edge_does_not_admit_an_orphan_target_node() -> None:
    public = _response(3, 0, 3)

    result = compose_semantic_graph(
        public_graph=public,
        candidate_nodes=(_node("focus"), _node("ordinary")),
        candidate_edges=(),
        semantic_nodes=(_node("semantic-target"),),
        semantic_edges=(_semantic_edge("semantic-target"),),
        focus_note_id="focus",
    )

    assert [node.id for node in result.nodes] == ["focus", "ordinary"]
    assert result.edges == []


@settings(max_examples=60, deadline=None)
@given(
    semantic_count=st.integers(min_value=0, max_value=8),
    membership_count=st.integers(min_value=0, max_value=8),
    max_nodes=st.integers(min_value=1, max_value=10),
    max_edges=st.integers(min_value=0, max_value=12),
    max_degree=st.integers(min_value=1, max_value=5),
)
def test_composition_never_exceeds_public_caps_and_is_deterministic(
    semantic_count: int,
    membership_count: int,
    max_nodes: int,
    max_edges: int,
    max_degree: int,
) -> None:
    public = _response(max_nodes, max_edges, max_degree)
    semantic_nodes = tuple(_node(f"semantic-{index}") for index in range(semantic_count))
    semantic_edges = tuple(
        _semantic_edge(f"semantic-{index}", score=0.9 - index / 100) for index in range(semantic_count)
    )
    membership_nodes = tuple(_node(f"tag:{index}", node_type="tag") for index in range(membership_count))
    membership_edges = tuple(
        GraphEdge(
            id=f"membership-{index}",
            source="focus",
            target=f"tag:{index}",
            type=EdgeType.tag_membership,
            directed=False,
        )
        for index in range(membership_count)
    )
    candidate_nodes = (_node("focus"), *membership_nodes)

    first = compose_semantic_graph(
        public_graph=public,
        candidate_nodes=candidate_nodes,
        candidate_edges=membership_edges,
        semantic_nodes=semantic_nodes,
        semantic_edges=semantic_edges,
        focus_note_id="focus",
    )
    second = compose_semantic_graph(
        public_graph=public,
        candidate_nodes=tuple(reversed(candidate_nodes)),
        candidate_edges=tuple(reversed(membership_edges)),
        semantic_nodes=tuple(reversed(semantic_nodes)),
        semantic_edges=tuple(reversed(semantic_edges)),
        focus_note_id="focus",
    )

    assert len(first.nodes) <= max_nodes
    assert len(first.edges) <= max_edges
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    node_ids = {node.id for node in first.nodes}
    assert all(edge.source in node_ids and edge.target in node_ids for edge in first.edges)
    degrees = Counter(endpoint for edge in first.edges for endpoint in (edge.source, edge.target))
    assert all(value <= max_degree for value in degrees.values())
    if max_nodes >= 1:
        assert first.nodes[0].id == "focus"
