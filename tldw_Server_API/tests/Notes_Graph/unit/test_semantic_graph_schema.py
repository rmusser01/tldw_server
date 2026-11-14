"""Contract tests for opt-in semantic Notes graph requests and responses."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas import notes_graph as schemas
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import (
    DEFAULT_SEMANTIC_INDEX_SETTINGS,
)

pytestmark = pytest.mark.unit


def _evidence(**overrides: object):
    values: dict[str, object] = {
        "similarity": 0.875,
        "qualitative_band": "high",
        "source_note_id": "note-source",
        "target_note_id": "note-target",
        "source_content_version": 3,
        "target_content_version": 4,
        "generation_id": "generation-1",
        "semantic_index_revision": 8,
        "configuration_revision": 5,
        "normalization_version": "notes-semantic-normalization-v1",
        "chunker_version": "notes-semantic-chunker-v1",
        "provider_label": "Local provider",
        "model_label": "Embedding model",
        "model_revision": "revision-7",
        "excerpt_pairs": [
            {
                "source": {
                    "field": "content",
                    "start_code_point": 0,
                    "end_code_point": 6,
                    "text": "source",
                },
                "target": {
                    "field": "title",
                    "start_code_point": 1,
                    "end_code_point": 7,
                    "text": "target",
                },
            }
        ],
    }
    values.update(overrides)
    return schemas.SemanticEdgeEvidence(**values)


def test_omitted_edge_types_resolve_to_the_frozen_legacy_set() -> None:
    request = schemas.NoteGraphRequest()

    assert schemas.EdgeType.semantic not in schemas.LEGACY_EDGE_TYPES
    assert frozenset(request.resolved_edge_types) == schemas.LEGACY_EDGE_TYPES
    assert request.edge_types is None
    assert frozenset(
        {
            schemas.EdgeType.manual,
            schemas.EdgeType.wikilink,
            schemas.EdgeType.backlink,
            schemas.EdgeType.tag_membership,
            schemas.EdgeType.source_membership,
        }
    ) == schemas.LEGACY_EDGE_TYPES
    assert frozenset(schemas.EdgeType) != schemas.LEGACY_EDGE_TYPES


def test_graph_response_defaults_manual_link_authority_to_false() -> None:
    response = schemas.NoteGraphResponse(
        limits=schemas.GraphLimits(max_nodes=20, max_edges=20, max_degree=20),
        active_note_count=0,
        all_notes_note_cap=20,
        all_notes_eligible=True,
    )

    assert response.manual_link_authorized is False
    assert response.model_dump(mode="json")["manual_link_authorized"] is False


def test_explicit_edge_types_are_sorted_and_deduplicated_for_identity() -> None:
    request = schemas.NoteGraphRequest(
        edge_types="semantic,manual,semantic,backlink,manual"
    )

    assert request.edge_types == [
        schemas.EdgeType.backlink,
        schemas.EdgeType.manual,
        schemas.EdgeType.semantic,
    ]
    assert request.resolved_edge_types == tuple(request.edge_types)
    assert request.semantic_requested is True


@pytest.mark.parametrize(
    ("values", "valid"),
    [
        ({"semantic_top_k": 1}, True),
        (
            {
                "semantic_top_k": DEFAULT_SEMANTIC_INDEX_SETTINGS.max_query_neighbors,
                "semantic_threshold": 0.0,
            },
            True,
        ),
        ({"semantic_threshold": 1.0}, True),
        ({"semantic_top_k": 0}, False),
        (
            {
                "semantic_top_k": (
                    DEFAULT_SEMANTIC_INDEX_SETTINGS.max_query_neighbors + 1
                )
            },
            False,
        ),
        ({"semantic_threshold": -0.001}, False),
        ({"semantic_threshold": 1.001}, False),
    ],
)
def test_semantic_controls_are_bounded(values: dict[str, object], valid: bool) -> None:
    payload = {"edge_types": ["semantic"], **values}

    if valid:
        schemas.NoteGraphRequest(**payload)
    else:
        with pytest.raises(ValidationError):
            schemas.NoteGraphRequest(**payload)


@pytest.mark.parametrize("field", ["semantic_top_k", "semantic_threshold"])
def test_semantic_controls_require_explicit_semantic_edge(field: str) -> None:
    value: int | float = 3 if field == "semantic_top_k" else 0.8

    with pytest.raises(ValidationError, match="semantic"):
        schemas.NoteGraphRequest(**{field: value})
    with pytest.raises(ValidationError, match="semantic"):
        schemas.NoteGraphRequest(edge_types=["manual"], **{field: value})


def test_semantic_evidence_is_typed_bounded_and_bound_to_the_edge() -> None:
    evidence = _evidence()
    edge = schemas.GraphEdge(
        id="semantic:source:target",
        source="note-source",
        target="note-target",
        type=schemas.EdgeType.semantic,
        directed=False,
        weight=evidence.similarity,
        evidence=evidence,
    )

    assert edge.evidence == evidence
    assert edge.model_dump(mode="json")["evidence"]["excerpt_pairs"][0]["source"] == {
        "field": "content",
        "start_code_point": 0,
        "end_code_point": 6,
        "text": "source",
    }

    with pytest.raises(ValidationError):
        schemas.SemanticEdgeEvidence(
            **{
                **evidence.model_dump(),
                "excerpt_pairs": evidence.model_dump()["excerpt_pairs"] * 4,
            }
        )
    with pytest.raises(ValidationError):
        schemas.SemanticExcerpt(
            field="content",
            start_code_point=0,
            end_code_point=481,
            text="x" * 481,
        )
    with pytest.raises(ValidationError):
        schemas.SemanticEdgeEvidence(**{**evidence.model_dump(), "properties": {}})
    with pytest.raises(ValidationError, match="source"):
        schemas.GraphEdge(
            id="semantic:wrong:target",
            source="wrong-source",
            target="note-target",
            type=schemas.EdgeType.semantic,
            directed=False,
            evidence=evidence,
        )

    omitted = schemas.GraphEdge(
        id="semantic:omitted",
        source="note-source",
        target="note-target",
        type=schemas.EdgeType.semantic,
        directed=False,
        evidence_omitted="response_byte_cap",
    )
    assert omitted.evidence is None
    assert schemas.GraphEdge.model_validate(omitted.model_dump(mode="python")) == omitted
    with pytest.raises(ValidationError, match="omission"):
        schemas.GraphEdge(
            id="semantic:ambiguous",
            source="note-source",
            target="note-target",
            type=schemas.EdgeType.semantic,
            directed=False,
            evidence=evidence,
            evidence_omitted="response_byte_cap",
        )


def test_semantic_status_is_typed_and_legacy_serialization_omits_new_fields() -> None:
    ordinary_edge = schemas.GraphEdge(
        id="manual-1",
        source="note-source",
        target="note-target",
        type=schemas.EdgeType.manual,
        directed=False,
    )
    ordinary_response = schemas.NoteGraphResponse(
        nodes=[],
        edges=[ordinary_edge],
        limits=schemas.GraphLimits(max_nodes=300, max_edges=1200, max_degree=40),
        active_note_count=0,
        all_notes_note_cap=100,
        all_notes_eligible=True,
    )

    assert "evidence" not in ordinary_edge.model_dump(mode="json")
    assert "semantic_status" not in ordinary_response.model_dump(mode="json")

    status = schemas.SemanticGraphStatus(
        available=True,
        state="ready",
        generation_id="generation-1",
        semantic_index_revision=8,
        configuration_revision=5,
        active_notes=15,
        indexed_notes=12,
        dirty_notes=1,
        excluded_notes=2,
        failed_notes=0,
        effective_top_k=10,
        effective_threshold=0.75,
        max_top_k=DEFAULT_SEMANTIC_INDEX_SETTINGS.max_query_neighbors,
        max_admission_nodes=50,
        max_admission_edges=50,
        max_evidence_pairs=3,
        max_excerpt_code_points=480,
        max_edge_evidence_code_points=2_880,
        max_response_evidence_bytes=256 * 1024,
        truncated_by=["semantic_candidates", "semantic_evidence_bytes"],
    )
    semantic_response = ordinary_response.model_copy(
        update={"semantic_status": status}
    )

    assert semantic_response.model_dump(mode="json")["semantic_status"] == status.model_dump(
        mode="json"
    )
    assert status.active_notes == 15
    with pytest.raises(ValidationError):
        schemas.SemanticGraphStatus(
            **{**status.model_dump(), "max_evidence_pairs": 4}
        )
