"""Endpoint integration for opt-in semantic projection and formatting."""

# ruff: noqa: F401, F811 - pytest collects the imported shared fixture by name.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import notes_graph as endpoint
from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    SemanticEdgeEvidence,
    SemanticGraphStatus,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_projector import (
    SemanticProjectionError,
)

from .test_graph_endpoint import (
    _create_note,
    _headers,
    client_and_db,
)

pytestmark = pytest.mark.integration


def _status() -> SemanticGraphStatus:
    return SemanticGraphStatus(
        available=True,
        state="ready",
        generation_id="generation-a",
        semantic_index_revision=11,
        configuration_revision=7,
        active_notes=2,
        indexed_notes=2,
        dirty_notes=0,
        excluded_notes=0,
        failed_notes=0,
        effective_top_k=2,
        effective_threshold=0.75,
        max_top_k=50,
        max_admission_nodes=2,
        max_admission_edges=2,
        max_evidence_pairs=3,
        max_excerpt_code_points=480,
        max_edge_evidence_code_points=2_880,
        max_response_evidence_bytes=256 * 1024,
    )


class _Projector:
    def __init__(self, target_note_id: str) -> None:
        self.target_note_id = target_note_id
        self.calls: list[object] = []

    async def project(self, request, ordinary, *, user):
        self.calls.append((request, str(user.id_str)))
        evidence = SemanticEdgeEvidence(
            similarity=0.9,
            qualitative_band="very_high",
            source_note_id=request.center_note_id,
            target_note_id=self.target_note_id,
            source_content_version=1,
            target_content_version=1,
            generation_id="generation-a",
            semantic_index_revision=11,
            configuration_revision=7,
            normalization_version="notes-semantic-normalization-v1",
            chunker_version="notes-semantic-chunker-v1",
            provider_label="Provider",
            model_label="Model",
        )
        return ordinary.model_copy(
            update={
                "nodes": [
                    *ordinary.nodes,
                    GraphNode(
                        id=self.target_note_id,
                        type="note",
                        label="Semantic target",
                        degree=1,
                    ),
                ],
                "edges": [
                    *ordinary.edges,
                    GraphEdge(
                        id="semantic-edge",
                        source=request.center_note_id,
                        target=self.target_note_id,
                        type=EdgeType.semantic,
                        directed=False,
                        weight=0.9,
                        evidence=evidence,
                    ),
                ],
                "semantic_status": _status(),
            }
        )


def test_ordinary_graph_does_not_build_semantic_runtime(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    note_id = _create_note(client, "Ordinary", "ordinary body")

    def unexpected_builder(**_kwargs):
        raise AssertionError("ordinary graph initialized semantic runtime")

    monkeypatch.setattr(endpoint, "_build_semantic_graph_projector", unexpected_builder)

    response = client.get(
        "/api/v1/notes/graph",
        params={"center_note_id": note_id, "edge_types": "manual"},
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    assert "semantic_status" not in response.json()


def test_semantic_graph_is_awaited_and_rate_limited_only_when_requested(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    focus = _create_note(client, "Focus", "focus body")
    target = _create_note(client, "Target", "target body")
    projector = _Projector(target)
    rate_calls: list[str] = []

    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    async def record_semantic_rate(_request, resource, _db_pool):
        rate_calls.append(resource)

    monkeypatch.setattr(endpoint, "enforce_rbac_rate_limit", record_semantic_rate)

    ordinary = client.get(
        "/api/v1/notes/graph",
        params={"center_note_id": focus, "edge_types": "manual"},
        headers=_headers(),
    )
    semantic = client.get(
        "/api/v1/notes/graph",
        params={
            "center_note_id": focus,
            "edge_types": "semantic",
            "semantic_top_k": 2,
            "semantic_threshold": 0.75,
        },
        headers=_headers(),
    )

    assert ordinary.status_code == 200, ordinary.text
    assert semantic.status_code == 200, semantic.text
    assert rate_calls == ["notes.graph.read", "notes.graph.semantic.read"]
    assert len(projector.calls) == 1
    assert semantic.json()["semantic_status"]["state"] == "ready"
    assert semantic.json()["edges"][0]["evidence"]["similarity"] == 0.9


def test_semantic_controls_without_semantic_edge_are_rejected(client_and_db) -> None:
    client, _db = client_and_db
    focus = _create_note(client, "Focus", "focus body")

    response = client.get(
        "/api/v1/notes/graph",
        params={
            "center_note_id": focus,
            "edge_types": "manual",
            "semantic_top_k": 2,
        },
        headers=_headers(),
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error_code"] == "notes_semantic_invalid_request"


def test_semantic_query_keys_do_not_remap_other_route_validation(client_and_db) -> None:
    client, _db = client_and_db

    response = client.get(
        "/api/v1/notes/graph/orphans",
        params={"limit": 0, "semantic_top_k": 2},
        headers=_headers(),
    )

    assert response.status_code == 422, response.text
    assert isinstance(response.json()["detail"], list)


def test_nonsemantic_graph_validation_keeps_framework_error_shape(client_and_db) -> None:
    client, _db = client_and_db

    response = client.get(
        "/api/v1/notes/graph",
        params={"edge_types": "semantic", "radius": 99},
        headers=_headers(),
    )

    assert response.status_code == 422, response.text
    assert isinstance(response.json()["detail"], list)


@pytest.mark.asyncio
async def test_graph_rate_limit_selects_exactly_one_request_specific_resource(
    monkeypatch,
) -> None:
    calls: list[str] = []

    async def record_rate(_request, resource, _db_pool):
        calls.append(resource)

    monkeypatch.setattr(endpoint, "enforce_rbac_rate_limit", record_rate)
    ordinary = Request({"type": "http", "method": "GET", "path": "/graph", "query_string": b"edge_types=manual"})
    semantic = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/graph",
            "query_string": b"edge_types=manual%2Csemantic",
        }
    )

    await endpoint._enforce_graph_request_rate_limit(ordinary, object())
    await endpoint._enforce_graph_request_rate_limit(semantic, object())

    assert calls == ["notes.graph.read", "notes.graph.semantic.read"]


def test_semantic_graph_rate_resource_is_registered() -> None:
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import load_catalog

    semantic_scope = next(scope for scope in load_catalog().scopes if scope.id == "notes.graph.semantic.read")

    assert semantic_scope.rate_limit_class == "standard"


def test_dynamic_graph_rate_resources_are_visible_to_privilege_introspection() -> None:
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import load_catalog
    from tldw_Server_API.app.core.PrivilegeMaps.introspection import (
        collect_privilege_route_registry,
    )

    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/notes")
    registry = collect_privilege_route_registry(app, load_catalog(), strict=True)
    graph_route = next(route for route in registry["notes.graph.read"] if route.path == "/api/v1/notes/graph")

    assert graph_route.rate_limit_resources == (
        "notes.graph.read",
        "notes.graph.semantic.read",
    )


def test_semantic_cursor_conflict_is_returned_as_typed_409(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    focus = _create_note(client, "Focus", "focus body")

    class _ConflictProjector:
        async def project(self, *_args, **_kwargs):
            raise SemanticProjectionError("notes_semantic_cursor_mismatch")

    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: _ConflictProjector(),
    )

    response = client.get(
        "/api/v1/notes/graph",
        params={"center_note_id": focus, "edge_types": "semantic"},
        headers=_headers(),
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == "notes_semantic_cursor_mismatch"


def test_cytoscape_semantic_response_preserves_status_and_evidence(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    focus = _create_note(client, "Focus", "focus body")
    target = _create_note(client, "Target", "target body")
    projector = _Projector(target)

    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    async def no_op_rate(_request, _resource, _db_pool):
        return None

    monkeypatch.setattr(endpoint, "enforce_rbac_rate_limit", no_op_rate)

    response = client.get(
        "/api/v1/notes/graph",
        params={
            "center_note_id": focus,
            "edge_types": "semantic",
            "format": "cytoscape",
        },
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["semantic_status"]["generation_id"] == "generation-a"
    semantic_edges = [item["data"] for item in data["elements"]["edges"] if item["data"]["type"] == "semantic"]
    assert semantic_edges[0]["evidence"]["similarity"] == 0.9
