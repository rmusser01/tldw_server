"""Nested HTTP contracts for Notes semantic-index management."""

from __future__ import annotations

from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import notes_semantic_index as endpoint
from tldw_Server_API.app.core.Notes_Graph.semantic_api import SemanticAPIError


class _FakeAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.failures: dict[str, SemanticAPIError] = {}

    def _record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        failure = self.failures.get(name)
        if failure is not None:
            raise failure

    def capabilities(self):
        self._record("capabilities")
        return {
            "active_note_count": 4,
            "estimated_chunk_count": 8,
            "estimated_run_count": 1,
            "provider_label": "OpenAI",
            "model": "text-embedding-3-small",
            "execution_boundary": "external",
            "storage_boundary": "local",
            "storage_label": "ChromaDB",
            "outbound_data_categories": ["note_content_chunks", "note_title"],
            "capability_revision": f"sha256:{'a' * 64}",
            "indexing_available": True,
            "unavailable_reason": None,
            "metric": "cosine",
            "resolved_dimensions": 1536,
        }

    def status(self):
        self._record("status")
        return {
            "state": "ready",
            "detail_reason": None,
            "desired_state": "enabled",
            "configuration_revision": 9,
            "semantic_index_revision": 2,
            "active_generation_id": "generation-a",
            "indexed_notes": 4,
            "excluded_notes": 0,
            "failed_notes": 0,
            "pending_notes": 0,
            "published_chunks": 8,
            "cleanup_pending": False,
            "active_run": None,
        }

    def enable(self, **kwargs):
        self._record("enable", **kwargs)
        return _mutation("build")

    def disable(self, **kwargs):
        self._record("disable", **kwargs)
        return _mutation("delete")

    def create_run(self, **kwargs):
        self._record("create_run", **kwargs)
        return _run(kwargs["mode"])

    def get_run(self, **kwargs):
        self._record("get_run", **kwargs)
        return _run("rebuild", run_id=str(kwargs["run_id"]))

    def cancel_run(self, **kwargs):
        self._record("cancel_run", **kwargs)
        return _mutation("rebuild", run_id=str(kwargs["run_id"]))


RUN_ID = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"


def _run(mode: str, *, run_id: str = RUN_ID) -> dict[str, object]:
    return {
        "run_id": run_id,
        "mode": mode,
        "status": "queued",
        "revision": 9,
        "indexed_notes": 0,
        "excluded_notes": 0,
        "failed_notes": 0,
        "pending_notes": 4,
        "published_chunks": 0,
        "cleanup_complete": False,
        "error_code": None,
        "link": f"/api/v1/notes/graph/semantic-index/runs/{run_id}",
    }


def _mutation(mode: str, *, run_id: str = RUN_ID) -> dict[str, object]:
    return {
        "resource": {
            "state": "preparing" if mode != "delete" else "off",
            "detail_reason": "building" if mode != "delete" else "cleanup_pending",
            "desired_state": "enabled" if mode != "delete" else "disabled",
            "configuration_revision": 9,
            "semantic_index_revision": 2,
            "active_generation_id": None,
            "indexed_notes": 0,
            "excluded_notes": 0,
            "failed_notes": 0,
            "pending_notes": 4,
            "published_chunks": 0,
            "cleanup_pending": mode == "delete",
            "active_run": None,
        },
        "run": _run(mode, run_id=run_id),
    }


@pytest.fixture
def client():
    api = _FakeAPI()
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/notes")
    app.dependency_overrides[endpoint.get_semantic_api] = lambda: api
    app.dependency_overrides[endpoint.require_semantic_read] = lambda: object()
    app.dependency_overrides[endpoint.require_semantic_manage] = lambda: object()
    with TestClient(app) as test_client:
        yield test_client, api, app


def test_all_seven_routes_are_nested_and_main_status_has_no_history(client) -> None:
    test_client, _api, app = client
    expected = {
        ("GET", "/api/v1/notes/graph/semantic-index/capabilities"),
        ("GET", "/api/v1/notes/graph/semantic-index"),
        ("PUT", "/api/v1/notes/graph/semantic-index"),
        ("DELETE", "/api/v1/notes/graph/semantic-index"),
        ("POST", "/api/v1/notes/graph/semantic-index/runs"),
        ("GET", "/api/v1/notes/graph/semantic-index/runs/{run_id}"),
        ("POST", "/api/v1/notes/graph/semantic-index/runs/{run_id}/cancel"),
    }
    actual = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if "semantic-index" in route.path
    }
    assert actual == expected

    response = test_client.get("/api/v1/notes/graph/semantic-index")
    assert response.status_code == 200
    assert "runs" not in response.json()
    assert response.json()["active_run"] is None


def test_enable_binds_capability_revision_and_returns_202(client) -> None:
    test_client, api, _app = client
    response = test_client.put(
        "/api/v1/notes/graph/semantic-index",
        headers={"Idempotency-Key": "enable-key"},
        json={
            "expected_revision": 0,
            "capability_revision": f"sha256:{'a' * 64}",
        },
    )

    assert response.status_code == 202
    UUID(response.json()["run"]["run_id"])
    assert api.calls[-1] == (
        "enable",
        {
            "expected_revision": 0,
            "capability_revision": f"sha256:{'a' * 64}",
            "idempotency_key": "enable-key",
        },
    )


def test_delete_and_cancel_return_202_with_revision_and_idempotency(client) -> None:
    test_client, api, _app = client
    deleted = test_client.request(
        "DELETE",
        "/api/v1/notes/graph/semantic-index",
        headers={"Idempotency-Key": "delete-key"},
        json={"expected_revision": 9},
    )
    cancelled = test_client.post(
        f"/api/v1/notes/graph/semantic-index/runs/{RUN_ID}/cancel",
        headers={"Idempotency-Key": "cancel-key"},
        json={"expected_revision": 9},
    )

    assert deleted.status_code == cancelled.status_code == 202
    assert api.calls[-2][0] == "disable"
    assert api.calls[-1] == (
        "cancel_run",
        {
            "run_id": UUID(RUN_ID),
            "expected_revision": 9,
            "idempotency_key": "cancel-key",
        },
    )


@pytest.mark.parametrize("mode", ["rebuild", "retry_failed"])
def test_run_creation_accepts_only_the_two_public_modes(client, mode: str) -> None:
    test_client, api, _app = client
    response = test_client.post(
        "/api/v1/notes/graph/semantic-index/runs",
        headers={"Idempotency-Key": f"run-{mode}"},
        json={"mode": mode, "expected_revision": 9},
    )

    assert response.status_code == 202
    assert response.json()["mode"] == mode
    assert api.calls[-1][1]["mode"] == mode


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("PUT", "/api/v1/notes/graph/semantic-index", {"expected_revision": 0, "capability_revision": "cap"}),
        ("DELETE", "/api/v1/notes/graph/semantic-index", {"expected_revision": 9}),
        ("POST", "/api/v1/notes/graph/semantic-index/runs", {"mode": "rebuild", "expected_revision": 9}),
        ("POST", f"/api/v1/notes/graph/semantic-index/runs/{RUN_ID}/cancel", {"expected_revision": 9}),
    ],
)
def test_every_mutation_requires_idempotency_key(client, method, path, body) -> None:
    test_client, _api, _app = client
    response = test_client.request(method, path, json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "notes_semantic_invalid_request"


def test_invalid_mode_and_revision_return_typed_422(client) -> None:
    test_client, _api, _app = client
    invalid_mode = test_client.post(
        "/api/v1/notes/graph/semantic-index/runs",
        headers={"Idempotency-Key": "invalid-mode"},
        json={"mode": "repair_everything", "expected_revision": 9},
    )
    missing_revision = test_client.request(
        "DELETE",
        "/api/v1/notes/graph/semantic-index",
        headers={"Idempotency-Key": "missing-revision"},
        json={},
    )

    assert invalid_mode.status_code == missing_revision.status_code == 422
    assert invalid_mode.json()["detail"]["error_code"] == "notes_semantic_invalid_request"
    assert missing_revision.json()["detail"]["error_code"] == "notes_semantic_invalid_request"


def test_foreign_run_is_404_and_conflicts_are_typed(client) -> None:
    test_client, api, _app = client
    api.failures["get_run"] = SemanticAPIError(404, "notes_semantic_run_not_found")
    missing = test_client.get(
        "/api/v1/notes/graph/semantic-index/runs/16f923f0-cfc5-455b-bc44-df610b433991"
    )
    api.failures["create_run"] = SemanticAPIError(409, "notes_semantic_writer_conflict")
    conflict = test_client.post(
        "/api/v1/notes/graph/semantic-index/runs",
        headers={"Idempotency-Key": "conflict"},
        json={"mode": "rebuild", "expected_revision": 9},
    )

    assert missing.status_code == 404
    assert missing.json()["detail"]["error_code"] == "notes_semantic_run_not_found"
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["error_code"] == "notes_semantic_writer_conflict"


def test_unavailable_enable_is_sanitized_503(client) -> None:
    test_client, api, _app = client
    api.failures["enable"] = SemanticAPIError(
        503,
        "notes_semantic_provider_unavailable",
    )
    response = test_client.put(
        "/api/v1/notes/graph/semantic-index",
        headers={"Idempotency-Key": "unavailable"},
        json={
            "expected_revision": 0,
            "capability_revision": f"sha256:{'a' * 64}",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "error_code": "notes_semantic_provider_unavailable",
        "message": "Semantic indexing is temporarily unavailable.",
    }
    assert "secret" not in response.text.lower()


def test_quota_rejection_is_typed_429(client) -> None:
    test_client, api, _app = client
    api.failures["create_run"] = SemanticAPIError(
        429,
        "notes_semantic_quota_exceeded",
    )

    response = test_client.post(
        "/api/v1/notes/graph/semantic-index/runs",
        headers={"Idempotency-Key": "quota"},
        json={"mode": "rebuild", "expected_revision": 9},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == {
        "error_code": "notes_semantic_quota_exceeded",
        "message": "The semantic indexing quota has been reached.",
    }


def test_manage_permission_failure_is_403(client) -> None:
    test_client, _api, app = client

    def forbidden():
        raise HTTPException(status_code=403, detail="forbidden")

    app.dependency_overrides[endpoint.require_semantic_manage] = forbidden
    response = test_client.post(
        "/api/v1/notes/graph/semantic-index/runs",
        headers={"Idempotency-Key": "forbidden"},
        json={"mode": "rebuild", "expected_revision": 9},
    )
    assert response.status_code == 403
