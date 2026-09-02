"""Semantic relationship conversion through the canonical manual-link route."""

# ruff: noqa: F401, F811 - pytest collects the imported shared fixture by name.

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes_graph as endpoint
from tldw_Server_API.app.core.Notes_Graph.semantic_projector import (
    SemanticProjectionError,
)

from .test_graph_endpoint import (
    _create_note,
    _headers,
    client_and_db,
)

pytestmark = pytest.mark.integration


class _ConversionProjector:
    def __init__(self, error_code: str | None = None) -> None:
        self.error_code = error_code
        self.calls: list[dict[str, str]] = []

    async def validate_conversion(self, **kwargs: str) -> None:
        self.calls.append(dict(kwargs))
        if self.error_code is not None:
            raise SemanticProjectionError(self.error_code)


@pytest.mark.asyncio
async def test_semantic_conversion_audit_uses_durable_content_free_record(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps

    audit_calls: list[dict[str, object]] = []
    flush_calls: list[bool] = []

    class _AuditService:
        async def log_event(self, **kwargs: object) -> str:
            audit_calls.append(dict(kwargs))
            return "event-a"

        async def flush(self, *, raise_on_failure: bool = False) -> bool:
            flush_calls.append(raise_on_failure)
            return True

    async def fake_service(user_id: str) -> _AuditService:
        assert user_id == "owner-a"
        return _AuditService()

    monkeypatch.setattr(
        Audit_DB_Deps,
        "get_or_create_audit_service_for_user_id_optional",
        fake_service,
    )

    await endpoint._audit_semantic_conversion(
        actor_user_id="owner-a",
        source_note_id="note-source",
        target_note_id="note-target",
        generation_id="generation-a",
        result="created",
    )

    assert len(audit_calls) == 1
    event = audit_calls[0]
    assert event["context"].user_id == "owner-a"
    assert event["resource_type"] == "notes_semantic_relationship"
    assert event["resource_id"] == "note-source"
    assert event["action"] == "notes_semantic.manual_conversion"
    assert event["result"] == "created"
    assert event["metadata"] == {
        "target_note_id": "note-target",
        "generation_id": "generation-a",
    }
    assert event["context"].endpoint is None
    serialized = repr(event).lower()
    for forbidden in ("excerpt", "content", "vector", "credential", "https://"):
        assert forbidden not in serialized
    assert flush_calls == [True]


def test_valid_semantic_conversion_uses_existing_link_writer_and_omits_context(
    client_and_db,
    monkeypatch,
) -> None:
    client, db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    projector = _ConversionProjector()
    audit_calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    async def record_audit(**kwargs: str) -> None:
        audit_calls.append(dict(kwargs))

    monkeypatch.setattr(endpoint, "_audit_semantic_conversion", record_audit)

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "directed": True,
            "weight": 0.25,
            "properties": {"kept": "value"},
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    edge = response.json()["edge"]
    assert edge["directed"] is False
    assert edge["weight"] == 1.0
    assert edge["properties"] == {"kept": "value"}
    stored = db.notes_link_store.get(edge["edge_id"])
    assert stored is not None
    assert stored.properties == {"kept": "value"}
    assert projector.calls == [
        {
            "source_note_id": source,
            "target_note_id": target,
            "generation_id": "generation-a",
        }
    ]
    assert audit_calls == [
        {
            "actor_user_id": "1",
            "source_note_id": source,
            "target_note_id": target,
            "generation_id": "generation-a",
            "result": "created",
        }
    ]


def test_existing_manual_link_returns_typed_semantic_conversion_conflict(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    created = client.post(
        f"/api/v1/notes/{source}/links",
        json={"to_note_id": target, "directed": False, "weight": 1.0},
        headers=_headers(),
    )
    assert created.status_code == 200, created.text
    projector = _ConversionProjector()
    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "notes_semantic_conversion_manual_link_exists"
    )
    assert len(projector.calls) == 1


def test_stale_semantic_generation_wins_over_existing_manual_link_refresh(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    created = client.post(
        f"/api/v1/notes/{source}/links",
        json={"to_note_id": target, "directed": False, "weight": 1.0},
        headers=_headers(),
    )
    assert created.status_code == 200, created.text
    projector = _ConversionProjector(
        "notes_semantic_conversion_generation_stale"
    )
    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "notes_semantic_conversion_generation_stale"
    )


def test_semantic_conversion_audit_failure_preserves_committed_manual_link(
    client_and_db,
    monkeypatch,
) -> None:
    client, db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    projector = _ConversionProjector()
    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    async def fail_audit(**_kwargs: str) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(endpoint, "_audit_semantic_conversion", fail_audit)

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    edge = response.json()["edge"]
    stored = db.notes_link_store.get(edge["edge_id"])
    assert stored is not None
    assert {stored.source_note_id, stored.target_note_id} == {source, target}


@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        ("notes_semantic_conversion_generation_stale", 409),
        ("notes_semantic_conversion_pair_mismatch", 409),
        ("notes_semantic_conversion_owner_mismatch", 404),
    ],
)
def test_stale_foreign_or_pair_mismatched_conversion_is_rejected_before_write(
    client_and_db,
    monkeypatch,
    error_code: str,
    expected_status: int,
) -> None:
    client, db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    projector = _ConversionProjector(error_code)
    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == expected_status, response.text
    assert response.json()["detail"]["error_code"] == error_code
    assert db.notes_link_store.list_page(limit=10, after_edge_id=None) == ()


def test_legacy_manual_link_caller_does_not_initialize_semantic_runtime(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")

    def unexpected_builder(**_kwargs):
        raise AssertionError("legacy manual link initialized semantic runtime")

    monkeypatch.setattr(endpoint, "_build_semantic_graph_projector", unexpected_builder)

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={"to_note_id": target, "directed": True, "weight": 0.5},
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    assert response.json()["edge"]["directed"] is True
    assert response.json()["edge"]["weight"] == 0.5
