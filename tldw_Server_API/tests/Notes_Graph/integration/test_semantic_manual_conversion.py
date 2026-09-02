"""Semantic relationship conversion through the canonical manual-link route."""

# ruff: noqa: F401, F811 - pytest collects the imported shared fixture by name.

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes_graph as endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Graph.semantic_projector import (
    SemanticProjectionError,
)
from tldw_Server_API.app.core.Sync.v2.adapters import (
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import (
    NotesLinkDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import (
    NotesLinkMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    SyncEnvelopeCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkPreflightError,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

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


def _activate_notes_link_sync(
    *,
    tmp_path: Path,
    db: CharactersRAGDB,
    note_ids: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SyncV2Store(
        SyncDatabase(sqlite_path=tmp_path / "sync.db", user_id="1")
    )
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain) for domain in M1_SYNC_DOMAINS]
        + [NotesLinkDomainAdapter()]
    )
    service = SyncV2Service(
        store=store,
        adapters=registry,
        materializers={"notes.link": NotesLinkMaterializer(db)},
        clock=lambda: "2026-08-10T12:00:00+00:00",
        id_factory=lambda prefix: f"{prefix}-semantic-conversion",
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            )
        ),
    )
    profile = service.bootstrap_profile(
        user_id="1",
        mode="offline_sync",
        device_id="device-semantic-conversion",
    )
    assert profile.dataset is not None
    dataset_id = profile.dataset.dataset_id
    for index, note_id in enumerate(note_ids, start=1):
        envelope = store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id=f"semantic-note-{index}",
                domain="notes.note",
                operation="upsert",
                object_id=note_id,
                device_id="device-semantic-conversion",
                object_revision=1,
                entity_version=1,
                payload={"title": note_id, "content": "body"},
                payload_hash=f"sha256:semantic-note-{index}",
                created_at_client="2026-08-10T12:00:00+00:00",
                status="accepted",
            )
        )
        assert envelope.server_cursor is not None
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=dataset_id,
                domain="notes.note",
                object_id=note_id,
                object_revision=1,
                object_hash=envelope.payload_hash or "",
                latest_server_cursor=envelope.server_cursor,
                deleted=False,
            )
        )
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="applied",
        )
    store.begin_notes_link_bootstrap(
        dataset_id,
        owner_user_id="1",
        bootstrap_id="semantic-links-ready",
    )
    store.transition_notes_link_bootstrap(
        dataset_id,
        bootstrap_id="semantic-links-ready",
        expected_state="initializing",
        state="ready",
        captured_count=0,
        expected_count=0,
        source_hash=None,
        ready_verifier=lambda: True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.notes_link_coordinator."
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: service if user_id == "1" else None,
    )


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


def test_active_sync_duplicate_returns_typed_semantic_conversion_conflict(
    client_and_db,
    monkeypatch,
    tmp_path: Path,
) -> None:
    client, db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    _activate_notes_link_sync(
        tmp_path=tmp_path,
        db=db,
        note_ids=(source, target),
        monkeypatch=monkeypatch,
    )
    created = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "directed": False,
            "weight": 1.0,
            "idempotency_key": "existing-manual-link",
        },
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
        f"/api/v1/notes/{target}/links",
        json={
            "to_note_id": source,
            "idempotency_key": "semantic-conversion-duplicate",
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "notes_semantic_conversion_manual_link_exists"
    )
    assert len(projector.calls) == 1


def test_preflight_failure_without_live_manual_link_remains_fail_closed(
    client_and_db,
    monkeypatch,
) -> None:
    client, _db = client_and_db
    source = _create_note(client, "Source", "source body")
    target = _create_note(client, "Target", "target body")
    projector = _ConversionProjector()

    class _PreflightCoordinator:
        def create(self, **_kwargs):
            raise NotesLinkPreflightError()

    monkeypatch.setattr(
        endpoint,
        "_build_semantic_graph_projector",
        lambda **_kwargs: projector,
    )
    monkeypatch.setattr(
        endpoint,
        "resolve_notes_link_coordinator",
        lambda **_kwargs: _PreflightCoordinator(),
    )

    response = client.post(
        f"/api/v1/notes/{source}/links",
        json={
            "to_note_id": target,
            "idempotency_key": "semantic-conversion-preflight",
            "semantic_conversion": {"generation_id": "generation-a"},
        },
        headers=_headers(),
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == "notes_link_preflight_failed"
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
