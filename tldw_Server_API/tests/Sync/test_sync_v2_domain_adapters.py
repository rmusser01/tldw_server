from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    SyncAdapterContext,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.chat import ChatDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.media import MediaCompatibilityAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.source_cache import SourceCacheAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.workspaces import WorkspacesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.models import SyncDataset, SyncEnvelope, SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _dataset(*, domains: list[str] | None = None) -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-1",
        owner_user_id="user-1",
        scope_type="personal",
        encryption_policy="client_private_v1",
        domains=domains or ["notes", "chat", "workspaces", "source_cache", "media"],
        workspace_id=None,
        metadata={},
        created_at="2026-05-10T00:00:00+00:00",
        updated_at="2026-05-10T00:00:00+00:00",
    )


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes",
        "entity_id": "note-1",
        "operation": "upsert",
        "adapter_version": 1,
        "device_id": "device-1",
        "stable_key": "note:note-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "base_version": "base-1",
        "entity_version": "v1",
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": "ciphertext:opaque",
        "payload_clear": {"entity_kind": "note", "status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 128,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _stored(envelope: SyncEnvelopeCreate, *, sequence: int = 1, status: str = "accepted") -> SyncEnvelope:
    return SyncEnvelope(
        server_sequence=sequence,
        dataset_id=envelope.dataset_id,
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.entity_id,
        operation=envelope.operation,
        adapter_version=envelope.adapter_version,
        server_timestamp="2026-05-10T00:00:00+00:00",
        device_id=envelope.device_id,
        stable_key=envelope.stable_key,
        client_timestamp=envelope.client_timestamp,
        base_version=envelope.base_version,
        entity_version=envelope.entity_version,
        dependencies=list(envelope.dependencies),
        routing_metadata=dict(envelope.routing_metadata),
        payload_ciphertext=envelope.payload_ciphertext,
        payload_clear=dict(envelope.payload_clear),
        payload_hash=envelope.payload_hash,
        payload_size_bytes=envelope.payload_size_bytes,
        status=status,
    )


def _context(*envelopes: SyncEnvelope) -> SyncAdapterContext:
    return SyncAdapterContext(prior_envelopes=list(envelopes))


def _adapter_for_domain(domain: str):
    return {
        "notes": NotesDomainAdapter,
        "chat": ChatDomainAdapter,
        "workspaces": WorkspacesDomainAdapter,
        "source_cache": SourceCacheAdapter,
    }[domain]()


def _domain_payload(domain: str) -> dict[str, object]:
    if domain == "source_cache":
        return {
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "payload_hash": "content-a",
        }
    return {"entity_kind": domain.rstrip("s")}


def test_notes_adapter_accepts_metadata_only_tag_status_merge():
    prior = _stored(
        _envelope(
            client_envelope_id="note-tags-a",
            payload_clear={"entity_kind": "note", "tag_ids": ["tag-1"], "status": "draft"},
            payload_hash="sha256:tags-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="note-tags-b",
        payload_clear={"entity_kind": "note", "tag_ids": ["tag-2"], "status": "active"},
        payload_hash="sha256:tags-b",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="note-tags-b")


def test_notes_adapter_conflicts_concurrent_encrypted_content_edits():
    prior = _stored(
        _envelope(
            client_envelope_id="note-content-a",
            routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
            payload_clear={"entity_kind": "note"},
            payload_hash="sha256:content-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="note-content-b",
        device_id="device-2",
        routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
        payload_clear={"entity_kind": "note"},
        payload_hash="sha256:content-b",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "encrypted_content_edit"
    assert outcome.entity_id == "note-1"


def test_notes_adapter_accepts_linear_encrypted_content_edit_against_current_head():
    prior = _stored(
        _envelope(
            client_envelope_id="note-content-a",
            routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
            payload_clear={"entity_kind": "note"},
            payload_hash="sha256:content-a",
            entity_version="note-v1",
        )
    )
    incoming = _envelope(
        client_envelope_id="note-content-b",
        routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
        payload_clear={"entity_kind": "note"},
        payload_hash="sha256:content-b",
        base_version="note-v1",
        entity_version="note-v2",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="note-content-b")


def test_notes_adapter_conflicts_stale_encrypted_content_edit():
    prior = _stored(
        _envelope(
            client_envelope_id="note-content-a",
            routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
            payload_clear={"entity_kind": "note"},
            payload_hash="sha256:content-a",
            entity_version="note-v2",
        )
    )
    incoming = _envelope(
        client_envelope_id="note-content-b",
        routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
        payload_clear={"entity_kind": "note"},
        payload_hash="sha256:content-b",
        base_version="note-v1",
        entity_version="note-v3",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "encrypted_content_edit"


def test_notes_adapter_accepts_linear_delete_after_upsert():
    prior = _stored(
        _envelope(
            client_envelope_id="note-update",
            payload_clear={"entity_kind": "note", "status": "active"},
            payload_hash="sha256:update",
            entity_version="note-v1",
        )
    )
    incoming = _envelope(
        client_envelope_id="note-delete",
        operation="delete",
        payload_clear={"entity_kind": "note", "deleted": True},
        payload_hash="sha256:delete",
        base_version="note-v1",
        entity_version="note-v2",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="note-delete")


def test_notes_adapter_conflicts_delete_vs_update():
    prior = _stored(
        _envelope(
            client_envelope_id="note-update",
            payload_clear={"entity_kind": "note", "status": "active"},
            payload_hash="sha256:update",
        )
    )
    incoming = _envelope(
        client_envelope_id="note-delete",
        operation="delete",
        payload_clear={"entity_kind": "note", "deleted": True},
        payload_hash="sha256:delete",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "delete_update_conflict"


@pytest.mark.parametrize("domain", ["notes", "chat", "workspaces", "source_cache"])
def test_domain_adapters_accept_linear_delete_after_upsert(domain: str):
    prior = _stored(
        _envelope(
            client_envelope_id=f"{domain}-update",
            domain=domain,
            payload_clear=_domain_payload(domain),
            payload_hash=f"sha256:{domain}-update",
            entity_version=f"{domain}-v1",
        )
    )
    incoming_payload = {**_domain_payload(domain), "deleted": True}
    incoming = _envelope(
        client_envelope_id=f"{domain}-delete",
        domain=domain,
        operation="delete",
        payload_clear=incoming_payload,
        payload_hash=f"sha256:{domain}-delete",
        base_version=f"{domain}-v1",
        entity_version=f"{domain}-v2",
    )

    outcome = _adapter_for_domain(domain).evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id=f"{domain}-delete")


@pytest.mark.parametrize("domain", ["notes", "chat", "workspaces", "source_cache"])
def test_domain_adapters_conflict_stale_delete_after_upsert(domain: str):
    prior = _stored(
        _envelope(
            client_envelope_id=f"{domain}-update",
            domain=domain,
            payload_clear=_domain_payload(domain),
            payload_hash=f"sha256:{domain}-update",
            entity_version=f"{domain}-v2",
        )
    )
    incoming_payload = {**_domain_payload(domain), "deleted": True}
    incoming = _envelope(
        client_envelope_id=f"{domain}-delete",
        domain=domain,
        operation="delete",
        payload_clear=incoming_payload,
        payload_hash=f"sha256:{domain}-delete",
        base_version=f"{domain}-v1",
        entity_version=f"{domain}-v3",
    )

    outcome = _adapter_for_domain(domain).evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "delete_update_conflict"


def test_chat_adapter_accepts_append_only_independent_message_ids():
    prior = _stored(
        _envelope(
            client_envelope_id="message-a",
            domain="chat",
            entity_id="message-1",
            stable_key="chat_message:message-1",
            routing_metadata={"entity_kind": "message"},
            payload_clear={"entity_kind": "message"},
            payload_hash="sha256:message-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="message-b",
        domain="chat",
        entity_id="message-2",
        stable_key="chat_message:message-2",
        routing_metadata={"entity_kind": "message"},
        payload_clear={"entity_kind": "message"},
        payload_hash="sha256:message-b",
    )

    outcome = ChatDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="message-b")


def test_chat_adapter_conflicts_same_message_id_with_different_hashes():
    prior = _stored(
        _envelope(
            client_envelope_id="message-a",
            domain="chat",
            entity_id="message-1",
            stable_key="chat_message:message-1",
            routing_metadata={"entity_kind": "message"},
            payload_clear={"entity_kind": "message"},
            payload_hash="sha256:message-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="message-b",
        domain="chat",
        entity_id="message-1",
        stable_key="chat_message:message-1",
        routing_metadata={"entity_kind": "message"},
        payload_clear={"entity_kind": "message"},
        payload_hash="sha256:message-b",
    )

    outcome = ChatDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "message_hash_mismatch"


def test_workspaces_adapter_accepts_source_ref_membership_by_source_id():
    prior = _stored(
        _envelope(
            client_envelope_id="workspace-source-a",
            domain="workspaces",
            entity_id="workspace-1:source-1",
            stable_key="workspace_source:workspace-1:source-1",
            operation="link",
            routing_metadata={"entity_kind": "workspace_source_ref", "workspace_id": "workspace-1"},
            payload_clear={
                "entity_kind": "workspace_source_ref",
                "workspace_id": "workspace-1",
                "source_id": "source-1",
            },
            payload_hash="sha256:source-1",
        )
    )
    incoming = _envelope(
        client_envelope_id="workspace-source-b",
        domain="workspaces",
        entity_id="workspace-1:source-2",
        stable_key="workspace_source:workspace-1:source-2",
        operation="link",
        routing_metadata={"entity_kind": "workspace_source_ref", "workspace_id": "workspace-1"},
        payload_clear={
            "entity_kind": "workspace_source_ref",
            "workspace_id": "workspace-1",
            "source_id": "source-2",
        },
        payload_hash="sha256:source-2",
    )

    outcome = WorkspacesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="workspace-source-b")


def test_workspaces_adapter_conflicts_ordered_or_rename_metadata_flags():
    incoming = _envelope(
        client_envelope_id="workspace-rename",
        domain="workspaces",
        entity_id="workspace-1",
        stable_key="workspace:workspace-1",
        routing_metadata={"entity_kind": "workspace", "conflict_kind": "rename"},
        payload_clear={"entity_kind": "workspace"},
        payload_hash="sha256:workspace-rename",
    )

    outcome = WorkspacesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "rename_conflict"


def test_source_cache_adapter_allows_same_source_id_with_different_content_hashes():
    prior = _stored(
        _envelope(
            client_envelope_id="cache-a",
            domain="source_cache",
            entity_id="source-1:content-a",
            stable_key="source_cache:source-1:content-a",
            routing_metadata={"entity_kind": "source_cache_entry"},
            payload_clear={
                "entity_kind": "source_cache_entry",
                "source_id": "source-1",
                "payload_hash": "content-a",
            },
            payload_hash="sha256:cache-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="cache-b",
        domain="source_cache",
        entity_id="source-1:content-b",
        stable_key="source_cache:source-1:content-b",
        routing_metadata={"entity_kind": "source_cache_entry"},
        payload_clear={
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "payload_hash": "content-b",
        },
        payload_hash="sha256:cache-b",
    )

    outcome = SourceCacheAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="cache-b")


def test_source_cache_adapter_conflicts_same_source_content_hash_with_different_payload():
    prior = _stored(
        _envelope(
            client_envelope_id="cache-a",
            domain="source_cache",
            entity_id="source-1:content-a",
            stable_key="source_cache:source-1:content-a",
            routing_metadata={"entity_kind": "source_cache_entry"},
            payload_clear={
                "entity_kind": "source_cache_entry",
                "source_id": "source-1",
                "payload_hash": "content-a",
            },
            payload_hash="sha256:cache-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="cache-b",
        domain="source_cache",
        entity_id="source-1:content-a",
        stable_key="source_cache:source-1:content-a",
        routing_metadata={"entity_kind": "source_cache_entry"},
        payload_clear={
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "payload_hash": "content-a",
        },
        payload_hash="sha256:cache-b",
    )

    outcome = SourceCacheAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "source_cache_hash_mismatch"


def test_service_persists_domain_adapter_conflicts(tmp_path: Path):
    registry = SyncAdapterRegistry([ChatDomainAdapter()])
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_domain_adapters.db")),
        adapters=registry,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["chat"])

    accepted = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="message-a",
                domain="chat",
                entity_id="message-1",
                stable_key="chat_message:message-1",
                routing_metadata={"entity_kind": "message"},
                payload_clear={"entity_kind": "message"},
                payload_hash="sha256:message-a",
            )
        ],
    )
    conflicted = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="message-b",
                domain="chat",
                entity_id="message-1",
                stable_key="chat_message:message-1",
                routing_metadata={"entity_kind": "message"},
                payload_clear={"entity_kind": "message"},
                payload_hash="sha256:message-b",
            )
        ],
    )

    conflicts = service.list_conflicts(user_id="user-1", dataset_id="dataset-1")
    assert accepted.accepted[0].client_envelope_id == "message-a"
    assert conflicted.conflicts[0].conflict_id == "conflict-generated"
    assert conflicts[0].conflict_type == "message_hash_mismatch"
    assert conflicts[0].local_envelope_id == "message-b"


def test_default_sync_v2_registry_uses_concrete_v1_domain_adapters():
    registry = sync_endpoint._default_sync_v2_registry()

    assert isinstance(registry.get("notes"), NotesDomainAdapter)
    assert isinstance(registry.get("chat"), ChatDomainAdapter)
    assert isinstance(registry.get("workspaces"), WorkspacesDomainAdapter)
    assert isinstance(registry.get("source_cache"), SourceCacheAdapter)
    assert isinstance(registry.get("media"), MediaCompatibilityAdapter)
