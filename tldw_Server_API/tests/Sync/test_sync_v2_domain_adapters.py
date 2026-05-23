from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    AttachmentRefAdapter,
    StaticSyncAdapter,
    SyncAdapterContext,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters._lineage import incoming_references_head
from tldw_Server_API.app.core.Sync.v2.domain_adapters.chat import ChatDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.media import MediaMetadataAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.source_cache import SourceCacheAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.workspaces import WorkspacesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    WORKSPACE_SYNC_DOMAINS,
    SyncDataset,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _dataset(*, domains: list[str] | None = None) -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-1",
        owner_user_id="user-1",
        scope_type="personal",
        encryption_policy="client_private_v1",
        domains=domains
        or [
            "notes",
            "chat",
            "workspaces",
            "source_cache.entry",
            "media.item",
            "media.keyword",
            "media.keyword_link",
            "media",
        ],
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


def _attachment_ref_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-attachment-1",
        "domain": "attachment.ref",
        "entity_id": "att-1",
        "operation": "upsert",
        "adapter_version": 1,
        "device_id": "device-1",
        "stable_key": "attachment:att-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "routing_metadata": {"entity_kind": "attachment_ref"},
        "payload_clear": {
            "attachment_id": "att-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "content_type": "image/png",
            "size_bytes": 512,
            "payload_hash": "sha256:blob-v1",
            "availability": "client_local",
        },
        "payload_hash": "sha256:blob-v1",
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
        "source_cache.entry": SourceCacheAdapter,
        "media.item": MediaMetadataAdapter,
        "media.keyword": MediaMetadataAdapter,
        "media.keyword_link": MediaMetadataAdapter,
    }[domain](domain=domain)


def _domain_payload(domain: str) -> dict[str, object]:
    if domain == "source_cache.entry":
        return {
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "content_hash": "sha256:content-a",
            "provenance": {"kind": "url", "uri": "https://example.test/source"},
        }
    if domain == "media.item":
        return {"media_id": "media-1", "media_type": "video", "title": "Lecture"}
    if domain == "media.keyword":
        return {"keyword_id": "keyword-1", "name": "research"}
    if domain == "media.keyword_link":
        return {"media_id": "media-1", "keyword_id": "keyword-1"}
    return {"entity_kind": domain.rstrip("s")}


def _domain_identity_kwargs(domain: str) -> dict[str, object]:
    if domain == "media.item":
        return {
            "entity_id": "media-1",
            "stable_key": "media.item:media-1",
            "routing_metadata": {"entity_kind": "media_item"},
        }
    if domain == "media.keyword":
        return {
            "entity_id": "keyword-1",
            "stable_key": "media.keyword:keyword-1",
            "routing_metadata": {"entity_kind": "media_keyword"},
        }
    if domain == "media.keyword_link":
        return {
            "entity_id": "media-1:keyword-1",
            "stable_key": "media.keyword_link:media-1:keyword-1",
            "routing_metadata": {"entity_kind": "media_keyword_link"},
        }
    return {}


def _ready_sync_settings() -> SyncV2Settings:
    return SyncV2Settings(
        server_trusted_encryption=server_trusted_encryption_status_from_config(
            mode="encrypted_volume",
            server_trusted_enabled=True,
            auth_mode="multi_user",
        )
    )


def test_default_attachment_ref_adapter_rejects_invalid_parent_domain():
    default_sync_v2_registry.cache_clear()
    adapter = default_sync_v2_registry().get("attachment.ref")

    outcome = adapter.evaluate_envelope(
        _attachment_ref_envelope(
            payload_clear={
                "attachment_id": "att-1",
                "parent_domain": "media",
                "parent_object_id": "media-1",
                "content_type": "image/png",
                "size_bytes": 512,
                "payload_hash": "sha256:blob-v1",
                "availability": "client_local",
            },
        ),
        dataset=_dataset(domains=list(M1_SYNC_DOMAINS)),
        context=_context(),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "attachment_ref_parent_domain_invalid"


def test_default_attachment_ref_adapter_conflicts_divergent_stable_payload_hash():
    default_sync_v2_registry.cache_clear()
    adapter = default_sync_v2_registry().get("attachment.ref")
    prior = _stored(_attachment_ref_envelope())

    outcome = adapter.evaluate_envelope(
        _attachment_ref_envelope(
            client_envelope_id="env-attachment-divergent",
            payload_clear={
                "attachment_id": "att-1",
                "parent_domain": "notes.note",
                "parent_object_id": "note-1",
                "content_type": "image/jpeg",
                "size_bytes": 512,
                "payload_hash": "sha256:blob-v2",
                "availability": "client_local",
            },
            payload_hash="sha256:blob-v2",
        ),
        dataset=_dataset(domains=list(M1_SYNC_DOMAINS)),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.domain == "attachment.ref"
    assert outcome.conflict_type == "attachment_ref_hash_mismatch"


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


def test_notes_adapter_accepts_content_edit_based_on_latest_content_head_after_metadata_head():
    content_v1 = _stored(
        _envelope(
            client_envelope_id="note-content-a",
            routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
            payload_clear={"entity_kind": "note"},
            payload_hash="sha256:content-a",
            entity_version="note-content-v1",
        ),
        sequence=1,
    )
    metadata_v2 = _stored(
        _envelope(
            client_envelope_id="note-tags-b",
            payload_clear={"entity_kind": "note", "tag_ids": ["tag-2"], "status": "active"},
            payload_hash="sha256:tags-b",
            entity_version="note-metadata-v2",
        ),
        sequence=2,
    )
    incoming = _envelope(
        client_envelope_id="note-content-c",
        routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
        payload_clear={"entity_kind": "note"},
        payload_hash="sha256:content-c",
        base_version="note-content-v1",
        entity_version="note-content-v2",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(content_v1, metadata_v2),
    )

    assert outcome == AdapterAccepted(client_envelope_id="note-content-c")


def test_notes_adapter_conflicts_content_edit_not_based_on_content_head_after_metadata_head():
    content_v1 = _stored(
        _envelope(
            client_envelope_id="note-content-a",
            routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
            payload_clear={"entity_kind": "note"},
            payload_hash="sha256:content-a",
            entity_version="note-content-v1",
        ),
        sequence=1,
    )
    metadata_v2 = _stored(
        _envelope(
            client_envelope_id="note-tags-b",
            payload_clear={"entity_kind": "note", "tag_ids": ["tag-2"], "status": "active"},
            payload_hash="sha256:tags-b",
            entity_version="note-metadata-v2",
        ),
        sequence=2,
    )
    incoming = _envelope(
        client_envelope_id="note-content-c",
        routing_metadata={"entity_kind": "note", "update_kind": "note_content"},
        payload_clear={"entity_kind": "note"},
        payload_hash="sha256:content-c",
        base_version="note-root",
        entity_version="note-content-v2",
    )

    outcome = NotesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(content_v1, metadata_v2),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "encrypted_content_edit"


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


def test_lineage_version_dependency_requires_matching_entity_identity():
    head = _stored(
        _envelope(
            client_envelope_id="note-head",
            entity_id="note-1",
            stable_key="note:note-1",
            entity_version="note-v2",
        ),
        sequence=7,
    )

    assert not incoming_references_head(
        _envelope(
            client_envelope_id="note-child",
            dependencies=[{"version": "note-v2"}],
        ),
        head,
    )
    assert incoming_references_head(
        _envelope(
            client_envelope_id="note-child",
            dependencies=[{"entity_id": "note-1", "version": "note-v2"}],
        ),
        head,
    )
    assert incoming_references_head(
        _envelope(
            client_envelope_id="note-child",
            dependencies=[{"stable_key": "note:note-1", "entity_version": "note-v2"}],
        ),
        head,
    )


def test_lineage_direct_dependency_identifiers_reference_head_without_entity_identity():
    head = _stored(
        _envelope(
            client_envelope_id="note-head",
            entity_id="note-1",
            stable_key="note:note-1",
            entity_version="note-v2",
        ),
        sequence=7,
    )

    for dependency in (
        {"client_envelope_id": "note-head"},
        {"envelope_id": "note-head"},
        {"base_envelope_id": "note-head"},
        {"server_sequence": 7},
    ):
        assert incoming_references_head(
            _envelope(client_envelope_id="note-child", dependencies=[dependency]),
            head,
        )


def test_lineage_m1_base_cursor_and_object_hash_reference_head():
    head = _stored(
        _envelope(
            client_envelope_id="note-head",
            entity_id="note-1",
            stable_key="note:note-1",
            entity_version="note-v2",
            object_revision=2,
            payload_hash="sha256:note-v2",
        ),
        sequence=7,
    )

    assert incoming_references_head(
        _envelope(
            client_envelope_id="note-child",
            base_server_cursor=7,
            base_object_revision=2,
            base_object_hash="sha256:note-v2",
        ),
        head,
    )


@pytest.mark.parametrize(
    "domain",
    [
        "notes",
        "chat",
        "workspaces",
        "source_cache.entry",
        "media.item",
        "media.keyword",
        "media.keyword_link",
    ],
)
def test_domain_adapters_accept_linear_tombstone_after_upsert(domain: str):
    prior = _stored(
        _envelope(
            client_envelope_id=f"{domain}-update",
            domain=domain,
            payload_clear=_domain_payload(domain),
            payload_hash=f"sha256:{domain}-update",
            entity_version=f"{domain}-v1",
            **_domain_identity_kwargs(domain),
        )
    )
    incoming_payload = {**_domain_payload(domain), "deleted": True}
    incoming = _envelope(
        client_envelope_id=f"{domain}-tombstone",
        domain=domain,
        operation="tombstone",
        payload_clear=incoming_payload,
        payload_hash=f"sha256:{domain}-tombstone",
        base_version=f"{domain}-v1",
        entity_version=f"{domain}-v2",
        **_domain_identity_kwargs(domain),
    )

    outcome = _adapter_for_domain(domain).evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id=f"{domain}-tombstone")


@pytest.mark.parametrize(
    "domain",
    [
        "notes",
        "chat",
        "workspaces",
        "source_cache.entry",
        "media.item",
        "media.keyword",
        "media.keyword_link",
    ],
)
def test_domain_adapters_conflict_stale_tombstone_after_upsert(domain: str):
    prior = _stored(
        _envelope(
            client_envelope_id=f"{domain}-update",
            domain=domain,
            payload_clear=_domain_payload(domain),
            payload_hash=f"sha256:{domain}-update",
            entity_version=f"{domain}-v2",
            **_domain_identity_kwargs(domain),
        )
    )
    incoming_payload = {**_domain_payload(domain), "deleted": True}
    incoming = _envelope(
        client_envelope_id=f"{domain}-tombstone",
        domain=domain,
        operation="tombstone",
        payload_clear=incoming_payload,
        payload_hash=f"sha256:{domain}-tombstone",
        base_version=f"{domain}-v1",
        entity_version=f"{domain}-v3",
        **_domain_identity_kwargs(domain),
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
            domain="workspaces.source_ref",
            entity_id="workspace-1:source-1",
            stable_key="workspace_source:workspace-1:source-1",
            operation="upsert",
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
        domain="workspaces.source_ref",
        entity_id="workspace-1:source-2",
        stable_key="workspace_source:workspace-1:source-2",
        operation="upsert",
        routing_metadata={"entity_kind": "workspace_source_ref", "workspace_id": "workspace-1"},
        payload_clear={
            "entity_kind": "workspace_source_ref",
            "workspace_id": "workspace-1",
            "source_id": "source-2",
        },
        payload_hash="sha256:source-2",
    )

    outcome = WorkspacesDomainAdapter(domain="workspaces.source_ref").evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert outcome == AdapterAccepted(client_envelope_id="workspace-source-b")


def test_workspaces_adapter_does_not_treat_resource_link_type_as_source_ref():
    incoming = _envelope(
        client_envelope_id="workspace-resource-link",
        domain="workspaces",
        entity_id="workspace-1:resource-1",
        stable_key="workspace_resource:workspace-1:resource-1",
        operation="link",
        routing_metadata={
            "entity_kind": "workspace_resource",
            "workspace_id": "workspace-1",
            "link_type": "resource",
        },
        payload_clear={
            "entity_kind": "workspace_resource",
            "workspace_id": "workspace-1",
            "link_type": "resource",
        },
        payload_hash="sha256:resource-1",
    )

    outcome = WorkspacesDomainAdapter().evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(),
    )

    assert outcome == AdapterAccepted(client_envelope_id="workspace-resource-link")


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
            domain="source_cache.entry",
            entity_id="source-1:content-a",
            stable_key="source_cache.entry:source-1:content-a",
            routing_metadata={"entity_kind": "source_cache_entry"},
            payload_clear={
                "entity_kind": "source_cache_entry",
                "source_id": "source-1",
                "content_hash": "sha256:content-a",
                "provenance": {"kind": "url", "uri": "https://example.test/a"},
            },
            payload_hash="sha256:cache-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="cache-b",
        domain="source_cache.entry",
        entity_id="source-1:content-b",
        stable_key="source_cache.entry:source-1:content-b",
        routing_metadata={"entity_kind": "source_cache_entry"},
        payload_clear={
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "content_hash": "sha256:content-b",
            "provenance": {"kind": "url", "uri": "https://example.test/b"},
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
            domain="source_cache.entry",
            entity_id="source-1:content-a",
            stable_key="source_cache.entry:source-1:content-a",
            routing_metadata={"entity_kind": "source_cache_entry"},
            payload_clear={
                "entity_kind": "source_cache_entry",
                "source_id": "source-1",
                "content_hash": "sha256:content-a",
                "provenance": {"kind": "url", "uri": "https://example.test/a"},
            },
            payload_hash="sha256:cache-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="cache-b",
        domain="source_cache.entry",
        entity_id="source-1:content-a",
        stable_key="source_cache.entry:source-1:content-a",
        routing_metadata={"entity_kind": "source_cache_entry"},
        payload_clear={
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "content_hash": "sha256:content-a",
            "provenance": {"kind": "url", "uri": "https://example.test/a"},
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


def test_source_cache_adapter_rejects_missing_provenance_metadata():
    outcome = SourceCacheAdapter().evaluate_envelope(
        _envelope(
            client_envelope_id="cache-missing-provenance",
            domain="source_cache.entry",
            entity_id="source-1:content-a",
            stable_key="source_cache.entry:source-1:content-a",
            routing_metadata={"entity_kind": "source_cache_entry"},
            payload_clear={
                "entity_kind": "source_cache_entry",
                "source_id": "source-1",
                "content_hash": "sha256:content-a",
            },
            payload_hash="sha256:cache-a",
        ),
        dataset=_dataset(),
        context=_context(),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "missing_source_cache_provenance"


def test_media_metadata_adapter_conflicts_divergent_stable_media_payload():
    prior = _stored(
        _envelope(
            client_envelope_id="media-a",
            domain="media.item",
            entity_id="media-1",
            stable_key="media.item:media-1",
            payload_clear={"media_id": "media-1", "media_type": "video", "title": "Lecture"},
            payload_hash="sha256:media-a",
        )
    )
    incoming = _envelope(
        client_envelope_id="media-b",
        domain="media.item",
        entity_id="media-1",
        stable_key="media.item:media-1",
        payload_clear={"media_id": "media-1", "media_type": "video", "title": "Other"},
        payload_hash="sha256:media-b",
    )

    outcome = MediaMetadataAdapter(domain="media.item").evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(prior),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "media_metadata_hash_mismatch"


@pytest.mark.parametrize(
    ("domain", "payload", "error_code"),
    [
        ("media.item", {"media_type": "video"}, "missing_media_item_metadata"),
        ("media.keyword", {"name": "research"}, "missing_media_keyword_metadata"),
        ("media.keyword_link", {"media_id": "media-1"}, "missing_media_keyword_link_metadata"),
    ],
)
def test_media_metadata_adapter_rejects_missing_stable_identity(
    domain: str,
    payload: dict[str, object],
    error_code: str,
):
    outcome = MediaMetadataAdapter(domain=domain).evaluate_envelope(
        _envelope(
            client_envelope_id=f"{domain}-missing-identity",
            domain=domain,
            entity_id="object-1",
            stable_key=f"{domain}:object-1",
            payload_clear=payload,
            payload_hash=f"sha256:{domain}",
        ),
        dataset=_dataset(),
        context=_context(),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == error_code


@pytest.mark.parametrize("domain", ["media.item", "media.keyword", "media.keyword_link"])
def test_media_metadata_adapter_rejects_raw_blob_payloads(domain: str):
    payload = {**_domain_payload(domain), "blob_ciphertext": "ciphertext:raw-media"}

    outcome = MediaMetadataAdapter(domain=domain).evaluate_envelope(
        _envelope(
            client_envelope_id=f"{domain}-raw-blob",
            domain=domain,
            payload_clear=payload,
            payload_hash=f"sha256:{domain}",
            **_domain_identity_kwargs(domain),
        ),
        dataset=_dataset(),
        context=_context(),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "media_metadata_payload_not_metadata_only"


def test_service_persists_domain_adapter_conflicts(tmp_path: Path):
    registry = SyncAdapterRegistry([ChatDomainAdapter(domain="chat.message")])
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_domain_adapters.db")),
        adapters=registry,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=_ready_sync_settings(),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["chat.message"],
    )

    accepted = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="message-a",
                domain="chat.message",
                operation="append",
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
                domain="chat.message",
                operation="append",
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


def test_default_sync_v2_registry_advertises_personal_and_workspace_metadata_domains():
    registry = sync_endpoint._default_sync_v2_registry()

    assert registry.supported_domains == sorted(SYNC_V2_SUPPORTED_DOMAINS)
    for domain in M1_SYNC_DOMAINS:
        if domain == "attachment.ref":
            assert isinstance(registry.get(domain), AttachmentRefAdapter)
        else:
            assert isinstance(registry.get(domain), StaticSyncAdapter)
    for domain in WORKSPACE_SYNC_DOMAINS:
        assert isinstance(registry.get(domain), WorkspacesDomainAdapter)
    assert isinstance(registry.get("source_cache.entry"), SourceCacheAdapter)
    for domain in ("media.item", "media.keyword", "media.keyword_link"):
        assert isinstance(registry.get(domain), MediaMetadataAdapter)
    with pytest.raises(KeyError):
        registry.get("source_cache")
    with pytest.raises(KeyError):
        registry.get("media")
