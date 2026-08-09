from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.mutation_group_validation import (
    mutation_group_plan_hash,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.restore import order_restore_envelopes
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-23T18:12:00+00:00"


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _test_user() -> User:
    return User(id="user-1", username="user-1")


@pytest.fixture()
def sync_service(tmp_path: Path) -> SyncV2Service:
    default_sync_v2_registry.cache_clear()
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_restore_preview.db")),
        adapters=default_sync_v2_registry(),
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )
    for user_id, device_id in (
        ("user-1", "device-1"),
        ("user-1", "device-2"),
        ("user-2", "other-device"),
    ):
        service.register_device(
            user_id=user_id,
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
        )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=list(M1_SYNC_DOMAINS),
    )
    return service


@pytest.fixture()
def client(sync_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: sync_service
    return TestClient(app)


def _note_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-note-1",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "object_revision": 1,
        "payload": {"title": "Research note", "content": "Body"},
        "payload_hash": "sha256:note-v1",
        "payload_size_bytes": 64,
        "created_at_client": "2026-05-23T18:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "note:note-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _conversation_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-chat-1",
        "domain": "chat.conversation",
        "operation": "upsert",
        "object_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 10,
        "object_revision": 1,
        "payload": {"title": "Research thread", "character_id": "character-1"},
        "payload_hash": "sha256:conversation-v1",
        "payload_size_bytes": 96,
        "created_at_client": "2026-05-23T18:01:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "chat:conversation-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _message_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-message-1",
        "domain": "chat.message",
        "operation": "append",
        "object_id": "message-1",
        "parent_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 20,
        "payload": {"conversation_id": "conversation-1", "role": "user", "content": "Hello"},
        "payload_hash": "sha256:message-v1",
        "payload_size_bytes": 80,
        "created_at_client": "2026-05-23T18:02:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "chat:message-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _attachment_ref_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-attachment-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": "attachment-1",
        "device_id": "device-1",
        "client_sequence": 30,
        "payload": {
            "attachment_id": "attachment-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "content_type": "image/png",
            "size_bytes": 512,
            "payload_hash": "sha256:attachment-v1",
            "availability": "client_local",
        },
        "payload_hash": "sha256:attachment-v1",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-23T18:03:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "attachment:attachment-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _source_cache_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-source-cache-1",
        "domain": "source_cache.entry",
        "operation": "upsert",
        "object_id": "source-1:sha256-source",
        "device_id": "device-1",
        "client_sequence": 40,
        "object_revision": 1,
        "payload": {
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "content_hash": "sha256:source",
            "provenance": {"kind": "url", "uri": "https://example.test/source"},
        },
        "payload_hash": "sha256:source-cache-entry",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-23T18:04:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "source_cache.entry:source-1:sha256-source",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _media_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-media-1",
        "domain": "media.item",
        "operation": "upsert",
        "object_id": "media-1",
        "device_id": "device-1",
        "client_sequence": 50,
        "object_revision": 1,
        "payload": {"media_id": "media-1", "media_type": "video", "title": "Lecture"},
        "payload_hash": "sha256:media-item-v1",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-23T18:05:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "media.item:media-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _push(service: SyncV2Service, *envelopes: SyncEnvelopeCreate) -> None:
    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=list(envelopes),
    )
    assert result.rejected == []
    assert result.conflicts == []
    assert [item.client_envelope_id for item in result.accepted] == [
        envelope.client_envelope_id for envelope in envelopes
    ]


def _enable_ready_notes_organization(service: SyncV2Service) -> None:
    service.store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ? "
        "WHERE dataset_id = ?",
        (
            json.dumps([*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]),
            json.dumps({"notes_organization_v1": {"state": "ready"}}),
            "dataset-1",
        ),
    )


def _organization_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-organization",
        "domain": "notes.keyword",
        "operation": "upsert",
        "object_id": "11111111-1111-4111-8111-111111111111",
        "device_id": "server-origin",
        "object_revision": 1,
        "payload": {"keyword": "Synthetic keyword"},
        "payload_hash": "sha256:organization",
        "payload_size_bytes": 32,
        "created_at_client": "2026-05-23T18:06:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _organization_group(
    group_id: str,
    *envelopes: SyncEnvelopeCreate,
) -> list[SyncEnvelopeCreate]:
    plan = [
        replace(
            envelope,
            mutation_group_id=group_id,
            mutation_step=index,
            mutation_step_count=len(envelopes),
            mutation_plan_hash="0" * 64,
        )
        for index, envelope in enumerate(envelopes)
    ]
    plan_hash = mutation_group_plan_hash(plan)
    return [replace(envelope, mutation_plan_hash=plan_hash) for envelope in plan]


def _keyword_link_envelope(
    *, note_id: str, keyword_id: str, client_envelope_id: str
) -> SyncEnvelopeCreate:
    return _organization_envelope(
        client_envelope_id=client_envelope_id,
        domain="notes.keyword_link",
        object_id=organization_link_id(
            "notes.keyword_link", ["note", note_id, keyword_id]
        ),
        payload={
            "subject_type": "note",
            "subject_id": note_id,
            "keyword_sync_id": keyword_id,
        },
        payload_hash=f"sha256:{client_envelope_id}",
    )


def _organization_tombstone(
    *, client_envelope_id: str, domain: str, object_id: str, object_revision: int = 2
) -> SyncEnvelopeCreate:
    return _organization_envelope(
        client_envelope_id=client_envelope_id,
        domain=domain,
        operation="tombstone",
        object_id=object_id,
        object_revision=object_revision,
        payload={},
        payload_hash=f"sha256:{client_envelope_id}",
    )


def _ordered_restore_heads(
    service: SyncV2Service, *heads: SyncEnvelope
) -> list[SyncEnvelope]:
    latest = {(head.domain, head.object_id): head for head in heads}
    expanded = service._expand_restore_mutation_groups(
        dataset_id="dataset-1",
        envelopes=list(heads),
        latest_object_envelopes=latest,
        selected_domains=set(),
        selected_object_ids=set(),
    )
    return order_restore_envelopes(expanded)


def test_restore_preview_empty_inventory_returns_safe_applies_ranges_counts_and_key_status(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope(), _conversation_envelope(), _message_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[],
    )

    assert preview.total_counts == {
        "chat.conversation": 1,
        "chat.message": 1,
        "notes.note": 1,
    }
    assert preview.datasets[0].latest_cursors == {
        "chat.conversation": 2,
        "chat.message": 3,
        "notes.note": 1,
    }
    assert [(item.domain, item.object_id, item.action) for item in preview.safe_applies] == [
        ("notes.note", "note-1", "apply"),
        ("chat.conversation", "conversation-1", "apply"),
        ("chat.message", "message-1", "append"),
    ]
    ranges = [(item.domain, item.from_cursor, item.to_cursor, item.envelope_count) for item in preview.envelope_ranges]
    assert ranges == [
        ("notes.note", 1, 1, 1),
        ("chat.conversation", 2, 2, 1),
        ("chat.message", 3, 3, 1),
    ]


def test_restore_preview_paginates_past_scan_limit_without_truncating_domain(
    sync_service: SyncV2Service,
) -> None:
    sync_service.settings = replace(sync_service.settings, restore_manifest_scan_limit=2)
    _push(
        sync_service,
        _note_envelope(
            object_id="note-1",
            client_envelope_id="env-note-1",
            client_sequence=1,
            payload_hash="sha256:note-1",
        ),
        _note_envelope(
            object_id="note-2",
            client_envelope_id="env-note-2",
            client_sequence=2,
            payload_hash="sha256:note-2",
            stable_key="note:note-2",
        ),
        _note_envelope(
            object_id="note-3",
            client_envelope_id="env-note-3",
            client_sequence=3,
            payload_hash="sha256:note-3",
            stable_key="note:note-3",
        ),
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note"],
        local_inventory=[],
    )

    assert [item.object_id for item in preview.safe_applies] == ["note-1", "note-2", "note-3"]
    assert preview.datasets[0].latest_cursors == {"notes.note": 3}
    assert preview.envelope_ranges[0].envelope_count == 3
    assert preview.object_conflicts == []
    assert preview.tombstones == []
    assert preview.encryption["policy"] == "server_trusted_v1"
    assert preview.encryption["ready"] is True
    assert preview.key_status == {"dataset-1": {"key_recovery_available": False}}


def test_restore_preview_matching_local_inventory_is_safe_noop(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[
            {
                "domain": "notes.note",
                "object_id": "note-1",
                "object_revision": 1,
                "object_hash": "sha256:note-v1",
                "deleted": False,
            }
        ],
    )

    assert [(item.domain, item.object_id, item.action) for item in preview.safe_applies] == [
        ("notes.note", "note-1", "noop")
    ]
    assert [
        (item.plan_index, item.action, item.domain, item.object_id, item.operation)
        for item in preview.ordered_actions
    ] == [(0, "noop", "notes.note", "note-1", "upsert")]
    assert preview.object_conflicts == []


def test_restore_preview_reports_whole_object_note_and_conversation_conflicts(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope(), _conversation_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[
            {
                "domain": "notes.note",
                "object_id": "note-1",
                "object_revision": 1,
                "object_hash": "sha256:local-note",
                "deleted": False,
            },
            {
                "domain": "chat.conversation",
                "object_id": "conversation-1",
                "object_revision": 1,
                "object_hash": "sha256:local-conversation",
                "deleted": False,
            },
        ],
    )

    assert [(item.domain, item.object_id, item.conflict_type) for item in preview.object_conflicts] == [
        ("notes.note", "note-1", "whole_object_conflict"),
        ("chat.conversation", "conversation-1", "whole_object_conflict"),
    ]
    assert [item.server_hash for item in preview.object_conflicts] == [
        "sha256:note-v1",
        "sha256:conversation-v1",
    ]
    assert preview.safe_applies == []


def test_restore_preview_surfaces_tombstones_as_delete_actions(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope())
    _push(
        sync_service,
        _note_envelope(
            client_envelope_id="env-note-delete",
            operation="tombstone",
            client_sequence=2,
            object_revision=2,
            payload={"deleted": True},
            payload_hash="sha256:note-deleted",
            base_server_cursor=1,
            base_object_revision=1,
            base_object_hash="sha256:note-v1",
        ),
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[
            {
                "domain": "notes.note",
                "object_id": "note-1",
                "object_revision": 1,
                "object_hash": "sha256:note-v1",
                "deleted": False,
            }
        ],
    )

    assert [(item.domain, item.object_id, item.action, item.server_cursor) for item in preview.tombstones] == [
        ("notes.note", "note-1", "delete", 2)
    ]
    assert preview.safe_applies == []
    assert preview.object_conflicts == []


def test_restore_preview_includes_attachment_refs_and_missing_blob_warning(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _attachment_ref_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[],
    )

    assert [(item.attachment_id, item.parent_domain, item.parent_object_id) for item in preview.attachment_refs] == [
        ("attachment-1", "notes.note", "note-1")
    ]
    assert [item.attachment_id for item in preview.missing_blobs] == ["attachment-1"]
    assert [warning.code for warning in preview.warnings] == [
        "sync_key_recovery_missing",
        "sync_attachment_blob_missing",
    ]


def test_restore_preview_includes_source_cache_entries_and_local_conflicts(
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=[*M1_SYNC_DOMAINS, "source_cache.entry"],
    )
    _push(sync_service, _source_cache_envelope())

    empty_inventory = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["source_cache.entry"],
        local_inventory=[],
    )
    divergent_inventory = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["source_cache.entry"],
        local_inventory=[
            {
                "domain": "source_cache.entry",
                "object_id": "source-1:sha256-source",
                "object_revision": 1,
                "object_hash": "sha256:local-source-cache",
                "deleted": False,
            }
        ],
    )

    assert empty_inventory.total_counts == {"source_cache.entry": 1}
    assert [(item.domain, item.object_id, item.action) for item in empty_inventory.safe_applies] == [
        ("source_cache.entry", "source-1:sha256-source", "apply")
    ]
    assert [
        (item.domain, item.object_id, item.conflict_type)
        for item in divergent_inventory.object_conflicts
    ] == [("source_cache.entry", "source-1:sha256-source", "stable_id_conflict")]


def test_restore_preview_includes_media_metadata_and_local_conflicts(
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=[*M1_SYNC_DOMAINS, "media.item", "media.keyword", "media.keyword_link"],
    )
    _push(sync_service, _media_envelope())

    empty_inventory = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["media.item"],
        local_inventory=[],
    )
    divergent_inventory = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["media.item"],
        local_inventory=[
            {
                "domain": "media.item",
                "object_id": "media-1",
                "object_revision": 1,
                "object_hash": "sha256:local-media-item",
                "deleted": False,
            }
        ],
    )

    assert empty_inventory.total_counts == {"media.item": 1}
    assert [(item.domain, item.object_id, item.action) for item in empty_inventory.safe_applies] == [
        ("media.item", "media-1", "apply")
    ]
    assert [
        (item.domain, item.object_id, item.conflict_type)
        for item in divergent_inventory.object_conflicts
    ] == [("media.item", "media-1", "stable_id_conflict")]


@pytest.mark.parametrize(
    ("domains", "selected_object_ids"),
    [
        (["notes.keyword"], None),
        (list(NOTES_ORGANIZATION_DOMAINS), ["11111111-1111-4111-8111-111111111111"]),
    ],
)
def test_spec_fix_restore_rejects_explicit_filter_that_splits_stored_group(
    sync_service: SyncV2Service,
    domains: list[str],
    selected_object_ids: list[str] | None,
) -> None:
    _enable_ready_notes_organization(sync_service)
    sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-filtered-group",
            _organization_envelope(
                client_envelope_id="env-filter-keyword",
                object_id="11111111-1111-4111-8111-111111111111",
            ),
            _organization_envelope(
                client_envelope_id="env-filter-folder",
                domain="notes.folder",
                object_id="22222222-2222-4222-8222-222222222222",
                payload={"name": "Folder", "parent_sync_id": None},
            ),
        )
    )

    with pytest.raises(SyncStoreError, match="sync_restore_plan_invalid"):
        sync_service.restore_preview(
            user_id="user-1",
            dataset_ids=["dataset-1"],
            domains=domains,
            selected_object_ids=selected_object_ids,
            local_inventory=[],
        )


def test_spec_fix_restore_keeps_historical_group_before_superseding_head(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    keyword_a = "11111111-1111-4111-8111-111111111111"
    keyword_b = "22222222-2222-4222-8222-222222222222"
    stored = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-history-group",
            _organization_envelope(
                client_envelope_id="env-history-a",
                object_id=keyword_a,
                payload={"keyword": "A"},
            ),
            _organization_envelope(
                client_envelope_id="env-history-b",
                object_id=keyword_b,
                payload={"keyword": "B"},
            ),
        )
    )
    superseding = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-history-b-v2",
            object_id=keyword_b,
            object_revision=2,
            payload={"keyword": "B2"},
            payload_hash="sha256:history-b-v2",
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=list(NOTES_ORGANIZATION_DOMAINS),
        local_inventory=[],
    )

    assert [item.server_cursor for item in preview.safe_applies] == [
        stored[0].server_cursor,
        stored[1].server_cursor,
        superseding.server_cursor,
    ]


def test_provider_selection_internal_group_precedes_later_resource_revision(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    note = sync_service.store.insert_envelope(
        _note_envelope(
            client_envelope_id="env-provider-note",
            object_id=note_id,
            payload_hash="sha256:provider-note",
        )
    )
    group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-provider-group",
            _organization_envelope(
                client_envelope_id="env-provider-keyword-v1",
                object_id=keyword_id,
                payload_hash="sha256:provider-keyword-v1",
            ),
            _keyword_link_envelope(
                note_id=note_id,
                keyword_id=keyword_id,
                client_envelope_id="env-provider-link",
            ),
        )
    )
    latest = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-provider-keyword-v2",
            object_id=keyword_id,
            object_revision=2,
            payload={"keyword": "Synthetic keyword v2"},
            payload_hash="sha256:provider-keyword-v2",
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=[],
    )

    assert [item.server_cursor for item in preview.safe_applies] == [
        note.server_cursor,
        group[0].server_cursor,
        group[1].server_cursor,
        latest.server_cursor,
    ]


def test_provider_selection_uses_only_earlier_external_resource(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-earlier-note", object_id=note_id)
    )
    keyword = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-earlier-keyword", object_id=keyword_id
        )
    )
    link = sync_service.store.insert_envelope(
        _keyword_link_envelope(
            note_id=note_id,
            keyword_id=keyword_id,
            client_envelope_id="env-earlier-link",
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=[],
    )

    assert [item.server_cursor for item in preview.safe_applies] == [
        note.server_cursor,
        keyword.server_cursor,
        link.server_cursor,
    ]


def test_provider_selection_uses_earliest_later_resource_when_none_is_earlier(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-later-note", object_id=note_id)
    )
    link = sync_service.store.insert_envelope(
        _keyword_link_envelope(
            note_id=note_id,
            keyword_id=keyword_id,
            client_envelope_id="env-later-link",
        )
    )
    keyword = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-later-keyword", object_id=keyword_id
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=[],
    )

    assert [item.server_cursor for item in preview.safe_applies] == [
        note.server_cursor,
        keyword.server_cursor,
        link.server_cursor,
    ]


def test_provider_selection_prefers_latest_earlier_revision_over_later_revision(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-history-note", object_id=note_id)
    )
    first_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-provider-history-one",
            _organization_envelope(
                client_envelope_id="env-history-keyword-v1",
                object_id=keyword_id,
                payload_hash="sha256:history-keyword-v1",
            ),
            _organization_envelope(
                client_envelope_id="env-history-folder-one",
                domain="notes.folder",
                object_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                payload={"name": "One", "parent_sync_id": None},
                payload_hash="sha256:history-folder-one",
            ),
        )
    )
    second_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-provider-history-two",
            _organization_envelope(
                client_envelope_id="env-history-keyword-v2",
                object_id=keyword_id,
                object_revision=2,
                payload={"keyword": "Synthetic keyword v2"},
                payload_hash="sha256:history-keyword-v2",
            ),
            _organization_envelope(
                client_envelope_id="env-history-folder-two",
                domain="notes.folder",
                object_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                payload={"name": "Two", "parent_sync_id": None},
                payload_hash="sha256:history-folder-two",
            ),
        )
    )
    link = sync_service.store.insert_envelope(
        _keyword_link_envelope(
            note_id=note_id,
            keyword_id=keyword_id,
            client_envelope_id="env-history-link",
        )
    )
    latest = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-history-keyword-v3",
            object_id=keyword_id,
            object_revision=3,
            payload={"keyword": "Synthetic keyword v3"},
            payload_hash="sha256:history-keyword-v3",
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=[],
    )

    assert [item.server_cursor for item in preview.safe_applies] == [
        note.server_cursor,
        first_group[0].server_cursor,
        first_group[1].server_cursor,
        second_group[0].server_cursor,
        second_group[1].server_cursor,
        link.server_cursor,
        latest.server_cursor,
    ]


def test_tombstone_graph_keeps_historical_group_before_later_exact_restore(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    keyword_id = "11111111-1111-4111-8111-111111111111"
    folder_id = "22222222-2222-4222-8222-222222222222"
    tombstone_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-tombstone-history",
            _organization_tombstone(
                client_envelope_id="env-history-keyword-tombstone",
                domain="notes.keyword",
                object_id=keyword_id,
            ),
            _organization_tombstone(
                client_envelope_id="env-history-folder-tombstone",
                domain="notes.folder",
                object_id=folder_id,
            ),
        )
    )
    restored = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-history-keyword-restore",
            object_id=keyword_id,
            object_revision=3,
            base_server_cursor=tombstone_group[0].server_cursor,
            base_object_revision=2,
            base_object_hash=tombstone_group[0].payload_hash,
            routing_metadata={"restore_intent": True},
            payload={"keyword": "Restored synthetic keyword"},
            payload_hash="sha256:history-keyword-restore",
        )
    )

    ordered = _ordered_restore_heads(sync_service, tombstone_group[1], restored)
    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=list(NOTES_ORGANIZATION_DOMAINS),
        local_inventory=[],
    )

    assert [item.server_cursor for item in ordered] == [
        tombstone_group[0].server_cursor,
        tombstone_group[1].server_cursor,
        restored.server_cursor,
    ]
    assert [item.server_cursor for item in preview.tombstones] == [
        tombstone_group[0].server_cursor,
        tombstone_group[1].server_cursor,
    ]
    assert [item.server_cursor for item in preview.safe_applies] == [
        restored.server_cursor
    ]


def test_restore_preview_ordered_actions_simulate_historical_tombstones_before_restore(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    keyword_id = "11111111-1111-4111-8111-111111111111"
    folder_id = "22222222-2222-4222-8222-222222222222"
    tombstone_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-public-plan-history",
            _organization_tombstone(
                client_envelope_id="env-public-keyword-tombstone",
                domain="notes.keyword",
                object_id=keyword_id,
            ),
            _organization_tombstone(
                client_envelope_id="env-public-folder-tombstone",
                domain="notes.folder",
                object_id=folder_id,
            ),
        )
    )
    restored = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-public-keyword-restore",
            object_id=keyword_id,
            object_revision=3,
            base_server_cursor=tombstone_group[0].server_cursor,
            base_object_revision=2,
            base_object_hash=tombstone_group[0].payload_hash,
            routing_metadata={"restore_intent": True},
            payload={"keyword": "Restored synthetic keyword"},
            payload_hash="sha256:public-keyword-restore",
        )
    )
    local_inventory = [
        {
            "domain": "notes.keyword",
            "object_id": keyword_id,
            "object_revision": 3,
            "object_hash": restored.payload_hash,
            "deleted": False,
        }
    ]

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=list(NOTES_ORGANIZATION_DOMAINS),
        local_inventory=local_inventory,
    )
    repeated = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=list(NOTES_ORGANIZATION_DOMAINS),
        local_inventory=local_inventory,
    )

    assert preview.ordered_actions == repeated.ordered_actions
    assert [
        (
            item.plan_index,
            item.action,
            item.domain,
            item.object_id,
            item.operation,
            item.server_cursor,
            item.mutation_group_id,
            item.mutation_step,
            item.mutation_step_count,
        )
        for item in preview.ordered_actions
    ] == [
        (
            0,
            "tombstone",
            "notes.keyword",
            keyword_id,
            "tombstone",
            tombstone_group[0].server_cursor,
            "server-origin-public-plan-history",
            0,
            2,
        ),
        (
            1,
            "tombstone",
            "notes.folder",
            folder_id,
            "tombstone",
            tombstone_group[1].server_cursor,
            "server-origin-public-plan-history",
            1,
            2,
        ),
        (
            2,
            "apply",
            "notes.keyword",
            keyword_id,
            "upsert",
            restored.server_cursor,
            None,
            None,
            None,
        ),
    ]
    assert [(item.object_id, item.action) for item in preview.safe_applies] == [
        (keyword_id, "apply")
    ]
    assert [item.object_id for item in preview.tombstones] == [keyword_id, folder_id]

    simulated_live: dict[tuple[str, str], bool] = {}
    for item in preview.ordered_actions:
        if item.action == "conflict":
            break
        if item.action == "tombstone":
            simulated_live[(item.domain, item.object_id)] = False
        elif item.action == "apply":
            simulated_live[(item.domain, item.object_id)] = True
    assert simulated_live == {
        ("notes.keyword", keyword_id): True,
        ("notes.folder", folder_id): False,
    }

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "domains": list(NOTES_ORGANIZATION_DOMAINS),
            "local_inventory": local_inventory,
        },
    )
    assert response.status_code == 200
    assert [
        (item["plan_index"], item["action"], item["server_cursor"])
        for item in response.json()["ordered_actions"]
    ] == [
        (0, "tombstone", tombstone_group[0].server_cursor),
        (1, "tombstone", tombstone_group[1].server_cursor),
        (2, "apply", restored.server_cursor),
    ]
    assert response.json()["safe_applies"][0]["action"] == "apply"


def test_restore_preview_endpoint_exposes_one_safe_canonical_ordered_plan(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    tombstoned_keyword_id = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    folder_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-public-plan-note", object_id=note_id)
    )
    keyword = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-public-plan-keyword",
            object_id=keyword_id,
        )
    )
    link = sync_service.store.insert_envelope(
        _keyword_link_envelope(
            note_id=note_id,
            keyword_id=keyword_id,
            client_envelope_id="env-public-plan-link",
        )
    )
    tombstone_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-public-plan-latest",
            _organization_tombstone(
                client_envelope_id="env-public-plan-keyword-tombstone",
                domain="notes.keyword",
                object_id=tombstoned_keyword_id,
            ),
            _organization_tombstone(
                client_envelope_id="env-public-plan-folder-tombstone",
                domain="notes.folder",
                object_id=folder_id,
            ),
        )
    )

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "domains": ["notes.note", *NOTES_ORGANIZATION_DOMAINS],
            "local_inventory": [
                {
                    "domain": "notes.note",
                    "object_id": note_id,
                    "object_revision": 1,
                    "object_hash": "sha256:local-divergent-note",
                    "deleted": False,
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["server_cursor"] for item in body["ordered_actions"]] == [
        note.server_cursor,
        keyword.server_cursor,
        link.server_cursor,
        tombstone_group[0].server_cursor,
        tombstone_group[1].server_cursor,
    ]
    assert [item["action"] for item in body["ordered_actions"]] == [
        "conflict",
        "apply",
        "apply",
        "tombstone",
        "tombstone",
    ]
    assert [item["plan_index"] for item in body["ordered_actions"]] == list(range(5))
    assert [
        (
            item.get("mutation_group_id"),
            item.get("mutation_step"),
            item.get("mutation_step_count"),
        )
        for item in body["ordered_actions"][-2:]
    ] == [
        ("server-origin-public-plan-latest", 0, 2),
        ("server-origin-public-plan-latest", 1, 2),
    ]
    assert set(body["ordered_actions"][0]) == {
        "plan_index",
        "action",
        "dataset_id",
        "domain",
        "object_id",
        "operation",
        "server_cursor",
        "mutation_group_id",
        "mutation_step",
        "mutation_step_count",
        "code",
    }
    assert body["ordered_actions"][0]["code"] == "whole_object_conflict"
    assert body["restore_status"] == "blocked_by_conflicts"
    assert len(body["safe_applies"]) == 2
    assert len(body["object_conflicts"]) == 1
    assert len(body["tombstones"]) == 2
    assert "Synthetic keyword" not in str(body["ordered_actions"])


def test_restore_preview_round5_distinguishes_same_object_across_datasets(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-2",
        domains=list(M1_SYNC_DOMAINS),
    )
    first = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-dataset-one-shared")
    )
    second = sync_service.store.insert_envelope(
        _note_envelope(
            dataset_id="dataset-2",
            client_envelope_id="env-dataset-two-shared",
            payload_hash="sha256:dataset-two-shared",
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1", "dataset-2"],
        domains=["notes.note"],
        local_inventory=[],
    )

    assert [
        (
            item.plan_index,
            item.dataset_id,
            item.domain,
            item.object_id,
            item.server_cursor,
        )
        for item in preview.ordered_actions
    ] == [
        (0, "dataset-1", "notes.note", "note-1", first.server_cursor),
        (1, "dataset-2", "notes.note", "note-1", second.server_cursor),
    ]

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1", "dataset-2"],
            "domains": ["notes.note"],
            "local_inventory": [],
        },
    )
    assert response.status_code == 200
    public_actions = response.json()["ordered_actions"]
    assert [
        (item["plan_index"], item["dataset_id"], item["domain"], item["object_id"])
        for item in public_actions
    ] == [
        (0, "dataset-1", "notes.note", "note-1"),
        (1, "dataset-2", "notes.note", "note-1"),
    ]
    assert set(public_actions[0]) == {
        "plan_index",
        "action",
        "dataset_id",
        "domain",
        "object_id",
        "operation",
        "server_cursor",
        "mutation_group_id",
        "mutation_step",
        "mutation_step_count",
        "code",
    }
    assert "Research note" not in str(public_actions)


def test_restore_preview_round5_replays_historical_live_group_before_matching_head(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-round5-note", object_id=note_id)
    )
    historical_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-round5-history",
            _organization_envelope(
                client_envelope_id="env-round5-keyword-v1",
                object_id=keyword_id,
                payload_hash="sha256:round5-keyword-v1",
            ),
            _keyword_link_envelope(
                note_id=note_id,
                keyword_id=keyword_id,
                client_envelope_id="env-round5-keyword-link",
            ),
        )
    )
    latest = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-round5-keyword-v2",
            object_id=keyword_id,
            object_revision=2,
            base_server_cursor=historical_group[0].server_cursor,
            base_object_revision=1,
            base_object_hash=historical_group[0].payload_hash,
            payload={"keyword": "Synthetic keyword v2"},
            payload_hash="sha256:round5-keyword-v2",
        )
    )
    local_inventory = [
        {
            "dataset_id": "dataset-1",
            "domain": "notes.keyword",
            "object_id": keyword_id,
            "object_revision": 2,
            "object_hash": latest.payload_hash,
            "deleted": False,
        }
    ]

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=local_inventory,
    )

    assert [item.server_cursor for item in preview.ordered_actions] == [
        note.server_cursor,
        historical_group[0].server_cursor,
        historical_group[1].server_cursor,
        latest.server_cursor,
    ]
    assert [item.action for item in preview.ordered_actions] == [
        "apply",
        "apply",
        "apply",
        "apply",
    ]
    assert [item.dataset_id for item in preview.ordered_actions] == ["dataset-1"] * 4
    assert [
        (item.mutation_group_id, item.mutation_step, item.mutation_step_count)
        for item in preview.ordered_actions[1:3]
    ] == [
        ("server-origin-round5-history", 0, 2),
        ("server-origin-round5-history", 1, 2),
    ]
    assert preview.object_conflicts == []
    assert [
        (item.server_cursor, item.action)
        for item in preview.safe_applies
        if item.domain == "notes.keyword"
    ] == [
        (historical_group[0].server_cursor, "apply"),
        (latest.server_cursor, "apply"),
    ]

    simulated_cursor: dict[tuple[str, str, str], int | None] = {}
    for item in preview.ordered_actions:
        key = (item.dataset_id, item.domain, item.object_id)
        simulated_cursor[key] = None if item.action == "tombstone" else item.server_cursor
    assert simulated_cursor[("dataset-1", "notes.keyword", keyword_id)] == latest.server_cursor

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "domains": ["notes.note", *NOTES_ORGANIZATION_DOMAINS],
            "local_inventory": local_inventory,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object_conflicts"] == []
    assert [
        (item["dataset_id"], item["action"], item["server_cursor"])
        for item in body["ordered_actions"]
    ] == [
        ("dataset-1", "apply", note.server_cursor),
        ("dataset-1", "apply", historical_group[0].server_cursor),
        ("dataset-1", "apply", historical_group[1].server_cursor),
        ("dataset-1", "apply", latest.server_cursor),
    ]
    assert "Synthetic keyword v2" not in str(body["ordered_actions"])

    divergent = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=[
            {
                "dataset_id": "dataset-1",
                "domain": "notes.keyword",
                "object_id": keyword_id,
                "object_revision": 9,
                "object_hash": "sha256:divergent-local-keyword",
                "deleted": False,
            }
        ],
    )
    assert [
        item.server_cursor
        for item in divergent.object_conflicts
        if item.domain == "notes.keyword"
    ] == [historical_group[0].server_cursor, latest.server_cursor]


def test_tombstone_graph_keeps_unrelated_live_before_latest_tombstone_group(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-unrelated-live")
    )
    tombstone_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-latest-tombstones",
            _organization_tombstone(
                client_envelope_id="env-latest-keyword-tombstone",
                domain="notes.keyword",
                object_id="11111111-1111-4111-8111-111111111111",
            ),
            _organization_tombstone(
                client_envelope_id="env-latest-folder-tombstone",
                domain="notes.folder",
                object_id="22222222-2222-4222-8222-222222222222",
            ),
        )
    )

    ordered = _ordered_restore_heads(sync_service, note, *tombstone_group)

    assert [item.server_cursor for item in ordered] == [
        note.server_cursor,
        tombstone_group[0].server_cursor,
        tombstone_group[1].server_cursor,
    ]


def test_tombstone_graph_keeps_latest_tombstones_after_live_relationship(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    note_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    keyword_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    note = sync_service.store.insert_envelope(
        _note_envelope(client_envelope_id="env-dormant-note", object_id=note_id)
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-dormant-keyword", object_id=keyword_id
        )
    )
    link = sync_service.store.insert_envelope(
        _keyword_link_envelope(
            note_id=note_id,
            keyword_id=keyword_id,
            client_envelope_id="env-dormant-link",
        )
    )
    tombstone_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-dormant-tombstones",
            _organization_tombstone(
                client_envelope_id="env-dormant-keyword-tombstone",
                domain="notes.keyword",
                object_id=keyword_id,
            ),
            _organization_tombstone(
                client_envelope_id="env-dormant-folder-tombstone",
                domain="notes.folder",
                object_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            ),
        )
    )

    ordered = _ordered_restore_heads(sync_service, note, link, *tombstone_group)

    assert [item.server_cursor for item in ordered] == [
        note.server_cursor,
        link.server_cursor,
        tombstone_group[0].server_cursor,
        tombstone_group[1].server_cursor,
    ]


def test_tombstone_graph_preserves_multiple_later_identity_revisions(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    keyword_id = "11111111-1111-4111-8111-111111111111"
    tombstone_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-multiple-history-tombstones",
            _organization_tombstone(
                client_envelope_id="env-multiple-keyword-tombstone",
                domain="notes.keyword",
                object_id=keyword_id,
            ),
            _organization_tombstone(
                client_envelope_id="env-multiple-folder-tombstone",
                domain="notes.folder",
                object_id="22222222-2222-4222-8222-222222222222",
            ),
        )
    )
    restore_group = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-multiple-restore",
            _organization_envelope(
                client_envelope_id="env-multiple-keyword-restore",
                object_id=keyword_id,
                object_revision=3,
                base_server_cursor=tombstone_group[0].server_cursor,
                base_object_revision=2,
                base_object_hash=tombstone_group[0].payload_hash,
                routing_metadata={"restore_intent": True},
                payload={"keyword": "Restored synthetic keyword"},
                payload_hash="sha256:multiple-keyword-restore",
            ),
            _organization_envelope(
                client_envelope_id="env-multiple-folder-live",
                domain="notes.folder",
                object_id="33333333-3333-4333-8333-333333333333",
                payload={"name": "Live", "parent_sync_id": None},
                payload_hash="sha256:multiple-folder-live",
            ),
        )
    )
    latest = sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-multiple-keyword-v4",
            object_id=keyword_id,
            object_revision=4,
            base_server_cursor=restore_group[0].server_cursor,
            base_object_revision=3,
            base_object_hash=restore_group[0].payload_hash,
            payload={"keyword": "Restored synthetic keyword v4"},
            payload_hash="sha256:multiple-keyword-v4",
        )
    )

    ordered = _ordered_restore_heads(
        sync_service,
        tombstone_group[1],
        restore_group[1],
        latest,
    )

    assert [item.server_cursor for item in ordered] == [
        tombstone_group[0].server_cursor,
        tombstone_group[1].server_cursor,
        restore_group[0].server_cursor,
        restore_group[1].server_cursor,
        latest.server_cursor,
    ]


def test_tombstone_graph_still_rejects_genuine_dependency_cycle(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    folder_a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    folder_b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-tombstone-cycle-a",
            domain="notes.folder",
            object_id=folder_a,
            payload={"name": "A", "parent_sync_id": folder_b},
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-tombstone-cycle-b",
            domain="notes.folder",
            object_id=folder_b,
            payload={"name": "B", "parent_sync_id": folder_a},
        )
    )
    sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-cycle-tombstones",
            _organization_tombstone(
                client_envelope_id="env-cycle-keyword-tombstone",
                domain="notes.keyword",
                object_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            ),
            _organization_tombstone(
                client_envelope_id="env-cycle-folder-tombstone",
                domain="notes.folder",
                object_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            ),
        )
    )

    with pytest.raises(SyncStoreError, match="sync_restore_plan_invalid"):
        sync_service.restore_preview(
            user_id="user-1",
            dataset_ids=["dataset-1"],
            domains=list(NOTES_ORGANIZATION_DOMAINS),
            local_inventory=[],
        )


def test_spec_fix_restore_rejects_incomplete_persisted_group(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    stored = sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-incomplete-group",
            _organization_envelope(client_envelope_id="env-incomplete-a"),
            _organization_envelope(
                client_envelope_id="env-incomplete-b",
                object_id="22222222-2222-4222-8222-222222222222",
            ),
        )
    )
    sync_service.store.db.execute(
        "DELETE FROM sync_envelopes WHERE server_sequence = ?",
        (stored[1].server_cursor,),
    )

    with pytest.raises(SyncStoreError, match="sync_restore_plan_invalid"):
        sync_service.restore_preview(
            user_id="user-1",
            dataset_ids=["dataset-1"],
            domains=list(NOTES_ORGANIZATION_DOMAINS),
            local_inventory=[],
        )


def test_spec_fix_restore_rejects_missing_dependency(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-missing-parent",
            domain="notes.folder",
            object_id="11111111-1111-4111-8111-111111111111",
            payload={
                "name": "Child",
                "parent_sync_id": "22222222-2222-4222-8222-222222222222",
            },
        )
    )

    with pytest.raises(SyncStoreError, match="sync_restore_plan_invalid"):
        sync_service.restore_preview(
            user_id="user-1",
            dataset_ids=["dataset-1"],
            domains=list(NOTES_ORGANIZATION_DOMAINS),
            local_inventory=[],
        )


def test_spec_fix_restore_rejects_group_dependency_contradiction(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    parent_id = "11111111-1111-4111-8111-111111111111"
    child_id = "22222222-2222-4222-8222-222222222222"
    sync_service.store.insert_envelopes_atomic(
        _organization_group(
            "server-origin-contradictory-group",
            _organization_envelope(
                client_envelope_id="env-child-first",
                domain="notes.folder",
                object_id=child_id,
                payload={"name": "Child", "parent_sync_id": parent_id},
            ),
            _organization_envelope(
                client_envelope_id="env-parent-second",
                domain="notes.folder",
                object_id=parent_id,
                payload={"name": "Parent", "parent_sync_id": None},
            ),
        )
    )

    with pytest.raises(SyncStoreError, match="sync_restore_plan_invalid"):
        sync_service.restore_preview(
            user_id="user-1",
            dataset_ids=["dataset-1"],
            domains=list(NOTES_ORGANIZATION_DOMAINS),
            local_inventory=[],
        )


def test_spec_fix_restore_rejects_cyclic_dependencies(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    folder_a = "11111111-1111-4111-8111-111111111111"
    folder_b = "22222222-2222-4222-8222-222222222222"
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-cycle-a",
            domain="notes.folder",
            object_id=folder_a,
            payload={"name": "A", "parent_sync_id": folder_b},
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-cycle-b",
            domain="notes.folder",
            object_id=folder_b,
            payload={"name": "B", "parent_sync_id": folder_a},
        )
    )

    with pytest.raises(SyncStoreError, match="sync_restore_plan_invalid"):
        sync_service.restore_preview(
            user_id="user-1",
            dataset_ids=["dataset-1"],
            domains=list(NOTES_ORGANIZATION_DOMAINS),
            local_inventory=[],
        )


def test_notes_organization_restore_preview_counts_and_orders_complete_state(
    sync_service: SyncV2Service,
) -> None:
    _enable_ready_notes_organization(sync_service)
    keyword_id = "11111111-1111-4111-8111-111111111111"
    collection_id = "22222222-2222-4222-8222-222222222222"
    parent_folder_id = "33333333-3333-4333-8333-333333333333"
    child_folder_id = "44444444-4444-4444-8444-444444444444"
    note_id = "55555555-5555-4555-8555-555555555555"
    keyword_link_id = organization_link_id(
        "notes.keyword_link", ["note", note_id, keyword_id]
    )
    collection_link_id = organization_link_id(
        "notes.keyword_collection_link", [collection_id, keyword_id]
    )
    folder_link_id = organization_link_id(
        "notes.folder_link", [note_id, child_folder_id]
    )

    sync_service.store.insert_envelope(
        _note_envelope(
            client_envelope_id="env-organization-note",
            object_id=note_id,
            payload_hash="sha256:organization-note",
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-keyword-link",
            domain="notes.keyword_link",
            object_id=keyword_link_id,
            payload={
                "subject_type": "note",
                "subject_id": note_id,
                "keyword_sync_id": keyword_id,
            },
            payload_hash="sha256:keyword-link",
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-folder-child",
            domain="notes.folder",
            object_id=child_folder_id,
            payload={"name": "Child", "parent_sync_id": parent_folder_id},
            payload_hash="sha256:folder-child",
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-folder-link",
            domain="notes.folder_link",
            object_id=folder_link_id,
            payload={"note_id": note_id, "folder_sync_id": child_folder_id},
            payload_hash="sha256:folder-link",
        )
    )
    group = _organization_group(
        "server-origin-group-restore",
        _organization_envelope(
            client_envelope_id="env-collection",
            domain="notes.keyword_collection",
            object_id=collection_id,
            payload={"name": "Synthetic collection", "parent_sync_id": None},
            payload_hash="sha256:collection",
        ),
        _organization_envelope(
            client_envelope_id="env-collection-link",
            domain="notes.keyword_collection_link",
            object_id=collection_link_id,
            payload={
                "collection_sync_id": collection_id,
                "keyword_sync_id": keyword_id,
            },
            payload_hash="sha256:collection-link",
        ),
    )
    sync_service.store.insert_envelopes_atomic(group)
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-keyword",
            object_id=keyword_id,
            payload_hash="sha256:keyword",
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-folder-parent",
            domain="notes.folder",
            object_id=parent_folder_id,
            payload={"name": "Parent", "parent_sync_id": None},
            payload_hash="sha256:folder-parent",
        )
    )
    sync_service.store.insert_envelope(
        _organization_envelope(
            client_envelope_id="env-keyword-tombstone",
            operation="tombstone",
            object_id=keyword_id,
            object_revision=2,
            payload={},
            payload_hash="sha256:keyword-tombstone",
        )
    )

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
        local_inventory=[],
    )

    assert preview.total_counts == {
        "notes.note": 1,
        "notes.folder": 2,
        "notes.folder_link": 1,
        "notes.keyword": 2,
        "notes.keyword_collection": 1,
        "notes.keyword_collection_link": 1,
        "notes.keyword_link": 1,
    }
    assert {
        detail.domain: (
            detail.safe_apply_count,
            detail.tombstone_count,
        )
        for detail in preview.domain_details
    } == {
        "notes.note": (1, 0),
        "notes.keyword": (0, 1),
        "notes.keyword_link": (1, 0),
        "notes.keyword_collection": (1, 0),
        "notes.keyword_collection_link": (1, 0),
        "notes.folder": (2, 0),
        "notes.folder_link": (1, 0),
    }
    ordered = [item.server_cursor for item in preview.safe_applies]
    collection_cursor = next(
        item.server_cursor
        for item in preview.safe_applies
        if item.object_id == collection_id
    )
    collection_link_cursor = next(
        item.server_cursor
        for item in preview.safe_applies
        if item.object_id == collection_link_id
    )
    assert ordered.index(collection_link_cursor) == ordered.index(collection_cursor) + 1
    assert ordered.index(
        next(item.server_cursor for item in preview.safe_applies if item.object_id == parent_folder_id)
    ) < ordered.index(
        next(item.server_cursor for item in preview.safe_applies if item.object_id == child_folder_id)
    )
    assert ordered.index(
        next(item.server_cursor for item in preview.safe_applies if item.object_id == child_folder_id)
    ) < ordered.index(
        next(item.server_cursor for item in preview.safe_applies if item.object_id == folder_link_id)
    )
    assert [(item.domain, item.object_id) for item in preview.tombstones] == [
        ("notes.keyword", keyword_id)
    ]
    assert any(item.object_id == keyword_link_id for item in preview.safe_applies)
    assert "Synthetic keyword" not in str(preview)
    assert "Synthetic collection" not in str(preview)


def test_restore_preview_endpoint_blocks_requested_cross_user_dataset(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-2",
        dataset_id="dataset-other",
        domains=list(M1_SYNC_DOMAINS),
    )
    result = sync_service.push(
        user_id="user-2",
        dataset_id="dataset-other",
        device_id="other-device",
        envelopes=[
            _note_envelope(
                dataset_id="dataset-other",
                device_id="other-device",
                object_id="other-note",
                payload_hash="sha256:other-user",
            )
        ],
    )
    assert [item.client_envelope_id for item in result.accepted] == ["env-note-1"]

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-other"], "local_inventory": []},
    )
    broad_response = client.post(
        "/api/v1/sync/restore/preview",
        json={"local_inventory": []},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "sync_resource_not_found"
    assert [dataset["dataset_id"] for dataset in broad_response.json()["datasets"]] == ["dataset-1"]
    assert "other-note" not in str(broad_response.json())
