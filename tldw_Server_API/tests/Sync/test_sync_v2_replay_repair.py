from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
    NotesOrganizationDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import MaterializationResult
from tldw_Server_API.app.core.Sync.v2.materializers.chat import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.mutation_group_validation import (
    mutation_group_plan_hash,
)
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
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(
        db_path=str(tmp_path / "ChaChaNotes.db"),
        client_id="server-user-1",
    )


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"))


@pytest.fixture()
def log_service(sync_store: SyncV2Store) -> SyncV2Service:
    service = _service(sync_store, materializers={})
    _register_and_enroll(service)
    return service


@pytest.fixture()
def repair_service(sync_store: SyncV2Store, chacha_db: CharactersRAGDB) -> SyncV2Service:
    return _service(
        sync_store,
        materializers={
            "notes.note": NotesMaterializer(chacha_db),
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
        },
    )


@pytest.fixture()
def repair_client(repair_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: repair_service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: repair_service
    return TestClient(app)


def _registry() -> SyncAdapterRegistry:
    return SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
        + [
            NotesOrganizationDomainAdapter(domain=domain)
            for domain in NOTES_ORGANIZATION_DOMAINS
        ]
    )


def _service(sync_store: SyncV2Store, *, materializers: dict[str, Any]) -> SyncV2Service:
    return SyncV2Service(
        store=sync_store,
        adapters=_registry(),
        materializers=materializers,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )


def _register_and_enroll(service: SyncV2Service) -> None:
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=list(M1_SYNC_DOMAINS),
    )


def _note_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-note-create",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "object_revision": 1,
        "payload": {"title": "Repair note", "content": "Rebuilt from log."},
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
        "client_envelope_id": "env-conversation-create",
        "domain": "chat.conversation",
        "operation": "upsert",
        "object_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 10,
        "object_revision": 1,
        "payload": {
            "title": "Repair chat",
            "assistant_kind": "persona",
            "assistant_id": "sync-assistant",
        },
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
        "client_envelope_id": "env-message-create",
        "domain": "chat.message",
        "operation": "append",
        "object_id": "message-1",
        "parent_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 20,
        "object_revision": 1,
        "payload": {
            "conversation_id": "conversation-1",
            "sender": "user",
            "content": "Replay this message.",
            "timestamp": "2026-05-23T18:02:00+00:00",
        },
        "payload_hash": "sha256:message-v1",
        "payload_size_bytes": 80,
        "created_at_client": "2026-05-23T18:02:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "chat:message-1",
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


def _enable_ready_notes_organization(store: SyncV2Store) -> None:
    store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ? "
        "WHERE dataset_id = ?",
        (
            json.dumps([*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]),
            json.dumps({"notes_organization_v1": {"state": "ready"}}),
            "dataset-1",
        ),
    )


def _keyword_group(
    *,
    group_id: str,
    count: int = 3,
) -> list[SyncEnvelopeCreate]:
    plan = [
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id=f"env-keyword-{index}",
            domain="notes.keyword",
            operation="upsert",
            object_id=f"{index + 1:08d}-1111-4111-8111-111111111111",
            device_id="server-origin",
            object_revision=1,
            payload={"keyword": f"Synthetic {index}"},
            payload_hash=f"sha256:keyword-{index}",
            payload_size_bytes=32,
            encryption_metadata={"policy": "server_trusted_v1"},
            mutation_group_id=group_id,
            mutation_step=index,
            mutation_step_count=count,
            mutation_plan_hash="0" * 64,
        )
        for index in range(count)
    ]
    plan_hash = mutation_group_plan_hash(plan)
    return [replace(envelope, mutation_plan_hash=plan_hash) for envelope in plan]


class _RecordingGroupMaterializer:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
        assert envelope.mutation_step is not None
        self.steps.append(envelope.mutation_step)
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="applied",
        )
        return MaterializationResult(status="applied")


class _RecordingMaterializer:
    def __init__(self) -> None:
        self.object_ids: list[str] = []

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
        self.object_ids.append(envelope.object_id)
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="applied",
        )
        return MaterializationResult(status="applied")


def _assert_blocked_group(
    result,
    materializer: _RecordingGroupMaterializer,
    *,
    group_id: str,
    failing_step: int,
    error_code: str,
) -> None:
    assert materializer.steps == []
    assert result.failed_count == 1
    assert result.repair_status["status"] == "repair_needed"
    assert result.repair_status["mutation_groups"] == [
        {
            "mutation_group_id": group_id,
            "failing_step": failing_step,
            "error_code": error_code,
            "retry_result": "blocked",
            "state": "failed",
        }
    ]


def test_repair_rebuilds_note_projection_from_accepted_envelopes(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())
    assert chacha_db.get_note_by_id("note-1") is None

    result = repair_service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Repair note"
    assert result.applied_count == 1
    assert result.failed_count == 0
    assert result.domain_results[0].domain == "notes.note"
    assert result.domain_results[0].applied_count == 1


def test_repair_rebuilds_chat_conversation_and_messages(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _conversation_envelope(), _message_envelope())

    result = repair_service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["chat.conversation", "chat.message"],
    )

    conversation = chacha_db.get_conversation_by_id("conversation-1")
    message = chacha_db.get_message_by_id("message-1")
    assert conversation is not None
    assert conversation["title"] == "Repair chat"
    assert message is not None
    assert message["conversation_id"] == "conversation-1"
    assert message["content"] == "Replay this message."
    assert result.applied_count == 2
    assert [item.domain for item in result.domain_results] == ["chat.conversation", "chat.message"]


def test_repair_retries_failed_apply_after_projection_issue_is_fixed(
    sync_store: SyncV2Store,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(
        sync_store,
        materializers={"notes.note": NotesMaterializer(chacha_db)},
    )
    _register_and_enroll(service)
    original_upsert = chacha_db.upsert_note_from_sync
    projection_available = False

    def _maybe_fail_projection(*args: Any, **kwargs: Any):
        if not projection_available:
            raise RuntimeError("projection unavailable")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(chacha_db, "upsert_note_from_sync", _maybe_fail_projection)
    _push(service, _note_envelope())
    before = service.profile_status(user_id="user-1", dataset_id="dataset-1")
    before_domains = {item.domain: item for item in before.domain_status}
    assert before_domains["notes.note"].failed_apply_count == 1
    assert before_domains["notes.note"].repair_status["status"] == "repair_needed"

    projection_available = True
    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        failed_only=True,
    )

    note = chacha_db.get_note_by_id("note-1")
    after = service.profile_status(user_id="user-1", dataset_id="dataset-1")
    after_domains = {item.domain: item for item in after.domain_status}
    assert note is not None
    assert result.applied_count == 1
    assert result.failed_count == 0
    assert after_domains["notes.note"].failed_apply_count == 0
    assert after_domains["notes.note"].repair_status["status"] == "healthy"
    assert after_domains["notes.note"].last_apply_result["status"] == "applied"


def test_repair_preserves_tombstones(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())
    _push(
        log_service,
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

    result = repair_service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    assert chacha_db.get_note_by_id("note-1") is None
    deleted = chacha_db.get_note_by_id("note-1", include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"]) is True
    assert result.applied_count == 2


def test_repair_never_replays_conflict_envelopes_as_accepted_changes(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())
    log_service.store.insert_envelope(
        _note_envelope(
            client_envelope_id="env-conflict",
            object_id="note-conflict",
            client_sequence=2,
            payload={"title": "Conflict note", "content": "Must not project."},
            payload_hash="sha256:conflict",
            status="conflict",
            apply_status="conflict",
        )
    )

    result = repair_service.repair(user_id="user-1", dataset_id="dataset-1")

    assert chacha_db.get_note_by_id("note-1") is not None
    assert chacha_db.get_note_by_id("note-conflict") is None
    assert result.applied_count == 1
    assert result.conflict_count == 0


def test_repair_endpoint_requires_owned_dataset_and_returns_status(
    log_service: SyncV2Service,
    repair_client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())

    repaired = repair_client.post(
        "/api/v1/sync/repair",
        json={"dataset_id": "dataset-1", "domains": ["notes.note"]},
    )
    forbidden = repair_client.post(
        "/api/v1/sync/repair",
        json={"dataset_id": "dataset-other", "domains": ["notes.note"]},
    )

    assert repaired.status_code == 200
    assert repaired.json()["applied_count"] == 1
    assert repaired.json()["domain_results"][0]["domain"] == "notes.note"
    assert chacha_db.get_note_by_id("note-1") is not None
    assert forbidden.status_code == 404
    assert forbidden.json()["detail"]["error_code"] == "sync_resource_not_found"


def test_notes_organization_repair_resumes_failed_group_without_skipping_pending_suffix(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(
        sync_store,
        materializers={"notes.keyword": materializer},
    )
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    stored = sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-group-retry")
    )
    sync_store.mark_envelope_apply_status(stored[0].server_cursor, apply_status="applied")
    sync_store.mark_envelope_apply_status(
        stored[1].server_cursor,
        apply_status="failed",
        apply_error_code="notes_organization_projection_failed",
        apply_error_message="Synthetic label /private/path raw database error",
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    assert materializer.steps == [1, 2]
    assert result.attempted_count == 2
    assert result.applied_count == 2
    assert result.failed_count == 0
    assert result.repair_status["mutation_groups"] == [
        {
            "mutation_group_id": "server-origin-group-retry",
            "failing_step": None,
            "error_code": None,
            "retry_result": "applied",
            "state": "applied",
        }
    ]
    serialized = str(asdict(result))
    assert "Synthetic label" not in serialized
    assert "/private/path" not in serialized
    assert "raw database error" not in serialized


def test_code_quality_i4_repair_limit_is_soft_for_one_complete_group(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(
        sync_store,
        materializers={"notes.keyword": materializer},
    )
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    stored = sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-soft-repair-limit")
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        limit=1,
    )

    assert materializer.steps == [0, 1, 2]
    assert result.scanned_count == 3
    assert result.to_cursor == stored[-1].server_cursor


def test_code_quality_i4_repair_propagates_shared_group_limit_failure(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(
        sync_store,
        materializers={"notes.keyword": materializer},
    )
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-oversized-repair-group")
    )

    def reject_oversized_group(
        _dataset_id: str,
        _mutation_group_id: str,
    ) -> list[Any]:
        raise SyncStoreError("sync_restore_group_limit_exceeded")

    monkeypatch.setattr(sync_store, "list_mutation_group", reject_oversized_group)

    with pytest.raises(SyncStoreError, match="sync_restore_group_limit_exceeded"):
        service.repair(
            user_id="user-1",
            dataset_id="dataset-1",
            domains=["notes.keyword"],
            limit=1,
        )

    assert materializer.steps == []


def test_spec_fix_repair_blocks_per_step_plan_hash_mismatch(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(sync_store, materializers={"notes.keyword": materializer})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    group_id = "server-origin-corrupt-step-hash"
    stored = sync_store.insert_envelopes_atomic(_keyword_group(group_id=group_id))
    sync_store.db.execute(
        "UPDATE sync_envelopes SET mutation_plan_hash = ? WHERE server_sequence = ?",
        ("f" * 64, stored[1].server_cursor),
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    _assert_blocked_group(
        result,
        materializer,
        group_id=group_id,
        failing_step=1,
        error_code="mutation_group_plan_hash_invalid",
    )


def test_spec_fix_repair_blocks_recomputed_group_fingerprint_mismatch(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(sync_store, materializers={"notes.keyword": materializer})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    group_id = "server-origin-corrupt-content"
    stored = sync_store.insert_envelopes_atomic(_keyword_group(group_id=group_id))
    sync_store.db.execute(
        "UPDATE sync_envelopes SET payload_json = ? WHERE server_sequence = ?",
        (json.dumps({"keyword": "Secret /private/path raw value"}), stored[1].server_cursor),
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    _assert_blocked_group(
        result,
        materializer,
        group_id=group_id,
        failing_step=0,
        error_code="mutation_group_fingerprint_invalid",
    )
    serialized = str(asdict(result))
    assert "Secret" not in serialized
    assert "/private/path" not in serialized
    assert "raw value" not in serialized


def test_spec_fix_repair_reports_first_missing_group_index(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(sync_store, materializers={"notes.keyword": materializer})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    group_id = "server-origin-missing-step"
    stored = sync_store.insert_envelopes_atomic(_keyword_group(group_id=group_id))
    sync_store.db.execute(
        "DELETE FROM sync_envelopes WHERE server_sequence = ?",
        (stored[1].server_cursor,),
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    _assert_blocked_group(
        result,
        materializer,
        group_id=group_id,
        failing_step=1,
        error_code="mutation_group_step_missing",
    )


def test_spec_fix_repair_blocks_duplicate_group_index(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(sync_store, materializers={"notes.keyword": materializer})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    group_id = "server-origin-duplicate-step"
    stored = sync_store.insert_envelopes_atomic(_keyword_group(group_id=group_id))
    duplicate = [stored[0], replace(stored[1], mutation_step=0), stored[2]]
    monkeypatch.setattr(sync_store, "list_mutation_group", lambda *_: duplicate)

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    _assert_blocked_group(
        result,
        materializer,
        group_id=group_id,
        failing_step=0,
        error_code="mutation_group_step_duplicate",
    )


def test_spec_fix_repair_resumes_pending_only_group_suffix(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(sync_store, materializers={"notes.keyword": materializer})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    stored = sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-pending-suffix")
    )
    sync_store.mark_envelope_apply_status(stored[0].server_cursor, apply_status="applied")

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    assert materializer.steps == [1, 2]
    assert result.attempted_count == 2
    assert result.applied_count == 2
    assert result.repair_status["status"] == "healthy"
    assert result.repair_status["mutation_groups"][0]["state"] == "applied"


def test_spec_fix_repair_resumes_pending_singleton(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingMaterializer()
    service = _service(sync_store, materializers={"notes.keyword": materializer})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    sync_store.insert_envelope(
        replace(
            _keyword_group(group_id="server-origin-unused", count=1)[0],
            mutation_group_id=None,
            mutation_step=None,
            mutation_step_count=None,
            mutation_plan_hash=None,
        )
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    assert materializer.object_ids == ["00000001-1111-4111-8111-111111111111"]
    assert result.applied_count == 1
    assert result.skipped_count == 0
    assert result.repair_status["status"] == "healthy"


def test_spec_fix_repair_reports_unmaterializable_pending_group_as_repair_needed(
    sync_store: SyncV2Store,
) -> None:
    service = _service(sync_store, materializers={})
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-pending-blocked")
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    assert result.repair_status["status"] == "repair_needed"
    assert result.repair_status["mutation_groups"] == [
        {
            "mutation_group_id": "server-origin-pending-blocked",
            "failing_step": 0,
            "error_code": "sync_materializer_unavailable",
            "retry_result": "blocked",
            "state": "pending",
        }
    ]


def test_notes_organization_repair_reports_blocked_group_without_applying_later_steps(
    sync_store: SyncV2Store,
) -> None:
    materializer = _RecordingGroupMaterializer()
    service = _service(
        sync_store,
        materializers={"notes.keyword": materializer},
    )
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    stored = sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-group-blocked")
    )
    sync_store.mark_envelope_apply_status(stored[0].server_cursor, apply_status="applied")
    sync_store.mark_envelope_apply_status(
        stored[1].server_cursor,
        apply_status="conflict",
        apply_error_code="notes_organization_base_conflict",
        apply_error_message="Secret label /private/path idempotency-key-plaintext",
    )

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
    )

    assert materializer.steps == []
    assert result.attempted_count == 0
    assert result.conflict_count == 1
    assert result.repair_status["mutation_groups"] == [
        {
            "mutation_group_id": "server-origin-group-blocked",
            "failing_step": 1,
            "error_code": "notes_organization_base_conflict",
            "retry_result": "blocked",
            "state": "conflict",
        }
    ]
    serialized = str(asdict(result))
    assert "Secret label" not in serialized
    assert "/private/path" not in serialized
    assert "idempotency-key-plaintext" not in serialized


def test_notes_organization_repair_exact_post_state_finishes_bookkeeping_once(
    sync_store: SyncV2Store,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = NotesOrganizationMaterializer(chacha_db, "notes.keyword")
    service = _service(
        sync_store,
        materializers={"notes.keyword": materializer},
    )
    _register_and_enroll(service)
    _enable_ready_notes_organization(sync_store)
    stored = sync_store.insert_envelopes_atomic(
        _keyword_group(group_id="server-origin-group-bookkeeping", count=1)
    )[0]
    product_writes = 0
    projection = NotesOrganizationSyncStore.apply_resource
    original_mark = sync_store.mark_envelope_apply_status
    fail_once = True

    def _count_product_write(*args: Any, **kwargs: Any):
        nonlocal product_writes
        product_writes += 1
        return projection(*args, **kwargs)

    def _fail_first_applied_status(*args: Any, **kwargs: Any):
        nonlocal fail_once
        if kwargs.get("apply_status") == "applied" and fail_once:
            fail_once = False
            raise RuntimeError("Sync bookkeeping unavailable")
        return original_mark(*args, **kwargs)

    monkeypatch.setattr(
        NotesOrganizationSyncStore,
        "apply_resource",
        _count_product_write,
    )
    monkeypatch.setattr(sync_store, "mark_envelope_apply_status", _fail_first_applied_status)
    first = materializer.apply(stored, store=sync_store)
    assert first.status == "failed"
    assert product_writes == 1
    monkeypatch.setattr(sync_store, "mark_envelope_apply_status", original_mark)

    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.keyword"],
        failed_only=True,
    )

    assert result.applied_count == 1
    assert product_writes == 1
    assert sync_store.list_mutation_group(
        "dataset-1", "server-origin-group-bookkeeping"
    )[0].apply_status == "applied"
