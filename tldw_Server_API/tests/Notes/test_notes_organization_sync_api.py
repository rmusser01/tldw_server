from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterRejected,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
    NotesOrganizationDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    ChatConversationMaterializer,
    NotesMaterializer,
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization_bootstrap import (
    NotesOrganizationBootstrapper,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import capture_server_origin_mutation
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchIdempotencyConflictError,
    SyncServerOriginBatchMaterializationError,
    server_origin_mutation_batch_group_id,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


class _NoopRateLimiter:
    async def check_user_rate_limit(self, user_id: int, endpoint: str, role: str = "user"):
        return True, {}


class _FailingOrganizationMaterializer:
    def __init__(self, domain: str = "notes.keyword") -> None:
        self.domain = domain

    def apply(self, envelope, *, store):
        raise RuntimeError("secret backend value")


class _FailingFolderChildMaterializer:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def apply(self, envelope, *, store):
        if envelope.payload.get("name") == "Child":
            raise RuntimeError("secret child materialization value")
        return self.delegate.apply(envelope, store=store)


class _FailingSecondRelationshipMaterializer:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.applies = 0

    def apply(self, envelope, *, store):
        self.applies += 1
        if self.applies == 2:
            raise RuntimeError("secret second relationship value")
        return self.delegate.apply(envelope, store=store)


def _assert_canonical_group_lineage(envelopes) -> None:
    assert envelopes
    groups: dict[str, list] = {}
    for envelope in envelopes:
        assert envelope.apply_status == "applied"
        assert envelope.mutation_group_id is not None
        groups.setdefault(envelope.mutation_group_id, []).append(envelope)
        if envelope.base_object_revision is None:
            assert envelope.object_revision == 1
            assert envelope.base_object_hash is None
        else:
            assert envelope.object_revision == envelope.base_object_revision + 1
            assert envelope.base_object_hash is not None
            assert envelope.base_server_cursor is not None
    for group in groups.values():
        assert [item.mutation_step for item in group] == list(range(len(group)))
        assert {item.mutation_step_count for item in group} == {len(group)}
        assert len({item.mutation_plan_hash for item in group}) == 1


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(db_path=str(tmp_path / "ChaChaNotes.db"), client_id="user-1")


@pytest.fixture()
def sync_service(tmp_path: Path, chacha_db: CharactersRAGDB) -> SyncV2Service:
    adapters = []
    for domain in M1_SYNC_DOMAINS:
        adapters.append(
            NotesDomainAdapter()
            if domain == "notes.note"
            else StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
        )
    adapters.extend(
        NotesOrganizationDomainAdapter(domain=domain)
        for domain in NOTES_ORGANIZATION_DOMAINS
    )
    materializers = {
        "chat.conversation": ChatConversationMaterializer(chacha_db),
        "notes.note": NotesMaterializer(chacha_db),
        **{
            domain: NotesOrganizationMaterializer(chacha_db, domain)
            for domain in NOTES_ORGANIZATION_DOMAINS
        },
    }
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=SyncAdapterRegistry(adapters),
        materializers=materializers,
        dataset_bootstrapper=NotesOrganizationBootstrapper(chacha_db),
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            )
        ),
        clock=lambda: "2026-08-09T08:00:00+00:00",
        id_factory=lambda prefix: f"{prefix}-test",
    )
    service.bootstrap_profile(
        user_id="user-1",
        mode="server_frontend",
        device_id="frontend-device",
        requested_domains=[*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS],
    )
    return service


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> TestClient:
    app = FastAPI()
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id="user-1", username="user-1", is_admin=True)

    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[notes_endpoint.get_request_user] = _user_override
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
    )
    return TestClient(app)


def test_direct_keyword_create_uses_sync_authority(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    response = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "keyword-create-1"},
        json={"keyword": " Project Alpha "},
    )

    assert response.status_code == 201, response.text
    created = response.json()
    assert created["keyword"] == "Project Alpha"
    assert chacha_db.get_keyword_by_id(created["id"])["sync_id"] == created["sync_id"]
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id, 0, domains=["notes.keyword"], limit=20
    )
    direct = [item for item in envelopes if item.routing_metadata.get("source") == "notes-api"]
    assert len(direct) == 1
    assert direct[0].object_id == created["sync_id"]
    assert direct[0].payload == {"keyword": "Project Alpha"}
    assert uuid.UUID(created["sync_id"]).version == 4
    assert "keyword-create-1" not in repr(direct[0])
    _assert_canonical_group_lineage(direct)


def test_direct_keyword_rename_delete_and_idempotent_replay_use_sync_authority(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    headers = {"Idempotency-Key": "keyword-lifecycle-create"}
    first = client.post(
        "/api/v1/notes/keywords/", headers=headers, json={"keyword": "Alpha"}
    )
    replay = client.post(
        "/api/v1/notes/keywords/", headers=headers, json={"keyword": "Alpha"}
    )
    assert first.status_code == replay.status_code == 201
    assert first.json()["id"] == replay.json()["id"]
    assert first.json()["sync_id"] == replay.json()["sync_id"]

    renamed = client.patch(
        f"/api/v1/notes/keywords/{first.json()['id']}",
        headers={
            "expected-version": str(first.json()["version"]),
            "Idempotency-Key": "keyword-lifecycle-rename",
        },
        json={"keyword": "Beta"},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["keyword"] == "Beta"
    deleted = client.delete(
        f"/api/v1/notes/keywords/{first.json()['id']}",
        headers={
            "expected-version": str(renamed.json()["version"]),
            "Idempotency-Key": "keyword-lifecycle-delete",
        },
    )
    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_keyword_by_id(first.json()["id"]) is None

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=["notes.keyword"], limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [(item.operation, item.payload) for item in direct] == [
        ("upsert", {"keyword": "Alpha"}),
        ("upsert", {"keyword": "Beta"}),
        ("tombstone", {}),
    ]
    _assert_canonical_group_lineage(direct)


def test_direct_folder_path_plans_missing_parents_before_child(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    response = client.post(
        "/api/v1/notes/folders",
        headers={"Idempotency-Key": "folder-project-reference"},
        json={"path": "Projects / Reference"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["path"] == "Projects/Reference"
    assert chacha_db.get_note_folder_by_path("Projects/Reference")["id"] == response.json()["id"]

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=["notes.folder"], limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.payload["name"] for item in direct] == ["Projects", "Reference"]
    assert direct[0].parent_id is None
    assert direct[1].parent_id == direct[0].object_id
    assert {item.mutation_group_id for item in direct} == {direct[0].mutation_group_id}
    _assert_canonical_group_lineage(direct)


def test_direct_collection_resource_and_keyword_links_use_sync_authority(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    keyword = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "collection-keyword"},
        json={"keyword": "Research"},
    ).json()
    created = client.post(
        "/api/v1/notes/collections",
        headers={"Idempotency-Key": "collection-create"},
        json={"name": "Reading"},
    )
    assert created.status_code == 201, created.text
    linked = client.post(
        f"/api/v1/notes/collections/{created.json()['id']}/keywords/{keyword['id']}",
        headers={"Idempotency-Key": "collection-link"},
    )
    assert linked.status_code == 200, linked.text
    assert [row["id"] for row in chacha_db.get_keywords_for_collection(created.json()["id"])] == [
        keyword["id"]
    ]
    unlinked = client.delete(
        f"/api/v1/notes/collections/{created.json()['id']}/keywords/{keyword['id']}",
        headers={"Idempotency-Key": "collection-unlink"},
    )
    assert unlinked.status_code == 200, unlinked.text
    assert chacha_db.get_keywords_for_collection(created.json()["id"]) == []
    updated = client.patch(
        f"/api/v1/notes/collections/{created.json()['id']}",
        headers={
            "expected-version": str(created.json()["version"]),
            "Idempotency-Key": "collection-update",
        },
        json={"name": "Reading List"},
    )
    assert updated.status_code == 200, updated.text
    deleted = client.delete(
        f"/api/v1/notes/collections/{created.json()['id']}",
        headers={
            "expected-version": str(updated.json()["version"]),
            "Idempotency-Key": "collection-delete",
        },
    )
    assert deleted.status_code == 204, deleted.text

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id,
            0,
            domains=["notes.keyword_collection", "notes.keyword_collection_link"],
            limit=30,
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.operation for item in direct] == [
        "upsert",
        "upsert",
        "tombstone",
        "upsert",
        "tombstone",
    ]
    _assert_canonical_group_lineage(direct)


def test_direct_note_and_conversation_keyword_links_use_sync_authority(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    keyword = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "subject-keyword"},
        json={"keyword": "Subject"},
    ).json()
    note_id = "d5deeb2b-7a73-4fdf-921e-cfe0f20bbb9f"
    note = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "subject-note"},
        json={"id": note_id, "title": "Subject note", "content": "Body"},
    )
    assert note.status_code == 201, note.text
    conversation_id = "conversation-subject"
    capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="chat.conversation",
        operation="upsert",
        object_id=conversation_id,
        payload={
            "title": "Subject chat",
            "assistant_kind": "persona",
            "assistant_id": "assistant-1",
            "scope_type": "global",
        },
        source="test-setup",
    )

    note_link = client.post(
        f"/api/v1/notes/{note_id}/keywords/{keyword['id']}",
        headers={"Idempotency-Key": "note-link"},
    )
    conversation_link = client.post(
        f"/api/v1/notes/conversations/{conversation_id}/keywords/{keyword['id']}",
        headers={"Idempotency-Key": "conversation-link"},
    )
    assert note_link.status_code == conversation_link.status_code == 200
    assert [row["id"] for row in chacha_db.get_keywords_for_note(note_id)] == [keyword["id"]]
    assert [row["id"] for row in chacha_db.get_keywords_for_conversation(conversation_id)] == [
        keyword["id"]
    ]
    note_unlink = client.delete(
        f"/api/v1/notes/{note_id}/keywords/{keyword['id']}",
        headers={"Idempotency-Key": "note-unlink"},
    )
    conversation_unlink = client.delete(
        f"/api/v1/notes/conversations/{conversation_id}/keywords/{keyword['id']}",
        headers={"Idempotency-Key": "conversation-unlink"},
    )
    assert note_unlink.status_code == conversation_unlink.status_code == 200
    assert chacha_db.get_keywords_for_note(note_id) == []
    assert chacha_db.get_keywords_for_conversation(conversation_id) == []

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=["notes.keyword_link"], limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [(item.operation, item.payload["subject_type"]) for item in direct] == [
        ("upsert", "note"),
        ("upsert", "conversation"),
        ("tombstone", "note"),
        ("tombstone", "conversation"),
    ]
    _assert_canonical_group_lineage(direct)


def test_direct_collection_with_inline_keywords_is_one_complete_group(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    response = client.post(
        "/api/v1/notes/collections",
        headers={"Idempotency-Key": "inline-collection"},
        json={"name": "Inline", "keywords": ["Alpha", "Beta"]},
    )
    assert response.status_code == 201, response.text
    assert [row["keyword"] for row in response.json()["keywords"]] == ["Alpha", "Beta"]
    assert [row["keyword"] for row in chacha_db.get_keywords_for_collection(response.json()["id"])] == [
        "Alpha",
        "Beta",
    ]

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id,
            0,
            domains=[
                "notes.keyword",
                "notes.keyword_collection",
                "notes.keyword_collection_link",
            ],
            limit=20,
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.domain for item in direct] == [
        "notes.keyword",
        "notes.keyword",
        "notes.keyword_collection",
        "notes.keyword_collection_link",
        "notes.keyword_collection_link",
    ]
    assert len({item.mutation_group_id for item in direct}) == 1
    _assert_canonical_group_lineage(direct)


@pytest.mark.parametrize(
    ("state", "expected_code"),
    [
        ("partial", "notes_organization_sync_domains_incomplete"),
        ("initializing", "notes_organization_sync_not_ready"),
        ("failed", "notes_organization_sync_not_ready"),
    ],
)
def test_direct_active_sync_readiness_failures_do_not_fall_back(
    state: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    if state == "partial":
        replacement = replace(
            dataset,
            domains=[domain for domain in dataset.domains if domain != "notes.folder_link"],
        )
    else:
        metadata = dict(dataset.metadata)
        organization = dict(metadata["notes_organization_v1"])
        organization["state"] = state
        organization["error_code"] = "safe_repair_code" if state == "failed" else None
        metadata["notes_organization_v1"] = organization
        replacement = replace(dataset, metadata=metadata)
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [replacement] if user_id == "user-1" else [],
    )

    response = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": f"readiness-{state}"},
        json={"keyword": "Must not write"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == expected_code
    assert chacha_db.get_keyword_by_text("Must not write") is None


def test_direct_idempotency_drift_is_safe_and_does_not_write_second_row(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    headers = {"Idempotency-Key": "keyword-drift"}
    first = client.post(
        "/api/v1/notes/keywords/", headers=headers, json={"keyword": "First"}
    )
    drift = client.post(
        "/api/v1/notes/keywords/", headers=headers, json={"keyword": "Second"}
    )
    assert first.status_code == 201
    assert drift.status_code == 409
    assert drift.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_idempotency_conflict"
    )
    assert chacha_db.get_keyword_by_text("First") is not None
    assert chacha_db.get_keyword_by_text("Second") is None
    assert "keyword-drift" not in drift.text


def test_direct_preflight_conflict_has_no_product_write_and_safe_error(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    first = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "preflight-first"},
        json={"keyword": "Unique name"},
    )
    conflict = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "preflight-second"},
        json={"keyword": "unique NAME"},
    )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["error_code"] == (
        "notes_organization_sync_preflight_failed"
    )
    assert "case-insensitive" not in conflict.text
    assert chacha_db.count_keywords() == 1


def test_direct_append_failure_has_no_product_write_and_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    def _fail_append(envelopes):
        raise SyncStoreError("secret append database value")

    monkeypatch.setattr(sync_service.store, "insert_envelopes_atomic", _fail_append)
    response = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "append-failure"},
        json={"keyword": "Not written"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "sync_server_origin_batch_append_failed"
    assert "secret" not in response.text
    assert chacha_db.get_keyword_by_text("Not written") is None


def test_direct_materialization_failure_has_no_fallback_write_and_safe_error(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    sync_service.materializers["notes.keyword"] = _FailingOrganizationMaterializer()
    response = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "materialization-failure"},
        json={"keyword": "Not projected"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_materialization_failed"
    )
    assert "secret backend value" not in response.text
    assert chacha_db.get_keyword_by_text("Not projected") is None


def test_direct_mid_group_materialization_failure_is_durable_blocked_and_resumable(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    original = sync_service.materializers["notes.keyword_collection"]
    sync_service.materializers["notes.keyword_collection"] = (
        _FailingOrganizationMaterializer("notes.keyword_collection")
    )
    headers = {"Idempotency-Key": "mid-group-resume"}
    body = {"name": "Blocked collection", "keywords": ["Durable keyword"]}

    failed = client.post("/api/v1/notes/collections", headers=headers, json=body)

    assert failed.status_code == 503
    assert failed.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_materialization_failed"
    )
    assert chacha_db.get_keyword_by_text("Durable keyword") is not None
    assert chacha_db.list_keyword_collections(limit=20, offset=0) == []
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    group = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=list(NOTES_ORGANIZATION_DOMAINS), limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.apply_status for item in group] == ["applied", "failed", "pending"]

    sync_service.materializers["notes.keyword_collection"] = original
    resumed = client.post("/api/v1/notes/collections", headers=headers, json=body)

    assert resumed.status_code == 201, resumed.text
    assert resumed.json()["name"] == "Blocked collection"
    assert [row["keyword"] for row in resumed.json()["keywords"]] == [
        "Durable keyword"
    ]
    resumed_group = sync_service.store.list_mutation_group(
        dataset_id, group[0].mutation_group_id or ""
    )
    _assert_canonical_group_lineage(resumed_group)


def test_direct_folder_retry_reuses_full_manifest_after_applied_ancestor(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    original = sync_service.materializers["notes.folder"]
    sync_service.materializers["notes.folder"] = _FailingFolderChildMaterializer(
        original
    )
    headers = {"Idempotency-Key": "folder-prefix-resume"}
    body = {"path": "Root/Child"}

    failed = client.post("/api/v1/notes/folders", headers=headers, json=body)

    assert failed.status_code == 503
    assert failed.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_materialization_failed"
    )
    assert chacha_db.get_note_folder_by_path("Root") is not None
    assert chacha_db.get_note_folder_by_path("Root/Child") is None
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    group = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=["notes.folder"], limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.payload["name"] for item in group] == ["Root", "Child"]
    assert [item.apply_status for item in group] == ["applied", "failed"]

    original_ids = [item.object_id for item in group]
    sync_service.materializers["notes.folder"] = original
    resumed = client.post("/api/v1/notes/folders", headers=headers, json=body)

    assert resumed.status_code == 201, resumed.text
    assert resumed.json()["path"] == "Root/Child"
    created_replay = client.post(
        "/api/v1/notes/folders", headers=headers, json=body
    )
    assert created_replay.status_code == 201, created_replay.text
    assert created_replay.json() == resumed.json()
    resumed_group = sync_service.store.list_mutation_group(
        dataset_id, group[0].mutation_group_id or ""
    )
    assert [item.object_id for item in resumed_group] == original_ids
    _assert_canonical_group_lineage(resumed_group)

    existing_headers = {"Idempotency-Key": "folder-existing-request"}
    existing = client.post(
        "/api/v1/notes/folders", headers=existing_headers, json=body
    )
    existing_replay = client.post(
        "/api/v1/notes/folders", headers=existing_headers, json=body
    )
    drift = client.post(
        "/api/v1/notes/folders",
        headers=existing_headers,
        json={"path": "Root/Other"},
    )
    assert existing.status_code == 200, existing.text
    assert existing_replay.status_code == 200, existing_replay.text
    assert existing_replay.json() == existing.json()
    assert drift.status_code == 409
    assert drift.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_idempotency_conflict"
    )

    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=["notes.folder"], limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert {
        item.routing_metadata["notes_organization_response_status"]
        for item in direct
    } == {200, 201}
    assert all(
        "Root/Child" not in repr(item.routing_metadata)
        and "folder-prefix-resume" not in repr(item.routing_metadata)
        and "folder-existing-request" not in repr(item.routing_metadata)
        for item in direct
    )


def test_direct_collection_retry_reuses_manifest_after_applied_link(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    collection = client.post(
        "/api/v1/notes/collections",
        headers={"Idempotency-Key": "collection-link-base"},
        json={"name": "Linked collection"},
    ).json()
    domain = "notes.keyword_collection_link"
    original = sync_service.materializers[domain]
    sync_service.materializers[domain] = _FailingSecondRelationshipMaterializer(
        original
    )
    headers = {"Idempotency-Key": "collection-link-resume"}
    headers["expected-version"] = str(collection["version"])
    body = {"keywords": ["Alpha", "Beta"]}

    failed = client.patch(
        f"/api/v1/notes/collections/{collection['id']}",
        headers=headers,
        json=body,
    )

    assert failed.status_code == 503
    assert failed.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_materialization_failed"
    )
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id, 0, domains=list(NOTES_ORGANIZATION_DOMAINS), limit=20
    )
    failed_envelope = next(item for item in envelopes if item.apply_status == "failed")
    group = sync_service.store.list_mutation_group(
        dataset_id, failed_envelope.mutation_group_id or ""
    )
    assert [item.apply_status for item in group][-2:] == ["applied", "failed"]
    original_shape = [
        (item.domain, item.operation, item.object_id) for item in group
    ]

    sync_service.materializers[domain] = original
    resumed = client.patch(
        f"/api/v1/notes/collections/{collection['id']}",
        headers=headers,
        json=body,
    )

    assert resumed.status_code == 200, resumed.text
    resumed_group = sync_service.store.list_mutation_group(
        dataset_id, group[0].mutation_group_id or ""
    )
    assert [
        (item.domain, item.operation, item.object_id) for item in resumed_group
    ] == original_shape
    _assert_canonical_group_lineage(resumed_group)


def test_direct_active_sync_keyword_merge_uses_sync_authority(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    source = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "merge-source"},
        json={"keyword": "Source"},
    ).json()
    target = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "merge-target"},
        json={"keyword": "Target"},
    ).json()

    response = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers={
            "expected-version": str(source["version"]),
            "Idempotency-Key": "merge-empty",
        },
        json={
            "target_keyword_id": target["id"],
            "expected_target_version": target["version"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["merged_note_links"] == 0
    assert response.json()["merged_conversation_links"] == 0
    assert response.json()["merged_collection_links"] == 0
    assert chacha_db.get_keyword_by_id(source["id"]) is None
    assert chacha_db.get_keyword_by_id(target["id"]) is not None
    group = _notes_api_group(sync_service, "merge-empty")
    assert [(item.domain, item.operation) for item in group] == [
        ("notes.keyword", "tombstone")
    ]


def test_direct_inactive_sync_preserves_local_keyword_collection_and_folder_writes(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: None,
    )

    keyword = client.post("/api/v1/notes/keywords/", json={"keyword": "Local"})
    collection = client.post(
        "/api/v1/notes/collections",
        json={"name": "Local collection", "keywords": ["Local"]},
    )
    folder = client.post("/api/v1/notes/folders", json={"path": "Local/Folder"})

    assert keyword.status_code == 201, keyword.text
    assert collection.status_code == 201, collection.text
    assert folder.status_code == 201, folder.text
    assert chacha_db.get_keyword_by_text("Local") is not None
    assert chacha_db.get_keyword_collection_by_id(collection.json()["id"]) is not None
    assert chacha_db.get_note_folder_by_path("Local/Folder") is not None
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=list(NOTES_ORGANIZATION_DOMAINS), limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert direct == []


def test_direct_coordinator_covers_non_exposed_folder_and_folder_link_planners(
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    coordinator = NotesOrganizationCoordinator(
        service=sync_service,
        note_db=chacha_db,
        user_id="user-1",
    )
    folder_plan = coordinator.plan_folder_path(
        "Root/Child", idempotency_key="coordinator-folders"
    )
    assert chacha_db.get_note_folder_by_path("Root") is None
    assert chacha_db.get_note_folder_by_path("Root/Child") is None
    coordinator.capture(
        steps=folder_plan.steps,
        source="coordinator-test",
        idempotency_key="coordinator-folders",
    )
    folder = folder_plan.load_result()
    assert isinstance(folder, dict)
    root = chacha_db.get_note_folder_by_path("Root")
    assert root is not None

    changed = coordinator.plan_folder_change(
        folder["id"],
        name="Moved",
        parent_id=None,
    )
    coordinator.capture(
        steps=changed.steps,
        source="coordinator-test",
        idempotency_key="coordinator-folder-move",
    )
    moved = changed.load_result()
    assert isinstance(moved, dict)
    assert moved["path"] == "Moved"

    note_id = "2510686b-7e97-48a1-aa90-419b57a7a37c"
    capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="notes.note",
        operation="upsert",
        object_id=note_id,
        payload={"title": "Folder link", "content": "Body"},
        source="coordinator-test",
    )
    link = coordinator.plan_relationship(
        "notes.folder_link",
        {"note_id": note_id, "folder_sync_id": moved["sync_id"]},
        True,
    )
    coordinator.capture(
        steps=link.steps,
        source="coordinator-test",
        idempotency_key="coordinator-folder-link",
    )
    assert link.load_result() is True
    unlink = coordinator.plan_relationship(
        "notes.folder_link",
        {"note_id": note_id, "folder_sync_id": moved["sync_id"]},
        False,
    )
    coordinator.capture(
        steps=unlink.steps,
        source="coordinator-test",
        idempotency_key="coordinator-folder-unlink",
    )
    assert unlink.load_result() is False

    deleted = coordinator.plan_resource_delete("notes.folder", moved["id"])
    coordinator.capture(
        steps=deleted.steps,
        source="coordinator-test",
        idempotency_key="coordinator-folder-delete",
    )
    assert chacha_db.get_note_folder_by_path("Moved") is None


def test_direct_relationship_relink_restores_current_tombstones(
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    coordinator = NotesOrganizationCoordinator(
        service=sync_service,
        note_db=chacha_db,
        user_id="user-1",
    )
    note_id = "69721de7-7165-4088-b57a-0cb1a8c69a8d"
    conversation_id = "relationship-restore-conversation"
    keyword_id = "2d89ab39-23db-43d5-9f5e-a6be38edb474"
    collection_id = "1aa34562-962b-4c3c-b14c-d5cecc44fbd9"
    folder_id = "87327cd9-1a35-4270-a60c-d32c7dc9f568"
    setup = [
        ("notes.note", note_id, {"title": "Restore note", "content": "Body"}),
        (
            "chat.conversation",
            conversation_id,
            {
                "title": "Restore conversation",
                "assistant_kind": "persona",
                "assistant_id": "assistant-1",
                "scope_type": "global",
            },
        ),
        ("notes.keyword", keyword_id, {"keyword": "Restore keyword"}),
        (
            "notes.keyword_collection",
            collection_id,
            {"name": "Restore collection", "parent_sync_id": None},
        ),
        (
            "notes.folder",
            folder_id,
            {"name": "Restore folder", "parent_sync_id": None},
        ),
    ]
    for domain, object_id, payload in setup:
        capture_server_origin_mutation(
            sync_service,
            user_id="user-1",
            domain=domain,
            operation="upsert",
            object_id=object_id,
            payload=payload,
            source="relationship-restore-setup",
        )
    relationships = [
        (
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": note_id,
                "keyword_sync_id": keyword_id,
            },
        ),
        (
            "notes.keyword_link",
            {
                "subject_type": "conversation",
                "subject_id": conversation_id,
                "keyword_sync_id": keyword_id,
            },
        ),
        (
            "notes.keyword_collection_link",
            {
                "collection_sync_id": collection_id,
                "keyword_sync_id": keyword_id,
            },
        ),
        (
            "notes.folder_link",
            {"note_id": note_id, "folder_sync_id": folder_id},
        ),
    ]

    for index, (domain, members) in enumerate(relationships):
        linked = coordinator.plan_relationship(domain, members, True)
        coordinator.capture(
            steps=linked.steps,
            source="relationship-restore-test",
            idempotency_key=f"relationship-link-{index}",
        )
        unlinked = coordinator.plan_relationship(domain, members, False)
        coordinator.capture(
            steps=unlinked.steps,
            source="relationship-restore-test",
            idempotency_key=f"relationship-unlink-{index}",
        )

        relink_key = f"relationship-relink-{index}"
        relinked = coordinator.plan_relationship(
            domain,
            members,
            True,
            source="relationship-restore-test",
            idempotency_key=relink_key,
        )

        assert relinked.steps[0].routing_metadata["restore_intent"] is True
        first_result = coordinator.capture(
            steps=relinked.steps,
            source="relationship-restore-test",
            idempotency_key=relink_key,
        )
        assert relinked.load_result() is True

        replayed = coordinator.plan_relationship(
            domain,
            members,
            True,
            source="relationship-restore-test",
            idempotency_key=relink_key,
        )
        assert [
            (step.domain, step.operation, step.object_id, step.payload)
            for step in replayed.steps
        ] == [
            (step.domain, step.operation, step.object_id, step.payload)
            for step in relinked.steps
        ]
        assert replayed.steps[0].routing_metadata["restore_intent"] is True
        assert replayed.steps[0].routing_metadata[
            "notes_organization_request_fingerprint"
        ] == relinked.steps[0].routing_metadata[
            "notes_organization_request_fingerprint"
        ]
        replay_result = coordinator.capture(
            steps=replayed.steps,
            source="relationship-restore-test",
            idempotency_key=relink_key,
        )
        assert [item.object_id for item in replay_result.envelopes] == [
            item.object_id for item in first_result.envelopes
        ]
        assert {item.apply_status for item in replay_result.envelopes} == {"applied"}
        assert replayed.load_result() is True

        with pytest.raises(SyncServerOriginBatchIdempotencyConflictError):
            coordinator.plan_relationship(
                domain,
                members,
                False,
                source="relationship-restore-test",
                idempotency_key=relink_key,
            )


def test_direct_update_and_delete_exact_requests_replay_before_mutable_preconditions(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    keyword = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "round1-keyword-create"},
        json={"keyword": "Before replay"},
    ).json()
    rename_headers = {
        "expected-version": str(keyword["version"]),
        "Idempotency-Key": "round1-keyword-rename",
    }
    renamed = client.patch(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers=rename_headers,
        json={"keyword": "After replay"},
    )
    rename_replay = client.patch(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers=rename_headers,
        json={"keyword": "After replay"},
    )
    rename_drift = client.patch(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers=rename_headers,
        json={"keyword": "Changed request"},
    )

    assert renamed.status_code == 200, renamed.text
    assert rename_replay.status_code == 200, rename_replay.text
    assert rename_replay.json() == renamed.json()
    assert rename_drift.status_code == 409
    assert rename_drift.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_idempotency_conflict"
    )

    delete_headers = {
        "expected-version": str(renamed.json()["version"]),
        "Idempotency-Key": "round1-keyword-delete",
    }
    deleted = client.delete(
        f"/api/v1/notes/keywords/{keyword['id']}", headers=delete_headers
    )
    delete_replay = client.delete(
        f"/api/v1/notes/keywords/{keyword['id']}", headers=delete_headers
    )
    delete_drift = client.delete(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers={
            "expected-version": str(renamed.json()["version"] + 1),
            "Idempotency-Key": "round1-keyword-delete",
        },
    )

    assert deleted.status_code == delete_replay.status_code == 204
    assert delete_drift.status_code == 409
    assert delete_drift.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_idempotency_conflict"
    )
    assert chacha_db.get_keyword_by_id(keyword["id"]) is None

    parent = client.post(
        "/api/v1/notes/collections",
        headers={"Idempotency-Key": "round1-parent-create"},
        json={"name": "Replay parent"},
    ).json()
    child = client.post(
        "/api/v1/notes/collections",
        headers={"Idempotency-Key": "round1-child-create"},
        json={"name": "Replay child"},
    ).json()
    collection_headers = {
        "expected-version": str(child["version"]),
        "Idempotency-Key": "round1-collection-update",
    }
    collection_body = {"name": "Replay child moved", "parent_id": parent["id"]}
    updated = client.patch(
        f"/api/v1/notes/collections/{child['id']}",
        headers=collection_headers,
        json=collection_body,
    )
    update_replay = client.patch(
        f"/api/v1/notes/collections/{child['id']}",
        headers=collection_headers,
        json=collection_body,
    )
    update_drift = client.patch(
        f"/api/v1/notes/collections/{child['id']}",
        headers=collection_headers,
        json={"name": "Different child", "parent_id": parent["id"]},
    )

    assert updated.status_code == 200, updated.text
    assert update_replay.status_code == 200, update_replay.text
    assert update_replay.json() == updated.json()
    assert update_drift.status_code == 409
    assert update_drift.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_idempotency_conflict"
    )

    collection_delete_headers = {
        "expected-version": str(updated.json()["version"]),
        "Idempotency-Key": "round1-collection-delete",
    }
    collection_deleted = client.delete(
        f"/api/v1/notes/collections/{child['id']}",
        headers=collection_delete_headers,
    )
    collection_delete_replay = client.delete(
        f"/api/v1/notes/collections/{child['id']}",
        headers=collection_delete_headers,
    )

    assert collection_deleted.status_code == collection_delete_replay.status_code == 204
    assert chacha_db.get_keyword_collection_by_id(child["id"]) is None


def test_direct_missing_deleted_and_stale_keywords_keep_stable_route_statuses(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    missing_rename = client.patch(
        "/api/v1/notes/keywords/999999",
        headers={"expected-version": "1"},
        json={"keyword": "Missing"},
    )
    missing_delete = client.delete(
        "/api/v1/notes/keywords/999999",
        headers={"expected-version": "1"},
    )

    assert missing_rename.status_code == missing_delete.status_code == 404
    assert missing_rename.json()["detail"]["error_code"] == (
        "notes_organization_resource_not_found"
    )
    assert missing_delete.json()["detail"]["error_code"] == (
        "notes_organization_resource_not_found"
    )

    keyword = client.post(
        "/api/v1/notes/keywords/",
        headers={"Idempotency-Key": "status-keyword-create"},
        json={"keyword": "Status keyword"},
    ).json()
    stale = client.patch(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers={"expected-version": str(keyword["version"] + 1)},
        json={"keyword": "Stale rename"},
    )
    deleted = client.delete(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers={"expected-version": str(keyword["version"])},
    )
    deleted_again = client.delete(
        f"/api/v1/notes/keywords/{keyword['id']}",
        headers={"expected-version": str(keyword["version"])},
    )

    assert stale.status_code == 409
    assert stale.json()["detail"]["error_code"] == (
        "notes_organization_version_conflict"
    )
    assert deleted.status_code == 204
    assert deleted_again.status_code == 404
    assert deleted_again.json()["detail"]["error_code"] == (
        "notes_organization_resource_not_found"
    )

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, 0, domains=["notes.keyword"], limit=20
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [(item.operation, item.payload) for item in direct] == [
        ("upsert", {"keyword": "Status keyword"}),
        ("tombstone", {}),
    ]


def _notes_api_group(sync_service: SyncV2Service, idempotency_key: str):
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    group_id = server_origin_mutation_batch_group_id(
        dataset_id=dataset_id,
        source="notes-api",
        idempotency_key=idempotency_key,
    )
    return sync_service.store.list_mutation_group(dataset_id, group_id)


def test_inline_note_create_captures_one_complete_ordered_group(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    headers = {"Idempotency-Key": "inline-note-create"}
    response = client.post(
        "/api/v1/notes/",
        headers=headers,
        json={
            "title": "Compound note",
            "content": "Body",
            "keywords": ["Alpha", "Beta"],
            "folder_paths": ["Projects/Research"],
        },
    )

    assert response.status_code == 201, response.text
    note = response.json()
    assert [item["keyword"] for item in note["keywords"]] == ["Alpha", "Beta"]
    assert [item["path"] for item in note["folders"]] == [
        "Projects",
        "Projects/Research",
    ]
    group = _notes_api_group(sync_service, "inline-note-create")
    assert [item.domain for item in group] == [
        "notes.note",
        "notes.keyword",
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_link",
        "notes.folder",
        "notes.folder",
        "notes.folder_link",
        "notes.folder_link",
    ]
    assert [item.payload.get("name") for item in group[5:7]] == [
        "Projects",
        "Research",
    ]
    assert all(item.apply_status == "applied" for item in group)
    assert chacha_db.get_note_by_id(note["id"])["version"] == 1


@pytest.mark.parametrize(("method", "path"), [("put", ""), ("patch", "")])
def test_inline_note_update_and_patch_capture_note_and_relationship_deltas(
    method: str,
    path: str,
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": f"inline-{method}-setup"},
        json={"id": note_id, "title": "Before", "content": "Body"},
    )
    assert created.status_code == 201, created.text
    key = f"inline-note-{method}"
    response = client.request(
        method,
        f"/api/v1/notes/{note_id}{path}",
        headers={
            "expected-version": str(created.json()["version"]),
            "Idempotency-Key": key,
        },
        json={
            "title": f"After {method}",
            "keywords": ["Delta"],
            "folder_paths": ["Work/Active"],
        },
    )

    assert response.status_code == 200, response.text
    group = _notes_api_group(sync_service, key)
    assert [item.domain for item in group] == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
        "notes.folder",
        "notes.folder",
        "notes.folder_link",
        "notes.folder_link",
    ]
    assert response.json()["title"] == f"After {method}"
    assert [item["keyword"] for item in response.json()["keywords"]] == ["Delta"]


@pytest.mark.parametrize("method", ["put", "patch"])
def test_inline_note_update_replays_original_response_after_later_mutation(
    method: str,
    client: TestClient,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": f"{method}-replay-setup"},
        json={"id": note_id, "title": "Before", "content": "Body"},
    )
    assert created.status_code == 201, created.text
    headers = {
        "Expected-Version": str(created.json()["version"]),
        "Idempotency-Key": f"{method}-durable-update",
    }
    payload = {
        "title": "Original update",
        "keywords": ["Original keyword"],
        "folder_paths": ["Original/Path"],
    }
    original = client.request(
        method,
        f"/api/v1/notes/{note_id}",
        headers=headers,
        json=payload,
    )
    assert original.status_code == 200, original.text

    changed = client.put(
        f"/api/v1/notes/{note_id}",
        headers={
            "Expected-Version": str(original.json()["version"]),
            "Idempotency-Key": f"{method}-later-update",
        },
        json={
            "title": "Later update",
            "keywords": ["Later keyword"],
            "folder_paths": ["Later/Path"],
        },
    )
    assert changed.status_code == 200, changed.text

    replay = client.request(
        method,
        f"/api/v1/notes/{note_id}",
        headers=headers,
        json=payload,
    )

    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()


def test_bulk_import_uses_one_group_per_note_and_isolates_invalid_items(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    response = client.post(
        "/api/v1/notes/bulk",
        headers={"Idempotency-Key": "bulk-import-groups"},
        json={
            "notes": [
                {
                    "title": "Valid compound",
                    "content": "Body",
                    "keywords": ["Bulk"],
                    "folder_paths": ["Imports/Valid"],
                },
                {
                    "title": "Invalid isolated",
                    "content": "Body",
                    "conversation_id": "missing-conversation",
                    "keywords": ["Must not exist"],
                },
            ]
        },
    )

    assert response.status_code == 207, response.text
    body = response.json()
    assert body["created_count"] == 1
    assert body["failed_count"] == 1
    assert body["results"][0]["success"] is True
    assert body["results"][1]["success"] is False
    first_group = _notes_api_group(sync_service, "bulk-import-groups:0")
    assert first_group
    assert {item.mutation_group_id for item in first_group} == {
        first_group[0].mutation_group_id
    }
    assert [item.domain for item in first_group][0] == "notes.note"


def test_literal_import_create_uses_one_compound_note_keyword_group(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    note_id = str(uuid.uuid4())
    content = json.dumps(
        [
            {
                "id": note_id,
                "title": "Imported",
                "content": "Body",
                "keywords": ["Research"],
            }
        ]
    )

    response = client.post(
        "/api/v1/notes/import",
        json={
            "duplicate_strategy": "overwrite",
            "items": [
                {
                    "file_name": "import.json",
                    "format": "json",
                    "content": content,
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 1
    assert [row["keyword"] for row in chacha_db.get_keywords_for_note(note_id)] == [
        "Research"
    ]
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(dataset_id, 0)
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.domain for item in direct] == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
    ]
    assert len({item.mutation_group_id for item in direct}) == 1


def test_literal_import_overwrite_uses_one_compound_note_keyword_group(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    created = client.post(
        "/api/v1/notes/",
        json={"title": "Before", "content": "Old body"},
    )
    assert created.status_code == 201, created.text
    note_id = created.json()["id"]
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    before_cursor = max(
        item.server_cursor or 0
        for item in sync_service.store.list_envelopes_after(dataset_id, 0)
    )
    content = json.dumps(
        [
            {
                "id": note_id,
                "title": "After",
                "content": "New body",
                "keywords": ["Updated"],
            }
        ]
    )

    response = client.post(
        "/api/v1/notes/import",
        json={
            "duplicate_strategy": "overwrite",
            "items": [
                {
                    "file_name": "overwrite.json",
                    "format": "json",
                    "content": content,
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["updated_count"] == 1
    note = chacha_db.get_note_by_id(note_id)
    assert note is not None and note["title"] == "After"
    assert [row["keyword"] for row in chacha_db.get_keywords_for_note(note_id)] == [
        "Updated"
    ]
    direct = [
        item
        for item in sync_service.store.list_envelopes_after(
            dataset_id, before_cursor
        )
        if item.routing_metadata.get("source") == "notes-api"
    ]
    assert [item.domain for item in direct] == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
    ]
    assert len({item.mutation_group_id for item in direct}) == 1


def test_literal_import_overwrite_replays_raw_request_before_mutable_lookup(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        json={"id": note_id, "title": "Before", "content": "Old body"},
    )
    assert created.status_code == 201, created.text
    reconciled_versions: list[int] = []

    def _record_reconciliation(**kwargs) -> None:
        reconciled_versions.append(int(kwargs["note_data"]["version"]))

    monkeypatch.setattr(
        notes_endpoint,
        "_reconcile_note_tasks_after_save",
        _record_reconciliation,
    )
    headers = {"Idempotency-Key": "literal-import-durable-overwrite"}
    imported_row = {
        "id": note_id,
        "title": "Imported original",
        "content": "Imported body",
        "keywords": ["Imported keyword"],
    }
    body = {
        "duplicate_strategy": "overwrite",
        "items": [
            {
                "file_name": "overwrite.json",
                "format": "json",
                "content": json.dumps([imported_row]),
            }
        ],
    }

    original = client.post("/api/v1/notes/import", headers=headers, json=body)
    assert original.status_code == 200, original.text
    original_group = _notes_api_group(
        sync_service, "literal-import-durable-overwrite:0:1"
    )
    original_ids = [item.client_envelope_id for item in original_group]

    conversation_id = chacha_db.add_conversation({"title": "Later conversation"})
    changed = client.put(
        f"/api/v1/notes/{note_id}",
        headers={
            "Expected-Version": str(chacha_db.get_note_by_id(note_id)["version"]),
            "Idempotency-Key": "literal-import-later-mutation",
        },
        json={
            "title": "Later title",
            "content": "Later body",
            "conversation_id": conversation_id,
            "keywords": ["Later keyword"],
        },
    )
    assert changed.status_code == 200, changed.text
    reconciliation_count = len(reconciled_versions)

    replay = client.post("/api/v1/notes/import", headers=headers, json=body)

    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()
    assert len(reconciled_versions) == reconciliation_count
    assert [
        item.client_envelope_id
        for item in _notes_api_group(
            sync_service, "literal-import-durable-overwrite:0:1"
        )
    ] == original_ids
    assert chacha_db.get_note_by_id(note_id)["title"] == "Later title"

    drifted_row = {**imported_row, "content": "Changed import request"}
    drifted = client.post(
        "/api/v1/notes/import",
        headers=headers,
        json={
            **body,
            "items": [
                {
                    **body["items"][0],
                    "content": json.dumps([drifted_row]),
                }
            ],
        },
    )
    assert drifted.status_code == 409, drifted.text

    strategy_drift = client.post(
        "/api/v1/notes/import",
        headers=headers,
        json={**body, "duplicate_strategy": "skip"},
    )
    assert strategy_drift.status_code == 409, strategy_drift.text


def test_literal_import_create_durable_replay_skips_stale_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    headers = {"Idempotency-Key": "literal-import-create-durable"}
    body = {
        "duplicate_strategy": "overwrite",
        "items": [
            {
                "file_name": "create.json",
                "format": "json",
                "content": json.dumps(
                    [{"title": "Imported create", "content": "Imported body"}]
                ),
            }
        ],
    }
    original = client.post("/api/v1/notes/import", headers=headers, json=body)
    assert original.status_code == 200, original.text
    group = _notes_api_group(sync_service, "literal-import-create-durable:0:1")
    note_id = next(item.object_id for item in group if item.domain == "notes.note")
    note = chacha_db.get_note_by_id(note_id)
    assert note is not None
    deleted = client.delete(
        f"/api/v1/notes/{note_id}",
        headers={"Expected-Version": str(note["version"])},
    )
    assert deleted.status_code == 204, deleted.text

    monkeypatch.setattr(
        notes_endpoint,
        "_reconcile_note_tasks_after_save",
        lambda **_kwargs: pytest.fail("durable replay must not reconcile"),
    )
    replay = client.post("/api/v1/notes/import", headers=headers, json=body)

    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()
    deleted_row = chacha_db.get_note_by_id(note_id, include_deleted=True)
    assert deleted_row is not None and deleted_row["deleted"] == 1


def test_inline_durable_original_response_replays_after_later_mutation(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    original_payload = {
        "title": "Original",
        "content": "Original body",
        "keywords": ["First"],
        "folder_paths": ["Archive/Original"],
    }
    original = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "durable-original-response"},
        json=original_payload,
    )
    assert original.status_code == 201, original.text
    original_body = original.json()

    changed = client.put(
        f"/api/v1/notes/{original_body['id']}",
        headers={
            "Idempotency-Key": "durable-later-mutation",
            "Expected-Version": str(original_body["version"]),
        },
        json={
            "title": "Later",
            "content": "Later body",
            "keywords": ["Second"],
            "folder_paths": ["Archive/Later"],
        },
    )
    assert changed.status_code == 200, changed.text

    original_keyword = original_body["keywords"][0]
    renamed_keyword = client.patch(
        f"/api/v1/notes/keywords/{original_keyword['id']}",
        headers={
            "Idempotency-Key": "durable-later-keyword-rename",
            "Expected-Version": str(original_keyword["version"]),
        },
        json={"keyword": "Renamed later"},
    )
    assert renamed_keyword.status_code == 200, renamed_keyword.text

    original_folder = chacha_db.get_note_folder_by_path("Archive/Original")
    parent_folder = chacha_db.get_note_folder_by_path("Archive")
    assert original_folder is not None
    assert parent_folder is not None
    coordinator = NotesOrganizationCoordinator(sync_service, chacha_db, "user-1")
    coordinator.capture(
        steps=(
            ServerOriginMutationStep(
                domain="notes.folder",
                operation="upsert",
                object_id=str(original_folder["sync_id"]),
                payload={
                    "name": "Renamed later",
                    "parent_sync_id": str(parent_folder["sync_id"]),
                },
                parent_id=str(parent_folder["sync_id"]),
            ),
        ),
        source="notes-api",
        idempotency_key="durable-later-folder-rename",
    )

    replay = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "durable-original-response"},
        json=original_payload,
    )

    assert replay.status_code == 201, replay.text
    assert replay.json() == original_body


def test_inline_durable_replay_finds_resource_before_over_100_later_revisions(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    payload = {
        "title": "History boundary",
        "content": "Body",
        "keywords": ["Original boundary keyword"],
    }
    headers = {"Idempotency-Key": "durable-history-boundary"}
    original = client.post("/api/v1/notes/", headers=headers, json=payload)
    assert original.status_code == 201, original.text
    keyword_sync_id = str(original.json()["keywords"][0]["sync_id"])
    coordinator = NotesOrganizationCoordinator(sync_service, chacha_db, "user-1")
    for revision in range(101):
        result = coordinator.capture(
            steps=(
                ServerOriginMutationStep(
                    domain="notes.keyword",
                    operation="upsert",
                    object_id=keyword_sync_id,
                    payload={"keyword": f"Later boundary keyword {revision}"},
                ),
            ),
            source="notes-api",
            idempotency_key=f"later-history-revision-{revision}",
        )
        assert result.fully_applied

    replay = client.post("/api/v1/notes/", headers=headers, json=payload)

    assert replay.status_code == 201, replay.text
    assert replay.json() == original.json()


def test_inline_durable_create_replays_after_later_note_deletion(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    payload = {
        "title": "Create before deletion",
        "content": "Body",
        "keywords": ["Deleted note keyword"],
        "folder_paths": ["Deleted/Create"],
    }
    headers = {"Idempotency-Key": "durable-create-before-note-deletion"}
    original = client.post("/api/v1/notes/", headers=headers, json=payload)
    assert original.status_code == 201, original.text
    note_id = original.json()["id"]
    deleted = client.delete(
        f"/api/v1/notes/{note_id}",
        headers={"Expected-Version": str(original.json()["version"])},
    )
    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_note_by_id(note_id) is None

    monkeypatch.setattr(
        notes_endpoint,
        "_reconcile_note_tasks_after_save",
        lambda **_kwargs: pytest.fail("durable replay must not reconcile"),
    )
    replay = client.post("/api/v1/notes/", headers=headers, json=payload)

    assert replay.status_code == 201, replay.text
    assert replay.json() == original.json()
    deleted_row = chacha_db.get_note_by_id(note_id, include_deleted=True)
    assert deleted_row is not None and deleted_row["deleted"] == 1


def test_inline_durable_update_replays_after_later_note_deletion(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    created = client.post(
        "/api/v1/notes/",
        json={"title": "Before update deletion", "content": "Body"},
    )
    assert created.status_code == 201, created.text
    note_id = created.json()["id"]
    payload = {
        "title": "Update before deletion",
        "keywords": ["Deleted update keyword"],
        "folder_paths": ["Deleted/Update"],
    }
    headers = {
        "Expected-Version": str(created.json()["version"]),
        "Idempotency-Key": "durable-update-before-note-deletion",
    }
    original = client.put(f"/api/v1/notes/{note_id}", headers=headers, json=payload)
    assert original.status_code == 200, original.text
    deleted = client.delete(
        f"/api/v1/notes/{note_id}",
        headers={"Expected-Version": str(original.json()["version"])},
    )
    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_note_by_id(note_id) is None

    monkeypatch.setattr(
        notes_endpoint,
        "_reconcile_note_tasks_after_save",
        lambda **_kwargs: pytest.fail("durable replay must not reconcile"),
    )
    replay = client.put(f"/api/v1/notes/{note_id}", headers=headers, json=payload)

    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()
    deleted_row = chacha_db.get_note_by_id(note_id, include_deleted=True)
    assert deleted_row is not None and deleted_row["deleted"] == 1


def test_inline_durable_replay_survives_later_organization_resource_deletion(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    payload = {
        "title": "Organization before deletion",
        "content": "Body",
        "keywords": ["Deleted original keyword"],
        "folder_paths": ["Deleted/Original"],
    }
    headers = {"Idempotency-Key": "durable-before-organization-deletion"}
    original = client.post("/api/v1/notes/", headers=headers, json=payload)
    assert original.status_code == 201, original.text
    body = original.json()

    changed = client.put(
        f"/api/v1/notes/{body['id']}",
        headers={
            "Expected-Version": str(body["version"]),
            "Idempotency-Key": "durable-remove-original-organization",
        },
        json={
            "title": "Organization later",
            "keywords": ["Replacement keyword"],
            "folder_paths": ["Replacement/Folder"],
        },
    )
    assert changed.status_code == 200, changed.text

    original_keyword = body["keywords"][0]
    keyword_deleted = client.delete(
        f"/api/v1/notes/keywords/{original_keyword['id']}",
        headers={"Expected-Version": str(original_keyword["version"])},
    )
    assert keyword_deleted.status_code == 204, keyword_deleted.text

    original_folder_rows = [
        chacha_db.get_note_folder_by_path(path)
        for path in ("Deleted/Original", "Deleted")
    ]
    assert all(row is not None for row in original_folder_rows)
    coordinator = NotesOrganizationCoordinator(sync_service, chacha_db, "user-1")
    deleted_folders = coordinator.capture(
        steps=tuple(
            ServerOriginMutationStep(
                domain="notes.folder",
                operation="tombstone",
                object_id=str(row["sync_id"]),
                payload={},
            )
            for row in original_folder_rows
            if row is not None
        ),
        source="notes-api",
        idempotency_key="durable-delete-original-folders",
    )
    assert deleted_folders.fully_applied

    replay = client.post("/api/v1/notes/", headers=headers, json=payload)

    assert replay.status_code == 201, replay.text
    assert replay.json() == body


def test_inline_auto_title_replay_looks_up_manifest_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    generated: list[str] = []

    def _generate_title(content: str, options=None) -> str:
        del content, options
        title = f"Generated {len(generated) + 1}"
        generated.append(title)
        return title

    monkeypatch.setattr(notes_endpoint, "generate_note_title", _generate_title)
    payload = {
        "content": "Auto-title body",
        "auto_title": True,
        "keywords": ["Generated"],
    }
    first = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "durable-auto-title"},
        json=payload,
    )
    assert first.status_code == 201, first.text

    replay = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "durable-auto-title"},
        json=payload,
    )

    assert replay.status_code == 201, replay.text
    assert replay.json() == first.json()
    assert generated == ["Generated 1"]


def test_inline_note_append_failure_writes_no_product_projection(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    monkeypatch.setattr(
        sync_service.store,
        "insert_envelopes_atomic",
        lambda _envelopes: (_ for _ in ()).throw(SyncStoreError("private append")),
    )
    response = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "inline-append-failure"},
        json={
            "title": "Not written",
            "content": "Body",
            "keywords": ["No keyword"],
            "folder_paths": ["No/Folder"],
        },
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_append_failed"
    )
    assert chacha_db.search_notes("Not written") == []
    assert chacha_db.get_keyword_by_text("No keyword") is None
    assert chacha_db.get_note_folder_by_path("No/Folder") is None


def test_inline_note_compound_preflight_rejection_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    original = sync_service._evaluate_envelope

    def _reject_keyword(dataset, envelope, *, context=None):
        if envelope.domain == "notes.keyword":
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="test_rejection",
                message="safe test rejection",
            )
        return original(dataset, envelope, context=context)

    monkeypatch.setattr(sync_service, "_evaluate_envelope", _reject_keyword)
    response = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "inline-preflight-rejection"},
        json={
            "title": "Rejected compound",
            "content": "Body",
            "keywords": ["Rejected keyword"],
            "folder_paths": ["Rejected/Folder"],
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "notes_organization_sync_preflight_failed"
    )
    assert chacha_db.search_notes("Rejected compound") == []
    assert chacha_db.get_keyword_by_text("Rejected keyword") is None
    assert chacha_db.get_note_folder_by_path("Rejected/Folder") is None
    assert _notes_api_group(sync_service, "inline-preflight-rejection") == []


def test_inline_note_same_key_drift_conflicts_without_second_projection(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    headers = {"Idempotency-Key": "inline-note-drift"}
    first = client.post(
        "/api/v1/notes/",
        headers=headers,
        json={
            "title": "Original drift note",
            "content": "Original body",
            "keywords": ["Original keyword"],
        },
    )
    drift = client.post(
        "/api/v1/notes/",
        headers=headers,
        json={
            "title": "Changed drift note",
            "content": "Changed body",
            "keywords": ["Changed keyword"],
        },
    )

    assert first.status_code == 201, first.text
    assert drift.status_code == 409, drift.text
    assert drift.json()["detail"]["error_code"] == (
        "sync_server_origin_batch_idempotency_conflict"
    )
    assert chacha_db.get_note_by_id(first.json()["id"])["title"] == (
        "Original drift note"
    )
    assert chacha_db.get_keyword_by_text("Changed keyword") is None


@pytest.mark.parametrize(
    ("state", "expected_code"),
    [
        ("partial", "notes_organization_sync_domains_incomplete"),
        ("initializing", "notes_organization_sync_not_ready"),
        ("failed", "notes_organization_sync_not_ready"),
    ],
)
def test_inline_note_active_sync_readiness_failures_do_not_fall_back(
    state: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    if state == "partial":
        replacement = replace(
            dataset,
            domains=[
                domain for domain in dataset.domains if domain != "notes.folder_link"
            ],
        )
    else:
        metadata = dict(dataset.metadata)
        organization = dict(metadata["notes_organization_v1"])
        organization["state"] = state
        organization["error_code"] = "safe_repair_code" if state == "failed" else None
        metadata["notes_organization_v1"] = organization
        replacement = replace(dataset, metadata=metadata)
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [replacement] if user_id == "user-1" else [],
    )

    response = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": f"inline-readiness-{state}"},
        json={
            "title": "Must not write",
            "content": "Body",
            "keywords": ["Must not write"],
            "folder_paths": ["Must/Not/Write"],
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == expected_code
    assert chacha_db.search_notes("Must not write") == []
    assert chacha_db.get_keyword_by_text("Must not write") is None


def test_bulk_compound_append_failure_isolated_without_product_projection(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    monkeypatch.setattr(
        sync_service.store,
        "insert_envelopes_atomic",
        lambda _envelopes: (_ for _ in ()).throw(SyncStoreError("private append")),
    )
    response = client.post(
        "/api/v1/notes/bulk",
        headers={"Idempotency-Key": "bulk-append-failure"},
        json={
            "notes": [
                {
                    "title": "Bulk not written",
                    "content": "Body",
                    "keywords": ["Bulk not written"],
                    "folder_paths": ["Bulk/Not/Written"],
                }
            ]
        },
    )

    assert response.status_code == 207, response.text
    assert response.json()["failed_count"] == 1
    assert chacha_db.search_notes("Bulk not written") == []
    assert chacha_db.get_keyword_by_text("Bulk not written") is None
    assert _notes_api_group(sync_service, "bulk-append-failure:0") == []


def test_bulk_compound_interruption_resumes_and_replays_without_duplicates(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    original = sync_service.materializers["notes.keyword"]
    sync_service.materializers["notes.keyword"] = _FailingOrganizationMaterializer()
    headers = {"Idempotency-Key": "bulk-interruption-replay"}
    payload = {
        "notes": [
            {
                "title": "Bulk resumes",
                "content": "Body",
                "keywords": ["Bulk resume keyword"],
                "folder_paths": ["Bulk/Resume"],
            }
        ]
    }

    failed = client.post("/api/v1/notes/bulk", headers=headers, json=payload)
    assert failed.status_code == 207, failed.text
    first_group = _notes_api_group(sync_service, "bulk-interruption-replay:0")
    assert [item.apply_status for item in first_group[:3]] == [
        "applied",
        "failed",
        "pending",
    ]
    note_id = first_group[0].object_id

    sync_service.materializers["notes.keyword"] = original
    resumed = client.post("/api/v1/notes/bulk", headers=headers, json=payload)
    replay = client.post("/api/v1/notes/bulk", headers=headers, json=payload)

    assert resumed.status_code == replay.status_code == 200, resumed.text
    assert replay.json() == resumed.json()
    resumed_group = _notes_api_group(sync_service, "bulk-interruption-replay:0")
    assert [item.client_envelope_id for item in resumed_group] == [
        item.client_envelope_id for item in first_group
    ]
    assert chacha_db.get_note_by_id(note_id)["version"] == 1
    assert len(chacha_db.get_keywords_for_note(note_id)) == 1
    assert len(chacha_db.get_note_folders_for_note(note_id)) == 2


def test_bulk_auto_title_replay_looks_up_manifest_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    generated: list[str] = []

    def _generate_title(content: str, options=None) -> str:
        del content, options
        title = f"Bulk generated {len(generated) + 1}"
        generated.append(title)
        return title

    monkeypatch.setattr(notes_endpoint, "generate_note_title", _generate_title)
    headers = {"Idempotency-Key": "bulk-auto-title-replay"}
    payload = {
        "notes": [
            {
                "content": "Bulk auto-title body",
                "auto_title": True,
                "keywords": ["Generated"],
            }
        ]
    }

    first = client.post("/api/v1/notes/bulk", headers=headers, json=payload)
    replay = client.post("/api/v1/notes/bulk", headers=headers, json=payload)

    assert first.status_code == replay.status_code == 200, replay.text
    assert replay.json() == first.json()
    assert generated == ["Bulk generated 1"]


def test_inline_note_retry_resumes_manifest_without_duplicate_note_versions(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    original = sync_service.materializers["notes.keyword"]
    sync_service.materializers["notes.keyword"] = _FailingOrganizationMaterializer()
    headers = {"Idempotency-Key": "inline-note-resume"}
    payload = {
        "title": "Resume compound",
        "content": "Body",
        "keywords": ["Resume keyword"],
        "folder_paths": ["Resume/Folder"],
    }

    failed = client.post("/api/v1/notes/", headers=headers, json=payload)
    assert failed.status_code == 503, failed.text
    group = _notes_api_group(sync_service, "inline-note-resume")
    assert [item.apply_status for item in group[:3]] == [
        "applied",
        "failed",
        "pending",
    ]
    note_id = group[0].object_id
    assert chacha_db.get_note_by_id(note_id)["version"] == 1

    sync_service.materializers["notes.keyword"] = original
    resumed = client.post("/api/v1/notes/", headers=headers, json=payload)

    assert resumed.status_code == 201, resumed.text
    resumed_group = _notes_api_group(sync_service, "inline-note-resume")
    assert [item.client_envelope_id for item in resumed_group] == [
        item.client_envelope_id for item in group
    ]
    assert chacha_db.get_note_by_id(note_id)["version"] == 1
    assert len(chacha_db.get_keywords_for_note(note_id)) == 1
    assert len(chacha_db.get_note_folders_for_note(note_id)) == 2


def test_inline_patch_without_version_replays_manifest_before_projection_checks(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        json={"id": note_id, "title": "Before", "content": "Body"},
    )
    assert created.status_code == 201, created.text
    headers = {"Idempotency-Key": "inline-patch-no-version"}
    payload = {"title": "After", "keywords": ["Replay"]}

    first = client.patch(f"/api/v1/notes/{note_id}", headers=headers, json=payload)
    replay = client.patch(f"/api/v1/notes/{note_id}", headers=headers, json=payload)

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["version"] == first.json()["version"]
    assert chacha_db.get_note_by_id(note_id)["version"] == first.json()["version"]


def test_inline_folder_omission_preserves_and_empty_list_removes_relationships(
    client: TestClient,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": "inline-folder-semantics-create"},
        json={
            "id": note_id,
            "title": "Folder semantics",
            "content": "Body",
            "folder_paths": ["Keep/Until/Removed"],
        },
    )
    assert created.status_code == 201, created.text

    preserved = client.patch(
        f"/api/v1/notes/{note_id}",
        headers={
            "expected-version": str(created.json()["version"]),
            "Idempotency-Key": "inline-folder-semantics-preserve",
        },
        json={"title": "Folder still present", "keywords": ["Trigger compound"]},
    )
    assert preserved.status_code == 200, preserved.text
    assert [item["path"] for item in preserved.json()["folders"]] == [
        "Keep",
        "Keep/Until",
        "Keep/Until/Removed",
    ]

    removed = client.patch(
        f"/api/v1/notes/{note_id}",
        headers={
            "expected-version": str(preserved.json()["version"]),
            "Idempotency-Key": "inline-folder-semantics-remove",
        },
        json={"folder_paths": []},
    )
    assert removed.status_code == 200, removed.text
    assert removed.json()["folders"] == []


def test_inline_manual_folder_intent_survives_final_source_removal(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        json={
            "id": note_id,
            "title": "Source folder",
            "content": "Body",
            "folder_paths": ["Sources/Shared"],
        },
    )
    assert created.status_code == 201, created.text
    folder = chacha_db.get_note_folder_by_path("Sources/Shared")
    assert folder is not None
    chacha_db.sync_note_folders(note_id, [])
    chacha_db.sync_note_source_folders(note_id, 41, [folder["path"]])

    saved = client.patch(
        f"/api/v1/notes/{note_id}",
        headers={
            "expected-version": str(created.json()["version"]),
            "Idempotency-Key": "inline-manual-over-source",
        },
        json={"folder_paths": [folder["path"]]},
    )
    assert saved.status_code == 200, saved.text

    with chacha_db.transaction() as conn:
        manual_count = conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0]
    assert manual_count == 1

    chacha_db.sync_note_source_folders(note_id, 41, [])

    assert [row["path"] for row in chacha_db.get_note_folders_for_note(note_id)] == [
        "Sources",
        "Sources/Shared",
    ]


def test_source_folder_read_set_race_preserves_concurrent_manual_membership(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    note_id = str(uuid.uuid4())
    created = client.post(
        "/api/v1/notes/",
        json={
            "id": note_id,
            "title": "Source race",
            "content": "Body",
            "folder_paths": ["Race"],
        },
    )
    assert created.status_code == 201, created.text
    folder = chacha_db.get_note_folder_by_path("Race")
    assert folder is not None
    chacha_db.sync_note_folders(note_id, [])
    coordinator = NotesOrganizationCoordinator(
        service=sync_service,
        user_id="user-1",
        note_db=chacha_db,
    )
    source_add = coordinator.plan_source_folder_change(
        note_id=note_id,
        source_id=73,
        folder_id=int(folder["id"]),
        present=True,
        idempotency_key="source-race-add",
    )
    coordinator.capture(
        steps=source_add.steps,
        source="notes-ingestion",
        idempotency_key="source-race-add",
    )

    stale_remove = coordinator.plan_source_folder_change(
        note_id=note_id,
        source_id=73,
        folder_id=int(folder["id"]),
        present=False,
        idempotency_key="source-race-remove",
    )
    provenance = stale_remove.steps[0].routing_metadata[
        "notes_folder_origin_provenance"
    ]
    assert set(provenance) == {
        "operation",
        "source_id",
        "pre_state_hash",
        "post_state_hash",
    }
    for key in ("pre_state_hash", "post_state_hash"):
        assert len(str(provenance[key])) == 64
        assert set(str(provenance[key])) <= set("0123456789abcdef")
    assert "Race" not in repr(stale_remove.steps[0].routing_metadata)
    chacha_db.sync_note_folders(note_id, [folder["path"]])

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        coordinator.capture(
            steps=stale_remove.steps,
            source="notes-ingestion",
            idempotency_key="source-race-remove",
        )

    with chacha_db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ? AND folder_id = ?",
            (note_id, 73, folder["id"]),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 0


@pytest.mark.parametrize("folder_paths", [["/private/note"], ["C:\\private\\note"], "a,b"])
def test_inline_folder_paths_reject_absolute_or_comma_delimited_input(
    client: TestClient,
    folder_paths,
) -> None:
    response = client.post(
        "/api/v1/notes/",
        json={
            "title": "Invalid folder",
            "content": "Body",
            "folder_paths": folder_paths,
        },
    )

    assert response.status_code == 422


def _setup_sync_merge(
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
    *,
    suffix: str,
) -> tuple[dict, dict]:
    note_id = str(uuid.uuid4())
    conversation_id = f"sync-merge-conversation-{suffix}"
    source_sync_id = str(uuid.uuid4())
    target_sync_id = str(uuid.uuid4())
    collection_sync_id = str(uuid.uuid4())
    setup = (
        ("notes.note", note_id, {"title": "Merge note", "content": "Body"}),
        (
            "chat.conversation",
            conversation_id,
            {
                "title": "Merge conversation",
                "assistant_kind": "persona",
                "assistant_id": "assistant-1",
                "scope_type": "global",
            },
        ),
        ("notes.keyword", source_sync_id, {"keyword": f"Merge source {suffix}"}),
        ("notes.keyword", target_sync_id, {"keyword": f"Merge target {suffix}"}),
        (
            "notes.keyword_collection",
            collection_sync_id,
            {"name": f"Merge collection {suffix}", "parent_sync_id": None},
        ),
    )
    for domain, object_id, payload in setup:
        capture_server_origin_mutation(
            sync_service,
            user_id="user-1",
            domain=domain,
            operation="upsert",
            object_id=object_id,
            payload=payload,
            source=f"sync-merge-setup-{suffix}",
        )
    coordinator = NotesOrganizationCoordinator(sync_service, chacha_db, "user-1")
    source = chacha_db.get_keyword_by_text(f"Merge source {suffix}")
    target = chacha_db.get_keyword_by_text(f"Merge target {suffix}")
    assert source is not None and target is not None
    source_members = (
        (
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": note_id,
                "keyword_sync_id": source_sync_id,
            },
        ),
        (
            "notes.keyword_link",
            {
                "subject_type": "conversation",
                "subject_id": conversation_id,
                "keyword_sync_id": source_sync_id,
            },
        ),
        (
            "notes.keyword_collection_link",
            {
                "collection_sync_id": collection_sync_id,
                "keyword_sync_id": source_sync_id,
            },
        ),
        (
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": note_id,
                "keyword_sync_id": target_sync_id,
            },
        ),
    )
    for index, (domain, members) in enumerate(source_members):
        plan = coordinator.plan_relationship(domain, members, True)
        coordinator.capture(
            steps=plan.steps,
            source=f"sync-merge-setup-{suffix}",
            idempotency_key=f"relationship-{index}",
        )
    return source, target


def test_merge_sync_group_moves_all_relationship_domains_and_replays_exactly(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    source, target = _setup_sync_merge(chacha_db, sync_service, suffix="complete")
    headers = {
        "expected-version": str(source["version"]),
        "Idempotency-Key": "sync-merge-complete",
    }
    payload = {
        "target_keyword_id": target["id"],
        "expected_target_version": target["version"],
    }

    first = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers=headers,
        json=payload,
    )
    replay = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers=headers,
        json=payload,
    )

    assert first.status_code == replay.status_code == 200, first.text
    assert replay.json() == first.json()
    assert first.json()["merged_note_links"] == 0
    assert first.json()["merged_conversation_links"] == 1
    assert first.json()["merged_collection_links"] == 1
    group = _notes_api_group(sync_service, "sync-merge-complete")
    assert [(item.domain, item.operation) for item in group] == [
        ("notes.keyword_link", "upsert"),
        ("notes.keyword_collection_link", "upsert"),
        ("notes.keyword_link", "tombstone"),
        ("notes.keyword_link", "tombstone"),
        ("notes.keyword_collection_link", "tombstone"),
        ("notes.keyword", "tombstone"),
    ]
    assert all(item.apply_status == "applied" for item in group)
    assert chacha_db.get_keyword_by_id(source["id"]) is None


def test_merge_sync_rejects_dormant_flashcard_dependency_before_append(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    source, target = _setup_sync_merge(chacha_db, sync_service, suffix="flashcard")
    card_uuid = chacha_db.add_flashcard({"front": "Front", "back": "Back"})
    with chacha_db.transaction() as conn:
        card = conn.execute(
            "SELECT id FROM flashcards WHERE uuid = ?", (card_uuid,)
        ).fetchone()
        conn.execute(
            "INSERT INTO flashcard_keywords(card_id, keyword_id, created_at) "
            "VALUES (?, ?, ?)",
            (card["id"], source["id"], "2026-08-09T08:00:00+00:00"),
        )
        conn.execute("UPDATE flashcards SET deleted = 1 WHERE id = ?", (card["id"],))

    response = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers={
            "expected-version": str(source["version"]),
            "Idempotency-Key": "sync-merge-flashcard",
        },
        json={
            "target_keyword_id": target["id"],
            "expected_target_version": target["version"],
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == (
        "notes_keyword_merge_unsynchronized_dependency"
    )
    assert _notes_api_group(sync_service, "sync-merge-flashcard") == []
    assert chacha_db.get_keyword_by_id(source["id"]) is not None


def test_merge_relationship_set_race_blocks_final_source_tombstone(
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    source, target = _setup_sync_merge(chacha_db, sync_service, suffix="race")
    coordinator = NotesOrganizationCoordinator(sync_service, chacha_db, "user-1")
    stale_merge = coordinator.plan_keyword_merge(
        source_keyword_id=int(source["id"]),
        target_keyword_id=int(target["id"]),
        expected_source_version=int(source["version"]),
        expected_target_version=int(target["version"]),
    )
    final_guard = stale_merge.steps[-1].routing_metadata[
        "notes_keyword_merge_precondition"
    ]
    assert set(final_guard) == {"relationship_set_hash"}
    assert len(str(final_guard["relationship_set_hash"])) == 64
    assert set(str(final_guard["relationship_set_hash"])) <= set(
        "0123456789abcdef"
    )
    late_note_id = str(uuid.uuid4())
    capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="notes.note",
        operation="upsert",
        object_id=late_note_id,
        payload={"title": "Late merge link", "content": "Body"},
        source="sync-merge-race-late-note",
    )
    late_link = coordinator.plan_relationship(
        "notes.keyword_link",
        {
            "subject_type": "note",
            "subject_id": late_note_id,
            "keyword_sync_id": str(source["sync_id"]),
        },
        True,
    )
    coordinator.capture(
        steps=late_link.steps,
        source="sync-merge-race-late-link",
        idempotency_key="late-link",
    )

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        coordinator.capture(
            steps=stale_merge.steps,
            source="notes-api",
            idempotency_key="sync-merge-race",
        )

    assert chacha_db.get_keyword_by_id(source["id"]) is not None
    assert [
        row["sync_id"] for row in chacha_db.get_keywords_for_note(late_note_id)
    ] == [source["sync_id"]]


def test_merge_sync_interruption_resumes_before_source_keyword_tombstone(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    source, target = _setup_sync_merge(chacha_db, sync_service, suffix="resume")
    original = sync_service.materializers["notes.keyword_link"]
    sync_service.materializers["notes.keyword_link"] = (
        _FailingSecondRelationshipMaterializer(original)
    )
    headers = {
        "expected-version": str(source["version"]),
        "Idempotency-Key": "sync-merge-resume",
    }
    payload = {
        "target_keyword_id": target["id"],
        "expected_target_version": target["version"],
    }

    failed = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers=headers,
        json=payload,
    )
    assert failed.status_code == 503, failed.text
    first_group = _notes_api_group(sync_service, "sync-merge-resume")
    assert [item.apply_status for item in first_group] == [
        "applied",
        "applied",
        "failed",
        "pending",
        "pending",
        "pending",
    ]
    assert chacha_db.get_keyword_by_id(source["id"]) is not None

    sync_service.materializers["notes.keyword_link"] = original
    resumed = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers=headers,
        json=payload,
    )

    assert resumed.status_code == 200, resumed.text
    resumed_group = _notes_api_group(sync_service, "sync-merge-resume")
    assert [item.client_envelope_id for item in resumed_group] == [
        item.client_envelope_id for item in first_group
    ]
    assert all(item.apply_status == "applied" for item in resumed_group)
    assert chacha_db.get_keyword_by_id(source["id"]) is None


def test_merge_sync_append_failure_preserves_source_projection(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> None:
    source, target = _setup_sync_merge(chacha_db, sync_service, suffix="append")
    before_note_ids = {
        row["id"] for row in chacha_db.get_notes_for_keyword(source["id"])
    }
    monkeypatch.setattr(
        sync_service.store,
        "insert_envelopes_atomic",
        lambda _envelopes: (_ for _ in ()).throw(SyncStoreError("private append")),
    )

    response = client.post(
        f"/api/v1/notes/keywords/{source['id']}/merge",
        headers={
            "expected-version": str(source["version"]),
            "Idempotency-Key": "sync-merge-append",
        },
        json={
            "target_keyword_id": target["id"],
            "expected_target_version": target["version"],
        },
    )

    assert response.status_code == 503, response.text
    assert chacha_db.get_keyword_by_id(source["id"]) is not None
    assert {
        row["id"] for row in chacha_db.get_notes_for_keyword(source["id"])
    } == before_note_ids
    assert _notes_api_group(sync_service, "sync-merge-append") == []
