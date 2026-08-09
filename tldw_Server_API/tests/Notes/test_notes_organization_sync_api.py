from __future__ import annotations

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
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
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


def test_direct_active_sync_keyword_merge_remains_fail_closed(
    client: TestClient,
    chacha_db: CharactersRAGDB,
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
        headers={"expected-version": str(source["version"])},
        json={
            "target_keyword_id": target["id"],
            "expected_target_version": target["version"],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_keyword_by_id(source["id"]) is not None
    assert chacha_db.get_keyword_by_id(target["id"]) is not None


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
