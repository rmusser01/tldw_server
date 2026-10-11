"""Finite real HTTP regressions for TASK13421.1's retained Notes acknowledgments."""

import json
from dataclasses import replace

import pytest

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import notes as endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.materializers import NotesMaterializer, NotesOrganizationMaterializer
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, NOTES_ORGANIZATION_DOMAINS, SyncDatasetCreate
from tldw_Server_API.app.core.Sync.v2.notes_organization_bootstrap import NotesOrganizationBootstrapper
from tldw_Server_API.app.core.Sync.v2.notes_provenance import NotesProvenanceMaterializer
from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import retain_notes_provenance
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    SyncServerOriginMaterializationError,
    capture_server_origin_mutation,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Notes import test_notes_organization_sync_api as fixtures

chacha_db = fixtures.chacha_db
client = fixtures.client
sync_service = fixtures.sync_service

BASE = "/api/v1/notes"
PROVENANCE = {"origin": "knowledge_qa", "question": "Original evidence"}


@pytest.fixture
def active(client, sync_service):
    sync_service.adapters = default_sync_v2_registry()
    return client


def _original(active, method):
    created = active.post(
        BASE + "/",
        json={
            "title": "Evidence",
            "content": "Body",
            "knowledge_provenance": PROVENANCE,
            "expected_provenance_version": 0,
        },
    )
    assert created.status_code == 201, created.text
    path = BASE + "/" + created.json()["id"]
    headers = {"expected-version": "1", "Idempotency-Key": "retained-original"}
    body = {"content": "Accepted", "keywords": ["Original"], "folder_paths": ["Original/Folder"]}
    accepted = getattr(active, method)(path, headers=headers, json=body)
    assert accepted.status_code == 200, accepted.text
    return path, headers, body, accepted


def _later(active, path, *, child=False):
    body = {"content": "Later", "keywords": ["Later"], "folder_paths": ["Later/Folder"]}
    if child:
        body.update(knowledge_provenance={"origin": "reviewed_sources"}, expected_provenance_version=1)
    response = active.put(path, headers={"expected-version": "2"}, json=body)
    assert response.status_code == 200, response.text
    return response


@pytest.mark.parametrize("method", ["put", "patch"])
@pytest.mark.parametrize("later_child", [False, True])
def test_omitted_provenance_ack_replays_original_group_after_later_edit(
    active, sync_service, chacha_db, record_property, method, later_child
):
    path, headers, body, accepted = _original(active, method)
    group = fixtures._notes_api_group(sync_service, headers["Idempotency-Key"])
    assert all(member.domain != "notes.provenance" for member in group)
    assert all(member.apply_status == "applied" for member in group)
    assert accepted.json()["knowledge_provenance"] == PROVENANCE
    later = _later(active, path, child=later_child)
    before = chacha_db.get_note_by_id(accepted.json()["id"])
    child_before = chacha_db.note_provenance_store.get(accepted.json()["id"])
    replay = getattr(active, method)(path, headers=headers, json=body)
    record_property("original_ack", accepted.text)
    record_property("later_ack", later.text)
    record_property("replay", replay.text)
    assert replay.status_code == 200, replay.text
    assert replay.json() == accepted.json()
    assert chacha_db.get_note_by_id(accepted.json()["id"]) == before
    assert chacha_db.note_provenance_store.get(accepted.json()["id"]) == child_before
    assert fixtures._notes_api_group(sync_service, headers["Idempotency-Key"]) == group
    current = active.get(path)
    assert current.status_code == 200, current.text
    for key in ("content", "version", "knowledge_provenance", "knowledge_provenance_version"):
        assert current.json()[key] == later.json()[key]
    assert [item["keyword"] for item in current.json()["keywords"]] == ["Later"]
    assert [item["path"] for item in current.json()["folders"]] == ["Later", "Later/Folder"]


def _absent_original(active, method):
    created = active.post(BASE + "/", json={"title": "No evidence", "content": "Body"})
    assert created.status_code == 201, created.text
    path = BASE + "/" + created.json()["id"]
    headers = {"expected-version": "1", "Idempotency-Key": "original-absent-child"}
    body = {"content": "Accepted without evidence", "keywords": ["Original"]}
    accepted = getattr(active, method)(path, headers=headers, json=body)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["knowledge_provenance_state"] == "absent"
    assert accepted.json()["knowledge_provenance_version"] == 0
    assert accepted.json()["knowledge_provenance"] is None
    return path, headers, body, accepted


def _first_later_child(active, path):
    later = active.put(
        path,
        headers={"expected-version": "2"},
        json={
            "content": "Later with first evidence",
            "knowledge_provenance": PROVENANCE,
            "expected_provenance_version": 0,
        },
    )
    assert later.status_code == 200, later.text
    assert later.json()["knowledge_provenance_version"] == 1
    return later


@pytest.mark.parametrize("method", ["put", "patch"])
def test_absent_provenance_ack_replays_after_first_later_child(
    active, sync_service, chacha_db, record_property, method
):
    path, headers, body, accepted = _absent_original(active, method)
    note_id = accepted.json()["id"]
    group = fixtures._notes_api_group(sync_service, headers["Idempotency-Key"])
    assert all(member.domain != "notes.provenance" for member in group)
    assert all(member.apply_status == "applied" for member in group)
    assert chacha_db.note_provenance_store.get(note_id, include_deleted=True) is None
    boundary = max(int(member.server_cursor or 0) for member in group)
    assert (
        sync_service.store.get_envelope_for_entity_at_or_before(
            group[0].dataset_id, "notes.provenance", entity_id=note_id, server_sequence=boundary
        )
        is None
    )
    later = _first_later_child(active, path)
    before = chacha_db.get_note_by_id(note_id)
    child_before = chacha_db.note_provenance_store.get(note_id)
    replay = getattr(active, method)(path, headers=headers, json=body)
    record_property("original_absent_ack", accepted.text)
    record_property("first_later_child_ack", later.text)
    record_property("absent_replay", replay.text)
    assert replay.status_code == 200, replay.text
    assert replay.json() == accepted.json()
    assert chacha_db.get_note_by_id(note_id) == before
    assert chacha_db.note_provenance_store.get(note_id) == child_before
    assert fixtures._notes_api_group(sync_service, headers["Idempotency-Key"]) == group
    history = sync_service.store.list_envelopes_for_entity(
        group[0].dataset_id, "notes.provenance", entity_id=note_id, limit=10
    )
    assert len(history) == 1
    assert history[0].object_revision == 1
    current = active.get(path)
    assert current.status_code == 200, current.text
    for key in ("content", "version", "knowledge_provenance_state", "knowledge_provenance_version"):
        assert current.json()[key] == later.json()[key]
    assert current.json()["knowledge_provenance"] == PROVENANCE


@pytest.mark.parametrize("method", ["put", "patch"])
def test_absent_provenance_ack_keeps_capability_gate(active, sync_service, method):
    path, headers, body, _ = _absent_original(active, method)
    _first_later_child(active, path)
    sync_service.settings = replace(
        sync_service.settings,
        server_trusted_encryption=server_trusted_encryption_status_from_config(
            mode=None, server_trusted_enabled=False, auth_mode="multi_user"
        ),
    )
    denied = getattr(active, method)(path, headers=headers, json=body)
    assert denied.status_code == 409, denied.text
    assert "Original evidence" not in denied.text


@pytest.mark.parametrize("method", ["put", "patch"])
def test_absent_provenance_ack_refuses_incompatible_parent_version(active, chacha_db, method):
    path, _, _, accepted = _absent_original(active, method)
    chacha_db.update_note(accepted.json()["id"], {"content": "Local version moved"}, expected_version=2)
    denied = getattr(active, method)(
        path,
        headers={"expected-version": "3", "Idempotency-Key": "incompatible-absent-parent"},
        json={"content": "Must not acknowledge mismatched version", "keywords": ["Version control"]},
    )
    assert denied.status_code == 409, denied.text
    assert "notes_provenance_projection_incomplete" in denied.text


@pytest.mark.parametrize("method", ["put", "patch"])
@pytest.mark.parametrize("change", ["key", "payload", "keywords", "folders", "version", "method"])
def test_omitted_provenance_replay_rejects_changed_request(active, chacha_db, method, change):
    path, headers, body, accepted = _original(active, method)
    _later(active, path, child=True)
    before = chacha_db.get_note_by_id(accepted.json()["id"])
    altered_headers, altered_body = dict(headers), dict(body)
    request_method = method
    if change == "key":
        altered_headers["Idempotency-Key"] = "unrelated-key"
    elif change == "payload":
        altered_body["content"] = "Must not persist"
    elif change == "keywords":
        altered_body["keywords"] = ["Must not exist"]
    elif change == "folders":
        altered_body["folder_paths"] = ["Must not exist"]
    elif change == "version":
        altered_headers["expected-version"] = "2"
    else:
        request_method = "patch" if method == "put" else "put"
    rejected = getattr(active, request_method)(path, headers=altered_headers, json=altered_body)
    assert rejected.status_code == 409, rejected.text
    assert chacha_db.get_note_by_id(accepted.json()["id"]) == before
    assert chacha_db.get_keyword_by_text("Must not exist") is None
    assert chacha_db.get_note_folder_by_path("Must not exist") is None


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_replay_does_not_cross_owner(active, sync_service, chacha_db, tmp_path, monkeypatch, method):
    path, headers, body, accepted = _original(active, method)
    foreign = CharactersRAGDB(str(tmp_path / "foreign-notes.db"), client_id="other-owner")
    foreign_service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "foreign-sync.db")),
        adapters=default_sync_v2_registry(),
        materializers={
            "notes.note": NotesMaterializer(foreign),
            **{domain: NotesOrganizationMaterializer(foreign, domain) for domain in NOTES_ORGANIZATION_DOMAINS},
        },
        dataset_bootstrapper=NotesOrganizationBootstrapper(foreign),
        settings=sync_service.settings,
        clock=sync_service.clock,
    )
    foreign_service.bootstrap_profile(
        user_id="other-owner",
        mode="server_frontend",
        device_id="other-device",
        requested_domains=[*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS],
    )

    async def foreign_db():
        return foreign

    async def foreign_user():
        return User(id="other-owner", username="other-owner", is_admin=True)

    active.app.dependency_overrides[endpoint.get_chacha_db_for_user] = foreign_db
    active.app.dependency_overrides[endpoint.get_request_user] = foreign_user
    monkeypatch.setattr(endpoint, "get_active_server_origin_sync_service_for_user", lambda _: foreign_service)
    try:
        denied = getattr(active, method)(path, headers=headers, json=body)
        assert denied.status_code == 404, denied.text
        assert "Original evidence" not in denied.text
        assert foreign.count_notes() == 0
        assert chacha_db.get_note_by_id(accepted.json()["id"])["content"] == "Accepted"
    finally:
        foreign.close_all_connections()


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_ack_keeps_organization_readiness_gate(active, sync_service, method):
    path, headers, body, _ = _original(active, method)
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset.dataset_id,
            owner_user_id="user-1",
            domains=list(dataset.domains),
            metadata={**dataset.metadata, "notes_organization_v1": {"state": "failed"}},
        )
    )
    denied = getattr(active, method)(path, headers=headers, json=body)
    assert denied.status_code == 409, denied.text
    assert "notes_organization_sync_not_ready" in denied.text
    assert "Original evidence" not in denied.text


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_ack_keeps_read_policy_gate(active, sync_service, method):
    path, headers, body, _ = _original(active, method)
    sync_service.settings = replace(
        sync_service.settings,
        server_trusted_encryption=(
            server_trusted_encryption_status_from_config(
                mode=None, server_trusted_enabled=False, auth_mode="multi_user"
            )
        ),
    )
    denied = getattr(active, method)(path, headers=headers, json=body)
    assert denied.status_code == 409, denied.text
    assert "Original evidence" not in denied.text


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_ack_refuses_unapplied_child_at_group_boundary(
    active, sync_service, chacha_db, monkeypatch, method
):
    path, _, _, accepted = _original(active, method)
    sync_service.materializers["notes.provenance"] = NotesProvenanceMaterializer(chacha_db)

    def fail(*args, **kwargs):
        raise RuntimeError("injected child projection failure")

    monkeypatch.setattr(chacha_db.note_provenance_store, "apply_sync", fail)
    with pytest.raises(SyncServerOriginMaterializationError):
        capture_server_origin_mutation(
            sync_service,
            user_id="user-1",
            domain="notes.provenance",
            operation="upsert",
            object_id=accepted.json()["id"],
            parent_id=accepted.json()["id"],
            payload={"origin": "reviewed_sources"},
            source="bounded-projection-control",
        )
    denied = getattr(active, method)(
        path,
        headers={"expected-version": "2", "Idempotency-Key": "must-not-ack"},
        json={"content": "Pending child", "keywords": ["Pending"]},
    )
    assert denied.status_code == 503, denied.text
    assert denied.json()["detail"]["error_code"] == "sync_server_origin_batch_materialization_failed"
    assert "Original evidence" not in denied.text


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_ack_refuses_incompatible_parent_version(active, chacha_db, method):
    path, _, _, accepted = _original(active, method)
    chacha_db.update_note(accepted.json()["id"], {"content": "Local version moved"}, expected_version=2)
    denied = getattr(active, method)(
        path,
        headers={"expected-version": "3", "Idempotency-Key": "incompatible-parent"},
        json={"content": "Must not acknowledge mismatched version", "keywords": ["Version control"]},
    )
    assert denied.status_code == 409, denied.text
    assert "notes_provenance_projection_incomplete" in denied.text


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_ack_retains_deleted_child_after_restore(active, method):
    path, _, _, accepted = _original(active, method)
    assert active.delete(path, headers={"expected-version": "2"}).status_code == 204
    restored = active.post(path + "/restore?expected_version=3")
    assert restored.status_code == 200, restored.text
    headers = {"expected-version": "4", "Idempotency-Key": "retained-deleted-child"}
    body = {"content": "Accepted with deleted evidence", "keywords": ["Deleted evidence"]}
    original = getattr(active, method)(path, headers=headers, json=body)
    assert original.status_code == 200, original.text
    assert original.json()["knowledge_provenance_state"] == "deleted"
    assert original.json()["knowledge_provenance_version"] == 2
    later = active.post(
        path + "/provenance/restore",
        headers={"expected-version": "5"},
        json={
            "expected_provenance_version": 2,
            "expected_provenance_hash": original.json()["knowledge_provenance_hash"],
        },
    )
    assert later.status_code == 200, later.text
    assert later.json()["knowledge_provenance_state"] == "active"
    replay = getattr(active, method)(path, headers=headers, json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()


@pytest.mark.parametrize("method", ["put", "patch"])
def test_omitted_provenance_ack_refuses_unfinished_organization_group(active, sync_service, method):
    created = active.post(
        BASE + "/",
        json={
            "title": "Evidence",
            "content": "Body",
            "knowledge_provenance": PROVENANCE,
            "expected_provenance_version": 0,
        },
    )
    assert created.status_code == 201, created.text
    sync_service.materializers["notes.keyword"] = fixtures._FailingOrganizationMaterializer()
    path = BASE + "/" + created.json()["id"]
    headers = {"expected-version": "1", "Idempotency-Key": "unfinished-organization"}
    body = {"content": "No acknowledgment yet", "keywords": ["Failing projection"]}
    for _ in range(2):
        denied = getattr(active, method)(path, headers=headers, json=body)
        assert denied.status_code == 503, denied.text
        assert denied.json()["detail"]["error_code"] == "sync_server_origin_batch_materialization_failed"
        assert "Original evidence" not in denied.text
    group = fixtures._notes_api_group(sync_service, headers["Idempotency-Key"])
    assert any(member.apply_status == "failed" for member in group)


@pytest.mark.parametrize("format", ["json", "markdown"])
@pytest.mark.parametrize("keyed", [False, True])
def test_personal_provenance_import_and_replay_without_organization(
    active, sync_service, chacha_db, record_property, format, keyed
):
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset.dataset_id,
            owner_user_id="user-1",
            domains=["notes.note"],
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    content = (
        json.dumps(
            [
                {
                    "title": "Imported",
                    "content": "Body",
                    "knowledge_provenance": PROVENANCE,
                    "expected_provenance_version": 0,
                }
            ]
        )
        if format == "json"
        else "# Imported\n\n" + retain_notes_provenance("Body", PROVENANCE)
    )
    body = {"items": [{"format": format, "content": content}]}
    headers = {"Idempotency-Key": "personal-import"} if keyed else {}
    first = active.post(BASE + "/import", headers=headers, json=body)
    record_property("import_response", first.text)
    assert first.status_code == 200, first.text
    assert first.json()["created_count"] == 1
    assert first.json()["failed_count"] == 0
    note = chacha_db.list_notes(limit=10, offset=0)[0]
    saved = active.get(BASE + "/" + note["id"])
    assert saved.status_code == 200, saved.text
    assert saved.json()["knowledge_provenance"] == PROVENANCE
    if keyed:
        later = active.put(BASE + "/" + note["id"], headers={"expected-version": "1"}, json={"content": "Later"})
        assert later.status_code == 200, later.text
        replay = active.post(BASE + "/import", headers=headers, json=body)
        record_property("import_replay", replay.text)
        assert replay.status_code == 200, replay.text
        assert replay.json() == first.json()
        assert chacha_db.count_notes() == 1
        assert chacha_db.get_note_by_id(note["id"])["content"] == "Later"
    current = sync_service.store.get_dataset(dataset.dataset_id)
    assert not set(NOTES_ORGANIZATION_DOMAINS).issubset(current.domains)
    assert "notes_organization_v1" not in current.metadata


@pytest.mark.parametrize("keywords", [[], ["Requested"]])
def test_personal_import_still_requires_organization_when_keywords_supplied(active, sync_service, chacha_db, keywords):
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset.dataset_id,
            owner_user_id="user-1",
            domains=["notes.note"],
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    content = json.dumps(
        [
            {
                "title": "Blocked",
                "content": "Body",
                "knowledge_provenance": PROVENANCE,
                "keywords": keywords,
                "expected_provenance_version": 0,
            }
        ]
    )
    denied = active.post(BASE + "/import", json={"items": [{"format": "json", "content": content}]})
    assert denied.status_code == 409, denied.text
    assert "notes_organization_sync_domains_incomplete" in denied.text
    assert chacha_db.count_notes() == 0
