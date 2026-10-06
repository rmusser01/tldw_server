"""Real REST provenance boundaries with and without canonical Sync capture."""

import csv
import io
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes as endpoint
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import (
    notes_provenance_object_hash,
    read_notes_provenance,
    retain_notes_provenance,
    validate_notes_provenance_payload,
)
from tldw_Server_API.tests.Notes import test_notes_organization_sync_api as organization_fixtures

chacha_db = organization_fixtures.chacha_db
client = organization_fixtures.client
sync_service = organization_fixtures.sync_service

PAYLOAD = {
    "origin": "knowledge_qa",
    "question": "What happened?",
    "scope": {"include_media_ids": [999]},
    "sources": [
        {
            "title": "Retained source",
            "excerpt": "Saved evidence",
            "type": "text",
            "originalId": "foreign-note",
            "mediaId": None,
            "sourceType": None,
        }
    ],
}
BASE = "/api/v1/notes"


@pytest.fixture(params=[False, True], ids=["inactive", "active"])
def api(request, client, sync_service, monkeypatch):
    sync_service.adapters = default_sync_v2_registry()
    if not request.param:
        monkeypatch.setattr(endpoint, "get_active_server_origin_sync_service_for_user", lambda _: None)
    return client, request.param


def create(api, **fields):
    client, _ = api
    response = client.post(
        BASE + "/",
        json={
            "title": "Evidence",
            "content": "Body",
            "knowledge_provenance": PAYLOAD,
            "expected_provenance_version": 0,
            **fields,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_roundtrip_read_list_search_and_omission(api):
    client, _ = api
    saved = create(api, keywords=["Source"], folder_paths=["Research"])
    assert saved["knowledge_provenance_state"] == "active"
    assert saved["knowledge_provenance_version"] == 1
    assert validate_notes_provenance_payload(saved["knowledge_provenance"]) == PAYLOAD
    assert saved["knowledge_provenance_hash"] == notes_provenance_object_hash(PAYLOAD)
    note_id = saved["id"]
    for path in [f"/{note_id}", "/", "/search?query=Evidence"]:
        response = client.get(BASE + path)
        assert response.status_code == 200, response.text
        value = response.json()
        if "notes" in value:
            value = value["notes"][0]
        assert value["knowledge_provenance"] == PAYLOAD
    update = client.put(BASE + f"/{note_id}", headers={"expected-version": "1"}, json={"content": "Rewritten"})
    assert update.status_code == 200, update.text
    assert update.json()["knowledge_provenance_version"] == 1
    assert update.json()["knowledge_provenance"] == PAYLOAD


@pytest.mark.parametrize("method", ["put", "patch"])
def test_independent_stale_save_changes_neither_parent_nor_child(api, chacha_db, method):
    client, _ = api
    saved = create(api)
    path = BASE + "/" + saved["id"]
    first = getattr(client, method)(
        path,
        headers={"expected-version": "1"},
        json={
            "content": "Second",
            "knowledge_provenance": {"origin": "reviewed_sources"},
            "expected_provenance_version": 1,
        },
    )
    assert first.status_code == 200, first.text
    stale = getattr(client, method)(
        path,
        headers={"expected-version": "2"},
        json={"content": "Must not persist", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 1},
    )
    assert stale.status_code == 409, stale.text
    assert chacha_db.get_note_by_id(saved["id"])["content"] == "Second"
    assert client.get(path).json()["knowledge_provenance_version"] == 2


def test_delete_core_restore_and_explicit_child_restore(api):
    client, _ = api
    saved = create(api, content=retain_notes_provenance("Body", PAYLOAD))
    path = BASE + "/" + saved["id"]
    assert client.delete(path, headers={"expected-version": "1"}).status_code == 204
    trash = client.get(BASE + "/trash").json()["notes"][0]
    assert trash["knowledge_provenance_state"] == "deleted"
    assert trash["knowledge_provenance_version"] == 2
    assert trash["knowledge_provenance"] is None
    restored = client.post(path + "/restore?expected_version=2")
    assert restored.status_code == 200, restored.text
    assert restored.json()["knowledge_provenance_state"] == "deleted"
    export = client.post(BASE + "/export", json={"note_ids": [saved["id"]]}).json()["notes"][0]
    assert read_notes_provenance(export["content"]) is None
    reject = client.put(
        path,
        headers={"expected-version": "3"},
        json={"content": "No revival", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 2},
    )
    assert reject.status_code == 409, reject.text
    response = client.post(
        path + "/provenance/restore",
        headers={"expected-version": "3"},
        json={"expected_provenance_version": 2, "expected_provenance_hash": trash["knowledge_provenance_hash"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["knowledge_provenance"] == PAYLOAD
    assert response.json()["knowledge_provenance_version"] == 3
    assert response.json()["version"] == 4


@pytest.mark.parametrize(
    "method,suffix", [("get", "/export"), ("post", "/export"), ("get", "/export.csv"), ("post", "/export.csv")]
)
def test_portable_exports_use_canonical_history(api, method, suffix):
    client, _ = api
    saved = create(api, content=retain_notes_provenance("Body", {"origin": "reviewed_sources"}))
    response = getattr(client, method)(
        BASE + suffix, **({"json": {"note_ids": [saved["id"]]}} if method == "post" else {})
    )
    assert response.status_code == 200, response.text
    row = next(csv.DictReader(io.StringIO(response.text))) if suffix.endswith("csv") else response.json()["notes"][0]
    assert read_notes_provenance(row["content"]) == PAYLOAD
    read = client.get(BASE + "/" + saved["id"]).json()
    assert read["knowledge_provenance_reconciliation"] == "canonical_wins"


@pytest.mark.parametrize(
    "field,value",
    [
        ("knowledge_provenance", {"origin": "knowledge_qa", "scope": {"sources": None}}),
        ("knowledge_provenance", {"origin": "knowledge_qa", "token": "secret"}),
        ("expected_provenance_version", True),
    ],
)
def test_strict_request_validation(api, field, value):
    client, _ = api
    response = client.post(
        BASE + "/",
        json={
            "title": "Reject",
            "content": "Body",
            "knowledge_provenance": PAYLOAD,
            "expected_provenance_version": 0,
            field: value,
        },
    )
    assert response.status_code == 422, response.text


def test_bulk_and_import_preserve_portable_history(api):
    client, _ = api
    bulk = client.post(
        BASE + "/bulk",
        json={
            "notes": [
                {"title": "Bulk", "content": "Body", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0}
            ]
        },
    )
    assert bulk.status_code == 200, bulk.text
    assert bulk.json()["results"][0]["note"]["knowledge_provenance"] == PAYLOAD
    imported = client.post(
        BASE + "/import",
        json={"items": [{"format": "markdown", "content": retain_notes_provenance("# Imported\n\nText", PAYLOAD)}]},
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["created_count"] == 1
    notes = client.get(BASE + "/").json()["notes"]
    assert all(note["knowledge_provenance"] == PAYLOAD for note in notes)


def test_plain_and_historical_reads_are_absent_until_exact_save(api, chacha_db):
    client, active = api
    note_id = chacha_db.add_note("Historical", retain_notes_provenance("Body", PAYLOAD))
    before = client.get(BASE + "/" + note_id).json()
    assert before["knowledge_provenance_state"] == "absent"
    assert before["knowledge_provenance_version"] == 0
    response = client.put(
        BASE + "/" + note_id,
        headers={"expected-version": "1"},
        json={"content": "Saved", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0},
    )
    if active:
        assert response.status_code == 409, response.text
        response = client.put(
            BASE + "/" + note_id,
            headers={"expected-version": "1"},
            json={"content": "Saved", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 1},
        )
    assert response.status_code == 200, response.text
    assert response.json()["knowledge_provenance"] == PAYLOAD


def test_lost_ack_replays_before_stale_checks_or_title_generation(api, monkeypatch):
    client, _ = api
    calls = []

    def title(*args, **kwargs):
        calls.append(1)
        return "Generated " + str(len(calls))

    monkeypatch.setattr(endpoint, "_generate_note_title_with_service_prompt", title)
    body = {"content": "Body", "auto_title": True, "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0}
    headers = {"Idempotency-Key": "lost-create"}
    first = client.post(BASE + "/", json=body, headers=headers)
    assert first.status_code == 201, first.text
    path = BASE + "/" + first.json()["id"]
    later = client.put(path, headers={"expected-version": "1"}, json={"content": "Later"})
    assert later.status_code == 200, later.text
    retry = client.post(BASE + "/", json=body, headers=headers)
    assert retry.status_code == 201, retry.text
    assert retry.json() == first.json()
    assert len(calls) == 1
    conflict = client.post(BASE + "/", json={**body, "expected_provenance_version": 1}, headers=headers)
    assert conflict.status_code == 409, conflict.text


def test_stale_parent_and_missing_patch_base_are_rejected(api, chacha_db):
    client, _ = api
    saved = create(api)
    path = BASE + "/" + saved["id"]
    missing = client.patch(path, json={"knowledge_provenance": PAYLOAD, "expected_provenance_version": 1})
    assert missing.status_code == 400
    stale = client.put(
        path,
        headers={"expected-version": "99"},
        json={
            "content": "Bad",
            "knowledge_provenance": PAYLOAD,
            "expected_provenance_version": 1,
            "keywords": ["Must not exist"],
            "folder_paths": ["Must not exist"],
        },
    )
    assert stale.status_code == 409
    assert chacha_db.get_note_by_id(saved["id"])["version"] == 1
    assert chacha_db.get_keyword_by_text("Must not exist") is None
    assert client.get(path).json()["knowledge_provenance_version"] == 1


def test_retained_pointers_grant_no_access_to_foreign_or_deleted_notes(api, chacha_db):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    client, _ = api
    foreign = CharactersRAGDB(str(Path(chacha_db.db_path).with_name("foreign.db")), client_id="other-owner")
    foreign.add_note("Private", "Never expose this", note_id="foreign-note")
    local = chacha_db.add_note("Deleted", "Deleted source", note_id="deleted-source")
    chacha_db.soft_delete_note(local, 1)
    saved = create(api)
    assert saved["knowledge_provenance"]["sources"][0]["excerpt"] == "Saved evidence"
    assert client.get(BASE + "/foreign-note").status_code == 404
    assert client.get(BASE + "/deleted-source").status_code == 404
    assert foreign.get_note_by_id("foreign-note")["content"] == "Never expose this"
    foreign.close_all_connections()


def test_restore_requires_active_parent_and_exact_head(api):
    client, _ = api
    saved = create(api)
    path = BASE + "/" + saved["id"]
    assert client.delete(path, headers={"expected-version": "1"}).status_code == 204
    trash = client.get(BASE + "/trash").json()["notes"][0]
    body = {"expected_provenance_version": 2, "expected_provenance_hash": trash["knowledge_provenance_hash"]}
    assert client.post(path + "/provenance/restore", headers={"expected-version": "2"}, json=body).status_code == 404
    assert client.post(path + "/restore?expected_version=2").status_code == 200
    assert (
        client.post(
            path + "/provenance/restore",
            headers={"expected-version": "3"},
            json={**body, "knowledge_provenance": {"origin": "reviewed_sources"}},
        ).status_code
        == 422
    )
    assert (
        client.post(
            path + "/provenance/restore",
            headers={"expected-version": "3"},
            json={**body, "expected_provenance_hash": "sha256:" + "0" * 64},
        ).status_code
        == 409
    )
    assert client.get(path).json()["knowledge_provenance_state"] == "deleted"


def test_json_import_roundtrip_overwrite_preserves_omission_and_rejects_stale_child(api):
    client, _ = api
    saved = create(api)
    exported = client.post(BASE + "/export", json={"note_ids": [saved["id"]]}).json()
    copy = client.post(BASE + "/import", json={"items": [{"format": "json", "content": json.dumps(exported)}]})
    assert copy.status_code == 200, copy.text
    assert copy.json()["created_count"] == 1
    rows = client.get(BASE + "/").json()["notes"]
    assert len(rows) == 2
    assert all(row["knowledge_provenance"] == PAYLOAD for row in rows)
    omitted = {"id": saved["id"], "title": "Import overwrite", "content": "New text"}
    response = client.post(
        BASE + "/import",
        json={"duplicate_strategy": "overwrite", "items": [{"format": "json", "content": json.dumps(omitted)}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["updated_count"] == 1
    path = BASE + "/" + saved["id"]
    assert client.get(path).json()["knowledge_provenance_version"] == 1
    stale = {**omitted, "content": "Do not write", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0}
    response = client.post(
        BASE + "/import",
        json={"duplicate_strategy": "overwrite", "items": [{"format": "json", "content": json.dumps(stale)}]},
    )
    assert response.status_code in {200, 409}, response.text
    if response.status_code == 200:
        assert response.json()["failed_count"] == 1
    assert client.get(path).json()["content"] == "New text"


@pytest.mark.parametrize("method", ["put", "patch"])
def test_lost_update_ack_returns_acknowledged_pair_after_later_child_edit(api, method):
    client, _ = api
    saved = create(api)
    path = BASE + "/" + saved["id"]
    headers = {"expected-version": "1", "Idempotency-Key": "update-lost"}
    body = {
        "content": "Accepted",
        "knowledge_provenance": {"origin": "reviewed_sources"},
        "expected_provenance_version": 1,
        "keywords": ["First"],
        "folder_paths": ["First"],
    }
    accepted = getattr(client, method)(path, headers=headers, json=body)
    assert accepted.status_code == 200, accepted.text
    later = client.put(
        path,
        headers={"expected-version": "2"},
        json={
            "content": "Later",
            "knowledge_provenance": PAYLOAD,
            "expected_provenance_version": 2,
            "keywords": ["Later"],
            "folder_paths": ["Later"],
        },
    )
    assert later.status_code == 200, later.text
    replay = getattr(client, method)(path, headers=headers, json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json() == accepted.json()
    assert client.get(path).json()["content"] == "Later"


def test_plain_provenance_save_without_organization_profile(client, sync_service):
    from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate

    sync_service.adapters = default_sync_v2_registry()
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset.dataset_id,
            owner_user_id="user-1",
            domains=["notes.note"],
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    response = client.post(
        BASE + "/",
        headers={"Idempotency-Key": "plain-only"},
        json={"title": "Plain", "content": "Body", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0},
    )
    assert response.status_code == 201, response.text
    replay = client.post(
        BASE + "/",
        headers={"Idempotency-Key": "plain-only"},
        json={"title": "Plain", "content": "Body", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0},
    )
    assert replay.status_code == 201, replay.text
    assert replay.json() == response.json()


def test_unavailable_encryption_never_falls_back_to_local(client, sync_service, chacha_db, monkeypatch):
    from dataclasses import replace

    sync_service.adapters = default_sync_v2_registry()
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    monkeypatch.setattr(
        sync_service.store, "list_datasets_for_user", lambda _: [replace(dataset, encryption_policy="client_e2ee_v1")]
    )
    response = client.post(
        BASE + "/",
        json={"title": "Reject", "content": "Body", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0},
    )
    assert response.status_code == 409, response.text
    assert chacha_db.count_notes() == 0


def test_capability_absent_is_distinct_from_no_history(api, chacha_db, monkeypatch):
    client, _ = api
    note_id = chacha_db.add_note("Plain", "Body")
    monkeypatch.setattr(chacha_db, "note_provenance_store", None)
    response = client.get(BASE + "/" + note_id)
    assert response.status_code == 200, response.text
    assert response.json()["knowledge_provenance_state"] == "unsupported"
    assert response.json()["knowledge_provenance_version"] is None


def test_marker_only_tombstone_exports_empty_content_without_reviving_history(api):
    client, _ = api
    saved = create(api, content=retain_notes_provenance("", PAYLOAD))
    path = BASE + "/" + saved["id"]
    assert client.delete(path, headers={"expected-version": "1"}).status_code == 204
    assert client.post(path + "/restore?expected_version=2").status_code == 200
    exported = client.post(BASE + "/export", json={"note_ids": [saved["id"]]})
    assert exported.status_code == 200, exported.text
    assert exported.json()["notes"][0]["content"] == ""
    assert exported.json()["notes"][0]["knowledge_provenance_state"] == "deleted"


def test_disabled_encryption_attestation_rejects_before_marker_backfill(client, sync_service, chacha_db):
    from dataclasses import replace

    from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config

    sync_service.adapters = default_sync_v2_registry()
    note_id = chacha_db.add_note("Historical", retain_notes_provenance("Body", PAYLOAD))
    sync_service.settings = replace(
        sync_service.settings,
        server_trusted_encryption=server_trusted_encryption_status_from_config(
            mode=None, server_trusted_enabled=False, auth_mode="multi_user"
        ),
    )
    response = client.put(
        BASE + "/" + note_id,
        headers={"expected-version": "1"},
        json={"content": "Must not save", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0},
    )
    assert response.status_code == 409, response.text
    assert chacha_db.get_note_by_id(note_id)["version"] == 1
    assert chacha_db.note_provenance_store.get(note_id, include_deleted=True) is None


def test_ready_profile_enrolls_existing_plain_target_before_organization_pair(client, sync_service, chacha_db):
    sync_service.adapters = default_sync_v2_registry()
    create((client, True))
    note_id = chacha_db.add_note("Unsynced", "First")
    chacha_db.update_note(note_id, {"content": "Second"}, 1)
    chacha_db.update_note(note_id, {"content": "Third"}, 2)
    response = client.put(
        BASE + "/" + note_id,
        headers={"expected-version": "3"},
        json={
            "content": "Fourth",
            "knowledge_provenance": PAYLOAD,
            "expected_provenance_version": 0,
            "keywords": ["Target"],
            "folder_paths": ["Target"],
        },
    )
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["version"] == 4
    assert saved["knowledge_provenance_version"] == 1
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    core = sync_service.store.get_current_head(dataset.dataset_id, "notes.note", note_id)
    child = sync_service.store.get_current_head(dataset.dataset_id, "notes.provenance", note_id)
    assert core.object_revision == 4
    assert core.base_object_revision == 3
    assert child.mutation_group_id == core.mutation_group_id
    assert child.mutation_step == core.mutation_step + 1
    assert "knowledge_provenance" not in core.payload


@pytest.mark.parametrize("policy", ["withdrawn", "unsupported"])
@pytest.mark.parametrize("state", ["active", "deleted", "marker"])
def test_read_and_exports_fail_closed_after_encryption_policy_change(
    client, sync_service, chacha_db, monkeypatch, policy, state
):
    from dataclasses import replace

    from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config

    sync_service.adapters = default_sync_v2_registry()
    if state == "marker":
        note_id = chacha_db.add_note("Historical", retain_notes_provenance("Body", PAYLOAD))
    else:
        saved = create((client, True), content=retain_notes_provenance("Body", PAYLOAD))
        note_id = saved["id"]
        if state == "deleted":
            assert client.delete(BASE + "/" + note_id, headers={"expected-version": "1"}).status_code == 204
            assert client.post(BASE + "/" + note_id + "/restore?expected_version=2").status_code == 200
    if policy == "withdrawn":
        sync_service.settings = replace(
            sync_service.settings,
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode=None, server_trusted_enabled=False, auth_mode="multi_user"
            ),
        )
    else:
        dataset = sync_service.store.list_datasets_for_user("user-1")[0]
        monkeypatch.setattr(
            sync_service.store,
            "list_datasets_for_user",
            lambda _: [replace(dataset, encryption_policy="client_e2ee_v1")],
        )
    for method, suffix in [
        ("get", "/" + note_id),
        ("get", "/"),
        ("get", "/export"),
        ("post", "/export"),
        ("get", "/export.csv"),
        ("post", "/export.csv"),
    ]:
        response = getattr(client, method)(
            BASE + suffix, **({"json": {"note_ids": [note_id]}} if method == "post" else {})
        )
        assert response.status_code == 409, response.text
        assert "Saved evidence" not in response.text
        assert "notes_provenance_encryption_unsupported" in response.text


def test_bulk_lost_ack_is_immutable_and_rejects_changed_input(api, monkeypatch):
    client, _ = api
    calls = []

    def title(*args, **kwargs):
        calls.append(1)
        return "Bulk generated " + str(len(calls))

    monkeypatch.setattr(endpoint, "_generate_note_title_with_service_prompt", title)
    item = {
        "content": "Body",
        "auto_title": True,
        "knowledge_provenance": PAYLOAD,
        "expected_provenance_version": 0,
        "keywords": ["Original"],
        "folder_paths": ["Original"],
    }
    body = {"notes": [item, {**item, "content": "Second"}]}
    headers = {"Idempotency-Key": "bulk-lost"}
    first = client.post(BASE + "/bulk", json=body, headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["created_count"] == 2, first.text
    notes = client.get(BASE + "/").json()["notes"]
    for note in notes:
        assert (
            client.put(
                BASE + "/" + note["id"],
                headers={"expected-version": "1"},
                json={"content": "Later", "keywords": ["Later"]},
            ).status_code
            == 200
        )
    replay = client.post(BASE + "/bulk", json=body, headers=headers)
    assert replay.json() == first.json(), replay.text
    assert len(calls) == 2
    assert len(client.get(BASE + "/").json()["notes"]) == 2
    changed = client.post(
        BASE + "/bulk", json={"notes": [{**item, "content": "Altered"}, body["notes"][1]]}, headers=headers
    )
    assert changed.status_code == 207, changed.text
    assert changed.json()["failed_count"] == 1
    assert len(calls) == 2


def test_inactive_receipt_rejects_retry_with_provenance_removed(client, monkeypatch):
    monkeypatch.setattr(endpoint, "get_active_server_origin_sync_service_for_user", lambda _: None)
    body = {"title": "First", "content": "Body", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0}
    headers = {"Idempotency-Key": "remove-provenance"}
    saved = client.post(BASE + "/", json=body, headers=headers)
    assert saved.status_code == 201, saved.text
    changed = client.post(BASE + "/", json={"title": "First", "content": "Body"}, headers=headers)
    assert changed.status_code == 409, changed.text
    assert len(client.get(BASE + "/").json()["notes"]) == 1


def test_inactive_rest_receipt_rolls_back_organization_then_replays_after_reopen(client, chacha_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

    monkeypatch.setattr(endpoint, "get_active_server_origin_sync_service_for_user", lambda _: None)
    store = chacha_db.note_provenance_store
    original = store.complete_receipt

    def fail(*args, **kwargs):
        raise CharactersRAGDBError("Injected acknowledgment failure")

    monkeypatch.setattr(store, "complete_receipt", fail)
    body = {
        "title": "First",
        "content": "Body",
        "knowledge_provenance": PAYLOAD,
        "expected_provenance_version": 0,
        "keywords": ["Atomic"],
        "folder_paths": ["Atomic"],
    }
    headers = {"Idempotency-Key": "rollback-reopen"}
    assert client.post(BASE + "/", json=body, headers=headers).status_code == 500
    assert chacha_db.count_notes() == 0
    assert chacha_db.get_keyword_by_text("Atomic") is None
    with chacha_db.transaction() as conn:
        assert conn.execute("SELECT * FROM notes_provenance_receipts").fetchall() == []
        assert conn.execute("SELECT * FROM notes_knowledge_provenance").fetchall() == []
        assert conn.execute("SELECT * FROM note_folders").fetchall() == []
    monkeypatch.setattr(store, "complete_receipt", original)
    first = client.post(BASE + "/", json=body, headers=headers)
    assert first.status_code == 201, first.text
    chacha_db.close_all_connections()
    reopened = CharactersRAGDB(chacha_db.db_path, client_id="user-1")
    client.app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: reopened
    try:
        assert client.post(BASE + "/", json=body, headers=headers).json() == first.json()
    finally:
        reopened.close_all_connections()


@pytest.mark.parametrize("policy", ["withdrawn", "unsupported"])
def test_acknowledgment_replay_cannot_bypass_read_policy(client, sync_service, monkeypatch, policy):
    from dataclasses import replace

    from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config

    sync_service.adapters = default_sync_v2_registry()
    body = {
        "title": "First",
        "content": retain_notes_provenance("Body", PAYLOAD),
        "knowledge_provenance": PAYLOAD,
        "expected_provenance_version": 0,
    }
    headers = {"Idempotency-Key": "policy-replay"}
    assert client.post(BASE + "/", json=body, headers=headers).status_code == 201
    if policy == "withdrawn":
        sync_service.settings = replace(
            sync_service.settings,
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode=None, server_trusted_enabled=False, auth_mode="multi_user"
            ),
        )
    else:
        dataset = sync_service.store.list_datasets_for_user("user-1")[0]
        monkeypatch.setattr(
            sync_service.store,
            "list_datasets_for_user",
            lambda _: [replace(dataset, encryption_policy="client_e2ee_v1")],
        )
    denied = client.post(BASE + "/", json=body, headers=headers)
    assert denied.status_code == 409, denied.text
    assert "Saved evidence" not in denied.text


@pytest.mark.parametrize("strategy", ["create_copy", "overwrite"])
def test_inactive_import_lost_ack_replays_before_duplicate_or_child_checks(client, monkeypatch, strategy):
    monkeypatch.setattr(endpoint, "get_active_server_origin_sync_service_for_user", lambda _: None)
    row = {
        "title": "Imported",
        "content": "Body",
        "knowledge_provenance": PAYLOAD,
        "expected_provenance_version": 0,
        "keywords": ["Imported"],
    }
    if strategy == "overwrite":
        saved = create((client, False))
        row.update(id=saved["id"], expected_provenance_version=1)
    body = {"duplicate_strategy": strategy, "items": [{"format": "json", "content": json.dumps(row)}]}
    headers = {"Idempotency-Key": "import-lost"}
    first = client.post(BASE + "/import", json=body, headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["failed_count"] == 0
    notes = client.get(BASE + "/").json()["notes"]
    for note in notes:
        assert (
            client.put(
                BASE + "/" + note["id"], json={"content": "Later"}, headers={"expected-version": str(note["version"])}
            ).status_code
            == 200
        )
    replay = client.post(BASE + "/import", json=body, headers=headers)
    assert replay.json() == first.json(), replay.text
    assert len(client.get(BASE + "/").json()["notes"]) == 1
    altered = {**body, "items": [{"format": "json", "content": json.dumps({**row, "content": "Changed"})}]}
    rejected = client.post(BASE + "/import", json=altered, headers=headers)
    assert rejected.json()["failed_count"] == 1, rejected.text
    assert len(client.get(BASE + "/").json()["notes"]) == 1


def test_lost_retained_restore_ack_uses_original_parent_child_pair(api):
    client, _ = api
    note = create(api)
    path = BASE + "/" + note["id"]
    assert client.delete(path, headers={"expected-version": "1"}).status_code == 204
    trash = client.get(BASE + "/trash").json()["notes"][0]
    assert client.post(path + "/restore?expected_version=2").status_code == 200
    body = {"expected_provenance_version": 2, "expected_provenance_hash": trash["knowledge_provenance_hash"]}
    headers = {"Idempotency-Key": "restore-lost", "expected-version": "3"}
    first = client.post(path + "/provenance/restore", json=body, headers=headers)
    assert first.status_code == 200, first.text
    assert client.put(path, json={"content": "Later"}, headers={"expected-version": "4"}).status_code == 200
    assert client.post(path + "/provenance/restore", json=body, headers=headers).json() == first.json()


def test_inactive_owner_receipts_and_created_ids_are_independent(client, chacha_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    monkeypatch.setattr(endpoint, "get_active_server_origin_sync_service_for_user", lambda _: None)
    body = {"title": "Owner", "content": "Body", "knowledge_provenance": PAYLOAD, "expected_provenance_version": 0}
    headers = {"Idempotency-Key": "same-key-for-each-owner"}
    first = client.post(BASE + "/", json=body, headers=headers)
    assert first.status_code == 201, first.text
    foreign = CharactersRAGDB(str(Path(chacha_db.db_path).with_name("receipt-owner.db")), client_id="other-owner")
    client.app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: foreign
    client.app.dependency_overrides[endpoint.get_request_user] = lambda: endpoint.User(
        id="other-owner", username="other-owner", is_active=True, is_admin=True
    )
    try:
        other = client.post(BASE + "/", json={**body, "title": "Other"}, headers=headers)
        assert other.status_code == 201, other.text
        assert other.json()["id"] != first.json()["id"]
        assert other.json()["title"] == "Other"
        assert client.post(BASE + "/", json={**body, "title": "Other"}, headers=headers).json() == other.json()
    finally:
        foreign.close_all_connections()
