"""Real API source imports, portable versions and complete Sync capture."""

import json
import secrets
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Notes import (
    test_notes_organization_sync_api as organization_tests,
)
from tldw_Server_API.tests.Sync import (
    test_sync_v2_server_origin_capture as origin_tests,
)


@pytest.fixture
def client_with_notes_db(tmp_path, monkeypatch):
    db = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="user-1")
    yield origin_tests._notes_app(monkeypatch, chacha_db=db, sync_service=None)
    db.close_connection()


@pytest.fixture
def origin(tmp_path):
    db = organization_tests.chacha_db.__wrapped__(tmp_path)
    service = organization_tests.sync_service.__wrapped__(tmp_path, db)
    service.settings = replace(service.settings, pull_token_signing_secret=secrets.token_urlsafe(32))
    yield service, db
    db.close_connection()


SOURCE = '\ufeff  {"x":9007199254740993,"huge":1e400,"html":"<>&"}\r\n '


def portable(language="json", version="tldw.note.v1", **fields):
    return {
        "format_version": version,
        "title": "  Data  ",
        "content": SOURCE,
        "content_language": language,
        "keywords": ["data"],
        **fields,
    }


def import_body(record, format="portable_note"):
    return {
        "duplicate_strategy": "overwrite",
        "items": [{"file_name": "Data.json", "format": format, "content": json.dumps(record)}],
    }


@pytest.mark.parametrize("language", [None, "json", "yaml", " future/😀 "])
def test_portable_import_preserves_source_and_language(client_with_notes_db, language):
    client = client_with_notes_db
    response = client.post("/api/v1/notes/import", json=import_body(portable(language)))
    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 1
    note = client.get("/api/v1/notes/").json()["notes"][0]
    assert (note["title"], note["content"], note["content_language"]) == (
        "  Data  ",
        SOURCE,
        language,
    )


def test_source_import_uses_filename_without_parsing_document(client_with_notes_db):
    client = client_with_notes_db
    response = client.post(
        "/api/v1/notes/import",
        json={"items": [{"file_name": "Data.json", "format": "source", "content": SOURCE}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 1
    note = client.get("/api/v1/notes/").json()["notes"][0]
    assert (note["title"], note["content"], note["content_language"]) == (
        "Data",
        SOURCE,
        "json",
    )


def test_unknown_portable_version_refuses_entire_batch(client_with_notes_db):
    client = client_with_notes_db
    body = import_body(portable(version="tldw.note.v99"))
    body["items"].insert(
        0,
        {
            "file_name": "old.json",
            "format": "json",
            "content": json.dumps({"title": "First", "content": "legacy"}),
        },
    )
    response = client.post("/api/v1/notes/import", json=body)
    assert response.status_code == 422, response.text
    assert client.get("/api/v1/notes/").json()["notes"] == []


def test_legacy_overwrite_omission_preserves_language_and_explicit_null_resets(
    client_with_notes_db,
):
    client = client_with_notes_db
    note = client.post(
        "/api/v1/notes/",
        json={"title": "Old", "content": "body", "content_language": "yaml"},
    ).json()
    for row, language in [
        ({"id": note["id"], "title": "Old", "content": "legacy"}, "yaml"),
        (portable(None, id=note["id"]), None),
    ]:
        response = client.post(
            "/api/v1/notes/import",
            json=import_body(row, "json" if "format_version" not in row else "portable_note"),
        )
        assert response.status_code == 200, response.text
        assert response.json()["updated_count"] == 1
        assert client.get(f"/api/v1/notes/{note['id']}").json()["content_language"] == language


def test_empty_portable_source_is_rejected_without_title_substitution(
    client_with_notes_db,
):
    client = client_with_notes_db
    response = client.post("/api/v1/notes/import", json=import_body(portable(content="")))
    assert response.status_code == 422, response.text
    assert client.get("/api/v1/notes/").json()["notes"] == []


def test_active_import_binds_language_to_sync_request_and_replay(origin, monkeypatch):
    service, db = origin
    client = origin_tests._notes_app(monkeypatch, chacha_db=db, sync_service=service)
    body = import_body(portable(id="d5deeb2b-7a73-4fdf-921e-cfe0f20bbb9f"))
    headers = {"Idempotency-Key": "structured-import"}
    response = client.post("/api/v1/notes/import", json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 1
    dataset = service.profile(user_id="user-1").active_dataset_id
    head = service.store.get_current_head(dataset, "notes.note", "d5deeb2b-7a73-4fdf-921e-cfe0f20bbb9f")
    assert (head.payload["content"], head.payload["content_language"]) == (
        SOURCE,
        "json",
    )
    assert (head.adapter_version, head.schema_version) == (2, 2)
    repeated = client.post("/api/v1/notes/import", json=body, headers=headers)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["created_count"] == 1
    changed = client.post(
        "/api/v1/notes/import",
        json=import_body(portable("yaml", id="d5deeb2b-7a73-4fdf-921e-cfe0f20bbb9f")),
        headers=headers,
    )
    assert changed.status_code == 409, changed.text
    assert db.get_note_by_id("d5deeb2b-7a73-4fdf-921e-cfe0f20bbb9f")["content_language"] == "json"


def test_source_without_filename_keeps_text_without_inferred_mode(client_with_notes_db):
    client = client_with_notes_db
    response = client.post(
        "/api/v1/notes/import",
        json={"items": [{"format": "source", "content": SOURCE}]},
    )
    assert response.status_code == 200, response.text
    note = client.get("/api/v1/notes/").json()["notes"][0]
    assert (note["title"], note["content"], note["content_language"]) == (
        "Imported source",
        SOURCE,
        None,
    )


@pytest.mark.parametrize(
    "record",
    [
        {"format_version": "tldw.note.v1", "title": "Data", "content": SOURCE},
        portable(content={"x": 1}),
        portable(""),
    ],
)
def test_invalid_portable_record_refuses_before_any_batch_write(client_with_notes_db, record):
    client = client_with_notes_db
    body = import_body(record, "json")
    body["items"].insert(
        0,
        {
            "format": "json",
            "content": json.dumps({"title": "First", "content": "legacy"}),
        },
    )
    response = client.post("/api/v1/notes/import", json=body)
    assert response.status_code == 422
    assert client.get("/api/v1/notes/").json()["notes"] == []


@pytest.mark.parametrize(("name", "source"), [("data.csv", SOURCE), ("data.json", "")])
def test_rejected_raw_sources_do_not_substitute_a_body(client_with_notes_db, name, source):
    client = client_with_notes_db
    response = client.post(
        "/api/v1/notes/import",
        json={"items": [{"format": "source", "file_name": name, "content": source}]},
    )
    if not source:
        assert response.status_code == 422
    else:
        assert response.status_code == 200, response.text
        assert response.json()["created_count"] == 0
        assert response.json()["failed_count"] == 1
    assert client.get("/api/v1/notes/").json()["notes"] == []


def test_import_overwrite_rechecks_captured_version_before_sync_append(origin, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import notes as endpoint

    service, db = origin
    client = origin_tests._notes_app(monkeypatch, chacha_db=db, sync_service=service)
    created = client.post(
        "/api/v1/notes/",
        json={"title": "Before", "content": "old", "content_language": "yaml"},
    )
    assert created.status_code == 201, created.text
    note_id = created.json()["id"]
    dataset = service.profile(user_id="user-1").active_dataset_id
    before = len(service.store.list_envelopes_after(dataset, 0))
    original = endpoint.plan_compound_note

    def concurrent_change(*args, **kwargs):
        db.update_note(note_id, {"content_language": "future/mode"}, expected_version=1)
        return original(*args, **kwargs)

    monkeypatch.setattr(endpoint, "plan_compound_note", concurrent_change)
    response = client.post(
        "/api/v1/notes/import",
        json=import_body(portable(id=note_id)),
        headers={"Idempotency-Key": "race-import"},
    )
    assert response.status_code == 409, response.text
    assert db.get_note_by_id(note_id)["content_language"] == "future/mode"
    assert db.get_note_by_id(note_id)["content"] == "old"
    assert len(service.store.list_envelopes_after(dataset, 0)) == before
    monkeypatch.setattr(endpoint, "plan_compound_note", original)
    retried = client.post(
        "/api/v1/notes/import",
        json=import_body(portable(id=note_id)),
        headers={"Idempotency-Key": "race-import"},
    )
    assert retried.status_code == 200, retried.text
    assert retried.json()["updated_count"] == 1
    head = service.store.get_current_head(dataset, "notes.note", note_id)
    assert (head.payload["content"], head.payload["content_language"]) == (
        SOURCE,
        "json",
    )


@pytest.mark.parametrize("container", [[portable(), 3], {"notes": [portable(), 3]}, []])
def test_declared_portable_container_refuses_discarded_records_before_batch_write(client_with_notes_db, container):
    client = client_with_notes_db
    body = import_body(container)
    body["items"].insert(0, {"format": "json", "content": json.dumps({"title": "First", "content": "legacy"})})
    response = client.post("/api/v1/notes/import", json=body)
    assert response.status_code == 422, response.text
    assert client.get("/api/v1/notes/").json()["notes"] == []


@pytest.mark.parametrize("wrapper", ["notes", "data", "items", "results"])
@pytest.mark.parametrize("format", ["json", "portable_note"])
def test_root_envelope_is_the_validated_and_executed_carrier(client_with_notes_db, wrapper, format):
    client = client_with_notes_db
    inner = portable(version="tldw.note.v99", title="Inner", content="unsupported inner source")
    outer = portable("yaml", title="Outer", **{wrapper: [inner, inner]})
    body = import_body(outer, format)
    body["items"].insert(0, {"format": "json", "content": json.dumps({"title": "First", "content": "legacy"})})
    response = client.post("/api/v1/notes/import", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 2
    notes = client.get("/api/v1/notes/").json()["notes"]
    assert {(note["title"], note["content"], note["content_language"]) for note in notes} == {
        ("First", "legacy", None),
        ("Outer", SOURCE, "yaml"),
    }
