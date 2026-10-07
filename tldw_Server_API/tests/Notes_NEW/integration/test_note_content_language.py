"""Canonical note language persists independently of exact source and versions."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_Server_API.tests.Notes_NEW.integration import test_notes_api


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="test")
    yield database
    database.close_connection()


@pytest.mark.parametrize("language", [None, "text", "markdown", "yaml", "json", "jsonl", "future/mode"])
def test_language_survives_create_read_list_and_copy(db, language):
    source = "\ufeffkey: 1\r\n# exact source  \r\n"
    note_id = db.add_note(title="Config", content=source, content_language=language)
    note = db.get_note_by_id(note_id)
    assert note["content"] == source
    assert note["content_language"] == language
    assert db.list_notes()[0]["content_language"] == language
    copy_id = db.add_note(title=note["title"], content=note["content"], content_language=note["content_language"])
    assert db.get_note_by_id(copy_id)["content_language"] == language


def test_body_update_preserves_language_and_language_only_change_versions(db):
    note_id = db.add_note(title="Config", content="a: 1", content_language="yaml")
    db.update_note(note_id, {"content": "a: 2"}, expected_version=1)
    assert db.get_note_by_id(note_id)["content_language"] == "yaml"
    db.update_note(note_id, {"content_language": "future/mode"}, expected_version=2)
    note = db.get_note_by_id(note_id)
    assert (note["version"], note["content"], note["content_language"]) == (3, "a: 2", "future/mode")
    with pytest.raises(ConflictError):
        db.update_note(note_id, {"content_language": None}, expected_version=2)
    assert db.get_note_by_id(note_id)["content_language"] == "future/mode"
    db.update_note(note_id, {"content_language": None}, expected_version=3)
    assert db.get_note_by_id(note_id)["content_language"] is None
    assert db.get_note_by_id(note_id)["version"] == 4


@pytest.mark.parametrize("language", ["", "x" * 101, 12, False, []])
def test_language_validation_rejects_invalid_metadata_before_write(db, language):
    with pytest.raises(InputError):
        db.add_note(title="Config", content="source", content_language=language)
    note_id = db.add_note(title="Config", content="source")
    with pytest.raises(InputError):
        db.update_note(note_id, {"content_language": language}, expected_version=1)
    assert db.get_note_by_id(note_id)["version"] == 1


def test_exact_title_is_preserved_on_create_and_update(db):
    note_id = db.add_note(title="  Config  ", content="source")
    assert db.get_note_by_id(note_id)["title"] == "  Config  "
    db.update_note(note_id, {"title": "  Next  "}, expected_version=1)
    assert db.get_note_by_id(note_id)["title"] == "  Next  "


@pytest.mark.parametrize("title", [" ", "x" * 256])
def test_title_boundary_rejects_without_rewriting(db, title):
    with pytest.raises(InputError):
        db.add_note(title=title, content="source")
    note_id = db.add_note(title="Valid", content="source")
    with pytest.raises(InputError):
        db.update_note(note_id, {"title": title}, expected_version=1)
    assert db.get_note_by_id(note_id)["version"] == 1


client_with_notes_db = test_notes_api.client_with_notes_db


def test_api_language_roundtrip_omission_null_conflict_and_copy(client_with_notes_db):
    client = client_with_notes_db
    capabilities = client.get("/api/v1/notes/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["content_language"] is True
    payload = {"title": "  Config  ", "content": "a: 1\r\n", "content_language": "yaml"}
    created = client.post("/api/v1/notes/", json=payload)
    assert created.status_code == 201, created.text
    note = created.json()
    assert {key: note[key] for key in payload} == payload
    note_id = note["id"]
    assert client.get(f"/api/v1/notes/{note_id}").json()["content_language"] == "yaml"
    assert client.get("/api/v1/notes/").json()["notes"][0]["content_language"] == "yaml"
    copy = client.post("/api/v1/notes/", json={key: note[key] for key in payload})
    assert copy.status_code == 201
    assert copy.json()["content_language"] == "yaml"
    changed = client.patch(f"/api/v1/notes/{note_id}", json={"content": "a: 2\r\n"}, headers={"expected-version": "1"})
    assert changed.status_code == 200, changed.text
    assert changed.json()["content_language"] == "yaml"
    changed = client.patch(
        f"/api/v1/notes/{note_id}", json={"content_language": "future/mode"}, headers={"expected-version": "2"}
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["version"] == 3
    stale = client.patch(f"/api/v1/notes/{note_id}", json={"content_language": None}, headers={"expected-version": "2"})
    assert stale.status_code == 409
    cleared = client.patch(
        f"/api/v1/notes/{note_id}", json={"content_language": None}, headers={"expected-version": "3"}
    )
    assert cleared.status_code == 200
    assert cleared.json()["content_language"] is None
    assert (
        client.post("/api/v1/notes/", json={"title": "Empty", "content": "", "content_language": "jsonl"}).status_code
        == 422
    )


def test_active_sync_rejects_language_and_language_bearing_body_before_storage(client_with_notes_db, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import notes

    client = client_with_notes_db
    created = client.post("/api/v1/notes/", json={"title": "Config", "content": "a: 1", "content_language": "yaml"})
    assert created.status_code == 201
    before = created.json()
    monkeypatch.setattr(notes, "_active_notes_sync_service", lambda user: object())
    assert client.get("/api/v1/notes/capabilities").json()["content_language"] is False
    for payload in (
        {"content_language": None},
        {"content_language": "json"},
        {"content": "a: 2"},
        {"title": "Changed"},
    ):
        response = client.patch(f"/api/v1/notes/{before['id']}", json=payload, headers={"expected-version": "1"})
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error_code"] == "notes_content_language_sync_unsupported"
    assert client.get(f"/api/v1/notes/{before['id']}").json() == before
    response = client.post("/api/v1/notes/", json={"title": "Config", "content": "source", "content_language": None})
    assert response.status_code == 409


def test_current_backup_restores_language_and_source(db, tmp_path):
    note_id = db.add_note(title="  Config  ", content="a: 1\r\n", content_language="future/mode")
    backup = tmp_path / "current-backup.db"
    assert db.backup_database(str(backup))
    restored = CharactersRAGDB(str(backup), client_id="test")
    try:
        assert restored.get_note_by_id(note_id) == db.get_note_by_id(note_id)
    finally:
        restored.close_connection()


def test_language_migration_and_old_backup_preserve_source(tmp_path, monkeypatch):
    path = tmp_path / "historical.db"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 78)
        old = CharactersRAGDB(str(path), client_id="test")
        try:
            conn = old.get_connection()
            assert "content_language" not in {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
            conn.execute(
                "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, ?, ?)",
                ("old", "  Config  ", "a: 1\r\n", "test"),
            )
            conn.commit()
            backup = tmp_path / "old-backup.db"
            assert old.backup_database(str(backup))
        finally:
            old.close_connection()
    for candidate in (path, backup):
        upgraded = CharactersRAGDB(str(candidate), client_id="test")
        try:
            note = upgraded.get_note_by_id("old")
            assert note["content_language"] is None
            assert (note["title"], note["content"], note["version"]) == ("  Config  ", "a: 1\r\n", 1)
            assert upgraded._get_db_version(upgraded.get_connection()) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        finally:
            upgraded.close_connection()


def test_exact_version_snapshot_contains_language(db):
    import json

    note_id = db.add_note(title="  Config  ", content="a: 1\r\n", content_language="yaml")
    for version, language in [(1, "yaml"), (2, "json")]:
        if version == 2:
            db.update_note(note_id, {"content_language": language}, expected_version=1)
        row = db.execute_query(
            "SELECT version, payload FROM sync_log WHERE entity = 'notes' AND entity_id = ? ORDER BY version DESC LIMIT 1",
            (note_id,),
        ).fetchone()
        assert (row["version"], json.loads(row["payload"])["content_language"]) == (version, language)
        assert json.loads(row["payload"])["content"] == "a: 1\r\n"


@pytest.fixture
def pg_db(pg_database_config, tmp_path):
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    database = CharactersRAGDB(str(tmp_path / "pg-notes.db"), client_id="test", backend=backend)
    try:
        yield database
    finally:
        database.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_storage_uses_same_language_contract(pg_db):
    test_language_survives_create_read_list_and_copy(pg_db, "future/mode")
    test_body_update_preserves_language_and_language_only_change_versions(pg_db)
    test_exact_title_is_preserved_on_create_and_update(pg_db)


def test_postgres_real_absent_column_upgrade(pg_db, tmp_path):
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

    note_id = pg_db.add_note(title="  Existing  ", content="\ufeffa: 1\r\n", content_language="yaml")
    config = pg_db.backend.config
    with pg_db.backend.transaction() as conn:
        pg_db.backend.execute("ALTER TABLE notes DROP COLUMN content_language", connection=conn)
        pg_db._set_schema_version_postgres(conn, 82)
        assert "content_language" not in {row["name"] for row in pg_db.backend.get_table_info("notes", connection=conn)}
    pg_db.close_all_connections()
    backend = DatabaseBackendFactory.create_backend(config)
    upgraded = CharactersRAGDB(str(tmp_path / "pg-upgraded.db"), client_id="test", backend=backend)
    try:
        assert upgraded._runtime_schema_version == 83
        note = upgraded.get_note_by_id(note_id)
        assert (note["title"], note["content"], note["version"], note["content_language"]) == (
            "  Existing  ",
            "\ufeffa: 1\r\n",
            1,
            None,
        )
        test_body_update_preserves_language_and_language_only_change_versions(upgraded)
    finally:
        upgraded.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_real_route_serialization(pg_db, client_with_notes_db):
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    client_with_notes_db.app.dependency_overrides[get_chacha_db_for_user] = lambda: pg_db
    test_api_language_roundtrip_omission_null_conflict_and_copy(client_with_notes_db)


def test_legacy_sync_projection_rejects_language_bearing_snapshot(db):
    note_id = db.add_note(title="Config", content="source", content_language="yaml")
    before = db.get_note_by_id(note_id)
    with pytest.raises(InputError, match="v1 Sync"):
        db.upsert_note_from_sync(
            note_id=note_id,
            title="Changed",
            content="source",
            conversation_id=None,
            message_id=None,
            sync_client_id="test",
            object_revision=2,
            object_hash="unused",
            expected_product_version=1,
        )
    assert db.get_note_by_id(note_id) == before
    ordinary = db.add_note(title="Ordinary", content="source")
    assert db.upsert_note_from_sync(
        note_id=ordinary,
        title="  Exact  ",
        content="source",
        conversation_id=None,
        message_id=None,
        sync_client_id="test",
        object_revision=2,
        object_hash="unused",
        expected_product_version=1,
    )
    assert db.get_note_by_id(ordinary)["title"] == "  Exact  "


def test_language_bearing_restore_is_rejected_before_mutation(client_with_notes_db, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import notes

    client = client_with_notes_db
    note = client.post(
        "/api/v1/notes/", json={"title": "Config", "content": "source", "content_language": "yaml"}
    ).json()
    assert client.delete(f"/api/v1/notes/{note['id']}", headers={"expected-version": "1"}).status_code == 204
    monkeypatch.setattr(notes, "_active_notes_sync_service", lambda user: object())
    restored = client.post(f"/api/v1/notes/{note['id']}/restore", params={"expected_version": 2})
    assert restored.status_code == 409, restored.text
    assert restored.json()["detail"]["error_code"] == "notes_content_language_sync_unsupported"
    assert client.get(f"/api/v1/notes/{note['id']}").status_code == 404


def test_capability_route_requires_authentication(client_with_notes_db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    client = client_with_notes_db
    override = client.app.dependency_overrides.pop(get_request_user)
    try:
        response = client.get("/api/v1/notes/capabilities", headers={"X-API-KEY": "invalid"})
        assert response.status_code in (401, 403)
    finally:
        client.app.dependency_overrides[get_request_user] = override


def test_interop_service_rejects_incomplete_active_sync_writes(db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Notes import Notes_Library

    service = Notes_Library.NotesInteropService(tmp_path, "test")
    monkeypatch.setattr(service, "_get_db", lambda user: db)
    monkeypatch.setattr(Notes_Library, "active_coordinator", lambda *args, **kwargs: object())
    with pytest.raises(InputError, match="cannot publish"):
        service.add_note("test", "Config", "source", content_language="yaml")
    with pytest.raises(InputError, match="cannot publish"):
        service.add_note("test", "Config", "source", content_language=None)
    note_id = db.add_note(title="Config", content="source", content_language="yaml")
    with pytest.raises(InputError, match="cannot publish"):
        service.update_note("test", note_id, {"title": "Changed"}, 1)
    with pytest.raises(InputError, match="cannot publish"):
        service.update_note(user_id="test", note_id=note_id, content_language=None, expected_version=1)
    assert db.get_note_by_id(note_id)["version"] == 1


@pytest.mark.asyncio
async def test_capability_does_not_claim_unmigrated_storage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.api.v1.endpoints import notes

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 78)
    old = CharactersRAGDB(str(tmp_path / "old-capability.db"), client_id="test")
    monkeypatch.setattr(notes, "_active_notes_sync_service", lambda user: None)
    try:
        assert await notes.notes_capabilities(db=old, current_user=SimpleNamespace(id=1)) == {"content_language": False}
    finally:
        old.close_connection()
