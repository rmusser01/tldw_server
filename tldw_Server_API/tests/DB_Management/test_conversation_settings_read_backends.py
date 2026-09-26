"""UAT213: real settings rows survive the backend result contract."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as chats
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def settings_db(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "settings.db", client_id="2", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _conversation(db):
    return db.add_conversation({
        "title": "Settings read control", "assistant_kind": "persona", "assistant_id": "fixture-persona"
    })


def test_missing_settings_return_none(settings_db):
    conversation_id = _conversation(settings_db)
    assert settings_db.get_conversation_settings(conversation_id) is None


@pytest.mark.parametrize("settings", [{}, {"temperature": 0.2, "memory": {"enabled": True}}])
def test_settings_roundtrip_preserves_document_version_and_timestamp(settings_db, settings):
    conversation_id = _conversation(settings_db)
    assert settings_db.upsert_conversation_settings(conversation_id, settings)
    row = settings_db.get_conversation_settings(conversation_id)
    assert row is not None
    assert row["settings"] == settings
    assert row["settings_version"] == 1
    assert row["last_modified"] is not None
    assert settings_db.upsert_conversation_settings(conversation_id, {"temperature": 0.7})
    updated = settings_db.get_conversation_settings(conversation_id)
    assert updated["settings"] == {"temperature": 0.7}
    assert updated["settings_version"] == 2


def test_invalid_settings_json_keeps_existing_none_fallback(settings_db):
    conversation_id = _conversation(settings_db)
    assert settings_db.upsert_conversation_settings(conversation_id, {})
    with settings_db.transaction() as conn:
        conn.execute("UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?",
                     ("invalid json", conversation_id))
    assert settings_db.get_conversation_settings(conversation_id) is None


def test_settings_read_does_not_commit_callers_pending_write(settings_db):
    conversation_id = _conversation(settings_db)
    assert settings_db.upsert_conversation_settings(conversation_id, {"temperature": 0.2})
    with pytest.raises(RuntimeError, match="caller rollback"):
        with settings_db.transaction() as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", ("Pending title", conversation_id))
            assert settings_db.get_conversation_settings(conversation_id)["settings"] == {"temperature": 0.2}
            raise RuntimeError("caller rollback")
    assert settings_db.get_conversation_by_id(conversation_id)["title"] == "Settings read control"


def test_actual_chat_settings_route_returns_persisted_public_settings(settings_db):
    conversation_id = _conversation(settings_db)
    assert settings_db.upsert_conversation_settings(conversation_id, {"temperature": 0.2, "greetingScope": "chat"})
    app = FastAPI()
    app.include_router(chats.router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: settings_db
    app.dependency_overrides[chats.get_request_user] = lambda: SimpleNamespace(id=2)
    app.dependency_overrides[chats.require_expected_user] = lambda: None
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/chats/{conversation_id}/settings")
    assert response.status_code == 200, response.text
    assert response.json()["settings"] == {"temperature": 0.2, "greetingScope": "chat"}
    assert response.json()["conversation_id"] == conversation_id


def test_postgres_standalone_settings_read_returns_connections_idle(settings_db, monkeypatch):
    if settings_db.backend_type != BackendType.POSTGRESQL:
        assert settings_db.get_conversation_settings("missing") is None
        return
    conversation_id = _conversation(settings_db)
    assert settings_db.upsert_conversation_settings(conversation_id, {"temperature": 0.2})
    pool = settings_db.backend.get_pool()
    returned_states = []
    original_return = pool.return_connection

    def observe_return(conn):
        returned_states.append(conn.info.transaction_status.name)
        return original_return(conn)

    monkeypatch.setattr(pool, "return_connection", observe_return)
    assert settings_db.get_conversation_settings(conversation_id)["settings"] == {"temperature": 0.2}
    assert returned_states and set(returned_states) == {"IDLE"}
