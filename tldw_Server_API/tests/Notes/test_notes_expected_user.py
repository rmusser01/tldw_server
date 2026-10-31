"""Notes requests assert the owner captured before a cookie-session switch."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import notes
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


class _RateLimiter:
    async def check_user_rate_limit(self, *_args):
        return True, {}


@pytest.fixture
def notes_client(tmp_path, monkeypatch):
    db = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="owner-84")
    note_id = db.add_note("Original", "Original private content")
    app = FastAPI()
    app.include_router(notes.router, prefix="/api/v1/notes")
    app.dependency_overrides[notes.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[notes.get_request_user] = lambda: SimpleNamespace(id=84, is_admin=True)
    app.dependency_overrides[get_auth_principal] = lambda: SimpleNamespace(user_id=84)
    app.dependency_overrides[notes.get_rate_limiter_dep] = lambda: _RateLimiter()
    monkeypatch.setattr(notes, "get_active_server_origin_sync_service_for_user", lambda _user_id: None)
    try:
        with TestClient(app) as client:
            yield client, db, note_id
    finally:
        db.close_connection()


@pytest.mark.parametrize("method", ["POST", "GET", "PUT"])
def test_notes_reject_changed_owner_before_read_or_write(notes_client, method):
    client, db, note_id = notes_client
    kwargs = {"headers": {"X-TLDW-Expected-User-ID": "42", "expected-version": "1"}}
    if method != "GET":
        kwargs["json"] = {"title": "Other account draft", "content": "Must not write"}
    path = "/api/v1/notes/" if method == "POST" else f"/api/v1/notes/{note_id}"
    response = client.request(method, path, **kwargs)
    assert response.status_code == 412, response.text
    assert response.json()["detail"]["code"] == "request_config_scope_changed"
    assert db.get_note_by_id(note_id)["content"] == "Original private content"


@pytest.mark.parametrize("expected_owner", [None, "84"])
def test_notes_allow_current_owner_and_legacy_unscoped_callers(notes_client, expected_owner):
    client, _, note_id = notes_client
    headers = {} if expected_owner is None else {"X-TLDW-Expected-User-ID": expected_owner}
    response = client.get(f"/api/v1/notes/{note_id}", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["content"] == "Original private content"
