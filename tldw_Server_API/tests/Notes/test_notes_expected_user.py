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


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/keywords", None),
        ("GET", "/keywords/", None),
        ("POST", "/keywords/", {"keyword": "Must not create in replacement account"}),
        ("DELETE", "/keywords/1", None),
        ("GET", "/collections", None),
        ("GET", "/collections/", None),
        ("POST", "/collections", {"name": "Must not create in replacement account"}),
        ("POST", "/collections/", {"name": "Must not create in replacement account"}),
        ("PATCH", "/collections/1", {"name": "Must not rename"}),
        ("DELETE", "/collections/1", None),
        ("GET", "/collections/keyword-links", None),
        ("GET", "/collections/1/keywords", None),
        ("GET", "/collections/1/keywords/", None),
        ("POST", "/collections/1/keywords/1", None),
        ("DELETE", "/collections/1/keywords/1", None),
        ("GET", "/conversations/keyword-links", None),
        ("GET", "/conversations/owned-chat/keywords", None),
        ("POST", "/conversations/owned-chat/keywords/1", None),
        ("DELETE", "/conversations/owned-chat/keywords/1", None),
    ],
)
def test_organization_rejects_replacement_cookie_principal_before_db(notes_client, method, path, body):
    """The header binds a captured operation even when cookie config is unchanged."""
    client, db, _ = notes_client
    db_reads = []

    def current_account_db():
        db_reads.append("loaded")
        return db

    client.app.dependency_overrides[notes.get_chacha_db_for_user] = current_account_db
    headers = {"X-TLDW-Expected-User-ID": "42", "expected-version": "1"}
    kwargs = {"headers": headers}
    if body is not None:
        kwargs["json"] = body
    response = client.request(method, f"/api/v1/notes{path}", **kwargs)
    assert response.status_code == 412, response.text
    assert response.json()["detail"]["code"] == "request_config_scope_changed"
    assert response.headers["cache-control"] == "no-store"
    assert db_reads == []


@pytest.mark.parametrize("expected_owner", [None, "84"])
def test_organization_current_and_legacy_account_can_create_and_read(notes_client, expected_owner):
    """Optional account assertions preserve legacy clients and current operations."""
    client, _, _ = notes_client
    headers = {} if expected_owner is None else {"X-TLDW-Expected-User-ID": expected_owner}
    keyword = client.post("/api/v1/notes/keywords/", json={"keyword": "Adventure"}, headers=headers)
    assert keyword.status_code == 201, keyword.text
    folder = client.post("/api/v1/notes/collections", json={"name": "Adventure"}, headers=headers)
    assert folder.status_code == 201, folder.text
    folders = client.get("/api/v1/notes/collections", headers=headers)
    assert folders.status_code == 200, folders.text
    assert folders.json()["collections"][0]["id"] == folder.json()["id"]
    keywords = client.get("/api/v1/notes/keywords", headers=headers)
    assert keywords.status_code == 200, keywords.text
    assert keywords.json()[0]["id"] == keyword.json()["id"]
