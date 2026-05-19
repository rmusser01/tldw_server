import io
import json
import zipfile
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    close_all_chacha_db_instances,
    get_chacha_db_for_user,
)


@pytest.fixture()
def client(tmp_path_factory, monkeypatch):
    tmp_dir = tmp_path_factory.mktemp("chatbooks_preview")
    db_path = tmp_dir / "ChaChaNotes.db"
    db_instance = CharactersRAGDB(db_path=str(db_path), client_id="chatbooks-preview-test")

    monkeypatch.setenv("TEST_MODE", "true")

    async def override_user():
        return User(id=1, username="tester", is_active=True)

    def override_db():
        return db_instance

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_db

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        try:
            db_instance.close_all_connections()
        except Exception:
            _ = None
        close_all_chacha_db_instances()


def _make_chatbook_bytes(version_str: str = "1.0.0") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        manifest = {
            "version": version_str,
            "name": "Preview Test",
            "description": "Test manifest",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "content_items": [],
            "configuration": {},
            "statistics": {},
            "metadata": {},
            "user_info": {"user_id": "test"},
        }
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def test_preview_manifest_version_coercion_legacy(client):
    data = _make_chatbook_bytes("1.0")
    files = {"file": ("test.chatbook", data, "application/zip")}
    resp = client.post("/api/v1/chatbooks/preview", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("manifest", {}).get("version") == "1.0.0"


def test_preview_manifest_version_ok(client):
    data = _make_chatbook_bytes("1.0.0")
    files = {"file": ("test.chatbook", data, "application/zip")}
    resp = client.post("/api/v1/chatbooks/preview", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("manifest", {}).get("version") == "1.0.0"


def test_preview_returns_400_for_invalid_manifest(client):
    data = _make_chatbook_bytes("invalid-version")
    files = {"file": ("broken.chatbook", data, "application/zip")}

    resp = client.post("/api/v1/chatbooks/preview", files=files)

    assert resp.status_code == 400
    assert resp.json().get("detail") == "Invalid chatbook manifest"


def test_preview_returns_500_for_internal_extract_failure(client, monkeypatch):
    data = _make_chatbook_bytes("1.0.0")
    files = {"file": ("test.chatbook", data, "application/zip")}

    def _boom(self, path, members=None, pwd=None):
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", _boom)

    resp = client.post("/api/v1/chatbooks/preview", files=files)

    assert resp.status_code == 500
    assert resp.json().get("detail") == "An error occurred while previewing the chatbook"


def test_preview_returns_500_for_internal_validator_failure(client, monkeypatch):
    data = _make_chatbook_bytes("1.0.0")
    files = {"file": ("test.chatbook", data, "application/zip")}
    original_testzip = zipfile.ZipFile.testzip
    validator_calls = {"count": 0}

    def _testzip_then_blow_up(self):
        validator_calls["count"] += 1
        if validator_calls["count"] == 1:
            return original_testzip(self)
        raise OSError("validator exploded")

    monkeypatch.setattr(zipfile.ZipFile, "testzip", _testzip_then_blow_up)

    resp = client.post("/api/v1/chatbooks/preview", files=files)

    assert resp.status_code == 500
    assert resp.json().get("detail") == "An error occurred while previewing the chatbook"
