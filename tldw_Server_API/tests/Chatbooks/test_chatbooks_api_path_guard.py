# test_chatbooks_api_path_guard.py
# API-level checks for chatbook upload path guard behavior

import io
import json
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Chatbooks.chatbook_validators import ChatbookValidator
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    close_all_chacha_db_instances,
    get_chacha_db_for_user,
)
from tldw_Server_API.tests.Chatbooks.test_chatbook_security import (
    build_dangerous_file_archive_bytes,
    build_symlink_archive_bytes,
    build_traversal_archive_bytes,
)


@pytest.fixture()
def client(tmp_path_factory):
    """Provide a TestClient with isolated ChaChaNotes DB + auth overrides."""
    tmp_dir = tmp_path_factory.mktemp("chatbooks_path_guard")
    db_path = tmp_dir / "ChaChaNotes.db"
    db_instance = CharactersRAGDB(db_path=str(db_path), client_id="chatbooks-path-guard-test")

    # Keep tests permissive by disabling rate limiting
    os.environ["TEST_MODE"] = "true"

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


def _make_chatbook_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        manifest = {
            "version": "1.0.0",
            "name": "Path Guard Test",
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


def _force_unsafe_filename(monkeypatch, unsafe_name: str) -> None:
    def _unsafe_validate(cls, _filename: str):
        return True, None, unsafe_name

    monkeypatch.setattr(ChatbookValidator, "validate_filename", classmethod(_unsafe_validate))


@pytest.mark.parametrize("unsafe_name", ["../evil.zip", "nested/evil.zip"])
def test_import_rejects_unsafe_safe_filename(monkeypatch, client, unsafe_name):
    _force_unsafe_filename(monkeypatch, unsafe_name)
    data = _make_chatbook_bytes()
    files = {"file": ("test.chatbook", data, "application/zip")}

    resp = client.post("/api/v1/chatbooks/import", files=files)

    assert resp.status_code == 400
    assert resp.json().get("detail") == "Invalid file path"


@pytest.mark.parametrize("unsafe_name", ["../evil.zip", "nested/evil.zip"])
def test_preview_rejects_unsafe_safe_filename(monkeypatch, client, unsafe_name):
    _force_unsafe_filename(monkeypatch, unsafe_name)
    data = _make_chatbook_bytes()
    files = {"file": ("test.chatbook", data, "application/zip")}

    resp = client.post("/api/v1/chatbooks/preview", files=files)

    assert resp.status_code == 400
    assert resp.json().get("detail") == "Invalid file path"


@pytest.mark.parametrize("endpoint", ["/api/v1/chatbooks/import", "/api/v1/chatbooks/preview"])
@pytest.mark.parametrize(
    ("archive_bytes", "expected_error"),
    [
        (build_symlink_archive_bytes(), "symlink"),
        (build_traversal_archive_bytes(), "unsafe"),
        (build_dangerous_file_archive_bytes(), "dangerous"),
    ],
    ids=["symlink", "path-traversal", "dangerous-file-type"],
)
def test_api_rejects_malicious_archives(client, endpoint, archive_bytes, expected_error):
    files = {"file": ("test.chatbook", archive_bytes, "application/zip")}

    resp = client.post(endpoint, files=files)

    assert resp.status_code == 400
    assert expected_error in resp.json().get("detail", "").lower()
