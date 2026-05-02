import importlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core import config as core_config
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit


@pytest.fixture()
def reading_app(monkeypatch):
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_READING", "1")
    monkeypatch.setenv("ROUTES_ENABLE", "reading")
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("TEST_MODE", "1")
    core_config.refresh_config_cache()

    base_dir = Path.cwd() / "Databases" / "test_reading_api"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    try:
        yield fastapi_app
    finally:
        fastapi_app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_reading_save_get_search_delete(reading_app):
    async def override_user():
        return User(id=333, username="reader", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/read",
            "title": "Reading Item",
            "tags": ["alpha"],
            "status": "saved",
            "content": "Example content body for reading list.",
            "notes": "Example notes.",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        item_id = body["id"]
        assert body["processing_status"] == "ready"

        r = client.get(f"/api/v1/reading/items/{item_id}")
        assert r.status_code == 200, r.text
        detail = r.json()
        assert "Example content body" in (detail.get("text") or "")
        assert detail["processing_status"] == "ready"

        r = client.get("/api/v1/reading/items", params={"q": "Example"})
        assert r.status_code == 200, r.text
        listed_payload = r.json()
        listed_ids = [item["id"] for item in listed_payload["items"]]
        assert item_id in listed_ids
        assert listed_payload["pagination"] == {
            "mode": "offset",
            "total": 1,
            "limit": 20,
            "offset": 0,
            "has_more": False,
            "next_offset": None,
        }
        assert listed_payload["has_more"] is False
        assert listed_payload["next_offset"] is None

        r = client.delete(f"/api/v1/reading/items/{item_id}")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "archived"

        r = client.delete(f"/api/v1/reading/items/{item_id}", params={"hard": "true"})
        assert r.status_code == 200, r.text
        assert r.json()["hard"] is True

        r = client.get(f"/api/v1/reading/items/{item_id}")
        assert r.status_code == 404


def test_reading_tts_requires_explicit_model(reading_app):
    async def override_user():
        return User(id=902, username="reading-tts", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    class FakeTTSService:
        async def generate_speech(self, *_args, **_kwargs):
            yield b"audiodata"

    async def fake_get_tts_service():
        return FakeTTSService()

    from tldw_Server_API.app.api.v1.endpoints import reading as reading_endpoint

    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/tts-required",
            "title": "TTS Required",
            "content": "Content for TTS validation.",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(reading_endpoint, "get_tts_service_v2", fake_get_tts_service)
        try:
            r = client.post(
                f"/api/v1/reading/items/{item_id}/tts",
                json={"stream": False},
            )
        finally:
            monkeypatch.undo()

        assert r.status_code == 422, r.text


def test_reading_save_returns_archive_requested_field(reading_app):
    async def override_user():
        return User(id=901, username="archive", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/archive",
            "title": "Archive Item",
            "content": "Archive body",
            "archive_mode": "always",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "archive_requested" in body
        assert "has_archive_copy" in body
        assert "last_fetch_error" in body
        assert body["archive_requested"] is True
        assert body["has_archive_copy"] is True


def test_reading_save_rejects_invalid_archive_mode(reading_app):
    async def override_user():
        return User(id=902, username="archive-invalid", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/archive-invalid",
            "title": "Archive Invalid",
            "content": "Archive body",
            "archive_mode": "bogus",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 422, r.text


def test_saved_search_endpoints_crud(reading_app):
    async def override_user():
        return User(id=903, username="saved-search", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        create_payload = {
            "name": "Morning",
            "query": {"q": "ai"},
            "sort": "updated_desc",
        }
        r = client.post("/api/v1/reading/saved-searches", json=create_payload)
        assert r.status_code == 201, r.text
        created = r.json()
        search_id = created["id"]
        assert created["name"] == "Morning"
        assert created["query"] == {"q": "ai"}

        r = client.get("/api/v1/reading/saved-searches", params={"limit": 10, "offset": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 1
        assert body["items"][0]["id"] == search_id
        assert body["pagination"] == {
            "mode": "offset",
            "total": 1,
            "limit": 10,
            "offset": 0,
            "has_more": False,
            "next_offset": None,
        }
        assert body["has_more"] is False
        assert body["next_offset"] is None

        r = client.patch(
            f"/api/v1/reading/saved-searches/{search_id}",
            json={"query": {"q": "ml"}},
        )
        assert r.status_code == 200, r.text
        updated = r.json()
        assert updated["query"] == {"q": "ml"}

        r = client.delete(f"/api/v1/reading/saved-searches/{search_id}")
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True

        r = client.get("/api/v1/reading/saved-searches", params={"limit": 10, "offset": 0})
        assert r.status_code == 200, r.text
        assert r.json() == {
            "items": [],
            "total": 0,
            "limit": 10,
            "offset": 0,
            "has_more": False,
            "next_offset": None,
            "pagination": {
                "mode": "offset",
                "total": 0,
                "limit": 10,
                "offset": 0,
                "has_more": False,
                "next_offset": None,
            },
        }


def test_saved_search_rejects_unsupported_query_key(reading_app):
    async def override_user():
        return User(id=911, username="saved-search-invalid-query", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        r = client.post(
            "/api/v1/reading/saved-searches",
            json={"name": "Invalid Query", "query": {"unknown_filter": "x"}},
        )
        assert r.status_code == 422, r.text


def test_saved_search_rejects_blank_name_values(reading_app):
    async def override_user():
        return User(id=912, username="saved-search-blank-name", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        r = client.post(
            "/api/v1/reading/saved-searches",
            json={"name": "   ", "query": {"q": "ai"}},
        )
        assert r.status_code == 422, r.text

        create = client.post(
            "/api/v1/reading/saved-searches",
            json={"name": "Keep Name", "query": {"q": "ai"}},
        )
        assert create.status_code == 201, create.text
        search_id = create.json()["id"]

        r = client.patch(
            f"/api/v1/reading/saved-searches/{search_id}",
            json={"name": "   "},
        )
        assert r.status_code == 422, r.text


def test_saved_search_endpoints_disabled_by_feature_flag(reading_app, monkeypatch):
    monkeypatch.setenv("COLLECTIONS_READING_SAVED_SEARCHES_ENABLED", "0")

    async def override_user():
        return User(id=906, username="saved-search-flag-off", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        r = client.post(
            "/api/v1/reading/saved-searches",
            json={"name": "Disabled", "query": {"q": "ai"}},
        )
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "reading_saved_searches_disabled"

        r = client.get("/api/v1/reading/saved-searches", params={"limit": 10, "offset": 0})
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "reading_saved_searches_disabled"


def test_note_link_endpoints_cover_post_get_delete(reading_app):
    async def override_user():
        return User(id=904, username="note-links", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        save_payload = {
            "url": "https://example.org/linkable",
            "title": "Linkable Item",
            "content": "Reading content",
        }
        r = client.post("/api/v1/reading/save", json=save_payload)
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        note_payload = {
            "title": "Linked Note",
            "content": "Note body",
        }
        r = client.post("/api/v1/notes/", json=note_payload)
        assert r.status_code == 201, r.text
        note_id = r.json()["id"]

        r = client.post(
            f"/api/v1/reading/items/{item_id}/links/note",
            json={"note_id": note_id},
        )
        assert r.status_code == 200, r.text
        linked = r.json()
        assert linked["item_id"] == item_id
        assert linked["note_id"] == note_id

        r = client.get(f"/api/v1/reading/items/{item_id}/links")
        assert r.status_code == 200, r.text
        links = r.json()["links"]
        assert any(link["note_id"] == note_id for link in links)

        r = client.delete(f"/api/v1/reading/items/{item_id}/links/note/{note_id}")
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True

        r = client.get(f"/api/v1/reading/items/{item_id}/links")
        assert r.status_code == 200, r.text
        assert r.json()["links"] == []


def test_note_link_endpoints_disabled_by_feature_flag(reading_app, monkeypatch):
    monkeypatch.setenv("COLLECTIONS_READING_NOTE_LINKS_ENABLED", "0")

    async def override_user():
        return User(id=907, username="note-links-flag-off", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        save_payload = {
            "url": "https://example.org/linkable-flag-off",
            "title": "Linkable Flag Off",
            "content": "Reading content",
        }
        r = client.post("/api/v1/reading/save", json=save_payload)
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        note_payload = {
            "title": "Linked Note Flag Off",
            "content": "Note body",
        }
        r = client.post("/api/v1/notes/", json=note_payload)
        assert r.status_code == 201, r.text
        note_id = r.json()["id"]

        r = client.post(
            f"/api/v1/reading/items/{item_id}/links/note",
            json={"note_id": note_id},
        )
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "reading_note_links_disabled"

        r = client.get(f"/api/v1/reading/items/{item_id}/links")
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "reading_note_links_disabled"


def test_note_link_rejects_missing_note(reading_app):
    async def override_user():
        return User(id=905, username="note-links-missing", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        save_payload = {
            "url": "https://example.org/linkable-missing",
            "title": "Linkable Missing",
            "content": "Reading content",
        }
        r = client.post("/api/v1/reading/save", json=save_payload)
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        r = client.post(
            f"/api/v1/reading/items/{item_id}/links/note",
            json={"note_id": "missing-note-id"},
        )
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "note_not_found"


def test_note_link_rejects_foreign_note(reading_app):
    async def user_a():
        return User(id=908, username="note-owner-a", email=None, is_active=True)

    async def user_b():
        return User(id=909, username="note-owner-b", email=None, is_active=True)

    with TestClient(reading_app) as client:
        reading_app.dependency_overrides[get_request_user] = user_a
        r = client.post(
            "/api/v1/notes/",
            json={"title": "User A Note", "content": "A body"},
        )
        assert r.status_code == 201, r.text
        note_id = r.json()["id"]

        reading_app.dependency_overrides[get_request_user] = user_b
        r = client.post(
            "/api/v1/reading/save",
            json={
                "url": "https://example.org/foreign-note",
                "title": "Foreign Note Item",
                "content": "Item body",
            },
        )
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        r = client.post(
            f"/api/v1/reading/items/{item_id}/links/note",
            json={"note_id": note_id},
        )
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "note_not_found"


def test_archive_mode_override_disabled_by_feature_flag(reading_app, monkeypatch):
    monkeypatch.setenv("COLLECTIONS_READING_ARCHIVE_CONTROLS_ENABLED", "0")

    async def override_user():
        return User(id=910, username="archive-flag-off", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        r = client.post(
            "/api/v1/reading/save",
            json={
                "url": "https://example.org/archive-default-ok",
                "title": "Archive Default Ok",
                "content": "Archive body",
            },
        )
        assert r.status_code == 200, r.text

        r = client.post(
            "/api/v1/reading/save",
            json={
                "url": "https://example.org/archive-override-blocked",
                "title": "Archive Override Blocked",
                "content": "Archive body",
                "archive_mode": "always",
            },
        )
        assert r.status_code == 404, r.text
        assert r.json()["detail"] == "reading_archive_controls_disabled"


def test_reading_user_isolation(reading_app):
    async def user_one():
        return User(id=444, username="alpha", email=None, is_active=True)

    async def user_two():
        return User(id=445, username="beta", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = user_one
    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/isolation",
            "title": "Isolation Item",
            "content": "Isolation content",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        reading_app.dependency_overrides[get_request_user] = user_two
        r = client.get("/api/v1/reading/items", params={"q": "Isolation"})
        assert r.status_code == 200, r.text
        assert not r.json()["items"]

        r = client.get(f"/api/v1/reading/items/{item_id}")
        assert r.status_code == 404, r.text


def test_reading_items_date_filters(reading_app):
    async def override_user():
        return User(id=446, username="dater", email=None, is_active=True)

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/one",
            "title": "Date One",
            "content": "Date content one",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        item_one = r.json()["id"]

        payload["url"] = "https://example.org/two"
        payload["title"] = "Date Two"
        payload["content"] = "Date content two"
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        item_two = r.json()["id"]

        cdb = CollectionsDatabase.for_user(user_id=446)
        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
        t2 = datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat()
        cdb.backend.execute(
            "UPDATE content_items SET created_at = ? WHERE id = ? AND user_id = ?",
            (t1, item_one, cdb.user_id),
        )
        cdb.backend.execute(
            "UPDATE content_items SET created_at = ? WHERE id = ? AND user_id = ?",
            (t2, item_two, cdb.user_id),
        )

        r = client.get("/api/v1/reading/items", params={"date_from": t2})
        assert r.status_code == 200, r.text
        ids = [item["id"] for item in r.json()["items"]]
        assert item_two in ids
        assert item_one not in ids


def test_reading_summarize_and_tts(reading_app, monkeypatch):
    async def override_user():
        return User(id=555, username="voice", email=None, is_active=True)

    def fake_analyze(*_args, **_kwargs):
        return "Mock summary output"

    class FakeTTSService:
        async def generate_speech(self, *_args, **_kwargs):
            yield b"audio"
            yield b"data"

    async def fake_get_tts_service():
        return FakeTTSService()

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.reading.summarize_analyze",
        fake_analyze,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.reading.get_tts_service_v2",
        fake_get_tts_service,
    )

    reading_app.dependency_overrides[get_request_user] = override_user

    with TestClient(reading_app) as client:
        payload = {
            "url": "https://example.org/tts",
            "title": "TTS Item",
            "content": "This is content for summary and TTS.",
        }
        r = client.post("/api/v1/reading/save", json=payload)
        assert r.status_code == 200, r.text
        item_id = r.json()["id"]

        r = client.post(
            f"/api/v1/reading/items/{item_id}/summarize",
            json={"provider": "openai"},
        )
        assert r.status_code == 200, r.text
        summary_body = r.json()
        assert summary_body["summary"] == "Mock summary output"
        assert summary_body["provider"] == "openai"
        assert summary_body["citations"][0]["item_id"] == item_id

        r = client.post(
            f"/api/v1/reading/items/{item_id}/tts",
            json={"model": "KittenML/kitten-tts-nano-0.8", "voice": "Bella", "stream": False},
        )
        assert r.status_code == 200, r.text
        assert r.content == b"audiodata"


def test_reading_tts_sanitizes_generic_tts_error():
    from tldw_Server_API.app.api.v1.endpoints import reading as reading_endpoint

    with pytest.raises(HTTPException) as excinfo:
        reading_endpoint._raise_for_tts_error(
            reading_endpoint.TTSError("provider backend exploded"),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "TTS generation failed"


def test_reading_tts_sanitizes_provider_not_configured_error():
    from tldw_Server_API.app.api.v1.endpoints import reading as reading_endpoint

    with pytest.raises(HTTPException) as excinfo:
        reading_endpoint._raise_for_tts_error(
            reading_endpoint.TTSProviderNotConfiguredError("tts provider secret path missing"),
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "TTS service unavailable"
