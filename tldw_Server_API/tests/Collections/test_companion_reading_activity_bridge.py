import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import reading as reading_ep
from tldw_Server_API.app.api.v1.endpoints import reading_highlights as reading_highlights_ep
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit

TEST_USER_ID = 222

fastapi_app = FastAPI()
fastapi_app.include_router(reading_ep.router, prefix="/api/v1")
fastapi_app.include_router(reading_highlights_ep.router, prefix="/api/v1")


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(message.format(*args) if args else message)


@pytest.fixture()
def client_with_companion_opt_in(monkeypatch):
    async def override_user():
        return User(id=TEST_USER_ID, username="reader", email=None, is_active=True)

    class _FakeNotesDB:
        def get_note_by_id(self, note_id: str):
            return {"id": str(note_id), "title": "Example note"}

    monkeypatch.setenv("TEST_MODE", "1")

    base_dir = Path.cwd() / "Databases" / "test_companion_reading_bridge"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    personalization_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(TEST_USER_ID)))
    personalization_db.update_profile(str(TEST_USER_ID), enabled=1)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _FakeNotesDB()
    try:
        with TestClient(fastapi_app) as client:
            yield client, personalization_db
    finally:
        fastapi_app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


@pytest.mark.asyncio
async def test_create_highlight_failure_log_is_sanitized(monkeypatch):
    class _FailingHighlightsDB:
        def get_content_item(self, item_id: int):
            raise KeyError(item_id)

        def create_highlight(self, **kwargs: Any):
            raise RuntimeError("highlight backend exploded at /private/highlights.db")

    logger = _LoggerStub()
    monkeypatch.setattr(reading_highlights_ep, "logger", logger)

    with pytest.raises(HTTPException) as exc_info:
        await reading_highlights_ep.create_highlight(
            item_id=1,
            payload=reading_highlights_ep.HighlightCreateRequest(
                item_id=1,
                quote="Important sentence",
                color="yellow",
                note="Capture this",
                anchor_strategy="fuzzy_quote",
            ),
            current_user=User(id=TEST_USER_ID, username="reader", email=None, is_active=True),
            db=_FailingHighlightsDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "highlight_create_failed"
    assert logger.errors == ["create_highlight failed"]
    error_text = "\n".join(logger.errors)
    assert "highlight backend exploded" not in error_text
    assert "/private/highlights.db" not in error_text


def test_reading_actions_record_companion_activity(client_with_companion_opt_in):
    client, personalization_db = client_with_companion_opt_in

    save_resp = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/article",
            "title": "Example Article",
            "content": "Example article body.",
            "tags": ["ai", "priority"],
        },
    )
    assert save_resp.status_code == 200, save_resp.text
    item = save_resp.json()
    item_id = item["id"]

    duplicate_save_resp = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/article",
            "title": "Example Article",
            "content": "Example article body.",
            "tags": ["ai"],
        },
    )
    assert duplicate_save_resp.status_code == 200, duplicate_save_resp.text

    update_resp = client.patch(
        f"/api/v1/reading/items/{item_id}",
        json={"status": "read", "favorite": True, "notes": "Finished"},
    )
    assert update_resp.status_code == 200, update_resp.text

    archive_resp = client.delete(f"/api/v1/reading/items/{item_id}")
    assert archive_resp.status_code == 200, archive_resp.text
    assert archive_resp.json()["hard"] is False

    events, total = personalization_db.list_companion_activity_events(str(TEST_USER_ID), limit=10)
    assert total == 3

    event_types = [event["event_type"] for event in events]
    assert event_types == [
        "reading_item_archived",
        "reading_item_updated",
        "reading_item_saved",
    ]

    saved_event = events[2]
    assert saved_event["source_type"] == "reading_item"
    assert saved_event["source_id"] == str(item_id)
    assert saved_event["surface"] == "api.reading"
    assert saved_event["tags"] == ["ai", "priority"]
    assert saved_event["provenance"]["capture_mode"] == "explicit"
    assert saved_event["provenance"]["route"] == "/api/v1/reading/save"
    assert saved_event["metadata"]["url"] == "https://example.com/article"
    assert saved_event["metadata"]["title"] == "Example Article"

    archived_event = events[0]
    assert archived_event["metadata"]["hard_delete"] is False
    assert archived_event["provenance"]["route"] == f"/api/v1/reading/items/{item_id}"


def test_reading_note_links_and_highlights_record_companion_activity(
    client_with_companion_opt_in,
):
    client, personalization_db = client_with_companion_opt_in

    save_resp = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/highlights",
            "title": "Annotated Article",
            "content": "Important sentence with surrounding context for highlights.",
            "tags": ["research"],
        },
    )
    assert save_resp.status_code == 200, save_resp.text
    item = save_resp.json()
    item_id = item["id"]

    link_resp = client.post(
        f"/api/v1/reading/items/{item_id}/links/note",
        json={"note_id": "note-1234"},
    )
    assert link_resp.status_code == 200, link_resp.text

    highlight_resp = client.post(
        f"/api/v1/reading/items/{item_id}/highlight",
        json={
            "item_id": item_id,
            "quote": "Important sentence",
            "start_offset": 0,
            "end_offset": 18,
            "color": "yellow",
            "note": "Capture this",
            "anchor_strategy": "fuzzy_quote",
        },
    )
    assert highlight_resp.status_code == 200, highlight_resp.text
    highlight = highlight_resp.json()
    highlight_id = int(highlight["id"])

    highlight_update_resp = client.patch(
        f"/api/v1/reading/highlights/{highlight_id}",
        json={"note": "Updated capture", "state": "active"},
    )
    assert highlight_update_resp.status_code == 200, highlight_update_resp.text

    highlight_delete_resp = client.delete(f"/api/v1/reading/highlights/{highlight_id}")
    assert highlight_delete_resp.status_code == 200, highlight_delete_resp.text

    unlink_resp = client.delete(
        f"/api/v1/reading/items/{item_id}/links/note/note-1234"
    )
    assert unlink_resp.status_code == 200, unlink_resp.text

    events, total = personalization_db.list_companion_activity_events(
        str(TEST_USER_ID),
        limit=10,
    )
    assert total == 6

    event_types = [event["event_type"] for event in events]
    assert event_types == [
        "reading_note_unlinked",
        "reading_highlight_deleted",
        "reading_highlight_updated",
        "reading_highlight_created",
        "reading_note_linked",
        "reading_item_saved",
    ]

    note_link_event = next(event for event in events if event["event_type"] == "reading_note_linked")
    assert note_link_event["source_type"] == "reading_note_link"
    assert note_link_event["source_id"] == f"{item_id}:note-1234"
    assert note_link_event["metadata"]["item_title"] == "Annotated Article"
    assert note_link_event["metadata"]["note_id"] == "note-1234"
    assert note_link_event["provenance"]["route"] == f"/api/v1/reading/items/{item_id}/links/note"

    highlight_create_event = next(
        event for event in events if event["event_type"] == "reading_highlight_created"
    )
    assert highlight_create_event["source_type"] == "reading_highlight"
    assert highlight_create_event["source_id"] == str(highlight_id)
    assert highlight_create_event["metadata"]["quote"] == "Important sentence"
    assert highlight_create_event["metadata"]["item_title"] == "Annotated Article"
    assert highlight_create_event["provenance"]["route"] == f"/api/v1/reading/items/{item_id}/highlight"

    highlight_delete_event = next(
        event for event in events if event["event_type"] == "reading_highlight_deleted"
    )
    assert highlight_delete_event["metadata"]["item_id"] == item_id
    assert highlight_delete_event["metadata"]["quote"] == "Important sentence"
