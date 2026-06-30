from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.media import item as item_mod
from tldw_Server_API.app.api.v1.schemas.media_request_models import MediaKeywordsUpdateRequest
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, DatabaseError, InputError

pytestmark = pytest.mark.unit


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/media/42",
            "headers": [],
        }
    )


@pytest.mark.asyncio
async def test_get_media_item_passes_media_context_to_http_error_mapper(monkeypatch):
    captured = {}

    def _raise_db_error(*args, **kwargs):
        raise DatabaseError("db exploded")

    def _fake_mapper(exc, **kwargs):
        captured["exc"] = exc
        captured.update(kwargs)
        return HTTPException(status_code=500, detail="Database error retrieving media details")

    monkeypatch.setattr(item_mod, "_is_test_mode", lambda: False)
    monkeypatch.setattr(item_mod, "get_full_media_details_rich", _raise_db_error)
    monkeypatch.setattr(item_mod, "map_db_error_to_http", _fake_mapper)

    with pytest.raises(HTTPException) as exc_info:
        await item_mod.get_media_item(
            request=_request(),
            response=Response(),
            media_id=42,
            db=object(),
            current_user=SimpleNamespace(id="user-1"),
        )

    assert exc_info.value.status_code == 500
    assert captured["default_detail"] == "Database error retrieving media details"
    assert captured["log_context"] == "get_media_item media_id=42"


@pytest.mark.asyncio
async def test_delete_media_item_uses_sanitized_conflict_message(monkeypatch):
    class _ConflictDb:
        def mark_as_trash(self, media_id: int):
            raise ConflictError(entity="Media", identifier=media_id)

    monkeypatch.setattr(item_mod, "get_media_by_id", lambda *args, **kwargs: {"is_trash": False})

    with pytest.raises(HTTPException) as exc_info:
        await item_mod.delete_media_item(
            media_id=42,
            db=_ConflictDb(),
            current_user=SimpleNamespace(id="user-1"),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Media was modified concurrently"


@pytest.mark.asyncio
async def test_update_media_keywords_returns_not_found_for_missing_media(monkeypatch):
    class _MissingMediaDb:
        def update_keywords_for_media(self, *, media_id: int, keywords: list[str]):
            raise InputError(f"Cannot update keywords: Media ID {media_id} not found or deleted.")

    monkeypatch.setattr(item_mod, "fetch_keywords_for_media", lambda *args, **kwargs: ["alpha"])

    with pytest.raises(HTTPException) as exc_info:
        await item_mod.update_media_keywords(
            payload=MediaKeywordsUpdateRequest(mode="set", keywords=["beta"]),
            media_id=42,
            db=_MissingMediaDb(),
            _current_user=SimpleNamespace(id="user-1"),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Media not found or deleted"


@pytest.mark.asyncio
async def test_update_media_keywords_returns_stable_500_for_unexpected_errors(monkeypatch):
    def _explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(item_mod, "fetch_keywords_for_media", _explode)

    with pytest.raises(HTTPException) as exc_info:
        await item_mod.update_media_keywords(
            payload=MediaKeywordsUpdateRequest(mode="set", keywords=["beta"]),
            media_id=42,
            db=object(),
            _current_user=SimpleNamespace(id="user-1"),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update keywords"
