import pytest
from fastapi import Depends, Request, status
from unittest.mock import MagicMock

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.TTS.utils import compute_tts_history_text_hash
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.api.v1.endpoints.audio import audio_history


pytestmark = [pytest.mark.unit]


def _override_media_db(db: MediaDatabase):
    async def _override(request: Request, current_user: User = Depends(get_request_user)):
        yield db
    return _override


def _media_db_dependency_keys():
    keys = [get_media_db_for_user]
    # When app.main is reloaded in other tests, router modules can retain a
    # different dependency function object identity.
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_history

    history_dep = getattr(audio_history, "get_media_db_for_user", None)
    if history_dep is not None and history_dep not in keys:
        keys.append(history_dep)
    return keys


def _set_media_db_override(fastapi_app, db: MediaDatabase):
    override = _override_media_db(db)
    keys = _media_db_dependency_keys()
    for key in keys:
        fastapi_app.dependency_overrides[key] = override
    return keys


def _clear_media_db_override(fastapi_app, keys):
    for key in keys:
        fastapi_app.dependency_overrides.pop(key, None)


def test_history_list_favorite_delete(test_client, auth_headers):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_test")
    fastapi_app = test_client.app
    entry_one = db.create_tts_history_entry(
        user_id="1",
        text_hash="hash_one",
        text="Hello world",
        text_length=11,
        provider="openai",
        model="tts-1",
        voice_name="alloy",
        format="mp3",
        status="success",
        favorite=False,
    )
    entry_two = db.create_tts_history_entry(
        user_id="1",
        text_hash="hash_two",
        text="Another entry",
        text_length=13,
        provider="openai",
        model="tts-1",
        voice_name="alloy",
        format="mp3",
        status="success",
        favorite=True,
    )

    dep_keys = _set_media_db_override(fastapi_app, db)
    try:
        resp = test_client.get("/api/v1/audio/history?favorite=true", headers=auth_headers)
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        assert len(payload["items"]) == 1
        assert payload["items"][0]["id"] == entry_two
        assert payload["pagination"] == {
            "mode": "cursor",
            "limit": 50,
            "cursor": None,
            "next_cursor": None,
            "has_more": False,
        }

        resp = test_client.patch(
            f"/api/v1/audio/history/{entry_one}",
            json={"favorite": True},
            headers=auth_headers,
        )
        assert resp.status_code == status.HTTP_200_OK

        resp = test_client.get("/api/v1/audio/history?favorite=true", headers=auth_headers)
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        ids = {item["id"] for item in payload["items"]}
        assert entry_one in ids
        assert entry_two in ids

        resp = test_client.delete(f"/api/v1/audio/history/{entry_two}", headers=auth_headers)
        assert resp.status_code == status.HTTP_204_NO_CONTENT

        resp = test_client.get("/api/v1/audio/history", headers=auth_headers)
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        ids = {item["id"] for item in payload["items"]}
        assert entry_two not in ids
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_q_rejected_when_text_disabled(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_q")
    fastapi_app = test_client.app
    dep_keys = _set_media_db_override(fastapi_app, db)
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", False, raising=False)
    try:
        resp = test_client.get("/api/v1/audio/history?q=hello", headers=auth_headers)
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_text_exact_search(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_text_exact")
    fastapi_app = test_client.app
    monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", False, raising=False)
    monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "test-history-key", raising=False)

    text_value = "Exact Match"
    text_hash = compute_tts_history_text_hash(text_value, "test-history-key")
    db.create_tts_history_entry(
        user_id="1",
        text_hash=text_hash,
        text=None,
        text_length=len(text_value),
        status="success",
    )

    dep_keys = _set_media_db_override(fastapi_app, db)
    try:
        resp = test_client.get(f"/api/v1/audio/history?text_exact={text_value}", headers=auth_headers)
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        assert len(payload["items"]) == 1
        assert payload["items"][0]["has_text"] is False
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_text_exact_hash_failure_log_is_sanitized(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_hash_failure")
    fastapi_app = test_client.app
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_history, "logger", fake_logger)

    def fail_hash(*_args, **_kwargs):
        raise RuntimeError("hash backend exploded at /private/tts-history.key")

    monkeypatch.setattr(audio_history, "compute_tts_history_text_hash", fail_hash)

    dep_keys = _set_media_db_override(fastapi_app, db)
    try:
        resp = test_client.get("/api/v1/audio/history?text_exact=Exact", headers=auth_headers)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["detail"] == "TTS history hash key not configured"
        fake_logger.debug.assert_called_once_with(
            "TTS history: failed to compute text_exact hash"
        )
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_next_cursor_failure_log_is_sanitized(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_cursor_failure")
    fastapi_app = test_client.app
    fake_logger = MagicMock()
    monkeypatch.setattr(audio_history, "logger", fake_logger)

    db.create_tts_history_entry(
        user_id="1",
        text_hash="cursor_hash_1",
        text="First",
        text_length=5,
        status="success",
    )
    db.create_tts_history_entry(
        user_id="1",
        text_hash="cursor_hash_2",
        text="Second",
        text_length=6,
        status="success",
    )

    def fail_encode_cursor(*_args, **_kwargs):
        raise RuntimeError("cursor encoder exploded at /private/tts-cursor")

    monkeypatch.setattr(audio_history, "_encode_cursor", fail_encode_cursor)

    dep_keys = _set_media_db_override(fastapi_app, db)
    try:
        resp = test_client.get("/api/v1/audio/history?limit=1", headers=auth_headers)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["detail"] == "Failed to list TTS history"
        fake_logger.error.assert_called_once()
        error_args, error_kwargs = fake_logger.error.call_args
        assert error_args[0] == "TTS history: failed to build next cursor for id={} created_at={}"
        assert error_args[1] == 2
        assert error_args[2]
        assert error_kwargs == {"exc_info": True}
        assert "/private/tts-cursor" not in repr(fake_logger.error.call_args)
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_voice_id_and_voice_name_filters(test_client, auth_headers):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_voice_filters")
    fastapi_app = test_client.app
    first_id = db.create_tts_history_entry(
        user_id="1",
        text_hash="voice_hash_1",
        text="OpenAI Alloy",
        text_length=12,
        provider="openai",
        model="tts-1",
        voice_id="alloy",
        voice_name="Alloy",
        status="success",
    )
    second_id = db.create_tts_history_entry(
        user_id="1",
        text_hash="voice_hash_2",
        text="ElevenLabs Rachel",
        text_length=17,
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        voice_id="rachel_v2",
        voice_name="Rachel",
        status="success",
    )
    db.create_tts_history_entry(
        user_id="1",
        text_hash="voice_hash_3",
        text="Custom voice",
        text_length=12,
        provider="openai",
        model="tts-1",
        voice_id="custom_demo",
        voice_name="Demo Voice",
        status="success",
    )

    dep_keys = _set_media_db_override(fastapi_app, db)
    try:
        resp = test_client.get("/api/v1/audio/history?voice_id=rachel_v2", headers=auth_headers)
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        assert [item["id"] for item in payload["items"]] == [second_id]

        resp = test_client.get("/api/v1/audio/history?voice_name=Alloy", headers=auth_headers)
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        assert [item["id"] for item in payload["items"]] == [first_id]

        resp = test_client.get(
            "/api/v1/audio/history?voice_id=alloy&voice_name=Alloy",
            headers=auth_headers,
        )
        assert resp.status_code == status.HTTP_200_OK
        payload = resp.json()
        assert [item["id"] for item in payload["items"]] == [first_id]
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_list_maps_database_error(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_list_error")
    fastapi_app = test_client.app
    dep_keys = _set_media_db_override(fastapi_app, db)

    def fail_list(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "list_tts_history", fail_list)
    try:
        resp = test_client.get("/api/v1/audio/history", headers=auth_headers)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["detail"] == "Failed to list TTS history"
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_detail_maps_database_error(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_detail_error")
    fastapi_app = test_client.app
    dep_keys = _set_media_db_override(fastapi_app, db)

    def fail_get(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "get_tts_history_entry", fail_get)
    try:
        resp = test_client.get("/api/v1/audio/history/42", headers=auth_headers)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["detail"] == "Failed to fetch TTS history entry"
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_update_maps_database_error(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_update_error")
    fastapi_app = test_client.app
    dep_keys = _set_media_db_override(fastapi_app, db)

    def fail_update(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "update_tts_history_favorite", fail_update)
    try:
        resp = test_client.patch(
            "/api/v1/audio/history/42",
            json={"favorite": True},
            headers=auth_headers,
        )
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["detail"] == "Failed to update TTS history entry"
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()


def test_history_delete_maps_database_error(test_client, auth_headers, monkeypatch):
    db = MediaDatabase(db_path=":memory:", client_id="tts_history_delete_error")
    fastapi_app = test_client.app
    dep_keys = _set_media_db_override(fastapi_app, db)

    def fail_delete(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise DatabaseError("driver failed")

    monkeypatch.setattr(MediaDatabase, "soft_delete_tts_history_entry", fail_delete)
    try:
        resp = test_client.delete("/api/v1/audio/history/42", headers=auth_headers)
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert resp.json()["detail"] == "Failed to delete TTS history entry"
    finally:
        _clear_media_db_override(fastapi_app, dep_keys)
        db.close_connection()
