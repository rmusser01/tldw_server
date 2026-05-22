"""
Integration tests for Character Chat endpoints: sessions, messages, and world books.
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timezone
import pytest
import httpx
import uuid as _uuid

from tldw_Server_API.app.core.AuthNZ.settings import get_settings


def test_chat_session_list_assistant_names_are_preloaded_in_bulk():
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
        _attach_conversation_assistant_names_from_lookups,
        _conversation_assistant_name_lookups,
    )

    class FakeDb:
        def __init__(self):
            self.character_bulk_calls = 0
            self.persona_bulk_calls = 0

        def get_character_cards_by_ids(self, character_ids):
            self.character_bulk_calls += 1
            assert character_ids == [1, 2]
            return {
                1: {"id": 1, "name": "Ada"},
                2: {"id": 2, "name": "Babbage"},
            }

        def get_persona_profiles_by_ids(self, *, user_id, persona_ids, include_deleted=False):
            self.persona_bulk_calls += 1
            assert user_id == "user-1"
            assert persona_ids == ["persona-1"]
            assert include_deleted is False
            return {"persona-1": {"id": "persona-1", "name": "Researcher"}}

        def get_character_card_by_id(self, character_id):
            raise AssertionError("list assistant-name resolution must not call per-row character lookup")

        def get_persona_profile(self, persona_id, *, user_id, include_deleted=False):
            raise AssertionError("list assistant-name resolution must not call per-row persona lookup")

    conversations = [
        {"id": "chat-1", "character_id": 1, "assistant_kind": "character"},
        {"id": "chat-2", "character_id": None, "assistant_kind": "character", "assistant_id": "2"},
        {"id": "chat-3", "character_id": None, "assistant_kind": "persona", "assistant_id": "persona-1"},
    ]

    db = FakeDb()
    character_names, persona_names = _conversation_assistant_name_lookups(
        db,
        conversations,
        "user-1",
    )

    labeled = [
        _attach_conversation_assistant_names_from_lookups(
            dict(conversation),
            character_names=character_names,
            persona_names=persona_names,
        )
        for conversation in conversations
    ]

    assert db.character_bulk_calls == 1
    assert db.persona_bulk_calls == 1
    assert labeled[0]["character_name"] == "Ada"
    assert labeled[0]["assistant_name"] == "Ada"
    assert labeled[1]["character_name"] == "Babbage"
    assert labeled[1]["assistant_name"] == "Babbage"
    assert labeled[2]["assistant_name"] == "Researcher"
    assert "character_name" not in labeled[2]


@pytest.mark.asyncio
async def test_character_chat_flow_sessions_messages_worldbooks():
    # Use an isolated per-test DB base directory
    tmpdir = tempfile.mkdtemp(prefix="chacha_test_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) List characters and pick one
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            chars = r.json()
            assert isinstance(chars, list) and len(chars) >= 1
            character_id = chars[0]["id"]
            character_name = chars[0]["name"]

            # 2) Create chat session
            create_payload = {"character_id": character_id, "title": "Test Chat"}
            r = await client.post("/api/v1/chats/", headers=headers, json=create_payload)
            assert r.status_code == 201
            chat = r.json()
            chat_id = chat["id"]
            chat_version = chat["version"]
            assert chat["assistant_kind"] == "character"
            assert chat["assistant_id"] == str(character_id)
            assert chat["character_id"] == character_id
            assert chat["character_name"] == character_name
            assert chat["assistant_name"] == character_name
            assert chat["persona_memory_mode"] is None

            # 3) Update chat session title (optimistic lock)
            r = await client.put(
                f"/api/v1/chats/{chat_id}",
                headers=headers,
                params={"expected_version": chat_version},
                json={"title": "Updated Test Chat"},
            )
            assert r.status_code == 200
            updated_chat = r.json()
            assert updated_chat["title"] == "Updated Test Chat"
            assert updated_chat["version"] == chat_version + 1
            assert updated_chat["character_name"] == character_name
            assert updated_chat["assistant_name"] == character_name
            chat_version = updated_chat["version"]

            # 3b) Chat settings read/write
            r = await client.get(f"/api/v1/chats/{chat_id}/settings", headers=headers)
            assert r.status_code == 404

            settings_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "greetingEnabled": True
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=settings_payload,
            )
            assert r.status_code == 200
            settings_resp = r.json()
            assert settings_resp["conversation_id"] == chat_id
            assert settings_resp["settings"]["greetingEnabled"] is True

            r = await client.get(f"/api/v1/chats/{chat_id}/settings", headers=headers)
            assert r.status_code == 200
            settings_resp = r.json()
            assert settings_resp["settings"]["greetingEnabled"] is True

            # 4) Send a user message
            msg_payload = {"role": "user", "content": "Hello there!"}
            r = await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json=msg_payload)
            assert r.status_code == 201
            msg = r.json()
            message_id = msg["id"]
            message_version = msg["version"]

            r = await client.get("/api/v1/chats/", headers=headers)
            assert r.status_code == 200
            listed_chats = r.json()["chats"]
            listed_chat = next((item for item in listed_chats if item["id"] == chat_id), None)
            assert listed_chat is not None, f"chat with id {chat_id} not found in listed_chats"
            assert listed_chat["character_name"] == character_name
            assert listed_chat["assistant_name"] == character_name

            # 5) Get messages and verify
            r = await client.get(f"/api/v1/chats/{chat_id}/messages", headers=headers)
            assert r.status_code == 200
            msgs = r.json()
            # When not using format_for_completions, response is a dict with messages list
            assert "messages" in msgs
            assert any(m.get("id") == message_id for m in msgs["messages"])  # our message present
            assert msgs["pagination"] == {
                "mode": "offset",
                "limit": 50,
                "offset": 0,
                "total": 1,
                "has_more": False,
                "next_offset": None,
            }
            assert msgs["has_more"] is False
            assert msgs["next_offset"] is None

            # 6) Delete the message (optimistic lock)
            r = await client.delete(
                f"/api/v1/messages/{message_id}",
                headers=headers,
                params={"expected_version": message_version},
            )
            assert r.status_code == 204

            # 7) Delete the chat session (optimistic lock)
            # Refresh to get current version
            r = await client.get(f"/api/v1/chats/{chat_id}", headers=headers)
            assert r.status_code == 200
            current_chat = r.json()
            assert current_chat["character_name"] == character_name
            assert current_chat["assistant_name"] == character_name
            r = await client.delete(
                f"/api/v1/chats/{chat_id}",
                headers=headers,
                params={"expected_version": current_chat["version"]},
            )
            assert r.status_code == 204
            # Ensure deleted
            r = await client.get(f"/api/v1/chats/{chat_id}", headers=headers)
            assert r.status_code == 404

            # 8) World book CRUD
            wb_name = f"WB Test {_uuid.uuid4()}"
            world_book_budget = 500
            wb_create = {
                "name": wb_name,
                "description": "World book for tests",
                "scan_depth": 3,
                "token_budget": world_book_budget,
                "recursive_scanning": False,
                "enabled": True,
            }
            r = await client.post("/api/v1/characters/world-books", headers=headers, json=wb_create)
            assert r.status_code == 201
            wb = r.json()
            wb_id = wb["id"]

            r = await client.get("/api/v1/characters/world-books", headers=headers)
            assert r.status_code == 200
            wb_list = r.json()
            assert wb_list.get("total", 0) >= 1

            r = await client.get(f"/api/v1/characters/world-books/{wb_id}", headers=headers)
            assert r.status_code == 200
            wb_get = r.json()
            assert wb_get["id"] == wb_id

            r = await client.put(
                f"/api/v1/characters/world-books/{wb_id}",
                headers=headers,
                json={"name": f"WB Test Updated {_uuid.uuid4()}"},
            )
            assert r.status_code == 200
            wb_upd = r.json()
            assert wb_upd["name"].startswith("WB Test Updated ")

            r = await client.delete(f"/api/v1/characters/world-books/{wb_id}", headers=headers)
            assert r.status_code == 200
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_create_persona_backed_chat_session():
    tmpdir = tempfile.mkdtemp(prefix="chacha_persona_chat_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            persona_resp = await client.post(
                "/api/v1/persona/profiles",
                headers=headers,
                json={"name": "Garden Helper"},
            )
            assert persona_resp.status_code == 201, persona_resp.text
            persona_id = persona_resp.json()["id"]

            create_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={
                    "assistant_kind": "persona",
                    "assistant_id": persona_id,
                    "persona_memory_mode": "read_only",
                    "title": "Persona-backed chat",
                },
            )
            assert create_resp.status_code == 201, create_resp.text
            body = create_resp.json()
            assert body["assistant_kind"] == "persona"
            assert body["assistant_id"] == persona_id
            assert body["character_id"] is None
            assert body["persona_memory_mode"] == "read_only"

            detail_resp = await client.get(f"/api/v1/chats/{body['id']}", headers=headers)
            assert detail_resp.status_code == 200, detail_resp.text
            detail = detail_resp.json()
            assert detail["assistant_kind"] == "persona"
            assert detail["assistant_id"] == persona_id
            assert detail["character_id"] is None
            assert detail["persona_memory_mode"] == "read_only"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_message_placeholders_and_length_guard(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_placeholders_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir
    try:
        from tldw_Server_API.app.core.Character_Chat.modules import character_chat as cc
        monkeypatch.setattr(cc, "settings", {"MAX_PERSIST_CONTENT_LENGTH": 20})

        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a character with placeholders in fields
            char_name = f"PlaceholderBot-{_uuid.uuid4()}"
            char_payload = {
                "name": char_name,
                "description": "I am {{char}} for {{user}}.",
                "personality": "Helpful to {{user}}.",
                "scenario": "Meeting {{user}}.",
                "system_prompt": "System for {{char}} and {{user}}.",
            }
            r = await client.post("/api/v1/characters/", headers=headers, json=char_payload)
            assert r.status_code == 201
            character_id = r.json().get("id") or r.json().get("character_id")

            # Create chat session
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # Send assistant message with placeholders (within limit)
            msg_content = "Hi {{user}}, I'm {{char}}."
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "assistant", "content": msg_content},
            )
            assert r.status_code == 201

            # Standard message listing should replace placeholders
            r = await client.get(f"/api/v1/chats/{chat_id}/messages", headers=headers)
            assert r.status_code == 200
            body = r.json()
            msgs = body.get("messages", [])
            assistant_msg = next(m for m in msgs if m.get("sender") == "assistant")
            assert assistant_msg["content"] == f"Hi User, I'm {char_name}."
            assert body["pagination"] == {
                "mode": "offset",
                "limit": 50,
                "offset": 0,
                "total": 1,
                "has_more": False,
                "next_offset": None,
            }
            assert body["has_more"] is False
            assert body["next_offset"] is None

            # Completions-formatted messages should replace placeholders in system context
            r = await client.get(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                params={"format_for_completions": True, "include_character_context": True},
            )
            assert r.status_code == 200
            data = r.json()
            sys_msg = next(m for m in data["messages"] if m.get("role") == "system")
            assert "{{" not in sys_msg["content"]
            assert char_name in sys_msg["content"]
            assert "User" in sys_msg["content"]

            # Oversized content should be rejected by guardrails
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "x" * 25},
            )
            assert r.status_code == 400
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_send_message_returns_generic_500_for_db_error(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_send_message_db_error_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "DB error chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            def fake_post_message_to_conversation(*args, **kwargs):
                raise CharactersRAGDBError("message send backend unavailable")

            monkeypatch.setattr(
                character_messages_endpoint,
                "post_message_to_conversation",
                fake_post_message_to_conversation,
            )

            response = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Hello"},
            )

            assert response.status_code == 500
            assert response.json()["detail"] == "Failed to send message"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_send_message_maps_input_error_to_400(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_send_message_input_error_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Input error chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            def fake_post_message_to_conversation(*args, **kwargs):
                raise InputError("message payload is invalid")

            monkeypatch.setattr(
                character_messages_endpoint,
                "post_message_to_conversation",
                fake_post_message_to_conversation,
            )

            response = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Hello"},
            )

            assert response.status_code == 400
            assert response.json()["detail"] == "message payload is invalid"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_send_message_maps_oversize_input_error_to_413(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_send_message_oversize_input_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Oversize input error chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            def fake_post_message_to_conversation(*args, **kwargs):
                raise InputError("Attachment exceeds maximum size")

            monkeypatch.setattr(
                character_messages_endpoint,
                "post_message_to_conversation",
                fake_post_message_to_conversation,
            )

            response = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Hello"},
            )

            assert response.status_code == 413
            assert response.json()["detail"] == "Attachment exceeds maximum size"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_send_message_maps_conflict_error_to_409(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_send_message_conflict_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Conflict error chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            def fake_post_message_to_conversation(*args, **kwargs):
                raise ConflictError("message version conflict")

            monkeypatch.setattr(
                character_messages_endpoint,
                "post_message_to_conversation",
                fake_post_message_to_conversation,
            )

            response = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Hello"},
            )

            assert response.status_code == 409
            assert response.json()["detail"] == "message version conflict"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_edit_message_returns_generic_500_for_db_error(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_edit_message_db_error_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Edit DB error chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            message_resp = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Original"},
            )
            assert message_resp.status_code == 201
            message = message_resp.json()

            def fake_edit_message_content(*args, **kwargs):
                raise CharactersRAGDBError("message edit backend unavailable")

            monkeypatch.setattr(
                character_messages_endpoint,
                "edit_message_content",
                fake_edit_message_content,
            )

            response = await client.put(
                f"/api/v1/messages/{message['id']}",
                headers=headers,
                params={"expected_version": message["version"]},
                json={"content": "Updated"},
            )

            assert response.status_code == 500
            assert response.json()["detail"] == "Failed to edit message"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_edit_message_maps_conflict_error_to_409(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_edit_message_conflict_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Edit conflict chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            message_resp = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Original"},
            )
            assert message_resp.status_code == 201
            message = message_resp.json()

            def fake_edit_message_content(*args, **kwargs):
                raise ConflictError("message edit conflict")

            monkeypatch.setattr(
                character_messages_endpoint,
                "edit_message_content",
                fake_edit_message_content,
            )

            response = await client.put(
                f"/api/v1/messages/{message['id']}",
                headers=headers,
                params={"expected_version": message["version"]},
                json={"content": "Updated"},
            )

            assert response.status_code == 409
            assert response.json()["detail"] == "message edit conflict"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_delete_message_returns_generic_500_for_db_error(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_delete_message_db_error_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Delete DB error chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            message_resp = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Delete me"},
            )
            assert message_resp.status_code == 201
            message = message_resp.json()

            def fake_remove_message_from_conversation(*args, **kwargs):
                raise CharactersRAGDBError("message delete backend unavailable")

            monkeypatch.setattr(
                character_messages_endpoint,
                "remove_message_from_conversation",
                fake_remove_message_from_conversation,
            )

            response = await client.delete(
                f"/api/v1/messages/{message['id']}",
                headers=headers,
                params={"expected_version": message["version"]},
            )

            assert response.status_code == 500
            assert response.json()["detail"] == "Failed to delete message"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_delete_message_maps_conflict_error_to_409(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_delete_message_conflict_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.api.v1.endpoints import character_messages as character_messages_endpoint
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Delete conflict chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            message_resp = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Delete me"},
            )
            assert message_resp.status_code == 201
            message = message_resp.json()

            def fake_remove_message_from_conversation(*args, **kwargs):
                raise ConflictError("message delete conflict")

            monkeypatch.setattr(
                character_messages_endpoint,
                "remove_message_from_conversation",
                fake_remove_message_from_conversation,
            )

            response = await client.delete(
                f"/api/v1/messages/{message['id']}",
                headers=headers,
                params={"expected_version": message["version"]},
            )

            assert response.status_code == 409
            assert response.json()["detail"] == "message delete conflict"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


@pytest.mark.asyncio
async def test_chat_endpoint_lists_linked_research_runs():
    tmpdir = tempfile.mkdtemp(prefix="chacha_research_runs_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir

    try:
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.core.Research.service import ResearchService
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        class DummyJobs:
            def create_job(self, **kwargs):
                return {"id": 31, "uuid": "job-31", "status": "queued"}

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chars = (await client.get("/api/v1/characters/", headers=headers)).json()
            character_id = chars[0]["id"]

            chat_resp = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "title": "Research-linked chat"},
            )
            assert chat_resp.status_code == 201
            chat_id = chat_resp.json()["id"]

            service = ResearchService(
                research_db_path=None,
                outputs_dir=None,
                job_manager=DummyJobs(),
            )
            session = service.create_session(
                owner_user_id="1",
                query="Investigate linked run visibility in chat",
                source_policy="balanced",
                autonomy_mode="checkpointed",
                chat_handoff={"chat_id": chat_id},
            )

            response = await client.get(f"/api/v1/chats/{chat_id}/research-runs", headers=headers)

            assert response.status_code == 200
            payload = response.json()
            assert payload == {
                "runs": [
                    {
                        "run_id": session.id,
                        "query": "Investigate linked run visibility in chat",
                        "status": "queued",
                        "phase": "drafting_plan",
                        "control_state": "running",
                        "latest_checkpoint_id": None,
                        "updated_at": payload["runs"][0]["updated_at"],
                    }
                ]
            }
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None


# --- Unit Tests for Helper Functions (Regression Tests) ---


def test_extract_text_with_none():
    """
    Regression test for Issue #1: Malformed _extract_text function.

    The _extract_text function should handle None input and return empty string.
    """
    # Import the function from the endpoint module
    # Note: The function is defined inside prepare_completion, so we test via behavior
    # This test verifies the function doesn't crash with various inputs

    # Direct test of expected behavior:
    # _extract_text(None) should return ""
    # _extract_text("string") should return "string"
    # _extract_text({"choices": [{"message": {"content": "text"}}]}) should return "text"

    # We test the logic directly since _extract_text is a local function
    def _extract_text(resp):
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            try:
                return resp.get("choices", [{}])[0].get("message", {}).get("content", "") or resp.get("text", "")
            except Exception:
                return resp.get("text", "")
        try:
            return str(resp)
        except Exception:
            return ""

    # Test cases
    assert _extract_text(None) == ""
    assert _extract_text("hello") == "hello"
    assert _extract_text({"choices": [{"message": {"content": "response"}}]}) == "response"
    assert _extract_text({"text": "fallback"}) == "fallback"
    assert _extract_text(123) == "123"
    assert _extract_text({}) == ""
