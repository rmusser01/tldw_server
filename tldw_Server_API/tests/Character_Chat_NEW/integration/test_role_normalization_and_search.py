"""
Integration tests for role normalization and message search placeholder handling.
"""

import base64
from datetime import datetime, timezone
import io
import shutil
import tempfile

import httpx
import pytest
from PIL import Image

from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_chat_context_and_prepare_roles_normalized(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_roles_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Use default character
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            # Create chat
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # Send user + assistant + system messages
            assert (await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "user", "content": "hello"})).status_code == 201
            assert (await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "assistant", "content": "hi there"})).status_code == 201
            assert (await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "system", "content": "note"})).status_code == 201

            # get_chat_context
            r = await client.get(f"/api/v1/chats/{chat_id}/context", headers=headers)
            assert r.status_code == 200
            msgs = r.json()["messages"]
            roles = {m["role"] for m in msgs}
            assert roles.issubset({"user", "assistant", "system"})

            # prepare_chat_completion (should include system preface + normalized roles)
            r = await client.post(
                f"/api/v1/chats/{chat_id}/completions",
                headers=headers,
                json={"include_character_context": True, "limit": 10, "offset": 0}
            )
            assert r.status_code == 200
            data = r.json()
            roles2 = [m["role"] for m in data["messages"]]
            assert roles2[0] == "system"
            assert set(roles2[1:]).issubset({"user", "assistant", "system", "tool"})
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_uses_normalized_roles_via_stubbed_provider(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_complete_v2_roles_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        # Monkeypatch provider call to capture messages
        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):

            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Setup character + chat + a couple of messages
            r = await client.get("/api/v1/characters/", headers=headers)
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            chat_id = r.json()["id"]
            await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "user", "content": "hello"})
            await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "assistant", "content": "hi"})

            # Use a non-local provider to trigger provider call path
            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={"provider": "openai", "model": "gpt-x", "append_user_message": "test", "save_to_db": False}
            )
            assert r.status_code == 200
            assert "messages" in captured
            roles = {m.get("role") for m in captured["messages"]}
            assert roles.issubset({"system", "user", "assistant", "tool"})
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_messages_format_for_completions_roles_and_search_placeholders(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_msgs_roles_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Setup
            r = await client.get("/api/v1/characters/", headers=headers)
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            chat_id = r.json()["id"]

            # Add messages including a placeholder in assistant content
            await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "user", "content": "Hi"})
            await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "assistant", "content": "Hello {{user}}"})
            await client.post(f"/api/v1/chats/{chat_id}/messages", headers=headers, json={"role": "system", "content": "sys note"})

            # GET messages with format_for_completions=true and context
            r = await client.get(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                params={"format_for_completions": True, "include_character_context": True}
            )
            assert r.status_code == 200
            data = r.json()
            assert data["pagination"] == {
                "mode": "offset",
                "limit": 50,
                "offset": 0,
                "total": 3,
                "has_more": False,
                "next_offset": None,
            }
            assert data["has_more"] is False
            assert data["next_offset"] is None
            roles = [m["role"] for m in data["messages"]]
            assert "system" in roles
            assert set(roles).issubset({"user", "assistant", "system", "tool"})

            # Search messages: verify placeholder replacement in response content
            r = await client.get(
                f"/api/v1/chats/{chat_id}/messages/search",
                headers=headers,
                params={"query": "Hello", "limit": 10}
            )
            assert r.status_code == 200
            res = r.json()
            assert any(m.get("content") == "Hello User" for m in res.get("messages", []))
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_injects_author_note_from_chat_settings(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_author_note_shared_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            char = r.json()[0]
            character_id = char["id"]
            character_name = char.get("name") or "Assistant"

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            settings_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "authorNote": "Keep responses concise for {{user}} while staying in character as {{char}}.",
                    "memoryScope": "shared",
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=settings_payload,
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "test",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200
            payload_messages = captured.get("messages") or []
            assert payload_messages, "Expected complete-v2 to forward messages to provider"
            first = payload_messages[0]
            assert first.get("role") == "system"
            content = first.get("content") or ""
            assert content.startswith("Author's note:")
            assert "User" in content
            assert character_name in content
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_roundtrip_persists_author_note_and_position(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_author_note_settings_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            expected_note = "Persist this author note for server-backed chat settings."
            expected_position = {"mode": "depth", "depth": 2}
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "authorNote": expected_note,
                    "authorNotePosition": expected_position,
                }
            }

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert r.status_code == 200

            r = await client.get(f"/api/v1/chats/{chat_id}/settings", headers=headers)
            assert r.status_code == 200
            settings_payload = r.json().get("settings") or {}
            assert settings_payload.get("authorNote") == expected_note
            assert settings_payload.get("authorNotePosition") == expected_position
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_roundtrip_persists_group_scope_fields(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_scope_settings_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "greetingScope": "character",
                    "presetScope": "chat",
                    "memoryScope": "both",
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert r.status_code == 200

            r = await client.get(f"/api/v1/chats/{chat_id}/settings", headers=headers)
            assert r.status_code == 200
            settings_payload = r.json().get("settings") or {}
            assert settings_payload.get("greetingScope") == "character"
            assert settings_payload.get("presetScope") == "chat"
            assert settings_payload.get("memoryScope") == "both"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_session_include_settings_for_detail_and_list(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_include_settings_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "greetingEnabled": True,
                        "authorNote": "include-settings-test",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.get(
                f"/api/v1/chats/{chat_id}",
                headers=headers,
                params={"include_settings": True},
            )
            assert r.status_code == 200
            detail_payload = r.json()
            assert isinstance(detail_payload.get("settings"), dict)
            assert detail_payload["settings"].get("greetingEnabled") is True
            assert detail_payload["settings"].get("authorNote") == "include-settings-test"

            r = await client.get(
                "/api/v1/chats/",
                headers=headers,
                params={"include_settings": True},
            )
            assert r.status_code == 200
            chats = r.json().get("chats") or []
            target = next((c for c in chats if c.get("id") == chat_id), None)
            assert target is not None
            assert isinstance(target.get("settings"), dict)
            assert target["settings"].get("greetingEnabled") is True
            assert target["settings"].get("authorNote") == "include-settings-test"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_server_wins_when_updated_at_equal(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_settings_server_wins_equal_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        shared_updated_at = "2026-02-06T00:00:00Z"
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            server_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": shared_updated_at,
                    "authorNote": "server-value",
                    "memoryScope": "shared",
                }
            }
            r = await client.put(f"/api/v1/chats/{chat_id}/settings", headers=headers, json=server_payload)
            assert r.status_code == 200

            stale_client_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": shared_updated_at,
                    "authorNote": "client-value-should-not-win",
                    "memoryScope": "both",
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=stale_client_payload,
            )
            assert r.status_code == 200

            merged = r.json().get("settings") or {}
            assert merged.get("authorNote") == "server-value"
            assert merged.get("memoryScope") == "shared"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_untimestamped_update_applies(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_settings_untimestamped_update_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        baseline_updated_at = "2026-02-06T00:00:00Z"
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            baseline_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": baseline_updated_at,
                    "authorNote": "server-value",
                    "memoryScope": "shared",
                }
            }
            r = await client.put(f"/api/v1/chats/{chat_id}/settings", headers=headers, json=baseline_payload)
            assert r.status_code == 200

            patch_payload = {
                "settings": {
                    "authorNote": "client-value-without-updated-at",
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=patch_payload,
            )
            assert r.status_code == 200

            merged = r.json().get("settings") or {}
            assert merged.get("authorNote") == "client-value-without-updated-at"
            assert merged.get("memoryScope") == "shared"
            assert merged.get("updatedAt") != baseline_updated_at
            assert datetime.fromisoformat(merged.get("updatedAt").replace("Z", "+00:00"))

            r = await client.get(f"/api/v1/chats/{chat_id}/settings", headers=headers)
            assert r.status_code == 200
            stored = r.json().get("settings") or {}
            assert stored.get("authorNote") == "client-value-without-updated-at"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_update_increments_conversation_version_once(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_settings_version_single_bump_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.get(f"/api/v1/chats/{chat_id}", headers=headers)
            assert r.status_code == 200
            before_version = int(r.json()["version"])

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-02-06T00:00:00Z",
                        "authorNote": "version-bump-check",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.get(f"/api/v1/chats/{chat_id}", headers=headers)
            assert r.status_code == 200
            after_first_update_version = int(r.json()["version"])
            assert after_first_update_version == before_version + 1

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={"settings": {"authorNote": "version-bump-check-2"}},
            )
            assert r.status_code == 200

            r = await client.get(f"/api/v1/chats/{chat_id}", headers=headers)
            assert r.status_code == 200
            after_second_update_version = int(r.json()["version"])
            assert after_second_update_version == after_first_update_version + 1
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_character_create_rejects_decodable_non_image_base64(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="character_create_non_image_reject_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        non_image_b64 = base64.b64encode(b"not-real-image-binary").decode("utf-8")
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Create Invalid Image", "image_base64": non_image_b64},
            )
            assert r.status_code == 400
            assert "not a valid image" in str(r.json().get("detail", "")).lower()
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_character_update_rejects_decodable_non_image_base64(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="character_update_non_image_reject_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        non_image_b64 = base64.b64encode(b"not-real-image-binary").decode("utf-8")
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Update Invalid Image Base"},
            )
            assert r.status_code == 201
            character = r.json()
            character_id = character["id"]
            original_version = int(character["version"])

            r = await client.put(
                f"/api/v1/characters/{character_id}",
                headers=headers,
                params={"expected_version": original_version},
                json={"image_base64": non_image_b64},
            )
            assert r.status_code == 400
            assert "not a valid image" in str(r.json().get("detail", "")).lower()

            r = await client.get(f"/api/v1/characters/{character_id}", headers=headers)
            assert r.status_code == 200
            assert int(r.json()["version"]) == original_version
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_character_update_accepts_valid_image_base64(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="character_update_valid_image_accept_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
        valid_png_1x1_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Update Valid Image Base"},
            )
            assert r.status_code == 201
            character = r.json()
            character_id = character["id"]
            original_version = int(character["version"])

            r = await client.put(
                f"/api/v1/characters/{character_id}",
                headers=headers,
                params={"expected_version": original_version},
                json={"image_base64": valid_png_1x1_b64},
            )
            assert r.status_code == 200
            payload = r.json()
            assert payload.get("image_present") is True
            assert int(payload["version"]) == original_version + 1
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_character_memory_merge_prefers_newer_entry(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_settings_memory_merge_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character_id = str(r.json()[0]["id"])

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Memory Merge Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = str(
                secondary_character.get("id") or secondary_character.get("character_id")
            )

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": int(primary_character_id)},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            first_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-02-06T00:00:00Z",
                    "characterMemoryById": {
                        primary_character_id: {
                            "note": "primary-server-note",
                            "updatedAt": "2026-02-06T00:00:00Z",
                        },
                        secondary_character_id: {
                            "note": "secondary-server-note",
                            "updatedAt": "2026-02-06T00:00:00Z",
                        },
                    },
                }
            }
            r = await client.put(f"/api/v1/chats/{chat_id}/settings", headers=headers, json=first_payload)
            assert r.status_code == 200

            second_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-02-06T00:00:00Z",
                    "characterMemoryById": {
                        primary_character_id: {
                            "note": "primary-client-older-note",
                            "updatedAt": "2026-02-05T23:59:59Z",
                        },
                        secondary_character_id: {
                            "note": "secondary-client-newer-note",
                            "updatedAt": "2026-02-06T00:00:01Z",
                        },
                    },
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=second_payload,
            )
            assert r.status_code == 200

            merged = r.json().get("settings") or {}
            memory_by_id = merged.get("characterMemoryById") or {}
            assert memory_by_id.get(primary_character_id, {}).get("note") == "primary-server-note"
            assert memory_by_id.get(secondary_character_id, {}).get("note") == "secondary-client-newer-note"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_invalid_scope_rejected(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_settings_invalid_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "greetingScope": "not-a-valid-scope",
                    }
                },
            )
            assert r.status_code == 422
            detail = str(r.json().get("detail") or "")
            assert "greetingScope" in detail
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_round_robin_turn_taking_persists_participant_senders(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_round_robin_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        call_count = {"value": 0}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            return {"choices": [{"message": {"content": f"ok-{call_count['value']}"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]
            primary_character_name = primary_character.get("name") or "Assistant"

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "RoundRobin Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = secondary_character.get("id") or secondary_character.get("character_id")
            assert secondary_character_id is not None
            secondary_character_name = secondary_character.get("name") or "RoundRobin Secondary"

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [secondary_character_id],
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            payload1 = r.json()
            assert payload1.get("character_id") == primary_character_id
            assert payload1.get("speaker_character_id") == primary_character_id
            assert payload1.get("speaker_character_name") == primary_character_name

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-2",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            payload2 = r.json()
            assert payload2.get("character_id") == secondary_character_id
            assert payload2.get("speaker_character_id") == secondary_character_id
            assert payload2.get("speaker_character_name") == secondary_character_name

            r = await client.get(f"/api/v1/chats/{chat_id}/messages", headers=headers)
            assert r.status_code == 200
            messages = r.json().get("messages") or []
            assistant_senders = [
                msg.get("sender")
                for msg in messages
                if msg.get("sender") not in {"user", "system"}
            ]
            assert assistant_senders[-2:] == [primary_character_name, secondary_character_name]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_directed_character_override_and_round_robin_continuation(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_directed_round_robin_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        call_count = {"value": 0}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            return {"choices": [{"message": {"content": f"ok-{call_count['value']}"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Directed Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = secondary_character.get("id") or secondary_character.get("character_id")
            assert secondary_character_id is not None

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [secondary_character_id],
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "directed_character_id": secondary_character_id,
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            payload1 = r.json()
            assert payload1.get("speaker_character_id") == secondary_character_id

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-2",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            payload2 = r.json()
            assert payload2.get("speaker_character_id") == primary_character_id
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_rejects_directed_character_not_in_participants(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_directed_invalid_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        call_count = {"value": 0}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Directed Included"},
            )
            assert r.status_code == 201
            included_character = r.json()
            included_character_id = included_character.get("id") or included_character.get("character_id")
            assert included_character_id is not None

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Directed Excluded"},
            )
            assert r.status_code == 201
            excluded_character = r.json()
            excluded_character_id = excluded_character.get("id") or excluded_character.get("character_id")
            assert excluded_character_id is not None

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [included_character_id],
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "directed_character_id": excluded_character_id,
                    "save_to_db": False,
                },
            )
            assert r.status_code == 400
            detail = r.json().get("detail")
            assert detail == "directed_character_id must reference a selected participant in this chat"
            assert call_count["value"] == 0
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_memory_scope_character_uses_directed_speaker_note(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_directed_memory_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Memory Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = secondary_character.get("id") or secondary_character.get("character_id")
            assert secondary_character_id is not None

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "participantCharacterIds": [secondary_character_id],
                        "memoryScope": "character",
                        "authorNote": "shared-note-should-not-appear",
                        "characterMemoryById": {
                            str(primary_character_id): {
                                "note": "primary-note-should-not-appear",
                                "updatedAt": datetime.now(timezone.utc).isoformat(),
                            },
                            str(secondary_character_id): {
                                "note": "secondary-note-should-appear",
                                "updatedAt": datetime.now(timezone.utc).isoformat(),
                            },
                        },
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "directed_character_id": secondary_character_id,
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200
            payload_messages = captured.get("messages") or []
            author_note_messages = [
                message
                for message in payload_messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content", "").startswith("Author's note:")
            ]
            assert len(author_note_messages) == 1
            note_content = author_note_messages[0]["content"]
            assert "secondary-note-should-appear" in note_content
            assert "primary-note-should-not-appear" not in note_content
            assert "shared-note-should-not-appear" not in note_content
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_character_greeting_scope_injects_per_participant_first_turn(
    monkeypatch,
):
    tmpdir = tempfile.mkdtemp(prefix="chacha_character_greeting_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured_payloads: list[list[dict[str, object]]] = []
        call_count = {"value": 0}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            captured_payloads.append(messages_payload)
            return {"choices": [{"message": {"content": f"ok-{call_count['value']}"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]
            primary_character_version = primary_character["version"]

            primary_greeting = "primary-stage-i2-greeting-token"
            r = await client.put(
                f"/api/v1/characters/{primary_character_id}",
                headers=headers,
                params={"expected_version": primary_character_version},
                json={"first_message": primary_greeting},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Greeting Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = (
                secondary_character.get("id") or secondary_character.get("character_id")
            )
            secondary_character_version = secondary_character.get("version")
            assert secondary_character_id is not None
            assert secondary_character_version is not None

            secondary_greeting = "secondary-stage-i2-greeting-token"
            r = await client.put(
                f"/api/v1/characters/{secondary_character_id}",
                headers=headers,
                params={"expected_version": secondary_character_version},
                json={"first_message": secondary_greeting},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "greetingScope": "character",
                        "greetingEnabled": True,
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [secondary_character_id],
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            assert r.json().get("speaker_character_id") == primary_character_id

            payload_messages_1 = captured_payloads[0]
            assistant_contents_1 = [
                str(message.get("content") or "")
                for message in payload_messages_1
                if message.get("role") == "assistant"
            ]
            assert any(primary_greeting in content for content in assistant_contents_1)
            assert not any(
                secondary_greeting in content for content in assistant_contents_1
            )

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-2",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            assert r.json().get("speaker_character_id") == secondary_character_id

            payload_messages_2 = captured_payloads[1]
            assistant_contents_2 = [
                str(message.get("content") or "")
                for message in payload_messages_2
                if message.get("role") == "assistant"
            ]
            assert any(secondary_greeting in content for content in assistant_contents_2)
            assert not any(primary_greeting in content for content in assistant_contents_2)
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_chat_greeting_scope_injects_once_for_group_chat(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_chat_greeting_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured_payloads: list[list[dict[str, object]]] = []
        call_count = {"value": 0}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            captured_payloads.append(messages_payload)
            return {"choices": [{"message": {"content": f"ok-{call_count['value']}"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]
            primary_character_version = primary_character["version"]

            primary_greeting = "primary-chat-scope-greeting-token"
            r = await client.put(
                f"/api/v1/characters/{primary_character_id}",
                headers=headers,
                params={"expected_version": primary_character_version},
                json={"first_message": primary_greeting},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Greeting Secondary Chat Scope"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = (
                secondary_character.get("id") or secondary_character.get("character_id")
            )
            secondary_character_version = secondary_character.get("version")
            assert secondary_character_id is not None
            assert secondary_character_version is not None

            secondary_greeting = "secondary-chat-scope-greeting-token"
            r = await client.put(
                f"/api/v1/characters/{secondary_character_id}",
                headers=headers,
                params={"expected_version": secondary_character_version},
                json={"first_message": secondary_greeting},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "greetingScope": "chat",
                        "greetingEnabled": True,
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [secondary_character_id],
                    }
                },
            )
            assert r.status_code == 200

            for turn in ("turn-1", "turn-2"):
                r = await client.post(
                    f"/api/v1/chats/{chat_id}/complete-v2",
                    headers=headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-x",
                        "append_user_message": turn,
                        "save_to_db": True,
                    },
                )
                assert r.status_code == 200

            first_call_assistant = [
                str(message.get("content") or "")
                for message in captured_payloads[0]
                if message.get("role") == "assistant"
            ]
            second_call_assistant = [
                str(message.get("content") or "")
                for message in captured_payloads[1]
                if message.get("role") == "assistant"
            ]

            assert any(primary_greeting in content for content in first_call_assistant)
            assert not any(
                primary_greeting in content or secondary_greeting in content
                for content in second_call_assistant
            )
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_character_greeting_scope_single_participant_injects_once(
    monkeypatch,
):
    tmpdir = tempfile.mkdtemp(prefix="chacha_character_greeting_single_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured_payloads: list[list[dict[str, object]]] = []
        call_count = {"value": 0}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            captured_payloads.append(messages_payload)
            return {"choices": [{"message": {"content": f"ok-{call_count['value']}"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]
            primary_character_version = primary_character["version"]

            greeting_token = "single-participant-greeting-token"
            r = await client.put(
                f"/api/v1/characters/{primary_character_id}",
                headers=headers,
                params={"expected_version": primary_character_version},
                json={"first_message": greeting_token},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "greetingScope": "character",
                        "greetingEnabled": True,
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-2",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200

            first_call_assistant = [
                str(message.get("content") or "")
                for message in captured_payloads[0]
                if message.get("role") == "assistant"
            ]
            second_call_assistant = [
                str(message.get("content") or "")
                for message in captured_payloads[1]
                if message.get("role") == "assistant"
            ]

            assert any(greeting_token in content for content in first_call_assistant)
            assert not any(greeting_token in content for content in second_call_assistant)
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_character_greeting_scope_respects_disabled_toggle(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_character_greeting_disabled_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]
            primary_character_version = primary_character["version"]

            greeting_token = "disabled-character-greeting-token"
            r = await client.put(
                f"/api/v1/characters/{primary_character_id}",
                headers=headers,
                params={"expected_version": primary_character_version},
                json={"first_message": greeting_token},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "greetingScope": "character",
                        "greetingEnabled": False,
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200
            payload_messages = captured.get("messages") or []
            assert not any(
                greeting_token in str(message.get("content") or "")
                for message in payload_messages
            )
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_applies_chat_generation_override(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_generation_override_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured_call_kwargs: dict[str, object] = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured_call_kwargs.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "chatGenerationOverride": {
                            "enabled": True,
                            "temperature": 1.2,
                            "top_p": 0.8,
                            "repetition_penalty": 1.1,
                            "stop": ["<END>"],
                        },
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            assert captured_call_kwargs.get("temp") == 1.2
            assert captured_call_kwargs.get("top_p") == 0.8
            assert captured_call_kwargs.get("repetition_penalty") == 1.1
            assert captured_call_kwargs.get("stop") == ["<END>"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_applies_preset_scope_chat_override(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_preset_scope_chat_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured_payloads: list[list[dict[str, object]]] = []

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured_payloads.append(messages_payload)
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]
            primary_character_version = primary_character["version"]

            r = await client.put(
                f"/api/v1/characters/{primary_character_id}",
                headers=headers,
                params={"expected_version": primary_character_version},
                json={"message_example": "preset-example-token"},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "presetScope": "chat",
                        "chatPresetOverrideId": "st_default",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-1",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            first_payload = captured_payloads[-1]
            first_system_message = next(
                (
                    str(message.get("content") or "")
                    for message in first_payload
                    if message.get("role") == "system"
                ),
                "",
            )
            assert "Example dialogue:" in first_system_message
            assert "preset-example-token" in first_system_message

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "presetScope": "character",
                        "chatPresetOverrideId": "st_default",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "turn-2",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            second_payload = captured_payloads[-1]
            second_system_message = next(
                (
                    str(message.get("content") or "")
                    for message in second_payload
                    if message.get("role") == "system"
                ),
                "",
            )
            assert "Example dialogue:" not in second_system_message
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_preview_reports_server_conflicts_and_steering_warning(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_prompt_preview_conflicts_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "authorNote": (
                            "temperature: 0.7. temperature: 0.2. "
                            "Speak tersely and be verbose."
                        ),
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/prompt-preview",
                headers=headers,
                json={
                    "include_character_context": True,
                    "continue_as_user": True,
                    "impersonate_user": True,
                    "force_narrate": False,
                },
            )
            assert r.status_code == 200
            payload = r.json()

            sections = payload.get("sections") or []
            assert sections
            for section in sections:
                assert "tokens_estimated" in section
                assert "tokens_effective" in section
                assert "budget" in section
                assert "truncated" in section

            by_name = {str(section.get("name")): section for section in sections}
            assert "message_steering" in by_name
            assert (by_name["message_steering"].get("tokens_effective") or 0) > 0

            warnings = payload.get("warnings") or []
            assert any(
                "impersonate_user overrides continue_as_user" in str(item)
                for item in warnings
            )

            conflicts = payload.get("conflicts") or []
            conflict_types = {str(item.get("type")) for item in conflicts}
            assert "scalar_conflict" in conflict_types
            assert "directive_conflict" in conflict_types
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_preview_applies_preset_scope_chat_override(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_prompt_preview_preset_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            character_id = primary_character["id"]
            character_version = primary_character["version"]

            example_token = "preview-preset-example-token"
            r = await client.put(
                f"/api/v1/characters/{character_id}",
                headers=headers,
                params={"expected_version": character_version},
                json={"message_example": example_token},
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "presetScope": "chat",
                        "chatPresetOverrideId": "st_default",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/prompt-preview",
                headers=headers,
                json={"include_character_context": True},
            )
            assert r.status_code == 200
            preview_with_chat_scope = r.json()
            preset_section_chat_scope = next(
                (
                    section
                    for section in (preview_with_chat_scope.get("sections") or [])
                    if section.get("name") == "preset"
                ),
                {},
            )
            preset_content_chat_scope = str(preset_section_chat_scope.get("content") or "")
            assert "Example dialogue:" in preset_content_chat_scope
            assert example_token in preset_content_chat_scope

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "presetScope": "character",
                        "chatPresetOverrideId": "st_default",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/prompt-preview",
                headers=headers,
                json={"include_character_context": True},
            )
            assert r.status_code == 200
            preview_with_character_scope = r.json()
            preset_section_character_scope = next(
                (
                    section
                    for section in (preview_with_character_scope.get("sections") or [])
                    if section.get("name") == "preset"
                ),
                {},
            )
            preset_content_character_scope = str(
                preset_section_character_scope.get("content") or ""
            )
            assert "Example dialogue:" not in preset_content_character_scope
            assert example_token not in preset_content_character_scope
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_export_uses_participant_alias_for_tool_calls_inline_fallback(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_export_round_robin_tools_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        call_count = {"value": 0}
        inline_tool_call_id = "call_secondary_inline_1"

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 2:
                return (
                    "secondary-turn\n[tool_calls]: "
                    f'[{{"id":"{inline_tool_call_id}","type":"function","function":{{"name":"lookup","arguments":"{{}}"}}}}]'
                )
            return "primary-turn"

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "Export Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = secondary_character.get("id") or secondary_character.get("character_id")
            assert secondary_character_id is not None
            secondary_character_name = secondary_character.get("name") or "Export Secondary"

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [secondary_character_id],
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "first-turn",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "second-turn",
                    "save_to_db": True,
                },
            )
            assert r.status_code == 200
            second_completion = r.json()
            assert second_completion.get("speaker_character_name") == secondary_character_name
            secondary_message_id = second_completion.get("assistant_message_id")
            assert secondary_message_id

            r = await client.get(
                f"/api/v1/chats/{chat_id}/export",
                headers=headers,
                params={"format": "json", "include_metadata": True},
            )
            assert r.status_code == 200
            export_payload = r.json()
            exported_messages = export_payload.get("messages") or []
            secondary_exported = next(
                (
                    message
                    for message in exported_messages
                    if message.get("id") == secondary_message_id
                ),
                None,
            )
            assert secondary_exported is not None
            assert secondary_exported.get("role") == secondary_character_name
            tool_calls = secondary_exported.get("tool_calls") or []
            assert any(call.get("id") == inline_tool_call_id for call in tool_calls)
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_author_note_memory_scope_character_overrides_shared(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_author_note_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            settings_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "authorNote": "shared-note-should-not-appear",
                    "memoryScope": "character",
                    "characterMemoryById": {
                        str(character_id): {
                            "note": "character-note-should-appear",
                            "updatedAt": datetime.now(timezone.utc).isoformat(),
                        }
                    },
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=settings_payload,
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "test",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200
            payload_messages = captured.get("messages") or []
            author_note_messages = [
                message
                for message in payload_messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content", "").startswith("Author's note:")
            ]
            assert len(author_note_messages) == 1
            note_content = author_note_messages[0]["content"]
            assert "character-note-should-appear" in note_content
            assert "shared-note-should-not-appear" not in note_content
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_uses_character_default_author_note_when_chat_note_empty(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_author_note_default_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character = r.json()[0]
            character_id = character["id"]
            character_version = character["version"]

            default_note = "default-note-from-character-extensions"
            r = await client.put(
                f"/api/v1/characters/{character_id}",
                headers=headers,
                params={"expected_version": character_version},
                json={"extensions": {"default_author_note": default_note}},
            )
            assert r.status_code == 200

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "authorNote": "   ",
                        "memoryScope": "shared",
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "test",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            payload_messages = captured.get("messages") or []
            author_note_messages = [
                message
                for message in payload_messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content", "").startswith("Author's note:")
            ]
            assert len(author_note_messages) == 1
            assert default_note in author_note_messages[0]["content"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_inserts_author_note_at_depth_position(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_author_note_depth_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "seed_first_message": False},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "first-user"},
            )
            assert r.status_code == 201
            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "assistant", "content": "first-assistant"},
            )
            assert r.status_code == 201

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "authorNote": "depth-note",
                        "authorNotePosition": {"mode": "depth", "depth": 1},
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "include_character_context": False,
                    "append_user_message": "latest-user",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            payload_messages = captured.get("messages") or []
            assert len(payload_messages) >= 4
            author_note_index = next(
                idx
                for idx, message in enumerate(payload_messages)
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content", "").startswith("Author's note:")
            )
            assert author_note_index == 1
            assert payload_messages[0].get("role") == "user"
            assert payload_messages[2].get("role") == "assistant"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prepare_completions_includes_author_note_from_settings(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_author_note_prepare_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            settings_payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "authorNote": "prep-note-should-appear",
                }
            }
            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=settings_payload,
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/completions",
                headers=headers,
                json={"include_character_context": False},
            )
            assert r.status_code == 200
            messages = r.json()["messages"]
            assert messages
            assert messages[0]["role"] == "system"
            assert messages[0]["content"].startswith("Author's note:")
            assert "prep-note-should-appear" in messages[0]["content"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_injects_message_steering_instruction(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_steering_complete_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "include_character_context": False,
                    "append_user_message": "test",
                    "continue_as_user": True,
                    "impersonate_user": True,
                    "force_narrate": True,
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200
            payload_messages = captured.get("messages") or []
            assert payload_messages

            steering_messages = [
                message
                for message in payload_messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message["content"].startswith("Steering instruction (single response):")
            ]
            assert len(steering_messages) == 1
            steering = steering_messages[0]["content"]
            assert "authored by the user" in steering
            assert "Continue the user's current thought" not in steering
            assert "narrative prose style" in steering

            steering_index = payload_messages.index(steering_messages[0])
            user_index = next(
                (
                    i
                    for i, message in enumerate(payload_messages)
                    if message.get("role") == "user"
                    and message.get("content") == "test"
                ),
                -1,
            )
            assert user_index > -1
            assert steering_index < user_index
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prepare_completions_includes_message_steering_instruction(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_steering_prepare_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/completions",
                headers=headers,
                json={
                    "include_character_context": False,
                    "append_user_message": "prep-msg",
                    "continue_as_user": True,
                    "force_narrate": True,
                },
            )
            assert r.status_code == 200
            messages = r.json()["messages"]
            assert messages

            steering_message = next(
                (
                    message
                    for message in messages
                    if message.get("role") == "system"
                    and isinstance(message.get("content"), str)
                    and message.get("content", "").startswith(
                        "Steering instruction (single response):"
                    )
                ),
                None,
            )
            assert steering_message is not None
            assert "Continue the user's current thought" in steering_message["content"]
            assert "narrative prose style" in steering_message["content"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_applies_character_generation_defaults_when_request_omits_fields(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_generation_defaults_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            captured["kwargs"] = kwargs
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character = r.json()[0]
            character_id = character["id"]
            character_version = character["version"]

            update_payload = {
                "extensions": {
                    "tldw": {
                        "generation": {
                            "temperature": 0.61,
                            "top_p": 0.83,
                            "repetition_penalty": 1.07,
                            "stop": ["###", "END"],
                        }
                    }
                }
            }
            r = await client.put(
                f"/api/v1/characters/{character_id}",
                headers=headers,
                params={"expected_version": character_version},
                json=update_payload,
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "test",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            kwargs = captured.get("kwargs") or {}
            assert kwargs.get("temp") == pytest.approx(0.61)
            assert kwargs.get("top_p") == pytest.approx(0.83)
            assert kwargs.get("repetition_penalty") == pytest.approx(1.07)
            assert kwargs.get("stop") == ["###", "END"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_request_generation_fields_override_character_defaults(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_generation_override_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            captured["kwargs"] = kwargs
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character = r.json()[0]
            character_id = character["id"]
            character_version = character["version"]

            update_payload = {
                "extensions": {
                    "tldw": {
                        "generation": {
                            "temperature": 0.2,
                            "top_p": 0.3,
                            "repetition_penalty": 1.01,
                            "stop": ["DEFAULT_STOP"],
                        }
                    }
                }
            }
            r = await client.put(
                f"/api/v1/characters/{character_id}",
                headers=headers,
                params={"expected_version": character_version},
                json=update_payload,
            )
            assert r.status_code == 200

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "append_user_message": "test",
                    "save_to_db": False,
                    "temperature": 0.91,
                    "top_p": 0.44,
                    "repetition_penalty": 1.4,
                    "stop": ["OVERRIDE_STOP"],
                },
            )
            assert r.status_code == 200

            kwargs = captured.get("kwargs") or {}
            assert kwargs.get("temp") == pytest.approx(0.91)
            assert kwargs.get("top_p") == pytest.approx(0.44)
            assert kwargs.get("repetition_penalty") == pytest.approx(1.4)
            assert kwargs.get("stop") == ["OVERRIDE_STOP"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_auto_summary_injects_and_persists_to_chat_settings(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_auto_summary_persist_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "seed_first_message": False},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            seeded_messages = [
                ("user", "older-user-1"),
                ("assistant", "older-assistant-1"),
                ("user", "older-user-2"),
                ("assistant", "older-assistant-2"),
                ("user", "recent-user-1"),
                ("assistant", "recent-assistant-1"),
            ]
            for role, content in seeded_messages:
                rr = await client.post(
                    f"/api/v1/chats/{chat_id}/messages",
                    headers=headers,
                    json={"role": role, "content": content},
                )
                assert rr.status_code == 201

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "autoSummaryEnabled": True,
                        "autoSummaryThresholdMessages": 4,
                        "autoSummaryWindowMessages": 2,
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "include_character_context": False,
                    "append_user_message": "latest-user",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            payload_messages = captured.get("messages") or []
            summary_messages = [
                message
                for message in payload_messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content", "").startswith("Conversation summary:")
            ]
            assert len(summary_messages) == 1
            assert "older-user-1" in summary_messages[0]["content"]
            assert "older-assistant-2" in summary_messages[0]["content"]

            r = await client.get(f"/api/v1/chats/{chat_id}/settings", headers=headers)
            assert r.status_code == 200
            summary_payload = (r.json().get("settings") or {}).get("summary")
            assert isinstance(summary_payload, dict)
            assert isinstance(summary_payload.get("content"), str)
            assert summary_payload["content"].strip()
            source_range = summary_payload.get("sourceRange")
            assert isinstance(source_range, dict)
            assert source_range.get("toMessageId")
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_auto_summary_excludes_pinned_messages_from_compression(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_auto_summary_pinned_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured = {}

        import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as mod

        def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
            captured["messages"] = messages_payload
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(mod, "perform_chat_api_call", _stub_chat_api_call)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "seed_first_message": False},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            message_ids: dict[str, tuple[str, int]] = {}
            seeded_messages = [
                ("user", "PINNED-OLDER-USER"),
                ("assistant", "older-assistant-1"),
                ("user", "older-user-2"),
                ("assistant", "older-assistant-2"),
                ("user", "recent-user-1"),
                ("assistant", "recent-assistant-1"),
            ]
            for idx, (role, content) in enumerate(seeded_messages):
                rr = await client.post(
                    f"/api/v1/chats/{chat_id}/messages",
                    headers=headers,
                    json={"role": role, "content": content},
                )
                assert rr.status_code == 201
                body = rr.json()
                message_ids[f"{role}:{idx}"] = (body["id"], body["version"])

            pinned_message_id, pinned_version = message_ids["user:0"]
            r = await client.put(
                f"/api/v1/messages/{pinned_message_id}",
                headers=headers,
                params={"expected_version": pinned_version},
                json={"content": "PINNED-OLDER-USER", "pinned": True},
            )
            assert r.status_code == 200

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "autoSummaryEnabled": True,
                        "autoSummaryThresholdMessages": 4,
                        "autoSummaryWindowMessages": 2,
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-x",
                    "include_character_context": False,
                    "append_user_message": "latest-user",
                    "save_to_db": False,
                },
            )
            assert r.status_code == 200

            payload_messages = captured.get("messages") or []
            summary_messages = [
                message
                for message in payload_messages
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message.get("content", "").startswith("Conversation summary:")
            ]
            assert len(summary_messages) == 1
            summary_text = summary_messages[0]["content"]

            assert "PINNED-OLDER-USER" not in summary_text
            assert any(
                message.get("role") == "user" and message.get("content") == "PINNED-OLDER-USER"
                for message in payload_messages
            )
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edit_message_pinned_updates_message_metadata(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_pin_message_metadata_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id, "seed_first_message": False},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/messages",
                headers=headers,
                json={"role": "user", "content": "pin-me"},
            )
            assert r.status_code == 201
            message_id = r.json()["id"]
            expected_version = r.json()["version"]

            r = await client.put(
                f"/api/v1/messages/{message_id}",
                headers=headers,
                params={"expected_version": expected_version},
                json={"pinned": True},
            )
            assert r.status_code == 200

            r = await client.get(
                f"/api/v1/messages/{message_id}",
                headers=headers,
                params={"include_metadata": "true"},
            )
            assert r.status_code == 200
            payload = r.json()
            metadata_extra = payload.get("metadata_extra")
            assert isinstance(metadata_extra, dict)
            assert metadata_extra.get("pinned") is True
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_v2_world_book_uses_active_directed_character_id(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_world_book_active_character_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        captured: dict[str, object] = {}

        import tldw_Server_API.app.core.Character_Chat.world_book_manager as wb_mod

        class _StubWorldBookService:
            def __init__(self, db):
                self._db = db

            def process_context(self, text, world_book_ids=None, character_id=None, scan_depth=3, token_budget=500, recursive_scanning=False, **kwargs):
                captured["character_id"] = character_id
                captured["include_diagnostics"] = kwargs.get("include_diagnostics")
                return {"processed_context": "", "diagnostics": []}

        monkeypatch.setattr(wb_mod, "WorldBookService", _StubWorldBookService)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            primary_character = r.json()[0]
            primary_character_id = primary_character["id"]

            r = await client.post(
                "/api/v1/characters/",
                headers=headers,
                json={"name": "WorldBook Secondary"},
            )
            assert r.status_code == 201
            secondary_character = r.json()
            secondary_character_id = secondary_character.get("id") or secondary_character.get("character_id")
            assert secondary_character_id is not None

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": primary_character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "turnTakingMode": "round_robin",
                        "participantCharacterIds": [secondary_character_id],
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "local-llm",
                    "model": "local-test",
                    "append_user_message": "trigger-world-book",
                    "save_to_db": False,
                    "directed_character_id": secondary_character_id,
                },
            )
            assert r.status_code == 200
            payload = r.json()
            assert payload.get("speaker_character_id") == secondary_character_id
            assert captured.get("character_id") == secondary_character_id
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_preview_applies_custom_chat_preset_override(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_custom_preset_chat_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)

        custom_preset_id = "custom_chat_scope_apply"
        marker = "CUSTOM-PRESET-MARKER"

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/chats/presets",
                headers=headers,
                json={
                    "preset_id": custom_preset_id,
                    "name": "Custom Chat Scope Apply",
                    "section_order": ["identity"],
                    "section_templates": {"identity": f"{marker}: {{char}} -> {{user}}"},
                },
            )
            assert r.status_code == 201

            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "presetScope": "chat",
                        "chatPresetOverrideId": custom_preset_id,
                    }
                },
            )
            assert r.status_code == 200

            r = await client.post(
                f"/api/v1/chats/{chat_id}/prompt-preview",
                headers=headers,
                json={"include_character_context": True},
            )
            assert r.status_code == 200
            preview = r.json()

            preset_section = next(
                (
                    section
                    for section in (preview.get("sections") or [])
                    if section.get("name") == "preset"
                ),
                {},
            )
            preset_content = str(preset_section.get("content") or "")
            assert marker in preset_content
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prompt_preview_applies_custom_request_preset_override(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_custom_preset_request_scope_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)

        custom_preset_id = "custom_request_scope_apply"
        marker = "REQUEST-PRESET-MARKER"

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/chats/presets",
                headers=headers,
                json={
                    "preset_id": custom_preset_id,
                    "name": "Custom Request Scope Apply",
                    "section_order": ["identity"],
                    "section_templates": {"identity": f"{marker}: {{char}}"},
                },
            )
            assert r.status_code == 201

            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]

            r = await client.post(
                "/api/v1/chats/",
                headers=headers,
                json={"character_id": character_id},
            )
            assert r.status_code == 201
            chat_id = r.json()["id"]

            r = await client.post(
                f"/api/v1/chats/{chat_id}/prompt-preview",
                headers=headers,
                json={"include_character_context": True, "prompt_preset": custom_preset_id},
            )
            assert r.status_code == 200
            preview = r.json()

            preset_section = next(
                (
                    section
                    for section in (preview.get("sections") or [])
                    if section.get("name") == "preset"
                ),
                {},
            )
            preset_content = str(preset_section.get("content") or "")
            assert marker in preset_content
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)
