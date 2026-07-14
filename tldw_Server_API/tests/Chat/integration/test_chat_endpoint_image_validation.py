"""
Endpoint-level integration tests for image-aware request size and image validation behavior.

Scenarios:
- Large data:image payloads are accepted by request-size validator (redaction) and request succeeds.
- When image size limit is configured too small, image validation fails and the saved user message
  contains a clear placeholder indicating the failure.
"""

import os
import json
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, DEFAULT_CHARACTER_NAME


@pytest.mark.unit
def test_chat_endpoint_large_data_image_accepted_by_size_and_flagged_by_image_validation(monkeypatch):
    # Configure strict request JSON size but allow endpoint to accept due to redaction
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", "1000")  # tight cap
    monkeypatch.setenv("CHAT_IMAGE_MAX_MB", "1")  # small image limit to trigger validation failure
    # Explicitly disable base64 enforcement to preserve the redaction-only path
    monkeypatch.setenv("CHAT_ENFORCE_BASE64_IMAGE_LIMIT", "0")

    # Create temp DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client")
    # Enable WAL mode and a generous busy timeout for concurrent test access
    try:
        conn = db.get_connection()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
    except Exception:
        _ = None
    try:
        # Add default character required by chat endpoint
        db.add_character_card(
            {
                "name": DEFAULT_CHARACTER_NAME,
                "description": "Default",
                "personality": "Helpful",
                "scenario": "Testing",
                "system_prompt": "You are helpful",
                "first_message": "Hello",
                "creator_notes": "test",
            }
        )

        with TestClient(app) as client:
            # CSRF token
            resp = client.get("/api/v1/health")
            csrf = resp.cookies.get("csrf_token", "")
            client.csrf_token = csrf

            # Override DB dependency
            _app = app
            _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

            # Ensure auth header
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings

            settings = get_settings()
            api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
            headers = {"X-API-KEY": api_key, "X-CSRF-Token": csrf}

            # Large base64 body to exceed 1k request size without redaction; redaction should allow it
            # Exceed base64 string threshold for 1 MB (set above)
            too_long_b64_len = int(1 * 1024 * 1024 * 4 / 3) + 200
            big_b64 = "A" * too_long_b64_len
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "save_to_db": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please analyze this image"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
                        ],
                    }
                ],
            }

            # Mock provider call to avoid external dependency
            mock_response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Processed"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

            with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=mock_response):
                r = client.post("/api/v1/chat/completions", json=body, headers=headers)
                assert r.status_code == 200, f"Unexpected status: {r.status_code}, {r.text}"
                data = r.json()
                assert isinstance(data, dict)
                conv_id = data.get("tldw_conversation_id")
                assert conv_id, "Expected conversation id in response"

                # Verify the user message was persisted with an image validation placeholder
                # (image validation failed due to CHAT_IMAGE_MAX_MB=1)
                # Load messages; history order for DB lookup is newest first in helper, we fetch directly via DB
                msgs = db.get_messages_for_conversation(conv_id, 50, 0, "ASC")
                # There should be at least one user message (with placeholder) and one assistant message
                assert len(msgs) >= 1
                found_placeholder = any(
                    isinstance(m.get("content"), str)
                    and ("<Image failed" in m.get("content") or "<Image failed validation" in m.get("content"))
                    for m in msgs
                    if m.get("sender", "").lower() == "user"
                )
                assert found_placeholder, "Expected user message to contain image validation placeholder"
    finally:
        # Cleanup db files
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        # Restore overrides
        try:
            getattr(app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_chat_endpoint_rejects_oversized_image_by_default(monkeypatch):
    # Make enforcement deterministic regardless of config defaults
    monkeypatch.setenv("CHAT_ENFORCE_BASE64_IMAGE_LIMIT", "1")
    monkeypatch.setenv("CHAT_IMAGE_MAX_MB", "1")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client_enforce")
    try:
        db.add_character_card(
            {
                "name": DEFAULT_CHARACTER_NAME,
                "description": "Default",
                "personality": "Helpful",
                "scenario": "Testing",
                "system_prompt": "You are helpful",
                "first_message": "Hello",
                "creator_notes": "test",
            }
        )

        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            csrf = resp.cookies.get("csrf_token", "")
            client.csrf_token = csrf

            _app = app
            _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

            from tldw_Server_API.app.core.AuthNZ.settings import get_settings

            settings = get_settings()
            api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
            headers = {"X-API-KEY": api_key, "X-CSRF-Token": csrf}

            too_long_b64_len = int(1 * 1024 * 1024 * 4 / 3) + 200
            big_b64 = "A" * too_long_b64_len
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "save_to_db": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please analyze this image"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
                        ],
                    }
                ],
            }

            r = client.post("/api/v1/chat/completions", json=body, headers=headers)
            assert r.status_code == 413, f"Unexpected status: {r.status_code}, {r.text}"
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        try:
            getattr(app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_chat_endpoint_request_size_returns_413(monkeypatch):
    monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", "200")
    from tldw_Server_API.app.api.v1.schemas import chat_validators
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 200)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client_req_size")
    try:
        db.add_character_card(
            {
                "name": DEFAULT_CHARACTER_NAME,
                "description": "Default",
                "personality": "Helpful",
                "scenario": "Testing",
                "system_prompt": "You are helpful",
                "first_message": "Hello",
                "creator_notes": "test",
            }
        )

        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            csrf = resp.cookies.get("csrf_token", "")
            client.csrf_token = csrf

            _app = app
            _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

            from tldw_Server_API.app.core.AuthNZ.settings import get_settings

            settings = get_settings()
            api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
            headers = {"X-API-KEY": api_key, "X-CSRF-Token": csrf}

            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "save_to_db": True,
                "messages": [
                    {
                        "role": "user",
                        "content": "X" * 500,
                    }
                ],
            }

            r = client.post("/api/v1/chat/completions", json=body, headers=headers)
            assert r.status_code == 413, f"Unexpected status: {r.status_code}, {r.text}"
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        try:
            getattr(app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_chat_endpoint_streaming_large_data_image_placeholder_in_db(monkeypatch):
    # Configure strict request JSON size but allow endpoint to accept due to redaction
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", "1000")  # tight cap
    monkeypatch.setenv("CHAT_IMAGE_MAX_MB", "1")  # small image limit to trigger validation failure

    # Create temp DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client_stream")
    try:
        conn = db.get_connection()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()
    except Exception:
        _ = None
    try:
        # Add default character required by chat endpoint
        db.add_character_card(
            {
                "name": DEFAULT_CHARACTER_NAME,
                "description": "Default",
                "personality": "Helpful",
                "scenario": "Testing",
                "system_prompt": "You are helpful",
                "first_message": "Hello",
                "creator_notes": "test",
            }
        )

        with TestClient(app) as client:
            # CSRF token
            resp = client.get("/api/v1/health")
            csrf = resp.cookies.get("csrf_token", "")
            client.csrf_token = csrf

            # Override DB dependency
            _app = app
            _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

            # Auth header
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings

            settings = get_settings()
            api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
            headers = {"X-API-KEY": api_key, "X-CSRF-Token": csrf}

            too_long_b64_len = int(1 * 1024 * 1024 * 4 / 3) + 200
            big_b64 = "A" * too_long_b64_len
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "save_to_db": True,
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please analyze this image"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
                        ],
                    }
                ],
            }

            # Upstream mock stream (OpenAI-like SSE) with two chunks and DONE
            chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
            chunk2 = {"choices": [{"delta": {"content": " world"}}]}

            def upstream_stream():
                yield f"data: {json.dumps(chunk1)}\n\n"
                yield f"data: {json.dumps(chunk2)}\n\n"
                yield "data: [DONE]\n\n"

            with patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()
            ):
                r = client.post("/api/v1/chat/completions", json=body, headers=headers)
                assert r.status_code == 200, f"Unexpected status: {r.status_code}"
                ctype = r.headers.get("content-type", "").lower()
                assert "text/event-stream" in ctype

                lines = r.text.splitlines()
                # Find conversation_id in the normalized stream metadata.
                conv_id = None
                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    payload = None
                    try:
                        payload = json.loads(line[6:].strip())
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        conv_id = payload.get("conversation_id") or payload.get("tldw_conversation_id")
                        if conv_id:
                            break
                assert conv_id, "Expected conversation_id in normalized stream metadata"

                # Verify normalized chunks appear
                normalized = []
                for line in lines:
                    if not line.startswith("data: "):
                        continue
                    data_s = line[6:].strip()
                    if data_s and data_s != "[DONE]":
                        obj = None
                        try:
                            obj = json.loads(data_s)
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict) and obj.get("choices"):
                            delta = obj["choices"][0].get("delta", {})
                            text = delta.get("content")
                            if text:
                                normalized.append(text)
                assert normalized, "Expected at least one normalized chunk"
                assert "".join(normalized) == "Hello world"

                # Verify DB placeholder for user image validation failure
                msgs = db.get_messages_for_conversation(conv_id, 50, 0, "ASC")
                assert len(msgs) >= 1
                found_placeholder = any(
                    isinstance(m.get("content"), str)
                    and ("<Image failed" in m.get("content") or "<Image failed validation" in m.get("content"))
                    for m in msgs
                    if m.get("sender", "").lower() == "user"
                )
                assert found_placeholder, "Expected placeholder in user message for failed image validation"
    finally:
        # Cleanup db files
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        try:
            getattr(app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None
