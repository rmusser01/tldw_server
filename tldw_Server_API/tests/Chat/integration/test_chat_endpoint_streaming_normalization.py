import os
import json
import tempfile
import pytest
from typing import Any
from fastapi.testclient import TestClient
from unittest.mock import patch

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Chat.tool_auto_exec import (
    ToolExecutionBatchResult,
    ToolExecutionRecord,
)


def _make_test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client")
    # Minimal default character required by endpoint
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
    return db, db_path


from contextlib import contextmanager


def _post_with_csrf(client: TestClient, url: str, **kwargs):
    """Helper to POST with CSRF header from client state.

    Avoids adding dynamic attributes to TestClient to keep type-checkers happy.
    """
    headers = kwargs.pop("headers", {}) or {}
    csrf = getattr(client, "csrf_token", "")
    return client.post(url, headers={"X-CSRF-Token": csrf, **headers}, **kwargs)


def _json_sse_payloads(text: str) -> list[dict[str, Any]]:
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


@contextmanager
def _make_test_client(db):
    with TestClient(app) as client:
        # CSRF token for POST
        resp = client.get("/api/v1/health")
        csrf = resp.cookies.get("csrf_token", "")
        client.csrf_token = csrf
        yield client


def _auth_headers(client):
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
    return {"X-API-KEY": api_key, "X-CSRF-Token": getattr(client, "csrf_token", "")}


@pytest.mark.unit
def test_endpoint_streaming_normalizes_openai_sse_frames():
    db, db_path = _make_test_db()
    try:
        # Cast to Any to avoid static analyzer complaints about dependency_overrides typing
        _app: Any = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        with _make_test_client(db) as client:

            # Upstream SSE frames (OpenAI-like)
            chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
            chunk2 = {"choices": [{"delta": {"content": " world"}}]}

            def upstream_stream():
                yield f"data: {json.dumps(chunk1)}\n\n"
                yield f"data: {json.dumps(chunk2)}\n\n"
                yield "data: [DONE]\n\n"

            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False),
                patch(
                    "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()
                ),
            ):
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                assert "text/event-stream" in r.headers.get("content-type", "").lower()

                # TestClient buffers entire SSE; parse normalized lines
                content = r.text.splitlines()
                normalized_chunks = []
                for line in content:
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("choices"):
                        delta = obj["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            normalized_chunks.append(text)
                assert normalized_chunks, "Expected at least one normalized chunk"
                assert "".join(normalized_chunks) == "Hello world"

    finally:
        # cleanup db files
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None

        _app = app  # type: ignore[assignment]
        try:
            getattr(_app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_endpoint_streaming_normalizes_multiline_event_and_data_frames():
    db, db_path = _make_test_db()
    try:
        _app: Any = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        with _make_test_client(db) as client:

            part_a = {"choices": [{"delta": {"content": "Part"}}]}
            part_b = {"choices": [{"delta": {"content": " A"}}]}
            multiline = (
                "event: chunk\n" f"data: {json.dumps(part_a)}\n" f"data: {json.dumps(part_b)}\n" "data: [DONE]\n\n"
            )

            def upstream_stream():

                # A single upstream chunk containing multiple frames
                yield multiline

            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False),
                patch(
                    "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()
                ),
            ):
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                assert "text/event-stream" in r.headers.get("content-type", "").lower()

                content = r.text.splitlines()
                normalized_chunks = []
                for line in content:
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("choices"):
                        delta = obj["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            normalized_chunks.append(text)
                assert normalized_chunks, "Expected at least one normalized chunk"
                assert "".join(normalized_chunks) == "Part A"

    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        _app = app  # type: ignore[assignment]
        try:
            getattr(_app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_endpoint_streaming_emits_single_terminal_done_frame():
    db, db_path = _make_test_db()
    try:
        _app: Any = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        with _make_test_client(db) as client:
            chunk = {"choices": [{"delta": {"content": "Hello"}}]}

            def upstream_stream():
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                yield "data: [DONE]\n\n"

            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False),
                patch(
                    "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                    return_value=upstream_stream(),
                ),
            ):
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                r = _post_with_csrf(
                    client,
                    "/api/v1/chat/completions",
                    json=body,
                    headers=_auth_headers(client),
                )
                assert r.status_code == 200
                assert "text/event-stream" in r.headers.get("content-type", "").lower()

                non_empty = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
                done_lines = [ln for ln in non_empty if ln.lower() == "data: [done]"]
                assert len(done_lines) == 1
                assert non_empty[-1].lower() == "data: [done]"

    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None

        _app = app  # type: ignore[assignment]
        try:
            getattr(_app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_endpoint_streaming_emits_tool_results_event_before_stream_end():
    db, db_path = _make_test_db()
    try:
        _app: Any = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        with _make_test_client(db) as client:
            tool_delta = {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "notes.search", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }

            def upstream_stream():
                yield f"data: {json.dumps(tool_delta)}\n\n"
                yield "data: [DONE]\n\n"

            async def fake_autoexec(**_kwargs):
                rec = ToolExecutionRecord(
                    tool_call_id="c1",
                    tool_name="notes.search",
                    ok=True,
                    result={"ok": True},
                    module="notes",
                    content='{"ok":true}',
                )
                return ToolExecutionBatchResult(
                    requested_calls=1,
                    processed_calls=1,
                    execution_attempts=1,
                    executed_calls=1,
                    truncated=False,
                    results=[rec],
                )

            with (
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "sk-test",
                        "CHAT_AUTO_EXECUTE_TOOLS": "1",
                        "CHAT_MAX_TOOL_CALLS": "2",
                        "CHAT_TOOL_TIMEOUT_MS": "4500",
                        "CHAT_TOOL_ALLOW_CATALOG": "notes.*",
                        "CHAT_TOOL_IDEMPOTENCY": "1",
                    },
                    clear=False,
                ),
                patch(
                    "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                    return_value=upstream_stream(),
                ),
                patch(
                    "tldw_Server_API.app.core.Chat.chat_service.execute_assistant_tool_calls",
                    side_effect=fake_autoexec,
                ),
            ):
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "save_to_db": True,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                assert "text/event-stream" in r.headers.get("content-type", "").lower()

                events = [evt for evt in r.text.split("\n\n") if evt.strip()]
                payloads = _json_sse_payloads(r.text)
                tool_idx = next(i for i, payload in enumerate(payloads) if "tool_results" in payload)
                end_idx = next(i for i, payload in enumerate(payloads) if payload.get("success") is True)
                done_idx = next(i for i, evt in enumerate(events) if evt.strip() == "data: [DONE]")
                assert tool_idx < end_idx
                assert events[done_idx].strip() == "data: [DONE]"
                assert done_idx == len(events) - 1

                payload = payloads[tool_idx]
                assert payload["tool_results"][0]["tool_call_id"] == "c1"
                assert payload["tool_results"][0]["ok"] is True
                assert payload.get("tldw_conversation_id")

    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        _app = app  # type: ignore[assignment]
        try:
            getattr(_app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None


@pytest.mark.unit
def test_endpoint_non_stream_auto_continue_returns_followup_assistant_content():
    db, db_path = _make_test_db()
    try:
        _app: Any = app
        _app.dependency_overrides[get_chacha_db_for_user] = lambda: db

        with _make_test_client(db) as client:
            first_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Calling tool",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "notes.search", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
            followup_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Final assistant answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }

            async def fake_autoexec(**_kwargs):
                rec = ToolExecutionRecord(
                    tool_call_id="c1",
                    tool_name="notes.search",
                    ok=True,
                    result={"ok": True},
                    module="notes",
                    content='{"ok":true}',
                )
                return ToolExecutionBatchResult(
                    requested_calls=1,
                    processed_calls=1,
                    execution_attempts=1,
                    executed_calls=1,
                    truncated=False,
                    results=[rec],
                )

            continuation_calls: list[dict[str, Any]] = []

            async def fake_followup_call(**kwargs):
                continuation_calls.append(kwargs)
                return followup_response

            with (
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "sk-test",
                        "CHAT_AUTO_EXECUTE_TOOLS": "1",
                        "CHAT_MAX_TOOL_CALLS": "2",
                        "CHAT_TOOL_TIMEOUT_MS": "4500",
                        "CHAT_TOOL_ALLOW_CATALOG": "notes.*",
                        "CHAT_TOOL_IDEMPOTENCY": "1",
                        "CHAT_TOOL_AUTO_CONTINUE_ONCE": "1",
                    },
                    clear=False,
                ),
                patch(
                    "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                    return_value=first_response,
                ),
                patch(
                    "tldw_Server_API.app.core.Chat.chat_service.execute_assistant_tool_calls",
                    side_effect=fake_autoexec,
                ),
                patch(
                    "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
                    side_effect=fake_followup_call,
                ),
            ):
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "save_to_db": True,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                payload = r.json()
                assert payload["choices"][0]["message"]["content"] == "Final assistant answer"
                assert payload["tldw_tool_results"][0]["tool_call_id"] == "c1"
                assert payload["tldw_tool_auto_continue"] == {"attempted": True, "succeeded": True}

            assert len(continuation_calls) == 1
            continuation_messages = continuation_calls[0]["messages_payload"]
            assert continuation_messages[-1]["role"] == "tool"
            assert continuation_messages[-1]["tool_call_id"] == "c1"

    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        _app = app  # type: ignore[assignment]
        try:
            getattr(_app, "dependency_overrides", {}).pop(get_chacha_db_for_user, None)
        except Exception:
            _ = None
