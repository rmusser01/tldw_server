"""
End-to-end streaming test for complete-v2 that monkeypatches the OpenAI requests.Session
to emit a deterministic multi-chunk SSE sequence. This avoids real network usage.
"""

import json as _json
import shutil
import tempfile

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings


class _FakeStreamingResponse:
    def __init__(self, lines):
        self._lines = list(lines)
        self.status_code = 200

    def raise_for_status(self):

        return None

    def iter_lines(self, decode_unicode=True):

        yield from self._lines

    def close(self):

        return None


class _FakeSession:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        return False

    def mount(self, *args, **kwargs):

        return None

    def post(self, url, headers=None, json=None, stream=False, timeout=30):

        # Provide a deterministic multi-chunk SSE response
        # 2 delta chunks and then [DONE]
        chunks = [
            "data: "
            + _json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            ),
            "data: "
            + _json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
                }
            ),
            "data: "
            + _json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
                }
            ),
            "data: [DONE]",
        ]
        return _FakeStreamingResponse(chunks)


@pytest.mark.asyncio
async def test_complete_v2_streaming_e2e_monkeypatched(
    monkeypatch,
    healthy_absent_provider_override_snapshot,
    character_provider_adapter_boundary,
):
    import tldw_Server_API.app.core.LLM_Calls.chat_calls as llm_mod

    monkeypatch.setattr(llm_mod, "create_session_with_retries", lambda *args, **kwargs: _FakeSession())
    provider_calls, bind_provider_call = character_provider_adapter_boundary

    streaming_payloads = [
        _json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        ),
        _json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }
        ),
        _json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
            }
        ),
    ]
    stream_chunks = [f"event: completion.chunk\ndata: {payload}\n\n" for payload in streaming_payloads]
    stream_chunks.append("event: close\n\n")
    stream_chunks.append("data: [DONE]\n\n")

    import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as chat_sessions_mod

    def _fake_perform_chat_api_call(*args, **kwargs):
        assert not args
        bind_provider_call(kwargs)

        def _generator():
            yield from stream_chunks

        return _generator()

    monkeypatch.setattr(chat_sessions_mod, "perform_chat_api_call", _fake_perform_chat_api_call)

    # Isolate DB and ensure env changes don't leak to other tests
    tmpdir = tempfile.mkdtemp(prefix="chacha_stream_e2e_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Point base URL to a non-routable host so any accidental network path fails fast
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://mock.local")
    try:
        from tldw_Server_API.app.main import app

        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Setup: character + chat
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            # Stream the completion via complete-v2
            url = f"/api/v1/chats/{chat_id}/complete-v2"
            expected_lines = [
                # Each line should be forwarded 1:1
                "data: " + streaming_payloads[0],
                "data: " + streaming_payloads[1],
                "data: " + streaming_payloads[2],
                "data: [DONE]",
            ]
            collected = []
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "append_user_message": "ping",
                    "save_to_db": False,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        collected.append(line)
            assert collected == expected_lines
            assert [call["api_endpoint"] for call in provider_calls] == ["openai"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_complete_v2_streaming_non_iterable_fallback(
    monkeypatch,
    healthy_absent_provider_override_snapshot,
    character_provider_adapter_boundary,
):
    tmpdir = tempfile.mkdtemp(prefix="chacha_stream_non_iterable_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("STREAMS_UNIFIED", raising=False)

    import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as chat_sessions_mod
    provider_calls, bind_provider_call = character_provider_adapter_boundary

    def _fake_perform_chat_api_call(*args, **kwargs):
        assert not args
        bind_provider_call(kwargs)
        return {
            "choices": [{"message": {"role": "assistant", "content": "Fallback content"}}]
        }

    monkeypatch.setattr(chat_sessions_mod, "perform_chat_api_call", _fake_perform_chat_api_call)
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

            async with client.stream(
                "POST",
                f"/api/v1/chats/{chat_id}/complete-v2",
                headers=headers,
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "append_user_message": "ping",
                    "save_to_db": False,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                ct = response.headers.get("content-type", "")
                assert ct.lower().startswith("text/event-stream")
                lines = [ln async for ln in response.aiter_lines() if ln]

        combined = "\n".join(lines)
        assert "Fallback content" in combined
        assert lines[-1].strip().lower() == "data: [done]"
        assert [call["api_endpoint"] for call in provider_calls] == ["openai"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
