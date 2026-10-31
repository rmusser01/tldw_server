"""
Validate character chat streaming under STREAMS_UNIFIED=1 with two providers
by monkeypatching the provider call to emit deterministic SSE chunks.
"""

import json as _json
import shutil
import tempfile

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings


@pytest.mark.asyncio
async def test_complete_v2_streaming_unified_flag_two_providers(
    monkeypatch,
    healthy_absent_provider_override_snapshot,
    character_provider_adapter_boundary,
):
    # Force unified streams and minimal app footprint
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    # Fake SSE chunks (as strings) from provider
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
    stream_chunks = [f"data: {payload}" for payload in streaming_payloads]
    stream_chunks.append("data: [DONE]")

    import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as chat_sessions_mod
    provider_calls, bind_provider_call = character_provider_adapter_boundary

    def _fake_perform_chat_api_call(*args, **kwargs):
        assert not args
        bind_provider_call(kwargs)

        def _generator():
            yield from stream_chunks

        return _generator()

    monkeypatch.setattr(chat_sessions_mod, "perform_chat_api_call", _fake_perform_chat_api_call)

    # Isolate DB/files
    tmpdir = tempfile.mkdtemp(prefix="chacha_stream_unified_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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

            async def _stream_and_collect(provider_name: str):
                url = f"/api/v1/chats/{chat_id}/complete-v2"
                collected = []
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json={
                        "provider": provider_name,
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
                return collected

            # Validate for two providers: openai and groq
            for provider_name in ("openai", "groq"):
                collected = await _stream_and_collect(provider_name)
                expected = [
                    f"data: {streaming_payloads[0]}",
                    f"data: {streaming_payloads[1]}",
                    f"data: {streaming_payloads[2]}",
                    "data: [DONE]",
                ]
                assert collected == expected
            assert [call["api_endpoint"] for call in provider_calls] == [
                "openai",
                "groq",
            ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
