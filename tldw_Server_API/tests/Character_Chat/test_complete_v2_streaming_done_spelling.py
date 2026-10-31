"""A provider DONE in any spelling ends a character stream with exactly one terminal frame (TASK-13370)."""

import json as _json
import shutil
import tempfile

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.LLM_Calls.sse import is_done_line


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"])
@pytest.mark.parametrize("done_line", ["data: [done]", "data:[DONE]"])
async def test_complete_v2_stream_emits_single_done_for_any_done_spelling(
    monkeypatch,
    healthy_absent_provider_override_snapshot,
    character_provider_adapter_boundary,
    streams_unified,
    done_line,
):
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    content = _json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
    )

    import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as chat_sessions_mod

    _provider_calls, bind_provider_call = character_provider_adapter_boundary

    def _fake_perform_chat_api_call(*args, **kwargs):
        bind_provider_call(kwargs)
        return iter([f"data: {content}", done_line])

    monkeypatch.setattr(chat_sessions_mod, "perform_chat_api_call", _fake_perform_chat_api_call)

    tmpdir = tempfile.mkdtemp(prefix="chacha_stream_done_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            chat_id = r.json()["id"]

            lines = []
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
                async for line in response.aiter_lines():
                    if line:
                        lines.append(line)

        assert f"data: {content}" in lines
        assert [line for line in lines if is_done_line(line)] == ["data: [DONE]"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
