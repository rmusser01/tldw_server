"""A character stream ends with exactly one terminal frame for a provider DONE in any spelling (TASK-13370),
and does not send OpenAI's usage chunk, which its request cannot ask for (TASK-13951)."""

import json as _json
import shutil
import tempfile

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.LLM_Calls.sse import is_done_line

CONTENT = _json.dumps(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    }
)


async def _complete_v2_stream_lines(monkeypatch, bind_provider_call, streams_unified, provider_lines):
    """Stream one complete-v2 request whose provider yields *provider_lines*; return the non-empty lines received."""
    monkeypatch.setenv("STREAMS_UNIFIED", streams_unified)
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as chat_sessions_mod

    def _fake_perform_chat_api_call(*args, **kwargs):
        bind_provider_call(kwargs)
        return iter(provider_lines)

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
        return lines
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    _provider_calls, bind_provider_call = character_provider_adapter_boundary

    lines = await _complete_v2_stream_lines(
        monkeypatch, bind_provider_call, streams_unified, [f"data: {CONTENT}", done_line]
    )

    assert f"data: {CONTENT}" in lines
    assert [line for line in lines if is_done_line(line)] == ["data: [DONE]"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streams_unified", ["0", "1"])
async def test_complete_v2_stream_does_not_send_an_unrequested_usage_chunk(
    monkeypatch,
    healthy_absent_provider_override_snapshot,
    character_provider_adapter_boundary,
    streams_unified,
):
    """OpenAI's usage chunk (empty choices) is dropped; lines that are not one pass unchanged (TASK-13951)."""
    _provider_calls, bind_provider_call = character_provider_adapter_boundary
    usage_chunk = 'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}'
    others = ["data: plain text", 'data: {"choices": [], "usage"', 'data: {"choices":[],"usage":null}']

    lines = await _complete_v2_stream_lines(
        monkeypatch,
        bind_provider_call,
        streams_unified,
        [f"data: {CONTENT}", usage_chunk, *others, "data: [DONE]"],
    )

    assert usage_chunk not in lines
    assert [line for line in (f"data: {CONTENT}", *others) if line not in lines] == []
