import asyncio
import threading
from typing import Any, Dict

import pytest

from tldw_Server_API.app.core.Chat import command_router


@pytest.fixture(autouse=True)
def _relax_chat_rate_limits_for_command_tests(monkeypatch, _reset_chat_rate_limiter_between_tests):
    monkeypatch.setenv("TEST_CHAT_PER_USER_RPM", "1000")
    monkeypatch.setenv("TEST_CHAT_PER_CONVERSATION_RPM", "1000")
    monkeypatch.setenv("TEST_CHAT_GLOBAL_RPM", "1000")
    monkeypatch.setenv("TEST_CHAT_TOKENS_PER_MINUTE", "1000000")
    monkeypatch.setenv("TEST_CHAT_BURST_MULTIPLIER", "1.0")
    from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter
    initialize_rate_limiter()
    yield


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_command_concurrency_respects_rate_limit(async_client, auth_headers, monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_USER", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "100")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    command_router._buckets.clear()
    command_router._global_buckets.clear()

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    lock = threading.Lock()
    results = {"ok": 0, "rate_limited": 0}

    original_async_dispatch = command_router.async_dispatch_command

    async def wrapped_async_dispatch(ctx, name, args):
        fixed_ctx = command_router.CommandContext(
            user_id="concurrency-user",
            conversation_id=ctx.conversation_id,
            request_meta=ctx.request_meta,
            auth_user_id=ctx.auth_user_id,
        )
        res = await original_async_dispatch(fixed_ctx, name, args)
        with lock:
            if res.ok:
                results["ok"] += 1
            else:
                results["rate_limited"] += 1
        return res

    monkeypatch.setattr(command_router, "async_dispatch_command", wrapped_async_dispatch)

    def fake_call(**kwargs):
        return {
            "id": "mock",
            "object": "chat.completion",
            "created": 0,
            "model": kwargs.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "/time"}],
        "stream": False,
    }

    async def call():
        return await async_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)

    responses = await asyncio.gather(*[call() for _ in range(10)])
    assert all(resp.status_code == 200 for resp in responses)
    assert results["ok"] + results["rate_limited"] == 10
    assert results["ok"] <= 1
    assert results["rate_limited"] >= 1
