import os
import time
import statistics

import pytest

from tldw_Server_API.app.core.Chat import command_router


pytestmark = pytest.mark.performance


def _perf_enabled() -> bool:
    return os.getenv("PERF", "0").lower() in {"1", "true", "yes", "y", "on"}


pytestmark = pytest.mark.skipif(not _perf_enabled(), reason="set PERF=1 to run performance checks")


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


@pytest.mark.asyncio
async def test_chat_command_p50_latency(async_client, auth_headers, monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_USER", "1000")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "1000")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    command_router._buckets.clear()
    command_router._global_buckets.clear()

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

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

    warmup = int(os.getenv("PERF_CHAT_COMMANDS_WARMUP", "5"))
    samples = int(os.getenv("PERF_CHAT_COMMANDS_SAMPLES", "25"))

    for _ in range(warmup):
        resp = await async_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        assert resp.status_code == 200

    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        resp = await async_client.post("/api/v1/chat/completions", json=payload, headers=auth_headers)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        timings.append(elapsed)

    p50 = statistics.median(timings)
    print(f"chat_command_p50_latency samples={samples} p50={p50:.4f}s")
