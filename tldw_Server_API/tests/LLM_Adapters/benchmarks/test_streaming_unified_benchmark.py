from __future__ import annotations

"""
Micro-benchmark for the unified streaming path (STREAMS_UNIFIED).

This measures endpoint streaming overhead by patching a provider adapter to
emit a fixed number of SSE chunks quickly, then consumes the endpoint stream
and records throughput via pytest-benchmark.

To compare parity, you can re-run with STREAMS_UNIFIED=0 to exercise the
non-unified path and compare benchmark results across runs.
"""

from typing import Iterator

import pytest
from _pytest.fixtures import FixtureLookupError

# Register chat fixtures (authenticated_client)
from tldw_Server_API.tests._plugins import chat_fixtures as _chat_pl  # noqa: F401


@pytest.fixture(autouse=True)
def _enable_unified(monkeypatch):
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


# Ensure this benchmark exercises the adapter path (not the built-in mock).
# The chat endpoint automatically switches to a mock provider in TEST_MODE for
# certain providers; disable that behavior here to use our patched adapter.
@pytest.fixture(autouse=True)
def _disable_test_mode_and_mock(monkeypatch):
    # Remove TEST_MODE (set globally in tests) and explicitly disable mock forcing
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("CHAT_FORCE_MOCK", "0")
    yield


def _payload() -> dict:
    return {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Benchmark"}],
        "stream": True,
    }


@pytest.mark.benchmark
def test_streaming_unified_throughput_benchmark(monkeypatch, request):
    try:
        benchmark = request.getfixturevalue("benchmark")
    except FixtureLookupError:
        pytest.skip("pytest-benchmark fixture is not available")
    authenticated_client = request.getfixturevalue("authenticated_client")

    import tldw_Server_API.app.api.v1.endpoints.chat as chat_endpoint
    chat_endpoint.API_KEYS = {**(chat_endpoint.API_KEYS or {}), "openai": "sk-openai-test"}

    # Patch OpenAIAdapter.stream to emit N chunks quickly
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
    N = 300

    def _fast_stream(*args, **kwargs) -> Iterator[str]:
        for i in range(N):
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(openai_mod.OpenAIAdapter, "stream", _fast_stream, raising=True)

    client = authenticated_client

    def _consume_once():
        with client.stream("POST", "/api/v1/chat/completions", json=_payload()) as resp:
            assert resp.status_code == 200
            count = 0
            for _ in resp.iter_lines():
                count += 1
            # There will be N chunks + 1 [DONE]
            assert count >= N
            return count

    result = benchmark(_consume_once)
    assert result >= N
