"""
Integration tests for /api/v1/chat/completions streaming under STREAMS_UNIFIED.
Mirrors character chat and doc-generation tests: asserts headers, single [DONE],
and validates heartbeat presence with a slow async producer.
"""

import asyncio
import os
import shutil
import tempfile
from typing import Any, AsyncIterator, Iterator

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings


def _fake_provider_stream_simple() -> Iterator[str]:


    yield (
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hello from chat completions\"},\"index\":0,\"finish_reason\":null}]}\n\n"
    )
    yield "data: [DONE]\n\n"


async def _fake_provider_stream_slow_async() -> AsyncIterator[str]:
    # Delay to allow unified SSE heartbeat to fire
    await asyncio.sleep(0.06)
    yield (
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Chunk A\"},\"index\":0,\"finish_reason\":null}]}\n\n"
    )
    await asyncio.sleep(0.02)
    yield "data: [DONE]\n\n"


def _fake_provider_stream_with_duplicate_done() -> Iterator[str]:


     # Emit a normal data chunk, then two provider DONEs; unified layer should still output exactly one DONE
    yield (
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Part 1\"},\"index\":0,\"finish_reason\":null}]}\n\n"
    )
    yield "data: [DONE]\n\n"
    yield "data: [DONE]\n\n"


@pytest.fixture(autouse=True)
def _isolate_chat_rate_limiter(monkeypatch):
    for key in (
        "TEST_CHAT_RATE_LIMIT_RPM",
        "TEST_CHAT_PER_USER_RPM",
        "TEST_CHAT_PER_CONVERSATION_RPM",
        "TEST_CHAT_TOKENS_PER_MINUTE",
        "TEST_CHAT_BURST_MULTIPLIER",
    ):
        monkeypatch.delenv(key, raising=False)

    from tldw_Server_API.app.core.Chat.rate_limiter import RateLimitConfig, initialize_rate_limiter

    initialize_rate_limiter(
        RateLimitConfig(
            global_rpm=1000,
            per_user_rpm=1000,
            per_conversation_rpm=1000,
            per_user_tokens_per_minute=1_000_000,
            burst_multiplier=1.0,
        )
    )
    yield
    initialize_rate_limiter(RateLimitConfig())


@pytest.mark.asyncio
async def test_chat_completions_streaming_unified_sse_simple(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="unified_sse_chat_simple_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir
    os.environ["STREAMS_UNIFIED"] = "1"
    os.environ["TEST_MODE"] = "true"
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        # Patch perform_chat_api_call in the chat endpoint to return a sync generator
        import tldw_Server_API.app.api.v1.endpoints.chat as chat_ep

        def _stub_perform_chat_api_call(*args, **kwargs):

            return _fake_provider_stream_simple()

        chat_ep.perform_chat_api_call = _stub_perform_chat_api_call  # type: ignore

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": "Say hi"},
                ],
                "stream": True,
                "provider": "openai",
            }
            async with client.stream(
                "POST",
                "/api/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                assert resp.status_code == 200
                # SSE headers
                ct = resp.headers.get("content-type", "").lower()
                assert ct.startswith("text/event-stream")
                assert resp.headers.get("Cache-Control") == "no-cache"
                assert resp.headers.get("X-Accel-Buffering") == "no"

                lines = []
                done_count = 0
                async for ln in resp.aiter_lines():
                    if not ln:
                        continue
                    lines.append(ln)
                    if ln.strip().lower() == "data: [done]":
                        done_count += 1

        # Assertions
        assert any(ln.startswith("data: ") and "[DONE]" not in ln for ln in lines)
        assert lines[-1].strip().lower() == "data: [done]"
        assert done_count == 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_completions_streaming_unified_sse_slow_async_heartbeat(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="unified_sse_chat_slow_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir
    os.environ["STREAMS_UNIFIED"] = "1"
    os.environ["TEST_MODE"] = "true"
    # Short heartbeats in unified SSE layer
    os.environ["STREAM_HEARTBEAT_INTERVAL_S"] = "0.02"
    os.environ["STREAM_HEARTBEAT_MODE"] = "data"
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        import tldw_Server_API.app.api.v1.endpoints.chat as chat_ep

        def _stub_perform_chat_api_call(*args, **kwargs):

            return _fake_provider_stream_slow_async()

        chat_ep.perform_chat_api_call = _stub_perform_chat_api_call  # type: ignore

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": "Slow please"},
                ],
                "stream": True,
                "provider": "openai",
            }
            async with client.stream(
                "POST",
                "/api/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                assert resp.status_code == 200
                ct = resp.headers.get("content-type", "").lower()
                assert ct.startswith("text/event-stream")
                assert resp.headers.get("Cache-Control") == "no-cache"
                assert resp.headers.get("X-Accel-Buffering") == "no"

                lines = []
                done_count = 0
                heartbeat_seen = False
                async for ln in resp.aiter_lines():
                    if not ln:
                        continue
                    lines.append(ln)
                    if ln.strip().lower() == "data: [done]":
                        done_count += 1
                    low = ln.lower()
                    if low.startswith("data:") and "heartbeat" in low:
                        heartbeat_seen = True

        assert heartbeat_seen is True
        assert any(ln.startswith("data: ") and "[DONE]" not in ln for ln in lines)
        assert lines[-1].strip().lower() == "data: [done]"
        assert done_count == 1
    finally:
        for k in ("STREAM_HEARTBEAT_INTERVAL_S", "STREAM_HEARTBEAT_MODE"):
            os.environ.pop(k, None)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_completions_streaming_unified_sse_provider_duplicate_done(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="unified_sse_chat_dupdone_")
    os.environ["USER_DB_BASE_DIR"] = tmpdir
    os.environ["STREAMS_UNIFIED"] = "1"
    os.environ["TEST_MODE"] = "true"
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        import tldw_Server_API.app.api.v1.endpoints.chat as chat_ep

        def _stub_perform_chat_api_call(*args, **kwargs):

            return _fake_provider_stream_with_duplicate_done()

        chat_ep.perform_chat_api_call = _stub_perform_chat_api_call  # type: ignore

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": "Emit duplicate DONE"},
                ],
                "stream": True,
                "provider": "openai",
            }
            async with client.stream(
                "POST",
                "/api/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                assert resp.status_code == 200
                ct = resp.headers.get("content-type", "").lower()
                assert ct.startswith("text/event-stream")
                assert resp.headers.get("Cache-Control") == "no-cache"
                assert resp.headers.get("X-Accel-Buffering") == "no"

                done_count = 0
                lines = []
                async for ln in resp.aiter_lines():
                    if not ln:
                        continue
                    lines.append(ln)
                    if ln.strip().lower() == "data: [done]":
                        done_count += 1

        # Unified stream should dedupe and emit a single terminal DONE
        assert any(ln.startswith("data: ") and "[DONE]" not in ln for ln in lines)
        assert lines[-1].strip().lower() == "data: [done]"
        assert done_count == 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
