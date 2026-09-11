"""
Integration test for chat document-generation streaming using unified SSEStream
behind STREAMS_UNIFIED. We stub the LLM call to return a simple async generator
of text chunks and assert SSE emission with a terminal [DONE].
"""

import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    bind_provider_call_credentials,
)


@pytest.fixture
def configured_openai_server_credential() -> Iterator[str]:
    """Expose one healthy configured key through the real credential runtime."""

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
    )

    with llm_provider_overrides._OVERRIDE_LOCK:
        original_overrides = dict(llm_provider_overrides._OVERRIDE_CACHE)
        original_healthy = llm_provider_overrides._OVERRIDE_CACHE_HEALTHY
        original_ttl_disabled = (
            llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
        )

    api_key = "test-openai-server-key"
    configured = dict(original_overrides)
    configured["openai"] = LLMProviderOverride(
        provider="openai",
        api_key=api_key,
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(configured)
    try:
        yield api_key
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            original_overrides,
            healthy=original_healthy,
            ttl_enabled=not original_ttl_disabled,
        )


def _recording_chat_adapter(
    stream_factory: Callable[[], AsyncIterator[str]],
    calls: list[tuple[str | None, bool, bool]],
) -> Callable[..., AsyncIterator[str]]:
    """Consume the runtime capability at the same seam as a real adapter."""

    def _adapter(**kwargs: Any) -> AsyncIterator[str]:
        provider = str(kwargs.get("api_endpoint") or "")
        bound, _credentials = bind_provider_call_credentials(
            provider,
            kwargs,
            consume=True,
        )
        calls.append(
            (
                bound.get("api_key"),
                bound.get("credentials_resolved") is True,
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in bound,
            )
        )
        return stream_factory()

    return _adapter


async def _async_text_stream() -> AsyncIterator[str]:
    yield "First line from doc-gen"
    # Simulate a slower producer emitting a later chunk
    await asyncio.sleep(0.02)
    yield "Second line from doc-gen"


def _dup_done_stream() -> AsyncIterator[str]:


    async def _gen():
        yield "Line before done"
        yield "[DONE]"
        yield "[DONE]"
    return _gen()


async def _async_text_stream_slow() -> AsyncIterator[str]:
    # Delay long enough to trigger at least one heartbeat from SSEStream
    await asyncio.sleep(0.06)
    yield "Delayed line 1"
    await asyncio.sleep(0.02)
    yield "Delayed line 2"


@pytest.mark.asyncio
async def test_chat_document_generation_streaming_unified_sse(
    monkeypatch,
    configured_openai_server_credential,
):
    tmpdir = tempfile.mkdtemp(prefix="unified_sse_doc_stream_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        # Keep the document service and credential bridge real; replace only the
        # provider adapter after it consumes the runtime capability.
        import tldw_Server_API.app.core.Chat.document_generator as gen_mod

        adapter_calls: list[tuple[str | None, bool, bool]] = []
        monkeypatch.setattr(
            gen_mod,
            "chat_api_call",
            _recording_chat_adapter(_async_text_stream, adapter_calls),
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Bootstrap: get default character + create chat to have a conversation id
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            conversation_id = r.json()["id"]
            # Ensure at least one message exists to satisfy doc generator
            msg_resp = await client.post(
                f"/api/v1/chats/{conversation_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Hello for doc-gen"},
            )
            assert msg_resp.status_code == 201

            payload = {
                "conversation_id": conversation_id,
                "document_type": "summary",
                "provider": "openai",
                "model": "gpt-x",
                "stream": True,
            }

            async with client.stream(
                "POST",
                "/api/v1/chat/documents/generate",
                headers=headers,
                json=payload,
            ) as resp:
                assert resp.status_code == 200
                # Header assertions
                ct = resp.headers.get("content-type", "")
                assert ct.lower().startswith("text/event-stream")
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

        # Should include our payload lines and finish with DONE
        assert any(ln.startswith("data: ") and "[DONE]" not in ln for ln in lines)
        assert lines[-1].strip().lower() == "data: [done]"
        assert done_count == 1
        assert adapter_calls == [
            (configured_openai_server_credential, True, True)
        ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_document_generation_streaming_unified_sse_provider_duplicate_done(
    monkeypatch,
    configured_openai_server_credential,
):
    tmpdir = tempfile.mkdtemp(prefix="unified_sse_doc_dupdone_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        import tldw_Server_API.app.core.Chat.document_generator as gen_mod

        adapter_calls: list[tuple[str | None, bool, bool]] = []
        monkeypatch.setattr(
            gen_mod,
            "chat_api_call",
            _recording_chat_adapter(_dup_done_stream, adapter_calls),
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/api/v1/characters/", headers=headers)
            assert r.status_code == 200
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            assert r.status_code == 201
            conversation_id = r.json()["id"]
            msg_resp = await client.post(
                f"/api/v1/chats/{conversation_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Seed message"},
            )
            assert msg_resp.status_code == 201

            payload = {
                "conversation_id": conversation_id,
                "document_type": "summary",
                "provider": "openai",
                "model": "gpt-x",
                "stream": True,
            }

            async with client.stream(
                "POST",
                "/api/v1/chat/documents/generate",
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
                async for ln in resp.aiter_lines():
                    if not ln:
                        continue
                    lines.append(ln)
                    if ln.strip().lower() == "data: [done]":
                        done_count += 1

        assert any(ln.startswith("data: ") and "[DONE]" not in ln for ln in lines)
        assert lines[-1].strip().lower() == "data: [done]"
        assert done_count == 1
        assert adapter_calls == [
            (configured_openai_server_credential, True, True)
        ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_document_generation_streaming_unified_sse_slow_async_heartbeat(
    monkeypatch,
    configured_openai_server_credential,
):
    tmpdir = tempfile.mkdtemp(prefix="unified_sse_doc_heartbeat_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    # Short heartbeat so it appears before first chunk
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_S", "0.02")
    monkeypatch.setenv("STREAM_HEARTBEAT_MODE", "data")
    try:
        from tldw_Server_API.app.main import app
        settings = get_settings()
        headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

        import tldw_Server_API.app.core.Chat.document_generator as gen_mod

        adapter_calls: list[tuple[str | None, bool, bool]] = []
        monkeypatch.setattr(
            gen_mod,
            "chat_api_call",
            _recording_chat_adapter(_async_text_stream_slow, adapter_calls),
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Bootstrap defaults
            r = await client.get("/api/v1/characters/", headers=headers)
            character_id = r.json()[0]["id"]
            r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
            conversation_id = r.json()["id"]
            msg_resp = await client.post(
                f"/api/v1/chats/{conversation_id}/messages",
                headers=headers,
                json={"role": "user", "content": "Slow seed"},
            )
            assert msg_resp.status_code == 201

            payload = {
                "conversation_id": conversation_id,
                "document_type": "summary",
                "provider": "openai",
                "model": "gpt-x",
                "stream": True,
            }

            async with client.stream(
                "POST",
                "/api/v1/chat/documents/generate",
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
                    if ln.lower().startswith("data:") and "heartbeat" in ln.lower():
                        heartbeat_seen = True

        assert heartbeat_seen is True
        assert any(ln.startswith("data: ") and "[DONE]" not in ln for ln in lines)
        assert lines[-1].strip().lower() == "data: [done]"
        assert done_count == 1
        assert adapter_calls == [
            (configured_openai_server_credential, True, True)
        ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
