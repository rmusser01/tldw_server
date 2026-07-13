import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.chat_metrics import ChatMetricsCollector
from tldw_Server_API.app.core.Chat.chat_service import (
    execute_non_stream_call,
    execute_streaming_call,
    merge_api_keys_for_provider,
)


class _DummyMetrics:
    def __init__(self):
        self.llm_calls = []
        self.fallback_successes = []
        self._collector = ChatMetricsCollector()

    def track_llm_call(self, provider, model, latency, success, error_type=None):

        self.llm_calls.append((provider, model, success, error_type))

    def track_provider_fallback_success(self, **metadata):

        self.fallback_successes.append(metadata)

    def track_tokens(self, **_kwargs):

        return None

    def track_streaming(self, conversation_id: str):
        return self._collector.track_streaming(conversation_id)


class _DummyProviderManager:
    def __init__(self):
        self.failure_records = []
        self.success_records = []
        self.fallback_requests = []

    def get_available_provider(self, exclude=None):

        self.fallback_requests.append(tuple(exclude or []))
        return "openai"

    def record_failure(self, provider, error):

        self.failure_records.append((provider, type(error).__name__))

    def record_success(self, provider, latency):

        self.success_records.append(provider)


class _DummyModeration:
    class _Policy:
        enabled = False
        output_enabled = False

    def get_effective_policy(self, *_args, **_kwargs):

        return self._Policy()

    def evaluate_action(self, *_args, **_kwargs):

        return None

    def check_text(self, *_args, **_kwargs):

        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):

        return text


@pytest.mark.asyncio
async def test_execute_non_stream_call_normalizes_raw_string(monkeypatch):
    monkeypatch.setenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES", "1")

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def llm_call_func():
        return "plain response"

    async def save_message_fn(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "streaming": False,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=False,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(response, dict)
    assert response["choices"][0]["message"]["content"] == "plain response"
    assert response["tldw_conversation_id"] == "conv-123"


@pytest.mark.asyncio
async def test_execute_non_stream_call_refreshes_credentials(monkeypatch):
    captured_kwargs = {}

    async def fake_perform_chat_api_call_async(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "choices": [
                {"message": {"content": "fallback success"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_perform_chat_api_call_async)
    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def failing_llm_call():

        raise ChatProviderError(provider="anthropic", message="primary failed", status_code=502)

    async def save_message_fn(*_args, **_kwargs):
        return None

    def refresh_provider(provider_name: str):
        assert provider_name == "openai"
        return (
            {
                "api_endpoint": "openai",
                "api_key": "fresh-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "anthropic",
            "api_key": "stale-key",
            "messages_payload": [],
            "model": "claude-3",
            "streaming": False,
        },
        selected_provider="anthropic",
        provider="anthropic",
        model="claude-3",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-123",
        character_card_for_context={},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="user-123",
        queue_execution_enabled=False,
        enable_provider_fallback=True,
        llm_call_func=failing_llm_call,
        refresh_provider_params=refresh_provider,
        moderation_getter=_DummyModeration,
    )

    assert captured_kwargs["api_endpoint"] == "openai"
    assert captured_kwargs["api_key"] == "fresh-key"
    assert captured_kwargs["model"] == "gpt-4o"
    assert response["tldw_conversation_id"] == "conv-123"
    assert provider_manager.fallback_requests == [("anthropic",)]
    assert provider_manager.success_records == ["openai"]
    assert any(
        entry.get("selected_provider") == "openai" and entry.get("streaming") is False
        for entry in metrics.fallback_successes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_error",
    [
        ChatAuthenticationError("provider rejected credentials", provider="anthropic"),
        ChatConfigurationError("provider configuration invalid", provider="anthropic"),
    ],
)
async def test_execute_non_stream_call_never_falls_back_for_terminal_provider_errors(
    monkeypatch,
    terminal_error,
):
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def failing_llm_call():
        raise terminal_error

    async def save_message_fn(*_args, **_kwargs):
        return None

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    with pytest.raises(type(terminal_error)):
        await execute_non_stream_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "anthropic",
                "api_key": "runtime-key",
                "credentials_resolved": True,
                "messages_payload": [],
                "model": "claude-3",
                "streaming": False,
            },
            selected_provider="anthropic",
            provider="anthropic",
            model="claude-3",
            request_json="{}",
            request=request,
            metrics=metrics,
            provider_manager=provider_manager,
            templated_llm_payload=[],
            should_persist=False,
            final_conversation_id="conv-terminal",
            character_card_for_context={},
            chat_db=None,
            save_message_fn=save_message_fn,
            audit_service=None,
            audit_context=None,
            client_id="user-terminal",
            queue_execution_enabled=False,
            enable_provider_fallback=True,
            llm_call_func=failing_llm_call,
            refresh_provider_params=lambda *_args, **_kwargs: None,
            moderation_getter=_DummyModeration,
        )

    assert provider_manager.fallback_requests == []


@pytest.mark.asyncio
async def test_execute_non_stream_call_attaches_continuation_metadata_and_parent_id(monkeypatch):
    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    metrics = _DummyMetrics()
    save_payloads: list[dict[str, object]] = []

    def llm_call_func():
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "continued"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    async def save_message_fn(*_args, **_kwargs):
        payload = _args[2] if len(_args) > 2 else _kwargs.get("payload", {})
        if isinstance(payload, dict):
            save_payloads.append(payload)
        return "msg-cont-1"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    continuation_meta = {
        "applied": True,
        "mode": "branch",
        "from_message_id": "anchor-msg-1",
    }

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "api_key": "test-key",
            "messages_payload": [{"role": "user", "content": "continue"}],
            "model": "gpt-4o-mini",
            "streaming": False,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "continue"}],
        should_persist=True,
        final_conversation_id="conv-123",
        character_card_for_context={"name": "Test"},
        chat_db=SimpleNamespace(),
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: _DummyModeration(),
        assistant_parent_message_id="anchor-msg-1",
        continuation_metadata=continuation_meta,
    )

    assert save_payloads
    assert save_payloads[0]["parent_message_id"] == "anchor-msg-1"
    assert response["tldw_continuation"] == continuation_meta
    assert response["tldw_message_id"] == "msg-cont-1"


@pytest.mark.asyncio
async def test_execute_streaming_call_preserves_http_exception(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    http_exc = HTTPException(status_code=429, detail="Rate limited")

    def failing_llm_call():

        raise http_exc

    async def save_message_fn(*_args, **_kwargs):
        return None

    # Disable queue path to exercise direct streaming behavior
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-test",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-test",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=failing_llm_call,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(resp, StreamingResponse)

    # Consume the StreamingResponse body iterator and validate error payload + DONE
    agen = resp.body_iterator
    chunks = []
    try:
        for _ in range(4):
            try:
                ln = await agen.__anext__()
            except StopAsyncIteration:
                break
            if not ln:
                continue
            chunks.append(ln)
    finally:
        try:
            await agen.aclose()
        except Exception:
            _ = None

    # Normalize to str for assertions
    chunks = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in chunks]
    assert any("\"error\"" in c for c in chunks), f"No error frame in chunks: {chunks}"
    assert any("HTTPException" in c and "Rate limited" in c for c in chunks), f"Missing HTTPException details in error frame: {chunks}"
    assert chunks and chunks[-1].strip() == "data: [DONE]"

    # The last llm call recorded should indicate an HTTPException error type
    assert metrics.llm_calls[-1][3] in ("HTTPException", "HTTPException")


@pytest.mark.asyncio
async def test_execute_streaming_call_queue_fallback(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()

    def failing_llm_call():
        raise ChatProviderError(provider="anthropic", message="primary failed", status_code=502)

    async def save_message_fn(*_args, **_kwargs):
        return None

    def refresh_provider(provider_name: str):
        assert provider_name == "openai"
        return (
            {
                "api_endpoint": "openai",
                "api_key": "fresh-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": True,
            },
            "gpt-4o",
        )

    def fake_perform_chat_api_call(**kwargs):
        assert kwargs.get("api_endpoint") == "openai"

        def _stream():
            yield 'data: {"choices": [{"delta": {"content": "fallback ok"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    class DummyQueue:
        def __init__(self):
            self._running = True

        def is_running(self):
            return True

        async def enqueue(self, *, processor, stream_channel, **_kwargs):
            async def _run():
                try:
                    result = await asyncio.get_running_loop().run_in_executor(None, processor)
                    if hasattr(result, "__aiter__"):
                        async for chunk in result:
                            await stream_channel.put(chunk)
                    else:
                        for chunk in result:
                            await stream_channel.put(chunk)
                finally:
                    await stream_channel.put(None)

            asyncio.create_task(_run())
            fut = asyncio.Future()
            fut.set_result({"status": "ok"})
            return fut

    monkeypatch.setattr(chat_service, "perform_chat_api_call", fake_perform_chat_api_call)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: DummyQueue())

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "anthropic",
            "api_key": "stale-key",
            "messages_payload": [],
            "model": "claude-3",
            "streaming": True,
        },
        selected_provider="anthropic",
        provider="anthropic",
        model="claude-3",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-queue",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=True,
        enable_provider_fallback=True,
        llm_call_func=failing_llm_call,
        refresh_provider_params=refresh_provider,
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(resp, StreamingResponse)

    agen = resp.body_iterator
    chunks = []
    try:
        for _ in range(8):
            try:
                ln = await agen.__anext__()
            except StopAsyncIteration:
                break
            if not ln:
                continue
            chunks.append(ln)
    finally:
        try:
            await agen.aclose()
        except Exception:
            _ = None

    chunks = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in chunks]
    joined = "".join(chunks)
    assert "fallback ok" in joined
    assert any("data: [DONE]" in c for c in chunks)
    assert provider_manager.success_records == ["openai"]
    assert any(
        entry.get("selected_provider") == "openai" and entry.get("queued") is True
        for entry in metrics.fallback_successes
    )


@pytest.mark.asyncio
async def test_execute_streaming_call_finalize_runs_without_refund_cb(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    captured: dict[str, object] = {}

    def llm_call_func():
        def _stream():
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    def fake_create_streaming_response_with_timeout(*_args, finalize_callback=None, **_kwargs):
        captured["finalize_callback"] = finalize_callback

        async def _gen():
            if callable(finalize_callback):
                await finalize_callback(success=False, cancelled=False, error=True)
            yield "data: [DONE]\n\n"

        return _gen()

    async def save_message_fn(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_service, "create_streaming_response_with_timeout", fake_create_streaming_response_with_timeout)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-finalize-1",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-test",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
    )

    assert isinstance(resp, StreamingResponse)

    agen = resp.body_iterator
    try:
        await agen.__anext__()
    except StopAsyncIteration:
        _ = None
    finally:
        with contextlib.suppress(Exception):
            # Best effort close. Some test wrappers may already exhaust/close the iterator.
            await agen.aclose()

    assert callable(captured.get("finalize_callback"))
    assert any(call[3] == "stream_error" for call in metrics.llm_calls)
    assert provider_manager.failure_records


@pytest.mark.asyncio
async def test_execute_streaming_call_refund_cb_still_conditional(monkeypatch):
    metrics = _DummyMetrics()
    provider_manager = _DummyProviderManager()
    refund_calls: list[dict[str, bool]] = []

    def llm_call_func():
        def _stream():
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    async def rg_refund_cb(*, cancelled: bool, error: bool):
        refund_calls.append({"cancelled": cancelled, "error": error})

    def fake_create_streaming_response_with_timeout(*_args, finalize_callback=None, **_kwargs):
        async def _gen():
            if callable(finalize_callback):
                await finalize_callback(success=False, cancelled=True, error=False)
            yield "data: [DONE]\n\n"

        return _gen()

    async def save_message_fn(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_service, "create_streaming_response_with_timeout", fake_create_streaming_response_with_timeout)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    resp = await execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={
            "api_endpoint": "openai",
            "messages_payload": [],
            "model": "gpt-test",
            "streaming": True,
        },
        selected_provider="openai",
        provider="openai",
        model="gpt-test",
        request_json="{}",
        request=request,
        metrics=metrics,
        provider_manager=provider_manager,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv-finalize-2",
        character_card_for_context=None,
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client-test",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=llm_call_func,
        refresh_provider_params=lambda _provider: ({}, None),
        moderation_getter=lambda: _DummyModeration(),
        rg_refund_cb=rg_refund_cb,
    )

    agen = resp.body_iterator
    try:
        await agen.__anext__()
    except StopAsyncIteration:
        _ = None
    finally:
        with contextlib.suppress(Exception):
            await agen.aclose()

    assert refund_calls == [{"cancelled": True, "error": False}]
    assert any(call[3] == "stream_cancelled" for call in metrics.llm_calls)


def test_merge_api_keys_prefers_dynamic_over_module():


    module_keys = {"openai": "module-key", "anthropic": "module-anthropic"}
    dynamic_keys = {"openai": "dynamic-key", "anthropic": ""}

    raw_openai, normalized_openai = merge_api_keys_for_provider(
        "openai",
        module_keys,
        dynamic_keys,
        {},
    )
    assert raw_openai == "dynamic-key"
    assert normalized_openai == "dynamic-key"

    raw_anthropic, normalized_anthropic = merge_api_keys_for_provider(
        "anthropic",
        module_keys,
        dynamic_keys,
        {},
    )
    assert raw_anthropic == "module-anthropic"
    assert normalized_anthropic == "module-anthropic"
