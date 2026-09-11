"""Credential-taxonomy regressions at Chat execution fallback boundaries."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    ProviderOverridePolicyError,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatProviderError,
    ProviderCredentialTerminalError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    execute_non_stream_call,
    execute_streaming_call,
)
from tldw_Server_API.app.core.Chat.provider_manager import ProviderManager
from tldw_Server_API.app.core.Chat.request_queue import RateLimitedQueue

_PRIVATE_PRIMARY_DETAIL = "fallback-taxonomy-secret-/srv/provider"
_CREDENTIAL_CODES = (
    "provider_disabled",
    "model_not_allowed",
    "invalid_provider_credentials",
    "credential_store_unavailable",
    "credential_scope_revoked",
)


class _StreamTracker:
    """No-op stream tracker matching the production metrics boundary."""

    def add_chunk(self) -> None:
        return None

    def add_heartbeat(self) -> None:
        return None


class _Metrics:
    """Minimal metrics recorder used by the execution-boundary tests."""

    def __init__(self) -> None:
        self.llm_calls: list[tuple[str, str, bool, str | None]] = []

    def track_llm_call(
        self,
        provider: str,
        model: str,
        _latency: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        self.llm_calls.append((provider, model, success, error_type))

    def track_provider_fallback_success(self, **_metadata: Any) -> None:
        return None

    def track_tokens(self, **_metadata: Any) -> None:
        return None

    def track_run_first_completion_proxy(self, **_metadata: Any) -> None:
        return None

    @asynccontextmanager
    async def track_streaming(
        self,
        _conversation_id: str,
    ) -> AsyncIterator[_StreamTracker]:
        """Provide the context-manager interface used by the real stream wrapper."""

        yield _StreamTracker()


class _ProviderManager:
    """Choose one deterministic fallback and retain only bounded failures."""

    def __init__(self) -> None:
        self.failure_errors: list[BaseException] = []
        self.failure_records: list[tuple[str, BaseException]] = []
        self.fallback_requests: list[tuple[str, ...]] = []

    def get_available_provider(self, exclude: list[str] | None = None) -> str:
        self.fallback_requests.append(tuple(exclude or ()))
        return "openai"

    def record_failure(self, provider: str, error: BaseException) -> None:
        self.failure_errors.append(error)
        self.failure_records.append((provider, error))

    def record_success(self, _provider: str, _latency: float) -> None:
        return None


class _DisabledModeration:
    """Small disabled moderation service accepted by both Chat paths."""

    class _Policy:
        enabled = False
        output_enabled = False

    def get_effective_policy(self, *_args: Any, **_kwargs: Any) -> _Policy:
        return self._Policy()

    def evaluate_action(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def check_text(self, *_args: Any, **_kwargs: Any) -> tuple[bool, None]:
        return False, None

    def redact_text(self, text: str, *_args: Any, **_kwargs: Any) -> str:
        return text


class _InlineThreadQueue:
    """Run a queue processor in its required sync worker and retain its error."""

    def __init__(
        self,
        *,
        release_after_errors: int = 0,
        error_release: threading.Event | None = None,
    ) -> None:
        self.errors: dict[str, BaseException] = {}
        self._release_after_errors = release_after_errors
        self._error_release = error_release

    def is_running(self) -> bool:
        return True

    async def enqueue(
        self,
        *,
        client_id: str,
        processor: Callable[[], Any],
        **_kwargs: Any,
    ) -> Any:
        try:
            return await asyncio.to_thread(processor)
        except Exception as exc:  # noqa: BLE001 - test observes the trust boundary
            self.errors[client_id] = exc
            if (
                self._error_release is not None
                and len(self.errors) >= self._release_after_errors
            ):
                self._error_release.set()
            raise


class _ObservedRateLimitedQueue(RateLimitedQueue):
    """Retain production queue futures and channels for boundary assertions."""

    def __init__(self, *, max_concurrent: int) -> None:
        super().__init__(
            max_queue_size=16,
            max_concurrent=max_concurrent,
            timeout=2.0,
            global_rate_limit=100,
            per_client_rate_limit=100,
        )
        self.futures: dict[str, asyncio.Future[Any]] = {}
        self.channels: dict[str, asyncio.Queue[Any]] = {}

    async def enqueue(self, *args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        future = await super().enqueue(*args, **kwargs)
        request_id = kwargs.get("request_id")
        assert isinstance(request_id, str)
        self.futures[request_id] = future
        stream_channel = kwargs.get("stream_channel")
        if isinstance(stream_channel, asyncio.Queue):
            self.channels[request_id] = stream_channel
        return future


def _credential_error(code: str) -> ByokResolutionError:
    """Build the canonical policy or BYOK terminal error for one code."""

    if code in {"provider_disabled", "model_not_allowed"}:
        return ProviderOverridePolicyError(code, "openai")
    return ByokResolutionError(code, "openai")


def _certified_primary_failure() -> ChatProviderError:
    """Return a provider failure explicitly certified safe for replay."""

    error = ChatProviderError(
        provider="anthropic",
        message=_PRIVATE_PRIMARY_DETAIL,
        status_code=502,
    )
    error.upstream_dispatched = False
    error.output_emitted = False
    error.allow_non_stream_fallback = True
    return error


def _failing_primary_call() -> None:
    raise _certified_primary_failure()


async def _noop_save_message(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> None:
    """Wait for a worker-thread gate without occupying another worker."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("Timed out waiting for worker-thread event")
        await asyncio.sleep(0.001)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )


def _common_call_kwargs(
    *,
    client_id: str,
    metrics: _Metrics,
    provider_manager: _ProviderManager,
    refresh_provider_params: Callable[[str], Any],
    streaming: bool,
) -> dict[str, Any]:
    return {
        "current_loop": asyncio.get_running_loop(),
        "cleaned_args": {
            "api_endpoint": "anthropic",
            "api_key": "primary-key",
            "messages_payload": [],
            "model": "claude-3",
            "streaming": streaming,
        },
        "selected_provider": "anthropic",
        "provider": "anthropic",
        "model": "claude-3",
        "request_json": "{}",
        "request": _request(),
        "metrics": metrics,
        "provider_manager": provider_manager,
        "templated_llm_payload": [],
        "should_persist": False,
        "final_conversation_id": client_id,
        "character_card_for_context": None,
        "chat_db": None,
        "save_message_fn": _noop_save_message,
        "audit_service": None,
        "audit_context": None,
        "client_id": client_id,
        "queue_execution_enabled": streaming,
        "enable_provider_fallback": True,
        "llm_call_func": _failing_primary_call,
        "refresh_provider_params": refresh_provider_params,
        "moderation_getter": lambda: _DisabledModeration(),
    }


async def _response_wire(response: StreamingResponse) -> str:
    """Consume one streaming response into its public wire representation."""

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, (bytes, bytearray)):
            chunks.append(chunk.decode())
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


async def _queue_future_outcome(
    future: asyncio.Future[Any],
    *,
    timeout: float = 2.0,
) -> Any:
    """Await one observed queue future without propagating its terminal error."""

    outcomes = await asyncio.wait_for(
        asyncio.gather(future, return_exceptions=True),
        timeout=timeout,
    )
    return outcomes[0]


def _terminal_wire_code(wire: str) -> str:
    """Return the code from the single public terminal SSE error frame."""

    error_payloads = [
        payload["error"]
        for line in wire.splitlines()
        if line.startswith("data: {")
        for payload in [json.loads(line.removeprefix("data: "))]
        if isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and isinstance(payload["error"].get("code"), str)
    ]
    assert len(error_payloads) == 1
    return str(error_payloads[0]["code"])


def _successful_nonstream_response(content: str) -> dict[str, object]:
    """Return the smallest provider response accepted by non-stream Chat."""

    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("code", _CREDENTIAL_CODES)
async def test_nonstream_fallback_preserves_typed_credential_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    """Direct fallback refresh keeps trusted credential codes terminal and safe."""

    metrics = _Metrics()
    provider_manager = _ProviderManager()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)

    def refresh_provider(_provider: str) -> None:
        raise _credential_error(code)

    with pytest.raises(ProviderCredentialTerminalError) as captured:
        await execute_non_stream_call(
            **_common_call_kwargs(
                client_id=f"nonstream-{code}",
                metrics=metrics,
                provider_manager=provider_manager,
                refresh_provider_params=refresh_provider,
                streaming=False,
            )
        )

    assert captured.value.code == code
    assert str(captured.value) == code
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert _PRIVATE_PRIMARY_DETAIL not in repr(captured.value)
    assert provider_manager.fallback_requests == [("anthropic",)]
    assert all(
        failed_provider != "openai"
        for failed_provider, _error in provider_manager.failure_records
    )
    assert _PRIVATE_PRIMARY_DETAIL not in "".join(
        str(error) for error in provider_manager.failure_errors
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("code", _CREDENTIAL_CODES)
async def test_queued_stream_fallback_preserves_typed_credential_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    """Queued refresh retains terminal type internally and exact code on SSE."""

    queue = _InlineThreadQueue()
    metrics = _Metrics()
    provider_manager = _ProviderManager()
    client_id = f"queued-{code}"
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)

    def refresh_provider(_provider: str) -> None:
        raise _credential_error(code)

    response = await execute_streaming_call(
        **_common_call_kwargs(
            client_id=client_id,
            metrics=metrics,
            provider_manager=provider_manager,
            refresh_provider_params=refresh_provider,
            streaming=True,
        ),
        provider_factory_timeout=1.0,
    )
    wire = await _response_wire(response)

    boundary_error = queue.errors[client_id]
    assert isinstance(boundary_error, ProviderCredentialTerminalError)
    assert boundary_error.code == code
    assert boundary_error.__cause__ is None
    assert boundary_error.__context__ is None
    assert _terminal_wire_code(wire) == code
    assert wire.count("data: [DONE]") == 1
    assert _PRIVATE_PRIMARY_DETAIL not in wire
    assert _PRIVATE_PRIMARY_DETAIL not in repr(boundary_error)
    assert all(
        failed_provider != "openai"
        for failed_provider, _error in provider_manager.failure_records
    )


@pytest.mark.asyncio
async def test_queued_stream_non_byok_refresh_failure_records_sanitized_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untyped refresh failures still update fallback health without leaking detail."""

    sentinel = "queued-refresh-secret-/srv/provider"
    queue = _InlineThreadQueue()
    provider_manager = _ProviderManager()
    client_id = "queued-non-byok-refresh"
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)

    def refresh_provider(_provider: str) -> None:
        raise ValueError(sentinel)

    response = await execute_streaming_call(
        **_common_call_kwargs(
            client_id=client_id,
            metrics=_Metrics(),
            provider_manager=provider_manager,
            refresh_provider_params=refresh_provider,
            streaming=True,
        ),
        provider_factory_timeout=1.0,
    )
    wire = await _response_wire(response)

    boundary_error = queue.errors[client_id]
    assert isinstance(boundary_error, SanitizedProviderStreamError)
    assert boundary_error.code == "provider_unavailable"
    fallback_failures = [
        error
        for failed_provider, error in provider_manager.failure_records
        if failed_provider == "openai"
    ]
    assert len(fallback_failures) == 1
    assert isinstance(fallback_failures[0], SanitizedProviderStreamError)
    assert fallback_failures[0].code == "provider_unavailable"
    assert _terminal_wire_code(wire) == "provider_unavailable"
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    assert sentinel not in repr(boundary_error)
    assert sentinel not in repr(fallback_failures[0])


@pytest.mark.asyncio
async def test_concurrent_queued_policy_rotation_keeps_codes_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent policy denials remain isolated across an event-gated refresh."""

    queue = _InlineThreadQueue()
    started = {
        "policy-disabled": asyncio.Event(),
        "policy-model": asyncio.Event(),
    }
    release = asyncio.Event()
    cases = {
        "policy-disabled": "provider_disabled",
        "policy-model": "model_not_allowed",
    }
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)

    async def invoke(client_id: str, code: str) -> tuple[str, str]:
        async def refresh_provider(_provider: str) -> None:
            started[client_id].set()
            await release.wait()
            raise _credential_error(code)

        response = await execute_streaming_call(
            **_common_call_kwargs(
                client_id=client_id,
                metrics=_Metrics(),
                provider_manager=_ProviderManager(),
                refresh_provider_params=refresh_provider,
                streaming=True,
            ),
            provider_factory_timeout=2.0,
        )
        return client_id, await _response_wire(response)

    tasks = [
        asyncio.create_task(invoke(client_id, code))
        for client_id, code in cases.items()
    ]
    try:
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), timeout=1.0) for event in started.values())
        )
        release.set()
        responses = dict(await asyncio.gather(*tasks))
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert set(queue.errors) == set(cases)
    for client_id, expected_code in cases.items():
        boundary_error = queue.errors[client_id]
        assert isinstance(boundary_error, ProviderCredentialTerminalError)
        assert boundary_error.code == expected_code
        assert _terminal_wire_code(responses[client_id]) == expected_code
        assert _PRIVATE_PRIMARY_DETAIL not in responses[client_id]


@pytest.mark.asyncio
async def test_concurrent_policy_denials_do_not_poison_allowed_fallback_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-local model policy denials never alter global provider health."""

    denial_count = 5
    allowed_primary_started = threading.Event()
    allowed_primary_release = threading.Event()
    queue = _InlineThreadQueue(
        release_after_errors=denial_count,
        error_release=allowed_primary_release,
    )
    provider_manager = ProviderManager(
        ["anthropic", "openai"],
        primary_provider="anthropic",
    )
    denial_started = [asyncio.Event() for _ in range(denial_count)]
    release_denials = asyncio.Event()
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    async def successful_fallback_call(**_kwargs: Any) -> dict[str, object]:
        return _successful_nonstream_response("allowed fallback")

    async def noop_usage_log(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        successful_fallback_call,
    )
    monkeypatch.setattr(chat_service, "log_llm_usage", noop_usage_log)

    async def invoke_denied(index: int) -> str:
        client_id = f"denied-model-{index}"

        async def denied_refresh(_provider: str) -> None:
            denial_started[index].set()
            await release_denials.wait()
            raise ProviderOverridePolicyError("model_not_allowed", "openai")

        response = await execute_streaming_call(
            **_common_call_kwargs(
                client_id=client_id,
                metrics=_Metrics(),
                provider_manager=provider_manager,
                refresh_provider_params=denied_refresh,
                streaming=True,
            ),
            provider_factory_timeout=2.0,
        )
        return await _response_wire(response)

    def gated_allowed_primary() -> None:
        allowed_primary_started.set()
        if not allowed_primary_release.wait(timeout=2.0):
            raise AssertionError("Policy denials never reached the fallback boundary")
        raise _certified_primary_failure()

    def allowed_refresh(provider: str) -> tuple[dict[str, Any], str]:
        assert provider == "openai"
        return (
            {
                "api_endpoint": provider,
                "api_key": "allowed-key",
                "messages_payload": [],
                "model": "gpt-4o",
                "streaming": False,
            },
            "gpt-4o",
        )

    denial_tasks = [
        asyncio.create_task(invoke_denied(index)) for index in range(denial_count)
    ]
    allowed_task = asyncio.create_task(
        execute_non_stream_call(
            **{
                **_common_call_kwargs(
                    client_id="allowed-model",
                    metrics=_Metrics(),
                    provider_manager=provider_manager,
                    refresh_provider_params=allowed_refresh,
                    streaming=False,
                ),
                "llm_call_func": gated_allowed_primary,
            }
        )
    )
    try:
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), timeout=1.0) for event in denial_started)
        )
        await _wait_for_thread_event(allowed_primary_started)
        release_denials.set()
        denial_wires = await asyncio.gather(*denial_tasks)
        allowed_result = await allowed_task
    finally:
        release_denials.set()
        allowed_primary_release.set()
        await asyncio.gather(*denial_tasks, allowed_task, return_exceptions=True)

    assert all(_terminal_wire_code(wire) == "model_not_allowed" for wire in denial_wires)
    assert allowed_result["choices"][0]["message"]["content"] == "allowed fallback"
    assert provider_manager.health_status["openai"].failure_count == 0
    assert provider_manager.health_status["openai"].consecutive_failures == 0
    assert provider_manager.circuit_breakers["openai"].state == "CLOSED"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", _CREDENTIAL_CODES)
async def test_real_queue_terminalizes_trusted_credential_code_once(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    """The production queue preserves one typed code through future and channel."""

    request_id = f"real-queue-{code}"
    sentinel = f"real-queue-{code}-secret-/srv/provider"
    queue = _ObservedRateLimitedQueue(max_concurrent=1)
    provider_manager = _ProviderManager()
    await queue.start(num_workers=1)
    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)

    def refresh_provider(_provider: str) -> None:
        error = _credential_error(code)
        error.args = (sentinel,)
        raise error

    try:
        response = await execute_streaming_call(
            **_common_call_kwargs(
                client_id=request_id,
                metrics=_Metrics(),
                provider_manager=provider_manager,
                refresh_provider_params=refresh_provider,
                streaming=True,
            ),
            queue_request_id=request_id,
            provider_factory_timeout=2.0,
        )
        wire = await asyncio.wait_for(_response_wire(response), timeout=2.0)
        future_outcome = await _queue_future_outcome(queue.futures[request_id])
    finally:
        await asyncio.wait_for(queue.stop(), timeout=2.0)

    assert isinstance(future_outcome, ProviderCredentialTerminalError)
    assert future_outcome.code == code
    assert future_outcome.__cause__ is None
    assert future_outcome.__context__ is None
    assert queue.channels[request_id].maxsize == 1
    assert queue.channels[request_id].empty()
    assert _terminal_wire_code(wire) == code
    assert wire.count(f'"code": "{code}"') == 1
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    assert sentinel not in repr(future_outcome)
    assert request_id not in queue._active_request_ids
    assert any(
        activity.get("request_id") == request_id
        and activity.get("result") == "error"
        and activity.get("error_type") == "ProviderCredentialTerminalError"
        for activity in queue.get_recent_activity()
    )
    assert all(
        failed_provider != "openai"
        for failed_provider, _error in provider_manager.failure_records
    )


@pytest.mark.asyncio
async def test_real_queue_non_byok_refresh_is_sanitized_and_recorded_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production queue bounds untyped refresh failures and records health."""

    request_id = "real-queue-non-byok-refresh"
    sentinel = "real-queue-refresh-secret-/srv/provider"
    queue = _ObservedRateLimitedQueue(max_concurrent=1)
    provider_manager = _ProviderManager()
    await queue.start(num_workers=1)
    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)

    def refresh_provider(_provider: str) -> None:
        raise ValueError(sentinel)

    try:
        response = await execute_streaming_call(
            **_common_call_kwargs(
                client_id=request_id,
                metrics=_Metrics(),
                provider_manager=provider_manager,
                refresh_provider_params=refresh_provider,
                streaming=True,
            ),
            queue_request_id=request_id,
            provider_factory_timeout=2.0,
        )
        wire = await asyncio.wait_for(_response_wire(response), timeout=2.0)
        future_outcome = await _queue_future_outcome(queue.futures[request_id])
    finally:
        await asyncio.wait_for(queue.stop(), timeout=2.0)

    assert isinstance(future_outcome, SanitizedProviderStreamError)
    assert future_outcome.code == "provider_unavailable"
    fallback_failures = [
        error
        for failed_provider, error in provider_manager.failure_records
        if failed_provider == "openai"
    ]
    assert len(fallback_failures) == 1
    assert isinstance(fallback_failures[0], SanitizedProviderStreamError)
    assert fallback_failures[0].code == "provider_unavailable"
    assert queue.channels[request_id].maxsize == 1
    assert queue.channels[request_id].empty()
    assert _terminal_wire_code(wire) == "provider_unavailable"
    assert wire.count('"code": "provider_unavailable"') == 1
    assert wire.count("data: [DONE]") == 1
    assert sentinel not in wire
    assert sentinel not in repr(future_outcome)
    assert sentinel not in repr(fallback_failures[0])


@pytest.mark.asyncio
async def test_real_queue_concurrent_policy_codes_and_breaker_health_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent queued policy denials cannot cross wires or poison fallback."""

    policy_codes = (
        "provider_disabled",
        "model_not_allowed",
        "provider_disabled",
        "model_not_allowed",
        "provider_disabled",
    )
    queue = _ObservedRateLimitedQueue(max_concurrent=len(policy_codes))
    provider_manager = ProviderManager(
        ["anthropic", "openai"],
        primary_provider="anthropic",
    )
    denial_started = [asyncio.Event() for _ in policy_codes]
    release_denials = asyncio.Event()
    allowed_primary_started = threading.Event()
    allowed_primary_release = threading.Event()
    responses: dict[str, StreamingResponse] = {}
    await queue.start(num_workers=len(policy_codes))
    monkeypatch.setenv("CHAT_STREAM_CHANNEL_MAXSIZE", "1")
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: queue)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    async def successful_fallback_call(**_kwargs: Any) -> dict[str, object]:
        return _successful_nonstream_response("allowed fallback")

    async def noop_usage_log(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        successful_fallback_call,
    )
    monkeypatch.setattr(chat_service, "log_llm_usage", noop_usage_log)

    def refresh_for(index: int, code: str) -> Callable[[str], Any]:
        async def refresh_provider(_provider: str) -> None:
            denial_started[index].set()
            await release_denials.wait()
            raise ProviderOverridePolicyError(code, "openai")

        return refresh_provider

    try:
        for index, code in enumerate(policy_codes):
            request_id = f"real-queue-policy-{index}-{code}"
            responses[request_id] = await execute_streaming_call(
                **_common_call_kwargs(
                    client_id=request_id,
                    metrics=_Metrics(),
                    provider_manager=provider_manager,
                    refresh_provider_params=refresh_for(index, code),
                    streaming=True,
                ),
                queue_request_id=request_id,
                provider_factory_timeout=3.0,
            )

        def gated_allowed_primary() -> None:
            allowed_primary_started.set()
            if not allowed_primary_release.wait(timeout=3.0):
                raise AssertionError("Queued policy denials never terminalized")
            raise _certified_primary_failure()

        def allowed_refresh(provider: str) -> tuple[dict[str, Any], str]:
            assert provider == "openai"
            return (
                {
                    "api_endpoint": provider,
                    "api_key": "allowed-key",
                    "messages_payload": [],
                    "model": "gpt-4o",
                    "streaming": False,
                },
                "gpt-4o",
            )

        allowed_task = asyncio.create_task(
            execute_non_stream_call(
                **{
                    **_common_call_kwargs(
                        client_id="real-queue-allowed-model",
                        metrics=_Metrics(),
                        provider_manager=provider_manager,
                        refresh_provider_params=allowed_refresh,
                        streaming=False,
                    ),
                    "llm_call_func": gated_allowed_primary,
                }
            )
        )
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), timeout=2.0) for event in denial_started)
        )
        await _wait_for_thread_event(allowed_primary_started, timeout=2.0)
        release_denials.set()
        future_outcomes = await asyncio.gather(
            *(
                _queue_future_outcome(queue.futures[request_id], timeout=3.0)
                for request_id in responses
            )
        )
        allowed_primary_release.set()
        denial_wires, allowed_result = await asyncio.gather(
            asyncio.gather(
                *(
                    asyncio.wait_for(_response_wire(response), timeout=2.0)
                    for response in responses.values()
                )
            ),
            allowed_task,
        )
    finally:
        release_denials.set()
        allowed_primary_release.set()
        if "allowed_task" in locals():
            await asyncio.gather(allowed_task, return_exceptions=True)
        await asyncio.wait_for(queue.stop(), timeout=2.0)

    assert len(future_outcomes) == len(policy_codes)
    assert len(denial_wires) == len(policy_codes)
    for index, (request_id, wire) in enumerate(zip(responses, denial_wires)):
        expected_code = policy_codes[index]
        future_outcome = future_outcomes[index]
        assert isinstance(future_outcome, ProviderCredentialTerminalError)
        assert future_outcome.code == expected_code
        assert _terminal_wire_code(wire) == expected_code
        assert wire.count(f'"code": "{expected_code}"') == 1
        assert wire.count("data: [DONE]") == 1
        assert queue.channels[request_id].maxsize == 1
        assert queue.channels[request_id].empty()
    assert allowed_result["choices"][0]["message"]["content"] == "allowed fallback"
    assert provider_manager.health_status["openai"].failure_count == 0
    assert provider_manager.health_status["openai"].consecutive_failures == 0
    assert provider_manager.circuit_breakers["openai"].state == "CLOSED"
