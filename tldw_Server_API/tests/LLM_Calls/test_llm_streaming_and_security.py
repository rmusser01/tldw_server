import asyncio
import json
import threading

import pytest
from loguru import logger


class _FailingSyncSseResponse:
    def __init__(
        self,
        *,
        sentinel: str,
        ready: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self._sentinel = sentinel
        self._ready = ready
        self._release = release

    def iter_lines(self, *, decode_unicode: bool = True):
        del decode_unicode
        yield 'data: {"choices":[{"delta":{"content":"safe"}}]}'
        if self._ready is not None:
            self._ready.set()
        if self._release is not None and not self._release.wait(1.0):
            raise TimeoutError("test release was not signalled")
        raise RuntimeError(self._sentinel)


class _FailingAsyncSseResponse:
    def __init__(
        self,
        *,
        sentinel: str,
        ready: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._sentinel = sentinel
        self._ready = ready
        self._release = release

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"safe"}}]}'
        if self._ready is not None:
            self._ready.set()
        if self._release is not None:
            await asyncio.wait_for(self._release.wait(), 1.0)
        raise RuntimeError(self._sentinel)


def _assert_bounded_stream_error(chunks: list[str], sentinel: str) -> None:
    rendered = "".join(chunks)
    assert "safe" in rendered
    assert sentinel not in rendered
    error_frames = [
        json.loads(chunk.removeprefix("data:").strip())
        for chunk in chunks
        if '"error"' in chunk
    ]
    assert error_frames == [
        {
            "error": {
                "code": "provider_unavailable",
                "message": "The chat service provider is currently unavailable.",
                "type": "moonshot_stream_error",
            }
        }
    ]


def test_sync_sse_helper_bounds_transport_exception_and_log_detail() -> None:
    from tldw_Server_API.app.core.LLM_Calls.streaming import iter_sse_lines_requests

    sentinel = "sync-upstream-secret-/srv/provider?key=secret"
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        chunks = list(
            iter_sse_lines_requests(
                _FailingSyncSseResponse(sentinel=sentinel),
                provider="moonshot",
            )
        )
    finally:
        logger.remove(sink_id)

    _assert_bounded_stream_error(chunks, sentinel)
    assert sentinel not in "".join(logs)
    assert "RuntimeError" in "".join(logs)


@pytest.mark.asyncio
async def test_async_sse_helper_bounds_transport_exception_and_log_detail() -> None:
    from tldw_Server_API.app.core.LLM_Calls.streaming import aiter_sse_lines_httpx

    sentinel = "async-upstream-secret-/srv/provider?key=secret"
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        chunks = [
            chunk
            async for chunk in aiter_sse_lines_httpx(
                _FailingAsyncSseResponse(sentinel=sentinel),
                provider="moonshot",
            )
        ]
    finally:
        logger.remove(sink_id)

    _assert_bounded_stream_error(chunks, sentinel)
    assert sentinel not in "".join(logs)
    assert "RuntimeError" in "".join(logs)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_sse_helpers_keep_transport_failures_bounded_and_isolated() -> None:
    from tldw_Server_API.app.core.LLM_Calls.streaming import (
        aiter_sse_lines_httpx,
        iter_sse_lines_requests,
    )

    sync_sentinel = "sync-concurrent-secret-/srv/alpha?key=one"
    async_sentinel = "async-concurrent-secret-/srv/beta?key=two"
    sync_ready = threading.Event()
    sync_release = threading.Event()
    async_ready = asyncio.Event()
    async_release = asyncio.Event()

    sync_task = asyncio.create_task(
        asyncio.to_thread(
            list,
            iter_sse_lines_requests(
                _FailingSyncSseResponse(
                    sentinel=sync_sentinel,
                    ready=sync_ready,
                    release=sync_release,
                ),
                provider="moonshot",
            ),
        )
    )

    async def collect_async() -> list[str]:
        return [
            chunk
            async for chunk in aiter_sse_lines_httpx(
                _FailingAsyncSseResponse(
                    sentinel=async_sentinel,
                    ready=async_ready,
                    release=async_release,
                ),
                provider="moonshot",
            )
        ]

    async_task = asyncio.create_task(collect_async())
    assert await asyncio.to_thread(sync_ready.wait, 1.0)
    await asyncio.wait_for(async_ready.wait(), 1.0)
    sync_release.set()
    async_release.set()
    sync_chunks, async_chunks = await asyncio.gather(sync_task, async_task)

    _assert_bounded_stream_error(sync_chunks, sync_sentinel)
    _assert_bounded_stream_error(async_chunks, async_sentinel)
    combined = "".join(sync_chunks + async_chunks)
    assert sync_sentinel not in combined
    assert async_sentinel not in combined


@pytest.mark.asyncio
async def test_real_openai_401_and_403_refresh_certificates_are_isolated(monkeypatch):
    import httpx

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat.Chat_Deps import SanitizedProviderStreamError
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    sentinels = {
        401: "openai-401-secret-/srv/provider",
        403: "openai-403-secret-/srv/provider",
    }
    both_requests_entered = threading.Barrier(2)

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.request = httpx.Request("POST", "https://api.openai.test/v1/chat/completions")
            self.text = sentinels[status_code]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def json(self):
            return {"error": {"message": sentinels[self.status_code]}}

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                sentinels[self.status_code],
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

        def iter_lines(self):
            return iter(())

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, _method, _url, *, headers, **_kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            both_requests_entered.wait(timeout=5.0)
            return Response(int(token))

    monkeypatch.setattr(openai_adapter, "http_client_factory", lambda **_kwargs: Client())

    def invoke(status_code):
        state = {}
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "api_key": str(status_code),
            "stream": True,
        }
        wrapped = chat_endpoint._sanitize_provider_stream_call(
            lambda: OpenAIAdapter().stream(request),
            state,
        )()
        with pytest.raises(SanitizedProviderStreamError) as captured:
            next(wrapped)
        return captured.value, state

    logs = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        (error_401, state_401), (error_403, state_403) = await asyncio.gather(
            asyncio.to_thread(invoke, 401),
            asyncio.to_thread(invoke, 403),
        )
    finally:
        logger.remove(sink_id)

    assert error_401.code == error_403.code == "provider_authentication_failed"
    assert getattr(error_401, "credential_refresh_retry_safe", False) is True
    assert getattr(error_403, "credential_refresh_retry_safe", False) is False
    assert state_401.get("credential_refresh_retry_certified") is True
    assert state_403.get("credential_refresh_retry_certified") is False
    assert error_401.__cause__ is None
    assert error_401.__context__ is None
    assert error_403.__cause__ is None
    assert error_403.__context__ is None
    rendered = str(error_401) + str(error_403) + "".join(logs)
    assert sentinels[401] not in rendered
    assert sentinels[403] not in rendered


def test_google_stream_emits_done_once(monkeypatch):
    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    return None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def iter_lines(self):
                    first_chunk = {
                        "candidates": [
                            {"content": {"parts": [{"text": "hello"}]}}
                        ]
                    }
                    return iter(
                        [
                            f"data: {json.dumps(first_chunk)}".encode(),
                            b"data: [DONE]",
                        ]
                    )

                def close(self):
                    return None

            return _Resp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.http_client_factory",
        lambda *a, **k: _Client(),
    )

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

    gen = perform_chat_api_call(
        api_provider="google",
        messages=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="gemini-2.5-flash",
        streaming=True,
    )
    chunks = list(gen)

    done_count = sum(1 for c in chunks if c.strip().lower() == "data: [done]")
    assert done_count == 1, f"Expected exactly one [DONE], got {done_count}. Chunks: {chunks}"


def test_huggingface_headers_are_masked(monkeypatch):
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call as _perform_chat

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"id": "ok", "choices": [{"message": {"content": "hi"}}]}

                def close(self):
                    return None

            return _Resp()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.http_client_factory",
        lambda *a, **k: _Client(),
    )

    captured_debug = []

    def _fake_debug(msg, *args, **kwargs):
        rendered = str(msg)
        if args:
            try:
                rendered = rendered.format(*args)
            except Exception:
                rendered = f"{msg} {args}"
        captured_debug.append(rendered)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.logger.debug",
        _fake_debug,
    )

    secret = "sk-ABCDEF1234567890"
    _perform_chat(
        api_provider="huggingface",
        messages=[{"role": "user", "content": "hi"}],
        api_key=secret,
        streaming=False,
        model="test/Model-Stub",
    )

    joined = "\n".join(captured_debug)
    assert "HuggingFace headers:" in joined
    assert secret not in joined
    assert "***" in joined


def test_http_400_logging_omits_prompt_and_request_body(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls import error_utils

    secret_prompt = "SECRET_PROMPT_CONTENT"
    secret_key = "sk-secret-should-not-log"
    body = {
        "error": {
            "message": f"Invalid request for prompt {secret_prompt}",
            "type": "invalid_request_error",
            "code": "bad_request",
        },
        "messages": [{"role": "user", "content": secret_prompt}],
        "api_key": secret_key,
    }

    class _Resp:
        status_code = 400
        text = json.dumps(body)

        def json(self):
            return body

    class _Exc(Exception):
        response = _Resp()

    warnings = []
    errors = []

    class _Logger:
        def warning(self, msg):
            warnings.append(str(msg))

        def error(self, msg):
            errors.append(str(msg))

    monkeypatch.setattr(error_utils, "logger", _Logger())

    exc = _Exc("upstream 400")
    error_utils.log_http_400_body("openai", exc)
    with pytest.raises(ChatBadRequestError):
        error_utils.raise_chat_error_from_http("openai", exc)

    rendered_logs = "\n".join(warnings + errors)
    assert secret_prompt not in rendered_logs
    assert secret_key not in rendered_logs
    assert "messages" not in rendered_logs
    assert "invalid_request_error" in rendered_logs


@pytest.mark.asyncio
async def test_wrap_sync_stream_applies_backpressure_and_closes_on_cancel():
    from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream

    class _FastIterator:
        def __init__(self):
            self.yielded = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.closed:
                raise StopIteration
            self.yielded += 1
            return f"chunk-{self.yielded}"

        def close(self):
            self.closed = True

    source = _FastIterator()
    stream = wrap_sync_stream(source, max_queue_size=1)

    assert await stream.__anext__() == "chunk-1"
    await asyncio.sleep(0.1)

    assert source.yielded <= 3
    await stream.aclose()
    await asyncio.sleep(0.1)

    yielded_after_close = source.yielded
    assert source.closed is True
    await asyncio.sleep(0.1)
    assert source.yielded == yielded_after_close


@pytest.mark.asyncio
async def test_wrap_sync_stream_does_not_use_default_executor_for_delivery(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import streaming

    async def fail_to_thread(*args, **kwargs):
        raise AssertionError("wrap_sync_stream should not use the default executor for chunk delivery")

    monkeypatch.setattr(streaming.asyncio, "to_thread", fail_to_thread)

    chunks = []
    async for chunk in streaming.wrap_sync_stream(iter(["chunk-1", "chunk-2"]), max_queue_size=1):
        chunks.append(chunk)

    assert chunks == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_wrap_sync_stream_logs_close_errors(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls import streaming

    sentinel = "sync-close-secret-/srv/provider"
    close_attempted = threading.Event()

    class _ClosingIterator:
        def __init__(self):
            self._done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            self._done = True
            return "chunk"

        def close(self):
            close_attempted.set()
            raise RuntimeError(sentinel)

    debug_messages = []

    def fake_debug(message, *args, **kwargs):
        debug_messages.append(str(message).format(*args))

    monkeypatch.setattr(streaming.logger, "debug", fake_debug)

    chunks = []
    async for chunk in streaming.wrap_sync_stream(_ClosingIterator()):
        chunks.append(chunk)

    assert await asyncio.to_thread(close_attempted.wait, 1.0)
    assert chunks == ["chunk"]
    assert any("RuntimeError" in message for message in debug_messages)
    assert sentinel not in "".join(debug_messages)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_wrap_sync_stream_admission_failure_closes_concurrently_and_recovers(
    monkeypatch,
):
    from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
    from tldw_Server_API.app.core.Chat.bounded_daemon import (
        BoundedDaemonPool,
        DaemonCapacityError,
    )
    from tldw_Server_API.app.core.LLM_Calls import streaming

    both_admissions_rejected = threading.Event()
    admission_lock = threading.Lock()
    admission_rejections = 0
    holder_started = threading.Event()
    release_holder = threading.Event()
    close_barrier = threading.Barrier(2)
    raw_next_called = [threading.Event(), threading.Event()]
    raw_closed = [threading.Event(), threading.Event()]

    class ObservingPool(BoundedDaemonPool):
        def start(self, target, *, name, released_event=None):
            nonlocal admission_rejections
            try:
                return super().start(
                    target,
                    name=name,
                    released_event=released_event,
                )
            except DaemonCapacityError:
                with admission_lock:
                    admission_rejections += 1
                    if admission_rejections >= 2:
                        both_admissions_rejected.set()
                raise

    class RawSyncIterator:
        def __init__(self, index: int) -> None:
            self.index = index

        def __iter__(self):
            return self

        def __next__(self) -> str:
            raw_next_called[self.index].set()
            raise StopIteration

        def close(self) -> None:
            close_barrier.wait(timeout=1.0)
            raw_closed[self.index].set()

    class HealthySyncIterator:
        def __init__(self) -> None:
            self.sent = False
            self.closed = threading.Event()

        def __iter__(self):
            return self

        def __next__(self) -> str:
            if self.sent:
                raise StopIteration
            self.sent = True
            return "healthy"

        def close(self) -> None:
            self.closed.set()

    async def wait_for_thread_event(
        event: threading.Event,
        timeout: float = 1.0,
    ) -> bool:
        """Poll a thread event without blocking the asyncio loop."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not event.is_set():
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0)
        return True

    pool = ObservingPool(capacity=1)
    cleanup_pool = BoundedDaemonPool(capacity=2)
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
    monkeypatch.setattr(
        bounded_daemon_module,
        "STREAM_CLEANUP_DAEMON_POOL",
        cleanup_pool,
        raising=False,
    )
    holder = pool.start(
        lambda: (holder_started.set(), release_holder.wait()),
        name="llm-bridge-admission-holder",
    )

    async def collect(index: int) -> list[str]:
        return [
            item
            async for item in streaming.wrap_sync_stream(RawSyncIterator(index))
        ]

    collect_tasks = [asyncio.create_task(collect(index)) for index in range(2)]
    try:
        assert await wait_for_thread_event(holder_started)
        assert await wait_for_thread_event(both_admissions_rejected)
        assert await wait_for_thread_event(raw_closed[0])
        assert await wait_for_thread_event(raw_closed[1])
        assert not release_holder.is_set()
        assert not any(event.is_set() for event in raw_next_called)

        results = await asyncio.wait_for(
            asyncio.gather(*collect_tasks, return_exceptions=True),
            timeout=1.0,
        )
        assert all(isinstance(result, DaemonCapacityError) for result in results)
    finally:
        release_holder.set()
        holder.join(timeout=1.0)
        for task in collect_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*collect_tasks, return_exceptions=True)

    healthy = HealthySyncIterator()
    chunks = [item async for item in streaming.wrap_sync_stream(healthy)]

    assert chunks == ["healthy"]
    assert await wait_for_thread_event(healthy.closed)
    assert pool.active_count == 0
    assert cleanup_pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_registered_openai_stream_disconnect_drains_blocked_read_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated disconnect cancellation cannot outlive a sync provider read."""
    from starlette.responses import StreamingResponse

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    provider_release = threading.Event()
    second_next_started = threading.Event()
    first_output = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []
    runtime_close_pool_counts: list[int] = []

    class RecordingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            super()._release_capacity()
            lifecycle.append("capacity_release")

    class BlockingIterator:
        def __init__(self) -> None:
            self.next_count = 0

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self.next_count += 1
            if self.next_count == 1:
                return 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            second_next_started.set()
            provider_release.wait(timeout=2.0)
            lifecycle.append("provider_exit")
            raise StopIteration

        def close(self) -> None:
            lifecycle.append("source_close")

    class Runtime:
        async def close(self) -> None:
            runtime_close_pool_counts.append(pool.active_count)
            lifecycle.append("runtime_close")
            runtime_closed.set()

    source = BlockingIterator()
    pool = RecordingPool(1)
    registry = ChatProviderRegistry(include_defaults=True)
    adapter = registry.get_adapter("openai")
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.async_chat_is_native is False
    monkeypatch.setattr(
        adapter,
        "stream",
        lambda _request, *, timeout=None: source,
    )
    monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)

    response = StreamingResponse(
        adapter.astream(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4o-mini",
                "api_key": "stream-secret",
            }
        )
    )
    chat_endpoint._attach_credential_runtime_cleanup(
        response,
        Runtime(),  # type: ignore[arg-type]
        sanitize_provider_errors=False,
    )
    chunks: list[str] = []

    async def consume() -> None:
        async for chunk in response.body_iterator:
            chunks.append(str(chunk))
            first_output.set()

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_output.wait(), timeout=1.0)
        deadline = asyncio.get_running_loop().time() + 1.0
        while not second_next_started.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)
        assert second_next_started.is_set()
        assert pool.active_count == 1

        consumer.cancel()
        await asyncio.sleep(0)
        consumer.cancel()
        await asyncio.sleep(0.08)

        assert consumer.done() is False
        assert runtime_closed.is_set() is False
        assert pool.active_count == 1
    finally:
        provider_release.set()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert chunks == ['data: {"choices":[{"delta":{"content":"hello"}}]}\n\n']
    assert runtime_closed.is_set()
    assert pool.active_count == 0
    assert runtime_close_pool_counts == [0]
    assert lifecycle == [
        "provider_exit",
        "source_close",
        "capacity_release",
        "runtime_close",
    ]
