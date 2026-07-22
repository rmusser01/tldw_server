import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint


class _DummyHTTPStatusError(Exception):
    def __init__(self, response):
        super().__init__("upstream status error")
        self.response = response


class _DummyRequestError(Exception):
    pass


class _FailingResponse:
    def __init__(self, *, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        raise _DummyHTTPStatusError(self)

    def json(self):
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class _FailingPostClient:
    def __init__(self, response: _FailingResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *_args, **_kwargs):
        return self._response


class _RaisingPostClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *_args, **_kwargs):
        raise _DummyRequestError("network down")


class _FailingStreamContext:
    def __init__(self, response: _FailingResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FailingStreamClient:
    def __init__(self, response: _FailingResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *_args, **_kwargs):
        return _FailingStreamContext(self._response)


class _TrackingStreamResponse:
    def __init__(self, items, *, terminal_error: Exception | None = None):
        self._items = list(items)
        self._terminal_error = terminal_error
        self.next_calls = 0

    def raise_for_status(self):
        return None

    def aiter_raw(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_calls += 1
        if self._items:
            return self._items.pop(0)
        if self._terminal_error is not None:
            raise self._terminal_error
        raise StopAsyncIteration


class _TrackingStreamContext:
    def __init__(self, response, *, enter_error: BaseException | None = None):
        self._response = response
        self._enter_error = enter_error
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self):
        self.enter_calls += 1
        if self._enter_error is not None:
            raise self._enter_error
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        return False


class _TrackingStreamClient:
    def __init__(self, stream_context, *, enter_error: BaseException | None = None):
        self._stream_context = stream_context
        self._enter_error = enter_error
        self.enter_calls = 0
        self.exit_calls = 0

    async def __aenter__(self):
        self.enter_calls += 1
        if self._enter_error is not None:
            raise self._enter_error
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        return False

    def stream(self, *_args, **_kwargs):
        return self._stream_context


async def _open_native_stream() -> object:
    return await messages_endpoint._prepare_native_stream_iterator(
        "https://example.invalid/v1/messages",
        {"x-api-key": "test"},
        {"model": "x"},
        timeout=30.0,
        provider="anthropic",
        operation="messages.stream",
    )


def _assert_bounded_detached_error(exc: HTTPException, sentinel: str) -> None:
    assert exc.status_code == 502
    assert exc.detail == {
        "error_code": "upstream_provider_error",
        "provider": "anthropic",
        "operation": "messages.stream",
        "message": "Upstream provider 'anthropic' request failed.",
    }
    assert sentinel not in str(exc)
    assert exc.__cause__ is None
    assert exc.__context__ is None


@pytest.mark.asyncio
async def test_native_post_json_maps_http_status_error(monkeypatch):
    sentinel = "sk-native-message-error-/private/provider-response.json"
    response = _FailingResponse(
        status_code=401,
        payload={"error": {"message": sentinel, "authorization": sentinel}},
        text=sentinel,
    )
    monkeypatch.setattr(messages_endpoint, "async_http_client_factory", lambda timeout=None: _FailingPostClient(response))

    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._native_post_json(
            "https://example.invalid/v1/messages",
            {"x-api-key": "test"},
            {"model": "x"},
            timeout=30.0,
            provider="anthropic",
            operation="messages",
        )

    assert exc_info.value.status_code == 502
    detail = exc_info.value.detail
    assert detail["provider"] == "anthropic"
    assert detail["operation"] == "messages"
    assert detail["message"] == "Upstream provider 'anthropic' request failed."
    assert "upstream_error" not in detail
    assert sentinel not in str(detail)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_native_post_json_maps_request_error_to_502(monkeypatch):
    monkeypatch.setattr(messages_endpoint, "async_http_client_factory", lambda timeout=None: _RaisingPostClient())

    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._native_post_json(
            "https://example.invalid/v1/messages",
            {"x-api-key": "test"},
            {"model": "x"},
            timeout=30.0,
            provider="anthropic",
            operation="messages",
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["provider"] == "anthropic"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "expected_status"),
    [(401, 502), (403, 502), (429, 429)],
)
async def test_prepare_native_stream_iterator_maps_preflight_status_error(
    monkeypatch,
    upstream_status,
    expected_status,
):
    sentinel = "sk-native-stream-error-/private/provider-stream.json"
    response = _FailingResponse(
        status_code=upstream_status,
        payload={"error": {"message": sentinel}},
        text=sentinel,
    )
    monkeypatch.setattr(messages_endpoint, "async_http_client_factory", lambda timeout=None: _FailingStreamClient(response))

    with pytest.raises(HTTPException) as exc_info:
        await messages_endpoint._prepare_native_stream_iterator(
            "https://example.invalid/v1/messages",
            {"x-api-key": "test"},
            {"model": "x"},
            timeout=30.0,
            provider="anthropic",
            operation="messages.stream",
        )

    assert exc_info.value.status_code == expected_status
    detail = exc_info.value.detail
    assert detail["provider"] == "anthropic"
    assert detail["operation"] == "messages.stream"
    assert "upstream_error" not in detail
    assert sentinel not in str(detail)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_native_post_auth_failures_are_bounded_across_concurrent_requests(
    monkeypatch,
):
    sentinels = {
        1.0: "sk-native-auth-one-/private/provider-one.json",
        2.0: "sk-native-auth-two-/private/provider-two.json",
    }
    responses = {
        timeout: _FailingResponse(
            status_code=status_code,
            payload={"error": {"message": sentinels[timeout]}},
            text=sentinels[timeout],
        )
        for timeout, status_code in ((1.0, 401), (2.0, 403))
    }
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: _FailingPostClient(responses[timeout]),
    )

    async def _call(timeout: float) -> HTTPException:
        with pytest.raises(HTTPException) as exc_info:
            await messages_endpoint._native_post_json(
                "https://example.invalid/v1/messages",
                {"x-api-key": "test"},
                {"model": "x"},
                timeout=timeout,
                provider="anthropic",
                operation="messages",
            )
        return exc_info.value

    errors = await asyncio.gather(_call(1.0), _call(2.0))

    assert [error.status_code for error in errors] == [502, 502]
    rendered = " ".join(str(error.detail) for error in errors)
    assert all(sentinel not in rendered for sentinel in sentinels.values())
    assert all(error.__cause__ is None for error in errors)
    assert all(error.__context__ is None for error in errors)


@pytest.mark.asyncio
async def test_prepare_native_stream_iterator_maps_factory_error_detached(monkeypatch):
    sentinel = "sk-native-factory-error-/private/client.json"

    def _factory(*, timeout=None):
        del timeout
        raise RuntimeError(sentinel)

    monkeypatch.setattr(messages_endpoint, "async_http_client_factory", _factory)

    with pytest.raises(HTTPException) as exc_info:
        await _open_native_stream()

    _assert_bounded_detached_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_prepare_native_stream_iterator_maps_client_enter_error_detached(
    monkeypatch,
):
    sentinel = "sk-native-client-enter-/private/client.json"
    stream_cm = _TrackingStreamContext(_TrackingStreamResponse([]))
    client_cm = _TrackingStreamClient(
        stream_cm,
        enter_error=RuntimeError(sentinel),
    )
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: client_cm,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _open_native_stream()

    _assert_bounded_detached_error(exc_info.value, sentinel)
    assert client_cm.enter_calls == 1
    assert client_cm.exit_calls == 0
    assert stream_cm.enter_calls == 0
    assert stream_cm.exit_calls == 0


@pytest.mark.asyncio
async def test_prepare_native_stream_iterator_cleans_entered_client_on_stream_enter_error(
    monkeypatch,
):
    sentinel = "sk-native-stream-enter-/private/stream.json"
    stream_cm = _TrackingStreamContext(
        _TrackingStreamResponse([]),
        enter_error=RuntimeError(sentinel),
    )
    client_cm = _TrackingStreamClient(stream_cm)
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: client_cm,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _open_native_stream()

    _assert_bounded_detached_error(exc_info.value, sentinel)
    assert client_cm.exit_calls == 1
    assert stream_cm.enter_calls == 1
    assert stream_cm.exit_calls == 0


@pytest.mark.asyncio
async def test_prepare_native_stream_iterator_preserves_cancel_and_cleans_entered_client(
    monkeypatch,
):
    enter_started = asyncio.Event()

    class _BlockingStreamContext(_TrackingStreamContext):
        async def __aenter__(self):
            self.enter_calls += 1
            enter_started.set()
            await asyncio.Event().wait()

    stream_cm = _BlockingStreamContext(_TrackingStreamResponse([]))
    client_cm = _TrackingStreamClient(stream_cm)
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: client_cm,
    )

    task = asyncio.create_task(_open_native_stream())
    await asyncio.wait_for(enter_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client_cm.exit_calls == 1
    assert stream_cm.exit_calls == 0


@pytest.mark.asyncio
async def test_native_mid_read_failure_emits_one_bounded_error_and_closes_once(
    monkeypatch,
):
    sentinel = "sk-native-mid-read-/private/transport.json"
    start_frame = _native_message_start_frame().encode()
    response = _TrackingStreamResponse(
        [start_frame],
        terminal_error=RuntimeError(sentinel),
    )
    stream_cm = _TrackingStreamContext(response)
    client_cm = _TrackingStreamClient(stream_cm)
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: client_cm,
    )

    stream = await _open_native_stream()
    body = "".join(
        [
            item
            async for item in messages_endpoint._sanitize_native_messages_stream(
                stream
            )
        ]
    )

    assert start_frame.decode() in body
    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert sentinel not in body
    assert stream_cm.exit_calls == 1
    assert client_cm.exit_calls == 1


@pytest.mark.asyncio
async def test_native_message_stop_ends_stream_before_later_transport_failure(
    monkeypatch,
):
    sentinel = "sk-native-after-stop-/private/transport.json"
    terminal_chunk = "".join(
        _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {"type": "text_delta", "text": "complete"},
                ),
                _native_block_stop_frame(0),
            ]
        )
    ).encode()
    response = _TrackingStreamResponse(
        [terminal_chunk],
        terminal_error=RuntimeError(sentinel),
    )
    stream_cm = _TrackingStreamContext(response)
    client_cm = _TrackingStreamClient(stream_cm)
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: client_cm,
    )

    stream = await _open_native_stream()
    body = "".join(
        [
            item
            async for item in messages_endpoint._sanitize_native_messages_stream(
                stream
            )
        ]
    )

    assert body == terminal_chunk.decode()
    assert body.count("event: message_stop") == 1
    assert "event: error" not in body
    assert sentinel not in body
    assert response.next_calls == 1
    assert stream_cm.exit_calls == 1
    assert client_cm.exit_calls == 1


@pytest.mark.asyncio
async def test_native_mid_read_cancellation_is_preserved_and_closes_once(monkeypatch):
    read_started = asyncio.Event()

    class _BlockingResponse(_TrackingStreamResponse):
        async def __anext__(self):
            self.next_calls += 1
            read_started.set()
            await asyncio.Event().wait()

    response = _BlockingResponse([])
    stream_cm = _TrackingStreamContext(response)
    client_cm = _TrackingStreamClient(stream_cm)
    monkeypatch.setattr(
        messages_endpoint,
        "async_http_client_factory",
        lambda timeout=None: client_cm,
    )
    stream = await _open_native_stream()
    sanitized = messages_endpoint._sanitize_native_messages_stream(stream)

    task = asyncio.create_task(sanitized.__anext__())
    await asyncio.wait_for(read_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream_cm.exit_calls == 1
    assert client_cm.exit_calls == 1


@pytest.mark.asyncio
async def test_concurrent_native_terminal_and_failure_streams_are_isolated():
    sentinels = {
        "terminal": "sk-native-terminal-late-/private/transport.json",
        "failure": "sk-native-concurrent-failure-/private/transport.json",
    }
    entered = {name: asyncio.Event() for name in sentinels}
    release = asyncio.Event()
    close_calls = dict.fromkeys(sentinels, 0)
    start_frame = _native_message_start_frame()
    terminal_chunk = "".join(
        _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {"type": "text_delta", "text": "complete"},
                ),
                _native_block_stop_frame(0),
            ]
        )
    )

    async def _source(name: str):
        try:
            entered[name].set()
            await release.wait()
            if name == "terminal":
                yield terminal_chunk.encode()
            else:
                yield start_frame.encode()
            raise RuntimeError(sentinels[name])
        finally:
            close_calls[name] += 1

    async def _consume(name: str) -> str:
        return "".join(
            [
                item
                async for item in messages_endpoint._sanitize_native_messages_stream(
                    _source(name)
                )
            ]
        )

    terminal_task = asyncio.create_task(_consume("terminal"))
    failure_task = asyncio.create_task(_consume("failure"))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        terminal_body, failure_body = await asyncio.wait_for(
            asyncio.gather(terminal_task, failure_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(
            terminal_task,
            failure_task,
            return_exceptions=True,
        )

    assert terminal_body == terminal_chunk
    assert "event: error" not in terminal_body
    assert failure_body.startswith(start_frame)
    assert failure_body.count("event: error") == 1
    assert all(
        sentinel not in terminal_body + failure_body
        for sentinel in sentinels.values()
    )
    assert close_calls == {"terminal": 1, "failure": 1}


class _NativeSSEBoundaryRuntime:
    """Request-local usage/close recorder for native SSE boundary tests."""

    def __init__(self, key: str, marks: list[str], lifecycle: list[str]) -> None:
        self.key = key
        self.marks = marks
        self.lifecycle = lifecycle

    async def mark_used(self, _credentials: Any) -> None:
        self.marks.append(self.key)

    async def close(self) -> None:
        self.lifecycle.append(f"runtime-close:{self.key}")


def _native_sse_frame(event: str, payload: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )


def _native_message_start_frame(*, diagnostic: str | None = None) -> str:
    payload: dict[str, Any] = {
        "type": "message_start",
        "message": {
            "id": "msg-native-boundary",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 0},
        },
    }
    if diagnostic is not None:
        payload["diagnostic"] = diagnostic
    return _native_sse_frame("message_start", payload)


def _native_message_delta_frame(
    *,
    stop_reason: str = "end_turn",
    output_tokens: int = 3,
) -> str:
    return _native_sse_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


def _native_message_stop_frame(*, diagnostic: str | None = None) -> str:
    payload: dict[str, Any] = {"type": "message_stop"}
    if diagnostic is not None:
        payload["diagnostic"] = diagnostic
    return _native_sse_frame("message_stop", payload)


def _native_block_start_frame(
    index: int,
    content_block: dict[str, Any],
) -> str:
    return _native_sse_frame(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        },
    )


def _native_block_delta_frame(index: int, delta: dict[str, Any]) -> str:
    return _native_sse_frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        },
    )


def _native_block_stop_frame(index: int) -> str:
    return _native_sse_frame(
        "content_block_stop",
        {"type": "content_block_stop", "index": index},
    )


def _native_complete_flow(
    content_frames: list[str],
    *,
    stop_reason: str = "end_turn",
) -> list[str]:
    return [
        _native_message_start_frame(),
        *content_frames,
        _native_message_delta_frame(stop_reason=stop_reason),
        _native_message_stop_frame(),
    ]


async def _consume_native_sse_boundary(
    *,
    key: str,
    frames: list[str],
    marks: list[str],
    lifecycle: list[str],
    entered: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> str:
    async def _source():
        try:
            if entered is not None:
                entered.set()
            if release is not None:
                await release.wait()
            for frame in frames:
                yield frame
        finally:
            lifecycle.append(f"source-close:{key}")

    runtime = _NativeSSEBoundaryRuntime(key, marks, lifecycle)
    credentials = SimpleNamespace(provider="anthropic", api_key=key)
    body = "".join(
        [
            item
            async for item in messages_endpoint._touch_on_first_stream_output(
                messages_endpoint._sanitize_native_messages_stream(_source()),
                runtime,
                credentials,
            )
        ]
    )
    source_close = f"source-close:{key}"
    runtime_close = f"runtime-close:{key}"
    assert lifecycle.count(source_close) == 1
    assert lifecycle.count(runtime_close) == 1
    assert lifecycle.index(source_close) < lifecycle.index(runtime_close)
    return body


def _native_structural_error_frame(case: str, sentinel: str) -> str:
    if case == "message-start-error":
        return _native_sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg-error",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                    "error": {"message": sentinel},
                },
            },
        )
    if case == "mixed-text-delta-error":
        return _native_sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "apparently valid",
                    "error": {"message": sentinel},
                },
            },
        )
    if case == "content-block-error":
        return _native_sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "error",
                    "error": {"message": sentinel},
                },
            },
        )
    raise AssertionError(f"Unknown native structural error case: {case}")


def _native_structural_error_frames(case: str, sentinel: str) -> list[str]:
    frame = _native_structural_error_frame(case, sentinel)
    if case == "mixed-text-delta-error":
        return [
            _native_message_start_frame(),
            _native_block_start_frame(0, {"type": "text", "text": ""}),
            frame,
        ]
    if case == "content-block-error":
        return [_native_message_start_frame(), frame]
    return [frame]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "message-start-error",
        "mixed-text-delta-error",
        "content-block-error",
    ],
)
async def test_native_sse_rejects_nested_and_mixed_provider_errors(
    case: str,
) -> None:
    """Nested provider diagnostics are bounded before output or usage marking."""
    sentinel = f"native-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_structural_error_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert marks == []


def _native_invalid_schema_frame(case: str, sentinel: str) -> str:
    if case in {"input-boolean", "input-negative"}:
        value: Any = True if case == "input-boolean" else -1
        return _native_sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg-invalid-usage",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "usage": {"input_tokens": value, "output_tokens": 0},
                },
            },
        )
    if case in {"output-boolean", "output-negative", "output-diagnostic"}:
        output_value: Any = {
            "output-boolean": False,
            "output-negative": -1,
            "output-diagnostic": {"error": {"message": sentinel}},
        }[case]
        return _native_sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_value},
            },
        )
    if case == "nonstring-text":
        return _native_sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": {"error": {"message": sentinel}},
                },
            },
        )
    if case == "nonobject-tool-input":
        return _native_sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-invalid",
                    "name": "lookup",
                    "input": [sentinel],
                },
            },
        )
    raise AssertionError(f"Unknown native invalid schema case: {case}")


def _native_invalid_schema_frames(case: str, sentinel: str) -> list[str]:
    frame = _native_invalid_schema_frame(case, sentinel)
    if case in {"input-boolean", "input-negative"}:
        return [frame]
    if case == "nonstring-text":
        return [
            _native_message_start_frame(),
            _native_block_start_frame(0, {"type": "text", "text": ""}),
            frame,
        ]
    return [_native_message_start_frame(), frame]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "input-boolean",
        "input-negative",
        "output-boolean",
        "output-negative",
        "output-diagnostic",
        "nonstring-text",
        "nonobject-tool-input",
    ],
)
async def test_native_sse_rejects_invalid_event_schema_before_mark(
    case: str,
) -> None:
    """Malformed usage, text, and tool events fail closed as bounded SSE."""
    sentinel = f"native-{case}-schema-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_invalid_schema_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "The upstream provider returned an error." in body
    assert sentinel not in body
    assert marks == []


def _native_domain_tool_frames() -> list[str]:
    tool_input = {
        "error": {
            "code": "city_not_found",
            "message": "No matching city",
        }
    }
    return _native_complete_flow(
        [
            _native_block_start_frame(
                0,
                {
                    "type": "tool_use",
                    "id": "tool-domain-error",
                    "name": "lookup_weather",
                    "input": {},
                },
            ),
            _native_block_delta_frame(
                0,
                {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tool_input, separators=(",", ":")),
                },
            ),
            _native_block_stop_frame(0),
        ],
        stop_reason="tool_use",
    )


@pytest.mark.asyncio
async def test_native_sse_preserves_domain_error_tool_input_control() -> None:
    """Schema-valid domain errors remain tool data and certify provider use."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="native-domain-control",
        frames=_native_domain_tool_frames(),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert "city_not_found" in body
    assert marks == ["native-domain-control"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_sse_validation_is_request_local() -> None:
    """An invalid native frame cannot mark or contaminate a valid stream."""
    sentinel = "native-concurrent-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("native-valid", "native-invalid")}
    release = asyncio.Event()

    valid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="native-valid",
            frames=_native_domain_tool_frames(),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["native-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="native-invalid",
            frames=_native_mixed_diagnostic_frames(
                "unknown-sse-field",
                sentinel,
            ),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["native-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert "city_not_found" in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["native-valid"]


def _native_text_start_frame(
    *,
    index: Any,
    text: Any,
    include_index: bool = True,
    block_type: str = "text",
) -> str:
    payload: dict[str, Any] = {
        "type": "content_block_start",
        "content_block": {"type": block_type, "text": text},
    }
    if include_index:
        payload["index"] = index
    return _native_sse_frame("content_block_start", payload)


def _native_text_delta_frame(
    *,
    index: Any,
    text: Any,
    include_index: bool = True,
    delta_type: str = "text_delta",
) -> str:
    payload: dict[str, Any] = {
        "type": "content_block_delta",
        "delta": {"type": delta_type, "text": text},
    }
    if include_index:
        payload["index"] = index
    return _native_sse_frame("content_block_delta", payload)


def _native_stop_frame(
    *,
    index: Any,
    sentinel: str | None = None,
) -> str:
    payload: dict[str, Any] = {"type": "content_block_stop", "index": index}
    if sentinel is not None:
        payload["diagnostic"] = sentinel
    return _native_sse_frame("content_block_stop", payload)


def _native_state_validation_frames(case: str, sentinel: str) -> list[str]:
    if case == "unknown-block-type":
        return [
            _native_sse_frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "provider_debug",
                        "diagnostic": sentinel,
                    },
                },
            )
        ]
    if case == "unknown-delta-type":
        return [
            _native_sse_frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "provider_debug_delta",
                        "diagnostic": sentinel,
                    },
                },
            )
        ]
    if case == "text-start-missing-index":
        return [
            _native_text_start_frame(
                index=0,
                text=sentinel,
                include_index=False,
            )
        ]
    if case == "text-start-negative-index":
        return [_native_text_start_frame(index=-1, text=sentinel)]
    if case == "text-start-noninteger-index":
        return [_native_text_start_frame(index="0", text=sentinel)]
    if case == "text-delta-noninteger-index":
        return [_native_text_delta_frame(index="0", text=sentinel)]
    if case == "orphan-text-delta":
        return [_native_text_delta_frame(index=7, text=sentinel)]
    if case == "orphan-stop":
        return [_native_stop_frame(index=7, sentinel=sentinel)]
    if case == "duplicate-open-index":
        return [
            _native_text_start_frame(index=0, text=""),
            _native_text_start_frame(index=0, text=sentinel),
        ]
    if case == "double-stop":
        return [
            _native_text_start_frame(index=0, text=""),
            _native_stop_frame(index=0),
            _native_stop_frame(index=0, sentinel=sentinel),
        ]
    if case == "delta-after-stop":
        return [
            _native_text_start_frame(index=0, text=""),
            _native_stop_frame(index=0),
            _native_text_delta_frame(index=0, text=sentinel),
        ]
    if case == "malformed-text-block":
        return [
            _native_text_start_frame(
                index=0,
                text={"diagnostic": sentinel},
            )
        ]
    if case == "malformed-text-delta":
        return [
            _native_text_delta_frame(
                index=0,
                text={"diagnostic": sentinel},
            )
        ]
    raise AssertionError(f"Unknown native state validation case: {case}")


def _native_state_validation_flow(case: str, sentinel: str) -> list[str]:
    prefix = [_native_message_start_frame()]
    if case in {
        "unknown-delta-type",
        "text-delta-noninteger-index",
        "malformed-text-delta",
    }:
        prefix.append(_native_text_start_frame(index=0, text=""))
    return [*prefix, *_native_state_validation_frames(case, sentinel)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "unknown-block-type",
        "unknown-delta-type",
        "text-start-missing-index",
        "text-start-negative-index",
        "text-start-noninteger-index",
        "text-delta-noninteger-index",
        "orphan-text-delta",
        "orphan-stop",
        "duplicate-open-index",
        "double-stop",
        "delta-after-stop",
        "malformed-text-block",
        "malformed-text-delta",
    ],
)
async def test_native_sse_rejects_unknown_schema_and_invalid_block_state(
    case: str,
) -> None:
    """Only schema-valid, ordered native content events cross the boundary."""
    sentinel = f"native-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_state_validation_flow(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert sentinel not in body
    assert marks == []


def _native_valid_text_frames() -> list[str]:
    return _native_complete_flow(
        [
            _native_text_start_frame(index=0, text=""),
            _native_text_delta_frame(index=0, text="valid native text"),
            _native_stop_frame(index=0),
        ]
    )


@pytest.mark.asyncio
async def test_native_sse_accepts_ordered_text_state_control() -> None:
    """A complete ordered native text stream remains compatible."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="native-state-control",
        frames=_native_valid_text_frames(),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert "valid native text" in body
    assert body.count("event: message_stop") == 1
    assert marks == ["native-state-control"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_message_lifecycle_is_request_local() -> None:
    """An invalid lifecycle cannot contaminate a valid concurrent stream."""
    sentinel = "native-concurrent-state-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("state-valid", "state-invalid")}
    release = asyncio.Event()

    valid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="state-valid",
            frames=_native_valid_text_frames(),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["state-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="state-invalid",
            frames=_native_lifecycle_violation_frames(
                "simultaneous-open-blocks",
                sentinel,
            ),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["state-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert "valid native text" in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["state-valid"]


def _native_lifecycle_violation_frames(case: str, sentinel: str) -> list[str]:
    if case == "pre-start-output":
        return [_native_text_start_frame(index=0, text=sentinel)]
    if case == "duplicate-message-start":
        return [
            _native_message_start_frame(),
            _native_message_start_frame(diagnostic=sentinel),
        ]
    if case == "gapped-first-index":
        return [
            _native_message_start_frame(),
            _native_text_start_frame(index=1, text=sentinel),
        ]
    if case == "reused-closed-index":
        return [
            _native_message_start_frame(),
            _native_text_start_frame(index=0, text=""),
            _native_text_delta_frame(index=0, text="valid prefix"),
            _native_stop_frame(index=0),
            _native_text_start_frame(index=0, text=sentinel),
        ]
    if case == "simultaneous-open-blocks":
        return [
            _native_message_start_frame(),
            _native_text_start_frame(index=0, text=""),
            _native_text_start_frame(index=1, text=sentinel),
        ]
    if case == "content-after-message-delta":
        return [
            _native_message_start_frame(),
            _native_message_delta_frame(),
            _native_text_start_frame(index=0, text=sentinel),
        ]
    if case == "stop-without-start":
        return [_native_message_stop_frame(diagnostic=sentinel)]
    if case == "complete-frame-premature-eof":
        return [
            _native_message_start_frame(),
            _native_text_start_frame(index=0, text=""),
            _native_text_delta_frame(index=0, text=sentinel),
            _native_stop_frame(index=0),
            _native_message_delta_frame(),
        ]
    if case == "buffered-frame-after-terminal":
        return [
            "".join(
                [
                    *_native_valid_text_frames(),
                    _native_message_stop_frame(diagnostic=sentinel),
                ]
            )
        ]
    raise AssertionError(f"Unknown native lifecycle violation: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "pre-start-output",
        "duplicate-message-start",
        "gapped-first-index",
        "reused-closed-index",
        "simultaneous-open-blocks",
        "content-after-message-delta",
        "stop-without-start",
        "complete-frame-premature-eof",
        "buffered-frame-after-terminal",
    ],
)
async def test_native_sse_rejects_invalid_message_lifecycle_before_mark(
    case: str,
) -> None:
    """Only one complete, sequential Messages lifecycle can certify use."""
    sentinel = f"native-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_lifecycle_violation_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert "event: message_stop" not in body
    if case != "complete-frame-premature-eof":
        assert sentinel not in body
    else:
        # Complete SSE events are intentionally forwarded as they arrive. A
        # later EOF can invalidate usage accounting, but cannot retract text
        # already delivered without buffering the entire response.
        assert sentinel in body
    assert marks == []


def _native_mixed_diagnostic_frames(case: str, sentinel: str) -> list[str]:
    payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "apparently valid"},
    }
    if case == "unknown-payload-field":
        payload["diagnostic"] = sentinel
        mixed_frame = _native_sse_frame("content_block_delta", payload)
    else:
        valid_frame = _native_sse_frame("content_block_delta", payload)
        prefix = f"diagnostic: {sentinel}\n" if case == "unknown-sse-field" else f"{sentinel}\n"
        mixed_frame = prefix + valid_frame
    return [
        _native_message_start_frame(),
        _native_text_start_frame(index=0, text=""),
        mixed_frame,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["unknown-sse-field", "plain-diagnostic-line", "unknown-payload-field"],
)
async def test_native_sse_rejects_mixed_raw_diagnostic_before_leak_or_mark(
    case: str,
) -> None:
    """A valid data event cannot smuggle an unrecognized diagnostic field."""
    sentinel = f"native-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_mixed_diagnostic_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert body.count("The upstream provider returned an error.") == 1
    assert sentinel not in body
    assert marks == []


def _native_text_content_frames(
    index: int,
    text: str,
    *,
    block_type: str = "text",
) -> list[str]:
    return [
        _native_block_start_frame(index, {"type": block_type, "text": ""}),
        _native_block_delta_frame(index, {"type": "text_delta", "text": text}),
        _native_block_stop_frame(index),
    ]


def _native_documented_content_flow(case: str) -> list[str]:
    if case == "thinking-signature":
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {"type": "thinking", "thinking": "", "signature": ""},
                ),
                _native_block_delta_frame(
                    0,
                    {"type": "thinking_delta", "thinking": "considering"},
                ),
                _native_block_delta_frame(
                    0,
                    {"type": "signature_delta", "signature": "signed-thinking"},
                ),
                _native_block_stop_frame(0),
                *_native_text_content_frames(1, "thinking complete"),
            ]
        )
    if case == "fallback-connector":
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {
                        "type": "fallback",
                        "from": {"model": "claude-fable-5"},
                        "to": {"model": "claude-opus-4-8"},
                        "trigger": {"type": "refusal", "category": "cyber"},
                    },
                ),
                _native_block_stop_frame(0),
                *_native_text_content_frames(
                    1,
                    "fallback narration",
                    block_type="connector_text",
                ),
            ]
        )
    if case == "server-tool-result-citation":
        tool_id = "srvtoolu_weather"
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {
                        "type": "server_tool_use",
                        "id": tool_id,
                        "name": "web_search",
                        "input": {},
                    },
                ),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "input_json_delta",
                        "partial_json": '{"query":"weather in San Francisco"}',
                    },
                ),
                _native_block_stop_frame(0),
                _native_block_start_frame(
                    1,
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "San Francisco forecast",
                                "url": "https://weather.example/san-francisco",
                                "encrypted_content": "encrypted-result",
                                "page_age": None,
                            }
                        ],
                    },
                ),
                _native_block_stop_frame(1),
                _native_block_start_frame(2, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    2,
                    {"type": "text_delta", "text": "It is sunny."},
                ),
                _native_block_delta_frame(
                    2,
                    {
                        "type": "citations_delta",
                        "citation": {
                            "type": "web_search_result_location",
                            "cited_text": "Sunny",
                            "encrypted_index": "encrypted-index",
                            "title": "San Francisco forecast",
                            "url": "https://weather.example/san-francisco",
                        },
                    },
                ),
                _native_block_stop_frame(2),
            ]
        )
    if case == "container-upload":
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {"type": "container_upload", "file_id": "file_01ABC"},
                ),
                _native_block_stop_frame(0),
                *_native_text_content_frames(1, "file uploaded"),
            ]
        )
    if case == "mcp-tool-result":
        tool_id = "mcp_toolu_weather"
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {
                        "type": "mcp_tool_use",
                        "id": tool_id,
                        "name": "lookup_weather",
                        "server_name": "weather-server",
                        "input": {},
                    },
                ),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "input_json_delta",
                        "partial_json": '{"city":"San Francisco"}',
                    },
                ),
                _native_block_stop_frame(0),
                _native_block_start_frame(
                    1,
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": tool_id,
                        "content": "Sunny",
                        "is_error": False,
                    },
                ),
                _native_block_stop_frame(1),
                *_native_text_content_frames(2, "MCP complete"),
            ],
        )
    if case == "compaction":
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {
                        "type": "compaction",
                        "content": None,
                        "encrypted_content": None,
                    },
                ),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "compaction_delta",
                        "content": "Earlier conversation summarized.",
                        "encrypted_content": "opaque-compaction-payload",
                    },
                ),
                _native_block_stop_frame(0),
                *_native_text_content_frames(1, "Compaction complete"),
            ]
        )
    if case == "advisor-tool-result":
        tool_id = "srvtoolu_advisor"
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {
                        "type": "server_tool_use",
                        "id": tool_id,
                        "name": "advisor",
                        "input": {},
                    },
                ),
                _native_block_stop_frame(0),
                _native_block_start_frame(
                    1,
                    {
                        "type": "advisor_tool_result",
                        "tool_use_id": tool_id,
                        "content": {
                            "type": "advisor_result",
                            "text": "Use a channel-based coordination pattern.",
                        },
                    },
                ),
                _native_block_stop_frame(1),
                *_native_text_content_frames(2, "Advisor complete"),
            ]
        )
    raise AssertionError(f"Unknown documented native content flow: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "thinking-signature",
        "fallback-connector",
        "server-tool-result-citation",
        "container-upload",
        "mcp-tool-result",
        "compaction",
        "advisor-tool-result",
    ],
)
async def test_native_sse_accepts_documented_content_block_flows(case: str) -> None:
    """Current Anthropic response blocks remain compatible after hardening."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_documented_content_flow(case),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert body.count("event: message_start") == 1
    assert body.count("event: message_stop") == 1
    assert marks == [case]


def _native_pre_output_refusal_flow() -> list[str]:
    return [
        _native_message_start_frame(),
        _native_sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "refusal",
                    "stop_sequence": None,
                    "stop_details": {
                        "type": "refusal",
                        "category": "cyber",
                        "explanation": "This request was declined.",
                    },
                },
                "usage": {"output_tokens": 0},
            },
        ),
        _native_message_stop_frame(),
    ]


@pytest.mark.asyncio
async def test_native_sse_pre_output_refusal_is_valid_and_marks_use() -> None:
    """A documented pre-output refusal is a semantic provider response."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="pre-output-refusal",
        frames=_native_pre_output_refusal_flow(),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert '"stop_reason":"refusal"' in body.replace(" ", "")
    assert body.count("event: message_stop") == 1
    assert marks == ["pre-output-refusal"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_refusal_accounting_is_request_local() -> None:
    """A valid refusal cannot inherit a concurrent structural failure."""
    sentinel = "native-refusal-concurrent-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("refusal-valid", "refusal-invalid")}
    release = asyncio.Event()
    valid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="refusal-valid",
            frames=_native_pre_output_refusal_flow(),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["refusal-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="refusal-invalid",
            frames=_native_nested_unknown_frames("message-delta", sentinel),
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["refusal-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["refusal-valid"]


def _native_nested_unknown_frames(case: str, sentinel: str) -> list[str]:
    message = {
        "id": "msg-native-nested",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 0},
    }
    if case == "message-start-message":
        message["diagnostic"] = sentinel
        return [
            _native_sse_frame(
                "message_start",
                {"type": "message_start", "message": message},
            ),
            *_native_text_content_frames(0, "apparently valid"),
            _native_message_delta_frame(),
            _native_message_stop_frame(),
        ]
    if case == "message-start-missing-required":
        return [
            _native_sse_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"diagnostic": sentinel},
                },
            ),
            *_native_text_content_frames(0, "apparently valid"),
            _native_message_delta_frame(),
            _native_message_stop_frame(),
        ]
    if case == "text-delta":
        return _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "text_delta",
                        "text": "apparently valid",
                        "diagnostic": sentinel,
                    },
                ),
                _native_block_stop_frame(0),
            ]
        )
    if case == "citation":
        return _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "citations_delta",
                        "citation": {
                            "type": "web_search_result_location",
                            "cited_text": "apparently valid",
                            "encrypted_index": "opaque-index",
                            "title": "Example",
                            "url": "https://example.invalid",
                            "diagnostic": sentinel,
                        },
                    },
                ),
                _native_block_stop_frame(0),
            ]
        )
    if case == "citation-unknown-type":
        return _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "citations_delta",
                        "citation": {
                            "type": "provider_debug",
                            "diagnostic": sentinel,
                        },
                    },
                ),
                _native_block_stop_frame(0),
            ]
        )
    if case == "message-delta":
        return [
            _native_message_start_frame(),
            *_native_text_content_frames(0, "apparently valid"),
            _native_sse_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "diagnostic": sentinel,
                    },
                    "usage": {"output_tokens": 3},
                },
            ),
            _native_message_stop_frame(),
        ]
    raise AssertionError(f"Unknown nested native case: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "message-start-message",
        "message-start-missing-required",
        "text-delta",
        "citation",
        "citation-unknown-type",
        "message-delta",
    ],
)
async def test_native_sse_rejects_nested_unknown_structural_fields_before_leak_or_mark(
    case: str,
) -> None:
    """Unknown structural siblings cannot ride a schema-valid native event."""
    sentinel = f"native-{case}-nested-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_nested_unknown_frames(case, sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body
    assert marks == []


def _native_documented_result_error_flow(case: str) -> list[str]:
    tool_id = f"srvtoolu_{case}"
    if case == "web-search":
        tool_type = "server_tool_use"
        tool_name = "web_search"
        result = {
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "too_many_requests",
            },
        }
    elif case == "advisor":
        tool_type = "server_tool_use"
        tool_name = "advisor"
        result = {
            "type": "advisor_tool_result",
            "tool_use_id": tool_id,
            "content": {
                "type": "advisor_tool_result_error",
                "error_code": "overloaded",
            },
        }
    elif case == "mcp":
        tool_type = "mcp_tool_use"
        tool_name = "lookup_weather"
        result = {
            "type": "mcp_tool_result",
            "tool_use_id": tool_id,
            "content": "Error: city not found",
            "is_error": True,
        }
    else:
        raise AssertionError(f"Unknown result-error case: {case}")

    tool_block: dict[str, Any] = {
        "type": tool_type,
        "id": tool_id,
        "name": tool_name,
        "input": {},
    }
    if case == "mcp":
        tool_block["server_name"] = "weather-server"
    return _native_complete_flow(
        [
            _native_block_start_frame(0, tool_block),
            _native_block_stop_frame(0),
            _native_block_start_frame(1, result),
            _native_block_stop_frame(1),
            *_native_text_content_frames(2, f"Handled {case} result"),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["web-search", "advisor", "mcp"])
async def test_native_sse_preserves_documented_domain_result_errors(
    case: str,
) -> None:
    """Typed tool failures are assistant content, not transport failures."""
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_documented_result_error_flow(case),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert f"Handled {case} result" in body
    assert body.count("event: message_stop") == 1
    assert marks == [case]


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("family", ["nested-structural", "domain-result"])
async def test_concurrent_native_new_boundary_families_are_request_local(
    family: str,
) -> None:
    """One native boundary decision cannot certify or poison another stream."""
    sentinel = f"native-{family}-concurrent-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("family-valid", "family-invalid")}
    release = asyncio.Event()
    if family == "nested-structural":
        valid_frames = _native_complete_flow(_native_text_content_frames(0, "valid"))
        invalid_frames = _native_nested_unknown_frames("text-delta", sentinel)
    else:
        valid_frames = _native_documented_result_error_flow("web-search")
        invalid_frames = [
            _native_message_start_frame(),
            _native_sse_frame(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": sentinel},
                },
            ),
        ]

    valid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="family-valid",
            frames=valid_frames,
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["family-valid"],
            release=release,
        )
    )
    invalid_task = asyncio.create_task(
        _consume_native_sse_boundary(
            key="family-invalid",
            frames=invalid_frames,
            marks=marks,
            lifecycle=lifecycle,
            entered=entered["family-invalid"],
            release=release,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, invalid_body = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert "event: error" not in valid_body
    assert invalid_body.count("event: error") == 1
    assert sentinel not in valid_body + invalid_body
    assert marks == ["family-valid"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_compaction_and_text_controls_are_request_local() -> None:
    """A current beta block and a stable text stream can complete concurrently."""
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("compat-compaction", "compat-text")}
    release = asyncio.Event()
    tasks = [
        asyncio.create_task(
            _consume_native_sse_boundary(
                key=key,
                frames=(
                    _native_documented_content_flow("compaction")
                    if key == "compat-compaction"
                    else _native_complete_flow(_native_text_content_frames(0, "valid"))
                ),
                marks=marks,
                lifecycle=lifecycle,
                entered=entered[key],
                release=release,
            )
        )
        for key in entered
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        bodies = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert all("event: error" not in body for body in bodies)
    assert sorted(marks) == ["compat-compaction", "compat-text"]


def _current_cache_creation() -> dict[str, int]:
    return {
        "ephemeral_1h_input_tokens": 1,
        "ephemeral_5m_input_tokens": 2,
    }


def _current_usage_iterations() -> list[dict[str, Any]]:
    common = {
        "input_tokens": 7,
        "output_tokens": 2,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 1,
        "cache_creation": _current_cache_creation(),
    }
    return [
        {"type": "message", "model": "claude-opus-4-8", **common},
        {"type": "compaction", **common},
        {
            "type": "advisor_message",
            "model": "claude-advisor-current",
            **common,
        },
        {
            "type": "fallback_message",
            "model": "claude-sonnet-current",
            **common,
        },
    ]


def _native_message_start_with_usage(usage: dict[str, Any]) -> str:
    return _native_sse_frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg-native-current-usage",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )


def _native_current_usage_flow(case: str) -> list[str]:
    start_usage: dict[str, Any] = {"input_tokens": 11, "output_tokens": 0}
    delta_usage: dict[str, Any] = {"output_tokens": 4}
    if case == "output-tokens-details":
        start_usage["output_tokens_details"] = {"thinking_tokens": 0}
        delta_usage["output_tokens_details"] = {"thinking_tokens": 2}
    elif case == "speed":
        start_usage["speed"] = "fast"
    elif case == "all-iteration-variants":
        iterations = _current_usage_iterations()
        start_usage["iterations"] = iterations
        delta_usage["iterations"] = iterations
    else:
        raise AssertionError(f"Unknown native usage compatibility case: {case}")
    return [
        _native_message_start_with_usage(start_usage),
        *_native_text_content_frames(0, "current usage response"),
        _native_sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": delta_usage,
            },
        ),
        _native_message_stop_frame(),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["output-tokens-details", "speed", "all-iteration-variants"],
)
async def test_native_sse_accepts_current_stable_and_beta_usage_shapes(
    case: str,
) -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_current_usage_flow(case),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert body.count("event: message_stop") == 1
    assert marks == [case]


def _native_current_refusal_flow(case: str) -> list[str]:
    if case == "optional-fields-omitted":
        stop_details: dict[str, Any] = {"type": "refusal"}
    elif case == "extended-beta-fields":
        stop_details = {
            "type": "refusal",
            "category": None,
            "explanation": None,
            "fallback_credit_token": "opaque-credit-token",
            "fallback_has_prefill_claim": False,
            "recommended_model": "claude-sonnet-current",
        }
    else:
        raise AssertionError(f"Unknown refusal stop-details case: {case}")
    return [
        _native_message_start_frame(),
        _native_sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "refusal",
                    "stop_sequence": None,
                    "stop_details": stop_details,
                },
                "usage": {"output_tokens": 0},
            },
        ),
        _native_message_stop_frame(),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["optional-fields-omitted", "extended-beta-fields"],
)
async def test_native_sse_accepts_current_refusal_stop_details(case: str) -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_current_refusal_flow(case),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert body.count("event: message_stop") == 1
    assert marks == [case]


def _native_context_management_flow() -> list[str]:
    return [
        _native_message_start_frame(),
        *_native_text_content_frames(0, "context managed"),
        _native_sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "context_management": {"applied_edits": []},
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        _native_message_stop_frame(),
    ]


@pytest.mark.asyncio
async def test_native_sse_accepts_context_management_on_message_delta_envelope() -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="context-management-envelope",
        frames=_native_context_management_flow(),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert body.count("event: message_stop") == 1
    assert marks == ["context-management-envelope"]


def _native_message_start_with_nested_metadata(case: str, sentinel: str) -> str:
    message: dict[str, Any] = {
        "id": "msg-native-nested-metadata",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 2, "output_tokens": 0},
    }
    if case == "diagnostics":
        message["diagnostics"] = {
            "cache_miss_reason": None,
            "provider_debug": sentinel,
        }
    elif case == "context-management":
        message["context_management"] = {
            "applied_edits": [],
            "provider_debug": sentinel,
        }
    else:
        raise AssertionError(f"Unknown nested metadata case: {case}")
    return _native_sse_frame(
        "message_start",
        {"type": "message_start", "message": message},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["diagnostics", "context-management"])
async def test_native_sse_rejects_unknown_nested_metadata_before_leak_or_mark(
    case: str,
) -> None:
    sentinel = f"native-{case}-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=[_native_message_start_with_nested_metadata(case, sentinel)],
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert sentinel not in body
    assert marks == []


def _native_optional_schema_flow(case: str) -> list[str]:
    if case.startswith("document-citation"):
        citation: dict[str, Any] = {
            "type": "char_location",
            "cited_text": "cited",
            "document_index": 0,
            "start_char_index": 0,
            "end_char_index": 5,
        }
        if case == "document-citation-null-optionals":
            citation["document_title"] = None
            citation["file_id"] = None
        return _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {"type": "text_delta", "text": "cited"},
                ),
                _native_block_delta_frame(
                    0,
                    {"type": "citations_delta", "citation": citation},
                ),
                _native_block_stop_frame(0),
            ]
        )
    if case == "web-citation-title-omitted":
        return _native_complete_flow(
            [
                _native_block_start_frame(0, {"type": "text", "text": ""}),
                _native_block_delta_frame(
                    0,
                    {"type": "text_delta", "text": "web cited"},
                ),
                _native_block_delta_frame(
                    0,
                    {
                        "type": "citations_delta",
                        "citation": {
                            "type": "web_search_result_location",
                            "cited_text": "web cited",
                            "encrypted_index": "opaque-index",
                            "url": "https://example.invalid/result",
                        },
                    },
                ),
                _native_block_stop_frame(0),
            ]
        )
    if case == "fallback-category-omitted":
        return _native_complete_flow(
            [
                _native_block_start_frame(
                    0,
                    {
                        "type": "fallback",
                        "from": {"model": "claude-fable-5"},
                        "to": {"model": "claude-opus-4-8"},
                        "trigger": {"type": "refusal"},
                    },
                ),
                _native_block_stop_frame(0),
                *_native_text_content_frames(1, "fallback completed"),
            ]
        )
    raise AssertionError(f"Unknown optional native schema case: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "document-citation-optionals-omitted",
        "document-citation-null-optionals",
        "web-citation-title-omitted",
        "fallback-category-omitted",
    ],
)
async def test_native_sse_accepts_documented_optional_citation_and_fallback_fields(
    case: str,
) -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=_native_optional_schema_flow(case),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert body.count("event: message_stop") == 1
    assert marks == [case]


def _native_client_tool_frames(*, stop_reason: str, partial_json: str) -> list[str]:
    return _native_complete_flow(
        [
            _native_block_start_frame(
                0,
                {
                    "type": "tool_use",
                    "id": "toolu-native-terminal",
                    "name": "lookup",
                    "input": {},
                },
            ),
            _native_block_delta_frame(
                0,
                {"type": "input_json_delta", "partial_json": partial_json},
            ),
            _native_block_stop_frame(0),
        ],
        stop_reason=stop_reason,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "frames"),
    [
        (
            "tool-finish-without-tool",
            _native_complete_flow(
                _native_text_content_frames(0, "not a tool"),
                stop_reason="tool_use",
            ),
        ),
        (
            "end-turn-with-client-tool",
            _native_client_tool_frames(stop_reason="end_turn", partial_json="{}"),
        ),
    ],
)
async def test_native_sse_rejects_inconsistent_client_tool_terminal_before_mark(
    case: str,
    frames: list[str],
) -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key=case,
        frames=frames,
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert marks == []


@pytest.mark.asyncio
async def test_native_sse_allows_max_tokens_with_partial_client_tool_input() -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="max-tokens-partial-tool",
        frames=_native_client_tool_frames(
            stop_reason="max_tokens",
            partial_json='{"query":',
        ),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert '"stop_reason":"max_tokens"' in body.replace(" ", "")
    assert body.count("event: message_stop") == 1
    assert marks == ["max-tokens-partial-tool"]


def _native_unknown_nonterminal_frame(sentinel: str) -> str:
    return _native_sse_frame(
        "future_progress",
        {
            "type": "future_progress",
            "provider_debug": sentinel,
        },
    )


def _native_valid_flow_with_unknown_event(sentinel: str) -> list[str]:
    return [
        _native_message_start_frame(),
        _native_unknown_nonterminal_frame(sentinel),
        *_native_text_content_frames(0, "valid after future event"),
        _native_message_delta_frame(),
        _native_message_stop_frame(),
    ]


@pytest.mark.asyncio
async def test_native_sse_ignores_unknown_nonterminal_event_until_valid_terminal() -> None:
    sentinel = "native-future-event-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="future-event-valid",
        frames=_native_valid_flow_with_unknown_event(sentinel),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert "event: error" not in body
    assert "future_progress" not in body
    assert sentinel not in body
    assert body.count("event: message_stop") == 1
    assert marks == ["future-event-valid"]


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_unknown_native_events_do_not_certify_or_poison_usage() -> None:
    sentinel = "native-future-concurrent-secret-/private/provider-stream.json"
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("unknown-valid", "unknown-only")}
    release = asyncio.Event()
    tasks = [
        asyncio.create_task(
            _consume_native_sse_boundary(
                key=key,
                frames=(
                    _native_valid_flow_with_unknown_event(sentinel)
                    if key == "unknown-valid"
                    else [_native_unknown_nonterminal_frame(sentinel)]
                ),
                marks=marks,
                lifecycle=lifecycle,
                entered=entered[key],
                release=release,
            )
        )
        for key in entered
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        valid_body, unknown_only_body = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert "event: error" not in valid_body
    assert unknown_only_body.count("event: error") == 1
    assert sentinel not in valid_body + unknown_only_body
    assert marks == ["unknown-valid"]


@pytest.mark.asyncio
async def test_native_sse_empty_end_turn_lifecycle_fails_closed() -> None:
    marks: list[str] = []
    lifecycle: list[str] = []

    body = await _consume_native_sse_boundary(
        key="empty-end-turn",
        frames=_native_complete_flow([]),
        marks=marks,
        lifecycle=lifecycle,
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert marks == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_native_empty_end_turn_and_valid_output_are_request_local() -> None:
    marks: list[str] = []
    lifecycle: list[str] = []
    entered = {key: asyncio.Event() for key in ("empty-end-turn", "valid-output")}
    release = asyncio.Event()
    tasks = {
        key: asyncio.create_task(
            _consume_native_sse_boundary(
                key=key,
                frames=(
                    _native_complete_flow([])
                    if key == "empty-end-turn"
                    else _native_complete_flow(
                        _native_text_content_frames(0, "valid concurrent output")
                    )
                ),
                marks=marks,
                lifecycle=lifecycle,
                entered=entered[key],
                release=release,
            )
        )
        for key in entered
    }
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release.set()
        empty_body, valid_body = await asyncio.wait_for(
            asyncio.gather(
                tasks["empty-end-turn"],
                tasks["valid-output"],
            ),
            timeout=1.0,
        )
    finally:
        release.set()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    assert empty_body.count("event: error") == 1
    assert "event: message_stop" not in empty_body
    assert "event: error" not in valid_body
    assert valid_body.count("event: message_stop") == 1
    assert marks == ["valid-output"]
