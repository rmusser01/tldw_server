import asyncio
import json
import threading

import pytest
from loguru import logger

from tldw_Server_API.app.core.LLM_Calls import anthropic_messages as anthropic_messages_module
from tldw_Server_API.app.core.LLM_Calls.anthropic_messages import (
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    openai_response_to_anthropic,
    openai_stream_to_anthropic,
)
from tldw_Server_API.app.core.LLM_Calls.sse import finalize_stream


def test_anthropic_messages_to_openai_tool_handling():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": [{"type": "text", "text": "ok"}]},
            ],
        },
    ]

    openai_messages, system_message = anthropic_messages_to_openai(messages, None)

    assert system_message is None
    assert openai_messages[0]["role"] == "user"
    assert openai_messages[1]["role"] == "assistant"
    assert "tool_calls" in openai_messages[1]
    assert openai_messages[2]["role"] == "tool"
    assert openai_messages[2]["tool_call_id"] == "tool_1"


def test_openai_response_to_anthropic_maps_tool_calls():
    response = {
        "id": "chatcmpl-1",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "content": "hello",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": "{\"q\": \"x\"}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }

    payload = openai_response_to_anthropic(response, model="gpt-4")

    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert any(block["type"] == "tool_use" for block in payload["content"])
    assert payload["usage"]["input_tokens"] == 2
    assert payload["usage"]["output_tokens"] == 1


def test_openai_response_to_anthropic_maps_legacy_function_call():
    """Legacy OpenAI function_call output remains valid Anthropic tool use."""
    response = {
        "id": "chatcmpl-legacy-function",
        "model": "legacy-model",
        "choices": [
            {
                "message": {
                    "content": None,
                    "function_call": {
                        "name": "lookup",
                        "arguments": '{"query":"weather"}',
                    },
                },
                "finish_reason": "function_call",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }

    payload = openai_response_to_anthropic(response, model="legacy-model")

    assert payload["content"] == [
        {
            "type": "tool_use",
            "id": "tool_0",
            "name": "lookup",
            "input": {"query": "weather"},
        }
    ]
    assert payload["stop_reason"] == "tool_use"
    assert payload["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_anthropic_tool_choice_any_maps_to_required():
    assert anthropic_tool_choice_to_openai("any") == "required"
    assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_emits_events():
    async def _stream():
        yield 'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}\n\n'
        yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}\n\n'

    chunks = [chunk async for chunk in openai_stream_to_anthropic(_stream(), model="gpt-4")]

    assert any("event: message_start" in chunk for chunk in chunks)
    assert any("event: content_block_delta" in chunk for chunk in chunks)
    assert any("event: message_stop" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_tool_calls_no_duplicate_start():
    async def _stream():
        yield 'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": "{\\"q\\":"}}]}, "finish_reason": null}]}\n\n'
        yield 'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"x\\"}"}}]}, "finish_reason": null}]}\n\n'
        yield 'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'

    chunks = [chunk async for chunk in openai_stream_to_anthropic(_stream(), model="gpt-4")]

    content_block_starts = [chunk for chunk in chunks if "event: content_block_start" in chunk]
    assert len(content_block_starts) == 1


def _openai_stream_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _collect_anthropic_stream(stream):
    return [
        chunk
        async for chunk in openai_stream_to_anthropic(stream, model="gpt-4")
    ]


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_rejects_tool_provider_index_drift():
    sentinel = "conversion-index-drift-secret-/private/provider-stream.json"
    arguments = json.dumps({"probe": sentinel}, separators=(",", ":"))
    split_at = max(1, len(arguments) // 2)

    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-shifted",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": arguments[:split_at],
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call-shifted",
                                    "function": {
                                        "arguments": arguments[split_at:]
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_preserves_ordered_parallel_tools():
    calls = [
        {
            "index": index,
            "id": f"call-ordered-{index}",
            "function": {
                "name": f"ordered_tool_{index}",
                "arguments": json.dumps(
                    {"position": index},
                    separators=(",", ":"),
                ),
            },
        }
        for index in range(2)
    ]

    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": calls},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )

    assert "event: error" not in body
    assert body.index("call-ordered-0") < body.index("call-ordered-1")
    assert body.index("ordered_tool_0") < body.index("ordered_tool_1")
    assert body.count("event: message_stop") == 1


@pytest.mark.asyncio
async def test_openai_stream_tool_before_text_has_sequential_anthropic_indexes():
    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-tool-before-text",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "text after tool identity"},
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield _openai_stream_frame(
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        )

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )
    starts = [
        json.loads(line.removeprefix("data: "))["index"]
        for frame in body.split("\n\n")
        if "event: content_block_start" in frame
        for line in frame.splitlines()
        if line.startswith("data: ")
    ]

    assert "event: error" not in body
    assert starts == [0, 1]
    assert body.count("event: message_stop") == 1


def _invalid_finish_stream(case: str, sentinel: str):
    if case == "unknown-reason":
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "apparently valid"},
                        "finish_reason": f"error: {sentinel}",
                    }
                ]
            }
        )
        return
    if case == "tool-finish-without-tool":
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "apparently valid"},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        return
    if case == "end-turn-with-tool":
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-inconsistent-stop",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        return
    raise AssertionError(f"Unknown finish case: {case}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["unknown-reason", "tool-finish-without-tool", "end-turn-with-tool"],
)
async def test_openai_stream_rejects_unknown_or_inconsistent_finish_reason(
    case: str,
):
    sentinel = f"conversion-{case}-secret-/private/provider-stream.json"

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _invalid_finish_stream(case, sentinel),
                model="gpt-4",
            )
        ]
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body
    assert sentinel not in body


@pytest.mark.asyncio
async def test_content_filter_maps_to_anthropic_refusal():
    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "filtered output"},
                        "finish_reason": "content_filter",
                    }
                ]
            }
        )

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )

    assert '"stop_reason": "refusal"' in body
    assert "event: error" not in body


@pytest.mark.asyncio
async def test_openai_stream_preserves_typed_error_prefix_and_rejects_raw_error():
    async def _typed_stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "Error: requested heading"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    async def _raw_stream():
        yield "Error: raw provider failure /private/provider.json\n"

    typed_body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _typed_stream(),
                model="gpt-4",
            )
        ]
    )
    raw_body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _raw_stream(),
                model="gpt-4",
            )
        ]
    )

    assert "Error: requested heading" in typed_body
    assert "event: error" not in typed_body
    assert raw_body.count("event: error") == 1
    assert "/private/provider.json" not in raw_body


@pytest.mark.asyncio
async def test_openai_stream_preserves_internal_error_code_as_assistant_text():
    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "provider_unavailable"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )

    assert "provider_unavailable" in body
    assert "event: error" not in body
    assert body.count("event: message_stop") == 1


@pytest.mark.asyncio
async def test_openai_stream_preserves_truncated_tool_json_on_max_tokens():
    partial_json = '{"query":"unfinished'

    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-truncated",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": partial_json,
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield _openai_stream_frame(
            {"choices": [{"delta": {}, "finish_reason": "length"}]}
        )

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert any(
        payload.get("delta", {}).get("partial_json") == partial_json
        for payload in payloads
    )
    assert '"stop_reason": "max_tokens"' in body
    assert "event: error" not in body
    assert body.count("event: message_stop") == 1


@pytest.mark.asyncio
async def test_openai_stream_rejects_bare_done_without_finish_reason():
    async def _stream():
        yield _openai_stream_frame(
            {
                "choices": [
                    {
                        "delta": {"content": "truncated output"},
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield "data: [DONE]\n\n"

    body = "".join(
        [
            chunk
            async for chunk in openai_stream_to_anthropic(
                _stream(),
                model="gpt-4",
            )
        ]
    )

    assert body.count("event: error") == 1
    assert "event: message_stop" not in body


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_closes_stream():
    class _ClosableStream:
        def __init__(self, items):
            self._items = list(items)
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

        async def aclose(self):
            self.closed = True

    stream = _ClosableStream(
        [
            'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}\n\n',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            'data: {"choices": [{"delta": {"content": "ignored"}, "finish_reason": null}]}\n\n',
        ]
    )

    chunks = [chunk async for chunk in openai_stream_to_anthropic(stream, model="gpt-4")]

    assert any("event: message_stop" in chunk for chunk in chunks)
    assert stream.closed is True


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_bounds_noncooperative_aclose():
    close_started = asyncio.Event()
    close_finished = asyncio.Event()
    release_close = asyncio.Event()

    class _ResistantAsyncCloseStream:
        def __init__(self):
            self._items = [
                'data: {"choices":[{"delta":{"content":"bounded"},"finish_reason":null}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

        async def aclose(self):
            close_started.set()
            try:
                while not release_close.is_set():
                    try:
                        await release_close.wait()
                    except asyncio.CancelledError:
                        continue
            finally:
                close_finished.set()

    stream = _ResistantAsyncCloseStream()
    collect_task = asyncio.create_task(
        _collect_anthropic_stream(stream),
    )
    try:
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        chunks = await asyncio.wait_for(asyncio.shield(collect_task), timeout=0.25)
        assert not release_close.is_set()
    finally:
        release_close.set()
        await asyncio.wait_for(close_finished.wait(), timeout=1.0)
        await asyncio.gather(collect_task, return_exceptions=True)

    assert any("event: message_stop" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_openai_stream_to_anthropic_runs_blocking_close_off_loop_and_isolates_peer():
    close_started = threading.Event()
    close_finished = threading.Event()
    release_close = threading.Event()
    loop_responsive = asyncio.Event()

    class _BlockingSyncCloseStream:
        def __init__(self):
            self._items = [
                'data: {"choices":[{"delta":{"content":"blocked"},"finish_reason":null}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

        def close(self):
            close_started.set()
            try:
                release_close.wait(timeout=2.0)
            finally:
                close_finished.set()

    async def _healthy_peer():
        yield 'data: {"choices":[{"delta":{"content":"healthy"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    blocked_task = asyncio.create_task(
        _collect_anthropic_stream(_BlockingSyncCloseStream())
    )
    healthy_task = asyncio.create_task(_collect_anthropic_stream(_healthy_peer()))
    asyncio.get_running_loop().call_later(0.02, loop_responsive.set)
    watchdog = threading.Timer(0.5, release_close.set)
    watchdog.start()
    try:
        await asyncio.wait_for(loop_responsive.wait(), timeout=0.25)
        assert close_started.is_set()
        assert not release_close.is_set()
        healthy_chunks = await asyncio.wait_for(
            asyncio.shield(healthy_task),
            timeout=0.25,
        )
        blocked_chunks = await asyncio.wait_for(
            asyncio.shield(blocked_task),
            timeout=0.25,
        )
    finally:
        release_close.set()
        watchdog.cancel()
        await asyncio.gather(blocked_task, healthy_task, return_exceptions=True)
        assert await asyncio.to_thread(close_finished.wait, 1.0)

    assert any("event: message_stop" in chunk for chunk in blocked_chunks)
    assert any("healthy" in chunk for chunk in healthy_chunks)
    assert any("event: message_stop" in chunk for chunk in healthy_chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_kind", ["aclose", "close"])
async def test_anthropic_stream_close_logs_only_error_type(close_kind):
    sentinel = "sk-anthropic-close-/private/provider-stream.json"
    records = []

    class _FailingCloseStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    stream = _FailingCloseStream()

    if close_kind == "aclose":
        async def _aclose():
            raise RuntimeError(sentinel)

        stream.aclose = _aclose
    else:
        def _close():
            raise RuntimeError(sentinel)

        stream.close = _close

    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        chunks = [
            chunk
            async for chunk in anthropic_messages_module.openai_stream_to_anthropic(
                stream,
                model="gpt-4",
            )
        ]
    finally:
        logger.remove(sink_id)

    assert len(chunks) == 1
    assert "event: error" in chunks[0]
    assert sentinel not in chunks[0]
    rendered = repr(records)
    assert f"Anthropic stream {close_kind} failed; error_type=RuntimeError" in rendered
    assert sentinel not in rendered
    assert "Traceback" not in rendered


def test_finalize_stream_close_logs_only_error_type():
    sentinel = "sk-sse-finalize-/private/provider-response.json"
    records = []

    class _FailingResponse:
        def close(self):
            raise RuntimeError(sentinel)

    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        assert list(finalize_stream(_FailingResponse())) == ["data: [DONE]\n\n"]
    finally:
        logger.remove(sink_id)

    rendered = repr(records)
    assert "SSE finalize_stream failed to close response; error_type=RuntimeError" in rendered
    assert sentinel not in rendered
    assert "Traceback" not in rendered
