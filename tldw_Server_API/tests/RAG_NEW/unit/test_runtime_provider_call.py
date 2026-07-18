import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.runtime_provider_call import (
    await_runtime_bound_provider_call,
    provider_result_succeeded,
)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("useful provider text", True),
        ({"choices": [{"message": {"content": "useful structured result"}}]}, True),
        ({"content": "useful top-level content"}, True),
        ({"text": "useful top-level text"}, True),
        (None, False),
        ("", False),
        ("   ", False),
        ({}, False),
        ({"usage": {"total_tokens": 3}}, False),
        ({"choices": []}, False),
        ({"choices": [{"message": {"content": ""}}]}, False),
        ({"content": "", "text": "   "}, False),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
            False,
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {"name": "lookup", "arguments": "{}"},
                        }
                    }
                ]
            },
            False,
        ),
        (["useful-looking list text"], False),
        (42, False),
        (object(), False),
        ("Error: provider unavailable", False),
        ("provider_unavailable", False),
        ('data: {"error":{"code":"provider_unavailable"}}\n\n', False),
        ('{"error":{"code":"provider_unavailable"}}', False),
    ],
    ids=[
        "valid-text",
        "valid-structured",
        "valid-top-level-content",
        "valid-top-level-text",
        "none",
        "empty-text",
        "whitespace-text",
        "empty-dict",
        "unrelated-dict",
        "empty-choices",
        "empty-choice-content",
        "empty-top-level-text",
        "tool-only",
        "function-only",
        "list",
        "scalar",
        "arbitrary-object",
        "error-prefix",
        "canonical-code",
        "sse-error",
        "serialized-error",
    ],
)
def test_provider_result_succeeded_accepts_only_supported_text_shapes(
    result: Any,
    expected: bool,
) -> None:
    assert provider_result_succeeded(result) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected_marks"),
    [
        ({"choices": [{"message": {"content": "usable"}}]}, 1),
        ({"content": "usable legacy content"}, 1),
        ({}, 0),
        ({"choices": [{"message": {"content": ""}}]}, 0),
    ],
    ids=["openai-text", "top-level-content", "empty-dict", "empty-content"],
)
async def test_runtime_bound_call_marks_only_semantic_normal_result(
    result: Any,
    expected_marks: int,
) -> None:
    handle = object()
    marked: list[object] = []

    class Runtime:
        async def mark_used(self, selected_handle: object) -> None:
            marked.append(selected_handle)

    async def provider_call() -> Any:
        return result

    returned = await await_runtime_bound_provider_call(
        provider_call(),
        credential_runtime=Runtime(),
        credential_handle=handle,
    )

    assert returned is result
    assert marked == [handle] * expected_marks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_result", "expected_marks"),
    [
        ({"choices": [{"message": {"content": "late usable text"}}]}, 1),
        ({"text": "late top-level text"}, 1),
        ({}, 0),
        ({"choices": []}, 0),
        ({"choices": [{"message": {"content": ""}}]}, 0),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"function": {"name": "lookup", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            },
            0,
        ),
        ("provider_unavailable", 0),
        ('data: {"error":{"code":"provider_unavailable"}}\n\n', 0),
    ],
    ids=[
        "openai-text",
        "top-level-text",
        "empty-dict",
        "empty-choices",
        "empty-content",
        "tool-only",
        "canonical-error",
        "sse-error",
    ],
)
async def test_runtime_bound_call_cancellation_drains_then_marks_only_semantic_late_result_before_scope_close(
    late_result: Any,
    expected_marks: int,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    lifecycle: list[str] = []
    handle = object()

    class Runtime:
        async def mark_used(self, selected_handle: object) -> None:
            assert selected_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    runtime = Runtime()

    async def provider_call() -> Any:
        entered.set()
        await release.wait()
        lifecycle.append("provider-exit")
        return late_result

    async def endpoint_scope() -> Any:
        try:
            return await await_runtime_bound_provider_call(
                provider_call(),
                credential_runtime=runtime,
                credential_handle=handle,
            )
        finally:
            await runtime.close()

    task = asyncio.create_task(endpoint_scope())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert lifecycle == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["provider-exit", *(["mark"] * expected_marks), "close"]


@pytest.mark.asyncio
async def test_concurrent_runtime_bound_calls_do_not_cross_mark_handles() -> None:
    release = asyncio.Event()
    entered = [asyncio.Event(), asyncio.Event()]
    valid_handle = object()
    malformed_handle = object()
    marked: list[object] = []
    valid_result = {"choices": [{"message": {"content": "usable"}}]}
    malformed_result: dict[str, Any] = {}

    class Runtime:
        async def mark_used(self, selected_handle: object) -> None:
            await asyncio.sleep(0)
            marked.append(selected_handle)

    runtime = Runtime()

    async def provider_call(index: int, result: Any) -> Any:
        entered[index].set()
        await release.wait()
        return result

    valid_task = asyncio.create_task(
        await_runtime_bound_provider_call(
            provider_call(0, valid_result),
            credential_runtime=runtime,
            credential_handle=valid_handle,
        )
    )
    malformed_task = asyncio.create_task(
        await_runtime_bound_provider_call(
            provider_call(1, malformed_result),
            credential_runtime=runtime,
            credential_handle=malformed_handle,
        )
    )
    try:
        await asyncio.gather(
            *(asyncio.wait_for(event.wait(), timeout=1.0) for event in entered)
        )
        release.set()
        returned_valid, returned_malformed = await asyncio.gather(
            valid_task,
            malformed_task,
        )
    finally:
        release.set()
        for task in (valid_task, malformed_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(valid_task, malformed_task, return_exceptions=True)

    assert returned_valid is valid_result
    assert returned_malformed is malformed_result
    assert marked == [valid_handle]
