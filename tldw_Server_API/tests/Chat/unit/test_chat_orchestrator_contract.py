from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


@dataclass
class _OrchestratorOutput:
    messages: list[dict[str, Any]]


async def run_orchestrator_stub(messages: list[dict[str, Any]]) -> _OrchestratorOutput:
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call

    captured: dict[str, Any] = {}

    def _fake_dispatch(**kwargs: Any) -> dict[str, Any]:
        captured["messages_payload"] = kwargs["messages_payload"]
        return {"ok": True}

    with patch(
        "tldw_Server_API.app.core.Chat.chat_orchestrator.perform_chat_api_call",
        side_effect=_fake_dispatch,
    ):
        chat_api_call(
            api_endpoint="openai",
            messages_payload=messages,
            streaming=False,
        )

    return _OrchestratorOutput(messages=captured["messages_payload"])


def run_stream_stub() -> Any:
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call

    chunks = [
        {"index": 0, "delta": "A"},
        {"index": 1, "delta": "B"},
    ]
    with patch(
        "tldw_Server_API.app.core.Chat.chat_orchestrator.perform_chat_api_call",
        return_value=iter(chunks),
    ):
        return chat_api_call(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "stream"}],
            streaming=True,
        )


@pytest.mark.asyncio
async def test_orchestrator_preserves_message_order():
    output = await run_orchestrator_stub(
        messages=[{"role": "user", "content": "A"}],
    )
    assert output.messages[0]["content"] == "A"


def test_orchestrator_emits_stream_chunks_in_order():
    chunks = list(run_stream_stub())
    assert chunks == sorted(chunks, key=lambda c: c["index"])


def test_legacy_orchestrator_forwards_authoritative_credential_marker():
    from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call

    captured: dict[str, Any] = {}

    def _fake_dispatch(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    with patch(
        "tldw_Server_API.app.core.Chat.chat_orchestrator.perform_chat_api_call",
        side_effect=_fake_dispatch,
    ):
        chat_api_call(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "hello"}],
            api_key=None,
            app_config={"openai_api": {}},
            credentials_resolved=True,
            streaming=False,
        )

    assert captured["api_key"] is None
    assert captured["app_config"] == {"openai_api": {}}
    assert captured["credentials_resolved"] is True


def test_provider_resolution_applies_default_provider_when_missing():
    from tldw_Server_API.app.core.Chat.orchestrator.provider_resolution import (
        resolve_provider,
    )

    provider = resolve_provider(model="gpt-4o-mini", provider=None)
    assert provider is not None


def test_stream_execution_maps_provider_errors_to_chat_error_shape():
    from tldw_Server_API.app.core.Chat.orchestrator.error_mapping import (
        map_stream_error,
    )

    err = map_stream_error(RuntimeError("provider exploded"))
    assert "message" in err
    assert "code" in err


def test_execute_stream_sync_wrapper_closes_underlying_iterator_on_early_close():
    from tldw_Server_API.app.core.Chat.orchestrator.stream_execution import execute_stream

    class _SyncClosableStream:
        def __init__(self) -> None:
            self._closed = False
            self._emitted = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._emitted:
                raise StopIteration
            self._emitted = True
            return {"delta": "x"}

        def close(self) -> None:
            self._closed = True

    stream = _SyncClosableStream()
    wrapped = execute_stream(stream)
    assert next(wrapped)["delta"] == "x"
    wrapped.close()
    assert stream._closed is True  # nosec B101


@pytest.mark.asyncio
async def test_execute_stream_async_wrapper_closes_underlying_iterator_on_early_close():
    from tldw_Server_API.app.core.Chat.orchestrator.stream_execution import execute_stream

    class _AsyncClosableStream:
        def __init__(self) -> None:
            self._closed = False
            self._emitted = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._emitted:
                raise StopAsyncIteration
            self._emitted = True
            return {"delta": "x"}

        async def aclose(self) -> None:
            self._closed = True

    stream = _AsyncClosableStream()
    wrapped = execute_stream(stream)
    first = await anext(wrapped)
    assert first["delta"] == "x"
    await wrapped.aclose()
    assert stream._closed is True  # nosec B101
