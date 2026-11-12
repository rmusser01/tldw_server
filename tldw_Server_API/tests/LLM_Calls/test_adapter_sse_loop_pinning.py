"""Pin client-visible SSE behaviour of provider adapter stream loops (TASK-13373).

Each adapter is driven through a real ``httpx.Client`` on a ``MockTransport``,
so decoding is whatever the production transport does. Pinned per adapter:
the frames for a normal stream (single DONE), invalid UTF-8 bytes, an in-band
SSE error event, and a transport failure mid-stream.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.LLM_Calls import chat_calls, http_helpers

pytestmark = pytest.mark.unit

_DELTA = b'data: {"choices":[{"delta":{"content":"hi"}}]}'
_BAD_UTF8 = b'data: {"choices":[{"delta":{"content":"a\xffb"}}]}'
_IN_BAND_ERROR = b'data: {"error":{"message":"slow down","type":"rate_limit_error","code":429}}'

# Adapters whose stream() raises typed, sanitized ChatAPIErrors on failure.
_RAISING_ADAPTERS = (
    ("openai_adapter", "OpenAIAdapter", "test-model"),
    ("groq_adapter", "GroqAdapter", "test-model"),
    ("openrouter_adapter", "OpenRouterAdapter", "test-model"),
    ("deepseek_adapter", "DeepSeekAdapter", "test-model"),
    ("huggingface_adapter", "HuggingFaceAdapter", "test-model"),
    ("qwen_adapter", "QwenAdapter", "test-model"),
    ("mistral_adapter", "MistralAdapter", "test-model"),
    ("bedrock_adapter", "BedrockAdapter", "meta.llama3-8b-instruct"),
)
# Adapters on the legacy session facade that report failures as an SSE error frame.
_FRAME_ADAPTERS = (
    ("zai_adapter", "ZaiAdapter", "zai"),
    ("moonshot_adapter", "MoonshotAdapter", "moonshot"),
)


class _BrokenStream(httpx.SyncByteStream):
    """Yields one delta, then fails like a dropped connection."""

    def __iter__(self) -> Iterator[bytes]:
        yield _DELTA + b"\n\n"
        raise httpx.ReadError("connection dropped")


def _transport(body: bytes | None) -> httpx.MockTransport:
    """Serve ``body`` as an SSE response, or a mid-stream drop when ``body`` is None."""
    def handler(request: httpx.Request) -> httpx.Response:
        if body is None:
            return httpx.Response(200, stream=_BrokenStream())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def _body(*lines: bytes) -> bytes:
    """Join SSE lines into one event-stream body."""
    return b"".join(line + b"\n\n" for line in lines)


def _run(monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, model: str, body: bytes | None) -> list[str]:
    """Stream one request through the real adapter on the mock transport; return its frames."""
    transport = _transport(body)
    module = importlib.import_module(f"tldw_Server_API.app.core.LLM_Calls.providers.{module_name}")

    def _client_on_mock_transport(*_args: Any, **_kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=transport)

    # Swap the project's own client factories (the sanctioned seam); the real
    # adapter code still builds requests and parses the stream.
    if hasattr(module, "http_client_factory"):
        monkeypatch.setattr(module, "http_client_factory", _client_on_mock_transport)
    monkeypatch.setattr(http_helpers, "_hc_create_client", _client_on_mock_transport)
    adapter = getattr(module, adapter_name)()
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": model,
        "api_key": "test-key",
        "app_config": {},
        "stream": True,
    }
    return list(adapter.stream(request))


def _contents(frames: list[str]) -> list[str]:
    """Delta contents carried by the OpenAI-shaped frames, in order."""
    out = []
    for frame in frames:
        if frame.startswith("data: {"):
            payload = json.loads(frame[len("data: "):])
            for choice in payload.get("choices", []):
                out.append(choice["delta"]["content"])
    return out


def _error_frame(provider: str) -> str:
    """The bounded frame a frame-reporting adapter emits for any stream failure."""
    error = {
        "error": {
            "code": "provider_unavailable",
            "message": "The chat service provider is currently unavailable.",
            "type": f"{provider}_stream_error",
        }
    }
    return f"data: {json.dumps(error)}\n\n"


_RAISING_IDS = [name for _m, name, _model in _RAISING_ADAPTERS]


@pytest.mark.parametrize(("module_name", "adapter_name", "model"), _RAISING_ADAPTERS, ids=_RAISING_IDS)
def test_raising_adapter_forwards_deltas_and_one_done(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, model: str
) -> None:
    """Provider keep-alives and duplicate DONEs collapse to the deltas plus one DONE."""
    frames = _run(monkeypatch, module_name, adapter_name, model, _body(b": ping", _DELTA, b"data: [done]", b"data: [DONE]"))
    assert _contents(frames) == ["hi"]
    assert frames[-1] == "data: [DONE]\n\n"
    assert sum("[DONE]" in f for f in frames) == 1


@pytest.mark.parametrize(("module_name", "adapter_name", "model"), _RAISING_ADAPTERS, ids=_RAISING_IDS)
def test_raising_adapter_replaces_invalid_utf8(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, model: str
) -> None:
    """Invalid UTF-8 from the provider arrives as U+FFFD, not an error."""
    frames = _run(monkeypatch, module_name, adapter_name, model, _body(_BAD_UTF8, b"data: [DONE]"))
    assert _contents(frames) == ["a�b"]


@pytest.mark.parametrize(("module_name", "adapter_name", "model"), _RAISING_ADAPTERS, ids=_RAISING_IDS)
def test_raising_adapter_raises_typed_error_for_in_band_event(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, model: str
) -> None:
    """An in-band SSE error event raises the sanitized 502 ChatProviderError."""
    with pytest.raises(ChatAPIError) as exc_info:
        _run(monkeypatch, module_name, adapter_name, model, _body(_DELTA, _IN_BAND_ERROR))
    assert type(exc_info.value).__name__ == "ChatProviderError"
    assert exc_info.value.status_code == 502


@pytest.mark.parametrize(("module_name", "adapter_name", "model"), _RAISING_ADAPTERS, ids=_RAISING_IDS)
def test_raising_adapter_raises_on_mid_stream_transport_error(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, model: str
) -> None:
    """A dropped connection mid-stream raises a typed ChatAPIError."""
    with pytest.raises(ChatAPIError):
        _run(monkeypatch, module_name, adapter_name, model, None)


@pytest.mark.parametrize(("module_name", "adapter_name", "provider"), _FRAME_ADAPTERS, ids=["zai", "moonshot"])
def test_frame_adapter_streams_over_httpx_facade(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, provider: str
) -> None:
    """Z.AI and Moonshot stream over the httpx session facade (red before the iter_lines fix)."""
    frames = _run(monkeypatch, module_name, adapter_name, "test-model", _body(_BAD_UTF8, b"data: [done]"))
    assert frames == [
        'data: {"choices":[{"delta":{"content":"a�b"}}]}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.mark.parametrize("body", [_body(_DELTA, _IN_BAND_ERROR), None], ids=["in_band", "transport"])
@pytest.mark.parametrize(("module_name", "adapter_name", "provider"), _FRAME_ADAPTERS, ids=["zai", "moonshot"])
def test_frame_adapter_emits_bounded_error_frame(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, provider: str, body: bytes | None
) -> None:
    """Any stream failure ends with the bounded error frame and one DONE, after the deltas."""
    frames = _run(monkeypatch, module_name, adapter_name, "test-model", body)
    assert frames[-2:] == [_error_frame(provider), "data: [DONE]\n\n"]
    assert _contents(frames) == ["hi"]


class _RequestsLikeResponse:
    """requests-style response: iter_lines(decode_unicode=...) yields str lines."""

    status_code = 200

    def __init__(self, lines: list[str] | None) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False) -> Iterator[str]:
        assert decode_unicode is True
        if self._lines is None:
            yield _DELTA.decode()
            raise ConnectionError("connection dropped")
        yield from self._lines

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    "lines",
    [[_DELTA.decode(), "", _IN_BAND_ERROR.decode()], None],
    ids=["in_band", "transport"],
)
@pytest.mark.parametrize(("module_name", "adapter_name", "provider"), _FRAME_ADAPTERS, ids=["zai", "moonshot"])
def test_frame_adapter_error_frame_on_requests_like_session(
    monkeypatch: pytest.MonkeyPatch, module_name: str, adapter_name: str, provider: str, lines: list[str] | None
) -> None:
    """The same failure frames through a requests-style session (iter_lines(decode_unicode=True))."""
    class _Session:
        """Minimal requests-style session returning the canned response."""

        def post(self, *_a: Any, **_k: Any) -> _RequestsLikeResponse:
            return _RequestsLikeResponse(lines)

        def close(self) -> None:
            return None

    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda *_a, **_k: _Session())
    frames = _run(monkeypatch, module_name, adapter_name, "test-model", b"")
    assert frames == [
        _DELTA.decode() + "\n\n",
        _error_frame(provider),
        "data: [DONE]\n\n",
    ]
