"""Gemini usageMetadata is costed in full: thinking tokens, tool-use prompt tokens, the service tier, and the
stream's usage chunk (TASK-13952)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.exceptions import ChatAPIError
from tldw_Server_API.app.core.LLM_Calls.providers import google_adapter as gmod
from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _StreamUsage
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage

pytestmark = pytest.mark.unit

# The repo fixture (tests/Chat/unit/test_chat_service_content.py): totalTokenCount 115 = 80 prompt + 30 candidates
# + 5 thoughts; 50 of the prompt tokens were cached.
USAGE_METADATA = {
    "promptTokenCount": 80,
    "candidatesTokenCount": 30,
    "totalTokenCount": 115,
    "cachedContentTokenCount": 50,
    "thoughtsTokenCount": 5,
}
EXPECTED_USAGE = {
    "prompt_tokens": 80,
    "completion_tokens": 35,
    "total_tokens": 115,
    "thoughtsTokenCount": 5,
    "cachedContentTokenCount": 50,
}


def _reply(usage_metadata: dict[str, Any]) -> dict[str, Any]:
    """A generateContent response with one text candidate and ``usage_metadata``."""
    return {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": usage_metadata,
    }


def test_thinking_tokens_are_counted_as_output() -> None:
    """Google bills thoughtsTokenCount as output, apart from candidatesTokenCount (AC #1); the block keeps the
    thinking share under its own name for the row's reasoning_tokens, and total_tokens is Gemini's total."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply(USAGE_METADATA))["usage"]

    assert usage == EXPECTED_USAGE


def test_the_normalized_block_is_not_double_counted() -> None:
    """The usage normalizer reads completion_tokens (thinking included) and reports the thinking share apart."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply(USAGE_METADATA))["usage"]

    normalized = normalize_llm_usage(provider="google", usage=usage)

    assert (normalized.input_tokens, normalized.output_tokens, normalized.total_tokens) == (80, 35, 115)
    assert (normalized.reasoning_tokens, normalized.cache_read_input_tokens) == (5, 50)


def test_tool_use_prompt_tokens_are_counted_as_input() -> None:
    """toolUsePromptTokenCount is outside promptTokenCount (Gemini's total = prompt + tool-use prompt + thoughts +
    candidates), so it joins prompt_tokens and the block's sides add up to its total."""
    usage = GoogleAdapter._normalize_to_openai_shape(
        _reply({**USAGE_METADATA, "toolUsePromptTokenCount": 7, "totalTokenCount": 122})
    )["usage"]

    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (87, 35, 122)
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_a_reply_without_thinking_or_cache_is_unchanged() -> None:
    """A plain usageMetadata gives the plain block, a missing count staying absent."""
    usage = GoogleAdapter._normalize_to_openai_shape(
        _reply({"promptTokenCount": 10, "candidatesTokenCount": 5})
    )["usage"]

    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": None}


@pytest.mark.parametrize(
    "usage_metadata",
    ["garbage", None, {"candidatesTokenCount": "30", "thoughtsTokenCount": True, "serviceTier": 3}],
    ids=["string", "missing", "wrong-types"],
)
def test_a_malformed_usage_metadata_never_raises(usage_metadata: Any) -> None:
    """A count that is not an int is left out of the sums; the reply is still normalized (not returned raw)."""
    shaped = GoogleAdapter._normalize_to_openai_shape({**_reply({}), "usageMetadata": usage_metadata})

    assert shaped["object"] == "chat.completion"
    assert "service_tier" not in shaped["usage"] and "thoughtsTokenCount" not in shaped["usage"]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [("flex", "flex"), ("priority", "priority"), ("SERVICE_TIER_PRIORITY", "priority"), ("unspecified", "unspecified")],
)
def test_the_service_tier_is_passed_on(reported: str, expected: str) -> None:
    """UsageMetadata.serviceTier (unspecified, standard, flex, priority) is kept as service_tier, lowercased and
    without an enum prefix, so the catalog's flex and priority rates apply (AC #3)."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply({**USAGE_METADATA, "serviceTier": reported}))["usage"]

    assert usage["service_tier"] == expected


# --- The stream ends with one usage chunk (AC #2) ---

# Two events; Gemini's usageMetadata is cumulative, so the last one is the total.
STREAM_EVENTS = [
    {
        "candidates": [{"content": {"parts": [{"text": "Hel"}]}, "index": 0}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 2, "totalTokenCount": 82},
    },
    {
        "candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {**USAGE_METADATA, "serviceTier": "flex"},
    },
]
STREAM_BODY = "".join(f"data: {json.dumps(event)}\n\n" for event in STREAM_EVENTS).encode()
STREAM_BODIES = {
    "gemini-end": STREAM_BODY,  # Gemini ends a stream by closing it
    # A proxy may add a [DONE] line, which ends the stream: a later event is not read.
    "proxy-done": STREAM_BODY + b'data: [DONE]\n\ndata: {"usageMetadata": {"promptTokenCount": 1}}\n\n',
    "json-array": json.dumps(STREAM_EVENTS).encode(),  # the body without alt=sse
}
STREAM_REQUEST = {"model": "gemini-2.5-flash", "api_key": "sk-test", "messages": [{"role": "user", "content": "hi"}]}


def _stream(monkeypatch: pytest.MonkeyPatch, body: bytes | Iterator[bytes]) -> Iterable[str]:
    """GoogleAdapter.stream over ``body``, through a stubbed transport (no network)."""
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(":streamGenerateContent")
        return httpx.Response(200, content=body)

    def fake_client(*_args: Any, **_kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gmod, "http_client_factory", fake_client)
    return GoogleAdapter().stream(dict(STREAM_REQUEST))


@pytest.fixture(params=list(STREAM_BODIES))
def gemini_stream(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> list[str]:
    """The lines GoogleAdapter.stream yields for each body of STREAM_EVENTS."""
    return list(_stream(monkeypatch, STREAM_BODIES[request.param]))


def _payloads(lines: list[str]) -> list[Any]:
    """The JSON payload of every data line, "[DONE]" as a string."""
    out: list[Any] = []
    for line in lines:
        for part in line.splitlines():
            if part.startswith("data:"):
                payload = part[5:].strip()
                out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_the_stream_ends_with_one_usage_chunk_from_the_last_usage_metadata(gemini_stream: list[str]) -> None:
    """The chunk is OpenAI's shape ({"choices": [], "usage": ...}) with the block the non-streaming reply gets,
    thinking tokens and the service tier included; it comes after the last delta and before [DONE]."""
    payloads = _payloads(gemini_stream)

    usage_chunks = [p for p in payloads if isinstance(p, dict) and "usage" in p]
    assert usage_chunks == [{"choices": [], "usage": {**EXPECTED_USAGE, "service_tier": "flex"}}]
    assert payloads[-2:] == [usage_chunks[0], "[DONE]"]
    assert [p["choices"][0]["delta"]["content"] for p in payloads[:-2]] == ["Hel", "lo"]


def test_the_provider_level_recorder_reads_the_usage_chunk(gemini_stream: list[str]) -> None:
    """_StreamUsage, which the provider-level recorder and the chat stream handler feed, takes the chunk as the
    stream's usage, so a streamed Gemini reply is no longer costed from the estimate."""
    seen = _StreamUsage()
    for line in gemini_stream:
        seen.feed(line)

    assert seen.usage == {**EXPECTED_USAGE, "service_tier": "flex"}
    assert seen.chars == len("Hello")


@pytest.mark.parametrize("failure", ["transport-error", "in-band-error"])
def test_a_stream_that_fails_after_usage_arrived_is_recorded_from_it(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A stream that fails mid-way sends the usage seen so far before the error, so the recorders cost it from
    Gemini's counts, thinking tokens included, not from the estimate."""

    def body() -> Iterator[bytes]:
        yield STREAM_BODY
        if failure == "transport-error":
            raise httpx.ReadError("connection reset")
        yield b'data: {"error": {"code": 500, "message": "internal", "status": "INTERNAL"}}\n\n'

    seen = _StreamUsage()
    with pytest.raises(ChatAPIError):
        for line in _stream(monkeypatch, body()):
            seen.feed(line)

    assert seen.usage == {**EXPECTED_USAGE, "service_tier": "flex"}


@pytest.mark.parametrize("extra", [{}, {"usageMetadata": {}}], ids=["absent", "empty"])
def test_a_stream_without_usage_metadata_sends_no_usage_chunk(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any]
) -> None:
    """No usageMetadata (a proxy that strips it, say), or an empty one: no chunk, and the recorders estimate as
    before."""
    event = {"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}], **extra}

    payloads = _payloads(list(_stream(monkeypatch, f"data: {json.dumps(event)}\n\n".encode())))

    assert [p for p in payloads if isinstance(p, dict) and "usage" in p] == []
    assert payloads[-1] == "[DONE]"
