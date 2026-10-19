"""Every chat provider call for a known user records one usage row (LLM usage accounting spec §2, §3)."""

import asyncio
import gc
import inspect
import json
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.LLM_Calls import usage_accounting
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context

pytestmark = pytest.mark.unit

REQUEST = {"model": "m1", "messages": [{"role": "user", "content": "hello there, how are you"}]}
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
LINES = [
    'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
    "data: [DONE]\n\n",
]


class _FakeProvider(ChatProvider):
    """A provider with native-looking async methods that call its sync ones."""

    name = "fake-accounted"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion with a usage block."""
        return {"choices": [{"message": {"content": "hello"}}], "usage": dict(USAGE)}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield two content chunks, the second with usage, then DONE."""
        yield from LINES

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Run the sync chat on a thread, as production adapters do."""
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """Iterate the sync stream, as production adapters do."""
        for line in self.stream(request, timeout=timeout):
            yield line


class _NoUsageProvider(_FakeProvider):
    """A provider that reports no usage and sends no usage chunk."""

    name = "fake-no-usage"

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion without a usage block."""
        return {"choices": [{"message": {"content": "x" * 40}}]}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield content chunks with no usage."""
        yield 'data: {"choices":[{"delta":{"content":"abcdefgh"}}]}\n\n'
        yield "data: [DONE]\n\n"


class _InheritingProvider(_FakeProvider):
    """Defines no call methods of its own."""

    name = "fake-inheriting"


class _SuperCallingProvider(_FakeProvider):
    """Overrides chat and delegates to the parent."""

    name = "fake-super"

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Delegate to the parent's chat."""
        return super().chat(request, timeout=timeout)


class _AwaitableAstreamProvider(_FakeProvider):
    """Its astream is a coroutine that resolves to an async iterator."""

    name = "fake-awaitable-astream"

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:  # type: ignore[override]
        """Return an async iterator instead of being one."""

        async def _gen() -> AsyncIterator[str]:
            """Yield the canned lines."""
            for line in LINES:
                yield line

        return _gen()


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, and make the test's loop the accounting loop."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    return captured


async def _settled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wait for scheduled writes, then return the rows."""
    await llm_accounting.drain_accounting()
    return rows


async def test_chat_on_a_thread_records_one_row_for_the_context_user(rows) -> None:
    """A sync chat on a worker thread writes one row with the provider's own usage."""
    with llm_call_context(user_id=7):
        response = await asyncio.to_thread(_FakeProvider().chat, REQUEST)
    assert response["usage"] == USAGE
    (row,) = await _settled(rows)
    assert (row["user_id"], row["provider"], row["model"]) == (7, "fake-accounted", "m1")
    assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
    assert (row["operation"], row["endpoint"], row["key_id"], row["status"]) == ("llm_call", "internal", None, 200)
    assert row["usage_metadata"] == USAGE and row["estimated"] is None


async def test_the_content_scope_user_is_recorded_and_labels_come_from_the_context(rows) -> None:
    """With only a content scope, that user is recorded; operation and endpoint follow the context."""
    with scoped_context(user_id=9), llm_call_context(operation="rag", endpoint="POST:/api/v1/rag/search"):
        _FakeProvider().chat(REQUEST)
    (row,) = await _settled(rows)
    assert (row["user_id"], row["operation"], row["endpoint"]) == (9, "rag", "POST:/api/v1/rag/search")


async def test_no_user_exempt_and_self_recorded_calls_write_nothing(rows) -> None:
    """No row without a user, for an exempt call, or for a call that records itself."""
    _FakeProvider().chat(REQUEST)
    with llm_call_context(user_id=7, exempt=True):
        _FakeProvider().chat(REQUEST)
    with llm_call_context(user_id=7, records_itself=True):
        _FakeProvider().chat(REQUEST)
        list(_FakeProvider().stream(REQUEST))
    assert await _settled(rows) == []


async def test_nested_provider_methods_record_once(rows) -> None:
    """achat calling chat, astream iterating stream, and super().chat() each write one row."""
    with llm_call_context(user_id=7):
        await _FakeProvider().achat(REQUEST)
        assert [line async for line in _FakeProvider().astream(REQUEST)] == LINES
        _SuperCallingProvider().chat(REQUEST)
    assert [row["provider"] for row in await _settled(rows)] == ["fake-accounted", "fake-accounted", "fake-super"]


async def test_an_inheriting_subclass_records_once(rows) -> None:
    """A subclass with no methods of its own is recorded once, under its own name."""
    with llm_call_context(user_id=7):
        _InheritingProvider().chat(REQUEST)
    (row,) = await _settled(rows)
    assert row["provider"] == "fake-inheriting"


async def test_a_fully_consumed_stream_records_the_usage_chunk(rows) -> None:
    """A stream read to the end writes one row from the provider's usage chunk."""
    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    assert inspect.isgenerator(stream)
    assert list(stream) == LINES
    (row,) = await _settled(rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 2)
    assert row["usage_metadata"] == {"prompt_tokens": 5, "completion_tokens": 2}


async def test_a_stream_closed_early_records_an_estimate(rows) -> None:
    """Closing after the first chunk still writes one row, estimated from what was received."""
    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    assert next(stream) == LINES[0]
    stream.close()
    (row,) = await _settled(rows)
    assert row["estimated"] is True and row["estimate_source"] == "stream_estimate"
    assert row["completion_tokens"] == 1 and row["prompt_tokens"] >= 1


async def test_an_abandoned_stream_still_records_and_never_raises(rows) -> None:
    """A stream dropped without close is finalized by the collector and still writes its row."""
    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    next(stream)
    del stream
    gc.collect()
    assert len(await _settled(rows)) == 1


async def test_a_response_without_usage_is_estimated(rows) -> None:
    """With no usage block, tokens are estimated at four characters each and the row is marked."""
    with llm_call_context(user_id=7):
        _NoUsageProvider().chat(REQUEST)
        list(_NoUsageProvider().stream(REQUEST))
    chat_row, stream_row = await _settled(rows)
    assert chat_row["estimated"] is True and chat_row["estimate_source"] == "response_estimate"
    assert chat_row["completion_tokens"] == 10 and chat_row["usage_metadata"] is None
    assert stream_row["estimate_source"] == "stream_estimate" and stream_row["completion_tokens"] == 2


async def test_an_awaitable_astream_still_works_and_records_once(rows) -> None:
    """An astream that returns an awaitable resolves to a working iterator and writes one row."""
    with llm_call_context(user_id=7):
        pending = _AwaitableAstreamProvider().astream(REQUEST)
        assert inspect.isawaitable(pending)
        assert [line async for line in await pending] == LINES
    assert len(await _settled(rows)) == 1


async def test_an_accounting_failure_does_not_change_the_call(monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """When the recorder raises, the provider's result is returned unchanged."""

    async def _boom(**kwargs: Any) -> None:
        """Fail like a database outage."""
        raise RuntimeError("db down")

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _boom)
    with llm_call_context(user_id=7):
        assert _FakeProvider().chat(REQUEST)["usage"] == USAGE
        assert list(_FakeProvider().stream(REQUEST)) == LINES
    await llm_accounting.drain_accounting()


async def test_a_midstream_provider_error_reraises_the_same_object_and_still_records(rows) -> None:
    """A provider that fails after one chunk raises its own exception, and the one row is still written."""
    failure = ValueError("upstream broke")

    class _FailingStreamProvider(_FakeProvider):
        """Yields one chunk, then fails."""

        name = "fake-failing-stream"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield one content chunk and raise."""
            yield LINES[0]
            raise failure

    with llm_call_context(user_id=7):
        stream = _FailingStreamProvider().stream(REQUEST)
    assert next(stream) == LINES[0]
    with pytest.raises(ValueError) as raised:
        next(stream)
    assert raised.value is failure
    (row,) = await _settled(rows)
    assert row["estimate_source"] == "stream_estimate" and row["completion_tokens"] == 1


async def test_a_non_iterable_stream_result_is_passed_through_untouched(rows) -> None:
    """A broken provider that returns None from stream() still returns None, and nothing is recorded."""

    class _BrokenStreamProvider(_FakeProvider):
        """Returns nothing iterable."""

        name = "fake-broken-stream"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:  # type: ignore[override]
            """Return None, as a misbehaving adapter might."""
            return None  # type: ignore[return-value]

    with llm_call_context(user_id=7):
        assert _BrokenStreamProvider().stream(REQUEST) is None
    assert await _settled(rows) == []


def test_instrumenting_keeps_type_identity_and_abstractness() -> None:
    """Instances keep their class, methods are marked, and an abstract subclass stays abstract."""
    assert type(_FakeProvider()) is _FakeProvider
    for method in ("chat", "stream", "achat", "astream"):
        assert getattr(_FakeProvider.__dict__[method], "__llm_accounted__", False) is True
    assert "chat" not in _InheritingProvider.__dict__

    class _StillAbstract(ChatProvider):
        """Implements only capabilities."""

        def capabilities(self) -> dict[str, Any]:
            """No capabilities."""
            return {}

    with pytest.raises(TypeError):
        _StillAbstract()  # type: ignore[abstract]


async def test_non_finite_usage_never_raises_and_counts_as_zero(rows) -> None:
    """An inf or NaN token count is skipped like a missing one, and the call is unaffected."""

    class _NonFiniteProvider(_FakeProvider):
        """Reports non-finite prompt tokens."""

        name = "fake-non-finite"

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Return inf prompt tokens."""
            return {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": float("inf"), "completion_tokens": 7},
            }

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield a usage chunk with NaN prompt tokens."""
            yield 'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":NaN,"completion_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

    with llm_call_context(user_id=7):
        assert _NonFiniteProvider().chat(REQUEST)["usage"]["completion_tokens"] == 7
        assert len(list(_NonFiniteProvider().stream(REQUEST))) == 2
    chat_row, stream_row = await _settled(rows)
    assert (chat_row["prompt_tokens"], chat_row["completion_tokens"]) == (0, 7)
    assert (stream_row["prompt_tokens"], stream_row["completion_tokens"]) == (0, 2)


async def test_a_recorder_bug_never_reaches_the_caller(monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """A failure inside the recorder neither changes a result nor masks a provider exception."""

    def _bug(*args: Any, **kwargs: Any) -> None:
        """Fail like a bug in the recorder."""
        raise RuntimeError("recorder bug")

    monkeypatch.setattr(usage_accounting, "_submit_record", _bug)
    failure = ValueError("upstream broke")

    class _FailingStreamProvider(_FakeProvider):
        """Yields one chunk, then fails."""

        name = "fake-failing-stream-2"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield one content chunk and raise."""
            yield LINES[0]
            raise failure

    with llm_call_context(user_id=7):
        assert _FakeProvider().chat(REQUEST)["usage"] == USAGE
        assert list(_FakeProvider().stream(REQUEST)) == LINES
        stream = _FailingStreamProvider().stream(REQUEST)
        assert next(stream) == LINES[0]
        with pytest.raises(ValueError) as raised:
            next(stream)
        assert raised.value is failure
        assert [line async for line in _FakeProvider().astream(REQUEST)] == LINES


async def test_a_stream_item_holding_two_events_counts_both(rows) -> None:
    """One yielded string carrying several SSE lines is parsed line by line."""

    class _BatchedProvider(_FakeProvider):
        """Yields two events in one string."""

        name = "fake-batched"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield a batched item, then DONE."""
            yield LINES[0] + LINES[1]
            yield LINES[2]

    with llm_call_context(user_id=7):
        list(_BatchedProvider().stream(REQUEST))
    (row,) = await _settled(rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 2)
    assert row["usage_metadata"] == {"prompt_tokens": 5, "completion_tokens": 2}


async def test_image_payload_bytes_are_not_counted_in_the_prompt_estimate(rows) -> None:
    """A base64 image in the messages is redacted before the estimate is taken."""
    image = "data:image/png;base64," + "A" * 100_000
    request = {
        "model": "m1",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}],
    }
    with llm_call_context(user_id=7):
        _NoUsageProvider().chat(request)
    (row,) = await _settled(rows)
    assert row["prompt_tokens"] < 100


async def test_the_system_prompt_is_counted_in_the_prompt_estimate(rows) -> None:
    """The adapter request's separate system_message adds to the estimate."""
    request = {"model": "m1", "messages": [{"role": "user", "content": "twenty chars here.."}], "system_message": "s" * 4000}
    with llm_call_context(user_id=7):
        _NoUsageProvider().chat(request)
    (row,) = await _settled(rows)
    assert 1000 <= row["prompt_tokens"] <= 1030


def test_the_prompt_estimate_stays_linear_on_a_crafted_prompt() -> None:
    """Many ``data:image`` markers with no comma must not make the redaction quadratic."""
    request = {"model": "m1", "messages": [{"role": "user", "content": "data:image" * 50_000}]}
    started = time.monotonic()
    estimate = usage_accounting._prompt_estimate(request)
    assert time.monotonic() - started < 2.0
    assert estimate > 100_000


async def test_a_usage_block_without_token_fields_is_estimated(rows) -> None:
    """A block holding only total_tokens is not recorded as real provider usage of zero."""

    class _TotalOnlyProvider(_FakeProvider):
        """Reports only a total."""

        name = "fake-total-only"

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Return a usage block with no prompt or completion count."""
            return {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 50}}

    with llm_call_context(user_id=7):
        _TotalOnlyProvider().chat(REQUEST)
    (row,) = await _settled(rows)
    assert row["estimated"] is True and row["estimate_source"] == "response_estimate"
    assert row["usage_metadata"] is None


async def test_the_stream_estimate_counts_reasoning_and_tool_call_arguments(rows) -> None:
    """Tool-call arguments and reasoning text count toward the completion estimate."""

    class _ToolStreamProvider(_FakeProvider):
        """Streams tool arguments and reasoning, with no usage chunk."""

        name = "fake-tool-stream"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield an arguments delta and a reasoning delta."""
            args = "x" * 40
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"' + args + '"}}]}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"reasoning_content":"' + "y" * 40 + '"}}]}\n\n'
            yield "data: [DONE]\n\n"

    with llm_call_context(user_id=7):
        list(_ToolStreamProvider().stream(REQUEST))
    (row,) = await _settled(rows)
    assert row["completion_tokens"] == 20


async def test_a_malformed_line_does_not_hide_a_later_usage_line(rows) -> None:
    """Each line is parsed on its own; role-only chunks change nothing."""

    class _NoisyProvider(_FakeProvider):
        """One item holding a role-only chunk, a broken line and a usage line."""

        name = "fake-noisy"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield one batched item."""
            yield (
                'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
                "data: {not json\n"
                'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
            )

    with llm_call_context(user_id=7):
        list(_NoisyProvider().stream(REQUEST))
    (row,) = await _settled(rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 2)
    assert row["usage_metadata"] == {"prompt_tokens": 5, "completion_tokens": 2}


def test_a_coroutine_astream_stays_a_coroutine_function() -> None:
    """The wrapped astream keeps the kind of the method it wraps."""
    assert inspect.iscoroutinefunction(_AwaitableAstreamProvider.astream)
    assert not inspect.iscoroutinefunction(_FakeProvider.astream)


# --- The check at the call (LLM usage accounting spec §4b) ---


class _EagerStreamProvider(_FakeProvider):
    """Sends its request inside stream(), before any item is read, as moonshot, zai and cohere do."""

    name = "fake-eager-stream"
    sent: list[str] = []

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Record that the provider was called."""
        self.sent.append("chat")
        return super().chat(request, timeout=timeout)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:  # type: ignore[override]
        """Record the request when stream() is called, then return the lines."""
        self.sent.append("stream")
        return iter(LINES)

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Record that the provider was called."""
        self.sent.append("achat")
        return {"choices": [{"message": {"content": "hello"}}], "usage": dict(USAGE)}

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """Record when the provider code runs, then yield the lines."""
        self.sent.append("astream")
        for line in LINES:
            yield line


@pytest.fixture
def refused(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Quotas on and user 7 over a limit; returns the users the decision ran for."""
    from tldw_Server_API.app.core.Usage import llm_quota_gate, quota_resolver
    from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

    seen: list[int] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7."""
        seen.append(user_id)
        if user_id == 7:
            return LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)
        return None

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    _EagerStreamProvider.sent = []
    return seen


async def test_a_refused_call_never_reaches_the_provider_and_records_nothing(rows, refused) -> None:
    """chat on a thread, achat, stream on a thread and astream all raise the 402 before the provider runs."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    provider = _EagerStreamProvider()
    with llm_call_context(user_id=7):
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.to_thread(provider.chat, REQUEST)
        with pytest.raises(LLMQuotaExceededError):
            await provider.achat(REQUEST)
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.to_thread(provider.stream, REQUEST)
        astream = provider.astream(REQUEST)
    with pytest.raises(LLMQuotaExceededError):
        await astream.__anext__()
    assert _EagerStreamProvider.sent == []
    assert await _settled(rows) == []
    assert refused == [7, 7, 7, 7]


async def test_an_admitted_user_is_checked_once_per_call_and_recorded(rows, refused) -> None:
    """A user under every limit goes through; nested provider methods are not checked again."""
    with llm_call_context(user_id=8):
        await asyncio.to_thread(_FakeProvider().chat, REQUEST)
        await _FakeProvider().achat(REQUEST)
        assert [line async for line in _FakeProvider().astream(REQUEST)] == LINES
    assert refused == [8, 8, 8]
    assert len(await _settled(rows)) == 3


async def test_a_call_that_records_itself_is_still_checked(rows, refused) -> None:
    """The claims engine records its own rows, and its calls are still refused."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    with llm_call_context(user_id=7, records_itself=True), pytest.raises(LLMQuotaExceededError):
        await asyncio.to_thread(_EagerStreamProvider().chat, REQUEST)
    assert _EagerStreamProvider.sent == []


async def test_an_admitted_request_is_not_refused_part_way(rows, refused) -> None:
    """After a boundary check admits the request, its calls are not checked, even for a refused user."""
    with llm_call_context(user_id=7, prechecked=True):
        await asyncio.to_thread(_FakeProvider().chat, REQUEST)
    assert refused == []
    assert len(await _settled(rows)) == 1


async def test_a_sync_call_on_the_loop_thread_is_recorded_but_not_checked(rows, refused) -> None:
    """Waiting on the accounting loop from its own thread would deadlock: the call goes through and is recorded."""
    with llm_call_context(user_id=7):
        assert _FakeProvider().chat(REQUEST)["usage"] == USAGE
    assert refused == []
    assert len(await _settled(rows)) == 1


async def test_a_stream_started_on_the_loop_thread_is_checked_at_its_first_item(rows, refused) -> None:
    """stream() called on the loop cannot check; the thread that reads the first item does, and records nothing."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    with pytest.raises(LLMQuotaExceededError):
        await asyncio.to_thread(next, stream)
    assert refused == [7]
    assert await _settled(rows) == []


async def test_an_awaitable_astream_is_checked_before_it_is_awaited(rows, refused) -> None:
    """An astream that returns an awaitable stays awaitable, and the refusal comes from awaiting it."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    with llm_call_context(user_id=7):
        pending = _AwaitableAstreamProvider().astream(REQUEST)
    assert inspect.isawaitable(pending)
    with pytest.raises(LLMQuotaExceededError):
        await pending
    assert await _settled(rows) == []


async def test_a_self_recorded_call_is_decided_once_even_when_achat_calls_chat(rows, refused) -> None:
    """The claims engine's calls record themselves; the nested sync chat is not checked a second time."""
    with llm_call_context(user_id=8, records_itself=True):
        await _FakeProvider().achat(REQUEST)
    assert refused == [8]
    assert await _settled(rows) == []


async def test_with_quotas_off_a_provider_call_hands_nothing_off(rows, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default costs a switch read: no hand-off to the accounting loop and no decision."""
    from tldw_Server_API.app.core.Usage import llm_quota_gate

    def _no_hand_off(*args: Any, **kwargs: Any) -> None:
        """Fail the test if the check hands anything to the loop."""
        raise AssertionError("handed off")

    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    monkeypatch.setattr(llm_quota_gate, "get_accounting_loop", _no_hand_off)
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _no_hand_off)
    with llm_call_context(user_id=7):
        assert (await asyncio.to_thread(_FakeProvider().chat, REQUEST))["usage"] == USAGE
        await _FakeProvider().achat(REQUEST)
    assert len(await _settled(rows)) == 2


def test_a_null_prompt_tokens_counts_the_same_streamed_and_not() -> None:
    """A null ``prompt_tokens`` is absent in both modes, so Anthropic-shaped input and cache counts add up in both."""
    usage = {"prompt_tokens": None, "input_tokens": 7, "cache_read_input_tokens": 5, "output_tokens": 2}
    seen = usage_accounting._StreamUsage()
    seen.feed("data: " + json.dumps({"usage": usage}) + "\n\n")
    assert seen.usage is not None
    assert usage_accounting._usage_tokens(usage) == usage_accounting._usage_tokens(seen.usage) == (12, 2)
