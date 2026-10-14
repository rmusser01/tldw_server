"""The LLM call state reaches pool threads and queued processors (LLM usage accounting spec §1)."""

import asyncio
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool, await_bounded_sync_call
from tldw_Server_API.app.core.Chat.request_queue import RequestPriority, RequestQueue
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)

pytestmark = pytest.mark.unit


def _probe() -> tuple[int | None, bool, bool]:
    """What the current thread sees."""
    context = current_llm_call_context()
    return current_llm_user_id(), context.records_itself, context.exempt


async def test_a_pool_thread_sees_the_starter_s_call_state() -> None:
    """A bounded sync call sees the scope user and the flags of the code that started it."""
    pool = BoundedDaemonPool(capacity=2)
    with scoped_context(user_id=6), llm_call_context(records_itself=True):
        marked = await await_bounded_sync_call(_probe, pool=pool, exhaustion_message="full")
    bare = await await_bounded_sync_call(_probe, pool=pool, exhaustion_message="full")
    assert marked == (6, True, False)
    assert bare == (None, False, False)


async def test_a_queued_processor_runs_with_the_enqueuer_s_call_state() -> None:
    """The queue's workers start with no call state; the processor still sees the enqueuer's."""
    queue = RequestQueue(max_queue_size=4, max_concurrent=1)
    await queue.start(num_workers=1)
    try:
        with llm_call_context(user_id=4, records_itself=True):
            marked: asyncio.Future[Any] = await queue.enqueue(
                "req-marked", {}, "client", RequestPriority.NORMAL, processor=_probe
            )
        bare: asyncio.Future[Any] = await queue.enqueue(
            "req-bare", {}, "client", RequestPriority.NORMAL, processor=_probe
        )
        assert await asyncio.wait_for(marked, timeout=5) == (4, True, False)
        assert await asyncio.wait_for(bare, timeout=5) == (None, False, False)
    finally:
        await queue.stop()


async def test_a_queued_stream_factory_runs_with_the_enqueuer_s_call_state() -> None:
    """A streaming request's factory also runs with the enqueuer's state."""
    queue = RequestQueue(max_queue_size=4, max_concurrent=1)
    await queue.start(num_workers=1)
    seen: list[tuple[int | None, bool, bool]] = []

    def _factory() -> Any:
        """Record the state, then return an empty stream."""
        seen.append(_probe())
        return iter(())

    channel: asyncio.Queue[Any] = asyncio.Queue()
    bare_channel: asyncio.Queue[Any] = asyncio.Queue()
    try:
        with llm_call_context(user_id=4, records_itself=True):
            future = await queue.enqueue(
                "req-stream", {}, "client", RequestPriority.NORMAL,
                processor=_factory, streaming=True, stream_channel=channel,
            )
        bare_future = await queue.enqueue(
            "req-stream-bare", {}, "client", RequestPriority.NORMAL,
            processor=_factory, streaming=True, stream_channel=bare_channel,
        )
        await asyncio.wait_for(asyncio.shield(future), timeout=5)
        await asyncio.wait_for(asyncio.shield(bare_future), timeout=5)
    finally:
        await queue.stop()
    assert seen == [(4, True, False), (None, False, False)]


# --- The shared call layer checks first (LLM usage accounting spec §4b) ---


class _ContextRecordingProvider(ChatProvider):
    """Records whether each call ran as already admitted."""

    name = "openai"
    seen: list[bool] = []

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Record the admitted mark and answer."""
        self.seen.append(current_llm_call_context().prechecked)
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield one line. Its body runs lazily, after the call layer has returned, so it records nothing."""
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'


_CALL_ARGS = {
    "api_provider": "openai",
    "messages": [{"role": "user", "content": "hi"}],
    "model": "gpt-4o-mini",
    "api_key": "test-key",
    "app_config": {},
}


@pytest.fixture
async def call_layer(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Route the shared call layer to the recording provider; refuse user 7; return who was decided."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
    from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

    decided: list[int] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7."""
        decided.append(user_id)
        if user_id == 7:
            return LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)
        return None

    adapter = _ContextRecordingProvider()
    _ContextRecordingProvider.seen = []
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _name: adapter))
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _no_write)
    return decided


async def _no_write(**_kwargs: Any) -> None:
    """Write nothing."""


async def test_the_shared_call_layer_refuses_before_its_try_even_with_privacy_safe_errors(
    call_layer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal reaches the caller as itself, not rebuilt as a provider error, and no provider runs."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    monkeypatch.setattr(chat_service, "_privacy_safe_provider_errors_enabled", lambda _request: True)
    with llm_call_context(user_id=7):
        with pytest.raises(LLMQuotaExceededError):
            await chat_service.perform_chat_api_call_async(**_CALL_ARGS)
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.to_thread(lambda: chat_service.perform_chat_api_call(**_CALL_ARGS))
    assert _ContextRecordingProvider.seen == []
    assert call_layer == [7, 7]


async def test_an_admitted_call_runs_its_adapter_as_admitted_and_is_decided_once(call_layer) -> None:
    """The check at the top settles it: the adapter calls are not checked again, streams included."""
    from tldw_Server_API.app.core.Chat import chat_service

    with llm_call_context(user_id=8):
        await chat_service.perform_chat_api_call_async(**_CALL_ARGS)
        stream = await chat_service.perform_chat_api_call_async(**_CALL_ARGS, streaming=True)
        assert [line async for line in stream]
        await asyncio.to_thread(lambda: chat_service.perform_chat_api_call(**_CALL_ARGS))
    assert _ContextRecordingProvider.seen == [True, True]
    assert call_layer == [8, 8, 8]
    assert current_llm_call_context().prechecked is False


async def test_a_sync_call_on_the_loop_thread_leaves_its_adapter_call_unmarked(call_layer) -> None:
    """Where the top check cannot run, the adapter call is not marked admitted, so it can still be checked."""
    from tldw_Server_API.app.core.Chat import chat_service

    with llm_call_context(user_id=8):
        chat_service.perform_chat_api_call(**_CALL_ARGS)
    assert _ContextRecordingProvider.seen == [False]
    assert call_layer == []
