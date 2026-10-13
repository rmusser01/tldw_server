"""The check at the call: who is checked, and where the decision runs (LLM usage accounting spec §4b, §5)."""

import asyncio
import threading
import time
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import llm_accounting, llm_quota_gate, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import (
    accounted_call,
    llm_call_context,
    mark_llm_quota_prechecked,
)
from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root
from tldw_Server_API.app.core.Usage.quota_checks import LLMQuotaRefusal

pytestmark = pytest.mark.unit

REFUSAL = LLMQuotaRefusal("llm_tokens_month", 100, 100, "Monthly LLM token limit reached", 60)


@pytest.fixture
def decisions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, asyncio.AbstractEventLoop]]:
    """Quotas on, no cached limits, and a decision that refuses user 7 and records where it ran."""
    seen: list[tuple[int, asyncio.AbstractEventLoop]] = []

    async def _decide(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Refuse user 7; admit everyone else."""
        seen.append((user_id, asyncio.get_running_loop()))
        return REFUSAL if user_id == 7 else None

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _decide)
    return seen


@pytest.fixture
async def accounting_loop(monkeypatch: pytest.MonkeyPatch) -> asyncio.AbstractEventLoop:
    """Make the test's loop the accounting loop."""
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    return loop


async def test_a_refused_user_gets_the_402_from_a_coroutine_on_the_accounting_loop(decisions, accounting_loop) -> None:
    """Awaited directly on the accounting loop: the refusal is the public 402."""
    with llm_call_context(user_id=7), pytest.raises(LLMQuotaExceededError) as refused:
        await llm_quota_gate.check_llm_call_async()
    assert refused.value.status_code == 402 and refused.value.headers == {"Retry-After": "60"}
    assert refused.value.detail["category"] == "llm_tokens_month"
    assert decisions == [(7, accounting_loop)]


async def test_a_sync_call_on_a_plain_thread_is_decided_on_the_accounting_loop(decisions, accounting_loop) -> None:
    """A thread with no loop hands the decision to the accounting loop and waits for it."""
    outcome: list[object] = []

    def _call() -> None:
        """Run the sync check as a provider call on a pool thread would."""
        with llm_call_context(user_id=7):
            try:
                llm_quota_gate.check_llm_call_sync()
            except LLMQuotaExceededError as error:
                outcome.append(error.status_code)

    await asyncio.to_thread(_call)
    assert outcome == [402] and decisions == [(7, accounting_loop)]


async def test_a_coroutine_on_another_loop_is_decided_on_the_accounting_loop(decisions, accounting_loop) -> None:
    """A workflow-engine-style thread with its own loop never runs the decision on that loop."""
    outcome: list[object] = []

    async def _other_loop_main() -> None:
        """Check from a second event loop."""
        with llm_call_context(user_id=8):
            await llm_quota_gate.check_llm_call_async()
        outcome.append("admitted")

    worker = threading.Thread(target=lambda: asyncio.run(_other_loop_main()))
    worker.start()
    await asyncio.to_thread(worker.join)
    assert outcome == ["admitted"] and decisions == [(8, accounting_loop)]


async def test_a_sync_call_on_the_accounting_loop_thread_is_not_checked(decisions, accounting_loop) -> None:
    """Waiting there would deadlock: the call is admitted unchecked and the caller is told."""
    with llm_call_context(user_id=7):
        assert llm_quota_gate.check_llm_call_sync() is False
    assert decisions == []


async def test_a_check_that_outlasts_the_timeout_admits_the_call(monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """A slow decision lets a sync call through after the timeout, and says it settled."""

    async def _slow(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Take longer than the timeout."""
        await asyncio.sleep(1)
        return REFUSAL

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _slow)
    monkeypatch.setattr(llm_quota_gate, "CHECK_TIMEOUT_SECONDS", 0.05)

    def _call() -> bool:
        """Check from a pool thread."""
        with llm_call_context(user_id=7):
            return llm_quota_gate.check_llm_call_sync()

    started = time.monotonic()
    assert await asyncio.to_thread(_call) is True
    assert time.monotonic() - started < 0.9


async def test_a_failing_decision_admits_the_call(monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """Anything but a refusal lets the call through."""

    async def _broken(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Fail like a broken pool."""
        raise RuntimeError("pool gone")

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _broken)
    with llm_call_context(user_id=7):
        await llm_quota_gate.check_llm_call_async()
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is True


async def test_with_no_accounting_loop_a_coroutine_uses_its_own_loop_and_sync_code_skips(
    decisions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the application lifespan: async checks on the running loop; sync calls go unchecked."""
    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    with llm_call_context(user_id=7):
        with pytest.raises(LLMQuotaExceededError):
            await llm_quota_gate.check_llm_call_async()
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is False
    assert decisions == [(7, asyncio.get_running_loop())]


async def test_no_user_exempt_admitted_and_nested_calls_are_not_checked(decisions, accounting_loop) -> None:
    """Nothing is decided without a user, for an exempt or admitted context, or inside an accounted call."""
    await llm_quota_gate.check_llm_call_async()
    with llm_call_context(user_id=7, exempt=True):
        await llm_quota_gate.check_llm_call_async()
    with llm_call_context(user_id=7, prechecked=True):
        await llm_quota_gate.check_llm_call_async()
    with llm_call_context(user_id=7), accounted_call():
        await llm_quota_gate.check_llm_call_async()
    assert decisions == []


async def test_with_quotas_off_nothing_is_handed_off_or_queried(monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """The common case costs one switch read in the calling thread."""

    def _no_hand_off(*args: Any, **kwargs: Any) -> None:
        """Fail the test if the check hands anything to the loop."""
        raise AssertionError("handed off")

    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    monkeypatch.setattr(llm_quota_gate, "get_accounting_loop", _no_hand_off)
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _no_hand_off)
    with llm_call_context(user_id=7):
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is True
        await llm_quota_gate.check_llm_call_async()


async def test_a_cached_user_with_no_llm_limit_is_not_handed_off(decisions, monkeypatch: pytest.MonkeyPatch, accounting_loop) -> None:
    """Quotas on, limits cached, none of them an LLM key: no hand-off; with an LLM key there is one."""

    async def _limits(user_id: int) -> dict:
        """A user with only a RAG limit, or user 9 with an LLM cost limit."""
        return {"limits.llm_cost_per_day_usd": 1.0} if user_id == 9 else {"limits.rag_queries_per_day": 5}

    monkeypatch.setattr(quota_resolver, "_load_limits", _limits)
    await quota_resolver.user_quota(7, "limits.rag_queries_per_day")
    await quota_resolver.user_quota(9, "limits.rag_queries_per_day")
    with llm_call_context(user_id=7):
        assert await asyncio.to_thread(llm_quota_gate.check_llm_call_sync) is True
    with llm_call_context(user_id=9):
        await asyncio.to_thread(llm_quota_gate.check_llm_call_sync)
    assert decisions == [(9, accounting_loop)]


async def test_a_root_is_never_admitted_by_the_request_that_started_it(decisions, accounting_loop) -> None:
    """A job or workflow started from an admitted request is still checked when it runs."""
    mark_llm_quota_prechecked()
    with llm_call_root(7), pytest.raises(LLMQuotaExceededError):
        await llm_quota_gate.check_llm_call_async()


async def test_a_decision_cancelled_by_the_accounting_loop_admits_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """At shutdown the accounting loop may cancel a handed-off decision; the caller is admitted, not cancelled."""
    other_loop = asyncio.new_event_loop()
    runner = threading.Thread(target=other_loop.run_forever, daemon=True)
    runner.start()

    async def _cancelled(user_id: Any, requested_tokens: int = 1) -> LLMQuotaRefusal | None:
        """Behave like a decision cancelled by a loop that is shutting down."""
        raise asyncio.CancelledError

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    monkeypatch.setattr(llm_quota_gate, "llm_quota_refusal", _cancelled)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", other_loop)
    try:
        with llm_call_context(user_id=7):
            await llm_quota_gate.check_llm_call_async()
    finally:
        other_loop.call_soon_threadsafe(other_loop.stop)
        await asyncio.to_thread(runner.join, 5)
        other_loop.close()
