"""Accounting writes run on one event loop, from any thread (LLM usage accounting spec §5)."""

import asyncio
import threading

import pytest

from tldw_Server_API.app.core.Usage import llm_accounting

pytestmark = pytest.mark.unit


@pytest.fixture
async def accounting_loop(monkeypatch: pytest.MonkeyPatch) -> asyncio.AbstractEventLoop:
    """Make the test's loop the accounting loop for this test only."""
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    return loop


async def test_a_write_from_the_loop_thread_runs_on_the_loop(accounting_loop) -> None:
    """Scheduling from the accounting loop's own thread runs the coroutine there."""
    ran: list[asyncio.AbstractEventLoop] = []

    async def _write() -> None:
        """Record which loop ran this."""
        ran.append(asyncio.get_running_loop())

    assert llm_accounting.submit_accounting(_write) is True
    await llm_accounting.drain_accounting()
    assert ran == [accounting_loop]


async def test_a_write_from_a_plain_thread_runs_on_the_accounting_loop(accounting_loop) -> None:
    """A thread with no loop hands the write to the accounting loop."""
    ran: list[asyncio.AbstractEventLoop] = []

    async def _write() -> None:
        """Record which loop ran this."""
        ran.append(asyncio.get_running_loop())

    results: list[bool] = []
    worker = threading.Thread(target=lambda: results.append(llm_accounting.submit_accounting(_write)))
    worker.start()
    await asyncio.to_thread(worker.join)
    await llm_accounting.drain_accounting()
    assert results == [True] and ran == [accounting_loop]


async def test_a_write_from_a_thread_with_its_own_loop_still_runs_on_the_accounting_loop(accounting_loop) -> None:
    """A workflow-engine-style thread with its own loop never runs accounting on that loop."""
    ran: list[asyncio.AbstractEventLoop] = []

    async def _write() -> None:
        """Record which loop ran this."""
        ran.append(asyncio.get_running_loop())

    async def _other_loop_main() -> None:
        """Schedule a write from a second event loop."""
        assert llm_accounting.submit_accounting(_write) is True

    worker = threading.Thread(target=lambda: asyncio.run(_other_loop_main()))
    worker.start()
    await asyncio.to_thread(worker.join)
    await llm_accounting.drain_accounting()
    assert ran == [accounting_loop]


def test_with_no_loop_anywhere_nothing_is_scheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no accounting loop and no running loop, the write is skipped and nothing raises."""
    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    called: list[int] = []

    def _make_coro() -> None:
        """Must never be called: nothing is built when there is nowhere to run it."""
        called.append(1)

    assert llm_accounting.submit_accounting(_make_coro) is False  # type: ignore[arg-type]
    assert called == []


async def test_a_failing_write_is_logged_and_never_raises(accounting_loop) -> None:
    """A write that raises does not escape to the caller or to drain."""

    async def _boom() -> None:
        """Fail like a database outage."""
        raise RuntimeError("db down")

    assert llm_accounting.submit_accounting(_boom) is True
    await llm_accounting.drain_accounting()
    assert not llm_accounting._pending


async def test_a_make_coro_that_raises_returns_false_and_raises_nothing(accounting_loop) -> None:
    """A factory that fails is reported as not scheduled, with the loop set."""

    def _make_coro() -> None:
        """Fail while building the coroutine."""
        raise RuntimeError("bad factory")

    assert llm_accounting.submit_accounting(_make_coro) is False  # type: ignore[arg-type]


def test_a_closed_accounting_loop_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured loop that has closed is not used."""
    loop = asyncio.new_event_loop()
    loop.close()
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    assert llm_accounting.get_accounting_loop() is None


def test_a_stopped_accounting_loop_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured loop that is no longer running is not used, so no write can sit forever."""
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0))
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    try:
        assert llm_accounting.get_accounting_loop() is None
        assert llm_accounting.submit_accounting(lambda: asyncio.sleep(0)) is False
    finally:
        loop.close()
