"""Bounded continuation, legacy migration, atomic writes and engine finalization."""

import ast
import asyncio
import inspect
import json
import os
import tempfile
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome
from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointError, CheckpointStore
from tldw_Server_API.app.core.Web_Scraping.crawl.engine import run_crawl
from tldw_Server_API.app.core.Web_Scraping.crawl.frontier import Frontier
from tldw_Server_API.app.core.Web_Scraping.crawl.models import (
    CheckpointLimits,
    CheckpointScope,
    CheckpointState,
    FrontierSnapshot,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import BudgetSnapshot, ParentStop, WorkHandle
from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import BASE, entry, page, plan

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path, request):
    local = getattr(request, "param", False)
    scope = CheckpointScope("owner", "crawl", root=None if local else tmp_path, trusted_local_path=local)
    value = CheckpointStore.open(tmp_path / "progress.json", scope=scope, limits=CheckpointLimits())
    yield value
    value.close()


def state(store, resolved=None):
    resolved = resolved or plan()
    return CheckpointState(
        store.scope,
        resolved,
        FrontierSnapshot(queued=(entry(BASE + "/second", 1),), inflight=(entry(BASE),)),
        BudgetSnapshot(1, 2, 3, 4, 5, 6, 123.0, discovered_url_bytes=789),
        "cancelled",
    )


def test_roundtrip_preserves_spent_budget_and_order(store):
    expected = state(store)
    store.save(expected)
    actual = store.load(plan=expected.plan, scope=store.scope)
    assert actual == expected
    frontier = Frontier.from_snapshot(actual.plan, actual.frontier)
    frontier.restore_unfinished()
    assert [item.url for item in frontier.snapshot().queued] == [BASE, BASE + "/second"]


def test_legacy_migration_preserves_articles_queue_pages_and_unknown_history(store, tmp_path):
    articles = [page(BASE + "/b").article, page(BASE + "/a").article]
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "visited": [BASE + "/a", BASE + "/b", BASE + "/failed"],
                "to_visit": [[BASE + "/first", 1], [BASE + "/second", 2]],
                "scraped_articles": articles,
                "pages_scraped": 2,
            }
        )
    )
    resolved = plan("legacy_recursive", delay=0)
    actual = store.load(plan=resolved, scope=store.scope)
    assert actual.frontier.results == tuple(articles)
    assert actual.frontier.visited == (BASE + "/a", BASE + "/b", BASE + "/failed")
    assert [item.url for item in actual.frontier.queued] == [BASE + "/first", BASE + "/second"]
    assert actual.pages_scraped == 2
    assert not actual.accounting_complete
    assert actual.budget.page_attempts == actual.budget.discovered_candidates == 0
    assert actual.budget.remaining_seconds == resolved.limits.execution_seconds
    assert not Frontier.from_snapshot(resolved, actual.frontier).admit(entry(BASE + "/failed", 99))
    store.save(actual)
    assert store.load(plan=resolved, scope=store.scope) == actual


@pytest.mark.parametrize("failure", ["replace", "serialization", "overflow"])
def test_failed_save_preserves_previous_file_and_removes_unique_temp(store, tmp_path, monkeypatch, failure):
    expected = state(store)
    store.save(expected)
    previous = (tmp_path / "progress.json").read_bytes()
    if failure == "replace":

        def fail(*args, **kwargs):
            raise OSError("private/path secret")

        monkeypatch.setattr("os.replace", fail)
        candidate = expected
    else:
        result = {"content": float("nan") if failure == "serialization" else "x" * 8000}
        candidate = replace(expected, frontier=FrontierSnapshot(completed=(entry(BASE),), results=(result,)))
        if failure == "overflow":
            store.limits = replace(store.limits, max_file_bytes=len(previous))
    with pytest.raises(CheckpointError, match="^checkpoint_failed$"):
        store.save(candidate)
    assert (tmp_path / "progress.json").read_bytes() == previous
    assert sorted(item.name for item in tmp_path.iterdir()) == ["progress.json", "progress.json.lock"]


def test_known_exhaustion_and_remaining_active_time_are_not_refreshed(store):
    expected = state(store)
    expected = replace(expected, budget=replace(expected.budget, page_attempts=1000, remaining_seconds=0))
    store.save(expected)
    actual = store.load(plan=expected.plan, scope=store.scope)
    context = CrawlContext(expected.plan.limits, clock=lambda: 99999)
    context.restore_budget(actual.budget, accounting_complete=actual.accounting_complete)
    with pytest.raises(ParentStop, match="deadline_exhausted"):
        context.check()
    assert context.snapshot().page_attempts == 1000


@pytest.mark.asyncio
async def test_engine_resume_readmits_unfinished_urls_and_deduplicates_committed(store):
    resolved = plan(max_pages=3)
    snapshot = FrontierSnapshot(
        queued=(entry(BASE + "/second", 2),),
        inflight=(entry(BASE + "/first", 1),),
        completed=(entry(BASE),),
        results=(page(BASE).article,),
    )
    store.save(replace(state(store, resolved), frontier=snapshot))
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    visited, executed = [], []

    class Policy:
        async def evaluate(self, item, *, phase):
            from tldw_Server_API.app.core.Web_Scraping.crawl.models import TraversalDecision

            visited.append((item.url, phase))
            return TraversalDecision()

    async def acquire(item):
        executed.append(item.url)
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=store, policy=Policy())
    assert executed == [BASE + "/first", BASE + "/second"]
    assert visited == [(url, phase) for url in executed for phase in ("discovery", "visit")]
    assert [item["url"] for item in report.articles] == [BASE, *executed]
    assert report.outcome.state == "completed"
    assert context.snapshot().discovered_url_bytes == 789


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["load", "save", "clear"])
async def test_actual_filesystem_call_retains_writer_after_caller_cancel(store, tmp_path, monkeypatch, operation):
    resolved = plan()
    entered, release = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    started, closed = loop.create_future(), loop.create_future()
    original = getattr(store, operation)
    original_close = store.close

    def held(*args, **kwargs):
        entered.set()
        loop.call_soon_threadsafe(started.set_result, None)
        if not release.wait(5):
            raise RuntimeError("test barrier expired")
        return original(*args, **kwargs)

    def close():
        original_close()
        if not closed.done():
            loop.call_soon_threadsafe(closed.set_result, None)

    monkeypatch.setattr(store, operation, held)
    monkeypatch.setattr(store, "close", close)
    context = CrawlContext(resolved.limits, clock=lambda: 0)

    async def acquire(item):
        if operation == "save":
            raise ParentStop("budget_exhausted")
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    task = asyncio.create_task(run_crawl(resolved, context=context, acquire=acquire, checkpoint=store))
    try:
        await asyncio.wait_for(asyncio.shield(started), 3)
        assert entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
        assert context.owned_handles()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(closed), 3)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["normal", "budget", "cancel"])
async def test_finalization_failure_precedence(store, monkeypatch, primary):
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)

    def fail(*args, **kwargs):
        raise OSError("secret checkpoint path")

    monkeypatch.setattr(store, "clear" if primary == "normal" else "save", fail)

    async def acquire(item):
        if primary == "cancel":
            raise asyncio.CancelledError
        if primary == "budget":
            raise ParentStop("budget_exhausted")
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    if primary == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run_crawl(resolved, context=context, acquire=acquire, checkpoint=store)
        assert context.publish(CrawlOutcome("cancelled", "cancelled", False)).stop_reason == "cancelled"
    else:
        report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=store)
        assert report.outcome.stop_reason == ("checkpoint_failed" if primary == "normal" else "budget_exhausted")
        assert report.diagnostics == ("checkpoint_failed",)


@pytest.mark.asyncio
@pytest.mark.parametrize("filesystem", ["open", "replace", "unlink"])
async def test_cancellation_during_actual_filesystem_syscall_holds_lock_and_restores_continuation(
    store,
    tmp_path,
    monkeypatch,
    filesystem,
):
    resolved = plan()
    store.save(state(store, resolved))
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    loop = asyncio.get_running_loop()
    started, closed = loop.create_future(), loop.create_future()
    release = threading.Event()
    original = getattr(os, filesystem)
    original_close = store.close
    intercepted = False

    def held(*args, **kwargs):
        nonlocal intercepted
        target = str(args[0])
        matches = filesystem != "open" or (target == "progress.json" and args[1] & os.O_ACCMODE == os.O_RDONLY)
        if matches and not intercepted:
            intercepted = True
            assert context.owned_handles(), "I/O dispatch preceded physical registration"
            loop.call_soon_threadsafe(started.set_result, None)
            if not release.wait(5):
                raise RuntimeError("test barrier expired")
        return original(*args, **kwargs)

    def close():
        original_close()
        if not closed.done():
            loop.call_soon_threadsafe(closed.set_result, None)

    monkeypatch.setattr(os, filesystem, held)
    monkeypatch.setattr(store, "close", close)

    async def acquire(item):
        if filesystem == "replace":
            raise ParentStop("budget_exhausted")
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    task = asyncio.create_task(run_crawl(resolved, context=context, acquire=acquire, checkpoint=store))
    try:
        await asyncio.wait_for(asyncio.shield(started), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(closed), 3)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    reopened = CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    try:
        loaded = reopened.load(plan=resolved, scope=reopened.scope)
        assert loaded is not None
        if filesystem == "replace":
            assert loaded.stop_reason == "cancelled"
            assert loaded.budget.page_attempts == 1
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_eager_task_factory_registers_checkpoint_before_filesystem_dispatch(store, monkeypatch):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    context = CrawlContext(plan().limits, clock=lambda: 0)
    original = store.load

    def load(**kwargs):
        assert context.owned_handles()
        return original(**kwargs)

    monkeypatch.setattr(store, "load", load)
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        report = await run_crawl(
            plan(), context=context, acquire=Mock(side_effect=ParentStop("budget_exhausted")), checkpoint=store
        )
        assert report.outcome.stop_reason == "budget_exhausted"
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
async def test_interrupted_transport_retains_file_and_writer_until_actual_page_drain(store, tmp_path):
    loop = asyncio.get_running_loop()
    drained = loop.create_future()
    context = CrawlContext(plan().limits, clock=lambda: 0)
    closed = loop.create_future()
    original_close = store.close

    def close():
        original_close()
        loop.call_soon_threadsafe(closed.set_result, None)

    store.close = close

    async def acquire(item):
        result = loop.create_future()
        result.set_result(
            replace(page(item.url), article={"error": "browser_transport_unavailable", "extraction_successful": False})
        )
        return WorkHandle(result, drained, lambda: None)

    try:
        report = await run_crawl(plan(), context=context, acquire=acquire, checkpoint=store)
        assert report.outcome.stop_reason == "transport_unavailable"
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
        assert (tmp_path / "progress.json").exists()
        assert not json.loads((tmp_path / "progress.json").read_text())["accounting_complete"]
    finally:
        drained.set_result(None)
        await asyncio.wait_for(asyncio.shield(closed), 3)
        store.close = original_close


@pytest.mark.asyncio
async def test_primary_close_failure_prevents_success_publication(store, monkeypatch):
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    original_close = store.close

    def fail():
        original_close()
        raise OSError("private close error")

    monkeypatch.setattr(store, "close", fail)

    async def acquire(item):
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=store)
    monkeypatch.setattr(store, "close", original_close)
    assert report.outcome.stop_reason == "checkpoint_failed"


def test_atomic_encoder_byte_ceiling_accepts_exact_size_and_rejects_one_less(store, tmp_path):
    candidate = state(store)
    store.save(candidate)
    size = (tmp_path / "progress.json").stat().st_size
    store.limits = replace(store.limits, max_file_bytes=size)
    store.save(candidate)
    assert store.load(plan=candidate.plan, scope=store.scope) == candidate
    store.limits = replace(store.limits, max_file_bytes=size - 1)
    with pytest.raises(CheckpointError):
        store.save(candidate)
    assert (tmp_path / "progress.json").stat().st_size == size


def test_legacy_duplicate_queue_entries_skip_visited_without_reordering_remaining(store, tmp_path):
    data = {
        "visited": [BASE],
        "to_visit": [[BASE + "/a", 1], [BASE, 0], [BASE + "/a", 2], [BASE + "/b", 1]],
        "scraped_articles": [page(BASE).article],
        "pages_scraped": 1,
    }
    (tmp_path / "progress.json").write_text(json.dumps(data))
    migrated = store.load(plan=plan("legacy_recursive"), scope=store.scope)
    assert [(item.url, item.depth) for item in migrated.frontier.queued] == [(BASE + "/a", 1), (BASE + "/b", 1)]


def test_legacy_cannot_invent_lost_inflight_or_historical_consumption(store, tmp_path):
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "visited": [BASE],
                "to_visit": [],
                "scraped_articles": [],
                "pages_scraped": 0,
            }
        )
    )
    migrated = store.load(plan=plan("legacy_recursive"), scope=store.scope)
    assert migrated.frontier.queued == migrated.frontier.inflight == migrated.frontier.results == ()
    assert migrated.frontier.visited == (BASE,)
    assert migrated.budget.page_attempts == 0 and not migrated.accounting_complete


def test_engine_functions_remain_within_new_function_size_ceiling():
    from tldw_Server_API.app.core.Web_Scraping.crawl import engine

    tree = ast.parse(inspect.getsource(engine))
    sizes = {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert all(size <= 300 for size in sizes.values()), sizes


@settings(max_examples=20, deadline=None)
@given(st.lists(st.text(alphabet="abcxyz012", min_size=1, max_size=8), unique=True, max_size=6))
def test_roundtrip_property_preserves_ordered_url_identities_without_refunding_usage(slugs):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        value = CheckpointStore.open(
            root / "progress.json", scope=CheckpointScope("owner", "crawl", root=root), limits=CheckpointLimits()
        )
        try:
            expected = replace(
                state(value),
                frontier=FrontierSnapshot(
                    queued=tuple(entry(BASE + "/" + slug, index) for index, slug in enumerate(slugs))
                ),
            )
            value.save(expected)
            assert value.load(plan=expected.plan, scope=value.scope) == expected
        finally:
            value.close()


@pytest.mark.asyncio
async def test_failed_clear_and_secondary_save_always_release_writer(store, tmp_path, monkeypatch):
    original_close = store.close
    loop = asyncio.get_running_loop()
    closed = loop.create_future()
    context = CrawlContext(plan().limits, clock=lambda: 0)

    def failed(*args):
        raise OSError("private filesystem error")

    def close():
        original_close()
        if not closed.done():
            loop.call_soon_threadsafe(closed.set_result, None)

    monkeypatch.setattr(store, "clear", failed)
    monkeypatch.setattr(store, "save", failed)
    monkeypatch.setattr(store, "close", close)

    async def acquire(item):
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(plan(), context=context, acquire=acquire, checkpoint=store)
    assert report.outcome.stop_reason == "checkpoint_failed"
    await asyncio.wait_for(asyncio.shield(closed), 1)
    await asyncio.gather(*(asyncio.shield(item.drained) for item in context.owned_handles()))
    other = CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    other.close()


def test_safe_legacy_transport_failure_code_survives_storage(store):
    candidate = replace(
        state(store),
        frontier=FrontierSnapshot(
            completed=(entry(BASE),),
            results=(
                {
                    "url": BASE,
                    "error": "browser_transport_unavailable",
                    "extraction_successful": False,
                },
            ),
        ),
    )
    store.save(candidate)
    actual = store.load(plan=candidate.plan, scope=store.scope)
    assert actual.frontier.results[0]["error"] == "browser_transport_unavailable"


@pytest.mark.asyncio
async def test_checkpoint_task_canceled_before_first_step_still_physically_drains(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.crawl import engine

    original = asyncio.create_task

    def create(coroutine, **kwargs):
        task = original(coroutine, **kwargs)
        task.cancel()
        return task

    context = CrawlContext(plan().limits, clock=lambda: 0)
    function = Mock()
    monkeypatch.setattr(asyncio, "create_task", create)
    handle = engine._checkpoint_operation(context, function)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(handle.result), 1)
    await asyncio.wait_for(asyncio.shield(handle.drained), 1)
    assert not function.called


@pytest.mark.asyncio
async def test_parent_cancel_then_held_write_is_registered_and_still_runs(store, tmp_path, monkeypatch):
    """Late cleanup attach must not turn parent cancellation into canceled I/O."""
    loop = asyncio.get_running_loop()
    page_started, write_started = loop.create_future(), loop.create_future()
    release = threading.Event()
    context = CrawlContext(plan().limits, clock=lambda: 0)
    original_replace = os.replace
    canceled_pages = []

    def held_replace(*args, **kwargs):
        assert context.owned_handles()
        loop.call_soon_threadsafe(write_started.set_result, None)
        if not release.wait(5):
            raise RuntimeError("test barrier expired")
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(os, "replace", held_replace)

    async def acquire(item):
        page_started.set_result(None)
        result, drained = loop.create_future(), loop.create_future()
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: canceled_pages.append(True))

    task = asyncio.create_task(run_crawl(plan(), context=context, acquire=acquire, checkpoint=store))
    try:
        await asyncio.wait_for(asyncio.shield(page_started), 3)
        task.cancel()
        await asyncio.wait_for(asyncio.shield(write_started), 3)
        with pytest.raises(asyncio.CancelledError):
            context.check()
        assert not task.done()
        assert context.owned_handles()
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        await asyncio.gather(*(asyncio.shield(item.drained) for item in context.owned_handles()))
    raw = json.loads((tmp_path / "progress.json").read_text())
    assert raw["stop_reason"] == "cancelled"
    assert [item["url"] for item in raw["frontier"]["inflight"]] == [BASE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,primary", [("load", "normal"), ("clear", "normal"), ("save", "budget"), ("close", "normal")]
)
async def test_port_internal_cancellation_is_failure_not_caller_cancellation(store, monkeypatch, operation, primary):
    context = CrawlContext(plan().limits, clock=lambda: 0)
    original_close = store.close

    def internal_cancel(*args, **kwargs):
        if operation == "close":
            original_close()
        raise asyncio.CancelledError("private port cancellation")

    monkeypatch.setattr(store, operation, internal_cancel)

    async def acquire(item):
        if primary == "budget":
            raise ParentStop("budget_exhausted")
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    try:
        report = await run_crawl(plan(), context=context, acquire=acquire, checkpoint=store)
        assert report.outcome.stop_reason == ("budget_exhausted" if primary == "budget" else "checkpoint_failed")
        assert report.outcome.state != "cancelled"
        await asyncio.gather(*(asyncio.shield(item.drained) for item in context.owned_handles()))
    finally:
        monkeypatch.setattr(store, "close", original_close)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["normal", "cancelled", "budget_exhausted", "deadline_exhausted"])
async def test_cleanup_stages_share_one_deadline_and_retain_actual_close_owner(store, tmp_path, monkeypatch, primary):
    """Advance virtual time at event barriers; never assert scheduler wall time."""
    from tldw_Server_API.app.core.Web_Scraping.crawl import engine

    loop = asyncio.get_running_loop()
    real_wait_for = asyncio.wait_for
    now, timeouts = [100.0], []
    first_wait, close_started, close_exited, expired = (loop.create_future() for _ in range(4))
    release_close = threading.Event()
    original_save, original_clear, original_close = store.save, store.clear, store.close
    active_clock = [0.0]
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: active_clock[0])
    cancellation = asyncio.CancelledError("caller") if primary == "cancelled" else None
    if cancellation is not None:
        context.cancel()
    elif primary == "deadline_exhausted":
        active_clock[0] = resolved.limits.execution_seconds
        with pytest.raises(ParentStop, match=primary):
            context.check()
    elif primary == "budget_exhausted":
        with pytest.raises(ParentStop, match=primary):
            context.reserve("page_attempts", resolved.limits.page_attempts + 1)

    async def virtual_wait_for(awaitable, timeout):
        timeouts.append((now[0], timeout))
        if not first_wait.done():
            first_wait.set_result(None)
        future = asyncio.ensure_future(awaitable)
        done, _ = await asyncio.wait((future, expired), return_when=asyncio.FIRST_COMPLETED)
        if future in done:
            return await future
        # Expire exactly at the supplied virtual deadline, canceling only the
        # shield/gather proxy just as wait_for would, never a physical future.
        assert now[0] >= timeouts[-1][0] + timeout
        future.cancel()
        raise TimeoutError

    def clear():
        original_clear()
        now[0] += 0.75

    def save(value):
        original_save(value)
        now[0] += 1.5

    def close():
        loop.call_soon_threadsafe(close_started.set_result, None)
        if not release_close.wait(5):
            raise RuntimeError("test close barrier expired")
        original_close()
        loop.call_soon_threadsafe(close_exited.set_result, None)

    class PrivateGraph:
        pass

    private_graph = PrivateGraph()
    retained = weakref.ref(private_graph)
    monkeypatch.setattr(engine, "monotonic", lambda: now[0], raising=False)
    monkeypatch.setattr(asyncio, "wait_for", virtual_wait_for)
    monkeypatch.setattr(store, "clear", clear)
    monkeypatch.setattr(store, "save", save)
    monkeypatch.setattr(store, "close", close)
    result, drained = loop.create_future(), loop.create_future()
    result.set_result(None)
    handles = []
    if primary == "normal":
        handle = WorkHandle(result, drained, lambda: None)
        context.attach(handle)
        handles.append(handle)
    task = asyncio.create_task(
        engine._finalize_crawl(
            resolved,
            context=context,
            frontier=Frontier(resolved),
            handles=handles,
            checkpoint=store,
            loading=False,
            count=0,
            reason="frontier_exhausted" if primary == "normal" else primary,
            cancellation=cancellation,
            diagnostics=[],
            owned=(private_graph,),
        )
    )
    del private_graph
    try:
        if primary == "normal":
            await real_wait_for(asyncio.shield(first_wait), 3)
            now[0] += 0.75
            drained.set_result(None)
        await real_wait_for(asyncio.shield(close_started), 3)
        assert timeouts[-1][1] == pytest.approx(0.5)
        assert all(start + timeout == pytest.approx(102.0) for start, timeout in timeouts)
        now[0] = 102.0
        expired.set_result(None)
        report, returned_cancellation = await real_wait_for(asyncio.shield(task), 3)
        assert returned_cancellation is cancellation
        assert report.outcome.stop_reason == ("checkpoint_failed" if primary == "normal" else primary)
        assert report.diagnostics == ("checkpoint_failed",)
        assert context.owned_handles() and retained() is not None
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    finally:
        if not drained.done():
            drained.set_result(None)
        release_close.set()
        await real_wait_for(asyncio.shield(close_exited), 3)
        await real_wait_for(asyncio.shield(task), 3)
        await real_wait_for(asyncio.gather(*(asyncio.shield(item.drained) for item in context.owned_handles())), 3)
        monkeypatch.setattr(store, "close", original_close)
    other = CheckpointStore.open(tmp_path / "progress.json", scope=store.scope, limits=store.limits)
    other.close()


@pytest.mark.asyncio
async def test_checkpoint_load_stays_under_active_context_not_cleanup_clock(store, monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.crawl import engine

    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    bounded_calls = []
    original = engine._bounded

    def cleanup_clock():
        pytest.fail("load consulted the finalization clock")

    async def bounded(awaitable, current):
        bounded_calls.append(current)
        return await original(awaitable, current)

    monkeypatch.setattr(engine, "monotonic", cleanup_clock, raising=False)
    monkeypatch.setattr(engine, "_bounded", bounded)
    assert (
        await engine._checkpoint_call(context, [], store.load, plan=resolved, scope=store.scope, loading=True) is None
    )
    assert bounded_calls and all(current is context for current in bounded_calls)
