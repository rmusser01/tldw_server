"""Legacy modulo saves use active owned I/O after atomic disposition."""

import asyncio
import threading
import time
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveredLink, DiscoveryResult, PageData
from tldw_Server_API.app.core.Web_Scraping.crawl import engine
from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointError, CheckpointStore
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CheckpointScope, CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import WorkHandle
from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import BASE, plan

pytestmark = pytest.mark.asyncio


def opened(tmp_path):
    return CheckpointStore.open(
        tmp_path / "progress.json", scope=CheckpointScope("owner", "crawl", root=tmp_path), limits=CheckpointLimits()
    )


def chain(context, successes, *, held=None, entered=None, transport=False):
    """Acquire inert pages, keeping the actual engine/frontier/store unmocked."""
    executed = []

    async def acquire(entry):
        index = 0 if entry.url == BASE else int(entry.url.rsplit("/", 1)[1])
        executed.append(index)
        context.reserve("page_attempts").commit()
        if index == held:
            entered.set()
            await asyncio.Event().wait()
        good = successes[index]
        article = {"url": entry.url, "content": "kept" if good else "", "extraction_successful": good}
        if transport:
            article["error"] = "browser_transport_unavailable"
        links = (DiscoveredLink(BASE + f"/{index + 1}", entry.url),) if index + 1 < len(successes) else ()
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(PageData(article, DiscoveryResult(entry.url, entry.url, links)))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    return acquire, executed


async def drain_background():
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task.get_name() == "crawl-owned-drain"))


async def test_ten_commits_have_real_active_checkpoint_and_resume_unknown_history(tmp_path):
    store = opened(tmp_path)
    resolved = plan("legacy_recursive", max_pages=20, max_depth=20, delay=0)
    context = CrawlContext(resolved.limits, clock=time.monotonic)
    entered = asyncio.Event()
    acquire, executed = chain(context, [True] * 11, held=10, entered=entered)
    task = asyncio.create_task(engine.run_crawl(resolved, context=context, acquire=acquire, checkpoint=store))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert (tmp_path / "progress.json").exists(), "ten committed pages need an active continuation"
        saved = store.load(plan=resolved, scope=store.scope)
        assert saved.pages_scraped == len(saved.frontier.results) == len(saved.frontier.completed) == 10
        assert [entry.url for entry in saved.frontier.queued] == [BASE + "/10"]
        assert saved.frontier.inflight == ()
        assert saved.budget.page_attempts == 10 and saved.budget.discovered_url_bytes == len(BASE)
        assert 0 < saved.budget.remaining_seconds <= resolved.limits.execution_seconds
        assert saved.stop_reason is None and not saved.accounting_complete
        assert context.snapshot().accounting_complete, "periodic uncertainty must not alter the live owner"
        (tmp_path / "resume.json").write_bytes((tmp_path / "progress.json").read_bytes())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await drain_background()
        store.close()
    resumed = CheckpointStore.open(tmp_path / "resume.json", scope=store.scope, limits=store.limits)
    context = CrawlContext(resolved.limits, clock=time.monotonic)
    acquire, executed = chain(context, [True] * 11)
    report = await engine.run_crawl(resolved, context=context, acquire=acquire, checkpoint=resumed)
    assert report.outcome.state == "completed" and len(report.articles) == 11
    assert executed == [10] and context.snapshot().page_attempts == 11
    assert not context.snapshot().accounting_complete


async def test_literal_zero_and_failed_disposition_modulo_cadence(tmp_path, monkeypatch):
    store = opened(tmp_path)
    observed = []
    save = store.save

    def observe(state):
        if state.stop_reason is None:
            observed.append((state.pages_scraped, len(state.frontier.completed), len(state.frontier.queued)))
        save(state)

    monkeypatch.setattr(store, "save", observe)
    resolved = plan("legacy_recursive", max_pages=20, max_depth=20, delay=0)
    context = CrawlContext(resolved.limits, clock=time.monotonic)
    acquire, _ = chain(context, [False, False] + [True] * 10 + [False])
    report = await engine.run_crawl(resolved, context=context, acquire=acquire, checkpoint=store)
    assert report.outcome.state == "completed"
    assert observed == [(0, 1, 1), (0, 2, 1), (10, 12, 1), (10, 13, 0)]


async def test_transport_unavailable_break_skips_periodic_save(tmp_path, monkeypatch):
    store = opened(tmp_path)
    observed = []
    save = store.save

    def observe(state):
        observed.append(state.stop_reason)
        save(state)

    monkeypatch.setattr(store, "save", observe)
    resolved = plan("legacy_recursive", delay=0)
    context = CrawlContext(resolved.limits, clock=time.monotonic)
    acquire, _ = chain(context, [False], transport=True)
    report = await engine.run_crawl(resolved, context=context, acquire=acquire, checkpoint=store)
    assert report.outcome.stop_reason == "transport_unavailable"
    assert observed == ["transport_unavailable"]


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
async def test_held_periodic_write_keeps_writer_and_serializes_final_save(tmp_path, monkeypatch, stop):
    store = opened(tmp_path)
    resolved = replace(
        plan("legacy_recursive", max_pages=20, max_depth=20, delay=0),
        limits=CrawlLimits(execution_seconds=0.15 if stop == "deadline" else 10),
    )
    context = CrawlContext(resolved.limits, clock=time.monotonic)
    entered, release = threading.Event(), threading.Event()
    writes, active = [], []
    save = store.save

    def held(state):
        assert not active, "checkpoint operations must not overlap"
        active.append(True)
        writes.append(state.stop_reason)
        try:
            if state.stop_reason is None:
                entered.set()
                assert release.wait(5)
            save(state)
        finally:
            active.pop()

    monkeypatch.setattr(store, "save", held)
    monkeypatch.setattr(engine, "_CLEANUP_SECONDS", 0.01)
    acquire, executed = chain(context, [True] * 11)
    task = asyncio.create_task(engine.run_crawl(resolved, context=context, acquire=acquire, checkpoint=store))
    try:
        assert await asyncio.to_thread(entered.wait, 2), "periodic write never started"
        if stop == "cancel":
            # Active I/O is not bounded by the much shorter cleanup allowance.
            await asyncio.sleep(0.03)
            assert not task.done()
            task.cancel("held periodic write")
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, 1)
            report = raised.value.report
        else:
            report = await asyncio.wait_for(task, 1)
        assert report.outcome.stop_reason == ("cancelled" if stop == "cancel" else "deadline_exhausted")
        assert len(report.articles) == 10 and executed == list(range(10))
        assert writes == [None] and context.owned_handles()
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            opened(tmp_path)
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(drain_background(), 2)
        store.close()
    assert writes == [None, report.outcome.stop_reason]
    reopened = opened(tmp_path)
    try:
        assert reopened.load(plan=resolved, scope=reopened.scope).stop_reason == report.outcome.stop_reason
    finally:
        reopened.close()


async def test_failed_periodic_save_preserves_prior_valid_file_and_primary_stop(tmp_path, monkeypatch):
    store = opened(tmp_path)
    resolved = plan("legacy_recursive", max_pages=30, max_depth=30, delay=0)
    context = CrawlContext(resolved.limits, clock=time.monotonic)
    save, seen = store.save, []
    entered = asyncio.Event()

    def fail_second(state):
        if state.stop_reason is None:
            seen.append(state.pages_scraped)
            if len(seen) == 2:
                raise OSError("private secondary save failure")
        save(state)

    monkeypatch.setattr(store, "save", fail_second)
    acquire, _ = chain(context, [True] * 21, held=20, entered=entered)
    task = asyncio.create_task(engine.run_crawl(resolved, context=context, acquire=acquire, checkpoint=store))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert seen == [10, 20]
        assert store.load(plan=resolved, scope=store.scope).pages_scraped == 10
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.report.outcome.stop_reason == "cancelled"
        assert "checkpoint_failed" in raised.value.report.diagnostics
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await drain_background()
        store.close()
