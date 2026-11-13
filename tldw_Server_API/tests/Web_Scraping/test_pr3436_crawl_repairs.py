"""Behavioral regressions for the PR3436 source-review repair wave."""

import asyncio
import json
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlInterrupted
from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointStore
from tldw_Server_API.app.core.Web_Scraping.crawl.engine import run_crawl
from tldw_Server_API.app.core.Web_Scraping.crawl.models import (
    CheckpointLimits,
    CheckpointScope,
    CheckpointState,
    FrontierSnapshot,
)
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import BudgetSnapshot, WorkHandle
from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness
from tldw_Server_API.tests.Web_Scraping.test_phase5_default_batch import batch as batch
from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import BASE, entry, page, plan

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize(
    "value, expected",
    [
        (" Yes ", True),
        ("enabled", True),
        ("false", False),
        ("unknown", False),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (None, False),
        ([], False),
        ([1], True),
    ],
)
async def test_shared_bool_coercion_preserves_crawl_robots_configuration(value, expected):
    harness = make_harness(profile="enhanced_sitemap", outcomes=[{"url": BASE}])
    harness.config["web_scraper_respect_robots"] = value
    results = await harness.run()
    assert len(results) == 1 and results[0]["extraction_successful"] is True
    assert harness.policy_calls[0]["respect_robots"] is expected


async def test_crawl_instance_route_agent_is_not_overridden_by_composition_default(batch):
    resolve = batch.scraper._resolve_scrape_plan

    def route(url):
        value, backend, handler = resolve(url)
        return replace(value, extra_headers={"User-Agent": "instance-route-agent"}), backend, handler

    batch.scraper._resolve_scrape_plan = route
    try:
        report = await batch.scraper._crawl_report(
            "https://93.184.216.34/article/one", profile="enhanced_recursive", options={"max_pages": 1}
        )
    finally:
        await batch.scraper.stop()
    assert report.articles[0]["extraction_successful"] is True
    assert batch.calls[0][1]["headers"]["User-Agent"] == "instance-route-agent"


async def test_overflow_checkpoint_reopen_preserves_known_exhaustion_before_any_work(tmp_path):
    resolved = plan(max_pages=3)
    resolved = replace(resolved, limits=replace(resolved.limits, transfer_bytes=100))
    scope = CheckpointScope("owner", "crawl", root=tmp_path)
    path = tmp_path / "overflow.json"
    store = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    attempts = []

    async def overflow(entry):
        attempts.append(entry.url)
        context.charge_bytes(80)
        context.charge_bytes(30)

    stopped = await run_crawl(resolved, context=context, acquire=overflow, checkpoint=store)
    assert stopped.outcome.stop_reason == "budget_exhausted"
    reopened = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
    resumed = CrawlContext(resolved.limits, clock=lambda: 0)
    policy_calls = []

    class Policy:
        async def evaluate(self, entry, *, phase):
            policy_calls.append((entry.url, phase))
            raise AssertionError("known exhaustion must precede policy")

    async def acquire(entry):
        attempts.append(entry.url)
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(entry.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=resumed, acquire=acquire, checkpoint=reopened, policy=Policy())
    assert (report.outcome.stop_reason, resumed.snapshot().transfer_bytes, attempts, policy_calls) == (
        "budget_exhausted",
        80,
        [BASE],
        [],
    )


@pytest.mark.parametrize("reason, complete", [(None, True), ("cancelled", True), ("execution_failed", False)])
async def test_periodic_cancelled_and_unknown_history_remain_resumable(tmp_path, reason, complete):
    resolved = plan(max_pages=1)
    scope = CheckpointScope("owner", "crawl", root=tmp_path)
    store = CheckpointStore.open(tmp_path / "continue.json", scope=scope, limits=CheckpointLimits())
    store.save(
        CheckpointState(
            scope,
            resolved,
            FrontierSnapshot(queued=(entry(BASE),)),
            BudgetSnapshot(1, 0, 0, 0, 80, 1, 100, accounting_complete=complete),
            reason,
            accounting_complete=complete,
        )
    )
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    attempts = []

    async def acquire(item):
        attempts.append(item.url)
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(item.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=store)
    assert (
        report.outcome.state,
        attempts,
        context.snapshot().transfer_bytes,
        context.snapshot().accounting_complete,
    ) == (
        "completed",
        [BASE],
        80,
        complete,
    )


async def test_known_exhaustion_resume_preserves_committed_partial_outcome(tmp_path):
    resolved = plan(max_pages=3)
    scope = CheckpointScope("owner", "crawl", root=tmp_path)
    path = tmp_path / "partial.json"
    store = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
    store.save(
        CheckpointState(
            scope,
            resolved,
            FrontierSnapshot(
                completed=(entry(BASE),), queued=(entry(BASE + "/next", 1),), results=(page(BASE).article,)
            ),
            BudgetSnapshot(1, 1, 0, 0, 80, 2, 100),
            "budget_exhausted",
        )
    )
    store.close()
    reopened = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    dispatched = []

    async def acquire(item):
        dispatched.append(item.url)
        raise AssertionError("known exhaustion must precede dispatch")

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=reopened)
    assert (report.outcome.state, report.outcome.has_partial_results, report.outcome.stop_reason) == (
        "partial",
        True,
        "budget_exhausted",
    )
    assert [item["url"] for item in report.articles] == [BASE] and dispatched == []


@pytest.mark.parametrize("profile", ["enhanced_recursive", "enhanced_sitemap"])
async def test_enhanced_seed_denial_raises_authoritative_report_without_fetch(profile):
    harness = make_harness(profile=profile, outcomes=[{"url": BASE}])
    harness.denied_urls.add(harness.sitemap_url if profile.endswith("sitemap") else harness.seed)
    with pytest.raises(CrawlInterrupted) as caught:
        await harness.run()
    assert (caught.value.report.outcome.stop_reason, harness.fetch_calls, harness.calls) == ("policy_denied", [], [])


async def test_watchlist_consumer_propagates_real_enhanced_denial_without_fallback(monkeypatch):
    from tldw_Server_API.app.core.Watchlists import fetchers
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    harness = make_harness(profile="enhanced_sitemap", outcomes=[{"url": BASE}])
    harness.denied_urls.add(harness.sitemap_url)
    with harness.boundaries():
        monkeypatch.setenv("TEST_MODE", "false")
        monkeypatch.setenv("TLDW_TEST_MODE", "false")
        monkeypatch.setattr(enhanced, "EnhancedWebScraper", lambda: harness.scraper)
        try:
            with pytest.raises(CrawlInterrupted) as caught:
                await fetchers.fetch_site_top_links(harness.sitemap_url, method="sitemap")
            assert (caught.value.report.outcome.stop_reason, harness.fetch_calls, harness.calls) == (
                "policy_denied",
                [],
                [],
            )
        finally:
            await harness.scraper.stop()


async def test_observation_snapshot_redacts_nested_execution_data_and_keeps_owner_and_extras(tmp_path):
    scraper = EnhancedWebScraper()
    scraper._progress["observation"] = {
        "owner_user_id": "owner",
        "current_url": "https://user:password-sentinel@example.com/?token=query-sentinel",
        "processed_urls": 2,
        "crawl_outcome": {"state": "partial", "stop_reason": "budget_exhausted", "has_partial_results": True},
        "extra": {
            "label": "keep",
            "custom_headers": {"Authorization": "header-sentinel"},
            "nested": ["https://example.com/?api_key=nested-sentinel"],
        },
    }
    path = tmp_path / "observation.json"
    await scraper.save_progress("observation", path)
    encoded = path.read_text()
    assert not any(
        secret in encoded for secret in ("password-sentinel", "query-sentinel", "header-sentinel", "nested-sentinel")
    )
    await scraper.load_progress("loaded", path)
    loaded = scraper.get_progress("loaded", owner_user_id="owner")
    assert (loaded["processed_urls"], loaded["extra"]["label"], loaded["crawl_outcome"]["state"]) == (
        2,
        "keep",
        "partial",
    )
    assert scraper.get_progress("loaded", owner_user_id="other") == {}
    assert json.loads(encoded) == loaded


@pytest.mark.parametrize("successes", [(False, False), (True, False), (False, True)])
async def test_sitemap_progress_counts_each_disposition_and_keeps_input_total(successes):
    urls = ["https://example.com/blog/first", "https://example.com/blog/second"]
    harness = make_harness(
        profile="enhanced_sitemap",
        outcomes=[{"url": url, "extraction_successful": success} for url, success in zip(urls, successes, strict=True)],
    )
    snapshots = []
    with harness.boundaries():
        original = harness.scraper._set_progress

        def observe(task_id, **updates):
            original(task_id, **updates)
            snapshots.append(dict(harness.scraper.get_progress(task_id)))

        harness.scraper._set_progress = observe
        try:
            results = await harness.scraper.scrape_sitemap(harness.sitemap_url, task_id="sitemap")
            final = harness.scraper.get_progress("sitemap")
        finally:
            await harness.scraper.stop()
    assert len(results) == 2
    assert (final["processed_urls"], final["total_urls"], final["status"]) == (2, 2, "completed")
    populated = [item for item in snapshots if item.get("total_urls")]
    assert {item["total_urls"] for item in populated} == {2}


@pytest.mark.parametrize("mode", ["denied", "retried", "filtered_capped"])
async def test_sitemap_progress_dispositions_exclude_discovery_and_retries(mode):
    urls = [f"https://example.com/blog/{name}" for name in ("first", "middle", "last")]
    harness = make_harness(
        profile="enhanced_sitemap", outcomes=[{"url": url, "extraction_successful": mode != "retried"} for url in urls]
    )
    if mode == "denied":
        harness.denied_urls.add(urls[1])
    options = {"filter_func": lambda url: url != urls[0], "max_urls": 1} if mode == "filtered_capped" else {}
    with harness.boundaries():
        try:
            result = await harness.scraper.scrape_sitemap(harness.sitemap_url, task_id="sitemap", **options)
            progress = dict(harness.scraper.get_progress("sitemap"))
        finally:
            await harness.scraper.stop()
    count = 1 if mode == "filtered_capped" else 3
    assert (progress["processed_urls"], progress["total_urls"], len(result)) == (count, count, count)
    if mode == "denied":
        assert result[1]["extraction_successful"] is False and result[1]["policy_reason"]
    if mode == "retried":
        assert len(harness.calls) > count and not any(item["extraction_successful"] for item in result)
