"""Private production composition parity; no external network or LLM work."""

import asyncio

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult, PageData
from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlHooks, CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import WorkHandle
from tldw_Server_API.tests.Web_Scraping.phase5_fakes import Phase5Harness

pytestmark = pytest.mark.asyncio
BASE = "https://example.com/blog/start"


async def run_report(
    monkeypatch,
    profile,
    outcomes,
    *,
    queued=False,
    production_queue=False,
    page_drains=None,
    config=None,
    limits=None,
    hooks=None,
    seeds=None,
    denied=(),
    sitemap_kind="namespaced",
    base_url=None,
    runtime_fetch=False,
    real_policy=False,
    transport_allowed=True,
    **options,
):
    fixture = Phase5Harness(profile=profile, outcomes=outcomes)
    fixture.denied_urls.update(denied)
    fixture.sitemap_kind = sitemap_kind
    credentials = {
        **fixture.credentials,
        **{key: options.pop(key) for key in ("user_agent", "custom_headers", "custom_cookies") if key in options},
    }
    submitted, executed = [], []
    queue = None
    production = None
    from tldw_Server_API.app.core.Web_Scraping.browser_transport import BrowserTransportDecision

    monkeypatch.setattr(
        crawl,
        "resolve_browser_transport_decision",
        lambda *_args, **_kwargs: BrowserTransportDecision(
            transport_allowed,
            "url_guarded" if transport_allowed else "disabled",
            "url_guarded" if transport_allowed else "disabled",
            False,
            "browser_transport_allowed_legacy" if transport_allowed else "browser_transport_disabled",
        ),
        raising=False,
    )

    async def policy(url, **kwargs):
        return fixture._policy(url, **kwargs)

    async def fetch(_dependencies, request):
        response = fixture._response(request.url, headers=dict(request.headers), cookies=dict(request.cookies))
        response.close()
        return FetchResponse(url=request.url, status=200, text=response.text, headers={}, backend="httpx")

    def start(url, cookies, *, controls, request_context, discover, **kwargs):
        executed.append(url)
        fixture.calls.append({"url": url, "kwargs": {"custom_cookies": cookies, **kwargs}})

        async def execute():
            controls.reserve("page_attempts").commit()
            article = fixture.article(url)
            if fixture.outcomes.get(url, {}).get("transport_error"):
                article.update(error="browser_transport_unavailable", extraction_successful=False)
            discovery = DiscoveryResult(url, url, ())
            if (not kwargs.get("discovery_on_success") or article.get("extraction_successful")) and (
                kwargs.get("discovery_gate") is None or kwargs["discovery_gate"](article)
            ):
                discovery = discover(fixture._html(url), url, url)
            return PageData(article, discovery)

        task = asyncio.create_task(execute())
        drained = asyncio.get_running_loop().create_future()
        if page_drains is None:
            task.add_done_callback(lambda _done: drained.set_result(None))
        else:
            page_drains.put_nowait((url, drained))
        return WorkHandle(task, drained, task.cancel)

    if real_policy:
        from types import SimpleNamespace

        from tldw_Server_API.app.core.Web_Scraping import outbound_policy

        monkeypatch.setattr(outbound_policy, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
        monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    else:
        monkeypatch.setattr(crawl, "decide_web_outbound_policy", policy)
    if runtime_fetch:
        from tldw_Server_API.app.core.Web_Scraping.runtime import fetch as fetch_module

        def http_fetch(url, **kwargs):
            if url.endswith("/robots.txt"):
                from tldw_Server_API.tests.Web_Scraping.phase5_fakes import FakeResponse

                fixture.fetch_calls.append(
                    {"url": url, "headers": kwargs.get("headers"), "cookies": kwargs.get("cookies")}
                )
                response = FakeResponse("User-agent: *\nDisallow: /blog/blocked\n")
            else:
                response = fixture._response(url, headers=kwargs.get("headers"), cookies=kwargs.get("cookies"))
            kwargs["on_response_chunk"](len(response.text.encode()))
            response.close()
            return {"url": url, "status": 200, "text": response.text, "headers": {}, "backend": "httpx"}

        monkeypatch.setattr(fetch_module, "http_fetch", http_fetch)
    else:
        monkeypatch.setattr(crawl.article, "_fetch_response", fetch)
    monkeypatch.setattr(crawl.article, "start_page", start)
    original = crawl.run_crawl

    async def engine(plan, *, acquire, **kwargs):
        nonlocal queue
        from tldw_Server_API.app.core.Web_Scraping.jobs.models import ScrapingJob
        from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue

        operations = {}

        def execute(payload, controls):
            entry = operations.pop(payload.job_id)
            loop = asyncio.get_running_loop()
            drained = loop.create_future()
            inner = None

            async def run():
                nonlocal inner
                try:
                    inner = await acquire(entry)
                    return await asyncio.shield(inner.result)
                finally:
                    if inner is None:
                        drained.set_result(None)
                    else:

                        async def drain():
                            await asyncio.shield(inner.drained)
                            drained.set_result(None)

                        asyncio.create_task(drain())

            task = asyncio.create_task(run())

            def cancel():
                if inner:
                    inner.cancel()
                task.cancel()

            return WorkHandle(task, drained, cancel)

        if queued:
            queue = ProcessLocalJobQueue(execute=execute, max_concurrent=1)
            await queue.start()

        async def accepted(entry):
            submitted.append(entry.url)
            if not queued:
                return await acquire(entry)
            # A test-only adapter around the real queue. No callable is placed in
            # a serialized job, options or credentials.
            job_id = f"crawl-{entry.sequence}"
            operations[job_id] = entry
            return await queue.submit_page(
                ScrapingJob(job_id, entry.url, "auto", max_retries=0), controls=kwargs["context"]
            )

        try:
            report = await original(plan, acquire=accepted, **kwargs)
            fixture.budget = kwargs["context"].snapshot()
            return report
        finally:
            if queue:
                await queue.stop()

    monkeypatch.setattr(crawl, "run_crawl", engine)
    if production_queue:
        from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue

        def forbidden_default(*_args):
            raise AssertionError("crawl lost its private execution binding")

        production = ProcessLocalJobQueue(execute=forbidden_default, completed_retention=0)
        await production.start()
    try:
        report = await crawl.execute_crawl(
            CrawlRequest(fixture.sitemap_url if profile.endswith("sitemap") else base_url or fixture.seed),
            profile=profile,
            options=options,
            supplied=frozenset(options),
            config=config or {},
            credentials=credentials,
            limits=limits or CrawlLimits(),
            hooks=hooks,
            seeds=seeds,
            **({"queue": production} if production_queue else {}),
        )
    finally:
        if production:
            await production.stop()
    return report, fixture, submitted, executed


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize(
    "profile",
    [
        "legacy_recursive",
        "enhanced_recursive",
        "legacy_sitemap",
        "legacy_url_level",
        "enhanced_sitemap",
        "enhanced_url_level",
        "enhanced_batch",
    ],
)
async def test_each_profile_report_retains_payload_and_executes_only_expected_url(monkeypatch, profile, queued):
    report, fixture, submitted, executed = await run_report(
        monkeypatch, profile, [{"url": BASE}], queued=queued, max_pages=3, max_depth=2, delay=0, level=3
    )
    assert [article["url"] for article in report.articles] == [BASE]
    assert report.articles[0]["custom_extra"] == {"rank": 7}
    assert report.articles[0]["metadata"]["nested"] == {"labels": ["baseline"]}
    assert executed == [BASE]
    assert submitted == ([fixture.sitemap_url, BASE] if profile.endswith("sitemap") else [BASE])


@pytest.mark.parametrize("profile", ["legacy_recursive", "enhanced_recursive"])
@pytest.mark.parametrize("boundary", ["page", "depth"])
async def test_unused_discovery_never_spends_candidate_budget(monkeypatch, profile, boundary):
    options = {"max_pages": 1 if boundary == "page" else 3, "max_depth": 0 if boundary == "depth" else 3, "delay": 0}
    report, _, _, executed = await run_report(
        monkeypatch,
        profile,
        [{"url": BASE, "links": ["/blog/child"] * 10}],
        limits=CrawlLimits(discovered_candidates=1),
        **options,
    )
    assert report.outcome.state == "completed"
    assert executed == [BASE]


@pytest.mark.parametrize("queued", [False, True])
async def test_lazy_batch_keeps_repeated_positions_failures_and_empty_truthy_payload(monkeypatch, queued):
    bad = "https://example.com/bad"
    report, _, submitted, executed = await run_report(
        monkeypatch,
        "enhanced_batch",
        [
            {"url": BASE, "content": "", "extraction_successful": "truthy"},
            {"url": bad, "error": "fixture", "extraction_successful": False},
        ],
        seeds=iter([BASE, bad, BASE]),
        queued=queued,
    )
    assert [article["url"] for article in report.articles] == [BASE, bad, BASE]
    assert submitted == executed == [BASE, bad, BASE]
    assert report.articles[1]["error"] == "fixture"
    assert not report.outcome.has_partial_results


async def test_explicit_empty_inputs_do_not_fall_back_to_request_seed(monkeypatch):
    report, _, submitted, executed = await run_report(monkeypatch, "enhanced_batch", [{"url": BASE}], seeds=iter(()))
    assert report.articles == () and report.outcome.state == "completed"
    assert submitted == executed == []


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize(
    "strategy,want",
    [
        ("default", [BASE, "https://example.com/short", "https://example.com/blog/a/deep"]),
        ("best_first", [BASE, "https://example.com/blog/a/deep", "https://example.com/short"]),
    ],
)
async def test_report_strategy_order_scores_and_payload_match_task1(monkeypatch, strategy, want, queued):
    outcomes = [
        {"url": BASE, "links": ["/short", "/blog/a/deep"]},
        {"url": "https://example.com/short"},
        {"url": "https://example.com/blog/a/deep"},
    ]
    legacy = Phase5Harness(profile="enhanced_recursive", outcomes=outcomes)
    expected = await legacy.run(max_pages=3, max_depth=2, crawl_strategy=strategy)
    report, _, submitted, executed = await run_report(
        monkeypatch, "enhanced_recursive", outcomes, queued=queued, max_pages=3, max_depth=2, crawl_strategy=strategy
    )
    assert list(report.articles) == expected
    assert submitted == executed == want


@pytest.mark.parametrize("profile", ["legacy_sitemap", "enhanced_sitemap"])
@pytest.mark.parametrize("kind", ["namespaced", "plain", "index", "malformed"])
async def test_report_sitemap_formats_match_task1(monkeypatch, profile, kind):
    outcomes = [{"url": BASE}, {"url": "https://example.com/blog/second"}]
    old = Phase5Harness(profile=profile, outcomes=outcomes)
    old.sitemap_kind = kind
    expected = await old.run()
    report, fixture, submitted, executed = await run_report(monkeypatch, profile, outcomes, sitemap_kind=kind)
    assert list(report.articles) == expected
    assert executed == [item["url"] for item in expected]
    assert submitted == [fixture.sitemap_url, *executed]
    assert len(fixture.fetch_calls) == 1


@pytest.mark.parametrize("profile,respect", [("legacy_sitemap", False), ("enhanced_sitemap", True)])
async def test_report_sitemap_seed_policy_denial_never_fetches_or_executes(monkeypatch, profile, respect):
    report, fixture, submitted, executed = await run_report(
        monkeypatch, profile, [{"url": BASE}], denied=["https://example.com/sitemap.xml"]
    )
    assert report.articles == () and report.outcome.stop_reason == "policy_denied"
    assert fixture.policy_calls[0]["respect_robots"] is respect
    assert fixture.fetch_calls == submitted == executed == []


async def test_report_sitemap_filter_precedes_cap_and_keeps_credentials(monkeypatch):
    outcomes = [{"url": "https://example.com/blog/skip"}, {"url": BASE}, {"url": "https://example.com/blog/last"}]
    seen = []

    def accept(url):
        seen.append(url)
        return not url.endswith("skip")

    report, fixture, submitted, executed = await run_report(
        monkeypatch,
        "enhanced_sitemap",
        outcomes,
        max_urls=1,
        user_agent="phase5-agent",
        custom_headers={"X-Phase5": "test"},
        hooks=CrawlHooks(url_filter=accept),
    )
    assert seen == [outcomes[0]["url"], BASE]
    assert [item["url"] for item in report.articles] == executed == [BASE]
    assert submitted == [fixture.sitemap_url, BASE]
    assert fixture.fetch_calls[0]["headers"] == {"User-Agent": "phase5-agent", "X-Phase5": "test"}
    assert fixture.fetch_calls[0]["cookies"] == {"session": "test-cookie"}


@pytest.mark.parametrize("profile", ["legacy_recursive", "enhanced_recursive"])
@pytest.mark.parametrize("queued", [False, True])
async def test_failed_seed_count_and_expansion_match_task1(monkeypatch, profile, queued):
    outcomes = [
        {"url": "https://example.com/bad", "extraction_successful": False, "links": ["/blog/good"]},
        {"url": "https://example.com/blog/good"},
    ]
    report, _, submitted, executed = await run_report(
        monkeypatch,
        profile,
        outcomes,
        queued=queued,
        max_pages=1,
        max_depth=3,
        delay=0,
        base_url="https://example.com/bad",
        seeds=[item["url"] for item in outcomes] if profile == "legacy_recursive" else None,
    )
    want = [item["url"] for item in outcomes] if profile == "legacy_recursive" else [outcomes[0]["url"]]
    assert submitted == executed == want
    assert [item["url"] for item in report.articles] == ([outcomes[1]["url"]] if profile == "legacy_recursive" else [])


@pytest.mark.parametrize("strategy", ["default", "best_first"])
@pytest.mark.parametrize("override", [False, True])
async def test_explicit_false_zero_and_omission_match_task1(monkeypatch, strategy, override):
    outcomes = [
        {"url": BASE, "links": ["https://external.example/blog/a", "/blog/child"]},
        {"url": "https://external.example/blog/a"},
        {"url": "https://example.com/blog/child"},
    ]
    options = {"include_external_override": False, "score_threshold_override": 0.0} if override else {}
    config = {"web_crawl_include_external": True, "web_crawl_score_threshold": 0.99}
    old = Phase5Harness(profile="enhanced_recursive", outcomes=outcomes)
    old.config.update(config)
    expected = await old.run(max_pages=3, crawl_strategy=strategy, **options)
    report, _, submitted, executed = await run_report(
        monkeypatch, "enhanced_recursive", outcomes, config=config, max_pages=3, crawl_strategy=strategy, **options
    )
    assert list(report.articles) == expected
    assert submitted == executed == ([BASE, "https://example.com/blog/child"] if override else [BASE])


@pytest.mark.parametrize("profile", ["legacy_url_level", "enhanced_url_level"])
async def test_absolute_url_level_filter_matches_task1(monkeypatch, profile):
    outcomes = [
        {"url": BASE, "links": ["/blog/child", "/blog/child/deep"]},
        {"url": "https://example.com/blog/child"},
        {"url": "https://example.com/blog/child/deep"},
    ]
    old = Phase5Harness(profile=profile, outcomes=outcomes)
    expected = await old.run(level=2)
    expected = expected["articles"] if profile == "enhanced_url_level" else expected
    report, fixture, submitted, executed = await run_report(
        monkeypatch, profile, outcomes, level=2, max_depth=2, max_pages=10
    )
    assert {item["url"] for item in report.articles} == {item["url"] for item in expected}
    if profile == "legacy_url_level":
        assert list(report.articles) == expected
    assert set(executed) == {BASE, "https://example.com/blog/child"}
    assert set(submitted) == (
        {BASE, "https://example.com/blog/child", "https://example.com/blog/child/deep"}
        if profile == "legacy_url_level"
        else set(executed)
    )


async def test_legacy_fifo_filter_cycle_and_progress_parity(monkeypatch):
    seed = "https://example.com"
    outcomes = [
        {"url": seed, "links": ["/blog/z", "/blog/a", "/login", "https://other.example/blog/x", "/blog/z"]},
        {"url": seed + "/blog/z", "links": [seed]},
        {"url": seed + "/blog/a"},
    ]
    messages = []
    report, _, submitted, executed = await run_report(
        monkeypatch,
        "legacy_recursive",
        outcomes,
        max_pages=4,
        max_depth=1,
        delay=0,
        hooks=CrawlHooks(progress_callback=messages.append),
    )
    assert [item["url"] for item in report.articles] == submitted == executed == [item["url"] for item in outcomes]
    assert messages[:2] == [f"Scraping page 1/4: {seed}", f"Scraping page 2/4: {seed}/blog/z"]
    assert messages[-1] == "Scraping completed. Total pages scraped: 3"
    assert all("crawl_strategy" not in item["metadata"] for item in report.articles)


async def test_legacy_transport_failure_is_retained_without_false_completion(monkeypatch):
    messages = []
    outcomes = [
        {"url": "https://example.com", "links": ["/blog/next"]},
        {"url": "https://example.com/blog/next", "transport_error": True},
    ]
    report, _, submitted, executed = await run_report(
        monkeypatch,
        "legacy_recursive",
        outcomes,
        max_pages=3,
        max_depth=3,
        delay=0,
        hooks=CrawlHooks(progress_callback=messages.append),
    )
    assert report.outcome.stop_reason == "transport_unavailable" and report.outcome.has_partial_results
    assert report.articles[1]["error"] == "browser_transport_unavailable"
    assert submitted == executed == [item["url"] for item in outcomes]
    assert not any("completed" in item for item in messages)


async def test_sitemap_real_runtime_controls_charge_transport_not_an_extracted_page(monkeypatch):
    report, fixture, submitted, executed = await run_report(
        monkeypatch, "enhanced_sitemap", [{"url": BASE}], runtime_fetch=True
    )
    assert report.outcome.state == "completed" and executed == [BASE]
    assert fixture.budget.page_attempts == 1
    assert fixture.budget.network_requests == 1
    assert fixture.budget.transfer_bytes > 0
    assert submitted == [fixture.sitemap_url, BASE]


async def test_candidate_robots_uses_private_identity_and_precedes_custom_filter(monkeypatch):
    outcomes = [
        {"url": BASE, "links": ["/login", "/blog/blocked", "/blog/child"]},
        {"url": "https://example.com/blog/child"},
    ]
    filtered = []
    report, fixture, submitted, executed = await run_report(
        monkeypatch,
        "enhanced_recursive",
        outcomes,
        max_pages=3,
        max_depth=2,
        runtime_fetch=True,
        real_policy=True,
        user_agent="phase5-agent",
        custom_headers={"Authorization": "private"},
        hooks=CrawlHooks(url_filter=lambda url: filtered.append(url) or True),
    )
    assert executed == submitted == [BASE, "https://example.com/blog/child"]
    assert filtered == ["https://example.com/blog/child"]
    assert fixture.fetch_calls == [
        {
            "url": "https://example.com/robots.txt",
            "headers": {"User-Agent": "phase5-agent", "Authorization": "private"},
            "cookies": {"session": "test-cookie"},
        }
    ]
    assert fixture.budget.network_requests == 1 and fixture.budget.page_attempts == 2
    assert report.outcome.state == "completed"


async def test_duplicate_encounters_exhaust_default_candidate_budget_before_page_commit(monkeypatch):
    report, fixture, submitted, executed = await run_report(
        monkeypatch, "enhanced_recursive", [{"url": BASE, "links": ["/blog/child"] * 50_001}], max_pages=3, max_depth=2
    )
    assert report.outcome.stop_reason == "budget_exhausted" and report.articles == ()
    assert submitted == executed == [BASE]
    assert fixture.budget.discovered_candidates == 50_000


async def test_cancellation_traceback_does_not_retain_live_hooks_after_physical_drain(monkeypatch):
    import weakref
    from types import SimpleNamespace

    class Callback:
        def __call__(self, _snapshot):
            pass

    callback = Callback()
    retained = weakref.ref(callback)

    class Availability:
        def __call__(self):
            return False

    availability = Availability()
    availability_retained = weakref.ref(availability)
    accepted = asyncio.Event()
    loop = asyncio.get_running_loop()
    drained = loop.create_future()
    monkeypatch.setattr(
        crawl,
        "decide_web_outbound_policy",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=SimpleNamespace(allowed=True)),
    )

    def start(*_args, **_kwargs):
        accepted.set()
        result = asyncio.create_task(asyncio.Event().wait())
        drained.add_done_callback(lambda _done, capability=_kwargs["browser_http_fallback"]: None)
        return WorkHandle(result, drained, result.cancel)

    monkeypatch.setattr(crawl.article, "start_page", start)
    task = asyncio.create_task(
        crawl.execute_crawl(
            CrawlRequest(BASE),
            profile="enhanced_recursive",
            options={},
            supplied=frozenset(),
            config={},
            credentials={},
            hooks=CrawlHooks(on_progress=callback),
            browser_http_fallback=availability,
        )
    )
    del callback
    del availability
    await accepted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert retained() is not None
    assert availability_retained() is not None
    drained.set_result(None)
    await asyncio.gather(*(item for item in asyncio.all_tasks() if item.get_name() == "crawl-owned-drain"))
    assert retained() is None
    assert availability_retained() is None
    assert caught.value is not None


async def test_failed_legacy_page_at_cap_can_still_discover_successful_child(monkeypatch):
    outcomes = [
        {"url": "https://example.com", "extraction_successful": False, "links": ["/blog/child"]},
        {"url": "https://example.com/blog/child"},
    ]
    report, _, submitted, executed = await run_report(
        monkeypatch, "legacy_recursive", outcomes, max_pages=1, max_depth=2, delay=0
    )
    assert [item["url"] for item in report.articles] == [outcomes[1]["url"]]
    assert submitted == executed == [item["url"] for item in outcomes]
    assert report.outcome.stop_reason == "page_cap_reached"


async def test_initial_legacy_transport_denial_returns_original_failure_without_execution(monkeypatch):
    report, fixture, submitted, executed = await run_report(
        monkeypatch, "legacy_recursive", [{"url": BASE}], max_pages=3, max_depth=2, delay=0, transport_allowed=False
    )
    assert report.articles[0]["error"] == "browser_transport_unavailable"
    assert report.articles[0]["url"] == BASE
    assert not report.outcome.has_partial_results
    assert submitted == executed == fixture.calls == []


async def test_url_level_uses_level_as_default_traversal_depth(monkeypatch):
    outcomes = [
        {"url": BASE, "links": ["/blog/a"]},
        {"url": "https://example.com/blog/a", "links": ["/blog/b"]},
        {"url": "https://example.com/blog/b", "links": ["/blog/c"]},
    ]
    report, fixture, submitted, executed = await run_report(
        monkeypatch, "enhanced_url_level", outcomes, level=2, max_pages=10
    )
    assert report.outcome.state == "completed"
    assert submitted == executed == [item["url"] for item in outcomes]
    assert fixture.calls[0]["kwargs"]["allow_llm_extraction"] is False


async def test_zero_page_cap_still_preserves_seed_policy_denial(monkeypatch):
    report, fixture, submitted, executed = await run_report(
        monkeypatch, "enhanced_recursive", [{"url": BASE}], max_pages=0, denied=[BASE]
    )
    assert report.outcome.stop_reason == "policy_denied"
    assert fixture.policy_calls[0]["url"] == BASE
    assert submitted == executed == []


async def test_route_planning_thread_is_owned_before_its_first_dispatch(monkeypatch):
    import threading
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _harness

    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    fixture = _harness()
    original = fixture.dependencies.resolve_plan
    handles = []

    def resolve(*args):
        entered.set()
        try:
            release.wait(3)
            return original(*args)
        finally:
            exited.set()

    fixture.dependencies = replace(fixture.dependencies, resolve_plan=resolve)
    monkeypatch.setattr(crawl.article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    monkeypatch.setattr(
        crawl,
        "decide_web_outbound_policy",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=SimpleNamespace(allowed=True)),
    )
    start = crawl.article.start_page

    def capture(*args, **kwargs):
        handle = start(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(crawl.article, "start_page", capture)
    task = asyncio.create_task(
        crawl.execute_crawl(
            CrawlRequest(BASE),
            profile="enhanced_recursive",
            options={},
            supplied=frozenset(),
            config={},
            credentials={},
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert len(handles) == 1
        assert not handles[0].drained.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not handles[0].drained.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        release.set()
        assert await asyncio.to_thread(exited.wait, 2)
        if handles:
            await asyncio.wait_for(asyncio.shield(handles[0].drained), 2)


async def test_explicit_browser_method_and_private_identity_reach_canonical_pipeline(monkeypatch):
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _harness

    fixture = _harness()
    monkeypatch.setattr(crawl.article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    report = await crawl.execute_crawl(
        CrawlRequest(BASE),
        profile="enhanced_batch",
        options={"method": "playwright", "allow_llm_extraction": False},
        supplied=frozenset({"method", "allow_llm_extraction"}),
        config={},
        credentials={
            "user_agent": "phase5-agent",
            "custom_headers": {"Authorization": "private"},
            "custom_cookies": [{"name": "sid", "value": "private", "domain": "example.com"}],
        },
    )
    assert report.outcome.state == "completed"
    assert fixture.fetch.requests == [] and len(fixture.browser.calls) == 1
    assert fixture.browser.calls[0][1].user_agent == "phase5-agent"
    assert fixture.extract.call_args.kwargs["allow_llm_extraction"] is False


async def test_visit_policy_readmits_restored_identity_before_page_acquisition(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.crawl.models import FrontierEntry
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
    from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import plan

    resolved = plan()
    calls = []

    async def deny(url, **kwargs):
        calls.append((url, kwargs["respect_robots"]))
        return SimpleNamespace(allowed=False)

    monkeypatch.setattr(crawl, "decide_web_outbound_policy", deny)
    policy = crawl.ProfileTraversalPolicy(
        resolved,
        config={},
        headers={},
        fetch_document=None,
        hooks=CrawlHooks(),
        controls=CrawlContext(resolved.limits, clock=lambda: 0),
    )
    decision = await policy.evaluate(FrontierEntry(BASE, None, 0, 0, 0), phase="visit")
    assert not decision.allowed and decision.reason == "policy_denied"
    assert calls == [(BASE, False)]


@pytest.mark.parametrize("phase", ["seed", "candidate"])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
async def test_robots_physical_owner_holds_checkpoint_until_thread_exit(monkeypatch, phase, stop):
    import threading
    import weakref
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping import outbound_policy
    from tldw_Server_API.app.core.Web_Scraping.runtime import fetch as fetch_module
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import physical_work
    from tldw_Server_API.tests.Web_Scraping.test_phase5_crawl_engine import MemoryCheckpoint

    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    checkpoint = MemoryCheckpoint()
    contexts, registered, executed, fetched = [], [], [], []
    context_type = crawl.CrawlContext

    class Callback:
        def __call__(self, _snapshot):
            pass

    callback = Callback()
    retained = weakref.ref(callback)

    def context(*args, **kwargs):
        result = context_type(*args, **kwargs)
        contexts.append(result)
        return result

    def http_fetch(url, **kwargs):
        fetched.append(url)
        owner = physical_work.get()
        registered.append(owner is not None and any(h.drained is owner.drained for h in contexts[0].owned_handles()))
        entered.set()
        try:
            assert release.wait(5), "test failed to release physical request"
            kwargs["on_response_chunk"](1)
            return {"url": url, "status": 200, "text": "User-agent: *\nAllow: /", "headers": {}, "backend": "httpx"}
        finally:
            exited.set()

    def start(url, cookies, *, discover, **kwargs):
        executed.append(url)
        kwargs["controls"].reserve("page_attempts").commit()
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(
            PageData(
                {"url": url, "content": "kept", "extraction_successful": True},
                discover('<a href="/blog/child">child</a>', url, url),
            )
        )
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    monkeypatch.setattr(crawl, "CrawlContext", context)
    monkeypatch.setattr(outbound_policy, "evaluate_url_policy", lambda _url: SimpleNamespace(allowed=True))
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    monkeypatch.setattr(fetch_module, "http_fetch", http_fetch)
    monkeypatch.setattr(crawl.article, "start_page", start)
    profile = "enhanced_sitemap" if phase == "seed" else "enhanced_recursive"
    task = asyncio.create_task(
        crawl.execute_crawl(
            CrawlRequest(BASE),
            profile=profile,
            options={"max_pages": 3},
            supplied=frozenset({"max_pages"}),
            config={},
            credentials={},
            limits=CrawlLimits(execution_seconds=0.5),
            checkpoint=checkpoint,
            hooks=CrawlHooks(on_progress=callback),
        )
    )
    del callback
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            report = await asyncio.wait_for(task, 2)
            assert report.outcome.stop_reason == "deadline_exhausted"
        assert not exited.is_set()
        assert not checkpoint.closed
        assert retained() is not None
        assert registered == [True]
        assert len(contexts[0].owned_handles()) == 1
        assert contexts[0].snapshot().page_attempts == int(phase == "candidate")
        assert fetched == ["https://example.com/robots.txt"]
        assert executed == ([] if phase == "seed" else [BASE])
        assert checkpoint.state.frontier.completed == ()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        release.set()
        assert await asyncio.to_thread(exited.wait, 2)
        await asyncio.wait_for(
            asyncio.gather(*(item for item in asyncio.all_tasks() if item.get_name() == "crawl-owned-drain")), 2
        )
    assert checkpoint.closed
    assert contexts[0].owned_handles() == ()
    assert retained() is None


@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="eager task dispatch requires Python 3.12+")
async def test_policy_transport_is_registered_before_even_eager_task_dispatch(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import physical_work

    contexts, registered = [], []
    original = crawl.CrawlContext

    def context(*args, **kwargs):
        result = original(*args, **kwargs)
        contexts.append(result)
        return result

    async def allow(*args, **kwargs):
        return SimpleNamespace(allowed=True)

    async def fetch(dependencies, request):
        owner = physical_work.get()
        registered.append(any(handle.drained is owner.drained for handle in contexts[0].owned_handles()))
        return FetchResponse(url=request.url, status=200, text="<urlset/>", headers={}, backend="httpx")

    monkeypatch.setattr(crawl, "CrawlContext", context)
    monkeypatch.setattr(crawl, "decide_web_outbound_policy", allow)
    monkeypatch.setattr(crawl.article, "_fetch_response", fetch)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        report = await crawl.execute_crawl(
            CrawlRequest(BASE), profile="legacy_sitemap", options={}, supplied=frozenset(), config={}, credentials={}
        )
    finally:
        loop.set_task_factory(previous)
    assert registered == [True]
    assert report.outcome.state == "completed"


async def test_production_seed_query_overflow_never_reaches_allocating_normalizer(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import url_utils

    encoded = []

    def forbidden(*args, **kwargs):
        encoded.append(True)
        raise AssertionError("oversized seed query reached urlencode")

    monkeypatch.setattr(url_utils, "urlencode", forbidden)
    seed = "https://e.test/x?q=" + "\u00e9" * 5
    report, _, submitted, executed = await run_report(
        monkeypatch, "enhanced_batch", [{"url": seed}], base_url=seed, limits=CrawlLimits(max_url_bytes=48)
    )
    assert report.outcome.stop_reason == "budget_exhausted"
    assert not encoded and submitted == executed == []


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("profile", ["enhanced_batch", "enhanced_sitemap", "legacy_sitemap"])
async def test_denied_middle_independent_item_retains_failure_and_continues_in_order(monkeypatch, queued, profile):
    first, denied, last = ("https://example.com/" + name for name in ("first", "denied", "last"))
    outcomes = [{"url": first, "summary": "original summary"}, {"url": denied}, {"url": last, "content": ""}]
    report, fixture, submitted, executed = await run_report(
        monkeypatch,
        profile,
        outcomes,
        queued=queued,
        denied=[denied],
        seeds=[first, denied, last] if profile == "enhanced_batch" else None,
    )
    failure = {
        "url": denied,
        "error": "Blocked by outbound policy",
        "extraction_successful": False,
        "policy_reason": "robots_unreachable",
        "policy_mode": "strict",
        "policy_stage": "visit",
        "policy_source": profile,
    }
    if profile == "legacy_sitemap":
        failure.update(title="N/A", author="N/A", date="N/A", content="")
    assert report.articles == (fixture.article(first), failure, fixture.article(last))
    assert executed == [first, last]
    assert submitted == ([fixture.sitemap_url, first, last] if profile.endswith("sitemap") else [first, last])
    assert denied not in [item["url"] for item in fixture.fetch_calls]
    assert fixture.budget.page_attempts == 2
    assert report.outcome.state == "completed" and report.outcome.stop_reason == "frontier_exhausted"
    assert not report.outcome.has_partial_results
