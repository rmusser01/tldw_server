"""Parent-owned canonical acquisition, without live network or browser dependencies."""

import asyncio
import threading
import time
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveredLink, DiscoveryResult
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration import article
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import URL, _harness

pytestmark = pytest.mark.asyncio


def _discover(document, request_url, final_url):
    assert document
    return DiscoveryResult(request_url, final_url, (DiscoveredLink(final_url + "/child", final_url),))


def _start(monkeypatch, harness, controls, **kwargs):
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: harness.dependencies)
    return article.start_page(
        URL,
        None,
        controls=controls,
        request_context=RuntimeRequestContext(source="private", stage="fetch"),
        discover=_discover,
        **kwargs,
    )


@pytest.mark.parametrize(
    "stage", ["preflight_http", "preflight_browser", "external_tool", "http_fallback", "browser_retry"]
)
async def test_parent_stop_escapes_all_fallbacks(monkeypatch, stage):
    stop = ParentStop("budget_exhausted")
    harness = _harness(
        preflight=stop if stage.startswith("preflight") or stage == "external_tool" else None,
        fetch_outcomes=[stop] if stage == "http_fallback" else None,
        browser_outcomes=[stop] if stage == "browser_retry" else None,
        backend="playwright" if stage == "browser_retry" else "httpx",
    )
    controls = CrawlContext(CrawlLimits(), clock=time.monotonic)
    handle = _start(monkeypatch, harness, controls)
    with pytest.raises(ParentStop) as caught:
        await handle.result
    assert caught.value is stop
    await handle.drained
    assert harness.extract.call_count == 0
    if stage != "browser_retry":
        assert harness.browser.calls == []


async def test_authenticated_discovery_uses_same_document(monkeypatch):
    final = "https://example.test/private/final"
    harness = _harness(
        fetch_outcomes=[FetchResponse(url=final, status=200, text="<a>private</a>", headers={}, backend="httpx")]
    )
    controls = CrawlContext(CrawlLimits(), clock=time.monotonic)
    page = await _start(monkeypatch, harness, controls).result
    assert len(harness.fetch.requests) == 1
    assert page.discovery.final_url == final
    assert page.discovery.links[0].parent_url == final
    assert "discovery" not in page.article and "<a>private</a>" not in repr(page.article)
    assert controls.snapshot().page_attempts == 1


@pytest.mark.parametrize("successful", [False, True, "truthy"])
async def test_success_only_discovery_uses_truthy_extraction_predicate(monkeypatch, successful):
    harness = _harness(extract_results=[{"content": "kept", "extraction_successful": successful}])
    seen = []
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: harness.dependencies)

    def discover(document, request, final):
        seen.append((document, request, final, harness.extract.call_count))
        return DiscoveryResult(request, final, ())

    handle = article.start_page(
        URL,
        None,
        controls=CrawlContext(CrawlLimits(), clock=time.monotonic),
        request_context=RuntimeRequestContext(),
        discover=discover,
        discovery_on_success=True,
    )
    await handle.result
    await handle.drained
    assert len(seen) == int(bool(successful))
    assert all(item[3] == 1 for item in seen)
    assert len(harness.fetch.requests) == 1


async def test_failed_enhanced_extraction_does_not_spend_discovery_budget(monkeypatch):
    harness = _harness(
        fetch_outcomes=[
            FetchResponse(url=URL, status=200, text='<a href="/child">x</a>' * 3, headers={}, backend="httpx")
        ],
        extract_results=[{"extraction_successful": False, "content": ""}],
    )
    controls = CrawlContext(CrawlLimits(discovered_candidates=1), clock=time.monotonic)
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: harness.dependencies)

    def discover(document, request, final):
        controls.reserve("discovered_candidates", 3).commit()
        return DiscoveryResult(request, final, ())

    handle = article.start_page(
        URL,
        None,
        controls=controls,
        request_context=RuntimeRequestContext(),
        discover=discover,
        discovery_on_success=True,
    )
    page = await handle.result
    await handle.drained
    assert not page.article["extraction_successful"]
    assert controls.snapshot().discovered_candidates == 0
    assert len(harness.fetch.requests) == 1


@pytest.mark.parametrize("successful", [False, True])
async def test_private_discovery_gate_observes_result_before_deciding_unused_expansion(monkeypatch, successful):
    harness = _harness(extract_results=[{"extraction_successful": successful, "content": "body"}])
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: harness.dependencies)
    seen = []

    def gate(result):
        seen.append((result["extraction_successful"], harness.extract.call_count))
        return not result["extraction_successful"]

    handle = article.start_page(
        URL,
        None,
        controls=CrawlContext(CrawlLimits(), clock=time.monotonic),
        request_context=RuntimeRequestContext(),
        discover=_discover,
        discovery_gate=gate,
    )
    page = await handle.result
    await handle.drained
    assert seen == [(successful, 1)]
    assert bool(page.discovery.links) is not successful


async def test_page_attempt_exact_limit_rejects_retry(monkeypatch):
    harness = _harness()
    controls = CrawlContext(CrawlLimits(page_attempts=1), clock=time.monotonic)
    await _start(monkeypatch, harness, controls).result
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await _start(monkeypatch, harness, controls).result
    assert len(harness.fetch.requests) == 1


async def test_sync_http_late_completion_is_owned_without_late_result(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    harness = _harness()
    original = harness.fetch.fetch

    def fetch(request):
        entered.set()
        release.wait(5)
        return original(request)

    harness.fetch.fetch = fetch
    handle = _start(monkeypatch, harness, CrawlContext(CrawlLimits(), clock=time.monotonic))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        handle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handle.result
        assert not handle.drained.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(handle.drained), 3)
    assert harness.extract.call_count == 0


async def test_active_timeout_stops_acquisition(monkeypatch):
    harness = _harness(backend="playwright")

    async def acquire(*_args):
        await asyncio.sleep(10)

    harness.browser.acquire = acquire
    controls = CrawlContext(CrawlLimits(execution_seconds=0.03), clock=time.monotonic)
    handle = _start(monkeypatch, harness, controls)
    with pytest.raises(ParentStop, match="deadline_exhausted"):
        await asyncio.wait_for(asyncio.shield(handle.result), 1)
    await handle.drained


async def test_analyzer_local_error_remains_fail_open(monkeypatch):
    harness = _harness(preflight=ValueError("local"))
    page = await _start(monkeypatch, harness, CrawlContext(CrawlLimits(), clock=time.monotonic)).result
    assert page.article["extraction_successful"]


async def test_explicit_plan_preserves_private_identity(monkeypatch):
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _plan

    harness = _harness()
    plan = replace(
        _plan(), headers={"User-Agent": "private-agent", "Authorization": "private"}, cookies={"sid": "secret"}
    )
    await _start(monkeypatch, harness, CrawlContext(CrawlLimits(), clock=time.monotonic), plan=plan).result
    request = harness.fetch.requests[0]
    assert request.headers["Authorization"] == "private"
    assert request.cookies["sid"] == "secret"
    assert request.context.source == "private"
    assert request.context.metadata == {}


@pytest.mark.parametrize("kind", ["http", "ws"])
async def test_browser_parent_counts_handshakes_and_stops_before_second_dispatch(kind):
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    url = URL if kind == "http" else "wss://example.com/socket"
    runtime = _FakeBrowserRuntime(dispatches=[(kind, url), (kind, url)])
    parent = CrawlContext(CrawlLimits(network_requests=1), clock=time.monotonic)
    browser = _adapter(runtime, _FakeGuard([True] * 10))
    browser.bind_parent(parent, RuntimeRequestContext())
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await browser.acquire(URL, _profile())
    assert parent.snapshot().network_requests == 1
    assert parent.snapshot().browser_attempts == 1


async def test_browser_parent_transfer_stop_prevents_retry_and_keeps_reason():
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    runtime = _FakeBrowserRuntime(cdp_events=[("Network.dataReceived", {"requestId": "main", "encodedDataLength": 4})])
    parent = CrawlContext(CrawlLimits(transfer_bytes=3), clock=time.monotonic)
    browser = _adapter(runtime, _FakeGuard([True] * 10))
    browser.bind_parent(parent, RuntimeRequestContext())
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await browser.acquire(URL, _profile())
    assert parent.snapshot().browser_attempts == 1
    assert parent.snapshot().transfer_bytes == 0


async def test_extraction_future_drain_is_per_page(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import ExtractionExecutorManager

    entered, release = threading.Event(), threading.Event()
    harness = _harness()

    def extract(*_args, **_kwargs):
        entered.set()
        release.wait(5)
        return {"content": "late", "extraction_successful": True}

    manager = ExtractionExecutorManager()
    harness.dependencies = replace(harness.dependencies, executor=manager, extract=extract)
    handle = _start(monkeypatch, harness, CrawlContext(CrawlLimits(), clock=time.monotonic))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        handle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handle.result
        assert not handle.drained.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(handle.drained), 3)
        await manager.shutdown()


async def test_browser_lease_retains_drain_through_cancellation_resistant_cleanup():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import ArticleFailure
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, physical_work
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    runtime = _FakeBrowserRuntime(cleanup_modes={"close:page": "resistant"})
    browser = _adapter(runtime, _FakeGuard([]), cleanup_grace_s=0.01)
    owner = PhysicalWork()
    token = physical_work.set(owner)
    try:
        with pytest.raises(ArticleFailure):
            await browser.acquire(URL, _profile())
        owner.finish()
        assert not owner.drained.done()
        runtime.resistant_cleanup_release.set()
        await asyncio.wait_for(asyncio.shield(owner.drained), 2)
    finally:
        runtime.resistant_cleanup_release.set()
        physical_work.reset(token)


async def test_direct_result_cancellation_closes_parent_admission(monkeypatch):
    harness = _harness(backend="playwright")
    captured = []
    entered = asyncio.Event()

    def context(*_args, **kwargs):
        captured.append(kwargs["parent"])
        return object()

    async def acquire(*_args):
        entered.set()
        await asyncio.Event().wait()

    harness.dependencies = replace(harness.dependencies, build_preflight_context=context)
    harness.browser.acquire = acquire
    handle = _start(monkeypatch, harness, CrawlContext(CrawlLimits(), clock=time.monotonic))
    await entered.wait()
    handle.result.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.result
    await handle.drained
    with pytest.raises(asyncio.CancelledError):
        captured[0].reserve("network_requests")


async def test_runtime_http_byte_stop_is_not_browser_fallback(monkeypatch):
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Web_Scraping.runtime import fetch as fetch_module
    from tldw_Server_API.app.core.Web_Scraping.runtime.fetch import DefaultFetchClient

    parent = CrawlContext(CrawlLimits(transfer_bytes=3), clock=time.monotonic)
    harness = _harness()
    harness.dependencies = replace(harness.dependencies, fetch_client=DefaultFetchClient())

    def fetch(*_args, **kwargs):
        http_client._observe_response_chunk(kwargs["on_response_chunk"], 4)
        raise AssertionError("no body may be returned after observer abort")

    monkeypatch.setattr(fetch_module, "http_fetch", fetch)
    handle = _start(monkeypatch, harness, parent)
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await handle.result
    await handle.drained
    assert parent.snapshot().network_requests == 1
    assert parent.snapshot().transfer_bytes == 0
    assert harness.browser.calls == []


async def test_browser_retry_requires_new_parent_attempt():
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    parent = CrawlContext(CrawlLimits(browser_attempts=1), clock=time.monotonic)
    runtime = _FakeBrowserRuntime(stage_errors={"goto": RuntimeError("local")})

    async def sleep(_delay):
        return None

    browser = _adapter(runtime, _FakeGuard([]), sleep=sleep)
    browser.bind_parent(parent, RuntimeRequestContext())
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await browser.acquire(URL, _profile())
    assert parent.snapshot().browser_attempts == 1


async def test_internal_http_awaiter_cancellation_does_not_claim_thread_exit(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, owned_to_thread, physical_work

    entered, release = threading.Event(), threading.Event()
    inner = []
    create_task = asyncio.create_task

    def capture(coroutine, **kwargs):
        task = create_task(coroutine, **kwargs)
        inner.append(task)
        return task

    def block():
        entered.set()
        release.wait(5)

    owner = PhysicalWork()
    token = physical_work.set(owner)
    monkeypatch.setattr(asyncio, "create_task", capture)
    outer = asyncio.get_running_loop().create_task(owned_to_thread(block))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        inner[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
        owner.finish()
        await asyncio.sleep(0)
        assert not owner.drained.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(owner.drained), 3)
        physical_work.reset(token)


async def test_page_cancel_between_http_timeout_and_commit_rolls_back_request(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.runtime import FetchRequest
    from tldw_Server_API.app.core.Web_Scraping.runtime import fetch as fetch_module
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PageControls

    parent = CrawlContext(CrawlLimits(), clock=lambda: 0)
    page = PageControls(parent)
    cap_timeout = page.cap_timeout
    dispatched = []

    def cancel_after_timeout(requested):
        timeout = cap_timeout(requested)
        page.cancel()
        return timeout

    def fetch(*args, **_kwargs):
        dispatched.append(args)
        return {"url": URL, "status": 200, "text": "late", "headers": {}, "backend": "httpx"}

    monkeypatch.setattr(page, "cap_timeout", cancel_after_timeout)
    monkeypatch.setattr(fetch_module, "http_fetch", fetch)
    with pytest.raises(asyncio.CancelledError):
        fetch_module.DefaultFetchClient(page).fetch(FetchRequest(url=URL, max_response_bytes=100))
    assert dispatched == []
    assert parent.snapshot().network_requests == 0
    sibling = PageControls(parent).reserve("network_requests")
    sibling.commit()
    sibling.close()
    assert parent.snapshot().network_requests == 1


async def test_websocket_stop_closes_admission_while_navigation_is_held(monkeypatch):
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _FakeHttpRoute,
        _FakePage,
        _FakeWebSocketRoute,
        _profile,
    )

    navigating, release = asyncio.Event(), asyncio.Event()

    async def held_navigation(_page, *_args, **_kwargs):
        navigating.set()
        await release.wait()

    monkeypatch.setattr(_FakePage, "goto", held_navigation)
    runtime = _FakeBrowserRuntime()
    guard = _FakeGuard([True, True])
    parent = CrawlContext(CrawlLimits(network_requests=1), clock=lambda: 0)
    parent.reserve("network_requests").commit()
    browser = _adapter(runtime, guard)
    browser.bind_parent(parent, RuntimeRequestContext())
    acquisition = asyncio.create_task(browser.acquire(URL, _profile()))
    try:
        await asyncio.wait_for(navigating.wait(), 1)
        socket = _FakeWebSocketRoute("wss://example.com/socket")
        await asyncio.wait_for(runtime.context.dispatch_websocket(socket), 1)
        await asyncio.wait_for(runtime.context_closed.wait(), 1)
        assert not release.is_set() and not acquisition.done()
        assert runtime.context_close_task_names == ["article-browser-emergency-context-shutdown"]
        assert socket.connect_calls == 0
        assert (1011, "Connection failed") not in socket.close_calls

        late = _FakeHttpRoute(URL)
        await asyncio.wait_for(runtime.context.dispatch_http(late), 1)
        assert late.abort_calls == 1 and late.continue_calls == []
        assert len(guard.calls) == 1
    finally:
        release.set()
        result = await asyncio.wait_for(asyncio.gather(acquisition, return_exceptions=True), 2)
    assert isinstance(result[0], ParentStop) and result[0].reason == "budget_exhausted"


@pytest.mark.parametrize("cancel_at", ["navigation", "cleanup"])
async def test_direct_caller_cancel_survives_parent_deadline_during_browser_cleanup(monkeypatch, cancel_at):
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _FakePage,
    )

    now = [0.0]
    navigating, cleanup_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_close = _FakePage.close

    async def navigation(_page, *_args, **_kwargs):
        navigating.set()
        if cancel_at == "navigation":
            await asyncio.Event().wait()

    async def held_close(page):
        cleanup_started.set()
        await release.wait()
        await original_close(page)

    monkeypatch.setattr(_FakePage, "goto", navigation)
    monkeypatch.setattr(_FakePage, "close", held_close)
    runtime = _FakeBrowserRuntime()
    harness = _harness(backend="playwright")
    harness.dependencies = replace(harness.dependencies, browser=_adapter(runtime, _FakeGuard([])))
    parent = CrawlContext(CrawlLimits(execution_seconds=10), clock=lambda: now[0])
    handle = _start(monkeypatch, harness, parent)
    try:
        await asyncio.wait_for(navigating.wait(), 1)
        if cancel_at == "navigation":
            handle.result.cancel("direct-caller-cancel")
        await asyncio.wait_for(cleanup_started.wait(), 1)
        if cancel_at == "cleanup":
            handle.result.cancel("direct-caller-cancel")
        now[0] = 11.0
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(handle.result, 2)
        if cancel_at == "navigation":
            assert caught.value.args == ("direct-caller-cancel",)
        assert harness.extract.call_count == 0
        with pytest.raises(ParentStop, match="deadline_exhausted"):
            parent.check()
    finally:
        release.set()
        await asyncio.gather(handle.result, return_exceptions=True)
        await asyncio.wait_for(asyncio.shield(handle.drained), 2)
