"""Real preflight Chromium dispatch and canonical stop propagation."""

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration import article
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.preflight import BrowserProbeOptions, PreflightRuntimeControls
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import GuardedPlaywrightBrowserProbe
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeError
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _allowed_target, _harness
from tldw_Server_API.tests.WebScraping.integration.test_phase5_enhanced_browser_native import _legacy_allowed, _sites

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def native_cdp_observation(monkeypatch):
    """Observe real CDP retirement, optionally holding native continuations."""
    from playwright.async_api import Browser

    state = SimpleNamespace(
        entered=asyncio.Event(),
        release=asyncio.Event(),
        exited=asyncio.Event(),
        hold_continuations=False,
        detached_connected=[],
        detached_after_release_and_exit=[],
    )
    new_session = Browser.new_browser_cdp_session

    async def session(browser):
        root = await new_session(browser)
        send, detach = root.send, root.detach
        browser.on("disconnected", lambda _: state.exited.set())

        async def observed_send(method, params=None):
            if state.hold_continuations and method == "Fetch.continueRequest":
                state.entered.set()
                await state.release.wait()
            return await send(method, params)

        async def observed_detach():
            state.detached_connected.append(browser.is_connected())
            state.detached_after_release_and_exit.append(state.release.is_set() and state.exited.is_set())
            await detach()

        monkeypatch.setattr(root, "send", observed_send)
        monkeypatch.setattr(root, "detach", observed_detach)
        return root

    monkeypatch.setattr(Browser, "new_browser_cdp_session", session)
    yield state
    state.release.set()


@pytest.mark.parametrize("mode", ["denied", "allowed", "budget"])
async def test_native_preflight_redirect_admission_checks_and_charges_each_dispatch_once(mode):
    checks = []

    class Guard:
        async def decide(self, url, **kwargs):
            checks.append(urlsplit(url).path)
            return SimpleNamespace(allowed=mode != "denied" or urlsplit(url).path != "/step")

    parent = CrawlContext(CrawlLimits(network_requests=1 if mode == "budget" else 10), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=Guard(), transport_decision=_legacy_allowed)
    failure = None
    async with _sites() as (bases, seen):
        try:
            async with probe.open_page(BrowserProbeOptions(capture_requests=True)) as page:
                try:
                    await page.goto(bases[0] + "/start", wait_until="domcontentloaded", timeout_ms=5000)
                except (ProbeError, ParentStop) as error:
                    failure = error
                if mode == "allowed":
                    assert "article preserved" in await page.content()
                    assert {urlsplit(url).path for url in page.captured_request_urls()} == {"/start", "/step", "/done"}
        finally:
            await controls.close()
        assert [path for _, path, _ in seen] == (["/start", "/step", "/done"] if mode == "allowed" else ["/start"])
        assert checks == (["/start", "/step", "/done"] if mode == "allowed" else ["/start", "/step"])
        assert parent.snapshot().network_requests == controls.consumed.requests == (3 if mode == "allowed" else 1)
        if mode == "budget":
            assert isinstance(failure, ParentStop) and failure.reason == "budget_exhausted"


async def test_native_preflight_redirect_stop_through_real_facade_prevents_canonical_fallback(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.preflight import facade, runner
    from tldw_Server_API.app.core.Web_Scraping.preflight.facade import PreflightAdapterOverrides
    from tldw_Server_API.app.core.Web_Scraping.preflight.options import PreflightOptions
    from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeHttpResponse

    async def noop(*args, **kwargs):
        return {}

    async def inert_http(request):
        return ProbeHttpResponse(url=request.url, status=200, headers={}, text="plain")

    class Guard:
        async def decide(self, url, **kwargs):
            return SimpleNamespace(allowed=True)

    async def target(url, **kwargs):
        return _allowed_target(url)

    def context(target, options, **kwargs):
        value = facade.build_execution_context(
            target,
            options,
            **kwargs,
            injected_adapters=PreflightAdapterOverrides(egress_guard=Guard(), http=SimpleNamespace(get=inert_http)),
        )
        value.browser._transport_decision = _legacy_allowed
        return value

    monkeypatch.setattr(runner, "_check_robots_txt", noop)
    monkeypatch.setattr(runner, "_analyze_tls_fingerprint", noop)
    harness = _harness()
    dependencies = replace(
        harness.dependencies,
        evaluate_target=target,
        build_preflight_context=context,
        run_preflight=facade.run_preflight,
        preflight_options=lambda _: PreflightOptions(enabled=True, timeout_s=10),
    )
    async with _sites() as (bases, seen):
        parent = CrawlContext(CrawlLimits(network_requests=1), clock=time.monotonic)
        handle = article.start_page(
            bases[0] + "/start",
            None,
            controls=parent,
            request_context=RuntimeRequestContext(),
            discover=lambda *args: DiscoveryResult(args[1], args[2], ()),
            dependencies=dependencies,
        )
        try:
            with pytest.raises(ParentStop, match="budget_exhausted"):
                await handle.result
        finally:
            await handle.drained
        assert [path for _, path, _ in seen] == ["/start"]
        assert harness.fetch.requests == harness.browser.calls == [] and harness.extract.call_count == 0


@pytest.mark.parametrize("close_mode", ["held", "rejected"])
async def test_native_preflight_guard_survives_held_or_rejected_close(monkeypatch, close_mode, native_cdp_observation):
    from playwright.async_api import Browser

    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters import browser as adapter
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, physical_work

    entered, release = asyncio.Event(), asyncio.Event()
    browsers, detached = [], native_cdp_observation.detached_connected
    close = Browser.close

    async def held_close(self, *args, **kwargs):
        browsers.append(self)
        entered.set()
        if close_mode == "rejected":
            raise RuntimeError("native close rejected")
        await release.wait()
        await close(self, *args, **kwargs)

    async def start():
        native = await adapter._DefaultPlaywrightLauncher().start()
        stop = native.stop

        async def held_stop():
            await release.wait()
            await stop()

        native.stop = held_stop
        return native

    monkeypatch.setattr(Browser, "close", held_close)
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", 0.05)
    parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)

    class Guard:
        async def decide(self, url, **kwargs):
            return SimpleNamespace(allowed=True)

    probe = GuardedPlaywrightBrowserProbe(
        controls=controls,
        egress_guard=Guard(),
        transport_decision=_legacy_allowed,
        launcher=SimpleNamespace(start=start),
    )
    owner = PhysicalWork()
    async with _sites() as (bases, seen):

        async def run():
            try:
                async with probe.open_page(BrowserProbeOptions()) as page:
                    await page.goto(bases[0] + "/done", wait_until="domcontentloaded", timeout_ms=5000)
            finally:
                owner.finish()

        token = physical_work.set(owner)
        task = asyncio.create_task(run())
        physical_work.reset(token)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(asyncio.shield(task), 2)
            assert browsers[0].is_connected() and detached == [] and not owner.drained.done()
            # The still-owned native guard blocks even a new context, with no
            # BrowserContext routing installed by this test.
            context = await browsers[0].new_context()
            page = await context.new_page()
            with pytest.raises(Exception, match="ERR_BLOCKED_BY_CLIENT"):
                await page.goto(bases[0] + "/deny", timeout=1000)
            assert not any(path == "/deny" for _, path, _ in seen)
            # Supply positive native retirement before releasing the separately
            # held connection stop; a connection loss alone is not close proof.
            await close(browsers[0])
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            done, _ = await asyncio.wait({owner.drained}, timeout=3)
            assert done, (
                browsers[0].is_connected(),
                detached,
                owner._count,
                [
                    (item.get_name(), [frame.f_code.co_name for frame in item.get_stack()])
                    for item in asyncio.all_tasks()
                    if item.get_name().startswith("preflight")
                ],
            )
            await controls.close()
        assert detached == [False] and not browsers[0].is_connected()


async def test_native_preflight_command_holds_physical_owner_past_browser_exit(monkeypatch, native_cdp_observation):
    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters import browser as adapter
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import PhysicalWork, observe_physical, physical_work

    observation = native_cdp_observation
    observation.hold_continuations = True
    entered, release, exited = observation.entered, observation.release, observation.exited
    detached, navigation = observation.detached_after_release_and_exit, []
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", 0.05)
    controls = PreflightRuntimeControls(RuntimeRequestContext())
    guard = SimpleNamespace(decide=lambda *args, **kwargs: asyncio.sleep(0, result=SimpleNamespace(allowed=True)))
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=guard, transport_decision=_legacy_allowed)
    owner = PhysicalWork()
    async with _sites() as (bases, seen):

        async def run():
            try:
                async with probe.open_page(BrowserProbeOptions()) as page:
                    # Keep the public navigation awaiter owned too: cancelling
                    # the test owner must not discard Playwright's eventual error.
                    async def navigate():
                        try:
                            await page.goto(bases[0] + "/done", wait_until="domcontentloaded", timeout_ms=5000)
                        except ProbeError as error:
                            # Python 3.14 shield logs late task exceptions even
                            # after retrieval. Return and assert this expected
                            # public error, rather than abandoning or hiding it.
                            return error

                    pending = asyncio.create_task(navigate())
                    navigation.append(pending)
                    observe_physical(pending)
                    await asyncio.shield(pending)
            finally:
                owner.finish()

        token = physical_work.set(owner)
        task = asyncio.create_task(run())
        physical_work.reset(token)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            await asyncio.wait_for(exited.wait(), 2)
            outcomes = await asyncio.gather(*navigation)
            assert all(isinstance(value, ProbeError) for value in outcomes)
            assert all(item.done() for item in navigation)
            assert detached == [] and not owner.drained.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.gather(*navigation, return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(owner.drained), 3)
            await controls.close()
        assert detached == [True]
        assert seen == [], "held continuation must never dispatch after cancellation"


@pytest.mark.parametrize("path", ["/children", "/websocket"])
async def test_native_preflight_keeps_settings_capture_blocks_and_child_policy(path):
    checks = []

    class Guard:
        async def decide(self, url, **kwargs):
            checks.append(urlsplit(url).path)
            return SimpleNamespace(allowed=urlsplit(url).path != "/deny")

    parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
    controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
    probe = GuardedPlaywrightBrowserProbe(controls=controls, egress_guard=Guard(), transport_decision=_legacy_allowed)
    async with _sites() as (bases, seen):
        try:
            async with probe.open_page(
                BrowserProbeOptions(
                    block_resource_types=("image",),
                    capture_requests=True,
                    extra_headers={"X-Preflight": "kept"},
                    init_scripts=("window.preflightMarker = 'kept';",),
                )
            ) as page:
                try:
                    await page.goto(bases[0] + path, wait_until="domcontentloaded", timeout_ms=5000)
                    await page.wait_for_timeout(100)
                except ProbeError as error:
                    pytest.fail(f"{error.error_code}: seen={[p for _, p, _ in seen]}; checks={checks}")
                assert await page.evaluate("window.preflightMarker") == "kept"
                captured = [urlsplit(url).path for url in page.captured_request_urls()]
            assert path in captured
            assert not any(received in {"/image", "/frame-image", "/deny"} for _, received, _ in seen)
            assert all(headers.get("x-preflight") == "kept" for _, _, headers in seen)
            if path == "/children":
                assert {"/frame", "/worker-fetch", "/popup"} <= {received for _, received, _ in seen}
            else:
                assert "/deny" in checks
            assert parent.snapshot().network_requests == len(seen)
        finally:
            await controls.close()


@pytest.mark.parametrize("boundary", ["redirect_exact", "redirect_over", "final"])
@pytest.mark.parametrize("enhanced", [False, True])
async def test_native_article_respects_trusted_url_bound_on_redirect_and_final_location(boundary, enhanced):
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import GuardedArticleBrowser
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import ArticlePlan

    requests, bases = [], []
    maximum = 64

    async def serve(reader, writer):
        try:
            line = (await reader.readuntil(b"\r\n\r\n")).decode().split("\r\n")[0]
            path = line.split(" ")[1]
            requests.append(path)
            target = "/" + "x" * (maximum - len(bases[0]) - 1 + int(boundary != "redirect_exact"))
            location = f"Location: {target}\r\n" if path == "/s" and boundary != "final" else ""
            status = "302 Found" if location else "200 OK"
            body = b"<html>bounded article</html>"
            writer.write(
                (
                    f"HTTP/1.1 {status}\r\n{location}Content-Length: {len(body)}\r\n"
                    "Content-Type: text/html\r\nConnection: close\r\n\r\n"
                ).encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    bases.append(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    url = bases[0] + "/s"
    parent = CrawlContext(CrawlLimits(max_url_bytes=maximum), clock=time.monotonic)

    class Guard:
        async def decide(self, *args, **kwargs):
            return SimpleNamespace(allowed=True)

    async def change_final(page):
        if boundary == "final":
            target = "/" + "x" * (maximum - len(bases[0]))
            await page.add_init_script(
                "document.addEventListener('DOMContentLoaded', () => history.replaceState(null, '', "
                + repr(target)
                + "));"
            )

    plan = ArticlePlan.from_routing_plan(
        SimpleNamespace(url=url, domain="127.0.0.1"), {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5}
    )
    if enhanced:
        plan = plan.for_enhanced_browser([])
    browser = GuardedArticleBrowser(
        egress_guard=Guard(),
        context=RuntimeRequestContext(),
        transport_decision=_legacy_allowed,
        stealth_hook=change_final,
    )
    browser.bind_parent(parent, RuntimeRequestContext())
    profile = replace(plan.browser, stealth_enabled=boundary == "final", stealth_wait_ms=0)
    try:
        if boundary == "redirect_exact":
            assert "bounded article" in await browser.acquire(url, profile)
            assert len(browser.final_url.encode()) == maximum and len(requests) == 2
        else:
            with pytest.raises(ParentStop, match="budget_exhausted"):
                await browser.acquire(url, profile)
            assert requests == ["/s"]
            signal = parent.stop_signal()
            await asyncio.sleep(0)
            assert signal.done(), "native URL stop must latch the shared crawl owner"
            with pytest.raises(ParentStop, match="budget_exhausted"):
                parent.reserve("network_requests").commit()
    finally:
        server.close()
        await server.wait_closed()
