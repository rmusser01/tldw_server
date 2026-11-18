"""Native acquisition regressions for PR3436, using production egress policy."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration import article
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import GuardedArticleBrowser
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
    ArticleLimits,
    ArticlePlan,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.policy.probe import DefaultProbeEgressGuard
from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser import GuardedPlaywrightBrowserProbe
from tldw_Server_API.app.core.Web_Scraping.preflight.context import PreflightRuntimeControls
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import BrowserProbeOptions
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.tests.WebScraping.integration.test_phase5_enhanced_browser_native import (
    _legacy_allowed,
    _sites,
)
from tldw_Server_API.tests.WebScraping.integration.test_phase5_enhanced_browser_native import (
    native_batch_network_policy as native_batch_network_policy,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@asynccontextmanager
async def site(status=200):
    seen = []
    base = ""

    async def serve(reader, writer):
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            path = raw.split(b" ")[1].decode()
            seen.append(path)
            body = (
                f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{base}/article</loc></url></urlset>'
                if path == "/sitemap.xml"
                else "<html><head><title>Kept</title></head><body><main>Visible fallback content</main></body></html>"
            ).encode()
            headers, response_status = "", status
            if path.startswith("/redirect"):
                target = base + "/" + "a" * (80 - len(base) - 1 + int(path.endswith("over")))
                headers, response_status = f"Location: {target}\r\n", 302
            if path == "/expanded":
                body = b"""<html><body><script>
                  document.body.appendChild(document.createTextNode('x'.repeat(16777217)));
                  Object.defineProperty(Element.prototype, 'outerHTML', {get: () => '<html>forged</html>'});
                  window.XMLSerializer = class {serializeToString() {return '';}};
                  String.prototype[Symbol.iterator] = function* () {yield 'x';};
                </script></body></html>"""
            writer.write(
                f"HTTP/1.1 {response_status} Test\r\n{headers}Content-Type: text/html\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        yield base, seen
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("status", [403, 404, 500])
async def test_enhanced_navigation_rejects_error_document_without_refetch(status, native_batch_network_policy):
    async with site(status) as (base, seen):
        native_batch_network_policy([base])
        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=base + "/article", domain="127.0.0.1"), {"web_scraper_retry_count": 1}
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=DefaultProbeEgressGuard(),
            context=RuntimeRequestContext(),
            transport_decision=_legacy_allowed,
            cleanup_grace_s=5,
        )
        with pytest.raises(ArticleFailure) as caught:
            await browser.acquire(plan.url, plan.browser)
        assert (caught.value.stage, seen, getattr(browser, "_rendered_fallback", None)) == (
            "navigation",
            ["/article"],
            None,
        )


@pytest.mark.parametrize("enhanced_profile", [False, True])
async def test_missing_navigation_response_is_rejected_only_for_enhanced(
    monkeypatch, native_batch_network_policy, enhanced_profile
):
    from playwright.async_api import Page

    goto = Page.goto

    async def missing_response(page, *args, **kwargs):
        await goto(page, *args, **kwargs)
        return None

    monkeypatch.setattr(Page, "goto", missing_response)
    async with site() as (base, seen):
        native_batch_network_policy([base])
        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=base + "/article", domain="127.0.0.1"), {"web_scraper_retry_count": 1}
        )
        if enhanced_profile:
            plan = plan.for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=DefaultProbeEgressGuard(),
            context=RuntimeRequestContext(),
            transport_decision=_legacy_allowed,
            cleanup_grace_s=5,
        )
        if enhanced_profile:
            with pytest.raises(ArticleFailure) as caught:
                await browser.acquire(plan.url, plan.browser)
            assert caught.value.stage == "navigation" and caught.value.code == "fetch_error"
        else:
            assert "Visible fallback content" in await browser.acquire(plan.url, plan.browser)
        assert seen == ["/article"]


@pytest.mark.parametrize("status", [403, 404, 500])
@pytest.mark.parametrize("entrypoint", ["batch", "recursive"])
async def test_native_enhanced_terminal_status_never_reaches_extraction(
    monkeypatch, tmp_path, native_batch_network_policy, status, entrypoint
):
    build = article._build_default_dependencies

    def dependencies(cookies):
        value = build(cookies)
        value.browser._transport_decision = _legacy_allowed
        value.browser._cleanup_grace_s = 5
        return value

    monkeypatch.setattr(article, "_build_default_dependencies", dependencies)
    async with site(status) as (base, seen):
        native_batch_network_policy([base])
        scraper = enhanced.EnhancedWebScraper(
            config={
                "web_scraper_default_backend": "playwright",
                "web_scraper_respect_robots": False,
                "web_scraper_retry_count": 1,
                "max_rps": 10000,
                "max_rpm": 10000,
                "max_rph": 10000,
            }
        )
        scraper.cookie_manager = enhanced.CookieManager(tmp_path / "cookies.json")
        scraper.deduplicator = enhanced.ContentDeduplicator(tmp_path / "dedup.json")
        extracted = []

        def extract(*args, **kwargs):
            extracted.append(True)
            return {"content": "must not be published", "extraction_successful": True}

        async def close():
            pass

        scraper._browser = SimpleNamespace(close=close)
        scraper._extract_from_html_with_pipeline = extract
        try:
            if entrypoint == "batch":
                report = await scraper._batch_report([base + "/article"], method="playwright")
                assert len(report.articles) == 1 and report.articles[0]["extraction_successful"] is False
            else:
                report = await scraper._crawl_report(
                    base + "/article", profile="enhanced_recursive", options={"max_pages": 1}
                )
                assert report.articles == () and report.outcome.state == "completed"
            # The established queue permits three retries; each attempt navigates once.
            assert seen == ["/article"] * 4 and extracted == []
        finally:
            await scraper.stop()


@pytest.mark.parametrize("over", [False, True])
async def test_native_preflight_http_redirect_byte_boundary_prevents_late_dispatch(native_batch_network_policy, over):
    from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http import GuardedHttpProbe
    from tldw_Server_API.app.core.Web_Scraping.preflight.probes import ProbeHttpRequest

    async with site() as (base, seen):
        native_batch_network_policy([base])
        parent = CrawlContext(CrawlLimits(max_url_bytes=80), clock=time.monotonic)
        controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
        probe = GuardedHttpProbe(controls=controls, egress_guard=DefaultProbeEgressGuard())
        try:
            if over:
                with pytest.raises(ParentStop, match="budget_exhausted"):
                    await probe.get(ProbeHttpRequest(base + "/redirect-over"))
                with pytest.raises(ParentStop):
                    await probe.get(ProbeHttpRequest(base + "/article"))
                assert seen == ["/redirect-over"] and parent.snapshot().network_requests == 1
            else:
                response = await probe.get(ProbeHttpRequest(base + "/redirect"))
                assert response.status == 200 and len(response.url.encode()) == 80
                assert len(seen) == parent.snapshot().network_requests == 2
        finally:
            await controls.close()


async def test_native_oversized_hostile_preflight_cannot_fail_open_to_article(monkeypatch, native_batch_network_policy):
    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult
    from tldw_Server_API.app.core.Web_Scraping.preflight import facade, runner
    from tldw_Server_API.app.core.Web_Scraping.preflight.options import PreflightOptions
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _allowed_target, _harness

    async def noop(*args, **kwargs):
        return {}

    def context(*args, **kwargs):
        from tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http import GuardedHttpProbe, HttpxProbeTransport

        value = facade.build_execution_context(*args, **kwargs)
        value.browser._transport_decision = _legacy_allowed
        return replace(
            value,
            http=GuardedHttpProbe(
                controls=value.controls,
                egress_guard=value.egress_guard,
                curl_transport=HttpxProbeTransport(value.controls),
            ),
        )

    async def target(url, **kwargs):
        return _allowed_target(url)

    monkeypatch.setattr(runner, "_check_robots_txt", noop)
    monkeypatch.setattr(runner, "_analyze_tls_fingerprint", noop)
    harness = _harness()
    dependencies = replace(
        harness.dependencies,
        evaluate_target=target,
        build_preflight_context=context,
        run_preflight=facade.run_preflight,
        preflight_options=lambda _: PreflightOptions(enabled=True, timeout_s=20),
    )
    async with site() as (base, seen):
        native_batch_network_policy([base])
        parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
        handle = article.start_page(
            base + "/expanded",
            None,
            controls=parent,
            request_context=RuntimeRequestContext(),
            dependencies=dependencies,
            discover=lambda _, request, final: DiscoveryResult(request, final, ()),
        )
        try:
            with pytest.raises(ParentStop, match="budget_exhausted"):
                await handle.result
        finally:
            await asyncio.wait_for(asyncio.shield(handle.drained), 10)
        assert seen == ["/expanded", "/expanded"] and parent.snapshot().network_requests == 2
        assert harness.fetch.requests == harness.browser.calls == [] and harness.extract.call_count == 0


@pytest.mark.parametrize("profile", ["enhanced_recursive", "enhanced_sitemap", "enhanced_url_level"])
async def test_enhanced_crawl_uses_instance_routing_pipeline_and_same_page_fallback(
    monkeypatch, tmp_path, native_batch_network_policy, profile
):
    build = article._build_default_dependencies

    def dependencies(cookies):
        value = build(cookies)
        value.browser._transport_decision = _legacy_allowed
        value.browser._cleanup_grace_s = 5
        return value

    monkeypatch.setattr(article, "_build_default_dependencies", dependencies)
    async with site() as (base, seen):
        native_batch_network_policy([base])
        scraper = enhanced.EnhancedWebScraper(
            config={
                "web_scraper_respect_robots": False,
                "web_scraper_retry_count": 1,
                "max_rps": 10000,
                "max_rpm": 10000,
                "max_rph": 10000,
            }
        )
        scraper.cookie_manager = enhanced.CookieManager(tmp_path / "cookies.json")
        scraper.deduplicator = enhanced.ContentDeduplicator(tmp_path / "dedup.json")

        async def close():
            pass

        scraper._browser = SimpleNamespace(close=close)
        resolver, routed, extracted = scraper._resolve_scrape_plan, [], []

        def route(url):
            routed.append(url)
            plan, _, handler = resolver(url)
            return replace(plan, backend="playwright"), "playwright", handler

        def extract(html, url, **kwargs):
            extracted.append((url, kwargs["method_label"]))
            return {"extraction_successful": False, "content": "", "extraction_trace": []}

        scraper._resolve_scrape_plan = route
        scraper._extract_from_html_with_pipeline = extract
        seed = base + ("/sitemap.xml" if profile.endswith("sitemap") else "/article")
        try:
            report = await scraper._crawl_report(
                seed,
                profile=profile,
                options={"max_pages": 1, "max_depth": 1, "level": 2, "allow_llm_extraction": False},
            )
        finally:
            await scraper.stop()
        assert report.outcome.state == "completed"
        assert [item["content"] for item in report.articles] == ["Visible fallback content"]
        assert routed == [base + "/article"] and extracted == [(base + "/article", "playwright")]
        assert seen == (["/sitemap.xml", "/article"] if profile.endswith("sitemap") else ["/article"])


@pytest.mark.parametrize("extra", [0, 1])
@pytest.mark.parametrize("hostile", [False, True])
async def test_preflight_rendered_html_utf8_bound_uses_isolated_world(extra, hostile, native_batch_network_policy):
    async with site() as (base, seen):
        native_batch_network_policy([base])
        parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
        controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
        probe = GuardedPlaywrightBrowserProbe(
            controls=controls, egress_guard=DefaultProbeEgressGuard(), transport_decision=_legacy_allowed
        )
        try:
            async with probe.open_page(BrowserProbeOptions()) as page:
                await page.goto(base + "/article", wait_until="load", timeout_ms=5000)
                size = ArticleLimits().max_article_bytes
                await page.evaluate(
                    """({size, extra, hostile}) => {
                  document.documentElement.innerHTML = '<head></head><body></body>';
                  const overhead = new TextEncoder().encode(document.documentElement.outerHTML).length;
                  document.body.textContent = '\u00e9'.repeat(Math.floor((size - overhead)/2)) + 'x'.repeat((size-overhead)%2 + extra);
                  if (hostile) {
                    Object.defineProperty(Element.prototype, 'outerHTML', {get: () => '<html>forged</html>'});
                    window.XMLSerializer = class {serializeToString() {return '';}};
                    String.prototype[Symbol.iterator] = function* () {yield 'x';};
                    window.TextEncoder = class {encode() {return new Uint8Array(0);}};
                  }
                }""",
                    {"size": size, "extra": extra, "hostile": hostile},
                )
                if extra:
                    with pytest.raises(ParentStop, match="budget_exhausted"):
                        await page.content()
                    with pytest.raises(ParentStop):
                        controls.cap_timeout(None)
                else:
                    html = await page.content()
                    assert len(html.encode("utf-8")) == size and "forged" not in html
        finally:
            await controls.close()
        assert seen == ["/article"] and parent.snapshot().network_requests == 1


@pytest.mark.parametrize("path", ["/stream-child", "/stream-body"])
async def test_preflight_child_transfer_stops_before_held_eof_and_native_exit(
    monkeypatch, native_batch_network_policy, path
):
    from playwright.async_api import Browser

    closed = asyncio.Event()
    close = Browser.close

    async def observed_close(browser, *args, **kwargs):
        await close(browser, *args, **kwargs)
        closed.set()

    monkeypatch.setattr(Browser, "close", observed_close)
    stream = SimpleNamespace(started=asyncio.Event(), peer_closed=asyncio.Event(), release=asyncio.Event())
    async with _sites(stream=stream) as (bases, seen):
        native_batch_network_policy(bases)
        monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "127.0.0.1,localhost")
        parent = CrawlContext(CrawlLimits(transfer_bytes=16384), clock=time.monotonic)
        controls = PreflightRuntimeControls(RuntimeRequestContext(), parent=parent)
        probe = GuardedPlaywrightBrowserProbe(
            controls=controls, egress_guard=DefaultProbeEgressGuard(), transport_decision=_legacy_allowed
        )
        try:
            with pytest.raises(ParentStop, match="budget_exhausted"):
                async with probe.open_page(BrowserProbeOptions()) as page:
                    await page.goto(bases[0] + path, wait_until="load", timeout_ms=10000)
                    await page.wait_for_load_state("networkidle", timeout_ms=10000)
            await asyncio.wait_for(closed.wait(), 5)
            await asyncio.wait_for(stream.peer_closed.wait(), 5)
            assert not stream.release.is_set() and stream.started.is_set()
            assert [received for _, received, _ in seen] == (
                ["/stream-child", "/stream-body"] if path == "/stream-child" else ["/stream-body"]
            )
        finally:
            stream.release.set()
            await controls.close()
