"""Native enhanced-profile requests through the existing guarded browser/session."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tldw_Server_API.app.core.Web_Scraping.browser_transport import decide_browser_transport
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import GuardedArticleBrowser
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import ArticleFailure, ArticlePlan
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext

pytestmark = [pytest.mark.integration]


@pytest.mark.asyncio
async def test_native_cancel_keeps_lease_until_guard_exits_after_owned_browser_retirement(monkeypatch):
    from playwright.async_api import Browser

    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import _BrowserAcquisitionPool

    entered, release = asyncio.Event(), asyncio.Event()
    disconnected = []
    close = Browser.close

    async def observed_close(self, *args, **kwargs):
        await close(self, *args, **kwargs)
        disconnected.append(not self.is_connected())

    monkeypatch.setattr(Browser, "close", observed_close)
    pool = _BrowserAcquisitionPool(1)
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, url, **kwargs):
                if urlsplit(url).path == "/stream-body":
                    entered.set()
                    while not release.is_set():
                        try:
                            await asyncio.shield(release.wait())
                        except asyncio.CancelledError:
                            pass
                return SimpleNamespace(allowed=True)

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=bases[0] + "/stream-child", domain="127.0.0.1"),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(),
            context=RuntimeRequestContext(),
            transport_decision=_legacy_allowed,
            acquisition_pool=pool,
        )
        task = asyncio.create_task(browser.acquire(plan.url, plan.browser))
        try:
            await asyncio.wait_for(entered.wait(), 8)
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=3)
            assert task in done and task.cancelled()
            assert disconnected == [True]
            assert pool.active_count == 1 and not release.is_set()
            assert [path for _, path, _ in seen] == ["/stream-child"]
            release.set()
            for _ in range(100):
                if pool.active_count == 0:
                    break
                await asyncio.sleep(0.01)
            assert pool.active_count == 0
            assert [path for _, path, _ in seen] == ["/stream-child"]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_proxy", [False, True])
async def test_native_default_batch_preserves_settings_and_same_page_dom_fallback(monkeypatch, tmp_path, use_proxy):
    from dataclasses import replace

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _allowed_target

    original = article._build_default_dependencies

    class Guard:
        async def decide(self, *args, **kwargs):
            return SimpleNamespace(allowed=True)

    async def allowed(url, *args, **kwargs):
        return _allowed_target(url)

    def local_dependencies(cookies):
        dependencies = original(cookies)
        dependencies.browser._egress_guard = Guard()
        dependencies.browser._transport_decision = _legacy_allowed
        return replace(dependencies, evaluate_target=allowed)

    monkeypatch.setattr(article, "_build_default_dependencies", local_dependencies)
    monkeypatch.setattr(http_client, "PROXY_ALLOWLIST", {"127.0.0.1"})
    async with _sites() as (bases, seen):
        target = "http://public.example/dom" if use_proxy else bases[0] + "/dom"
        scraper = enhanced.EnhancedWebScraper(
            config={
                "max_concurrent": 1,
                "max_rps": 10000,
                "max_rpm": 10000,
                "max_rph": 10000,
                "web_scraper_respect_robots": False,
                "web_scraper_retry_count": 1,
                "web_scraper_retry_timeout": 5,
            }
        )
        scraper.cookie_manager = enhanced.CookieManager(tmp_path / "cookies.json")
        scraper.deduplicator = enhanced.ContentDeduplicator(tmp_path / "dedup.json")

        async def close_legacy_capability():
            pass

        scraper._browser = SimpleNamespace(close=close_legacy_capability)
        scraper.cookie_manager.add_cookies(urlsplit(target).netloc, [{"name": "sid", "value": "stored"}])
        resolve = scraper._resolve_scrape_plan

        def routed(url):
            plan, backend, handler = resolve(url)
            return (
                replace(
                    plan,
                    extra_headers={"X-Route": "route"},
                    cookies={"sid": "route"},
                    proxies={"http": bases[0]} if use_proxy else {},
                ),
                backend,
                handler,
            )

        scraper._resolve_scrape_plan = routed
        scraper._extract_from_html_with_pipeline = lambda *args, **kwargs: {
            "extraction_successful": False,
            "content": "",
            "extraction_trace": [{"strategy": "pipeline", "status": "failed"}],
        }
        try:
            results = await scraper.scrape_multiple(
                [target],
                method="playwright",
                custom_cookies=[{"name": "sid", "value": "explicit"}],
                custom_headers={"Authorization": "secret"},
            )
        finally:
            await scraper.stop()
        assert len(results) == 1 and results[0]["extraction_successful"]
        data = results[0]
        assert data["content"] == "Visible longest content\nSecond rendered line"
        assert (data["title"], data["author"], data["date"]) == ("Native title", "Native author", "2026-11-02")
        assert data["method"] == data["extraction_strategy"] == "playwright"
        assert data["extraction_trace"][-1]["reason"] == "fallback_extracted"
        first = next(h for _, path, h in seen if urlsplit(path).path == "/dom")
        assert first.get("authorization") == "secret" and first.get("x-route") == "route"
        assert "sid=route" in first.get("cookie", "") and "sid=explicit" not in first["cookie"]
        if use_proxy:
            assert [path for _, path, _ in seen if path.startswith("http:")] == [target]
            assert all(
                "authorization" not in h and "x-route" not in h and "cookie" not in h
                for _, path, h in seen
                if path != target
            )
        else:
            assert [path for _, path, _ in seen] == ["/dom"]


@pytest.mark.asyncio
async def test_native_same_owned_document_captures_bounded_live_dom_snapshot():
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, *args, **kwargs):
                return SimpleNamespace(allowed=True)

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=bases[0] + "/dom", domain="127.0.0.1"),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
        )
        html = await browser.acquire(plan.url, plan.browser)
        assert "THIS HIDDEN" in html
        assert dict(browser._rendered_fallback) == {
            "title": "Native title",
            "author": "Native author",
            "date": "2026-11-02",
            "content": "Visible longest content\nSecond rendered line",
        }
        with pytest.raises(TypeError):
            browser._rendered_fallback["content"] = "mutation"
        assert [path for _, path, _ in seen] == ["/dom"]


@asynccontextmanager
async def _sites(*, stream=None, wire=None):
    seen, bases = [], []

    async def serve(reader, writer):
        try:
            try:
                raw = (await reader.readuntil(b"\r\n\r\n")).decode("latin1").split("\r\n")
            except asyncio.IncompleteReadError as error:
                if not error.partial:
                    return  # Chromium may close an unused preconnect socket.
                raise
            path = raw[0].split()[1]
            headers = {k.lower(): v.strip() for field in raw[1:] if ":" in field for k, v in [field.split(":", 1)]}
            seen.append((writer.get_extra_info("sockname")[1], path, headers))
            path = urlsplit(path).path
            redirects = {
                "/start": bases[0] + "/step",
                "/step": bases[0] + "/done",
                "/cross": bases[1] + "/return",
                "/return": bases[0] + "/done",
                "/host": bases[1].replace("127.0.0.1", "localhost") + "/done",
                "/frame-redirect": bases[1].replace("127.0.0.1", "localhost") + "/deny",
                "/worker-redirect": bases[0] + "/deny",
                "/denied": bases[1] + "/deny",
            }
            body = b"<html><main>article preserved</main></html>"
            mime, status, location = "text/html", "200 OK", ""
            if path in redirects:
                status, location = "302 Found", f"Location: {redirects[path]}\r\n"
            elif path == "/children":
                body = (
                    f'<html><iframe src="{bases[1].replace("127.0.0.1", "localhost")}/frame"></iframe>'
                    '<img src="/image"><script>new Worker("/worker.js");window.open("/popup");</script></html>'
                ).encode()
            elif path == "/websocket":
                body = f'<html>article preserved<script>new WebSocket("{bases[0].replace("http:", "ws:")}/deny");</script></html>'.encode()
            elif path == "/bad-frame":
                body = (
                    f'<html><iframe src="{bases[1].replace("127.0.0.1", "localhost")}/bad-frame-body"></iframe></html>'
                ).encode()
            elif path in {"/popup-chain", "/popup-denied"}:
                target = "/cross" if path == "/popup-chain" else "/denied"
                body = f'<html>article preserved<script>window.open("{target}");</script></html>'.encode()
            elif path == "/large-child":
                body = b'<html>article preserved<script>window.open("/large-body");</script></html>'
            elif path == "/large-frame":
                body = f'<html><iframe src="{bases[1].replace("127.0.0.1", "localhost")}/large-body"></iframe></html>'.encode()
            elif path == "/large-body":
                body = b"<html>" + b"x" * 2_097_152 + b"</html>"
            elif path == "/stream-child":
                body = (
                    f'<html>article preserved<iframe src="{bases[1].replace("127.0.0.1", "localhost")}/stream-body"></iframe></html>'
                ).encode()
            elif path == "/stream-body":
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 4194304\r\n\r\n<html><body>"
                )
                await writer.drain()
                stream.started.set()
                await asyncio.sleep(0.1)
                writer.write(b"x" * 1048576)
                await writer.drain()
                await reader.read()
                stream.peer_closed.set()
                await stream.release.wait()
                return
            elif path == "/dom":
                body = (
                    '<html><head><title>Native title</title><meta name="author" content="Native author">'
                    '<meta property="article:published_time" content="2026-11-02"></head><body>'
                    '<main>short</main><article>medium text</article><div class="content">'
                    '<span style="display:none">THIS HIDDEN TEXT MUST NOT WIN ' + "x" * 1000 + "</span>"
                    'Visible longest content<br>Second rendered line</div><div class="content">tiny</div>'
                    "</body></html>"
                ).encode()
            elif path == "/bad-frame-body":
                body = b'<html><img src="/frame-redirect"></html>'
            elif path == "/bad-worker":
                body = b'<html><script>new Worker("/bad-worker.js");</script></html>'
            elif path == "/frame":
                body = b'<html><img src="/frame-image"></html>'
            elif path == "/worker.js":
                mime, body = "application/javascript", b'fetch("/worker-fetch");'
            elif path == "/bad-worker.js":
                mime, body = "application/javascript", b'fetch("/worker-redirect");'
            response = (
                f"HTTP/1.1 {status}\r\n{location}Content-Type: {mime}\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            ).encode() + body
            if wire is not None:
                wire.append((path, len(response)))
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    servers = [await asyncio.start_server(serve, "127.0.0.1", 0) for _ in range(2)]
    bases[:] = [f"http://127.0.0.1:{s.sockets[0].getsockname()[1]}" for s in servers]
    try:
        yield bases, seen
    finally:
        for server in servers:
            server.close()
            await server.wait_closed()


def _legacy_allowed():
    return decide_browser_transport(
        configured_mode="url_guarded", auth_mode="single_user", outbound_policy_mode="compat"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/start", "/cross", "/host", "/denied", "/children", "/bad-frame", "/bad-worker", "/websocket"]
)
async def test_native_enhanced_settings_and_guarded_dispatch(path):
    async with _sites() as (bases, seen):
        checks = []

        class Guard:
            async def decide(self, url, **kwargs):
                checks.append(url)
                return SimpleNamespace(allowed=urlsplit(url).path != "/deny")

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(
                url=bases[0] + path,
                domain="127.0.0.1",
                extra_headers={"Authorization": "secret", "X-Private": "private"},
                cookies={"route": "route-cookie"},
            ),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser(
            [
                {"name": "sid", "value": "explicit", "url": bases[0] + "/"},
                {"name": "narrow", "value": "private-path", "domain": "127.0.0.1", "path": "/private"},
            ]
        )
        browser = GuardedArticleBrowser(
            egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
        )
        if path in {"/denied", "/bad-frame", "/bad-worker", "/websocket"}:
            with pytest.raises(ArticleFailure):
                await browser.acquire(plan.url, plan.browser)
            assert not any(p == "/deny" for _, p, _ in seen)
            assert any(url.endswith("/deny") for url in checks)
            return
        try:
            html = await browser.acquire(plan.url, plan.browser)
        except ArticleFailure as failure:
            pytest.fail(f"Native acquisition failed: {failure.code}/{failure.stage}; paths={[p for _, p, _ in seen]}")
        assert "article preserved" in html or path == "/children"
        first = next(h for _, p, h in seen if p == path)
        assert first.get("authorization") == "secret" and first.get("x-private") == "private"
        assert "sid=explicit" in first.get("cookie", "") and "route=route-cookie" in first["cookie"]
        assert "narrow=" not in first["cookie"]
        if path == "/start":
            assert browser.final_url == bases[0] + "/done"
            assert all(h.get("authorization") == "secret" for _, p, h in seen if p in {"/step", "/done"})
        elif path in {"/cross", "/host"}:
            assert all(
                "authorization" not in h and "x-private" not in h for _, p, h in seen if p in {"/return", "/done"}
            )
            final = next(h for _, p, h in seen if p == "/done")
            # Browser cookies use host/path rules, not a manually invented port scope.
            assert ("sid=explicit" in final.get("cookie", "")) is (path == "/cross")
        else:
            assert {"/frame", "/frame-image", "/image", "/worker-fetch", "/popup"} <= {p for _, p, _ in seen}
            assert all(
                any(url.endswith(p) for url in checks) for p in ("/frame", "/frame-image", "/image", "/worker-fetch")
            )
            assert all(
                "authorization" not in h and "x-private" not in h for _, p, h in seen if p in {"/frame", "/frame-image"}
            )
            assert all(
                h.get("authorization") == "secret" for _, p, h in seen if p in {"/image", "/worker-fetch", "/popup"}
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/popup-chain", "/popup-denied"])
async def test_native_popup_first_request_and_redirect_boundaries(path):
    async with _sites() as (bases, seen):
        checks = []

        class Guard:
            async def decide(self, url, **kwargs):
                checks.append(url)
                return SimpleNamespace(allowed=urlsplit(url).path != "/deny")

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(
                url=bases[0] + path,
                domain="127.0.0.1",
                extra_headers={"Authorization": "secret", "X-Private": "private"},
            ),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
        )
        if path == "/popup-denied":
            failure = None
            try:
                await browser.acquire(plan.url, plan.browser)
            except ArticleFailure as error:
                failure = error
            assert failure is not None, {
                "checks": [urlsplit(url).path for url in checks],
                "received": [p for _, p, _ in seen],
            }
            assert any(url.endswith("/deny") for url in checks)
            assert not any(p == "/deny" for _, p, _ in seen)
        else:
            assert "article preserved" in await browser.acquire(plan.url, plan.browser)
            assert {"/cross", "/return", "/done"} <= {p for _, p, _ in seen}
            assert all(
                "authorization" not in h and "x-private" not in h for _, p, h in seen if p in {"/return", "/done"}
            )
        initial = next(h for _, p, h in seen if p in {"/cross", "/denied"})
        assert initial["authorization"] == "secret" and initial["x-private"] == "private"


@pytest.mark.asyncio
async def test_native_selected_proxy_is_used_without_fabricated_attestation(monkeypatch):
    from tldw_Server_API.app.core import http_client

    monkeypatch.setattr(http_client, "PROXY_ALLOWLIST", {"127.0.0.1"})
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, url, **kwargs):
                return SimpleNamespace(allowed=True)

        url = "http://public.example/article"
        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(
                url=url, domain="public.example", proxies={"http": bases[0]}, extra_headers={"X-Custom": "proxy-test"}
            ),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
        )
        assert "article preserved" in await browser.acquire(url, plan.browser)
        actual = next(headers for _, path, headers in seen if path == url)
        assert actual["x-custom"] == "proxy-test" and actual["host"] == "public.example"
        assert browser.transport_capability()["dns_peer_attested"] is False


@pytest.mark.asyncio
async def test_native_crawl_parent_counts_each_transport_once_and_child_transfer():
    import time

    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, url, **kwargs):
                return SimpleNamespace(allowed=True)

        def request(path):
            plan = ArticlePlan.from_routing_plan(
                SimpleNamespace(url=bases[0] + path, domain="127.0.0.1"),
                {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
            ).for_enhanced_browser([])
            browser = GuardedArticleBrowser(
                egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
            )
            return plan, browser

        plan, browser = request("/start")
        parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
        browser.bind_parent(parent, RuntimeRequestContext())
        await browser.acquire(plan.url, plan.browser)
        assert parent.snapshot().network_requests == len(seen)
        assert parent.snapshot().transfer_bytes > 0
        plan, browser = request("/large-child")
        parent = CrawlContext(CrawlLimits(transfer_bytes=1024), clock=time.monotonic)
        browser.bind_parent(parent, RuntimeRequestContext())
        with pytest.raises(ParentStop, match="budget_exhausted"):
            await browser.acquire(plan.url, plan.browser)
        assert any(path == "/large-body" for _, path, _ in seen)


@pytest.mark.asyncio
async def test_native_browser_scope_guard_is_installed_before_context_creation(monkeypatch):
    from playwright.async_api import Browser

    enabled = []
    original_session, original_context = Browser.new_browser_cdp_session, Browser.new_context

    async def observed_session(self):
        session = await original_session(self)
        send = session.send

        async def observed_send(method, params=None):
            result = await send(method, params)
            if method == "Fetch.enable":
                enabled.append(True)
            return result

        monkeypatch.setattr(session, "send", observed_send)
        return session

    async def observed_context(self, **kwargs):
        assert enabled == [True], "this acquisition's browser guard must precede context I/O"
        return await original_context(self, **kwargs)

    monkeypatch.setattr(Browser, "new_browser_cdp_session", observed_session)
    monkeypatch.setattr(Browser, "new_context", observed_context)
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, url, **kwargs):
                return SimpleNamespace(allowed=True)

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=bases[0] + "/start", domain="127.0.0.1"),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
        )
        assert "article preserved" in await browser.acquire(plan.url, plan.browser)
        assert enabled == [True] and len(seen) == 3


@pytest.mark.asyncio
async def test_native_child_stream_budget_stops_before_held_eof_and_physically_drains(monkeypatch):
    import time

    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import (
        _BrowserAcquisitionPool,
        _BrowserTransferLedger,
    )
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

    chunks = []
    original = _BrowserTransferLedger.on_request_data

    def observed(self, session, event):
        chunks.append((session, event.get("encodedDataLength") or event.get("dataLength", 0)))
        return original(self, session, event)

    monkeypatch.setattr(_BrowserTransferLedger, "on_request_data", observed)

    stream = SimpleNamespace(started=asyncio.Event(), peer_closed=asyncio.Event(), release=asyncio.Event())
    async with _sites(stream=stream) as (bases, seen):

        class Guard:
            async def decide(self, *args, **kwargs):
                return SimpleNamespace(allowed=True)

        pool = _BrowserAcquisitionPool(1)
        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=bases[0] + "/stream-child", domain="127.0.0.1"),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(),
            context=RuntimeRequestContext(),
            transport_decision=_legacy_allowed,
            acquisition_pool=pool,
        )
        parent = CrawlContext(CrawlLimits(transfer_bytes=1024), clock=time.monotonic)
        browser.bind_parent(parent, RuntimeRequestContext())
        task = asyncio.create_task(browser.acquire(plan.url, plan.browser))
        try:
            await asyncio.wait_for(stream.started.wait(), 8)
            with pytest.raises(ParentStop, match="budget_exhausted"):
                await asyncio.wait_for(asyncio.shield(task), 8)
            await asyncio.wait_for(stream.peer_closed.wait(), 2)
            assert not stream.release.is_set(), "EOF remains held until physical-drain assertions"
            assert any(session != "main" and size > 1024 for session, size in chunks)
            assert 0 < parent.snapshot().transfer_bytes <= 1024
            assert pool.active_count == 0
            assert [path for _, path, _ in seen] == ["/stream-child", "/stream-body"]
        finally:
            stream.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/done", "/large-frame"])
async def test_native_main_transfer_cumulative_eof_matches_actual_wire_without_duplicate_session(path):
    import time

    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext

    wire = []
    async with _sites(wire=wire) as (bases, seen):

        class Guard:
            async def decide(self, *args, **kwargs):
                return SimpleNamespace(allowed=True)

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=bases[0] + path, domain="127.0.0.1"),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(), context=RuntimeRequestContext(), transport_decision=_legacy_allowed
        )
        parent = CrawlContext(CrawlLimits(), clock=time.monotonic)
        browser.bind_parent(parent, RuntimeRequestContext())
        await browser.acquire(plan.url, plan.browser)
        assert len(seen) == parent.snapshot().network_requests == (1 if path == "/done" else 2)
        assert parent.snapshot().transfer_bytes == sum(size for _, size in wire)
