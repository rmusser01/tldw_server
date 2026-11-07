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
@pytest.mark.parametrize("close_mode", ["failed", "held"])
async def test_native_failed_or_held_close_keeps_guard_and_lease_until_actual_browser_exit(monkeypatch, close_mode):
    from playwright.async_api import Browser

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as owner

    entered, release = asyncio.Event(), asyncio.Event()
    native, policies, detach_states = [], [], []
    original_close = Browser.close
    bind = owner._BrowserRequestPolicy.bind
    detach = owner._BrowserRequestPolicy._detach

    async def retained_bind(self, *args):
        policies.append(self)
        return await bind(self, *args)

    async def failed_or_held(self, *args, **kwargs):
        native.append(self)
        entered.set()
        if close_mode == "failed":
            raise RuntimeError("close rejected while browser still connected")
        await release.wait()
        await original_close(self, *args, **kwargs)

    async def observed_detach(session):
        detach_states.append(native[0].is_connected())
        await detach(session)

    monkeypatch.setattr(Browser, "close", failed_or_held)
    monkeypatch.setattr(owner._BrowserRequestPolicy, "bind", retained_bind)
    monkeypatch.setattr(owner._BrowserRequestPolicy, "_detach", staticmethod(observed_detach))
    pool = owner._BrowserAcquisitionPool(1)
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, url, **kwargs):
                return SimpleNamespace(allowed=urlsplit(url).path != "/deny")

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(url=bases[0] + "/done", domain="127.0.0.1"),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser([])
        browser = GuardedArticleBrowser(
            egress_guard=Guard(),
            context=RuntimeRequestContext(),
            transport_decision=_legacy_allowed,
            acquisition_pool=pool,
            cleanup_grace_s=0.1,
        )
        task = asyncio.create_task(browser.acquire(plan.url, plan.browser))
        evaluation = None
        try:
            await asyncio.wait_for(entered.wait(), 8)
            with pytest.raises(ArticleFailure, match="browser_error"):
                await asyncio.wait_for(asyncio.shield(task), 2)
            assert native[0].is_connected() and pool.active_count == 1 and detach_states == []
            assert len(native) == 1, "no new close retries or polling"
            assert pool.acquire(callback_capacity=2) is None
            paused = asyncio.Event()
            policies[0]._root.on(
                "Fetch.requestPaused",
                lambda event: paused.set() if urlsplit(event["request"]["url"]).path == "/deny" else None,
            )
            # Isolate the still-owned browser guard, rather than the earlier context route.
            await native[0].contexts[0].unroute_all(behavior="wait")
            evaluation = asyncio.create_task(
                native[0].contexts[0].pages[0].evaluate("fetch('/deny').catch(() => null)")
            )
            await asyncio.wait_for(paused.wait(), 2)
            assert not any(path == "/deny" for _, path, _ in seen)
            assert detach_states == [] and pool.active_count == 1
            if close_mode == "failed":
                await original_close(native[0])
            else:
                release.set()
            for _ in range(200):
                if pool.active_count == 0:
                    break
                await asyncio.sleep(0.01)
            assert not native[0].is_connected() and pool.active_count == 0
            assert detach_states and not any(detach_states)
            assert not any(path == "/deny" for _, path, _ in seen)
        finally:
            release.set()
            if native and native[0].is_connected():
                await original_close(native[0])
            if evaluation is not None:
                await asyncio.gather(evaluation, return_exceptions=True)
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/children", "/cookie-port", "/cross", "/popup-chain"])
async def test_native_raw_cookie_override_is_origin_scoped_without_losing_browser_jar(path):
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, *args, **kwargs):
                return SimpleNamespace(allowed=True)

        plan = ArticlePlan.from_routing_plan(
            SimpleNamespace(
                url=bases[0] + path,
                domain="127.0.0.1",
                extra_headers={"Cookie": "raw-secret=original", "Authorization": "secret"},
            ),
            {"web_scraper_retry_count": 1, "web_scraper_retry_timeout": 5},
        ).for_enhanced_browser(
            [
                {"name": "localjar", "value": "original", "url": bases[0] + "/"},
                {"name": "framejar", "value": "natural", "domain": "localhost", "path": "/"},
                {"name": "narrow", "value": "hidden", "domain": "localhost", "path": "/private"},
            ]
        )
        browser = GuardedArticleBrowser(
            egress_guard=Guard(),
            context=RuntimeRequestContext(),
            transport_decision=_legacy_allowed,
            cleanup_grace_s=5,
        )
        await browser.acquire(plan.url, plan.browser)
        original_requests = list(seen)
        initial = next(headers for _, received, headers in seen if received == path)
        assert initial.get("cookie") == "raw-secret=original"
        if path in {"/children", "/cookie-port"}:
            frames = [headers for _, received, headers in seen if received in {"/frame", "/frame-image"}]
            assert len(frames) == 2
            assert all(
                "raw-secret" not in headers.get("cookie", "") and "narrow" not in headers.get("cookie", "")
                for headers in frames
            )
            if path == "/cookie-port":
                assert all(headers.get("cookie") == "localjar=original" for headers in frames)
            else:
                # Compare actual Chromium third-party/SameSite selection; do not invent a jar override.
                from dataclasses import replace

                count = len(seen)
                control = replace(plan.browser, extra_headers={"Authorization": "secret"})
                await browser.acquire(plan.url, control)
                natural = [
                    headers.get("cookie")
                    for _, received, headers in seen[count:]
                    if received in {"/frame", "/frame-image"}
                ]
                assert [headers.get("cookie") for headers in frames] == natural
            assert all(
                headers.get("cookie") == "raw-secret=original"
                for _, received, headers in original_requests
                if received in {"/image", "/worker-fetch", "/popup"}
            )
        else:
            redirected = [headers for _, received, headers in seen if received in {"/return", "/done"}]
            assert len(redirected) == 2
            assert all(
                "raw-secret" not in headers.get("cookie", "") and "authorization" not in headers
                for headers in redirected
            )
            assert all(headers.get("cookie") == "localjar=original" for headers in redirected)
            if path == "/popup-chain":
                assert next(h for _, received, h in seen if received == "/cross")["cookie"] == "raw-secret=original"


@pytest.mark.asyncio
@pytest.mark.parametrize("hold_stage", ["guard", "continue", "early_continue_error"])
async def test_native_cancel_keeps_lease_until_guard_exits_after_owned_browser_retirement(monkeypatch, hold_stage):
    from playwright._impl._errors import TargetClosedError
    from playwright.async_api import Browser, Route

    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import _BrowserAcquisitionPool

    entered, release = asyncio.Event(), asyncio.Event()
    disconnected = []
    native_closed = asyncio.Event()
    close = Browser.close
    continue_ = Route.continue_
    abort = Route.abort

    async def held():
        entered.set()
        while not release.is_set():
            try:
                await asyncio.shield(release.wait())
            except asyncio.CancelledError:
                pass

    async def held_continue(self, *args, **kwargs):
        if urlsplit(self.request.url).path == "/stream-body":
            if hold_stage == "continue":
                await held()
            elif hold_stage == "early_continue_error":
                raise RuntimeError("continuation failed before browser exit")
        return await continue_(self, *args, **kwargs)

    async def early_abort(self, *args, **kwargs):
        if hold_stage == "early_continue_error" and urlsplit(self.request.url).path == "/stream-body":
            entered.set()
            raise TargetClosedError("page route unavailable before browser exit")
        return await abort(self, *args, **kwargs)

    async def observed_close(self, *args, **kwargs):
        if hold_stage == "early_continue_error":
            await release.wait()
        await close(self, *args, **kwargs)
        disconnected.append(not self.is_connected())
        native_closed.set()

    monkeypatch.setattr(Browser, "close", observed_close)
    monkeypatch.setattr(Route, "continue_", held_continue)
    monkeypatch.setattr(Route, "abort", early_abort)
    pool = _BrowserAcquisitionPool(1)
    async with _sites() as (bases, seen):

        class Guard:
            async def decide(self, url, **kwargs):
                if hold_stage == "guard" and urlsplit(url).path == "/stream-body":
                    await held()
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
            assert pool.active_count == 1 and not release.is_set()
            if hold_stage == "early_continue_error":
                assert disconnected == []
                assert [path for _, path, _ in seen] == ["/stream-child"]
                release.set()
            await asyncio.wait_for(native_closed.wait(), 8)
            assert disconnected == [True]
            if hold_stage != "early_continue_error":
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
            elif path == "/cookie-port":
                body = f'<html>article preserved<iframe src="{bases[1]}/frame"></iframe></html>'.encode()
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
        drained = asyncio.Event()
        release_lease = pool._release

        def observed_release(lease):
            release_lease(lease)
            drained.set()

        monkeypatch.setattr(pool, "_release", observed_release)
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
            await asyncio.wait_for(drained.wait(), 5)
            assert not stream.release.is_set(), "EOF remains held through actual pool release"
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
