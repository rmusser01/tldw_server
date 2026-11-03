"""Enhanced crawl browser settings stay distinct from direct capture defaults."""

import asyncio
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Web_Scraping.orchestration import article, crawl
from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import URL, _harness

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("already_disconnected", [False, True])
async def test_native_disconnection_observed_before_retirement_is_not_missed_after_close_rejects(already_disconnected):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    handlers, retired = {}, []

    class Native:
        connected = True

        def on(self, name, callback):
            handlers[name] = callback

        def remove_listener(self, name, callback):
            handlers.pop(name)

        def is_connected(self):
            return self.connected

        async def close(self):
            raise RuntimeError("exit was already observed")

    async def retire():
        retired.append(True)

    native = Native()
    outcome = browser._AcquisitionOutcome()
    native.connected = not already_disconnected
    outcome.watch_browser(native)
    native.connected = False
    if not already_disconnected:
        handlers["disconnected"]()
    pool = browser._BrowserAcquisitionPool(1)
    lease = pool.acquire(callback_capacity=2)
    lifecycle = browser._CallbackLifecycle(outcome, lease, lambda: None)
    owners = {
        "browser": browser._OwnedResource(native, "close", "browser"),
        "requests": browser._OwnedResource(SimpleNamespace(retire=retire), "close", "requests"),
    }
    assert await asyncio.wait_for(
        _adapter(_FakeBrowserRuntime(), _FakeGuard([]), cleanup_grace_s=0.02)._retire_browser_owner(
            owners, lifecycle, lease
        ),
        1,
    )
    lease.seal()
    lease.owner_finished()
    if already_disconnected:
        assert not outcome.native_browser_disconnected and retired == [] and pool.active_count == 1
        handlers["disconnected"]()
        for _ in range(100):
            if pool.active_count == 0:
                break
            await asyncio.sleep(0.01)
    assert retired == [True] and outcome.native_browser_disconnected and handlers == {} and pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sealed", "creation_failure", "unobserved"])
async def test_retirement_task_startup_disposes_listener_or_retains_native_work(monkeypatch, mode):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    handlers, calls = {}, []
    pool = browser._BrowserAcquisitionPool(1)
    lease = pool.acquire(callback_capacity=2)

    class Native:
        def on(self, name, callback):
            handlers[name] = callback

        def remove_listener(self, name, callback):
            handlers.pop(name)

        def is_connected(self):
            return False

        async def close(self):
            calls.append("native-closed")

    class Unobserved(asyncio.Task):
        def add_done_callback(self, *args, **kwargs):
            raise RuntimeError("completion observation unavailable")

    create = asyncio.create_task

    def create_task(operation, *, name=None):
        if name != "article-browser-owner-retirement":
            return create(operation, name=name)
        if mode == "creation_failure":
            raise RuntimeError("cannot create task")
        return Unobserved(operation, loop=asyncio.get_running_loop())

    async def retire():
        calls.append("retired")

    monkeypatch.setattr(asyncio, "create_task", create_task)
    if mode == "sealed":
        lease.seal()
    outcome = browser._AcquisitionOutcome()
    lifecycle = browser._CallbackLifecycle(outcome, lease, lambda: None)
    owners = {
        "browser": browser._OwnedResource(Native(), "close", "browser"),
        "requests": browser._OwnedResource(SimpleNamespace(retire=retire), "close", "requests"),
    }
    assert await _adapter(_FakeBrowserRuntime(), _FakeGuard([]))._retire_browser_owner(owners, lifecycle, lease)
    if mode == "unobserved":
        for _ in range(10):
            await asyncio.sleep(0)
        assert calls == ["native-closed", "retired"]
        assert all(task.done() for task in lease._tasks)
        lease.seal()
        lease.owner_finished()
        assert pool.active_count == 1, "unobservable task cannot refund its capacity"
    else:
        assert calls == []
        lease.seal()
        lease.owner_finished()
        assert pool.active_count == 0
    assert handlers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["failed", "held"])
async def test_live_native_after_close_failure_keeps_guard_graph_and_lease_until_actual_exit(close_mode):
    import weakref

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    calls, handlers = [], {}
    release_close = asyncio.Event()

    class Native:
        connected = True

        def on(self, event, callback):
            handlers[event] = callback

        def remove_listener(self, event, callback):
            handlers.pop(event, None)

        def is_connected(self):
            return self.connected

        async def close(self):
            calls.append("native-close")
            if close_mode == "failed":
                raise RuntimeError("native close rejected")
            await release_close.wait()

    class Requests:
        def close(self):
            calls.append("admission-closed")

        async def detach(self):
            calls.append("guard-detached")

        async def retire(self):
            assert not native.connected
            calls.append("retired")

    class Driver:
        async def stop(self):
            calls.append("driver-stopped")

    native, policy = Native(), Requests()
    retained = weakref.ref(policy)
    pool = browser._BrowserAcquisitionPool(1)
    lease = pool.acquire(callback_capacity=2)
    outcome = browser._AcquisitionOutcome()
    lifecycle = browser._CallbackLifecycle(outcome, lease, lambda: None)
    adapter = _adapter(_FakeBrowserRuntime(), _FakeGuard([]), cleanup_grace_s=0.02)
    resources = [
        browser._OwnedResource(native, "close", "browser"),
        browser._OwnedResource(policy, "close", "requests"),
        browser._OwnedResource(Driver(), "stop", "playwright"),
    ]
    try:
        assert await asyncio.wait_for(
            adapter._teardown(resources, context=None, http_routing_started=False, lifecycle=lifecycle, lease=lease),
            0.5,
        )
        lease.seal()
        lease.owner_finished()
        del policy, resources
        assert pool.active_count == 1 and retained() is not None
        assert calls == ["admission-closed", "native-close"] and not outcome.native_browser_disconnected
        assert pool.acquire(callback_capacity=2) is None
        native.connected = False
        handlers["disconnected"]()
        if close_mode == "held":
            for _ in range(3):
                await asyncio.sleep(0)
            assert pool.active_count == 1, "actual close-command work must also drain"
        release_close.set()
        for _ in range(100):
            if pool.active_count == 0:
                break
            await asyncio.sleep(0.01)
        assert pool.active_count == 0 and outcome.native_browser_disconnected
        assert calls == ["admission-closed", "native-close", "retired", "driver-stopped"]
        assert retained() is None
    finally:
        native.connected = False
        callback = handlers.get("disconnected")
        if callback:
            callback()
        release_close.set()


@pytest.mark.asyncio
async def test_continue_failure_uses_terminal_route_completion_after_native_exit():
    from types import SimpleNamespace

    from playwright._impl._errors import TargetClosedError

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    completed = []

    class Route:
        request = SimpleNamespace(url=URL)

        async def continue_(self):
            raise RuntimeError("in-flight continuation failed")

        async def abort(self):
            raise TargetClosedError("owned native browser exited")

        async def fallback(self):
            completed.append(True)

    outcome = browser._AcquisitionOutcome()
    outcome.native_browser_disconnected = True
    await _adapter(_FakeBrowserRuntime(), _FakeGuard([True]))._http_handler(Route(), outcome)
    assert completed == [True]


@pytest.mark.asyncio
async def test_terminal_route_error_waits_for_owned_native_exit_before_local_completion():
    from types import SimpleNamespace

    from playwright._impl._errors import TargetClosedError

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    handlers, completed = {}, []
    aborted = asyncio.Event()

    class Native:
        connected = True

        def on(self, name, callback):
            handlers[name] = callback

        def remove_listener(self, name, callback):
            handlers.pop(name)

        def is_connected(self):
            return self.connected

    class Route:
        request = SimpleNamespace(url=URL)

        async def continue_(self):
            raise RuntimeError("continuation failed before native retirement")

        async def abort(self):
            aborted.set()
            raise TargetClosedError("page closed before browser exit")

        async def fallback(self):
            completed.append(True)

    native, outcome = Native(), browser._AcquisitionOutcome()
    outcome.watch_browser(native)
    task = asyncio.create_task(_adapter(_FakeBrowserRuntime(), _FakeGuard([True]))._http_handler(Route(), outcome))
    try:
        await asyncio.wait_for(aborted.wait(), 1)
        assert not task.done() and completed == []
        native.connected = False
        handlers["disconnected"]()
        await asyncio.wait_for(task, 1)
        assert completed == [True]
    finally:
        native.connected = False
        handlers["disconnected"]()
        await asyncio.gather(task, return_exceptions=True)
        outcome.unwatch_browser(native)


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [False, True])
async def test_terminal_http_route_completion_requires_owned_browser_retirement(retired):
    from playwright._impl._errors import TargetClosedError

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    completed = []

    class Route:
        async def abort(self):
            raise TargetClosedError("native owner closed")

        async def fallback(self):
            completed.append("framework-completed-locally")

    outcome = browser._AcquisitionOutcome()
    outcome.native_browser_disconnected = retired
    adapter = _adapter(_FakeBrowserRuntime(), _FakeGuard([]))
    await adapter._abort_http(Route(), outcome, failure_stage="callback")
    assert completed == (["framework-completed-locally"] if retired else [])
    assert outcome.failure is not None


@pytest.mark.asyncio
async def test_failed_native_close_cannot_authorize_local_route_completion():
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    retired = []
    handlers = {}

    class Requests:
        def close(self):
            pass

        async def detach(self):
            pass

        async def retire(self):
            retired.append(True)

    class Native:
        def on(self, event, callback):
            handlers[event] = callback

        def remove_listener(self, event, callback):
            handlers.pop(event, None)

        async def close(self):
            raise RuntimeError("close did not acknowledge retirement")

        def is_connected(self):
            return False

    pool = browser._BrowserAcquisitionPool(1)
    lease = pool.acquire(callback_capacity=2)
    outcome = browser._AcquisitionOutcome()
    lifecycle = browser._CallbackLifecycle(outcome, lease, lambda: None)
    adapter = _adapter(_FakeBrowserRuntime(), _FakeGuard([]), cleanup_grace_s=0.02)
    failed = await adapter._teardown(
        [browser._OwnedResource(Requests(), "close", "requests"), browser._OwnedResource(Native(), "close", "browser")],
        context=None,
        http_routing_started=False,
        lifecycle=lifecycle,
        lease=lease,
    )
    lease.seal()
    lease.owner_finished()
    assert failed and not outcome.native_browser_disconnected and retired == []
    assert pool.active_count == 1
    handlers["disconnected"]()
    for _ in range(100):
        if pool.active_count == 0:
            break
        await asyncio.sleep(0.01)
    assert pool.active_count == 0 and retired == [True]


@pytest.mark.asyncio
async def test_retirement_authority_is_not_shared_between_acquisitions_on_same_adapter():
    from playwright._impl._errors import TargetClosedError

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _adapter, _FakeBrowserRuntime, _FakeGuard

    completed = []

    class Route:
        async def abort(self):
            raise TargetClosedError("closed route does not prove owner retirement")

        async def fallback(self):
            completed.append(True)

    adapter = _adapter(_FakeBrowserRuntime(), _FakeGuard([]))
    retired, live = browser._AcquisitionOutcome(), browser._AcquisitionOutcome()
    retired.native_browser_disconnected = True
    await adapter._abort_http(Route(), retired, failure_stage="callback")
    await adapter._abort_http(Route(), live, failure_stage="callback")
    assert completed == [True] and not live.native_browser_disconnected


@pytest.mark.asyncio
async def test_native_pause_after_admission_closes_is_aborted_under_same_callback_bound():
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser

    entered, release = asyncio.Event(), asyncio.Event()
    commands, handlers = [], {}

    class Root:
        def on(self, name, handler):
            handlers[name] = handler

        async def send(self, method, params=None):
            commands.append(method)
            if method == "Fetch.failRequest":
                entered.set()
                await release.wait()
            return {}

    async def shutdown():
        pass

    pool = browser._BrowserAcquisitionPool(1)
    lease = pool.acquire(callback_capacity=2)
    outcome = browser._AcquisitionOutcome()
    lifecycle = browser._CallbackLifecycle(outcome, lease, shutdown)
    policy = browser._BrowserRequestPolicy(URL, {}, None, outcome)
    await policy.bind(Root(), lifecycle, None, lease)
    policy.close()
    lifecycle.close_admission()
    handlers["Fetch.requestPaused"]({"requestId": "late-request"})
    try:
        await asyncio.wait_for(entered.wait(), 1)
        lease.seal()
        lease.owner_finished()
        assert pool.active_count == 1
        assert "Fetch.continueRequest" not in commands
    finally:
        release.set()
        await lifecycle.drain(1)
        lease.seal()
        lease.owner_finished()
    for _ in range(3):
        await asyncio.sleep(0)
    assert commands.count("Fetch.failRequest") == 1 and pool.active_count == 0


@pytest.mark.asyncio
async def test_late_browser_scope_guard_cannot_continue_after_admission_closes():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser

    entered, release = asyncio.Event(), asyncio.Event()
    commands = []

    async def decision(url):
        entered.set()
        await release.wait()
        return True

    async def send(method, params):
        commands.append(method)

    pool = browser._BrowserAcquisitionPool(1)
    lease = pool.acquire(callback_capacity=2)
    outcome = browser._AcquisitionOutcome()
    lifecycle = browser._CallbackLifecycle(outcome, lease, lambda: None)
    policy = browser._BrowserRequestPolicy(
        URL, {"Authorization": "secret"}, SimpleNamespace(_decision_allowed=decision), outcome
    )
    callback = lifecycle.event_handler(lambda event: policy.handle(event, SimpleNamespace(send=send)))
    callback({"requestId": "request", "request": {"url": URL, "headers": {"Authorization": "secret"}}})
    await entered.wait()
    policy.close()
    lifecycle.close_admission()
    lease.seal()
    lease.owner_finished()
    assert pool.active_count == 1
    release.set()
    assert not await lifecycle.drain(1)
    for _ in range(3):
        await asyncio.sleep(0)
    assert commands == ["Fetch.failRequest"] and pool.active_count == 0
    assert policy._requests == {} and policy._headers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dom",
    [
        None,
        {"content": "partial"},
        {"title": "t", "author": "a", "date": "d", "content": ["nested"]},
        {"title": "t", "author": "a", "date": "d", "content": "x" * 20},
    ],
)
async def test_live_dom_snapshot_rejects_unbounded_or_non_string_values(dom):
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser import GuardedArticleBrowser

    class Session:
        async def send(self, method, params=None):
            return {
                "Page.getFrameTree": {"frameTree": {"frame": {"id": "main"}}},
                "Page.createIsolatedWorld": {"executionContextId": 1},
                "Runtime.evaluate": {"result": {"value": {"ok": True, "html": "html", "dom": dom}}},
            }[method]

    with pytest.raises(article.ArticleFailure, match="browser_error"):
        await GuardedArticleBrowser._serialize_document(Session(), 10, capture_dom=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("route_cookies", [{}, {"sid": "route"}])
async def test_public_recursive_binds_enhanced_browser_settings(monkeypatch, tmp_path, route_cookies):
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import CookieManager, EnhancedWebScraper

    fixture = _harness(backend="playwright")
    resolve = fixture.dependencies.resolve_plan
    fixture.dependencies = replace(
        fixture.dependencies,
        resolve_plan=lambda *args: replace(
            resolve(*args),
            headers={"X-Route": "route-header"},
            cookies=route_cookies,
            proxies={"https": "http://proxy.example:8080"},
        ),
    )
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    scraper = EnhancedWebScraper(config={})

    async def close_legacy_capability():
        pass

    from types import SimpleNamespace

    scraper._browser = SimpleNamespace(close=close_legacy_capability)
    scraper.cookie_manager = CookieManager(tmp_path / "cookies.json")
    scraper.cookie_manager.add_cookies("example.com", [{"name": "sid", "value": "stored"}])
    try:
        result = await scraper.recursive_scrape(
            URL,
            max_pages=1,
            custom_cookies=[{"name": "sid", "value": "explicit"}],
            custom_headers={"Authorization": "secret", "X-Custom": "custom-header"},
        )
    finally:
        await scraper.stop()
    assert result[0]["extraction_successful"] is True
    profile = fixture.browser.calls[0][1]
    assert profile.custom_cookies == (
        {"name": "sid", "value": route_cookies.get("sid", "explicit"), "url": "https://example.com/"},
    )
    assert profile.extra_headers["X-Route"] == "route-header"
    assert profile.extra_headers["Authorization"] == "secret"
    assert profile.extra_headers["X-Custom"] == "custom-header"
    assert profile.proxies == {"https": "http://proxy.example:8080"}


def test_direct_plan_keeps_original_browser_settings_ineffective():
    from types import SimpleNamespace

    cookie = {
        "name": "sid",
        "value": "caller",
        "domain": "example.com",
        "path": "/private",
        "secure": True,
        "sameSite": "Strict",
    }
    plan = article.ArticlePlan.from_routing_plan(
        SimpleNamespace(
            url=URL,
            domain="example.com",
            extra_headers={"Authorization": "route"},
            cookies={"sid": "route"},
            proxies={"https": "http://proxy.example:8080"},
        ),
        {},
        [cookie],
    )
    assert plan.browser.custom_cookies == (cookie,)
    assert plan.browser.extra_headers is None
    assert plan.browser.proxies == {}


def test_enhanced_cookie_normalization_preserves_browser_scope_attributes():
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _plan

    cookie = {
        "name": "sid",
        "value": "caller",
        "domain": "example.com",
        "path": "/private",
        "secure": True,
        "sameSite": "Strict",
    }
    prepared = crawl._prepare_page_plan(
        _plan(),
        headers={"User-Agent": "caller"},
        legacy=False,
        method=None,
        explicit_agent=None,
        enhanced=True,
        custom_cookies=[cookie],
    )
    assert prepared.browser.custom_cookies == (cookie,)
    with pytest.raises(TypeError):
        prepared.browser.extra_headers["Authorization"] = "mutation"
    public = prepared.for_public_capture()
    assert public.browser.custom_cookies == () and public.browser.extra_headers is None and public.browser.proxies == {}


@pytest.mark.asyncio
async def test_guarded_browser_uses_validated_selected_legacy_proxy(monkeypatch):
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    monkeypatch.setattr(http_client, "PROXY_ALLOWLIST", {"proxy.example"})
    runtime = _FakeBrowserRuntime()
    browser = _adapter(runtime, _FakeGuard([]))
    profile = replace(_profile(), stealth_enabled=False, proxies={"https": "http://user:password@proxy.example:8080"})
    await browser.acquire(URL, profile)
    assert runtime.context_options["proxy"] == {
        "server": "http://proxy.example:8080",
        "username": "user",
        "password": "password",
    }
    assert "extra_http_headers" not in runtime.context_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "binding", [None, {"server": "http://other.example:8080"}, {"server": "http://proxy.example:8080"}]
)
async def test_selected_attested_proxy_requires_matching_hidden_binding(monkeypatch, binding):
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
        BrowserTransportAttestation,
        decide_browser_transport,
    )
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    monkeypatch.setattr(http_client, "PROXY_ALLOWLIST", {"proxy.example"})
    attestation = BrowserTransportAttestation("governed_proxy", True, True, True, proxy_binding=binding)
    decision = decide_browser_transport(
        configured_mode="attested_proxy", auth_mode="multi_user", outbound_policy_mode="strict", attestation=attestation
    )
    runtime = _FakeBrowserRuntime()
    browser = _adapter(runtime, _FakeGuard([]), transport_decision=lambda: decision)
    profile = replace(_profile(), proxies={"https": "http://proxy.example:8080"})
    if binding == {"server": "http://proxy.example:8080"}:
        await browser.acquire(URL, profile)
        assert runtime.context_options["proxy"] == binding
    else:
        with pytest.raises(article.ArticleFailure) as raised:
            await browser.acquire(URL, profile)
        assert raised.value.code == "browser_transport_unavailable"
        assert runtime.events == []
    assert "proxy_binding" not in decision.to_capability_metadata()
    assert "proxy.example" not in repr(decision) and "proxy.example" not in repr(attestation)


@pytest.mark.asyncio
async def test_invalid_selected_proxy_is_not_silently_dropped(monkeypatch):
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import (
        _adapter,
        _FakeBrowserRuntime,
        _FakeGuard,
        _profile,
    )

    monkeypatch.setattr(http_client, "PROXY_ALLOWLIST", set())
    runtime = _FakeBrowserRuntime()
    with pytest.raises(article.ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            URL, replace(_profile(), proxies={"https": "http://denied.example:8080"})
        )
    assert raised.value.code == "browser_transport_unavailable"
    assert runtime.events == []


@pytest.mark.asyncio
async def test_child_transfer_command_dispatch_precedes_next_navigation_turn():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser_dispatch import ChildSessions
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _pool_type

    trace = []

    class Root:
        def on(self, *args):
            pass

        async def send(self, *args):
            trace.append("guard-dispatched")
            await asyncio.sleep(0)
            ports.pending[1][1].set_result({})

    pool = _pool_type()(1)
    lease = pool.acquire(callback_capacity=2)
    ports = ChildSessions(Root(), lease, lambda: None)
    child = ports.adopt("child", ports.root)
    asyncio.get_running_loop().call_soon(trace.append, "navigation-turn")
    await child.send("Network.enable", {})
    lease.seal()
    lease.owner_finished()
    assert trace == ["guard-dispatched", "navigation-turn"]


@pytest.mark.asyncio
async def test_child_command_cancellation_holds_lease_until_actual_reply():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser_dispatch import ChildSessions
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _pool_type

    entered = asyncio.Event()

    class Root:
        def on(self, *args):
            pass

        async def send(self, *args):
            entered.set()

    pool = _pool_type()(1)
    lease = pool.acquire(callback_capacity=2)
    ports = ChildSessions(Root(), lease, lambda: None)
    child = ports.adopt("child", ports.root)
    task = asyncio.create_task(child.send("Fetch.continueRequest", {}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lease.seal()
    lease.owner_finished()
    assert pool.active_count == 1 and len(ports.pending) == 1
    ports.pending[1][1].set_result({})
    for _ in range(10):
        await asyncio.sleep(0)
    assert pool.active_count == 0 and ports.pending == {}


@pytest.mark.asyncio
async def test_transfer_detached_native_session_is_not_detached_twice():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser_dispatch import TransferSessions
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _pool_type

    class Root:
        def on(self, *args):
            pass

        async def send(self, *args):
            raise AssertionError("native session already destroyed")

    pool = _pool_type()(1)
    lease = pool.acquire(callback_capacity=2)
    transfers = TransferSessions(Root(), lease, None, lambda: None)
    child = transfers.ports.adopt("destroyed", transfers.root)
    child.on("Network.dataReceived", lambda _: None)
    transfers.ports.disposed("destroyed")
    await transfers.detach()
    lease.seal()
    lease.owner_finished()
    assert transfers.ports.sessions == {} and child.handlers == {}
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_transfer_context_destroyed_before_detach_event_has_native_retirement_proof():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser_dispatch import TransferSessions
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_browser import _pool_type

    commands = []

    class Root:
        def on(self, *args):
            pass

        async def send(self, method, params=None):
            commands.append(method)
            assert method == "Target.getTargets", "destroyed target cannot accept another detach"
            return {"targetInfos": []}

    pool = _pool_type()(1)
    lease = pool.acquire(callback_capacity=2)
    transfers = TransferSessions(Root(), lease, None, lambda: None)
    session = transfers.ports.adopt("delayed-detach-event", transfers.root)
    session.target = "destroyed-target"
    await transfers.detach()
    lease.seal()
    lease.owner_finished()
    assert commands == ["Target.getTargets"] and transfers.ports.sessions == {}
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_transfer_target_discovery_excludes_already_owned_main_page():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser_dispatch import TransferSessions

    attached = []

    class Root:
        def on(self, *args):
            pass

        async def send(self, method):
            return {"targetInfos": [{"targetId": "main", "type": "page", "browserContextId": "own"}]}

    transfers = TransferSessions(Root(), None, None, lambda: None)
    transfers.context_id, transfers.main_target = "own", "main"

    async def attach(identifier):
        attached.append(identifier)

    transfers._attach = attach
    await transfers.track()
    assert attached == []


def test_native_intercepted_data_length_fallback_is_charged():
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext

    class Lease:
        _callback_capacity = 64

        def ensure_emergency_shutdown(self, *args):
            raise AssertionError("no exhaustion")

    parent = CrawlContext(CrawlLimits(), clock=lambda: 0)
    outcome = browser._AcquisitionOutcome()
    ledger = browser._BrowserTransferLedger(
        limit=1000, outcome=outcome, lease=Lease(), shutdown_factory=lambda: None, parent=parent
    )
    ledger.on_request_data("child", {"requestId": "document", "encodedDataLength": 0, "dataLength": 120})
    assert parent.snapshot().transfer_bytes == 120


@pytest.mark.parametrize(
    "chunks, finished, expected", [([40, 60], 160, 160), ([], 2097253, 2097253), ([120], 100, 120)]
)
def test_enhanced_transfer_cumulative_finish_only_charges_unseen_bytes(chunks, finished, expected):
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext

    class Lease:
        _callback_capacity = 64

        def ensure_emergency_shutdown(self, *args):
            raise AssertionError("no exhaustion")

    parent = CrawlContext(CrawlLimits(), clock=lambda: 0)
    outcome = browser._AcquisitionOutcome()
    ledger = browser._BrowserTransferLedger(
        limit=3000000, outcome=outcome, lease=Lease(), shutdown_factory=lambda: None, parent=parent
    )
    for amount in chunks:
        ledger.on_request_data("page", {"requestId": "document", "encodedDataLength": amount, "dataLength": amount})
    ledger.on_request_finished("page", {"requestId": "document", "encodedDataLength": finished})
    assert parent.snapshot().transfer_bytes == expected
    assert outcome.failure is None


def test_enhanced_transfer_streaming_stop_occurs_without_loading_finished():
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser as browser
    from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext

    shutdown = []

    class Lease:
        _callback_capacity = 64

        def ensure_emergency_shutdown(self, *args):
            shutdown.append(True)

        def close_callback_admission(self):
            pass

    parent = CrawlContext(CrawlLimits(transfer_bytes=100), clock=lambda: 0)
    outcome = browser._AcquisitionOutcome()
    ledger = browser._BrowserTransferLedger(
        limit=1000, outcome=outcome, lease=Lease(), shutdown_factory=lambda: None, parent=parent
    )
    ledger.on_request_data("page", {"requestId": "stream", "encodedDataLength": 0, "dataLength": 120})
    assert outcome.parent_stop.reason == "budget_exhausted" and shutdown
