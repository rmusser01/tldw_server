from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
    DirectBrowserProfile,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

_TARGET = "https://article.example/start"


def _browser_type() -> type[Any]:
    try:
        module = importlib.import_module("tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser")
    except ModuleNotFoundError:
        pytest.fail("GuardedArticleBrowser implementation is missing", pytrace=False)
    return module.GuardedArticleBrowser


def _profile() -> DirectBrowserProfile:
    return DirectBrowserProfile(
        user_agent="Task15 Browser/1.0",
        custom_cookies=(
            {
                "name": "caller-cookie",
                "value": "secret",
                "domain": "article.example",
                "path": "/",
            },
        ),
        retries=3,
        timeout_ms=4_000,
        stealth_enabled=True,
        stealth_wait_ms=250,
        viewport_width=1024,
        viewport_height=768,
    )


class _FakeGuard:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, RuntimeRequestContext]] = []

    async def decide(
        self,
        url: str,
        *,
        context: RuntimeRequestContext,
    ) -> ProbeEgressDecision:
        self.calls.append((url, context))
        if not self.outcomes:
            raise AssertionError("unexpected egress decision")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, bool):
            return ProbeEgressDecision(allowed=outcome, reason="test")
        assert isinstance(outcome, ProbeEgressDecision)
        return outcome


class _FakeHttpRoute:
    def __init__(
        self,
        url: str,
        *,
        continue_error: BaseException | None = None,
        abort_error: BaseException | None = None,
    ) -> None:
        self.request = SimpleNamespace(url=url, resource_type="document")
        self.continue_error = continue_error
        self.abort_error = abort_error
        self.continue_calls: list[dict[str, object]] = []
        self.abort_calls = 0

    async def continue_(self, **kwargs: object) -> None:
        self.continue_calls.append(kwargs)
        if self.continue_error is not None:
            raise self.continue_error

    async def abort(self) -> None:
        self.abort_calls += 1
        if self.abort_error is not None:
            raise self.abort_error


class _FakeWebSocketRoute:
    def __init__(
        self,
        url: str,
        *,
        connect_error: BaseException | None = None,
        close_error: BaseException | None = None,
        missing_connect: bool = False,
    ) -> None:
        self.url = url
        self.connect_error = connect_error
        self.close_error = close_error
        self.connect_calls = 0
        self.close_calls: list[tuple[int | None, str | None]] = []
        if missing_connect:
            self.connect_to_server = None  # type: ignore[assignment]

    async def connect_to_server(self) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    async def close(
        self,
        *,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.close_calls.append((code, reason))
        if self.close_error is not None:
            raise self.close_error


class _FakeBrowserRuntime:
    def __init__(
        self,
        *,
        dispatches: list[tuple[str, object]] | None = None,
        stage_errors: dict[str, BaseException] | None = None,
        html: str = "<!doctype html><html><body>ok</body></html>",
        missing_context_capability: str | None = None,
    ) -> None:
        self.dispatches = list(dispatches or [])
        self.stage_errors = dict(stage_errors or {})
        self.html = html
        self.missing_context_capability = missing_context_capability
        self.events: list[str] = []
        self.launch_options: dict[str, object] | None = None
        self.context_options: dict[str, object] | None = None
        self.http_routes: list[_FakeHttpRoute] = []
        self.websocket_routes: list[_FakeWebSocketRoute] = []
        self.context: _FakeContext | None = None
        self.launcher = _FakeLauncher(self)

    def raise_at(self, stage: str) -> None:
        error = self.stage_errors.get(stage)
        if error is not None:
            raise error


class _FakePage:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime

    async def goto(self, url: str, **kwargs: object) -> None:
        self.runtime.events.append(f"goto:{url}")
        self.runtime.events.append(f"goto-options:{sorted(kwargs)}")
        self.runtime.raise_at("goto")
        context = self.runtime.context
        assert context is not None
        for kind, value in self.runtime.dispatches:
            if kind == "http":
                route = value if isinstance(value, _FakeHttpRoute) else _FakeHttpRoute(str(value))
                self.runtime.http_routes.append(route)
                assert context.http_handler is not None
                await context.http_handler(route)
            else:
                route = value if isinstance(value, _FakeWebSocketRoute) else _FakeWebSocketRoute(str(value))
                self.runtime.websocket_routes.append(route)
                assert context.websocket_handler is not None
                await context.websocket_handler(route)

    async def content(self) -> str:
        self.runtime.events.append("content")
        self.runtime.raise_at("content")
        return self.runtime.html

    async def close(self) -> None:
        self.runtime.events.append("close:page")
        self.runtime.raise_at("page_close")


class _FakeContext:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime
        self.http_handler: Callable[[_FakeHttpRoute], Awaitable[None]] | None = None
        self.websocket_handler: Callable[[_FakeWebSocketRoute], Awaitable[None]] | None = None
        if runtime.missing_context_capability == "route":
            self.route = None  # type: ignore[assignment]
        if runtime.missing_context_capability == "route_web_socket":
            self.route_web_socket = None  # type: ignore[assignment]

    async def route(
        self,
        pattern: str,
        handler: Callable[[_FakeHttpRoute], Awaitable[None]],
    ) -> None:
        self.runtime.events.append(f"route:{pattern}")
        self.runtime.raise_at("route")
        self.http_handler = handler

    async def route_web_socket(
        self,
        pattern: str,
        handler: Callable[[_FakeWebSocketRoute], Awaitable[None]],
    ) -> None:
        self.runtime.events.append(f"route_web_socket:{pattern}")
        self.runtime.raise_at("route_web_socket")
        self.websocket_handler = handler

    async def new_page(self) -> _FakePage:
        self.runtime.events.append("new_page")
        self.runtime.raise_at("new_page")
        return _FakePage(self.runtime)

    async def new_cdp_session(self, page: _FakePage) -> object:
        self.runtime.events.append("new_cdp_session")
        return SimpleNamespace(page=page)

    async def close(self) -> None:
        self.runtime.events.append("close:context")
        self.runtime.raise_at("context_close")


class _FakeBrowser:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime

    async def new_context(self, **kwargs: object) -> _FakeContext:
        self.runtime.events.append("new_context")
        self.runtime.context_options = dict(kwargs)
        self.runtime.raise_at("context")
        context = _FakeContext(self.runtime)
        self.runtime.context = context
        return context

    async def close(self) -> None:
        self.runtime.events.append("close:browser")
        self.runtime.raise_at("browser_close")


class _FakeChromium:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime

    async def launch(self, **kwargs: object) -> _FakeBrowser:
        self.runtime.events.append("launch")
        self.runtime.launch_options = dict(kwargs)
        self.runtime.raise_at("launch")
        return _FakeBrowser(self.runtime)


class _FakePlaywright:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime
        self.chromium = _FakeChromium(runtime)

    async def stop(self) -> None:
        self.runtime.events.append("stop:playwright")
        self.runtime.raise_at("playwright_stop")


class _FakeLauncher:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime

    async def start(self) -> _FakePlaywright:
        self.runtime.events.append("start")
        self.runtime.raise_at("start")
        return _FakePlaywright(self.runtime)


def _adapter(
    runtime: _FakeBrowserRuntime,
    guard: _FakeGuard,
    *,
    capability_check: Callable[[], bool] | None = None,
) -> Any:
    context = RuntimeRequestContext(
        source="article_extract",
        stage="browser",
        user_id="7",
        request_id="request-15",
    )
    return _browser_type()(
        egress_guard=guard,
        context=context,
        launcher=runtime.launcher,
        capability_check=capability_check or (lambda: True),
    )


def _assert_failure(exc: ArticleFailure, *, stage: str) -> None:
    assert exc.code == "browser_error"
    assert exc.stage == stage
    assert str(exc) == "browser_error"
    assert "secret" not in str(exc)
    assert "token" not in str(exc)


@pytest.mark.unit
async def test_acquire_guards_target_redirect_and_subresource_fresh_without_transport_pinning() -> None:
    urls = [
        _TARGET,
        "https://redirect.example/final",
        "https://static.example/app.js",
        "https://static.example/app.js",
    ]
    runtime = _FakeBrowserRuntime(dispatches=[("http", url) for url in urls])
    guard = _FakeGuard(
        [
            ProbeEgressDecision(True, "allowed", ("203.0.113.10",)),
            ProbeEgressDecision(True, "allowed", ("203.0.113.11",)),
            ProbeEgressDecision(True, "allowed", ("203.0.113.12",)),
            ProbeEgressDecision(True, "allowed", ("203.0.113.12",)),
        ]
    )

    html = await _adapter(runtime, guard).acquire(_TARGET, _profile())

    assert html == "<!doctype html><html><body>ok</body></html>"
    assert [url for url, _ in guard.calls] == urls
    assert [context.stage for _, context in guard.calls] == ["fetch"] * 4
    assert [context.source for _, context in guard.calls] == [
        "article_extract",
        "article_extract",
        "article_extract",
        "article_extract",
    ]
    assert [route.continue_calls for route in runtime.http_routes] == [[{}], [{}], [{}], [{}]]
    assert [route.abort_calls for route in runtime.http_routes] == [0, 0, 0, 0]
    assert runtime.launch_options == {"headless": True}
    assert runtime.context_options == {
        "service_workers": "block",
        "user_agent": "Task15 Browser/1.0",
        "viewport": {"width": 1024, "height": 768},
    }
    assert runtime.events.index("route:**/*") < runtime.events.index(f"goto:{_TARGET}")
    assert runtime.events.index("route_web_socket:**/*") < runtime.events.index(f"goto:{_TARGET}")
    assert runtime.events[-5:] == [
        "content",
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]
    assert runtime.events[-1] == "stop:playwright"


@pytest.mark.unit
async def test_websocket_destinations_use_transport_equivalent_policy_urls_only() -> None:
    sockets = [
        _FakeWebSocketRoute("wss://socket.example/live?channel=one"),
        _FakeWebSocketRoute("ws://socket.example/plain"),
    ]
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", route) for route in sockets])
    guard = _FakeGuard([True, True])

    await _adapter(runtime, guard).acquire(_TARGET, _profile())

    assert [url for url, _ in guard.calls] == [
        "https://socket.example/live?channel=one",
        "http://socket.example/plain",
    ]
    assert [route.url for route in sockets] == [
        "wss://socket.example/live?channel=one",
        "ws://socket.example/plain",
    ]
    assert [route.connect_calls for route in sockets] == [1, 1]
    assert [route.close_calls for route in sockets] == [[], []]


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome",
    [
        False,
        RuntimeError("https://secret.example/path?token=raw"),
    ],
    ids=["denied", "guard-error"],
)
async def test_http_denial_or_guard_error_aborts_and_surfaces_sanitized_failure(
    outcome: object,
) -> None:
    route = _FakeHttpRoute("https://secret.example/path?token=raw")
    runtime = _FakeBrowserRuntime(dispatches=[("http", route)])
    guard = _FakeGuard([outcome])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, guard).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="egress")
    assert route.abort_calls == 1
    assert route.continue_calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "file:///private/etc/passwd",
        "https://bad.example:invalid/path",
        "https:///missing-host",
    ],
)
async def test_invalid_http_destination_fails_closed_before_guard(url: str) -> None:
    route = _FakeHttpRoute(url)
    runtime = _FakeBrowserRuntime(dispatches=[("http", route)])
    guard = _FakeGuard([])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, guard).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="egress")
    assert guard.calls == []
    assert route.abort_calls == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "outcome"),
    [
        ("wss://secret.example/socket?token=raw", False),
        (
            "wss://secret.example/socket?token=raw",
            RuntimeError("https://secret.example/socket?token=raw"),
        ),
        ("https://not-a-websocket.example/socket", True),
        ("wss:///missing-host", True),
    ],
    ids=["denied", "guard-error", "wrong-scheme", "missing-host"],
)
async def test_websocket_denial_guard_error_or_invalid_url_closes_and_fails_sanitized(
    url: str,
    outcome: object,
) -> None:
    route = _FakeWebSocketRoute(url)
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", route)])
    guard = _FakeGuard([outcome] if url.startswith("wss://secret") else [])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, guard).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="egress")
    assert route.connect_calls == 0
    assert route.close_calls == [(1008, "Policy denied")]


@pytest.mark.unit
async def test_http_continue_failure_attempts_abort_and_surfaces_route_stage() -> None:
    route = _FakeHttpRoute(
        "https://secret.example/path?token=raw",
        continue_error=RuntimeError("https://secret.example/path?token=raw"),
    )
    runtime = _FakeBrowserRuntime(dispatches=[("http", route)])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([True])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="http_route")
    assert route.continue_calls == [{}]
    assert route.abort_calls == 1


@pytest.mark.unit
async def test_http_abort_failure_surfaces_route_stage() -> None:
    route = _FakeHttpRoute(
        "https://secret.example/path?token=raw",
        abort_error=RuntimeError("https://secret.example/path?token=raw"),
    )
    runtime = _FakeBrowserRuntime(dispatches=[("http", route)])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([False])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="http_route")
    assert route.abort_calls == 1


@pytest.mark.unit
@pytest.mark.parametrize("close_fails", [False, True], ids=["close-ok", "close-fails"])
async def test_websocket_connect_failure_closes_and_surfaces_route_stage(
    close_fails: bool,
) -> None:
    route = _FakeWebSocketRoute(
        "wss://secret.example/socket?token=raw",
        connect_error=RuntimeError("https://secret.example/path?token=raw"),
        close_error=RuntimeError("raw close") if close_fails else None,
    )
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", route)])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([True])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="websocket_route")
    assert route.connect_calls == 1
    assert route.close_calls == [(1011, "Connection failed")]


@pytest.mark.unit
async def test_missing_websocket_connect_capability_fails_with_capability_boundary() -> None:
    route = _FakeWebSocketRoute("wss://socket.example/live", missing_connect=True)
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", route)])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([True])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="capability")
    assert route.close_calls == [(1008, "Policy denied")]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["false", "raises"])
async def test_static_route_capability_absence_fails_before_browser_launch(mode: str) -> None:
    runtime = _FakeBrowserRuntime()
    guard = _FakeGuard([])

    def capability_check() -> bool:
        if mode == "raises":
            raise RuntimeError("raw capability error")
        return False

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, guard, capability_check=capability_check).acquire(
            _TARGET,
            _profile(),
        )

    _assert_failure(raised.value, stage="capability")
    assert runtime.events == []


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["route", "route_web_socket"])
async def test_runtime_route_capability_absence_closes_started_resources(
    missing: str,
) -> None:
    runtime = _FakeBrowserRuntime(missing_context_capability=missing)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="capability")
    assert not any(event.startswith("goto:") for event in runtime.events)
    assert runtime.events[-3:] == ["close:context", "close:browser", "stop:playwright"]


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["route", "route_web_socket"])
async def test_route_install_failure_is_sanitized_and_prevents_navigation(stage: str) -> None:
    runtime = _FakeBrowserRuntime(stage_errors={stage: RuntimeError("https://secret.example/path?token=raw")})

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="routing")
    assert not any(event.startswith("goto:") for event in runtime.events)
    assert runtime.events[-3:] == ["close:context", "close:browser", "stop:playwright"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stage", "closed"),
    [
        ("start", []),
        ("launch", ["stop:playwright"]),
        ("context", ["close:browser", "stop:playwright"]),
        ("route", ["close:context", "close:browser", "stop:playwright"]),
        ("route_web_socket", ["close:context", "close:browser", "stop:playwright"]),
        ("new_page", ["close:context", "close:browser", "stop:playwright"]),
        (
            "goto",
            ["close:page", "close:context", "close:browser", "stop:playwright"],
        ),
        (
            "content",
            ["close:page", "close:context", "close:browser", "stop:playwright"],
        ),
    ],
)
async def test_owner_cancellation_propagates_and_closes_every_acquired_resource(
    stage: str,
    closed: list[str],
) -> None:
    runtime = _FakeBrowserRuntime(stage_errors={stage: asyncio.CancelledError()})

    with pytest.raises(asyncio.CancelledError):
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    close_events = [event for event in runtime.events if event.startswith("close:") or event == "stop:playwright"]
    assert close_events == closed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "guard_outcome", "route_error"),
    [
        ("http", asyncio.CancelledError(), None),
        ("http", True, ("continue", asyncio.CancelledError())),
        ("http", False, ("abort", asyncio.CancelledError())),
        ("websocket", asyncio.CancelledError(), None),
        ("websocket", True, ("connect", asyncio.CancelledError())),
        ("websocket", False, ("close", asyncio.CancelledError())),
    ],
    ids=[
        "http-guard",
        "http-continue",
        "http-abort",
        "websocket-guard",
        "websocket-connect",
        "websocket-close",
    ],
)
async def test_route_cancellation_propagates_after_cleanup(
    kind: str,
    guard_outcome: object,
    route_error: tuple[str, BaseException] | None,
) -> None:
    if kind == "http":
        route = _FakeHttpRoute(
            _TARGET,
            continue_error=(route_error[1] if route_error and route_error[0] == "continue" else None),
            abort_error=(route_error[1] if route_error and route_error[0] == "abort" else None),
        )
    else:
        route = _FakeWebSocketRoute(
            "wss://socket.example/live",
            connect_error=(route_error[1] if route_error and route_error[0] == "connect" else None),
            close_error=(route_error[1] if route_error and route_error[0] == "close" else None),
        )
    runtime = _FakeBrowserRuntime(dispatches=[(kind, route)])

    with pytest.raises(asyncio.CancelledError):
        await _adapter(runtime, _FakeGuard([guard_outcome])).acquire(_TARGET, _profile())

    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]


@pytest.mark.unit
async def test_navigation_failure_is_sanitized_and_resources_close_in_reverse_order() -> None:
    runtime = _FakeBrowserRuntime(stage_errors={"goto": RuntimeError("https://secret.example/path?token=raw")})

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="navigation")
    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]


@pytest.mark.unit
async def test_cleanup_failure_attempts_all_resources_and_is_sanitized() -> None:
    runtime = _FakeBrowserRuntime(stage_errors={"page_close": RuntimeError("https://secret.example/path?token=raw")})

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="cleanup")
    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]
