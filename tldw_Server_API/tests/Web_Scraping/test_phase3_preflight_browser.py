from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.preflight import (
    BrowserProbeOptions,
    PreflightDeadlineExceeded,
    PreflightLimits,
    PreflightRuntimeControls,
    ProbeBudgetExhausted,
    ProbeError,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import (
    FakeBrowserContext,
    FakeBrowserPage,
    FakeClock,
    FakePlaywrightLauncher,
    FakeProbeEgressGuard,
    FakeWebSocketRoute,
)

pytestmark = pytest.mark.unit

_ADAPTER_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser"


def _adapter_module() -> Any | None:
    try:
        return importlib.import_module(_ADAPTER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in {_ADAPTER_MODULE, _ADAPTER_MODULE.rpartition(".")[0]}:
            return None
        raise


def _required(name: str) -> Any:
    module = _adapter_module()
    assert module is not None, "Task 5 governed browser adapter module is missing"
    assert hasattr(module, name), f"Task 5 governed browser adapter {name} is missing"
    return getattr(module, name)


def _controls(
    *,
    browsers: int | None = None,
    deadline: float | None = None,
    clock: FakeClock | None = None,
) -> PreflightRuntimeControls:
    return PreflightRuntimeControls(
        RuntimeRequestContext(
            source="preflight",
            stage="preflight",
            user_id="7",
            request_id="request-browser-1",
            metadata={"scope": "task-5"},
        ),
        limits=PreflightLimits(browsers=browsers),
        deadline=deadline,
        clock=clock or FakeClock(),
    )


def _probe(
    *,
    controls: PreflightRuntimeControls,
    guard: FakeProbeEgressGuard,
    launcher: FakePlaywrightLauncher,
    capability: bool = True,
    no_sandbox: bool = False,
) -> Any:
    return _required("GuardedPlaywrightBrowserProbe")(
        controls=controls,
        egress_guard=guard,
        launcher=launcher,
        capability_check=lambda: capability,
        no_sandbox=no_sandbox,
    )


@pytest.mark.asyncio
async def test_browser_routes_before_page_and_blocks_service_workers() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions(user_agent="UA")):
        pass
    await controls.close()

    assert launcher.events[:6] == [
        "launch",
        "launch_browser",
        "new_context:service_workers=block",
        "route_http",
        "route_web_socket",
        "new_page",
    ]
    assert launcher.browser.context_options == [
        {
            "service_workers": "block",
            "user_agent": "UA",
            "extra_http_headers": {},
            "viewport": {"width": 1280, "height": 720},
        }
    ]


@pytest.mark.asyncio
async def test_missing_websocket_capability_is_unavailable_before_budget_or_launch() -> None:
    controls = _controls(browsers=1)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        capability=False,
    )

    with pytest.raises(ProbeUnavailable) as raised:
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert raised.value.error_code == "unavailable"
    assert controls.consumed.browsers == 0
    assert launcher.events == []


def test_capability_check_requires_http_websocket_and_server_connect_callables() -> None:
    capability_check = _required("_playwright_has_required_routing")

    class CompleteContext:
        def route(self) -> None:
            return None

        def route_web_socket(self) -> None:
            return None

    class CompleteWebSocketRoute:
        def connect_to_server(self) -> None:
            return None

    class MissingHttp:
        route = None

        def route_web_socket(self) -> None:
            return None

    class MissingWebSocket:
        def route(self) -> None:
            return None

    class MissingConnect:
        pass

    assert capability_check(
        context_type=CompleteContext,
        websocket_route_type=CompleteWebSocketRoute,
    )
    assert not capability_check(
        context_type=MissingHttp,
        websocket_route_type=CompleteWebSocketRoute,
    )
    assert not capability_check(
        context_type=MissingWebSocket,
        websocket_route_type=CompleteWebSocketRoute,
    )
    assert not capability_check(
        context_type=CompleteContext,
        websocket_route_type=MissingConnect,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_connect", [False, True])
async def test_websocket_server_connect_is_invoked_once(
    awaitable_connect: bool,
) -> None:
    route = FakeWebSocketRoute(
        "wss://allowed.example/socket",
        [],
        awaitable_connect=awaitable_connect,
    )

    await _required("_connect_web_socket_to_server")(route)

    assert route.connect_calls == 1


@pytest.mark.asyncio
async def test_http_route_allows_navigation_redirect_and_subresource() -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([True, True, True])
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions()):
        context = launcher.browser.contexts[0]
        routes = [
            await context.dispatch_http("https://start.example"),
            await context.dispatch_http("https://redirect.example"),
            await context.dispatch_http("https://cdn.example/script.js", "script"),
        ]
    await controls.close()

    assert [route.continue_calls for route in routes] == [1, 1, 1]
    assert [route.abort_calls for route in routes] == [0, 0, 0]
    assert guard.urls == [
        "https://start.example",
        "https://redirect.example",
        "https://cdn.example/script.js",
    ]
    assert all(context.stage == "preflight_subrequest" for context in guard.contexts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [False, RuntimeError("https://secret.example/?token=raw")],
)
async def test_http_route_denial_and_guard_error_fail_closed(decision: object) -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([decision])  # type: ignore[list-item]
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions()):
        route = await launcher.browser.contexts[0].dispatch_http("https://secret.example/?token=raw")
    await controls.close()

    assert route.abort_calls == 1
    assert route.continue_calls == 0


@pytest.mark.asyncio
async def test_http_route_guard_cancellation_propagates() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([asyncio.CancelledError()]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        with pytest.raises(asyncio.CancelledError):
            await launcher.browser.contexts[0].dispatch_http("https://allowed.example")
    await controls.close()


@pytest.mark.asyncio
async def test_blocked_resource_type_aborts_without_guard_decision() -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([])
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions(block_resource_types=("image",))):
        route = await launcher.browser.contexts[0].dispatch_http(
            "https://cdn.example/image.png",
            "image",
        )
    await controls.close()

    assert route.abort_calls == 1
    assert guard.urls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable_connect", [False, True])
async def test_websocket_route_allows_and_connects(
    awaitable_connect: bool,
) -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([True])
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions()):
        route = await launcher.browser.contexts[0].dispatch_websocket(
            "wss://socket.example/path",
            awaitable_connect=awaitable_connect,
        )
    await controls.close()

    assert route.connect_calls == 1
    assert route.close_calls == []
    assert guard.contexts[0].stage == "preflight_subrequest"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [False, RuntimeError("wss://secret.example/?token=raw")],
)
async def test_websocket_denial_and_guard_error_close_with_policy_code(
    decision: object,
) -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([decision]),  # type: ignore[list-item]
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        route = await launcher.browser.contexts[0].dispatch_websocket("wss://secret.example/?token=raw")
    await controls.close()

    assert route.close_calls == [(1008, "Policy denied")]
    assert route.connect_calls == 0


@pytest.mark.asyncio
async def test_websocket_guard_cancellation_propagates() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([asyncio.CancelledError()]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        with pytest.raises(asyncio.CancelledError):
            await launcher.browser.contexts[0].dispatch_websocket("wss://allowed.example/socket")
    await controls.close()


@pytest.mark.asyncio
async def test_browser_budget_is_reserved_exactly_once_before_launch() -> None:
    controls = _controls(browsers=1)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        assert controls.consumed.browsers == 1
    await controls.close()

    assert launcher.start_calls == 1
    assert len(launcher.playwright.chromium.launch_calls) == 1


@pytest.mark.asyncio
async def test_browser_budget_exhaustion_prevents_launch() -> None:
    controls = _controls(browsers=0)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    with pytest.raises(ProbeBudgetExhausted):
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert launcher.events == []


@pytest.mark.asyncio
async def test_exhausted_deadline_reserves_once_but_prevents_launch() -> None:
    controls = _controls(deadline=0.0, clock=FakeClock())
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    with pytest.raises(PreflightDeadlineExceeded):
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert controls.consumed.browsers == 1
    assert launcher.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("no_sandbox", "expected"),
    [(False, {"headless": True}), (True, {"headless": True, "args": ["--no-sandbox"]})],
)
async def test_no_sandbox_launch_arg_requires_explicit_opt_in(
    no_sandbox: bool,
    expected: dict[str, Any],
) -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        no_sandbox=no_sandbox,
    )

    async with probe.open_page(BrowserProbeOptions()):
        pass
    await controls.close()

    assert launcher.playwright.chromium.launch_calls == [expected]


@pytest.mark.asyncio
async def test_init_scripts_and_request_capture_are_installed_before_page() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([True]),
        launcher=launcher,
    )
    options = BrowserProbeOptions(
        extra_headers={"X-Test": "value"},
        viewport_width=900,
        viewport_height=600,
        init_scripts=("window.first = true", "window.second = true"),
        capture_requests=True,
    )

    async with probe.open_page(options) as page:
        context = launcher.browser.contexts[0]
        await context.dispatch_http("https://capture.example/path")
        assert page.captured_request_urls() == ("https://capture.example/path",)
        page.clear_captured_request_urls()
        assert page.captured_request_urls() == ()
    await controls.close()

    assert context.init_scripts == ["window.first = true", "window.second = true"]
    assert launcher.events.index("init_script") < launcher.events.index("new_page")
    assert launcher.events.index("capture_requests") < launcher.events.index("new_page")
    assert launcher.browser.context_options[0]["extra_http_headers"] == {"X-Test": "value"}
    assert launcher.browser.context_options[0]["viewport"] == {"width": 900, "height": 600}


@pytest.mark.asyncio
async def test_page_wrapper_delegates_operations_and_caps_timeout_ms() -> None:
    clock = FakeClock(8.0)
    controls = _controls(deadline=10.0, clock=clock)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        raw = launcher.browser.contexts[0].pages[0]
        raw.results["content"] = "<html>marker</html>"
        raw.results["evaluate"] = {"ok": True}
        await page.goto("https://allowed.example", wait_until="domcontentloaded", timeout_ms=9000)
        await page.reload(wait_until="load", timeout_ms=8000)
        await page.wait_for_load_state("networkidle", timeout_ms=7000)
        assert await page.content() == "<html>marker</html>"
        assert await page.evaluate("value => value", {"a": 1}) == {"ok": True}
        assert await page.link_count() == 2
        assert await page.link_is_visible(0) is True
        assert await page.link_is_visible(1) is False
    await controls.close()

    assert raw.calls[:3] == [
        (
            "goto",
            ("https://allowed.example",),
            {"wait_until": "domcontentloaded", "timeout": 2000.0},
        ),
        ("reload", (), {"wait_until": "load", "timeout": 2000.0}),
        (
            "wait_for_load_state",
            ("networkidle",),
            {"timeout": 2000.0},
        ),
    ]


@pytest.mark.asyncio
async def test_wait_for_timeout_uses_remaining_deadline_and_then_expires() -> None:
    clock = FakeClock(8.0)
    controls = _controls(deadline=10.0, clock=clock)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        with pytest.raises(PreflightDeadlineExceeded):
            await page.wait_for_timeout(5000)
        raw = launcher.browser.contexts[0].pages[0]
    await controls.close()

    assert raw.calls[0] == ("wait_for_timeout", (2000.0,), {})


@pytest.mark.asyncio
async def test_playwright_timeout_is_probe_timeout_while_deadline_remains() -> None:
    playwright = pytest.importorskip("playwright.async_api")
    controls = _controls(deadline=10.0, clock=FakeClock(1.0))
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        launcher.browser.contexts[0].pages[0].errors["goto"] = playwright.TimeoutError(
            "https://secret.example/?token=raw"
        )
        with pytest.raises(ProbeTimeout) as raised:
            await page.goto("https://allowed.example", wait_until="load", timeout_ms=1000)
    await controls.close()

    assert raised.value.public_message == "Probe timed out."


@pytest.mark.asyncio
async def test_playwright_timeout_is_deadline_error_when_deadline_expires() -> None:
    playwright = pytest.importorskip("playwright.async_api")
    clock = FakeClock(1.0)

    class DeadlinePage(FakeBrowserPage):
        async def goto(self, url: str, **kwargs: Any) -> Any:
            clock.advance(9.0)
            raise playwright.TimeoutError("https://secret.example/?token=raw")

    controls = _controls(deadline=10.0, clock=clock)
    launcher = FakePlaywrightLauncher(
        context_factory=lambda events: FakeBrowserContext(
            events,
            page_factory=DeadlinePage,
        )
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        with pytest.raises(PreflightDeadlineExceeded):
            await page.goto("https://allowed.example", wait_until="load", timeout_ms=1000)
    await controls.close()


@pytest.mark.asyncio
async def test_arbitrary_page_error_is_sanitized_without_timeout_conversion() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        launcher.browser.contexts[0].pages[0].errors["reload"] = RuntimeError("https://secret.example/?token=raw")
        with pytest.raises(ProbeError) as raised:
            await page.reload(wait_until="load", timeout_ms=1000)
    await controls.close()

    assert raised.value.error_code == "probe_error"
    assert raised.value.public_message == "Probe failed."


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["file:///etc/passwd", "data:text/html,secret", "ftp://host/file"])
async def test_non_http_navigation_fails_before_page_navigation(url: str) -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        with pytest.raises(ProbeError) as raised:
            await page.goto(url, wait_until="load", timeout_ms=1000)
        raw = launcher.browser.contexts[0].pages[0]
    await controls.close()

    assert raised.value.error_code == "policy_denied"
    assert not [call for call in raw.calls if call[0] == "goto"]


@pytest.mark.asyncio
async def test_about_blank_is_the_only_internal_navigation_allowed() -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([])
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions()) as page:
        await page.goto("about:blank", wait_until="load", timeout_ms=1000)
        raw = launcher.browser.contexts[0].pages[0]
    await controls.close()

    assert raw.calls[0][0:2] == ("goto", ("about:blank",))
    assert guard.urls == []


@pytest.mark.asyncio
async def test_page_scope_and_request_cleanup_are_idempotent_and_ordered() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        pass
    async with probe.open_page(BrowserProbeOptions()):
        pass
    await controls.close()

    assert launcher.events.count("close:page") == 2
    assert [event for event in launcher.events if event.startswith("close:")] == [
        "close:page",
        "close:page",
        "close:context",
        "close:browser",
        "close:playwright",
        "close:context",
        "close:browser",
        "close:playwright",
    ]
    assert launcher.playwright.stop_calls == 2
    assert all(page.close_calls == 1 for context in launcher.browser.contexts for page in context.pages)


@pytest.mark.asyncio
async def test_partial_startup_failure_closes_created_resources() -> None:
    class FailingContext(FakeBrowserContext):
        async def route_web_socket(self, pattern: str, handler: Any) -> None:
            raise RuntimeError("https://secret.example/?token=raw")

    controls = _controls()
    launcher = FakePlaywrightLauncher(context_factory=FailingContext)
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    with pytest.raises(ProbeError) as raised:
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert raised.value.public_message == "Probe failed."
    assert launcher.events[-3:] == [
        "close:context",
        "close:browser",
        "close:playwright",
    ]


@pytest.mark.asyncio
async def test_startup_cancellation_closes_created_resources_and_propagates() -> None:
    class CancellingContext(FakeBrowserContext):
        async def new_page(self) -> FakeBrowserPage:
            raise asyncio.CancelledError()

    controls = _controls()
    launcher = FakePlaywrightLauncher(context_factory=CancellingContext)
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    with pytest.raises(asyncio.CancelledError):
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert launcher.events[-3:] == [
        "close:context",
        "close:browser",
        "close:playwright",
    ]


@pytest.mark.asyncio
async def test_controls_force_cleanup_reaches_a_stuck_page_within_one_grace() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher(
        context_factory=lambda events: FakeBrowserContext(
            events,
            page_factory=lambda context, page_events: FakeBrowserPage(
                context,
                page_events,
                block_close=True,
                suppress_close_cancellation=True,
            ),
        )
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )
    manager = probe.open_page(BrowserProbeOptions())
    await manager.__aenter__()

    await controls.close(grace_s=0.02)
    await manager.__aexit__(None, None, None)

    page = launcher.browser.contexts[0].pages[0]
    assert page.force_close_calls == 1
    assert page.close_cancellations >= 1
    assert launcher.browser.contexts[0].force_close_calls == 1
    assert launcher.browser.force_close_calls == 1
    assert launcher.playwright.force_close_calls == 1


@pytest.mark.asyncio
async def test_guard_and_cleanup_logs_never_include_urls_or_raw_errors() -> None:
    class FailingCloseContext(FakeBrowserContext):
        async def close(self) -> None:
            raise RuntimeError("https://secret.example/?token=raw")

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    controls = _controls()
    launcher = FakePlaywrightLauncher(context_factory=FailingCloseContext)
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([RuntimeError("https://secret.example/?token=raw")]),
        launcher=launcher,
    )
    try:
        async with probe.open_page(BrowserProbeOptions()):
            route = await launcher.browser.contexts[0].dispatch_http("https://secret.example/?token=raw")
        await controls.close()
    finally:
        logger.remove(sink)

    rendered = "".join(messages)
    assert route.abort_calls == 1
    assert "secret.example" not in rendered
    assert "token=raw" not in rendered
    assert "RuntimeError" not in rendered
