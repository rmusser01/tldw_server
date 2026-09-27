from __future__ import annotations

import asyncio
import importlib
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    BrowserTransportAttestation,
    decide_browser_transport,
)
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
from tldw_Server_API.app.core.Web_Scraping.preflight.asyncio_compat import timeout as asyncio_timeout
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import (
    FakeBrowserContext,
    FakeBrowserPage,
    FakeBrowserRequest,
    FakeBrowserRoute,
    FakeCleanupHandle,
    FakeClock,
    FakePlaywrightLauncher,
    FakeProbeEgressGuard,
    FakeWebSocketRoute,
    RealLikeBrowserContext,
    RealLikeBrowserPage,
    RealLikePlaywrightLauncher,
)

pytestmark = pytest.mark.unit

_ADAPTER_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.adapters.browser"
_ASYNCIO_COMPAT_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.asyncio_compat"


class _LegacyAsyncioTimeoutError(Exception):
    """Simulate Python 3.10's distinct asyncio timeout exception."""


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


def _asyncio_compat_module() -> Any:
    return importlib.import_module(_ASYNCIO_COMPAT_MODULE)


@pytest.mark.asyncio
async def test_asyncio_timeout_compatibility_uses_stdlib_context_with_expired_method() -> None:
    compat = _asyncio_compat_module()

    context = compat.timeout(1.0)
    async with context:
        pass

    assert context.expired() is False


@pytest.mark.asyncio
async def test_asyncio_timeout_compatibility_uses_async_timeout_when_stdlib_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat = _asyncio_compat_module()
    created: list[float | None] = []

    class LegacyTimeout:
        @property
        def expired(self) -> bool:
            return False

        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    def legacy_timeout(delay: float | None) -> LegacyTimeout:
        created.append(delay)
        return LegacyTimeout()

    legacy_module = types.ModuleType("async_timeout")
    legacy_module.timeout = legacy_timeout
    monkeypatch.delattr(compat.asyncio, "timeout", raising=False)
    monkeypatch.setitem(sys.modules, "async_timeout", legacy_module)

    context = compat.timeout(0.25)
    async with context:
        pass

    assert created == [0.25]
    assert context.expired() is False


@pytest.mark.asyncio
async def test_shared_deadline_normalizes_python310_asyncio_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _adapter_module()
    assert module is not None
    controls = _live_controls(deadline_s=1.0)

    class ExpiredTimeout:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def expired(self) -> bool:
            return True

    async def legacy_timeout() -> None:
        raise _LegacyAsyncioTimeoutError()

    monkeypatch.setattr(module.asyncio, "TimeoutError", _LegacyAsyncioTimeoutError)
    monkeypatch.setattr(module, "_asyncio_timeout", lambda _delay: ExpiredTimeout())

    with pytest.raises(PreflightDeadlineExceeded):
        await module._await_shared_deadline(
            controls,
            legacy_timeout,
            check_after=False,
        )


@pytest.mark.asyncio
async def test_shared_deadline_preserves_nested_timeout_error() -> None:
    controls = _live_controls(deadline_s=1.0)

    async def nested_timeout() -> None:
        raise TimeoutError("nested operation timeout")

    with pytest.raises(TimeoutError, match="nested operation timeout"):
        await _required("_await_shared_deadline")(
            controls,
            nested_timeout,
            check_after=False,
        )


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


def _live_controls(*, deadline_s: float) -> PreflightRuntimeControls:
    loop = asyncio.get_running_loop()
    return PreflightRuntimeControls(
        RuntimeRequestContext(source="preflight", stage="preflight"),
        deadline=loop.time() + deadline_s,
        clock=loop.time,
    )


def _probe(
    *,
    controls: PreflightRuntimeControls,
    guard: FakeProbeEgressGuard,
    launcher: FakePlaywrightLauncher,
    transport_decision: Callable[[], object] | None = None,
    capability: bool = True,
    no_sandbox: bool = False,
) -> Any:
    return _required("GuardedPlaywrightBrowserProbe")(
        controls=controls,
        egress_guard=guard,
        launcher=launcher,
        transport_decision=transport_decision
        or (
            lambda: decide_browser_transport(
                configured_mode="auto",
                auth_mode="single_user",
                outbound_policy_mode="compat",
            )
        ),
        capability_check=lambda: capability,
        no_sandbox=no_sandbox,
    )


@pytest.mark.asyncio
async def test_browser_routes_before_page_and_blocks_service_workers() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    attested = decide_browser_transport(
        configured_mode="attested_proxy",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
        attestation=BrowserTransportAttestation(
            mechanism="governed_proxy",
            routes_all_requests=True,
            dns_pinned=True,
            peer_verified=True,
        ),
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        transport_decision=lambda: attested,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_reason"),
    [
        (
            decide_browser_transport(
                configured_mode="disabled",
                auth_mode="single_user",
                outbound_policy_mode="compat",
            ),
            "browser_transport_disabled",
        ),
        (
            decide_browser_transport(
                configured_mode="auto",
                auth_mode="multi_user",
                outbound_policy_mode="strict",
            ),
            "browser_transport_unattested",
        ),
        (
            decide_browser_transport(
                configured_mode="bogus",
                auth_mode="single_user",
                outbound_policy_mode="compat",
            ),
            "browser_transport_config_invalid",
        ),
    ],
)
async def test_browser_transport_denial_precedes_budget_and_launch(
    decision: object,
    expected_reason: str,
) -> None:
    controls = _controls(browsers=1)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        transport_decision=lambda: decision,
    )

    with pytest.raises(ProbeUnavailable) as raised:
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert raised.value.error_code == expected_reason
    assert raised.value.public_message == "Safe browser transport is unavailable."
    assert controls.consumed.browsers == 0
    assert launcher.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_decision",
    [
        lambda: (_ for _ in ()).throw(RuntimeError("secret config error")),
        lambda: object(),
    ],
    ids=["provider-error", "wrong-type"],
)
async def test_invalid_browser_transport_provider_fails_closed_before_budget(
    transport_decision: Callable[[], object],
) -> None:
    controls = _controls(browsers=1)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        transport_decision=transport_decision,
    )

    with pytest.raises(ProbeUnavailable) as raised:
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert raised.value.error_code == "browser_transport_config_invalid"
    assert controls.consumed.browsers == 0
    assert launcher.events == []


def test_transport_provider_failure_logs_only_safe_diagnostics() -> None:
    """Provider failures should be observable without leaking exception details."""
    secret = "transport-secret"
    records: list[Any] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        probe = _probe(
            controls=_controls(),
            guard=FakeProbeEgressGuard([]),
            launcher=FakePlaywrightLauncher(),
            transport_decision=lambda: (_ for _ in ()).throw(RuntimeError(secret)),
        )

        capability = probe.transport_capability()
    finally:
        logger.remove(sink_id)

    assert capability["reason"] == "browser_transport_config_invalid"
    matching = [record for record in records if record["extra"].get("operation") == "resolve_transport"]
    assert len(matching) == 1
    assert matching[0]["extra"]["component"] == "preflight_browser_probe"
    assert matching[0]["extra"]["operation"] == "resolve_transport"
    assert matching[0]["extra"]["exception_type"] == "RuntimeError"
    assert secret not in matching[0]["message"]


def test_browser_transport_capability_is_exactly_bounded() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    denied = decide_browser_transport(
        configured_mode="auto",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
        transport_decision=lambda: denied,
    )

    assert probe.transport_capability() == denied.to_capability_metadata()


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
@pytest.mark.parametrize(
    ("allowed", "resource_type", "continue_error", "abort_error"),
    [
        (
            True,
            "document",
            RuntimeError("https://secret.example/continue"),
            RuntimeError("https://secret.example/fallback-abort"),
        ),
        (False, "document", None, RuntimeError("https://secret.example/abort")),
        (True, "image", None, RuntimeError("https://secret.example/blocked")),
    ],
)
async def test_http_route_action_failures_are_contained_sanitized_and_fail_closed(
    allowed: bool,
    resource_type: str,
    continue_error: BaseException | None,
    abort_error: BaseException | None,
) -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    guard = FakeProbeEgressGuard([] if resource_type == "image" else [allowed])
    probe = _probe(controls=controls, guard=guard, launcher=launcher)
    try:
        async with probe.open_page(BrowserProbeOptions(block_resource_types=("image",))):
            context = launcher.browser.contexts[0]
            assert context.http_handler is not None
            route = FakeBrowserRoute(
                FakeBrowserRequest("https://secret.example/?token=raw", resource_type),
                launcher.events,
                continue_error=continue_error,
                abort_error=abort_error,
            )
            await context.http_handler(route)
        await controls.close()
    finally:
        logger.remove(sink)

    assert route.abort_calls == 1
    assert route.continue_calls == (1 if allowed and resource_type != "image" else 0)
    rendered = "".join(messages)
    assert "secret.example" not in rendered
    assert "token=raw" not in rendered
    assert "RuntimeError" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["continue", "abort"])
async def test_http_route_action_cancellation_propagates(action: str) -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([action == "continue"]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        context = launcher.browser.contexts[0]
        assert context.http_handler is not None
        route = FakeBrowserRoute(
            FakeBrowserRequest("https://allowed.example"),
            launcher.events,
            continue_error=asyncio.CancelledError() if action == "continue" else None,
            abort_error=asyncio.CancelledError() if action == "abort" else None,
        )
        with pytest.raises(asyncio.CancelledError):
            await context.http_handler(route)
    await controls.close()


@pytest.mark.asyncio
async def test_decision_allowed_accessor_failure_is_contained_and_sanitized() -> None:
    class RaisingDecision:
        @property
        def allowed(self) -> bool:
            raise RuntimeError("https://secret.example/?token=raw")

    class RaisingDecisionGuard:
        async def decide(self, *_args: Any, **_kwargs: Any) -> RaisingDecision:
            return RaisingDecision()

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=RaisingDecisionGuard(),  # type: ignore[arg-type]
        launcher=launcher,
    )
    try:
        async with probe.open_page(BrowserProbeOptions()):
            context = launcher.browser.contexts[0]
            assert context.http_handler is not None
            route = FakeBrowserRoute(
                FakeBrowserRequest("https://allowed.example"),
                launcher.events,
            )
            await context.http_handler(route)
        await controls.close()
    finally:
        logger.remove(sink)

    assert route.abort_calls == 1
    assert route.continue_calls == 0
    rendered = "".join(messages)
    assert rendered
    assert "secret.example" not in rendered
    assert "token=raw" not in rendered
    assert "RuntimeError" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_accessor", ["request", "resource_type", "url"])
async def test_http_accessor_failure_is_contained_sanitized_and_aborted(
    failing_accessor: str,
) -> None:
    class RaisingRequest:
        @property
        def resource_type(self) -> str:
            if failing_accessor == "resource_type":
                raise RuntimeError("https://secret.example/resource")
            return "document"

        @property
        def url(self) -> str:
            if failing_accessor == "url":
                raise RuntimeError("https://secret.example/?token=raw")
            return "https://allowed.example"

    class RaisingRoute:
        def __init__(self) -> None:
            self.abort_calls = 0
            self.continue_calls = 0

        @property
        def request(self) -> RaisingRequest:
            if failing_accessor == "request":
                raise RuntimeError("https://secret.example/request")
            return RaisingRequest()

        async def abort(self) -> None:
            self.abort_calls += 1

        async def continue_(self) -> None:
            self.continue_calls += 1

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([True]),
        launcher=launcher,
    )
    route = RaisingRoute()
    try:
        async with probe.open_page(BrowserProbeOptions()):
            context = launcher.browser.contexts[0]
            assert context.http_handler is not None
            await context.http_handler(route)  # type: ignore[arg-type]
        await controls.close()
    finally:
        logger.remove(sink)

    assert route.abort_calls == 1
    assert route.continue_calls == 0
    rendered = "".join(messages)
    assert rendered
    assert "secret.example" not in rendered
    assert "token=raw" not in rendered
    assert "RuntimeError" not in rendered


@pytest.mark.asyncio
async def test_http_accessor_cancellation_propagates() -> None:
    class CancellingRoute:
        @property
        def request(self) -> Any:
            raise asyncio.CancelledError()

        async def abort(self) -> None:
            pytest.fail("abort must not replace cancellation")

    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        context = launcher.browser.contexts[0]
        assert context.http_handler is not None
        with pytest.raises(asyncio.CancelledError):
            await context.http_handler(CancellingRoute())  # type: ignore[arg-type]
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
async def test_websocket_url_accessor_failure_is_contained_sanitized_and_closed() -> None:
    class RaisingWebSocketRoute:
        def __init__(self) -> None:
            self.close_calls: list[tuple[int | None, str | None]] = []
            self.connect_calls = 0

        @property
        def url(self) -> str:
            raise RuntimeError("wss://secret.example/?token=raw")

        def connect_to_server(self) -> None:
            self.connect_calls += 1

        async def close(
            self,
            *,
            code: int | None = None,
            reason: str | None = None,
        ) -> None:
            self.close_calls.append((code, reason))

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )
    route = RaisingWebSocketRoute()
    try:
        async with probe.open_page(BrowserProbeOptions()):
            context = launcher.browser.contexts[0]
            assert context.websocket_handler is not None
            await context.websocket_handler(route)  # type: ignore[arg-type]
        await controls.close()
    finally:
        logger.remove(sink)

    assert route.close_calls == [(1008, "Policy denied")]
    assert route.connect_calls == 0
    rendered = "".join(messages)
    assert rendered
    assert "secret.example" not in rendered
    assert "token=raw" not in rendered
    assert "RuntimeError" not in rendered


@pytest.mark.asyncio
async def test_websocket_url_accessor_deadline_propagates() -> None:
    class DeadlineWebSocketRoute:
        @property
        def url(self) -> str:
            raise PreflightDeadlineExceeded()

        async def close(self, **_kwargs: Any) -> None:
            pytest.fail("close must not replace the deadline")

    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        context = launcher.browser.contexts[0]
        assert context.websocket_handler is not None
        with pytest.raises(PreflightDeadlineExceeded):
            await context.websocket_handler(DeadlineWebSocketRoute())  # type: ignore[arg-type]
    await controls.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_url", "guard_url"),
    [
        ("ws://socket.example/path?q=1", "http://socket.example/path?q=1"),
        ("wss://socket.example/path?q=1", "https://socket.example/path?q=1"),
        ("ws://socket.example:80/path", "http://socket.example:80/path"),
        ("wss://socket.example:443/path", "https://socket.example:443/path"),
    ],
)
async def test_websocket_policy_uses_transport_equivalent_http_url(
    route_url: str,
    guard_url: str,
) -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([True])
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions()):
        route = await launcher.browser.contexts[0].dispatch_websocket(route_url)
    await controls.close()

    assert guard.urls == [guard_url]
    assert route.connect_calls == 1
    assert route.close_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_url",
    [
        "https://socket.example/path",
        "ws:///missing-host",
        "wss://socket.example:invalid/path",
        "wss://socket.example/path#fragment",
    ],
)
async def test_malformed_or_non_websocket_url_fails_closed_without_guard(
    route_url: str,
) -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([])
    launcher = FakePlaywrightLauncher()
    probe = _probe(controls=controls, guard=guard, launcher=launcher)

    async with probe.open_page(BrowserProbeOptions()):
        route = await launcher.browser.contexts[0].dispatch_websocket(route_url)
    await controls.close()

    assert guard.urls == []
    assert route.connect_calls == 0
    assert route.close_calls == [(1008, "Policy denied")]


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_websocket_action_failures_are_contained_sanitized_and_fail_closed(
    allowed: bool,
) -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([allowed]),
        launcher=launcher,
    )
    try:
        async with probe.open_page(BrowserProbeOptions()):
            context = launcher.browser.contexts[0]
            assert context.websocket_handler is not None
            route = FakeWebSocketRoute(
                "wss://secret.example/?token=raw",
                launcher.events,
                connect_error=(RuntimeError("wss://secret.example/connect") if allowed else None),
                close_error=RuntimeError("wss://secret.example/close"),
            )
            await context.websocket_handler(route)
        await controls.close()
    finally:
        logger.remove(sink)

    assert route.connect_calls == (1 if allowed else 0)
    assert route.close_calls == [(1011, "Connection failed") if allowed else (1008, "Policy denied")]
    rendered = "".join(messages)
    assert "secret.example" not in rendered
    assert "token=raw" not in rendered
    assert "RuntimeError" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["connect", "close"])
async def test_websocket_action_cancellation_propagates(action: str) -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([action == "connect"]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        context = launcher.browser.contexts[0]
        assert context.websocket_handler is not None
        route = FakeWebSocketRoute(
            "wss://allowed.example/socket",
            launcher.events,
            connect_error=asyncio.CancelledError() if action == "connect" else None,
            close_error=asyncio.CancelledError() if action == "close" else None,
        )
        with pytest.raises(asyncio.CancelledError):
            await context.websocket_handler(route)
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
    "stage",
    [
        "launcher_start",
        "launch_browser",
        "new_context",
        "route_http",
        "route_web_socket",
        "init_script",
        "new_page",
    ],
)
async def test_each_startup_await_is_bounded_by_the_shared_deadline(stage: str) -> None:
    controls = _live_controls(deadline_s=0.02)
    launcher = FakePlaywrightLauncher(block_at=stage)
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )
    options = BrowserProbeOptions(init_scripts=("window.test = true",))

    with pytest.raises(PreflightDeadlineExceeded):
        async with asyncio_timeout(0.2):
            async with probe.open_page(options):
                pytest.fail("page must not be created")

    assert launcher.startup_gate.started.is_set()
    assert controls.consumed.browsers == 1
    await controls.close(grace_s=0.02)


@pytest.mark.asyncio
async def test_startup_rechecks_deadline_before_the_next_side_effect() -> None:
    clock = FakeClock(1.0)

    class BoundaryLauncher(FakePlaywrightLauncher):
        async def start(self) -> Any:
            result = await super().start()
            clock.advance(1.0)
            return result

    controls = _controls(deadline=2.0, clock=clock)
    launcher = BoundaryLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    with pytest.raises(PreflightDeadlineExceeded):
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    assert launcher.playwright.chromium.launch_calls == []
    assert launcher.playwright.stop_calls == 1


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
async def test_zero_timeout_uses_positive_shared_deadline_cap() -> None:
    clock = FakeClock(8.0)
    controls = _controls(deadline=10.0, clock=clock)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        await page.reload(wait_until="load", timeout_ms=0)
        raw = launcher.browser.contexts[0].pages[0]
    await controls.close()

    assert raw.calls[0] == ("reload", (), {"wait_until": "load", "timeout": 2000.0})


@pytest.mark.asyncio
async def test_zero_timeout_passes_through_without_shared_deadline() -> None:
    controls = _controls()
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        await page.reload(wait_until="load", timeout_ms=0)
        raw = launcher.browser.contexts[0].pages[0]
    await controls.close()

    assert raw.calls[0] == ("reload", (), {"wait_until": "load", "timeout": 0.0})


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
async def test_wait_for_timeout_checks_exact_deadline_boundary_after_completion() -> None:
    clock = FakeClock(8.0)

    class BoundaryWaitPage(FakeBrowserPage):
        async def wait_for_timeout(self, timeout_ms: float) -> Any:
            result = await super().wait_for_timeout(timeout_ms)
            clock.advance(timeout_ms / 1000.0)
            return result

    controls = _controls(deadline=10.0, clock=clock)
    launcher = FakePlaywrightLauncher(
        context_factory=lambda events: FakeBrowserContext(
            events,
            page_factory=BoundaryWaitPage,
        )
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        with pytest.raises(PreflightDeadlineExceeded):
            await page.wait_for_timeout(2000)
    await controls.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["content", "evaluate", "count", "visibility"])
async def test_operations_without_playwright_timeout_are_deadline_bounded(
    operation: str,
) -> None:
    blocker = asyncio.Event()

    class BlockingLocator:
        def __init__(self, index: int | None = None) -> None:
            self.index = index

        def nth(self, index: int) -> BlockingLocator:
            return BlockingLocator(index)

        async def count(self) -> int:
            await blocker.wait()
            return 0

        async def is_visible(self) -> bool:
            await blocker.wait()
            return False

    class BlockingPage(FakeBrowserPage):
        async def content(self) -> str:
            await blocker.wait()
            return ""

        async def evaluate(self, expression: str, argument: Any = None) -> Any:
            await blocker.wait()
            return None

        def locator(self, selector: str) -> BlockingLocator:
            assert selector == "a"
            return BlockingLocator()

    controls = _live_controls(deadline_s=0.02)
    launcher = FakePlaywrightLauncher(
        context_factory=lambda events: FakeBrowserContext(
            events,
            page_factory=BlockingPage,
        )
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()) as page:
        call = {
            "content": page.content,
            "evaluate": lambda: page.evaluate("1"),
            "count": page.link_count,
            "visibility": lambda: page.link_is_visible(0),
        }[operation]
        with pytest.raises(PreflightDeadlineExceeded):
            async with asyncio_timeout(0.2):
                await call()
    await controls.close(grace_s=0.02)


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
        "close:context",
        "close:browser",
        "close:playwright",
        "close:page",
        "close:context",
        "close:browser",
        "close:playwright",
    ]
    assert launcher.playwright.stop_calls == 2
    assert all(page.close_calls == 1 for context in launcher.browser.contexts for page in context.pages)


@pytest.mark.asyncio
async def test_page_scope_cleanup_does_not_close_unrelated_registered_adapter() -> None:
    controls = _controls()
    unrelated = FakeCleanupHandle(name="unrelated")
    controls.register_cleanup(unrelated)
    launcher = FakePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async with probe.open_page(BrowserProbeOptions()):
        pass

    assert unrelated.close_calls == 0
    assert launcher.playwright.stop_calls == 1
    await controls.close()
    assert unrelated.close_calls == 1


@pytest.mark.asyncio
async def test_page_scopes_share_one_non_additive_cleanup_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter_module()
    assert adapter is not None
    grace_s = 0.05
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", grace_s)
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

    started_at = asyncio.get_running_loop().time()
    async with probe.open_page(BrowserProbeOptions()):
        pass
    async with probe.open_page(BrowserProbeOptions()):
        pass
    elapsed_s = asyncio.get_running_loop().time() - started_at
    await controls.close(grace_s=grace_s)

    assert elapsed_s < grace_s * 1.6
    assert all(page.close_calls == 1 for context in launcher.browser.contexts for page in context.pages)


@pytest.mark.asyncio
async def test_analyzer_time_does_not_consume_shared_cleanup_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter_module()
    assert adapter is not None
    grace_s = 0.03
    analyzer_delay_s = 0.05
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", grace_s)

    class SlowClosePage(FakeBrowserPage):
        async def close(self) -> None:
            self.close_calls += 1
            self.events.append("close:page")
            self.close_started.set()
            try:
                await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                self.close_cancellations += 1
                raise
            self.results["close_complete"] = True

    controls = _controls()
    launcher = FakePlaywrightLauncher(
        context_factory=lambda events: FakeBrowserContext(
            events,
            page_factory=SlowClosePage,
        )
    )
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    started_at = asyncio.get_running_loop().time()
    async with probe.open_page(BrowserProbeOptions()):
        pass
    await asyncio.sleep(analyzer_delay_s)
    async with probe.open_page(BrowserProbeOptions()):
        pass
    elapsed_s = asyncio.get_running_loop().time() - started_at
    await controls.close(grace_s=grace_s)

    pages = [page for context in launcher.browser.contexts for page in context.pages]
    assert [page.close_calls for page in pages] == [1, 1]
    assert [page.close_cancellations for page in pages] == [0, 0]
    assert [page.results.get("close_complete") for page in pages] == [True, True]
    assert [page.force_close_calls for page in pages] == [0, 0]
    assert elapsed_s - analyzer_delay_s < grace_s


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
async def test_fresh_caller_cancellation_wins_earlier_startup_error_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter_module()
    assert adapter is not None
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", 0.03)

    class FailingContext(FakeBrowserContext):
        def __init__(self, events: list[str]) -> None:
            super().__init__(events)
            self.close_started = asyncio.Event()
            self.release = asyncio.Event()

        async def route_web_socket(self, pattern: str, handler: Any) -> None:
            raise RuntimeError("https://secret.example/startup")

        async def close(self) -> None:
            self.close_calls += 1
            self.close_started.set()
            await self.release.wait()

    controls = _controls()
    launcher = FakePlaywrightLauncher(context_factory=FailingContext)
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )

    async def open_failing_page() -> None:
        async with probe.open_page(BrowserProbeOptions()):
            pytest.fail("page must not be created")

    caller = asyncio.create_task(open_failing_page())
    while not launcher.browser.contexts:
        await asyncio.sleep(0)
    context = launcher.browser.contexts[0]
    assert isinstance(context, FailingContext)
    await context.close_started.wait()
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio_timeout(0.2):
            await caller


@pytest.mark.asyncio
async def test_page_and_request_cleanup_share_one_idempotent_close_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter_module()
    assert adapter is not None
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", 0.02, raising=False)

    class RealLikePage(FakeBrowserPage):
        force_close = None

    controls = _controls()
    launcher = FakePlaywrightLauncher(
        context_factory=lambda events: FakeBrowserContext(
            events,
            page_factory=lambda context, page_events: RealLikePage(
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
    scope_cleanup = asyncio.create_task(manager.__aexit__(None, None, None))
    request_cleanup = asyncio.create_task(controls.close(grace_s=0.02))
    done, pending = await asyncio.wait(
        {scope_cleanup, request_cleanup},
        timeout=0.2,
    )
    if pending:
        page = launcher.browser.contexts[0].pages[0]
        page.release_close()
        await asyncio.gather(*pending, return_exceptions=True)

    page = launcher.browser.contexts[0].pages[0]
    assert not pending
    assert all(task.exception() is None for task in done)
    assert page.close_calls == 1
    assert page.force_close_calls == 0
    assert page.close_cancellations == 0
    assert launcher.browser.contexts[0].close_calls == 0
    assert launcher.browser.close_calls == 0
    assert launcher.playwright.stop_calls == 0
    assert launcher.browser.contexts[0].force_close_calls == 1
    assert launcher.browser.force_close_calls == 1
    assert launcher.playwright.force_close_calls == 1
    assert not {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task.get_name().startswith("preflight-browser-close")
    }


@pytest.mark.asyncio
async def test_all_real_like_graph_uses_reserved_parent_teardown_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter_module()
    assert adapter is not None
    grace_s = 0.08
    monkeypatch.setattr(adapter, "_BROWSER_CLEANUP_GRACE_S", grace_s)
    controls = _controls()
    launcher = RealLikePlaywrightLauncher()
    probe = _probe(
        controls=controls,
        guard=FakeProbeEgressGuard([]),
        launcher=launcher,
    )
    manager = probe.open_page(BrowserProbeOptions())
    await manager.__aenter__()
    context = launcher.browser.contexts[0]
    assert isinstance(context, RealLikeBrowserContext)
    page = context.pages[0]
    assert isinstance(page, RealLikeBrowserPage)
    resources = (page, context, launcher.browser, launcher.playwright)
    assert all(not callable(getattr(resource, "force_close", None)) for resource in resources)

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    scope_cleanup = asyncio.create_task(manager.__aexit__(None, None, None))
    await page.close_started.wait()
    request_cleanup = asyncio.create_task(controls.close(grace_s=grace_s))
    scope_cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio_timeout(grace_s * 3):
            await scope_cleanup
    async with asyncio_timeout(grace_s * 3):
        await request_cleanup
    elapsed_s = loop.time() - started_at
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert [resource.close_calls for resource in resources[:3]] == [1, 1, 1]
    assert launcher.playwright.stop_calls == 1
    assert page.closed
    assert context.closed
    assert launcher.browser.closed
    assert launcher.playwright.stopped
    assert context.close_started_at is not None
    assert context.close_started_at - started_at < grace_s * 0.75
    assert elapsed_s < grace_s * 1.25
    assert not {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name().startswith(("preflight-browser-close", "preflight-cleanup"))
    }


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
