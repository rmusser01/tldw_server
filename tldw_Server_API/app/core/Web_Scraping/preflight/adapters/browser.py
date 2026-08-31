"""Guarded async Playwright browser probes."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    BrowserTransportDecision,
    default_browser_transport_decision,
    resolve_browser_transport_decision,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.asyncio_compat import timeout as _asyncio_timeout
from tldw_Server_API.app.core.Web_Scraping.preflight.context import (
    PreflightDeadlineExceeded,
    PreflightRuntimeControls,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import (
    BrowserProbeOptions,
    BrowserProbePage,
    ProbeError,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.browser import (
    RuntimeBrowserPage,
    RuntimeBrowserRoute,
    RuntimeWebSocketRoute,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressGuard
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import (
    RuntimeRequestContext,
)

_HTTP_ROUTE_PATTERN = "**/*"
_HTTP_SCHEMES = frozenset({"http", "https"})
_WEBSOCKET_POLICY_SCHEMES = {"ws": "http", "wss": "https"}
_INTERNAL_BLANK_PAGE = "about:blank"
_BROWSER_CLEANUP_GRACE_S = 2.0


class _DefaultPlaywrightLauncher:
    async def start(self) -> Any:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ProbeUnavailable(error_code="missing_dependency") from None
        return await async_playwright().start()


def _playwright_has_required_routing(
    *,
    context_type: type[Any] | None = None,
    websocket_route_type: type[Any] | None = None,
) -> bool:
    """Return whether the installed async API has every required route hook."""
    if context_type is None or websocket_route_type is None:
        try:
            from playwright.async_api import BrowserContext, WebSocketRoute
        except ImportError:
            return False
        context_type = BrowserContext
        websocket_route_type = WebSocketRoute
    return (
        callable(getattr(context_type, "route", None))
        and callable(getattr(context_type, "route_web_socket", None))
        and callable(getattr(websocket_route_type, "connect_to_server", None))
    )


async def _connect_web_socket_to_server(route: RuntimeWebSocketRoute) -> None:
    connect = getattr(route, "connect_to_server", None)
    if not callable(connect):
        raise ProbeUnavailable()
    result = connect()
    if inspect.isawaitable(result):
        await result


async def _invoke_first(resource: Any, method_names: tuple[str, ...]) -> None:
    for method_name in method_names:
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


def _first_callable(resource: Any, method_names: tuple[str, ...]) -> Any | None:
    for method_name in method_names:
        method = getattr(resource, method_name, None)
        if callable(method):
            return method
    return None


async def _await_shared_deadline(
    controls: PreflightRuntimeControls,
    operation: Callable[[], Awaitable[Any]],
    *,
    check_after: bool,
) -> Any:
    """Run one await against the request's absolute monotonic deadline."""
    remaining_s = controls.cap_timeout(None)
    if remaining_s is None:
        result = await operation()
    else:
        timeout = _asyncio_timeout(remaining_s)
        try:
            async with timeout:
                result = await operation()
        except (TimeoutError, asyncio.TimeoutError):
            if timeout.expired():
                raise PreflightDeadlineExceeded() from None
            raise
    if check_after:
        controls.cap_timeout(None)
    return result


class _BrowserCleanupHandle:
    """Share one native close operation across graceful and forced cleanup."""

    def __init__(
        self,
        resource: Any,
        *,
        close_methods: tuple[str, ...],
        force_methods: tuple[str, ...],
        kind: str,
    ) -> None:
        self._resource = resource
        self._close_methods = close_methods
        self._force_methods = force_methods
        self._kind = kind
        self._operation_lock = asyncio.Lock()
        self._force_lock = asyncio.Lock()
        self._operation_task: asyncio.Task[None] | None = None
        self._terminal = False
        self._force_started = False

    @staticmethod
    def _consume_operation(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def _operation(self) -> tuple[asyncio.Task[None] | None, bool]:
        async with self._operation_lock:
            if self._terminal:
                return None, False
            if self._operation_task is None:
                self._operation_task = asyncio.create_task(
                    _invoke_first(self._resource, self._close_methods),
                    name=f"preflight-browser-close-{self._kind}",
                )
                self._operation_task.add_done_callback(self._consume_operation)
                return self._operation_task, True
            return self._operation_task, False

    async def close(self) -> None:
        operation, _ = await self._operation()
        if operation is None:
            return
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise

    async def force_close(self) -> None:
        async with self._force_lock:
            if self._force_started:
                return
            self._force_started = True
            force_method = _first_callable(self._resource, self._force_methods)
            if force_method is not None:
                async with self._operation_lock:
                    self._terminal = True
                result = force_method()
                if inspect.isawaitable(result):
                    await result
                return

            operation, _ = await self._operation()
            if operation is None:
                return
            async with self._operation_lock:
                self._terminal = True
            await operation


def _cleanup_handle(resource: Any, *, kind: str) -> _BrowserCleanupHandle:
    if kind == "playwright":
        return _BrowserCleanupHandle(
            resource,
            close_methods=("stop",),
            force_methods=("force_close",),
            kind=kind,
        )
    return _BrowserCleanupHandle(
        resource,
        close_methods=("close",),
        force_methods=("force_close",),
        kind=kind,
    )


def _is_playwright_timeout(exc: BaseException) -> bool:
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    except ImportError:
        return False
    return isinstance(exc, PlaywrightTimeoutError)


def _websocket_policy_url(url: str) -> str | None:
    """Map a valid WebSocket URL to the guard's transport-equivalent URL."""
    try:
        parsed = urlsplit(url)
        policy_scheme = _WEBSOCKET_POLICY_SCHEMES.get(parsed.scheme.lower())
        if policy_scheme is None or not parsed.netloc or parsed.hostname is None or parsed.fragment:
            return None
        _ = parsed.port
    except (TypeError, ValueError):
        return None
    return urlunsplit((policy_scheme, parsed.netloc, parsed.path, parsed.query, ""))


class _GuardedPlaywrightPage:
    def __init__(
        self,
        *,
        page: RuntimeBrowserPage,
        page_handle: _BrowserCleanupHandle,
        controls: PreflightRuntimeControls,
        captured_urls: list[str],
    ) -> None:
        self._page = page
        self._page_handle = page_handle
        self._controls = controls
        self._captured_urls = captured_urls

    @staticmethod
    def _normalize_timeout_ms(requested_ms: float) -> float:
        if isinstance(requested_ms, bool):
            raise ValueError("timeout_ms must be a non-negative finite number")
        normalized = float(requested_ms)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("timeout_ms must be a non-negative finite number")
        return normalized

    def _cap_timeout_ms(self, requested_ms: float) -> float:
        normalized = self._normalize_timeout_ms(requested_ms)
        if normalized == 0:
            remaining_s = self._controls.remaining_seconds()
            if remaining_s is None:
                return 0.0
            if remaining_s <= 0:
                raise PreflightDeadlineExceeded()
            return remaining_s * 1000.0
        capped_s = self._controls.cap_timeout(normalized / 1000.0)
        return normalized if capped_s is None else capped_s * 1000.0

    def _cap_delay_ms(self, requested_ms: float) -> float:
        normalized = self._normalize_timeout_ms(requested_ms)
        if normalized == 0:
            return 0.0
        capped_s = self._controls.cap_timeout(normalized / 1000.0)
        return normalized if capped_s is None else capped_s * 1000.0

    async def _invoke(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        try:
            return await _await_shared_deadline(
                self._controls,
                operation,
                check_after=True,
            )
        except asyncio.CancelledError:
            raise
        except (ProbeError, PreflightDeadlineExceeded):
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize the Playwright boundary
            if _is_playwright_timeout(exc):
                if self._controls.deadline_exhausted():
                    raise PreflightDeadlineExceeded() from None
                raise ProbeTimeout() from None
            raise ProbeError("probe_error", "Probe failed.") from None

    async def goto(
        self,
        url: str,
        *,
        wait_until: str,
        timeout_ms: float,
    ) -> None:
        normalized_url = str(url)
        if normalized_url != _INTERNAL_BLANK_PAGE:
            try:
                scheme = urlsplit(normalized_url).scheme.lower()
            except (TypeError, ValueError):
                scheme = ""
            if scheme not in _HTTP_SCHEMES:
                raise ProbeError("policy_denied", "Probe destination was denied.")
        effective_ms = self._cap_timeout_ms(timeout_ms)
        await self._invoke(
            lambda: self._page.goto(
                normalized_url,
                wait_until=wait_until,
                timeout=effective_ms,
            )
        )

    async def reload(self, *, wait_until: str, timeout_ms: float) -> None:
        effective_ms = self._cap_timeout_ms(timeout_ms)
        await self._invoke(lambda: self._page.reload(wait_until=wait_until, timeout=effective_ms))

    async def wait_for_load_state(self, state: str, *, timeout_ms: float) -> None:
        effective_ms = self._cap_timeout_ms(timeout_ms)
        await self._invoke(lambda: self._page.wait_for_load_state(state, timeout=effective_ms))

    async def wait_for_timeout(self, timeout_ms: float) -> None:
        effective_ms = self._cap_delay_ms(timeout_ms)
        await self._invoke(lambda: self._page.wait_for_timeout(effective_ms))
        if effective_ms < float(timeout_ms):
            raise PreflightDeadlineExceeded()

    async def content(self) -> str:
        return str(await self._invoke(self._page.content))

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        return await self._invoke(lambda: self._page.evaluate(expression, argument))

    async def link_count(self) -> int:
        return int(await self._invoke(lambda: self._page.locator("a").count()))

    async def link_is_visible(self, index: int) -> bool:
        return bool(await self._invoke(lambda: self._page.locator("a").nth(index).is_visible()))

    def captured_request_urls(self) -> tuple[str, ...]:
        return tuple(self._captured_urls)

    def clear_captured_request_urls(self) -> None:
        self._captured_urls.clear()

    async def close(self) -> None:
        await self._page_handle.close()


class GuardedPlaywrightBrowserProbe:
    """Create one routed async Playwright page under request controls."""

    def __init__(
        self,
        *,
        controls: PreflightRuntimeControls,
        egress_guard: ProbeEgressGuard,
        launcher: Any | None = None,
        transport_decision: Callable[[], BrowserTransportDecision] = default_browser_transport_decision,
        capability_check: Callable[[], bool] = _playwright_has_required_routing,
        no_sandbox: bool = False,
    ) -> None:
        self._controls = controls
        self._guard = egress_guard
        self._launcher = launcher or _DefaultPlaywrightLauncher()
        self._transport_decision = transport_decision
        self._capability_check = capability_check
        self._no_sandbox = bool(no_sandbox)

    def _resolve_transport_decision(self) -> BrowserTransportDecision:
        """Resolve browser admission without exposing provider failures."""
        return resolve_browser_transport_decision(
            self._transport_decision,
            component="preflight_browser_probe",
        )

    def transport_capability(self) -> dict[str, str | bool]:
        """Return the current bounded browser-transport capability snapshot."""
        return self._resolve_transport_decision().to_capability_metadata()

    def _subrequest_context(self) -> RuntimeRequestContext:
        return replace(
            self._controls.request_context,
            stage="preflight_subrequest",
        )

    async def _decision_allowed(self, url: str) -> bool:
        try:
            decision = await _await_shared_deadline(
                self._controls,
                lambda: self._guard.decide(
                    url,
                    context=self._subrequest_context(),
                ),
                check_after=True,
            )
            return bool(decision.allowed)
        except asyncio.CancelledError:
            raise
        except PreflightDeadlineExceeded:
            raise
        except Exception:  # noqa: BLE001 - egress policy failures fail closed
            logger.warning("Browser egress decision failed.")
            return False

    async def _abort_http(self, route: RuntimeBrowserRoute) -> None:
        try:
            await _await_shared_deadline(
                self._controls,
                route.abort,
                check_after=True,
            )
        except (asyncio.CancelledError, PreflightDeadlineExceeded):
            raise
        except Exception:  # noqa: BLE001 - route failures stay at the boundary
            logger.warning("Browser HTTP route action failed.")

    async def _continue_http(self, route: RuntimeBrowserRoute) -> None:
        try:
            await _await_shared_deadline(
                self._controls,
                route.continue_,
                check_after=True,
            )
        except (asyncio.CancelledError, PreflightDeadlineExceeded):
            raise
        except Exception:  # noqa: BLE001 - fail closed after a routing failure
            logger.warning("Browser HTTP route action failed.")
            await self._abort_http(route)

    async def _close_websocket(
        self,
        route: RuntimeWebSocketRoute,
        *,
        code: int,
        reason: str,
    ) -> None:
        try:
            await _await_shared_deadline(
                self._controls,
                lambda: route.close(code=code, reason=reason),
                check_after=True,
            )
        except (asyncio.CancelledError, PreflightDeadlineExceeded):
            raise
        except Exception:  # noqa: BLE001 - route failures stay at the boundary
            logger.warning("Browser WebSocket route action failed.")

    def _http_handler(
        self,
        options: BrowserProbeOptions,
    ) -> Callable[[RuntimeBrowserRoute], Any]:
        async def _route_http(route: RuntimeBrowserRoute) -> None:
            try:
                request = route.request
                if request.resource_type in options.block_resource_types:
                    await self._abort_http(route)
                    return
                if await self._decision_allowed(request.url):
                    await self._continue_http(route)
                else:
                    await self._abort_http(route)
            except (asyncio.CancelledError, PreflightDeadlineExceeded):
                raise
            except Exception:  # noqa: BLE001 - route accessors fail closed
                logger.warning("Browser HTTP route evaluation failed.")
                await self._abort_http(route)

        return _route_http

    def _websocket_handler(self) -> Callable[[RuntimeWebSocketRoute], Any]:
        async def _route_web_socket(route: RuntimeWebSocketRoute) -> None:
            try:
                policy_url = _websocket_policy_url(route.url)
                if policy_url is None or not await self._decision_allowed(policy_url):
                    await self._close_websocket(
                        route,
                        code=1008,
                        reason="Policy denied",
                    )
                    return
            except (asyncio.CancelledError, PreflightDeadlineExceeded):
                raise
            except Exception:  # noqa: BLE001 - route accessors fail closed
                logger.warning("Browser WebSocket route evaluation failed.")
                await self._close_websocket(
                    route,
                    code=1008,
                    reason="Policy denied",
                )
                return
            try:
                await _await_shared_deadline(
                    self._controls,
                    lambda: _connect_web_socket_to_server(route),
                    check_after=True,
                )
            except (asyncio.CancelledError, PreflightDeadlineExceeded):
                raise
            except Exception:  # noqa: BLE001 - fail closed after connect failure
                logger.warning("Browser WebSocket route action failed.")
                await self._close_websocket(
                    route,
                    code=1011,
                    reason="Connection failed",
                )

        return _route_web_socket

    def _register(
        self,
        handles: list[_BrowserCleanupHandle],
        resource: Any,
        *,
        kind: str,
    ) -> _BrowserCleanupHandle:
        handle = _cleanup_handle(resource, kind=kind)
        handles.append(handle)
        self._controls.register_cleanup(handle)
        return handle

    async def _cleanup_owned_resources(
        self,
        handles: list[_BrowserCleanupHandle],
    ) -> None:
        try:
            await self._controls.cleanup_handles(
                handles,
                grace_s=_BROWSER_CLEANUP_GRACE_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - cleanup cannot replace an outcome
            logger.warning("Browser probe cleanup failed.")

    @asynccontextmanager
    async def open_page(
        self,
        options: BrowserProbeOptions,
    ) -> Any:
        decision = self._resolve_transport_decision()
        if not decision.allowed:
            raise ProbeUnavailable(error_code=decision.reason)
        try:
            available = bool(self._capability_check())
        except Exception:  # noqa: BLE001 - capability introspection is optional
            available = False
        if not available:
            raise ProbeUnavailable()

        await self._controls.reserve("browser")
        handles: list[_BrowserCleanupHandle] = []
        startup_stage = "launch"
        try:
            playwright = await _await_shared_deadline(
                self._controls,
                self._launcher.start,
                check_after=False,
            )
            self._register(handles, playwright, kind="playwright")

            launch_options: dict[str, Any] = {"headless": True}
            if self._no_sandbox:
                launch_options["args"] = ["--no-sandbox"]
            browser = await _await_shared_deadline(
                self._controls,
                lambda: playwright.chromium.launch(**launch_options),
                check_after=False,
            )
            self._register(handles, browser, kind="browser")
            startup_stage = "context"

            context_options: dict[str, Any] = {
                "service_workers": "block",
                "user_agent": options.user_agent,
                "extra_http_headers": dict(options.extra_headers),
                "viewport": {
                    "width": options.viewport_width,
                    "height": options.viewport_height,
                },
            }
            context = await _await_shared_deadline(
                self._controls,
                lambda: browser.new_context(**context_options),
                check_after=False,
            )
            self._register(handles, context, kind="context")
            await _await_shared_deadline(
                self._controls,
                lambda: context.route(
                    _HTTP_ROUTE_PATTERN,
                    self._http_handler(options),
                ),
                check_after=False,
            )
            await _await_shared_deadline(
                self._controls,
                lambda: context.route_web_socket(
                    _HTTP_ROUTE_PATTERN,
                    self._websocket_handler(),
                ),
                check_after=False,
            )
            for script in options.init_scripts:
                await _await_shared_deadline(
                    self._controls,
                    lambda script=script: context.add_init_script(script=script),
                    check_after=False,
                )

            captured_urls: list[str] = []
            if options.capture_requests:
                self._controls.cap_timeout(None)
                context.on(
                    "request",
                    lambda request: captured_urls.append(str(request.url)),
                )

            page = await _await_shared_deadline(
                self._controls,
                context.new_page,
                check_after=False,
            )
            page_handle = self._register(handles, page, kind="page")
            self._controls.cap_timeout(None)
            wrapped: BrowserProbePage = _GuardedPlaywrightPage(
                page=page,
                page_handle=page_handle,
                controls=self._controls,
                captured_urls=captured_urls,
            )
        except asyncio.CancelledError:
            await self._cleanup_owned_resources(handles)
            raise
        except (ProbeError, PreflightDeadlineExceeded):
            await self._cleanup_owned_resources(handles)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize startup failures
            await self._cleanup_owned_resources(handles)
            if _is_playwright_timeout(exc):
                if self._controls.deadline_exhausted():
                    raise PreflightDeadlineExceeded() from None
                raise ProbeTimeout() from None
            if startup_stage == "launch":
                raise ProbeUnavailable() from None
            raise ProbeError("probe_error", "Probe failed.") from None

        try:
            yield wrapped
        finally:
            await self._cleanup_owned_resources(handles)
