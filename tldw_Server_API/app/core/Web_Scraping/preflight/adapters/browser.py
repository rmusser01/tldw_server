"""Guarded async Playwright browser probes."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

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
_INTERNAL_BLANK_PAGE = "about:blank"


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


class _BrowserCleanupHandle:
    """Idempotent graceful/force hooks shared by page and request cleanup."""

    def __init__(
        self,
        resource: Any,
        *,
        close_methods: tuple[str, ...],
        force_methods: tuple[str, ...],
    ) -> None:
        self._resource = resource
        self._close_methods = close_methods
        self._force_methods = force_methods
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._force_started = False

    async def close(self) -> None:
        if self._closed:
            return
        async with self._close_lock:
            if self._closed:
                return
            await _invoke_first(self._resource, self._close_methods)
            self._closed = True

    async def force_close(self) -> None:
        if self._closed or self._force_started:
            return
        self._force_started = True
        await _invoke_first(self._resource, self._force_methods)
        self._closed = True


def _cleanup_handle(resource: Any, *, kind: str) -> _BrowserCleanupHandle:
    if kind == "playwright":
        return _BrowserCleanupHandle(
            resource,
            close_methods=("stop",),
            force_methods=("force_close", "stop"),
        )
    return _BrowserCleanupHandle(
        resource,
        close_methods=("close",),
        force_methods=("force_close", "close"),
    )


async def _close_partial_resources(
    handles: list[_BrowserCleanupHandle],
) -> None:
    for handle in reversed(handles):
        try:
            await handle.close()
        except BaseException:  # noqa: BLE001 - preserve the established outcome
            logger.warning("Browser probe cleanup failed.")


def _is_playwright_timeout(exc: BaseException) -> bool:
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    except ImportError:
        return False
    return isinstance(exc, PlaywrightTimeoutError)


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

    def _cap_timeout_ms(self, requested_ms: float) -> float:
        if isinstance(requested_ms, bool):
            raise ValueError("timeout_ms must be a non-negative finite number")
        normalized = float(requested_ms)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("timeout_ms must be a non-negative finite number")
        capped_s = self._controls.cap_timeout(normalized / 1000.0)
        return normalized if capped_s is None else capped_s * 1000.0

    async def _invoke(self, operation: Any) -> Any:
        try:
            return await operation
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
            self._page.goto(
                normalized_url,
                wait_until=wait_until,
                timeout=effective_ms,
            )
        )

    async def reload(self, *, wait_until: str, timeout_ms: float) -> None:
        effective_ms = self._cap_timeout_ms(timeout_ms)
        await self._invoke(self._page.reload(wait_until=wait_until, timeout=effective_ms))

    async def wait_for_load_state(self, state: str, *, timeout_ms: float) -> None:
        effective_ms = self._cap_timeout_ms(timeout_ms)
        await self._invoke(self._page.wait_for_load_state(state, timeout=effective_ms))

    async def wait_for_timeout(self, timeout_ms: float) -> None:
        effective_ms = self._cap_timeout_ms(timeout_ms)
        await self._invoke(self._page.wait_for_timeout(effective_ms))
        if effective_ms < float(timeout_ms):
            raise PreflightDeadlineExceeded()

    async def content(self) -> str:
        return str(await self._invoke(self._page.content()))

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        return await self._invoke(self._page.evaluate(expression, argument))

    async def link_count(self) -> int:
        return int(await self._invoke(self._page.locator("a").count()))

    async def link_is_visible(self, index: int) -> bool:
        return bool(await self._invoke(self._page.locator("a").nth(index).is_visible()))

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
        capability_check: Callable[[], bool] = _playwright_has_required_routing,
        no_sandbox: bool = False,
    ) -> None:
        self._controls = controls
        self._guard = egress_guard
        self._launcher = launcher or _DefaultPlaywrightLauncher()
        self._capability_check = capability_check
        self._no_sandbox = bool(no_sandbox)

    def _subrequest_context(self) -> RuntimeRequestContext:
        return replace(
            self._controls.request_context,
            stage="preflight_subrequest",
        )

    async def _decision_allowed(self, url: str) -> bool:
        try:
            decision = await self._guard.decide(
                url,
                context=self._subrequest_context(),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - egress policy failures fail closed
            return False
        return bool(decision.allowed)

    def _http_handler(
        self,
        options: BrowserProbeOptions,
    ) -> Callable[[RuntimeBrowserRoute], Any]:
        async def _route_http(route: RuntimeBrowserRoute) -> None:
            if route.request.resource_type in options.block_resource_types:
                await route.abort()
                return
            if await self._decision_allowed(route.request.url):
                await route.continue_()
            else:
                await route.abort()

        return _route_http

    def _websocket_handler(self) -> Callable[[RuntimeWebSocketRoute], Any]:
        async def _route_web_socket(route: RuntimeWebSocketRoute) -> None:
            if not await self._decision_allowed(route.url):
                await route.close(code=1008, reason="Policy denied")
                return
            await _connect_web_socket_to_server(route)

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

    @asynccontextmanager
    async def open_page(
        self,
        options: BrowserProbeOptions,
    ) -> Any:
        try:
            available = bool(self._capability_check())
        except Exception:  # noqa: BLE001 - capability introspection is optional
            available = False
        if not available:
            raise ProbeUnavailable()

        await self._controls.reserve("browser")
        self._controls.cap_timeout(None)
        handles: list[_BrowserCleanupHandle] = []
        page_handle: _BrowserCleanupHandle | None = None
        startup_stage = "launch"
        try:
            playwright = await self._launcher.start()
            self._register(handles, playwright, kind="playwright")

            launch_options: dict[str, Any] = {"headless": True}
            if self._no_sandbox:
                launch_options["args"] = ["--no-sandbox"]
            browser = await playwright.chromium.launch(**launch_options)
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
            context = await browser.new_context(**context_options)
            self._register(handles, context, kind="context")
            await context.route(_HTTP_ROUTE_PATTERN, self._http_handler(options))
            await context.route_web_socket(
                _HTTP_ROUTE_PATTERN,
                self._websocket_handler(),
            )
            for script in options.init_scripts:
                await context.add_init_script(script=script)

            captured_urls: list[str] = []
            if options.capture_requests:
                context.on(
                    "request",
                    lambda request: captured_urls.append(str(request.url)),
                )

            page = await context.new_page()
            page_handle = self._register(handles, page, kind="page")
            wrapped: BrowserProbePage = _GuardedPlaywrightPage(
                page=page,
                page_handle=page_handle,
                controls=self._controls,
                captured_urls=captured_urls,
            )
        except asyncio.CancelledError:
            await _close_partial_resources(handles)
            raise
        except (ProbeError, PreflightDeadlineExceeded):
            await _close_partial_resources(handles)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize startup failures
            await _close_partial_resources(handles)
            if _is_playwright_timeout(exc):
                if self._controls.deadline_exhausted():
                    raise PreflightDeadlineExceeded() from None
                raise ProbeTimeout() from None
            if startup_stage == "launch":
                raise ProbeUnavailable() from None
            raise ProbeError("probe_error", "Probe failed.") from None

        established_error: BaseException | None = None
        try:
            yield wrapped
        except BaseException as exc:
            established_error = exc
            raise
        finally:
            if page_handle is not None:
                try:
                    await page_handle.close()
                except asyncio.CancelledError:
                    if established_error is None:
                        raise
                except Exception:  # noqa: BLE001 - request cleanup can retry
                    logger.warning("Browser probe cleanup failed.")
