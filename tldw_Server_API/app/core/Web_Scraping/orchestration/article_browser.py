"""Fail-closed direct-browser acquisition for governed article requests."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tldw_Server_API.app.core.Web_Scraping.runtime.browser import (
    RuntimeBrowserRoute,
    RuntimeWebSocketRoute,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressGuard
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

from .article_models import ArticleFailure, DirectBrowserProfile

_ROUTE_PATTERN = "**/*"
_HTTP_SCHEMES = frozenset({"http", "https"})
_WEBSOCKET_POLICY_SCHEMES = {"ws": "http", "wss": "https"}


class _DefaultPlaywrightLauncher:
    async def start(self) -> Any:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ArticleFailure("browser_error", "capability") from None
        return await async_playwright().start()


def _playwright_has_required_routing() -> bool:
    try:
        from playwright.async_api import BrowserContext, WebSocketRoute
    except ImportError:
        return False
    return (
        callable(getattr(BrowserContext, "route", None))
        and callable(getattr(BrowserContext, "route_web_socket", None))
        and callable(getattr(WebSocketRoute, "connect_to_server", None))
    )


def _http_url_is_valid(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in _HTTP_SCHEMES:
            return False
        if not parsed.netloc or parsed.hostname is None or parsed.fragment:
            return False
        _ = parsed.port
    except (TypeError, ValueError):
        return False
    return True


def _websocket_policy_url(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        policy_scheme = _WEBSOCKET_POLICY_SCHEMES.get(parsed.scheme.lower())
        if policy_scheme is None or not parsed.netloc or parsed.hostname is None or parsed.fragment:
            return None
        _ = parsed.port
    except (TypeError, ValueError):
        return None
    return urlunsplit((policy_scheme, parsed.netloc, parsed.path, parsed.query, ""))


class _RouteOutcome:
    """Latch the first sanitized route failure for one acquisition."""

    def __init__(self) -> None:
        self.failure: ArticleFailure | None = None

    def fail(self, stage: str) -> None:
        if self.failure is None:
            self.failure = ArticleFailure("browser_error", stage)


async def _connect_websocket(route: RuntimeWebSocketRoute) -> None:
    connect = getattr(route, "connect_to_server", None)
    if not callable(connect):
        raise ArticleFailure("browser_error", "capability")
    result = connect()
    if inspect.isawaitable(result):
        await result


class GuardedArticleBrowser:
    """Own one Chromium acquisition and guard every browser destination."""

    def __init__(
        self,
        *,
        egress_guard: ProbeEgressGuard,
        context: RuntimeRequestContext,
        launcher: Any | None = None,
        capability_check: Any = _playwright_has_required_routing,
    ) -> None:
        self._egress_guard = egress_guard
        self._context = context
        self._launcher = launcher or _DefaultPlaywrightLauncher()
        self._capability_check = capability_check

    async def _decision_allowed(self, url: str) -> bool:
        decision = await self._egress_guard.decide(
            url,
            context=replace(self._context, stage="fetch"),
        )
        return bool(decision.allowed)

    async def _abort_http(
        self,
        route: RuntimeBrowserRoute,
        outcome: _RouteOutcome,
        *,
        failure_stage: str,
    ) -> None:
        try:
            await route.abort()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - sanitize the browser route boundary
            outcome.fail("http_route")
            return
        outcome.fail(failure_stage)

    async def _http_handler(
        self,
        route: RuntimeBrowserRoute,
        outcome: _RouteOutcome,
    ) -> None:
        try:
            request = route.request
            url = request.url
            if not _http_url_is_valid(url):
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            try:
                allowed = await self._decision_allowed(url)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - egress guard failures fail closed
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            if not allowed:
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            try:
                await route.continue_()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - route action failures fail closed
                outcome.fail("http_route")
                try:
                    await route.abort()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - original route failure is authoritative
                    outcome.fail("http_route")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - route accessors fail closed
            await self._abort_http(route, outcome, failure_stage="egress")

    @staticmethod
    async def _close_websocket(
        route: RuntimeWebSocketRoute,
        outcome: _RouteOutcome,
        *,
        code: int,
        reason: str,
        failure_stage: str,
    ) -> None:
        try:
            await route.close(code=code, reason=reason)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - sanitize the browser route boundary
            outcome.fail("websocket_route")
            return
        outcome.fail(failure_stage)

    async def _websocket_handler(
        self,
        route: RuntimeWebSocketRoute,
        outcome: _RouteOutcome,
    ) -> None:
        try:
            policy_url = _websocket_policy_url(route.url)
            if policy_url is None:
                await self._close_websocket(
                    route,
                    outcome,
                    code=1008,
                    reason="Policy denied",
                    failure_stage="egress",
                )
                return
            try:
                allowed = await self._decision_allowed(policy_url)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - egress guard failures fail closed
                await self._close_websocket(
                    route,
                    outcome,
                    code=1008,
                    reason="Policy denied",
                    failure_stage="egress",
                )
                return
            if not allowed:
                await self._close_websocket(
                    route,
                    outcome,
                    code=1008,
                    reason="Policy denied",
                    failure_stage="egress",
                )
                return
            if not callable(getattr(route, "connect_to_server", None)):
                await self._close_websocket(
                    route,
                    outcome,
                    code=1008,
                    reason="Policy denied",
                    failure_stage="capability",
                )
                return
            try:
                await _connect_websocket(route)
            except asyncio.CancelledError:
                raise
            except ArticleFailure:
                await self._close_websocket(
                    route,
                    outcome,
                    code=1008,
                    reason="Policy denied",
                    failure_stage="capability",
                )
            except Exception:  # noqa: BLE001 - route action failures fail closed
                await self._close_websocket(
                    route,
                    outcome,
                    code=1011,
                    reason="Connection failed",
                    failure_stage="websocket_route",
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - route accessors fail closed
            await self._close_websocket(
                route,
                outcome,
                code=1008,
                reason="Policy denied",
                failure_stage="egress",
            )

    @staticmethod
    async def _cleanup(resources: list[tuple[Any, str]]) -> BaseException | None:
        first_error: BaseException | None = None
        for resource, method_name in reversed(resources):
            method = getattr(resource, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError as exc:
                first_error = exc
            except Exception as exc:  # noqa: BLE001 - attempt every owned cleanup
                if first_error is None:
                    first_error = exc
        return first_error

    async def acquire(self, url: str, profile: DirectBrowserProfile) -> str:
        """Return rendered HTML after freshly guarding every browser dispatch."""
        try:
            available = bool(self._capability_check())
        except Exception:  # noqa: BLE001 - capability introspection fails closed
            available = False
        if not available:
            raise ArticleFailure("browser_error", "capability")
        if not _http_url_is_valid(url):
            raise ArticleFailure("browser_error", "egress")

        resources: list[tuple[Any, str]] = []
        route_outcome = _RouteOutcome()
        primary_error: BaseException | None = None
        html: str | None = None
        stage = "launch"
        try:
            playwright = await self._launcher.start()
            resources.append((playwright, "stop"))
            browser = await playwright.chromium.launch(headless=True)
            resources.append((browser, "close"))
            stage = "context"
            context = await browser.new_context(
                service_workers="block",
                user_agent=profile.user_agent,
                viewport={
                    "width": profile.viewport_width,
                    "height": profile.viewport_height,
                },
            )
            resources.append((context, "close"))

            route = getattr(context, "route", None)
            route_web_socket = getattr(context, "route_web_socket", None)
            if not callable(route) or not callable(route_web_socket):
                raise ArticleFailure("browser_error", "capability")

            stage = "routing"
            await route(
                _ROUTE_PATTERN,
                lambda intercepted: self._http_handler(intercepted, route_outcome),
            )
            await route_web_socket(
                _ROUTE_PATTERN,
                lambda intercepted: self._websocket_handler(intercepted, route_outcome),
            )
            page = await context.new_page()
            resources.append((page, "close"))

            stage = "navigation"
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=profile.timeout_ms,
            )
            if route_outcome.failure is not None:
                raise route_outcome.failure
            stage = "content"
            html = str(await page.content())
            if route_outcome.failure is not None:
                raise route_outcome.failure
        except asyncio.CancelledError as exc:
            primary_error = exc
        except ArticleFailure as exc:
            primary_error = exc
        except Exception:  # noqa: BLE001 - sanitize the Playwright boundary
            primary_error = route_outcome.failure or ArticleFailure("browser_error", stage)

        cleanup_error = await self._cleanup(resources)
        if isinstance(primary_error, asyncio.CancelledError):
            raise primary_error
        if isinstance(cleanup_error, asyncio.CancelledError):
            raise cleanup_error
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise ArticleFailure("browser_error", "cleanup")
        if html is None:
            raise ArticleFailure("browser_error", "content")
        return html


__all__ = ["GuardedArticleBrowser"]
