"""Fail-closed direct-browser acquisition for governed article requests."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
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
_DEFAULT_CLEANUP_GRACE_S = 1.0
_CALLBACK_IDLE_TURNS = 3
_FORCED_CALLBACK_CANCEL = object()


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
        and callable(getattr(BrowserContext, "unroute_all", None))
        and callable(getattr(WebSocketRoute, "connect_to_server", None))
    )


def _normalize_grace(value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_CLEANUP_GRACE_S
    if not math.isfinite(normalized) or normalized <= 0:
        return _DEFAULT_CLEANUP_GRACE_S
    return normalized


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


class _AcquisitionOutcome:
    """Latch sanitized route failure and callback cancellation state."""

    def __init__(self) -> None:
        self.failure: ArticleFailure | None = None
        self.callback_cancelled = False

    def fail(self, stage: str) -> None:
        if self.failure is None:
            self.failure = ArticleFailure("browser_error", stage)


class _CallbackLifecycle:
    """Track independently scheduled Playwright callback invocations."""

    def __init__(self, outcome: _AcquisitionOutcome) -> None:
        self._outcome = outcome
        self._active: set[asyncio.Task[Any]] = set()
        self._generation = 0

    def handler(
        self,
        operation: Callable[[Any], Awaitable[None]],
    ) -> Callable[[Any], Awaitable[None]]:
        def _tracked(route: Any) -> Awaitable[None]:
            task = asyncio.current_task()
            if task is None:
                self._outcome.fail("callback")

                async def _missing_task() -> None:
                    return None

                return _missing_task()
            self._generation += 1
            self._active.add(task)
            task.add_done_callback(_consume_task_exception)

            async def _invoke() -> None:
                try:
                    await operation(route)
                except asyncio.CancelledError as exc:
                    if exc.args and exc.args[0] is _FORCED_CALLBACK_CANCEL:
                        self._outcome.fail("callback_drain")
                    else:
                        self._outcome.callback_cancelled = True
                except Exception:  # noqa: BLE001 - callback failures are sanitized
                    self._outcome.fail("callback")
                finally:
                    self._active.discard(task)

            return _invoke()

        return _tracked

    async def drain(self, grace_s: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_s
        stable_idle_turns = 0
        while loop.time() < deadline:
            active = tuple(task for task in self._active if not task.done())
            if active:
                stable_idle_turns = 0
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                _, pending = await asyncio.wait(active, timeout=remaining)
                if pending:
                    break
                continue

            generation = self._generation
            await asyncio.sleep(0)
            if any(not task.done() for task in self._active) or self._generation != generation:
                stable_idle_turns = 0
                continue
            stable_idle_turns += 1
            if stable_idle_turns >= _CALLBACK_IDLE_TURNS:
                return False

        self._outcome.fail("callback_drain")
        for task in tuple(self._active):
            if not task.done():
                task.add_done_callback(_consume_task_exception)
                task.cancel(_FORCED_CALLBACK_CANCEL)
        return True


@dataclass(slots=True)
class _OwnedResource:
    resource: Any
    method_name: str
    kind: str


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    task.exception()


async def _invoke_method(
    resource: Any,
    method_name: str,
    kwargs: dict[str, Any] | None = None,
) -> None:
    method = getattr(resource, method_name)
    if not callable(method):
        raise TypeError("cleanup method is unavailable")
    result = method(**(kwargs or {}))
    if inspect.isawaitable(result):
        await result


async def _bounded_method(
    owner: _OwnedResource,
    *,
    grace_s: float,
    kwargs: dict[str, Any] | None = None,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_s
    task = asyncio.create_task(
        _invoke_method(owner.resource, owner.method_name, kwargs),
        name=f"article-browser-cleanup-{owner.kind}",
    )
    _, pending = await asyncio.wait({task}, timeout=max(0.0, deadline - loop.time()))
    if pending:
        task.add_done_callback(_consume_task_exception)
        task.cancel()
        return True
    if task.cancelled():
        return True
    return task.exception() is not None


class GuardedArticleBrowser:
    """Own one Chromium acquisition and guard every browser destination."""

    def __init__(
        self,
        *,
        egress_guard: ProbeEgressGuard,
        context: RuntimeRequestContext,
        launcher: Any | None = None,
        capability_check: Any = _playwright_has_required_routing,
        cleanup_grace_s: float = _DEFAULT_CLEANUP_GRACE_S,
    ) -> None:
        self._egress_guard = egress_guard
        self._context = context
        self._launcher = launcher or _DefaultPlaywrightLauncher()
        self._capability_check = capability_check
        self._cleanup_grace_s = _normalize_grace(cleanup_grace_s)

    async def _decision_allowed(self, url: str) -> bool:
        decision = await self._egress_guard.decide(
            url,
            context=replace(self._context, stage="fetch"),
        )
        return bool(decision.allowed)

    async def _abort_http(
        self,
        route: RuntimeBrowserRoute,
        outcome: _AcquisitionOutcome,
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
        outcome: _AcquisitionOutcome,
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
        outcome: _AcquisitionOutcome,
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
        outcome: _AcquisitionOutcome,
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
            connect = getattr(route, "connect_to_server", None)
            if not callable(connect):
                await self._close_websocket(
                    route,
                    outcome,
                    code=1008,
                    reason="Policy denied",
                    failure_stage="capability",
                )
                return
            try:
                # Playwright starts its browser-owned connection task synchronously.
                # Later transport failures are not exposed through this public API.
                connect()
            except Exception:  # noqa: BLE001 - immediate invocation failures fail closed
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

    async def _teardown(
        self,
        resources: list[_OwnedResource],
        *,
        context: Any | None,
        http_routing_started: bool,
        lifecycle: _CallbackLifecycle,
    ) -> bool:
        cleanup_failed = False
        by_kind = {owner.kind: owner for owner in resources}
        if context is not None and http_routing_started:
            cleanup_failed |= await _bounded_method(
                _OwnedResource(context, "unroute_all", "http-routes"),
                grace_s=self._cleanup_grace_s,
                kwargs={"behavior": "wait"},
            )
        for kind in ("page", "context"):
            owner = by_kind.get(kind)
            if owner is not None:
                cleanup_failed |= await _bounded_method(
                    owner,
                    grace_s=self._cleanup_grace_s,
                )
        cleanup_failed |= await lifecycle.drain(self._cleanup_grace_s)
        for kind in ("browser", "playwright"):
            owner = by_kind.get(kind)
            if owner is not None:
                cleanup_failed |= await _bounded_method(
                    owner,
                    grace_s=self._cleanup_grace_s,
                )
        return cleanup_failed

    @staticmethod
    async def _await_teardown(task: asyncio.Task[bool]) -> tuple[bool, bool]:
        caller_cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                caller_cancelled = True
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        return task.result(), caller_cancelled

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

        resources: list[_OwnedResource] = []
        outcome = _AcquisitionOutcome()
        lifecycle = _CallbackLifecycle(outcome)
        primary_error: BaseException | None = None
        html: str | None = None
        context: Any | None = None
        http_routing_started = False
        stage = "launch"
        try:
            playwright = await self._launcher.start()
            resources.append(_OwnedResource(playwright, "stop", "playwright"))
            browser = await playwright.chromium.launch(headless=True)
            resources.append(_OwnedResource(browser, "close", "browser"))
            stage = "context"
            context = await browser.new_context(
                service_workers="block",
                user_agent=profile.user_agent,
                viewport={
                    "width": profile.viewport_width,
                    "height": profile.viewport_height,
                },
            )
            resources.append(_OwnedResource(context, "close", "context"))

            route = getattr(context, "route", None)
            route_web_socket = getattr(context, "route_web_socket", None)
            unroute_all = getattr(context, "unroute_all", None)
            if not all(callable(item) for item in (route, route_web_socket, unroute_all)):
                raise ArticleFailure("browser_error", "capability")

            stage = "routing"
            await route(
                _ROUTE_PATTERN,
                lifecycle.handler(lambda intercepted: self._http_handler(intercepted, outcome)),
            )
            http_routing_started = True
            await route_web_socket(
                _ROUTE_PATTERN,
                lifecycle.handler(lambda intercepted: self._websocket_handler(intercepted, outcome)),
            )
            page = await context.new_page()
            resources.append(_OwnedResource(page, "close", "page"))

            stage = "navigation"
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=profile.timeout_ms,
            )
            stage = "content"
            html = str(await page.content())
        except asyncio.CancelledError as exc:
            primary_error = exc
        except ArticleFailure as exc:
            primary_error = exc
        except Exception:  # noqa: BLE001 - sanitize the Playwright boundary
            primary_error = outcome.failure or ArticleFailure("browser_error", stage)

        teardown_task = asyncio.create_task(
            self._teardown(
                resources,
                context=context,
                http_routing_started=http_routing_started,
                lifecycle=lifecycle,
            ),
            name="article-browser-teardown",
        )
        cleanup_failed, teardown_cancelled = await self._await_teardown(teardown_task)

        if isinstance(primary_error, asyncio.CancelledError) or teardown_cancelled or outcome.callback_cancelled:
            raise asyncio.CancelledError()
        if primary_error is not None:
            raise primary_error
        if outcome.failure is not None:
            raise outcome.failure
        if cleanup_failed:
            raise ArticleFailure("browser_error", "cleanup")
        if html is None:
            raise ArticleFailure("browser_error", "content")
        return html


__all__ = ["GuardedArticleBrowser"]
