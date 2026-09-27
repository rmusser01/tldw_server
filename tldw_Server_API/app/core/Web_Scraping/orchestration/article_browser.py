"""Fail-closed direct-browser acquisition for governed article requests."""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import math
import threading
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    BrowserTransportDecision,
    default_browser_transport_decision,
    resolve_browser_transport_decision,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.browser import (
    RuntimeBrowserRoute,
    RuntimeWebSocketRoute,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressGuard
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

from .article_models import ArticleFailure, ArticleLimits, DirectBrowserProfile

_ROUTE_PATTERN = "**/*"
_HTTP_SCHEMES = frozenset({"http", "https"})
_WEBSOCKET_POLICY_SCHEMES = {"ws": "http", "wss": "https"}
_DEFAULT_CLEANUP_GRACE_S = 1.0
_DEFAULT_ACQUISITION_CAPACITY = 32
_MAX_ACQUISITION_CAPACITY = 256
_DEFAULT_CALLBACK_CAPACITY = 64
_MAX_CALLBACK_CAPACITY = 1024
_CALLBACK_IDLE_TURNS = 3
_BROWSER_RETRY_DELAY_S = 2.0
_RETRYABLE_BROWSER_STAGES = frozenset({"launch", "context", "page", "navigation", "stealth", "wait", "content"})
_ISOLATED_WORLD_NAME = "tldw-article-serialization-v1"
_MAX_SAFE_JAVASCRIPT_INTEGER = 2**53 - 1
_BASE64_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
_HTML_SERIALIZATION_EXPRESSION = """(maxBytes) => {
  const doctype = document.doctype
    ? new XMLSerializer().serializeToString(document.doctype) + "\\n"
    : "";
  const html = doctype + document.documentElement.outerHTML;
  let size = 0;
  for (const character of html) {
    const codePoint = character.codePointAt(0);
    size += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;
    if (size > maxBytes) {
      return { ok: false, size };
    }
  }
  return { ok: true, html };
}"""


def _strict_capacity(value: object, *, maximum: int, label: str) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"Browser {label} capacity must be an integer from 1 to {maximum}")
    return value


class _BrowserAcquisitionPool:
    """Hold finite process-wide capacity until admitted async work really exits."""

    def __init__(self, capacity: int) -> None:
        self.capacity = _strict_capacity(
            capacity,
            maximum=_MAX_ACQUISITION_CAPACITY,
            label="acquisition",
        )
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._lock = threading.Lock()
        self._leases: set[_BrowserAcquisitionLease] = set()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._leases)

    def acquire(self, *, callback_capacity: int) -> _BrowserAcquisitionLease | None:
        if not self._slots.acquire(blocking=False):
            return None
        try:
            lease = _BrowserAcquisitionLease(
                pool=self,
                callback_capacity=callback_capacity,
            )
        except BaseException:
            self._slots.release()
            raise
        with self._lock:
            self._leases.add(lease)
        return lease

    def _release(self, lease: _BrowserAcquisitionLease) -> None:
        with self._lock:
            if lease not in self._leases:
                return
            self._leases.remove(lease)
        self._slots.release()


class _BrowserAcquisitionLease:
    """Own one acquisition and every callback or cleanup task it admits."""

    def __init__(
        self,
        *,
        pool: _BrowserAcquisitionPool,
        callback_capacity: int,
    ) -> None:
        self._pool = pool
        self._callback_capacity = callback_capacity
        self._lock = threading.Lock()
        self._tasks: dict[asyncio.Task[Any], bool] = {}
        self._active_callbacks = 0
        self._callback_generation = 0
        self._callback_admission_open = True
        self._shutdown_claimed = False
        self._shutdown_task: asyncio.Task[Any] | None = None
        self._shutdown_task_observable = False
        self._sealed = False
        self._owner_finished = False
        self._released = False

    def admit_callback(self, task: asyncio.Task[Any]) -> str:
        with self._lock:
            self._callback_generation += 1
            if self._sealed or not self._callback_admission_open:
                return "closed"
            if self._active_callbacks >= self._callback_capacity:
                return "capacity"
            self._active_callbacks += 1
            self._tasks[task] = True
        try:
            task.add_done_callback(self._task_done)
        except BaseException:  # noqa: BLE001 - failed observation must consume capacity
            # The task remains strongly owned forever because completion cannot
            # be observed safely enough to release process capacity.
            return "callback"
        return "admitted"

    def start_cleanup(
        self,
        operation: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> tuple[asyncio.Task[Any] | None, str]:
        """Create and retain cleanup atomically with respect to sealing."""
        with self._lock:
            if self._sealed:
                if inspect.iscoroutine(operation):
                    operation.close()
                return None, "sealed"
            try:
                task = asyncio.create_task(operation, name=name)
            except BaseException:  # noqa: BLE001 - creation failures are sanitized
                if inspect.iscoroutine(operation):
                    operation.close()
                return None, "failed"
            self._tasks[task] = False
        try:
            task.add_done_callback(self._task_done)
        except BaseException:  # noqa: BLE001 - failed observation must consume capacity
            # Never detach cleanup work. A task without a completion callback
            # permanently consumes this lease and its process capacity.
            return task, "unobserved"
        return task, "retained"

    def ensure_emergency_shutdown(
        self,
        factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> str:
        """Atomically own at most one context shutdown for rejected callbacks."""
        operation: Awaitable[None] | None = None
        with self._lock:
            if self._sealed:
                return "sealed"
            if self._shutdown_claimed:
                return "existing"
            self._shutdown_claimed = True
            try:
                operation = factory()
                task = asyncio.create_task(
                    operation,
                    name="article-browser-emergency-context-shutdown",
                )
            except BaseException:  # noqa: BLE001 - normal teardown remains available
                if inspect.iscoroutine(operation):
                    operation.close()
                return "failed"
            self._tasks[task] = False
            self._shutdown_task = task
        try:
            task.add_done_callback(self._task_done)
        except BaseException:  # noqa: BLE001 - permanent retention is fail closed
            return "unobserved"
        with self._lock:
            if task is self._shutdown_task:
                self._shutdown_task_observable = True
        return "started"

    def close_callback_admission(self) -> tuple[asyncio.Task[Any] | None, bool]:
        with self._lock:
            self._callback_admission_open = False
            self._shutdown_claimed = True
            return self._shutdown_task, self._shutdown_task_observable

    def seal(self) -> None:
        """Permanently reject new tasks before acquisition ownership can end."""
        with self._lock:
            self._callback_admission_open = False
            self._sealed = True

    def callback_snapshot(self) -> tuple[int, tuple[asyncio.Task[Any], ...]]:
        with self._lock:
            tasks = tuple(task for task, is_callback in self._tasks.items() if is_callback and not task.done())
            return self._callback_generation, tasks

    def owner_finished(self) -> None:
        release = False
        with self._lock:
            self._owner_finished = True
            release = self._mark_released_if_idle()
        if release:
            self._pool._release(self)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        try:
            _task_failed(task)
        finally:
            release = False
            with self._lock:
                is_callback = self._tasks.pop(task, None)
                if is_callback:
                    self._active_callbacks -= 1
                if task is self._shutdown_task:
                    self._shutdown_task = None
                    self._shutdown_task_observable = False
                release = self._mark_released_if_idle()
            if release:
                self._pool._release(self)

    def _mark_released_if_idle(self) -> bool:
        if self._released or not self._sealed or not self._owner_finished or self._tasks:
            return False
        self._released = True
        return True


_BROWSER_ACQUISITION_POOL = _BrowserAcquisitionPool(_DEFAULT_ACQUISITION_CAPACITY)


class _DefaultPlaywrightLauncher:
    async def start(self) -> Any:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ArticleFailure("browser_error", "capability") from None
        return await async_playwright().start()


async def _default_stealth_hook(page: Any) -> None:
    try:
        from playwright_stealth import stealth_async
    except ImportError:
        return
    await stealth_async(page)


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


def _http_policy_url(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in _HTTP_SCHEMES:
            return None
        if not parsed.netloc or parsed.hostname is None:
            return None
        _ = parsed.port
    except (TypeError, ValueError):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _http_url_is_valid(url: str) -> bool:
    return _http_policy_url(url) is not None


def _uncancel_task(task: Any) -> None:
    uncancel = getattr(task, "uncancel", None)
    if callable(uncancel):
        uncancel()


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
        self._lock = threading.Lock()

    def fail(self, stage: str) -> None:
        self.latch(ArticleFailure("browser_error", stage))

    def latch(self, failure: ArticleFailure) -> None:
        with self._lock:
            if self.failure is None:
                self.failure = failure


class _BrowserTransferLedger:
    """Synchronously account CDP transfer events and latch one terminal failure."""

    def __init__(
        self,
        *,
        limit: int,
        outcome: _AcquisitionOutcome,
        lease: _BrowserAcquisitionLease,
        shutdown_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self._limit = limit
        self._outcome = outcome
        self._lease = lease
        self._shutdown_factory = shutdown_factory
        self._lock = threading.Lock()
        self._total = 0
        self._terminal = False

    def on_data_received(self, payload: object) -> None:
        try:
            if not isinstance(payload, Mapping):
                raise TypeError("invalid CDP event envelope")
            amount = self._non_negative_integer(payload.get("encodedDataLength"))
        except (TypeError, ValueError, OverflowError):
            self._fail(ArticleFailure("browser_error", "capability"))
            return
        self._add(amount)

    def on_websocket_frame(self, payload: object) -> None:
        try:
            if not isinstance(payload, Mapping):
                raise TypeError("invalid CDP event envelope")
            response = payload.get("response")
            if not isinstance(response, Mapping):
                raise TypeError("missing websocket frame")
            opcode = response.get("opcode")
            data = response.get("payloadData")
            if type(opcode) is not int or not isinstance(data, str):
                raise TypeError("invalid websocket frame")
            if opcode == 1:
                self._add_text(data)
                return
            elif opcode == 2:
                self._add_binary(data)
                return
            else:
                raise ValueError("unsupported websocket opcode")
        except (binascii.Error, TypeError, ValueError, UnicodeError):
            self._fail(ArticleFailure("browser_error", "capability"))

    @staticmethod
    def _non_negative_integer(value: object) -> int:
        if type(value) is int:
            if value < 0:
                raise ValueError("invalid transfer length")
            return value
        if type(value) is not float:
            raise TypeError("invalid transfer length")
        if not math.isfinite(value) or value < 0 or not value.is_integer() or value > _MAX_SAFE_JAVASCRIPT_INTEGER:
            raise ValueError("invalid transfer length")
        return int(value)

    def _add_text(self, data: str) -> None:
        failure: ArticleFailure | None = None
        with self._lock:
            if self._terminal:
                return
            remaining = self._limit - self._total
            amount = 0
            try:
                for character in data:
                    codepoint = ord(character)
                    if codepoint <= 0x7F:
                        width = 1
                    elif codepoint <= 0x7FF:
                        width = 2
                    elif 0xD800 <= codepoint <= 0xDFFF:
                        raise UnicodeError("surrogate is not valid UTF-8 text")
                    elif codepoint <= 0xFFFF:
                        width = 3
                    elif codepoint <= 0x10FFFF:
                        width = 4
                    else:
                        raise UnicodeError("invalid Unicode code point")
                    amount += width
                    if amount > remaining:
                        self._terminal = True
                        failure = ArticleFailure(
                            "response_too_large",
                            "browser_transfer",
                        )
                        break
                else:
                    self._total += amount
            except Exception:  # noqa: BLE001 - malformed CDP text fails closed
                self._terminal = True
                failure = ArticleFailure("browser_error", "capability")
        if failure is not None:
            self._latch_and_shutdown(failure)

    def _add_binary(self, data: str) -> None:
        failure: ArticleFailure | None = None
        with self._lock:
            if self._terminal:
                return
            try:
                encoded_length = len(data)
                if encoded_length % 4 != 0:
                    raise ValueError("invalid base64 length")
                remaining = self._limit - self._total
                minimum_decoded_length = max(0, (encoded_length // 4) * 3 - 2)
                if minimum_decoded_length > remaining:
                    self._terminal = True
                    failure = ArticleFailure(
                        "response_too_large",
                        "browser_transfer",
                    )
                else:
                    padding = 0
                    saw_padding = False
                    for character in data:
                        if character == "=":
                            saw_padding = True
                            padding += 1
                            if padding > 2:
                                raise ValueError("invalid base64 padding")
                        elif character in _BASE64_ALPHABET:
                            if saw_padding:
                                raise ValueError("base64 data follows padding")
                        else:
                            raise ValueError("invalid base64 character")
                    amount = (encoded_length // 4) * 3 - padding
                    if amount > remaining:
                        self._terminal = True
                        failure = ArticleFailure(
                            "response_too_large",
                            "browser_transfer",
                        )
                    else:
                        decoded = base64.b64decode(data, validate=True)
                        if len(decoded) != amount:
                            raise ValueError("invalid base64 decoded length")
                        self._total += amount
            except (binascii.Error, TypeError, ValueError, UnicodeError):
                self._terminal = True
                failure = ArticleFailure("browser_error", "capability")
        if failure is not None:
            self._latch_and_shutdown(failure)

    def _add(self, amount: int) -> None:
        failure: ArticleFailure | None = None
        with self._lock:
            if self._terminal:
                return
            self._total += amount
            if self._total > self._limit:
                self._terminal = True
                failure = ArticleFailure("response_too_large", "browser_transfer")
        if failure is not None:
            self._latch_and_shutdown(failure)

    def _fail(self, failure: ArticleFailure) -> None:
        with self._lock:
            if self._terminal:
                return
            self._terminal = True
        self._latch_and_shutdown(failure)

    def _latch_and_shutdown(self, failure: ArticleFailure) -> None:
        self._outcome.latch(failure)
        self._lease.ensure_emergency_shutdown(self._shutdown_factory)


class _CallbackLifecycle:
    """Track independently scheduled Playwright callback invocations."""

    def __init__(
        self,
        outcome: _AcquisitionOutcome,
        lease: _BrowserAcquisitionLease,
        shutdown_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self._outcome = outcome
        self._lease = lease
        self._shutdown_factory = shutdown_factory

    def handler(
        self,
        operation: Callable[[Any], Awaitable[None]],
        rejection: Callable[[Any], Awaitable[None]],
    ) -> Callable[[Any], Awaitable[None]]:
        def _tracked(route: Any) -> Awaitable[None]:
            task = asyncio.current_task()
            if task is None:
                self._outcome.fail("callback")
                admission = "callback"
            else:
                admission = self._lease.admit_callback(task)
            if admission != "admitted":
                failure_stage = "capacity" if admission == "capacity" else "callback"
                self._outcome.fail(failure_stage)
                self._lease.ensure_emergency_shutdown(self._shutdown_factory)

                async def _reject() -> None:
                    try:
                        await rejection(route)
                    except asyncio.CancelledError:
                        self._outcome.callback_cancelled = True
                    except Exception:  # noqa: BLE001 - callback failures are sanitized
                        self._outcome.fail("callback")

                return _reject()

            async def _invoke() -> None:
                try:
                    await operation(route)
                except asyncio.CancelledError:
                    self._outcome.callback_cancelled = True
                except Exception:  # noqa: BLE001 - callback failures are sanitized
                    self._outcome.fail("callback")

            return _invoke()

        return _tracked

    def close_admission(self) -> tuple[asyncio.Task[Any] | None, bool]:
        return self._lease.close_callback_admission()

    async def drain(self, grace_s: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_s
        stable_idle_turns = 0
        while loop.time() < deadline:
            generation, active = self._lease.callback_snapshot()
            if active:
                stable_idle_turns = 0
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                _, pending = await asyncio.wait(active, timeout=remaining)
                if pending:
                    break
                continue

            await asyncio.sleep(0)
            next_generation, next_active = self._lease.callback_snapshot()
            if next_active or next_generation != generation:
                stable_idle_turns = 0
                continue
            stable_idle_turns += 1
            if stable_idle_turns >= _CALLBACK_IDLE_TURNS:
                return False

        self._outcome.fail("callback_drain")
        return True


@dataclass(slots=True)
class _OwnedResource:
    resource: Any
    method_name: str
    kind: str


def _task_failed(task: asyncio.Task[Any]) -> bool:
    if task.cancelled():
        return True
    try:
        return task.exception() is not None
    except BaseException:  # noqa: BLE001 - task bookkeeping must remain fail closed
        return True


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
    lease: _BrowserAcquisitionLease,
    grace_s: float,
    kwargs: dict[str, Any] | None = None,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_s
    operation = _invoke_method(owner.resource, owner.method_name, kwargs)
    task, retention = lease.start_cleanup(
        operation,
        name=f"article-browser-cleanup-{owner.kind}",
    )
    if task is None:
        return True
    if retention == "unobserved":
        task.cancel()
        return True
    _, pending = await asyncio.wait({task}, timeout=max(0.0, deadline - loop.time()))
    if pending:
        task.cancel()
        return True
    return _task_failed(task)


async def _bounded_retained_task(task: asyncio.Task[Any], *, grace_s: float) -> bool:
    _, pending = await asyncio.wait({task}, timeout=grace_s)
    if pending:
        task.cancel()
        return True
    return _task_failed(task)


class GuardedArticleBrowser:
    """Own one Chromium acquisition and guard every browser destination."""

    def __init__(
        self,
        *,
        egress_guard: ProbeEgressGuard,
        context: RuntimeRequestContext,
        launcher: Any | None = None,
        transport_decision: Callable[[], BrowserTransportDecision] = default_browser_transport_decision,
        capability_check: Any = _playwright_has_required_routing,
        cleanup_grace_s: float = _DEFAULT_CLEANUP_GRACE_S,
        acquisition_pool: _BrowserAcquisitionPool = _BROWSER_ACQUISITION_POOL,
        callback_capacity: int = _DEFAULT_CALLBACK_CAPACITY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        stealth_hook: Callable[[Any], Awaitable[None]] = _default_stealth_hook,
    ) -> None:
        self._egress_guard = egress_guard
        self._context = context
        self._launcher = launcher or _DefaultPlaywrightLauncher()
        self._transport_decision = transport_decision
        self._capability_check = capability_check
        self._cleanup_grace_s = _normalize_grace(cleanup_grace_s)
        self._acquisition_pool = acquisition_pool
        self._sleep = sleep
        self._stealth_hook = stealth_hook
        self._callback_capacity = _strict_capacity(
            callback_capacity,
            maximum=_MAX_CALLBACK_CAPACITY,
            label="callback",
        )

    def _resolve_transport_decision(self) -> BrowserTransportDecision:
        """Resolve browser admission without exposing provider failures."""
        return resolve_browser_transport_decision(
            self._transport_decision,
            component="article_browser",
        )

    def transport_capability(self) -> dict[str, str | bool]:
        """Return the current bounded browser-transport capability snapshot."""
        return self._resolve_transport_decision().to_capability_metadata()

    @staticmethod
    def _should_retry(failure: ArticleFailure) -> bool:
        return (
            failure.code == "browser_error"
            and failure.stage in _RETRYABLE_BROWSER_STAGES
            and not failure.retry_suppressed
        )

    @staticmethod
    async def _install_transfer_accounting(
        session: Any,
        ledger: _BrowserTransferLedger,
    ) -> None:
        on = getattr(session, "on", None)
        send = getattr(session, "send", None)
        if not callable(on) or not callable(send):
            raise ArticleFailure("browser_error", "capability")
        try:
            for event, handler in (
                ("Network.dataReceived", ledger.on_data_received),
                ("Network.webSocketFrameReceived", ledger.on_websocket_frame),
                ("Network.webSocketFrameSent", ledger.on_websocket_frame),
            ):
                registered = on(event, handler)
                if inspect.isawaitable(registered):
                    await registered
            enabled = send("Network.enable")
            if not inspect.isawaitable(enabled):
                raise TypeError("CDP send is not awaitable")
            await enabled
        except asyncio.CancelledError:
            raise
        except ArticleFailure:
            raise
        except Exception:  # noqa: BLE001 - CDP capability failures are sanitized
            raise ArticleFailure("browser_error", "capability") from None

    @staticmethod
    async def _serialize_html(session: Any, max_bytes: int) -> str:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ArticleFailure("browser_error", "capability")
        send = getattr(session, "send", None)
        if not callable(send):
            raise ArticleFailure("browser_error", "capability")

        async def command(
            method: str,
            params: dict[str, object] | None = None,
        ) -> Mapping[str, Any]:
            try:
                operation = send(method) if params is None else send(method, params)
                if not inspect.isawaitable(operation):
                    raise TypeError("CDP send is not awaitable")
                response = await operation
            except asyncio.CancelledError:
                raise
            except ArticleFailure:
                raise
            except Exception:  # noqa: BLE001 - CDP failures are sanitized
                raise ArticleFailure("browser_error", "capability") from None
            if not isinstance(response, Mapping) or "exceptionDetails" in response:
                raise ArticleFailure("browser_error", "capability")
            return response

        frame_tree_response = await command("Page.getFrameTree")
        frame_tree = frame_tree_response.get("frameTree")
        if not isinstance(frame_tree, Mapping):
            raise ArticleFailure("browser_error", "capability")
        frame = frame_tree.get("frame")
        if not isinstance(frame, Mapping):
            raise ArticleFailure("browser_error", "capability")
        frame_id = frame.get("id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ArticleFailure("browser_error", "capability")

        isolated_world_response = await command(
            "Page.createIsolatedWorld",
            {
                "frameId": frame_id,
                "worldName": _ISOLATED_WORLD_NAME,
                "grantUniveralAccess": False,
            },
        )
        context_id = isolated_world_response.get("executionContextId")
        if type(context_id) is not int or context_id < 0:
            raise ArticleFailure("browser_error", "capability")

        evaluation_response = await command(
            "Runtime.evaluate",
            {
                "expression": f"({_HTML_SERIALIZATION_EXPRESSION})({max_bytes})",
                "contextId": context_id,
                "returnByValue": True,
            },
        )
        remote_result = evaluation_response.get("result")
        if not isinstance(remote_result, Mapping):
            raise ArticleFailure("browser_error", "capability")
        result = remote_result.get("value")
        if not isinstance(result, Mapping) or type(result.get("ok")) is not bool:
            raise ArticleFailure("browser_error", "capability")
        if result["ok"] is True:
            html = result.get("html")
            if not isinstance(html, str):
                raise ArticleFailure("browser_error", "capability")
            return html
        if "html" in result:
            raise ArticleFailure("browser_error", "capability")
        try:
            _BrowserTransferLedger._non_negative_integer(result.get("size"))
        except (TypeError, ValueError, OverflowError):
            raise ArticleFailure("browser_error", "capability") from None
        raise ArticleFailure("response_too_large", "rendered_html")

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
            policy_url = _http_policy_url(url)
            if policy_url is None:
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            try:
                allowed = await self._decision_allowed(policy_url)
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
                connection = connect()
                if inspect.isawaitable(connection):
                    await connection
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
        lease: _BrowserAcquisitionLease,
    ) -> bool:
        cleanup_failed = False
        by_kind = {owner.kind: owner for owner in resources}
        emergency_shutdown, shutdown_observable = lifecycle.close_admission()
        if emergency_shutdown is not None:
            if shutdown_observable:
                cleanup_failed |= await _bounded_retained_task(
                    emergency_shutdown,
                    grace_s=self._cleanup_grace_s,
                )
            else:
                emergency_shutdown.cancel()
                cleanup_failed = True
        if context is not None and http_routing_started:
            cleanup_failed |= await _bounded_method(
                _OwnedResource(context, "unroute_all", "http-routes"),
                lease=lease,
                grace_s=self._cleanup_grace_s,
                kwargs={"behavior": "wait"},
            )
        for kind in ("cdp", "page", "context"):
            owner = by_kind.get(kind)
            if owner is not None:
                cleanup_failed |= await _bounded_method(
                    owner,
                    lease=lease,
                    grace_s=self._cleanup_grace_s,
                )
        cleanup_failed |= await lifecycle.drain(self._cleanup_grace_s)
        for kind in ("browser", "playwright"):
            owner = by_kind.get(kind)
            if owner is not None:
                cleanup_failed |= await _bounded_method(
                    owner,
                    lease=lease,
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
                    _uncancel_task(current)
        return task.result(), caller_cancelled

    async def acquire(
        self,
        url: str,
        profile: DirectBrowserProfile,
        limits: ArticleLimits | None = None,
    ) -> str:
        """Return rendered HTML after freshly guarding every browser dispatch."""
        if limits is None:
            limits = ArticleLimits()
        elif not isinstance(limits, ArticleLimits):
            raise TypeError("limits must be an ArticleLimits instance")

        attempts = profile.retries
        if attempts == 0:
            return ""
        decision = self._resolve_transport_decision()
        if not decision.allowed:
            raise ArticleFailure(
                "browser_transport_unavailable",
                decision.reason,
                capability=decision.to_capability_metadata(),
            )
        for attempt in range(attempts):
            lease = self._acquisition_pool.acquire(
                callback_capacity=self._callback_capacity,
            )
            if lease is None:
                raise ArticleFailure("browser_error", "capacity")
            try:
                return await self._acquire_with_lease(url, profile, limits, lease)
            except ArticleFailure as failure:
                if attempt + 1 >= attempts or not self._should_retry(failure):
                    raise
            finally:
                lease.seal()
                lease.owner_finished()
            await self._sleep(_BROWSER_RETRY_DELAY_S)

        raise ArticleFailure("browser_error", "launch")

    async def _acquire_with_lease(
        self,
        url: str,
        profile: DirectBrowserProfile,
        limits: ArticleLimits,
        lease: _BrowserAcquisitionLease,
    ) -> str:
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
        primary_error: BaseException | None = None
        html: str | None = None
        context: Any | None = None

        async def _shutdown_context() -> None:
            if context is None:
                raise TypeError("browser context is unavailable")
            await _invoke_method(context, "close")

        lifecycle = _CallbackLifecycle(outcome, lease, _shutdown_context)
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

            if profile.custom_cookies:
                add_cookies = getattr(context, "add_cookies", None)
                if not callable(add_cookies):
                    raise ArticleFailure("browser_error", "capability")
                await add_cookies([dict(cookie) for cookie in profile.custom_cookies])

            route = getattr(context, "route", None)
            route_web_socket = getattr(context, "route_web_socket", None)
            unroute_all = getattr(context, "unroute_all", None)
            if not callable(route) or not callable(route_web_socket) or not callable(unroute_all):
                raise ArticleFailure("browser_error", "capability")

            stage = "routing"
            await route(
                _ROUTE_PATTERN,
                lifecycle.handler(
                    lambda intercepted: self._http_handler(intercepted, outcome),
                    lambda intercepted: self._abort_http(
                        intercepted,
                        outcome,
                        failure_stage="callback",
                    ),
                ),
            )
            http_routing_started = True
            await route_web_socket(
                _ROUTE_PATTERN,
                lifecycle.handler(
                    lambda intercepted: self._websocket_handler(intercepted, outcome),
                    lambda intercepted: self._close_websocket(
                        intercepted,
                        outcome,
                        code=1008,
                        reason="Policy denied",
                        failure_stage="callback",
                    ),
                ),
            )
            stage = "page"
            page = await context.new_page()
            resources.append(_OwnedResource(page, "close", "page"))

            new_cdp_session = getattr(context, "new_cdp_session", None)
            if not callable(new_cdp_session):
                raise ArticleFailure("browser_error", "capability")
            try:
                cdp_session = await new_cdp_session(page)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - CDP capability failures are sanitized
                raise ArticleFailure("browser_error", "capability") from None
            if not callable(getattr(cdp_session, "detach", None)):
                raise ArticleFailure("browser_error", "capability")
            resources.append(_OwnedResource(cdp_session, "detach", "cdp"))
            transfer = _BrowserTransferLedger(
                limit=limits.max_browser_transfer_bytes,
                outcome=outcome,
                lease=lease,
                shutdown_factory=_shutdown_context,
            )
            await self._install_transfer_accounting(cdp_session, transfer)

            if profile.stealth_enabled:
                stage = "stealth"
                await self._stealth_hook(page)

            stage = "navigation"
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=profile.timeout_ms,
            )
            if outcome.failure is not None:
                raise outcome.failure
            if profile.stealth_enabled:
                wait_for_timeout = getattr(page, "wait_for_timeout", None)
                if not callable(wait_for_timeout):
                    raise ArticleFailure("browser_error", "capability")
                stage = "wait"
                await wait_for_timeout(profile.stealth_wait_ms)
            else:
                wait_for_load_state = getattr(page, "wait_for_load_state", None)
                if not callable(wait_for_load_state):
                    raise ArticleFailure("browser_error", "capability")
                stage = "wait"
                await wait_for_load_state("networkidle", timeout=profile.timeout_ms)
            if outcome.failure is not None:
                raise outcome.failure
            stage = "content"
            html = await self._serialize_html(cdp_session, limits.max_article_bytes)
        except asyncio.CancelledError as exc:
            primary_error = exc
        except ArticleFailure as exc:
            primary_error = exc
        except Exception:  # noqa: BLE001 - sanitize the Playwright boundary
            primary_error = outcome.failure or ArticleFailure("browser_error", stage)

        teardown_operation = self._teardown(
            resources,
            context=context,
            http_routing_started=http_routing_started,
            lifecycle=lifecycle,
            lease=lease,
        )
        cleanup_failed = False
        teardown_cancelled = False
        try:
            teardown_task = asyncio.create_task(
                teardown_operation,
                name="article-browser-teardown",
            )
        except Exception:  # noqa: BLE001 - creation failures are sanitized
            teardown_operation.close()
            cleanup_failed = True
            fallback_operation = self._teardown(
                resources,
                context=context,
                http_routing_started=http_routing_started,
                lifecycle=lifecycle,
                lease=lease,
            )
            try:
                teardown_task = asyncio.get_running_loop().create_task(
                    fallback_operation,
                    name="article-browser-teardown-fallback",
                )
            except Exception:  # noqa: BLE001 - lease finalization remains authoritative
                fallback_operation.close()
            else:
                fallback_failed, teardown_cancelled = await self._await_teardown(teardown_task)
                cleanup_failed |= fallback_failed
        else:
            cleanup_failed, teardown_cancelled = await self._await_teardown(teardown_task)

        if isinstance(primary_error, asyncio.CancelledError) or teardown_cancelled or outcome.callback_cancelled:
            raise asyncio.CancelledError()
        transfer_failure = outcome.failure
        if (
            transfer_failure is not None
            and transfer_failure.code == "response_too_large"
            and transfer_failure.stage == "browser_transfer"
        ):
            raise transfer_failure
        if primary_error is not None:
            if cleanup_failed and isinstance(primary_error, ArticleFailure):
                primary_error.retry_suppressed = True
            raise primary_error
        if outcome.failure is not None:
            raise outcome.failure
        if cleanup_failed:
            raise ArticleFailure("browser_error", "cleanup")
        if html is None:
            raise ArticleFailure("browser_error", "content")
        return html


__all__ = ["GuardedArticleBrowser"]
