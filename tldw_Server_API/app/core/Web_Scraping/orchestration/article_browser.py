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
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    BrowserTransportDecision,
    bind_selected_browser_proxy,
    default_browser_transport_decision,
    resolve_browser_transport_decision,
    selected_browser_proxy,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.browser import (
    RuntimeBrowserRoute,
    RuntimeWebSocketRoute,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressGuard
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

from ..runtime.controls import ParentControls, ParentStop, physical_work
from .article_browser_dispatch import TransferSessions
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
_DOCUMENT_SERIALIZATION_EXPRESSION = """(maxBytes) => {
  const serialized = (HTML_SERIALIZER)(maxBytes);
  if (!serialized.ok) return serialized;
  let content = "";
  for (const selector of ["main", "article", '[role="main"]', "#content", ".content"]) {
    for (const element of document.querySelectorAll(selector)) {
      const text = element.innerText;
      if (text.length > content.length) content = text;
    }
  }
  if (!content) content = document.body.innerText;
  const author = document.querySelector('meta[name="author"]');
  const date = document.querySelector('meta[property="article:published_time"]');
  const dom = {title: document.title, author: author ? author.content : "Unknown",
               date: date ? date.content : "", content};
  let size = 0;
  for (const text of Object.values(dom)) {
    for (const character of text) {
      const codePoint = character.codePointAt(0);
      size += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;
      if (size > maxBytes) return {ok: false, size};
    }
  }
  return {...serialized, dom};
}""".replace("HTML_SERIALIZER", _HTML_SERIALIZATION_EXPRESSION)


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
        if lease._physical_release is not None:
            lease._physical_release()


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
        self._failed_cleanup_owners: tuple[_OwnedResource, ...] = ()
        owner = physical_work.get()
        self._physical_release = owner.acquire() if owner is not None else None

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

    def start_callback(self, operation: Coroutine[Any, Any, None], *, allow_closed: bool = False) -> str:
        """Admit CDP event work before scheduling, sharing the route callback bound."""
        ready = asyncio.Event()

        async def admitted() -> None:
            try:
                await ready.wait()
                await operation
            finally:
                operation.close()

        with self._lock:
            self._callback_generation += 1
            if self._sealed or (not self._callback_admission_open and not allow_closed):
                operation.close()
                return "closed"
            if self._active_callbacks >= self._callback_capacity:
                operation.close()
                return "capacity"
            wrapper = admitted()
            try:
                task = asyncio.create_task(wrapper, name="article-browser-request-callback")
            except BaseException:  # noqa: BLE001 - creation failure must not retain an unstarted coroutine
                wrapper.close()
                operation.close()
                return "callback"
            self._active_callbacks += 1
            self._tasks[task] = True
        try:
            task.add_done_callback(self._task_done)
        except BaseException:  # noqa: BLE001 - unobservable completion conservatively retains this lease
            task.cancel()
            return "callback"
        ready.set()
        return "admitted"

    def start_cleanup(
        self,
        operation: Coroutine[Any, Any, None],
        *,
        name: str,
    ) -> tuple[asyncio.Task[Any] | None, str]:
        """Create and retain cleanup atomically with respect to sealing."""
        with self._lock:
            if self._sealed and asyncio.current_task() not in self._tasks:
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

    def retain_failed_cleanup(self, owners: Mapping[str, _OwnedResource]) -> None:
        """Permanently retain unobservable cleanup, as for an unobservable task."""
        with self._lock:
            self._failed_cleanup_owners = tuple(owners.values())

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
        if self._released or not self._sealed or not self._owner_finished or self._tasks or self._failed_cleanup_owners:
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
        self.parent_stop: BaseException | None = None
        self.callback_cancelled = False
        self.native_browser_disconnected = False
        self.native_exit = asyncio.Event()
        self.native_exit_handler = None
        self._lock = threading.Lock()

    def watch_browser(self, native: Any) -> None:
        def exited(*_args):
            if native.is_connected() is False:
                self.native_browser_disconnected = True
                self.native_exit.set()

        self.native_exit_handler = exited
        native.on("disconnected", exited)

    def unwatch_browser(self, native: Any) -> None:
        if self.native_exit_handler is not None:
            native.remove_listener("disconnected", self.native_exit_handler)
            self.native_exit_handler = None

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
        parent: ParentControls | None = None,
    ) -> None:
        self._limit = limit
        self._outcome = outcome
        self._lease = lease
        self._shutdown_factory = shutdown_factory
        self._lock = threading.Lock()
        self._total = 0
        self._terminal = False
        self._parent = parent
        self._request_lock = threading.Lock()
        self._request_bytes: dict[tuple[str, str], int] = {}
        self._finished_requests: set[tuple[str, str]] = set()

    def _charge_parent(self, amount: int) -> bool:
        if self._parent is None:
            return True
        try:
            self._parent.check()
            if amount:
                self._parent.charge_bytes(amount)
        except ParentStop as exc:
            self._outcome.parent_stop = exc
            self._lease.ensure_emergency_shutdown(self._shutdown_factory)
            self._lease.close_callback_admission()
            return False
        except asyncio.CancelledError as exc:
            self._outcome.parent_stop = exc
            self._lease.ensure_emergency_shutdown(self._shutdown_factory)
            self._lease.close_callback_admission()
            return False
        return True

    def on_data_received(self, payload: object) -> None:
        try:
            if not isinstance(payload, Mapping):
                raise TypeError("invalid CDP event envelope")
            amount = self._non_negative_integer(payload.get("encodedDataLength"))
        except (TypeError, ValueError, OverflowError):
            self._fail(ArticleFailure("browser_error", "capability"))
            return
        self._add(amount)

    @staticmethod
    def _request_key(session: str, payload: object) -> tuple[str, str]:
        if not isinstance(payload, Mapping):
            raise TypeError("invalid CDP event envelope")
        identifier = payload.get("requestId")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 1024:
            raise ValueError("invalid request identifier")
        return session, identifier

    def on_request_data(self, session: str, payload: object) -> None:
        try:
            key = self._request_key(session, payload)
            amount = self._non_negative_integer(payload.get("encodedDataLength"))
            if amount == 0:
                amount = self._non_negative_integer(payload.get("dataLength"))
            with self._request_lock:
                if key in self._finished_requests:
                    return
                if key not in self._request_bytes and len(self._request_bytes) >= self._lease._callback_capacity:
                    raise ValueError("request observation capacity")
                self._request_bytes[key] = self._request_bytes.get(key, 0) + amount
            self._add(amount)
        except (TypeError, ValueError, OverflowError):
            self._fail(ArticleFailure("browser_error", "capability"))

    def on_request_finished(self, session: str, payload: object) -> None:
        try:
            key = self._request_key(session, payload)
            total = self._non_negative_integer(payload.get("encodedDataLength"))
            with self._request_lock:
                if key in self._finished_requests:
                    return
                charged = self._request_bytes.pop(key, 0)
                if len(self._finished_requests) >= self._lease._callback_capacity:
                    self._finished_requests.pop()
                self._finished_requests.add(key)
            self._add(max(0, total - charged))
        except (TypeError, ValueError, OverflowError):
            self._fail(ArticleFailure("browser_error", "capability"))

    def on_request_failed(self, session: str, payload: object) -> None:
        try:
            key = self._request_key(session, payload)
            with self._request_lock:
                self._request_bytes.pop(key, None)
        except (TypeError, ValueError):
            self._fail(ArticleFailure("browser_error", "capability"))

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
        if not self._charge_parent(0):
            return
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
                    if self._charge_parent(amount):
                        self._total += amount
            except Exception:  # noqa: BLE001 - malformed CDP text fails closed
                self._terminal = True
                failure = ArticleFailure("browser_error", "capability")
        if failure is not None:
            self._latch_and_shutdown(failure)

    def _add_binary(self, data: str) -> None:
        if not self._charge_parent(0):
            return
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
                        if not self._charge_parent(amount):
                            return
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
        if not self._charge_parent(amount):
            return
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
                    except ParentStop as exc:
                        self._outcome.parent_stop = exc
                    except Exception:  # noqa: BLE001 - callback failures are sanitized
                        self._outcome.fail("callback")

                return _reject()

            async def _invoke() -> None:
                try:
                    await operation(route)
                except ParentStop as exc:
                    self._outcome.parent_stop = exc
                    self._lease.ensure_emergency_shutdown(self._shutdown_factory)
                    self._lease.close_callback_admission()
                except asyncio.CancelledError:
                    self._outcome.callback_cancelled = True
                except Exception:  # noqa: BLE001 - callback failures are sanitized
                    self._outcome.fail("callback")

            return _invoke()

        return _tracked

    def close_admission(self) -> tuple[asyncio.Task[Any] | None, bool]:
        return self._lease.close_callback_admission()

    def event_handler(
        self, operation: Callable[[Any], Awaitable[None]], *, allow_closed: bool = False
    ) -> Callable[[Any], None]:
        """Own synchronously delivered CDP events without an unbounded task queue."""

        def dispatch(event: Any) -> None:
            async def invoke() -> None:
                try:
                    await operation(event)
                except ParentStop as exc:
                    self._outcome.parent_stop = exc
                    self._lease.ensure_emergency_shutdown(self._shutdown_factory)
                    self._lease.close_callback_admission()
                except asyncio.CancelledError:
                    self._outcome.callback_cancelled = True
                    self._lease.ensure_emergency_shutdown(self._shutdown_factory)
                except Exception:  # noqa: BLE001 - event/command failures fail closed
                    self._outcome.fail("callback")
                    self._lease.ensure_emergency_shutdown(self._shutdown_factory)

            admission = self._lease.start_callback(invoke(), allow_closed=allow_closed)
            if admission != "admitted":
                self._outcome.fail("capacity" if admission == "capacity" else "callback")
                self._lease.ensure_emergency_shutdown(self._shutdown_factory)

        return dispatch

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


class _BrowserRequestPolicy:
    """Before-dispatch policy, retaining direct versus enhanced credentials."""

    def __init__(
        self, url: str, headers: Mapping[str, str] | None, browser: GuardedArticleBrowser, outcome: _AcquisitionOutcome
    ) -> None:
        self._enhanced = headers is not None
        headers = headers or {}
        self._origin = http_client._url_origin(url)
        self._headers = {k: v for k, v in headers.items() if k.lower() not in {"cookie", "host", "proxy-authorization"}}
        self._raw_cookie = next((value for key, value in headers.items() if key.lower() == "cookie"), None)
        self._browser = browser
        self._outcome = outcome
        self._requests: dict[str, bool] = {}
        self._closed = False
        self._transfer_sessions: TransferSessions | None = None
        self.page_session: Any = None

    async def install(self, session: Any, lifecycle: _CallbackLifecycle) -> None:
        request = lifecycle.event_handler(lambda event: self.handle(event, session))
        abort = lifecycle.event_handler(lambda event: self._abort(session, event["requestId"]), allow_closed=True)
        session.on("Fetch.requestPaused", lambda event: abort(event) if self._closed else request(event))
        metadata = lifecycle.event_handler(self._transfer_sessions.created)
        session.on("Target.targetCreated", lambda event: None if self._closed else metadata(event))
        await session.send("Target.setDiscoverTargets", {"discover": True})
        await session.send(
            "Fetch.enable",
            {
                "patterns": [
                    {"urlPattern": "*", "requestStage": "Request"},
                    {"urlPattern": "*", "requestStage": "Response"},
                ]
            },
        )

    async def bind(
        self,
        session: Any,
        lifecycle: _CallbackLifecycle,
        transfer: _BrowserTransferLedger,
        lease: _BrowserAcquisitionLease,
    ) -> None:
        self._lifecycle, self._lease = lifecycle, lease
        self._root = session
        self._transfer_sessions = TransferSessions(session, lease, transfer, self._fail)
        await self.install(session, lifecycle)

    def _fail(self) -> None:
        self._outcome.fail("capability")
        self._lease.ensure_emergency_shutdown(self._lifecycle._shutdown_factory)

    async def prepare(self, request: Any) -> None:
        if self._closed:
            raise ArticleFailure("browser_error", "callback")

    def close(self) -> None:
        self._closed = True
        if self._transfer_sessions is not None:
            self._transfer_sessions.closed = True
        self._requests.clear()
        self._headers.clear()
        self._raw_cookie = None

    async def retire(self) -> None:
        self._transfer_sessions.retire(native_browser_disconnected=self._outcome.native_browser_disconnected)
        if self.page_session is not None:
            await self._detach(self.page_session)
        await self._detach(self._root)

    @staticmethod
    async def _detach(session: Any) -> None:
        # Closing the page/context disposes its native session before detach.
        from playwright._impl._errors import TargetClosedError

        try:
            await session.detach()
        except TargetClosedError:
            pass

    async def detach(self) -> None:
        if self._transfer_sessions is not None:
            await self._transfer_sessions.detach()
        if self.page_session is not None:
            await self._detach(self.page_session)
        await self._detach(self._root)

    async def _abort(self, session: Any, request_id: str) -> None:
        await session.send("Fetch.failRequest", {"requestId": request_id, "errorReason": "BlockedByClient"})

    async def handle(self, event: Mapping[str, Any], session: Any) -> None:
        request_id = event["requestId"]
        request = event["request"]
        if self._browser._parent is not None:
            self._browser._parent.check_url(request["url"])
        url = _http_policy_url(request["url"])
        if (
            not isinstance(request_id, str)
            or len(request_id) > 1024
            or url is None
            or (self._browser._parent is None and len(url.encode("utf-8")) > 65_536)
        ):
            raise ArticleFailure("browser_error", "capability")
        if self._closed or self._outcome.failure is not None:
            await self._abort(session, request_id)
            return
        if "responseStatusCode" in event or "responseErrorReason" in event:
            if event.get("responseStatusCode") not in {301, 302, 303, 307, 308}:
                self._requests.pop(request_id, None)
            await session.send("Fetch.continueRequest", {"requestId": request_id})
            return
        if not await self._browser._decision_allowed(url):
            self._outcome.fail("egress")
            await self._abort(session, request_id)
            return
        if self._closed or self._outcome.failure is not None:
            await self._abort(session, request_id)
            return
        if not self._enhanced:
            # Controlled direct pages need the same redirect admission, but keep
            # their browser-owned headers/cookies and legacy serialization.
            await self._transfer_sessions.track()
            self._browser._reserve_parent("network_requests")
            await session.send("Fetch.continueRequest", {"requestId": request_id})
            return
        previous = event.get("redirectedRequestId")
        permitted = self._requests.pop(previous, True)
        if request_id not in self._requests and len(self._requests) >= 64:
            self._outcome.fail("capacity")
            await self._abort(session, request_id)
            raise ArticleFailure("browser_error", "capacity")
        same = http_client._url_origin(url) == self._origin
        permitted = permitted and same
        self._requests[request_id] = permitted
        headers = dict(request["headers"])
        if permitted and self._raw_cookie is not None:
            headers = {key: value for key, value in headers.items() if key.lower() != "cookie"}
            headers["Cookie"] = self._raw_cookie
        if not permitted:
            source = f"{self._origin[0]}://{self._origin[1]}:{self._origin[2]}" if self._origin else ""
            # Cookie was already selected by Chromium's domain/path/secure rules.
            # Do not synthesize, rescope or remove that browser-owned header.
            browser_cookies = {k: v for k, v in headers.items() if k.lower() == "cookie"}
            headers = http_client._strip_sensitive_headers_for_cross_origin(
                {k: v for k, v in headers.items() if k.lower() != "cookie"}, original_url=source, target_url=url
            )
            headers.update(browser_cookies)
            private = {key.lower() for key in self._headers} - {"user-agent", "accept-encoding"}
            headers = {k: v for k, v in headers.items() if k.lower() not in private or k.lower() == "cookie"}
        await self._transfer_sessions.track()
        if self._closed or self._outcome.failure is not None:
            await self._abort(session, request_id)
            return
        self._browser._reserve_parent("network_requests")
        params = {"requestId": request_id, "headers": [{"name": k, "value": str(v)} for k, v in headers.items()]}
        await session.send("Fetch.continueRequest", params)


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
        self._parent: ParentControls | None = None
        self.final_url: str | None = None
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

    def bind_parent(self, parent: ParentControls, context: RuntimeRequestContext) -> None:
        """Bind one private acquisition instance, never a serialized profile."""
        self._parent = parent
        self._context = context

    def _reserve_parent(self, counter: str) -> None:
        if self._parent is not None:
            reservation = self._parent.reserve(counter)
            try:
                reservation.commit()
            finally:
                reservation.close()

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
        *,
        enhanced: bool = False,
    ) -> None:
        on = getattr(session, "on", None)
        send = getattr(session, "send", None)
        if not callable(on) or not callable(send):
            raise ArticleFailure("browser_error", "capability")
        try:
            handlers = (
                ("Network.dataReceived", ledger.on_data_received),
                ("Network.webSocketFrameReceived", ledger.on_websocket_frame),
                ("Network.webSocketFrameSent", ledger.on_websocket_frame),
            )
            if enhanced:
                handlers = (
                    ("Network.dataReceived", lambda event: ledger.on_request_data("main", event)),
                    ("Network.loadingFinished", lambda event: ledger.on_request_finished("main", event)),
                    ("Network.loadingFailed", lambda event: ledger.on_request_failed("main", event)),
                    *handlers[1:],
                )
            for event, handler in handlers:
                registered = on(event, handler)
                if inspect.isawaitable(registered):
                    await registered
            enabled = send("Network.enable")
            if not inspect.isawaitable(enabled):
                raise TypeError("CDP send is not awaitable")
            await enabled
        except asyncio.CancelledError:
            raise
        except ParentStop:
            raise
        except ArticleFailure:
            raise
        except Exception:  # noqa: BLE001 - CDP capability failures are sanitized
            raise ArticleFailure("browser_error", "capability") from None

    @staticmethod
    async def _serialize_html(session: Any, max_bytes: int) -> str:
        html, _ = await GuardedArticleBrowser._serialize_document(session, max_bytes)
        return html

    @staticmethod
    async def _serialize_document(
        session: Any, max_bytes: int, *, capture_dom: bool = False
    ) -> tuple[str, Mapping[str, str] | None]:
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
            except ParentStop:
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
                "expression": f"({(_DOCUMENT_SERIALIZATION_EXPRESSION if capture_dom else _HTML_SERIALIZATION_EXPRESSION)})({max_bytes})",
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
            if not isinstance(html, str) or len(html.encode("utf-8")) > max_bytes:
                raise ArticleFailure("browser_error", "capability")
            if not capture_dom:
                return html, None
            dom = result.get("dom")
            if (
                not isinstance(dom, Mapping)
                or set(dom) != {"title", "author", "date", "content"}
                or any(not isinstance(value, str) for value in dom.values())
                or sum(len(value.encode("utf-8")) for value in dom.values()) > max_bytes
            ):
                raise ArticleFailure("browser_error", "capability")
            return html, MappingProxyType(dict(dom))
        if "html" in result:
            raise ArticleFailure("browser_error", "capability")
        try:
            _BrowserTransferLedger._non_negative_integer(result.get("size"))
        except (TypeError, ValueError, OverflowError):
            raise ArticleFailure("browser_error", "capability") from None
        raise ArticleFailure("response_too_large", "rendered_html")

    async def _decision_allowed(self, url: str) -> bool:
        if self._parent is not None:
            self._parent.check_url(url)
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
        except ParentStop:
            raise
        except Exception as error:  # noqa: BLE001 - sanitize the browser route boundary
            if outcome.native_browser_disconnected or outcome.native_exit_handler is not None:
                from playwright._impl._errors import TargetClosedError

                if isinstance(error, TargetClosedError):
                    if not outcome.native_browser_disconnected:
                        await outcome.native_exit.wait()
                    # The owned browser has exited; finish only Playwright's local route future.
                    await route.fallback()
            outcome.fail("http_route")
            return
        outcome.fail(failure_stage)

    async def _http_handler(
        self,
        route: RuntimeBrowserRoute,
        outcome: _AcquisitionOutcome,
        request_policy: _BrowserRequestPolicy | None = None,
    ) -> None:
        try:
            request = route.request
            url = request.url
            if self._parent is not None:
                self._parent.check_url(url)
            policy_url = _http_policy_url(url)
            if policy_url is None:
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            if request_policy is not None:
                await request_policy.prepare(request)
                await route.continue_()
                return
            try:
                allowed = await self._decision_allowed(policy_url)
            except asyncio.CancelledError:
                raise
            except ParentStop:
                raise
            except Exception:  # noqa: BLE001 - egress guard failures fail closed
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            if not allowed:
                await self._abort_http(route, outcome, failure_stage="egress")
                return
            try:
                self._reserve_parent("network_requests")
                await route.continue_()
            except asyncio.CancelledError:
                raise
            except ParentStop:
                raise
            except Exception:  # noqa: BLE001 - route action failures fail closed
                outcome.fail("http_route")
                try:
                    await self._abort_http(route, outcome, failure_stage="http_route")
                except asyncio.CancelledError:
                    raise
                except ParentStop:
                    raise
                except Exception:  # noqa: BLE001 - original route failure is authoritative
                    outcome.fail("http_route")
        except asyncio.CancelledError:
            raise
        except ParentStop:
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
        except ParentStop:
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
            except ParentStop:
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
                self._reserve_parent("network_requests")
                connection = connect()
                if inspect.isawaitable(connection):
                    await connection
            except ParentStop:
                raise
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
        except ParentStop:
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
        if "requests" in by_kind:
            by_kind["requests"].resource.close()
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
        if "requests" in by_kind:
            return cleanup_failed | await self._retire_browser_owner(by_kind, lifecycle, lease)
        if context is not None and http_routing_started:
            cleanup_failed |= await _bounded_method(
                _OwnedResource(context, "unroute_all", "http-routes"),
                lease=lease,
                grace_s=self._cleanup_grace_s,
                kwargs={"behavior": "wait"},
            )
        for kind in ("page", "context", "cdp") if "requests" in by_kind else ("cdp", "page", "context"):
            owner = by_kind.get(kind)
            if kind == "cdp" and "requests" in by_kind:
                owner = _OwnedResource(by_kind["requests"].resource, "detach", "cdp")
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

    async def _retire_browser_owner(self, owners, lifecycle, lease) -> bool:
        """Retain the guarded owner until actual exit, even when close rejects or stalls."""
        native = owners["browser"].resource
        outcome = lifecycle._outcome
        if outcome.native_exit_handler is None:
            outcome.watch_browser(native)

        async def finish():
            try:
                close_failed = False
                try:
                    await _invoke_method(native, "close")
                except Exception:  # noqa: BLE001 - a rejected close never proves native retirement
                    close_failed = True
                if close_failed or native.is_connected() is not False:
                    await outcome.native_exit.wait()
                outcome.native_browser_disconnected = True
                return close_failed | await self._retire_browser_resources(owners, lifecycle, lease)
            finally:
                outcome.unwatch_browser(native)

        task, retention = lease.start_cleanup(finish(), name="article-browser-owner-retirement")
        if task is None:
            lease.retain_failed_cleanup(owners)
            return True
        if retention == "unobserved":
            return True
        _, pending = await asyncio.wait({task}, timeout=self._cleanup_grace_s)
        # Never cancel native close or detach its guard to meet a logical timeout.
        return bool(pending) or _task_failed(task) or task.result()

    async def _retire_browser_resources(self, owners, lifecycle, lease) -> bool:
        """Retire local handles only after this acquisition's native browser exits."""
        failed = False

        async def retire(owner):
            from playwright._impl._errors import TargetClosedError

            try:
                await _invoke_method(owner.resource, owner.method_name)
            except TargetClosedError:
                if not lifecycle._outcome.native_browser_disconnected:
                    raise

        for kind in ("requests", "page", "context"):
            owner = owners.get(kind)
            if owner is None:
                continue
            if kind == "requests":
                owner = _OwnedResource(owner.resource, "retire", kind)
            task, retention = lease.start_cleanup(retire(owner), name=f"article-browser-retire-{kind}")
            if task is None:
                failed = True
            elif retention == "unobserved":
                task.cancel()
                failed = True
            else:
                failed |= await _bounded_retained_task(task, grace_s=self._cleanup_grace_s)
        failed |= await lifecycle.drain(self._cleanup_grace_s)
        if "playwright" in owners:
            failed |= await _bounded_method(owners["playwright"], lease=lease, grace_s=self._cleanup_grace_s)
        return failed

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
        self._rendered_fallback = None
        if limits is None:
            limits = ArticleLimits()
        elif not isinstance(limits, ArticleLimits):
            raise TypeError("limits must be an ArticleLimits instance")

        if self._parent is not None:
            self._parent.check_url(url)
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
        try:
            proxy = selected_browser_proxy(url, profile.proxies)
        except Exception:  # noqa: BLE001 - proxy policy/configuration errors are sanitized
            raise ArticleFailure("browser_transport_unavailable", "capability", capability={}) from None
        decision = bind_selected_browser_proxy(decision, proxy)
        if not decision.allowed:
            raise ArticleFailure(
                "browser_transport_unavailable", decision.reason, capability=decision.to_capability_metadata()
            )
        for attempt in range(attempts):
            if self._parent is not None:
                self._parent.check()
            lease = self._acquisition_pool.acquire(
                callback_capacity=self._callback_capacity,
            )
            if lease is None:
                raise ArticleFailure("browser_error", "capacity")
            try:
                self._reserve_parent("browser_attempts")
                return await self._acquire_with_lease(url, profile, limits, lease, proxy=proxy)
            except ArticleFailure as failure:
                if attempt + 1 >= attempts or not self._should_retry(failure):
                    raise
            finally:
                lease.seal()
                lease.owner_finished()
            await self._sleep(
                self._parent.cap_timeout(_BROWSER_RETRY_DELAY_S) if self._parent else _BROWSER_RETRY_DELAY_S
            )

        raise ArticleFailure("browser_error", "launch")

    async def _acquire_with_lease(
        self,
        url: str,
        profile: DirectBrowserProfile,
        limits: ArticleLimits,
        lease: _BrowserAcquisitionLease,
        *,
        proxy: Mapping[str, str] | None = None,
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
        requests: _BrowserRequestPolicy | None = None

        async def _shutdown_context() -> None:
            if context is None:
                raise TypeError("browser context is unavailable")
            await _invoke_method(context, "close")

        lifecycle = _CallbackLifecycle(outcome, lease, _shutdown_context)
        transfer = _BrowserTransferLedger(
            limit=limits.max_browser_transfer_bytes,
            outcome=outcome,
            lease=lease,
            shutdown_factory=_shutdown_context,
            parent=self._parent,
        )
        http_routing_started = False
        stage = "launch"
        try:
            playwright = await self._launcher.start()
            resources.append(_OwnedResource(playwright, "stop", "playwright"))
            browser = await playwright.chromium.launch(
                headless=True, **({"channel": "chromium"} if profile.extra_headers is not None else {})
            )
            resources.append(_OwnedResource(browser, "close", "browser"))
            if profile.extra_headers is not None or self._parent is not None:
                guard_session = await browser.new_browser_cdp_session()
                requests = _BrowserRequestPolicy(url, profile.extra_headers, self, outcome)
                resources.append(_OwnedResource(requests, "close", "requests"))
                outcome.watch_browser(browser)
                await requests.bind(guard_session, lifecycle, transfer, lease)
            stage = "context"
            context = await browser.new_context(
                **({"proxy": dict(proxy)} if proxy is not None else {}),
                **(
                    {
                        "extra_http_headers": {
                            key: value for key, value in profile.extra_headers.items() if key.lower() != "cookie"
                        }
                    }
                    if profile.extra_headers is not None
                    else {}
                ),
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

            async def install_http_routes() -> None:
                nonlocal http_routing_started
                await route(
                    _ROUTE_PATTERN,
                    lifecycle.handler(
                        lambda intercepted: self._http_handler(intercepted, outcome, requests),
                        lambda intercepted: self._abort_http(
                            intercepted,
                            outcome,
                            failure_stage="callback",
                        ),
                    ),
                )
                http_routing_started = True

            await install_http_routes()
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
            except ParentStop:
                raise
            except Exception:  # noqa: BLE001 - CDP capability failures are sanitized
                raise ArticleFailure("browser_error", "capability") from None
            if not callable(getattr(cdp_session, "detach", None)):
                raise ArticleFailure("browser_error", "capability")
            resources.append(_OwnedResource(cdp_session, "detach", "cdp"))
            if requests is None:
                await self._install_transfer_accounting(cdp_session, transfer)
            else:
                requests.page_session = cdp_session
                info = (await cdp_session.send("Target.getTargetInfo"))["targetInfo"]
                requests._transfer_sessions.context_id = info["browserContextId"]
                requests._transfer_sessions.main_target = info["targetId"]
                await self._install_transfer_accounting(cdp_session, transfer, enhanced=True)

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
            if requests is not None:
                await requests._transfer_sessions.acknowledged(
                    self._parent.cap_timeout(max(self._cleanup_grace_s, profile.timeout_ms / 1000))
                    if self._parent
                    else max(self._cleanup_grace_s, profile.timeout_ms / 1000)
                )
            if profile.extra_headers is None:
                html = await self._serialize_html(cdp_session, limits.max_article_bytes)
            else:
                html, self._rendered_fallback = await self._serialize_document(
                    cdp_session, limits.max_article_bytes, capture_dom=True
                )
            final_url = str(getattr(page, "url", url) or url)
            if self._parent is not None:
                self._parent.check_url(final_url)
            self.final_url = final_url
        except asyncio.CancelledError as exc:
            primary_error = exc
        except ArticleFailure as exc:
            primary_error = exc
        except ParentStop as exc:
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

        if outcome.parent_stop is not None:
            raise outcome.parent_stop
        if isinstance(primary_error, ParentStop):
            raise primary_error
        if isinstance(primary_error, asyncio.CancelledError):
            raise primary_error
        if teardown_cancelled or outcome.callback_cancelled:
            raise asyncio.CancelledError()
        if self._parent is not None:
            self._parent.check()
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
