from __future__ import annotations

import asyncio
import gc
import importlib
import re
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    BrowserTransportAttestation,
    decide_browser_transport,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleFailure,
    ArticleLimits,
    DirectBrowserProfile,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressDecision
from tldw_Server_API.app.core.Web_Scraping.runtime.requests import RuntimeRequestContext

_TARGET = "https://article.example/start"
_DEFAULT_EVALUATION = object()


def _browser_module() -> Any:
    try:
        return importlib.import_module("tldw_Server_API.app.core.Web_Scraping.orchestration.article_browser")
    except ModuleNotFoundError:
        pytest.fail("GuardedArticleBrowser implementation is missing", pytrace=False)


def _browser_type() -> type[Any]:
    return _browser_module().GuardedArticleBrowser


def _pool_type() -> type[Any]:
    pool_type = getattr(_browser_module(), "_BrowserAcquisitionPool", None)
    if pool_type is None:
        pytest.fail("Browser acquisition pool implementation is missing", pytrace=False)
    return pool_type


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


def test_uncancel_compatibility_accepts_python310_style_task() -> None:
    class _LegacyTask:
        pass

    _browser_module()._uncancel_task(_LegacyTask())


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
        if isinstance(
            outcome,
            (_GatedOutcome, _DispatchingOutcome, _CancellationResistantOutcome),
        ):
            outcome = await outcome.resolve()
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, bool):
            return ProbeEgressDecision(allowed=outcome, reason="test")
        assert isinstance(outcome, ProbeEgressDecision)
        return outcome


class _GatedOutcome:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancellation_args: tuple[object, ...] | None = None

    async def resolve(self) -> object:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError as exc:
            self.cancellation_args = exc.args
            raise
        return self.outcome


class _DispatchingOutcome:
    def __init__(
        self,
        runtime: _FakeBrowserRuntime,
        route: _FakeWebSocketRoute,
        outcome: object,
    ) -> None:
        self.runtime = runtime
        self.route = route
        self.outcome = outcome

    async def resolve(self) -> object:
        context = self.runtime.context
        assert context is not None
        handler = context.websocket_handler
        assert handler is not None
        await self.runtime.context_closed.wait()
        await asyncio.sleep(0)
        context._schedule(handler, self.route, kind="websocket")
        return self.outcome


class _CancellationResistantOutcome:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellation_args: tuple[object, ...] | None = None

    async def resolve(self) -> object:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError as exc:
            self.cancellation_args = exc.args
            await self.release.wait()
        self.finished.set()
        raise RuntimeError("https://secret.example/path?token=raw")


class _CancelRecordingTask(asyncio.Task[Any]):
    def __init__(
        self,
        coroutine: Coroutine[Any, Any, Any],
        *,
        loop: asyncio.AbstractEventLoop,
        context: Any = None,
        name: str | None = None,
    ) -> None:
        kwargs = {"loop": loop}
        if context is not None:
            kwargs["context"] = context
        if name is not None:
            kwargs["name"] = name
        super().__init__(coroutine, **kwargs)
        self.cancel_messages: list[object | None] = []

    def cancel(self, msg: object | None = None) -> bool:
        self.cancel_messages.append(msg)
        return super().cancel(msg)


class _DoneCallbackRejectingTask(asyncio.Task[Any]):
    def add_done_callback(
        self,
        callback: Callable[[asyncio.Future[Any]], object],
        *,
        context: Any = None,
    ) -> None:
        raise RuntimeError("callback registration failed")


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
        self.handled = asyncio.Event()

    async def continue_(self, **kwargs: object) -> None:
        self.continue_calls.append(kwargs)
        self.handled.set()
        if self.continue_error is not None:
            raise self.continue_error

    async def abort(self) -> None:
        self.abort_calls += 1
        self.handled.set()
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

    def connect_to_server(self) -> object:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error
        return SimpleNamespace(kind="server-route")

    async def close(
        self,
        *,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.close_calls.append((code, reason))
        if self.close_error is not None:
            raise self.close_error


class _AwaitableWebSocketRoute(_FakeWebSocketRoute):
    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.connected = asyncio.Event()

    async def connect_to_server(self) -> object:
        self.connect_calls += 1
        await asyncio.sleep(0)
        self.connected.set()
        return SimpleNamespace(kind="server-route")


class _ResistantRejectedHttpRoute(_FakeHttpRoute):
    def __init__(self, url: str, target_closed: asyncio.Event) -> None:
        super().__init__(url)
        self.target_closed = target_closed
        self.abort_started = asyncio.Event()

    async def abort(self) -> None:
        self.abort_calls += 1
        self.handled.set()
        self.abort_started.set()
        await self.target_closed.wait()


class _ResistantRejectedWebSocketRoute(_FakeWebSocketRoute):
    def __init__(self, url: str, target_closed: asyncio.Event) -> None:
        super().__init__(url)
        self.target_closed = target_closed
        self.close_started = asyncio.Event()

    async def close(
        self,
        *,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.close_calls.append((code, reason))
        self.close_started.set()
        await self.target_closed.wait()


class _ExplodingTailText(str):
    def encode(self, *_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("text accounting must not allocate an encoded copy")

    def __iter__(self) -> Any:
        yield "é"
        raise AssertionError("text accounting must stop after crossing the limit")


class _UnscannableBinary(str):
    def __iter__(self) -> Any:
        raise AssertionError("obvious binary outcomes must not scan payload data")


class _FakeBrowserRuntime:
    def __init__(
        self,
        *,
        dispatches: list[tuple[str, object]] | None = None,
        stage_errors: dict[str, BaseException | list[BaseException | None]] | None = None,
        html: str = "<!doctype html><html><body>ok</body></html>",
        missing_context_capability: str | None = None,
        cleanup_modes: dict[str, str] | None = None,
        callback_start_turns_after_close: list[int | None] | None = None,
        release_callbacks_after_close_timer: bool = False,
        cdp_events: list[tuple[str, object]] | None = None,
        cdp_missing: str | None = None,
        cdp_responses: dict[str, object] | None = None,
        cdp_command_gates: dict[str, asyncio.Event] | None = None,
        evaluation_result: object = _DEFAULT_EVALUATION,
    ) -> None:
        self.dispatches = list(dispatches or [])
        self.stage_errors = dict(stage_errors or {})
        self.html = html
        self.missing_context_capability = missing_context_capability
        self.cleanup_modes = dict(cleanup_modes or {})
        self.callback_start_turns_after_close = list(callback_start_turns_after_close or [])
        self.release_callbacks_after_close_timer = release_callbacks_after_close_timer
        self.cdp_events = list(cdp_events or [])
        self.cdp_missing = cdp_missing
        self.cdp_responses = dict(cdp_responses or {})
        self.cdp_command_gates = dict(cdp_command_gates or {})
        self.evaluation_result = evaluation_result
        self.events: list[str] = []
        self.launch_options: dict[str, object] | None = None
        self.context_options: dict[str, object] | None = None
        self.http_routes: list[_FakeHttpRoute] = []
        self.websocket_routes: list[_FakeWebSocketRoute] = []
        self.context: _FakeContext | None = None
        self.installed_http_handler: Callable[[_FakeHttpRoute], Awaitable[None]] | None = None
        self.installed_websocket_handler: Callable[[_FakeWebSocketRoute], Awaitable[None]] | None = None
        self.callback_tasks: set[asyncio.Task[None]] = set()
        self.all_callback_tasks: list[asyncio.Task[None]] = []
        self.cleanup_tasks: dict[str, asyncio.Task[Any]] = {}
        self.stuck_cleanup_cancelled: list[str] = []
        self.cleanup_started: dict[str, asyncio.Event] = {}
        self.content_returned = asyncio.Event()
        self.context_closed = asyncio.Event()
        self.target_closed = asyncio.Event()
        self.context_close_task_names: list[str] = []
        self.cookies: list[dict[str, object]] = []
        self.evaluate_calls: list[tuple[str, object]] = []
        self.cdp_command_calls: list[tuple[str, dict[str, object] | None]] = []
        self.cdp_command_started: dict[str, asyncio.Event] = {}
        self.cdp_sessions: list[_FakeCDPSession] = []
        self.sleep_delays: list[float] = []
        self.stealth_pages: list[_FakePage] = []
        self.resistant_cleanup_release = asyncio.Event()
        self.resistant_cleanup_finished = asyncio.Event()
        self._callback_schedule_count = 0
        self.launcher = _FakeLauncher(self)

    def raise_at(self, stage: str) -> None:
        error = self.stage_errors.get(stage)
        if isinstance(error, list):
            error = error.pop(0) if error else None
        if error is not None:
            raise error

    async def cleanup(self, stage: str) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.cleanup_tasks[stage] = task
        self.events.append(stage)
        self.cleanup_started.setdefault(stage, asyncio.Event()).set()
        if self.cleanup_modes.get(stage) == "stuck":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.stuck_cleanup_cancelled.append(stage)
                raise
        if self.cleanup_modes.get(stage) == "resistant":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.stuck_cleanup_cancelled.append(stage)
            await self.resistant_cleanup_release.wait()
            self.resistant_cleanup_finished.set()
            raise RuntimeError("https://secret.example/path?token=raw")
        self.raise_at(stage)


class _CleanupControlled:
    cleanup_method = "close"
    cleanup_stage = ""

    def __getattribute__(self, name: str) -> Any:
        cleanup_method = object.__getattribute__(self, "cleanup_method")
        if name == cleanup_method:
            runtime = object.__getattribute__(self, "runtime")
            mode = runtime.cleanup_modes.get(object.__getattribute__(self, "cleanup_stage"))
            if mode == "raising_accessor":
                raise RuntimeError("https://secret.example/path?token=raw")
            if mode == "missing":
                return None
        return object.__getattribute__(self, name)


class _FakeCDPSession:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime
        self.handlers: dict[str, Callable[[object], None]] = {}
        self.detached = False
        if runtime.cdp_missing == "on":
            self.on = None  # type: ignore[assignment]
        if runtime.cdp_missing == "send":
            self.send = None  # type: ignore[assignment]

    def on(
        self,
        event: str,
        handler: Callable[[object], None],
    ) -> None:
        self.runtime.events.append(f"cdp-on:{event}")
        self.runtime.raise_at(f"cdp-on:{event}")
        self.handlers[event] = handler

    async def send(
        self,
        method: str,
        params: dict[str, object] | None = None,
    ) -> object:
        self.runtime.events.append(f"cdp-send:{method}")
        self.runtime.cdp_command_calls.append((method, params))
        self.runtime.cdp_command_started.setdefault(method, asyncio.Event()).set()
        self.runtime.raise_at(f"cdp-send:{method}")
        gate = self.runtime.cdp_command_gates.get(method)
        if gate is not None:
            await gate.wait()
            self.runtime.raise_at(f"cdp-send-after:{method}")
        if method in self.runtime.cdp_responses:
            return self.runtime.cdp_responses[method]
        if method == "Network.enable":
            return {}
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "main-frame"}}}
        if method == "Page.createIsolatedWorld":
            return {"executionContextId": 73}
        if method == "Runtime.evaluate":
            assert params is not None
            expression = params.get("expression")
            assert isinstance(expression, str)
            match = re.search(r"\)\((\d+)\)\s*$", expression)
            assert match is not None
            limit = int(match.group(1))
            if self.runtime.evaluation_result is _DEFAULT_EVALUATION:
                size = len(self.runtime.html.encode("utf-8"))
                value: object = (
                    {"ok": True, "html": self.runtime.html} if size <= limit else {"ok": False, "size": size}
                )
            else:
                value = self.runtime.evaluation_result
            self.runtime.content_returned.set()
            return {"result": {"type": "object", "value": value}}
        raise AssertionError(f"unexpected CDP command: {method}")

    def emit(self, event: str, payload: object) -> None:
        handler = self.handlers.get(event)
        assert handler is not None, f"CDP handler not installed for {event}"
        handler(payload)

    async def detach(self) -> None:
        self.detached = True
        self.runtime.raise_at("detach:cdp")


class _FakePage(_CleanupControlled):
    cleanup_stage = "close:page"

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
                context.dispatch_http(route)
            else:
                route = value if isinstance(value, _FakeWebSocketRoute) else _FakeWebSocketRoute(str(value))
                self.runtime.websocket_routes.append(route)
                context.dispatch_websocket(route)
        for event, payload in self.runtime.cdp_events:
            assert self.runtime.cdp_sessions
            self.runtime.cdp_sessions[-1].emit(event, payload)
        await asyncio.sleep(0)

    async def content(self) -> str:
        self.runtime.events.append("content")
        self.runtime.raise_at("content")
        self.runtime.content_returned.set()
        return self.runtime.html

    async def evaluate(self, expression: str, argument: object) -> object:
        self.runtime.events.append("evaluate")
        self.runtime.evaluate_calls.append((expression, argument))
        self.runtime.raise_at("evaluate")
        self.runtime.raise_at("content")
        self.runtime.content_returned.set()
        if self.runtime.evaluation_result is not _DEFAULT_EVALUATION:
            return self.runtime.evaluation_result
        size = len(self.runtime.html.encode("utf-8"))
        return {"ok": True, "html": self.runtime.html} if size <= argument else {"ok": False, "size": size}

    async def wait_for_timeout(self, timeout_ms: int) -> None:
        self.runtime.events.append(f"wait_for_timeout:{timeout_ms}")
        self.runtime.raise_at("wait_for_timeout")

    async def wait_for_load_state(self, state: str, **kwargs: object) -> None:
        self.runtime.events.append(f"wait_for_load_state:{state}")
        self.runtime.events.append(f"wait-options:{sorted(kwargs)}")
        self.runtime.raise_at("wait_for_load_state")

    async def close(self) -> None:
        await self.runtime.cleanup("close:page")


class _FakeContext(_CleanupControlled):
    cleanup_stage = "close:context"

    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime
        self.http_handler: Callable[[_FakeHttpRoute], Awaitable[None]] | None = None
        self.websocket_handler: Callable[[_FakeWebSocketRoute], Awaitable[None]] | None = None
        if runtime.missing_context_capability == "route":
            self.route = None  # type: ignore[assignment]
        if runtime.missing_context_capability == "route_web_socket":
            self.route_web_socket = None  # type: ignore[assignment]
        if runtime.missing_context_capability == "unroute_all":
            self.unroute_all = None  # type: ignore[assignment]
        if runtime.missing_context_capability == "new_cdp_session":
            self.new_cdp_session = None  # type: ignore[assignment]

    def _schedule(
        self,
        handler: Callable[[Any], Awaitable[None]],
        route: object,
        *,
        kind: str,
    ) -> asyncio.Task[None]:
        index = self.runtime._callback_schedule_count
        self.runtime._callback_schedule_count += 1
        start_turns = (
            self.runtime.callback_start_turns_after_close[index]
            if index < len(self.runtime.callback_start_turns_after_close)
            else None
        )

        async def invoke() -> None:
            if start_turns is not None:
                await self.runtime.context_closed.wait()
                for _ in range(start_turns):
                    await asyncio.sleep(0)
            await handler(route)
            if kind == "http":
                assert isinstance(route, _FakeHttpRoute)
                await route.handled.wait()

        task = asyncio.create_task(invoke(), name=f"fake-playwright-{kind}-callback")
        self.runtime.callback_tasks.add(task)
        self.runtime.all_callback_tasks.append(task)
        task.add_done_callback(self.runtime.callback_tasks.discard)
        return task

    def dispatch_http(self, route: _FakeHttpRoute) -> asyncio.Task[None]:
        assert self.http_handler is not None
        return self._schedule(self.http_handler, route, kind="http")

    def dispatch_websocket(self, route: _FakeWebSocketRoute) -> asyncio.Task[None]:
        assert self.websocket_handler is not None
        return self._schedule(self.websocket_handler, route, kind="websocket")

    async def route(
        self,
        pattern: str,
        handler: Callable[[_FakeHttpRoute], Awaitable[None]],
    ) -> None:
        self.runtime.events.append(f"route:{pattern}")
        self.runtime.raise_at("route")
        self.http_handler = handler
        self.runtime.installed_http_handler = handler

    async def route_web_socket(
        self,
        pattern: str,
        handler: Callable[[_FakeWebSocketRoute], Awaitable[None]],
    ) -> None:
        self.runtime.events.append(f"route_web_socket:{pattern}")
        self.runtime.raise_at("route_web_socket")
        self.websocket_handler = handler
        self.runtime.installed_websocket_handler = handler

    async def unroute_all(self, *, behavior: str | None = None) -> None:
        self.runtime.events.append(f"unroute_all:{behavior}")
        self.runtime.raise_at("unroute_all")
        tasks = tuple(
            task for task in self.runtime.callback_tasks if task.get_name() == "fake-playwright-http-callback"
        )
        if behavior == "wait" and tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.http_handler = None

    async def new_page(self) -> _FakePage:
        self.runtime.events.append("new_page")
        self.runtime.raise_at("new_page")
        return _FakePage(self.runtime)

    async def new_cdp_session(self, page: _FakePage) -> _FakeCDPSession:
        self.runtime.events.append("new_cdp_session")
        self.runtime.raise_at("new_cdp_session")
        session = _FakeCDPSession(self.runtime)
        self.runtime.cdp_sessions.append(session)
        return session

    async def add_cookies(self, cookies: list[dict[str, object]]) -> None:
        self.runtime.events.append("add_cookies")
        self.runtime.raise_at("add_cookies")
        self.runtime.cookies.extend(cookies)

    async def close(self) -> None:
        self.websocket_handler = None
        self.runtime.target_closed.set()
        task = asyncio.current_task()
        assert task is not None
        self.runtime.context_close_task_names.append(task.get_name())
        try:
            await self.runtime.cleanup("close:context")
        finally:
            if self.runtime.release_callbacks_after_close_timer:
                asyncio.get_running_loop().call_later(0, self.runtime.context_closed.set)
            else:
                self.runtime.context_closed.set()


class _FakeBrowser(_CleanupControlled):
    cleanup_stage = "close:browser"

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
        await self.runtime.cleanup("close:browser")


class _FakeChromium:
    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime

    async def launch(self, **kwargs: object) -> _FakeBrowser:
        self.runtime.events.append("launch")
        self.runtime.launch_options = dict(kwargs)
        self.runtime.raise_at("launch")
        return _FakeBrowser(self.runtime)


class _FakePlaywright(_CleanupControlled):
    cleanup_method = "stop"
    cleanup_stage = "stop:playwright"

    def __init__(self, runtime: _FakeBrowserRuntime) -> None:
        self.runtime = runtime
        self.chromium = _FakeChromium(runtime)

    async def stop(self) -> None:
        await self.runtime.cleanup("stop:playwright")


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
    transport_decision: Callable[[], object] | None = None,
    capability_check: Callable[[], bool] | None = None,
    cleanup_grace_s: float | None = None,
    acquisition_pool: Any | None = None,
    callback_capacity: int | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    stealth_hook: Callable[[Any], Awaitable[None]] | None = None,
) -> Any:
    context = RuntimeRequestContext(
        source="article_extract",
        stage="browser",
        user_id="7",
        request_id="request-15",
    )
    kwargs: dict[str, object] = {}
    if cleanup_grace_s is not None:
        kwargs["cleanup_grace_s"] = cleanup_grace_s
    if acquisition_pool is not None:
        kwargs["acquisition_pool"] = acquisition_pool
    if callback_capacity is not None:
        kwargs["callback_capacity"] = callback_capacity

    async def immediate_sleep(_delay: float) -> None:
        await asyncio.sleep(0)

    kwargs["sleep"] = sleep or immediate_sleep
    if stealth_hook is not None:
        kwargs["stealth_hook"] = stealth_hook
    return _browser_type()(
        egress_guard=guard,
        context=context,
        launcher=runtime.launcher,
        transport_decision=transport_decision
        or (
            lambda: decide_browser_transport(
                configured_mode="auto",
                auth_mode="single_user",
                outbound_policy_mode="compat",
            )
        ),
        capability_check=capability_check or (lambda: True),
        **kwargs,
    )


def _assert_failure(exc: ArticleFailure, *, stage: str) -> None:
    assert exc.code == "browser_error"
    assert exc.stage == stage
    assert str(exc) == "browser_error"
    assert "secret" not in str(exc)
    assert "token" not in str(exc)


def _assert_limit_failure(exc: ArticleFailure, *, stage: str) -> None:
    assert exc.code == "response_too_large"
    assert exc.stage == stage
    assert str(exc) == "response_too_large"


@pytest.mark.unit
async def test_attested_acquire_preserves_fresh_target_redirect_and_subresource_guards() -> None:
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

    html = await _adapter(
        runtime,
        guard,
        transport_decision=lambda: attested,
    ).acquire(_TARGET, _profile())

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
    assert runtime.events[-6:] == [
        "cdp-send:Runtime.evaluate",
        "unroute_all:wait",
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

    await _adapter(
        runtime,
        guard,
        transport_decision=lambda: attested,
    ).acquire(_TARGET, _profile())

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
async def test_transport_denial_precedes_pool_guard_and_launcher() -> None:
    runtime = _FakeBrowserRuntime()
    guard = _FakeGuard([])
    pool = _pool_type()(1)
    denied = decide_browser_transport(
        configured_mode="auto",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            guard,
            acquisition_pool=pool,
            transport_decision=lambda: denied,
        ).acquire(_TARGET, _profile())

    assert raised.value.code == "browser_transport_unavailable"
    assert raised.value.stage == "browser_transport_unattested"
    assert dict(raised.value.capability) == denied.to_capability_metadata()
    assert runtime.events == []
    assert guard.calls == []
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "transport_decision",
    [
        lambda: (_ for _ in ()).throw(RuntimeError("secret config error")),
        lambda: object(),
    ],
    ids=["provider-error", "wrong-type"],
)
async def test_invalid_transport_provider_fails_closed_before_launch(
    transport_decision: Callable[[], object],
) -> None:
    runtime = _FakeBrowserRuntime()
    guard = _FakeGuard([])
    pool = _pool_type()(1)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            guard,
            acquisition_pool=pool,
            transport_decision=transport_decision,
        ).acquire(_TARGET, _profile())

    assert raised.value.code == "browser_transport_unavailable"
    assert raised.value.stage == "browser_transport_config_invalid"
    assert dict(raised.value.capability) == {
        "name": "safe_browser_transport",
        "available": False,
        "configured_mode": "disabled",
        "effective_mode": "disabled",
        "dns_peer_attested": False,
        "reason": "browser_transport_config_invalid",
    }
    assert runtime.events == []
    assert guard.calls == []
    assert pool.active_count == 0


@pytest.mark.unit
def test_transport_provider_failure_logs_only_safe_diagnostics() -> None:
    """Provider failures should be observable without leaking exception details."""
    secret = "transport-secret"
    records: list[Any] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        adapter = _adapter(
            _FakeBrowserRuntime(),
            _FakeGuard([]),
            transport_decision=lambda: (_ for _ in ()).throw(RuntimeError(secret)),
        )

        capability = adapter.transport_capability()
    finally:
        logger.remove(sink_id)

    assert capability["reason"] == "browser_transport_config_invalid"
    matching = [record for record in records if record["extra"].get("operation") == "resolve_transport"]
    assert len(matching) == 1
    assert matching[0]["extra"]["component"] == "article_browser"
    assert matching[0]["extra"]["operation"] == "resolve_transport"
    assert matching[0]["extra"]["exception_type"] == "RuntimeError"
    assert secret not in matching[0]["message"]


@pytest.mark.unit
async def test_http_fragment_navigation_guards_fragmentless_transport_url() -> None:
    target = f"{_TARGET}#section"
    route = _FakeHttpRoute(target)
    runtime = _FakeBrowserRuntime(dispatches=[("http", route)])
    guard = _FakeGuard([True])

    html = await _adapter(runtime, guard).acquire(target, _profile())

    assert html == "<!doctype html><html><body>ok</body></html>"
    assert [url for url, _context in guard.calls] == [_TARGET]
    assert f"goto:{target}" in runtime.events
    assert route.continue_calls == [{}]


@pytest.mark.unit
async def test_websocket_awaitable_connection_is_completed_before_acquisition_returns() -> None:
    route = _AwaitableWebSocketRoute("wss://socket.example/live")
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", route)])

    await _adapter(runtime, _FakeGuard([True])).acquire(_TARGET, _profile())

    assert route.connect_calls == 1
    assert route.connected.is_set()
    assert route.close_calls == []


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
async def test_websocket_connect_immediate_exception_closes_and_surfaces_route_stage(
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
@pytest.mark.parametrize("capacity", [True, False, 0, -1, 1.0, "1", 257])
def test_browser_acquisition_pool_rejects_non_strict_or_unbounded_capacity(
    capacity: object,
) -> None:
    with pytest.raises(ValueError, match="capacity"):
        _pool_type()(capacity)


@pytest.mark.unit
@pytest.mark.parametrize("callback_capacity", [True, False, 0, -1, 1.0, "1", 1025])
async def test_browser_adapter_rejects_non_strict_or_unbounded_callback_capacity(
    callback_capacity: object,
) -> None:
    runtime = _FakeBrowserRuntime()

    with pytest.raises(ValueError, match="callback capacity"):
        _browser_type()(
            egress_guard=_FakeGuard([]),
            context=RuntimeRequestContext(source="article_extract", stage="browser"),
            launcher=runtime.launcher,
            capability_check=lambda: True,
            acquisition_pool=_pool_type()(1),
            callback_capacity=callback_capacity,
        )

    assert runtime.events == []


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
@pytest.mark.parametrize(
    ("url", "capability", "stage"),
    [
        (_TARGET, False, "capability"),
        ("file:///private/etc/passwd", True, "egress"),
    ],
)
async def test_prelaunch_rejection_releases_reserved_acquisition_capacity(
    url: str,
    capability: bool,
    stage: str,
) -> None:
    pool = _pool_type()(1)
    runtime = _FakeBrowserRuntime()

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            _FakeGuard([]),
            capability_check=lambda: capability,
            acquisition_pool=pool,
        ).acquire(url, _profile())

    _assert_failure(raised.value, stage=stage)
    assert runtime.events == []
    assert pool.active_count == 0


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["route", "route_web_socket", "unroute_all"])
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
            "cdp-send:Runtime.evaluate",
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
        ("websocket", False, ("close", asyncio.CancelledError())),
    ],
    ids=[
        "http-guard",
        "http-continue",
        "http-abort",
        "websocket-guard",
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
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=1),
        )

    _assert_failure(raised.value, stage="navigation")
    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]


@pytest.mark.unit
async def test_cleanup_failure_attempts_all_resources_and_is_sanitized() -> None:
    runtime = _FakeBrowserRuntime(stage_errors={"close:page": RuntimeError("https://secret.example/path?token=raw")})

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="cleanup")
    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]


async def _assert_no_browser_tasks() -> None:
    await asyncio.sleep(0)
    current = asyncio.current_task()
    assert [
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not current
        and not task.done()
        and (task.get_name().startswith("article-browser-") or task.get_name().startswith("fake-playwright-"))
    ] == []


async def _assert_pool_active_count(pool: Any, expected: int) -> None:
    for _ in range(100):
        if pool.active_count == expected:
            return
        await asyncio.sleep(0)
    assert pool.active_count == expected


def _retained_task_count(pool: Any) -> int:
    with pool._lock:
        leases = tuple(pool._leases)
    total = 0
    for lease in leases:
        with lease._lock:
            total += len(lease._tasks)
    return total


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["http", "websocket"])
async def test_late_denial_after_content_cannot_return_html(kind: str) -> None:
    gated = _GatedOutcome(False)
    route: object
    if kind == "http":
        route = _FakeHttpRoute("https://late.example/resource")
    else:
        route = _FakeWebSocketRoute("wss://late.example/socket")
    runtime = _FakeBrowserRuntime(dispatches=[(kind, route)])
    acquisition = asyncio.create_task(
        _adapter(runtime, _FakeGuard([gated])).acquire(_TARGET, _profile()),
        name="test-late-denial-acquisition",
    )

    await gated.started.wait()
    await runtime.content_returned.wait()
    await asyncio.sleep(0)
    assert acquisition.done() is False
    gated.release.set()

    with pytest.raises(ArticleFailure) as raised:
        await acquisition

    _assert_failure(raised.value, stage="egress")
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_callback_cancelled_error_propagates_after_deterministic_cleanup() -> None:
    gated = _GatedOutcome(asyncio.CancelledError())
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", _FakeWebSocketRoute("wss://late.example/socket"))])
    acquisition = asyncio.create_task(
        _adapter(runtime, _FakeGuard([gated])).acquire(_TARGET, _profile()),
        name="test-callback-cancellation-acquisition",
    )

    await gated.started.wait()
    await runtime.content_returned.wait()
    gated.release.set()

    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_callback_drain_is_bounded_without_internally_cancelling_late_work() -> None:
    resistant = _CancellationResistantOutcome()
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", _FakeWebSocketRoute("wss://late.example/socket"))])
    pool = _pool_type()(1)
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            _FakeGuard([resistant]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
        ).acquire(_TARGET, _profile())

    assert loop.time() - started < 0.5
    _assert_failure(raised.value, stage="callback_drain")
    assert resistant.cancellation_args is None
    assert pool.active_count == 1

    blocked_runtime = _FakeBrowserRuntime()
    with pytest.raises(ArticleFailure) as blocked:
        await _adapter(
            blocked_runtime,
            _FakeGuard([]),
            acquisition_pool=pool,
        ).acquire(_TARGET, _profile())

    _assert_failure(blocked.value, stage="capacity")
    assert blocked_runtime.events == []

    resistant.release.set()
    await resistant.finished.wait()
    await _assert_pool_active_count(pool, 0)
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_callback_starting_after_teardown_admission_closes_is_rejected() -> None:
    route = _FakeWebSocketRoute("wss://late.example/queued")
    runtime = _FakeBrowserRuntime(
        dispatches=[("websocket", route)],
        callback_start_turns_after_close=[0],
        release_callbacks_after_close_timer=True,
    )
    guard = _FakeGuard([])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, guard).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="callback")
    assert guard.calls == []
    assert route.connect_calls == 0
    assert route.close_calls == [(1008, "Policy denied")]
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_admitted_callback_cannot_admit_new_callback_during_teardown() -> None:
    first = _FakeWebSocketRoute("wss://late.example/first")
    second = _FakeWebSocketRoute("wss://late.example/second")
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", first)])
    guard = _FakeGuard([_DispatchingOutcome(runtime, second, True)])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, guard).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="callback")
    assert second.connect_calls == 0
    assert second.close_calls == [(1008, "Policy denied")]
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_external_callback_task_cancellation_remains_acquisition_cancellation() -> None:
    gated = _GatedOutcome(True)
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", _FakeWebSocketRoute("wss://late.example/external-cancel"))])
    acquisition = asyncio.create_task(
        _adapter(runtime, _FakeGuard([gated])).acquire(_TARGET, _profile()),
        name="test-external-callback-cancellation-acquisition",
    )

    await gated.started.wait()
    await runtime.content_returned.wait()
    callback = next(task for task in runtime.callback_tasks if not task.done())
    callback.cancel("external-callback")

    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert gated.cancellation_args == ("external-callback",)
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_external_callback_cancellation_near_drain_deadline_has_no_internal_cancel() -> None:
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(
        lambda loop, coroutine, **kwargs: _CancelRecordingTask(
            coroutine,
            loop=loop,
            **kwargs,
        )
    )
    gated = _GatedOutcome(True)
    runtime = _FakeBrowserRuntime(
        dispatches=[
            (
                "websocket",
                _FakeWebSocketRoute("wss://late.example/external-deadline"),
            )
        ]
    )
    acquisition = asyncio.create_task(
        _adapter(runtime, _FakeGuard([gated]), cleanup_grace_s=0.03).acquire(
            _TARGET,
            _profile(),
        ),
        name="test-deadline-callback-cancellation-acquisition",
    )

    try:
        await gated.started.wait()
        await runtime.context_closed.wait()
        callback = next(task for task in runtime.callback_tasks if not task.done())
        assert isinstance(callback, _CancelRecordingTask)
        loop.call_later(0.02, callback.cancel, "external-callback")

        with pytest.raises(asyncio.CancelledError):
            await acquisition

        assert callback.cancel_messages == ["external-callback"]
        assert gated.cancellation_args == ("external-callback",)
    finally:
        loop.set_task_factory(previous_factory)

    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_callback_capacity_rejection_fails_closed_without_dispatch() -> None:
    first = _FakeWebSocketRoute("wss://late.example/first")
    rejected = _FakeWebSocketRoute("wss://late.example/rejected")
    gated = _GatedOutcome(True)
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", first), ("websocket", rejected)])
    pool = _pool_type()(1)
    acquisition = asyncio.create_task(
        _adapter(
            runtime,
            _FakeGuard([gated]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
            callback_capacity=1,
        ).acquire(_TARGET, _profile()),
        name="test-callback-capacity-acquisition",
    )

    await gated.started.wait()
    with pytest.raises(ArticleFailure) as raised:
        await acquisition

    _assert_failure(raised.value, stage="capacity")
    assert rejected.connect_calls == 0
    assert rejected.close_calls == [(1008, "Policy denied")]
    assert gated.cancellation_args is None

    gated.release.set()
    await _assert_pool_active_count(pool, 0)
    await _assert_no_browser_tasks()


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["http", "websocket"])
async def test_callback_overflow_burst_uses_one_retained_emergency_shutdown(kind: str) -> None:
    gated = _GatedOutcome(True)
    runtime = _FakeBrowserRuntime(cleanup_modes={"close:context": "resistant"})
    if kind == "http":
        admitted_kind = "websocket"
        admitted: object = _FakeWebSocketRoute("wss://late.example/admitted")
        rejected: list[Any] = [
            _ResistantRejectedHttpRoute(
                f"https://late.example/rejected-{index}",
                runtime.target_closed,
            )
            for index in range(12)
        ]
    else:
        admitted_kind = "websocket"
        admitted = _FakeWebSocketRoute("wss://late.example/admitted")
        rejected = [
            _ResistantRejectedWebSocketRoute(
                f"wss://late.example/rejected-{index}",
                runtime.target_closed,
            )
            for index in range(12)
        ]
    runtime.dispatches = [
        (admitted_kind, admitted),
        *((kind, route) for route in rejected),
    ]
    pool = _pool_type()(1)
    acquisition = asyncio.create_task(
        _adapter(
            runtime,
            _FakeGuard([gated]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
            callback_capacity=1,
        ).acquire(_TARGET, _profile()),
        name=f"test-{kind}-callback-overflow-burst",
    )

    try:
        await gated.started.wait()
        with pytest.raises(ArticleFailure) as raised:
            await acquisition

        _assert_failure(raised.value, stage="capacity")
        assert runtime.context_close_task_names.count("article-browser-emergency-context-shutdown") == 1
        assert _retained_task_count(pool) == 3
        if kind == "http":
            assert [route.abort_calls for route in rejected] == [1] * len(rejected)
            assert [route.continue_calls for route in rejected] == [[] for _ in rejected]
            assert all(route.handled.is_set() for route in rejected)
            assert all(
                task.done() for task in runtime.all_callback_tasks if task.get_name() == "fake-playwright-http-callback"
            )
        else:
            assert [route.close_calls for route in rejected] == [[(1008, "Policy denied")] for _ in rejected]
            assert [route.connect_calls for route in rejected] == [0] * len(rejected)
    finally:
        gated.release.set()
        runtime.resistant_cleanup_release.set()

    await _assert_pool_active_count(pool, 0)
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_http_callback_after_released_lease_handles_closed_target_without_attaching() -> None:
    runtime = _FakeBrowserRuntime()
    pool = _pool_type()(1)

    html = await _adapter(
        runtime,
        _FakeGuard([]),
        acquisition_pool=pool,
    ).acquire(_TARGET, _profile())

    assert html == "<!doctype html><html><body>ok</body></html>"
    assert pool.active_count == 0
    for _ in range(8):
        await asyncio.sleep(0)

    late = _FakeHttpRoute("https://late.example/after-release")
    handler = runtime.installed_http_handler
    assert handler is not None
    context = runtime.context
    assert context is not None
    late_task = context._schedule(
        handler,
        late,
        kind="http",
    )
    await asyncio.wait_for(late_task, timeout=0.1)

    assert late.abort_calls == 1
    assert late.continue_calls == []
    assert late.handled.is_set()
    assert pool.active_count == 0

    second_runtime = _FakeBrowserRuntime()
    second_html = await _adapter(
        second_runtime,
        _FakeGuard([]),
        acquisition_pool=pool,
    ).acquire(_TARGET, _profile())
    assert second_html == "<!doctype html><html><body>ok</body></html>"
    assert pool.active_count == 0
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_callback_late_exception_is_durably_retained_and_consumed() -> None:
    resistant = _CancellationResistantOutcome()
    runtime = _FakeBrowserRuntime(dispatches=[("websocket", _FakeWebSocketRoute("wss://late.example/resistant"))])
    pool = _pool_type()(1)
    loop = asyncio.get_running_loop()
    unobserved: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unobserved.append(context))
    acquisition = asyncio.create_task(
        _adapter(
            runtime,
            _FakeGuard([resistant]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
        ).acquire(_TARGET, _profile()),
        name="test-resistant-callback-acquisition",
    )
    started = loop.time()

    try:
        with pytest.raises(ArticleFailure) as raised:
            await asyncio.wait_for(asyncio.shield(acquisition), timeout=0.2)
        assert loop.time() - started < 0.2
        _assert_failure(raised.value, stage="callback_drain")
        assert resistant.cancellation_args is None
        callback = next(task for task in runtime.callback_tasks if not task.done())
        callback_ref = weakref.ref(callback)
        runtime.callback_tasks.discard(callback)
        del callback
        gc.collect()
        assert callback_ref() is not None
        assert pool.active_count == 1
    finally:
        resistant.release.set()
        await resistant.finished.wait()
        await _assert_pool_active_count(pool, 0)
        await asyncio.sleep(0)
        loop.set_exception_handler(previous_handler)

    assert unobserved == []
    await _assert_no_browser_tasks()


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["raising_accessor", "missing"])
async def test_cleanup_accessor_or_method_failure_is_sanitized_and_all_owners_attempted(
    mode: str,
) -> None:
    runtime = _FakeBrowserRuntime(cleanup_modes={"close:page": mode})

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="cleanup")
    assert runtime.events[-3:] == ["close:context", "close:browser", "stop:playwright"]
    await _assert_no_browser_tasks()


@pytest.mark.unit
@pytest.mark.parametrize(
    "failing_name",
    ["article-browser-teardown", "article-browser-cleanup-page"],
)
async def test_teardown_task_creation_failure_attempts_owners_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
    failing_name: str,
) -> None:
    runtime = _FakeBrowserRuntime()
    pool = _pool_type()(1)
    original_create_task = asyncio.create_task

    def create_task(
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        if name == failing_name:
            coroutine.close()
            raise RuntimeError("https://secret.example/path?token=raw")
        kwargs: dict[str, object] = {"name": name}
        if context is not None:
            kwargs["context"] = context
        return original_create_task(coroutine, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create_task)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            _FakeGuard([]),
            acquisition_pool=pool,
        ).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="cleanup")
    assert runtime.events[-3:] == ["close:context", "close:browser", "stop:playwright"]
    assert pool.active_count == 0


@pytest.mark.unit
async def test_cleanup_callback_registration_failure_keeps_permanent_lease_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeBrowserRuntime()
    pool = _pool_type()(1)
    original_create_task = asyncio.create_task
    rejected_tasks: list[asyncio.Task[Any]] = []

    def create_task(
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        if name == "article-browser-cleanup-page":
            task = _DoneCallbackRejectingTask(
                coroutine,
                loop=asyncio.get_running_loop(),
                name=name,
            )
            rejected_tasks.append(task)
            return task
        kwargs: dict[str, object] = {"name": name}
        if context is not None:
            kwargs["context"] = context
        return original_create_task(coroutine, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create_task)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            _FakeGuard([]),
            acquisition_pool=pool,
        ).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="cleanup")
    assert len(rejected_tasks) == 1
    task_ref = weakref.ref(rejected_tasks[0])
    rejected_tasks.clear()
    gc.collect()
    assert task_ref() is not None
    assert pool.active_count == 1
    assert _retained_task_count(pool) == 1


@pytest.mark.unit
async def test_emergency_shutdown_creation_failure_falls_back_to_normal_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _GatedOutcome(True)
    rejected = _FakeWebSocketRoute("wss://late.example/rejected")
    runtime = _FakeBrowserRuntime(
        dispatches=[
            ("websocket", _FakeWebSocketRoute("wss://late.example/admitted")),
            ("websocket", rejected),
        ]
    )
    pool = _pool_type()(1)
    original_create_task = asyncio.create_task
    emergency_attempts = 0

    def create_task(
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        nonlocal emergency_attempts
        if name == "article-browser-emergency-context-shutdown":
            emergency_attempts += 1
            coroutine.close()
            raise RuntimeError("https://secret.example/path?token=raw")
        kwargs: dict[str, object] = {"name": name}
        if context is not None:
            kwargs["context"] = context
        return original_create_task(coroutine, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create_task)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            _FakeGuard([admitted]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
            callback_capacity=1,
        ).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="capacity")
    assert emergency_attempts == 1
    assert rejected.connect_calls == 0
    assert rejected.close_calls == [(1008, "Policy denied")]
    assert "article-browser-cleanup-context" in runtime.context_close_task_names

    admitted.release.set()
    await _assert_pool_active_count(pool, 0)


@pytest.mark.unit
async def test_emergency_shutdown_callback_registration_failure_keeps_permanent_lease_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _GatedOutcome(True)
    rejected = _FakeWebSocketRoute("wss://late.example/rejected")
    runtime = _FakeBrowserRuntime(
        dispatches=[
            ("websocket", _FakeWebSocketRoute("wss://late.example/admitted")),
            ("websocket", rejected),
        ]
    )
    pool = _pool_type()(1)
    original_create_task = asyncio.create_task
    rejected_tasks: list[asyncio.Task[Any]] = []

    def create_task(
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        if name == "article-browser-emergency-context-shutdown":
            task = _DoneCallbackRejectingTask(
                coroutine,
                loop=asyncio.get_running_loop(),
                name=name,
            )
            rejected_tasks.append(task)
            return task
        kwargs: dict[str, object] = {"name": name}
        if context is not None:
            kwargs["context"] = context
        return original_create_task(coroutine, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create_task)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(
            runtime,
            _FakeGuard([admitted]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
            callback_capacity=1,
        ).acquire(_TARGET, _profile())

    _assert_failure(raised.value, stage="capacity")
    assert rejected.close_calls == [(1008, "Policy denied")]
    assert len(rejected_tasks) == 1

    admitted.release.set()
    for _ in range(100):
        if _retained_task_count(pool) == 1:
            break
        await asyncio.sleep(0)

    task_ref = weakref.ref(rejected_tasks[0])
    rejected_tasks.clear()
    gc.collect()
    assert task_ref() is not None
    assert pool.active_count == 1
    assert _retained_task_count(pool) == 1


@pytest.mark.unit
async def test_stuck_cleanup_is_bounded_cancelled_consumed_and_all_owners_attempted() -> None:
    runtime = _FakeBrowserRuntime(cleanup_modes={"close:page": "stuck"})
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([]), cleanup_grace_s=0.01).acquire(
            _TARGET,
            _profile(),
        )

    assert loop.time() - started < 0.5
    _assert_failure(raised.value, stage="cleanup")
    assert runtime.stuck_cleanup_cancelled == ["close:page"]
    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]
    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_cancellation_resistant_cleanup_cannot_block_later_owners_or_teardown_bound() -> None:
    runtime = _FakeBrowserRuntime(cleanup_modes={"close:page": "resistant"})
    pool = _pool_type()(1)
    loop = asyncio.get_running_loop()
    unobserved: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unobserved.append(context))
    acquisition = asyncio.create_task(
        _adapter(
            runtime,
            _FakeGuard([]),
            cleanup_grace_s=0.01,
            acquisition_pool=pool,
        ).acquire(_TARGET, _profile()),
        name="test-resistant-cleanup-acquisition",
    )

    try:
        with pytest.raises(ArticleFailure) as raised:
            await asyncio.wait_for(asyncio.shield(acquisition), timeout=0.2)
        _assert_failure(raised.value, stage="cleanup")
        assert runtime.events[-3:] == ["close:context", "close:browser", "stop:playwright"]
        assert runtime.stuck_cleanup_cancelled == ["close:page"]
        assert pool.active_count == 1

        cleanup_task = runtime.cleanup_tasks.pop("close:page")
        cleanup_ref = weakref.ref(cleanup_task)
        del cleanup_task
        gc.collect()
        assert cleanup_ref() is not None

        blocked_runtime = _FakeBrowserRuntime()
        with pytest.raises(ArticleFailure) as blocked:
            await _adapter(
                blocked_runtime,
                _FakeGuard([]),
                acquisition_pool=pool,
            ).acquire(_TARGET, _profile())
        _assert_failure(blocked.value, stage="capacity")
        assert blocked_runtime.events == []

        runtime.resistant_cleanup_release.set()
        await runtime.resistant_cleanup_finished.wait()
        await _assert_pool_active_count(pool, 0)
        await asyncio.sleep(0)
        assert unobserved == []
    finally:
        runtime.resistant_cleanup_release.set()
        if not acquisition.done():
            with pytest.raises(ArticleFailure):
                await acquisition
        loop.set_exception_handler(previous_handler)

    await _assert_no_browser_tasks()


@pytest.mark.unit
async def test_repeated_caller_cancellation_preserves_cancellation_and_attempts_all_cleanup() -> None:
    runtime = _FakeBrowserRuntime(cleanup_modes={"close:page": "stuck", "close:context": "stuck"})
    acquisition = asyncio.create_task(
        _adapter(runtime, _FakeGuard([]), cleanup_grace_s=0.03).acquire(
            _TARGET,
            _profile(),
        ),
        name="test-repeated-cancellation-acquisition",
    )

    await runtime.cleanup_started.setdefault("close:page", asyncio.Event()).wait()
    acquisition.cancel()
    await runtime.cleanup_started.setdefault("close:context", asyncio.Event()).wait()
    acquisition.cancel()

    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert runtime.events[-4:] == [
        "close:page",
        "close:context",
        "close:browser",
        "stop:playwright",
    ]
    assert runtime.stuck_cleanup_cancelled == ["close:page", "close:context"]
    await _assert_no_browser_tasks()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("missing_context", "cdp_missing", "failing_stage"),
    [
        ("new_cdp_session", None, None),
        (None, "on", None),
        (None, "send", None),
        (None, None, "cdp-on:Network.dataReceived"),
        (None, None, "cdp-send:Network.enable"),
    ],
    ids=["session", "event-api", "command-api", "event-registration", "network-enable"],
)
async def test_task16_cdp_capability_failures_are_terminal_and_precede_navigation(
    missing_context: str | None,
    cdp_missing: str | None,
    failing_stage: str | None,
) -> None:
    errors = {failing_stage: RuntimeError("raw cdp error")} if failing_stage else None
    runtime = _FakeBrowserRuntime(
        missing_context_capability=missing_context,
        cdp_missing=cdp_missing,
        stage_errors=errors,
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            _profile(),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
        )

    _assert_failure(raised.value, stage="capability")
    assert runtime.events.count("launch") == 1
    assert not any(event.startswith("goto:") for event in runtime.events)


@pytest.mark.unit
async def test_task16_installs_all_transfer_accounting_before_navigation() -> None:
    runtime = _FakeBrowserRuntime()

    await _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        _profile(),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    goto_index = runtime.events.index(f"goto:{_TARGET}")
    assert runtime.events.index("new_cdp_session") < goto_index
    assert runtime.events.index("cdp-on:Network.dataReceived") < goto_index
    assert runtime.events.index("cdp-on:Network.webSocketFrameReceived") < goto_index
    assert runtime.events.index("cdp-on:Network.webSocketFrameSent") < goto_index
    assert runtime.events.index("cdp-send:Network.enable") < goto_index


@pytest.mark.unit
async def test_task16_uses_isolated_world_cdp_commands_after_navigation_wait() -> None:
    runtime = _FakeBrowserRuntime(html="<html></html>")

    html = await _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        replace(_profile(), stealth_enabled=False),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    assert html == "<html></html>"
    methods = [method for method, _ in runtime.cdp_command_calls]
    assert methods == [
        "Network.enable",
        "Page.getFrameTree",
        "Page.createIsolatedWorld",
        "Runtime.evaluate",
    ]
    wait_index = runtime.events.index("wait_for_load_state:networkidle")
    assert wait_index < runtime.events.index("cdp-send:Page.getFrameTree")
    assert runtime.evaluate_calls == []
    assert runtime.cdp_command_calls[2] == (
        "Page.createIsolatedWorld",
        {
            "frameId": "main-frame",
            "worldName": "tldw-article-serialization-v1",
            "grantUniveralAccess": False,
        },
    )
    method, params = runtime.cdp_command_calls[3]
    assert method == "Runtime.evaluate"
    assert params is not None
    assert params["contextId"] == 73
    assert params["returnByValue"] is True
    assert isinstance(params["expression"], str)
    assert str(params["expression"]).endswith(")(128)")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "response"),
    [
        ("Page.getFrameTree", None),
        (
            "Page.getFrameTree",
            {
                "exceptionDetails": {},
                "frameTree": {"frame": {"id": "main-frame"}},
            },
        ),
        ("Page.getFrameTree", {"frameTree": {"frame": {"id": ""}}}),
        ("Page.createIsolatedWorld", None),
        (
            "Page.createIsolatedWorld",
            {"exceptionDetails": {}, "executionContextId": 73},
        ),
        ("Page.createIsolatedWorld", {"executionContextId": True}),
        ("Runtime.evaluate", None),
        ("Runtime.evaluate", {"exceptionDetails": {}, "result": {"value": {"ok": True, "html": "x"}}}),
        ("Runtime.evaluate", {"result": None}),
        ("Runtime.evaluate", {"result": {"value": []}}),
    ],
    ids=[
        "frame-tree-nonmapping",
        "frame-tree-exception",
        "empty-frame-id",
        "isolated-world-nonmapping",
        "isolated-world-exception",
        "boolean-context-id",
        "evaluate-nonmapping",
        "evaluate-exception",
        "evaluate-result-nonmapping",
        "evaluate-value-nonmapping",
    ],
)
async def test_task16_malformed_isolated_world_command_envelope_is_terminal(
    method: str,
    response: object,
) -> None:
    runtime = _FakeBrowserRuntime(cdp_responses={method: response})

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3, stealth_enabled=False),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
        )

    _assert_failure(raised.value, stage="capability")
    assert runtime.events.count("launch") == 1
    assert runtime.evaluate_calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("encoded_length", "limit", "expected_stage"),
    [(8, 8, None), (8.0, 8, None), (9, 8, "browser_transfer")],
    ids=["exact-int", "exact-safe-float", "over"],
)
async def test_task16_http_encoded_transfer_accepts_exact_and_rejects_over(
    encoded_length: object,
    limit: int,
    expected_stage: str | None,
) -> None:
    runtime = _FakeBrowserRuntime(
        cdp_events=[("Network.dataReceived", {"encodedDataLength": encoded_length})],
        html="<html></html>",
    )
    operation = _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        replace(_profile(), retries=3),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=limit),
    )

    if expected_stage is None:
        assert await operation == "<html></html>"
    else:
        with pytest.raises(ArticleFailure) as raised:
            await operation
        _assert_limit_failure(raised.value, stage=expected_stage)
        assert runtime.events.count("launch") == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("encoded_length", "expected_code"),
    [
        (10**400, "response_too_large"),
        (float(2**53), "browser_error"),
        (1.5, "browser_error"),
        (float("inf"), "browser_error"),
    ],
    ids=["huge-exact-int", "unsafe-float", "fractional-float", "infinite-float"],
)
async def test_task16_http_transfer_numeric_validation_is_exact(
    encoded_length: object,
    expected_code: str,
) -> None:
    runtime = _FakeBrowserRuntime(cdp_events=[("Network.dataReceived", {"encodedDataLength": encoded_length})])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=8),
        )

    assert raised.value.code == expected_code
    assert raised.value.stage == ("browser_transfer" if expected_code == "response_too_large" else "capability")
    assert runtime.events.count("launch") == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event", "frame", "limit"),
    [
        ("Network.webSocketFrameReceived", {"opcode": 1, "payloadData": "éé"}, 4),
        ("Network.webSocketFrameReceived", {"opcode": 2, "payloadData": "AAEC"}, 3),
        ("Network.webSocketFrameSent", {"opcode": 1, "payloadData": "éé"}, 4),
        ("Network.webSocketFrameSent", {"opcode": 2, "payloadData": "AAEC"}, 3),
    ],
    ids=["received-text", "received-binary", "sent-text", "sent-binary"],
)
async def test_task16_websocket_frames_count_decoded_payload_bytes_exactly(
    event: str,
    frame: dict[str, object],
    limit: int,
) -> None:
    runtime = _FakeBrowserRuntime(
        cdp_events=[(event, {"response": frame})],
        html="<html></html>",
    )

    html = await _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        _profile(),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=limit),
    )

    assert html == "<html></html>"


@pytest.mark.unit
async def test_task16_text_transfer_rejects_surrogates_without_encoding_copy() -> None:
    runtime = _FakeBrowserRuntime(
        cdp_events=[
            (
                "Network.webSocketFrameReceived",
                {"response": {"opcode": 1, "payloadData": "ok\ud800"}},
            )
        ]
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            _profile(),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
        )

    _assert_failure(raised.value, stage="capability")


@pytest.mark.unit
async def test_task16_huge_text_short_circuits_after_crossing_remaining_budget() -> None:
    runtime = _FakeBrowserRuntime(
        cdp_events=[
            (
                "Network.webSocketFrameReceived",
                {"response": {"opcode": 1, "payloadData": _ExplodingTailText("ignored")}},
            )
        ]
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            _profile(),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=1),
        )

    _assert_limit_failure(raised.value, stage="browser_transfer")


@pytest.mark.unit
async def test_task16_huge_binary_rejects_without_scan_or_base64_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_module = _browser_module()

    def forbidden_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("over-limit base64 must not be decoded")

    monkeypatch.setattr(browser_module.base64, "b64decode", forbidden_decode)
    runtime = _FakeBrowserRuntime(
        cdp_events=[
            (
                "Network.webSocketFrameReceived",
                {
                    "response": {
                        "opcode": 2,
                        "payloadData": _UnscannableBinary("AAAA" * 10_000),
                    }
                },
            )
        ]
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            _profile(),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=1),
        )

    _assert_limit_failure(raised.value, stage="browser_transfer")


@pytest.mark.unit
async def test_task16_non_multiple_binary_length_rejects_without_scan_or_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_module = _browser_module()

    def forbidden_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("invalid-length base64 must not be decoded")

    monkeypatch.setattr(browser_module.base64, "b64decode", forbidden_decode)
    runtime = _FakeBrowserRuntime(
        cdp_events=[
            (
                "Network.webSocketFrameReceived",
                {
                    "response": {
                        "opcode": 2,
                        "payloadData": _UnscannableBinary("AAA"),
                    }
                },
            )
        ]
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
        )

    _assert_failure(raised.value, stage="capability")
    assert runtime.events.count("launch") == 1


@pytest.mark.unit
async def test_task16_zero_retries_preserves_legacy_no_attempt_result() -> None:
    runtime = _FakeBrowserRuntime()

    html = await _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        replace(_profile(), retries=0),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    assert html == ""
    assert runtime.events == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event", "payload"),
    [
        ("Network.dataReceived", []),
        ("Network.dataReceived", {}),
        ("Network.dataReceived", {"encodedDataLength": True}),
        ("Network.dataReceived", {"encodedDataLength": -1}),
        ("Network.webSocketFrameReceived", {}),
        ("Network.webSocketFrameReceived", {"response": []}),
        ("Network.webSocketFrameReceived", {"response": {"opcode": 1, "payloadData": 1}}),
        ("Network.webSocketFrameReceived", {"response": {"opcode": 2, "payloadData": "***"}}),
        ("Network.webSocketFrameReceived", {"response": {"opcode": 9, "payloadData": "x"}}),
    ],
    ids=[
        "nonmapping-http-envelope",
        "missing-http-length",
        "boolean-http-length",
        "negative-http-length",
        "missing-frame",
        "nonmapping-frame-envelope",
        "non-string-frame",
        "invalid-base64",
        "unsupported-opcode",
    ],
)
async def test_task16_invalid_transfer_payload_fails_closed_without_retry(
    event: str,
    payload: object,
) -> None:
    runtime = _FakeBrowserRuntime(cdp_events=[(event, payload)])

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=8),
        )

    _assert_failure(raised.value, stage="capability")
    assert runtime.events.count("launch") == 1


@pytest.mark.unit
async def test_task16_transfer_overflow_reuses_one_retained_emergency_shutdown() -> None:
    runtime = _FakeBrowserRuntime(
        cdp_events=[
            ("Network.dataReceived", {"encodedDataLength": 9}),
            ("Network.dataReceived", {"encodedDataLength": 9}),
            (
                "Network.webSocketFrameReceived",
                {"response": {"opcode": 1, "payloadData": "overflow"}},
            ),
        ]
    )

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=8),
        )

    _assert_limit_failure(raised.value, stage="browser_transfer")
    assert runtime.context_close_task_names.count("article-browser-emergency-context-shutdown") == 1
    assert runtime.events.count("launch") == 1


@pytest.mark.unit
async def test_task16_latched_transfer_overflow_wins_over_pending_cdp_and_teardown_errors() -> None:
    release_evaluate = asyncio.Event()
    runtime = _FakeBrowserRuntime(
        cdp_command_gates={"Runtime.evaluate": release_evaluate},
        stage_errors={
            "cdp-send-after:Runtime.evaluate": RuntimeError("CDP target closed"),
            "detach:cdp": RuntimeError("CDP detach failed"),
        },
    )
    operation = asyncio.create_task(
        _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=8),
        )
    )

    await runtime.cdp_command_started.setdefault("Runtime.evaluate", asyncio.Event()).wait()
    session = runtime.cdp_sessions[0]
    session.emit("Network.dataReceived", {"encodedDataLength": 9})
    await runtime.context_closed.wait()
    release_evaluate.set()

    with pytest.raises(ArticleFailure) as raised:
        await operation

    _assert_limit_failure(raised.value, stage="browser_transfer")
    assert runtime.context_close_task_names.count("article-browser-emergency-context-shutdown") == 1
    assert session.detached is True
    assert runtime.events.count("launch") == 1


@pytest.mark.unit
@pytest.mark.parametrize("delta", [0, 1], ids=["exact", "over"])
async def test_task16_rendered_html_is_measured_in_browser_without_page_content(
    delta: int,
) -> None:
    html = "<!DOCTYPE html>\n<html><body>é</body></html>"
    size = len(html.encode("utf-8"))
    runtime = _FakeBrowserRuntime(html=html)
    limit = size - delta

    operation = _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        _profile(),
        ArticleLimits(max_article_bytes=limit, max_browser_transfer_bytes=128),
    )
    if delta == 0:
        assert await operation == html
    else:
        with pytest.raises(ArticleFailure) as raised:
            await operation
        _assert_limit_failure(raised.value, stage="rendered_html")

    assert "content" not in runtime.events
    assert runtime.evaluate_calls == []
    method, params = next(call for call in runtime.cdp_command_calls if call[0] == "Runtime.evaluate")
    assert method == "Runtime.evaluate"
    assert params is not None
    expression = params["expression"]
    assert isinstance(expression, str)
    assert "new XMLSerializer().serializeToString(document.doctype)" in expression
    assert "document.documentElement.outerHTML" in expression
    assert "TextEncoder" not in expression
    assert "for (const character of html)" in expression
    assert "if (size > maxBytes)" in expression
    assert expression.endswith(f")({limit})")


@pytest.mark.unit
@pytest.mark.parametrize(
    "result",
    [None, {}, {"ok": 1}, {"ok": True}, {"ok": True, "html": 1}, {"ok": False}, {"ok": False, "html": "leak"}],
)
async def test_task16_malformed_browser_serialization_result_fails_closed_without_retry(
    result: object,
) -> None:
    runtime = _FakeBrowserRuntime(evaluation_result=result)

    with pytest.raises(ArticleFailure) as raised:
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
        )

    _assert_failure(raised.value, stage="capability")
    assert runtime.events.count("launch") == 1
    assert "content" not in runtime.events
    assert runtime.evaluate_calls == []


@pytest.mark.unit
async def test_task16_direct_browser_profile_preserves_cookies_stealth_launch_and_viewport() -> None:
    runtime = _FakeBrowserRuntime(html="<html></html>")

    async def stealth(page: Any) -> None:
        runtime.events.append("stealth")
        runtime.stealth_pages.append(page)

    await _adapter(runtime, _FakeGuard([]), stealth_hook=stealth).acquire(
        _TARGET,
        _profile(),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    assert runtime.launch_options == {"headless": True}
    assert runtime.context_options == {
        "service_workers": "block",
        "user_agent": "Task15 Browser/1.0",
        "viewport": {"width": 1024, "height": 768},
    }
    assert runtime.cookies == [
        {
            "name": "caller-cookie",
            "value": "secret",
            "domain": "article.example",
            "path": "/",
        }
    ]
    assert len(runtime.stealth_pages) == 1
    assert "wait_for_timeout:250" in runtime.events
    assert not any(event.startswith("wait_for_load_state:") for event in runtime.events)
    assert runtime.events.index("cdp-send:Network.enable") < runtime.events.index("stealth")
    assert runtime.events.index("stealth") < runtime.events.index(f"goto:{_TARGET}")
    assert runtime.events.index(f"goto:{_TARGET}") < runtime.events.index("wait_for_timeout:250")


@pytest.mark.unit
async def test_task16_non_stealth_navigation_waits_for_networkidle_with_timeout() -> None:
    runtime = _FakeBrowserRuntime(html="<html></html>")

    await _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        replace(_profile(), stealth_enabled=False),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    assert "wait_for_load_state:networkidle" in runtime.events
    assert "wait-options:['timeout']" in runtime.events
    assert not any(event.startswith("wait_for_timeout:") for event in runtime.events)


@pytest.mark.unit
async def test_task16_ordinary_navigation_failure_retries_with_injected_sleep() -> None:
    runtime = _FakeBrowserRuntime(
        stage_errors={"goto": [RuntimeError("first navigation failed"), None]},
        html="<html></html>",
    )

    async def sleep(delay: float) -> None:
        runtime.sleep_delays.append(delay)

    html = await _adapter(runtime, _FakeGuard([]), sleep=sleep).acquire(
        _TARGET,
        replace(_profile(), retries=2, stealth_enabled=False),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    assert html == "<html></html>"
    assert runtime.events.count("launch") == 2
    assert runtime.sleep_delays == [2.0]
    assert len(runtime.cdp_sessions) == 2
    assert all(session.detached for session in runtime.cdp_sessions)


@pytest.mark.unit
async def test_task16_caller_cancellation_never_retries() -> None:
    runtime = _FakeBrowserRuntime(stage_errors={"goto": asyncio.CancelledError()})

    with pytest.raises(asyncio.CancelledError):
        await _adapter(runtime, _FakeGuard([])).acquire(
            _TARGET,
            replace(_profile(), retries=3),
            ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
        )

    assert runtime.events.count("launch") == 1


@pytest.mark.unit
async def test_task16_cdp_session_is_owned_and_detached_on_success() -> None:
    runtime = _FakeBrowserRuntime(html="<html></html>")

    await _adapter(runtime, _FakeGuard([])).acquire(
        _TARGET,
        _profile(),
        ArticleLimits(max_article_bytes=128, max_browser_transfer_bytes=128),
    )

    assert len(runtime.cdp_sessions) == 1
    assert runtime.cdp_sessions[0].detached is True
