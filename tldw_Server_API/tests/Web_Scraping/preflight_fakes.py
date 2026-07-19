"""Deterministic fakes for governed preflight contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from tldw_Server_API.app.core.Web_Scraping.runtime import (
    ProbeEgressDecision,
    RuntimeRequestContext,
)


class FakeClock:
    """Mutable monotonic clock controlled by tests."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSleep:
    """Sleep callable that advances a fake clock without wall-clock delay."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay_s: float) -> None:
        self.delays.append(delay_s)
        self.clock.advance(delay_s)


class EventSleep:
    """Sleep callable controlled by events and cancelled only by its caller task."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, delay_s: float) -> None:
        self.delays.append(delay_s)
        self.started.set()
        await self.release.wait()
        self.clock.advance(delay_s)


class FakeCleanupHandle:
    """Cleanup handle with controllable graceful-close behavior."""

    def __init__(
        self,
        *,
        block_close: bool = False,
        suppress_close_cancellation: bool = False,
        block_force_close: bool = False,
        suppress_force_cancellation: bool = False,
        force_releases_close: bool = True,
        close_error: BaseException | None = None,
        events: list[str] | None = None,
        name: str = "cleanup",
    ) -> None:
        self.block_close = block_close
        self.suppress_close_cancellation = suppress_close_cancellation
        self.block_force_close = block_force_close
        self.suppress_force_cancellation = suppress_force_cancellation
        self.force_releases_close = force_releases_close
        self.close_error = close_error
        self.events = events
        self.name = name
        self.close_calls = 0
        self.close_cancellations = 0
        self.close_tasks: list[asyncio.Task[object]] = []
        self.force_close_calls = 0
        self.force_close_cancellations = 0
        self.force_close_tasks: list[asyncio.Task[object]] = []
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.force_close_started = asyncio.Event()
        self.force_close_finished = asyncio.Event()
        self._release_close = asyncio.Event()
        self._release_force_close = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        task = asyncio.current_task()
        if task is not None:
            self.close_tasks.append(task)
        if self.events is not None:
            self.events.append(f"close:{self.name}")
        self.close_started.set()
        try:
            if self.close_error is not None:
                raise self.close_error
            if self.block_close:
                while not self._release_close.is_set():
                    try:
                        await self._release_close.wait()
                    except asyncio.CancelledError:
                        self.close_cancellations += 1
                        if not self.suppress_close_cancellation:
                            raise
        finally:
            self.close_finished.set()

    async def force_close(self) -> None:
        self.force_close_calls += 1
        task = asyncio.current_task()
        if task is not None:
            self.force_close_tasks.append(task)
        if self.events is not None:
            self.events.append(f"force:{self.name}")
        self.force_close_started.set()
        if self.force_releases_close:
            self._release_close.set()
        try:
            if self.block_force_close:
                while not self._release_force_close.is_set():
                    try:
                        await self._release_force_close.wait()
                    except asyncio.CancelledError:
                        self.force_close_cancellations += 1
                        if not self.suppress_force_cancellation:
                            raise
        finally:
            self.force_close_finished.set()

    def release_close(self) -> None:
        self._release_close.set()

    def release_force_close(self) -> None:
        self._release_force_close.set()


class FakeIdentitySelector:
    """Identity selector that records how often it is consulted."""

    def __init__(self, identity: Mapping[str, str]) -> None:
        self.identity = dict(identity)
        self.calls = 0

    def __call__(self) -> Mapping[str, str]:
        self.calls += 1
        return self.identity


class FakePolicyChecker:
    async def decide(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("policy evaluation is not expected in contract tests")


class FakeEgressGuard:
    async def decide(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("egress evaluation is not expected in contract tests")


class FakeHttpProbe:
    async def get(self, _request: Any) -> Any:
        raise AssertionError("HTTP probing is not expected in contract tests")


class FakeBrowserProbe:
    def open_page(self, _options: Any) -> Any:
        raise AssertionError("browser probing is not expected in contract tests")


class FakeExternalToolProbe:
    async def run_waf(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("external probing is not expected in contract tests")


class FakeCallRecorder:
    """Record synchronous observer calls and optionally raise a fixed error."""

    def __init__(
        self,
        *,
        error: BaseException | None = None,
        events: list[str] | None = None,
        name: str = "call",
    ) -> None:
        self.error = error
        self.events = events
        self.name = name
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, dict(kwargs)))
        if self.events is not None:
            self.events.append(self.name)
        if self.error is not None:
            raise self.error

    def observe(self) -> None:
        self()


class FakeWhich:
    """Deterministic executable lookup that never inspects the host system."""

    def __init__(
        self,
        result: str | None,
        *,
        error: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.events = events
        self.calls: list[str] = []

    def __call__(self, executable: str) -> str | None:
        self.calls.append(executable)
        if self.events is not None:
            self.events.append("which")
        if self.error is not None:
            raise self.error
        return self.result


class FakeCoercionValue:
    """Raise fixed errors when policy values are coerced."""

    def __init__(
        self,
        *,
        bool_error: BaseException | None = None,
        str_error: BaseException | None = None,
    ) -> None:
        self.bool_error = bool_error
        self.str_error = str_error

    def __bool__(self) -> bool:
        if self.bool_error is not None:
            raise self.bool_error
        return True

    def __str__(self) -> str:
        if self.str_error is not None:
            raise self.str_error
        return "allowed"


class FakeExternalToolDecision:
    """Controllable policy decision with failing accessors when requested."""

    def __init__(
        self,
        *,
        allowed: Any = True,
        reason: Any = "allowed",
        allowed_error: BaseException | None = None,
        reason_error: BaseException | None = None,
    ) -> None:
        self._allowed = allowed
        self._reason = reason
        self.allowed_error = allowed_error
        self.reason_error = reason_error

    @property
    def allowed(self) -> Any:
        if self.allowed_error is not None:
            raise self.allowed_error
        return self._allowed

    @property
    def reason(self) -> Any:
        if self.reason_error is not None:
            raise self.reason_error
        return self._reason


class FakeExternalProcess:
    """Controllable async subprocess without host process creation."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes | str = b"",
        stderr: bytes | str = b"",
        block_communicate: bool = False,
        communicate_error: BaseException | None = None,
        communicate_hook: Callable[[], None] | None = None,
        terminate_completes: bool = True,
        terminate_error: BaseException | None = None,
        kill_error: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.planned_returncode = returncode
        self.returncode: int | None = None
        self.stdout = stdout
        self.stderr = stderr
        self.block_communicate = block_communicate
        self.communicate_error = communicate_error
        self.communicate_hook = communicate_hook
        self.terminate_completes = terminate_completes
        self.terminate_error = terminate_error
        self.kill_error = kill_error
        self.events = events
        self.communicate_calls = 0
        self.communicate_cancellations = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.wait_cancellations = 0
        self.communicate_started = asyncio.Event()
        self.wait_started = asyncio.Event()
        self._release_communicate = asyncio.Event()
        self._terminal = asyncio.Event()

    async def communicate(self) -> tuple[bytes | str, bytes | str]:
        self.communicate_calls += 1
        self.communicate_started.set()
        if self.events is not None:
            self.events.append("process:communicate")
        try:
            if self.block_communicate:
                await self._release_communicate.wait()
            if self.communicate_error is not None:
                raise self.communicate_error
        except asyncio.CancelledError:
            self.communicate_cancellations += 1
            raise
        self.returncode = self.planned_returncode
        self._terminal.set()
        if self.communicate_hook is not None:
            self.communicate_hook()
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.events is not None:
            self.events.append("process:terminate")
        if self.terminate_error is not None:
            raise self.terminate_error
        if self.terminate_completes:
            self.returncode = -15
            self._terminal.set()
            self._release_communicate.set()

    def kill(self) -> None:
        self.kill_calls += 1
        if self.events is not None:
            self.events.append("process:kill")
        if self.kill_error is not None:
            raise self.kill_error
        self.returncode = -9
        self._terminal.set()
        self._release_communicate.set()

    async def wait(self) -> int:
        self.wait_calls += 1
        self.wait_started.set()
        if self.events is not None:
            self.events.append("process:wait")
        try:
            await self._terminal.wait()
        except asyncio.CancelledError:
            self.wait_cancellations += 1
            raise
        assert self.returncode is not None
        return self.returncode

    def release_communicate(self) -> None:
        self._release_communicate.set()


class FakeProcessFactory:
    """Queue-backed create_subprocess_exec replacement with call capture."""

    def __init__(
        self,
        processes: list[FakeExternalProcess] | None = None,
        *,
        error: BaseException | None = None,
        block_creation: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.processes = list(processes or [])
        self.error = error
        self.block_creation = block_creation
        self.events = events
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.creation_started = asyncio.Event()
        self.creation_cancellations = 0
        self._release_creation = asyncio.Event()

    async def __call__(self, *args: Any, **kwargs: Any) -> FakeExternalProcess:
        self.calls.append((args, dict(kwargs)))
        self.creation_started.set()
        if self.events is not None:
            self.events.append("process:create")
        try:
            if self.block_creation:
                await self._release_creation.wait()
        except asyncio.CancelledError:
            self.creation_cancellations += 1
            raise
        if self.error is not None:
            raise self.error
        if not self.processes:
            raise AssertionError("unexpected process creation")
        return self.processes.pop(0)

    def release_creation(self) -> None:
        self._release_creation.set()


class FakeProbeEgressGuard:
    """Queue-backed probe guard that records each fresh decision."""

    def __init__(
        self,
        decisions: list[bool | str | ProbeEgressDecision | FakeExternalToolDecision | BaseException],
        *,
        events: list[str] | None = None,
    ) -> None:
        self._decisions = list(decisions)
        self.events = events
        self.urls: list[str] = []
        self.contexts: list[RuntimeRequestContext] = []

    async def decide(
        self,
        url: str,
        *,
        context: RuntimeRequestContext,
    ) -> ProbeEgressDecision:
        self.urls.append(url)
        self.contexts.append(context)
        if self.events is not None:
            self.events.append(f"guard:{url}")
        if not self._decisions:
            raise AssertionError("unexpected egress decision")
        decision = self._decisions.pop(0)
        if isinstance(decision, BaseException):
            raise decision
        if isinstance(decision, FakeExternalToolDecision):
            return decision  # type: ignore[return-value]
        if isinstance(decision, ProbeEgressDecision):
            return decision
        if isinstance(decision, str):
            return ProbeEgressDecision(allowed=False, reason=decision)
        return ProbeEgressDecision(
            allowed=decision,
            reason="allowed" if decision else "address_forbidden",
        )


class FakeRawResponse:
    """Minimal async response with deterministic close controls."""

    def __init__(
        self,
        status_code: int,
        *,
        headers: Mapping[str, str] | None = None,
        text: str = "",
        url: str | None = None,
        close_error: BaseException | None = None,
        block_close: bool = False,
        suppress_close_cancellation: bool = False,
        close_error_after_release: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.text = text
        self.url = url
        self.close_error = close_error
        self.block_close = block_close
        self.suppress_close_cancellation = suppress_close_cancellation
        self.close_error_after_release = close_error_after_release
        self.events = events
        self.close_calls = 0
        self.close_cancellations = 0
        self.closed = False
        self.close_started = asyncio.Event()
        self._release_close = asyncio.Event()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.events is not None:
            self.events.append("response:close")
        if self.block_close:
            while not self._release_close.is_set():
                try:
                    await self._release_close.wait()
                except asyncio.CancelledError:
                    self.close_cancellations += 1
                    if not self.suppress_close_cancellation:
                        raise
        self.closed = True
        if self.close_error is not None:
            raise self.close_error
        if self.close_error_after_release is not None:
            raise self.close_error_after_release

    def release_close(self) -> None:
        self._release_close.set()


class FakeHttpTransport:
    """Queue-backed async HTTP transport with immutable call capture."""

    def __init__(
        self,
        responses: list[Any],
        *,
        events: list[str] | None = None,
        block_send: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.events = events
        self.block_send = block_send
        self.calls: list[Any] = []
        self.send_started = asyncio.Event()
        self._release_send = asyncio.Event()

    async def send(self, request: Any) -> Any:
        self.calls.append(request)
        self.send_started.set()
        if self.events is not None:
            self.events.append(f"transport:{request.url}")
        if self.block_send:
            await self._release_send.wait()
        if not self.responses:
            raise AssertionError("unexpected HTTP transport dispatch")
        result = self.responses.pop(0)
        if callable(result):
            result = result()
        if isinstance(result, BaseException):
            raise result
        return result

    def release_send(self) -> None:
        self._release_send.set()


class FakeBrowserRequest:
    """Minimal browser request passed to HTTP route handlers."""

    def __init__(self, url: str, resource_type: str = "document") -> None:
        self.url = url
        self.resource_type = resource_type


class FakeBrowserRoute:
    """HTTP route fake recording the selected routing action."""

    def __init__(
        self,
        request: FakeBrowserRequest,
        events: list[str],
        *,
        abort_error: BaseException | None = None,
        continue_error: BaseException | None = None,
    ) -> None:
        self.request = request
        self.events = events
        self.abort_error = abort_error
        self.continue_error = continue_error
        self.abort_calls = 0
        self.continue_calls = 0

    async def abort(self) -> None:
        self.abort_calls += 1
        self.events.append("http:abort")
        if self.abort_error is not None:
            raise self.abort_error

    async def continue_(self) -> None:
        self.continue_calls += 1
        self.events.append("http:continue")
        if self.continue_error is not None:
            raise self.continue_error


class FakeWebSocketRoute:
    """WebSocket route fake supporting synchronous or awaitable connection."""

    def __init__(
        self,
        url: str,
        events: list[str],
        *,
        awaitable_connect: bool = True,
        connect_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.url = url
        self.events = events
        self.awaitable_connect = awaitable_connect
        self.connect_error = connect_error
        self.close_error = close_error
        self.connect_calls = 0
        self.close_calls: list[tuple[int | None, str | None]] = []

    def connect_to_server(self) -> Awaitable[None] | None:
        self.connect_calls += 1
        self.events.append("websocket:connect")
        if self.connect_error is not None:
            raise self.connect_error
        if not self.awaitable_connect:
            return None

        async def _complete() -> None:
            return None

        return _complete()

    async def close(
        self,
        *,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.close_calls.append((code, reason))
        self.events.append("websocket:close")
        if self.close_error is not None:
            raise self.close_error


class FakeBrowserStartupGate:
    """Block one selected startup await until its caller is cancelled."""

    def __init__(self, block_at: str | None = None) -> None:
        self.block_at = block_at
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def wait(self, stage: str) -> None:
        if stage != self.block_at:
            return
        self.started.set()
        await self._release.wait()


class FakeBrowserLocator:
    """Locator fake for link count and visibility wrapper methods."""

    def __init__(self, visible_links: tuple[bool, ...]) -> None:
        self._visible_links = visible_links
        self._index: int | None = None

    def nth(self, index: int) -> FakeBrowserLocator:
        locator = FakeBrowserLocator(self._visible_links)
        locator._index = index
        return locator

    async def count(self) -> int:
        return len(self._visible_links)

    async def is_visible(self) -> bool:
        if self._index is None:
            raise AssertionError("nth() must be called before is_visible()")
        return self._visible_links[self._index]


class FakeBrowserPage:
    """Async page fake with operation capture and controllable closure."""

    def __init__(
        self,
        context: FakeBrowserContext,
        events: list[str],
        *,
        block_close: bool = False,
        suppress_close_cancellation: bool = False,
        max_suppressed_close_cancellations: int = 1,
    ) -> None:
        self.context = context
        self.events = events
        self.block_close = block_close
        self.suppress_close_cancellation = suppress_close_cancellation
        self.max_suppressed_close_cancellations = max_suppressed_close_cancellations
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.results: dict[str, Any] = {
            "content": "<html>fake</html>",
            "evaluate": None,
        }
        self.errors: dict[str, BaseException] = {}
        self.visible_links = (True, False)
        self.close_calls = 0
        self.force_close_calls = 0
        self.close_cancellations = 0
        self.close_started = asyncio.Event()
        self._release_close = asyncio.Event()

    async def _operation(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args, dict(kwargs)))
        error = self.errors.get(name)
        if error is not None:
            raise error
        return self.results.get(name)

    async def goto(self, url: str, **kwargs: Any) -> Any:
        return await self._operation("goto", url, **kwargs)

    async def reload(self, **kwargs: Any) -> Any:
        return await self._operation("reload", **kwargs)

    async def wait_for_load_state(self, state: str, **kwargs: Any) -> Any:
        return await self._operation("wait_for_load_state", state, **kwargs)

    async def wait_for_timeout(self, timeout_ms: float) -> Any:
        return await self._operation("wait_for_timeout", timeout_ms)

    async def content(self) -> str:
        return str(await self._operation("content"))

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        return await self._operation("evaluate", expression, argument)

    def locator(self, selector: str) -> FakeBrowserLocator:
        if selector != "a":
            raise AssertionError(f"unexpected locator selector: {selector}")
        return FakeBrowserLocator(self.visible_links)

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("close:page")
        self.close_started.set()
        if self.block_close:
            while not self._release_close.is_set():
                try:
                    await self._release_close.wait()
                except asyncio.CancelledError:
                    self.close_cancellations += 1
                    if (
                        not self.suppress_close_cancellation
                        or self.close_cancellations > self.max_suppressed_close_cancellations
                    ):
                        raise

    async def force_close(self) -> None:
        self.force_close_calls += 1
        self.events.append("force:page")
        self._release_close.set()

    def release_close(self) -> None:
        self._release_close.set()


class FakeBrowserContext:
    """Browser context fake retaining installed handlers for direct dispatch."""

    def __init__(
        self,
        events: list[str],
        *,
        page_factory: Callable[[FakeBrowserContext, list[str]], FakeBrowserPage] | None = None,
        startup_gate: FakeBrowserStartupGate | None = None,
    ) -> None:
        self.events = events
        self.page_factory = page_factory or FakeBrowserPage
        self.startup_gate = startup_gate or FakeBrowserStartupGate()
        self.http_handler: Callable[[FakeBrowserRoute], Awaitable[None]] | None = None
        self.websocket_handler: Callable[[FakeWebSocketRoute], Awaitable[None]] | None = None
        self.request_handler: Callable[[FakeBrowserRequest], None] | None = None
        self.init_scripts: list[str] = []
        self.pages: list[FakeBrowserPage] = []
        self.close_calls = 0
        self.force_close_calls = 0

    async def route(self, pattern: str, handler: Callable[[FakeBrowserRoute], Awaitable[None]]) -> None:
        await self.startup_gate.wait("route_http")
        assert pattern == "**/*"
        self.http_handler = handler
        self.events.append("route_http")

    async def route_web_socket(
        self,
        pattern: str,
        handler: Callable[[FakeWebSocketRoute], Awaitable[None]],
    ) -> None:
        await self.startup_gate.wait("route_web_socket")
        assert pattern == "**/*"
        self.websocket_handler = handler
        self.events.append("route_web_socket")

    async def add_init_script(self, *, script: str) -> None:
        await self.startup_gate.wait("init_script")
        self.init_scripts.append(script)
        self.events.append("init_script")

    def on(self, event: str, handler: Callable[[FakeBrowserRequest], None]) -> None:
        assert event == "request"
        self.request_handler = handler
        self.events.append("capture_requests")

    async def new_page(self) -> FakeBrowserPage:
        await self.startup_gate.wait("new_page")
        self.events.append("new_page")
        page = self.page_factory(self, self.events)
        self.pages.append(page)
        return page

    async def dispatch_http(
        self,
        url: str,
        resource_type: str = "document",
        *,
        abort_error: BaseException | None = None,
        continue_error: BaseException | None = None,
    ) -> FakeBrowserRoute:
        if self.http_handler is None:
            raise AssertionError("HTTP handler was not installed")
        request = FakeBrowserRequest(url, resource_type)
        if self.request_handler is not None:
            self.request_handler(request)
        route = FakeBrowserRoute(
            request,
            self.events,
            abort_error=abort_error,
            continue_error=continue_error,
        )
        await self.http_handler(route)
        return route

    async def dispatch_websocket(
        self,
        url: str,
        *,
        awaitable_connect: bool = True,
        connect_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> FakeWebSocketRoute:
        if self.websocket_handler is None:
            raise AssertionError("WebSocket handler was not installed")
        route = FakeWebSocketRoute(
            url,
            self.events,
            awaitable_connect=awaitable_connect,
            connect_error=connect_error,
            close_error=close_error,
        )
        await self.websocket_handler(route)
        return route

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("close:context")

    async def force_close(self) -> None:
        self.force_close_calls += 1
        self.events.append("force:context")
        for page in self.pages:
            page.release_close()


class FakeBrowser:
    """Browser fake creating one context from supplied options."""

    def __init__(
        self,
        events: list[str],
        *,
        context_factory: Callable[[list[str]], FakeBrowserContext] | None = None,
        startup_gate: FakeBrowserStartupGate | None = None,
    ) -> None:
        self.events = events
        self.context_factory = context_factory or FakeBrowserContext
        self.startup_gate = startup_gate or FakeBrowserStartupGate()
        self.contexts: list[FakeBrowserContext] = []
        self.context_options: list[dict[str, Any]] = []
        self.close_calls = 0
        self.force_close_calls = 0

    async def new_context(self, **kwargs: Any) -> FakeBrowserContext:
        await self.startup_gate.wait("new_context")
        self.context_options.append(dict(kwargs))
        self.events.append(f"new_context:service_workers={kwargs.get('service_workers')}")
        context = self.context_factory(self.events)
        context.startup_gate = self.startup_gate
        self.contexts.append(context)
        return context

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("close:browser")

    async def force_close(self) -> None:
        self.force_close_calls += 1
        self.events.append("force:browser")


class FakeChromium:
    """Chromium fake recording secure launch options."""

    def __init__(
        self,
        browser: FakeBrowser,
        events: list[str],
        startup_gate: FakeBrowserStartupGate,
    ) -> None:
        self.browser = browser
        self.events = events
        self.startup_gate = startup_gate
        self.launch_calls: list[dict[str, Any]] = []

    async def launch(self, **kwargs: Any) -> FakeBrowser:
        await self.startup_gate.wait("launch_browser")
        self.launch_calls.append(dict(kwargs))
        self.events.append("launch_browser")
        return self.browser


class FakePlaywright:
    """Started Playwright fake exposing Chromium and stop hooks."""

    def __init__(
        self,
        browser: FakeBrowser,
        events: list[str],
        startup_gate: FakeBrowserStartupGate,
    ) -> None:
        self.events = events
        self.chromium = FakeChromium(browser, events, startup_gate)
        self.stop_calls = 0
        self.force_close_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        self.events.append("close:playwright")

    async def force_close(self) -> None:
        self.force_close_calls += 1
        self.events.append("force:playwright")


class FakePlaywrightLauncher:
    """Injected Playwright starter with inspectable resource graph."""

    def __init__(
        self,
        *,
        context_factory: Callable[[list[str]], FakeBrowserContext] | None = None,
        block_at: str | None = None,
    ) -> None:
        self.events: list[str] = []
        self.startup_gate = FakeBrowserStartupGate(block_at)
        self.browser = FakeBrowser(
            self.events,
            context_factory=context_factory,
            startup_gate=self.startup_gate,
        )
        self.playwright = FakePlaywright(
            self.browser,
            self.events,
            self.startup_gate,
        )
        self.start_calls = 0

    async def start(self) -> FakePlaywright:
        await self.startup_gate.wait("launcher_start")
        self.start_calls += 1
        self.events.append("launch")
        return self.playwright


class RealLikeBrowserPage(FakeBrowserPage):
    """Page exposing only Playwright's public close operation."""

    force_close = None

    def __init__(self, context: FakeBrowserContext, events: list[str]) -> None:
        super().__init__(context, events)
        self.closed = False
        self.close_started_at: float | None = None

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("close:page")
        self.close_started_at = asyncio.get_running_loop().time()
        self.close_started.set()
        await self._release_close.wait()
        self.closed = True


class RealLikeBrowserContext(FakeBrowserContext):
    """Context close releases page closes and awaits browser teardown."""

    force_close = None

    def __init__(self, events: list[str]) -> None:
        super().__init__(events, page_factory=RealLikeBrowserPage)
        self.closed = False
        self.close_started_at: float | None = None
        self._release_close = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("close:context")
        self.close_started_at = asyncio.get_running_loop().time()
        for page in self.pages:
            page.release_close()
        await self._release_close.wait()
        self.closed = True

    def release_close(self) -> None:
        self._release_close.set()


class RealLikeBrowser(FakeBrowser):
    """Browser close releases context closes and awaits Playwright stop."""

    force_close = None

    def __init__(
        self,
        events: list[str],
        startup_gate: FakeBrowserStartupGate,
    ) -> None:
        super().__init__(
            events,
            context_factory=RealLikeBrowserContext,
            startup_gate=startup_gate,
        )
        self.closed = False
        self.close_started_at: float | None = None
        self._release_close = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("close:browser")
        self.close_started_at = asyncio.get_running_loop().time()
        for context in self.contexts:
            assert isinstance(context, RealLikeBrowserContext)
            context.release_close()
        await self._release_close.wait()
        self.closed = True

    def release_close(self) -> None:
        self._release_close.set()


class RealLikePlaywright(FakePlaywright):
    """Playwright exposing only stop, which completes browser teardown."""

    force_close = None

    def __init__(
        self,
        browser: RealLikeBrowser,
        events: list[str],
        startup_gate: FakeBrowserStartupGate,
    ) -> None:
        super().__init__(browser, events, startup_gate)
        self.browser = browser
        self.stopped = False
        self.stop_started_at: float | None = None

    async def stop(self) -> None:
        self.stop_calls += 1
        self.events.append("close:playwright")
        self.stop_started_at = asyncio.get_running_loop().time()
        self.browser.release_close()
        self.stopped = True


class RealLikePlaywrightLauncher(FakePlaywrightLauncher):
    """All-public-API resource graph with parent-driven teardown."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.startup_gate = FakeBrowserStartupGate()
        self.browser = RealLikeBrowser(self.events, self.startup_gate)
        self.playwright = RealLikePlaywright(
            self.browser,
            self.events,
            self.startup_gate,
        )
        self.start_calls = 0
