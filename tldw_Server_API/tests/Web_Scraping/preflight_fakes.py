"""Deterministic fakes for governed preflight contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any


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
        force_releases_close: bool = True,
        close_error: BaseException | None = None,
        events: list[str] | None = None,
        name: str = "cleanup",
    ) -> None:
        self.block_close = block_close
        self.suppress_close_cancellation = suppress_close_cancellation
        self.block_force_close = block_force_close
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
                await self._release_force_close.wait()
        except asyncio.CancelledError:
            self.force_close_cancellations += 1
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
