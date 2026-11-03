"""Bounded, PID-owned moderation child supervision, never inline execution.

An entry is reserved before Popen and released only after the execution owner,
child and both pipe helpers finish. A single reaper retries quarantined entries;
quarantine closes admission rather than accumulating detached work.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import struct

# Fixed isolated interpreter/file argv; no shell or content arguments.
import subprocess  # nosec B404
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.execution_models import (
    ExecutionBounds,
    ExecutionScope,
    FailureMetadata,
    WorkerRequest,
    WorkerResponse,
    request_input_chars,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import decode_response, encode_request

_ownership_lock = threading.Lock()
_owned_scopes: dict[int, int] = {}
_owned_total = 0
_quarantined_total = 0
_instances: weakref.WeakSet = weakref.WeakSet()
_fork_condition = threading.Condition()
_active_spawns = 0
_fork_waiters = 0


def _scope_is_owned(scope: ExecutionScope) -> bool:
    """Include injected runners when retaining an async orchestration lease."""
    with _ownership_lock:
        return id(scope) in _owned_scopes


@dataclass(eq=False)
class _Owned:
    scope: ExecutionScope
    process: subprocess.Popen | None = None
    helpers: list[threading.Thread] = field(default_factory=list)
    helper_done: list[threading.Event] = field(default_factory=list)
    frame: bytes | None = None
    reason: str | None = None
    io_done: threading.Event = field(default_factory=threading.Event)
    owner_done: bool = False
    quarantined: bool = False
    spawning: bool = True


class ModerationExecutionRunner:
    """Immediately admit at most four owned children with one shared deadline."""

    def __init__(
        self,
        bounds: ExecutionBounds,
        *,
        clock: Callable[[], float] = time.monotonic,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.bounds = bounds
        self._clock = clock
        self._process_factory = process_factory
        self._reset_pid()
        with _ownership_lock:
            _instances.add(self)

    def _reset_pid(self) -> None:
        # Never acquire an inherited lock: its owner may not exist after fork.
        if getattr(self, "_pid", os.getpid()) != os.getpid():
            for entry in self._entries:
                if entry.process is not None:
                    for stream in (entry.process.stdin, entry.process.stdout):
                        if stream is not None:
                            stream.close()
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._entries: set[_Owned] = set()
        self._closed = False
        self._wake = threading.Event()
        self._reaper: threading.Thread | None = None

    def _check_pid(self) -> None:
        if self._pid != os.getpid():
            self._reset_pid()

    def _check_scope(self, scope: ExecutionScope, deadline: float | None = None) -> None:
        if scope.cancelled.is_set():
            raise asyncio.CancelledError
        if self._clock() >= min(scope.deadline, deadline if deadline is not None else scope.deadline):
            raise ModerationExecutionError("deadline_exceeded")

    @contextmanager
    def _publishing_spawn(self, scope: ExecutionScope, deadline: float) -> Iterator[None]:
        global _active_spawns
        with _fork_condition:
            while _fork_waiters:
                self._check_scope(scope, deadline)
                _fork_condition.wait(0.01)
            self._check_scope(scope, deadline)
            _active_spawns += 1
        try:
            yield
        finally:
            with _fork_condition:
                _active_spawns -= 1
                _fork_condition.notify_all()

    def execute(self, request: WorkerRequest, scope: ExecutionScope) -> WorkerResponse:
        """Run one framed batch, retaining ownership through every failure race."""
        global _owned_total
        self._check_pid()
        deadline = min(scope.deadline, self._clock() + min(self.bounds.timeout_s, request.bounds.timeout_s))
        self._check_scope(scope, deadline)
        if threading.current_thread().daemon or multiprocessing.current_process().daemon:
            raise ModerationExecutionError("worker_unavailable")
        with self._lock:
            if self._closed:
                raise ModerationExecutionError("worker_unavailable")
            if len(self._entries) >= self.bounds.max_workers or any(e.quarantined for e in self._entries):
                raise ModerationExecutionError("capacity_exhausted")
            entry = _Owned(scope)
            with _ownership_lock:
                if _owned_total >= 4 or _quarantined_total:
                    raise ModerationExecutionError("capacity_exhausted")
                _owned_total += 1
                _owned_scopes[id(scope)] = _owned_scopes.get(id(scope), 0) + 1
            self._entries.add(entry)
        try:
            # Runner ceilings cannot be widened by a supplied request.
            for name, value in vars(self.bounds).items():
                if name not in {"max_workers", "timeout_s"} and getattr(request.bounds, name) > value:
                    raise ModerationExecutionError("size_exceeded")
            frame = encode_request(request)
            self._check_scope(scope, deadline)
            entry_path = Path(__file__).with_name("regex_worker.py").resolve()
            environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
            try:
                # Fork preparation waits through native creation AND publication.
                # Concurrent admitted spawns remain bounded but are not serialized.
                with self._publishing_spawn(scope, deadline):
                    entry.process = self._process_factory(
                        [sys.executable, "-I", "-S", str(entry_path)],
                        shell=False,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                        env=environment,
                        bufsize=0,
                    )
            except (OSError, ValueError, subprocess.SubprocessError):
                raise ModerationExecutionError("worker_unavailable") from None
            finally:
                entry.spawning = False
            self._check_scope(scope, deadline)
            with self._lock:
                if self._closed:
                    raise ModerationExecutionError("worker_unavailable")
            self._start_io(entry, frame, request.bounds.max_response_bytes)
            while True:
                self._check_scope(scope, deadline)
                with self._lock:
                    if self._closed:
                        raise ModerationExecutionError("worker_unavailable")
                code = entry.process.poll()
                if code is not None and code != 0:
                    raise ModerationExecutionError("worker_crashed")
                if entry.reason:
                    raise ModerationExecutionError(entry.reason)
                if code == 0 and entry.io_done.is_set() and all(not t.is_alive() for t in entry.helpers):
                    response = decode_response(entry.frame or b"", request)
                    if response.reason:
                        raise ModerationExecutionError(response.reason)
                    posture = response.resource_limits
                    required = (
                        ("address_space", "cpu", "core")
                        if sys.platform.startswith("linux")
                        else (("cpu", "core") if os.name == "posix" else ())
                    )
                    if posture is None or any(not posture[key] for key in required):
                        raise ModerationExecutionError("worker_unavailable")
                    self._check_scope(scope, deadline)
                    return response
                scope.cancelled.wait(min(0.01, max(0, deadline - self._clock())))
        finally:
            # Also covers BaseException/cancellation during process creation.
            entry.spawning = False
            self._kill(entry)
            end = time.monotonic() + 0.5
            while not self._settled(entry) and time.monotonic() < end:
                time.sleep(0.005)
            with self._lock:
                entry.owner_done = True
                if self._settled(entry):
                    self._release_locked(entry)
                else:
                    entry.quarantined = True
                    self._mark_quarantined()
                    self._ensure_reaper_locked()
            self._wake.set()
            if entry.quarantined:
                # The original error/cancellation may never reach a caller reporter.
                from tldw_Server_API.app.core.Moderation.execution_incidents import report_execution_failure_background

                report_execution_failure_background(
                    ModerationExecutionError("cleanup_failed"),
                    metadata=FailureMetadata(request.operation, request.phase, request_input_chars(request)),
                    scope=scope,
                )

    def _release_locked(self, entry: _Owned) -> None:
        global _owned_total, _quarantined_total
        self._entries.remove(entry)
        with _ownership_lock:
            _owned_total -= 1
            if entry.quarantined:
                _quarantined_total -= 1
            count = _owned_scopes[id(entry.scope)] - 1
            if count:
                _owned_scopes[id(entry.scope)] = count
            else:
                del _owned_scopes[id(entry.scope)]

    @staticmethod
    def _mark_quarantined() -> None:
        global _quarantined_total
        with _ownership_lock:
            _quarantined_total += 1

    def _start_io(self, entry: _Owned, frame: bytes, cap: int) -> None:
        def write() -> None:
            try:
                view = memoryview(frame)
                while view:
                    written = entry.process.stdin.write(view[:65536])
                    if not written:
                        raise OSError
                    view = view[written:]
            except (OSError, ValueError):
                entry.reason = "worker_crashed"
            finally:
                entry.process.stdin.close()

        def read() -> None:
            try:
                stream = entry.process.stdout

                def exact(count: int) -> bytes:
                    parts = bytearray()
                    while len(parts) < count:
                        part = stream.read(min(65536, count - len(parts)))
                        if not part:
                            raise ModerationExecutionError("invalid_response")
                        parts.extend(part)
                    return bytes(parts)

                header = exact(4)
                length = struct.unpack(">I", header)[0]
                if length + 4 > cap:
                    raise ModerationExecutionError("size_exceeded")
                payload = exact(length)
                if stream.read(1):
                    raise ModerationExecutionError("invalid_response")
                entry.frame = header + payload
            except ModerationExecutionError as error:
                entry.reason = error.reason
            except (OSError, ValueError):
                entry.reason = "invalid_response"
            finally:
                entry.process.stdout.close()
                entry.io_done.set()

        for target, name in ((write, "moderation-write"), (read, "moderation-read")):
            done = threading.Event()

            def owned_target(target=target, done=done) -> None:
                try:
                    target()
                except Exception:  # noqa: BLE001 - never log pipe exceptions or child data.
                    entry.reason = "invalid_response"
                finally:
                    done.set()

            helper = threading.Thread(target=owned_target, name=name, daemon=True)
            # Register before native thread creation too. An interruption of
            # Thread.start cannot turn an eventual reader into detached work.
            entry.helpers.append(helper)
            entry.helper_done.append(done)
            try:
                helper.start()
            except (RuntimeError, OSError):
                done.set()
                raise ModerationExecutionError("worker_unavailable") from None

    @staticmethod
    def _kill(entry: _Owned) -> None:
        if entry.process is not None:
            try:
                entry.process.kill()
            except OSError:
                # ESRCH is normal; poll still must confirm reaping.
                pass

    @staticmethod
    def _settled(entry: _Owned) -> bool:
        if entry.spawning:
            return False
        if entry.process is not None and entry.process.poll() is None:
            return False
        if any(not done.is_set() for done in entry.helper_done):
            return False
        if any(helper.is_alive() for helper in entry.helpers):
            return False
        if entry.process is not None:
            for stream in (entry.process.stdin, entry.process.stdout):
                if stream is not None and not stream.closed:
                    stream.close()
        return True

    def _ensure_reaper_locked(self) -> None:
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper = threading.Thread(target=self._reap, name="moderation-reaper", daemon=True)
            try:
                self._reaper.start()
            except (RuntimeError, OSError):
                # Preserve the original deadline/cancellation and all leases.
                # close() can retry reaping without allocating another thread.
                self._closed = True
                self._reaper = None

    def _reap(self) -> None:
        while True:
            with self._lock:
                entries = [e for e in self._entries if e.owner_done]
                for entry in entries:
                    self._kill(entry)
                    if self._settled(entry):
                        self._release_locked(entry)
                if not any(e.owner_done for e in self._entries):
                    self._reaper = None
                    return
            self._wake.wait(0.05)
            self._wake.clear()

    def close(self, *, timeout_s: float = 0.5) -> None:
        """Reject admission immediately, kill registered children and boundedly reap."""
        self._check_pid()
        end = time.monotonic() + min(0.5, max(0, timeout_s))
        with self._lock:
            self._closed = True
            for entry in list(self._entries):
                # Do not mark scope cancelled: shutdown is an engine failure,
                # while caller cancellation must keep CancelledError semantics.
                self._kill(entry)
                if entry.owner_done and self._settled(entry):
                    self._release_locked(entry)
        self._wake.set()
        while time.monotonic() < end:
            with self._lock:
                for entry in list(self._entries):
                    if entry.owner_done and self._settled(entry):
                        self._release_locked(entry)
                if not self._entries:
                    return
            time.sleep(0.005)


_runner: ModerationExecutionRunner | None = None
_singleton_lock = threading.Lock()


def _before_fork() -> None:
    global _fork_waiters
    _fork_condition.acquire()
    _fork_waiters += 1
    while _active_spawns:
        _fork_condition.wait()
    # Held across the actual fork: no new descriptor creation can slip in.


def _after_fork_parent() -> None:
    global _fork_waiters
    _fork_waiters -= 1
    _fork_condition.notify_all()
    _fork_condition.release()


def _after_fork() -> None:
    global _runner, _singleton_lock, _ownership_lock, _owned_scopes, _owned_total, _quarantined_total
    global _fork_condition, _active_spawns, _fork_waiters
    # Never reuse/release locks inherited from a vanished publication owner.
    _fork_condition, _active_spawns, _fork_waiters = threading.Condition(), 0, 0
    _runner = None
    _singleton_lock = threading.Lock()
    _ownership_lock, _owned_scopes, _owned_total = threading.Lock(), {}, 0
    _quarantined_total = 0
    for runner in list(_instances):
        runner._reset_pid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork)


def get_execution_runner() -> ModerationExecutionRunner:
    """Return the lazy PID-owned singleton without launching any process."""
    global _runner
    with _singleton_lock:
        if _runner is None:
            _runner = ModerationExecutionRunner(ExecutionBounds())
        return _runner
