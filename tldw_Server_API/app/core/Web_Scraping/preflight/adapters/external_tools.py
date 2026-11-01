"""Governed async adapter for optional external preflight tools."""

from __future__ import annotations

import asyncio
import shutil
import threading
from collections.abc import Callable
from dataclasses import replace
from time import monotonic as _monotonic
from typing import Any, cast

from loguru import logger

from tldw_Server_API.app.core.Security.egress import public_url_policy_active
from tldw_Server_API.app.core.Web_Scraping.preflight.asyncio_compat import timeout as _asyncio_timeout
from tldw_Server_API.app.core.Web_Scraping.preflight.context import (
    PreflightDeadlineExceeded,
    PreflightRuntimeControls,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import (
    ExternalToolResult,
    ProbeError,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressGuard

from ...runtime.controls import observe_physical

_WAF_TIMEOUT_SECONDS = 60.0
_PROCESS_CLEANUP_GRACE_SECONDS = 2.0
_LEGACY_DEFAULT_WARNING = "Preflight external tool used because its config key is absent"
_LEGACY_DEFAULT_METRIC = "web_scraping_preflight_legacy_external_tool_default_total"
_OBSERVER_FAILURE_MESSAGES = {
    "legacy_default_warning": "Preflight external-tool observer failed: legacy_default_warning.",
    "legacy_default_metric": "Preflight external-tool observer failed: legacy_default_metric.",
    "legacy_default_observer": "Preflight external-tool observer failed: legacy_default_observer.",
}


def _run_best_effort(callback: Callable[[], Any], *, label: str) -> None:
    try:
        callback()
    except Exception:  # noqa: BLE001 - observer failures are deliberately secondary
        logger.warning(_OBSERVER_FAILURE_MESSAGES[label])


class _LegacyExternalToolDefaultObserver:
    """Emit the absent-config transition signal at most once per process."""

    def __init__(
        self,
        *,
        warning: Callable[[str], Any] = logger.warning,
        increment_counter: Callable[..., Any] | None = None,
    ) -> None:
        self._warning = warning
        self._increment_counter = increment_counter
        self._lock = threading.Lock()
        self._observed = False

    def observe(self) -> None:
        with self._lock:
            if self._observed:
                return
            self._observed = True

        _run_best_effort(
            lambda: self._warning(_LEGACY_DEFAULT_WARNING),
            label="legacy_default_warning",
        )
        _run_best_effort(
            self._increment_legacy_default_metric,
            label="legacy_default_metric",
        )

    def _increment_legacy_default_metric(self) -> None:
        increment_counter = self._increment_counter
        if increment_counter is None:
            from tldw_Server_API.app.core.Metrics import increment_counter as default_increment_counter

            increment_counter = default_increment_counter

        increment_counter(
            _LEGACY_DEFAULT_METRIC,
            labels={"tool": "wafw00f"},
        )


_LEGACY_DEFAULT_OBSERVER = _LegacyExternalToolDefaultObserver()


def _subrequest_context(controls: PreflightRuntimeControls) -> Any:
    return replace(controls.request_context, stage="preflight_subrequest")


def _denied_error(reason: str) -> ProbeError:
    code = "policy_error" if reason == "policy_error" else "policy_denied"
    return ProbeError(code, "Probe destination was denied.")


def _decision_fields(decision: Any) -> tuple[bool, str]:
    try:
        return bool(decision.allowed), str(decision.reason)
    except (asyncio.CancelledError, ParentStop):
        raise
    except Exception:  # noqa: BLE001 - policy values must fail closed
        raise ProbeError(
            "policy_error",
            "Probe destination was denied.",
        ) from None


async def _fresh_decision(
    guard: ProbeEgressGuard,
    url: str,
    *,
    controls: PreflightRuntimeControls,
) -> Any:
    try:
        return await guard.decide(
            url,
            context=_subrequest_context(controls),
        )
    except (asyncio.CancelledError, ParentStop):
        raise
    except Exception:  # noqa: BLE001 - policy failures fail closed
        raise ProbeError(
            "policy_error",
            "Probe destination was denied.",
        ) from None


def _consume_wait_task(task: asyncio.Task[int]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, ParentStop):
        pass


async def _shield_process_wait(task: asyncio.Task[int]) -> None:
    try:
        await asyncio.shield(task)
    except (asyncio.CancelledError, ParentStop):
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise


class _ProcessCleanupHandle:
    """Coordinate one terminate, kill, and wait sequence across cleanup races."""

    def __init__(self, process: Any) -> None:
        self._process = process
        self._state_lock = asyncio.Lock()
        self._terminate_started = False
        self._kill_started = False
        self._wait_task: asyncio.Task[int] | None = None

    def _wait_task_locked(self) -> asyncio.Task[int]:
        if self._wait_task is None:
            self._wait_task = asyncio.create_task(
                self._process.wait(),
                name="preflight-external-tool-wait",
            )
            self._wait_task.add_done_callback(_consume_wait_task)
            observe_physical(self._wait_task)
        return self._wait_task

    async def close(self) -> None:
        async with self._state_lock:
            if self._process.returncode is not None:
                return
            if not self._terminate_started:
                self._terminate_started = True
                try:
                    self._process.terminate()
                except ProcessLookupError:
                    pass
            wait_task = self._wait_task_locked()
        await _shield_process_wait(wait_task)

    async def force_close(self) -> None:
        async with self._state_lock:
            if self._process.returncode is not None:
                return
            if not self._kill_started:
                self._kill_started = True
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            wait_task = self._wait_task_locked()
        await _shield_process_wait(wait_task)


def _decode_output(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


class GuardedExternalToolProbe:
    """Apply enablement, dependency, policy, budget, and process governance."""

    def __init__(
        self,
        *,
        controls: PreflightRuntimeControls,
        egress_guard: ProbeEgressGuard,
        which: Callable[[str], str | None] = shutil.which,
        process_factory: Callable[..., Any] = asyncio.create_subprocess_exec,
        legacy_default_observer: _LegacyExternalToolDefaultObserver = (_LEGACY_DEFAULT_OBSERVER),
    ) -> None:
        self._controls = controls
        self._egress_guard = egress_guard
        self._which = which
        self._process_factory = process_factory
        self._legacy_default_observer = legacy_default_observer

    async def _create_process(
        self,
        executable: str,
        url: str,
        *,
        find_all: bool,
        timeout_s: float,
    ) -> Any:
        argv = (executable, url, *(("-a",) if find_all else ()))
        spawn_task: asyncio.Task[Any] | None = None

        async def spawn() -> Any:
            return await self._process_factory(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        def reclaim() -> None:
            if spawn_task is None:
                return

            async def cleanup_late_process() -> None:
                try:
                    process = await asyncio.shield(spawn_task)
                except (Exception, asyncio.CancelledError):  # noqa: BLE001 - spawn failure has no live process
                    return
                await self._cleanup_process(_ProcessCleanupHandle(process))

            cleanup_task = asyncio.create_task(cleanup_late_process(), name="preflight-late-spawn-cleanup")
            observe_physical(cleanup_task)
            cleanup_task.add_done_callback(lambda done: None if done.cancelled() else done.exception())

        try:
            async with _asyncio_timeout(timeout_s):
                if self._controls.parent is None:
                    return await spawn()
                spawn_task = asyncio.create_task(spawn(), name="preflight-owned-tool-spawn")
                observe_physical(spawn_task)
                return await asyncio.shield(spawn_task)
        except (asyncio.CancelledError, ParentStop):
            reclaim()
            raise
        except (TimeoutError, asyncio.TimeoutError):
            reclaim()
            self._controls.cap_timeout(None)
            if self._controls.deadline_exhausted():
                raise PreflightDeadlineExceeded() from None
            raise ProbeTimeout() from None
        except (ProbeError, PreflightDeadlineExceeded):
            raise
        except Exception:  # noqa: BLE001 - sanitize the process boundary
            raise ProbeError("probe_error", "Probe failed.") from None

    async def _cleanup_process(self, handle: _ProcessCleanupHandle) -> None:
        try:
            await self._controls.cleanup_handles(
                (handle,),
                grace_s=_PROCESS_CLEANUP_GRACE_SECONDS,
            )
        except (asyncio.CancelledError, ParentStop):
            raise
        except Exception:  # noqa: BLE001 - cleanup cannot replace the outcome
            logger.warning("External tool process cleanup failed.")

    def _observe_legacy_default(self) -> None:
        _run_best_effort(
            self._legacy_default_observer.observe,
            label="legacy_default_observer",
        )

    async def run_waf(
        self,
        url: str,
        *,
        find_all: bool,
        enabled: bool | None,
    ) -> ExternalToolResult:
        if enabled is False:
            raise ProbeError(
                "external_tool_disabled",
                "External tool probing is disabled.",
            )

        # This subprocess API has no pinned DNS, credential isolation, or body bound.
        if public_url_policy_active():
            raise ProbeUnavailable()
        try:
            executable = self._which("wafw00f")
        except Exception:  # noqa: BLE001 - dependency inspection is optional
            executable = None
        if not executable:
            raise ProbeUnavailable(error_code="missing_dependency")

        if self._controls.parent is None:
            await self._controls.reserve("active_probe")
        decision = await _fresh_decision(
            self._egress_guard,
            url,
            controls=self._controls,
        )
        allowed, reason = _decision_fields(decision)
        if not allowed:
            raise _denied_error(reason)
        effective_timeout_s = cast(
            float,
            self._controls.cap_timeout(_WAF_TIMEOUT_SECONDS),
        )
        operation_deadline = _monotonic() + effective_timeout_s

        if self._controls.parent is not None:
            await self._controls.reserve("active_probe")

        process = await self._create_process(
            executable,
            url,
            find_all=find_all,
            timeout_s=effective_timeout_s,
        )
        handle = _ProcessCleanupHandle(process)
        try:
            self._controls.register_cleanup(handle)
        except Exception:  # noqa: BLE001 - created processes remain owned
            await self._cleanup_process(handle)
            raise ProbeError("probe_error", "Probe failed.") from None

        try:
            if enabled is None:
                self._observe_legacy_default()
            timeout_s = operation_deadline - _monotonic()
            if timeout_s <= 0:
                if self._controls.deadline_exhausted():
                    raise PreflightDeadlineExceeded()
                raise ProbeTimeout()
            stdout, stderr = await asyncio.wait_for(
                self._read_output(process) if self._controls.parent is not None else process.communicate(),
                timeout=timeout_s,
            )
            if _monotonic() >= operation_deadline:
                if self._controls.deadline_exhausted():
                    raise PreflightDeadlineExceeded()
                raise ProbeTimeout()
            self._controls.cap_timeout(None)
            result = ExternalToolResult(
                returncode=process.returncode,
                stdout=_decode_output(stdout),
                stderr=_decode_output(stderr),
            )
        except (asyncio.CancelledError, ParentStop):
            await self._cleanup_process(handle)
            raise
        except (TimeoutError, asyncio.TimeoutError):
            deadline_exhausted = self._controls.deadline_exhausted()
            await self._cleanup_process(handle)
            self._controls.cap_timeout(None)
            if deadline_exhausted:
                raise PreflightDeadlineExceeded() from None
            raise ProbeTimeout() from None
        except (ProbeError, PreflightDeadlineExceeded):
            await self._cleanup_process(handle)
            raise
        except Exception:  # noqa: BLE001 - captured process details stay private
            await self._cleanup_process(handle)
            raise ProbeError("probe_error", "Probe failed.") from None

        await self._cleanup_process(handle)
        return result

    async def _read_output(self, process: Any) -> tuple[bytes, bytes]:
        """Bound opaque tool output, not its hidden HTTP requests or wire bytes."""
        total = 0

        async def read(stream: Any) -> bytes:
            nonlocal total
            body = bytearray()
            while True:
                self._controls.cap_timeout(None)
                chunk = await stream.read(65_536)
                if not chunk:
                    return bytes(body)
                self._controls.charge_bytes(len(chunk))
                total += len(chunk)
                if total > 16_777_216:
                    raise ProbeError("probe_error", "Probe output exceeds limit.")
                body.extend(chunk)

        tasks = [asyncio.create_task(read(stream)) for stream in (process.stdout, process.stderr)]
        for task in tasks:
            observe_physical(task)
        try:
            stdout, stderr = await asyncio.gather(*tasks)
            await process.wait()
            return stdout, stderr
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["GuardedExternalToolProbe"]
