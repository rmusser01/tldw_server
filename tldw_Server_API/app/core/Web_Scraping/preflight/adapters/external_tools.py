"""Governed async adapter for optional external preflight tools."""

from __future__ import annotations

import asyncio
import shutil
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from loguru import logger

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
from tldw_Server_API.app.core.Web_Scraping.runtime.policy import ProbeEgressGuard

_WAF_TIMEOUT_SECONDS = 60.0
_PROCESS_CLEANUP_GRACE_SECONDS = 2.0
_LEGACY_DEFAULT_WARNING = "Preflight external tool used because its config key is absent"
_LEGACY_DEFAULT_METRIC = "web_scraping_preflight_legacy_external_tool_default_total"


def _run_best_effort(callback: Callable[[], Any]) -> None:
    try:
        callback()
    except Exception:  # noqa: BLE001 - observer failures are deliberately secondary
        return


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

        _run_best_effort(lambda: self._warning(_LEGACY_DEFAULT_WARNING))
        _run_best_effort(self._increment_legacy_default_metric)

    def _increment_legacy_default_metric(self) -> None:
        increment_counter = self._increment_counter
        if increment_counter is None:
            from tldw_Server_API.app.core.Metrics import increment_counter

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
    except asyncio.CancelledError:
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
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - policy failures fail closed
        raise ProbeError(
            "policy_error",
            "Probe destination was denied.",
        ) from None


def _consume_wait_task(task: asyncio.Task[int]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _shield_process_wait(task: asyncio.Task[int]) -> None:
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
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
    ) -> Any:
        timeout_s = self._controls.cap_timeout(None)
        argv = (executable, url, *(("-a",) if find_all else ()))
        try:
            if timeout_s is None:
                return await self._process_factory(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            async with asyncio.timeout(timeout_s):
                return await self._process_factory(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if self._controls.deadline_exhausted():
                raise PreflightDeadlineExceeded() from None
            raise ProbeError("probe_error", "Probe failed.") from None
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
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - cleanup cannot replace the outcome
            logger.warning("External tool process cleanup failed.")

    def _observe_legacy_default(self) -> None:
        _run_best_effort(self._legacy_default_observer.observe)

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

        try:
            executable = self._which("wafw00f")
        except Exception:  # noqa: BLE001 - dependency inspection is optional
            executable = None
        if not executable:
            raise ProbeUnavailable(error_code="missing_dependency")

        await self._controls.reserve("active_probe")
        decision = await _fresh_decision(
            self._egress_guard,
            url,
            controls=self._controls,
        )
        allowed, reason = _decision_fields(decision)
        if not allowed:
            raise _denied_error(reason)
        self._controls.cap_timeout(None)

        process = await self._create_process(
            executable,
            url,
            find_all=find_all,
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
            timeout_s = self._controls.cap_timeout(_WAF_TIMEOUT_SECONDS)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_s,
            )
            self._controls.cap_timeout(None)
            result = ExternalToolResult(
                returncode=process.returncode,
                stdout=_decode_output(stdout),
                stderr=_decode_output(stderr),
            )
        except asyncio.CancelledError:
            await self._cleanup_process(handle)
            raise
        except TimeoutError:
            deadline_exhausted = self._controls.deadline_exhausted()
            await self._cleanup_process(handle)
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


__all__ = ["GuardedExternalToolProbe"]
