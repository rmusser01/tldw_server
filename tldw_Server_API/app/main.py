# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
import asyncio
import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

#
# Local Imports
#
# Early logging configuration to keep startup output consistent
import os as _early_os
import os as _env_os

#
# 3rd-party Libraries
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette import status as _starlette_status
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from tldw_Server_API.app.core.startup_logging import (
    startup_api_key_log_value as _startup_api_key_log_value,
)
from tldw_Server_API.app.api.v1.router_registry import include_router_idempotent
from tldw_Server_API.app.services.app_lifecycle import (
    mark_lifecycle_shutdown,
    mark_lifecycle_startup,
    get_or_create_lifecycle_state,
)
from tldw_Server_API.app.core.testing import (
    env_flag_enabled as _shared_env_flag_enabled,
)
from tldw_Server_API.app.core.testing import (
    is_explicit_pytest_runtime as _shared_is_explicit_pytest_runtime,
)
from tldw_Server_API.app.core.testing import (
    is_truthy as _shared_is_truthy,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    get_user_media_db_path,
)
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    ensure_chacha_rls,
    ensure_prompt_studio_rls,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    managed_media_database,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
    list_claims_rebuild_media_ids,
)

# Backward-compat for Starlette variants that expose 413 as
# HTTP_413_REQUEST_ENTITY_TOO_LARGE instead of HTTP_413_CONTENT_TOO_LARGE.
if not hasattr(_starlette_status, "HTTP_413_CONTENT_TOO_LARGE"):
    setattr(
        _starlette_status,
        "HTTP_413_CONTENT_TOO_LARGE",
        getattr(_starlette_status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413),
    )

_LOGGING_SETUP_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_TRACE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_IMPORT_EXCEPTIONS = (
    AssertionError,
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_IO_EXCEPTIONS = (
    OSError,
    ValueError,
    AttributeError,
)
_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_REQUEST_GUARD_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_READINESS_GUARD_EXCEPTIONS = _REQUEST_GUARD_EXCEPTIONS + (
    ImportError,
    ModuleNotFoundError,
)


@contextmanager
def _claims_rebuild_db_session(
    app_settings: Mapping[str, Any],
) -> Iterator[tuple[int, str, Any]]:
    """Yield one managed Media DB session for the claims rebuild worker loop."""
    user_id = int(app_settings.get("SINGLE_USER_FIXED_ID", "1"))
    db_path = str(get_user_media_db_path(user_id))
    client_id = str(app_settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"))
    with managed_media_database(
        client_id=client_id,
        db_path=db_path,
        initialize=False,
    ) as db:
        yield user_id, db_path, db


def _run_pg_rls_auto_ensure(backend: Any) -> tuple[bool, bool]:
    """Apply both PostgreSQL RLS installers and log the combined result."""
    prompt_ok = ensure_prompt_studio_rls(backend)
    chacha_ok = ensure_chacha_rls(backend)
    logger.info(
        "PG RLS ensure invoked (prompt_studio_applied={}, chacha_applied={})",
        prompt_ok,
        chacha_ok,
    )
    return prompt_ok, chacha_ok


def _apply_shutdown_transition_gate(app: FastAPI, readiness_state: Any | None) -> None:
    """Move the app into draining mode and gate new jobs."""
    try:
        lifecycle_state = get_or_create_lifecycle_state(app)
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        lifecycle_state = None
        logger.debug(f"Shutdown transition gate: lifecycle state lookup skipped: {exc}")

    try:
        if lifecycle_state is None or lifecycle_state.phase != "draining" or not lifecycle_state.draining:
            mark_lifecycle_shutdown(app, readiness_state)
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Shutdown transition gate: failed to mark lifecycle shutdown: {exc}")

    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JM

        _JM.set_acquire_gate(True)
    except _IMPORT_EXCEPTIONS as exc:
        logger.debug(f"Shutdown transition gate: job acquire gate unavailable: {exc}")


def _build_legacy_shutdown_context(
    *,
    readiness_state: Any | None,
    usage_task: Any = None,
    llm_usage_task: Any = None,
    authnz_scheduler_started: bool = False,
    chatbooks_cleanup_task: Any = None,
    chatbooks_cleanup_stop_event: Any = None,
    storage_cleanup_service: Any = None,
) -> "LegacyShutdownContext":
    """Collect the explicit shutdown dependencies used by legacy adapters."""
    from tldw_Server_API.app.services.shutdown_legacy_adapters import LegacyShutdownContext

    return LegacyShutdownContext(
        readiness_state=readiness_state,
        usage_task=usage_task,
        llm_usage_task=llm_usage_task,
        authnz_scheduler_started=authnz_scheduler_started,
        chatbooks_cleanup_task=chatbooks_cleanup_task,
        chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
        storage_cleanup_service=storage_cleanup_service,
    )


def _build_coordinated_shutdown_coordinator(
    app: FastAPI,
    legacy_shutdown_plan: list[Any],
    *,
    transport_registry: Any | None = None,
) -> tuple["ShutdownCoordinator", list["ShutdownComponent"], list["ShutdownComponent"]]:
    """Assemble the production drain coordinator with legacy and transport owners."""
    from tldw_Server_API.app.services.shutdown_coordinator import ShutdownCoordinator
    from tldw_Server_API.app.services.shutdown_transport_registry import (
        build_shutdown_components,
    )

    coordinator = ShutdownCoordinator(profile="prod_drain")
    legacy_components: list[Any] = []
    try:
        from tldw_Server_API.app.services.shutdown_legacy_adapters import (
            register_legacy_shutdown_components,
        )

        legacy_components = register_legacy_shutdown_components(
            coordinator,
            legacy_shutdown_plan,
        )
    except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS):
        legacy_components = []
    transport_components = build_shutdown_components(transport_registry)
    for component in transport_components:
        coordinator.register(component)

    try:
        app.state._tldw_shutdown_transport_component_names = [
            component.name for component in transport_components
        ]
    except _STARTUP_GUARD_EXCEPTIONS:
        pass

    return coordinator, legacy_components, transport_components


async def _run_coordinated_shutdown(
    app: FastAPI,
    legacy_shutdown_plan: list[Any],
    *,
    transport_registry: Any | None = None,
) -> set[str]:
    """Run the coordinated shutdown slice used by the real lifespan teardown."""
    try:
        from tldw_Server_API.app.services.shutdown_legacy_adapters import (
            get_legacy_shutdown_suppressed_component_names,
        )
    except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS):
        get_legacy_shutdown_suppressed_component_names = lambda _summary: set()

    (
        coordinated_legacy_coordinator,
        coordinated_legacy_components,
        coordinated_transport_components,
    ) = _build_coordinated_shutdown_coordinator(
        app,
        legacy_shutdown_plan,
        transport_registry=transport_registry,
    )
    all_components = list(coordinated_legacy_components) + list(coordinated_transport_components)
    if not all_components:
        return set()

    coordinated_legacy_summary = await coordinated_legacy_coordinator.shutdown()
    legacy_component_name_set = {component.name for component in coordinated_legacy_components}
    coordinated_legacy_component_names = {
        name
        for name in get_legacy_shutdown_suppressed_component_names(coordinated_legacy_summary)
        if name in legacy_component_name_set
    }
    try:
        app.state._tldw_shutdown_legacy_coordinator_summary = coordinated_legacy_summary
        app.state._tldw_shutdown_legacy_coordinator_component_names = [
            component.name for component in all_components
        ]
        app.state._tldw_shutdown_legacy_coordinator_phase_groups = {
            phase.value: phase_summary.component_names
            for phase, phase_summary in coordinated_legacy_summary.phases.items()
        }
    except _STARTUP_GUARD_EXCEPTIONS:
        pass
    logger.info(
        "App Shutdown: legacy coordinator summary components={} phase_groups={} wall_time_ms={}",
        [component.name for component in all_components],
        {
            phase.value: phase_summary.component_names
            for phase, phase_summary in coordinated_legacy_summary.phases.items()
        },
        coordinated_legacy_summary.wall_time_ms,
    )
    return coordinated_legacy_component_names


@dataclass
class _ManagedJobPoller:
    """Shutdown-owned in-process job poller handle."""

    name: str
    task: asyncio.Task[Any]
    stop_event: asyncio.Event | None = None
    timeout_sec: float = 5.0


def _publish_shutdown_job_poller_inventory(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
) -> None:
    """Expose shutdown-owned job poller metadata on app.state."""
    inventory = [
        {
            "name": handle.name,
            "task_name": handle.task.get_name(),
            "has_stop_event": handle.stop_event is not None,
            "timeout_sec": handle.timeout_sec,
        }
        for handle in handles
    ]
    try:
        app.state._tldw_shutdown_job_poller_inventory = inventory
    except _STARTUP_GUARD_EXCEPTIONS:
        pass


def _register_owned_job_poller(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
    *,
    name: str,
    task: asyncio.Task[Any] | None,
    stop_event: asyncio.Event | None = None,
    timeout_sec: float = 5.0,
) -> None:
    """Register one shutdown-owned job poller and refresh app-state inventory."""
    if task is None:
        return
    handles.append(
        _ManagedJobPoller(
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
        )
    )
    _publish_shutdown_job_poller_inventory(app, handles)

def _replace_owned_job_poller_inventory(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
    *,
    registrations: list[tuple[str, asyncio.Task[Any] | None, asyncio.Event | None, float]],
) -> None:
    """Replace the managed job-poller inventory with the current owned poller set."""
    handles.clear()
    _publish_shutdown_job_poller_inventory(app, handles)
    for name, task, stop_event, timeout_sec in registrations:
        _register_owned_job_poller(
            app,
            handles,
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
        )
def _record_shutdown_timing_segment(
    app: FastAPI,
    segment: str,
    duration_ms: int,
    **extra: object,
) -> None:
    """Store one shutdown timing segment and emit a consistent log line."""
    payload = {"segment": segment, "duration_ms": max(int(duration_ms), 0), **extra}
    segments = getattr(app.state, "_tldw_shutdown_timing_segments", None)
    if not isinstance(segments, list):
        segments = []
        try:
            app.state._tldw_shutdown_timing_segments = segments
        except _STARTUP_GUARD_EXCEPTIONS:
            return
    segments.append(payload)
    extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
    if extra_text:
        logger.info(f"App Shutdown Timing: segment={segment} duration_ms={payload['duration_ms']} {extra_text}")
    else:
        logger.info(f"App Shutdown Timing: segment={segment} duration_ms={payload['duration_ms']}")


@contextmanager
def _timed_shutdown_segment(
    app: FastAPI,
    segment: str,
    **extra: object,
) -> Iterator[None]:
    """Measure a shutdown block with monotonic time and record it on app.state."""
    started = time.monotonic()
    try:
        yield
    finally:
        duration_ms = int((time.monotonic() - started) * 1000)
        _record_shutdown_timing_segment(app, segment, duration_ms, **extra)


def _record_shutdown_timing_total(app: FastAPI, duration_ms: int) -> None:
    """Record total teardown time and summarize the slowest non-total segment."""
    segments = getattr(app.state, "_tldw_shutdown_timing_segments", [])
    non_total_segments = [
        entry
        for entry in segments
        if isinstance(entry, dict) and entry.get("segment") != "total app teardown"
    ]
    if non_total_segments:
        slowest = max(non_total_segments, key=lambda entry: int(entry.get("duration_ms", 0)))
        slowest_segment = str(slowest.get("segment", ""))
        slowest_duration_ms = int(slowest.get("duration_ms", 0))
    else:
        slowest_segment = "total app teardown"
        slowest_duration_ms = max(int(duration_ms), 0)
    _record_shutdown_timing_segment(app, "total app teardown", duration_ms)
    summary = {
        "duration_ms": max(int(duration_ms), 0),
        "slowest_segment": slowest_segment,
        "slowest_duration_ms": slowest_duration_ms,
    }
    try:
        app.state._tldw_shutdown_timing_total = summary
    except _STARTUP_GUARD_EXCEPTIONS:
        pass
    logger.info(
        "App Shutdown Timing: total duration_ms={} slowest_segment={} slowest_duration_ms={}",
        summary["duration_ms"],
        summary["slowest_segment"],
        summary["slowest_duration_ms"],
    )


async def _stop_registered_job_pollers(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
) -> None:
    """Stop registered job pollers, preferring explicit stop events."""
    async def _await_job_poller_shutdown(handle: _ManagedJobPoller) -> bool:
        try:
            await asyncio.wait_for(asyncio.shield(handle.task), timeout=handle.timeout_sec)
        except asyncio.CancelledError:
            return bool(handle.task.done())
        except asyncio.TimeoutError:
            logger.warning(
                "App Shutdown: Timed out waiting for job poller {} after {}s; cancelling",
                handle.name,
                handle.timeout_sec,
            )
            handle.task.cancel()
            try:
                await asyncio.wait_for(handle.task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "App Shutdown: Job poller {} did not cancel within 1.0s after timeout",
                    handle.name,
                )
            except Exception as exc:
                logger.warning(
                    "App Shutdown: Job poller {} raised after cancellation: {}",
                    handle.name,
                    exc,
                )
            except _STARTUP_GUARD_EXCEPTIONS as exc:
                logger.debug(
                    "App Shutdown: Job poller cancel guard triggered for {}: {}",
                    handle.name,
                    exc,
                )
        except _STARTUP_GUARD_EXCEPTIONS as exc:
            logger.debug(
                "App Shutdown: Job poller stop guard triggered for {}: {}",
                handle.name,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "App Shutdown: Job poller {} exited during shutdown: {}",
                handle.name,
                exc,
            )
        return bool(handle.task.done())

    for handle in handles:
        if handle.stop_event is not None:
            handle.stop_event.set()
        else:
            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                handle.task.cancel()

    quiesce_results = await asyncio.gather(
        *(_await_job_poller_shutdown(handle) for handle in handles),
        return_exceptions=False,
    )
    try:
        app.state._tldw_shutdown_quiesced_job_poller_names = [
            handle.name
            for handle, quiesced in zip(handles, quiesce_results)
            if quiesced
        ]
    except _STARTUP_GUARD_EXCEPTIONS:
        pass


async def _quiesce_owned_job_pollers_for_shutdown(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
    *,
    wait_for_leases_sec: int | float,
    count_active_processing: Any,
) -> None:
    """Optionally wait for active leases, then quiesce owned job pollers.

    This helper runs only after the shutdown transition handoff has enabled the
    Jobs acquire gate, so the bounded wait drains already-leased work rather
    than allowing new in-process pollers to claim fresh jobs.
    """
    lease_wait_started = time.monotonic()
    initial_active = 0
    if wait_for_leases_sec > 0 and handles:
        try:
            initial_active = max(int(count_active_processing()), 0)
        except _STARTUP_GUARD_EXCEPTIONS:
            initial_active = 0
        if initial_active > 0:
            deadline = time.monotonic() + float(wait_for_leases_sec)
            active = initial_active
            while active > 0:
                remaining_sec = deadline - time.monotonic()
                if remaining_sec <= 0:
                    break
                await asyncio.sleep(min(0.5, remaining_sec))
                try:
                    active = max(int(count_active_processing()), 0)
                except _STARTUP_GUARD_EXCEPTIONS:
                    active = 0
            duration_ms = int((time.monotonic() - lease_wait_started) * 1000)
            _record_shutdown_timing_segment(
                app,
                "optional_lease_wait",
                duration_ms,
                skipped=False,
                initial_active=initial_active,
                wait_for_leases_sec=float(wait_for_leases_sec),
            )
        else:
            _record_shutdown_timing_segment(
                app,
                "optional_lease_wait",
                0,
                skipped=True,
                initial_active=0,
                wait_for_leases_sec=float(wait_for_leases_sec),
            )
    else:
        _record_shutdown_timing_segment(
            app,
            "optional_lease_wait",
            0,
            skipped=True,
            initial_active=0,
            wait_for_leases_sec=float(wait_for_leases_sec),
        )

    with _timed_shutdown_segment(app, "job_poller_quiesce", poller_count=len(handles)):
        await _stop_registered_job_pollers(app, handles)

_early_os.environ.setdefault("MCP_INHERIT_GLOBAL_LOGGER", "1")
try:
    # Route warnings through stdlib logging so they inherit the Loguru format.
    logging.captureWarnings(True)
except _LOGGING_SETUP_EXCEPTIONS:
    logger.debug("Failed to enable warning capture via stdlib logging")


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk back through frames to skip logging/loguru internals
        frame, depth = logging.currentframe(), 2
        try:
            import os as _os

            _logging_file = _os.path.abspath(getattr(logging, "__file__", ""))
        except _LOGGING_SETUP_EXCEPTIONS:
            _logging_file = ""
        # Move at least one frame back (currentframe() points to this emit())
        if frame is not None:
            frame = frame.f_back
        while frame is not None:
            fname = getattr(frame.f_code, "co_filename", "")
            if _logging_file and _logging_file == fname:
                depth += 1
                frame = frame.f_back
                continue
            # Skip frames inside loguru internals as well
            if "loguru" in (fname or ""):
                depth += 1
                frame = frame.f_back
                continue
            break
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class _SafeExtra(dict):
    """A dict that returns empty string for missing keys.

    Prevents KeyError when format strings reference {extra[missing_key]}."""

    def __getitem__(self, key):  # type: ignore[override]
        try:
            return super().__getitem__(key)
        except KeyError:
            return ""


def _trace_log_patcher(record):
    try:
        from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager as _get_tm

        span = _get_tm().get_current_span()
        trace_id = span.get_span_context().trace_id if span else 0
        span_id = span.get_span_context().span_id if span else 0
        record.setdefault("extra", {})
        record["extra"].setdefault("trace_id", f"{trace_id:032x}" if trace_id else "")
        record["extra"].setdefault("span_id", f"{span_id:016x}" if span_id else "")
        if record["extra"].get("trace_id") and record["extra"].get("span_id"):
            record["extra"].setdefault(
                "traceparent", f"00-{record['extra']['trace_id']}-{record['extra']['span_id']}-01"
            )
        else:
            record["extra"].setdefault("traceparent", "")
        try:
            req = _get_tm().get_baggage("request_id")
            ses = _get_tm().get_baggage("session_id")
            if req:
                record["extra"].setdefault("request_id", req)
            if ses:
                record["extra"].setdefault("session_id", ses)
        except _TRACE_EXCEPTIONS:
            pass
    except _TRACE_EXCEPTIONS:
        record.setdefault("extra", {})
        record["extra"].setdefault("trace_id", "")
        record["extra"].setdefault("span_id", "")
        record["extra"].setdefault("traceparent", "")
        record["extra"].setdefault("request_id", "")
        record["extra"].setdefault("session_id", "")
    # Ensure commonly referenced extra keys exist to avoid formatter KeyErrors
    # and wrap with SafeExtra so any unknown keys resolve to an empty string.
    try:
        record["extra"].setdefault("event_id", "")
        record["extra"].setdefault("event_type", "")
        record["extra"].setdefault("category", "")
        record["extra"].setdefault("action", "")
        # Replace the mapping with a tolerant wrapper
        if not isinstance(record["extra"], _SafeExtra):
            record["extra"] = _SafeExtra(record["extra"])  # type: ignore[assignment]
    except _TRACE_EXCEPTIONS:
        # As a last resort, provide an empty tolerant mapping
        record["extra"] = _SafeExtra()
    try:
        import re as _re

        msg = record.get("message", "")
        msg = _re.sub(r"sk-[A-Za-z0-9-_]{8,}", "sk-***REDACTED***", msg)
        msg = _re.sub(r"(?i)(api[_-]?key|authorization|token|password)\s*[:=]\s*[^\s,;]+", r"\1=***REDACTED***", msg)
        record["message"] = msg
    except _TRACE_EXCEPTIONS:
        pass
    # Normalize extra values for JSON serialization and log safety
    try:
        from datetime import datetime as _dt

        extra = record.get("extra", {})
        if isinstance(extra, dict):
            for _k, _v in list(extra.items()):
                if isinstance(_v, _dt):
                    extra[_k] = _v.isoformat()
                elif isinstance(_v, (set, tuple)):
                    extra[_k] = list(_v)
    except _TRACE_EXCEPTIONS:
        pass


def _safe_log_format(record: dict) -> str:
    """
    Build a safe format template for Loguru which defers insertion of
    dynamic values (especially the message) to Loguru's own formatting.

    Returning a template with placeholders avoids embedding the raw message
    into the format string. This prevents Loguru's colorizer from parsing
    curly braces coming from messages (e.g., JSON dicts) which previously
    caused recursive parsing and "Max string recursion exceeded" errors.
    """
    # Note: Markup tags (<level>, <dim>, etc.) are parsed before placeholders
    # are formatted, so the inserted {message} content will not be re-parsed
    # for markup. This removes the need to strip '<' or '>' from messages.
    return (
        "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> | "
        "<level>{level: <8}</level> | "
        "<cyan>trace={extra[trace_id]}</cyan> <cyan>span={extra[span_id]}</cyan> "
        "<cyan>tp={extra[traceparent]}</cyan> "
        "<yellow>req={extra[request_id]}</yellow> <yellow>job={extra[job_id]}</yellow> "
        "<yellow>ps={extra[ps_component]}:{extra[ps_job_kind]}</yellow> | "
        "<blue>{name}</blue>:<magenta>{function}</magenta>:<cyan>{line}</cyan> - {message}{exception}"
    )


def _safe_debug(message: str) -> None:
    try:
        logger.debug(message)
    except _LOGGING_SETUP_EXCEPTIONS:
        with suppress(_IO_EXCEPTIONS):
            sys.__stderr__.write(message + "\n")


class _StderrInterceptor:
    """Intercept writes to stderr and route through Loguru."""

    def __init__(self, stream):
        self._stream = stream
        self._local = threading.local()

    def write(self, message: str) -> None:
        if message is None:
            return
        if getattr(self._local, "in_write", False):
            try:
                self._stream.write(message)
            except _IO_EXCEPTIONS as exc:
                _safe_debug(f"StderrInterceptor direct write failed: {exc}")
            return
        try:
            self._local.in_write = True
            buf = getattr(self._local, "buffer", "")
            buf += message
            lines = buf.splitlines(keepends=True)
            new_buf = ""
            for line in lines:
                if line.endswith("\n") or line.endswith("\r"):
                    text = line.rstrip("\r\n")
                    if text:
                        self._log_line(text)
                else:
                    new_buf += line
            self._local.buffer = new_buf
        finally:
            self._local.in_write = False

    def _log_line(self, text: str) -> None:
        level = "warning"
        msg = text
        for prefix, lvl in (
            ("WARNING:", "warning"),
            ("ERROR:", "error"),
            ("CRITICAL:", "critical"),
            ("INFO:", "info"),
            ("DEBUG:", "debug"),
        ):
            if text.startswith(prefix):
                msg = text[len(prefix):].lstrip()
                level = lvl
                break
        try:
            if level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "critical":
                logger.critical(msg)
            elif level == "info":
                logger.info(msg)
            elif level == "debug":
                logger.debug(msg)
            else:
                logger.info(msg)
        except _LOGGING_SETUP_EXCEPTIONS:
            try:
                self._stream.write(text + "\n")
            except _IO_EXCEPTIONS as exc:
                _safe_debug(f"StderrInterceptor fallback write failed: {exc}")

    def writelines(self, lines) -> None:
        if lines is None:
            return
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            buf = getattr(self._local, "buffer", "")
            if buf:
                try:
                    self._local.buffer = ""
                except _IO_EXCEPTIONS as exc:
                    _safe_debug(f"StderrInterceptor failed to clear buffer: {exc}")
                text = buf.rstrip("\r\n")
                if text:
                    in_write = getattr(self._local, "in_write", False)
                    if in_write:
                        try:
                            self._stream.write(text + "\n")
                        except _IO_EXCEPTIONS as exc:
                            _safe_debug(f"StderrInterceptor buffer flush write failed: {exc}")
                    else:
                        try:
                            self._local.in_write = True
                            self._log_line(text)
                        except _LOGGING_SETUP_EXCEPTIONS:
                            try:
                                self._stream.write(text + "\n")
                            except _IO_EXCEPTIONS as exc:
                                _safe_debug(f"StderrInterceptor buffer fallback write failed: {exc}")
                        finally:
                            self._local.in_write = False
            self._stream.flush()
        except _IO_EXCEPTIONS as exc:
            _safe_debug(f"StderrInterceptor flush failed: {exc}")

    def isatty(self) -> bool:
        try:
            return bool(getattr(self._stream, "isatty", lambda: False)())
        except _IO_EXCEPTIONS:
            _safe_debug("StderrInterceptor isatty check failed")
            return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self._stream, "errors", None)

    def fileno(self):
        fn = getattr(self._stream, "fileno", None)
        if fn is None:
            import io
            raise io.UnsupportedOperation("fileno")
        return fn()

    def __getattr__(self, name):
        return getattr(self._stream, name)

def _redirect_external_loggers() -> None:
    """Ensure third-party loggers route through our Loguru interceptor."""
    try:
        warn_logger = logging.getLogger("py.warnings")
        warn_logger.handlers = [InterceptHandler()]
        warn_logger.propagate = False
        warn_logger.setLevel(0)
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to configure warning logger interception: {exc}")
    # Pre-create known external loggers so they propagate to root interception.
    prefixes = (
        "kokoro",
        "huggingface_hub",
        "transformers",
        "torch",
        "sentence_transformers",
        "accelerate",
    )
    for name in prefixes:
        try:
            ext_logger = logging.getLogger(name)
            ext_logger.handlers = []
            ext_logger.propagate = True
            ext_logger.setLevel(0)
        except _LOGGING_SETUP_EXCEPTIONS as exc:
            _safe_debug(f"Failed to redirect logger '{name}': {exc}")
    # Sweep any dynamically-created external loggers.
    try:
        for lname, lgr in list(logging.root.manager.loggerDict.items()):
            if isinstance(lgr, logging.Logger) and lname.startswith(prefixes):
                lgr.handlers = []
                lgr.propagate = True
                lgr.setLevel(0)
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to sweep external loggers for redirection: {exc}")
    try:
        level_name = os.getenv("TLDW_AIOSQLITE_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.getLogger("aiosqlite").setLevel(level)
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to set aiosqlite log level: {exc}")


def _install_stderr_redirect() -> None:
    try:
        if os.getenv("TLDW_CAPTURE_STDERR", "1").lower() in {"0", "false", "no", "off"}:
            return
        if isinstance(sys.stderr, _StderrInterceptor):
            return
        base = sys.__stderr__ or sys.stderr
        sys.stderr = _StderrInterceptor(base)
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to install stderr redirect: {exc}")


def _unwrap_stderr(stream):
    if isinstance(stream, _StderrInterceptor):
        return stream._stream
    return stream

# Reset Loguru and configure a single, thread-safe sink
logger.remove()
_log_level = "DEBUG"
_force_color = _shared_env_flag_enabled("FORCE_COLOR") or _shared_env_flag_enabled("PY_COLORS")
_sink_choice = _early_os.getenv("LOG_STREAM", "stderr").lower()
_stderr = _unwrap_stderr(sys.__stderr__ or sys.stderr)
_sink = sys.stdout if _shared_is_truthy(_sink_choice) or _sink_choice == "stdout" else _stderr
_use_color = _force_color or (
    _sink.isatty() and _early_os.getenv("LOG_COLOR", "1").lower() not in {"0", "false", "no", "off"}
)


# Use synchronous logging during import-time initialization to avoid Loguru's background
# queue thread taking the import lock while startup modules are still being loaded.
class _SafeStreamWrapper:
    def __init__(self, stream):
        self._stream = stream

    def write(self, message: str):
        try:
            # Normalize line endings and ensure a newline terminator
            if message and not message.endswith("\n"):
                message = message[:-1] + "\n" if message.endswith("\r") else message + "\n"
            self._stream.write(message)
            # Flush to avoid line coalescing in buffered environments
            with suppress(_IO_EXCEPTIONS):
                self._stream.flush()
        except _IO_EXCEPTIONS:
            # Swallow closed-file or teardown-time errors
            pass

    def flush(self):
        with suppress(_IO_EXCEPTIONS):
            self._stream.flush()

    def isatty(self):
        try:
            return bool(getattr(self._stream, "isatty", lambda: False)())
        except _IO_EXCEPTIONS:
            return False


def _unwrap_logger_add(func):
    """Follow wrapper attributes to locate the underlying Loguru ``logger.add``."""
    seen = set()
    candidate = func
    while True:
        next_candidate = getattr(candidate, "_tldw_safe_original", None) or getattr(candidate, "__wrapped__", None)
        if not next_candidate or next_candidate is candidate or next_candidate in seen:
            return candidate
        seen.add(candidate)
        candidate = next_candidate


def _unwrap_loguru_wrapper(func):
    """Follow wrapper attributes to locate the underlying Loguru callable."""
    seen = set()
    candidate = func
    while True:
        next_candidate = getattr(candidate, "_tldw_safe_original", None) or getattr(candidate, "__wrapped__", None)
        if not next_candidate or next_candidate is candidate or next_candidate in seen:
            return candidate
        seen.add(candidate)
        candidate = next_candidate


def _unwrap_stdlib_wrapper(func):
    """Follow wrapper attributes to locate the underlying stdlib function."""
    seen = set()
    candidate = func
    while True:
        next_candidate = getattr(candidate, "_tldw_original", None) or getattr(candidate, "__wrapped__", None)
        if not next_candidate or next_candidate is candidate or next_candidate in seen:
            return candidate
        seen.add(candidate)
        candidate = next_candidate


# Guard against third-party loguru reconfiguration. These are the only
# modules that may reconfigure Loguru sinks in production; allow overrides
# via TLDW_ALLOW_LOGURU_RECONFIG for local troubleshooting.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tldw_Server_API/
_ALLOWED_LOGURU_CALLERS = {
    Path(__file__).resolve(),
    (_PROJECT_ROOT / "app" / "core" / "Logging" / "system_log_buffer.py").resolve(),
    (_PROJECT_ROOT / "app" / "core" / "Ingestion_Media_Processing" / "MediaWiki" / "Media_Wiki.py").resolve(),
}


def _caller_allowed_for_loguru_config() -> bool:
    if _shared_env_flag_enabled("TLDW_ALLOW_LOGURU_RECONFIG"):
        return True
    if _shared_is_explicit_pytest_runtime():
        return True
    frame = logging.currentframe()
    if frame is not None:
        frame = frame.f_back
    while frame is not None:
        fname = getattr(frame.f_code, "co_filename", "") or ""
        func_name = getattr(frame.f_code, "co_name", "") or ""
        if not fname:
            frame = frame.f_back
            continue
        if "loguru" in fname:
            frame = frame.f_back
            continue
        if fname == __file__ and func_name in {"_safe_logger_add", "_safe_logger_remove", "_safe_logger_configure"}:
            frame = frame.f_back
            continue
        try:
            return Path(fname).resolve() in _ALLOWED_LOGURU_CALLERS
        except _LOGGING_SETUP_EXCEPTIONS as exc:
            _safe_debug(f"Failed to resolve Loguru config caller path: {exc}")
            return False
    return False


# Ensure any subsequent logger.add calls wrap raw streams with SafeStreamWrapper
_ROOT_LOGGER = logger
_original_logger_add = _ROOT_LOGGER.add
_original_unwrapped_logger_add = _unwrap_logger_add(_original_logger_add)


def _safe_logger_add(sink, *args, **kwargs):
    if not _caller_allowed_for_loguru_config():
        _safe_debug("Blocked Loguru add from unauthorized caller")
        return None
    try:
        if hasattr(sink, "write") and not isinstance(sink, _SafeStreamWrapper):
            sink = _SafeStreamWrapper(sink)
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to wrap Loguru sink; using original sink: {exc}")
    target = _unwrap_logger_add(_original_logger_add)
    return target(sink, *args, **kwargs)


_ROOT_LOGGER.add = _safe_logger_add  # type: ignore[assignment]
_ROOT_LOGGER.add._tldw_safe_original = _original_unwrapped_logger_add  # type: ignore[attr-defined]
_ROOT_LOGGER.add.__wrapped__ = _original_unwrapped_logger_add  # type: ignore[attr-defined]


# Sink-level filter to guarantee presence of common extra fields
def _ensure_log_extra_fields(record: dict) -> bool:
    try:
        extra = record.setdefault("extra", {})
        # Provide defaults to avoid KeyError in format templates
        extra.setdefault("trace_id", "")
        extra.setdefault("span_id", "")
        extra.setdefault("request_id", "")
        extra.setdefault("session_id", "")
        # Ensure W3C trace context placeholder exists even before patcher runs
        extra.setdefault("traceparent", "")
        # Structured context defaults (Prompt Studio/jobs)
        extra.setdefault("job_id", "")
        extra.setdefault("ps_component", "")
        extra.setdefault("ps_job_kind", "")
        extra.setdefault("optimization_id", "")
        extra.setdefault("evaluation_id", "")
    except _TRACE_EXCEPTIONS:
        # Never block a log line due to filter errors
        pass
    return True


_ROOT_LOGGER.add(
    _SafeStreamWrapper(_sink),
    level=_log_level,
    format=_safe_log_format,
    colorize=_use_color,
    filter=_ensure_log_extra_fields,
    enqueue=False,
)
logger = _ROOT_LOGGER.patch(_trace_log_patcher)
_redirect_external_loggers()
_install_stderr_redirect()

if not hasattr(_ROOT_LOGGER, "_tldw_original_remove"):
    _ROOT_LOGGER._tldw_original_remove = _unwrap_loguru_wrapper(_ROOT_LOGGER.remove)  # type: ignore[attr-defined]
_original_logger_remove = getattr(_ROOT_LOGGER, "_tldw_original_remove", None)
_root_configure = getattr(_ROOT_LOGGER, "configure", None)
if callable(_root_configure) and not hasattr(_ROOT_LOGGER, "_tldw_original_configure"):
    _ROOT_LOGGER._tldw_original_configure = _unwrap_loguru_wrapper(_root_configure)  # type: ignore[attr-defined]
_original_logger_configure = getattr(_ROOT_LOGGER, "_tldw_original_configure", None)


def _safe_logger_remove(sink_id=None):
    if not _caller_allowed_for_loguru_config():
        _safe_debug("Blocked Loguru remove from unauthorized caller")
        return None
    target = getattr(_ROOT_LOGGER, "_tldw_original_remove", None)
    try:
        if target is None or target is _safe_logger_remove:
            try:
                return _ROOT_LOGGER.__class__.remove(_ROOT_LOGGER, sink_id)
            except _LOGGING_SETUP_EXCEPTIONS:
                return None
        return target(sink_id)
    finally:
        _redirect_external_loggers()


def _safe_logger_configure(*args, **kwargs):
    if not _caller_allowed_for_loguru_config():
        _safe_debug("Blocked Loguru configure from unauthorized caller")
        return None
    try:
        target = getattr(_ROOT_LOGGER, "_tldw_original_configure", None)
        if callable(target) and target is not _safe_logger_configure:
            return target(*args, **kwargs)
        if hasattr(_ROOT_LOGGER.__class__, "configure"):
            return _ROOT_LOGGER.__class__.configure(_ROOT_LOGGER, *args, **kwargs)
        _safe_debug("Loguru configure target unavailable; skipping")
        return None
    finally:
        _redirect_external_loggers()


_ROOT_LOGGER.remove = _safe_logger_remove  # type: ignore[assignment]
if callable(_original_logger_configure):
    _ROOT_LOGGER.configure = _safe_logger_configure  # type: ignore[assignment]
_ROOT_LOGGER.remove._tldw_safe_original = _original_logger_remove  # type: ignore[attr-defined]
_ROOT_LOGGER.remove.__wrapped__ = _original_logger_remove  # type: ignore[attr-defined]
if callable(_original_logger_configure):
    _ROOT_LOGGER.configure._tldw_safe_original = _original_logger_configure  # type: ignore[attr-defined]
    _ROOT_LOGGER.configure.__wrapped__ = _original_logger_configure  # type: ignore[attr-defined]
logger.remove = _safe_logger_remove  # type: ignore[assignment]
if callable(_original_logger_configure):
    logger.configure = _safe_logger_configure  # type: ignore[assignment]

# Prevent third-party stdlib loggers from attaching their own handlers.
if not hasattr(logging, "_tldw_original_addHandler"):
    logging._tldw_original_addHandler = logging.Logger.addHandler  # type: ignore[attr-defined]
_original_logging_addHandler = logging._tldw_original_addHandler  # type: ignore[attr-defined]


def _safe_logging_addHandler(self: logging.Logger, hdlr: logging.Handler) -> None:
    target = getattr(logging, "_tldw_original_addHandler", None)
    if target is None or target is _safe_logging_addHandler:
        _safe_debug("Stdlib addHandler hook unavailable; dropping handler to preserve Loguru interception")
        return
    if isinstance(hdlr, InterceptHandler) or _caller_allowed_for_loguru_config():
        target(self, hdlr)
    else:
        # Drop third-party handlers and rely on root interception.
        try:
            self.handlers = []
            self.propagate = True
            self.setLevel(0)
            _safe_debug(f"Dropped handler from stdlib logger '{self.name}' to preserve Loguru interception")
        except _LOGGING_SETUP_EXCEPTIONS as exc:
            _safe_debug(f"Failed to drop stdlib logger handlers for '{self.name}': {exc}")


if logging.Logger.addHandler is not _safe_logging_addHandler:
    logging.Logger.addHandler = _safe_logging_addHandler  # type: ignore[assignment]
    _safe_logging_addHandler.__wrapped__ = _original_logging_addHandler  # type: ignore[attr-defined]
    _safe_logging_addHandler._tldw_original = _original_logging_addHandler  # type: ignore[attr-defined]

# Intercept stdlib and uvicorn logs early
try:
    for _h in list(logging.root.handlers):
        logging.root.removeHandler(_h)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
except _LOGGING_SETUP_EXCEPTIONS:
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger().setLevel(0)

for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _lg = logging.getLogger(_name)
    _lg.handlers = [InterceptHandler()]
    _lg.propagate = False


# Guard against later reconfiguration by uvicorn or libraries
def _reinstall_intercept_handlers():
    try:
        logging.root.handlers = [InterceptHandler()]
        logging.root.setLevel(0)
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to reinstall root intercept handler: {exc}")
    # Replace handlers on all known loggers to avoid mixed formats
    try:
        for _lname, _logger in list(logging.root.manager.loggerDict.items()):
            if isinstance(_logger, logging.Logger):
                _logger.handlers = [InterceptHandler()]
                _logger.propagate = False
    except _LOGGING_SETUP_EXCEPTIONS as exc:
        _safe_debug(f"Failed to reinstall intercept handlers for stdlib loggers: {exc}")
    for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        try:
            _lg = logging.getLogger(_name)
            _lg.handlers = [InterceptHandler()]
            _lg.propagate = False
        except _LOGGING_SETUP_EXCEPTIONS as exc:
            _safe_debug(f"Failed to reinstall intercept handler for logger '{_name}': {exc}")
    _redirect_external_loggers()


try:
    import logging.config as _logcfg

    if not hasattr(logging, "_tldw_original_basicConfig"):
        logging._tldw_original_basicConfig = _unwrap_stdlib_wrapper(logging.basicConfig)  # type: ignore[attr-defined]
    else:
        _orig_basic = getattr(logging, "_tldw_original_basicConfig", None)
        if _orig_basic is logging.basicConfig:
            logging._tldw_original_basicConfig = _unwrap_stdlib_wrapper(logging.basicConfig)  # type: ignore[attr-defined]
    logging._tldw_reinstall = _reinstall_intercept_handlers  # type: ignore[attr-defined]

    if not getattr(logging, "_tldw_basic_config_wrapped", False):

        def _basic_config_wrapper(*args, **kwargs):
            try:
                _orig = getattr(logging, "_tldw_original_basicConfig", None)
                if callable(_orig):
                    _orig(*args, **kwargs)  # type: ignore[misc]
            finally:
                _maybe_reinstall = getattr(logging, "_tldw_reinstall", None)
                if callable(_maybe_reinstall):
                    _maybe_reinstall()

        logging.basicConfig = _basic_config_wrapper  # type: ignore[assignment]
        _basic_config_wrapper.__wrapped__ = getattr(logging, "_tldw_original_basicConfig", None)
        _basic_config_wrapper._tldw_original = getattr(logging, "_tldw_original_basicConfig", None)
        logging._tldw_basic_config_wrapped = True  # type: ignore[attr-defined]

    if hasattr(_logcfg, "dictConfig"):
        if not hasattr(_logcfg, "_tldw_original_dictConfig"):
            _logcfg._tldw_original_dictConfig = _unwrap_stdlib_wrapper(_logcfg.dictConfig)  # type: ignore[attr-defined]
        else:
            _orig_dict = getattr(_logcfg, "_tldw_original_dictConfig", None)
            if _orig_dict is _logcfg.dictConfig:
                _logcfg._tldw_original_dictConfig = _unwrap_stdlib_wrapper(_logcfg.dictConfig)  # type: ignore[attr-defined]
        _logcfg._tldw_reinstall = _reinstall_intercept_handlers  # type: ignore[attr-defined]

        if not getattr(_logcfg, "_tldw_dict_config_wrapped", False):

            def _dict_config_wrapper(config):
                try:
                    _orig = getattr(_logcfg, "_tldw_original_dictConfig", None)
                    if callable(_orig):
                        _orig(config)  # type: ignore[misc]
                finally:
                    _maybe_reinstall = getattr(_logcfg, "_tldw_reinstall", None)
                    if callable(_maybe_reinstall):
                        _maybe_reinstall()

            _logcfg.dictConfig = _dict_config_wrapper  # type: ignore[assignment]
            _dict_config_wrapper.__wrapped__ = getattr(_logcfg, "_tldw_original_dictConfig", None)
            _dict_config_wrapper._tldw_original = getattr(_logcfg, "_tldw_original_dictConfig", None)
            _logcfg._tldw_dict_config_wrapped = True  # type: ignore[attr-defined]
except _LOGGING_SETUP_EXCEPTIONS as _log_wrap_err:
    logger.debug(
        "Failed to wrap logging.config.dictConfig for interception: {}",
        _log_wrap_err,
    )

# Apply once now as well
_reinstall_intercept_handlers()

logger.info("Logging configured (Loguru + stdlib interception)")

#
# Auth Endpoint (NEW)
"""
Initialize feature flags up-front so later references in route inclusion do not
raise NameError when running under ULTRA/MINIMAL test modes or when optional
routers fail to import.
"""
_HAS_HEALTH = False
_HAS_AUDIO = False
_HAS_AUDIO_JOBS = False
_HAS_MEDIA = False
_HAS_SANDBOX = False
_HAS_OUTPUT_TEMPLATES = False
_HAS_OUTPUTS = False
_HAS_PROMPT_STUDIO = False
_HAS_WORKFLOWS = False
_HAS_CHAT_WORKFLOWS = False
_HAS_UNIFIED_EVALUATIONS = False
_HAS_SCHEDULER_WF = False
_HAS_JOBS_ADMIN = False
_HAS_CHUNKING = False
_HAS_NOTES_GRAPH = False
_HAS_READING_HIGHLIGHTS = False
_HAS_KANBAN = False
_HAS_DATA_TABLES = False
_HAS_MEETINGS = False

# Minimal test-app gating: when enabled, skip importing heavy routers
from tldw_Server_API.app.api.v1.endpoints.auth import router as auth_router
from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled

_MINIMAL_TEST_APP = _env_flag_enabled("MINIMAL_TEST_APP")
# Ultra-minimal diagnostic mode: only import health endpoints
_ULTRA_MINIMAL_APP = _env_flag_enabled("ULTRA_MINIMAL_APP")
# Opt-in startup tracing
_STARTUP_TRACE = _env_flag_enabled("STARTUP_TRACE")


def _startup_trace(msg: str) -> None:
    if _STARTUP_TRACE:
        try:
            logger.info(f"[startup-trace] {msg}")
        except _LOGGING_SETUP_EXCEPTIONS as _startup_log_err:
            logger.debug(f"Startup trace logging failed: {_startup_log_err}")


_startup_trace(f"Endpoint import gating: ULTRA_MINIMAL_APP={_ULTRA_MINIMAL_APP}, MINIMAL_TEST_APP={_MINIMAL_TEST_APP}")
#
if _ULTRA_MINIMAL_APP:
    # Keep ultra-minimal import surface tiny; health is provided by the
    # control-plane routes registered later in this module.
    _startup_trace("ULTRA_MINIMAL_APP enabled: skipping API router imports (control-plane health only).")
elif _MINIMAL_TEST_APP:
    # Defer to the dedicated minimal import block below.
    # This avoids importing heavyweight optional modules (e.g., torch-backed
    # audio dependencies) during pytest collection.
    _startup_trace("MINIMAL_TEST_APP enabled: deferring heavyweight router imports.")
else:
    _in_pytest_cmd = _shared_is_explicit_pytest_runtime() or any("pytest" in str(arg or "") for arg in sys.argv)
    _full_audio_import_enabled = True
    if _in_pytest_cmd and not _env_flag_enabled("MINIMAL_TEST_INCLUDE_AUDIO"):
        _full_audio_import_enabled = False
        logger.info("Skipping audio endpoint imports in pytest full startup (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)")

    # Audio Endpoint (includes WebSocket streaming transcription)
    if _full_audio_import_enabled:
        try:
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import ws_router as audio_ws_router

            _HAS_AUDIO = True
        except _IMPORT_EXCEPTIONS as _audio_err:
            # guard non-critical endpoints in tests
            logger.warning(f"Audio endpoints unavailable; skipping import: {_audio_err}")
            _HAS_AUDIO = False
        # Guard audio_jobs import to avoid unrelated test breakages
        try:
            from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import router as audio_jobs_router

            _HAS_AUDIO_JOBS = True
        except _IMPORT_EXCEPTIONS as _audio_jobs_err:
            logger.warning(f"Audio jobs endpoints unavailable; skipping import: {_audio_jobs_err}")
            _HAS_AUDIO_JOBS = False
    else:
        _HAS_AUDIO = False
        _HAS_AUDIO_JOBS = False
    # Chat Endpoint
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import router as character_chat_sessions_router
    from tldw_Server_API.app.api.v1.endpoints.character_memory import router as character_memory_router
    from tldw_Server_API.app.api.v1.endpoints.character_messages import router as character_messages_router

    # Workspace Endpoints
    from tldw_Server_API.app.api.v1.endpoints.workspace_migrations import router as workspace_migrations_router
    from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router

    # Character Endpoints
    from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as character_router
    from tldw_Server_API.app.api.v1.endpoints.chat import (
        conversations_alias_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.chat import (
        router as chat_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.chat_loop import (
        router as chat_loop_router,
    )

    # Metrics Endpoint
    from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router

    # Sandbox Endpoint (scaffold)
    try:
        from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

        _HAS_SANDBOX = True
    except _IMPORT_EXCEPTIONS as _sandbox_err:
        logger.warning(f"Sandbox endpoints unavailable; skipping import: {_sandbox_err}")
        _HAS_SANDBOX = False
    # Chunking Endpoints (guard to avoid failures from optional summarization deps)
    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking import chunking_router as chunking_router

        _HAS_CHUNKING = True
    except _IMPORT_EXCEPTIONS as _chunk_err:
        logger.warning(f"Chunking endpoints unavailable; skipping import: {_chunk_err}")
        _HAS_CHUNKING = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router
    except _IMPORT_EXCEPTIONS as _chunk_tpl_err:
        logger.warning(f"Chunking templates endpoints unavailable; skipping import: {_chunk_tpl_err}")
    # Embeddings / Vector stores / Claims
    from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router
    from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router
    from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router as vector_stores_router

    # Collections (stubs to anchor PRD)
    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs_templates import router as outputs_templates_router

        _HAS_OUTPUT_TEMPLATES = True
    except _IMPORT_EXCEPTIONS as _ot_err:
        logger.warning(f"Outputs templates endpoints unavailable; skipping import: {_ot_err}")
        _HAS_OUTPUT_TEMPLATES = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

        _HAS_OUTPUTS = True
    except _IMPORT_EXCEPTIONS as _o_err:
        logger.warning(f"Outputs endpoints unavailable; skipping import: {_o_err}")
        _HAS_OUTPUTS = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.meetings import router as meetings_router

        _HAS_MEETINGS = True
    except _IMPORT_EXCEPTIONS as _meetings_err:
        logger.warning(f"Meetings endpoints unavailable; skipping import: {_meetings_err}")
        _HAS_MEETINGS = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as collections_feeds_router

        _HAS_COLLECTIONS_FEEDS = True
    except _IMPORT_EXCEPTIONS as _cf_err:
        logger.warning(f"Collections feeds endpoints unavailable; skipping import: {_cf_err}")
        _HAS_COLLECTIONS_FEEDS = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import (
            callback_router as websub_callback_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import (
            router as collections_websub_router,
        )

        _HAS_COLLECTIONS_WEBSUB = True
    except _IMPORT_EXCEPTIONS as _cw_err:
        logger.warning(f"Collections WebSub endpoints unavailable; skipping import: {_cw_err}")
        _HAS_COLLECTIONS_WEBSUB = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.slack import router as slack_router

        _HAS_SLACK = True
    except _IMPORT_EXCEPTIONS as _slack_err:
        logger.warning(f"Slack endpoints unavailable; skipping import: {_slack_err}")
        _HAS_SLACK = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.discord import router as discord_router

        _HAS_DISCORD = True
    except _IMPORT_EXCEPTIONS as _discord_err:
        logger.warning(f"Discord endpoints unavailable; skipping import: {_discord_err}")
        _HAS_DISCORD = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.telegram import router as telegram_router

        _HAS_TELEGRAM = True
    except _IMPORT_EXCEPTIONS as _telegram_err:
        logger.warning(f"Telegram endpoints unavailable; skipping import: {_telegram_err}")
        _HAS_TELEGRAM = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.files import router as files_router

        _HAS_FILES = True
    except ImportError as _files_err:
        logger.warning(f"Files endpoints unavailable; skipping import: {_files_err}")
        _HAS_FILES = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router

        _HAS_DATA_TABLES = True
    except ImportError as _dt_err:
        logger.warning(f"Data tables endpoints unavailable; skipping import: {_dt_err}")
        _HAS_DATA_TABLES = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.reading_highlights import router as reading_highlights_router

        _HAS_READING_HIGHLIGHTS = True
    except _IMPORT_EXCEPTIONS as _rh_err:
        logger.warning(f"Reading highlights endpoints unavailable; skipping import: {_rh_err}")
        _HAS_READING_HIGHLIGHTS = False
    # Media Endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
        from tldw_Server_API.app.api.v1.endpoints.web_scraping import (
            router as web_scraping_router,
        )

        _HAS_MEDIA = True
    except _IMPORT_EXCEPTIONS as _media_import_err:
        logger.warning(f"Media endpoints unavailable; skipping import: {_media_import_err}")
        _HAS_MEDIA = False
    from tldw_Server_API.app.api.v1.endpoints.media_embeddings import router as media_embeddings_router

    # Unified items endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.items import router as items_router

        _HAS_ITEMS = True
    except _IMPORT_EXCEPTIONS as _items_err:
        logger.warning(f"Items endpoints unavailable; skipping import: {_items_err}")
        _HAS_ITEMS = False
    # Notes / Prompts / Translation
    from tldw_Server_API.app.api.v1.endpoints.ingestion_sources import router as ingestion_sources_router
    from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router
    from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router
    from tldw_Server_API.app.api.v1.endpoints.translate import router as translate_router
    try:
        from tldw_Server_API.app.api.v1.endpoints.web_clipper import router as web_clipper_router

        _HAS_WEB_CLIPPER = True
    except _IMPORT_EXCEPTIONS as _wc_err:
        logger.warning(f"Web clipper endpoints unavailable; skipping import: {_wc_err}")
        _HAS_WEB_CLIPPER = False

    # Notes Graph (stub, RBAC-wired)
    try:
        from tldw_Server_API.app.api.v1.endpoints.notes_graph import router as notes_graph_router

        _HAS_NOTES_GRAPH = True
    except _IMPORT_EXCEPTIONS as _ng_err:
        logger.warning(f"Notes Graph endpoints unavailable; skipping import: {_ng_err}")
        _HAS_NOTES_GRAPH = False
    from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router

    # Kanban Board endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_boards import router as kanban_boards_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_cards import router as kanban_cards_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_checklists import router as kanban_checklists_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_comments import router as kanban_comments_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_labels import router as kanban_labels_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_links import router as kanban_links_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_lists import router as kanban_lists_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_search import router as kanban_search_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_workflow import router as kanban_workflow_router

        _HAS_KANBAN = True
    except ImportError as _kanban_err:
        logger.warning(f"Kanban endpoints unavailable; skipping import: {_kanban_err}")
        _HAS_KANBAN = False

    # Prompt Studio (guarded)
    try:
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations import (
            router as prompt_studio_evaluations_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization import (
            router as prompt_studio_optimization_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_projects import (
            router as prompt_studio_projects_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts import (
            router as prompt_studio_prompts_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_status import (
            router as prompt_studio_status_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_test_cases import (
            router as prompt_studio_test_cases_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
            router as prompt_studio_websocket_router,
        )

        _HAS_PROMPT_STUDIO = True
    except _IMPORT_EXCEPTIONS as _ps_import_err:
        logger.warning(f"Prompt Studio endpoints unavailable; skipping import: {_ps_import_err}")
        _HAS_PROMPT_STUDIO = False
    # RAG & Workflows
    from tldw_Server_API.app.api.v1.endpoints.feedback import router as feedback_router
    from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router
    from tldw_Server_API.app.api.v1.endpoints.rag_unified import router as rag_unified_router

    try:
        from tldw_Server_API.app.api.v1.endpoints.workflows import router as workflows_router

        _HAS_WORKFLOWS = True
    except _IMPORT_EXCEPTIONS as _wf_import_err:
        logger.warning(f"Workflows endpoints unavailable; skipping import: {_wf_import_err}")
        _HAS_WORKFLOWS = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.chat_workflows import (
            router as chat_workflows_router,
        )

        _HAS_CHAT_WORKFLOWS = True
    except _IMPORT_EXCEPTIONS as _chat_wf_import_err:
        logger.warning(
            f"Chat workflows endpoints unavailable; skipping import: {_chat_wf_import_err}"
        )
        _HAS_CHAT_WORKFLOWS = False
# Legacy RAG Endpoint (Deprecated)
# from tldw_Server_API.app.api.v1.endpoints.rag import router as retrieval_agent_router
#
# Research/Paper Search and heavy routers/imports
# In minimal test-app mode, import only what is needed for lightweight tests.
if _ULTRA_MINIMAL_APP:
    # Keep ultra-minimal import surface tiny; this mode intentionally avoids
    # endpoint imports beyond control-plane health handling.
    pass
elif _MINIMAL_TEST_APP:
    # Research Endpoint (lightweight subset for tests)
    # Paper Search Endpoint (provider-specific)
    from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router
    from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router
    from tldw_Server_API.app.api.v1.endpoints.research import router as research_router
    from tldw_Server_API.app.api.v1.endpoints.research_runs import router as research_runs_router
    try:
        from tldw_Server_API.app.api.v1.endpoints.setup import router as setup_router
    except _IMPORT_EXCEPTIONS as _setup_min_import_err:
        logger.debug(
            "Skipping setup router import in minimal test app: {}",
            _setup_min_import_err,
        )
        setup_router = None  # type: ignore[assignment]

    # Admin endpoints are used by several pytest modules; import for minimal app
    try:
        from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router

        _HAS_ADMIN_MIN = True
    except _IMPORT_EXCEPTIONS as _admin_min_err:
        logger.debug(f"Skipping admin router import in minimal test app: {_admin_min_err}")
        _HAS_ADMIN_MIN = False
    _HAS_UNIFIED_EVALUATIONS = False
    # Minimal chat/character endpoints to support lightweight tests
    # These are relatively lightweight and safe to import under MINIMAL_TEST_APP
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import router as character_chat_sessions_router
    from tldw_Server_API.app.api.v1.endpoints.character_memory import router as character_memory_router
    from tldw_Server_API.app.api.v1.endpoints.character_messages import router as character_messages_router
    from tldw_Server_API.app.api.v1.endpoints.workspace_migrations import router as workspace_migrations_router
    from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router
    from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as character_router
    from tldw_Server_API.app.api.v1.endpoints.chat import (
        conversations_alias_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.chat import (
        router as chat_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.chat_loop import (
        router as chat_loop_router,
    )

    # Sandbox endpoint is optional; guard import so minimal startup never fails
    try:
        from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

        _HAS_SANDBOX = True
    except _IMPORT_EXCEPTIONS as _sb_err:
        logger.warning(f"Sandbox endpoints unavailable; skipping import: {_sb_err}")
        _HAS_SANDBOX = False
    # MCP Unified Endpoint (safe to import for tests)
    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router
    except _IMPORT_EXCEPTIONS as _mcp_imp_err:
        logger.debug(f"Skipping MCP unified import in minimal test app: {_mcp_imp_err}")
    # LlamaCpp endpoints for reranking tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import (
            public_router as llamacpp_public_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import (
            router as llamacpp_router,
        )
    except _IMPORT_EXCEPTIONS as _llama_imp_err:
        logger.debug(f"Skipping llamacpp import in minimal test app: {_llama_imp_err}")
        llamacpp_router = None  # type: ignore[assignment]
        llamacpp_public_router = None  # type: ignore[assignment]
else:
    # Research Endpoint
    # Note: Evaluations, OCR, and VLM are imported later inside route-enabled gates
    # Benchmark Endpoint
    from tldw_Server_API.app.api.v1.endpoints.benchmark_api import router as benchmark_router

    # Paper Search Endpoint (provider-specific)
    from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router
    from tldw_Server_API.app.api.v1.endpoints.research import router as research_router
    from tldw_Server_API.app.api.v1.endpoints.research_runs import router as research_runs_router

    # Sync Endpoint
    from tldw_Server_API.app.api.v1.endpoints.sync import router as sync_router
    from tldw_Server_API.app.api.v1.endpoints.text2sql import router as text2sql_router

    # Tools Endpoint (optional; guard import to avoid startup failure on optional module issues)
    try:
        from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router
    except _IMPORT_EXCEPTIONS as _tools_import_err:
        logger.warning(f"Tools endpoints unavailable at import time; deferring: {_tools_import_err}")
        tools_router = None  # type: ignore[assignment]
    # Agent Client Protocol (ACP) runner endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import router as acp_router
    except _IMPORT_EXCEPTIONS as _acp_import_err:
        logger.warning(f"ACP endpoints unavailable at import time; deferring: {_acp_import_err}")
        acp_router = None  # type: ignore[assignment]
    # ACP sub-module routers (schedules, triggers, permissions)
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_schedules import router as acp_schedules_router
    except _IMPORT_EXCEPTIONS as _acp_sched_err:
        logger.warning(f"ACP schedules endpoints unavailable at import time; deferring: {_acp_sched_err}")
        acp_schedules_router = None  # type: ignore[assignment]
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_triggers import router as acp_triggers_router
    except _IMPORT_EXCEPTIONS as _acp_trig_err:
        logger.warning(f"ACP triggers endpoints unavailable at import time; deferring: {_acp_trig_err}")
        acp_triggers_router = None  # type: ignore[assignment]
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_permissions import router as acp_permissions_router
    except _IMPORT_EXCEPTIONS as _acp_perm_err:
        logger.warning(f"ACP permissions endpoints unavailable at import time; deferring: {_acp_perm_err}")
        acp_permissions_router = None  # type: ignore[assignment]
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_multiplex import router as acp_multiplex_router
    except _IMPORT_EXCEPTIONS as _acp_mpx_err:
        logger.warning(f"ACP multiplex endpoints unavailable at import time; deferring: {_acp_mpx_err}")
        acp_multiplex_router = None  # type: ignore[assignment]
    # Users Endpoint (NEW)
    # Chatbooks Endpoint
    from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router
    # Sharing Endpoint
    from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router
    from tldw_Server_API.app.api.v1.endpoints.consent import router as consent_router

    # Flashcards Endpoint (V5 - ChaChaNotes)
    from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router
    from tldw_Server_API.app.api.v1.endpoints.study_suggestions import (
        router as study_suggestions_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.llamacpp import (
        public_router as llamacpp_public_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.llamacpp import (
        router as llamacpp_router,
    )

    # LLM Providers Endpoint
    from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_providers_router

    ## Trash Endpoint
    # from tldw_Server_API.app.api.v1.endpoints.trash import router as trash_router
    # MCP Unified Endpoint (Production-ready, secure implementation)
    from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router
    from tldw_Server_API.app.api.v1.endpoints.messages import (
        public_router as messages_public_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.messages import (
        router as messages_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.mlx import router as mlx_router

    # Privilege Maps Endpoint
    from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router

    # Quizzes Endpoint (ChaChaNotes)
    from tldw_Server_API.app.api.v1.endpoints.quizzes import router as quizzes_router
    from tldw_Server_API.app.api.v1.endpoints.setup import router as setup_router
    from tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped import router as shared_keys_scoped_router
    from tldw_Server_API.app.api.v1.endpoints.user_keys import router as user_keys_router
    try:
        from tldw_Server_API.app.api.v1.endpoints.users import router as users_router
    except _IMPORT_EXCEPTIONS as _users_import_err:
        logger.warning(f"Users endpoints unavailable at import time; deferring: {_users_import_err}")
        users_router = None  # type: ignore[assignment]

    # Web Scraping Management Endpoints
    from tldw_Server_API.app.api.v1.endpoints.web_scraping import router as web_scraping_router

    # Writing Playground Endpoint (ChaChaNotes)
    try:
        from tldw_Server_API.app.api.v1.endpoints.writing import router as writing_router
    except _IMPORT_EXCEPTIONS as _writing_import_err:
        logger.warning(f"Writing endpoints unavailable at import time; deferring: {_writing_import_err}")
        writing_router = None  # type: ignore[assignment]

    # Manuscript Management Endpoints (ChaChaNotes)
    try:
        from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import router as manuscripts_router
    except _IMPORT_EXCEPTIONS as _manuscripts_import_err:
        logger.warning(f"Manuscript endpoints unavailable at import time; deferring: {_manuscripts_import_err}")
        manuscripts_router = None  # type: ignore[assignment]

    # Sandbox Endpoint (scaffold)
    try:
        from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

        _HAS_SANDBOX = True
    except _IMPORT_EXCEPTIONS as _sb_err:
        logger.warning(f"Sandbox endpoints unavailable; skipping import: {_sb_err}")
        _HAS_SANDBOX = False

# Metrics and Telemetry - import directly and fail fast on errors
from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
from tldw_Server_API.app.core.AuthNZ.startup_integrity import (
    verify_authnz_sqlite_startup_integrity,
)

# Core helpers - import directly (fail fast if missing)
from tldw_Server_API.app.core.Evaluations.evaluation_manager import get_cached_evaluation_manager
from tldw_Server_API.app.core.Metrics import (
    OTEL_AVAILABLE,
    get_metrics_registry,
    initialize_telemetry,
    shutdown_telemetry,
    track_metrics,
)
from tldw_Server_API.app.core.Setup.setup_manager import needs_setup

# MCP Unified config validation (fail-fast hardening)
try:
    from tldw_Server_API.app.core.MCP_unified.config import (
        get_config as get_mcp_config,
    )
    from tldw_Server_API.app.core.MCP_unified.config import (
        validate_config as validate_mcp_config,
    )
except _IMPORT_EXCEPTIONS:
    # MCP module may be optional in some minimal deployments; guard import
    validate_mcp_config = None  # type: ignore[assignment]
    get_mcp_config = None  # type: ignore[assignment]
#
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#
########################################################################################################################
#
# Functions:


"""
Optional JSON-structured logs sink (enable with LOG_JSON=true)
- Adds an additional sink which serializes records as JSON to stdout.
"""
try:
    if _shared_env_flag_enabled("LOG_JSON") or _shared_env_flag_enabled("ENABLE_JSON_LOGS"):
        logger.add(
            _SafeStreamWrapper(sys.stdout),
            level=_log_level,
            serialize=True,
            backtrace=False,
            diagnose=False,
            filter=_ensure_log_extra_fields,
            enqueue=True,
        )
        with suppress(_LOGGING_SETUP_EXCEPTIONS):
            logger.info("JSON logging enabled (serialize=True, async enqueue)")
except _LOGGING_SETUP_EXCEPTIONS as _e:
    with suppress(_LOGGING_SETUP_EXCEPTIONS):
        logger.debug(f"Failed to enable JSON logs sink: {_e}")

# Best-effort: capture recent logs in an in-memory ring buffer for admin queries.
try:
    from tldw_Server_API.app.core.Logging.system_log_buffer import ensure_system_log_buffer

    ensure_system_log_buffer()
except _IMPORT_EXCEPTIONS as _e:
    with suppress(_LOGGING_SETUP_EXCEPTIONS):
        logger.debug(f"Failed to enable system log buffer: {_e}")


BASE_DIR = Path(__file__).resolve().parent
FAVICON_PATH = BASE_DIR / "static" / "favicon.ico"

############################# TEST DB Handling #####################################
# --- TEST DB Instance ---
test_db_instance_ref = None  # Global or context variable to hold the test DB instance

# Global readiness state (flips false during graceful shutdown)
READINESS_STATE = {"ready": True}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown for the given FastAPI app, performing validations, initializing services, scheduling deferred non-critical startup tasks, and running background workers.

    Parameters:
        app (FastAPI): The FastAPI application instance whose lifespan is managed.

    Returns:
        None: Yields once to allow the application to run; when resumed performs orderly shutdown and resource cleanup.
    """
    _startup_trace("lifespan: entered")
    # Security hard-stop: test-mode flags must only be active during explicit
    # pytest runtime (PYTEST_CURRENT_TEST).
    try:
        from tldw_Server_API.app.core.testing import validate_test_runtime_flags

        validate_test_runtime_flags()
    except RuntimeError as _test_guard_err:
        logger.critical(f"Startup aborted due to unsafe test-mode flags: {_test_guard_err}")
        raise
    except _IMPORT_EXCEPTIONS as _test_guard_import_err:
        logger.debug(f"Test-mode runtime guard import skipped: {_test_guard_import_err}")
    # Ensure in-process restarts (common in tests) reset readiness and job acquisition gates.
    # In production, the process typically exits after shutdown; in tests we reuse the app object.
    try:
        mark_lifecycle_startup(app, READINESS_STATE)
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JM

        _JM.set_acquire_gate(False)
    except _IMPORT_EXCEPTIONS:
        pass
    usage_task = None
    llm_usage_task = None
    chatbooks_cleanup_task = None
    chatbooks_cleanup_stop_event = None
    storage_cleanup_service = None
    _authnz_sched_started = False
    # Fail fast if the assembled app contains duplicate method+path route registrations.
    _fail_on_duplicate_route_method_pairs(app, context="lifespan startup")
    # Run environmental preflight checks before heavy initialization.
    # Executed in a worker thread because some checks (e.g. database
    # connectivity) perform blocking network I/O.
    try:
        import asyncio as _preflight_asyncio
        from tldw_Server_API.app.core.startup_preflight import run_preflight_checks

        preflight = await _preflight_asyncio.to_thread(run_preflight_checks)
        logger.info(
            "Preflight: {} checks, {} warnings, {} failures",
            len(preflight.checks),
            len(preflight.warnings),
            len(preflight.failures),
        )
    except RuntimeError:
        raise
    except _STARTUP_GUARD_EXCEPTIONS as _preflight_err:
        logger.debug(f"Preflight checks skipped: {_preflight_err}")
    # Determine if heavy (non-critical) startup should be deferred to background
    # Read environment knobs with precedence:
    # - DISABLE_HEAVY_STARTUP=true  => force synchronous (no deferral)
    # - else DEFER_HEAVY_STARTUP=true => defer heavy startup
    # - default => synchronous (no deferral)
    try:
        import os as _env_os
        _disable = _shared_is_truthy(_env_os.getenv("DISABLE_HEAVY_STARTUP"))
        _defer_heavy = False if _disable else _shared_is_truthy(_env_os.getenv("DEFER_HEAVY_STARTUP"))
        # Default to synchronous (False) if neither flag is set
        _defer_heavy = bool(_defer_heavy)
    except _STARTUP_GUARD_EXCEPTIONS:
        # On any error determining flags, default to synchronous startup
        _defer_heavy = False

    # Container for background startup tasks (used during shutdown)
    with suppress(_STARTUP_GUARD_EXCEPTIONS):
        app.state.bg_tasks = {}
    chat_config: dict[str, object] = {}
    # Startup: initialize Prompts DB close worker on a running event loop.
    try:
        from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import (
            start_prompts_pending_close_worker,
        )

        start_prompts_pending_close_worker()
    except _STARTUP_GUARD_EXCEPTIONS as _prompts_start_err:
        logger.debug(f"App Startup: Prompts close worker startup skipped/failed: {_prompts_start_err}")
    # Startup: Validate MCP configuration in production (fail fast)
    try:
        if get_mcp_config and validate_mcp_config:
            mcp_cfg = get_mcp_config()
            if not mcp_cfg.debug_mode:
                ok = validate_mcp_config()
                if not ok:
                    raise RuntimeError("MCP configuration validation failed; refusing to start in production")
    except _STARTUP_GUARD_EXCEPTIONS as _mcp_val_err:
        # Abort startup on validation errors
        logger.exception(f"Startup aborted due to insecure MCP configuration: {_mcp_val_err}")
        raise

    # Startup: Validate ACP runner configuration (non-fatal warnings)
    try:
        if route_enabled("acp", default_stable=False):
            from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
                load_acp_runner_config as _load_acp_cfg,
                validate_acp_config as _validate_acp_cfg,
            )

            _acp_cfg = _load_acp_cfg()
            _acp_warnings = _validate_acp_cfg(_acp_cfg)
            for _acp_w in _acp_warnings:
                logger.warning("ACP config: {}", _acp_w)
            if not _acp_warnings:
                logger.info("App Startup: ACP runner configuration validated")
    except _STARTUP_GUARD_EXCEPTIONS as _acp_val_err:
        logger.debug("App Startup: ACP config validation skipped: {}", _acp_val_err)

    # Startup: Validate Postgres content backend when enabled
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Manager import (
            validate_postgres_content_backend as _validate_content_backend,
        )

        _validate_content_backend()
        logger.info("App Startup: PostgreSQL content backend validated")
    except RuntimeError as _content_err:
        logger.exception(f"Startup aborted: {_content_err}")
        raise
    except ImportError as _content_import_err:
        logger.debug(f"Content backend validation skipped (import error): {_content_import_err}")

    # Startup: validate claims extraction prompt templates (configurable off|warning|error).
    try:
        from tldw_Server_API.app.core.Claims_Extraction.prompt_validation import (
            ClaimsPromptValidationError,
            claims_prompt_report_has_issues,
            validate_claims_prompt_preflight,
        )
        from tldw_Server_API.app.core.config import settings as _claims_settings

        _claims_prompt_report = validate_claims_prompt_preflight(_claims_settings)
        if claims_prompt_report_has_issues(_claims_prompt_report) and _claims_prompt_report.mode != "off":
            logger.warning(
                "App Startup: Claims prompt validation found {} issue(s) (mode={}, strict={})",
                len(_claims_prompt_report.issues),
                _claims_prompt_report.mode,
                _claims_prompt_report.strict,
            )
        else:
            logger.info(
                "App Startup: Claims prompt validation completed (mode={}, strict={})",
                _claims_prompt_report.mode,
                _claims_prompt_report.strict,
            )
    except ClaimsPromptValidationError as _claims_prompt_err:
        logger.exception("Startup aborted due to claims prompt validation error")
        raise
    except _STARTUP_GUARD_EXCEPTIONS as _claims_prompt_exc:
        logger.debug("App Startup: Claims prompt validation skipped/failed: {}", _claims_prompt_exc)

    # Startup: preserve fail-fast semantics for critical lazy subsystems in non-test runtime.
    # Warm lazy managers early so configuration errors surface at startup.
    try:
        if not globals().get("_TEST_MODE") and route_enabled("evaluations"):
            from tldw_Server_API.app.core.Evaluations.connection_pool import (
                get_connection_manager as _get_evaluations_connection_manager,
            )
            from tldw_Server_API.app.core.Evaluations.webhook_manager import (
                get_webhook_manager as _get_webhook_manager,
            )

            _get_evaluations_connection_manager()
            _get_webhook_manager()
            logger.info("App Startup: Warmed lazy Evaluations managers (fail-fast enabled)")
    except _STARTUP_GUARD_EXCEPTIONS as _lazy_warmup_err:
        logger.exception(f"Startup aborted: lazy subsystem warmup failed: {_lazy_warmup_err}")
        raise

    # Startup: Initialize telemetry and metrics
    logger.info("App Startup: Initializing telemetry and metrics...")
    try:
        telemetry_manager = initialize_telemetry()
        if OTEL_AVAILABLE:
            logger.info(f"App Startup: OpenTelemetry initialized for service: {telemetry_manager.config.service_name}")
        else:
            logger.warning("App Startup: OpenTelemetry not available, using fallback metrics")
        try:
            from tldw_Server_API.app.core.Metrics.telemetry import instrument_fastapi_app

            if instrument_fastapi_app(app, telemetry_manager):
                logger.info("App Startup: FastAPI instrumentation enabled")
        except _STARTUP_GUARD_EXCEPTIONS as _otel_app_err:
            logger.debug(f"App Startup: FastAPI instrumentation skipped: {_otel_app_err}")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Startup: Failed to initialize telemetry: {e}")

    # Startup: Initialize Sentry error tracking (optional)
    _sentry_dsn = os.getenv("SENTRY_DSN", "")
    if _sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=_sentry_dsn,
                traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
                environment=os.getenv("DEPLOYMENT_ENV", "development"),
                release=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
                send_default_pii=False,
            )
            logger.info("App Startup: Sentry error tracking initialized")
        except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _sentry_err:
            logger.warning("App Startup: Sentry initialization failed: {}", _sentry_err)

    # Startup: Warn if first-time setup is enabled (local-only, no proxies)
    try:
        if needs_setup():
            logger.warning(
                "First-time setup is enabled. The setup API is local-only and blocks proxied requests. "
                "If running behind a reverse proxy, ensure /setup and /api/v1/setup are not publicly exposed, or "
                "set TLDW_SETUP_ALLOW_REMOTE=1 temporarily on trusted networks."
            )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.debug(f"Setup status check failed during startup: {e}")

    # Startup: Initialize auth services
    logger.info("App Startup: Initializing authentication services...")
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_auth_settings

        _auth_settings = _get_auth_settings()
        _allow_corrupt_startup = _shared_is_truthy(
            os.getenv("TLDW_ALLOW_CORRUPT_AUTHNZ_STARTUP")
        )
        await verify_authnz_sqlite_startup_integrity(
            database_url=str(getattr(_auth_settings, "DATABASE_URL", "")),
            auth_mode=str(getattr(_auth_settings, "AUTH_MODE", "single_user")),
            dispatch_alerts=True,
            fail_on_error=not _allow_corrupt_startup,
        )
        if _allow_corrupt_startup:
            logger.warning(
                "App Startup: Corrupt AuthNZ DB fail-open mode enabled via "
                "TLDW_ALLOW_CORRUPT_AUTHNZ_STARTUP=true"
            )
    except _STARTUP_GUARD_EXCEPTIONS as _integrity_err:
        logger.exception(
            f"App Startup: AuthNZ SQLite integrity preflight failed: {_integrity_err}"
        )
        raise

    try:
        # Initialize database pool for auth
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        db_pool = await get_db_pool()
        logger.info("App Startup: Database pool initialized")

        # Ensure AuthNZ schema/migrations (centralized helper for SQLite; PG extras as before)
        try:
            from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

            await ensure_authnz_schema_ready_once()
        except _IMPORT_EXCEPTIONS as _e:
            logger.debug(f"App Startup: Skipped AuthNZ SQLite migration ensure: {_e}")
        # Postgres-only: ensure additive extras (tool catalogs, privilege snapshots, usage tables, VK counters)
        try:
            if getattr(db_pool, "pool", None):
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_api_keys_tables_pg,
                    ensure_authnz_core_tables_pg,
                    ensure_generated_files_table_pg,
                    ensure_llm_provider_overrides_pg,
                    ensure_privilege_snapshots_table_pg,
                    ensure_tool_catalogs_tables_pg,
                    ensure_usage_tables_pg,
                    ensure_virtual_key_counters_pg,
                )

                ok_authnz_core_pg = await ensure_authnz_core_tables_pg(db_pool)
                if ok_authnz_core_pg:
                    logger.info("App Startup: Ensured PG AuthNZ core tables")
                ok_generated_files_pg = await ensure_generated_files_table_pg(db_pool)
                if ok_generated_files_pg:
                    logger.info("App Startup: Ensured PG generated_files table")
                ok_catalogs = await ensure_tool_catalogs_tables_pg(db_pool)
                if ok_catalogs:
                    logger.info("App Startup: Ensured PG tool catalogs tables")
                ok_priv_snapshots = await ensure_privilege_snapshots_table_pg(db_pool)
                if ok_priv_snapshots:
                    logger.info("App Startup: Ensured PG privilege_snapshots table")
                ok_api_keys_pg = await ensure_api_keys_tables_pg(db_pool)
                if ok_api_keys_pg:
                    logger.info("App Startup: Ensured PG api_keys tables")
                ok_usage_pg = await ensure_usage_tables_pg(db_pool)
                if ok_usage_pg:
                    logger.info("App Startup: Ensured PG usage tables")
                ok_vk_pg = await ensure_virtual_key_counters_pg(db_pool)
                if ok_vk_pg:
                    logger.info("App Startup: Ensured PG virtual-key counters tables")
                ok_overrides_pg = await ensure_llm_provider_overrides_pg(db_pool)
                if ok_overrides_pg:
                    logger.info("App Startup: Ensured PG llm_provider_overrides table")
        except _STARTUP_GUARD_EXCEPTIONS as _pg_e:
            logger.debug(f"App Startup: PG extras ensure failed/skipped: {_pg_e}")
        # Ensure RBAC seed exists in single-user mode (idempotent; both backends)
        try:
            await ensure_single_user_rbac_seed_if_needed()
            logger.info("App Startup: Ensured single-user RBAC seed (baseline roles/permissions)")
        except _IMPORT_EXCEPTIONS as _e:
            logger.debug(f"App Startup: RBAC single-user seed ensure skipped: {_e}")

        # Load LLM provider overrides into memory for runtime enforcement.
        try:
            from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
                refresh_llm_provider_overrides as _refresh_llm_provider_overrides,
            )

            await _refresh_llm_provider_overrides(db_pool)
            logger.info("App Startup: Loaded LLM provider overrides")
        except _IMPORT_EXCEPTIONS as _e:
            logger.debug(f"App Startup: LLM provider overrides load skipped: {_e}")

        # Initialize ResourceGovernor policy loader (file or DB store)
        try:
            from tldw_Server_API.app.core.config import (
                rg_backend as _rg_backend_sel,
            )
            from tldw_Server_API.app.core.config import (
                rg_policy_path as _rg_policy_path,
            )
            from tldw_Server_API.app.core.config import (
                rg_policy_reload_enabled as _rg_reload_enabled,
            )
            from tldw_Server_API.app.core.config import (
                rg_policy_reload_interval_sec as _rg_reload_interval,
            )
            from tldw_Server_API.app.core.config import (
                rg_policy_store as _rg_store_sel,
            )
            from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RedisResourceGovernor
            from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                PolicyLoader as _RGPolicyLoader,
            )
            from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                PolicyReloadConfig as _RGReloadCfg,
            )
            from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                db_policy_loader as _rg_db_loader,
            )
            from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                default_policy_loader as _rg_default_loader,
            )

            _store_mode = _rg_store_sel()
            if _store_mode == "db":
                try:
                    from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import (
                        AuthNZPolicyStore as _RGDBStore,
                    )

                    _store = _RGDBStore()
                    _interval = _rg_reload_interval()
                    rg_loader = _rg_db_loader(_store, _RGReloadCfg(enabled=True, interval_sec=_interval))
                    logger.info("ResourceGovernor policy loader configured for AuthNZ DB store")
                except _STARTUP_GUARD_EXCEPTIONS as _rg_db_err:
                    logger.warning(f"Failed to configure DB-backed policy store, falling back to file: {_rg_db_err}")
                    rg_loader = _rg_default_loader()
                    _store_mode = "file"
            else:
                # File-based policy store: use config-driven path and reload settings
                _enabled = _rg_reload_enabled()
                _interval = _rg_reload_interval()
                _path = _rg_policy_path()
                rg_loader = _RGPolicyLoader(_path, _RGReloadCfg(enabled=_enabled, interval_sec=_interval))
                _store_mode = "file"

            await rg_loader.load_once()
            try:
                if _rg_reload_enabled():
                    await rg_loader.start_auto_reload()
            except _STARTUP_GUARD_EXCEPTIONS as _rg_reload_err:
                logger.debug(f"Policy auto-reload not started: {_rg_reload_err}")
            app.state.rg_policy_loader = rg_loader
            app.state.rg_policy_store = _store_mode
            try:
                _backend = _rg_backend_sel()
                if _backend == "redis":
                    # Boot-time health guard: when Redis backend is selected and
                    # fail mode is fail_closed, require a real Redis connection
                    # and refuse to start if unreachable (no stub fallback).
                    try:
                        from tldw_Server_API.app.core.config import rg_redis_fail_mode as _rg_fail_mode

                        if str(_rg_fail_mode() or "").strip().lower() == "fail_closed":
                            from tldw_Server_API.app.core.Infrastructure.redis_factory import (
                                create_async_redis_client as _create_async_redis_client,
                            )
                            from tldw_Server_API.app.core.Infrastructure.redis_factory import (
                                ensure_async_client_closed as _ensure_async_client_closed,
                            )

                            _start = logger.bind(component="rg_boot_health")
                            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                                _start.info("RG boot health: verifying Redis connectivity (fail_closed mode)")
                            _rc = await _create_async_redis_client(fallback_to_fake=False, context="rg_boot_health")
                            try:
                                # Extra sanity ping; factory already pings
                                res = getattr(_rc, "ping", None)
                                if res:
                                    pr = res()
                                    if hasattr(pr, "__await__"):
                                        await pr
                            finally:
                                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                                    await _ensure_async_client_closed(_rc)
                    except _STARTUP_GUARD_EXCEPTIONS as _rg_boot_err:
                        logger.exception(
                            f"ResourceGovernor boot health failed (Redis unreachable, fail_closed): {_rg_boot_err}"
                        )
                        raise RuntimeError(
                            "Redis backend selected with fail_closed, but Redis is unreachable; refusing to start"
                        ) from _rg_boot_err
                    app.state.rg_governor = RedisResourceGovernor(policy_loader=rg_loader)
                    logger.info("ResourceGovernor initialized (redis backend)")
                else:
                    app.state.rg_governor = MemoryResourceGovernor(policy_loader=rg_loader)
                    logger.info("ResourceGovernor initialized (memory backend)")
            except _STARTUP_GUARD_EXCEPTIONS as _rg_gov_err:
                logger.warning(f"ResourceGovernor initialization failed/skipped: {_rg_gov_err}")
            try:
                snap = rg_loader.get_snapshot()
                app.state.rg_policy_version = int(getattr(snap, "version", 0) or 0)
                app.state.rg_policy_count = len(getattr(snap, "policies", {}) or {})
            except _STARTUP_GUARD_EXCEPTIONS:
                app.state.rg_policy_version = 0
                app.state.rg_policy_count = 0

            # Keep version fresh on reloads
            try:

                def _on_rg_change(snap):
                    try:
                        app.state.rg_policy_version = int(getattr(snap, "version", 0) or 0)
                        app.state.rg_policy_count = len(getattr(snap, "policies", {}) or {})
                    except _STARTUP_GUARD_EXCEPTIONS:
                        pass

                rg_loader.add_on_change(_on_rg_change)
            except _STARTUP_GUARD_EXCEPTIONS:
                pass

            # Best-effort audit: warn on API routes not covered by RG route_map.
            try:
                def _should_audit_rg_route_map() -> bool:
                    return _shared_is_truthy(os.getenv("RG_ROUTE_MAP_AUDIT", "true"))

                def _route_map_matches(path: str, by_path: dict) -> bool:
                    for pat in by_path:
                        pat = str(pat)
                        if pat.endswith("*"):
                            if path.startswith(pat[:-1]):
                                return True
                        elif path == pat:
                            return True
                    return False

                if _should_audit_rg_route_map():
                    snap = rg_loader.get_snapshot()
                    route_map = getattr(snap, "route_map", {}) or {}
                    by_path = dict(route_map.get("by_path") or {})
                    by_tag = dict(route_map.get("by_tag") or {})
                    if by_path or by_tag:
                        skip_prefixes = ("/docs", "/openapi.json", "/redoc", "/static", "/favicon.ico")
                        missing: list[tuple[str, list[str]]] = []
                        seen_paths: set[str] = set()
                        for route in getattr(app, "routes", []):
                            path = getattr(route, "path", None)
                            if not path or path in seen_paths:
                                continue
                            if path.startswith(skip_prefixes):
                                continue
                            # Focus on API-ish endpoints and health/setup roots.
                            if not (
                                path.startswith("/api/")
                                or path.startswith("/v1/")
                                or path.startswith("/health")
                                or path.startswith("/readyz")
                                or path.startswith("/metrics")
                                or path.startswith("/setup")
                            ):
                                continue
                            if _route_map_matches(path, by_path):
                                seen_paths.add(path)
                                continue
                            tags = list(getattr(route, "tags", []) or [])
                            if tags and any(t in by_tag for t in tags):
                                seen_paths.add(path)
                                continue
                            missing.append((path, tags))
                            seen_paths.add(path)
                        if missing:
                            sample = ", ".join(
                                f"{p} (tags={tags})" for p, tags in missing[:10]
                            )
                            logger.warning(
                                f"RG route_map missing coverage for {len(missing)} routes; sample: {sample}"
                            )
            except _IMPORT_EXCEPTIONS as _rg_audit_err:
                logger.debug(f"RG route_map audit skipped: {_rg_audit_err}")
        except _IMPORT_EXCEPTIONS as _rg_err:
            logger.warning(f"ResourceGovernor policy loader initialization skipped: {_rg_err}")
        try:
            from tldw_Server_API.app.core.config import (
                rg_backend as _rg_backend_sel,
            )
            from tldw_Server_API.app.core.config import (
                rg_enabled as _rg_enabled_flag,
            )
            from tldw_Server_API.app.core.config import (
                rg_policy_path as _rg_policy_path,
            )
            from tldw_Server_API.app.core.config import (
                rg_policy_store as _rg_store_sel,
            )

            if bool(_rg_enabled_flag(False)) and getattr(app.state, "rg_governor", None) is None:
                logger.warning(
                    "ResourceGovernor enabled but not initialized; rate limiting will fail closed. "
                    f"policy_path={_rg_policy_path()} backend={_rg_backend_sel()} "
                    f"store={_rg_store_sel()} cwd={os.getcwd()}"
                )
        except _IMPORT_EXCEPTIONS as _rg_warn_err:
            logger.debug(f"ResourceGovernor init warning skipped: {_rg_warn_err}")

        # Production hard-stop: require RG coverage/availability for auth endpoints.
        from tldw_Server_API.app.core.AuthNZ.rg_startup_guard import (
            validate_auth_rg_startup_guards,
        )

        validate_auth_rg_startup_guards(app)

        # Initialize session manager
        from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

        session_manager = await get_session_manager()
        logger.info("App Startup: Session manager initialized")

        try:
            from tldw_Server_API.app.core.AuthNZ.alerting import get_security_alert_dispatcher

            dispatcher = get_security_alert_dispatcher()
            dispatcher.validate_configuration()
            if dispatcher.enabled:
                logger.info("App Startup: Security alert configuration validated")
        except ValueError as config_error:
            logger.exception(f"App Startup: Security alert configuration invalid: {config_error}")
            raise
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.exception(f"App Startup: Security alert validation / auth services init failed: {exc}")
        # Continue startup even if auth services fail (for backward compatibility)

    # Startup: Warm ChaChaNotes to remove request-path blocking for the default user
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
            reset_chacha_shutdown_state,
            warm_chacha_db_for_user,
        )
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_auth_settings
        from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode

        reset_chacha_shutdown_state()
        if is_single_user_mode():
            _auth_settings = _get_auth_settings()
            _single_user_id = int(getattr(_auth_settings, "SINGLE_USER_FIXED_ID", 1))
            asyncio.create_task(warm_chacha_db_for_user(_single_user_id, str(_single_user_id)))
            logger.info(f"App Startup: scheduled ChaChaNotes warm-up for single-user id={_single_user_id}")
        else:
            logger.debug("ChaChaNotes warm-up skipped (multi-user mode)")
    except _STARTUP_GUARD_EXCEPTIONS as _warm_err:
        # warm-up is best-effort
        logger.warning(f"ChaChaNotes warm-up scheduling failed: {_warm_err}")

    # Startup: Validate privilege catalog and route metadata (fail fast on mismatch)
    try:
        from tldw_Server_API.app.core.PrivilegeMaps.startup import validate_privilege_metadata_on_startup

        validate_privilege_metadata_on_startup(app)
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.exception(f"App Startup: Privilege metadata validation failed: {exc}")
        raise

    # Heavy initializations: helpers and shared runner to avoid duplication
    # Ensure resources are bound in the enclosing scope so shutdown can detect them
    mcp_server = None
    provider_manager = None
    request_queue = None

    async def _init_local_llm_manager(*, deferred: bool) -> None:
        try:
            if getattr(app.state, "llm_manager", None) is not None:
                return

            # Lazy skip: only initialize if local-LLM routes are available in the current policy.
            try:
                _llm_routes_enabled = route_enabled("llamacpp") or route_enabled("llm")
            except _STARTUP_GUARD_EXCEPTIONS:
                _llm_routes_enabled = True
            if not _llm_routes_enabled:
                logger.debug("Local LLM inference manager skipped (llm/llamacpp routes disabled)")
                return

            from tldw_Server_API.app.core.config import get_llamacpp_handler_config
            from tldw_Server_API.app.core.Local_LLM import LLMInferenceManager, LLMManagerConfig

            _llama_cfg = get_llamacpp_handler_config()
            cfg_kwargs = {}
            if _llama_cfg:
                cfg_kwargs["llamacpp"] = _llama_cfg

            manager = await asyncio.to_thread(LLMInferenceManager, LLMManagerConfig(**cfg_kwargs))
            app.state.llm_manager = manager
            try:
                from tldw_Server_API.app.api.v1.endpoints import llamacpp as _llamacpp_module

                _llamacpp_module.llm_manager = manager
            except _STARTUP_GUARD_EXCEPTIONS as _llm_ep_err:
                logger.debug(f"LLM manager initialized but not injected into llama.cpp endpoints: {_llm_ep_err}")
            logger.info(
                ("Deferred startup: " if deferred else "App Startup: ")
                + "Local LLM inference manager initialized"
            )
        except _STARTUP_GUARD_EXCEPTIONS as _llm_init_err:
            if deferred:
                logger.debug(f"Deferred startup: local LLM manager skipped/failed: {_llm_init_err}")
            else:
                logger.warning(
                    "Local LLM inference manager not initialized; llama.cpp endpoints will return 503: "
                    f"{_llm_init_err}"
                )

    async def _init_mcp_server(*, deferred: bool) -> None:
        nonlocal mcp_server
        try:
            from tldw_Server_API.app.core.MCP_unified import get_mcp_server

            mcp_server = get_mcp_server()
            if not deferred:
                logger.info("App Startup: Initializing MCP Unified server...")
            await mcp_server.initialize()
            logger.info(
                ("Deferred startup: " if deferred else "App Startup: ") + "MCP Unified server initialized successfully"
            )
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: MCP Unified server skipped/failed: {e}")
            else:
                logger.exception(f"App Startup: Failed to initialize MCP Unified server: {e}")
                logger.warning("Ensure MCP_JWT_SECRET and MCP_API_KEY_SALT environment variables are set")

    async def _init_provider_manager(*, deferred: bool) -> None:
        nonlocal provider_manager
        try:
            from tldw_Server_API.app.core.Chat.provider_manager import initialize_provider_manager
            from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

            providers = get_registry().list_providers()
            provider_manager = initialize_provider_manager(
                providers, primary_provider=providers[0] if providers else None
            )
            await provider_manager.start_health_checks()
            if deferred:
                logger.info(f"Deferred startup: Provider manager ready ({len(providers)} providers)")
            else:
                logger.info(f"App Startup: Provider manager initialized with {len(providers)} providers")
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: provider manager skipped/failed: {e}")
            else:
                logger.exception(f"App Startup: Failed to initialize provider manager: {e}")

    async def _init_request_queue(*, deferred: bool) -> None:
        nonlocal request_queue
        try:
            from tldw_Server_API.app.core.Chat.request_queue import initialize_request_queue
            from tldw_Server_API.app.core.config import load_comprehensive_config

            cfg = load_comprehensive_config()
            chat_cfg = {}
            if cfg and cfg.has_section("Chat-Module"):
                chat_cfg = dict(cfg.items("Chat-Module"))
            queued_execution_enabled = False
            try:
                env_queued = os.getenv("CHAT_QUEUED_EXECUTION")
                if env_queued is not None:
                    queued_execution_enabled = _shared_is_truthy(env_queued)
                else:
                    queued_execution_enabled = _shared_is_truthy(
                        str(chat_cfg.get("queued_execution", "False"))
                    )
            except _STARTUP_GUARD_EXCEPTIONS:
                queued_execution_enabled = False
            if queued_execution_enabled:
                request_queue = initialize_request_queue(
                    max_queue_size=int(chat_cfg.get("max_queue_size", 100)),
                    max_concurrent=int(chat_cfg.get("max_concurrent_requests", 10)),
                    global_rate_limit=int(chat_cfg.get("rate_limit_per_minute", 60)),
                    per_client_rate_limit=int(chat_cfg.get("rate_limit_per_conversation_per_minute", 20)),
                )
                await request_queue.start(num_workers=4)
                if deferred:
                    logger.info("Deferred startup: Request queue online")
                else:
                    logger.info("App Startup: Request queue initialized with 4 workers")
            else:
                if not deferred:
                    logger.info("App Startup: Request queue disabled (QUEUED_EXECUTION is off)")
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: request queue skipped/failed: {e}")
            else:
                logger.exception(f"App Startup: Failed to initialize request queue: {e}")

    async def _init_rate_limiter(*, deferred: bool) -> None:
        try:
            from tldw_Server_API.app.core.config import rg_enabled as _rg_enabled_flag
            if _rg_enabled_flag(False):
                logger.info(
                    ("Deferred startup: " if deferred else "App Startup: ")
                    + "Rate limiter skipped (RG enabled)"
                )
                return
            from tldw_Server_API.app.core.Chat.rate_limiter import RateLimitConfig, initialize_rate_limiter
            from tldw_Server_API.app.core.config import load_comprehensive_config

            cfg = load_comprehensive_config()
            chat_cfg = {}
            if cfg and cfg.has_section("Chat-Module"):
                chat_cfg = dict(cfg.items("Chat-Module"))
            rl_cfg = RateLimitConfig(
                global_rpm=int(chat_cfg.get("rate_limit_per_minute", 60)),
                per_user_rpm=int(chat_cfg.get("rate_limit_per_user_per_minute", 20)),
                per_conversation_rpm=int(chat_cfg.get("rate_limit_per_conversation_per_minute", 10)),
                per_user_tokens_per_minute=int(chat_cfg.get("rate_limit_tokens_per_minute", 10000)),
            )
            initialize_rate_limiter(rl_cfg)
            logger.info(
                ("Deferred startup: " if deferred else "App Startup: ")
                + "Rate limiter "
                + ("online" if deferred else "initialized")
            )
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: rate limiter skipped/failed: {e}")
            else:
                logger.exception(f"App Startup: Failed to initialize rate limiter: {e}")

    async def _init_tts_service(*, deferred: bool) -> None:
        try:
            from tldw_Server_API.app.core.config import load_comprehensive_config_with_tts
            from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
            from tldw_Server_API.app.core.TTS.voice_manager import init_voice_manager

            cfg_obj = load_comprehensive_config_with_tts()
            tts_cfg_dict = cfg_obj.get_tts_config() if hasattr(cfg_obj, "get_tts_config") else None
            await get_tts_service_v2(config=tts_cfg_dict)
            await init_voice_manager()
            logger.info(
                ("Deferred startup: " if deferred else "App Startup: ")
                + "TTS service "
                + ("ready" if deferred else "initialized successfully")
            )
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: TTS skipped/failed: {e}")
            else:
                logger.exception(f"App Startup: Failed to initialize TTS service: {e}")
                logger.warning("TTS functionality will be unavailable")

    async def _init_chunking_templates(*, deferred: bool) -> None:
        try:
            from tldw_Server_API.app.core.Chunking.template_initialization import ensure_templates_initialized

            ok = ensure_templates_initialized()
            if ok:
                logger.info(
                    ("Deferred startup: " if deferred else "App Startup: ")
                    + "Chunking templates "
                    + ("ready" if deferred else "initialized successfully")
                )
            else:
                if deferred:
                    logger.debug("Deferred startup: Chunking templates incomplete")
                else:
                    logger.warning("App Startup: Chunking templates initialization incomplete")
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: chunking templates skipped/failed: {e}")
            else:
                logger.exception(f"App Startup: Failed to initialize chunking templates: {e}")

    async def _init_embeddings_dim_check(*, deferred: bool) -> None:
        try:
            enabled = os.getenv("EMBEDDINGS_STARTUP_DIM_CHECK_ENABLED", "false").lower() in {
                "true",
                "1",
                "yes",
                "y",
                "on",
            }
            if not enabled:
                return
            strict_mode = os.getenv("EMBEDDINGS_DIM_CHECK_STRICT", "false").lower() in {"true", "1", "yes", "y", "on"}
            if not deferred:
                logger.info("App Startup: Running embeddings dimension sanity check (opt-in)")
            from pathlib import Path as _Path

            from tldw_Server_API.app.core.config import settings as _emb_settings
            from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

            def _check_user(user_id: str) -> list[tuple[str, int, int, str]]:
                mms: list[tuple[str, int, int, str]] = []
                mgr = ChromaDBManager(user_id=user_id, user_embedding_config=_emb_settings)
                client = getattr(mgr, "client", None)
                list_fn = getattr(client, "list_collections", None)
                collections = list_fn() if callable(list_fn) else []
                for col in collections:
                    try:
                        name = getattr(col, "name", None) or (col.get("name") if isinstance(col, dict) else None)
                        if not name:
                            continue
                        get_fn = getattr(client, "get_collection", None)
                        c = get_fn(name=name) if callable(get_fn) else col
                        meta = getattr(c, "metadata", None) or {}
                        expected = None
                        if isinstance(meta, dict) and meta.get("embedding_dimension"):
                            try:
                                expected = int(meta.get("embedding_dimension"))
                            except _STARTUP_GUARD_EXCEPTIONS:
                                expected = None
                        actual = None
                        if hasattr(c, "get") and callable(c.get):
                            try:
                                res = c.get(limit=1, include=["embeddings"])
                                embs = res.get("embeddings") if isinstance(res, dict) else None
                                if embs and len(embs) > 0:
                                    first = embs[0]
                                    if first and hasattr(first, "__len__"):
                                        actual = len(first)
                            except _STARTUP_GUARD_EXCEPTIONS:
                                pass
                        if expected is not None and actual is not None and expected != actual:
                            mms.append((name, expected, actual, user_id))
                    except _STARTUP_GUARD_EXCEPTIONS:
                        pass
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    mgr.close()
                return mms

            auth_mode = str(_emb_settings.get("AUTH_MODE", os.getenv("AUTH_MODE", "single_user")))
            mismatches: list[tuple[str, int, int, str]] = []
            if auth_mode == "multi_user":
                base: _Path = _emb_settings.get("USER_DB_BASE_DIR")
                if base and _Path(base).exists():
                    for entry in _Path(base).iterdir():
                        if entry.is_dir():
                            user_id = entry.name
                            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                                mismatches.extend(_check_user(user_id))
                else:
                    if not deferred:
                        logger.warning(
                            "Embeddings dimension check: USER_DB_BASE_DIR missing or does not exist in multi_user mode"
                        )
            else:
                user_id = str(_emb_settings.get("SINGLE_USER_FIXED_ID", "1"))
                mismatches.extend(_check_user(user_id))

            if mismatches:
                for n, e, a, u in mismatches:
                    logger.error(
                        ("Deferred startup: " if deferred else "")
                        + f"Embeddings dimension mismatch{' (deferred)' if deferred else ' at startup'} (user={u}) in collection '{n}': expected={e}, actual={a}"
                    )
                if strict_mode:
                    raise RuntimeError("EMBEDDINGS_STARTUP_DIM_CHECK_FAILED")
            else:
                logger.info(
                    ("Deferred startup: " if deferred else "")
                    + (
                        "Embeddings dimension check OK"
                        if deferred
                        else "Embeddings dimension sanity check: OK (no mismatches)"
                    )
                )
        except _STARTUP_GUARD_EXCEPTIONS as e:
            if deferred:
                logger.debug(f"Deferred startup: embeddings dimension check skipped/failed: {e}")
            else:
                logger.exception(f"Embeddings dimension sanity check failed: {e}")
                # Do not raise except in strict mode (handled above)

    async def _run_heavy_initializations(*, deferred: bool) -> None:
        if deferred:
            logger.info("Deferred startup: beginning non-critical initializations in background")
        # Local LLM manager
        await _init_local_llm_manager(deferred=deferred)
        # MCP Unified server
        await _init_mcp_server(deferred=deferred)
        # Provider manager
        await _init_provider_manager(deferred=deferred)
        # Request queue
        await _init_request_queue(deferred=deferred)
        # Rate limiter
        await _init_rate_limiter(deferred=deferred)
        # TTS service
        await _init_tts_service(deferred=deferred)
        # Chunking templates
        await _init_chunking_templates(deferred=deferred)
        # Embeddings dimension check
        await _init_embeddings_dim_check(deferred=deferred)
        if deferred:
            logger.info("Deferred startup: completed non-critical initializations")

    # Load persona archetypes and MCP server catalog (lightweight YAML reads).
    try:
        from pathlib import Path as _ArchPath

        from tldw_Server_API.app.core.Persona.archetype_loader import load_archetypes_from_directory
        from tldw_Server_API.app.core.MCP_unified.catalog_loader import load_mcp_catalog

        _config_dir = _ArchPath(__file__).resolve().parent.parent / "Config_Files"
        load_archetypes_from_directory(_config_dir / "persona_archetypes")
        load_mcp_catalog(_config_dir / "mcp_server_catalog.yaml")
    except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _archetype_err:
        logger.debug("Archetype/catalog loading skipped: {}", _archetype_err)

    # Initialize Chat Module Components (single log retained)
    logger.info("App Startup: Initializing Chat module components...")

    # Run heavy initializations synchronously or schedule in background
    if _defer_heavy:
        import asyncio as _asyncio

        try:
            app.state.bg_tasks["deferred_startup"] = _asyncio.create_task(_run_heavy_initializations(deferred=True))
        except _STARTUP_GUARD_EXCEPTIONS as _ds_e:
            logger.debug(f"Failed to schedule deferred startup task: {_ds_e}")
    else:
        await _run_heavy_initializations(deferred=False)

    # Note: Audit service now uses dependency injection
    # No need to initialize globally - use get_audit_service_for_user dependency in endpoints
    logger.info("App Startup: Audit service available via dependency injection")

    # Start background workers: ephemeral collections cleanup, core Jobs (chatbooks), audio Jobs (MVP), claims rebuild
    cleanup_task = None
    chatbooks_cleanup_task = None
    core_jobs_task = None
    audiobook_jobs_task = None
    files_jobs_task = None
    data_tables_jobs_task = None
    prompt_studio_jobs_task = None
    privilege_snapshot_task = None
    audio_jobs_task = None
    presentation_render_jobs_task = None
    media_ingest_jobs_task = None
    media_ingest_heavy_jobs_task = None
    reading_digest_jobs_task = None
    study_pack_jobs_task = None
    study_suggestions_jobs_task = None
    companion_reflection_jobs_task = None
    reminder_jobs_task = None
    admin_backup_jobs_task = None
    admin_byok_validation_jobs_task = None
    connectors_jobs_task = None
    evals_abtest_jobs_task = None
    admin_maintenance_rotation_jobs_task = None
    recipe_run_jobs_task = None
    jobs_metrics_reconcile_task = None
    loop_lag_task = None
    jobs_crypto_rotate_task = None
    jobs_webhooks_task = None
    meetings_webhook_dlq_task = None
    workflows_dlq_task = None
    workflows_gc_task = None
    workflows_maint_task = None
    jobs_integrity_task = None
    _tts_history_cleanup_task = None
    jobs_notifications_bridge_task = None
    chatbooks_cleanup_stop_event = None
    audiobook_jobs_stop_event = None
    core_jobs_stop_event = None
    files_jobs_stop_event = None
    data_tables_jobs_stop_event = None
    prompt_studio_jobs_stop_event = None
    privilege_snapshot_stop_event = None
    embeddings_compactor_stop_event = None
    audio_jobs_stop_event = None
    presentation_render_jobs_stop_event = None
    media_ingest_jobs_stop_event = None
    media_ingest_heavy_jobs_stop_event = None
    reading_digest_jobs_stop_event = None
    study_pack_jobs_stop_event = None
    study_suggestions_jobs_stop_event = None
    companion_reflection_jobs_stop_event = None
    reminder_jobs_stop_event = None
    admin_backup_jobs_stop_event = None
    admin_byok_validation_jobs_stop_event = None
    admin_maintenance_rotation_jobs_stop_event = None
    connectors_jobs_stop_event = None
    evals_abtest_jobs_stop_event = None
    recipe_run_jobs_stop_event = None
    jobs_metrics_stop_event = None
    jobs_metrics_reconcile_stop = None
    loop_lag_stop_event = None
    jobs_crypto_rotate_stop_event = None
    jobs_webhooks_stop_event = None
    meetings_webhook_dlq_stop_event = None
    workflows_dlq_stop_event = None
    workflows_gc_stop_event = None
    workflows_maint_stop_event = None
    jobs_integrity_stop_event = None
    _tts_history_cleanup_stop_event = None
    claims_task = None
    jobs_metrics_task = None
    reminders_sched_task = None
    owned_job_pollers: list[_ManagedJobPoller] = []
    _publish_shutdown_job_poller_inventory(app, owned_job_pollers)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.core.config import legacy_get as _legacy_get
        from tldw_Server_API.app.core.config import settings as _app_settings
        from tldw_Server_API.app.core.DB_Management.DB_Manager import (
            create_evaluations_database as _create_evals_db,
        )
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DBP
        from tldw_Server_API.app.core.RAG.rag_service.vector_stores import (
            create_from_settings_for_user as _create_vs_from_settings,
        )

        def _env_flag(key: str, default: bool) -> bool:
            raw = _os.getenv(key)
            if raw is None or str(raw).strip() == "":
                return bool(default)
            return str(raw).strip().lower() in {"true", "1", "yes", "y", "on"}

        def _route_default(route_key: str, *, default_stable: bool = True) -> bool:
            try:
                if globals().get("_TEST_MODE"):
                    return False
            except _STARTUP_GUARD_EXCEPTIONS:
                return False
            try:
                return bool(route_enabled(route_key, default_stable=default_stable))
            except _STARTUP_GUARD_EXCEPTIONS:
                return bool(default_stable)

        _sidecar_mode = _env_flag("TLDW_WORKERS_SIDECAR_MODE", False)

        def _should_start_worker(flag_key: str, route_key: str, *, default_stable: bool = True) -> bool:
            if _sidecar_mode:
                return False
            return _env_flag(flag_key, _route_default(route_key, default_stable=default_stable))

        if _sidecar_mode:
            logger.info("Sidecar worker mode enabled; in-process Jobs workers are disabled")

        # Use per-user evaluations DB for cleanup; default to single-user ID
        _single_uid = int(_app_settings.get("SINGLE_USER_FIXED_ID", "1"))
        _db_path = str(_DBP.get_evaluations_db_path(_single_uid))
        # Read settings
        _enabled = bool(_app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
        _interval_sec = int(_app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", 1800))

        async def _ephemeral_cleanup_loop():
            logger.info(f"Starting ephemeral collections cleanup worker (every {_interval_sec}s)")
            # Use backend-aware factory so Postgres content backend is honored
            db = _create_evals_db(db_path=_db_path)
            adapter = _create_vs_from_settings(_app_settings, str(_app_settings.get("SINGLE_USER_FIXED_ID", "1")))
            await adapter.initialize()
            while True:
                try:
                    # Re-read settings each cycle
                    _enabled_dyn = bool(_app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
                    _interval_dyn = int(_app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", _interval_sec))
                    if not _enabled_dyn:
                        await _asyncio.sleep(_interval_sec)
                        continue
                    expired = db.list_expired_ephemeral_collections()
                    if expired:
                        deleted = 0
                        for cname in expired:
                            try:
                                await adapter.delete_collection(cname)
                                db.mark_ephemeral_deleted(cname)
                                deleted += 1
                            except _STARTUP_GUARD_EXCEPTIONS as ce:
                                logger.warning(f"Ephemeral cleanup: failed to delete {cname}: {ce}")
                        if deleted:
                            logger.info(f"Ephemeral cleanup: deleted {deleted}/{len(expired)} expired collections")
                except _STARTUP_GUARD_EXCEPTIONS as ce:
                    logger.warning(f"Ephemeral cleanup loop error: {ce}")
                await _asyncio.sleep(_interval_dyn)

        if _enabled:
            cleanup_task = _asyncio.create_task(_ephemeral_cleanup_loop())
        else:
            logger.info("Ephemeral cleanup worker disabled by settings")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start ephemeral cleanup worker: {e}")

    # Chatbooks cleanup worker (scheduled retention cleanup)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.chatbooks_cleanup_service import (
            run_chatbooks_cleanup_loop as _run_chatbooks_cleanup,
        )

        _interval_sec = int(_os.getenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "0") or "0")
        if _interval_sec > 0:
            chatbooks_cleanup_stop_event = _asyncio.Event()
            chatbooks_cleanup_task = _asyncio.create_task(_run_chatbooks_cleanup(chatbooks_cleanup_stop_event))
            logger.info("Chatbooks cleanup worker started")
        else:
            logger.info("Chatbooks cleanup worker disabled by settings")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start chatbooks cleanup worker: {e}")

    # Storage cleanup worker (expired files, trash purge)
    storage_cleanup_service = None
    try:
        import os as _os

        from tldw_Server_API.app.services.storage_cleanup_service import get_cleanup_service as _get_storage_cleanup

        _storage_cleanup_default = "false" if globals().get("_TEST_MODE") else "true"
        _storage_cleanup_enabled = _os.getenv("STORAGE_CLEANUP_ENABLED", _storage_cleanup_default).lower() in {
            "true", "1", "yes", "y", "on"
        }
        if _storage_cleanup_enabled:
            storage_cleanup_service = _get_storage_cleanup()
            await storage_cleanup_service.start()
            logger.info("Storage cleanup worker started")
        else:
            logger.info("Storage cleanup worker disabled by settings")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start storage cleanup worker: {e}")

    # Core Jobs worker (Chatbooks, if backend=core)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.core_jobs_worker import run_chatbooks_core_jobs_worker as _run_cb_jobs

        _backend = (_os.getenv("CHATBOOKS_JOBS_BACKEND") or _os.getenv("TLDW_JOBS_BACKEND") or "").lower()
        _is_core = (_backend == "core") or (not _backend)
        _core_worker_enabled = _os.getenv("CHATBOOKS_CORE_WORKER_ENABLED", "true").lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if _sidecar_mode:
            _core_worker_enabled = False
        if _is_core and _core_worker_enabled:
            core_jobs_stop_event = _asyncio.Event()
            core_jobs_task = _asyncio.create_task(_run_cb_jobs(core_jobs_stop_event))
            logger.info("Core Jobs worker (Chatbooks) started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="core_jobs_task",
                task=core_jobs_task,
                stop_event=core_jobs_stop_event,
            )
        else:
            logger.info("Core Jobs worker (Chatbooks) disabled by backend selection or flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start core Jobs worker (Chatbooks): {e}")

    # File Artifacts Jobs worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.core.File_Artifacts.jobs_worker import (
            run_file_artifacts_jobs_worker as _run_files_jobs,
        )

        _enabled = _should_start_worker("FILES_JOBS_WORKER_ENABLED", "files")
        if _enabled:
            files_jobs_stop_event = _asyncio.Event()
            files_jobs_task = _asyncio.create_task(_run_files_jobs(files_jobs_stop_event))
            logger.info("File Artifacts Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="files_jobs_task",
                task=files_jobs_task,
                stop_event=files_jobs_stop_event,
            )
        else:
            logger.info("File Artifacts Jobs worker disabled by flag (FILES_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        # startup/shutdown guard; log and continue
        logger.warning(f"Failed to start File Artifacts Jobs worker: {e}")

    # Data Tables Jobs worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.core.Data_Tables.jobs_worker import (
            run_data_tables_jobs_worker as _run_data_tables_jobs,
        )

        _enabled = _should_start_worker("DATA_TABLES_JOBS_WORKER_ENABLED", "data-tables")
        if _enabled:
            data_tables_jobs_stop_event = _asyncio.Event()
            data_tables_jobs_task = _asyncio.create_task(_run_data_tables_jobs(data_tables_jobs_stop_event))
            logger.info("Data Tables Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="data_tables_jobs_task",
                task=data_tables_jobs_task,
                stop_event=data_tables_jobs_stop_event,
            )
        else:
            logger.info("Data Tables Jobs worker disabled by flag (DATA_TABLES_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        # startup/shutdown guard; log and continue
        logger.warning(f"Failed to start Data Tables Jobs worker: {e}")

    # Prompt Studio Jobs worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services.jobs_worker import (
            run_prompt_studio_jobs_worker as _run_prompt_studio_jobs,
        )

        _enabled = _should_start_worker("PROMPT_STUDIO_JOBS_WORKER_ENABLED", "prompt-studio")
        if _enabled:
            prompt_studio_jobs_stop_event = _asyncio.Event()
            prompt_studio_jobs_task = _asyncio.create_task(_run_prompt_studio_jobs(prompt_studio_jobs_stop_event))
            logger.info("Prompt Studio Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="prompt_studio_jobs_task",
                task=prompt_studio_jobs_task,
                stop_event=prompt_studio_jobs_stop_event,
            )
        else:
            logger.info("Prompt Studio Jobs worker disabled by flag (PROMPT_STUDIO_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        # startup/shutdown guard; log and continue
        logger.warning(f"Failed to start Prompt Studio Jobs worker: {e}")

    # Study-pack Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("STUDY_PACK_JOBS_WORKER_ENABLED", "flashcards")
        if _enabled:
            from tldw_Server_API.app.services.study_pack_jobs_worker import (
                run_study_pack_jobs_worker as _run_study_pack_jobs,
            )

            study_pack_jobs_stop_event = _asyncio.Event()
            study_pack_jobs_task = _asyncio.create_task(_run_study_pack_jobs(study_pack_jobs_stop_event))
            logger.info("Study-pack Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="study_pack_jobs_task",
                task=study_pack_jobs_task,
                stop_event=study_pack_jobs_stop_event,
            )
        else:
            logger.info("Study-pack Jobs worker disabled by flag (STUDY_PACK_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Study-pack Jobs worker: {e}")

    # Study-suggestions Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED", "study-suggestions")
        if _enabled:
            from tldw_Server_API.app.services.study_suggestions_jobs_worker import (
                run_study_suggestions_jobs_worker as _run_study_suggestions_jobs,
            )

            study_suggestions_jobs_stop_event = _asyncio.Event()
            study_suggestions_jobs_task = _asyncio.create_task(
                _run_study_suggestions_jobs(study_suggestions_jobs_stop_event)
            )
            logger.info("Study-suggestions Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="study_suggestions_jobs_task",
                task=study_suggestions_jobs_task,
                stop_event=study_suggestions_jobs_stop_event,
            )
        else:
            logger.info("Study-suggestions Jobs worker disabled by flag (STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Study-suggestions Jobs worker: {e}")

    # Privilege snapshot worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.privilege_snapshot_worker import (
            run_privilege_snapshot_worker as _run_priv_snapshot,
        )

        _enabled = _should_start_worker("PRIVILEGE_SNAPSHOT_WORKER_ENABLED", "privileges")
        if _enabled:
            privilege_snapshot_stop_event = _asyncio.Event()
            privilege_snapshot_task = _asyncio.create_task(_run_priv_snapshot(privilege_snapshot_stop_event))
            logger.info("Privilege snapshot worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="privilege_snapshot_task",
                task=privilege_snapshot_task,
                stop_event=privilege_snapshot_stop_event,
            )
        else:
            logger.info("Privilege snapshot worker disabled by flag (PRIVILEGE_SNAPSHOT_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        # startup/shutdown guard; log and continue
        logger.warning(f"Failed to start privilege snapshot worker: {e}")

    # Embeddings Vector Compactor (soft-delete propagation)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.core.Embeddings.services.vector_compactor import run as _run_vec_compactor

        _enabled = _os.getenv("EMBEDDINGS_COMPACTOR_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _enabled:
            embeddings_compactor_stop_event = _asyncio.Event()
            embeddings_compactor_task = _asyncio.create_task(_run_vec_compactor(embeddings_compactor_stop_event))
            logger.info("Embeddings Vector Compactor started with explicit stop_event signal")
        else:
            logger.info("Embeddings Vector Compactor disabled by flag (EMBEDDINGS_COMPACTOR_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Embeddings Vector Compactor: {e}")

    # WebSub lease renewal worker
    try:
        import asyncio as _asyncio
        import os as _os

        _websub_enabled = _os.getenv("WEBSUB_CALLBACK_BASE_URL", "").strip() and _should_start_worker(
            "WEBSUB_RENEWAL_WORKER_ENABLED", "collections-websub"
        )
        if _websub_enabled:
            from tldw_Server_API.app.core.Watchlists.websub import websub_renewal_loop

            websub_renewal_task = _asyncio.create_task(websub_renewal_loop())
            logger.info("WebSub lease renewal worker started")
        else:
            logger.info("WebSub renewal worker disabled (no WEBSUB_CALLBACK_BASE_URL or flag off)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start WebSub renewal worker: {e}")

    # Audio Jobs worker (MVP)
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("AUDIO_JOBS_WORKER_ENABLED", "audio-jobs")
        if _enabled:
            from tldw_Server_API.app.services.audio_jobs_worker import run_audio_jobs_worker as _run_audio_jobs

            audio_jobs_stop_event = _asyncio.Event()
            audio_jobs_task = _asyncio.create_task(_run_audio_jobs(audio_jobs_stop_event))
            logger.info("Audio Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="audio_jobs_task",
                task=audio_jobs_task,
                stop_event=audio_jobs_stop_event,
            )
        else:
            logger.info("Audio Jobs worker disabled by flag (AUDIO_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Audio Jobs worker: {e}")

    # Audiobook Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("AUDIOBOOK_JOBS_WORKER_ENABLED", "audiobooks")
        if _enabled:
            from tldw_Server_API.app.services.audiobook_jobs_worker import (
                run_audiobook_jobs_worker as _run_audiobook_jobs,
            )

            audiobook_jobs_stop_event = _asyncio.Event()
            audiobook_jobs_task = _asyncio.create_task(_run_audiobook_jobs(audiobook_jobs_stop_event))
            logger.info("Audiobook Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="audiobook_jobs_task",
                task=audiobook_jobs_task,
                stop_event=audiobook_jobs_stop_event,
            )
        else:
            logger.info("Audiobook Jobs worker disabled by flag (AUDIOBOOK_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Audiobook Jobs worker: {e}")

    # Presentation Render Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("PRESENTATION_RENDER_JOBS_WORKER_ENABLED", "slides")
        if _enabled:
            from tldw_Server_API.app.services.presentation_render_jobs_worker import (
                run_presentation_render_jobs_worker as _run_presentation_render_jobs,
            )

            presentation_render_jobs_stop_event = _asyncio.Event()
            presentation_render_jobs_task = _asyncio.create_task(
                _run_presentation_render_jobs(presentation_render_jobs_stop_event)
            )
            logger.info("Presentation Render Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="presentation_render_jobs_task",
                task=presentation_render_jobs_task,
                stop_event=presentation_render_jobs_stop_event,
            )
        else:
            logger.info(
                "Presentation Render Jobs worker disabled by flag (PRESENTATION_RENDER_JOBS_WORKER_ENABLED)"
            )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Presentation Render Jobs worker: {e}")

    # Media Ingest Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("MEDIA_INGEST_JOBS_WORKER_ENABLED", "media")
        if _enabled:
            from tldw_Server_API.app.services.media_ingest_jobs_worker import (
                run_media_ingest_jobs_worker as _run_media_jobs,
            )

            media_ingest_jobs_stop_event = _asyncio.Event()
            media_ingest_jobs_task = _asyncio.create_task(_run_media_jobs(media_ingest_jobs_stop_event))
            logger.info("Media Ingest Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="media_ingest_jobs_task",
                task=media_ingest_jobs_task,
                stop_event=media_ingest_jobs_stop_event,
            )
        else:
            logger.info("Media Ingest Jobs worker disabled by flag (MEDIA_INGEST_JOBS_WORKER_ENABLED)")

        _heavy_enabled = _should_start_worker(
            "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
            "media-ingest-heavy-jobs",
            default_stable=False,
        )
        if _heavy_enabled:
            from tldw_Server_API.app.services.media_ingest_jobs_worker import (
                run_media_ingest_heavy_jobs_worker as _run_media_heavy_jobs,
            )

            media_ingest_heavy_jobs_stop_event = _asyncio.Event()
            media_ingest_heavy_jobs_task = _asyncio.create_task(
                _run_media_heavy_jobs(media_ingest_heavy_jobs_stop_event)
            )
            logger.info("Media Ingest Heavy Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="media_ingest_heavy_jobs_task",
                task=media_ingest_heavy_jobs_task,
                stop_event=media_ingest_heavy_jobs_stop_event,
            )
        else:
            logger.info(
                "Media Ingest Heavy Jobs worker disabled by flag (MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED)"
            )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Media Ingest Jobs worker: {e}")

    # Reading Digest Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("READING_DIGEST_JOBS_WORKER_ENABLED", "reading")
        if _enabled:
            from tldw_Server_API.app.core.Collections.reading_digest_jobs_worker import (
                run_reading_digest_jobs_worker as _run_reading_digest_jobs,
            )

            reading_digest_jobs_stop_event = _asyncio.Event()
            reading_digest_jobs_task = _asyncio.create_task(
                _run_reading_digest_jobs(reading_digest_jobs_stop_event)
            )
            logger.info("Reading digest Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="reading_digest_jobs_task",
                task=reading_digest_jobs_task,
                stop_event=reading_digest_jobs_stop_event,
            )
        else:
            logger.info("Reading digest Jobs worker disabled by flag (READING_DIGEST_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Reading digest Jobs worker: {e}")

    # Companion reflection Jobs worker
    try:
        import asyncio as _asyncio

        _enabled = _should_start_worker("COMPANION_REFLECTION_JOBS_WORKER_ENABLED", "companion")
        if _enabled:
            from tldw_Server_API.app.core.Personalization.companion_reflection_jobs_worker import (
                run_companion_reflection_jobs_worker as _run_companion_reflection_jobs,
            )

            companion_reflection_jobs_stop_event = _asyncio.Event()
            companion_reflection_jobs_task = _asyncio.create_task(
                _run_companion_reflection_jobs(companion_reflection_jobs_stop_event)
            )
            logger.info("Companion reflection Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="companion_reflection_jobs_task",
                task=companion_reflection_jobs_task,
                stop_event=companion_reflection_jobs_stop_event,
            )
        else:
            logger.info(
                "Companion reflection Jobs worker disabled by flag (COMPANION_REFLECTION_JOBS_WORKER_ENABLED)"
            )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Companion reflection Jobs worker: {e}")

    # Reminder Jobs worker
    try:
        if _sidecar_mode:
            logger.info("Reminder Jobs worker disabled in sidecar mode")
        else:
            from tldw_Server_API.app.services.reminder_jobs_worker import start_reminder_jobs_worker

            reminder_jobs_stop_event = _asyncio.Event()
            reminder_jobs_task = await start_reminder_jobs_worker(stop_event=reminder_jobs_stop_event)
            if reminder_jobs_task:
                logger.info("Reminder Jobs worker started")
                _register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="reminder_jobs_task",
                    task=reminder_jobs_task,
                    stop_event=reminder_jobs_stop_event,
                )
            else:
                logger.info("Reminder Jobs worker disabled (REMINDER_JOBS_WORKER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Reminder Jobs worker: {e}")

    # Admin backup Jobs worker
    try:
        if _sidecar_mode:
            logger.info("Admin backup Jobs worker disabled in sidecar mode")
        else:
            from tldw_Server_API.app.services.admin_backup_jobs_worker import start_admin_backup_jobs_worker

            admin_backup_jobs_stop_event = _asyncio.Event()
            admin_backup_jobs_task = await start_admin_backup_jobs_worker(
                stop_event=admin_backup_jobs_stop_event
            )
            if admin_backup_jobs_task:
                logger.info("Admin backup Jobs worker started")
                _register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="admin_backup_jobs_task",
                    task=admin_backup_jobs_task,
                    stop_event=admin_backup_jobs_stop_event,
                )
            else:
                logger.info("Admin backup Jobs worker disabled (ADMIN_BACKUP_JOBS_WORKER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Admin backup Jobs worker: {e}")

    # Admin BYOK validation Jobs worker
    try:
        if _sidecar_mode:
            logger.info("Admin BYOK validation Jobs worker disabled in sidecar mode")
        else:
            from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
                start_admin_byok_validation_jobs_worker,
            )

            admin_byok_validation_jobs_stop_event = _asyncio.Event()
            admin_byok_validation_jobs_task = await start_admin_byok_validation_jobs_worker(
                stop_event=admin_byok_validation_jobs_stop_event
            )
            if admin_byok_validation_jobs_task:
                logger.info("Admin BYOK validation Jobs worker started")
                _register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="admin_byok_validation_jobs_task",
                    task=admin_byok_validation_jobs_task,
                    stop_event=admin_byok_validation_jobs_stop_event,
                )
            else:
                logger.info(
                    "Admin BYOK validation Jobs worker disabled "
                    "(ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED != true)"
                )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Admin BYOK validation Jobs worker: {e}")

    # Admin maintenance rotation Jobs worker
    try:
        if _sidecar_mode:
            logger.info("Admin maintenance rotation Jobs worker disabled in sidecar mode")
        else:
            from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
                start_admin_maintenance_rotation_jobs_worker,
            )

            admin_maintenance_rotation_jobs_stop_event = _asyncio.Event()
            admin_maintenance_rotation_jobs_task = await start_admin_maintenance_rotation_jobs_worker(
                stop_event=admin_maintenance_rotation_jobs_stop_event
            )
            if admin_maintenance_rotation_jobs_task:
                logger.info("Admin maintenance rotation Jobs worker started")
                _register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="admin_maintenance_rotation_jobs_task",
                    task=admin_maintenance_rotation_jobs_task,
                    stop_event=admin_maintenance_rotation_jobs_stop_event,
                )
            else:
                logger.info(
                    "Admin maintenance rotation Jobs worker disabled "
                    "(ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED != true)"
                )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning("Failed to start Admin maintenance rotation Jobs worker: {}", e)

    # Evaluations recipe-run Jobs worker
    try:
        if _sidecar_mode:
            logger.info("Evaluation recipe-run Jobs worker disabled in sidecar mode")
        else:
            from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker import (
                start_recipe_run_jobs_worker,
            )

            recipe_run_jobs_stop_event = _asyncio.Event()
            recipe_run_jobs_task = await start_recipe_run_jobs_worker(
                stop_event=recipe_run_jobs_stop_event
            )
            if recipe_run_jobs_task:
                logger.info("Evaluation recipe-run Jobs worker started")
                _register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="recipe_run_jobs_task",
                    task=recipe_run_jobs_task,
                    stop_event=recipe_run_jobs_stop_event,
                )
            else:
                logger.info(
                    "Evaluation recipe-run Jobs worker disabled "
                    "(EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED != true)"
                )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start evaluation recipe-run Jobs worker: {e}")

    # Jobs notifications bridge worker
    try:
        if _sidecar_mode:
            logger.info("Jobs notifications bridge worker disabled in sidecar mode")
        else:
            from tldw_Server_API.app.services.jobs_notifications_service import start_jobs_notifications_service

            jobs_notifications_bridge_task = await start_jobs_notifications_service()
            if jobs_notifications_bridge_task:
                logger.info("Jobs notifications bridge worker started")
            else:
                logger.info(
                    "Jobs notifications bridge worker disabled (JOBS_NOTIFICATIONS_BRIDGE_ENABLED != true)"
                )
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs notifications bridge worker: {e}")

    # Evaluations Embeddings A/B Jobs worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs_worker import (
            run_embeddings_abtest_jobs_worker as _run_abtest_jobs,
        )

        _enabled = _os.getenv("EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if not _enabled:
            _enabled = _os.getenv("EVALS_ABTEST_JOBS_WORKER_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _sidecar_mode:
            _enabled = False
        if _enabled:
            evals_abtest_jobs_stop_event = _asyncio.Event()
            evals_abtest_jobs_task = _asyncio.create_task(_run_abtest_jobs(evals_abtest_jobs_stop_event))
            logger.info("Embeddings A/B Jobs worker started with explicit stop_event signal")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="evals_abtest_jobs_task",
                task=evals_abtest_jobs_task,
                stop_event=evals_abtest_jobs_stop_event,
            )
        else:
            logger.info("Embeddings A/B Jobs worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Embeddings A/B Jobs worker: {e}")

    # Jobs metrics gauges worker (SLO percentiles)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.jobs_metrics_service import run_jobs_metrics_gauges as _run_jobs_metrics

        _enabled = _os.getenv("JOBS_METRICS_GAUGES_ENABLED", "true").lower() in {"true", "1", "yes", "y", "on"}
        if _enabled:
            jobs_metrics_stop_event = _asyncio.Event()
            jobs_metrics_task = _asyncio.create_task(_run_jobs_metrics(jobs_metrics_stop_event))
            logger.info("Jobs metrics gauge worker started with explicit stop_event signal")
        else:
            logger.info("Jobs metrics gauge worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs metrics gauge worker: {e}")

    # Event loop lag watchdog (lightweight)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.loop_lag_watchdog import run_loop_lag_watchdog as _run_loop_lag_watchdog

        _enabled = _os.getenv("EVENT_LOOP_LAG_WATCHDOG_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _enabled:
            loop_lag_stop_event = _asyncio.Event()
            loop_lag_task = _asyncio.create_task(_run_loop_lag_watchdog(loop_lag_stop_event))
            logger.info("Event loop lag watchdog started")
        else:
            logger.info("Event loop lag watchdog disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start event loop lag watchdog: {e}")

    # Jobs metrics reconcile worker (job_counters/gauges amortized refresh)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.jobs_metrics_service import run_jobs_metrics_reconcile as _run_jobs_reconcile

        _enabled_recon = _os.getenv("JOBS_METRICS_RECONCILE_ENABLE", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _enabled_recon:
            jobs_metrics_reconcile_stop = _asyncio.Event()
            jobs_metrics_reconcile_task = _asyncio.create_task(_run_jobs_reconcile(jobs_metrics_reconcile_stop))
            logger.info("Jobs metrics reconcile worker started with explicit stop_event signal")
        else:
            logger.info("Jobs metrics reconcile worker disabled by flag (JOBS_METRICS_RECONCILE_ENABLE)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs metrics reconcile worker: {e}")

    # Jobs crypto rotate worker (optional staged rotation)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.jobs_crypto_rotate_service import run_jobs_crypto_rotate as _run_jobs_crypto

        _enabled = _os.getenv("JOBS_CRYPTO_ROTATE_SERVICE_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _enabled:
            jobs_crypto_rotate_stop_event = _asyncio.Event()
            jobs_crypto_rotate_task = _asyncio.create_task(_run_jobs_crypto(jobs_crypto_rotate_stop_event))
            logger.info("Jobs crypto rotate worker started with explicit stop_event signal")
        else:
            logger.info("Jobs crypto rotate worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs crypto rotate worker: {e}")

    # Jobs webhooks worker (signed callbacks)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.jobs_webhooks_service import run_jobs_webhooks_worker as _run_jobs_webhooks

        _enabled = (_os.getenv("JOBS_WEBHOOKS_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}) and bool(
            _os.getenv("JOBS_WEBHOOKS_URL")
        )
        if _enabled:
            jobs_webhooks_stop_event = _asyncio.Event()
            jobs_webhooks_task = _asyncio.create_task(_run_jobs_webhooks(jobs_webhooks_stop_event))
            logger.info("Jobs webhooks worker started with explicit stop_event signal")
        else:
            logger.info("Jobs webhooks worker disabled by flag or missing URL")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs webhooks worker: {e}")

    # Meetings webhook DLQ worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.meetings_webhook_dlq_service import (
            run_meetings_webhook_dlq_worker as _run_meetings_dlq,
        )

        _meetings_dlq_enabled = _os.getenv("MEETINGS_WEBHOOK_DLQ_ENABLED", "false").lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if _meetings_dlq_enabled:
            meetings_webhook_dlq_stop_event = _asyncio.Event()
            meetings_webhook_dlq_task = _asyncio.create_task(
                _run_meetings_dlq(meetings_webhook_dlq_stop_event)
            )
            logger.info("Meetings webhook DLQ worker started with explicit stop_event signal")
        else:
            logger.info("Meetings webhook DLQ worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Meetings webhook DLQ worker: {e}")

    # Workflows webhook DLQ retry worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.workflows_webhook_dlq_service import (
            run_workflows_webhook_dlq_worker as _run_wf_dlq,
        )

        _wf_enabled = _os.getenv("WORKFLOWS_WEBHOOK_DLQ_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _wf_enabled:
            workflows_dlq_stop_event = _asyncio.Event()
            workflows_dlq_task = _asyncio.create_task(_run_wf_dlq(workflows_dlq_stop_event))
            logger.info("Workflows webhook DLQ worker started with explicit stop_event signal")
        else:
            logger.info("Workflows webhook DLQ worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Workflows webhook DLQ worker: {e}")

    # Workflows artifact GC worker
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.workflows_artifact_gc_service import (
            run_workflows_artifact_gc_worker as _run_wf_gc,
        )

        _wf_gc_enabled = _os.getenv("WORKFLOWS_ARTIFACT_GC_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _wf_gc_enabled:
            workflows_gc_stop_event = _asyncio.Event()
            workflows_gc_task = _asyncio.create_task(_run_wf_gc(workflows_gc_stop_event))
            logger.info("Workflows artifact GC worker started with explicit stop_event signal")
        else:
            logger.info("Workflows artifact GC worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Workflows artifact GC worker: {e}")

    # Workflows DB maintenance worker (checkpoint/VACUUM)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.workflows_db_maintenance import run_workflows_db_maintenance as _run_wf_maint

        _wf_maint_enabled = _os.getenv("WORKFLOWS_DB_MAINTENANCE_ENABLED", "false").lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if _wf_maint_enabled:
            workflows_maint_stop_event = _asyncio.Event()
            workflows_maint_task = _asyncio.create_task(_run_wf_maint(workflows_maint_stop_event))
            logger.info("Workflows DB maintenance worker started with explicit stop_event signal")
        else:
            logger.info("Workflows DB maintenance worker disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Workflows DB maintenance worker: {e}")

    # Jobs integrity sweeper (periodic validator)
    try:
        import asyncio as _asyncio
        import os as _os

        from tldw_Server_API.app.services.jobs_integrity_service import (
            run_jobs_integrity_sweeper as _run_jobs_integrity,
        )

        _enabled = _os.getenv("JOBS_INTEGRITY_SWEEP_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        if _enabled:
            jobs_integrity_stop_event = _asyncio.Event()
            jobs_integrity_task = _asyncio.create_task(_run_jobs_integrity(jobs_integrity_stop_event))
            logger.info("Jobs integrity sweeper started with explicit stop_event signal")
        else:
            logger.info("Jobs integrity sweeper disabled by flag")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs integrity sweeper: {e}")

    # Claims rebuild worker (periodic)
    try:
        import asyncio as _asyncio

        from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import (
            get_claims_rebuild_service as _get_claims_svc,
        )
        from tldw_Server_API.app.core.config import settings as _app_settings

        _claims_enabled = bool(_app_settings.get("CLAIMS_REBUILD_ENABLED", False))
        _claims_interval = int(_app_settings.get("CLAIMS_REBUILD_INTERVAL_SEC", 3600))
        _claims_policy = str(_app_settings.get("CLAIMS_REBUILD_POLICY", "missing")).lower()

        async def _claims_rebuild_loop():
            if not _claims_enabled:
                logger.info("Claims rebuild worker disabled by settings")
                return
            logger.info(f"Starting claims rebuild worker (every {_claims_interval}s, policy={_claims_policy})")
            svc = _get_claims_svc()
            while True:
                try:
                    with _claims_rebuild_db_session(_app_settings) as (_, db_path, db):
                        mids = list_claims_rebuild_media_ids(
                            db,
                            policy=_claims_policy,
                            stale_days=int(_app_settings.get("CLAIMS_STALE_DAYS", 7)),
                            compare_media_last_modified=False,
                            limit=25,
                        )
                        for mid in mids:
                            svc.submit(media_id=mid, db_path=db_path)
                except _STARTUP_GUARD_EXCEPTIONS as e:
                    logger.warning(f"Claims rebuild loop error: {e}")
                await _asyncio.sleep(_claims_interval)

        if _claims_enabled:
            claims_task = _asyncio.create_task(_claims_rebuild_loop())
        else:
            logger.info("Claims rebuild worker disabled by settings")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start claims rebuild worker: {e}")

    # Claims alerts scheduler (periodic)
    try:
        from tldw_Server_API.app.services.claims_alerts_scheduler import start_claims_alerts_scheduler

        _claims_alerts_task = await start_claims_alerts_scheduler()
        if _claims_alerts_task:
            logger.info("Claims alerts scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start claims alerts scheduler: {e}")

    # Claims review metrics scheduler (periodic)
    try:
        from tldw_Server_API.app.services.claims_review_metrics_scheduler import (
            start_claims_review_metrics_scheduler,
        )

        _claims_review_metrics_task = await start_claims_review_metrics_scheduler()
        if _claims_review_metrics_task:
            logger.info("Claims review metrics scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start claims review metrics scheduler: {e}")

    # Start usage aggregator (if enabled, and not disabled via env or test-mode)
    try:
        _disable_usage_agg = _shared_env_flag_enabled("DISABLE_USAGE_AGGREGATOR")
        if _disable_usage_agg:
            logger.info("Usage aggregator disabled via DISABLE_USAGE_AGGREGATOR")
        else:
            from tldw_Server_API.app.services.usage_aggregator import start_usage_aggregator

            usage_task = await start_usage_aggregator()
            if usage_task:
                logger.info("Usage aggregator started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start usage aggregator: {e}")

    # Start LLM usage aggregator (if enabled, and not disabled via env or test-mode)
    try:
        _disable_llm_usage_agg = _shared_env_flag_enabled("DISABLE_LLM_USAGE_AGGREGATOR")
        if _disable_llm_usage_agg:
            logger.info("LLM usage aggregator disabled via DISABLE_LLM_USAGE_AGGREGATOR")
        else:
            from tldw_Server_API.app.services.llm_usage_aggregator import start_llm_usage_aggregator

            llm_usage_task = await start_llm_usage_aggregator()
            if llm_usage_task:
                logger.info("LLM usage aggregator started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start LLM usage aggregator: {e}")

    # Start personalization consolidation service if enabled
    try:
        _personalization_enabled = bool(
            _legacy_get("PERSONALIZATION_ENABLED", _app_settings.get("PERSONALIZATION_ENABLED", True))
        )
        _skip_consolidation = _shared_env_flag_enabled("DISABLE_PERSONALIZATION_CONSOLIDATION")
        if not _personalization_enabled or _skip_consolidation:
            logger.info("Personalization consolidation disabled (flag or env)")
        else:
            from tldw_Server_API.app.services.personalization_consolidation import get_consolidation_service

            _consol = get_consolidation_service()
            await _consol.start()
            logger.info("Personalization consolidation service started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start personalization consolidation: {e}")

    # Ensure PG RLS policies (optional, guarded by env)
    try:
        _ensure_rls = _shared_env_flag_enabled("RAG_ENSURE_PG_RLS")
        if _ensure_rls:
            from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
            from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

            _cfg = DatabaseConfig.from_env()
            _backend = DatabaseBackendFactory.create_backend(_cfg)
            try:
                _run_pg_rls_auto_ensure(_backend)
            except DatabaseError as e:
                logger.warning(f"Failed to apply PG RLS policies automatically: {e}")
        else:
            logger.info("PG RLS auto-ensure disabled (set RAG_ENSURE_PG_RLS=true to enable)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to apply PG RLS policies automatically: {e}")

    # Start RAG quality eval scheduler (nightly dashboards)
    try:
        _disable_quality_eval = not _shared_is_truthy(_env_os.getenv("RAG_QUALITY_EVAL_ENABLED", "false"))
        if _disable_quality_eval:
            logger.info("RAG quality eval scheduler disabled (RAG_QUALITY_EVAL_ENABLED != true)")
        else:
            from tldw_Server_API.app.services.quality_eval_scheduler import start_quality_eval_scheduler

            _quality_task = await start_quality_eval_scheduler()
            if _quality_task:
                logger.info("RAG quality eval scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start RAG quality eval scheduler: {e}")

    # Start Outputs purge scheduler (daily maintenance)
    try:
        _enable_outputs_purge = _shared_is_truthy(_env_os.getenv("OUTPUTS_PURGE_ENABLED", "false"))
        if not _enable_outputs_purge:
            logger.info("Outputs purge scheduler disabled (OUTPUTS_PURGE_ENABLED != true)")
        else:
            from tldw_Server_API.app.services.outputs_purge_scheduler import start_outputs_purge_scheduler

            _purge_task = await start_outputs_purge_scheduler()
            if _purge_task:
                logger.info("Outputs purge scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Outputs purge scheduler: {e}")

    # Start TTS history cleanup scheduler (retention cleanup)
    try:
        from tldw_Server_API.app.services.tts_history_cleanup_service import run_tts_history_cleanup_loop
        _tts_history_cleanup_stop_event = _asyncio.Event()
        _tts_history_cleanup_task = _asyncio.create_task(run_tts_history_cleanup_loop(_tts_history_cleanup_stop_event))
        logger.info("TTS history cleanup worker started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start TTS history cleanup worker: {e}")

    # Start Kanban activity cleanup scheduler (retention cleanup)
    try:
        _enable_kanban_activity_cleanup = _shared_is_truthy(
            _env_os.getenv("KANBAN_ACTIVITY_CLEANUP_ENABLED", "false")
        )
        if not _enable_kanban_activity_cleanup:
            logger.info("Kanban activity cleanup scheduler disabled (KANBAN_ACTIVITY_CLEANUP_ENABLED != true)")
        else:
            from tldw_Server_API.app.services.kanban_activity_cleanup_service import (
                start_kanban_activity_cleanup_scheduler,
            )

            _kanban_cleanup_task = await start_kanban_activity_cleanup_scheduler()
            if _kanban_cleanup_task:
                logger.info("Kanban activity cleanup scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Kanban activity cleanup scheduler: {e}")

    # Start ingestion source archive cleanup scheduler (retention cleanup)
    try:
        _enable_ingestion_sources_cleanup = _shared_is_truthy(
            _env_os.getenv("INGESTION_SOURCES_CLEANUP_ENABLED", "false")
        )
        if not _enable_ingestion_sources_cleanup:
            logger.info(
                "Ingestion source archive cleanup scheduler disabled "
                "(INGESTION_SOURCES_CLEANUP_ENABLED != true)"
            )
        else:
            from tldw_Server_API.app.services.ingestion_sources_cleanup_service import (
                start_ingestion_sources_cleanup_scheduler,
            )

            _ingestion_sources_cleanup_task = await start_ingestion_sources_cleanup_scheduler()
            if _ingestion_sources_cleanup_task:
                logger.info("Ingestion source archive cleanup scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start ingestion source archive cleanup scheduler: {e}")

    # Start Kanban soft-delete purge scheduler
    try:
        _enable_kanban_purge = _shared_is_truthy(_env_os.getenv("KANBAN_PURGE_ENABLED", "false"))
        if not _enable_kanban_purge:
            logger.info("Kanban purge scheduler disabled (KANBAN_PURGE_ENABLED != true)")
        else:
            from tldw_Server_API.app.services.kanban_purge_service import start_kanban_purge_scheduler

            _kanban_purge_task = await start_kanban_purge_scheduler()
            if _kanban_purge_task:
                logger.info("Kanban purge scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Kanban purge scheduler: {e}")

    # Start File artifacts export GC scheduler (expired export cleanup)
    try:
        _enable_files_export_gc = _shared_is_truthy(_env_os.getenv("FILES_EXPORT_GC_ENABLED", "false"))
        if not _enable_files_export_gc:
            logger.info("File artifacts export GC scheduler disabled (FILES_EXPORT_GC_ENABLED != true)")
        else:
            from tldw_Server_API.app.services.file_artifacts_export_gc_service import (
                start_file_artifacts_export_gc_scheduler,
            )

            _files_gc_task = await start_file_artifacts_export_gc_scheduler()
            if _files_gc_task:
                logger.info("File artifacts export GC scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        # startup/shutdown guard; log and continue
        logger.warning(f"Failed to start File artifacts export GC scheduler: {e}")

    # Start Notifications prune scheduler (retention cleanup)
    try:
        _enable_notifications_prune = _shared_is_truthy(_env_os.getenv("NOTIFICATIONS_PRUNE_ENABLED", "false"))
        if not _enable_notifications_prune:
            logger.info("Notifications prune scheduler disabled (NOTIFICATIONS_PRUNE_ENABLED != true)")
        else:
            from tldw_Server_API.app.services.notifications_prune_service import start_notifications_prune_scheduler

            _notifications_prune_task = await start_notifications_prune_scheduler()
            if _notifications_prune_task:
                logger.info("Notifications prune scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Notifications prune scheduler: {e}")

    # Start Jobs prune scheduler (daily maintenance)
    try:
        _enable_jobs_prune = _shared_is_truthy(_env_os.getenv("JOBS_PRUNE_ENFORCE", "false"))
        if not _enable_jobs_prune:
            logger.info("Jobs prune scheduler disabled (JOBS_PRUNE_ENFORCE != true)")
        else:
            from tldw_Server_API.app.services.jobs_prune_scheduler import start_jobs_prune_scheduler

            jobs_prune_task = await start_jobs_prune_scheduler()
            if jobs_prune_task:
                logger.info("Jobs prune scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Jobs prune scheduler: {e}")

    # Start Connectors worker (scaffold; opt-in via env)
    try:
        import asyncio as _asyncio

        from tldw_Server_API.app.services.connectors_worker import start_connectors_worker

        connectors_jobs_stop_event = _asyncio.Event()
        connectors_jobs_task = await start_connectors_worker(stop_event=connectors_jobs_stop_event)
        if connectors_jobs_task:
            logger.info("Connectors worker started")
            _register_owned_job_poller(
                app,
                owned_job_pollers,
                name="connectors_jobs_task",
                task=connectors_jobs_task,
                stop_event=connectors_jobs_stop_event,
            )
        else:
            logger.info("Connectors worker disabled (CONNECTORS_WORKER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Connectors worker: {e}")

    _replace_owned_job_poller_inventory(
        app,
        owned_job_pollers,
        registrations=[
            ("core_jobs_task", core_jobs_task, core_jobs_stop_event, 5.0),
            ("files_jobs_task", files_jobs_task, files_jobs_stop_event, 5.0),
            ("data_tables_jobs_task", data_tables_jobs_task, data_tables_jobs_stop_event, 5.0),
            ("prompt_studio_jobs_task", prompt_studio_jobs_task, prompt_studio_jobs_stop_event, 5.0),
            ("study_pack_jobs_task", study_pack_jobs_task, study_pack_jobs_stop_event, 5.0),
            (
                "study_suggestions_jobs_task",
                study_suggestions_jobs_task,
                study_suggestions_jobs_stop_event,
                5.0,
            ),
            ("privilege_snapshot_task", privilege_snapshot_task, privilege_snapshot_stop_event, 5.0),
            ("audio_jobs_task", audio_jobs_task, audio_jobs_stop_event, 5.0),
            ("audiobook_jobs_task", audiobook_jobs_task, audiobook_jobs_stop_event, 5.0),
            (
                "presentation_render_jobs_task",
                presentation_render_jobs_task,
                presentation_render_jobs_stop_event,
                5.0,
            ),
            ("media_ingest_jobs_task", media_ingest_jobs_task, media_ingest_jobs_stop_event, 5.0),
            (
                "media_ingest_heavy_jobs_task",
                media_ingest_heavy_jobs_task,
                media_ingest_heavy_jobs_stop_event,
                5.0,
            ),
            ("reading_digest_jobs_task", reading_digest_jobs_task, reading_digest_jobs_stop_event, 5.0),
            (
                "companion_reflection_jobs_task",
                companion_reflection_jobs_task,
                companion_reflection_jobs_stop_event,
                5.0,
            ),
            ("reminder_jobs_task", reminder_jobs_task, reminder_jobs_stop_event, 5.0),
            ("admin_backup_jobs_task", admin_backup_jobs_task, admin_backup_jobs_stop_event, 5.0),
            (
                "admin_byok_validation_jobs_task",
                admin_byok_validation_jobs_task,
                admin_byok_validation_jobs_stop_event,
                5.0,
            ),
            (
                "admin_maintenance_rotation_jobs_task",
                admin_maintenance_rotation_jobs_task,
                admin_maintenance_rotation_jobs_stop_event,
                5.0,
            ),
            ("recipe_run_jobs_task", recipe_run_jobs_task, recipe_run_jobs_stop_event, 5.0),
            ("evals_abtest_jobs_task", evals_abtest_jobs_task, evals_abtest_jobs_stop_event, 5.0),
            ("connectors_jobs_task", connectors_jobs_task, connectors_jobs_stop_event, 5.0),
        ],
    )

    # Start AuthNZ scheduler (retention/cleanup tasks) with env guard
    _authnz_sched_started = False
    try:
        _disable_authnz_sched = _shared_env_flag_enabled("DISABLE_AUTHNZ_SCHEDULER")
        if _disable_authnz_sched:
            logger.info("AuthNZ scheduler disabled via DISABLE_AUTHNZ_SCHEDULER env var")
        else:
            from tldw_Server_API.app.core.AuthNZ.scheduler import start_authnz_scheduler

            await start_authnz_scheduler()
            _authnz_sched_started = True
            logger.info("AuthNZ scheduler started")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start AuthNZ scheduler: {e}")

    # Start Workflows recurring scheduler (cron-based submission into core Scheduler)
    workflows_sched_task = None
    try:
        from tldw_Server_API.app.services.workflows_scheduler import start_workflows_scheduler

        workflows_sched_task = await start_workflows_scheduler()
        if workflows_sched_task:
            logger.info("Workflows recurring scheduler started")
        else:
            logger.info("Workflows recurring scheduler disabled (WORKFLOWS_SCHEDULER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Workflows recurring scheduler: {e}")

    # Start Reading digest scheduler (cron-based submission into Jobs)
    reading_digest_sched_task = None
    try:
        try:
            _rd_sched_enabled = _env_flag("READING_DIGEST_SCHEDULER_ENABLED", True)
            if globals().get("_TEST_MODE") and _os.getenv("READING_DIGEST_SCHEDULER_ENABLED") is None:
                _rd_sched_enabled = False
        except _STARTUP_GUARD_EXCEPTIONS:
            _rd_sched_enabled = _shared_is_truthy(_os.getenv("READING_DIGEST_SCHEDULER_ENABLED", "true"))
        if _rd_sched_enabled:
            from tldw_Server_API.app.services.reading_digest_scheduler import start_reading_digest_scheduler

            reading_digest_sched_task = await start_reading_digest_scheduler(enabled=True)
            if reading_digest_sched_task:
                logger.info("Reading digest scheduler started")
            else:
                logger.info("Reading digest scheduler disabled (READING_DIGEST_SCHEDULER_ENABLED != true)")
        else:
            logger.info("Reading digest scheduler disabled (READING_DIGEST_SCHEDULER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Reading digest scheduler: {e}")

    # Start Admin backup scheduler (platform backup schedule submission into Jobs)
    admin_backup_sched_task = None
    try:
        from tldw_Server_API.app.services.admin_backup_scheduler import start_admin_backup_scheduler

        admin_backup_sched_task = await start_admin_backup_scheduler()
        if admin_backup_sched_task:
            logger.info("Admin backup scheduler started")
        else:
            logger.info("Admin backup scheduler disabled (ADMIN_BACKUP_SCHEDULER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Admin backup scheduler: {e}")

    # Start Companion reflection scheduler (cron-based submission into Jobs)
    companion_reflection_sched_task = None
    try:
        try:
            _companion_reflection_sched_enabled = _env_flag("COMPANION_REFLECTION_SCHEDULER_ENABLED", False)
        except _STARTUP_GUARD_EXCEPTIONS:
            _companion_reflection_sched_enabled = _shared_is_truthy(
                _os.getenv("COMPANION_REFLECTION_SCHEDULER_ENABLED", "false")
            )
        if _companion_reflection_sched_enabled:
            from tldw_Server_API.app.services.companion_reflection_scheduler import (
                start_companion_reflection_scheduler,
            )

            companion_reflection_sched_task = await start_companion_reflection_scheduler(enabled=True)
            if companion_reflection_sched_task:
                logger.info("Companion reflection scheduler started")
            else:
                logger.info("Companion reflection scheduler disabled (COMPANION_REFLECTION_SCHEDULER_ENABLED != true)")
        else:
            logger.info("Companion reflection scheduler disabled (COMPANION_REFLECTION_SCHEDULER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Companion reflection scheduler: {e}")

    # Start Reminders scheduler (cron/date based submission into Jobs)
    try:
        from tldw_Server_API.app.services.reminders_scheduler import start_reminders_scheduler

        reminders_sched_task = await start_reminders_scheduler()
        if reminders_sched_task:
            logger.info("Reminders scheduler started")
        else:
            logger.info("Reminders scheduler disabled (REMINDERS_SCHEDULER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Reminders scheduler: {e}")

    # Start Connectors sync scheduler (periodic submission into Jobs)
    connectors_sync_sched_task = None
    try:
        from tldw_Server_API.app.services.connectors_sync_scheduler import (
            start_connectors_sync_scheduler,
        )

        connectors_sync_sched_task = await start_connectors_sync_scheduler()
        if connectors_sync_sched_task:
            logger.info("Connectors sync scheduler started")
        else:
            logger.info("Connectors sync scheduler disabled (CONNECTORS_SYNC_SCHEDULER_ENABLED != true)")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.warning(f"Failed to start Connectors sync scheduler: {e}")

    # Display authentication mode (API key masked by default unless explicitly requested)
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings, is_single_user_mode

        settings = get_settings()

        logger.info("=" * 60)
        logger.info("🚀 TLDW Server Started Successfully")
        logger.info("=" * 60)

        if is_single_user_mode():
            logger.info("🔐 Authentication Mode: SINGLE USER")
            _display_key = _startup_api_key_log_value(settings.SINGLE_USER_API_KEY)
            _masked_note = ""
            if _display_key != settings.SINGLE_USER_API_KEY:
                _masked_note = " (masked; set SHOW_API_KEY_ON_STARTUP=true to display once)"
            logger.info(f"🔑 API Key: {_display_key}{_masked_note}")
            logger.info("=" * 60)
            logger.info("Use this API key in the X-API-KEY header for requests")
        else:
            logger.info("🔐 Authentication Mode: MULTI USER")
            try:
                from tldw_Server_API.app.core.AuthNZ.database import get_db_pool as _get_db_pool

                _pool = await _get_db_pool()
                _is_pg = bool(getattr(_pool, "pool", None) is not None)
                if _is_pg:
                    logger.info("JWT Bearer tokens required for authentication")
                else:
                    logger.info("JWT Bearer tokens or X-API-KEY (per-user) supported for SQLite setups")
            except _STARTUP_GUARD_EXCEPTIONS:
                logger.info("JWT Bearer tokens required for authentication")
            logger.info("=" * 60)

        logger.info("📍 API Documentation: http://127.0.0.1:8000/docs")
        logger.info("🧭 Quickstart: http://127.0.0.1:8000/api/v1/config/quickstart")
        logger.info("🛠 Setup UI: http://127.0.0.1:8000/setup (if required)")
        logger.info("=" * 60)
    except _IMPORT_EXCEPTIONS as e:
        logger.exception(f"Failed to display startup info: {e}")

    # Preflight environment report (non-blocking)
    try:
        import os as _os

        from tldw_Server_API.app.core.AuthNZ.csrf_protection import global_settings as _csrf_globals
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings
        from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager as _get_pm
        from tldw_Server_API.app.core.config import (
            get_cors_runtime_diagnostics as _get_cors_runtime_diagnostics,
        )
        from tldw_Server_API.app.core.Metrics import OTEL_AVAILABLE as _OTEL

        _s = _get_settings()
        _prod = _os.getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}
        _auth_mode = _s.AUTH_MODE
        _db_url = _s.DATABASE_URL
        # Use the unified backend detector for the engine label in diagnostics
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool as _get_db_pool

            _pool = await _get_db_pool()
            _is_pg = bool(getattr(_pool, "pool", None) is not None)
            _db_engine = "postgresql" if _is_pg else ("sqlite" if str(_db_url).startswith("sqlite") else "other")
        except _STARTUP_GUARD_EXCEPTIONS:
            _db_engine = "other"
        _redis_url = _s.REDIS_URL or ""
        _redis_enabled = bool(_s.REDIS_URL) or bool(
            _os.getenv("REDIS_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
        )
        _csrf_enabled = (_auth_mode == "multi_user") or (_csrf_globals.get("CSRF_ENABLED", None) is True)
        _cors_diagnostics = _get_cors_runtime_diagnostics()
        _cors_disable = bool(_cors_diagnostics.get("disable_cors", False))
        _cors_disable_source = str(_cors_diagnostics.get("disable_cors_source", "unknown"))
        _cors_allow_credentials = bool(_cors_diagnostics.get("allow_credentials", False))
        _cors_allow_credentials_source = str(_cors_diagnostics.get("allow_credentials_source", "unknown"))
        _cors_count = int(_cors_diagnostics.get("allowed_origins_count", 0))
        _cors_allowed_origins_source = str(_cors_diagnostics.get("allowed_origins_source", "unknown"))
        _cors_allowed_origins = _cors_diagnostics.get("allowed_origins", [])
        if not isinstance(_cors_allowed_origins, list):
            _cors_allowed_origins = []
        _cors_config_path = _cors_diagnostics.get("config_path")
        _cors_config_loaded = bool(_cors_diagnostics.get("config_loaded", False))
        _has_limiter = hasattr(app.state, "limiter")
        _pm = _get_pm()
        _providers = len(_pm.providers) if _pm and hasattr(_pm, "providers") else 0

        logger.info("Preflight Environment Report ─────────────────────────────────────────")
        logger.info(f"• Mode: {_auth_mode} | Production: {_prod}")
        logger.info(f"• Database: engine={_db_engine}")
        if _db_engine == "sqlite" and _auth_mode == "multi_user":
            if _prod:
                logger.error("• Database check: FAIL (SQLite in multi-user prod not supported)")
            else:
                logger.warning("• Database check: WARN (SQLite in multi-user; prefer PostgreSQL)")
        else:
            logger.info("• Database check: OK")
        logger.info(f"• Redis: enabled={_redis_enabled}")
        logger.info(f"• CSRF: enabled={_csrf_enabled}")
        if _cors_disable:
            logger.info("• CORS: disabled")
        else:
            logger.info(
                f"• CORS: allowed_origins={_cors_count} | allow_credentials={_cors_allow_credentials}"
            )
        logger.info(
            "• CORS effective settings: "
            f"disable={_cors_disable} (source={_cors_disable_source}) | "
            f"allow_credentials={_cors_allow_credentials} (source={_cors_allow_credentials_source}) | "
            f"origins={_cors_count} (source={_cors_allowed_origins_source})"
        )
        logger.info(
            f"• CORS config file: path={_cors_config_path or '(unknown)'} | loaded={_cors_config_loaded}"
        )
        if _cors_allowed_origins:
            _origin_preview_max = 6
            _origin_preview = ", ".join(str(o) for o in _cors_allowed_origins[:_origin_preview_max])
            if len(_cors_allowed_origins) > _origin_preview_max:
                _origin_preview += f", ... (+{len(_cors_allowed_origins) - _origin_preview_max} more)"
            logger.info(f"• CORS origins preview: {_origin_preview}")
        logger.info(f"• Global rate limiter: {_has_limiter}")
        logger.info(f"• Providers configured: {_providers}")
        logger.info(f"• OpenTelemetry available: {bool(_OTEL)}")
        logger.info("──────────────────────────────────────────────────────────────────────")

        # Warn if test-mode gates/toggles are enabled in a production environment
        try:
            if _prod:
                _test_flags = {
                    "TEST_MODE": _os.getenv("TEST_MODE", ""),
                    "TLDW_TEST_MODE": _os.getenv("TLDW_TEST_MODE", ""),
                }
                _enabled = [k for k, v in _test_flags.items() if _shared_is_truthy(v)]
                if _enabled:
                    logger.warning(
                        f"Test-mode toggles enabled in production: {', '.join(_enabled)} - disable these for secure deployments"
                    )
        except _STARTUP_GUARD_EXCEPTIONS:
            pass
    except _STARTUP_GUARD_EXCEPTIONS as _pf_e:
        logger.warning(f"Preflight report could not be generated: {_pf_e}")

    yield

    # Build and record the legacy shutdown inventory first.
    # Execute only the narrow transition gate handoff through the coordinator;
    # the remaining legacy teardown paths stay on the existing direct teardown path.
    shutdown_started = time.monotonic()
    try:
        app.state._tldw_shutdown_timing_segments = []
    except _STARTUP_GUARD_EXCEPTIONS:
        pass
    transition_gate_applied = False
    legacy_shutdown_plan: list[Any] = []
    coordinated_legacy_component_names: set[str] = set()
    with _timed_shutdown_segment(app, "transition_handoff"):
        try:
            from tldw_Server_API.app.services.shutdown_coordinator import ShutdownCoordinator, ShutdownPhase
            from tldw_Server_API.app.services.shutdown_legacy_adapters import (
                build_legacy_shutdown_plan,
            )

            shutdown_context = _build_legacy_shutdown_context(
                readiness_state=READINESS_STATE,
                usage_task=usage_task,
                llm_usage_task=llm_usage_task,
                authnz_scheduler_started=_authnz_sched_started,
                chatbooks_cleanup_task=chatbooks_cleanup_task,
                chatbooks_cleanup_stop_event=chatbooks_cleanup_stop_event,
                storage_cleanup_service=storage_cleanup_service,
            )
            legacy_shutdown_plan = build_legacy_shutdown_plan(app, shutdown_context)
            legacy_phase_groups: dict[str, list[str]] = {}
            for component in legacy_shutdown_plan:
                legacy_phase_groups.setdefault(component.phase.value, []).append(component.name)

            try:
                app.state._tldw_shutdown_legacy_plan = legacy_shutdown_plan
                app.state._tldw_shutdown_legacy_phase_groups = legacy_phase_groups
                app.state._tldw_shutdown_legacy_inventory_visible = bool(legacy_shutdown_plan)
            except _STARTUP_GUARD_EXCEPTIONS:
                pass

            logger.info(
                "App Shutdown: legacy inventory visible={} phase_groups={}",
                bool(legacy_shutdown_plan),
                legacy_phase_groups,
            )

            transition_coordinator = ShutdownCoordinator(profile="dev_fast")
            for component in legacy_shutdown_plan:
                if component.phase == ShutdownPhase.TRANSITION:
                    transition_coordinator.register(component)
            if legacy_shutdown_plan:
                transition_summary = await transition_coordinator.shutdown()
                transition_gate_summary = transition_summary.components.get("lifecycle_gate")
                transition_gate_applied = bool(
                    transition_gate_summary is not None and transition_gate_summary.result == "stopped"
                )
                if transition_gate_applied:
                    logger.info("App Shutdown: legacy transition gate handoff executed via coordinator")
                else:
                    logger.warning(
                        "App Shutdown: legacy transition gate handoff did not complete cleanly; "
                        "falling back to direct drain",
                    )
        except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _legacy_shutdown_err:
            logger.debug(f"Legacy shutdown inventory skipped: {_legacy_shutdown_err}")
        finally:
            if not transition_gate_applied:
                _apply_shutdown_transition_gate(app, READINESS_STATE)

    _max_wait = 0
    _count_active_processing = lambda: 0
    try:
        _max_wait = int(_env_os.getenv("JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC", "0") or "0")
    except _STARTUP_GUARD_EXCEPTIONS:
        _max_wait = 0
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _ShutdownJM

        _count_active_processing = _ShutdownJM().count_active_processing
    except _IMPORT_EXCEPTIONS:
        pass

    await _quiesce_owned_job_pollers_for_shutdown(
        app,
        owned_job_pollers,
        wait_for_leases_sec=_max_wait,
        count_active_processing=_count_active_processing,
    )
    early_quiesced_job_poller_names = set(
        getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names", [])
    )

    def _should_run_late_stop(task_name: str, task: Any) -> bool:
        return bool(task) and task_name not in early_quiesced_job_poller_names

    # Execute the migrated legacy shutdown components through the coordinator.
    try:
        non_transition_legacy_shutdown_plan = [
            component
            for component in legacy_shutdown_plan
            if getattr(getattr(component, "phase", None), "value", None) != "transition"
        ]
        coordinated_legacy_component_names = await _run_coordinated_shutdown(
            app,
            non_transition_legacy_shutdown_plan,
        )
    except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _coordinated_legacy_shutdown_err:
        logger.debug(f"Legacy coordinator shutdown skipped: {_coordinated_legacy_shutdown_err}")

    # Cancel/stop background worker(s)
    try:
        bg = getattr(app.state, "bg_tasks", None)
        if isinstance(bg, dict):
            task = bg.get("deferred_startup")
            if task:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS:
        pass
    try:
        if "cleanup_task" in locals() and cleanup_task:
            cleanup_task.cancel()
        if "chatbooks_cleanup" not in coordinated_legacy_component_names:
            if "chatbooks_cleanup_stop_event" in locals() and chatbooks_cleanup_stop_event:
                chatbooks_cleanup_stop_event.set()
            if "chatbooks_cleanup_task" in locals() and chatbooks_cleanup_task:
                chatbooks_cleanup_task.cancel()
        # Storage cleanup service shutdown
        if "storage_cleanup_service" in locals() and storage_cleanup_service:
            if "storage_cleanup_service" not in coordinated_legacy_component_names:
                try:
                    await storage_cleanup_service.stop()
                    logger.info("Storage cleanup worker stopped")
                except _STARTUP_GUARD_EXCEPTIONS:
                    pass
        try:
            from tldw_Server_API.app.services.storage_cleanup_service import (
                reset_cleanup_service as _reset_cleanup_service,
            )
            from tldw_Server_API.app.services.storage_quota_service import (
                reset_storage_service as _reset_storage_service,
            )

            await _reset_cleanup_service()
            await _reset_storage_service()
            logger.info("Storage service singletons reset")
        except _STARTUP_GUARD_EXCEPTIONS:
            pass
        try:
            from tldw_Server_API.app.core.AuthNZ.rate_limiter import (
                reset_rate_limiter as _reset_authnz_rate_limiter,
            )

            await _reset_authnz_rate_limiter()
            logger.info("AuthNZ limiter singletons reset")
        except _STARTUP_GUARD_EXCEPTIONS:
            pass
        if "core_jobs_task" in locals() and _should_run_late_stop("core_jobs_task", core_jobs_task):
            # Prefer graceful stop via explicit stop_event
            if "core_jobs_stop_event" in locals() and core_jobs_stop_event:
                try:
                    core_jobs_stop_event.set()
                    await _asyncio.wait_for(core_jobs_task, timeout=5.0)
                    logger.info("Core Jobs worker (Chatbooks) stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    core_jobs_task.cancel()
            else:
                core_jobs_task.cancel()
        if "files_jobs_task" in locals() and _should_run_late_stop("files_jobs_task", files_jobs_task):
            # Prefer graceful stop via explicit stop_event
            if "files_jobs_stop_event" in locals() and files_jobs_stop_event:
                try:
                    files_jobs_stop_event.set()
                    await _asyncio.wait_for(files_jobs_task, timeout=5.0)
                    logger.info("File Artifacts Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    # startup/shutdown guard; log and continue
                    files_jobs_task.cancel()
            else:
                files_jobs_task.cancel()
        if "data_tables_jobs_task" in locals() and _should_run_late_stop("data_tables_jobs_task", data_tables_jobs_task):
            # Prefer graceful stop via explicit stop_event
            if "data_tables_jobs_stop_event" in locals() and data_tables_jobs_stop_event:
                try:
                    data_tables_jobs_stop_event.set()
                    await _asyncio.wait_for(data_tables_jobs_task, timeout=5.0)
                    logger.info("Data Tables Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    data_tables_jobs_task.cancel()
            else:
                data_tables_jobs_task.cancel()
        if "prompt_studio_jobs_task" in locals() and _should_run_late_stop("prompt_studio_jobs_task", prompt_studio_jobs_task):
            # Prefer graceful stop via explicit stop_event
            if "prompt_studio_jobs_stop_event" in locals() and prompt_studio_jobs_stop_event:
                try:
                    prompt_studio_jobs_stop_event.set()
                    await _asyncio.wait_for(prompt_studio_jobs_task, timeout=5.0)
                    logger.info("Prompt Studio Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    prompt_studio_jobs_task.cancel()
            else:
                prompt_studio_jobs_task.cancel()
        if "privilege_snapshot_task" in locals() and _should_run_late_stop("privilege_snapshot_task", privilege_snapshot_task):
            # Prefer graceful stop via explicit stop_event
            if "privilege_snapshot_stop_event" in locals() and privilege_snapshot_stop_event:
                try:
                    privilege_snapshot_stop_event.set()
                    await _asyncio.wait_for(privilege_snapshot_task, timeout=5.0)
                    logger.info("Privilege snapshot worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    privilege_snapshot_task.cancel()
            else:
                privilege_snapshot_task.cancel()
        if "audio_jobs_task" in locals() and _should_run_late_stop("audio_jobs_task", audio_jobs_task):
            # Prefer graceful stop via explicit stop_event
            if "audio_jobs_stop_event" in locals() and audio_jobs_stop_event:
                try:
                    audio_jobs_stop_event.set()
                    await _asyncio.wait_for(audio_jobs_task, timeout=5.0)
                    logger.info("Audio Jobs worker stopped via stop_event")
                except _asyncio.CancelledError:
                    raise
                except _STARTUP_GUARD_EXCEPTIONS:
                    audio_jobs_task.cancel()
                except Exception as e:
                    logger.warning(
                        f"Audio Jobs worker exited with exception before shutdown completion: {e}"
                    )
                    with suppress(_STARTUP_GUARD_EXCEPTIONS):
                        audio_jobs_task.cancel()
            else:
                audio_jobs_task.cancel()
        if "presentation_render_jobs_task" in locals() and _should_run_late_stop("presentation_render_jobs_task", presentation_render_jobs_task):
            if "presentation_render_jobs_stop_event" in locals() and presentation_render_jobs_stop_event:
                try:
                    presentation_render_jobs_stop_event.set()
                    await _asyncio.wait_for(presentation_render_jobs_task, timeout=5.0)
                    logger.info("Presentation Render Jobs worker stopped via stop_event")
                except _asyncio.CancelledError:
                    raise
                except _STARTUP_GUARD_EXCEPTIONS:
                    presentation_render_jobs_task.cancel()
                except Exception as e:
                    logger.warning(
                        f"Presentation Render Jobs worker exited with exception before shutdown completion: {e}"
                    )
                    with suppress(_STARTUP_GUARD_EXCEPTIONS):
                        presentation_render_jobs_task.cancel()
            else:
                presentation_render_jobs_task.cancel()
        if "media_ingest_jobs_task" in locals() and _should_run_late_stop("media_ingest_jobs_task", media_ingest_jobs_task):
            # Prefer graceful stop via explicit stop_event
            if "media_ingest_jobs_stop_event" in locals() and media_ingest_jobs_stop_event:
                try:
                    media_ingest_jobs_stop_event.set()
                    await _asyncio.wait_for(media_ingest_jobs_task, timeout=5.0)
                    logger.info("Media Ingest Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    media_ingest_jobs_task.cancel()
            else:
                media_ingest_jobs_task.cancel()
        if "media_ingest_heavy_jobs_task" in locals() and _should_run_late_stop("media_ingest_heavy_jobs_task", media_ingest_heavy_jobs_task):
            if (
                "media_ingest_heavy_jobs_stop_event" in locals()
                and media_ingest_heavy_jobs_stop_event
            ):
                try:
                    media_ingest_heavy_jobs_stop_event.set()
                    await _asyncio.wait_for(media_ingest_heavy_jobs_task, timeout=5.0)
                    logger.info("Media Ingest Heavy Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    media_ingest_heavy_jobs_task.cancel()
            else:
                media_ingest_heavy_jobs_task.cancel()
        if "reading_digest_jobs_task" in locals() and _should_run_late_stop("reading_digest_jobs_task", reading_digest_jobs_task):
            if "reading_digest_jobs_stop_event" in locals() and reading_digest_jobs_stop_event:
                try:
                    reading_digest_jobs_stop_event.set()
                    await _asyncio.wait_for(reading_digest_jobs_task, timeout=5.0)
                    logger.info("Reading digest Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    reading_digest_jobs_task.cancel()
            else:
                reading_digest_jobs_task.cancel()
        if "study_pack_jobs_task" in locals() and _should_run_late_stop("study_pack_jobs_task", study_pack_jobs_task):
            if "study_pack_jobs_stop_event" in locals() and study_pack_jobs_stop_event:
                try:
                    study_pack_jobs_stop_event.set()
                    await _asyncio.wait_for(study_pack_jobs_task, timeout=5.0)
                    logger.info("Study-pack Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    study_pack_jobs_task.cancel()
            else:
                study_pack_jobs_task.cancel()
        if "study_suggestions_jobs_task" in locals() and _should_run_late_stop("study_suggestions_jobs_task", study_suggestions_jobs_task):
            if "study_suggestions_jobs_stop_event" in locals() and study_suggestions_jobs_stop_event:
                try:
                    study_suggestions_jobs_stop_event.set()
                    await _asyncio.wait_for(study_suggestions_jobs_task, timeout=5.0)
                    logger.info("Study-suggestions Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    study_suggestions_jobs_task.cancel()
            else:
                study_suggestions_jobs_task.cancel()
        if "companion_reflection_jobs_task" in locals() and _should_run_late_stop("companion_reflection_jobs_task", companion_reflection_jobs_task):
            if "companion_reflection_jobs_stop_event" in locals() and companion_reflection_jobs_stop_event:
                try:
                    companion_reflection_jobs_stop_event.set()
                    await _asyncio.wait_for(companion_reflection_jobs_task, timeout=5.0)
                    logger.info("Companion reflection Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    companion_reflection_jobs_task.cancel()
            else:
                companion_reflection_jobs_task.cancel()
        if "reminder_jobs_task" in locals() and _should_run_late_stop("reminder_jobs_task", reminder_jobs_task):
            try:
                reminder_jobs_task.cancel()
                await _asyncio.wait_for(reminder_jobs_task, timeout=5.0)
                logger.info("Reminder Jobs worker cancelled")
            except _asyncio.CancelledError:
                pass
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    reminder_jobs_task.cancel()
        if "admin_backup_jobs_task" in locals() and _should_run_late_stop("admin_backup_jobs_task", admin_backup_jobs_task):
            try:
                admin_backup_jobs_task.cancel()
                await _asyncio.wait_for(admin_backup_jobs_task, timeout=5.0)
                logger.info("Admin backup Jobs worker cancelled")
            except _asyncio.CancelledError:
                pass
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    admin_backup_jobs_task.cancel()
        if (
            "admin_maintenance_rotation_jobs_task" in locals()
            and _should_run_late_stop(
                "admin_maintenance_rotation_jobs_task",
                admin_maintenance_rotation_jobs_task,
            )
        ):
            if (
                "admin_maintenance_rotation_jobs_stop_event" in locals()
                and admin_maintenance_rotation_jobs_stop_event
            ):
                try:
                    admin_maintenance_rotation_jobs_stop_event.set()
                    await _asyncio.wait_for(admin_maintenance_rotation_jobs_task, timeout=5.0)
                    logger.info("Admin maintenance rotation Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    admin_maintenance_rotation_jobs_task.cancel()
            else:
                admin_maintenance_rotation_jobs_task.cancel()
        if "jobs_notifications_bridge_task" in locals() and jobs_notifications_bridge_task:
            try:
                jobs_notifications_bridge_task.cancel()
                await _asyncio.wait_for(jobs_notifications_bridge_task, timeout=5.0)
                logger.info("Jobs notifications bridge worker cancelled")
            except _asyncio.CancelledError:
                pass
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    jobs_notifications_bridge_task.cancel()
        if "recipe_run_jobs_task" in locals() and _should_run_late_stop("recipe_run_jobs_task", recipe_run_jobs_task):
            if "recipe_run_jobs_stop_event" in locals() and recipe_run_jobs_stop_event:
                try:
                    recipe_run_jobs_stop_event.set()
                    await _asyncio.wait_for(recipe_run_jobs_task, timeout=5.0)
                    logger.info("Evaluation recipe-run Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    recipe_run_jobs_task.cancel()
            else:
                recipe_run_jobs_task.cancel()
        if "evals_abtest_jobs_task" in locals() and _should_run_late_stop("evals_abtest_jobs_task", evals_abtest_jobs_task):
            if "evals_abtest_jobs_stop_event" in locals() and evals_abtest_jobs_stop_event:
                try:
                    evals_abtest_jobs_stop_event.set()
                    await _asyncio.wait_for(evals_abtest_jobs_task, timeout=5.0)
                    logger.info("Embeddings A/B Jobs worker stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    evals_abtest_jobs_task.cancel()
            else:
                evals_abtest_jobs_task.cancel()
        if "claims_task" in locals() and claims_task:
            claims_task.cancel()
        if "jobs_prune_task" in locals() and jobs_prune_task:
            jobs_prune_task.cancel()
        if "_files_gc_task" in locals() and _files_gc_task:
            _files_gc_task.cancel()
        if "_notifications_prune_task" in locals() and _notifications_prune_task:
            _notifications_prune_task.cancel()
        if "embeddings_compactor_task" in locals() and embeddings_compactor_task:
            if "embeddings_compactor_stop_event" in locals() and embeddings_compactor_stop_event:
                try:
                    embeddings_compactor_stop_event.set()
                    await _asyncio.wait_for(embeddings_compactor_task, timeout=5.0)
                    logger.info("Embeddings Vector Compactor stopped via stop_event")
                except _STARTUP_GUARD_EXCEPTIONS:
                    embeddings_compactor_task.cancel()
            else:
                embeddings_compactor_task.cancel()
        if "websub_renewal_task" in locals() and websub_renewal_task:
            websub_renewal_task.cancel()
            logger.info("WebSub renewal worker cancelled")
        # Stop usage aggregators gracefully
        try:
            if "usage_aggregator" not in coordinated_legacy_component_names:
                if "usage_task" in locals() and usage_task:
                    from tldw_Server_API.app.services.usage_aggregator import (
                        stop_usage_aggregator as _stop_usage,
                    )

                    await _stop_usage(usage_task)
                    usage_task = None
        except _STARTUP_GUARD_EXCEPTIONS:
            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                usage_task.cancel()
                usage_task = None
        try:
            if "llm_usage_aggregator" not in coordinated_legacy_component_names:
                if "llm_usage_task" in locals() and llm_usage_task:
                    from tldw_Server_API.app.services.llm_usage_aggregator import (
                        stop_llm_usage_aggregator as _stop_llm,
                    )

                    await _stop_llm(llm_usage_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                llm_usage_task.cancel()
        # Stop Workflows recurring scheduler
        try:
            if "workflows_sched_task" in locals() and workflows_sched_task:
                from tldw_Server_API.app.services.workflows_scheduler import stop_workflows_scheduler as _stop_wf_sched

                await _stop_wf_sched(workflows_sched_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            try:
                if "workflows_sched_task" in locals() and workflows_sched_task:
                    workflows_sched_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                pass
        # Stop Reading digest scheduler
        try:
            if "reading_digest_sched_task" in locals() and reading_digest_sched_task:
                from tldw_Server_API.app.services.reading_digest_scheduler import (
                    stop_reading_digest_scheduler as _stop_rd_sched,
                )

                await _stop_rd_sched(reading_digest_sched_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            try:
                if "reading_digest_sched_task" in locals() and reading_digest_sched_task:
                    reading_digest_sched_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                pass
        # Stop Admin backup scheduler
        try:
            if "admin_backup_sched_task" in locals() and admin_backup_sched_task:
                from tldw_Server_API.app.services.admin_backup_scheduler import (
                    stop_admin_backup_scheduler as _stop_admin_backup_scheduler,
                )

                await _stop_admin_backup_scheduler(admin_backup_sched_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            try:
                if "admin_backup_sched_task" in locals() and admin_backup_sched_task:
                    admin_backup_sched_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                pass
        # Stop Companion reflection scheduler
        try:
            if "companion_reflection_sched_task" in locals() and companion_reflection_sched_task:
                from tldw_Server_API.app.services.companion_reflection_scheduler import (
                    stop_companion_reflection_scheduler as _stop_companion_reflection_scheduler,
                )

                await _stop_companion_reflection_scheduler(companion_reflection_sched_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            try:
                if "companion_reflection_sched_task" in locals() and companion_reflection_sched_task:
                    companion_reflection_sched_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                pass
        # Stop Reminders scheduler
        try:
            if "reminders_sched_task" in locals():
                from tldw_Server_API.app.services.reminders_scheduler import (
                    stop_reminders_scheduler as _stop_reminders_scheduler,
                )

                await _stop_reminders_scheduler(reminders_sched_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            try:
                if "reminders_sched_task" in locals() and reminders_sched_task:
                    reminders_sched_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                pass
        # Stop Connectors sync scheduler
        try:
            if "connectors_sync_sched_task" in locals():
                from tldw_Server_API.app.services.connectors_sync_scheduler import (
                    stop_connectors_sync_scheduler as _stop_connectors_sync_scheduler,
                )

                await _stop_connectors_sync_scheduler(connectors_sync_sched_task)
        except _STARTUP_GUARD_EXCEPTIONS:
            try:
                if "connectors_sync_sched_task" in locals() and connectors_sync_sched_task:
                    connectors_sync_sched_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                pass
        # Jobs metrics gauges worker shutdown
        if "jobs_metrics_task" in locals() and jobs_metrics_task:
            try:
                if "jobs_metrics_stop_event" in locals() and jobs_metrics_stop_event:
                    jobs_metrics_stop_event.set()
                    await _asyncio.wait_for(jobs_metrics_task, timeout=5.0)
                    logger.info("Jobs metrics gauge worker stopped via stop_event")
                else:
                    jobs_metrics_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    jobs_metrics_task.cancel()

        # Jobs metrics reconcile worker shutdown
        if "jobs_metrics_reconcile_task" in locals() and jobs_metrics_reconcile_task:
            try:
                if "jobs_metrics_reconcile_stop" in locals() and jobs_metrics_reconcile_stop:
                    jobs_metrics_reconcile_stop.set()
                    await _asyncio.wait_for(jobs_metrics_reconcile_task, timeout=5.0)
                    logger.info("Jobs metrics reconcile worker stopped via stop_event")
                else:
                    jobs_metrics_reconcile_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    jobs_metrics_reconcile_task.cancel()

        # Event loop lag watchdog shutdown
        if "loop_lag_task" in locals() and loop_lag_task:
            try:
                if "loop_lag_stop_event" in locals() and loop_lag_stop_event:
                    loop_lag_stop_event.set()
                    await _asyncio.wait_for(loop_lag_task, timeout=2.0)
                    logger.info("Event loop lag watchdog stopped via stop_event")
                else:
                    loop_lag_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                try:
                    loop_lag_task.cancel()
                except _STARTUP_GUARD_EXCEPTIONS as _lag_cancel_err:
                    logger.debug(f"Event loop lag watchdog cancel failed: {_lag_cancel_err}")

        # Personalization consolidation service shutdown
        try:
            from tldw_Server_API.app.services.personalization_consolidation import get_consolidation_service

            _consol = get_consolidation_service()
            await _consol.stop()
            logger.info("Personalization consolidation service stopped")
        except _STARTUP_GUARD_EXCEPTIONS:
            pass

        # Jobs crypto rotate worker shutdown
        if "jobs_crypto_rotate_task" in locals() and jobs_crypto_rotate_task:
            try:
                if "jobs_crypto_rotate_stop_event" in locals() and jobs_crypto_rotate_stop_event:
                    jobs_crypto_rotate_stop_event.set()
                    await _asyncio.wait_for(jobs_crypto_rotate_task, timeout=5.0)
                    logger.info("Jobs crypto rotate worker stopped via stop_event")
                else:
                    jobs_crypto_rotate_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    jobs_crypto_rotate_task.cancel()

        # Jobs integrity sweeper shutdown
        if "jobs_integrity_task" in locals() and jobs_integrity_task:
            try:
                if "jobs_integrity_stop_event" in locals() and jobs_integrity_stop_event:
                    jobs_integrity_stop_event.set()
                    await _asyncio.wait_for(jobs_integrity_task, timeout=5.0)
                    logger.info("Jobs integrity sweeper stopped via stop_event")
                else:
                    jobs_integrity_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    jobs_integrity_task.cancel()

        # Jobs webhooks worker shutdown
        if "jobs_webhooks_task" in locals() and jobs_webhooks_task:
            try:
                if "jobs_webhooks_stop_event" in locals() and jobs_webhooks_stop_event:
                    jobs_webhooks_stop_event.set()
                    await _asyncio.wait_for(jobs_webhooks_task, timeout=5.0)
                    logger.info("Jobs webhooks worker stopped via stop_event")
                else:
                    jobs_webhooks_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    jobs_webhooks_task.cancel()

        # Meetings webhook DLQ worker shutdown
        if "meetings_webhook_dlq_task" in locals() and meetings_webhook_dlq_task:
            try:
                if "meetings_webhook_dlq_stop_event" in locals() and meetings_webhook_dlq_stop_event:
                    meetings_webhook_dlq_stop_event.set()
                    await _asyncio.wait_for(meetings_webhook_dlq_task, timeout=5.0)
                    logger.info("Meetings webhook DLQ worker stopped via stop_event")
                else:
                    meetings_webhook_dlq_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    meetings_webhook_dlq_task.cancel()

        # Workflows webhook DLQ worker shutdown
        if "workflows_dlq_task" in locals() and workflows_dlq_task:
            try:
                if "workflows_dlq_stop_event" in locals() and workflows_dlq_stop_event:
                    workflows_dlq_stop_event.set()
                    await _asyncio.wait_for(workflows_dlq_task, timeout=5.0)
                    logger.info("Workflows webhook DLQ worker stopped via stop_event")
                else:
                    workflows_dlq_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    workflows_dlq_task.cancel()

        # Workflows artifact GC worker shutdown
        if "workflows_gc_task" in locals() and workflows_gc_task:
            try:
                if "workflows_gc_stop_event" in locals() and workflows_gc_stop_event:
                    workflows_gc_stop_event.set()
                    await _asyncio.wait_for(workflows_gc_task, timeout=5.0)
                    logger.info("Workflows artifact GC worker stopped via stop_event")
                else:
                    workflows_gc_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    workflows_gc_task.cancel()

        # Workflows DB maintenance worker shutdown
        if "workflows_maint_task" in locals() and workflows_maint_task:
            try:
                if "workflows_maint_stop_event" in locals() and workflows_maint_stop_event:
                    workflows_maint_stop_event.set()
                    await _asyncio.wait_for(workflows_maint_task, timeout=5.0)
                    logger.info("Workflows DB maintenance worker stopped via stop_event")
                else:
                    workflows_maint_task.cancel()
            except _STARTUP_GUARD_EXCEPTIONS:
                with suppress(_STARTUP_GUARD_EXCEPTIONS):
                    workflows_maint_task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS:
        pass

    # Stop AuthNZ scheduler early so it can't keep the loop alive during shutdown.
    try:
        if (
            "_authnz_sched_started" in locals()
            and _authnz_sched_started
            and "authnz_scheduler" not in coordinated_legacy_component_names
        ):
            from tldw_Server_API.app.core.AuthNZ.scheduler import stop_authnz_scheduler

            await stop_authnz_scheduler()
            _authnz_sched_started = False
            logger.info("AuthNZ scheduler stopped")
    except _STARTUP_GUARD_EXCEPTIONS as _e:
        logger.debug(f"AuthNZ scheduler shutdown skipped: {_e}")

    # Shutdown: Clean up resources
    logger.info("App Shutdown: Cleaning up resources...")

    # Note: Audit service cleanup handled via dependency injection
    # No global shutdown needed
    logger.info("App Shutdown: Audit services cleanup handled by dependency injection")

    # Close auth database pool (skip in test contexts to avoid closing shared pool)
    try:
        if "db_pool" in locals():
            import os as _os

            _in_pytest = _shared_is_explicit_pytest_runtime()
            if not _in_pytest:
                await db_pool.close()
                logger.info("App Shutdown: Auth database pool closed")
            else:
                logger.info("App Shutdown: Skipping DB pool close in test context")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error closing auth database pool: {e}")

    # Shutdown session manager
    try:
        if "session_manager" in locals():
            await session_manager.shutdown()
            logger.info("App Shutdown: Session manager shutdown")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error shutting down session manager: {e}")

    # Shutdown MCP Unified server
    try:
        if "mcp_server" in locals() and mcp_server is not None:
            await mcp_server.shutdown()
            logger.info("App Shutdown: MCP Unified server shutdown")
    except _IMPORT_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error shutting down MCP Unified server: {e}")

    # Shutdown MCP Unified rate limiter cleanup task (if any)
    try:
        from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import (
            shutdown_rate_limiter as _mcp_shutdown_rl,
        )

        await _mcp_shutdown_rl()
        logger.info("App Shutdown: MCP rate limiter cleanup task cancelled")
    except _IMPORT_EXCEPTIONS as e:
        logger.debug(f"App Shutdown: MCP rate limiter shutdown skipped/failed: {e}")

    _in_pytest_for_tts_shutdown = _shared_is_explicit_pytest_runtime()

    # Shutdown TTS Service
    if _in_pytest_for_tts_shutdown:
        logger.info("App Shutdown: Skipping TTS service shutdown in test context")
    else:
        try:
            from tldw_Server_API.app.core.TTS.tts_service_v2 import close_tts_service_v2
            from tldw_Server_API.app.core.TTS.voice_manager import shutdown_voice_manager

            await shutdown_voice_manager()
            await close_tts_service_v2()
            logger.info("App Shutdown: TTS service shutdown complete")
        except _IMPORT_EXCEPTIONS as e:
            logger.exception(f"App Shutdown: Error shutting down TTS service: {e}")

    # Shutdown TTS Resource Manager (memory monitor, sessions, HTTP clients)
    if _in_pytest_for_tts_shutdown:
        logger.info("App Shutdown: Skipping TTS resource manager shutdown in test context")
    else:
        try:
            from tldw_Server_API.app.core.TTS.tts_resource_manager import (
                close_resource_manager as _close_tts_resource_manager,
            )

            await _close_tts_resource_manager()
            logger.info("App Shutdown: TTS resource manager shutdown complete")
        except _IMPORT_EXCEPTIONS as e:
            logger.exception(f"App Shutdown: Error shutting down TTS resource manager: {e}")

    # Shutdown shared HTTP client sessions (aiohttp)
    try:
        from tldw_Server_API.app.core.http_client import shutdown_http_client

        await shutdown_http_client()
        logger.info("App Shutdown: HTTP client sessions shutdown complete")
    except _IMPORT_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error shutting down HTTP client sessions: {e}")

    # Shutdown ChaChaNotes executor and cached instances
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
            shutdown_chacha_resources,
        )

        await shutdown_chacha_resources()
        logger.info("App Shutdown: ChaChaNotes resources cleaned up")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error shutting down ChaChaNotes resources: {e}")

    # Shutdown Prompts DB cache and close worker
    try:
        from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import (
            close_all_cached_prompts_db_instances,
            stop_prompts_pending_close_worker,
        )

        await close_all_cached_prompts_db_instances()
        await stop_prompts_pending_close_worker()
        logger.info("App Shutdown: Prompts DB resources cleaned up")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error shutting down Prompts DB resources: {e}")

    # Shutdown Chat Workflows DB cache
    try:
        from tldw_Server_API.app.api.v1.API_Deps.chat_workflows_deps import (
            shutdown_chat_workflows_deps,
        )

        shutdown_chat_workflows_deps(app)
        logger.info("App Shutdown: Chat workflows resources cleaned up")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error shutting down chat workflows resources: {e}")

    # Shutdown Chat Module Components
    logger.info("App Shutdown: Cleaning up Chat module components...")

    # Shutdown Provider Manager
    try:
        if "provider_manager" in locals() and provider_manager is not None:
            await provider_manager.stop_health_checks()
            logger.info("App Shutdown: Provider manager stopped")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error stopping provider manager: {e}")

    # Shutdown Request Queue
    try:
        if "request_queue" in locals() and request_queue is not None:
            await request_queue.stop()
            logger.info("App Shutdown: Request queue stopped")
    except _IMPORT_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error stopping request queue: {e}")

    # Shutdown Local LLM manager (best effort)
    try:
        _llm_manager = getattr(app.state, "llm_manager", None)
        if _llm_manager is not None and hasattr(_llm_manager, "cleanup_on_exit"):
            await asyncio.to_thread(_llm_manager.cleanup_on_exit)
            logger.info("App Shutdown: Local LLM manager cleanup complete")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error cleaning up local LLM manager: {e}")

    with _timed_shutdown_segment(app, "evaluations_pool_shutdown"):
        # Shutdown Evaluations pool via lazy helper (no-op if never initialized)
        try:
            from tldw_Server_API.app.core.Evaluations.connection_pool import (
                shutdown_evaluations_pool_if_initialized as _shutdown_evals,
            )

            _shutdown_evals()
            logger.info("App Shutdown: Evaluations connection manager shutdown (lazy)")
        except _IMPORT_EXCEPTIONS as e:
            logger.debug(f"App Shutdown: Evaluations pool shutdown skipped/failed: {e}")

        # Shutdown Evaluations webhook manager (no-op if never initialized)
        try:
            from tldw_Server_API.app.core.Evaluations.webhook_manager import (
                shutdown_webhook_manager_if_initialized as _shutdown_webhooks,
            )

            _shutdown_webhooks()
            logger.info("App Shutdown: Evaluations webhook manager shutdown (lazy)")
        except _IMPORT_EXCEPTIONS as e:
            logger.debug(f"App Shutdown: Evaluations webhook manager shutdown skipped/failed: {e}")

    with _timed_shutdown_segment(app, "unified_audit_and_executor_shutdown"):
        # Shutdown Unified Audit Services (via DI cache)
        try:
            from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
                shutdown_all_audit_services,
            )

            logger.info("App Shutdown: Shutting down unified audit services...")
            await shutdown_all_audit_services()
            logger.info("App Shutdown: Unified audit services stopped")

            try:
                from tldw_Server_API.app.api.v1.endpoints.sharing import (
                    shutdown_sharing_audit_service,
                )

                await shutdown_sharing_audit_service()
                logger.info("App Shutdown: Sharing audit service stopped")
            except (*_STARTUP_GUARD_EXCEPTIONS, ImportError, ModuleNotFoundError) as _e:
                logger.debug(f"Sharing audit service shutdown skipped: {_e}")

            try:
                from tldw_Server_API.app.core.Embeddings.audit_adapter import (
                    shutdown_local_audit_adapter_loop,
                )

                shutdown_local_audit_adapter_loop()
                logger.info("App Shutdown: Embeddings audit adapter loop stopped")
            except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _e:
                logger.debug("Embeddings audit adapter loop shutdown skipped: {}", _e)

            try:
                from tldw_Server_API.app.core.Evaluations.audit_adapter import (
                    shutdown_local_evaluations_audit_loop,
                )

                shutdown_local_evaluations_audit_loop()
                logger.info("App Shutdown: Evaluations audit adapter loop stopped")
            except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _e:
                logger.debug("Evaluations audit adapter loop shutdown skipped: {}", _e)
        except _IMPORT_EXCEPTIONS as e:
            logger.exception(f"App Shutdown: Error stopping unified audit services: {e}")

        # Shutdown registered executors (thread/process pools)
        try:
            from tldw_Server_API.app.core.Utils.executor_registry import shutdown_all_registered_executors

            await shutdown_all_registered_executors(wait=True, cancel_futures=True)
            logger.info("App Shutdown: Registered executors shutdown")
            try:
                loop = asyncio.get_running_loop()
                if hasattr(loop, "shutdown_default_executor"):
                    await loop.shutdown_default_executor()
                    logger.info("App Shutdown: Default executor shutdown")
            except _STARTUP_GUARD_EXCEPTIONS as e:
                logger.debug(f"App Shutdown: Default executor shutdown skipped/failed: {e}")
        except _IMPORT_EXCEPTIONS as e:
            logger.exception(f"App Shutdown: Error shutting down executors: {e}")

        # Cleanup CPU pools
        try:
            from tldw_Server_API.app.core.Utils.cpu_bound_handler import cleanup_pools

            cleanup_pools()
            logger.info("App Shutdown: CPU pools cleaned up")
        except _STARTUP_GUARD_EXCEPTIONS as e:
            logger.exception(f"App Shutdown: Error cleaning up CPU pools: {e}")

    # Stop usage aggregator
    try:
        if "usage_aggregator" not in coordinated_legacy_component_names:
            if "usage_task" in locals() and usage_task:
                from tldw_Server_API.app.services.usage_aggregator import stop_usage_aggregator

                await stop_usage_aggregator(usage_task)
                logger.info("Usage aggregator stopped")
                usage_task = None
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.exception(f"App Shutdown: Error stopping usage aggregator: {e}")

    with _timed_shutdown_segment(app, "telemetry_shutdown"):
        # Shutdown telemetry
        try:
            # Stop AuthNZ scheduler if started
            try:
                if (
                    "_authnz_sched_started" in locals()
                    and _authnz_sched_started
                    and "authnz_scheduler" not in coordinated_legacy_component_names
                ):
                    from tldw_Server_API.app.core.AuthNZ.scheduler import stop_authnz_scheduler

                    await stop_authnz_scheduler()
                    logger.info("AuthNZ scheduler stopped")
            except _IMPORT_EXCEPTIONS as _e:
                logger.debug(f"AuthNZ scheduler shutdown skipped: {_e}")

            shutdown_telemetry()
            logger.info("App Shutdown: Telemetry shutdown")
        except _IMPORT_EXCEPTIONS as e:
            logger.exception(f"App Shutdown: Error shutting down telemetry: {e}")

    # Close cached MediaDatabase instances so Postgres pooled connections are released
    try:
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import reset_media_db_cache

        reset_media_db_cache()
        logger.info("App Shutdown: Media DB cache cleared")
    except _IMPORT_EXCEPTIONS as e:
        logger.debug(f"App Shutdown: Media DB cache cleanup skipped/failed: {e}")

    # Close shared content database backend pool (PostgreSQL content mode)
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Manager import (
            shutdown_content_backend as _shutdown_content_backend,
        )

        _shutdown_content_backend()
        logger.info("App Shutdown: Content DB backend pool closed")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.debug(f"App Shutdown: Content backend pool close skipped/failed: {e}")

    # Original test DB cleanup
    global test_db_instance_ref
    if test_db_instance_ref and hasattr(test_db_instance_ref, "close_all_connections"):
        logger.info("App Shutdown: Closing test DB connections")
        test_db_instance_ref.close_all_connections()
    else:
        logger.info("App Shutdown: No test DB instance found to close")

    # Reset the global jobs acquire gate after shutdown completes.
    # The gate is a process-wide flag meant to prevent new acquisitions during shutdown.
    # Leaving it enabled breaks in-process reuse patterns (e.g., pytest running workers after a TestClient closes).
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JM

        _JM.set_acquire_gate(False)
    except _STARTUP_GUARD_EXCEPTIONS:
        pass
    _record_shutdown_timing_total(
        app,
        int((time.monotonic() - shutdown_started) * 1000),
    )


#
############################# End of Test DB Handling###################

# Create FastAPI app with lifespan
# --- OpenAPI / Docs configuration ---
# Curated tag metadata to improve /docs grouping and clarity

from fastapi.openapi.utils import get_openapi


# Build absolute externalDocs URLs for OpenAPI (Pydantic v2 requires absolute URLs)
def _ext_url(path: str) -> str:
    base = _env_os.getenv("OPENAPI_EXTERNAL_DOCS_BASE_URL")
    if base and (base.startswith("http://") or base.startswith("https://")):
        return base.rstrip("/") + path
    fallback = _env_os.getenv("OPENAPI_SERVER_BASE_URL", "http://127.0.0.1:8000")
    return fallback.rstrip("/") + path


OPENAPI_TAGS = [
    {"name": "health", "description": "Health and status checks."},
    {
        "name": "authentication",
        "description": "AuthNZ endpoints for API key and JWT-based auth.",
        "externalDocs": {"description": "AuthNZ usage", "url": _ext_url("/docs-static/AUTHNZ_USAGE_EXAMPLES.md")},
    },
    {
        "name": "users",
        "description": "User management: create, list, roles, and profiles.",
        "externalDocs": {
            "description": "Permission matrix",
            "url": _ext_url("/docs-static/AUTHNZ_PERMISSION_MATRIX.md"),
        },
    },
    {
        "name": "organizations",
        "description": "Organization management: create orgs, manage membership, teams, and roles.",
    },
    {
        "name": "invites",
        "description": "Organization invite codes: preview, redeem, and audit.",
    },
    {
        "name": "billing",
        "description": "Billing and subscription management (plans, invoices, webhooks).",
    },
    {
        "name": "kanban",
        "description": "Kanban board endpoints: boards, lists, cards, and actions.",
    },
    {
        "name": "admin",
        "description": "Administrative operations and diagnostics (non-Jobs). For Jobs Admin endpoints (stats, prune, TTL sweep, requeue quarantined, integrity sweep), see the 'jobs' tag.",
        "externalDocs": {
            "description": "Jobs Admin Examples",
            "url": _ext_url("/docs-static/Code_Documentation/Jobs_Admin_Examples.md"),
        },
    },
    {
        "name": "jobs",
        "description": "Jobs queue manager and admin (SQLite/PG).",
        "externalDocs": {
            "description": "Jobs Manager ordering",
            "url": _ext_url("/docs-static/Code_Documentation/Jobs_Manager.md"),
        },
    },
    {
        "name": "media",
        "description": "Ingest and process media (video/audio/PDF/EPUB/HTML/Markdown).",
        "externalDocs": {"description": "Overview", "url": _ext_url("/docs-static/Documentation.md")},
    },
    {
        "name": "audio",
        "description": "Audio transcription and TTS (OpenAI-compatible).",
        "externalDocs": {"description": "Nemo STT setup", "url": _ext_url("/docs-static/NEMO_STT_DOCUMENTATION.md")},
    },
    {
        "name": "audio-websocket",
        "description": "Real-time streaming transcription over WebSocket.",
        "externalDocs": {
            "description": "Streaming STT",
            "url": _ext_url("/docs-static/NEMO_STREAMING_DOCUMENTATION.md"),
        },
    },
    {
        "name": "audio-jobs",
        "description": "Background audio processing via Jobs (fan-out pipeline).",
        "externalDocs": {
            "description": "Audio Jobs API",
            "url": _ext_url("/docs-static/API-related/Audio_Jobs_API.md"),
        },
    },
    {
        "name": "chat",
        "description": "Chat completions and conversation management (OpenAI-compatible).",
        "externalDocs": {
            "description": "Chat API",
            "url": _ext_url("/docs-static/API-related/Chat_API_Documentation.md"),
        },
    },
    {
        "name": "characters",
        "description": "Character cards/personas and related operations.",
        "externalDocs": {
            "description": "Character Chat API",
            "url": _ext_url("/docs-static/CHARACTER_CHAT_API_DOCUMENTATION.md"),
        },
    },
    {
        "name": "character-chat-sessions",
        "description": "Character chat sessions lifecycle management.",
        "externalDocs": {
            "description": "Character Chat API",
            "url": _ext_url("/docs-static/CHARACTER_CHAT_API_DOCUMENTATION.md"),
        },
    },
    {
        "name": "character-messages",
        "description": "Character message creation, retrieval, and search.",
        "externalDocs": {
            "description": "Character Chat API",
            "url": _ext_url("/docs-static/CHARACTER_CHAT_API_DOCUMENTATION.md"),
        },
    },
    {
        "name": "metrics",
        "description": "Metrics and monitoring endpoints.",
        "externalDocs": {"description": "Metrics design", "url": _ext_url("/docs-static/Design/Metrics.md")},
    },
    {"name": "monitoring", "description": "OpenTelemetry/metrics reporting in JSON."},
    {
        "name": "audit",
        "description": "Audit export, count, and tools. Includes /audit/export and /audit/count.",
        "externalDocs": {
            "description": "Audit Export & Count API",
            "url": _ext_url("/docs-static/API/Audit_Export.md"),
        },
    },
    {
        "name": "chunking",
        "description": "Content chunking operations and utilities.",
        "externalDocs": {"description": "Chunking design", "url": _ext_url("/docs-static/Design/Chunking.md")},
    },
    {
        "name": "chunking-templates",
        "description": "Chunking template management (create, list, update).",
        "externalDocs": {"description": "Templates", "url": _ext_url("/docs-static/Chunking_Templates.md")},
    },
    {
        "name": "embeddings",
        "description": "OpenAI-compatible embeddings generation.",
        "externalDocs": {
            "description": "Embeddings API Guide",
            "url": _ext_url("/docs-static/Embeddings/Embeddings-API-Guide.md"),
        },
    },
    {
        "name": "vector-stores",
        "description": "OpenAI-compatible vector store APIs (indexes, vectors).",
        "externalDocs": {
            "description": "Embedding & Vector Store Config",
            "url": _ext_url("/docs-static/Development/Embedding-and-Vectorstore-Config.md"),
        },
    },
    {
        "name": "claims",
        "description": "Claims extraction, indexing, and maintenance for media.",
        "externalDocs": {"description": "Claims design", "url": _ext_url("/docs-static/Design/ingestion_claims.md")},
    },
    {
        "name": "media-embeddings",
        "description": "Generate embeddings for uploaded/ingested media.",
        "externalDocs": {
            "description": "Embeddings docs",
            "url": _ext_url("/docs-static/Embeddings/Embeddings-Documentation.md"),
        },
    },
    {"name": "notes", "description": "Notes and knowledge management."},
    {"name": "writing", "description": "Writing Playground sessions, templates, themes, and token utilities."},
    {
        "name": "data-tables",
        "description": "Data table generation jobs and CRUD.",
    },
    {
        "name": "notes-graph",
        "description": "Graph of notes, tags, and sources.",
        "externalDocs": {"description": "Graphing PRD", "url": _ext_url("/docs-static/Design/Graphing-Notes-PRD.md")},
    },
    {
        "name": "prompts",
        "description": "Prompt library management (import/export).",
        "externalDocs": {"description": "Prompts design", "url": _ext_url("/docs-static/Design/Prompts.md")},
    },
    {
        "name": "prompt-studio",
        "description": "Projects, prompts, tests, optimization, and background jobs (experimental).",
        "externalDocs": {
            "description": "Prompt Studio API",
            "url": _ext_url("/docs-static/API-related/Prompt_Studio_API.md"),
        },
    },
    {
        "name": "rag-health",
        "description": "RAG health, caching, and metrics.",
        "externalDocs": {"description": "RAG notes", "url": _ext_url("/docs-static/RAG_Notes.md")},
    },
    {
        "name": "rag-unified",
        "description": "Unified RAG: FTS5 + embeddings + re-ranking.",
        "externalDocs": {"description": "RAG notes", "url": _ext_url("/docs-static/RAG_Notes.md")},
    },
    {
        "name": "feedback",
        "description": "User feedback capture for RAG quality and relevance signals.",
        "externalDocs": {
            "description": "Feedback system design",
            "url": _ext_url("/docs-static/Design/Feedback_System.md"),
        },
    },
    {
        "name": "workflows",
        "description": "Workflow definitions and execution (scaffolding, experimental).",
        "externalDocs": {"description": "Workflows", "url": _ext_url("/docs-static/Design/Workflows.md")},
    },
    {
        "name": "research",
        "description": "Research providers and web data collection.",
        "externalDocs": {"description": "Researcher", "url": _ext_url("/docs-static/Design/Researcher.md")},
    },
    {
        "name": "paper-search",
        "description": "Provider-specific paper search (arXiv, BioRxiv/MedRxiv, PubMed, Semantic Scholar).",
        "externalDocs": {"description": "Paper Search", "url": _ext_url("/docs-static/Design/PaperSearch.md")},
    },
    {
        "name": "evaluations",
        "description": "Unified evaluation APIs (geval, batch, metrics).",
        "externalDocs": {"description": "Eval report", "url": _ext_url("/docs-static/EVALUATION_TEST_REPORT.md")},
    },
    {
        "name": "benchmarks",
        "description": "Benchmarking endpoints and utilities.",
        "externalDocs": {"description": "RAG benchmarks", "url": _ext_url("/docs-static/RAG_Benchmarks.md")},
    },
    {"name": "config", "description": "Server configuration and capability info."},
    {"name": "sync", "description": "Synchronization operations and helpers."},
    {"name": "tools", "description": "Tooling endpoints (utilities)."},
    {
        "name": "mcp-unified",
        "description": "MCP server + endpoints (JWT/RBAC) - experimental surface in 0.1.",
        "externalDocs": {
            "description": "MCP Unified Developer Guide",
            "url": _ext_url("/docs-static/MCP/Unified/Developer_Guide.md"),
        },
    },
    {"name": "flashcards", "description": "Flashcards/Decks (ChaChaNotes)"},
    {"name": "quizzes", "description": "Quizzes (ChaChaNotes)"},
    {
        "name": "chatbooks",
        "description": "Import/export chatbooks (backup/restore).",
        "externalDocs": {
            "description": "Chatbooks API",
            "url": _ext_url("/docs-static/API-related/Chatbook_Features_API_Documentation.md"),
        },
    },
    {
        "name": "llm",
        "description": "LLM provider configuration and discovery.",
        "externalDocs": {
            "description": "Chat developer guide",
            "url": _ext_url("/docs-static/Code_Documentation/Chat_Developer_Guide.md"),
        },
    },
    {
        "name": "llamacpp",
        "description": "Llama.cpp helpers and management.",
        "externalDocs": {
            "description": "Inference engines",
            "url": _ext_url("/docs-static/Design/Inference_Engines.md"),
        },
    },
    {
        "name": "web-scraping",
        "description": "Web scraping management and job control.",
        "externalDocs": {"description": "Web scraping design", "url": _ext_url("/docs-static/Design/WebScraping.md")},
    },
    {
        "name": "chat-dictionaries",
        "description": "Per-user/domain dictionaries for chat preprocessing and postprocessing.",
        "externalDocs": {
            "description": "Character Chat API",
            "url": _ext_url("/docs-static/CHARACTER_CHAT_API_DOCUMENTATION.md"),
        },
    },
    {"name": "chat-documents", "description": "Generate documents from conversations and templates."},
    {
        "name": "personalization",
        "description": "Opt-in user profiles, memories, and RAG biasing.",
        "externalDocs": {
            "description": "Personalization design",
            "url": _ext_url("/docs-static/Design/Personalization_Design.md"),
        },
    },
    {
        "name": "persona",
        "description": "Persona agent (voice, tools, MCP).",
        "externalDocs": {
            "description": "Persona design",
            "url": _ext_url("/docs-static/Design/Persona_Agent_Design.md"),
        },
    },
]


_prod_flag = _env_os.getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}

APP_DESCRIPTION = """
    Too Long; Didn't Watch Server (tldw_server) - unified research assistant and media analysis platform.

    Auth: Click the “Authorize” button.
    - Single-user mode: use header X-API-KEY with the printed key.
    - Multi-user mode: use Bearer JWT tokens (login endpoints under authentication).

    Highlights
    - Media ingestion (video/audio/docs) with automatic metadata
    - STT (file + real-time WS) and TTS (OpenAI-compatible)
    - RAG: SQLite FTS5 + embeddings + re-ranking
    - Chat: OpenAI-compatible /chat/completions across providers
    - Notes, prompts, evaluations, MCP Unified server

    Helpful paths
    - Quickstart: /api/v1/config/quickstart
    - Setup UI: /setup
    - OpenAPI JSON: /openapi.json
    - Metrics: /metrics and /api/v1/metrics
    """.strip()

# Always expose docs and redoc; remove ENABLE_OPENAPI toggle
_docs_url = "/docs"
_redoc_url = "/redoc"
# Always serve OpenAPI JSON regardless of docs gating
_openapi_url = "/openapi.json"

_startup_trace("Creating FastAPI app instance")

# Prefer locally-served Swagger UI assets when available to avoid CSP/CDN issues
_swagger_static_dir = BASE_DIR / "static" / "swagger"
_swagger_bundle = _swagger_static_dir / "swagger-ui-bundle.js"
_swagger_css = _swagger_static_dir / "swagger-ui.css"
_swagger_use_local = _swagger_bundle.exists() and _swagger_css.exists()
_swagger_ui_js_url = "/static/swagger/swagger-ui-bundle.js" if _swagger_use_local else None
_swagger_ui_css_url = "/static/swagger/swagger-ui.css" if _swagger_use_local else None

# Merge Swagger UI parameters and include our overrides via customCssUrl
_swagger_ui_params = {
    "displayRequestDuration": True,
    "deepLinking": True,
    "docExpansion": "none",
    "defaultModelsExpandDepth": -1,
    "defaultModelExpandDepth": 2,
    "persistAuthorization": True,
    "tryItOutEnabled": True,
    "tagsSorter": "alpha",
    "operationsSorter": "alpha",
    "filter": True,
    # Inject our optional overrides stylesheet without replacing the base CSS
    "customCssUrl": "/static/swagger-overrides.css",
}

app = FastAPI(
    title="tldw API",
    version="0.1.0",
    description=APP_DESCRIPTION,
    terms_of_service="https://github.com/cpacker/tldw_server",
    contact={
        "name": "tldw_server Maintainers",
        "url": "https://github.com/cpacker/tldw_server/issues",
    },
    license_info={
        "name": "GNU GPL v2.0",
        "url": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html",
    },
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters=_swagger_ui_params,
    swagger_ui_js_url=_swagger_ui_js_url,
    swagger_ui_css_url=_swagger_ui_css_url,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
    lifespan=lifespan,
)
_startup_trace("FastAPI app created")


def _iter_route_method_pairs(app: FastAPI) -> list[tuple[str, str, str]]:
    """Return explicit route method/path pairs for duplicate detection."""
    rows: list[tuple[str, str, str]] = []
    for route in getattr(app, "routes", []):
        if not isinstance(route, APIRoute):
            continue
        path = str(getattr(route, "path", "") or "")
        if not path:
            continue
        methods = set(getattr(route, "methods", set()) or set())
        for method in sorted(methods):
            method_upper = str(method).upper()
            # Ignore framework-generated methods to keep duplicate checks focused.
            if method_upper in {"HEAD", "OPTIONS"}:
                continue
            rows.append((path, method_upper, str(getattr(route, "name", "<unnamed>"))))
    return rows


def _fail_on_duplicate_route_method_pairs(app: FastAPI, *, context: str) -> None:
    seen: dict[tuple[str, str], str] = {}
    duplicates: list[tuple[str, str, str, str]] = []
    for path, method, route_name in _iter_route_method_pairs(app):
        key = (path, method)
        previous = seen.get(key)
        if previous is None:
            seen[key] = route_name
            continue
        duplicates.append((path, method, previous, route_name))
    if not duplicates:
        return

    sample = "; ".join(
        f"{method} {path} ({first} vs {second})"
        for path, method, first, second in duplicates[:10]
    )
    message = (
        f"Duplicate route registrations detected during {context}: "
        f"{len(duplicates)} duplicate (path, method) pairs. Sample: {sample}"
    )
    logger.critical(message)
    raise RuntimeError(message)


def _resolve_cors_origins_or_raise(allowed_origins: list[str] | None) -> list[str]:
    origins = [str(origin).strip() for origin in (allowed_origins or []) if str(origin).strip()]
    if origins:
        return origins
    message = (
        "CORS is enabled but ALLOWED_ORIGINS is empty. "
        "Set ALLOWED_ORIGINS to a non-empty list (for example: ['http://localhost:3000']) "
        "or set ALLOWED_ORIGINS='*' with CORS_ALLOW_CREDENTIALS=false for local development."
    )
    logger.critical(message)
    raise RuntimeError(message)


def _validate_cors_configuration_or_raise(
    origins: list[str],
    *,
    allow_credentials: bool,
    enforce_explicit_origins: bool = False,
) -> None:
    """Reject invalid CORS combinations at startup."""
    if enforce_explicit_origins and "*" in origins:
        message = (
            "Invalid CORS configuration: ALLOWED_ORIGINS cannot include '*' in production. "
            "Configure explicit origins instead."
        )
        logger.critical(message)
        raise RuntimeError(message)

    if allow_credentials and "*" in origins:
        message = (
            "Invalid CORS configuration: ALLOWED_ORIGINS cannot include '*' "
            "when credentials are enabled. Configure explicit origins instead."
        )
        logger.critical(message)
        raise RuntimeError(message)


_DEV_PRIVATE_NETWORK_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost"
    r"|127(?:\.\d{1,3}){3}"
    r"|10(?:\.\d{1,3}){3}"
    r"|192\.168(?:\.\d{1,3}){2}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
    r")(?::\d{1,5})?$"
)


def _compute_dev_cors_origin_regex(
    origins: list[str],
    *,
    enforce_explicit_origins: bool,
) -> str | None:
    """Allow common localhost/private-LAN web UIs in non-production without widening prod CORS."""
    if enforce_explicit_origins or "*" in origins:
        return None
    return _DEV_PRIVATE_NETWORK_ORIGIN_REGEX


def _compute_openapi_cors_allow_origin(
    origin: str | None,
    *,
    allow_all_origins: bool,
    allow_credentials: bool,
    allowed_openapi_origins: set[str],
) -> str | None:
    """Return the value to emit for Access-Control-Allow-Origin on OpenAPI responses."""
    if allow_all_origins:
        if allow_credentials:
            return origin or None
        return "*"
    if not origin:
        return None
    normalized_origin = str(origin).rstrip("/")
    if normalized_origin in allowed_openapi_origins:
        return origin
    return None


_cors_allow_all_origins = False
_cors_allow_credentials = False
_cors_allow_origin_regex: str | None = None
_cors_allowed_openapi_origins: set[str] = set()


def _compute_runtime_cors_allow_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    if _cors_allow_all_origins and not _cors_allow_credentials:
        return "*"

    normalized_origin = str(origin).rstrip("/")
    if normalized_origin in _cors_allowed_openapi_origins:
        return origin

    if _cors_allow_origin_regex:
        try:
            import re as _re

            if _re.match(_cors_allow_origin_regex, origin):
                return origin
        except _REQUEST_GUARD_EXCEPTIONS:
            return None

    if _cors_allow_all_origins:
        return origin
    return None


def _apply_runtime_cors_headers(request: Request, response: Any) -> Any:
    allow_origin = _compute_runtime_cors_allow_origin(request.headers.get("origin"))
    if not allow_origin:
        return response

    response.headers.setdefault("Access-Control-Allow-Origin", allow_origin)
    if allow_origin != "*":
        response.headers.setdefault("Vary", "Origin")
    if _cors_allow_credentials:
        response.headers.setdefault("Access-Control-Allow-Credentials", "true")
    response.headers.setdefault(
        "Access-Control-Expose-Headers",
        "X-Request-ID, traceparent, X-Trace-Id"
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler – surfaces tracebacks that BaseHTTPMiddleware
# layers would otherwise swallow, producing only a bare
# "Exception in ASGI application" in the uvicorn log.
# ---------------------------------------------------------------------------
from fastapi.responses import JSONResponse as _JSONResponse  # noqa: E402


@app.exception_handler(Exception)
async def _global_unhandled_exception_handler(request, exc):
    if isinstance(exc, ClientDisconnect):
        logger.debug(
            "Client disconnected during {method} {url}",
            method=request.method,
            url=request.url,
        )
        return _JSONResponse(
            status_code=499,
            content={"detail": "Client disconnected"},
        )

    logger.opt(exception=exc).error(
        "Unhandled exception on {method} {url}: {exc}",
        method=request.method,
        url=request.url,
        exc=exc,
    )
    return _apply_runtime_cors_headers(
        request,
        _JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        ),
    )


@app.exception_handler(ClientDisconnect)
async def _client_disconnect_exception_handler(request: Request, exc: ClientDisconnect):
    logger.debug(
        "Client disconnected during {method} {url}",
        method=request.method,
        url=request.url,
    )
    return _apply_runtime_cors_headers(
        request,
        _JSONResponse(
            status_code=499,
            content={"detail": "Client disconnected"},
        ),
    )


# Early middleware to guard workflow templates path traversal attempts

from starlette.responses import JSONResponse  # noqa: E402

try:
    # Determine whether to enable RGSimpleMiddleware.
    # - When global RG is enabled (RG_ENABLED / config), ingress enforcement is on by default.
    # - Tests that want RG ingress should set RG_ENABLED=1 explicitly; we avoid
    #   enabling middleware purely due to pytest/minimal-test settings to prevent
    #   unintended 429s in unrelated suites.
    from tldw_Server_API.app.core.config import rg_enabled as _rg_enabled_flag  # noqa: E402

    try:
        _rg_global_enabled = bool(_rg_enabled_flag(False))
    except _STARTUP_GUARD_EXCEPTIONS:
        _rg_global_enabled = False

    if _rg_global_enabled:
        from tldw_Server_API.app.core.Resource_Governance.middleware_simple import (
            RGSimpleMiddleware as _RGMw,
        )  # noqa: E402

        # Avoid double-adding
        try:
            already = any(getattr(m, "cls", None) is _RGMw for m in getattr(app, "user_middleware", []))
        except _STARTUP_GUARD_EXCEPTIONS:
            already = False
        if not already:
            app.add_middleware(_RGMw)
            logger.info("RGSimpleMiddleware enabled (RG_ENABLED)")
except _STARTUP_GUARD_EXCEPTIONS as _rg_mw_err:
    logger.debug(f"RGSimpleMiddleware not enabled: {_rg_mw_err}")


@app.middleware("http")
async def _guard_workflow_templates_traversal(request, call_next):
    try:
        p = request.url.path or ""
        # Only inspect under the workflows templates prefix
        prefix = "/api/v1/workflows/templates/"
        if p.startswith(prefix):
            tail = p[len(prefix) :]
            # If any traversal segments are found in the raw path, reject early with 400
            # This runs before route resolution so it also handles router-level 404 shortcuts.
            if ".." in tail.split("/"):
                return JSONResponse({"detail": "Invalid template name"}, status_code=400)
    except _REQUEST_GUARD_EXCEPTIONS:
        pass
    return await call_next(request)


# Early middleware to guard sandbox artifact path traversal/double-slash before Starlette routing
@app.middleware("http")
async def _guard_sandbox_artifact_path(request: Request, call_next):
    try:
        # Inspect raw ASGI path first to avoid client/Starlette normalization
        raw_path = request.scope.get("raw_path")
        path_raw = (
            raw_path.decode("utf-8", "ignore") if isinstance(raw_path, (bytes, bytearray)) else (request.url.path or "")
        )
        # Debug logging removed after verification
        # Quick filter: only check sandbox artifact endpoints
        # Example: /api/v1/sandbox/runs/{run_id}/artifacts/{path}
        if "/api/v1/sandbox/runs/" in path_raw and "/artifacts/" in path_raw:
            from urllib.parse import unquote

            # Segment after /artifacts/
            idx = path_raw.find("/artifacts/")
            tail = path_raw[idx + len("/artifacts/") :]
            tail_unquoted = unquote(tail)
            # Reject traversal attempts and absolute/double-slash paths
            if ".." in tail_unquoted.split("/") or tail_unquoted.startswith("/") or "//" in tail:
                return JSONResponse({"detail": "invalid_path"}, status_code=400)
    except _REQUEST_GUARD_EXCEPTIONS:
        # Fail open: if guard fails, let the request proceed
        pass
    return await call_next(request)


# Add global security schemes, servers, and branding to the generated OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Servers for common deployments
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "http://127.0.0.1:8000", "description": "Loopback"},
    ]

    # Security schemes to document both supported auth modes
    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes.update(
        {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-KEY",
                "description": "Single-user mode API key authentication.",
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "Multi-user mode JWT bearer token.",
            },
        }
    )

    # Optional: top-level external docs and logo
    openapi_schema["externalDocs"] = {
        "description": "Project documentation",
        "url": "/docs-static",
    }
    openapi_schema.setdefault("info", {}).setdefault("x-logo", {"url": "/static/favicon.ico"})

    # Default security: show lock icons by default in Swagger UI
    # Endpoints can override with openapi_extra={"security": []} to be public
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"BearerAuth": []},
    ]

    # ReDoc tag grouping for better navigation in /redoc
    openapi_schema["x-tagGroups"] = [
        {
            "name": "Core",
            "tags": ["health", "authentication", "users", "admin"],
        },
        {
            "name": "Media",
            "tags": ["media", "audio", "media-embeddings", "web-scraping", "research", "paper-search"],
        },
        {
            "name": "Chat & TTS",
            "tags": [
                "chat",
                "chat-dictionaries",
                "chat-documents",
                "audio-websocket",
                "characters",
                "character-chat-sessions",
                "character-messages",
                "persona",
            ],
        },
        {
            "name": "RAG & Evals",
            "tags": ["rag-health", "rag-unified", "feedback", "evaluations", "benchmarks"],
        },
        {
            "name": "Embeddings & Vectors",
            "tags": ["embeddings", "vector-stores", "claims"],
        },
        {
            "name": "Studio & Knowledge",
            "tags": ["prompt-studio", "prompts", "notes", "personalization", "chatbooks", "tools"],
        },
        {
            "name": "Infra",
            "tags": ["metrics", "monitoring", "config", "sync", "llm", "llamacpp", "mcp-unified", "workflows"],
        },
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Display API key information on startup for single user mode
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, is_single_user_mode


async def _display_startup_info_and_warm():
    """Startup banner and optional warm-up tasks (moved to lifespan)."""
    if is_single_user_mode():
        settings = get_settings()
        api_key = settings.SINGLE_USER_API_KEY
        _display_key = _startup_api_key_log_value(api_key)
        _masked_note = ""
        if _display_key != api_key:
            _masked_note = " (masked; set SHOW_API_KEY_ON_STARTUP=true to display once)"
        logger.info("=" * 70)
        logger.info("🚀 TLDW Server Started in SINGLE USER MODE")
        logger.info("=" * 70)
        logger.info("📌 API Key for authentication:")
        logger.info(f"   {_display_key}{_masked_note}")
        logger.info("🌐 Access URLs:")
        logger.info("   Quickstart: http://localhost:8000/api/v1/config/quickstart")
        logger.info("   Setup UI:   http://localhost:8000/setup (if required)")
        logger.info("   API Docs:   http://localhost:8000/docs")
        logger.info("   ReDoc:      http://localhost:8000/redoc")
        logger.info("💡 Use this API key in X-API-KEY for API requests")
        logger.info("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("🚀 TLDW Server Started in MULTI-USER MODE")
        logger.info("=" * 70)
        logger.info("Authentication required via JWT tokens")
        logger.info("=" * 70)

    # Optional pre-warm
    try:
        await asyncio.to_thread(get_cached_evaluation_manager)
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.exception(f"Failed to initialize evaluation manager during startup: {exc}")
    try:
        if needs_setup():
            logger.info("First-time setup is enabled. Open http://localhost:8000/setup to configure the server.")
    except FileNotFoundError:
        logger.warning("Configuration file missing; unable to determine setup state. Ensure config.txt exists.")


# --- FIX: Add CORS Middleware ---
# Import from config
from tldw_Server_API.app.core.config import (
    ALLOWED_ORIGINS,
    API_V1_PREFIX,
    resolve_runtime_allowed_origins,
    is_production_environment,
    route_enabled,
    should_allow_cors_credentials,
    should_disable_cors,
)

# FIXME - CORS
if should_disable_cors():
    logger.warning("CORS middleware disabled via configuration/ENV flag.")
else:
    origins, _cors_origin_source, _cors_origin_fallback = resolve_runtime_allowed_origins(ALLOWED_ORIGINS)
    if _cors_origin_fallback:
        logger.warning(
            "ALLOWED_ORIGINS resolved to an empty list outside production. "
            "Using local browser defaults (localhost/127.0.0.1) so self-hosted setup keeps working. "
            "Set ALLOWED_ORIGINS only if you need a different browser origin."
        )

    # C1: Auto-add common localhost origins in single-user mode when no explicit
    # ALLOWED_ORIGINS env var is set. In multi-user mode, require explicit origins.
    _env_allowed_origins_set = os.getenv("ALLOWED_ORIGINS") is not None
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_cors_settings
        _cors_auth_mode = _get_cors_settings().AUTH_MODE
    except Exception:
        _cors_auth_mode = os.getenv("AUTH_MODE", "single_user")

    _SINGLE_USER_LOCALHOST_ORIGINS = [
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:8080",
    ]

    if str(_cors_auth_mode) == "single_user" and not _env_allowed_origins_set:
        _auto_added = []
        for _origin in _SINGLE_USER_LOCALHOST_ORIGINS:
            if _origin not in origins:
                origins.append(_origin)
                _auto_added.append(_origin)
        if _auto_added:
            logger.info(
                f"CORS single-user auto-detect: added localhost origins {_auto_added}"
            )
        else:
            logger.info("CORS single-user auto-detect: all common localhost origins already present.")
    elif str(_cors_auth_mode) == "multi_user" and not origins:
        logger.warning(
            "CORS multi-user mode: ALLOWED_ORIGINS is empty. "
            "Set ALLOWED_ORIGINS explicitly for multi-user deployments."
        )
    origins = _resolve_cors_origins_or_raise(origins)
    _cors_allow_credentials = should_allow_cors_credentials()
    _cors_enforce_explicit_origins = is_production_environment()
    _validate_cors_configuration_or_raise(
        origins,
        allow_credentials=_cors_allow_credentials,
        enforce_explicit_origins=_cors_enforce_explicit_origins,
    )
    _cors_allow_all_origins = "*" in origins
    _cors_allow_origin_regex = _compute_dev_cors_origin_regex(
        origins,
        enforce_explicit_origins=_cors_enforce_explicit_origins,
    )
    _cors_allowed_openapi_origins = {str(o).rstrip("/") for o in origins if isinstance(o, str)}
    try:
        app.state._tldw_drain_gate_cors_config = {
            "allow_all_origins": _cors_allow_all_origins,
            "allow_origin_regex": _cors_allow_origin_regex,
            "allow_credentials": _cors_allow_credentials,
            "allowed_origins": _cors_allowed_openapi_origins,
            "expose_headers": "X-Request-ID, traceparent, X-Trace-Id",
        }
    except _STARTUP_GUARD_EXCEPTIONS:
        pass
    # # -- If you have any global middleware, add it here --
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=_cors_allow_origin_regex,
        allow_credentials=_cors_allow_credentials,
        allow_methods=["*"],  # Must include OPTIONS, GET, POST, DELETE etc.
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "traceparent", "X-Trace-Id"],
    )

    # Ensure OpenAPI schema is consumable across common local origins (helpful when docs are
    # viewed via alternate hostnames like 127.0.0.1 vs localhost). We only set headers if the
    # CORS middleware didn't already do so.
    @app.middleware("http")
    async def _openapi_cors_helper(request, call_next):
        response = await call_next(request)
        try:
            if request.url.path == _openapi_url:
                origin = request.headers.get("origin")
                allow_origin = _compute_openapi_cors_allow_origin(
                    origin,
                    allow_all_origins=_cors_allow_all_origins,
                    allow_credentials=_cors_allow_credentials,
                    allowed_openapi_origins=_cors_allowed_openapi_origins,
                )
                if allow_origin:
                    response.headers.setdefault("Access-Control-Allow-Origin", allow_origin)
                    if allow_origin != "*":
                        response.headers.setdefault("Vary", "Origin")
                response.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
                response.headers.setdefault("Access-Control-Allow-Headers", "*")
                response.headers.setdefault("Access-Control-Expose-Headers", "X-Request-ID, traceparent, X-Trace-Id")
        except _REQUEST_GUARD_EXCEPTIONS:
            pass
        return response


# Add CSRF Protection Middleware (NEW) with friendly error logging for misconfiguration
from tldw_Server_API.app.core.AuthNZ.csrf_protection import add_csrf_protection

try:
    add_csrf_protection(app)
except _STARTUP_GUARD_EXCEPTIONS as _csrf_e:
    logger.exception(f"Failed to configure CSRF middleware: {_csrf_e}")
    logger.exception(
        "Auth configuration error. If running in single-user mode, ensure SINGLE_USER_API_KEY is set.\n"
        "If running in multi-user mode, ensure JWT_SECRET_KEY is set (>=32 chars).\n"
        "See README: Authentication Setup and .env templates."
    )
    raise

# Static files serving
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Security middleware (headers + request size limit)
from tldw_Server_API.app.core.AuthNZ.llm_budget_middleware import LLMBudgetMiddleware
from tldw_Server_API.app.core.AuthNZ.usage_logging_middleware import UsageLoggingMiddleware
from tldw_Server_API.app.core.Metrics.http_middleware import HTTPMetricsMiddleware
from tldw_Server_API.app.core.Sandbox.middleware import SandboxArtifactTraversalGuardMiddleware
from tldw_Server_API.app.core.Security.drain_gate_middleware import DrainGateMiddleware
from tldw_Server_API.app.core.Security.middleware import SecurityHeadersMiddleware
from tldw_Server_API.app.core.Security.request_id_middleware import RequestIDMiddleware
from tldw_Server_API.app.core.Security.setup_access_guard import SetupAccessGuardMiddleware
from tldw_Server_API.app.core.Security.setup_csp import SetupCSPMiddleware
from tldw_Server_API.app.core.testing import (
    env_flag_enabled as _test_env_flag_enabled,
)
from tldw_Server_API.app.core.testing import (
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
)
from tldw_Server_API.app.core.testing import (
    is_test_mode as _shared_is_test_mode,
)

_TEST_FLAGS_SET = _shared_is_test_mode() or _test_env_flag_enabled("TESTING")
_EXPLICIT_PYTEST_RUNTIME = _is_explicit_pytest_runtime()
_TEST_MODE = _EXPLICIT_PYTEST_RUNTIME and (
    _TEST_FLAGS_SET or bool(_env_os.getenv("PYTEST_CURRENT_TEST"))
)

if _TEST_FLAGS_SET and not _EXPLICIT_PYTEST_RUNTIME:
    logger.warning(
        "Test flags are set without explicit pytest runtime; startup guard will reject this configuration."
    )

if _TEST_MODE:
    logger.info("TEST_MODE detected: Skipping non-essential middlewares (security headers, metrics, usage logging)")
    # Apply Setup CSP nonce injection even in tests to keep behavior consistent
    try:
        app.add_middleware(SetupCSPMiddleware)
    except _STARTUP_GUARD_EXCEPTIONS as _e:
        logger.debug(f"Skipping SetupCSPMiddleware in tests: {_e}")
    # Guard Setup remote access in tests too (should evaluate loopback as allowed)
    try:
        app.add_middleware(SetupAccessGuardMiddleware)
    except _STARTUP_GUARD_EXCEPTIONS as _e:
        logger.debug(f"Skipping SetupAccessGuardMiddleware in tests: {_e}")

    # Sandbox artifact traversal guard (pre-routing)
    try:
        app.add_middleware(SandboxArtifactTraversalGuardMiddleware)
    except _IMPORT_EXCEPTIONS as _e:
        logger.debug(f"Skipping SandboxArtifactTraversalGuardMiddleware in tests: {_e}")

    @app.middleware("http")
    async def _trace_headers_middleware(request: Request, call_next):
        from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager

        tm = get_tracing_manager()
        # Ensure request_id is in baggage (RequestIDMiddleware already set it)
        try:
            req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID")
            if req_id:
                tm.set_baggage("request_id", str(req_id))
        except _REQUEST_GUARD_EXCEPTIONS as _baggage_err:
            logger.debug(f"Trace headers: failed to set baggage request_id: {_baggage_err}")
        response = await call_next(request)
        # Add trace headers to response
        try:
            span = tm.get_current_span()
            if span:
                ctx = span.get_span_context()
                if ctx and getattr(ctx, "is_valid", False):
                    trace_id = f"{ctx.trace_id:032x}"
                    span_id = f"{ctx.span_id:016x}"
                    response.headers.setdefault("X-Trace-Id", trace_id)
                    response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
                else:
                    # No active span; synthesize a valid W3C traceparent for tests
                    try:
                        from secrets import token_hex as _th

                        trace_id = _th(16)  # 32 hex chars
                        span_id = _th(8)  # 16 hex chars
                        response.headers.setdefault("X-Trace-Id", trace_id)
                        response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
                    except _REQUEST_GUARD_EXCEPTIONS as _synth_err:
                        logger.debug(f"Trace headers: failed to synthesize traceparent: {_synth_err}")
            else:
                # No span; synthesize trace headers
                try:
                    from secrets import token_hex as _th

                    trace_id = _th(16)
                    span_id = _th(8)
                    response.headers.setdefault("X-Trace-Id", trace_id)
                    response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
                except _REQUEST_GUARD_EXCEPTIONS as _synth_err2:
                    logger.debug(f"Trace headers: failed to synthesize trace headers (no-span case): {_synth_err2}")
        except _REQUEST_GUARD_EXCEPTIONS as _trace_hdr_err:
            logger.debug(f"Trace headers: middleware error while setting headers: {_trace_hdr_err}")
        return response

else:
    _enable_sec_headers_env = _env_os.getenv("ENABLE_SECURITY_HEADERS")
    _enable_sec_headers = (
        True
        if (_prod_flag and _enable_sec_headers_env is None)
        else ((_enable_sec_headers_env or "true").lower() in {"true", "1", "yes", "y", "on"})
    )
    # Apply Setup CSP nonce injection before security headers
    try:
        app.add_middleware(SetupCSPMiddleware)
    except _STARTUP_GUARD_EXCEPTIONS as _e:
        logger.debug(f"Skipping SetupCSPMiddleware: {_e}")
    # Enforce Setup remote access policy
    try:
        app.add_middleware(SetupAccessGuardMiddleware)
    except _STARTUP_GUARD_EXCEPTIONS as _e:
        logger.debug(f"Skipping SetupAccessGuardMiddleware: {_e}")

    if _enable_sec_headers:
        app.add_middleware(SecurityHeadersMiddleware, enabled=True)

    # HTTP request metrics middleware (records count and latency per route)
    app.add_middleware(HTTPMetricsMiddleware)

    # Structured access logs (request_id, method, host, status, duration)
    try:
        from tldw_Server_API.app.core.Logging.access_log_middleware import AccessLogMiddleware

        app.add_middleware(AccessLogMiddleware)
    except _IMPORT_EXCEPTIONS as _e:
        logger.debug(f"Skipping AccessLogMiddleware: {_e}")

    # Sandbox artifact traversal guard (pre-routing)
    try:
        app.add_middleware(SandboxArtifactTraversalGuardMiddleware)
    except _IMPORT_EXCEPTIONS as _e:
        logger.debug(f"Skipping SandboxArtifactTraversalGuardMiddleware: {_e}")

    # Per-request usage logging (guarded by settings flag)
    app.add_middleware(UsageLoggingMiddleware)

    # Add trace headers middleware: propagate trace context to HTTP responses
    @app.middleware("http")
    async def _trace_headers_middleware(request: Request, call_next):
        from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager

        tm = get_tracing_manager()
        # Ensure request_id is in baggage (RequestIDMiddleware already set it)
        try:
            req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID")
            if req_id:
                tm.set_baggage("request_id", str(req_id))
        except _REQUEST_GUARD_EXCEPTIONS:
            pass
        response = await call_next(request)
        # Add trace headers to response
        try:
            span = tm.get_current_span()
            if span:
                ctx = span.get_span_context()
                if ctx and getattr(ctx, "is_valid", False):
                    trace_id = f"{ctx.trace_id:032x}"
                    span_id = f"{ctx.span_id:016x}"
                    response.headers.setdefault("X-Trace-Id", trace_id)
                    response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
                else:
                    # No active span; synthesize a valid W3C traceparent
                    try:
                        from secrets import token_hex as _th

                        trace_id = _th(16)
                        span_id = _th(8)
                        response.headers.setdefault("X-Trace-Id", trace_id)
                        response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
                    except _REQUEST_GUARD_EXCEPTIONS:
                        pass
            else:
                # No span; synthesize trace headers
                try:
                    from secrets import token_hex as _th

                    trace_id = _th(16)
                    span_id = _th(8)
                    response.headers.setdefault("X-Trace-Id", trace_id)
                    response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
                except _REQUEST_GUARD_EXCEPTIONS:
                    pass
        except _REQUEST_GUARD_EXCEPTIONS:
            pass
        return response


# Always apply LLM budget middleware (guarded by settings) even in tests so allowlists/budgets are enforced
try:
    app.add_middleware(LLMBudgetMiddleware)
except _STARTUP_GUARD_EXCEPTIONS as _e:
    logger.debug(f"Skipping LLMBudgetMiddleware: {_e}")

# Request ID context should be available before the drain gate, and the drain gate
# should reject work before the LLM budget middleware gets a chance to do heavier setup.
app.add_middleware(DrainGateMiddleware)
app.add_middleware(RequestIDMiddleware)

# Keep Setup UI HTML outside the static mounts to avoid bypassing the
# /setup gating via direct file access.
SETUP_PAGE_PATH = BASE_DIR / "Setup_UI" / "setup.html"


async def serve_setup_page():
    """Serve the first-time setup UI when required."""
    try:
        setup_required = needs_setup()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Configuration file missing; cannot render setup UI.") from None

    if not setup_required:
        return RedirectResponse(url="/api/v1/config/quickstart", status_code=307)

    if not SETUP_PAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="Setup UI assets missing. Reinstall the setup UI bundle.")

    return FileResponse(SETUP_PAGE_PATH)


# Register setup UI route conditionally
try:
    if route_enabled("setup"):
        app.add_api_route(
            "/setup", serve_setup_page, methods=["GET"], include_in_schema=False, openapi_extra={"security": []}
        )
    else:
        logger.info("Route disabled by policy: setup (UI)")
except _STARTUP_GUARD_EXCEPTIONS as _setup_rt_err:
    logger.warning(f"Route gating error for setup UI; including by default. Error: {_setup_rt_err}")
    app.add_api_route(
        "/setup", serve_setup_page, methods=["GET"], include_in_schema=False, openapi_extra={"security": []}
    )

# Mount project Docs (read-only) for UI links, if present
DOCS_DIR = BASE_DIR.parent.parent / "Docs"
if DOCS_DIR.exists():
    app.mount("/docs-static", StaticFiles(directory=str(DOCS_DIR), html=False), name="docs-static")
    logger.info(f"Docs mounted at /docs-static from {DOCS_DIR}")
else:
    logger.warning(f"Docs directory not found at {DOCS_DIR}")


# Favicon serving
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")


@app.get("/", openapi_extra={"security": []})
async def root():
    try:
        if needs_setup():
            try:
                if route_enabled("setup"):
                    return RedirectResponse(url="/setup", status_code=307)
            except _REQUEST_GUARD_EXCEPTIONS:
                pass
    except FileNotFoundError:
        logger.warning("config.txt missing while handling root request; serving default message.")

    return {
        "message": "Welcome to the tldw API; if you're seeing this, the server is running! "
        "Check out /api/v1/config/quickstart, /docs, or /metrics to get started."
    }


# Metrics endpoint for Prometheus scraping (registered conditionally below)
async def metrics():
    from tldw_Server_API.app.api.v1.endpoints.metrics import build_prometheus_metrics_response

    return await build_prometheus_metrics_response()


# OpenTelemetry metrics endpoint (if using OTLP) - registered conditionally below
@track_metrics(labels={"endpoint": "metrics"})
async def api_metrics():
    """Get current metrics in JSON format."""
    registry = get_metrics_registry()
    return registry.get_all_metrics()


# Router for health monitoring endpoints (NEW)
if _ULTRA_MINIMAL_APP:
    # Ultra-minimal mode relies exclusively on control-plane health routes
    # (/health, /ready, /health/ready) registered below.
    logger.info("ULTRA_MINIMAL_APP enabled: using control-plane health routes only.")
elif _MINIMAL_TEST_APP:
    # Minimal set for paper_search tests
    include_router_idempotent(app, research_router, prefix=f"{API_V1_PREFIX}/research", tags=["research"])
    include_router_idempotent(app, research_runs_router, prefix=f"{API_V1_PREFIX}", tags=["research-runs"])
    include_router_idempotent(app, paper_search_router, prefix=f"{API_V1_PREFIX}/paper-search", tags=["paper-search"])
    # Include lightweight chat/character routes needed by tests
    include_router_idempotent(app, chat_router, prefix=f"{API_V1_PREFIX}/chat")
    include_router_idempotent(app, chat_loop_router, prefix=f"{API_V1_PREFIX}")
    include_router_idempotent(app, conversations_alias_router, prefix=f"{API_V1_PREFIX}/chats", tags=["chat"])
    include_router_idempotent(app, character_router, prefix=f"{API_V1_PREFIX}/characters", tags=["characters"])
    include_router_idempotent(app, character_memory_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character-memory"])
    include_router_idempotent(
        app, character_chat_sessions_router, prefix=f"{API_V1_PREFIX}/chats", tags=["character-chat-sessions"]
    )
    include_router_idempotent(app, character_messages_router, prefix=f"{API_V1_PREFIX}", tags=["character-messages"])
    include_router_idempotent(app, workspace_migrations_router, prefix=f"{API_V1_PREFIX}/workspaces", tags=["workspaces"])
    include_router_idempotent(app, workspaces_router, prefix=f"{API_V1_PREFIX}/workspaces", tags=["workspaces"])
    # Include audio endpoints (REST + WebSocket) only when enabled by route policy.
    # In pytest + MINIMAL_TEST_APP, default to skipping audio router imports unless
    # explicitly requested. This avoids importing heavy optional transcriber deps
    # that may hard-abort in constrained local test environments.
    _minimal_audio_enabled = route_enabled("audio") or route_enabled("audio-websocket")
    _in_pytest_cmd = _shared_is_explicit_pytest_runtime() or any("pytest" in str(arg or "") for arg in sys.argv)
    if _in_pytest_cmd and not _env_flag_enabled("MINIMAL_TEST_INCLUDE_AUDIO"):
        _minimal_audio_enabled = False
        logger.info("Skipping audio routers in minimal test app (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)")

    if _minimal_audio_enabled:
        try:
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import ws_router as audio_ws_router

            # Mount under /api/v1/audio to match test expectations and non-minimal routing
            app.include_router(audio_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio"])
            app.include_router(audio_ws_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio-ws"])
        except _IMPORT_EXCEPTIONS as _audio_min_err:
            logger.debug(f"Skipping audio routers in minimal test app: {_audio_min_err}")
    else:
        logger.info("Route disabled by policy: audio/audio-websocket (minimal test app)")
    # Health endpoints (required by AuthNZ integration tests)
    try:
        from tldw_Server_API.app.api.v1.endpoints.health import router as health_router

        app.include_router(
            health_router, prefix=f"{API_V1_PREFIX}", tags=["health"]
        )  # /api/v1/health*, /api/v1/healthz, /api/v1/readyz
    except _IMPORT_EXCEPTIONS as _health_min_err:
        logger.debug(f"Skipping health router in minimal test app: {_health_min_err}")
    # Media endpoints (permission enforcement tests call /api/v1/media/add)
    _minimal_media_enabled = route_enabled("media")

    if _minimal_media_enabled:
        try:
            from tldw_Server_API.app.api.v1.endpoints.media import router as media_router

            app.include_router(media_router, prefix=f"{API_V1_PREFIX}/media", tags=["media"])
        except _IMPORT_EXCEPTIONS as _media_min_err:
            logger.debug(f"Skipping media router in minimal test app: {_media_min_err}")
    else:
        logger.info("Route disabled by policy: media (minimal test app)")
    # Email search endpoint (normalized email tables)
    try:
        from tldw_Server_API.app.api.v1.endpoints.email import router as email_router

        app.include_router(email_router, prefix=f"{API_V1_PREFIX}/email", tags=["email"])
    except _IMPORT_EXCEPTIONS as _email_min_err:
        logger.debug(f"Skipping email router in minimal test app: {_email_min_err}")
    # LLM Providers endpoints (used by Chat_NEW unit tests)
    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_providers_router

        app.include_router(llm_providers_router, prefix=f"{API_V1_PREFIX}", tags=["llm"])  # /api/v1/llm/providers
    except _IMPORT_EXCEPTIONS as _llm_min_err:
        logger.debug(f"Skipping llm providers router in minimal test app: {_llm_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.mlx import router as mlx_router

        app.include_router(mlx_router, prefix=f"{API_V1_PREFIX}", tags=["llm"])
    except _IMPORT_EXCEPTIONS as _mlx_min_err:
        logger.debug(f"Skipping mlx router in minimal test app: {_mlx_min_err}")
    # Vector Stores (OpenAI-compatible admin + stores API)
    try:
        from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router as vector_stores_router

        app.include_router(vector_stores_router, prefix=f"{API_V1_PREFIX}", tags=["vector-stores"])
    except _IMPORT_EXCEPTIONS as _vs_min_err:
        logger.debug(f"Skipping vector-stores router in minimal test app: {_vs_min_err}")
    # Embeddings (OpenAI-compatible) endpoints for policy/budget tests and OpenAPI presence
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router

        app.include_router(embeddings_router, prefix=f"{API_V1_PREFIX}", tags=["embeddings"])
    except _IMPORT_EXCEPTIONS as _emb_min_err:
        logger.debug(f"Skipping embeddings router in minimal test app: {_emb_min_err}")
    # Media Embeddings endpoints (/api/v1/media/*/embeddings and jobs listing)
    try:
        from tldw_Server_API.app.api.v1.endpoints.media_embeddings import router as media_embeddings_router

        app.include_router(media_embeddings_router, prefix=f"{API_V1_PREFIX}", tags=["media-embeddings"])
    except _IMPORT_EXCEPTIONS as _me_min_err:
        logger.debug(f"Skipping media_embeddings router in minimal test app: {_me_min_err}")
    # Chunking Templates endpoints (CRUD + apply)
    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router

        app.include_router(chunking_templates_router, prefix=f"{API_V1_PREFIX}", tags=["chunking-templates"])
    except _IMPORT_EXCEPTIONS as _chunk_tpl_min_err:
        logger.debug(f"Skipping chunking templates router in minimal test app: {_chunk_tpl_min_err}")
    # Prompts endpoints (includes collections subpaths)
    try:
        from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router

        app.include_router(prompt_router, prefix=f"{API_V1_PREFIX}/prompts", tags=["prompts"])
    except _IMPORT_EXCEPTIONS as _prompts_min_err:
        logger.debug(f"Skipping prompts router in minimal test app: {_prompts_min_err}")
    # Claims endpoints (status, list, rebuild)
    try:
        from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router

        app.include_router(claims_router, prefix=f"{API_V1_PREFIX}", tags=["claims"])
    except _IMPORT_EXCEPTIONS as _claims_min_err:
        logger.debug(f"Skipping claims router in minimal test app: {_claims_min_err}")
    # RAG unified endpoints (router has its own /api/v1/rag prefix)
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_unified import router as rag_unified_router

        app.include_router(rag_unified_router, tags=["rag-unified"])
    except _IMPORT_EXCEPTIONS as _rag_min_err:
        logger.debug(f"Skipping rag_unified router in minimal test app: {_rag_min_err}")
    # Standalone text2sql endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.text2sql import router as text2sql_router

        app.include_router(text2sql_router, prefix=f"{API_V1_PREFIX}", tags=["text2sql"])
    except _IMPORT_EXCEPTIONS as _text2sql_min_err:
        logger.debug(f"Skipping text2sql router in minimal test app: {_text2sql_min_err}")
    # Explicit feedback endpoints (shared chat/RAG)
    try:
        from tldw_Server_API.app.api.v1.endpoints.feedback import router as feedback_router

        app.include_router(feedback_router, prefix=f"{API_V1_PREFIX}/feedback", tags=["feedback"])
    except _IMPORT_EXCEPTIONS as _feedback_min_err:
        logger.debug(f"Skipping feedback router in minimal test app: {_feedback_min_err}")
    # Vision-language backends listing (lightweight; needed for smoke tests)
    try:
        from tldw_Server_API.app.api.v1.endpoints.vlm import router as vlm_router

        app.include_router(vlm_router, prefix=f"{API_V1_PREFIX}", tags=["vlm"])
    except _IMPORT_EXCEPTIONS as _vlm_min_err:
        logger.debug(f"Skipping vlm router in minimal test app: {_vlm_min_err}")
    # RAG health endpoints (lightweight; required by RAG integration tests)
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router

        app.include_router(rag_health_router, tags=["rag-health"])
    except _IMPORT_EXCEPTIONS as _rag_health_min_err:
        logger.debug(f"Skipping rag_health router in minimal test app: {_rag_health_min_err}")
    # Consent management endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.consent import router as consent_router

        app.include_router(consent_router, prefix=f"{API_V1_PREFIX}", tags=["consent"])
    except _IMPORT_EXCEPTIONS as _consent_min_err:
        logger.debug("Skipping consent router in minimal test app: {}", _consent_min_err)
    # Collections endpoints (treated as lightweight; always included in minimal app)
    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs_templates import router as outputs_templates_router

        app.include_router(outputs_templates_router, prefix=f"{API_V1_PREFIX}", tags=["outputs-templates"])
    except _IMPORT_EXCEPTIONS as _ot_min_err:
        logger.debug(f"Skipping outputs_templates router in minimal test app: {_ot_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

        app.include_router(outputs_router, prefix=f"{API_V1_PREFIX}", tags=["outputs"])
    except _IMPORT_EXCEPTIONS as _outputs_min_err:
        logger.debug(f"Skipping outputs router in minimal test app: {_outputs_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as collections_feeds_router

        app.include_router(collections_feeds_router, prefix=f"{API_V1_PREFIX}", tags=["collections-feeds"])
    except _IMPORT_EXCEPTIONS as _feeds_min_err:
        logger.debug(f"Skipping collections_feeds router in minimal test app: {_feeds_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import (
            callback_router as websub_callback_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import (
            router as collections_websub_router,
        )

        app.include_router(collections_websub_router, prefix=f"{API_V1_PREFIX}", tags=["collections-websub"])
        app.include_router(websub_callback_router, prefix=f"{API_V1_PREFIX}", tags=["collections-websub"])
    except _IMPORT_EXCEPTIONS as _websub_min_err:
        logger.debug(f"Skipping collections_websub router in minimal test app: {_websub_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.slack import router as slack_router

        app.include_router(slack_router, prefix=f"{API_V1_PREFIX}", tags=["slack"])
    except _IMPORT_EXCEPTIONS as _slack_min_err:
        logger.debug(f"Skipping slack router in minimal test app: {_slack_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.discord import router as discord_router

        app.include_router(discord_router, prefix=f"{API_V1_PREFIX}", tags=["discord"])
    except _IMPORT_EXCEPTIONS as _discord_min_err:
        logger.debug(f"Skipping discord router in minimal test app: {_discord_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.telegram import router as telegram_router

        app.include_router(telegram_router, prefix=f"{API_V1_PREFIX}", tags=["telegram"])
    except _IMPORT_EXCEPTIONS as _telegram_min_err:
        logger.debug(f"Skipping telegram router in minimal test app: {_telegram_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.files import router as files_router

        app.include_router(files_router, prefix=f"{API_V1_PREFIX}", tags=["files"])
    except ImportError as _files_min_err:
        logger.debug(f"Skipping files router in minimal test app: {_files_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.storage import router as storage_router

        app.include_router(storage_router, prefix=f"{API_V1_PREFIX}", tags=["storage"])
    except ImportError as _storage_min_err:
        logger.debug(f"Skipping storage router in minimal test app: {_storage_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router

        app.include_router(data_tables_router, prefix=f"{API_V1_PREFIX}", tags=["data-tables"])
    except ImportError as _dt_min_err:
        logger.debug(f"Skipping data_tables router in minimal test app: {_dt_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.reading_highlights import router as reading_highlights_router

        app.include_router(reading_highlights_router, prefix=f"{API_V1_PREFIX}", tags=["reading-highlights"])
    except _IMPORT_EXCEPTIONS as _rh_min_err:
        logger.debug(f"Skipping reading_highlights router in minimal test app: {_rh_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.items import router as items_router

        app.include_router(items_router, prefix=f"{API_V1_PREFIX}", tags=["items"])
    except _IMPORT_EXCEPTIONS as _items_min_err:
        logger.debug(f"Skipping items router in minimal test app: {_items_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.reminders import router as reminders_router

        app.include_router(reminders_router, prefix=f"{API_V1_PREFIX}", tags=["tasks"])
    except _IMPORT_EXCEPTIONS as _reminders_min_err:
        logger.debug(f"Skipping reminders router in minimal test app: {_reminders_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.integrations_control_plane import (
            router as integrations_control_plane_router,
        )

        app.include_router(integrations_control_plane_router, prefix=f"{API_V1_PREFIX}", tags=["integrations"])
    except _IMPORT_EXCEPTIONS as _integrations_cp_min_err:
        logger.debug(f"Skipping integrations control plane router in minimal test app: {_integrations_cp_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane import (
            router as scheduled_tasks_control_plane_router,
        )

        app.include_router(scheduled_tasks_control_plane_router, prefix=f"{API_V1_PREFIX}", tags=["scheduled-tasks"])
    except _IMPORT_EXCEPTIONS as _scheduled_tasks_cp_min_err:
        logger.debug(f"Skipping scheduled tasks control plane router in minimal test app: {_scheduled_tasks_cp_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.notifications import router as notifications_router

        app.include_router(notifications_router, prefix=f"{API_V1_PREFIX}", tags=["notifications"])
    except _IMPORT_EXCEPTIONS as _notifications_min_err:
        logger.debug("Skipping notifications router in minimal test app: {}", _notifications_min_err)
    # Chatbooks endpoints (export/import, jobs, download)
    try:
        from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router

        app.include_router(chatbooks_router, prefix=f"{API_V1_PREFIX}", tags=["chatbooks"])
    except _IMPORT_EXCEPTIONS as _chatbooks_min_err:
        logger.debug(f"Skipping chatbooks router in minimal test app: {_chatbooks_min_err}")
    # Sharing endpoints (workspace sharing, tokens, admin)
    try:
        from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router

        app.include_router(sharing_router, prefix=f"{API_V1_PREFIX}", tags=["sharing"])
    except _IMPORT_EXCEPTIONS as _sharing_min_err:
        logger.debug("Skipping sharing router in minimal test app: {}", _sharing_min_err)
    # Personalization scaffold endpoints (opt-in/profile/memories) needed for unit tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.personalization import router as personalization_router

        app.include_router(personalization_router, prefix=f"{API_V1_PREFIX}/personalization", tags=["personalization"])
    except _IMPORT_EXCEPTIONS as _pers_min_err:
        logger.debug(f"Skipping personalization router in minimal test app: {_pers_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.companion import router as companion_router

        app.include_router(companion_router, prefix=f"{API_V1_PREFIX}/companion", tags=["companion"])
    except _IMPORT_EXCEPTIONS as _companion_min_err:
        logger.debug(f"Skipping companion router in minimal test app: {_companion_min_err}")
    # Guardian controls (parental/supervised account controls)
    try:
        from tldw_Server_API.app.api.v1.endpoints.guardian_controls import router as guardian_controls_router
        from tldw_Server_API.app.api.v1.endpoints.family_wizard import router as family_wizard_router

        app.include_router(guardian_controls_router, prefix=f"{API_V1_PREFIX}/guardian", tags=["guardian"])
        app.include_router(family_wizard_router, prefix=f"{API_V1_PREFIX}/guardian", tags=["guardian"])
    except _IMPORT_EXCEPTIONS as _guard_min_err:
        logger.debug(f"Skipping guardian controls router in minimal test app: {_guard_min_err}")
    # Self-monitoring (awareness notifications, crisis resources)
    try:
        from tldw_Server_API.app.api.v1.endpoints.self_monitoring import router as self_monitoring_router

        app.include_router(self_monitoring_router, prefix=f"{API_V1_PREFIX}/self-monitoring", tags=["self-monitoring"])
    except _IMPORT_EXCEPTIONS as _selfmon_min_err:
        logger.debug(f"Skipping self-monitoring router in minimal test app: {_selfmon_min_err}")
    # Persona scaffold endpoints (catalog/session/WS) used by unit tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.persona import router as persona_router

        app.include_router(persona_router, prefix=f"{API_V1_PREFIX}/persona", tags=["persona"])
    except _IMPORT_EXCEPTIONS as _persona_min_err:
        logger.debug(f"Skipping persona router in minimal test app: {_persona_min_err}")
    # Archetype template endpoints (list / detail / preview)
    try:
        from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import router as archetype_router

        app.include_router(archetype_router, prefix=f"{API_V1_PREFIX}/persona/archetypes", tags=["persona-archetypes"])
    except _IMPORT_EXCEPTIONS as _archetype_min_err:
        logger.debug("Skipping archetype router in minimal test app: {}", _archetype_min_err)
    # Notes endpoints (health + CRUD)
    try:
        from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router

        app.include_router(notes_router, prefix=f"{API_V1_PREFIX}/notes", tags=["notes"])
    except _IMPORT_EXCEPTIONS as _notes_min_err:
        logger.debug(f"Skipping notes router in minimal test app: {_notes_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.web_clipper import router as web_clipper_router

        app.include_router(web_clipper_router, prefix=f"{API_V1_PREFIX}/web-clipper", tags=["web-clipper"])
    except _IMPORT_EXCEPTIONS as _web_clipper_min_err:
        logger.debug(f"Skipping web clipper router in minimal test app: {_web_clipper_min_err}")
    # Skills endpoints (SKILL.md management)
    try:
        from tldw_Server_API.app.api.v1.endpoints.skills import router as skills_router

        app.include_router(skills_router, prefix=f"{API_V1_PREFIX}/skills", tags=["skills"])
    except _IMPORT_EXCEPTIONS as _skills_min_err:
        logger.debug(f"Skipping skills router in minimal test app: {_skills_min_err}")
    # Translation endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.translate import router as translate_router

        app.include_router(translate_router, prefix=f"{API_V1_PREFIX}", tags=["translation"])
    except _IMPORT_EXCEPTIONS as _translate_min_err:
        logger.debug(f"Skipping translate router in minimal test app: {_translate_min_err}")
    # Slides endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router

        app.include_router(slides_router, prefix=f"{API_V1_PREFIX}", tags=["slides"])
    except _IMPORT_EXCEPTIONS as _slides_min_err:
        logger.debug(f"Skipping slides router in minimal test app: {_slides_min_err}")
    # Kanban Board endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_boards import router as kanban_boards_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_cards import router as kanban_cards_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_checklists import router as kanban_checklists_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_comments import router as kanban_comments_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_labels import router as kanban_labels_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_links import router as kanban_links_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_lists import router as kanban_lists_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_search import router as kanban_search_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_workflow import router as kanban_workflow_router

        app.include_router(kanban_boards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_lists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_cards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_labels_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_checklists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_comments_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_search_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_links_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        app.include_router(kanban_workflow_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
    except _IMPORT_EXCEPTIONS as _kanban_min_err:
        logger.debug(f"Skipping kanban router in minimal test app: {_kanban_min_err}")
    # Auth endpoints (login/register/refresh/logout/me)
    try:
        app.include_router(auth_router, prefix=f"{API_V1_PREFIX}", tags=["authentication"])
        logger.info("Auth router consolidated: endpoints/auth.py (minimal test app)")
    except _IMPORT_EXCEPTIONS as _auth_min_err:
        logger.debug(f"Skipping auth router in minimal test app: {_auth_min_err}")
    # Users endpoints (sessions, change-password, storage, me)
    try:
        from tldw_Server_API.app.api.v1.endpoints.users import router as users_router

        app.include_router(users_router, prefix=f"{API_V1_PREFIX}", tags=["users"])
    except _IMPORT_EXCEPTIONS as _users_min_err:
        logger.debug(f"Skipping users router in minimal test app: {_users_min_err}")

    # Include BYOK and shared-key routes independently so optional users.py deps
    # do not suppress keys endpoints in minimal test mode.
    try:
        from tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped import router as shared_keys_scoped_router
        from tldw_Server_API.app.api.v1.endpoints.user_keys import router as user_keys_router

        app.include_router(user_keys_router, prefix=f"{API_V1_PREFIX}", tags=["users"])
        app.include_router(shared_keys_scoped_router, prefix=f"{API_V1_PREFIX}", tags=["organizations"])
    except _IMPORT_EXCEPTIONS as _keys_min_err:
        logger.debug(f"Skipping BYOK/shared keys routers in minimal test app: {_keys_min_err}")
    # Include Jobs admin endpoints for tests that exercise jobs stats/counters
    try:
        from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router

        app.include_router(jobs_admin_router, prefix=f"{API_V1_PREFIX}", tags=["jobs"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.debug(f"Skipping jobs_admin router in minimal test app: {_e}")
    # Include Audio Jobs (admin + listing) for tests under minimal mode when enabled.
    _minimal_audio_jobs_enabled = route_enabled("audio-jobs")
    if _in_pytest_cmd and not _env_flag_enabled("MINIMAL_TEST_INCLUDE_AUDIO_JOBS"):
        _minimal_audio_jobs_enabled = False
        logger.info(
            "Skipping audio-jobs router in minimal test app (set MINIMAL_TEST_INCLUDE_AUDIO_JOBS=1 to enable)"
        )

    if _minimal_audio_jobs_enabled:
        try:
            from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import router as audio_jobs_router

            app.include_router(audio_jobs_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio-jobs"])
        except _IMPORT_EXCEPTIONS as _audio_jobs_min_err:
            logger.debug(f"Skipping audio_jobs router in minimal test app: {_audio_jobs_min_err}")
    else:
        logger.info("Route disabled by policy: audio-jobs (minimal test app)")
    # Include Audit endpoints in minimal test app so tests relying on /api/v1/audit/* don't 404
    try:
        from tldw_Server_API.app.api.v1.endpoints.audit import router as audit_router

        app.include_router(audit_router, prefix=f"{API_V1_PREFIX}", tags=["audit"])
    except _IMPORT_EXCEPTIONS as _audit_min_err:
        logger.debug(f"Skipping audit router in minimal test app: {_audit_min_err}")
    # Config info endpoints (includes /api/v1/config/jobs used by OpenAPI tests)
    try:
        app.include_router(setup_router, prefix=f"{API_V1_PREFIX}", tags=["setup"])
    except _IMPORT_EXCEPTIONS as _setup_min_err:
        logger.debug("Skipping setup router in minimal test app: {}", _setup_min_err)
    try:
        from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router

        app.include_router(config_info_router, prefix=f"{API_V1_PREFIX}", tags=["config"])
    except _IMPORT_EXCEPTIONS as _config_min_err:
        logger.debug(f"Skipping config_info router in minimal test app: {_config_min_err}")
    # Admin config diagnostics endpoint (effective config)
    try:
        from tldw_Server_API.app.api.v1.endpoints.config_admin import router as config_admin_router

        app.include_router(config_admin_router, prefix=f"{API_V1_PREFIX}", tags=["config", "admin"])
    except _IMPORT_EXCEPTIONS as _config_admin_min_err:
        logger.debug(f"Skipping config_admin router in minimal test app: {_config_admin_min_err}")
    # Flashcards endpoints (ChaChaNotes-backed) for integration tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

        app.include_router(flashcards_router, prefix=f"{API_V1_PREFIX}", tags=["flashcards"])
    except _IMPORT_EXCEPTIONS as _flash_min_err:
        logger.debug(f"Skipping flashcards router in minimal test app: {_flash_min_err}")
    # Quizzes endpoints (ChaChaNotes-backed) for integration tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.quizzes import router as quizzes_router

        app.include_router(quizzes_router, prefix=f"{API_V1_PREFIX}", tags=["quizzes"])
    except _IMPORT_EXCEPTIONS as _quiz_min_err:
        logger.debug(f"Skipping quizzes router in minimal test app: {_quiz_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.study_suggestions import (
            router as study_suggestions_router,
        )

        app.include_router(study_suggestions_router, prefix=f"{API_V1_PREFIX}", tags=["study-suggestions"])
    except _IMPORT_EXCEPTIONS as _study_suggestions_min_err:
        logger.debug(f"Skipping study_suggestions router in minimal test app: {_study_suggestions_min_err}")
    # Writing Playground endpoints (ChaChaNotes-backed) for integration tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.writing import router as writing_router

        app.include_router(writing_router, prefix=f"{API_V1_PREFIX}/writing", tags=["writing"])
    except _IMPORT_EXCEPTIONS as _writing_min_err:
        logger.debug(f"Skipping writing router in minimal test app: {_writing_min_err}")
    # Manuscript Management endpoints (ChaChaNotes-backed) for integration tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import router as manuscripts_router

        app.include_router(manuscripts_router, prefix=f"{API_V1_PREFIX}/writing/manuscripts", tags=["manuscripts"])
    except _IMPORT_EXCEPTIONS as _manuscripts_min_err:
        logger.debug(f"Skipping manuscripts router in minimal test app: {_manuscripts_min_err}")
    # Metrics endpoints (/api/v1/metrics/text)
    try:
        from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router

        app.include_router(metrics_router, prefix=f"{API_V1_PREFIX}", tags=["metrics"])
    except _IMPORT_EXCEPTIONS as _metrics_min_err:
        logger.debug(f"Skipping metrics router in minimal test app: {_metrics_min_err}")
    # AuthNZ debug routes for tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.authnz_debug import router as authnz_debug_router

        app.include_router(authnz_debug_router, prefix=f"{API_V1_PREFIX}", tags=["authnz-debug"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.debug(f"Skipping authnz_debug router in tests: {_e}")
    # Sandbox (scaffold) - include in minimal test app to support sandbox tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

        app.include_router(sandbox_router, prefix=f"{API_V1_PREFIX}", tags=["sandbox"])
    except _IMPORT_EXCEPTIONS as _sandbox_err:
        # Never let optional sandbox break startup in tests
        logger.debug(f"Skipping sandbox router in minimal test app: {_sandbox_err}")
    # Include MCP Unified WS/HTTP endpoints for tests (auth typically disabled via env/fixtures)
    try:
        # mcp_unified_router may already be imported above; if not, import here guarded
        if "mcp_unified_router" not in locals():
            from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router
        app.include_router(mcp_unified_router, prefix=f"{API_V1_PREFIX}", tags=["mcp-unified"])
        # MCP tool catalogs admin (lightweight) for unit tests
        try:
            from tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage import router as mcp_catalogs_manage_router

            app.include_router(mcp_catalogs_manage_router, prefix=f"{API_V1_PREFIX}", tags=["mcp-catalogs"])
        except _IMPORT_EXCEPTIONS as _mcp_cat_err:
            logger.debug(f"Skipping MCP catalogs router in minimal test app: {_mcp_cat_err}")
        try:
            from tldw_Server_API.app.api.v1.endpoints.mcp_hub_management import router as mcp_hub_management_router

            app.include_router(mcp_hub_management_router, prefix=f"{API_V1_PREFIX}", tags=["mcp-hub"])
        except _IMPORT_EXCEPTIONS as _mcp_hub_err:
            logger.debug(f"Skipping MCP hub router in minimal test app: {_mcp_hub_err}")
        # Privileges endpoints used by tests that introspect RBAC snapshots
        try:
            from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router

            app.include_router(privileges_router, prefix=f"{API_V1_PREFIX}", tags=["privileges"])
        except _IMPORT_EXCEPTIONS as _priv_min_err:
            logger.debug(f"Skipping privileges router in minimal test app: {_priv_min_err}")
    except _IMPORT_EXCEPTIONS as _mcp_min_err:
        logger.debug(f"Skipping MCP unified router in minimal test app: {_mcp_min_err}")
    # Tools endpoints (MCP-backed) needed for permission enforcement tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router

        app.include_router(tools_router, prefix=f"{API_V1_PREFIX}", tags=["tools"])
    except _IMPORT_EXCEPTIONS as _tools_min_err:
        logger.debug(f"Skipping tools router in minimal test app: {_tools_min_err}")
    # ACP runner endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import router as acp_router

        app.include_router(acp_router, prefix=f"{API_V1_PREFIX}", tags=["acp"])
    except _IMPORT_EXCEPTIONS as _acp_min_err:
        logger.debug(f"Skipping ACP router in minimal test app: {_acp_min_err}")
    # ACP sub-module routers (schedules, triggers, permissions)
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_schedules import router as acp_schedules_router

        app.include_router(acp_schedules_router, prefix=f"{API_V1_PREFIX}", tags=["acp-schedules"])
    except _IMPORT_EXCEPTIONS as _acp_sched_min_err:
        logger.debug(f"Skipping ACP schedules router in minimal test app: {_acp_sched_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_triggers import router as acp_triggers_router

        app.include_router(acp_triggers_router, prefix=f"{API_V1_PREFIX}", tags=["acp-triggers"])
    except _IMPORT_EXCEPTIONS as _acp_trig_min_err:
        logger.debug(f"Skipping ACP triggers router in minimal test app: {_acp_trig_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_permissions import router as acp_permissions_router

        app.include_router(acp_permissions_router, prefix=f"{API_V1_PREFIX}", tags=["acp-permissions"])
    except _IMPORT_EXCEPTIONS as _acp_perm_min_err:
        logger.debug(f"Skipping ACP permissions router in minimal test app: {_acp_perm_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_multiplex import router as acp_multiplex_router

        app.include_router(acp_multiplex_router, prefix=f"{API_V1_PREFIX}", tags=["acp-multiplex"])
    except _IMPORT_EXCEPTIONS as _acp_mpx_min_err:
        logger.debug(f"Skipping ACP multiplex router in minimal test app: {_acp_mpx_min_err}")
    # Agent Orchestration endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import router as orch_router

        app.include_router(orch_router, prefix=f"{API_V1_PREFIX}", tags=["agent-orchestration"])
    except _IMPORT_EXCEPTIONS as _orch_min_err:
        logger.debug(f"Skipping orchestration router in minimal test app: {_orch_min_err}")
    # Include admin router in minimal mode if available (ensure not gated by MCP import)
    try:
        if "admin_router" not in locals():
            from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router
        app.include_router(admin_router, prefix=f"{API_V1_PREFIX}", tags=["admin"])
    except _IMPORT_EXCEPTIONS as _adm_inc_err:
        logger.debug(f"Skipping admin router include in minimal test app: {_adm_inc_err}")
        # Keep BYOK admin controls available even when broader admin router
        # dependencies are unavailable (e.g., optional MFA deps in tests).
        try:
            from tldw_Server_API.app.api.v1.endpoints.admin.admin_byok import (
                router as admin_byok_router,
            )

            app.include_router(admin_byok_router, prefix=f"{API_V1_PREFIX}/admin", tags=["admin"])
        except _IMPORT_EXCEPTIONS as _adm_byok_min_err:
            logger.debug(f"Skipping admin BYOK router in minimal test app: {_adm_byok_min_err}")
    # Organization endpoints used by AuthNZ integration tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.orgs import router as orgs_router

        app.include_router(orgs_router, prefix=f"{API_V1_PREFIX}", tags=["organizations"])
    except _IMPORT_EXCEPTIONS as _orgs_min_err:
        logger.debug(f"Skipping orgs router in minimal test app: {_orgs_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.org_invites import router as org_invites_router

        app.include_router(org_invites_router, prefix=f"{API_V1_PREFIX}", tags=["invites"])
    except _IMPORT_EXCEPTIONS as _org_inv_min_err:
        logger.debug(f"Skipping org_invites router in minimal test app: {_org_inv_min_err}")
    # Resource Governor admin/diag endpoints are required for RG tests in minimal app
    try:
        from tldw_Server_API.app.api.v1.endpoints.resource_governor import router as resource_governor_router

        app.include_router(resource_governor_router, prefix=f"{API_V1_PREFIX}", tags=["resource-governor"])
    except _IMPORT_EXCEPTIONS as _rg_min_err:
        logger.debug(f"Skipping resource_governor router in minimal test app: {_rg_min_err}")
    # LlamaCpp endpoints for reranking tests
    try:
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import (
            public_router as llamacpp_public_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import (
            router as llamacpp_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.messages import (
            public_router as messages_public_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.messages import (
            router as messages_router,
        )

        app.include_router(llamacpp_router, prefix=f"{API_V1_PREFIX}", tags=["llamacpp"])
        app.include_router(llamacpp_public_router, prefix="", tags=["llamacpp"])
        app.include_router(messages_router, prefix=f"{API_V1_PREFIX}", tags=["messages"])
        app.include_router(messages_public_router, prefix="", tags=["messages"])
    except _IMPORT_EXCEPTIONS as _llama_min_err:
        logger.debug(f"Skipping llamacpp router in minimal test app: {_llama_min_err}")
    # Workflows + scheduler routers are lightweight enough to enable in minimal
    # test mode so unit tests do not see 404s.
    try:
        from tldw_Server_API.app.api.v1.endpoints.workflows import router as _wf_router

        app.include_router(_wf_router, prefix="", tags=["workflows"])
    except _IMPORT_EXCEPTIONS as _wf_min_err:
        logger.debug(f"Skipping workflows router in minimal test app: {_wf_min_err}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.chat_workflows import (
            router as _chat_wf_router,
        )

        app.include_router(_chat_wf_router, prefix="", tags=["chat-workflows"])
    except _IMPORT_EXCEPTIONS as _chat_wf_min_err:
        logger.debug(
            f"Skipping chat workflows router in minimal test app: {_chat_wf_min_err}"
        )
    try:
        from tldw_Server_API.app.api.v1.endpoints.scheduler_workflows import router as _sch_wf_router

        app.include_router(_sch_wf_router, prefix="", tags=["scheduler"])
    except _IMPORT_EXCEPTIONS as _sch_min_err:
        logger.debug(f"Skipping scheduler workflows router in minimal test app: {_sch_min_err}")
    # Evaluations endpoints in minimal mode: policy-gated by route toggles.
    try:
        if route_enabled("evaluations"):
            from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import (
                router as _evaluations_router,
            )

            app.include_router(_evaluations_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])
        else:
            logger.info("Route disabled by policy: evaluations (minimal test app)")
    except _STARTUP_GUARD_EXCEPTIONS as _evals_min_err:
        logger.debug(f"Skipping evaluations routers in minimal test app: {_evals_min_err}")
    try:
        if route_enabled("monitoring"):
            from tldw_Server_API.app.api.v1.endpoints.monitoring import router as _monitoring_router

            app.include_router(_monitoring_router, prefix=f"{API_V1_PREFIX}", tags=["monitoring"])
        else:
            logger.info("Route disabled by policy: monitoring (minimal test app)")
    except _STARTUP_GUARD_EXCEPTIONS as _monitoring_min_err:
        logger.debug(f"Skipping monitoring router in minimal test app: {_monitoring_min_err}")
else:
    # Small helper to guard route inclusion via config.txt and ENV
    def _include_if_enabled(
        route_key: str, router, *, prefix: str = "", tags: list | None = None, default_stable: bool = True
    ) -> None:
        try:
            # In explicit pytest runtime, force-include certain routes even if
            # config gating would normally disable them (e.g., workflows/scheduler).
            _test_ctx = bool(_TEST_MODE)
            if _test_ctx and route_key in {"workflows", "scheduler"}:
                include_router_idempotent(app, router, prefix=prefix, tags=tags)
                return
            if route_enabled(route_key, default_stable=default_stable):
                include_router_idempotent(app, router, prefix=prefix, tags=tags)
            else:
                logger.info(f"Route disabled by policy: {route_key}")
        except _STARTUP_GUARD_EXCEPTIONS as _rt_err:
            logger.warning(f"Route gating error for {route_key}; including by default. Error: {_rt_err}")
            include_router_idempotent(app, router, prefix=prefix, tags=tags)

    try:
        from tldw_Server_API.app.api.v1.endpoints.health import router as health_router

        _HAS_HEALTH = True
    except _IMPORT_EXCEPTIONS as _health_import_err:
        logger.warning(f"Health endpoints unavailable; skipping import: {_health_import_err}")
        _HAS_HEALTH = False
    from tldw_Server_API.app.api.v1.endpoints.moderation import router as moderation_router
    from tldw_Server_API.app.api.v1.endpoints.monitoring import router as monitoring_router

    if _HAS_HEALTH:
        _include_if_enabled(
            "health", health_router, prefix=f"{API_V1_PREFIX}", tags=["health"]
        )  # /api/v1/healthz, /api/v1/readyz
    _include_if_enabled("moderation", moderation_router, prefix=f"{API_V1_PREFIX}", tags=["moderation"])
    _include_if_enabled("monitoring", monitoring_router, prefix=f"{API_V1_PREFIX}", tags=["monitoring"])
    from tldw_Server_API.app.api.v1.endpoints.audit import router as audit_router

    _include_if_enabled("audit", audit_router, prefix=f"{API_V1_PREFIX}", tags=["audit"])
    _include_if_enabled("auth", auth_router, prefix=f"{API_V1_PREFIX}", tags=["authentication"])
    _include_if_enabled("consent", consent_router, prefix=f"{API_V1_PREFIX}", tags=["consent"])
    logger.info("Auth router consolidated: endpoints/auth.py")
    if "users_router" in locals() and users_router is not None:
        _include_if_enabled("users", users_router, prefix=f"{API_V1_PREFIX}", tags=["users"])
    _include_if_enabled("users", user_keys_router, prefix=f"{API_V1_PREFIX}", tags=["users"])

    # Include AuthNZ debug endpoints once via the gated path.
    # Force-enable when _TEST_MODE is true; otherwise respect route policy.
    try:
        from tldw_Server_API.app.api.v1.endpoints.authnz_debug import router as authnz_debug_router

        _include_if_enabled(
            "authnz-debug",
            authnz_debug_router,
            prefix=f"{API_V1_PREFIX}",
            tags=["authnz-debug"],
            default_stable=bool(_TEST_MODE),
        )
    except _IMPORT_EXCEPTIONS as _e:
        logger.debug(f"Skipping authnz_debug router: {_e}")
    _include_if_enabled("privileges", privileges_router, prefix=f"{API_V1_PREFIX}", tags=["privileges"])
    try:
        from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router
    except _IMPORT_EXCEPTIONS as _admin_import_err:
        logger.warning(f"Admin endpoints unavailable at import time; deferring: {_admin_import_err}")
        admin_router = None  # type: ignore[assignment]
    from tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage import router as mcp_catalogs_manage_router
    from tldw_Server_API.app.api.v1.endpoints.mcp_hub_management import router as mcp_hub_management_router

    if admin_router is not None:
        _include_if_enabled("admin", admin_router, prefix=f"{API_V1_PREFIX}", tags=["admin"])
    # Billing / subscription management endpoints (admin-only)
    try:
        from tldw_Server_API.app.api.v1.endpoints.billing import router as billing_router

        _include_if_enabled("billing", billing_router, prefix=f"{API_V1_PREFIX}", tags=["billing"])
    except _IMPORT_EXCEPTIONS as _billing_import_err:
        logger.warning(f"Billing endpoints unavailable; skipping: {_billing_import_err}")
    _include_if_enabled("mcp-catalogs", mcp_catalogs_manage_router, prefix=f"{API_V1_PREFIX}")
    _include_if_enabled("mcp-hub", mcp_hub_management_router, prefix=f"{API_V1_PREFIX}", tags=["mcp-hub"])
    # Self-service organization management endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.orgs import router as orgs_router

        _include_if_enabled("orgs", orgs_router, prefix=f"{API_V1_PREFIX}", tags=["organizations"])
        _include_if_enabled("orgs", shared_keys_scoped_router, prefix=f"{API_V1_PREFIX}", tags=["organizations"])
    except ImportError as _orgs_err:
        logger.warning(f"Skipping orgs router due to import error: {_orgs_err}")
    # Organization invite preview and redemption endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.org_invites import router as org_invites_router

        _include_if_enabled("org-invites", org_invites_router, prefix=f"{API_V1_PREFIX}", tags=["invites"])
    except ImportError as _inv_err:
        logger.warning(f"Skipping org_invites router due to import error: {_inv_err}")
    if _HAS_MEDIA:
        _include_if_enabled("media", media_router, prefix=f"{API_V1_PREFIX}/media", tags=["media"])
    try:
        from tldw_Server_API.app.api.v1.endpoints.email import router as email_router

        _include_if_enabled("email", email_router, prefix=f"{API_V1_PREFIX}/email", tags=["email"])
    except _IMPORT_EXCEPTIONS as _email_route_err:
        logger.debug(f"Email endpoints unavailable; skipping import: {_email_route_err}")
    if _HAS_AUDIO:
        _include_if_enabled("audio", audio_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio"])
    if _HAS_AUDIO_JOBS:
        _include_if_enabled("audio-jobs", audio_jobs_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio-jobs"])
    if _HAS_AUDIO:
        _include_if_enabled(
            "audio-websocket", audio_ws_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio-websocket"]
        )
    # Voice Assistant endpoints (REST + WebSocket)
    try:
        from tldw_Server_API.app.api.v1.endpoints.voice_assistant import (
            router as voice_assistant_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.voice_assistant import (
            ws_router as voice_assistant_ws_router,
        )

        _include_if_enabled(
            "voice-assistant", voice_assistant_router, prefix=f"{API_V1_PREFIX}/voice", tags=["voice-assistant"]
        )
        _include_if_enabled(
            "voice-assistant-ws",
            voice_assistant_ws_router,
            prefix=f"{API_V1_PREFIX}/voice",
            tags=["voice-assistant-ws"],
        )
    except ImportError as _voice_err:
        logger.debug(f"Voice assistant endpoints not available: {_voice_err}")
    # Guard optional routers that may not be imported in ULTRA_MINIMAL_APP
    if "chat_router" in locals():
        _include_if_enabled("chat", chat_router, prefix=f"{API_V1_PREFIX}/chat")
    if "chat_loop_router" in locals():
        _include_if_enabled("chat", chat_loop_router, prefix=f"{API_V1_PREFIX}")
    if "conversations_alias_router" in locals():
        _include_if_enabled("chat", conversations_alias_router, prefix=f"{API_V1_PREFIX}/chats", tags=["chat"])
    # Tools (MCP-backed server tool execution) - include if initial guarded import succeeded
    if "tools_router" in locals() and tools_router is not None:
        _include_if_enabled("tools", tools_router, prefix=f"{API_V1_PREFIX}", tags=["tools"], default_stable=False)
    if "acp_router" in locals() and acp_router is not None:
        _include_if_enabled("acp", acp_router, prefix=f"{API_V1_PREFIX}", tags=["acp"], default_stable=False)
    if "acp_schedules_router" in locals() and acp_schedules_router is not None:
        _include_if_enabled("acp", acp_schedules_router, prefix=f"{API_V1_PREFIX}", tags=["acp-schedules"], default_stable=False)
    if "acp_triggers_router" in locals() and acp_triggers_router is not None:
        _include_if_enabled("acp", acp_triggers_router, prefix=f"{API_V1_PREFIX}", tags=["acp-triggers"], default_stable=False)
    if "acp_permissions_router" in locals() and acp_permissions_router is not None:
        _include_if_enabled("acp", acp_permissions_router, prefix=f"{API_V1_PREFIX}", tags=["acp-permissions"], default_stable=False)
    if "acp_multiplex_router" in locals() and acp_multiplex_router is not None:
        _include_if_enabled("acp", acp_multiplex_router, prefix=f"{API_V1_PREFIX}", tags=["acp-multiplex"], default_stable=False)
    if "character_router" in locals():
        _include_if_enabled("characters", character_router, prefix=f"{API_V1_PREFIX}/characters", tags=["characters"])
    if "character_memory_router" in locals():
        _include_if_enabled(
            "character-memory", character_memory_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character-memory"]
        )
    if "workspaces_router" in locals():
        if "workspace_migrations_router" in locals():
            _include_if_enabled(
                "workspaces",
                workspace_migrations_router,
                prefix=f"{API_V1_PREFIX}/workspaces",
                tags=["workspaces"],
            )
        _include_if_enabled(
            "workspaces", workspaces_router, prefix=f"{API_V1_PREFIX}/workspaces", tags=["workspaces"]
        )
    if "character_chat_sessions_router" in locals():
        _include_if_enabled(
            "character-chat-sessions",
            character_chat_sessions_router,
            prefix=f"{API_V1_PREFIX}/chats",
            tags=["character-chat-sessions"],
        )
    if "character_messages_router" in locals():
        _include_if_enabled(
            "character-messages", character_messages_router, prefix=f"{API_V1_PREFIX}", tags=["character-messages"]
        )
    if "metrics_router" in locals():
        _include_if_enabled("metrics", metrics_router, prefix=f"{API_V1_PREFIX}", tags=["metrics"])
    if _HAS_CHUNKING and "chunking_router" in locals():
        _include_if_enabled("chunking", chunking_router, prefix=f"{API_V1_PREFIX}/chunking", tags=["chunking"])
    if "chunking_templates_router" in locals():
        _include_if_enabled(
            "chunking-templates", chunking_templates_router, prefix=f"{API_V1_PREFIX}", tags=["chunking-templates"]
        )
    if _HAS_OUTPUT_TEMPLATES and "outputs_templates_router" in locals():
        _include_if_enabled(
            "outputs-templates", outputs_templates_router, prefix=f"{API_V1_PREFIX}", tags=["outputs-templates"]
        )
    if _HAS_COLLECTIONS_FEEDS and "collections_feeds_router" in locals():
        _include_if_enabled(
            "collections-feeds", collections_feeds_router, prefix=f"{API_V1_PREFIX}", tags=["collections-feeds"]
        )
    if _HAS_COLLECTIONS_WEBSUB and "collections_websub_router" in locals():
        _include_if_enabled(
            "collections-websub", collections_websub_router, prefix=f"{API_V1_PREFIX}", tags=["collections-websub"]
        )
    if _HAS_COLLECTIONS_WEBSUB and "websub_callback_router" in locals():
        _include_if_enabled(
            "collections-websub", websub_callback_router, prefix=f"{API_V1_PREFIX}", tags=["collections-websub"]
        )
    if _HAS_SLACK and "slack_router" in locals():
        _include_if_enabled("slack", slack_router, prefix=f"{API_V1_PREFIX}", tags=["slack"], default_stable=False)
    if _HAS_DISCORD and "discord_router" in locals():
        _include_if_enabled("discord", discord_router, prefix=f"{API_V1_PREFIX}", tags=["discord"], default_stable=False)
    if _HAS_TELEGRAM and "telegram_router" in locals():
        _include_if_enabled(
            "telegram", telegram_router, prefix=f"{API_V1_PREFIX}", tags=["telegram"], default_stable=False
        )
    try:
        # Optional outputs artifacts endpoint
        from tldw_Server_API.app.api.v1.endpoints.outputs import router as _outputs_router

        _include_if_enabled("outputs", _outputs_router, prefix=f"{API_V1_PREFIX}", tags=["outputs"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Outputs endpoint not available: {_e}")
    if _HAS_MEETINGS and "meetings_router" in locals():
        _include_if_enabled(
            "meetings",
            meetings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=["meetings"],
            default_stable=False,
        )
    try:
        # Optional audiobook creation endpoint
        from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router

        _include_if_enabled(
            "audiobooks",
            audiobooks_router,
            prefix=f"{API_V1_PREFIX}",
            tags=["audiobooks"],
            default_stable=False,
        )
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Audiobooks endpoint not available: {_e}")
    try:
        # Optional files artifacts endpoint
        from tldw_Server_API.app.api.v1.endpoints.files import router as _files_router

        _include_if_enabled("files", _files_router, prefix=f"{API_V1_PREFIX}", tags=["files"])
    except ImportError as _e:
        logger.warning(f"Files endpoint not available: {_e}")
    try:
        # Optional data tables endpoint
        from tldw_Server_API.app.api.v1.endpoints.data_tables import router as _data_tables_router

        _include_if_enabled("data-tables", _data_tables_router, prefix=f"{API_V1_PREFIX}", tags=["data-tables"])
    except ImportError as _e:
        logger.warning(f"Data tables endpoint not available: {_e}")
    if "embeddings_router" in locals():
        _include_if_enabled("embeddings", embeddings_router, prefix=f"{API_V1_PREFIX}", tags=["embeddings"])
    if "vector_stores_router" in locals():
        _include_if_enabled("vector-stores", vector_stores_router, prefix=f"{API_V1_PREFIX}", tags=["vector-stores"])
    # External connectors (Drive/Notion) scaffold
    try:
        from tldw_Server_API.app.api.v1.endpoints.connectors import router as connectors_router

        _include_if_enabled(
            "connectors", connectors_router, prefix=f"{API_V1_PREFIX}", tags=["connectors"], default_stable=False
        )
    except _IMPORT_EXCEPTIONS as _conn_e:
        logger.warning(f"Connectors endpoints unavailable; skipping import: {_conn_e}")
    _include_if_enabled(
        "ingestion-sources",
        ingestion_sources_router,
        prefix=f"{API_V1_PREFIX}",
        tags=["ingestion-sources"],
        default_stable=False,
    )
    if "claims_router" in locals():
        _include_if_enabled("claims", claims_router, prefix=f"{API_V1_PREFIX}")
    if "media_embeddings_router" in locals():
        _include_if_enabled(
            "media-embeddings", media_embeddings_router, prefix=f"{API_V1_PREFIX}", tags=["media-embeddings"]
        )
    try:
        # Unified items endpoint
        from tldw_Server_API.app.api.v1.endpoints.items import router as _items_router

        _include_if_enabled("items", _items_router, prefix=f"{API_V1_PREFIX}", tags=["items"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Items endpoint not available: {_e}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.reminders import router as _reminders_router

        _include_if_enabled("tasks", _reminders_router, prefix=f"{API_V1_PREFIX}", tags=["tasks"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Reminders endpoint not available: {_e}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.integrations_control_plane import (
            router as _integrations_control_plane_router,
        )

        _include_if_enabled(
            "integrations",
            _integrations_control_plane_router,
            prefix=f"{API_V1_PREFIX}",
            tags=["integrations"],
            default_stable=False,
        )
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Integrations control plane endpoint not available: {_e}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane import (
            router as _scheduled_tasks_control_plane_router,
        )

        _include_if_enabled(
            "scheduled-tasks",
            _scheduled_tasks_control_plane_router,
            prefix=f"{API_V1_PREFIX}",
            tags=["scheduled-tasks"],
            default_stable=False,
        )
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Scheduled tasks control plane endpoint not available: {_e}")
    try:
        from tldw_Server_API.app.api.v1.endpoints.notifications import router as _notifications_router

        _include_if_enabled("notifications", _notifications_router, prefix=f"{API_V1_PREFIX}", tags=["notifications"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Notifications endpoint not available: {_e}")
    _reading_import_enabled = True
    if (
        _EXPLICIT_PYTEST_RUNTIME
        and _MINIMAL_TEST_APP
        and not _test_env_flag_enabled("MINIMAL_TEST_INCLUDE_READING")
    ):
        _reading_import_enabled = False
        logger.info("Skipping reading endpoint imports in pytest startup (set MINIMAL_TEST_INCLUDE_READING=1 to enable)")
    if _reading_import_enabled:
        try:
            from tldw_Server_API.app.api.v1.endpoints.reading import router as _reading_router

            _include_if_enabled("reading", _reading_router, prefix=f"{API_V1_PREFIX}", tags=["reading"])
        except _IMPORT_EXCEPTIONS as _e:
            logger.warning(f"Reading endpoint not available: {_e}")
    # Watchlists endpoints (sources/groups/tags/jobs/runs)
    try:
        from tldw_Server_API.app.api.v1.endpoints.watchlists import router as _watchlists_router

        _include_if_enabled("watchlists", _watchlists_router, prefix=f"{API_V1_PREFIX}", tags=["watchlists"])
    except _IMPORT_EXCEPTIONS as _e:
        logger.warning(f"Watchlists endpoint not available: {_e}")
    # Include Notes Graph routes before generic notes routes so /graph is not shadowed by /{note_id}
    if _HAS_NOTES_GRAPH:
        _include_if_enabled(
            "notes", notes_graph_router, prefix=f"{API_V1_PREFIX}/notes", tags=["notes"]
        )  # /api/v1/notes/graph
    _include_if_enabled("notes", notes_router, prefix=f"{API_V1_PREFIX}/notes", tags=["notes"])
    if _HAS_WEB_CLIPPER:
        _include_if_enabled("web-clipper", web_clipper_router, prefix=f"{API_V1_PREFIX}/web-clipper", tags=["web-clipper"])
    _include_if_enabled("translation", translate_router, prefix=f"{API_V1_PREFIX}", tags=["translation"])
    _include_if_enabled("slides", slides_router, prefix=f"{API_V1_PREFIX}", tags=["slides"])
    _include_if_enabled("prompts", prompt_router, prefix=f"{API_V1_PREFIX}/prompts", tags=["prompts"])
    # Kanban Board endpoints
    if _HAS_KANBAN:
        _include_if_enabled(
            "kanban", kanban_boards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_lists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_cards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_labels_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_checklists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_comments_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_search_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_links_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
        _include_if_enabled(
            "kanban", kanban_workflow_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"]
        )
    if _HAS_READING_HIGHLIGHTS:
        _include_if_enabled(
            "reading-highlights", reading_highlights_router, prefix=f"{API_V1_PREFIX}", tags=["reading-highlights"]
        )
    if _HAS_PROMPT_STUDIO:
        _include_if_enabled("prompt-studio", prompt_studio_projects_router, tags=["prompt-studio"])
        _include_if_enabled("prompt-studio", prompt_studio_prompts_router, tags=["prompt-studio"])
        _include_if_enabled("prompt-studio", prompt_studio_test_cases_router, tags=["prompt-studio"])
        _include_if_enabled("prompt-studio", prompt_studio_optimization_router, tags=["prompt-studio"])
        _include_if_enabled("prompt-studio", prompt_studio_status_router, tags=["prompt-studio"])
        _include_if_enabled("prompt-studio", prompt_studio_evaluations_router, tags=["prompt-studio"])
        _include_if_enabled("prompt-studio", prompt_studio_websocket_router, tags=["prompt-studio"])
    _include_if_enabled("rag-health", rag_health_router, tags=["rag-health"])
    _include_if_enabled("rag-unified", rag_unified_router, tags=["rag-unified"])
    if "text2sql_router" in locals():
        _include_if_enabled("text2sql", text2sql_router, prefix=f"{API_V1_PREFIX}", tags=["text2sql"])
    _include_if_enabled("feedback", feedback_router, prefix=f"{API_V1_PREFIX}/feedback", tags=["feedback"])
    if _HAS_WORKFLOWS:
        # In test contexts, force-include workflows regardless of policy to avoid 404s.
        _test_ctx = bool(_TEST_MODE)
        if _test_ctx:
            app.include_router(workflows_router, prefix="", tags=["workflows"])
        else:
            _include_if_enabled("workflows", workflows_router, tags=["workflows"], default_stable=False)
    if _HAS_CHAT_WORKFLOWS:
        _test_ctx = bool(_TEST_MODE)
        if _test_ctx:
            app.include_router(chat_workflows_router, prefix="", tags=["chat-workflows"])
        else:
            _include_if_enabled("chat-workflows", chat_workflows_router, tags=["chat-workflows"])
    try:
        from tldw_Server_API.app.api.v1.endpoints.scheduler_workflows import router as scheduler_workflows_router

        _HAS_SCHEDULER_WF = True
    except _IMPORT_EXCEPTIONS as _sch_import_err:
        logger.warning(f"Scheduler Workflows endpoints unavailable; skipping import: {_sch_import_err}")
        _HAS_SCHEDULER_WF = False
    if _HAS_SCHEDULER_WF:
        _test_ctx = bool(_TEST_MODE)
        if _test_ctx:
            app.include_router(scheduler_workflows_router, prefix="", tags=["scheduler"])
        else:
            _include_if_enabled("scheduler", scheduler_workflows_router, tags=["scheduler"], default_stable=False)
    _include_if_enabled("research", research_router, prefix=f"{API_V1_PREFIX}/research", tags=["research"])
    _include_if_enabled("research", research_runs_router, prefix=f"{API_V1_PREFIX}", tags=["research-runs"])
    _include_if_enabled(
        "paper-search", paper_search_router, prefix=f"{API_V1_PREFIX}/paper-search", tags=["paper-search"]
    )
    # Heavy routers: import only when enabled to avoid import-time side effects
    try:
        if route_enabled("evaluations"):
            from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import (
                router as _evaluations_router,
            )

            app.include_router(_evaluations_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])
        else:
            logger.info("Route disabled by policy: evaluations")
    except _IMPORT_EXCEPTIONS as _evals_rt_err:
        logger.warning(f"Route gating error for evaluations; skipping import. Error: {_evals_rt_err}")

    try:
        if route_enabled("ocr"):
            from tldw_Server_API.app.api.v1.endpoints.ocr import router as _ocr_router

            app.include_router(_ocr_router, prefix=f"{API_V1_PREFIX}", tags=["ocr"])
        else:
            logger.info("Route disabled by policy: ocr")
    except _IMPORT_EXCEPTIONS as _ocr_rt_err:
        logger.warning(f"Route gating error for ocr; skipping import. Error: {_ocr_rt_err}")

    try:
        if route_enabled("vlm"):
            from tldw_Server_API.app.api.v1.endpoints.vlm import router as _vlm_router

            app.include_router(_vlm_router, prefix=f"{API_V1_PREFIX}", tags=["vlm"])
        else:
            logger.info("Route disabled by policy: vlm")
    except _IMPORT_EXCEPTIONS as _vlm_rt_err:
        logger.warning(f"Route gating error for vlm; skipping import. Error: {_vlm_rt_err}")
    _include_if_enabled(
        "benchmarks", benchmark_router, prefix=f"{API_V1_PREFIX}", tags=["benchmarks"], default_stable=False
    )
    from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router

    try:
        from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router

        _HAS_JOBS_ADMIN = True
    except _IMPORT_EXCEPTIONS as _e:
        _HAS_JOBS_ADMIN = False
        try:
            from loguru import logger as _logger

            _logger.warning(f"Skipping jobs_admin router due to import error: {_e}")
        except _LOGGING_SETUP_EXCEPTIONS:
            pass
    _include_if_enabled("setup", setup_router, prefix=f"{API_V1_PREFIX}", tags=["setup"])
    _include_if_enabled("config", config_info_router, prefix=f"{API_V1_PREFIX}", tags=["config"])
    try:
        from tldw_Server_API.app.api.v1.endpoints.config_admin import router as config_admin_router

        _include_if_enabled("config", config_admin_router, prefix=f"{API_V1_PREFIX}", tags=["config", "admin"])
    except _IMPORT_EXCEPTIONS as _config_admin_err:
        logger.warning(f"Admin config endpoint unavailable; skipping import: {_config_admin_err}")
    # Resource Governor policy snapshot endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.resource_governor import router as resource_governor_router

        _include_if_enabled(
            "resource-governor", resource_governor_router, prefix=f"{API_V1_PREFIX}", tags=["resource-governor"]
        )
    except _IMPORT_EXCEPTIONS as _rg_ep_err:
        logger.warning(f"Resource Governor endpoint unavailable; skipping import: {_rg_ep_err}")
    if _HAS_JOBS_ADMIN:
        _include_if_enabled(
            "jobs",
            jobs_admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=["jobs"],
            default_stable=False,
        )
    _include_if_enabled("sync", sync_router, prefix=f"{API_V1_PREFIX}/sync", tags=["sync"])
    # Tools router included above with prefix f"{API_V1_PREFIX}"; avoid duplicate nested path
    # Sandbox (scaffold)
    if _HAS_SANDBOX:
        if _TEST_MODE:
            # In tests, force-include sandbox endpoints regardless of route policy
            app.include_router(sandbox_router, prefix=f"{API_V1_PREFIX}", tags=["sandbox"])
        else:
            _include_if_enabled(
                "sandbox", sandbox_router, prefix=f"{API_V1_PREFIX}", tags=["sandbox"], default_stable=False
            )
    # Flashcards are now considered stable; include by default unless disabled
    _include_if_enabled(
        "flashcards", flashcards_router, prefix=f"{API_V1_PREFIX}", tags=["flashcards"], default_stable=True
    )
    _include_if_enabled(
        "quizzes", quizzes_router, prefix=f"{API_V1_PREFIX}", tags=["quizzes"], default_stable=True
    )
    _include_if_enabled(
        "study-suggestions",
        study_suggestions_router,
        prefix=f"{API_V1_PREFIX}",
        tags=["study-suggestions"],
        default_stable=True,
    )
    if "writing_router" in locals() and writing_router is not None:
        _include_if_enabled(
            "writing", writing_router, prefix=f"{API_V1_PREFIX}/writing", tags=["writing"], default_stable=True
        )
    if "manuscripts_router" in locals() and manuscripts_router is not None:
        _include_if_enabled(
            "manuscripts", manuscripts_router, prefix=f"{API_V1_PREFIX}/writing/manuscripts",
            tags=["manuscripts"], default_stable=True
        )
    from tldw_Server_API.app.api.v1.endpoints.persona import (
        router as persona_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.personalization import (
        router as personalization_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.companion import (
        router as companion_router,
    )

    _include_if_enabled(
        "personalization",
        personalization_router,
        prefix=f"{API_V1_PREFIX}/personalization",
        tags=["personalization"],
        default_stable=False,
    )
    _include_if_enabled(
        "companion",
        companion_router,
        prefix=f"{API_V1_PREFIX}/companion",
        tags=["companion"],
        default_stable=False,
    )
    try:
        from tldw_Server_API.app.api.v1.endpoints.guardian_controls import router as guardian_controls_router_full
        from tldw_Server_API.app.api.v1.endpoints.family_wizard import router as family_wizard_router_full
        from tldw_Server_API.app.api.v1.endpoints.self_monitoring import router as self_monitoring_router_full

        _include_if_enabled(
            "guardian",
            guardian_controls_router_full,
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=["guardian"],
            default_stable=False,
        )
        _include_if_enabled(
            "guardian",
            family_wizard_router_full,
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=["guardian"],
            default_stable=False,
        )
        _include_if_enabled(
            "self-monitoring",
            self_monitoring_router_full,
            prefix=f"{API_V1_PREFIX}/self-monitoring",
            tags=["self-monitoring"],
            default_stable=False,
        )
    except _STARTUP_GUARD_EXCEPTIONS as _guardian_full_err:
        logger.debug(f"Guardian/self-monitoring routers unavailable in full app: {_guardian_full_err}")
    # In tests, force-include persona endpoints regardless of route policy for WS/unit coverage
    if _TEST_MODE:
        app.include_router(persona_router, prefix=f"{API_V1_PREFIX}/persona", tags=["persona"])
    else:
        _include_if_enabled(
            "persona", persona_router, prefix=f"{API_V1_PREFIX}/persona", tags=["persona"], default_stable=True
        )
    # Archetype template endpoints are always available (read-only catalog data)
    try:
        from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import router as archetype_router  # noqa: F811

        include_router_idempotent(
            app, archetype_router, prefix=f"{API_V1_PREFIX}/persona/archetypes", tags=["persona-archetypes"]
        )
    except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _arch_full_err:
        logger.debug("Archetype router unavailable in full app: {}", _arch_full_err)
    _include_if_enabled("mcp-unified", mcp_unified_router, prefix=f"{API_V1_PREFIX}", tags=["mcp-unified"])
    _include_if_enabled("chatbooks", chatbooks_router, prefix=f"{API_V1_PREFIX}", tags=["chatbooks"])
    _include_if_enabled("sharing", sharing_router, prefix=f"{API_V1_PREFIX}", tags=["sharing"])
    _include_if_enabled("llm", mlx_router, prefix=f"{API_V1_PREFIX}", tags=["llm"])
    _include_if_enabled("llm", llm_providers_router, prefix=f"{API_V1_PREFIX}", tags=["llm"])
    _include_if_enabled("llm", messages_router, prefix=f"{API_V1_PREFIX}", tags=["messages"])
    _include_if_enabled("llm", messages_public_router, prefix="", tags=["messages"])
    _include_if_enabled("llamacpp", llamacpp_router, prefix=f"{API_V1_PREFIX}", tags=["llamacpp"])
    _include_if_enabled("llamacpp", llamacpp_public_router, prefix="", tags=["llamacpp"])
    _include_if_enabled("web-scraping", web_scraping_router, tags=["web-scraping"])
    _include_if_enabled("web-scraping", web_scraping_router, prefix=f"{API_V1_PREFIX}", tags=["web-scraping"])

# Register control-plane metrics endpoints (works in both minimal and full modes)
if _shared_env_flag_enabled("ENABLE_ADMIN_E2E_TEST_MODE"):
    try:
        from tldw_Server_API.app.api.v1.endpoints.test_support.admin_e2e import (
            router as admin_e2e_test_support_router,
        )

        include_router_idempotent(
            app,
            admin_e2e_test_support_router,
            prefix=f"{API_V1_PREFIX}/test-support/admin-e2e",
            tags=["test-support"],
        )
    except _IMPORT_EXCEPTIONS as _admin_e2e_err:
        logger.warning(f"Failed to include admin e2e test-support router: {_admin_e2e_err}")

try:
    if route_enabled("metrics"):
        app.add_api_route("/metrics", metrics, include_in_schema=False)
        app.add_api_route(f"{API_V1_PREFIX}/metrics", api_metrics, methods=["GET"], tags=["monitoring"])
    else:
        logger.info("Route disabled by policy: metrics")
except _STARTUP_GUARD_EXCEPTIONS as _metrics_rt_err:
    logger.warning(f"Route gating error for metrics; including by default. Error: {_metrics_rt_err}")
    app.add_api_route("/metrics", metrics, include_in_schema=False)
    app.add_api_route(f"{API_V1_PREFIX}/metrics", api_metrics, methods=["GET"], tags=["monitoring"])

# Router for trash endpoints - deletion of media items / trash file handling (FIXME: Secure delete vs lag on delete?)
# app.include_router(trash_router, prefix=f"{API_V1_PREFIX}/trash", tags=["trash"])

# Router for authentication endpoint
# app.include_router(auth_router, prefix=f"{API_V1_PREFIX}/auth", tags=["auth"])
# The docs at http://localhost:8000/docs will show an “Authorize” button. You can log in by calling POST /api/v1/auth/login with a form that includes username and password. The docs interface is automatically aware because we used OAuth2PasswordBearer.


# Health check (registered conditionally below)
async def health_check():
    body = {"status": "healthy"}
    # Always attempt to include RG policy snapshot: prefer app.state, fallback to configured file
    try:
        rgv = getattr(app.state, "rg_policy_version", None)
        if rgv is not None:
            body["rg_policy_version"] = int(rgv)
            body["rg_policy_store"] = getattr(app.state, "rg_policy_store", None)
            body["rg_policy_count"] = getattr(app.state, "rg_policy_count", None)
        else:
            # Fallback to RG_POLICY_PATH (file-based) when loader not initialized
            import os as _os
            from pathlib import Path as _Path

            import yaml as _yaml

            p = _os.getenv("RG_POLICY_PATH")
            if p and _Path(p).exists():
                try:
                    with _Path(p).open("r", encoding="utf-8") as _f:
                        _data = _yaml.safe_load(_f) or {}
                    body["rg_policy_version"] = int(_data.get("version") or 1)
                    body["rg_policy_store"] = _os.getenv("RG_POLICY_STORE", "file")
                    body["rg_policy_count"] = len((_data.get("policies") or {}).keys())
                except _REQUEST_GUARD_EXCEPTIONS:
                    pass
    except _REQUEST_GUARD_EXCEPTIONS:
        pass
    return body


# Readiness check (verifies critical dependencies) - registered conditionally below
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe for orchestrators and load balancers."""
    try:
        lifecycle = get_or_create_lifecycle_state(request.app)
        if lifecycle.draining or lifecycle.phase == "draining":
            return JSONResponse(
                {"status": "not_ready", "reason": "shutdown_in_progress"},
                status_code=503,
            )
        # Engine stats
        try:
            from tldw_Server_API.app.core.Workflows.engine import WorkflowScheduler as _WS

            engine_stats = _WS.instance().stats()
        except _REQUEST_GUARD_EXCEPTIONS:
            engine_stats = {"queue_depth": None, "active_tenants": None, "active_workflows": None}

        # DB health (AuthNZ pool basic health for API; Workflows DB schema check below)
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        db_pool = await get_db_pool()
        db_health = await db_pool.health_check()

        # Workflows backend schema check
        try:
            from tldw_Server_API.app.core.DB_Management.DB_Manager import (
                create_workflows_database,
                get_content_backend_instance,
            )
            from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase as _WDB

            backend = get_content_backend_instance()
            wdb: _WDB = create_workflows_database(backend=backend)
            if wdb._using_backend():
                with wdb.backend.transaction() as conn:  # type: ignore[union-attr]
                    try:
                        wf_schema_version = int(wdb._get_backend_schema_version(conn))  # type: ignore[attr-defined]
                        wf_expected_version = int(wdb._CURRENT_SCHEMA_VERSION)  # type: ignore[attr-defined]
                    except _REQUEST_GUARD_EXCEPTIONS:
                        wf_schema_version = None
                        wf_expected_version = None
            else:
                wf_schema_version = None
                wf_expected_version = None
        except _REQUEST_GUARD_EXCEPTIONS:
            wf_schema_version = None
            wf_expected_version = None

        # Provider manager health (if initialized)
        try:
            from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager

            pm = get_provider_manager()
            provider_health = pm.get_health_report() if pm else {}
            providers_ok = pm is not None
        except _REQUEST_GUARD_EXCEPTIONS:
            provider_health = {}
            providers_ok = False

        # OTEL status
        from tldw_Server_API.app.core.Metrics import OTEL_AVAILABLE

        ready = db_health.get("status") == "healthy"
        # If workflows backend reports schema version, ensure it matches expected
        if wf_schema_version is not None and wf_expected_version is not None:
            ready = ready and (wf_schema_version == wf_expected_version)
        body = {
            "status": "ready" if ready else "not_ready",
            "database": db_health,
            "workflows_db": {
                "schema_version": wf_schema_version,
                "expected_version": wf_expected_version,
            },
            "engine": engine_stats,
            "providers_initialized": providers_ok,
            "provider_health": provider_health,
            "otel_available": bool(OTEL_AVAILABLE),
        }
        # Include Resource Governor policy metadata; prefer app.state and fallback to RG_POLICY_PATH
        try:
            rgv = getattr(app.state, "rg_policy_version", None)
            if rgv is not None:
                body["rg_policy"] = {
                    "version": int(rgv),
                    "store": getattr(app.state, "rg_policy_store", None),
                    "policies": getattr(app.state, "rg_policy_count", None),
                }
            else:
                import os as _os
                from pathlib import Path as _Path

                import yaml as _yaml

                p = _os.getenv("RG_POLICY_PATH")
                if p and _Path(p).exists():
                    try:
                        with _Path(p).open("r", encoding="utf-8") as _f:
                            _data = _yaml.safe_load(_f) or {}
                        body["rg_policy"] = {
                            "version": int(_data.get("version") or 1),
                            "store": _os.getenv("RG_POLICY_STORE", "file"),
                            "policies": len((_data.get("policies") or {}).keys()),
                        }
                    except _REQUEST_GUARD_EXCEPTIONS:
                        pass
        except _REQUEST_GUARD_EXCEPTIONS:
            pass
        return JSONResponse(body, status_code=(200 if ready else 503))
    except _READINESS_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Readiness check failed: {type(exc).__name__}: {exc}")
        return JSONResponse(
            {"status": "not_ready", "reason": "dependency_check_failed"},
            status_code=503,
        )


# /health/ready alias for some orchestrators (registered conditionally below)
async def readiness_alias(request: Request) -> JSONResponse:
    return await readiness_check(request)


# Register control-plane health endpoints (works in both minimal and full modes)
try:
    if route_enabled("health"):
        app.add_api_route("/health", health_check, methods=["GET", "HEAD"], openapi_extra={"security": []})
        app.add_api_route("/ready", readiness_check, methods=["GET", "HEAD"], openapi_extra={"security": []})
        app.add_api_route("/health/ready", readiness_alias, methods=["GET", "HEAD"], openapi_extra={"security": []})
    else:
        logger.info("Route disabled by policy: health (/health, /ready, /health/ready)")
except _STARTUP_GUARD_EXCEPTIONS as _health_rt_err:
    logger.warning(f"Route gating error for health; including by default. Error: {_health_rt_err}")
    app.add_api_route("/health", health_check, methods=["GET", "HEAD"], openapi_extra={"security": []})
    app.add_api_route("/ready", readiness_check, methods=["GET", "HEAD"], openapi_extra={"security": []})
    app.add_api_route("/health/ready", readiness_alias, methods=["GET", "HEAD"], openapi_extra={"security": []})

# Import-time CI/startup guard: fail immediately if the route table contains duplicates.
_fail_on_duplicate_route_method_pairs(app, context="module import")


#
## Entry point for running the server
########################################################################################################################
def run_server():
    """Run the FastAPI server using uvicorn."""
    import uvicorn

    uvicorn.run("tldw_Server_API.app.main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    run_server()

#
## End of main.py
########################################################################################################################
