# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
import asyncio
import logging
import os
from collections.abc import Iterator

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
from tldw_Server_API.app.api.v1.router_registry import include_router_idempotent, register_router_specs
from tldw_Server_API.app.services.app_lifecycle import (
    mark_lifecycle_shutdown,
    mark_lifecycle_startup,  # noqa: F401 - re-exported for lifecycle contract tests.
    get_or_create_lifecycle_state,
)
from tldw_Server_API.app.services import shutdown_coordinated_runtime as _shutdown_coordinated_runtime
from tldw_Server_API.app.services import shutdown_owned_job_pollers as _shutdown_owned_job_pollers
from tldw_Server_API.app.services import startup_pg_rls as _startup_pg_rls
from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS
from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase as WorkerShutdownPhase,
    publish_worker_inventory,
    stop_registered_workers,
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
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    ensure_chacha_rls,
    ensure_prompt_studio_rls,
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
_STARTUP_GUARD_EXCEPTIONS = LIFECYCLE_GUARD_EXCEPTIONS
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


def _run_pg_rls_auto_ensure(backend: Any) -> tuple[bool, bool]:
    """Apply both PostgreSQL RLS installers and log the combined result."""
    return _startup_pg_rls.run_pg_rls_auto_ensure(
        backend,
        ensure_prompt_studio_rls=ensure_prompt_studio_rls,
        ensure_chacha_rls=ensure_chacha_rls,
        logger_obj=logger,
    )


def _apply_shutdown_transition_gate(app: FastAPI, readiness_state: Any | None) -> None:
    """Move the app into draining mode and gate new jobs."""
    def _set_job_acquire_gate(enabled: bool) -> None:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JM

        _JM.set_acquire_gate(enabled)

    _shutdown_coordinated_runtime.apply_shutdown_transition_gate(
        app,
        readiness_state,
        get_or_create_lifecycle_state=get_or_create_lifecycle_state,
        mark_lifecycle_shutdown=mark_lifecycle_shutdown,
        set_job_acquire_gate=_set_job_acquire_gate,
        logger_obj=logger,
        startup_guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        import_exceptions=_IMPORT_EXCEPTIONS,
    )


def _build_legacy_shutdown_context(
    *,
    readiness_state: Any | None,
) -> "LegacyShutdownContext":
    """Collect the explicit shutdown dependencies used by legacy adapters."""
    from tldw_Server_API.app.services.shutdown_legacy_adapters import LegacyShutdownContext

    return LegacyShutdownContext(
        readiness_state=readiness_state,
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

    def _register_legacy_shutdown_components(coordinator: Any, plan: list[Any]) -> list[Any]:
        from tldw_Server_API.app.services.shutdown_legacy_adapters import (
            register_legacy_shutdown_components,
        )

        return register_legacy_shutdown_components(coordinator, plan)

    return _shutdown_coordinated_runtime.build_coordinated_shutdown_coordinator(
        app,
        legacy_shutdown_plan,
        transport_registry=transport_registry,
        coordinator_factory=ShutdownCoordinator,
        register_legacy_shutdown_components=_register_legacy_shutdown_components,
        build_shutdown_components=build_shutdown_components,
        startup_guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        import_exceptions=_IMPORT_EXCEPTIONS,
    )


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
    return await _shutdown_coordinated_runtime.run_coordinated_shutdown(
        app,
        legacy_shutdown_plan,
        transport_registry=transport_registry,
        build_coordinated_shutdown_coordinator=_build_coordinated_shutdown_coordinator,
        get_legacy_shutdown_suppressed_component_names=get_legacy_shutdown_suppressed_component_names,
        logger_obj=logger,
        startup_guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        import_exceptions=_IMPORT_EXCEPTIONS,
    )

_ManagedJobPoller = ManagedWorker


def _is_job_poller_quiesce(handle: ManagedWorker) -> bool:
    """Return whether a lifecycle-managed worker belongs to the job-poller phase."""
    shutdown_phase = getattr(handle, "shutdown_phase", WorkerShutdownPhase.JOB_POLLER_QUIESCE)
    return shutdown_phase in {
        WorkerShutdownPhase.JOB_POLLER_QUIESCE,
        WorkerShutdownPhase.JOB_POLLER_QUIESCE.value,
    }


def _publish_shutdown_job_poller_inventory(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
) -> None:
    """Expose shutdown-owned worker metadata on app.state."""
    publish_worker_inventory(app, handles)


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
        ManagedWorker(
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
            shutdown_phase=WorkerShutdownPhase.JOB_POLLER_QUIESCE,
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
    replacement_handles = [
        ManagedWorker(
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
            shutdown_phase=WorkerShutdownPhase.JOB_POLLER_QUIESCE,
        )
        for name, task, stop_event, timeout_sec in registrations
        if task is not None
    ]
    handles[:] = [
        handle
        for handle in handles
        if not _is_job_poller_quiesce(handle)
    ]
    handles.extend(replacement_handles)
    _publish_shutdown_job_poller_inventory(app, handles)

def _record_shutdown_timing_segment(
    app: FastAPI,
    segment: str,
    duration_ms: int,
    **extra: object,
) -> None:
    """Store one shutdown timing segment and emit a consistent log line."""
    _shutdown_owned_job_pollers.record_shutdown_timing_segment(
        app,
        segment,
        duration_ms,
        logger_obj=logger,
        guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        **extra,
    )


@contextmanager
def _timed_shutdown_segment(
    app: FastAPI,
    segment: str,
    **extra: object,
) -> Iterator[None]:
    """Measure a shutdown block with monotonic time and record it on app.state."""
    with _shutdown_owned_job_pollers.timed_shutdown_segment(
        app,
        segment,
        monotonic=time.monotonic,
        record_shutdown_timing_segment=_record_shutdown_timing_segment,
        **extra,
    ):
        yield


def _record_shutdown_timing_total(app: FastAPI, duration_ms: int) -> None:
    """Record total teardown time and summarize the slowest non-total segment."""
    _shutdown_owned_job_pollers.record_shutdown_timing_total(
        app,
        duration_ms,
        record_shutdown_timing_segment=_record_shutdown_timing_segment,
        logger_obj=logger,
        guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
    )


async def _stop_registered_job_pollers(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
) -> None:
    """Stop registered job pollers, preferring explicit stop events."""
    job_poller_handles = [
        handle
        for handle in handles
        if _is_job_poller_quiesce(handle)
    ]
    await stop_registered_workers(
        app,
        job_poller_handles,
        stopped_names_attr="_tldw_shutdown_quiesced_job_poller_names",
        log_label="job poller",
    )


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
    job_poller_handles = [
        handle
        for handle in handles
        if _is_job_poller_quiesce(handle)
    ]
    await _shutdown_owned_job_pollers.quiesce_owned_job_pollers_for_shutdown(
        app,
        job_poller_handles,
        wait_for_leases_sec=wait_for_leases_sec,
        count_active_processing=count_active_processing,
        stop_registered_job_pollers=_stop_registered_job_pollers,
        record_shutdown_timing_segment=_record_shutdown_timing_segment,
        timed_shutdown_segment=_timed_shutdown_segment,
        guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        monotonic=time.monotonic,
        asyncio_module=asyncio,
    )

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
# Minimal test-app gating: when enabled, skip importing heavy routers
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
    _startup_trace("Minimal test app router imports delegated to router groups.")
else:
    _startup_trace("Full app router imports delegated to router groups.")

# Metrics and Telemetry - import directly and fail fast on errors
# Core helpers - import directly (fail fast if missing)
from tldw_Server_API.app.core.Metrics import (
    get_metrics_registry,
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
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    worker_runtime = LifespanWorkerRuntimeState()
    # Fail fast if the assembled app contains duplicate method+path route registrations.
    _fail_on_duplicate_route_method_pairs(app, context="lifespan startup")
    from tldw_Server_API.app.services.lifespan_startup_sequence import (
        run_lifespan_startup_sequence,
    )

    startup_sequence_handles = await run_lifespan_startup_sequence(
        app=app,
        worker_runtime=worker_runtime,
        module_file=__file__,
        logger=logger,
        readiness_state=READINESS_STATE,
        shared_is_truthy=_shared_is_truthy,
        route_enabled=route_enabled,
        get_mcp_config=get_mcp_config,
        validate_mcp_config=validate_mcp_config,
        test_mode=bool(globals().get("_TEST_MODE")),
        run_pg_rls_auto_ensure=_run_pg_rls_auto_ensure,
        register_owned_job_poller=_register_owned_job_poller,
        replace_owned_job_poller_inventory=_replace_owned_job_poller_inventory,
        startup_api_key_log_value=_startup_api_key_log_value,
        startup_guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        import_exceptions=_IMPORT_EXCEPTIONS,
    )
    db_pool = startup_sequence_handles.db_pool
    session_manager = startup_sequence_handles.session_manager
    heavy_startup_handles = startup_sequence_handles.heavy_startup_handles

    # Note: Audit service now uses dependency injection
    # No need to initialize globally - use get_audit_service_for_user dependency in endpoints
    logger.info("App Startup: Audit service available via dependency injection")

    _run_startup_config_validation()

    yield
    from tldw_Server_API.app.services.lifespan_shutdown_sequence import (
        run_lifespan_shutdown_sequence,
    )

    await run_lifespan_shutdown_sequence(
        app=app,
        worker_runtime=worker_runtime,
        readiness_state=READINESS_STATE,
        db_pool=locals().get("db_pool"),
        session_manager=locals().get("session_manager"),
        heavy_startup_handles=locals().get("heavy_startup_handles"),
        build_legacy_shutdown_context=_build_legacy_shutdown_context,
        apply_shutdown_transition_gate=_apply_shutdown_transition_gate,
        quiesce_owned_job_pollers_for_shutdown=_quiesce_owned_job_pollers_for_shutdown,
        run_coordinated_shutdown=_run_coordinated_shutdown,
        startup_guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        import_exceptions=_IMPORT_EXCEPTIONS,
        in_pytest_runtime=_shared_is_explicit_pytest_runtime(),
        test_db_instance_ref=test_db_instance_ref,
        timed_shutdown_segment=_timed_shutdown_segment,
        record_shutdown_timing_total=_record_shutdown_timing_total,
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
from tldw_Server_API.app.api.v1.utils.exception_handlers import (  # noqa: E402
    client_disconnect_handler as _client_disconnect_handler,
    global_unhandled_exception_handler as _global_handler,
)


def _run_startup_config_validation() -> None:
    """Run best-effort startup config validation without blocking app startup."""
    try:
        from tldw_Server_API.app.core.config import validate_config

        validate_config()
    except (_STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS) as _vc_e:
        logger.warning(f"Config validation could not run: {_vc_e}")


@app.exception_handler(Exception)
async def _global_unhandled_exception_handler(request, exc):
    response = await _global_handler(request, exc)
    return _apply_runtime_cors_headers(request, response)


@app.exception_handler(ClientDisconnect)
async def _client_disconnect_exception_handler(request: Request, exc: ClientDisconnect):
    response = await _client_disconnect_handler(request, exc)
    return _apply_runtime_cors_headers(request, response)


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


_OPENAPI_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}


def _ensure_openapi_operation_tags_declared(openapi_schema: dict[str, Any]) -> None:
    tags = openapi_schema.setdefault("tags", [])
    declared_tags = {
        tag["name"]
        for tag in tags
        if isinstance(tag, dict) and isinstance(tag.get("name"), str)
    }
    operation_tags: set[str] = set()

    for operations in openapi_schema.get("paths", {}).values():
        if not isinstance(operations, dict):
            continue
        for method, operation in operations.items():
            if method.lower() not in _OPENAPI_HTTP_METHODS or not isinstance(operation, dict):
                continue
            op_tags = operation.get("tags", [])
            if isinstance(op_tags, (list, tuple, set)):
                operation_tags.update(tag for tag in op_tags if isinstance(tag, str))

    for missing_tag in sorted(operation_tags - declared_tags):
        tags.append({"name": missing_tag})


# Add global security schemes, servers, and branding to the generated OpenAPI schema
def custom_openapi():
    # All schema normalization below is covered by FastAPI's OpenAPI schema cache.
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )
    _ensure_openapi_operation_tags_declared(openapi_schema)

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


@app.get(
    "/",
    openapi_extra={"security": []},
    responses={
        _starlette_status.HTTP_307_TEMPORARY_REDIRECT: {
            "description": "Redirect to the first-time setup page when setup is required.",
            "headers": {
                "Location": {
                    "description": "Setup page URL.",
                    "schema": {"type": "string"},
                },
            },
        },
    },
)
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
    from tldw_Server_API.app.api.v1.router_groups.minimal import (
        iter_minimal_optional_router_specs,
        iter_minimal_test_router_specs,
    )

    _minimal_grouped_count = register_router_specs(app, iter_minimal_test_router_specs())
    logger.info(f"Registered {_minimal_grouped_count} minimal-test routers from router groups")
    _minimal_optional_count = register_router_specs(app, iter_minimal_optional_router_specs())
    logger.info(f"Registered {_minimal_optional_count} optional minimal-test routers from router groups")
    # Config info endpoints (includes /api/v1/config/jobs used by OpenAPI tests)
else:
    # Register grouped routers first (idempotent — won't conflict with later registrations)
    from tldw_Server_API.app.api.v1.router_registry import register_all_routers as _register_grouped

    _grouped_count = _register_grouped(app)
    logger.info(f"Registered {_grouped_count} routers from router groups")

    logger.info("Auth router consolidated: endpoints/auth.py")
    # Tools router included above with prefix f"{API_V1_PREFIX}"; avoid duplicate nested path
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


def _add_public_control_plane_route(path: str, endpoint: Any) -> None:
    route_kwargs = {"tags": ["health"], "openapi_extra": {"security": []}}
    app.add_api_route(path, endpoint, methods=["GET"], **route_kwargs)
    app.add_api_route(path, endpoint, methods=["HEAD"], **route_kwargs)


# Register control-plane health endpoints (works in both minimal and full modes)
try:
    if route_enabled("health"):
        _add_public_control_plane_route("/health", health_check)
        _add_public_control_plane_route("/ready", readiness_check)
        _add_public_control_plane_route("/health/ready", readiness_alias)
    else:
        logger.info("Route disabled by policy: health (/health, /ready, /health/ready)")
except _STARTUP_GUARD_EXCEPTIONS as _health_rt_err:
    logger.warning(f"Route gating error for health; including by default. Error: {_health_rt_err}")
    _add_public_control_plane_route("/health", health_check)
    _add_public_control_plane_route("/ready", readiness_check)
    _add_public_control_plane_route("/health/ready", readiness_alias)

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
