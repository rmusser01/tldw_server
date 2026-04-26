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
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette import status as _starlette_status
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse, Response
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
from tldw_Server_API.app.services import shutdown_coordinated_runtime as _shutdown_coordinated_runtime
from tldw_Server_API.app.services import shutdown_owned_job_pollers as _shutdown_owned_job_pollers
from tldw_Server_API.app.services import startup_pg_rls as _startup_pg_rls
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

if TYPE_CHECKING:
    from tldw_Server_API.app.services.shutdown_coordinator import (
        ShutdownComponent,
        ShutdownCoordinator,
    )
    from tldw_Server_API.app.services.shutdown_legacy_adapters import LegacyShutdownContext

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

_ManagedJobPoller = _shutdown_owned_job_pollers.ManagedJobPoller


def _publish_shutdown_job_poller_inventory(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
) -> None:
    """Expose shutdown-owned job poller metadata on app.state."""
    _shutdown_owned_job_pollers.publish_shutdown_job_poller_inventory(
        app,
        handles,
        guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
    )


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
    _shutdown_owned_job_pollers.register_owned_job_poller(
        app,
        handles,
        name=name,
        task=task,
        stop_event=stop_event,
        timeout_sec=timeout_sec,
        publish_inventory=_publish_shutdown_job_poller_inventory,
    )


def _replace_owned_job_poller_inventory(
    app: FastAPI,
    handles: list[_ManagedJobPoller],
    *,
    registrations: list[tuple[str, asyncio.Task[Any] | None, asyncio.Event | None, float]],
) -> None:
    """Replace the managed job-poller inventory with the current owned poller set."""
    _shutdown_owned_job_pollers.replace_owned_job_poller_inventory(
        app,
        handles,
        registrations=registrations,
        register_owned_job_poller_fn=_register_owned_job_poller,
        publish_inventory=_publish_shutdown_job_poller_inventory,
    )

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
    await _shutdown_owned_job_pollers.stop_registered_job_pollers(
        app,
        handles,
        logger_obj=logger,
        guard_exceptions=_STARTUP_GUARD_EXCEPTIONS,
        asyncio_module=asyncio,
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
    await _shutdown_owned_job_pollers.quiesce_owned_job_pollers_for_shutdown(
        app,
        handles,
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
                msg = text[len(prefix) :].lstrip()
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
        logger.info(
            "Skipping audio endpoint imports in pytest full startup (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)"
        )

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
        logger.warning(f"Chat workflows endpoints unavailable; skipping import: {_chat_wf_import_err}")
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
# Core helpers - import directly (fail fast if missing)
from tldw_Server_API.app.core.Evaluations.evaluation_manager import get_cached_evaluation_manager
from tldw_Server_API.app.core.Metrics import (
    OTEL_AVAILABLE,
    get_metrics_registry,
    initialize_telemetry,
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
        publish_shutdown_job_poller_inventory=_publish_shutdown_job_poller_inventory,
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

    sample = "; ".join(f"{method} {path} ({first} vs {second})" for path, method, first, second in duplicates[:10])
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
    response.headers.setdefault("Access-Control-Expose-Headers", "X-Request-ID, traceparent, X-Trace-Id")
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
    except _STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS as _vc_e:
        logger.warning(f"Config validation could not run: {_vc_e}")


@app.exception_handler(Exception)
async def _global_unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> Response:
    response = await _global_handler(request, exc)
    return _apply_runtime_cors_headers(request, response)


@app.exception_handler(ClientDisconnect)
async def _client_disconnect_exception_handler(
    request: Request,
    exc: ClientDisconnect,
) -> Response:
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
            logger.info(f"CORS single-user auto-detect: added localhost origins {_auto_added}")
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
_TEST_MODE = _EXPLICIT_PYTEST_RUNTIME and (_TEST_FLAGS_SET or bool(_env_os.getenv("PYTEST_CURRENT_TEST")))

if _TEST_FLAGS_SET and not _EXPLICIT_PYTEST_RUNTIME:
    logger.warning("Test flags are set without explicit pytest runtime; startup guard will reject this configuration.")

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
    include_router_idempotent(
        app, character_memory_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character-memory"]
    )
    include_router_idempotent(
        app, character_chat_sessions_router, prefix=f"{API_V1_PREFIX}/chats", tags=["character-chat-sessions"]
    )
    include_router_idempotent(app, character_messages_router, prefix=f"{API_V1_PREFIX}", tags=["character-messages"])
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
        logger.debug(
            f"Skipping scheduled tasks control plane router in minimal test app: {_scheduled_tasks_cp_min_err}"
        )
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
        logger.info("Skipping audio-jobs router in minimal test app (set MINIMAL_TEST_INCLUDE_AUDIO_JOBS=1 to enable)")

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
        logger.debug(f"Skipping chat workflows router in minimal test app: {_chat_wf_min_err}")
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
        _include_if_enabled(
            "acp", acp_schedules_router, prefix=f"{API_V1_PREFIX}", tags=["acp-schedules"], default_stable=False
        )
    if "acp_triggers_router" in locals() and acp_triggers_router is not None:
        _include_if_enabled(
            "acp", acp_triggers_router, prefix=f"{API_V1_PREFIX}", tags=["acp-triggers"], default_stable=False
        )
    if "acp_permissions_router" in locals() and acp_permissions_router is not None:
        _include_if_enabled(
            "acp", acp_permissions_router, prefix=f"{API_V1_PREFIX}", tags=["acp-permissions"], default_stable=False
        )
    if "acp_multiplex_router" in locals() and acp_multiplex_router is not None:
        _include_if_enabled(
            "acp", acp_multiplex_router, prefix=f"{API_V1_PREFIX}", tags=["acp-multiplex"], default_stable=False
        )
    if "character_router" in locals():
        _include_if_enabled("characters", character_router, prefix=f"{API_V1_PREFIX}/characters", tags=["characters"])
    if "character_memory_router" in locals():
        _include_if_enabled(
            "character-memory", character_memory_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character-memory"]
        )
    if "workspaces_router" in locals():
        _include_if_enabled("workspaces", workspaces_router, prefix=f"{API_V1_PREFIX}/workspaces", tags=["workspaces"])
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
        _include_if_enabled(
            "discord", discord_router, prefix=f"{API_V1_PREFIX}", tags=["discord"], default_stable=False
        )
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
    if _EXPLICIT_PYTEST_RUNTIME and _MINIMAL_TEST_APP and not _test_env_flag_enabled("MINIMAL_TEST_INCLUDE_READING"):
        _reading_import_enabled = False
        logger.info(
            "Skipping reading endpoint imports in pytest startup (set MINIMAL_TEST_INCLUDE_READING=1 to enable)"
        )
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
        _include_if_enabled(
            "web-clipper", web_clipper_router, prefix=f"{API_V1_PREFIX}/web-clipper", tags=["web-clipper"]
        )
    _include_if_enabled("translation", translate_router, prefix=f"{API_V1_PREFIX}", tags=["translation"])
    _include_if_enabled("slides", slides_router, prefix=f"{API_V1_PREFIX}", tags=["slides"])
    _include_if_enabled("prompts", prompt_router, prefix=f"{API_V1_PREFIX}/prompts", tags=["prompts"])
    # Kanban Board endpoints
    if _HAS_KANBAN:
        _include_if_enabled("kanban", kanban_boards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_lists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_cards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_labels_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_checklists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_comments_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_search_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_links_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
        _include_if_enabled("kanban", kanban_workflow_router, prefix=f"{API_V1_PREFIX}/kanban", tags=["kanban"])
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
    _include_if_enabled("quizzes", quizzes_router, prefix=f"{API_V1_PREFIX}", tags=["quizzes"], default_stable=True)
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
            "manuscripts",
            manuscripts_router,
            prefix=f"{API_V1_PREFIX}/writing/manuscripts",
            tags=["manuscripts"],
            default_stable=True,
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
    except _STARTUP_GUARD_EXCEPTIONS + _IMPORT_EXCEPTIONS as _arch_full_err:
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
