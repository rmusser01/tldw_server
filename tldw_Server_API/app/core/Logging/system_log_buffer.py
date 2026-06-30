from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread, local
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.Utils import get_database_dir

_DEFAULT_BUFFER_SIZE = 2000


def _coerce_bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse an integer setting and clamp it to configured bounds."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


_BUFFER_SIZE = _coerce_bounded_int(
    os.getenv("SYSTEM_LOG_BUFFER_SIZE"),
    default=_DEFAULT_BUFFER_SIZE,
    minimum=100,
)
_BUFFER: deque[dict[str, Any]] = deque(maxlen=_BUFFER_SIZE)
_BUFFER_LOCK = Lock()
_SINK_LOCK = Lock()
_SINK_ATTR = "_tldw_system_log_buffer_sink_id"
_SINK_ID: int | None = getattr(logger, _SINK_ATTR, None)
_LOG_FILE_QUEUE_ATTR = "_tldw_system_log_buffer_file_queue"
_LOG_FILE_WORKER_ATTR = "_tldw_system_log_buffer_file_worker"

_DEFAULT_LOG_FILE_ENTRIES = 5000
_DEFAULT_LOG_FILE_COMPACT_EVERY_WRITES = 250
_DEFAULT_LOG_FILE_QUEUE_SIZE = 1000
_LOG_FILE_SETTINGS_LOCK = Lock()
_LOG_FILE_SETTINGS_THREAD_STATE = local()
_LOG_FILE_SETTINGS_INITIALIZED = False
_LOG_FILE_MAX_ENTRIES = _DEFAULT_LOG_FILE_ENTRIES
_LOG_FILE_COMPACT_EVERY_WRITES = _DEFAULT_LOG_FILE_COMPACT_EVERY_WRITES
_LOG_FILE_APPENDS_SINCE_COMPACT = 0
_LOG_FILE_ENABLED = True
_LOG_FILE_PATH = Path(get_database_dir()) / "system_logs.jsonl"
_LOG_FILE_LOCK_TIMEOUT = 5.0
_LOG_FILE_LOCK_POLL_INTERVAL = 0.05
_LOG_FILE_STALE_LOCK_MIN_SECONDS = 1.0
_LOG_FILE_COMPACTION_COUNTER_LOCK = Lock()
_LOG_FILE_QUEUE_SIZE = _coerce_bounded_int(
    os.getenv("SYSTEM_LOG_FILE_QUEUE_SIZE"),
    default=_DEFAULT_LOG_FILE_QUEUE_SIZE,
    minimum=100,
)
_existing_log_file_queue = getattr(logger, _LOG_FILE_QUEUE_ATTR, None)
_LOG_FILE_QUEUE: Queue[dict[str, Any]] = (
    _existing_log_file_queue
    if isinstance(_existing_log_file_queue, Queue)
    else Queue(maxsize=_LOG_FILE_QUEUE_SIZE)
)
with suppress(Exception):
    setattr(logger, _LOG_FILE_QUEUE_ATTR, _LOG_FILE_QUEUE)
_LOG_FILE_WORKER: Thread | None = getattr(logger, _LOG_FILE_WORKER_ATTR, None)
_LOG_FILE_WORKER_LOCK = Lock()

_LOGURU_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9-_]{8,}")
_AUTH_HEADER_RE = re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:bearer\s+)?[^\s,;]+")
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"
)
_REDACTED_VALUE = "***REDACTED***"
_SENSITIVE_LOG_KEY_FRAGMENTS = (
    "password",
    "secret",
    "apikey",
    "authorization",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "bearertoken",
    "apitoken",
    "idtoken",
)

try:
    import fcntl  # type: ignore

    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

_EXTRA_FIELDS = {
    "request_id",
    "org_id",
    "user_id",
    "trace_id",
    "span_id",
    "correlation_id",
    "event",
}


def _coerce_optional_int(value: Any) -> int | None:
    """Return an integer value when coercion succeeds, otherwise None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: str | None, default: bool) -> bool:
    """Parse a truthy configuration value with a caller-provided default."""
    if value is None:
        return default
    return is_truthy(value)


def _normalize_log_level(value: str | None, default: str = "DEBUG") -> str:
    """Return a Loguru level name, falling back when the value is invalid."""
    raw = str(value or default).strip().upper()
    return raw if raw in _LOGURU_LEVELS else default


def _redact_log_text(value: str) -> str:
    """Redact common secret patterns before admin log exposure."""
    text = _OPENAI_KEY_RE.sub(f"sk-{_REDACTED_VALUE}", value)
    text = _AUTH_HEADER_RE.sub(f"authorization={_REDACTED_VALUE}", text)
    text = _BEARER_TOKEN_RE.sub(f"Bearer {_REDACTED_VALUE}", text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}={_REDACTED_VALUE}", text)
    return text


def _is_sensitive_log_key(key: Any) -> bool:
    """Return True when a structured log field name usually carries a secret."""
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return normalized == "token" or any(
        fragment in normalized for fragment in _SENSITIVE_LOG_KEY_FRAGMENTS
    )


def _redact_log_value(value: Any) -> Any:
    """Redact string values while leaving structured non-strings unchanged."""
    if isinstance(value, str):
        return _redact_log_text(value)
    return value


def _redact_log_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a captured log entry with sensitive text redacted."""
    redacted: dict[str, Any] = {}
    for key, value in entry.items():
        redacted[key] = (
            _REDACTED_VALUE if _is_sensitive_log_key(key) else _redact_log_value(value)
        )
    return redacted


def _extract_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """Extract supported structured fields and sensitive extras from Loguru."""
    payload: dict[str, Any] = {}
    for key, val in extra.items():
        if key not in _EXTRA_FIELDS and not _is_sensitive_log_key(key):
            continue
        if key in {"org_id", "user_id"}:
            payload[key] = _coerce_optional_int(val)
        else:
            payload[key] = val if val is not None else None
    return payload


def _log_file_settings_init_active() -> bool:
    """Return True while this thread is initializing log-file settings."""
    return bool(getattr(_LOG_FILE_SETTINGS_THREAD_STATE, "initializing", False))


def _init_log_file_settings() -> None:
    """Load file-backed system-log settings from env/config once per module."""
    global _LOG_FILE_MAX_ENTRIES
    global _LOG_FILE_COMPACT_EVERY_WRITES
    global _LOG_FILE_APPENDS_SINCE_COMPACT
    global _LOG_FILE_ENABLED
    global _LOG_FILE_PATH
    global _LOG_FILE_LOCK_TIMEOUT
    global _LOG_FILE_SETTINGS_INITIALIZED

    if _LOG_FILE_SETTINGS_INITIALIZED or _log_file_settings_init_active():
        return
    with _LOG_FILE_SETTINGS_LOCK:
        if _LOG_FILE_SETTINGS_INITIALIZED:
            return
        _LOG_FILE_SETTINGS_THREAD_STATE.initializing = True
        try:
            env_enabled = os.getenv("SYSTEM_LOG_FILE_ENABLED")
            env_path = os.getenv("SYSTEM_LOG_FILE_PATH")
            env_max_entries = os.getenv("SYSTEM_LOG_FILE_MAX_ENTRIES")
            env_compact_every = os.getenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES")
            env_lock_timeout = os.getenv("SYSTEM_LOG_FILE_LOCK_TIMEOUT")

            config_path = None
            config_max_entries = None
            config_compact_every = None
            if env_path is None or env_max_entries is None or env_compact_every is None:
                try:
                    from tldw_Server_API.app.core.config import load_comprehensive_config

                    parser = load_comprehensive_config()
                    if hasattr(parser, "has_section") and parser.has_section("Logging"):
                        if env_path is None:
                            config_path = parser.get("Logging", "system_log_file_path", fallback=None)
                        if env_max_entries is None:
                            config_max_entries = parser.get("Logging", "system_log_file_max_entries", fallback=None)
                        if env_compact_every is None:
                            config_compact_every = parser.get(
                                "Logging",
                                "system_log_file_compact_every_writes",
                                fallback=None,
                            )
                except Exception:
                    logger.debug("System log settings config read failed")

            _LOG_FILE_ENABLED = _coerce_bool(env_enabled, True)
            path_value = env_path or config_path
            _LOG_FILE_PATH = Path(path_value) if path_value else Path(get_database_dir()) / "system_logs.jsonl"

            max_raw = env_max_entries if env_max_entries is not None else config_max_entries
            try:
                max_entries = int(str(max_raw).strip()) if max_raw else _DEFAULT_LOG_FILE_ENTRIES
            except (TypeError, ValueError):
                max_entries = _DEFAULT_LOG_FILE_ENTRIES
            _LOG_FILE_MAX_ENTRIES = max(100, max_entries)

            compact_raw = env_compact_every if env_compact_every is not None else config_compact_every
            try:
                compact_every = (
                    int(str(compact_raw).strip()) if compact_raw else _DEFAULT_LOG_FILE_COMPACT_EVERY_WRITES
                )
            except (TypeError, ValueError):
                compact_every = _DEFAULT_LOG_FILE_COMPACT_EVERY_WRITES
            compact_every = max(1, compact_every)
            if _LOG_FILE_MAX_ENTRIES > 0:
                compact_every = min(compact_every, _LOG_FILE_MAX_ENTRIES)
            _LOG_FILE_COMPACT_EVERY_WRITES = compact_every
            _LOG_FILE_APPENDS_SINCE_COMPACT = 0

            if env_lock_timeout:
                try:
                    _LOG_FILE_LOCK_TIMEOUT = float(env_lock_timeout)
                except (TypeError, ValueError):
                    _LOG_FILE_LOCK_TIMEOUT = 5.0

            _LOG_FILE_SETTINGS_INITIALIZED = True
        finally:
            _LOG_FILE_SETTINGS_THREAD_STATE.initializing = False


@contextmanager
def _log_file_lock(timeout: float | None = None) -> Iterator[None]:
    """Acquire the shared system-log file lock within the configured timeout."""
    _init_log_file_settings()
    timeout_seconds = _LOG_FILE_LOCK_TIMEOUT if timeout is None else max(0.0, float(timeout))
    lock_path = _LOG_FILE_PATH.with_suffix(_LOG_FILE_PATH.suffix + ".lock")
    lock_fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        stale_after = max(timeout_seconds * 2, _LOG_FILE_STALE_LOCK_MIN_SECONDS)
        if _HAS_FCNTL:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    remaining = timeout_seconds - (time.monotonic() - start_time)
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Failed to acquire system log lock within {timeout_seconds}s"
                        ) from None
                    time.sleep(min(_LOG_FILE_LOCK_POLL_INTERVAL, remaining))
        else:
            while True:
                try:
                    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    break
                except FileExistsError:
                    try:
                        lock_stat = os.stat(lock_path)
                        if time.time() - lock_stat.st_mtime > stale_after:
                            os.unlink(lock_path)
                            continue
                    except (OSError, FileNotFoundError):
                        pass
                    remaining = timeout_seconds - (time.monotonic() - start_time)
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Failed to acquire system log lock within {timeout_seconds}s"
                        ) from None
                    time.sleep(min(_LOG_FILE_LOCK_POLL_INTERVAL, remaining))
        yield
    finally:
        if lock_fd is not None:
            if _HAS_FCNTL:
                with suppress(Exception):
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            with suppress(Exception):
                os.close(lock_fd)
        if not _HAS_FCNTL:
            with suppress(Exception):
                lock_path.unlink(missing_ok=True)


def _coerce_timestamp(value: Any) -> datetime | None:
    """Convert supported timestamp payloads into datetime objects."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _normalize_timestamp(value: Any) -> datetime | None:
    """Normalize a supported timestamp payload to timezone-aware UTC."""
    timestamp = _coerce_timestamp(value)
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _emit_internal_diagnostic(message: str) -> None:
    """Write sink-internal diagnostics without re-entering Loguru."""
    # Must not use Loguru here: this function is called from inside a Loguru sink.
    with suppress(Exception):
        stream = sys.__stderr__ or sys.stderr
        stream.write(message.rstrip() + "\n")


def _exception_type_name(exc: BaseException) -> str:
    """Return a stable exception descriptor that omits raw exception text."""
    return type(exc).__name__


def _sort_timestamp_value(entry: dict[str, Any]) -> float:
    """Return a sortable UTC epoch value for a log entry timestamp."""
    timestamp = _normalize_timestamp(entry.get("timestamp"))
    if timestamp is None:
        return float("-inf")
    return timestamp.timestamp()


def _should_compact_after_append() -> bool:
    """Return True when this append should trigger file compaction."""
    if _LOG_FILE_MAX_ENTRIES <= 0:
        return False
    if _LOG_FILE_COMPACT_EVERY_WRITES <= 1:
        return True
    global _LOG_FILE_APPENDS_SINCE_COMPACT
    with _LOG_FILE_COMPACTION_COUNTER_LOCK:
        _LOG_FILE_APPENDS_SINCE_COMPACT += 1
        if _LOG_FILE_APPENDS_SINCE_COMPACT < _LOG_FILE_COMPACT_EVERY_WRITES:
            return False
        _LOG_FILE_APPENDS_SINCE_COMPACT = 0
    return True


def _compact_log_file_locked() -> None:
    """Trim the log file to the newest configured entries while locked."""
    if _LOG_FILE_MAX_ENTRIES <= 0:
        return
    try:
        lines = _LOG_FILE_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    if len(lines) <= _LOG_FILE_MAX_ENTRIES:
        return
    trimmed = lines[-_LOG_FILE_MAX_ENTRIES:]
    tmp_path = _LOG_FILE_PATH.with_suffix(_LOG_FILE_PATH.suffix + ".tmp")
    tmp_path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
    tmp_path.replace(_LOG_FILE_PATH)


def _append_log_file(entry: dict[str, Any]) -> None:
    """Append one already-redacted entry to the shared JSONL log file."""
    if _log_file_settings_init_active():
        return
    _init_log_file_settings()
    if not _LOG_FILE_ENABLED:
        return
    payload = dict(entry)
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, datetime):
        payload["timestamp"] = timestamp.isoformat()
    try:
        _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _log_file_lock():
            with _LOG_FILE_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            if _should_compact_after_append():
                _compact_log_file_locked()
    except Exception as exc:
        _emit_internal_diagnostic(
            f"system_log_buffer append failed: {_exception_type_name(exc)}"
        )


def _log_file_worker_loop(queue: Queue[dict[str, Any]]) -> None:
    """Drain queued file-backed log entries on a daemon writer thread."""
    while True:
        entry = queue.get()
        try:
            _append_log_file(entry)
        except Exception as exc:
            _emit_internal_diagnostic(
                f"system_log_buffer worker append failed: {_exception_type_name(exc)}"
            )
        finally:
            queue.task_done()


def _ensure_log_file_worker() -> None:
    """Start or reuse the shared daemon worker for file-backed log writes."""
    global _LOG_FILE_WORKER
    worker = _LOG_FILE_WORKER
    if worker is not None and worker.is_alive():
        return
    with _LOG_FILE_WORKER_LOCK:
        worker = _LOG_FILE_WORKER
        if worker is not None and worker.is_alive():
            return
        _LOG_FILE_WORKER = Thread(
            target=_log_file_worker_loop,
            args=(_LOG_FILE_QUEUE,),
            name="tldw-system-log-writer",
            daemon=True,
        )
        _LOG_FILE_WORKER.start()
        with suppress(Exception):
            setattr(logger, _LOG_FILE_WORKER_ATTR, _LOG_FILE_WORKER)


def _enqueue_log_file(entry: dict[str, Any]) -> None:
    """Queue a log entry for file persistence without raising into Loguru."""
    try:
        if _log_file_settings_init_active():
            return
        _init_log_file_settings()
        if not _LOG_FILE_ENABLED:
            return
        _ensure_log_file_worker()
        _LOG_FILE_QUEUE.put_nowait(dict(entry))
    except Full:
        _emit_internal_diagnostic("system_log_buffer queue full; dropping file-backed log entry")
    except Exception as exc:
        _emit_internal_diagnostic(
            f"system_log_buffer enqueue failed: {_exception_type_name(exc)}"
        )


def _read_log_file_entries() -> list[dict[str, Any]]:
    """Read file-backed log entries, skipping malformed lines safely."""
    _init_log_file_settings()
    if not _LOG_FILE_ENABLED:
        return []
    if not _LOG_FILE_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with _log_file_lock():
            lines = _LOG_FILE_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        logger.debug("Failed to read system log file")
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        timestamp = _normalize_timestamp(entry.get("timestamp"))
        if timestamp is not None:
            entry["timestamp"] = timestamp
        entries.append(entry)
    return entries


def _entry_dedupe_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable key for merging file and in-memory log entries."""
    timestamp = _normalize_timestamp(entry.get("timestamp"))
    timestamp_key = timestamp.isoformat() if timestamp else str(entry.get("timestamp") or "")
    return (
        timestamp_key,
        entry.get("level"),
        entry.get("message"),
        entry.get("logger"),
        entry.get("module"),
        entry.get("function"),
        entry.get("line"),
        entry.get("request_id"),
        entry.get("org_id"),
        entry.get("user_id"),
        entry.get("trace_id"),
        entry.get("span_id"),
        entry.get("correlation_id"),
        entry.get("event"),
    )


def _log_sink(message: Any) -> None:
    """Capture a Loguru record into the memory buffer and async file queue."""
    record = message.record
    extra = _extract_extra(record.get("extra", {}))
    entry = _redact_log_entry({
        "timestamp": record.get("time"),
        "level": record.get("level").name if record.get("level") else None,
        "message": record.get("message"),
        "logger": record.get("name"),
        "module": record.get("module"),
        "function": record.get("function"),
        "line": record.get("line"),
        **extra,
    })
    with _BUFFER_LOCK:
        _BUFFER.append(entry)
    try:
        _enqueue_log_file(entry)
    except Exception as exc:
        _emit_internal_diagnostic(
            f"system_log_buffer sink enqueue failed: {_exception_type_name(exc)}"
        )


def _sink_still_present(sink_id: int) -> bool:
    """Check if a loguru sink with the given ID still exists.

    WARNING: This function accesses loguru's private internals and may break
    with future loguru versions. It returns False if the sink is not found
    or if any error occurs during the check.

    Args:
        sink_id: The integer ID returned by logger.add().

    Returns:
        True if the sink appears to still exist, False otherwise.
    """
    # Loguru doesn't expose a public API for checking removed sinks.
    try:
        core = getattr(logger, "_core", None)
        handlers = getattr(core, "handlers", None)
        return isinstance(handlers, dict) and sink_id in handlers
    except (AttributeError, TypeError, KeyError):
        return False


def ensure_system_log_buffer() -> None:
    """Attach a Loguru sink to capture recent logs into an in-memory ring buffer."""
    global _SINK_ID
    _init_log_file_settings()
    with _SINK_LOCK:
        if _SINK_ID is not None and _sink_still_present(_SINK_ID):
            return
        _SINK_ID = logger.add(
            _log_sink,
            level=_normalize_log_level(os.getenv("SYSTEM_LOG_LEVEL"), "DEBUG"),
            backtrace=False,
            diagnose=False,
            enqueue=False,
        )
        with suppress(Exception):
            setattr(logger, _SINK_ATTR, _SINK_ID)


def query_system_logs(
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    level: str | None = None,
    service: str | None = None,
    query: str | None = None,
    org_id: int | None = None,
    org_ids: list[int] | None = None,
    user_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    ensure_system_log_buffer()
    start_norm = _normalize_timestamp(start)
    end_norm = _normalize_timestamp(end)
    level_norm = level.strip().upper() if level else None
    service_norm = service.strip().lower() if service else None
    query_norm = query.strip().lower() if query else None

    entries = _read_log_file_entries()
    with _BUFFER_LOCK:
        buffer_entries = list(_BUFFER)
    if buffer_entries:
        seen = {_entry_dedupe_key(entry) for entry in entries}
        for entry in buffer_entries:
            key = _entry_dedupe_key(entry)
            if key not in seen:
                entries.append(entry)
                seen.add(key)

    filtered: list[dict[str, Any]] = []
    org_id_set = {org_id} if org_id is not None else set(org_ids or [])
    for entry in entries:
        timestamp = _normalize_timestamp(entry.get("timestamp"))
        if timestamp:
            if start_norm and timestamp < start_norm:
                continue
            if end_norm and timestamp > end_norm:
                continue
        if level_norm and (entry.get("level") or "").upper() != level_norm:
            continue
        if service_norm:
            logger_name = (entry.get("logger") or "").lower()
            module_name = (entry.get("module") or "").lower()
            if service_norm not in logger_name and service_norm not in module_name:
                continue
        if query_norm and query_norm not in (entry.get("message") or "").lower():
            continue
        if org_id_set and entry.get("org_id") not in org_id_set:
            continue
        if user_id is not None and entry.get("user_id") != user_id:
            continue
        if timestamp is not None:
            entry["timestamp"] = timestamp
        filtered.append(entry)

    filtered.sort(key=_sort_timestamp_value, reverse=True)
    total = len(filtered)
    safe_offset = max(0, offset)
    safe_limit = max(1, limit)
    return filtered[safe_offset:safe_offset + safe_limit], total
