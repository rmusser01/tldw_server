from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, local
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.Utils import get_database_dir

_DEFAULT_BUFFER_SIZE = 2000
_BUFFER_SIZE = max(100, int(os.getenv("SYSTEM_LOG_BUFFER_SIZE", str(_DEFAULT_BUFFER_SIZE))))
_BUFFER: deque[dict[str, Any]] = deque(maxlen=_BUFFER_SIZE)
_BUFFER_LOCK = Lock()
_SINK_ID: int | None = None

_DEFAULT_LOG_FILE_ENTRIES = 5000
_DEFAULT_LOG_FILE_COMPACT_EVERY_WRITES = 250
_LOG_FILE_SETTINGS_LOCK = Lock()
_LOG_FILE_SETTINGS_THREAD_STATE = local()
_LOG_FILE_SETTINGS_INITIALIZED = False
_LOG_FILE_MAX_ENTRIES = _DEFAULT_LOG_FILE_ENTRIES
_LOG_FILE_COMPACT_EVERY_WRITES = _DEFAULT_LOG_FILE_COMPACT_EVERY_WRITES
_LOG_FILE_APPENDS_SINCE_COMPACT = 0
_LOG_FILE_ENABLED = True
_LOG_FILE_PATH = Path(get_database_dir()) / "system_logs.jsonl"
_LOG_FILE_LOCK_TIMEOUT = 5.0
_LOG_FILE_COMPACTION_COUNTER_LOCK = Lock()

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
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return is_truthy(value)


def _extract_extra(extra: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in _EXTRA_FIELDS:
        if key not in extra:
            continue
        val = extra.get(key)
        if key in {"org_id", "user_id"}:
            payload[key] = _coerce_optional_int(val)
        else:
            payload[key] = val if val is not None else None
    return payload


def _log_file_settings_init_active() -> bool:
    return bool(getattr(_LOG_FILE_SETTINGS_THREAD_STATE, "initializing", False))


def _init_log_file_settings() -> None:
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
def _log_file_lock(timeout: float | None = None):
    _init_log_file_settings()
    timeout_seconds = _LOG_FILE_LOCK_TIMEOUT if timeout is None else max(0.0, float(timeout))
    lock_path = _LOG_FILE_PATH.with_suffix(_LOG_FILE_PATH.suffix + ".lock")
    lock_fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        if _HAS_FCNTL:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - start_time > timeout_seconds:
                        raise RuntimeError(f"Failed to acquire system log lock within {timeout_seconds}s")
                    time.sleep(0.05)
        else:
            while True:
                try:
                    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    break
                except FileExistsError:
                    try:
                        lock_stat = os.stat(lock_path)
                        if time.time() - lock_stat.st_mtime > timeout_seconds * 2:
                            os.unlink(lock_path)
                            continue
                    except (OSError, FileNotFoundError):
                        pass
                    if time.time() - start_time > timeout_seconds:
                        raise RuntimeError(f"Failed to acquire system log lock within {timeout_seconds}s")
                    time.sleep(0.05)
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
    timestamp = _coerce_timestamp(value)
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _emit_internal_diagnostic(message: str) -> None:
    # Must not use Loguru here: this function is called from inside a Loguru sink.
    with suppress(Exception):
        stream = sys.__stderr__ or sys.stderr
        stream.write(message.rstrip() + "\n")


def _sort_timestamp_value(entry: dict[str, Any]) -> float:
    timestamp = _normalize_timestamp(entry.get("timestamp"))
    if timestamp is None:
        return float("-inf")
    return timestamp.timestamp()


def _should_compact_after_append() -> bool:
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
        _emit_internal_diagnostic(f"system_log_buffer append failed: {exc}")


def _read_log_file_entries() -> list[dict[str, Any]]:
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


def _log_sink(message: Any) -> None:
    record = message.record
    extra = _extract_extra(record.get("extra", {}))
    entry = {
        "timestamp": record.get("time"),
        "level": record.get("level").name if record.get("level") else None,
        "message": record.get("message"),
        "logger": record.get("name"),
        "module": record.get("module"),
        "function": record.get("function"),
        "line": record.get("line"),
        **extra,
    }
    with _BUFFER_LOCK:
        _BUFFER.append(entry)
    _append_log_file(entry)


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
    if _SINK_ID is not None and _sink_still_present(_SINK_ID):
        return
    _SINK_ID = logger.add(
        _log_sink,
        level=os.getenv("SYSTEM_LOG_LEVEL", "DEBUG"),
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )


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
    if not entries:
        with _BUFFER_LOCK:
            entries = list(_BUFFER)

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
