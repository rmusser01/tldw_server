from __future__ import annotations

import configparser
import importlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _reload_log_buffer() -> ModuleType:
    from loguru import logger

    sink_attr = "_tldw_system_log_buffer_sink_id"
    sink_id = getattr(logger, sink_attr, None)
    if sink_id is not None:
        with suppress(Exception):
            logger.remove(sink_id)
        with suppress(Exception):
            delattr(logger, sink_attr)

    import tldw_Server_API.app.core.Logging.system_log_buffer as log_buffer

    return importlib.reload(log_buffer)


def _message_for_log_sink(
    message: str,
    *,
    extra: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        record={
            "time": datetime.now(timezone.utc),
            "level": SimpleNamespace(name="INFO"),
            "message": message,
            "name": "test.logger",
            "module": "test_module",
            "function": "test_function",
            "line": 10,
            "extra": dict(extra or {}),
        }
    )


class _CapturingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str, *args: object, **_kwargs: object) -> None:
        if args:
            message = message.format(*args)
        self.messages.append(message)


@pytest.mark.unit
def test_system_log_file_query_reads_shared_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "system_logs.jsonl"
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(log_path))
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "10")

    log_buffer = _reload_log_buffer()

    entry = {
        "timestamp": datetime.now(timezone.utc),
        "level": "INFO",
        "message": "file-backed entry",
        "logger": "test",
        "module": "test_module",
        "function": "test_fn",
        "line": 1,
    }

    log_buffer._append_log_file(entry)

    items, total = log_buffer.query_system_logs(limit=10, offset=0)
    assert total >= 1
    assert any(item.get("message") == "file-backed entry" for item in items)
    assert log_path.exists()


@pytest.mark.unit
def test_log_sink_redacts_secrets_before_buffer_and_file_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_buffer = _reload_log_buffer()
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(log_buffer, "_enqueue_log_file", lambda entry: captured.append(dict(entry)))

    secret_message = (
        "provider failed api_key=sk-secretvalue123 password:open-sesame "
        "Authorization: Bearer secret-token-123"
    )
    secret_extra = {
        "api_key": "sk-extra-secretvalue123",
        "db_password": "extra-open-sesame",
        "access_token": "extra-token-secret",
        "request_id": "req-secret-test",
    }

    with log_buffer._BUFFER_LOCK:
        log_buffer._BUFFER.clear()
    log_buffer._log_sink(_message_for_log_sink(secret_message, extra=secret_extra))

    with log_buffer._BUFFER_LOCK:
        buffered = list(log_buffer._BUFFER)
    serialized = "\n".join(str(entry) for entry in buffered + captured)
    assert "sk-secretvalue123" not in serialized
    assert "open-sesame" not in serialized
    assert "secret-token-123" not in serialized
    assert "sk-extra-secretvalue123" not in serialized
    assert "extra-open-sesame" not in serialized
    assert "extra-token-secret" not in serialized
    assert "***REDACTED***" in serialized
    assert buffered[0]["api_key"] == "***REDACTED***"
    assert buffered[0]["db_password"] == "***REDACTED***"
    assert buffered[0]["access_token"] == "***REDACTED***"
    assert buffered[0]["request_id"] == "req-secret-test"


@pytest.mark.unit
def test_log_sink_returns_before_log_file_append_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    log_buffer = _reload_log_buffer()

    def _slow_append(_entry: dict[str, object]) -> None:
        time.sleep(0.2)

    monkeypatch.setattr(log_buffer, "_append_log_file", _slow_append)

    started = time.perf_counter()
    log_buffer._log_sink(_message_for_log_sink("non-blocking sink check"))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05


@pytest.mark.unit
def test_log_sink_swallows_enqueue_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    log_buffer = _reload_log_buffer()
    diagnostics: list[str] = []

    def _raise_enqueue(_entry: dict[str, object]) -> None:
        raise RuntimeError("worker start failed")

    monkeypatch.setattr(log_buffer, "_enqueue_log_file", _raise_enqueue)
    monkeypatch.setattr(
        log_buffer,
        "_emit_internal_diagnostic",
        lambda message: diagnostics.append(message),
    )

    log_buffer._log_sink(_message_for_log_sink("sink failure remains internal"))

    assert any(
        "system_log_buffer sink enqueue failed: RuntimeError" in message
        for message in diagnostics
    )
    assert all("worker start failed" not in message for message in diagnostics)


@pytest.mark.unit
def test_append_log_file_skips_recursive_append_during_settings_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "system_logs.jsonl"
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(log_path))
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "100")
    monkeypatch.delenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", raising=False)

    log_buffer = _reload_log_buffer()

    import tldw_Server_API.app.core.config as config_mod

    nested_calls: list[int] = []

    def _fake_load_comprehensive_config() -> configparser.ConfigParser:
        nested_calls.append(1)
        log_buffer._append_log_file(
            {
                "timestamp": datetime.now(timezone.utc),
                "level": "INFO",
                "message": "nested",
            }
        )
        parser = configparser.ConfigParser()
        parser.add_section("Logging")
        parser.set("Logging", "system_log_file_compact_every_writes", "10")
        return parser

    monkeypatch.setattr(config_mod, "load_comprehensive_config", _fake_load_comprehensive_config, raising=True)

    log_buffer._append_log_file(
        {
            "timestamp": datetime.now(timezone.utc),
            "level": "INFO",
            "message": "outer",
        }
    )

    assert nested_calls
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert any('"message": "outer"' in line for line in lines)


@pytest.mark.unit
def test_malformed_system_log_buffer_size_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_BUFFER_SIZE", "not-an-int")

    log_buffer = _reload_log_buffer()

    assert log_buffer._BUFFER.maxlen == log_buffer._DEFAULT_BUFFER_SIZE


@pytest.mark.unit
def test_invalid_system_log_level_falls_back_to_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_LEVEL", "definitely-not-a-level")
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "false")
    log_buffer = _reload_log_buffer()
    added: list[str] = []

    class _Logger:
        def add(self, _sink: object, *, level: str, **_kwargs: object) -> int:
            added.append(level)
            return 99

    monkeypatch.setattr(log_buffer, "logger", _Logger())
    monkeypatch.setattr(log_buffer, "_init_log_file_settings", lambda: None)

    log_buffer.ensure_system_log_buffer()

    assert added == ["DEBUG"]


@pytest.mark.unit
def test_ensure_system_log_buffer_installs_one_sink_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "false")
    log_buffer = _reload_log_buffer()
    add_calls: list[int] = []
    start = threading.Barrier(8)

    class _Logger:
        def add(self, *_args: object, **_kwargs: object) -> int:
            time.sleep(0.05)
            add_calls.append(1)
            return len(add_calls)

    monkeypatch.setattr(log_buffer, "logger", _Logger())
    monkeypatch.setattr(log_buffer, "_init_log_file_settings", lambda: None)
    monkeypatch.setattr(log_buffer, "_sink_still_present", lambda _sink_id: True)
    monkeypatch.setattr(log_buffer, "_SINK_ID", None)

    def _install() -> None:
        start.wait(timeout=1)
        log_buffer.ensure_system_log_buffer()

    threads = [threading.Thread(target=_install) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert len(add_calls) == 1


@pytest.mark.unit
def test_reload_reuses_log_file_queue_and_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from loguru import logger

    log_buffer = _reload_log_buffer()
    queue = log_buffer._LOG_FILE_QUEUE

    class _Worker:
        def is_alive(self) -> bool:
            return True

    worker = _Worker()
    monkeypatch.setattr(logger, log_buffer._LOG_FILE_QUEUE_ATTR, queue, raising=False)
    monkeypatch.setattr(logger, log_buffer._LOG_FILE_WORKER_ATTR, worker, raising=False)

    reloaded = importlib.reload(log_buffer)

    assert reloaded._LOG_FILE_QUEUE is queue
    assert reloaded._LOG_FILE_WORKER is worker


@pytest.mark.unit
def test_append_log_file_compacts_periodically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "system_logs.jsonl"
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(log_path))
    # Runtime minimum is 100 entries, so use that floor for deterministic assertions.
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "100")
    monkeypatch.setenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", "1")

    log_buffer = _reload_log_buffer()
    base_time = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)

    for idx in range(101):
        log_buffer._append_log_file(
            {
                "timestamp": base_time + timedelta(minutes=idx),
                "level": "INFO",
                "message": f"m{idx}",
            }
        )

    lines_after_compact = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_compact) == 100
    assert all('"message": "m0"' not in line for line in lines_after_compact)
    assert any('"message": "m100"' in line for line in lines_after_compact)


@pytest.mark.unit
def test_append_log_file_failure_avoids_sink_reentry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system_logs.jsonl"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "100")
    monkeypatch.setenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", "10")
    log_buffer = _reload_log_buffer()

    called: list[str] = []
    monkeypatch.setattr(log_buffer, "_emit_internal_diagnostic", lambda message: called.append(message))

    @contextmanager
    def _failing_lock(_timeout: object = None) -> Iterator[None]:
        raise PermissionError("write denied")
        yield

    monkeypatch.setattr(log_buffer, "_log_file_lock", _failing_lock)

    log_buffer._append_log_file(
        {
            "timestamp": datetime.now(timezone.utc),
            "level": "INFO",
            "message": "should not recurse",
        }
    )

    assert called
    assert "system_log_buffer append failed: PermissionError" in called[0]
    assert "write denied" not in called[0]


@pytest.mark.unit
def test_system_log_settings_config_read_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_buffer = _reload_log_buffer()

    import tldw_Server_API.app.core.config as config_mod

    def _raise_config_error() -> None:
        raise RuntimeError("config read failed at /private/system-log-config.ini")

    monkeypatch.delenv("SYSTEM_LOG_FILE_PATH", raising=False)
    monkeypatch.delenv("SYSTEM_LOG_FILE_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", raising=False)
    monkeypatch.setattr(config_mod, "load_comprehensive_config", _raise_config_error, raising=True)
    capture = _CapturingLogger()
    monkeypatch.setattr(log_buffer, "logger", capture)

    log_buffer._init_log_file_settings()

    joined = "\n".join(capture.messages)
    assert "System log settings config read failed" in joined
    assert "config read failed at" not in joined
    assert "/private/system-log-config.ini" not in joined


@pytest.mark.unit
def test_read_log_file_failure_log_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    log_buffer = _reload_log_buffer()

    class FailingLogPath:
        def exists(self) -> bool:
            return True

        def read_text(self, encoding: str) -> str:
            raise OSError("system log read failed at /private/system_logs.jsonl")

    @contextmanager
    def _noop_lock(_timeout: object = None) -> Iterator[None]:
        yield

    capture = _CapturingLogger()
    monkeypatch.setattr(log_buffer, "logger", capture)
    monkeypatch.setattr(log_buffer, "_LOG_FILE_SETTINGS_INITIALIZED", True)
    monkeypatch.setattr(log_buffer, "_LOG_FILE_ENABLED", True)
    monkeypatch.setattr(log_buffer, "_LOG_FILE_PATH", FailingLogPath())
    monkeypatch.setattr(log_buffer, "_log_file_lock", _noop_lock)

    entries = log_buffer._read_log_file_entries()

    joined = "\n".join(capture.messages)
    assert entries == []
    assert "Failed to read system log file" in joined
    assert "system log read failed" not in joined
    assert "/private/system_logs.jsonl" not in joined


@pytest.mark.unit
def test_query_system_logs_handles_naive_start_with_aware_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "false")
    log_buffer = _reload_log_buffer()

    now_aware = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    target_message = "aware-entry-isolated"
    with log_buffer._BUFFER_LOCK:
        log_buffer._BUFFER.clear()
        log_buffer._BUFFER.append(
            {
                "timestamp": now_aware,
                "level": "INFO",
                "message": target_message,
                "logger": "test",
                "module": "test_module",
            }
        )

    items, total = log_buffer.query_system_logs(
        start=datetime(2026, 1, 1),
        query=target_message,
        limit=10,
        offset=0,
    )
    assert total == 1
    assert items[0]["message"] == target_message
    assert items[0]["timestamp"].tzinfo is not None


@pytest.mark.unit
def test_query_system_logs_sorts_with_malformed_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "false")
    log_buffer = _reload_log_buffer()

    with log_buffer._BUFFER_LOCK:
        log_buffer._BUFFER.clear()
        log_buffer._BUFFER.extend(
            [
                {
                    "timestamp": "not-a-time",
                    "level": "INFO",
                    "message": "sortcase-bad-timestamp",
                },
                {
                    "timestamp": datetime(2026, 2, 1, 11, 0, tzinfo=timezone.utc),
                    "level": "INFO",
                    "message": "sortcase-older",
                },
                {
                    "timestamp": datetime(2026, 2, 1, 12, 0),
                    "level": "INFO",
                    "message": "sortcase-newer-naive",
                },
            ]
        )

    items, total = log_buffer.query_system_logs(query="sortcase-", limit=10, offset=0)
    assert total == 3
    assert items[0]["message"] == "sortcase-newer-naive"
    assert items[-1]["message"] == "sortcase-bad-timestamp"


@pytest.mark.unit
def test_query_system_logs_dedupes_with_tenant_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_buffer = _reload_log_buffer()
    base_entry = {
        "timestamp": datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc),
        "level": "INFO",
        "message": "tenant-collision",
        "logger": "test",
        "module": "test_module",
        "function": "test_fn",
        "line": 1,
        "request_id": "same-request",
    }
    file_entry = {**base_entry, "org_id": 1, "user_id": 10}
    buffer_entry = {**base_entry, "org_id": 2, "user_id": 20}

    monkeypatch.setattr(log_buffer, "ensure_system_log_buffer", lambda: None)
    monkeypatch.setattr(log_buffer, "_read_log_file_entries", lambda: [file_entry])
    with log_buffer._BUFFER_LOCK:
        log_buffer._BUFFER.clear()
        log_buffer._BUFFER.append(buffer_entry)

    items, total = log_buffer.query_system_logs(
        query="tenant-collision",
        org_id=2,
        limit=10,
        offset=0,
    )

    assert total == 1
    assert items[0]["org_id"] == 2
    assert items[0]["user_id"] == 20


@pytest.mark.unit
def test_query_system_logs_keeps_entries_with_different_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_buffer = _reload_log_buffer()
    base_entry = {
        "timestamp": datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc),
        "level": "INFO",
        "message": "event-collision",
        "logger": "test",
        "module": "test_module",
        "function": "test_fn",
        "line": 1,
        "request_id": "same-request",
    }
    file_entry = {**base_entry, "event": "file-event"}
    buffer_entry = {**base_entry, "event": "buffer-event"}

    monkeypatch.setattr(log_buffer, "ensure_system_log_buffer", lambda: None)
    monkeypatch.setattr(log_buffer, "_read_log_file_entries", lambda: [file_entry])
    with log_buffer._BUFFER_LOCK:
        log_buffer._BUFFER.clear()
        log_buffer._BUFFER.append(buffer_entry)

    items, total = log_buffer.query_system_logs(
        query="event-collision",
        limit=10,
        offset=0,
    )

    assert total == 2
    assert {item["event"] for item in items} == {"file-event", "buffer-event"}


@pytest.mark.unit
def test_log_file_lock_uses_runtime_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "system_logs.jsonl"
    lock_path = log_path.with_suffix(log_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("locked", encoding="utf-8")

    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(log_path))
    monkeypatch.setenv("SYSTEM_LOG_FILE_LOCK_TIMEOUT", "0.2")

    log_buffer = _reload_log_buffer()
    monkeypatch.setattr(log_buffer, "_HAS_FCNTL", False)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Failed to acquire system log lock"):
        with log_buffer._log_file_lock():
            pass
    elapsed = time.monotonic() - started
    assert elapsed < 0.5
