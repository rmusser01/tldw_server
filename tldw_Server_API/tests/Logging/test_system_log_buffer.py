from __future__ import annotations

import configparser
import importlib
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest


def _reload_log_buffer():
    import tldw_Server_API.app.core.Logging.system_log_buffer as log_buffer

    return importlib.reload(log_buffer)


class _CapturingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str, *args, **_kwargs) -> None:
        if args:
            message = message.format(*args)
        self.messages.append(message)


def test_system_log_file_query_reads_shared_file(tmp_path, monkeypatch):
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


def test_append_log_file_skips_recursive_append_during_settings_init(tmp_path, monkeypatch):
    log_path = tmp_path / "system_logs.jsonl"
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(log_path))
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "100")
    monkeypatch.delenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", raising=False)

    log_buffer = _reload_log_buffer()

    import tldw_Server_API.app.core.config as config_mod

    nested_calls: list[int] = []

    def _fake_load_comprehensive_config():
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


def test_append_log_file_compacts_periodically(tmp_path, monkeypatch):
    log_path = tmp_path / "system_logs.jsonl"
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(log_path))
    # Runtime minimum is 100 entries, so use that floor for deterministic assertions.
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "100")
    monkeypatch.setenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", "3")

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

    lines_after_overflow = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_overflow) == 101

    log_buffer._append_log_file(
        {
            "timestamp": base_time + timedelta(minutes=101),
            "level": "INFO",
            "message": "m101",
        }
    )

    lines_after_compact = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_compact) == 100
    assert all('"message": "m0"' not in line for line in lines_after_compact)
    assert all('"message": "m1"' not in line for line in lines_after_compact)
    assert any('"message": "m2"' in line for line in lines_after_compact)
    assert any('"message": "m101"' in line for line in lines_after_compact)


def test_append_log_file_failure_avoids_sink_reentry(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEM_LOG_FILE_ENABLED", "true")
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system_logs.jsonl"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_MAX_ENTRIES", "100")
    monkeypatch.setenv("SYSTEM_LOG_FILE_COMPACT_EVERY_WRITES", "10")
    log_buffer = _reload_log_buffer()

    called: list[str] = []
    monkeypatch.setattr(log_buffer, "_emit_internal_diagnostic", lambda message: called.append(message))

    @contextmanager
    def _failing_lock(_timeout=None):
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
    assert "write denied" in called[0]


def test_system_log_settings_config_read_failure_log_is_sanitized(monkeypatch):
    log_buffer = _reload_log_buffer()

    import tldw_Server_API.app.core.config as config_mod

    def _raise_config_error():
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


def test_read_log_file_failure_log_is_sanitized(monkeypatch):
    log_buffer = _reload_log_buffer()

    class FailingLogPath:
        def exists(self):
            return True

        def read_text(self, encoding):
            raise OSError("system log read failed at /private/system_logs.jsonl")

    @contextmanager
    def _noop_lock(_timeout=None):
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


def test_query_system_logs_handles_naive_start_with_aware_entries(monkeypatch):
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


def test_query_system_logs_sorts_with_malformed_timestamps(monkeypatch):
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


def test_log_file_lock_uses_runtime_timeout_from_env(monkeypatch, tmp_path):
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
