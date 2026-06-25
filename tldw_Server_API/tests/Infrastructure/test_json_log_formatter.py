"""Tests for the JSON log formatter."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Logging.json_log_formatter import json_log_format


def _make_record(**overrides) -> dict:
    """Build a minimal loguru-style record dict."""
    base = {
        "time": datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc),
        "level": type("Level", (), {"name": "INFO"})(),
        "message": "hello world",
        "module": "test_module",
        "function": "test_func",
        "line": 42,
        "exception": None,
        "extra": {},
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestJsonLogFormat:
    def test_basic_format(self) -> None:
        record = _make_record()
        line = json_log_format(record)
        assert line.endswith("\n")
        parsed = json.loads(line)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_func"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_exception_included(self) -> None:
        record = _make_record(exception="ValueError: bad input")
        parsed = json.loads(json_log_format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_extra_fields(self) -> None:
        record = _make_record(extra={"request_id": "abc-123", "user_id": 7})
        parsed = json.loads(json_log_format(record))
        assert parsed["extra"]["request_id"] == "abc-123"
        assert parsed["extra"]["user_id"] == "7"  # stringified

    def test_no_extra_key_when_empty(self) -> None:
        record = _make_record(extra={})
        parsed = json.loads(json_log_format(record))
        assert "extra" not in parsed

    def test_timestamp_format(self) -> None:
        record = _make_record()
        parsed = json.loads(json_log_format(record))
        ts = parsed["timestamp"]
        # Should be ISO-8601-ish with microseconds and Z suffix
        assert ts.endswith("Z")
        assert "T" in ts

    def test_timestamp_is_converted_to_utc_before_z_suffix(self) -> None:
        record = _make_record(
            time=datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone(timedelta(hours=-7)))
        )

        parsed = json.loads(json_log_format(record))

        assert parsed["timestamp"] == "2026-03-14T19:00:00.000000Z"
