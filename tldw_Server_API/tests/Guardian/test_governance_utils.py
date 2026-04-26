"""
test_governance_utils.py

Tests for is_schedule_active() and chat_type_matches() pure utility functions.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import tldw_Server_API.app.core.Moderation.governance_utils as governance_module
from tldw_Server_API.app.core.Moderation.governance_utils import (
    chat_type_matches,
    is_schedule_active,
)


# ── is_schedule_active ─────────────────────────────────────────


class TestScheduleAllNoneAlwaysActive:
    def test_all_none(self):
        assert is_schedule_active(None, None, None, None) is True

    def test_all_empty_strings(self):
        assert is_schedule_active("", "", "", "") is True

    def test_mixed_none_and_empty(self):
        assert is_schedule_active(None, "", None, "") is True


class TestScheduleDayOfWeek:
    def test_matching_day(self):
        """If today's day is in the schedule, should be active."""
        now = datetime.now(ZoneInfo("UTC"))
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        today_name = day_names[now.weekday()]
        assert is_schedule_active(None, None, today_name, "UTC") is True

    def test_non_matching_day(self):
        """If today's day is NOT in the schedule, should be inactive."""
        now = datetime.now(ZoneInfo("UTC"))
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        today_idx = now.weekday()
        other_day = day_names[(today_idx + 3) % 7]
        assert is_schedule_active(None, None, other_day, "UTC") is False

    def test_multiple_days_with_today(self):
        now = datetime.now(ZoneInfo("UTC"))
        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        today_name = day_names[now.weekday()]
        schedule = f"{today_name},sat,sun"
        assert is_schedule_active(None, None, schedule, "UTC") is True

    def test_full_day_names(self):
        now = datetime.now(ZoneInfo("UTC"))
        full_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        today_name = full_names[now.weekday()]
        assert is_schedule_active(None, None, today_name, "UTC") is True


class TestScheduleTimeOfDay:
    def test_within_normal_range(self):
        """If current time is within start-end, should be active."""
        now = datetime.now(ZoneInfo("UTC"))
        # Create a range that includes current time
        start_min = max(0, now.hour * 60 + now.minute - 30)
        end_min = min(1439, now.hour * 60 + now.minute + 30)
        start_str = f"{start_min // 60:02d}:{start_min % 60:02d}"
        end_str = f"{end_min // 60:02d}:{end_min % 60:02d}"
        assert is_schedule_active(start_str, end_str, None, "UTC") is True

    def test_outside_normal_range(self):
        """If current time is outside start-end, should be inactive."""
        now = datetime.now(ZoneInfo("UTC"))
        current_min = now.hour * 60 + now.minute
        # Create a range that excludes current time (2 hours from now, 1 hour window)
        start_min = (current_min + 120) % 1440
        end_min = (current_min + 180) % 1440
        # Only test non-overnight case
        if start_min < end_min:
            start_str = f"{start_min // 60:02d}:{start_min % 60:02d}"
            end_str = f"{end_min // 60:02d}:{end_min % 60:02d}"
            assert is_schedule_active(start_str, end_str, None, "UTC") is False

    def test_overnight_range_active(self):
        """Overnight range (e.g. 22:00-06:00) active at 23:00."""
        with patch("tldw_Server_API.app.core.Moderation.governance_utils.datetime") as mock_dt:
            mock_now = datetime(2026, 2, 7, 23, 0, tzinfo=ZoneInfo("UTC"))
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_schedule_active("22:00", "06:00", None, "UTC") is True

    def test_overnight_range_active_early_morning(self):
        """Overnight range (e.g. 22:00-06:00) active at 03:00."""
        with patch("tldw_Server_API.app.core.Moderation.governance_utils.datetime") as mock_dt:
            mock_now = datetime(2026, 2, 7, 3, 0, tzinfo=ZoneInfo("UTC"))
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_schedule_active("22:00", "06:00", None, "UTC") is True

    def test_overnight_range_inactive_midday(self):
        """Overnight range (e.g. 22:00-06:00) inactive at 12:00."""
        with patch("tldw_Server_API.app.core.Moderation.governance_utils.datetime") as mock_dt:
            mock_now = datetime(2026, 2, 7, 12, 0, tzinfo=ZoneInfo("UTC"))
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert is_schedule_active("22:00", "06:00", None, "UTC") is False


class TestScheduleTimezone:
    def test_invalid_timezone_falls_back_to_utc(self):
        """Invalid timezone should fail-open (treat as active via UTC)."""
        assert is_schedule_active(None, None, None, "Invalid/Timezone") is True

    def test_invalid_timezone_log_sanitizes_input_details(self):
        messages: list[str] = []
        sink_id = governance_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            is_schedule_active(None, None, "mon", "Private/Timezone")
        finally:
            governance_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "Invalid schedule_timezone" in joined
        assert "Private/Timezone" not in joined

    def test_valid_timezone(self):
        """Valid timezone should be used without error."""
        result = is_schedule_active(None, None, None, "America/New_York")
        assert result is True


class TestScheduleMalformed:
    def test_malformed_time_fails_open(self):
        """Malformed HH:MM should fail-open (active)."""
        assert is_schedule_active("bad", "bad", None, "UTC") is True

    def test_partial_time_fails_open(self):
        """Only start set, end malformed should fail-open."""
        assert is_schedule_active("09:00", "bad", None, "UTC") is True

    def test_only_start_time(self):
        """Only start set, no end -> fail-open."""
        assert is_schedule_active("09:00", None, None, "UTC") is True

    def test_schedule_evaluation_log_sanitizes_exception_details(self, monkeypatch):
        def fail_get_tz(_timezone):
            raise RuntimeError("schedule backend failed at /private/governance-schedule.db")

        monkeypatch.setattr(governance_module, "_get_tz", fail_get_tz)

        messages: list[str] = []
        sink_id = governance_module.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            assert is_schedule_active("09:00", "17:00", None, "UTC") is True
        finally:
            governance_module.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "Schedule evaluation error, failing open" in joined
        assert "governance-schedule.db" not in joined


# ── chat_type_matches ──────────────────────────────────────────


class TestChatTypeAll:
    def test_scope_all_matches_anything(self):
        assert chat_type_matches("all", "regular") is True
        assert chat_type_matches("all", "character") is True
        assert chat_type_matches("all", "rag") is True

    def test_scope_none_matches_anything(self):
        assert chat_type_matches(None, "regular") is True

    def test_scope_empty_matches_anything(self):
        assert chat_type_matches("", "regular") is True


class TestChatTypeNoneDefault:
    def test_none_chat_type_defaults_to_regular(self):
        assert chat_type_matches("regular", None) is True
        assert chat_type_matches("character", None) is False


class TestChatTypeCommaSeparated:
    def test_single_scope_match(self):
        assert chat_type_matches("regular", "regular") is True

    def test_single_scope_no_match(self):
        assert chat_type_matches("character", "regular") is False

    def test_comma_separated_match(self):
        assert chat_type_matches("regular,character", "character") is True

    def test_comma_separated_no_match(self):
        assert chat_type_matches("regular,character", "rag") is False


class TestChatTypeCaseInsensitive:
    def test_case_insensitive_scope(self):
        assert chat_type_matches("Regular", "regular") is True

    def test_case_insensitive_chat_type(self):
        assert chat_type_matches("regular", "REGULAR") is True

    def test_case_insensitive_both(self):
        assert chat_type_matches("CHARACTER,RAG", "rag") is True
