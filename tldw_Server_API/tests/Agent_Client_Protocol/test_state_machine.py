"""Tests for ACP state machine with Phase/Activity separation."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.state_machine import (
    SessionPhase,
    SessionActivity,
    is_sticky,
    is_platform_set,
    validate_activity,
    should_update_activity,
    ActivityDetail,
    migrate_legacy_status,
)


# ── Enum membership ──────────────────────────────────────────────────

class TestSessionPhaseEnum:
    def test_all_phases_exist(self):
        expected = {"queued", "provisioning", "starting", "running",
                    "stopping", "stopped", "error"}
        actual = {p.value for p in SessionPhase}
        assert actual == expected

    def test_phase_is_str(self):
        assert isinstance(SessionPhase.RUNNING, str)
        assert SessionPhase.RUNNING == "running"


class TestSessionActivityEnum:
    def test_all_activities_exist(self):
        expected = {"idle", "thinking", "executing", "waiting_for_input",
                    "blocked", "completed", "limits_exceeded",
                    "stalled", "offline"}
        actual = {a.value for a in SessionActivity}
        assert actual == expected

    def test_activity_is_str(self):
        assert isinstance(SessionActivity.IDLE, str)
        assert SessionActivity.IDLE == "idle"


# ── Sticky / platform-set predicates ─────────────────────────────────

class TestIsSticky:
    @pytest.mark.parametrize("activity", [
        SessionActivity.WAITING_FOR_INPUT,
        SessionActivity.BLOCKED,
        SessionActivity.COMPLETED,
        SessionActivity.LIMITS_EXCEEDED,
    ])
    def test_sticky_activities(self, activity):
        assert is_sticky(activity) is True

    @pytest.mark.parametrize("activity", [
        SessionActivity.IDLE,
        SessionActivity.THINKING,
        SessionActivity.EXECUTING,
        SessionActivity.STALLED,
        SessionActivity.OFFLINE,
    ])
    def test_non_sticky_activities(self, activity):
        assert is_sticky(activity) is False


class TestIsPlatformSet:
    @pytest.mark.parametrize("activity", [
        SessionActivity.STALLED,
        SessionActivity.OFFLINE,
    ])
    def test_platform_set_activities(self, activity):
        assert is_platform_set(activity) is True

    @pytest.mark.parametrize("activity", [
        SessionActivity.IDLE,
        SessionActivity.THINKING,
        SessionActivity.EXECUTING,
        SessionActivity.WAITING_FOR_INPUT,
        SessionActivity.BLOCKED,
        SessionActivity.COMPLETED,
        SessionActivity.LIMITS_EXCEEDED,
    ])
    def test_non_platform_set_activities(self, activity):
        assert is_platform_set(activity) is False


# ── validate_activity ─────────────────────────────────────────────────

class TestValidateActivity:
    def test_running_phase_allows_any_activity(self):
        for act in SessionActivity:
            validate_activity(SessionPhase.RUNNING, act)  # no raise

    def test_none_activity_allowed_for_any_phase(self):
        for phase in SessionPhase:
            validate_activity(phase, None)  # no raise

    @pytest.mark.parametrize("phase", [
        SessionPhase.QUEUED,
        SessionPhase.PROVISIONING,
        SessionPhase.STARTING,
        SessionPhase.STOPPING,
        SessionPhase.STOPPED,
        SessionPhase.ERROR,
    ])
    def test_non_running_phase_rejects_activity(self, phase):
        with pytest.raises(ValueError, match="only valid when phase is 'running'"):
            validate_activity(phase, SessionActivity.IDLE)


# ── should_update_activity ────────────────────────────────────────────

class TestShouldUpdateActivity:
    def test_new_work_always_updates(self):
        """New work events should clear even sticky activities."""
        assert should_update_activity(
            SessionActivity.BLOCKED, SessionActivity.THINKING,
            is_new_work=True,
        ) is True

    def test_none_current_always_updates(self):
        assert should_update_activity(
            None, SessionActivity.THINKING,
        ) is True

    def test_tool_start_clears_waiting_for_input(self):
        assert should_update_activity(
            SessionActivity.WAITING_FOR_INPUT, SessionActivity.EXECUTING,
            is_tool_start=True,
        ) is True

    def test_tool_start_does_not_clear_other_sticky(self):
        assert should_update_activity(
            SessionActivity.BLOCKED, SessionActivity.EXECUTING,
            is_tool_start=True,
        ) is False

    def test_normal_event_blocked_by_sticky(self):
        for sticky in [SessionActivity.WAITING_FOR_INPUT,
                       SessionActivity.BLOCKED,
                       SessionActivity.COMPLETED,
                       SessionActivity.LIMITS_EXCEEDED]:
            assert should_update_activity(
                sticky, SessionActivity.THINKING,
            ) is False

    def test_normal_event_updates_non_sticky(self):
        assert should_update_activity(
            SessionActivity.IDLE, SessionActivity.THINKING,
        ) is True


# ── ActivityDetail ────────────────────────────────────────────────────

class TestActivityDetail:
    def test_defaults(self):
        detail = ActivityDetail()
        assert detail.tool_name is None
        assert detail.message is None
        assert detail.task_summary is None

    def test_roundtrip_dict(self):
        original = ActivityDetail(
            tool_name="bash", message="running ls", task_summary="list files"
        )
        rebuilt = ActivityDetail.from_dict(original.to_dict())
        assert rebuilt.tool_name == "bash"
        assert rebuilt.message == "running ls"
        assert rebuilt.task_summary == "list files"

    def test_roundtrip_json(self):
        original = ActivityDetail(
            tool_name="python", message="exec script", task_summary="run test"
        )
        rebuilt = ActivityDetail.from_json(original.to_json())
        assert rebuilt.tool_name == "python"
        assert rebuilt.message == "exec script"
        assert rebuilt.task_summary == "run test"

    def test_from_dict_none_returns_empty(self):
        assert ActivityDetail.from_dict(None) == ActivityDetail()

    def test_from_dict_empty_returns_empty(self):
        assert ActivityDetail.from_dict({}) == ActivityDetail()

    def test_from_json_none_returns_empty(self):
        assert ActivityDetail.from_json(None) == ActivityDetail()

    def test_from_json_empty_string_returns_empty(self):
        assert ActivityDetail.from_json("") == ActivityDetail()


# ── migrate_legacy_status ─────────────────────────────────────────────

class TestMigrateLegacyStatus:
    def test_active_maps_to_running_idle(self):
        phase, activity = migrate_legacy_status("active")
        assert phase == SessionPhase.RUNNING
        assert activity == SessionActivity.IDLE

    def test_closed_maps_to_stopped_none(self):
        phase, activity = migrate_legacy_status("closed")
        assert phase == SessionPhase.STOPPED
        assert activity is None

    def test_error_maps_to_error_none(self):
        phase, activity = migrate_legacy_status("error")
        assert phase == SessionPhase.ERROR
        assert activity is None

    def test_unknown_status_defaults_to_error(self):
        phase, activity = migrate_legacy_status("banana")
        assert phase == SessionPhase.ERROR
        assert activity is None
