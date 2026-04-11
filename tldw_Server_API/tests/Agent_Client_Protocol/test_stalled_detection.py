"""Tests for stalled session detection in AgentHealthMonitor."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from tldw_Server_API.app.core.Agent_Client_Protocol.health_monitor import (
    AgentHealthMonitor,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _make_session(
    session_id: str = "sess-1",
    phase: str = "running",
    activity: str | None = "thinking",
    last_activity_at: str | None = None,
) -> dict:
    """Build a minimal session dict matching ACPSessionsDB output."""
    if last_activity_at is None:
        last_activity_at = _iso(datetime.now(timezone.utc) - timedelta(minutes=10))
    return {
        "session_id": session_id,
        "phase": phase,
        "activity": activity,
        "last_activity_at": last_activity_at,
    }


class TestStalledDetection:
    """Tests for _check_stalled_sessions integration."""

    def test_stalled_detection_marks_inactive_session(self) -> None:
        """Session with stale last_activity_at is marked stalled."""
        db = MagicMock()
        stale_session = _make_session(
            session_id="sess-stale",
            activity="executing",
            last_activity_at=_iso(datetime.now(timezone.utc) - timedelta(minutes=10)),
        )
        db.list_stalled_sessions.return_value = [stale_session]
        db.mark_session_stalled.return_value = True

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 1
        db.list_stalled_sessions.assert_called_once()
        # Verify threshold_str was passed
        threshold_arg = db.list_stalled_sessions.call_args[0][0]
        # The threshold should be roughly 5 minutes ago
        threshold_dt = datetime.fromisoformat(threshold_arg)
        assert (datetime.now(timezone.utc) - threshold_dt).total_seconds() == pytest.approx(
            300, abs=5
        )
        db.mark_session_stalled.assert_called_once_with(
            "sess-stale",
            stalled_from_activity="executing",
        )

    def test_stalled_detection_skips_recent_activity(self) -> None:
        """Session with recent activity is not marked stalled."""
        db = MagicMock()
        # The DB query itself filters these out -- list_stalled_sessions
        # should return nothing if all sessions are recent.
        db.list_stalled_sessions.return_value = []

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 0
        db.mark_session_stalled.assert_not_called()

    def test_stalled_detection_skips_non_running(self) -> None:
        """Session with phase != running is not marked stalled.

        The DB query filters by phase='running', so non-running sessions
        should never appear in the results.
        """
        db = MagicMock()
        db.list_stalled_sessions.return_value = []

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 0
        db.mark_session_stalled.assert_not_called()

    def test_stalled_detection_preserves_stalled_from_activity(self) -> None:
        """Pre-stall activity is saved to stalled_from_activity."""
        db = MagicMock()
        session = _make_session(
            session_id="sess-preserve",
            activity="waiting_for_input",
        )
        db.list_stalled_sessions.return_value = [session]
        db.mark_session_stalled.return_value = True

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 1
        db.mark_session_stalled.assert_called_once_with(
            "sess-preserve",
            stalled_from_activity="waiting_for_input",
        )

    def test_stalled_detection_no_db_returns_zero(self) -> None:
        """Returns 0 when no DB configured."""
        monitor = AgentHealthMonitor(db=None, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 0

    def test_stalled_detection_handles_none_activity(self) -> None:
        """Session with activity=None still gets marked stalled."""
        db = MagicMock()
        session = _make_session(
            session_id="sess-none-activity",
            activity=None,
        )
        db.list_stalled_sessions.return_value = [session]
        db.mark_session_stalled.return_value = True

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 1
        db.mark_session_stalled.assert_called_once_with(
            "sess-none-activity",
            stalled_from_activity=None,
        )

    def test_stalled_detection_multiple_sessions(self) -> None:
        """Multiple stalled sessions are all marked."""
        db = MagicMock()
        sessions = [
            _make_session(session_id=f"sess-{i}", activity="idle")
            for i in range(3)
        ]
        db.list_stalled_sessions.return_value = sessions
        db.mark_session_stalled.return_value = True

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 3
        assert db.mark_session_stalled.call_count == 3

    def test_stalled_detection_db_query_error_returns_zero(self) -> None:
        """Returns 0 if the DB query raises an exception."""
        db = MagicMock()
        db.list_stalled_sessions.side_effect = RuntimeError("DB error")

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        assert count == 0

    def test_stalled_detection_mark_error_continues(self) -> None:
        """If marking one session fails, others still get processed."""
        db = MagicMock()
        sessions = [
            _make_session(session_id="sess-ok-1", activity="idle"),
            _make_session(session_id="sess-fail", activity="executing"),
            _make_session(session_id="sess-ok-2", activity="thinking"),
        ]
        db.list_stalled_sessions.return_value = sessions
        db.mark_session_stalled.side_effect = [
            True,
            RuntimeError("DB write error"),
            True,
        ]

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=300)
        count = monitor._check_stalled_sessions()

        # Only 2 succeeded
        assert count == 2
        assert db.mark_session_stalled.call_count == 3

    def test_custom_stall_threshold(self) -> None:
        """Custom stall_threshold_seconds is respected in threshold calculation."""
        db = MagicMock()
        db.list_stalled_sessions.return_value = []

        monitor = AgentHealthMonitor(db=db, stall_threshold_seconds=60)
        monitor._check_stalled_sessions()

        threshold_arg = db.list_stalled_sessions.call_args[0][0]
        threshold_dt = datetime.fromisoformat(threshold_arg)
        # With 60s threshold, the cutoff should be ~1 minute ago
        delta = (datetime.now(timezone.utc) - threshold_dt).total_seconds()
        assert delta == pytest.approx(60, abs=5)
