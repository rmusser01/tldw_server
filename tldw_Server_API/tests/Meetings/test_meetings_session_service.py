from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase
from tldw_Server_API.app.core.Meetings.session_service import MeetingSessionService


pytestmark = pytest.mark.unit


@pytest.fixture()
def session_service(tmp_path):
    db = MeetingsDatabase(db_path=tmp_path / "Media_DB_v2.db", client_id="tester", user_id="1")
    service = MeetingSessionService(db=db)
    try:
        yield service
    finally:
        db.close_connection()


def test_session_state_machine_blocks_invalid_transition(session_service):
    created = session_service.create_session(title="Standup", meeting_type="standup")
    with pytest.raises(ValueError):
        session_service.transition(session_id=created["id"], to_status="completed")


def test_session_state_machine_allows_valid_transition(session_service):
    created = session_service.create_session(title="Sprint Planning", meeting_type="planning")
    live = session_service.transition(session_id=created["id"], to_status="live")
    assert live["status"] == "live"
    processing = session_service.transition(session_id=created["id"], to_status="processing")
    assert processing["status"] == "processing"
    completed = session_service.transition(session_id=created["id"], to_status="completed")
    assert completed["status"] == "completed"


def test_session_transition_raises_for_missing_session(session_service):
    with pytest.raises(KeyError):
        session_service.transition(session_id="sess_missing", to_status="live")


def test_session_transition_rejects_stale_current_status(
    session_service: MeetingSessionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = session_service.create_session(title="Race Review", meeting_type="standup")
    real_update = session_service._db.update_session_status

    def racing_update(
        *,
        session_id: str,
        status: str,
        user_id: int | str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        if expected_status == "scheduled":
            assert real_update(session_id=session_id, status="processing", user_id=user_id) is True
        return real_update(
            session_id=session_id,
            status=status,
            user_id=user_id,
            expected_status=expected_status,
        )

    monkeypatch.setattr(session_service._db, "update_session_status", racing_update)

    with pytest.raises(ValueError, match="changed concurrently"):
        session_service.transition(session_id=created["id"], to_status="live")

    row = session_service.get_session(session_id=created["id"])
    assert row["status"] == "processing"
