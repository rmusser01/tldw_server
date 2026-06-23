"""Session-level domain logic for Meetings."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase

ALLOWED_STATUS_TRANSITIONS = {
    "scheduled": {"live", "processing", "failed"},
    "live": {"processing", "completed", "failed"},
    "processing": {"completed", "failed"},
}


class MeetingSessionService:
    """High-level operations for meeting sessions."""

    def __init__(self, db: MeetingsDatabase) -> None:
        self._db = db

    def create_session(
        self,
        *,
        title: str,
        meeting_type: str,
        source_type: str = "upload",
        language: str | None = None,
        template_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_id = self._db.create_session(
            title=title,
            meeting_type=meeting_type,
            source_type=source_type,
            language=language,
            template_id=template_id,
            metadata=metadata,
        )
        return self.get_session(session_id=session_id)

    def get_session(self, *, session_id: str) -> dict[str, Any]:
        row = self._db.get_session(session_id=session_id)
        if row is None:
            raise KeyError(f"meeting session not found: {session_id}")
        return row

    def list_sessions(self, *, status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return self._db.list_sessions(status=status, limit=limit, offset=offset)

    def transition(self, *, session_id: str, to_status: str) -> dict[str, Any]:
        current = self.get_session(session_id=session_id)
        current_status = str(current.get("status") or "").strip().lower()
        target_status = str(to_status).strip().lower()

        if current_status == target_status:
            return current

        allowed_targets = ALLOWED_STATUS_TRANSITIONS.get(current_status, set())
        if target_status not in allowed_targets:
            raise ValueError(
                f"Invalid meeting session status transition: {current_status!r} -> {target_status!r}"
            )

        updated = self._db.update_session_status(
            session_id=session_id,
            status=target_status,
            expected_status=current_status,
        )
        if not updated:
            latest = self._db.get_session(session_id=session_id)
            if latest is None:
                raise KeyError(f"meeting session not found: {session_id}")
            latest_status = str(latest.get("status") or "").strip().lower()
            if latest_status == target_status:
                return latest
            raise ValueError(
                "Meeting session status changed concurrently: "
                f"{current_status!r} -> {latest_status!r}; requested {target_status!r}"
            )
        return self.get_session(session_id=session_id)
