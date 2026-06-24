from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Meetings.stream_adapter import to_sse_frame


pytestmark = pytest.mark.unit


def _create_session(meetings_api_client: TestClient) -> str:
    resp = meetings_api_client.post(
        "/api/v1/meetings/sessions",
        json={"title": "SSE Session", "meeting_type": "standup"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_sse_events_streams_structured_frames(meetings_api_client: TestClient) -> None:
    session_id = _create_session(meetings_api_client)
    resp = meetings_api_client.get(f"/api/v1/meetings/sessions/{session_id}/events")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "event:" in resp.text
    assert "\"session_id\"" in resp.text


def test_sse_frame_sanitizes_control_fields() -> None:
    frame = to_sse_frame(
        {
            "id": "evt-1\nretry: 0",
            "type": "transcript.final\ndata: forged",
            "session_id": "sess_1",
            "timestamp": "2026-06-23T00:00:00+00:00",
            "data": {"text": "hello"},
        }
    )

    lines = frame.splitlines()
    assert lines[0] == "id: evt-1_retry:_0"
    assert lines[1] == "event: transcript.final_data:_forged"
    assert all(line != "retry: 0" for line in lines)
    assert all(line != "data: forged" for line in lines)
