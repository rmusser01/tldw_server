from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.unit


def _create_session(meetings_api_client: TestClient) -> str:
    resp = meetings_api_client.post(
        "/api/v1/meetings/sessions",
        json={"title": "Finalize Session", "meeting_type": "standup"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_finalize_session_generates_summary_and_actions(meetings_api_client: TestClient) -> None:
    session_id = _create_session(meetings_api_client)
    transcript = (
        "Team discussed blockers. TODO: Alice will update the API docs. "
        "TODO: Bob will validate deployment checklist."
    )

    commit_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={"transcript_text": transcript},
    )
    assert commit_resp.status_code == 200

    body = commit_resp.json()
    artifacts_by_kind = {artifact["kind"]: artifact for artifact in body["artifacts"]}
    assert "summary" in artifacts_by_kind
    assert "action_items" in artifacts_by_kind
    assert "decisions" in artifacts_by_kind
    assert "speaker_stats" in artifacts_by_kind
    assert artifacts_by_kind["action_items"]["payload_json"]["items"] == [
        "Alice will update the API docs",
        "Bob will validate deployment checklist",
    ]
    assert artifacts_by_kind["decisions"]["payload_json"]["items"] == []


def test_finalize_session_rejects_unsupported_final_kind_without_partial_artifacts(
    meetings_api_client: TestClient,
) -> None:
    session_id = _create_session(meetings_api_client)

    commit_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={
            "transcript_text": "Team discussed launch readiness.",
            "include": ["summary", "sentiment"],
        },
    )

    assert commit_resp.status_code == 400
    assert "not finalizable" in commit_resp.json()["detail"]

    artifacts_resp = meetings_api_client.get(f"/api/v1/meetings/sessions/{session_id}/artifacts")
    assert artifacts_resp.status_code == 200
    assert artifacts_resp.json() == []


def test_finalize_session_respects_empty_include_list(meetings_api_client: TestClient) -> None:
    session_id = _create_session(meetings_api_client)

    commit_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={"transcript_text": "Team discussed launch readiness.", "include": []},
    )

    assert commit_resp.status_code == 200
    assert commit_resp.json()["artifacts"] == []

    artifacts_resp = meetings_api_client.get(f"/api/v1/meetings/sessions/{session_id}/artifacts")
    assert artifacts_resp.status_code == 200
    assert artifacts_resp.json() == []


def test_finalize_session_empty_include_clears_existing_final_artifacts(
    meetings_api_client: TestClient,
) -> None:
    session_id = _create_session(meetings_api_client)
    transcript = "Team discussed blockers. TODO: Alice will update the API docs."

    create_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={"transcript_text": transcript},
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["artifacts"]

    clear_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={"transcript_text": transcript, "include": []},
    )

    assert clear_resp.status_code == 200
    assert clear_resp.json()["artifacts"] == []

    artifacts_resp = meetings_api_client.get(f"/api/v1/meetings/sessions/{session_id}/artifacts")
    assert artifacts_resp.status_code == 200
    assert artifacts_resp.json() == []


def test_finalize_session_replaces_existing_final_artifacts(meetings_api_client: TestClient) -> None:
    session_id = _create_session(meetings_api_client)
    transcript = "Team discussed blockers. TODO: Alice will update the API docs."

    first_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={"transcript_text": transcript},
    )
    second_resp = meetings_api_client.post(
        f"/api/v1/meetings/sessions/{session_id}/commit",
        json={"transcript_text": transcript},
    )

    assert first_resp.status_code == 200
    assert second_resp.status_code == 200

    artifacts_resp = meetings_api_client.get(f"/api/v1/meetings/sessions/{session_id}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()
    kinds = [artifact["kind"] for artifact in artifacts]
    assert sorted(kinds) == ["action_items", "decisions", "speaker_stats", "summary"]
