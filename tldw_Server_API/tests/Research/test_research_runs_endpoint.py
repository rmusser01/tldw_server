from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


pytestmark = pytest.mark.unit


def test_create_and_approve_research_run():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def create_session(self, **kwargs):
            assert kwargs["owner_user_id"] == "1"
            return {
                "id": "rs_1",
                "status": "queued",
                "phase": "drafting_plan",
                "control_state": "running",
                "progress_percent": None,
                "progress_message": None,
                "active_job_id": "9",
                "latest_checkpoint_id": None,
            }

        def approve_checkpoint(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            assert kwargs["checkpoint_id"] == "cp_1"
            return {
                "id": kwargs["session_id"],
                "phase": "collecting",
                "status": "queued",
                "control_state": "running",
                "progress_percent": 10.0,
                "progress_message": "planning research",
                "active_job_id": "10",
                "latest_checkpoint_id": kwargs["checkpoint_id"],
            }

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        create_resp = client.post("/api/v1/research/runs", json={"query": "Test deep research run"})
        assert create_resp.status_code == 200
        assert create_resp.json()["id"] == "rs_1"

        approve_resp = client.post(
            "/api/v1/research/runs/rs_1/checkpoints/cp_1/patch-and-approve",
            json={"patch_payload": {"focus_areas": ["background", "counterevidence"]}},
        )
        assert approve_resp.status_code == 200
        assert approve_resp.json()["phase"] == "collecting"


def test_patch_and_approve_research_run_triggers_research_checkpoint_workflow_resume_bridge(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    bridge_calls: list[dict[str, str]] = []

    async def _fake_bridge(*, research_run_id: str, checkpoint_id: str) -> None:
        bridge_calls.append(
            {
                "research_run_id": research_run_id,
                "checkpoint_id": checkpoint_id,
            }
        )

    class StubService:
        def approve_checkpoint(self, **kwargs):
            return {
                "id": kwargs["session_id"],
                "phase": "collecting",
                "status": "queued",
                "control_state": "running",
                "progress_percent": 10.0,
                "progress_message": "planning research",
                "active_job_id": "10",
                "latest_checkpoint_id": kwargs["checkpoint_id"],
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.research_wait_bridge.resume_workflows_waiting_on_research_checkpoint",
        _fake_bridge,
        raising=False,
    )
    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        approve_resp = client.post(
            "/api/v1/research/runs/rs_1/checkpoints/cp_1/patch-and-approve",
            json={},
        )

    assert approve_resp.status_code == 200
    assert bridge_calls == [
        {
            "research_run_id": "rs_1",
            "checkpoint_id": "cp_1",
        }
    ]


def test_patch_and_approve_research_run_returns_400_for_invalid_checkpoint_patch():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def approve_checkpoint(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            assert kwargs["checkpoint_id"] == "cp_1"
            raise ValueError("invalid_checkpoint_patch:plan_review:query")

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        approve_resp = client.post(
            "/api/v1/research/runs/rs_1/checkpoints/cp_1/patch-and-approve",
            json={"patch_payload": {"query": "Mutated query"}},
        )

    assert approve_resp.status_code == 400
    assert approve_resp.json()["detail"] == "invalid_checkpoint_patch:plan_review:query"


def test_create_research_run_passes_provider_overrides():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def create_session(self, **kwargs):
            assert kwargs["provider_overrides"]["web"]["engine"] == "duckduckgo"
            assert kwargs["provider_overrides"]["local"]["top_k"] == 4
            return {
                "id": "rs_2",
                "status": "queued",
                "phase": "drafting_plan",
                "control_state": "running",
                "progress_percent": None,
                "progress_message": None,
                "active_job_id": "11",
                "latest_checkpoint_id": None,
            }

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/research/runs",
            json={
                "query": "Provider override test",
                "provider_overrides": {
                    "local": {"top_k": 4, "sources": ["media_db"]},
                    "web": {"engine": "duckduckgo", "result_count": 3},
                },
            },
        )
        assert create_resp.status_code == 200
        assert create_resp.json()["id"] == "rs_2"


def test_create_research_run_passes_follow_up_payload_to_service():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def create_session(self, **kwargs):
            assert kwargs["chat_handoff"]["chat_id"] == "chat_123"
            assert kwargs["follow_up"]["question"] == "What should we check next?"
            assert kwargs["follow_up"]["background"]["outline"][0]["title"] == "Evidence gap"
            return {
                "id": "rs_3",
                "status": "queued",
                "phase": "drafting_plan",
                "control_state": "running",
                "progress_percent": None,
                "progress_message": None,
                "active_job_id": "12",
                "latest_checkpoint_id": None,
            }

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/research/runs",
            json={
                "query": "What should we check next?",
                "chat_handoff": {"chat_id": "chat_123"},
                "follow_up": {
                    "question": "What should we check next?",
                    "background": {
                        "question": "What did the attached research conclude?",
                        "outline": [{"title": "Evidence gap", "focus_area": "background"}],
                        "key_claims": [
                            {
                                "claim_id": "clm_1",
                                "text": "The evidence gap remains unresolved.",
                            }
                        ],
                        "unresolved_questions": ["Which claim needs verification next?"],
                        "verification_summary": {
                            "supported_claim_count": 1,
                            "unsupported_claim_count": 0,
                        },
                        "source_trust_summary": {
                            "high_trust_count": 1,
                            "low_trust_count": 0,
                        },
                    },
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["id"] == "rs_3"


def test_create_research_run_returns_400_for_invalid_chat_handoff():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def create_session(self, **_kwargs):
            raise ValueError(
                "chat_handoff.chat_id is not owned by the current user or was not found"
            )

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/research/runs",
            json={
                "query": "What should we check next?",
                "chat_handoff": {"chat_id": "chat_missing"},
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "chat_handoff.chat_id is not owned by the current user or was not found"
    )


def test_research_run_create_request_rejects_malformed_follow_up_background():
    from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchRunCreateRequest

    with pytest.raises(ValidationError):
        ResearchRunCreateRequest.model_validate(
            {
                "query": "What should we check next?",
                "follow_up": {
                    "question": "What should we check next?",
                    "background": {
                        "question": "Seed context",
                        "outline": "not-a-list",
                    },
                },
            }
        )


def test_research_run_create_request_rejects_oversized_follow_up_background():
    from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchRunCreateRequest

    with pytest.raises(ValidationError):
        ResearchRunCreateRequest.model_validate(
            {
                "query": "What should we check next?",
                "follow_up": {
                    "question": "What should we check next?",
                    "background": {
                        "question": "Seed context",
                        "outline": [
                            {"title": f"Section {index}", "focus_area": "background"}
                            for index in range(8)
                        ],
                        "key_claims": [],
                        "unresolved_questions": [],
                        "verification_summary": {
                            "supported_claim_count": 1,
                            "unsupported_claim_count": 0,
                        },
                        "source_trust_summary": {
                            "high_trust_count": 1,
                            "low_trust_count": 0,
                        },
                    },
                },
            }
        )


def test_list_research_runs_endpoint_returns_recent_created_runs():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def list_sessions(self, **kwargs):
            assert kwargs["owner_user_id"] == "1"
            assert kwargs["limit"] == 25
            return [
                {
                    "id": "rs_new",
                    "query": "Newest created run",
                    "status": "queued",
                    "phase": "drafting_plan",
                    "control_state": "running",
                    "progress_percent": 5.0,
                    "progress_message": "planning research",
                    "active_job_id": "15",
                    "latest_checkpoint_id": None,
                    "completed_at": None,
                    "created_at": "2026-03-07T08:00:00+00:00",
                    "updated_at": "2026-03-07T08:00:00+00:00",
                },
                {
                    "id": "rs_old",
                    "query": "Older created run",
                    "status": "waiting_human",
                    "phase": "awaiting_plan_review",
                    "control_state": "running",
                    "progress_percent": 10.0,
                    "progress_message": "planning research",
                    "active_job_id": None,
                    "latest_checkpoint_id": "cp_1",
                    "completed_at": None,
                    "created_at": "2026-03-07T07:00:00+00:00",
                    "updated_at": "2026-03-07T07:30:00+00:00",
                },
            ]

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        response = client.get("/api/v1/research/runs")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == ["rs_new", "rs_old"]
    assert body[0]["query"] == "Newest created run"
    assert body[0]["created_at"] == "2026-03-07T08:00:00+00:00"
    assert body[1]["updated_at"] == "2026-03-07T07:30:00+00:00"


def test_read_research_run_bundle_and_artifact():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def get_session(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            return {
                "id": "rs_1",
                "status": "completed",
                "phase": "completed",
                "control_state": "running",
                "progress_percent": 100.0,
                "progress_message": "packaging results",
                "active_job_id": None,
                "latest_checkpoint_id": "cp_1",
                "completed_at": "2026-03-07T00:00:00+00:00",
                "chat_id": "chat_123",
            }

        def get_bundle(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            return {
                "question": "What changed?",
                "report_markdown": "# Report",
                "claims": [],
            }

        def get_artifact(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            assert kwargs["artifact_name"] == "report_v1.md"
            return {
                "artifact_name": "report_v1.md",
                "content_type": "text/markdown",
                "content": "# Report",
            }

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        run_resp = client.get("/api/v1/research/runs/rs_1")
        bundle_resp = client.get("/api/v1/research/runs/rs_1/bundle")
        artifact_resp = client.get("/api/v1/research/runs/rs_1/artifacts/report_v1.md")

        assert run_resp.status_code == 200
        assert run_resp.json()["completed_at"] == "2026-03-07T00:00:00+00:00"
        assert run_resp.json()["control_state"] == "running"
        assert run_resp.json()["progress_percent"] == 100.0
        assert run_resp.json()["progress_message"] == "packaging results"
        assert run_resp.json()["chat_id"] == "chat_123"
        assert bundle_resp.status_code == 200
        assert bundle_resp.json()["question"] == "What changed?"
        assert artifact_resp.status_code == 200
        assert artifact_resp.json()["content_type"] == "text/markdown"


def test_pause_resume_and_cancel_research_run_endpoints():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def pause_run(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            return {
                "id": "rs_1",
                "status": "queued",
                "phase": "collecting",
                "control_state": "pause_requested",
                "progress_percent": 45.0,
                "progress_message": "collecting sources",
                "active_job_id": "22",
                "latest_checkpoint_id": None,
                "completed_at": None,
            }

        def resume_run(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            return {
                "id": "rs_1",
                "status": "waiting_human",
                "phase": "awaiting_plan_review",
                "control_state": "running",
                "progress_percent": 10.0,
                "progress_message": "planning research",
                "active_job_id": None,
                "latest_checkpoint_id": "cp_1",
                "completed_at": None,
            }

        def cancel_run(self, **kwargs):
            assert kwargs["session_id"] == "rs_1"
            return {
                "id": "rs_1",
                "status": "cancelled",
                "phase": "awaiting_plan_review",
                "control_state": "cancelled",
                "progress_percent": 10.0,
                "progress_message": "planning research",
                "active_job_id": None,
                "latest_checkpoint_id": "cp_1",
                "completed_at": None,
            }

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        pause_resp = client.post("/api/v1/research/runs/rs_1/pause")
        resume_resp = client.post("/api/v1/research/runs/rs_1/resume")
        cancel_resp = client.post("/api/v1/research/runs/rs_1/cancel")

        assert pause_resp.status_code == 200
        assert pause_resp.json()["control_state"] == "pause_requested"
        assert pause_resp.json()["progress_percent"] == 45.0
        assert resume_resp.status_code == 200
        assert resume_resp.json()["status"] == "waiting_human"
        assert resume_resp.json()["control_state"] == "running"
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"
        assert cancel_resp.json()["control_state"] == "cancelled"


def test_read_research_artifact_maps_not_found_and_disallowed_errors():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def get_session(self, **kwargs):
            raise KeyError(kwargs["session_id"])

        def get_bundle(self, **kwargs):
            raise KeyError("bundle.json")

        def get_artifact(self, **kwargs):
            if kwargs["artifact_name"] == "bad.bin":
                raise ValueError("artifact_not_allowed")
            raise KeyError(kwargs["artifact_name"])

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        run_resp = client.get("/api/v1/research/runs/rs_missing")
        bundle_resp = client.get("/api/v1/research/runs/rs_1/bundle")
        bad_artifact_resp = client.get("/api/v1/research/runs/rs_1/artifacts/bad.bin")
        missing_artifact_resp = client.get("/api/v1/research/runs/rs_1/artifacts/report_v1.md")

        assert run_resp.status_code == 404
        assert bundle_resp.status_code == 404
        assert bad_artifact_resp.status_code == 400
        assert missing_artifact_resp.status_code == 404


def test_run_control_endpoints_map_invalid_transition_and_missing_run_errors():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def pause_run(self, **kwargs):
            if kwargs["session_id"] == "rs_missing":
                raise KeyError(kwargs["session_id"])
            raise ValueError("pause_not_allowed")

        def resume_run(self, **kwargs):
            if kwargs["session_id"] == "rs_missing":
                raise KeyError(kwargs["session_id"])
            raise ValueError("resume_not_allowed")

        def cancel_run(self, **kwargs):
            if kwargs["session_id"] == "rs_missing":
                raise KeyError(kwargs["session_id"])
            raise ValueError("cancel_not_allowed")

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        assert client.post("/api/v1/research/runs/rs_1/pause").status_code == 400
        assert client.post("/api/v1/research/runs/rs_1/resume").status_code == 400
        assert client.post("/api/v1/research/runs/rs_1/cancel").status_code == 400
        assert client.post("/api/v1/research/runs/rs_missing/pause").status_code == 404
        assert client.post("/api/v1/research/runs/rs_missing/resume").status_code == 404
        assert client.post("/api/v1/research/runs/rs_missing/cancel").status_code == 404


def test_research_run_events_stream_emits_snapshot_then_terminal_for_completed_run():
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchRunSnapshotResponse

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def get_stream_snapshot(self, **kwargs):
            assert kwargs["owner_user_id"] == "1"
            assert kwargs["session_id"] == "rs_1"
            return ResearchRunSnapshotResponse.model_validate(
                {
                    "run": {
                        "id": "rs_1",
                        "status": "completed",
                        "phase": "completed",
                        "control_state": "running",
                        "progress_percent": 100.0,
                        "progress_message": "packaging results",
                        "active_job_id": None,
                        "latest_checkpoint_id": "cp_1",
                        "completed_at": "2026-03-07T00:00:00+00:00",
                    },
                    "latest_event_id": 0,
                    "checkpoint": {
                        "checkpoint_id": "cp_1",
                        "checkpoint_type": "outline_review",
                        "status": "resolved",
                        "proposed_payload": {"outline": {"sections": [{"title": "Background"}]}},
                        "resolution": "approved",
                    },
                    "artifacts": [
                        {
                            "artifact_name": "bundle.json",
                            "artifact_version": 1,
                            "content_type": "application/json",
                            "phase": "packaging",
                            "job_id": "14",
                        }
                    ],
                }
            )

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        with client.stream("GET", "/api/v1/research/runs/rs_1/events/stream") as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "event: snapshot" in body
        assert "event: terminal" in body
        assert body.index("event: snapshot") < body.index("event: terminal")
        assert "\"latest_event_id\":0" in body.replace(" ", "")


def test_research_run_events_stream_replays_persisted_rows_with_after_id():
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchRunSnapshotResponse
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchRunEventRow

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def get_stream_snapshot(self, **kwargs):
            assert kwargs["owner_user_id"] == "1"
            assert kwargs["session_id"] == "rs_1"
            return ResearchRunSnapshotResponse.model_validate(
                {
                    "run": {
                        "id": "rs_1",
                        "status": "completed",
                        "phase": "completed",
                        "control_state": "running",
                        "progress_percent": 100.0,
                        "progress_message": "packaging results",
                        "active_job_id": None,
                        "latest_checkpoint_id": None,
                        "completed_at": "2026-03-07T00:00:00+00:00",
                    },
                    "latest_event_id": 5,
                    "checkpoint": None,
                    "artifacts": [],
                }
            )

        def list_run_events_after(self, **kwargs):
            assert kwargs["owner_user_id"] == "1"
            assert kwargs["session_id"] == "rs_1"
            assert kwargs["after_id"] == 3
            return [
                ResearchRunEventRow(
                    id=4,
                    session_id="rs_1",
                    owner_user_id="1",
                    event_type="artifact",
                    event_payload={
                        "artifact_name": "report_v1.md",
                        "artifact_version": 1,
                        "content_type": "text/markdown",
                        "phase": "synthesizing",
                        "job_id": "81",
                    },
                    phase="synthesizing",
                    job_id="81",
                    created_at="2026-03-07T00:00:00+00:00",
                ),
                ResearchRunEventRow(
                    id=5,
                    session_id="rs_1",
                    owner_user_id="1",
                    event_type="terminal",
                    event_payload={
                        "id": "rs_1",
                        "status": "completed",
                        "phase": "completed",
                        "control_state": "running",
                        "active_job_id": None,
                        "latest_checkpoint_id": None,
                        "completed_at": "2026-03-07T00:00:00+00:00",
                    },
                    phase="completed",
                    job_id=None,
                    created_at="2026-03-07T00:00:01+00:00",
                ),
            ]

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        with client.stream("GET", "/api/v1/research/runs/rs_1/events/stream?after_id=3") as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        compact = body.replace(" ", "").replace("\n", "")
        assert response.status_code == 200
        assert "event:snapshot" in compact
        assert "\"latest_event_id\":5" in compact
        assert "id:4" in compact
        assert "event:artifact" in compact
        assert "\"event_id\":4" in compact
        assert "\"replayed\":true" in compact
        assert "id:5" in compact
        assert "event:terminal" in compact


def test_research_run_events_stream_maps_missing_run_to_404():
    from tldw_Server_API.app.api.v1.endpoints import research_runs

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    class StubService:
        def get_stream_snapshot(self, **kwargs):
            raise KeyError(kwargs["session_id"])

    app.dependency_overrides[research_runs.get_research_service] = lambda: StubService()

    with TestClient(app) as client:
        response = client.get("/api/v1/research/runs/rs_missing/events/stream")

    assert response.status_code == 404
