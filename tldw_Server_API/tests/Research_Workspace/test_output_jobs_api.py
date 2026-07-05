from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.schemas.research_workspace_outputs import (
    ResearchWorkspaceOutputSubmitRequest,
)
from tldw_Server_API.app.core.Research_Workspace.output_jobs import (
    RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
    ResearchWorkspaceOutputJobError,
    get_research_workspace_output_job_status,
    submit_research_workspace_output_job,
)

pytestmark = pytest.mark.integration


class FakeWorkspaceDB:
    def __init__(self) -> None:
        self.sources = [
            {
                "id": "src-1",
                "workspace_id": "ws-1",
                "media_id": 1,
                "title": "Source",
                "source_type": "pdf",
            }
        ]
        self.added_artifacts: list[dict[str, Any]] = []
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.artifact_lookups = 0

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        return {"id": workspace_id, "name": "Workspace"}

    def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
        return [source for source in self.sources if source["workspace_id"] == workspace_id]

    def add_workspace_artifact(self, workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = {
            "workspace_id": workspace_id,
            "root_artifact_id": payload["id"],
            "artifact_version_id": f"{payload['id']}:v1",
            "created_at": "2026-07-05T00:00:00Z",
            "version": 1,
            **payload,
        }
        self.added_artifacts.append(payload)
        self.artifacts[payload["id"]] = row
        return row

    def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any] | None:
        self.artifact_lookups += 1
        artifact = self.artifacts.get(artifact_id)
        if artifact and artifact["workspace_id"] == workspace_id:
            return artifact
        return None


class FakeJobManager:
    def __init__(self) -> None:
        self.created_jobs: list[dict[str, Any]] = []
        self.jobs: dict[int, dict[str, Any]] = {}

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        job_id = len(self.created_jobs) + 100
        row = {
            "id": job_id,
            "status": "queued",
            "progress_percent": 0.0,
            "progress_message": "queued",
            **kwargs,
        }
        self.created_jobs.append(row)
        self.jobs[job_id] = row
        return row

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        return self.jobs.get(int(job_id))


def test_submit_output_rejects_unknown_artifact_type() -> None:
    with pytest.raises(ValidationError):
        ResearchWorkspaceOutputSubmitRequest(artifact_type="not-real", source_ids=["src-1"])


def test_submit_output_rejects_empty_source_ids() -> None:
    with pytest.raises(ValidationError):
        ResearchWorkspaceOutputSubmitRequest(artifact_type="infographic", source_ids=[])


def test_submit_output_creates_pending_artifact_and_job() -> None:
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager()

    result = submit_research_workspace_output_job(
        workspace_id="ws-1",
        request=ResearchWorkspaceOutputSubmitRequest(
            artifact_type="infographic",
            source_ids=["src-1"],
        ),
        workspace_db=workspace_db,
        job_manager=job_manager,
        user_id="42",
    )

    assert result.artifact_type == "infographic"
    assert result.status == "queued"
    assert workspace_db.added_artifacts[0]["status"] == "pending"
    assert job_manager.created_jobs[0]["job_type"] == RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE
    assert job_manager.created_jobs[0]["payload"]["source_ids"] == ["src-1"]


def test_submit_output_rejects_missing_sources_before_creating_artifact() -> None:
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager()

    with pytest.raises(ResearchWorkspaceOutputJobError) as excinfo:
        submit_research_workspace_output_job(
            workspace_id="ws-1",
            request=ResearchWorkspaceOutputSubmitRequest(
                artifact_type="infographic",
                source_ids=["missing"],
            ),
            workspace_db=workspace_db,
            job_manager=job_manager,
            user_id="42",
        )

    assert excinfo.value.public_code == "workspace_sources_not_found"
    assert workspace_db.added_artifacts == []
    assert job_manager.created_jobs == []


def test_status_returns_job_progress_plus_artifact() -> None:
    workspace_db = FakeWorkspaceDB()
    artifact = workspace_db.add_workspace_artifact(
        "ws-1",
        {
            "id": "infographic-abc",
            "artifact_type": "infographic",
            "title": "Infographic",
            "status": "pending",
            "content": None,
            "content_type": "image/png",
        },
    )
    job_manager = FakeJobManager()
    job_manager.jobs[101] = {
        "id": 101,
        "status": "processing",
        "owner_user_id": "42",
        "job_type": RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
        "payload": {
            "workspace_id": "ws-1",
            "artifact_id": artifact["id"],
            "artifact_type": "infographic",
        },
        "result": {"step": "render"},
        "progress_percent": 42.5,
        "progress_message": "generate_infographic",
    }

    result = get_research_workspace_output_job_status(
        workspace_id="ws-1",
        job_id=101,
        workspace_db=workspace_db,
        job_manager=job_manager,
        user_id="42",
    )

    assert result.status == "processing"
    assert result.progress_percent == 42.5
    assert result.progress_message == "generate_infographic"
    assert result.artifact_id == "infographic-abc"
    assert result.artifact is not None
    assert result.artifact.id == "infographic-abc"
    assert result.result == {"step": "render"}


@pytest.mark.parametrize(
    "job",
    [
        {
            "id": 101,
            "status": "queued",
            "owner_user_id": "42",
            "job_type": RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
            "payload": {
                "workspace_id": "other-ws",
                "artifact_id": "infographic-abc",
                "artifact_type": "infographic",
            },
        },
        {
            "id": 101,
            "status": "queued",
            "owner_user_id": "99",
            "job_type": RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
            "payload": {
                "workspace_id": "ws-1",
                "artifact_id": "infographic-abc",
                "artifact_type": "infographic",
            },
        },
    ],
)
def test_status_route_rejects_workspace_or_owner_mismatch_before_artifact_lookup(
    job: dict[str, Any],
) -> None:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager()
    job_manager.jobs[101] = job

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=42)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: workspace_db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/ws-1/outputs/101")

    assert response.status_code == 404
    assert workspace_db.artifact_lookups == 0
