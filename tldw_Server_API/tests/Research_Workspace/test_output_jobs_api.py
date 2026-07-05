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
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.schemas.research_workspace_outputs import (
    ResearchWorkspaceOutputSubmitRequest,
)
from tldw_Server_API.app.core.Research_Workspace.output_jobs import (
    RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
    RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
    ResearchWorkspaceOutputJobError,
    get_research_workspace_output_job_status,
    submit_research_workspace_output_job,
)

pytestmark = pytest.mark.integration


class FakeWorkspaceDB:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
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
        self.deleted_artifacts: list[tuple[str, str]] = []
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

    def delete_workspace_artifact(self, workspace_id: str, artifact_id: str) -> None:
        self.deleted_artifacts.append((workspace_id, artifact_id))
        if self.fail_delete:
            raise RuntimeError("cleanup unavailable")
        self.artifacts.pop(artifact_id, None)


class FakeJobManager:
    def __init__(self, *, fail_create: bool = False, fail_get: bool = False) -> None:
        self.fail_create = fail_create
        self.fail_get = fail_get
        self.created_jobs: list[dict[str, Any]] = []
        self.jobs: dict[int, dict[str, Any]] = {}

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_create:
            raise RuntimeError("queue full")
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
        if self.fail_get:
            raise RuntimeError("jobs backend unavailable")
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


def test_submit_output_rolls_back_pending_artifact_when_job_create_fails() -> None:
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager(fail_create=True)

    with pytest.raises(ResearchWorkspaceOutputJobError) as excinfo:
        submit_research_workspace_output_job(
            workspace_id="ws-1",
            request=ResearchWorkspaceOutputSubmitRequest(
                artifact_type="infographic",
                source_ids=["src-1"],
            ),
            workspace_db=workspace_db,
            job_manager=job_manager,
            user_id="42",
        )

    assert excinfo.value.public_code == "output_job_enqueue_failed"
    assert excinfo.value.status_code == 503
    assert workspace_db.added_artifacts[0]["status"] == "pending"
    assert workspace_db.deleted_artifacts == [("ws-1", workspace_db.added_artifacts[0]["id"])]
    assert workspace_db.artifacts == {}


def test_submit_output_preserves_job_create_error_when_rollback_cleanup_fails() -> None:
    workspace_db = FakeWorkspaceDB(fail_delete=True)
    job_manager = FakeJobManager(fail_create=True)

    with pytest.raises(ResearchWorkspaceOutputJobError) as excinfo:
        submit_research_workspace_output_job(
            workspace_id="ws-1",
            request=ResearchWorkspaceOutputSubmitRequest(
                artifact_type="infographic",
                source_ids=["src-1"],
            ),
            workspace_db=workspace_db,
            job_manager=job_manager,
            user_id="42",
        )

    assert excinfo.value.public_code == "output_job_enqueue_failed"
    assert excinfo.value.status_code == 503
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "queue full"
    assert workspace_db.deleted_artifacts == [("ws-1", workspace_db.added_artifacts[0]["id"])]
    assert workspace_db.artifacts


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
        "domain": RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
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


def test_status_rejects_missing_job_domain_before_artifact_lookup() -> None:
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager()
    job_manager.jobs[101] = {
        "id": 101,
        "status": "queued",
        "owner_user_id": "42",
        "job_type": RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
        "payload": {
            "workspace_id": "ws-1",
            "artifact_id": "infographic-abc",
            "artifact_type": "infographic",
        },
    }

    with pytest.raises(ResearchWorkspaceOutputJobError) as excinfo:
        get_research_workspace_output_job_status(
            workspace_id="ws-1",
            job_id=101,
            workspace_db=workspace_db,
            job_manager=job_manager,
            user_id="42",
        )

    assert excinfo.value.public_code == "job_not_found"
    assert workspace_db.artifact_lookups == 0


def test_submit_output_route_creates_job() -> None:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager()

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=42)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: workspace_db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/workspaces/ws-1/outputs",
            json={"artifact_type": "infographic", "source_ids": ["src-1"]},
        )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["artifact_type"] == "infographic"
    assert job_manager.created_jobs[0]["owner_user_id"] == "42"


def test_submit_output_route_maps_job_create_failure_to_stable_error() -> None:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager(fail_create=True)

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=42)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: workspace_db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/workspaces/ws-1/outputs",
            json={"artifact_type": "infographic", "source_ids": ["src-1"]},
        )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "output_job_enqueue_failed"
    assert workspace_db.artifacts == {}


def test_status_route_maps_job_lookup_failure_to_stable_error() -> None:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    workspace_db = FakeWorkspaceDB()
    job_manager = FakeJobManager(fail_get=True)

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=42)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: workspace_db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/ws-1/outputs/101")

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "output_job_status_unavailable"
    assert workspace_db.artifact_lookups == 0


@pytest.mark.parametrize(
    "job",
    [
        {
            "id": 101,
            "status": "queued",
            "owner_user_id": "42",
            "domain": RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
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
            "domain": RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
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
