import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


class _StubDB:
    def __init__(self):
        self.client_id = "test-client"

    def get_prompt_with_project(self, prompt_id: int, include_deleted: bool = False):
        return {"id": prompt_id, "project_id": 321}

    def create_optimization(self, **kwargs):

        return {"id": 555, **kwargs}

    def update_optimization(self, optimization_id: int, updates: dict):
        return {"id": optimization_id, **updates}


@pytest.fixture
def override_db_dependency(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps

    async def _override_db():
        return _StubDB()

    app.dependency_overrides[deps.get_prompt_studio_db] = _override_db

    # Skip actual permission checks
    async def _ok(*args, **kwargs):
        return True

    monkeypatch.setattr(deps, "require_project_write_access", _ok, raising=True)

    yield
    app.dependency_overrides.pop(deps.get_prompt_studio_db, None)


def test_create_optimization_includes_request_id_in_job_payload(monkeypatch, override_db_dependency):


    # Force TEST_MODE for deterministic behavior (skip background task spawn)
    monkeypatch.setenv("TEST_MODE", "true")

    captured = {}

    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import jobs_adapter as ps_jobs

    def fake_create_job(  # noqa: D401
        self,
        *,
        user_id=None,
        job_type=None,
        entity_id=None,
        payload=None,
        project_id=None,
        priority=5,
        max_retries=3,
        request_id=None,
        trace_id=None,
    ):
        captured["payload"] = payload
        return {"id": 999, "status": "queued"}

    monkeypatch.setattr(ps_jobs.PromptStudioJobsAdapter, "create_job", fake_create_job, raising=True)

    client = TestClient(app)
    body = {
        "project_id": 1,
        "initial_prompt_id": 1,
        "optimization_config": {
            "optimizer_type": "iterative",
            "target_metric": "accuracy"
        },
        "test_case_ids": [1, 2],
        "name": "Refine"
    }
    r = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=body,
        headers={
            "X-Request-ID": "req-ps-create-001",
            "X-API-KEY": "test-key",
        },
    )
    assert r.status_code in (200, 201), r.text
    assert captured.get("payload", {}).get("request_id") == "req-ps-create-001"
