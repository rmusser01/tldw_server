import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


class _StubDB:
    def __init__(self):
        self.client_id = "test-client"
        self._next_opt_id = 100
        self.created_optimizations = 0

    def get_prompt_with_project(self, prompt_id: int, include_deleted: bool = False):
        # Always return a fake project association for the provided prompt
        return {"id": prompt_id, "project_id": 123, "project_user_id": "1"}

    def get_test_cases_by_ids(self, test_case_ids: list[int]):
        return [
            {"id": test_case_id, "project_id": 123, "deleted": False}
            for test_case_id in test_case_ids
        ]

    def create_optimization(self, **kwargs):
        self.created_optimizations += 1
        oid = self._next_opt_id
        self._next_opt_id += 1
        return {"id": oid, "uuid": f"optimization-{oid}", **kwargs}

    def update_optimization(self, optimization_id: int, updates: dict):
        return {
            "id": optimization_id,
            "uuid": f"optimization-{optimization_id}",
            **updates,
        }


@pytest.fixture
def override_db_dependency(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_optimization as pso,
    )

    db = _StubDB()

    async def _override_db():
        return db

    app.dependency_overrides[deps.get_prompt_studio_db] = _override_db

    # Permit write access without real checks for this unit test
    async def _ok(*args, **kwargs):
        return True

    monkeypatch.setattr(pso, "require_project_write_access", _ok, raising=True)

    yield db
    app.dependency_overrides.pop(deps.get_prompt_studio_db, None)


def test_compare_strategies_propagates_request_id_for_each_job(monkeypatch, override_db_dependency):


    captured_payloads = []

    # Patch Prompt Studio Jobs adapter to capture each payload
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
        captured_payloads.append(payload)
        return {"id": 9000 + len(captured_payloads), "status": "queued"}

    monkeypatch.setattr(ps_jobs.PromptStudioJobsAdapter, "create_job", fake_create_job, raising=True)

    client = TestClient(app)
    strategies = ["iterative", "mipro", "bootstrap"]
    r = client.post(
        "/api/v1/prompt-studio/optimizations/compare-strategies",
        json={
            "prompt_id": 1,
            "test_case_ids": [1, 2],
            "strategies": strategies,
            "model_configuration": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
            },
        },
        headers={
            "X-API-KEY": "test-api-key-12345",
            "X-Request-ID": "req-ps-compare-001",
        },
    )
    assert r.status_code == 200, r.text
    assert len(captured_payloads) == len(strategies)
    assert all(p.get("request_id") == "req-ps-compare-001" for p in captured_payloads)
    assert [p.get("optimization_uuid") for p in captured_payloads] == [
        "optimization-100",
        "optimization-101",
        "optimization-102",
    ]


def test_compare_strategies_mixed_case_request_id_header(monkeypatch, override_db_dependency):


    captured_payloads = []

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
        captured_payloads.append(payload)
        return {"id": 8000 + len(captured_payloads), "status": "queued"}

    monkeypatch.setattr(ps_jobs.PromptStudioJobsAdapter, "create_job", fake_create_job, raising=True)

    client = TestClient(app)
    r = client.post(
        "/api/v1/prompt-studio/optimizations/compare-strategies",
        json={
            "prompt_id": 1,
            "test_case_ids": [3, 4],
            "strategies": ["iterative", "mipro"],
            "model_configuration": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
            },
        },
        headers={
            "X-API-KEY": "test-api-key-12345",
            "x-request-id": "req-ps-compare-mixed-01",  # mixed-case header
        },
    )
    assert r.status_code == 200, r.text
    assert len(captured_payloads) == 2
    assert all(
        payload.get("request_id") == "req-ps-compare-mixed-01"
        for payload in captured_payloads
    )


def test_compare_rejects_unsupported_strategy_before_side_effects(
    monkeypatch,
    override_db_dependency,
):
    captured_payloads = []

    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import jobs_adapter as ps_jobs

    def fake_create_job(self, **kwargs):
        captured_payloads.append(kwargs.get("payload"))
        return {"id": 7001, "status": "queued"}

    monkeypatch.setattr(
        ps_jobs.PromptStudioJobsAdapter,
        "create_job",
        fake_create_job,
        raising=True,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/prompt-studio/optimizations/compare-strategies",
        json={
            "prompt_id": 1,
            "test_case_ids": [1],
            "strategies": ["iterative", "quantum_search"],
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422
    assert override_db_dependency.created_optimizations == 0
    assert captured_payloads == []
