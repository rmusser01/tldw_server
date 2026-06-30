from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_benchmarks as benchmarks_ep,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_unified as eval_unified,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.identity import EvaluationIdentity


def _build_app():
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            roles=["admin"],
            permissions=["system.configure", "evals.read", "evals.manage"],
        )

    async def _rate_limit_dep_override():
        return None

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.check_evaluation_rate_limit] = _rate_limit_dep_override
    return app


def test_benchmark_catalog_exposed_under_evaluations_namespace():
    app = _build_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/evaluations/benchmarks")
    assert response.status_code == 200
    names = [b["name"] for b in response.json()["data"]]
    assert "bullshit_benchmark" in names


def test_benchmark_run_endpoint_exists(monkeypatch):
    class _Reg:
        def get(self, name):
            if name != "bullshit_benchmark":
                return None
            return type("Cfg", (), {"evaluation_type": "bullshit_detection"})()

        def create_evaluator(self, _name):
            class _Eval:
                def format_for_custom_metric(self, item):
                    return {
                        "name": "bullshit_detection",
                        "description": "d",
                        "evaluation_prompt": "{question}",
                        "input_data": {"question": item["question"]},
                        "scoring_criteria": {"incoherence_identification": "x"},
                    }

            return _Eval()

    monkeypatch.setattr(benchmarks_ep, "get_registry", lambda: _Reg())
    monkeypatch.setattr(
        benchmarks_ep,
        "load_benchmark_dataset",
        lambda *args, **kwargs: [
            {
                "id": "q1",
                "question": "q",
                "nonsensical_element": "n",
                "domain": "d",
            }
        ],
    )

    class _Manager:
        evaluate_custom_metric = AsyncMock(return_value={"score": 0.9, "explanation": "ok"})
        store_evaluation = AsyncMock(return_value="eval_1")

    monkeypatch.setattr(
        benchmarks_ep,
        "_get_evaluation_manager_for_user",
        lambda _user: _Manager(),
    )

    app = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/benchmarks/bullshit_benchmark/run",
            json={"limit": 1, "api_name": "openai", "parallel": 1, "save_results": False},
        )
    assert response.status_code == 200
    assert response.json()["benchmark"] == "bullshit_benchmark"


def test_benchmark_manager_preserves_string_user_scope(monkeypatch):
    captured = {}

    class _Manager:
        def __init__(self, user_id=None):
            captured["user_id"] = user_id

    monkeypatch.setattr(benchmarks_ep, "EvaluationManager", _Manager)

    identity = EvaluationIdentity(
        user_scope="tenant-user",
        created_by="tenant-user",
        rate_limit_subject="tenant-user",
        webhook_user_id="user_tenant-user",
    )

    _ = benchmarks_ep._get_evaluation_manager_for_user(identity)
    assert captured["user_id"] == "tenant-user"
