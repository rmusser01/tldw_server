from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
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


def test_benchmark_run_counts_an_unscored_item_apart(monkeypatch):
    """A scored, an unscored (score None) and a raised item: only the scored one is a success, and only it
    reaches the averages, by_category and the stored scores (owner, 2026-11-10; TASK-15485)."""
    replies = {
        "scored": {"score": 0.8, "explanation": "ok"},
        "unscored": {"score": None, "explanation": "Error: 429 rate limited"},
        "raised": RuntimeError("benchmark item exploded"),
    }

    async def _evaluate(**kwargs):
        reply = replies[kwargs["input_data"]["question"]]
        if isinstance(reply, Exception):
            raise reply
        return reply

    class _Reg:
        def get(self, name):
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

    class _Manager:
        evaluate_custom_metric = staticmethod(_evaluate)
        store_evaluation = AsyncMock(return_value="eval_1")

    monkeypatch.setattr(benchmarks_ep, "get_registry", lambda: _Reg())
    monkeypatch.setattr(
        benchmarks_ep,
        "load_benchmark_dataset",
        lambda *args, **kwargs: [{"question": question, "domain": "d"} for question in replies],
    )
    monkeypatch.setattr(benchmarks_ep, "_get_evaluation_manager_for_user", lambda _user: _Manager())

    app = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/benchmarks/bullshit_benchmark/run",
            json={"limit": 3, "api_name": "openai", "parallel": 3, "save_results": True},
        )
    assert response.status_code == 200, response.text
    summary = response.json()["results_summary"]
    assert [summary[k] for k in ("total_evaluated", "successful", "failed", "unscored")] == [3, 1, 1, 1]
    assert [summary[k] for k in ("average_score", "min_score", "max_score")] == [0.8, 0.8, 0.8]
    assert summary["by_category"] == {"d": {"count": 1, "average": 0.8}}
    assert _Manager.store_evaluation.await_args.kwargs["results"]["scores"] == [0.8]


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


@pytest.mark.parametrize("own_key", [True, False], ids=["own-key", "server-key"])
def test_benchmark_run_names_its_provider_to_the_llm_quota(monkeypatch, own_key):
    """Benchmark runs check the LLM quota at the request boundary, as evaluation runs do (TASK-13638), naming the
    run's provider: on the user's own key the run goes on and its items carry that key; on the server's key it is
    refused before the dataset loads (TASK-15426)."""
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work, users_own_key

    refuse_llm_work(monkeypatch)
    asked = users_own_key(monkeypatch, "openai" if own_key else "anthropic")
    loaded, seen = [], []
    item = {"name": "m", "description": "d", "evaluation_prompt": "{q}", "input_data": {"q": "q"}, "scoring_criteria": {}}
    monkeypatch.setattr(
        benchmarks_ep,
        "get_registry",
        lambda: SimpleNamespace(
            get=lambda _name: SimpleNamespace(evaluation_type="demo"),
            create_evaluator=lambda _name: SimpleNamespace(format_for_custom_metric=lambda _item: item),
        ),
    )
    monkeypatch.setattr(benchmarks_ep, "load_benchmark_dataset", lambda *_a, **_k: loaded.append(True) or [{"q": "q"}])

    async def _evaluate(**kwargs):
        seen.append((kwargs["provider_credentials"].api_key, current_llm_call_context().prechecked))
        return {"score": 0.8, "explanation": "ok"}

    manager = SimpleNamespace(evaluate_custom_metric=_evaluate)
    monkeypatch.setattr(benchmarks_ep, "_get_evaluation_manager_for_user", lambda _identity: manager)

    app = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/benchmarks/bullshit_benchmark/run",
            json={"limit": 1, "api_name": "openai", "parallel": 1, "save_results": False},
        )

    assert asked == ["openai"]
    if own_key:
        assert response.status_code == 200, response.text
        assert seen == [("sk-users-own-key", False)]
    else:
        assert_llm_quota_402(response)
        assert response.json()["detail"]["error"] == "limit_exceeded"
        assert loaded == []
