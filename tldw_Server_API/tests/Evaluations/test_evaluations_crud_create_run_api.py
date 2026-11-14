import os
from typing import Tuple

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Evaluations.audit_adapter import MandatoryAuditWriteError


@pytest.fixture()
def evals_crud_client() -> Tuple[TestClient, dict]:
    os.environ.setdefault("AUTH_MODE", "single_user")
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("TEST_MODE", "true")

    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_crud import crud_router
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    app = FastAPI()
    app.include_router(crud_router, prefix="/api/v1/evaluations")

    api_key = get_settings().SINGLE_USER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    client = TestClient(app)
    return client, headers


@pytest.mark.integration
def test_create_run_uses_typed_model(evals_crud_client, monkeypatch):
    client, headers = evals_crud_client

    # Patch evaluation service with minimal stub
    class _SvcStub:
        async def get_evaluation(self, eval_id, **_kwargs):
            return None

        async def create_run(
            self,
            eval_id,
            target_model,
            config=None,
            dataset_override=None,
            webhook_url=None,
            created_by=None,
            webhook_user_id=None,
            provider_credentials=None,
        ):
            return {
                "id": "run_123",
                "object": "run",
                "eval_id": eval_id,
                "status": "pending",
                "target_model": target_model or "default-model",
                "created": 1700000000,
            }

    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_crud as crud
    monkeypatch.setattr(crud, "get_unified_evaluation_service_for_user", lambda uid: _SvcStub())

    payload = {
        "target_model": "gpt-4o-mini",
        "config": {"max_workers": 2},
    }
    r = client.post("/api/v1/evaluations/eval_abc/runs", json=payload, headers=headers)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["object"] == "run"
    assert body["eval_id"] == "eval_abc"
    assert body["target_model"] == "gpt-4o-mini"


@pytest.mark.integration
def test_create_run_forbids_extra_keys(evals_crud_client, monkeypatch):
    client, headers = evals_crud_client

    # No need to patch service; validation happens before invocation
    payload = {
        "target_model": "x",
        "config": {},
        "extra_key": True,
    }
    r = client.post("/api/v1/evaluations/e1/runs", json=payload, headers=headers)
    # Pydantic extra='forbid' should 422 on extra keys
    assert r.status_code == 422


@pytest.mark.integration
def test_create_run_returns_503_on_mandatory_audit_failure(evals_crud_client, monkeypatch):
    client, headers = evals_crud_client

    class _SvcStub:
        async def get_evaluation(self, eval_id, **_kwargs):
            return None

        async def create_run(self, *args, **kwargs):
            raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_crud as crud

    monkeypatch.setattr(crud, "get_unified_evaluation_service_for_user", lambda uid: _SvcStub())

    response = client.post(
        "/api/v1/evaluations/eval_abc/runs",
        json={"target_model": "gpt-4o-mini"},
        headers=headers,
    )

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["error"]["message"] == "Mandatory audit persistence unavailable"
    assert body["detail"]["error"]["type"] == "service_unavailable"
    assert body["detail"]["error"]["code"] == "audit_persistence_failure"


_GRADED = {"eval_type": "model_graded", "eval_spec": {"sub_type": "summarization"}}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("evaluation", "config", "stored", "admitted", "asks"),
    [
        (_GRADED, {"api_name": "openai"}, "openai", True, ["openai"]),
        (_GRADED, {"api_name": "openai"}, "anthropic", False, ["openai"]),
        (_GRADED, {"api_name": "openai", "api_key": "sk-raw"}, "openai", False, []),
        ({"eval_type": "model_graded", "eval_spec": {"sub_type": "rag_pipeline"}}, {}, "openai", False, []),
        ({"eval_type": "exact_match", "eval_spec": {}}, {}, "openai", False, []),
    ],
    ids=["own-key", "server-key", "raw-api-key", "rag-pipeline", "no-llm"],
)
def test_create_run_names_its_judge_provider_to_the_llm_quota(
    evals_crud_client, monkeypatch, evaluation, config, stored, admitted, asks
):
    """A run's samples are LLM calls made in a task the request starts, so the run is checked at creation (§4a).

    Past a limit, a run whose judge (named by the run config here) runs on the user's own key starts on that key,
    unmarked. A run on the server's key, on a raw ``api_key``, a ``rag_pipeline`` run, or one that makes no LLM
    call is refused before it starts (TASK-15426).
    """
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
    from tldw_Server_API.tests.helpers.llm_quota import assert_llm_quota_402, refuse_llm_work, users_own_key

    client, headers = evals_crud_client
    started: list[tuple] = []

    class _SvcStub:
        async def get_evaluation(self, eval_id, **_kwargs):
            return evaluation

        async def create_run(self, eval_id, target_model, provider_credentials=None, **_kwargs):
            started.append((getattr(provider_credentials, "api_key", None), current_llm_call_context().prechecked))
            return {"id": "run_1", "object": "run", "eval_id": eval_id, "status": "pending",
                    "target_model": target_model, "created": 1700000000}

    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_crud as crud

    monkeypatch.setattr(crud, "get_unified_evaluation_service_for_user", lambda uid: _SvcStub())
    refuse_llm_work(monkeypatch)
    asked = users_own_key(monkeypatch, stored)

    r = client.post(
        "/api/v1/evaluations/eval_abc/runs", json={"target_model": "gpt-4o-mini", "config": config}, headers=headers
    )

    assert asked == asks
    if admitted:
        assert r.status_code == 202, r.text
        assert started == [("sk-users-own-key", False)]
    else:
        assert_llm_quota_402(r)
        assert started == []
