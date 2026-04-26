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
        async def create_run(
            self,
            eval_id,
            target_model,
            config=None,
            dataset_override=None,
            webhook_url=None,
            created_by=None,
            webhook_user_id=None,
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
