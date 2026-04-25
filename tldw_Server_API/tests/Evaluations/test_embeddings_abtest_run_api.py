import json
import os
from typing import Tuple

import pytest
from fastapi import Response
from fastapi.testclient import TestClient


@pytest.fixture()
def evals_client() -> Tuple[TestClient, dict]:
    """Provide a minimal FastAPI app mounting the evaluations router directly.

    Avoid main.py route gating by including the router here.
    """
    os.environ.setdefault("AUTH_MODE", "single_user")
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("TEST_MODE", "true")

    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evals_router
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    app = FastAPI()
    app.include_router(evals_router, prefix="/api/v1")

    api_key = get_settings().SINGLE_USER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    client = TestClient(app)
    return client, headers


@pytest.mark.integration
def test_run_embeddings_abtest_synchronous_success(evals_client, monkeypatch):
    client, headers = evals_client

    # Patch evaluation service and runner to no-op
    class _DBStub:
        def get_abtest(self, test_id, created_by=None):
            return {"test_id": test_id, "created_by": created_by or "tester"}
        def lookup_idempotency(self, *a, **kw):
            return None
        def record_idempotency(self, *a, **kw):
            return None
        def set_abtest_status(self, *a, **kw):
            return None

    class _SvcStub:
        def __init__(self):
            self.db = _DBStub()

    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_embeddings_abtest as ab
    monkeypatch.setattr(ab, "get_unified_evaluation_service_for_user", lambda uid: _SvcStub())

    async def _fake_run_abtest_full(db, cfg, test_id, user_id, media_db):
        return None

    monkeypatch.setattr(ab, "run_abtest_full", _fake_run_abtest_full)

    payload = {
        "config": {
            "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
            "media_ids": [],
            "retrieval": {"k": 5, "search_mode": "vector"},
            "queries": [{"text": "hello"}],
            "metric_level": "media",
            "reuse_existing": True
        }
    }
    # Request should complete synchronously in TESTING mode
    r = client.post("/api/v1/evaluations/embeddings/abtest/mytest/run", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["test_id"] == "mytest"
    assert body["status"] in ("completed", "running")
    # In TESTING env this should be completed
    if body["status"] == "completed":
        assert body.get("progress", {}).get("phase") == 1.0


@pytest.mark.integration
def test_run_embeddings_abtest_treats_testing_y_as_synchronous(evals_client, monkeypatch):
    client, headers = evals_client
    monkeypatch.setenv("TESTING", "y")

    class _DBStub:
        def get_abtest(self, test_id, created_by=None):
            return {"test_id": test_id, "created_by": created_by or "tester"}
        def lookup_idempotency(self, *a, **kw):
            return None
        def record_idempotency(self, *a, **kw):
            return None
        def set_abtest_status(self, *a, **kw):
            return None

    class _SvcStub:
        def __init__(self):
            self.db = _DBStub()

    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_embeddings_abtest as ab

    monkeypatch.setattr(ab, "get_unified_evaluation_service_for_user", lambda uid: _SvcStub())

    def _unexpected_jobs_manager():
        raise AssertionError("jobs path should not run")

    monkeypatch.setattr(ab, "abtest_jobs_manager", _unexpected_jobs_manager)

    calls = {"run": 0}

    async def _fake_run_abtest_full(db, cfg, test_id, user_id, media_db):
        calls["run"] += 1
        return None

    monkeypatch.setattr(ab, "run_abtest_full", _fake_run_abtest_full)

    payload = {
        "config": {
            "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
            "media_ids": [],
            "retrieval": {"k": 5, "search_mode": "vector"},
            "queries": [{"text": "hello"}],
            "metric_level": "media",
            "reuse_existing": True,
        }
    }
    r = client.post("/api/v1/evaluations/embeddings/abtest/mytest/run", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    assert calls["run"] == 1


@pytest.mark.asyncio
async def test_run_embeddings_abtest_idempotent_replay_uses_stored_normalized_status(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_embeddings_abtest as ab
    from tldw_Server_API.app.api.v1.schemas.embeddings_abtest_schemas import EmbeddingsABTestRunRequest
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    class _DBStub:
        def get_abtest(self, test_id, created_by=None):
            return {
                "test_id": test_id,
                "created_by": created_by or "tenant-user",
                "status": "canceled",
                "stats_json": json.dumps({"progress": {"phase": 0.75}}),
            }

        def lookup_idempotency(self, *args, **kwargs):
            _ = (args, kwargs)
            return "mytest"

    class _SvcStub:
        def __init__(self):
            self.db = _DBStub()

    monkeypatch.setattr(ab, "enforce_heavy_evaluations_admin", lambda _principal: None)
    monkeypatch.setattr(ab, "get_unified_evaluation_service_for_user", lambda _user_id: _SvcStub())

    result = await ab.run_embeddings_abtest(
        test_id="mytest",
        payload=EmbeddingsABTestRunRequest(
            config={
                "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
                "media_ids": [],
                "retrieval": {"k": 5, "search_mode": "vector"},
                "queries": [{"text": "hello"}],
            }
        ),
        user_ctx="tenant-user",
        _=None,
        __=None,
        media_db=None,
        principal=object(),
        current_user=User(id="tenant-user", username="tester", email=None, is_active=True),
        idempotency_key="idem-run-1",
        response=Response(),
    )

    assert result.status == "cancelled"
    assert result.progress == {"phase": 0.75}
