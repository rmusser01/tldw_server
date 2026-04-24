from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_datasets as eval_datasets
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_MANAGE, EVALS_READ


def test_health_public_and_metrics_private(monkeypatch):
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    class _Service:
        async def health_check(self):
            return {
                "status": "healthy",
                "version": "test",
                "uptime": 1,
                "database": "connected",
            }

        async def get_metrics_summary(self):
            return {"total_requests": 7}

    monkeypatch.setattr(eval_unified, "get_unified_evaluation_service_for_user", lambda _uid: _Service())

    with TestClient(app) as client:
        health_response = client.get("/api/v1/evaluations/health")
        assert health_response.status_code == 200, health_response.text
        assert health_response.json()["status"] == "healthy"

        unauth_metrics = client.get("/api/v1/evaluations/metrics")
        assert unauth_metrics.status_code == 401

    current_permissions: list[str] = []

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            permissions=list(current_permissions),
            is_admin=False,
        )

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override

    with TestClient(app) as client:
        current_permissions[:] = []
        forbidden_metrics = client.get("/api/v1/evaluations/metrics")
        assert forbidden_metrics.status_code == 403

        current_permissions[:] = [EVALS_READ]
        allowed_metrics = client.get("/api/v1/evaluations/metrics")
        assert allowed_metrics.status_code == 200, allowed_metrics.text
        assert allowed_metrics.json()["total_requests"] == 7


def test_dataset_routes_require_read_vs_manage_permissions(monkeypatch):
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    stored_datasets = {
        "ds_1": {
            "id": "ds_1",
            "object": "dataset",
            "name": "seed_dataset",
            "description": "seed",
            "sample_count": 1,
            "samples": [{"input": {"text": "hello"}, "expected": "hello", "metadata": {}}],
            "created": 1700000000,
            "created_at": 1700000000,
            "created_by": "1",
            "metadata": {},
        }
    }

    class _FakeDB:
        def lookup_idempotency(self, _scope, _idempotency_key, _user_id):
            return None

        def record_idempotency(self, _scope, _idempotency_key, _resource_id, _user_id):
            return None

        def list_datasets(self, *, limit, offset, created_by):
            _ = (limit, offset, created_by)
            return [stored_datasets["ds_1"]], False

        def get_dataset(
            self,
            dataset_id,
            created_by,
            include_samples=True,
            sample_limit=None,
            sample_offset=0,
        ):
            _ = (created_by, include_samples, sample_limit, sample_offset)
            return stored_datasets.get(dataset_id)

    class _Service:
        def __init__(self):
            self.db = _FakeDB()

        async def create_dataset(self, *, name, samples, description, metadata, created_by):
            stored_datasets["ds_2"] = {
                "id": "ds_2",
                "object": "dataset",
                "name": name,
                "description": description,
                "sample_count": len(samples),
                "samples": samples,
                "created": 1700000100,
                "created_at": 1700000100,
                "created_by": created_by,
                "metadata": metadata or {},
            }
            return "ds_2"

        async def get_dataset(self, dataset_id, created_by):
            _ = created_by
            return stored_datasets.get(dataset_id)

        async def delete_dataset(self, dataset_id, *, deleted_by, created_by):
            _ = (deleted_by, created_by)
            return stored_datasets.pop(dataset_id, None) is not None

    monkeypatch.setattr(eval_datasets, "get_unified_evaluation_service_for_user", lambda _uid: _Service())

    current_permissions: list[str] = []

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            permissions=list(current_permissions),
            is_admin=False,
        )

    app.dependency_overrides[eval_datasets.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_datasets.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override

    dataset_body = {
        "name": "new_ds",
        "description": "for tests",
        "samples": [{"input": {"text": "foo"}, "expected": "bar", "metadata": {}}],
        "metadata": {"k": "v"},
    }

    with TestClient(app) as client:
        current_permissions[:] = [EVALS_READ]
        list_resp = client.get("/api/v1/evaluations/datasets")
        assert list_resp.status_code == 200, list_resp.text
        get_resp = client.get("/api/v1/evaluations/datasets/ds_1")
        assert get_resp.status_code == 200, get_resp.text

        create_forbidden = client.post("/api/v1/evaluations/datasets", json=dataset_body)
        assert create_forbidden.status_code == 403
        delete_forbidden = client.delete("/api/v1/evaluations/datasets/ds_1")
        assert delete_forbidden.status_code == 403

        current_permissions[:] = [EVALS_READ, EVALS_MANAGE]
        create_ok = client.post("/api/v1/evaluations/datasets", json=dataset_body)
        assert create_ok.status_code == 201, create_ok.text
        assert create_ok.json()["id"] == "ds_2"

        delete_ok = client.delete("/api/v1/evaluations/datasets/ds_1")
        assert delete_ok.status_code == 204, delete_ok.text
