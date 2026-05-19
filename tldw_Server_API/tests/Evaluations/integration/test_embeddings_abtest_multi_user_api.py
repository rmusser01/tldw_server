import pytest
from fastapi import Depends, FastAPI, Header, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth as eval_auth
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_embeddings_abtest as abtest_ep
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evals_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module

pytestmark = [pytest.mark.integration]


def _abtest_create_payload() -> dict:
    return {
        "name": "tenant-scope-abtest",
        "config": {
            "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
            "media_ids": [],
            "retrieval": {"k": 3, "search_mode": "vector"},
            "queries": [{"text": "hello"}],
            "metric_level": "media",
        },
    }


def test_dedicated_abtest_create_uses_string_tenant_scope_db(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    service_module._service_instance = None
    try:
        service_module._service_instances_by_user.clear()
    except Exception:
        service_module._service_instances_by_user = {}  # type: ignore[assignment]

    app = FastAPI()
    app.include_router(evals_router, prefix="/api/v1")

    async def _verify_api_key(
        request: Request,
        x_api_key: str = Header(None, alias="X-API-KEY"),
    ) -> str:
        _ = x_api_key
        return request.headers.get("X-User-Scope", "tenant-user")

    class _EvalUser:
        id = 7
        id_str = "tenant-user"
        id_int = 7
        roles = ["admin"]
        permissions = ["system.configure", "evals.read", "evals.manage"]
        is_admin = True

    async def _get_eval_request_user(
        _user_ctx: str = Depends(_verify_api_key),
    ) -> _EvalUser:
        return _EvalUser()

    async def _rate_limit_override():
        return None

    async def _admin_principal() -> AuthPrincipal:
        return AuthPrincipal(
            kind="user",
            user_id=7,
            username="tenant-user",
            subject="user:tenant-user",
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
        )

    app.dependency_overrides[eval_auth.verify_api_key] = _verify_api_key
    app.dependency_overrides[eval_auth.get_eval_request_user] = _get_eval_request_user
    app.dependency_overrides[eval_auth.check_evaluation_rate_limit] = _rate_limit_override
    app.dependency_overrides[abtest_ep.verify_api_key] = _verify_api_key
    app.dependency_overrides[abtest_ep.get_eval_request_user] = _get_eval_request_user
    app.dependency_overrides[abtest_ep.check_evaluation_rate_limit] = _rate_limit_override
    app.dependency_overrides[get_auth_principal] = _admin_principal

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/embeddings/abtest",
            json=_abtest_create_payload(),
            headers={
                "Authorization": "Bearer test-token",
                "X-API-KEY": "test",
                "X-User-Scope": "tenant-user",
            },
        )
        assert response.status_code == 200, response.text
        test_id = response.json()["test_id"]

    tenant_db = EvaluationsDatabase(str(DatabasePaths.get_evaluations_db_path("tenant-user")))
    numeric_db = EvaluationsDatabase(str(DatabasePaths.get_evaluations_db_path(7)))
    assert tenant_db.get_abtest(test_id, created_by="tenant-user") is not None
    assert numeric_db.get_abtest(test_id, created_by="tenant-user") is None

    app.dependency_overrides.clear()
    service_module._service_instance = None
    try:
        service_module._service_instances_by_user.clear()
    except Exception:
        service_module._service_instances_by_user = {}  # type: ignore[assignment]
