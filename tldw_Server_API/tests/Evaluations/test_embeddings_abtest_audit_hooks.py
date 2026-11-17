import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evals_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_embeddings_abtest as abtest_endpoints
import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified as evals_endpoints
import tldw_Server_API.app.core.Evaluations.embeddings_abtest_service as abtest_service


@pytest.mark.integration
def test_abtest_audit_hooks_emitted(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", str(tmp_path / "evals.db"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_db"))
    reset_settings()

    calls = {"create": 0, "run": 0, "delete": 0, "export": 0}

    def _bump(key):
        def _inner(*args, **kwargs):
            calls[key] += 1
        return _inner

    monkeypatch.setattr("tldw_Server_API.app.core.Evaluations.audit_adapter.log_evaluation_created", _bump("create"))
    monkeypatch.setattr("tldw_Server_API.app.core.Evaluations.audit_adapter.log_run_started", _bump("run"))
    monkeypatch.setattr("tldw_Server_API.app.core.Evaluations.audit_adapter.log_evaluation_deleted", _bump("delete"))
    monkeypatch.setattr("tldw_Server_API.app.core.Evaluations.audit_adapter.log_evaluation_exported", _bump("export"))
    monkeypatch.setattr(abtest_service, "cleanup_abtest_resources", lambda *args, **kwargs: {"abtests_deleted": 0})

    app = FastAPI()
    app.include_router(evals_router, prefix="/api/v1")
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        is_admin=True,
    )

    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY, "Content-Type": "application/json"}

    payload = {
        "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
        "media_ids": [],
        "retrieval": {"k": 3, "search_mode": "vector"},
        "queries": [{"text": "hello"}],
        "metric_level": "media",
    }

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/evaluations/embeddings/abtest",
            json={"name": "audit-test", "config": payload},
            headers=headers,
        )
        assert created.status_code == 200, created.text
        test_id = created.json()["test_id"]
        assert calls["create"] == 1

        run_resp = client.post(
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/run",
            json={"config": payload},
            headers=headers,
        )
        assert run_resp.status_code == 200, run_resp.text
        assert calls["run"] == 1

        export_resp = client.get(
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/export?format=json",
            headers=headers,
        )
        assert export_resp.status_code == 200, export_resp.text
        assert calls["export"] == 1

        delete_resp = client.delete(
            f"/api/v1/evaluations/embeddings/abtest/{test_id}",
            headers=headers,
        )
        assert delete_resp.status_code == 200, delete_resp.text
        assert calls["delete"] == 1

    reset_settings()
