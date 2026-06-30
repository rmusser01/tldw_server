import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evals_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import (
    get_unified_evaluation_service_for_user,
)


@pytest.mark.integration
def test_abtest_results_endpoint_returns_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", str(tmp_path / "evals.db"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_db"))
    reset_settings()

    app = FastAPI()
    app.include_router(evals_router, prefix="/api/v1")
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="user",
        user_id=1,
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
            json={"name": "results-test", "config": payload},
            headers=headers,
        )
        assert created.status_code == 200, created.text
        test_id = created.json()["test_id"]

        svc = get_unified_evaluation_service_for_user(1)
        arms = svc.db.get_abtest_arms(test_id)
        queries = svc.db.get_abtest_queries(test_id)
        assert arms and queries
        arm_id = arms[0]["arm_id"]
        query_id = queries[0]["query_id"]
        svc.db.insert_abtest_result(
            test_id=test_id,
            arm_id=arm_id,
            query_id=query_id,
            ranked_ids=["m1", "m2"],
            scores=[0.9, 0.8],
            metrics={"ndcg": 0.5},
            latency_ms=12.5,
            ranked_distances=[0.1, 0.2],
            ranked_metadatas=[{"title": "a"}, {"title": "b"}],
            ranked_documents=["doc1", "doc2"],
            rerank_scores=[0.7, 0.6],
        )

        resp = client.get(
            f"/api/v1/evaluations/embeddings/abtest/{test_id}/results",
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 50,
            "total": 1,
            "total_pages": 1,
            "has_more": False,
        }
        assert len(data["results"]) == 1
        row = data["results"][0]
        assert row["arm_id"] == arm_id
        assert row["ranked_ids"] == ["m1", "m2"]
        assert row["metrics"]["ndcg"] == 0.5

    reset_settings()
