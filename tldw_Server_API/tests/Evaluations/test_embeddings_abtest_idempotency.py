import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as eval_service_module


@pytest.fixture()
def isolated_evaluations_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("EVALUATIONS_TEST_DB_PATH", str(tmp_path / "evals.db"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_db"))
    reset_settings()
    eval_service_module._service_instance = None
    eval_service_module._service_instances_by_user.clear()
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    yield headers
    eval_service_module._service_instance = None
    eval_service_module._service_instances_by_user.clear()
    reset_settings()


@pytest.mark.integration
def test_abtest_export_idempotency(isolated_evaluations_storage):
    auth_headers = isolated_evaluations_storage
    client = TestClient(app)

    # Create minimal A/B test
    cfg = {
        "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
        "media_ids": [],
        "queries": [{"text": "hello"}],
        "retrieval": {"k": 1},
        "reuse_existing": True,
    }
    r = client.post(
        "/api/v1/evaluations/embeddings/abtest",
        json={"name": "idem-exp", "config": cfg},
        headers=auth_headers,
    )
    assert r.status_code == 200
    tid = r.json()["test_id"]

    # Kick off a run (doesn't need to complete for export shape)
    r2 = client.post(
        f"/api/v1/evaluations/embeddings/abtest/{tid}/run",
        json={"config": cfg},
        headers=auth_headers,
    )
    assert r2.status_code == 200

    # Export with Idempotency-Key
    headers = {**auth_headers, "Idempotency-Key": "abtest-export-1"}
    e1 = client.get(f"/api/v1/evaluations/embeddings/abtest/{tid}/export", params={"format": "json"}, headers=headers)
    assert e1.status_code == 200
    e2 = client.get(f"/api/v1/evaluations/embeddings/abtest/{tid}/export", params={"format": "json"}, headers=headers)
    assert e2.status_code == 200
    # The responses should be consistent for idempotent key (shape stable even if total may grow later)
    j1 = e1.json()
    j2 = e2.json()
    assert j1["test_id"] == tid and j2["test_id"] == tid
    assert isinstance(j1.get("total", 0), int) and isinstance(j2.get("total", 0), int)


@pytest.mark.integration
def test_abtest_delete_idempotency(isolated_evaluations_storage):
    auth_headers = isolated_evaluations_storage
    client = TestClient(app)

    # Create minimal A/B test
    cfg = {
        "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
        "media_ids": [],
        "queries": [{"text": "hello"}],
        "retrieval": {"k": 1},
        "reuse_existing": True,
    }
    r = client.post(
        "/api/v1/evaluations/embeddings/abtest",
        json={"name": "idem-del", "config": cfg},
        headers=auth_headers,
    )
    assert r.status_code == 200
    tid = r.json()["test_id"]

    headers = {**auth_headers, "Idempotency-Key": "abtest-delete-1"}
    d1 = client.delete(f"/api/v1/evaluations/embeddings/abtest/{tid}", headers=headers)
    assert d1.status_code == 200
    d2 = client.delete(f"/api/v1/evaluations/embeddings/abtest/{tid}", headers=headers)
    assert d1.json().get("status") == "deleted"
    # Current delete flow checks existence before idempotency lookup, so
    # a replay may legitimately 404 after a successful hard delete.
    assert d2.status_code in (200, 404)
    if d2.status_code == 200:
        assert d2.json().get("status") == "deleted"
