from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_ready_endpoint_returns_503_when_draining() -> None:
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.services.app_lifecycle import get_or_create_lifecycle_state

    with TestClient(app) as client:
        state = get_or_create_lifecycle_state(app)
        state.phase = "draining"
        state.ready = False
        state.draining = True

        response = client.get("/ready")

    if response.status_code != 503:
        raise AssertionError(f"expected 503, got {response.status_code}")
    payload = response.json()
    if payload["reason"] != "shutdown_in_progress":
        raise AssertionError(f"unexpected reason: {payload['reason']!r}")


@pytest.mark.integration
def test_ready_endpoint_returns_non_success_when_dependency_raises(monkeypatch) -> None:
    from tldw_Server_API.app.main import app

    def _fake_get_db_pool() -> object:
        raise RuntimeError("boom")

    with TestClient(app) as client:
        monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _fake_get_db_pool)
        response = client.get("/ready")

    payload = response.json()
    if response.status_code != 503:
        raise AssertionError(f"expected 503, got {response.status_code}")
    if payload["status"] != "not_ready":
        raise AssertionError(f"unexpected status: {payload['status']!r}")
    if payload["reason"] != "dependency_check_failed":
        raise AssertionError(f"unexpected reason: {payload['reason']!r}")
    if "error" in payload:
        raise AssertionError(f"readiness probe should not echo raw errors: {payload!r}")


@pytest.mark.integration
def test_ready_endpoint_returns_503_for_import_error(monkeypatch) -> None:
    from tldw_Server_API.app.main import app

    def _fake_create_workflows_database(*args, **kwargs) -> object:
        raise ImportError("missing readiness dependency")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.DB_Manager.create_workflows_database",
        _fake_create_workflows_database,
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    payload = response.json()
    if response.status_code != 503:
        raise AssertionError(f"expected 503, got {response.status_code}")
    if payload["status"] != "not_ready":
        raise AssertionError(f"unexpected status: {payload['status']!r}")
    if payload["reason"] != "dependency_check_failed":
        raise AssertionError(f"unexpected reason: {payload['reason']!r}")
    if "error" in payload:
        raise AssertionError(f"readiness probe should not echo raw errors: {payload!r}")
