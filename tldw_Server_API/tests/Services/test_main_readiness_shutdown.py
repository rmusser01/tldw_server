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
    from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
    from tldw_Server_API.app.main import app

    async def _fake_get_db_pool() -> object:
        raise DatabaseError("private database failure")

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
def test_ready_endpoint_sanitizes_unhealthy_database_payload(monkeypatch) -> None:
    from tldw_Server_API.app.main import app

    marker = "postgresql://admin:secret@private-host/authnz"

    class _Pool:
        async def health_check(self) -> dict[str, str]:
            return {
                "status": "unhealthy",
                "type": "postgresql",
                "error": marker,
            }

    async def _fake_get_db_pool() -> _Pool:
        return _Pool()

    with TestClient(app) as client:
        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            _fake_get_db_pool,
        )
        response = client.get("/ready")

    payload = response.json()
    assert response.status_code == 503
    assert payload["database"] == {
        "status": "unhealthy",
        "type": "postgresql",
        "error": "database_unavailable",
    }
    assert marker not in response.text
    assert "secret" not in response.text


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
