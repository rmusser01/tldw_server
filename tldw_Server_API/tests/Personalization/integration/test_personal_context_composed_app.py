"""Production-assembly integration coverage for Personal Context routes."""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit

pytestmark = pytest.mark.integration

_OPERATIONS = {
    "/api/v1/personal-context/status": {"get"},
    "/api/v1/personal-context/manifest": {"get", "post"},
    "/api/v1/personal-context/scopes": {"get"},
    "/api/v1/personal-context/scopes/workspace": {"post"},
    "/api/v1/personal-context/records": {"get", "post"},
    "/api/v1/personal-context/records/{record_id}": {"get", "patch", "delete"},
    "/api/v1/personal-context/records/{record_id}/archive": {"post"},
    "/api/v1/personal-context/records/{record_id}/restore": {"post"},
    "/api/v1/personal-context/proposals": {"get", "post"},
    "/api/v1/personal-context/proposals/{proposal_id}/review": {"post"},
    "/api/v1/personal-context/runtime": {"get", "patch"},
    "/api/v1/personal-context/export": {"post"},
    "/api/v1/personal-context/purge": {"post"},
}


def test_composed_application_exposes_every_personal_context_contract() -> None:
    """Every route must be present in the production app with a modeled success body."""

    from tldw_Server_API.app.main import app

    app.openapi_schema = None
    paths = app.openapi()["paths"]
    for path, methods in _OPERATIONS.items():
        assert methods <= paths[path].keys()
        for method in methods:
            responses = paths[path][method]["responses"]
            success = next(
                response
                for code, response in responses.items()
                if 200 <= int(code) < 300
            )
            assert "schema" in success["content"]["application/json"]


def test_composed_status_uses_auth_middleware_and_rate_guard(
    tmp_path,
    monkeypatch,
) -> None:
    """The real app must authenticate status calls and honor the route rate guard."""

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    )
    from tldw_Server_API.app.main import app

    api_key = os.environ["SINGLE_USER_TEST_API_KEY"]
    with TestClient(app) as client:
        unauthenticated = client.get("/api/v1/personal-context/status")
        assert unauthenticated.status_code == 401

        response = client.get(
            "/api/v1/personal-context/status",
            headers={"X-API-KEY": api_key},
        )
        assert response.status_code == 200
        assert response.json() == {
            "state": "absent",
            "profile_id": None,
            "revision": None,
            "purge_generation": None,
        }

        async def deny_rate_limit() -> None:
            raise HTTPException(status_code=429, detail="test rate limit")

        app.dependency_overrides[check_rate_limit] = deny_rate_limit
        try:
            limited = client.get(
                "/api/v1/personal-context/status",
                headers={"X-API-KEY": api_key},
            )
        finally:
            app.dependency_overrides.pop(check_rate_limit, None)
        assert limited.status_code == 429
        assert limited.json()["detail"] == "test rate limit"
