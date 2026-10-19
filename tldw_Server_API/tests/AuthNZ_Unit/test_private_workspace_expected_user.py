"""Unit HTTP checks for captured private-workspace identity before database access."""

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps, kanban_deps
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import admin, claims
from tldw_Server_API.app.api.v1.endpoints.admin.admin_system import get_db_transaction

pytestmark = pytest.mark.unit


@pytest.fixture(params=["claims", "kanban", "admin"])
def workspace_request(request, monkeypatch):
    """Use real route/dependency guards with a recorded database-admission sentinel."""
    app = FastAPI()
    accesses = []

    async def principal(_request):
        return SimpleNamespace(user_id=84)

    async def database():
        accesses.append("database")
        raise HTTPException(409, "Database admission reached")

    monkeypatch.setattr(auth_deps, "get_auth_principal", principal)
    app.dependency_overrides[auth_deps.get_request_user] = lambda: SimpleNamespace(id=84)
    if request.param == "claims":
        app.include_router(claims.router)
        app.dependency_overrides[get_media_db_for_user] = database
        path = "/claims"
    elif request.param == "admin":
        app.include_router(admin.router)
        # Authorization is an independent unit boundary; exercise the real owner guard.
        for dependency in admin.router.dependencies:
            if dependency.dependency is not auth_deps.require_expected_user_if_present:
                app.dependency_overrides[dependency.dependency] = lambda: None
        app.dependency_overrides[get_db_transaction] = database
        path = "/admin/stats"
    else:

        async def kanban_database(_user_id):
            return await database()

        monkeypatch.setattr(kanban_deps, "_get_or_init_db_instance", kanban_database)
        app.dependency_overrides[kanban_deps.get_request_user] = lambda: SimpleNamespace(id=84)

        @app.get("/kanban")
        async def kanban_probe(_db=Depends(kanban_deps.get_kanban_db_for_user)):
            return {}

        path = "/kanban"
    with TestClient(app) as client:
        yield client, path, accesses


def test_changed_owner_is_rejected_before_database_access(workspace_request):
    client, path, accesses = workspace_request
    response = client.get(path, headers={"X-TLDW-Expected-User-ID": "42"})
    assert response.status_code == 412  # nosec B101 # pytest ownership contract
    assert response.json()["detail"]["code"] == "request_config_scope_changed"  # nosec B101
    assert accesses == []  # nosec B101


@pytest.mark.parametrize("expected_owner", [None, "84"])
def test_matching_and_legacy_requests_reach_database_admission(workspace_request, expected_owner):
    client, path, accesses = workspace_request
    headers = {} if expected_owner is None else {"X-TLDW-Expected-User-ID": expected_owner}
    response = client.get(path, headers=headers)
    assert response.status_code == 409  # nosec B101 # sentinel proves unchanged admission
    assert accesses == ["database"]  # nosec B101
