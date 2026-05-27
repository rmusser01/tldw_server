from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import chain

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType


def _user_dep(user_id: int = 1, *, is_admin: bool = False):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    roles = ["admin"] if is_admin else ["user"]
    return lambda: User(
        id=user_id,
        username=f"user-{user_id}",
        roles=roles,
        is_admin=is_admin,
    )


def _client(monkeypatch, *, user_id: int = 1, is_admin: bool = False) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        pass

    from tldw_Server_API.app.api.v1.endpoints import sandbox_workspace_diagnostics as sb

    app = FastAPI()
    app.include_router(sb.router, prefix="/api/v1")
    app.dependency_overrides[sb.get_request_user] = _user_dep(
        user_id,
        is_admin=is_admin,
    )
    return TestClient(app)


def _sandbox_api_client(monkeypatch, *, user_id: int = 1, is_admin: bool = False) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        pass

    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb_api
    from tldw_Server_API.app.api.v1.endpoints import sandbox_workspace_diagnostics as sb_diag

    app = FastAPI()
    app.include_router(sb_api.router, prefix="/api/v1")
    app.include_router(sb_diag.router, prefix="/api/v1")
    user_dep = _user_dep(user_id, is_admin=is_admin)
    app.dependency_overrides[sb_api.get_request_user] = user_dep
    app.dependency_overrides[sb_diag.get_request_user] = user_dep
    app.dependency_overrides[sb_api.get_audit_service_for_user] = lambda: None
    return TestClient(app)


def _seed_run(
    run_id: str,
    user_id: int,
    *,
    workspace_id: str,
    workspace_group_id: str | None = None,
    scope_snapshot_id: str | None = None,
    phase: str = "completed",
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox_workspace_diagnostics as sb

    st = RunStatus(
        id=run_id,
        phase=RunPhase(phase),
        spec_version="1.0",
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        exit_code=0,
        started_at=(datetime.now(timezone.utc) - timedelta(seconds=120)),
        finished_at=datetime.now(timezone.utc),
        message="ok",
        image_digest="workspace-diagnostics-digest",
        policy_hash="deadbeefcafebabe",
        workspace_id=workspace_id,
        workspace_group_id=workspace_group_id,
        scope_snapshot_id=scope_snapshot_id,
    )
    sb._service._orch._store.put_run(user_id, st)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_workspace_diagnostics_route_survives_disabled_sandbox_admin_policy(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.admin import iter_admin_router_specs
    from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
    from tldw_Server_API.app.api.v1.router_registry import register_router_specs

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda route_key, default_stable=True: route_key != "sandbox" and default_stable,
    )

    app = FastAPI()
    selected_specs = [
        spec
        for spec in chain(iter_content_router_specs(), iter_admin_router_specs())
        if spec.name in {"sandbox", "sandbox_workspace_diagnostics"}
    ]
    register_router_specs(app, selected_specs)

    assert "/api/v1/sandbox/workspaces/{workspace_id}/diagnostics" in {
        getattr(route, "path", "") for route in app.routes
    }


@pytest.mark.unit
@pytest.mark.sandbox_no_auth
def test_workspace_diagnostics_filters_runs_to_authenticated_user_and_workspace(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox_workspace_diagnostics as sb

    monkeypatch.setattr(
        sb._service,
        "runtime_diagnostics_summary",
        lambda: {
            "source": "feature_discovery",
            "summary": {
                "total": 0,
                "ready": 0,
                "unavailable": 0,
                "host_gated": 0,
                "scaffold": 0,
            },
            "runtimes": [],
        },
    )
    _seed_run("workspace_diag_target", 1, workspace_id="workspace-alpha")
    _seed_run("workspace_diag_other_user", 2, workspace_id="workspace-alpha")
    _seed_run("workspace_diag_other_workspace", 1, workspace_id="workspace-beta")

    with _client(monkeypatch, user_id=1, is_admin=False) as client:
        response = client.get(
            "/api/v1/sandbox/workspaces/workspace-alpha/diagnostics",
            params={"source_label": "research_workspace", "limit": 10},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["workspace_id"] == "workspace-alpha"
    assert data["source_label"] == "research_workspace"
    assert "workspace_playground" not in str(data)
    assert data["runtime"]["state"] == "not_configured"
    assert data["admission"]["state"] == "blocked"
    assert data["runs"]["total"] == 1
    assert [item["id"] for item in data["runs"]["items"]] == ["workspace_diag_target"]
    assert data["runs"]["items"][0]["workspace_id"] == "workspace-alpha"


@pytest.mark.unit
@pytest.mark.sandbox_no_auth
def test_workspace_diagnostics_includes_run_created_through_sandbox_api(
    monkeypatch,
) -> None:
    workspace_id = "workspace-api-created-diagnostics"

    with _sandbox_api_client(monkeypatch, user_id=11, is_admin=False) as client:
        create_response = client.post(
            "/api/v1/sandbox/runs",
            headers={"Idempotency-Key": "workspace-diagnostics-created-run"},
            json={
                "spec_version": "1.0",
                "runtime": "docker",
                "base_image": "python:3.11-slim",
                "command": ["python", "-c", "print('workspace diagnostics')"],
                "timeout_sec": 5,
                "workspace_id": workspace_id,
                "workspace_group_id": "research-workspace",
                "scope_snapshot_id": "scope-api-created-diagnostics",
            },
        )

        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["id"]
        assert created["workspace_id"] == workspace_id
        assert created["workspace_group_id"] == "research-workspace"
        assert created["scope_snapshot_id"] == "scope-api-created-diagnostics"

        diagnostics_response = client.get(
            f"/api/v1/sandbox/workspaces/{workspace_id}/diagnostics",
            params={"source_label": "research_workspace", "limit": 10},
        )

    assert diagnostics_response.status_code == 200
    diagnostics = diagnostics_response.json()
    assert diagnostics["workspace_id"] == workspace_id
    assert diagnostics["source_label"] == "research_workspace"
    assert diagnostics["runs"]["total"] == 1
    assert [item["id"] for item in diagnostics["runs"]["items"]] == [run_id]
    assert diagnostics["runs"]["items"][0]["workspace_id"] == workspace_id
    assert diagnostics["runs"]["items"][0]["workspace_group_id"] == "research-workspace"
    assert diagnostics["runs"]["items"][0]["scope_snapshot_id"] == "scope-api-created-diagnostics"


@pytest.mark.unit
@pytest.mark.sandbox_no_auth
def test_workspace_diagnostics_reports_unavailable_runtime(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox_workspace_diagnostics as sb

    monkeypatch.setattr(
        sb._service,
        "runtime_diagnostics_summary",
        lambda: {
            "source": "feature_discovery",
            "summary": {
                "total": 2,
                "ready": 0,
                "unavailable": 2,
                "host_gated": 0,
                "scaffold": 0,
            },
            "runtimes": [
                {
                    "name": "docker",
                    "readiness": "unavailable",
                    "normalized_reasons": ["runtime_unavailable"],
                }
            ],
        },
    )

    with _client(monkeypatch, user_id=1, is_admin=False) as client:
        response = client.get("/api/v1/sandbox/workspaces/workspace-empty/diagnostics")

    assert response.status_code == 200
    data = response.json()
    assert data["runtime"]["state"] == "unavailable"
    assert data["runtime"]["reason_code"] == "sandbox_runtime_unavailable"
    assert data["admission"]["state"] == "blocked"
    assert data["runs"]["total"] == 0
    assert data["runs"]["items"] == []


@pytest.mark.unit
@pytest.mark.sandbox_no_auth
def test_workspace_diagnostics_blocks_admission_when_sandbox_route_disabled(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox_workspace_diagnostics as sb

    monkeypatch.setattr(
        sb._service,
        "runtime_diagnostics_summary",
        lambda: {
            "source": "feature_discovery",
            "summary": {
                "total": 1,
                "ready": 1,
                "unavailable": 0,
                "host_gated": 0,
                "scaffold": 0,
            },
            "runtimes": [
                {
                    "name": "docker",
                    "readiness": "available",
                    "normalized_reasons": [],
                }
            ],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda route_key, default_stable=True: False
        if route_key == "sandbox"
        else default_stable,
    )

    with _client(monkeypatch, user_id=1, is_admin=False) as client:
        response = client.get("/api/v1/sandbox/workspaces/workspace-route-gated/diagnostics")

    assert response.status_code == 200
    data = response.json()
    assert data["runtime"]["state"] == "available"
    assert data["admission"]["state"] == "blocked"
    assert data["admission"]["reason_code"] == "sandbox_route_disabled"
    assert "route policy" in data["admission"]["message"]
