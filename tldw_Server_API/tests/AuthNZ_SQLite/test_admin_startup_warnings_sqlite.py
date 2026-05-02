from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.unit


def _build_app(*, roles: list[str]) -> FastAPI:
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    app = FastAPI()
    app.include_router(admin_router, prefix="/api/v1")

    async def _principal_override(request=None):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="adminuser",
            token_type="access",
            jti=None,
            roles=roles,
            permissions=["system.configure"],
            is_admin="admin" in roles,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override
    return app


def _warning(*, code: str, startup_action: str = "warn"):
    from tldw_Server_API.app.services.startup_warning_models import StartupWarningRecord

    return StartupWarningRecord(
        component="sandbox.vz_linux",
        severity="error" if startup_action == "block_startup" else "warning",
        startup_action=startup_action,
        code=code,
        summary=f"summary for {code}",
        remediation="follow the operator notes",
        details={"count": 1},
        detected_at=datetime(2026, 4, 30, 12, 0, tzinfo=UTC),
    )


def test_admin_startup_warnings_returns_current_process_registry_summary() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    app = _build_app(roles=["admin"])
    registry = StartupWarningRegistry(startup_id="boot-1")
    registry.add_warning(_warning(code="vz_stale_session_controls_detected"))
    registry.add_warning(_warning(code="vz_helper_protocol_mismatch", startup_action="block_startup"))
    app.state.startup_warning_registry = registry

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/startup-warnings")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["startup_id"] == "boot-1"
    assert payload["scope"] == "current_process"
    assert payload["warnings_present"] is True
    assert payload["blocking_present"] is True
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["by_component"] == {"sandbox.vz_linux": 2}
    assert payload["summary"]["by_severity"] == {"error": 1, "warning": 1}
    assert [item["code"] for item in payload["items"]] == [
        "vz_helper_protocol_mismatch",
        "vz_stale_session_controls_detected",
    ]


def test_admin_startup_warnings_is_admin_only() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    app = _build_app(roles=["user"])
    app.state.startup_warning_registry = StartupWarningRegistry(startup_id="boot-1")

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/startup-warnings")

    assert response.status_code == 403, response.text


def test_admin_startup_warnings_reports_current_process_scope() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    app = _build_app(roles=["admin"])
    app.state.startup_warning_registry = StartupWarningRegistry(startup_id="boot-1")

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/startup-warnings")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["scope"] == "current_process"
    assert payload["warnings_present"] is False
    assert payload["blocking_present"] is False
    assert payload["summary"]["total"] == 0
    assert payload["items"] == []
