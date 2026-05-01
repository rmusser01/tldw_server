from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_mod
from tldw_Server_API.app.core.AuthNZ.permissions import ROLE_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    is_admin: bool,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=[ROLE_ADMIN] if is_admin else ["user"],
        permissions=[],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def _build_app_with_overrides(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.include_router(sandbox_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        request.state.auth = AuthContext(
            principal=principal,
            ip=(request.client.host if getattr(request, "client", None) else None),
            user_agent=(request.headers.get("User-Agent") if getattr(request, "headers", None) else None),
            request_id=(request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None),
        )
        return principal

    async def _fake_get_request_user() -> SimpleNamespace:
        return SimpleNamespace(
            id=1,
            username="sandbox-admin",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
            tenant_id="default",
        )

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[sandbox_mod.get_request_user] = _fake_get_request_user
    return app


def _diagnostics_payload() -> dict:
    return {
        "host": {
            "os": "darwin",
            "arch": "arm64",
            "apple_silicon": True,
            "macos_version": "15.0",
            "supported": True,
            "reasons": [],
        },
        "helper": {
            "configured": False,
            "path": None,
            "exists": False,
            "executable": False,
            "ready": False,
            "transport": None,
            "protocol_version": None,
            "helper_version": None,
            "reasons": ["macos_helper_missing"],
        },
        "templates": {
            "vz_linux": {
                "configured": False,
                "ready": False,
                "source": None,
                "reasons": ["vz_linux_template_missing"],
            }
        },
        "runtimes": {
            "vz_linux": {
                "available": False,
                "supported_trust_levels": ["trusted", "standard", "untrusted"],
                "reasons": ["macos_helper_missing", "vz_linux_template_missing"],
                "execution_mode": "none",
                "remediation": "Configure the macOS virtualization helper and mark it ready.",
            }
        },
        "reconciliation": {
            "computed": False,
            "persisted_sessions": 0,
            "live_vms": 0,
            "healthy_session_ids": [],
            "stale_session_ids": [],
            "unhealthy_session_ids": [],
            "skipped_active_session_ids": [],
            "orphaned_vm_ids": [],
            "owned_orphaned_vm_ids": [],
            "unknown_orphaned_vm_ids": [],
            "foreign_orphaned_vm_ids": [],
            "items": [],
            "reasons": ["macos_virtualization_helper_unavailable"],
        },
        "image_store": {
            "configured": False,
            "root_path": None,
            "registered_templates": 0,
            "run_manifests": 0,
            "gc_candidates": 0,
            "items": [],
            "reasons": [],
        },
    }


def test_admin_macos_diagnostics_returns_structured_payload(monkeypatch) -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_models import (
        StartupWarningRecord,
    )

    fake_service = SimpleNamespace(macos_diagnostics=lambda: _diagnostics_payload())
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))
    registry = StartupWarningRegistry(startup_id="boot-1")
    registry.add_warning(
        StartupWarningRecord(
            component="sandbox.vz_linux",
            severity="warning",
            startup_action="warn",
            code="vz_stale_session_controls_detected",
            summary="stale sessions detected",
            remediation="review diagnostics",
            details={"stale_session_controls": 1},
        )
    )
    registry.add_warning(
        StartupWarningRecord(
            component="jobs.integrity",
            severity="error",
            startup_action="block_startup",
            code="jobs_integrity_blocker",
            summary="jobs blocker",
            remediation="inspect jobs",
            details={},
        )
    )
    app.state.startup_warning_registry = registry

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/macos-diagnostics")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "host",
        "helper",
        "templates",
        "runtimes",
        "reconciliation",
        "image_store",
        "startup_warning_summary",
    }
    assert body["host"]["supported"] is True
    assert body["runtimes"]["vz_linux"]["execution_mode"] == "none"
    assert body["helper"]["protocol_version"] is None
    assert body["reconciliation"]["computed"] is False
    assert body["reconciliation"]["healthy_session_ids"] == []
    assert body["reconciliation"]["unhealthy_session_ids"] == []
    assert body["reconciliation"]["skipped_active_session_ids"] == []
    assert body["reconciliation"]["items"] == []
    assert body["image_store"]["configured"] is False
    assert body["image_store"]["items"] == []
    assert body["startup_warning_summary"] == {
        "present": True,
        "blocking": False,
        "codes": ["vz_stale_session_controls_detected"],
    }


def test_admin_macos_diagnostics_allows_real_vz_linux_execution_mode(monkeypatch) -> None:
    payload = _diagnostics_payload()
    payload["helper"]["configured"] = True
    payload["helper"]["ready"] = True
    payload["helper"]["protocol_version"] = "1"
    payload["helper"]["helper_version"] = "0.1.0"
    payload["helper"]["reasons"] = []
    payload["templates"]["vz_linux"]["configured"] = True
    payload["templates"]["vz_linux"]["ready"] = True
    payload["templates"]["vz_linux"]["reasons"] = []
    payload["runtimes"]["vz_linux"]["available"] = True
    payload["runtimes"]["vz_linux"]["reasons"] = []
    payload["runtimes"]["vz_linux"]["execution_mode"] = "real"
    payload["runtimes"]["vz_linux"]["remediation"] = None
    payload["reconciliation"]["computed"] = True
    payload["reconciliation"]["persisted_sessions"] = 1
    payload["reconciliation"]["live_vms"] = 1
    payload["reconciliation"]["healthy_session_ids"] = ["sess-live"]
    payload["reconciliation"]["items"] = [
        {
            "status": "healthy",
            "session_id": "sess-live",
            "vm_id": "vm-live",
            "state": "running",
            "healthy": True,
        }
    ]
    payload["reconciliation"]["reasons"] = []
    payload["image_store"]["configured"] = True
    payload["image_store"]["root_path"] = "/tmp/image-store"
    payload["image_store"]["registered_templates"] = 1
    payload["image_store"]["run_manifests"] = 1
    payload["image_store"]["gc_candidates"] = 0
    payload["image_store"]["items"] = [
        {
            "run_id": "run-live",
            "template_id": "vz_linux:ubuntu-24.04",
            "run_manifest_path": "/tmp/image-store/runs/run-live/manifest.json",
            "run_manifest_present": True,
            "gc_reason": None,
            "matched_vm_id": "vm-live",
            "matched_reconciliation_status": "healthy",
        }
    ]

    fake_service = SimpleNamespace(macos_diagnostics=lambda: payload)
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/macos-diagnostics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["runtimes"]["vz_linux"]["available"] is True
    assert body["runtimes"]["vz_linux"]["execution_mode"] == "real"
    assert body["helper"]["protocol_version"] == "1"
    assert body["helper"]["helper_version"] == "0.1.0"
    assert body["reconciliation"]["computed"] is True
    assert body["reconciliation"]["healthy_session_ids"] == ["sess-live"]
    assert body["reconciliation"]["items"][0]["status"] == "healthy"
    assert body["image_store"]["configured"] is True
    assert body["image_store"]["items"][0]["matched_vm_id"] == "vm-live"
