from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
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
        "recovery_summary": {
            "status": "unavailable",
            "severity": "error",
            "codes": ["vz_recovery_unavailable"],
            "counts": {
                "persisted_session_controls": 0,
                "healthy_session_controls": 0,
                "stale_session_controls": 0,
                "unhealthy_session_controls": 0,
                "skipped_active_session_controls": 0,
                "orphaned_vms": 0,
                "owned_orphaned_vms": 0,
                "unknown_orphaned_vms": 0,
                "foreign_orphaned_vms": 0,
                "image_store_gc_candidates": 0,
                "live_vms": 0,
            },
            "recommended_action": "Fix helper and runtime diagnostics before running repair.",
            "repair_endpoint": None,
            "cleanup_plan_endpoint": None,
            "notes": ["Reconciliation did not compute."],
        },
    }


def test_admin_macos_diagnostics_returns_structured_payload(monkeypatch) -> None:
    from tldw_Server_API.app.services.startup_warning_models import (
        StartupWarningRecord,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
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
        "observability",
        "recovery_summary",
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
    assert body["observability"] is None
    assert body["recovery_summary"]["status"] == "unavailable"
    assert body["recovery_summary"]["codes"] == ["vz_recovery_unavailable"]
    assert body["startup_warning_summary"] == {
        "present": True,
        "blocking": False,
        "codes": ["vz_stale_session_controls_detected"],
    }


def test_admin_runtime_diagnostics_returns_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services.startup_warning_models import (
        StartupWarningRecord,
    )
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    fake_service = SimpleNamespace(
        runtime_diagnostics_summary=lambda: {
            "source": "feature_discovery",
            "summary": {
                "total": 2,
                "ready": 1,
                "unavailable": 1,
                "host_gated": 1,
                "scaffold": 0,
                "host_local_warning_runtimes": [],
                "repair_supported_runtimes": ["vz_linux"],
            },
            "runtimes": [
                {
                    "name": "docker",
                    "available": True,
                    "implementation_state": "supported",
                    "readiness": "ready",
                    "reasons": [],
                    "normalized_reasons": [],
                    "boundary_class": "container",
                    "vm_grade_isolation": False,
                    "untrusted_eligible": True,
                    "isolation_warnings": [],
                    "strict_deny_all_supported": True,
                    "strict_allowlist_supported": False,
                    "session_reuse_model": "workspace_only",
                    "requires_live_health_check": False,
                    "repair_supported": False,
                    "recommended_action": "none",
                },
                {
                    "name": "vz_linux",
                    "available": False,
                    "implementation_state": "host_gated",
                    "readiness": "host_gated",
                    "reasons": ["macos_virtualization_helper_unavailable"],
                    "normalized_reasons": ["helper_unavailable"],
                    "boundary_class": "vm_grade",
                    "vm_grade_isolation": True,
                    "untrusted_eligible": True,
                    "isolation_warnings": [],
                    "strict_deny_all_supported": False,
                    "strict_allowlist_supported": False,
                    "session_reuse_model": "warm_vm",
                    "requires_live_health_check": True,
                    "repair_supported": True,
                    "recommended_action": "check_helper",
                },
            ],
        }
    )
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
    app.state.startup_warning_registry = registry

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/runtime-diagnostics")

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "feature_discovery"
    assert body["summary"]["repair_supported_runtimes"] == ["vz_linux"]
    assert [row["name"] for row in body["runtimes"]] == ["docker", "vz_linux"]
    assert body["runtimes"][1]["recommended_action"] == "check_helper"
    assert body["startup_warning_summary"] == {
        "present": True,
        "blocking": False,
        "codes": ["vz_stale_session_controls_detected"],
    }


def test_admin_runtime_diagnostics_offloads_runtime_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_to_thread(
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        calls.append(getattr(func, "__name__", "unknown"))
        return func(*args, **kwargs)

    fake_service = SimpleNamespace(
        runtime_diagnostics_summary=lambda: {
            "source": "feature_discovery",
            "summary": {
                "total": 0,
                "ready": 0,
                "unavailable": 0,
                "host_gated": 0,
                "scaffold": 0,
                "host_local_warning_runtimes": [],
                "repair_supported_runtimes": [],
            },
            "runtimes": [],
        }
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)
    monkeypatch.setattr(sandbox_mod.asyncio, "to_thread", fake_to_thread)

    app = _build_app_with_overrides(_make_principal(is_admin=True))
    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/runtime-diagnostics")

    assert resp.status_code == 200
    assert calls == ["<lambda>"]


def test_sandbox_startup_warning_summary_fails_open_without_registry() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    assert sandbox_mod._sandbox_startup_warning_summary(request) == {
        "present": False,
        "blocking": False,
        "codes": [],
    }


def test_sandbox_startup_warning_summary_fails_open_on_registry_error() -> None:
    class BrokenRegistry:
        def list_warnings(self, *, component_prefix: str) -> list[object]:
            raise RuntimeError("registry unavailable")

        def summary(self, *, component_prefix: str) -> dict[str, object]:
            raise RuntimeError("registry unavailable")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(startup_warning_registry=BrokenRegistry())
        )
    )

    assert sandbox_mod._sandbox_startup_warning_summary(request) == {
        "present": False,
        "blocking": False,
        "codes": [],
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
    payload["recovery_summary"] = {
        "status": "healthy",
        "severity": "ok",
        "codes": [],
        "counts": {
            "persisted_session_controls": 1,
            "healthy_session_controls": 1,
            "stale_session_controls": 0,
            "unhealthy_session_controls": 0,
            "skipped_active_session_controls": 0,
            "orphaned_vms": 0,
            "owned_orphaned_vms": 0,
            "unknown_orphaned_vms": 0,
            "foreign_orphaned_vms": 0,
            "image_store_gc_candidates": 0,
            "live_vms": 1,
        },
        "recommended_action": "No recovery action needed.",
        "repair_endpoint": None,
        "cleanup_plan_endpoint": None,
        "notes": [],
    }

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
    assert body["recovery_summary"]["status"] == "healthy"
