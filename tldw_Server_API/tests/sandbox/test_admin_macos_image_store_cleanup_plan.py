from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_mod
from tldw_Server_API.app.core.AuthNZ.permissions import ROLE_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _make_principal(*, is_admin: bool) -> AuthPrincipal:
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


def _diagnostics_payload() -> dict[str, object]:
    return {
        "host": {"supported": True},
        "helper": {"ready": True, "protocol_version": "1", "helper_version": "0.1.0"},
        "templates": {},
        "runtimes": {},
        "reconciliation": {"computed": True, "items": [], "reasons": []},
        "image_store": {
            "configured": True,
            "root_path": "/tmp/image-store",
            "registered_templates": 1,
            "run_manifests": 4,
            "gc_candidates": 3,
            "items": [
                {
                    "run_id": "run-live",
                    "template_id": "vz_linux:bundle-live",
                    "run_manifest_path": "/tmp/image-store/runs/run-live/manifest.json",
                    "run_manifest_present": True,
                    "gc_reason": None,
                    "gc_path": None,
                    "matched_vm_id": "vm-live",
                    "matched_reconciliation_status": "healthy",
                    "matched_reconciliation_reason": None,
                },
                {
                    "run_id": "run-manifest-only",
                    "template_id": "vz_linux:bundle-manifest",
                    "run_manifest_path": "/tmp/image-store/runs/run-manifest-only/manifest.json",
                    "run_manifest_present": True,
                    "gc_reason": "planning_only_run_manifest",
                    "gc_path": "/tmp/image-store/runs/run-manifest-only",
                    "matched_vm_id": None,
                    "matched_reconciliation_status": None,
                    "matched_reconciliation_reason": None,
                },
                {
                    "run_id": "run-inactive",
                    "template_id": "vz_linux:bundle-inactive",
                    "run_manifest_path": "/tmp/image-store/runs/run-inactive/manifest.json",
                    "run_manifest_present": True,
                    "gc_reason": "inactive_run",
                    "gc_path": "/tmp/image-store/runs/run-inactive",
                    "matched_vm_id": None,
                    "matched_reconciliation_status": None,
                    "matched_reconciliation_reason": None,
                },
                {
                    "run_id": "run-legacy",
                    "template_id": None,
                    "run_manifest_path": None,
                    "run_manifest_present": False,
                    "gc_reason": "legacy_run_directory",
                    "gc_path": "/tmp/image-store/runs/run-legacy",
                    "matched_vm_id": None,
                    "matched_reconciliation_status": None,
                    "matched_reconciliation_reason": None,
                },
            ],
            "reasons": [],
        },
    }


def test_admin_macos_image_store_cleanup_plan_endpoint_returns_structured_payload(monkeypatch) -> None:
    fake_service = SimpleNamespace(
        plan_macos_image_store_cleanup=lambda: {
            "dry_run": True,
            "image_store": {"configured": True, "root_path": "/tmp/image-store"},
            "summary": {
                "total_candidates": 3,
                "planned_actions": 3,
                "blocked_live_matches": 0,
                "planning_only_run_manifests": 1,
                "inactive_runs": 1,
                "legacy_run_directories": 1,
            },
            "actions": [
                {
                    "type": "remove_run_manifest",
                    "run_id": "run-manifest-only",
                    "status": "planned",
                    "gc_reason": "planning_only_run_manifest",
                }
            ],
            "reasons": [],
        }
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/macos-image-store/cleanup-plan")

    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["summary"]["total_candidates"] == 3
    assert body["actions"][0]["type"] == "remove_run_manifest"


def test_admin_macos_image_store_cleanup_plan_requires_admin(monkeypatch) -> None:
    fake_service = SimpleNamespace(plan_macos_image_store_cleanup=lambda: {})
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=False))

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/macos-image-store/cleanup-plan")

    assert resp.status_code == 403


def test_service_plans_image_store_cleanup_actions_from_diagnostics(monkeypatch) -> None:
    service = SandboxService(enable_background_tasks=False)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload())

    result = service.plan_macos_image_store_cleanup()

    assert result["dry_run"] is True
    assert result["image_store"]["configured"] is True
    assert result["summary"] == {
        "total_candidates": 3,
        "planned_actions": 3,
        "blocked_live_matches": 0,
        "planning_only_run_manifests": 1,
        "inactive_runs": 1,
        "legacy_run_directories": 1,
    }
    assert result["actions"] == [
        {
            "type": "remove_run_manifest",
            "run_id": "run-manifest-only",
            "template_id": "vz_linux:bundle-manifest",
            "run_manifest_path": "/tmp/image-store/runs/run-manifest-only/manifest.json",
            "run_manifest_present": True,
            "gc_reason": "planning_only_run_manifest",
            "gc_path": "/tmp/image-store/runs/run-manifest-only",
            "matched_vm_id": None,
            "matched_reconciliation_status": None,
            "matched_reconciliation_reason": None,
            "status": "planned",
        },
        {
            "type": "remove_run_directory",
            "run_id": "run-inactive",
            "template_id": "vz_linux:bundle-inactive",
            "run_manifest_path": "/tmp/image-store/runs/run-inactive/manifest.json",
            "run_manifest_present": True,
            "gc_reason": "inactive_run",
            "gc_path": "/tmp/image-store/runs/run-inactive",
            "matched_vm_id": None,
            "matched_reconciliation_status": None,
            "matched_reconciliation_reason": None,
            "status": "planned",
        },
        {
            "type": "remove_legacy_run_directory",
            "run_id": "run-legacy",
            "template_id": None,
            "run_manifest_path": None,
            "run_manifest_present": False,
            "gc_reason": "legacy_run_directory",
            "gc_path": "/tmp/image-store/runs/run-legacy",
            "matched_vm_id": None,
            "matched_reconciliation_status": None,
            "matched_reconciliation_reason": None,
            "status": "planned",
        },
    ]


def test_service_does_not_plan_cleanup_for_live_matched_runs(monkeypatch) -> None:
    service = SandboxService(enable_background_tasks=False)
    payload = _diagnostics_payload()
    payload["image_store"]["items"].append(
        {
            "run_id": "run-blocked",
            "template_id": "vz_linux:bundle-blocked",
            "run_manifest_path": "/tmp/image-store/runs/run-blocked/manifest.json",
            "run_manifest_present": True,
            "gc_reason": "inactive_run",
            "gc_path": "/tmp/image-store/runs/run-blocked",
            "matched_vm_id": "vm-blocked",
            "matched_reconciliation_status": "owned_orphaned_vm",
            "matched_reconciliation_reason": "owned_orphan",
        }
    )
    payload["image_store"]["gc_candidates"] = 4
    monkeypatch.setattr(service, "macos_diagnostics", lambda: payload)

    result = service.plan_macos_image_store_cleanup()

    assert result["summary"]["total_candidates"] == 4
    assert result["summary"]["planned_actions"] == 3
    assert result["summary"]["blocked_live_matches"] == 1
    assert all(action["run_id"] != "run-blocked" for action in result["actions"])
    assert result["reasons"] == ["live_vm_matches_blocked_cleanup"]
