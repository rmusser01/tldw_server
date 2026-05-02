from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_mod
from tldw_Server_API.app.core.AuthNZ.permissions import ROLE_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Sandbox import service as service_mod
from tldw_Server_API.app.core.Sandbox.image_store import SandboxImageStore
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


def _cleanup_payload(*, dry_run: bool, action_status: str) -> dict[str, object]:
    return {
        "dry_run": dry_run,
        "image_store": {"configured": True, "root_path": "/tmp/image-store"},
        "summary": {
            "total_candidates": 1,
            "planned_actions": 1,
            "deleted_actions": 0 if dry_run else 1,
            "blocked_live_matches": 0,
            "planning_only_run_manifests": 1,
            "inactive_runs": 0,
            "legacy_run_directories": 0,
        },
        "actions": [
            {
                "type": "remove_run_manifest",
                "run_id": "run-manifest-only",
                "status": action_status,
                "gc_reason": "planning_only_run_manifest",
            }
        ],
        "reasons": [],
    }


def _diagnostics_payload(root: Path) -> dict[str, object]:
    return {
        "host": {"supported": True},
        "helper": {"ready": True, "protocol_version": "1", "helper_version": "0.1.0"},
        "templates": {},
        "runtimes": {},
        "reconciliation": {"computed": True, "items": [], "reasons": []},
        "image_store": {
            "configured": True,
            "root_path": str(root),
            "registered_templates": 1,
            "run_manifests": 3,
            "gc_candidates": 3,
            "items": [
                {
                    "run_id": "run-manifest-only",
                    "template_id": "vz_linux:bundle-manifest",
                    "run_manifest_path": str(root / "runs" / "run-manifest-only" / "manifest.json"),
                    "run_manifest_present": True,
                    "gc_reason": "planning_only_run_manifest",
                    "gc_path": str(root / "runs" / "run-manifest-only"),
                    "matched_vm_id": None,
                    "matched_reconciliation_status": None,
                    "matched_reconciliation_reason": None,
                },
                {
                    "run_id": "run-inactive",
                    "template_id": "vz_linux:bundle-inactive",
                    "run_manifest_path": str(root / "runs" / "run-inactive" / "manifest.json"),
                    "run_manifest_present": True,
                    "gc_reason": "inactive_run",
                    "gc_path": str(root / "runs" / "run-inactive"),
                    "matched_vm_id": None,
                    "matched_reconciliation_status": None,
                    "matched_reconciliation_reason": None,
                },
                {
                    "run_id": "run-blocked",
                    "template_id": "vz_linux:bundle-blocked",
                    "run_manifest_path": str(root / "runs" / "run-blocked" / "manifest.json"),
                    "run_manifest_present": True,
                    "gc_reason": "inactive_run",
                    "gc_path": str(root / "runs" / "run-blocked"),
                    "matched_vm_id": "vm-blocked",
                    "matched_reconciliation_status": "owned_orphaned_vm",
                    "matched_reconciliation_reason": "owned_orphan",
                },
            ],
            "reasons": [],
        },
    }


def _seed_store(root: Path) -> None:
    store = SandboxImageStore(root_path=root)
    template_disk = root.parent / "rootfs.img"
    template_disk.write_bytes(b"rootfs")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="bundle-manifest",
        disk_paths=[str(template_disk)],
    )
    store.prepare_run_clone(template_id=template_id, run_id="run-manifest-only")
    store.prepare_run_clone(template_id=template_id, run_id="run-inactive")
    store.prepare_run_clone(template_id=template_id, run_id="run-blocked")
    (root / "runs" / "run-inactive" / "rootfs.img").write_bytes(b"clone")
    (root / "runs" / "run-blocked" / "rootfs.img").write_bytes(b"clone")


def test_admin_macos_image_store_cleanup_defaults_to_dry_run(monkeypatch) -> None:
    fake_service = SimpleNamespace(
        cleanup_macos_image_store=lambda **kwargs: _cleanup_payload(
            dry_run=kwargs["dry_run"],
            action_status="planned",
        )
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post("/api/v1/sandbox/admin/macos-image-store/cleanup", json={})

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True
    assert resp.json()["actions"][0]["status"] == "planned"


def test_admin_macos_image_store_cleanup_dry_run_false_passes_through(monkeypatch) -> None:
    seen_kwargs: dict[str, object] = {}

    def _cleanup(**kwargs) -> dict[str, object]:
        seen_kwargs.update(kwargs)
        return _cleanup_payload(dry_run=kwargs["dry_run"], action_status="deleted")

    monkeypatch.setattr(sandbox_mod, "_service", SimpleNamespace(cleanup_macos_image_store=_cleanup), raising=True)
    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sandbox/admin/macos-image-store/cleanup",
            json={"dry_run": False, "confirm_all": True},
        )

    assert resp.status_code == 200
    assert seen_kwargs["dry_run"] is False
    assert seen_kwargs["confirm_all"] is True
    assert resp.json()["dry_run"] is False
    assert resp.json()["summary"]["deleted_actions"] == 1
    assert resp.json()["actions"][0]["status"] == "deleted"


def test_admin_macos_image_store_cleanup_filter_fields_pass_through(monkeypatch) -> None:
    seen_kwargs: dict[str, object] = {}

    def _cleanup(**kwargs) -> dict[str, object]:
        seen_kwargs.update(kwargs)
        return _cleanup_payload(dry_run=bool(kwargs["dry_run"]), action_status="planned")

    monkeypatch.setattr(sandbox_mod, "_service", SimpleNamespace(cleanup_macos_image_store=_cleanup), raising=True)
    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sandbox/admin/macos-image-store/cleanup",
            json={
                "action_types": ["remove_run_manifest"],
                "run_ids": ["run-manifest-only"],
            },
        )

    assert resp.status_code == 200
    assert seen_kwargs["action_types"] == ["remove_run_manifest"]
    assert seen_kwargs["run_ids"] == ["run-manifest-only"]


def test_admin_macos_image_store_cleanup_maps_confirmation_error(monkeypatch) -> None:
    def _cleanup(**kwargs) -> dict[str, object]:
        raise service_mod.SandboxImageStoreCleanupError(
            "image_store_cleanup_confirmation_required",
            400,
        )

    monkeypatch.setattr(sandbox_mod, "_service", SimpleNamespace(cleanup_macos_image_store=_cleanup), raising=True)
    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post("/api/v1/sandbox/admin/macos-image-store/cleanup", json={"dry_run": False})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "image_store_cleanup_confirmation_required"


def test_admin_macos_image_store_cleanup_requires_admin(monkeypatch) -> None:
    fake_service = SimpleNamespace(cleanup_macos_image_store=lambda **kwargs: _cleanup_payload(dry_run=True, action_status="planned"))
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)
    app = _build_app_with_overrides(_make_principal(is_admin=False))

    with TestClient(app) as client:
        resp = client.post("/api/v1/sandbox/admin/macos-image-store/cleanup", json={})

    assert resp.status_code == 403


def test_service_cleanup_macos_image_store_dry_run_reuses_planning(monkeypatch, tmp_path: Path) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store"
    _seed_store(root)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))

    result = service.cleanup_macos_image_store(dry_run=True)

    assert result["dry_run"] is True
    assert result["summary"]["planned_actions"] == 2
    assert result["summary"]["deleted_actions"] == 0
    assert [action["status"] for action in result["actions"]] == ["planned", "planned"]
    assert (root / "runs" / "run-manifest-only" / "manifest.json").exists()
    assert (root / "runs" / "run-inactive").exists()


def test_service_cleanup_macos_image_store_mutates_planned_candidates_only(monkeypatch, tmp_path: Path) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store"
    _seed_store(root)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))

    result = service.cleanup_macos_image_store(dry_run=False, confirm_all=True)

    assert result["dry_run"] is False
    assert result["summary"] == {
        "total_candidates": 3,
        "planned_actions": 2,
        "deleted_actions": 2,
        "blocked_live_matches": 1,
        "planning_only_run_manifests": 1,
        "inactive_runs": 2,
        "legacy_run_directories": 0,
    }
    assert [action["status"] for action in result["actions"]] == ["deleted", "deleted"]
    assert not (root / "runs" / "run-manifest-only").exists()
    assert not (root / "runs" / "run-inactive").exists()
    assert (root / "runs" / "run-blocked").exists()
    assert result["reasons"] == ["live_vm_matches_blocked_cleanup"]


def test_service_cleanup_macos_image_store_reports_per_action_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store"
    _seed_store(root)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))
    original_cleanup = SandboxImageStore.cleanup_run_candidate

    def _cleanup(self: SandboxImageStore, *, run_id: str, reason: str) -> bool:
        if run_id == "run-manifest-only":
            raise OSError("locked")
        return original_cleanup(self, run_id=run_id, reason=reason)

    monkeypatch.setattr(SandboxImageStore, "cleanup_run_candidate", _cleanup)

    result = service.cleanup_macos_image_store(dry_run=False, confirm_all=True)

    assert result["summary"]["deleted_actions"] == 1
    assert [action["status"] for action in result["actions"]] == ["error", "deleted"]
    assert result["actions"][0]["error"] == "locked"
    assert (root / "runs" / "run-manifest-only").exists()
    assert not (root / "runs" / "run-inactive").exists()


def test_service_cleanup_macos_image_store_maps_store_init_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store-file"
    root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))

    with pytest.raises(service_mod.SandboxImageStoreCleanupError) as exc_info:
        service.cleanup_macos_image_store(dry_run=False, confirm_all=True)

    assert exc_info.value.reason == "image_store_cleanup_unavailable"
    assert exc_info.value.status_code == 503


def test_service_cleanup_macos_image_store_requires_confirmation_for_unfiltered_mutation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store"
    _seed_store(root)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))

    with pytest.raises(service_mod.SandboxImageStoreCleanupError, match="image_store_cleanup_confirmation_required"):
        service.cleanup_macos_image_store(dry_run=False)

    assert (root / "runs" / "run-manifest-only").exists()
    assert (root / "runs" / "run-inactive").exists()


def test_service_cleanup_macos_image_store_filters_by_action_type(monkeypatch, tmp_path: Path) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store"
    _seed_store(root)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))

    result = service.cleanup_macos_image_store(
        dry_run=False,
        action_types=["remove_run_manifest"],
    )

    assert result["summary"]["total_candidates"] == 3
    assert result["summary"]["planned_actions"] == 1
    assert result["summary"]["deleted_actions"] == 1
    assert [action["run_id"] for action in result["actions"]] == ["run-manifest-only"]
    assert not (root / "runs" / "run-manifest-only").exists()
    assert (root / "runs" / "run-inactive").exists()


def test_service_cleanup_macos_image_store_filters_by_run_id(monkeypatch, tmp_path: Path) -> None:
    service = SandboxService(enable_background_tasks=False)
    root = tmp_path / "store"
    _seed_store(root)
    monkeypatch.setattr(service, "macos_diagnostics", lambda: _diagnostics_payload(root))

    result = service.cleanup_macos_image_store(
        dry_run=True,
        run_ids=["run-inactive"],
    )

    assert result["dry_run"] is True
    assert result["summary"]["planned_actions"] == 1
    assert result["summary"]["deleted_actions"] == 0
    assert [action["run_id"] for action in result["actions"]] == ["run-inactive"]
