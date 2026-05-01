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
from tldw_Server_API.app.core.Sandbox import service as service_mod
from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.service import SandboxService


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


def _repair_payload(*, dry_run: bool, action_status: str) -> dict[str, object]:
    return {
        "dry_run": dry_run,
        "helper": {"ready": True, "protocol_version": "1", "helper_version": "0.1.0"},
        "summary": {
            "stale_session_controls": 1,
            "unhealthy_session_controls": 0,
            "deleted_session_controls": 0 if dry_run else 1,
            "skipped_active_sessions": 0,
            "orphaned_vms": 0,
            "terminated_orphaned_vms": 0,
        },
        "actions": [{"type": "delete_session_control", "session_id": "sess-stale", "status": action_status}],
        "reasons": [],
    }


def _reconciliation_report(
    *,
    items: list[dict[str, object]] | None = None,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "computed": not bool(reasons),
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
            "items": list(items or []),
            "reasons": list(reasons or []),
        }


def _service_with_orchestrator(orch: SimpleNamespace) -> SandboxService:
    service = SandboxService(enable_background_tasks=False)
    service._orch = orch
    return service


def test_admin_reconciliation_repair_defaults_to_dry_run(monkeypatch) -> None:
    fake_service = SimpleNamespace(
        repair_macos_reconciliation=lambda **kwargs: _repair_payload(
            dry_run=kwargs["dry_run"],
            action_status="planned",
        )
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post("/api/v1/sandbox/admin/macos-reconciliation/repair", json={})

    assert resp.status_code == 200
    assert resp.json()["dry_run"] is True
    assert resp.json()["actions"][0]["status"] == "planned"


def test_admin_reconciliation_repair_dry_run_false_passes_through(monkeypatch) -> None:
    seen_kwargs: dict[str, object] = {}

    def _repair(**kwargs) -> dict[str, object]:
        seen_kwargs.update(kwargs)
        return _repair_payload(dry_run=kwargs["dry_run"], action_status="deleted")

    monkeypatch.setattr(sandbox_mod, "_service", SimpleNamespace(repair_macos_reconciliation=_repair), raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sandbox/admin/macos-reconciliation/repair",
            json={"dry_run": False},
        )

    assert resp.status_code == 200
    assert seen_kwargs["dry_run"] is False
    assert resp.json()["dry_run"] is False
    assert resp.json()["actions"][0]["status"] == "deleted"


def test_admin_reconciliation_repair_runs_service_in_thread(monkeypatch) -> None:
    to_thread_calls: list[dict[str, object]] = []

    async def _fake_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append({"func": func, "args": args, "kwargs": dict(kwargs)})
        return func(*args, **kwargs)

    fake_service = SimpleNamespace(
        repair_macos_reconciliation=lambda **kwargs: _repair_payload(
            dry_run=kwargs["dry_run"],
            action_status="planned",
        )
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)
    monkeypatch.setattr(sandbox_mod.asyncio, "to_thread", _fake_to_thread)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post("/api/v1/sandbox/admin/macos-reconciliation/repair", json={})

    assert resp.status_code == 200
    assert to_thread_calls
    assert to_thread_calls[0]["func"] == fake_service.repair_macos_reconciliation
    assert to_thread_calls[0]["kwargs"]["dry_run"] is True


def test_admin_reconciliation_repair_orphan_termination_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_kwargs: dict[str, object] = {}

    def _repair(**kwargs: object) -> dict[str, object]:
        seen_kwargs.update(kwargs)
        return _repair_payload(dry_run=bool(kwargs["dry_run"]), action_status="planned")

    monkeypatch.setattr(sandbox_mod, "_service", SimpleNamespace(repair_macos_reconciliation=_repair), raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sandbox/admin/macos-reconciliation/repair",
            json={"terminate_orphaned_vms": True},
        )

    assert resp.status_code == 200
    assert seen_kwargs["terminate_orphaned_vms"] is True


def test_admin_reconciliation_repair_maps_service_unavailable_error(monkeypatch) -> None:
    def _repair(**kwargs) -> dict[str, object]:
        raise service_mod.SandboxReconciliationRepairError("macos_virtualization_helper_unavailable", 503)

    monkeypatch.setattr(sandbox_mod, "_service", SimpleNamespace(repair_macos_reconciliation=_repair), raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=True))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/sandbox/admin/macos-reconciliation/repair",
            json={"dry_run": False},
        )

    assert resp.status_code == 503
    assert resp.json()["detail"] == "macos_virtualization_helper_unavailable"


def test_admin_reconciliation_repair_requires_admin(monkeypatch) -> None:
    fake_service = SimpleNamespace(repair_macos_reconciliation=lambda **kwargs: _repair_payload(dry_run=True, action_status="planned"))
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(_make_principal(is_admin=False))

    with TestClient(app) as client:
        resp = client.post("/api/v1/sandbox/admin/macos-reconciliation/repair", json={})

    assert resp.status_code == 403


def test_repair_stale_row_dry_run_plans_delete_without_mutation(monkeypatch) -> None:
    deleted_session_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=deleted_session_ids.append)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "probe_helper",
        lambda: {"ready": True, "protocol_version": "1", "helper_version": "0.1.0"},
        raising=True,
    )
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "stale_session",
                    "session_id": "sess-stale",
                    "vm_id": "vm-missing",
                    "reason": "vm_missing",
                    "persisted_template_id": "vz_linux:bundle-stale",
                    "helper_template_id": "vz_linux:bundle-old",
                    "template_id_matches_persisted": False,
                }
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation()

    assert deleted_session_ids == []
    assert result["dry_run"] is True
    assert result["helper"] == {"ready": True, "protocol_version": "1", "helper_version": "0.1.0"}
    assert result["summary"]["stale_session_controls"] == 1
    assert result["summary"]["deleted_session_controls"] == 0
    assert result["actions"] == [
        {
            "type": "delete_session_control",
            "session_id": "sess-stale",
            "vm_id": "vm-missing",
            "status": "planned",
            "reason": "vm_missing",
            "persisted_template_id": "vz_linux:bundle-stale",
            "helper_template_id": "vz_linux:bundle-old",
            "template_id_matches_persisted": False,
        }
    ]


def test_repair_stale_row_delete_calls_orchestrator(monkeypatch) -> None:
    deleted_session_ids: list[str] = []

    def _delete_vz_session_control(session_id: str) -> bool:
        deleted_session_ids.append(session_id)
        return True

    orch = SimpleNamespace(delete_vz_session_control=_delete_vz_session_control)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "stale_session",
                    "session_id": "sess-stale",
                    "vm_id": "vm-missing",
                    "reason": "vm_missing",
                }
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation(dry_run=False)

    assert deleted_session_ids == ["sess-stale"]
    assert result["summary"]["deleted_session_controls"] == 1
    assert result["actions"][0]["status"] == "deleted"


def test_repair_unhealthy_row_delete_calls_orchestrator(monkeypatch) -> None:
    deleted_session_ids: list[str] = []

    def _delete_vz_session_control(session_id: str) -> bool:
        deleted_session_ids.append(session_id)
        return True

    orch = SimpleNamespace(delete_vz_session_control=_delete_vz_session_control)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "unhealthy_vm",
                    "session_id": "sess-unhealthy",
                    "vm_id": "vm-unhealthy",
                    "reason": "vm_unhealthy",
                }
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation(dry_run=False)

    assert deleted_session_ids == ["sess-unhealthy"]
    assert result["summary"]["unhealthy_session_controls"] == 1
    assert result["summary"]["deleted_session_controls"] == 1
    assert result["actions"] == [
        {
            "type": "delete_session_control",
            "session_id": "sess-unhealthy",
            "vm_id": "vm-unhealthy",
            "status": "deleted",
            "reason": "vm_unhealthy",
        }
    ]


def test_repair_disabled_delete_flags_suppress_actions_and_mutation(monkeypatch) -> None:
    deleted_session_ids: list[str] = []

    def _delete_vz_session_control(session_id: str) -> bool:
        deleted_session_ids.append(session_id)
        return True

    orch = SimpleNamespace(delete_vz_session_control=_delete_vz_session_control)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "stale_session",
                    "session_id": "sess-stale",
                    "vm_id": "vm-missing",
                    "reason": "vm_missing",
                },
                {
                    "status": "unhealthy_vm",
                    "session_id": "sess-unhealthy",
                    "vm_id": "vm-unhealthy",
                    "reason": "vm_unhealthy",
                },
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation(
        delete_stale_session_controls=False,
        delete_unhealthy_session_controls=False,
        dry_run=False,
    )

    assert deleted_session_ids == []
    assert result["summary"]["stale_session_controls"] == 1
    assert result["summary"]["unhealthy_session_controls"] == 1
    assert result["summary"]["deleted_session_controls"] == 0
    assert result["actions"] == []


def test_repair_delete_false_reports_missing_without_incrementing_deleted(monkeypatch) -> None:
    deleted_session_ids: list[str] = []

    def _delete_vz_session_control(session_id: str) -> bool:
        deleted_session_ids.append(session_id)
        return False

    orch = SimpleNamespace(delete_vz_session_control=_delete_vz_session_control)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "stale_session",
                    "session_id": "sess-stale",
                    "vm_id": "vm-missing",
                    "reason": "vm_missing",
                }
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation(dry_run=False)

    assert deleted_session_ids == ["sess-stale"]
    assert result["summary"]["deleted_session_controls"] == 0
    assert result["actions"] == [
        {
            "type": "delete_session_control",
            "session_id": "sess-stale",
            "vm_id": "vm-missing",
            "status": "missing",
            "reason": "vm_missing",
        }
    ]


def test_repair_delete_exception_maps_to_structured_error(monkeypatch) -> None:
    def _delete_vz_session_control(session_id: str) -> bool:
        raise RuntimeError("database locked")

    orch = SimpleNamespace(delete_vz_session_control=_delete_vz_session_control)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "stale_session",
                    "session_id": "sess-stale",
                    "vm_id": "vm-missing",
                    "reason": "vm_missing",
                }
            ]
        ),
        raising=True,
    )

    with pytest.raises(service_mod.SandboxReconciliationRepairError) as exc_info:
        service.repair_macos_reconciliation(dry_run=False)

    assert exc_info.value.status_code == 503
    assert exc_info.value.reason == "vz_session_control_delete_failed"


def test_repair_active_session_item_is_skipped(monkeypatch) -> None:
    deleted_session_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=deleted_session_ids.append)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 1)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "skipped_active_session",
                    "session_id": "sess-active",
                    "vm_id": "vm-missing",
                    "reason": "active_session",
                }
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation(dry_run=False)

    assert deleted_session_ids == []
    assert result["summary"]["skipped_active_sessions"] == 1
    assert result["actions"][0]["status"] == "skipped"
    assert result["actions"][0]["reason"] == "active_session"


def test_repair_orphan_vm_dry_run_plans_termination_without_helper_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminated_vm_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "owned_orphaned_vm",
                    "vm_id": "vm-orphan",
                    "reason": "owned_orphan",
                    "termination_eligible": True,
                    "run_id": "run-orphan",
                    "template_id": "vz_linux:bundle-owned",
                    "planning_source": "image_store",
                    "run_manifest_path": "/tmp/image-store/runs/run-orphan/manifest.json",
                    "run_manifest_present": True,
                }
            ]
        ),
        raising=True,
    )
    monkeypatch.setattr(
        service_mod,
        "MacOSVirtualizationHelperClient",
        lambda: SimpleNamespace(terminate_vm=terminated_vm_ids.append),
        raising=True,
    )

    result = service.repair_macos_reconciliation(terminate_orphaned_vms=True)

    assert terminated_vm_ids == []
    assert result["summary"]["orphaned_vms"] == 1
    assert result["summary"]["terminated_orphaned_vms"] == 0
    assert result["actions"] == [
        {
            "type": "terminate_orphaned_vm",
            "session_id": None,
            "vm_id": "vm-orphan",
            "status": "planned",
            "reason": "owned_orphan",
            "termination_eligible": True,
            "run_id": "run-orphan",
            "template_id": "vz_linux:bundle-owned",
            "planning_source": "image_store",
            "run_manifest_path": "/tmp/image-store/runs/run-orphan/manifest.json",
            "run_manifest_present": True,
        }
    ]


def test_repair_orphan_vm_mutating_run_calls_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    terminated_vm_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "owned_orphaned_vm",
                    "vm_id": "vm-orphan",
                    "reason": "owned_orphan",
                    "termination_eligible": True,
                }
            ]
        ),
        raising=True,
    )

    def _terminate_vm(vm_id: str) -> bool:
        terminated_vm_ids.append(vm_id)
        return True

    monkeypatch.setattr(
        service_mod,
        "MacOSVirtualizationHelperClient",
        lambda: SimpleNamespace(terminate_vm=_terminate_vm),
        raising=True,
    )

    result = service.repair_macos_reconciliation(
        terminate_orphaned_vms=True,
        dry_run=False,
    )

    assert terminated_vm_ids == ["vm-orphan"]
    assert result["summary"]["orphaned_vms"] == 1
    assert result["summary"]["terminated_orphaned_vms"] == 1
    assert result["actions"] == [
        {
            "type": "terminate_orphaned_vm",
            "session_id": None,
            "vm_id": "vm-orphan",
            "status": "terminated",
            "reason": "owned_orphan",
            "termination_eligible": True,
        }
    ]


def test_repair_orphan_vm_termination_false_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    terminated_vm_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "owned_orphaned_vm",
                    "vm_id": "vm-orphan",
                    "reason": "owned_orphan",
                    "termination_eligible": True,
                }
            ]
        ),
        raising=True,
    )

    def _terminate_vm(vm_id: str) -> bool:
        terminated_vm_ids.append(vm_id)
        return False

    monkeypatch.setattr(
        service_mod,
        "MacOSVirtualizationHelperClient",
        lambda: SimpleNamespace(terminate_vm=_terminate_vm),
        raising=True,
    )

    result = service.repair_macos_reconciliation(
        terminate_orphaned_vms=True,
        dry_run=False,
    )

    assert terminated_vm_ids == ["vm-orphan"]
    assert result["summary"]["orphaned_vms"] == 1
    assert result["summary"]["terminated_orphaned_vms"] == 0
    assert result["actions"][0]["status"] == "missing"
    assert result["actions"][0]["termination_eligible"] is True


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("owned_orphaned_vm", "owned_orphan"),
        ("unknown_orphaned_vm", "unknown_ownership"),
        ("foreign_orphaned_vm", "foreign_owner"),
        ("orphaned_vm", "session_missing"),
    ],
)
def test_repair_ineligible_or_unknown_orphan_vm_skips_without_helper_call(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    reason: str,
) -> None:
    terminated_vm_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": status,
                    "vm_id": "vm-skip",
                    "reason": reason,
                    "termination_eligible": False,
                    "template_id": "vz_linux:bundle-skip",
                    "planning_source": "image_store",
                    "run_manifest_path": "/tmp/image-store/runs/run-skip/manifest.json",
                    "run_manifest_present": False,
                }
            ]
        ),
        raising=True,
    )
    monkeypatch.setattr(
        service_mod,
        "MacOSVirtualizationHelperClient",
        lambda: SimpleNamespace(terminate_vm=terminated_vm_ids.append),
        raising=True,
    )

    result = service.repair_macos_reconciliation(
        terminate_orphaned_vms=True,
        dry_run=False,
    )

    assert terminated_vm_ids == []
    assert result["summary"]["orphaned_vms"] == 1
    assert result["summary"]["terminated_orphaned_vms"] == 0
    assert result["actions"] == [
        {
            "type": "skip_orphaned_vm",
            "session_id": None,
            "vm_id": "vm-skip",
            "status": "skipped",
            "reason": reason,
            "termination_eligible": False,
            "template_id": "vz_linux:bundle-skip",
            "planning_source": "image_store",
            "run_manifest_path": "/tmp/image-store/runs/run-skip/manifest.json",
            "run_manifest_present": False,
        }
    ]


def test_repair_summary_counts_all_orphan_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {"status": "owned_orphaned_vm", "vm_id": "vm-owned", "termination_eligible": True},
                {"status": "unknown_orphaned_vm", "vm_id": "vm-unknown", "termination_eligible": False},
                {"status": "foreign_orphaned_vm", "vm_id": "vm-foreign", "termination_eligible": False},
                {"status": "orphaned_vm", "vm_id": "vm-legacy", "termination_eligible": False},
            ]
        ),
        raising=True,
    )

    result = service.repair_macos_reconciliation()

    assert result["summary"]["orphaned_vms"] == 4
    assert result["actions"] == []


@pytest.mark.parametrize(
    ("helper_exc", "expected_reason"),
    [
        (
            MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable"),
            "macos_virtualization_helper_unavailable",
        ),
        (
            MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_error"),
            "macos_virtualization_helper_protocol_mismatch",
        ),
        (
            MacOSVirtualizationHelperFailure("vm_shutdown_denied", "helper refused termination"),
            "vm_shutdown_denied",
        ),
    ],
)
def test_repair_orphan_vm_termination_helper_errors_preserve_reason(
    monkeypatch: pytest.MonkeyPatch,
    helper_exc: Exception,
    expected_reason: str,
) -> None:
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "owned_orphaned_vm",
                    "vm_id": "vm-orphan",
                    "reason": "owned_orphan",
                    "termination_eligible": True,
                }
            ]
        ),
        raising=True,
    )

    def _terminate_vm(vm_id: str) -> bool:
        raise helper_exc

    monkeypatch.setattr(
        service_mod,
        "MacOSVirtualizationHelperClient",
        lambda: SimpleNamespace(terminate_vm=_terminate_vm),
        raising=True,
    )

    with pytest.raises(service_mod.SandboxReconciliationRepairError) as exc_info:
        service.repair_macos_reconciliation(
            terminate_orphaned_vms=True,
            dry_run=False,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.reason == expected_reason


def test_repair_orphan_vm_termination_unexpected_exception_maps_to_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = SimpleNamespace(delete_vz_session_control=lambda session_id: True)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(service, "_active_session_run_count", lambda session_id: 0)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(
            items=[
                {
                    "status": "owned_orphaned_vm",
                    "vm_id": "vm-orphan",
                    "reason": "owned_orphan",
                    "termination_eligible": True,
                }
            ]
        ),
        raising=True,
    )

    def _terminate_vm(vm_id: str) -> bool:
        raise RuntimeError("helper disconnected")

    monkeypatch.setattr(
        service_mod,
        "MacOSVirtualizationHelperClient",
        lambda: SimpleNamespace(terminate_vm=_terminate_vm),
        raising=True,
    )

    with pytest.raises(service_mod.SandboxReconciliationRepairError) as exc_info:
        service.repair_macos_reconciliation(
            terminate_orphaned_vms=True,
            dry_run=False,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.reason == "vz_orphan_vm_termination_failed"


@pytest.mark.parametrize(
    "reason",
    [
        "macos_virtualization_helper_unavailable",
        "macos_virtualization_helper_protocol_mismatch",
    ],
)
def test_repair_helper_unavailable_or_protocol_mismatch_raises_for_mutating_run(monkeypatch, reason: str) -> None:
    deleted_session_ids: list[str] = []
    orch = SimpleNamespace(delete_vz_session_control=deleted_session_ids.append)
    service = _service_with_orchestrator(orch)
    monkeypatch.setattr(
        service_mod,
        "collect_vz_reconciliation",
        lambda *args, **kwargs: _reconciliation_report(reasons=[reason]),
        raising=True,
    )

    with pytest.raises(service_mod.SandboxReconciliationRepairError) as exc_info:
        service.repair_macos_reconciliation(dry_run=False)

    assert exc_info.value.status_code == 503
    assert exc_info.value.reason == reason
    assert deleted_session_ids == []
