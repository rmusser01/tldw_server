from __future__ import annotations

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperVMListReply,
    HelperVMStatusReply,
)
from tldw_Server_API.app.core.Sandbox.vz_reconciliation import collect_vz_reconciliation


class _FakeOrchestrator:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_vz_session_controls(self) -> list[dict[str, object]]:
        return list(self._rows)


class _NoListerOrchestrator:
    pass


class _RaisingOrchestrator:
    def list_vz_session_controls(self) -> list[dict[str, object]]:
        raise RuntimeError("store unavailable")


class _FakeHelper:
    def __init__(self, vms: list[HelperVMStatusReply]) -> None:
        self._vms = vms
        self.terminated_vm_ids: list[str] = []
        self.deleted_vm_ids: list[str] = []

    def list_vms(self) -> HelperVMListReply:
        return HelperVMListReply(
            protocol_version="1",
            helper_version="0.1.0",
            vms=list(self._vms),
        )

    def terminate_vm(self, vm_id: str) -> bool:
        self.terminated_vm_ids.append(vm_id)
        return True

    def delete_vm(self, vm_id: str) -> bool:
        self.deleted_vm_ids.append(vm_id)
        return True


class _UnavailableHelper:
    def list_vms(self) -> HelperVMListReply:
        raise MacOSVirtualizationHelperUnavailable("helper socket missing")


class _ProtocolMismatchHelper:
    def list_vms(self) -> HelperVMListReply:
        raise MacOSVirtualizationHelperProtocolError("helper protocol mismatch")


def _vm(vm_id: str, *, state: str = "running", healthy: bool = True) -> HelperVMStatusReply:
    return HelperVMStatusReply(
        protocol_version="1",
        helper_version="0.1.0",
        vm_id=vm_id,
        state=state,
        healthy=healthy,
    )


def test_reconciliation_reports_healthy_stale_unhealthy_and_orphaned_vms():
    helper = _FakeHelper(
        [
            _vm("vm-live", state="running", healthy=True),
            _vm("vm-unhealthy", state="running", healthy=False),
            _vm("vm-orphan", state="running", healthy=True),
        ]
    )

    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator(
            [
                {"id": "sess-live", "vm_id": "vm-live"},
                {"id": "sess-stale", "vm_id": "vm-missing"},
                {"id": "sess-unhealthy", "vm_id": "vm-unhealthy"},
            ]
        ),
        helper_client=helper,
    )

    assert report["computed"] is True
    assert report["persisted_sessions"] == 3
    assert report["live_vms"] == 3
    assert report["healthy_session_ids"] == ["sess-live"]
    assert report["stale_session_ids"] == ["sess-stale"]
    assert report["unhealthy_session_ids"] == ["sess-unhealthy"]
    assert report["orphaned_vm_ids"] == ["vm-orphan"]
    assert {item["status"] for item in report["items"]} >= {
        "healthy",
        "stale_session",
        "unhealthy_vm",
        "orphaned_vm",
    }
    assert helper.terminated_vm_ids == []
    assert helper.deleted_vm_ids == []


def test_reconciliation_classifies_helper_unavailable():
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([{"id": "sess-live", "vm_id": "vm-live"}]),
        helper_client=_UnavailableHelper(),
    )

    assert report["computed"] is False
    assert "macos_virtualization_helper_unavailable" in report["reasons"]


def test_reconciliation_classifies_protocol_mismatch():
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([{"id": "sess-live", "vm_id": "vm-live"}]),
        helper_client=_ProtocolMismatchHelper(),
    )

    assert report["computed"] is False
    assert "macos_virtualization_helper_protocol_mismatch" in report["reasons"]


def test_reconciliation_marks_active_stale_sessions_as_skipped():
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator(
            [
                {"id": "sess-active", "vm_id": "vm-missing"},
                {"id": "sess-stale", "vm_id": "vm-also-missing"},
            ]
        ),
        helper_client=_FakeHelper([]),
        active_session_checker=lambda sid: sid == "sess-active",
    )

    assert report["computed"] is True
    assert report["skipped_active_session_ids"] == ["sess-active"]
    assert report["stale_session_ids"] == ["sess-stale"]
    assert any(item["status"] == "skipped_active_session" for item in report["items"])


def test_reconciliation_marks_active_unhealthy_sessions_as_skipped():
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator(
            [
                {"id": "sess-active", "vm_id": "vm-unhealthy"},
                {"id": "sess-unhealthy", "vm_id": "vm-also-unhealthy"},
            ]
        ),
        helper_client=_FakeHelper(
            [
                _vm("vm-unhealthy", state="running", healthy=False),
                _vm("vm-also-unhealthy", state="running", healthy=False),
            ]
        ),
        active_session_checker=lambda sid: sid == "sess-active",
    )

    assert report["computed"] is True
    assert report["skipped_active_session_ids"] == ["sess-active"]
    assert report["unhealthy_session_ids"] == ["sess-unhealthy"]
    assert any(item["status"] == "skipped_active_session" for item in report["items"])
    assert not any(
        item["status"] == "unhealthy_vm" and item.get("session_id") == "sess-active"
        for item in report["items"]
    )


def test_reconciliation_unavailable_without_orchestrator():
    report = collect_vz_reconciliation(orchestrator=None, helper_client=_FakeHelper([]))

    assert report == {
        "computed": False,
        "persisted_sessions": 0,
        "live_vms": 0,
        "healthy_session_ids": [],
        "stale_session_ids": [],
        "unhealthy_session_ids": [],
        "skipped_active_session_ids": [],
        "orphaned_vm_ids": [],
        "items": [],
        "reasons": ["vz_reconciliation_unavailable"],
    }


def test_reconciliation_unavailable_when_lister_missing_or_raising():
    for orchestrator in (_NoListerOrchestrator(), _RaisingOrchestrator()):
        report = collect_vz_reconciliation(orchestrator=orchestrator, helper_client=_FakeHelper([]))

        assert report["computed"] is False
        assert report["reasons"] == ["vz_reconciliation_unavailable"]


def test_reconciliation_orders_id_lists_and_items_deterministically():
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator(
            [
                {"id": "sess-stale-b", "vm_id": "vm-missing-b"},
                {"id": "sess-unhealthy-b", "vm_id": "vm-unhealthy-b"},
                {"id": "sess-stale-a", "vm_id": "vm-missing-a"},
                {"id": "sess-healthy-b", "vm_id": "vm-healthy-b"},
                {"id": "sess-unhealthy-a", "vm_id": "vm-unhealthy-a"},
                {"id": "sess-healthy-a", "vm_id": "vm-healthy-a"},
            ]
        ),
        helper_client=_FakeHelper(
            [
                _vm("vm-orphan-b", healthy=True),
                _vm("vm-unhealthy-b", healthy=False),
                _vm("vm-healthy-b", healthy=True),
                _vm("vm-unhealthy-a", healthy=False),
                _vm("vm-orphan-a", healthy=True),
                _vm("vm-healthy-a", healthy=True),
            ]
        ),
    )

    assert report["healthy_session_ids"] == ["sess-healthy-a", "sess-healthy-b"]
    assert report["stale_session_ids"] == ["sess-stale-a", "sess-stale-b"]
    assert report["unhealthy_session_ids"] == ["sess-unhealthy-a", "sess-unhealthy-b"]
    assert report["orphaned_vm_ids"] == ["vm-orphan-a", "vm-orphan-b"]
    assert report["items"] == sorted(
        report["items"],
        key=lambda item: (
            str(item.get("status") or ""),
            str(item.get("session_id") or ""),
            str(item.get("vm_id") or ""),
        ),
    )
