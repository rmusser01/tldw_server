from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperVMListReply,
    HelperVMMetadata,
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


class _FailureHelper:
    def list_vms(self) -> HelperVMListReply:
        raise MacOSVirtualizationHelperFailure("helper_internal_error", "list failed")


def _metadata(
    *,
    owner: str = "tldw",
    runtime: str = "vz_linux",
    run_id: str = "run-owned",
    session_id: str = "",
    session_mode: bool = False,
    template_id: str = "",
    run_manifest_path: str = "",
    planning_source: str = "",
    created_at: str = "2026-04-30T18:00:00Z",
) -> HelperVMMetadata:
    return HelperVMMetadata(
        owner=owner,
        runtime=runtime,
        run_id=run_id,
        session_id=session_id,
        session_mode=session_mode,
        template_id=template_id,
        template_path="/tmp/template",
        run_manifest_path=run_manifest_path,
        planning_source=planning_source,
        workspace_path="/tmp/workspace",
        created_at=created_at,
    )


def _vm(
    vm_id: str,
    *,
    state: str = "running",
    healthy: bool = True,
    metadata: HelperVMMetadata | None = None,
) -> HelperVMStatusReply:
    return HelperVMStatusReply(
        protocol_version="1",
        helper_version="0.1.0",
        vm_id=vm_id,
        state=state,
        healthy=healthy,
        metadata=metadata or HelperVMMetadata(),
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
        "unknown_orphaned_vm",
    }
    assert helper.terminated_vm_ids == []
    assert helper.deleted_vm_ids == []


def test_reconciliation_reports_template_context_for_persisted_sessions() -> None:
    helper = _FakeHelper(
        [
            _vm(
                "vm-live",
                metadata=_metadata(
                    run_id="run-live",
                    template_id="vz_linux:bundle-a",
                ),
            ),
            _vm(
                "vm-unhealthy",
                healthy=False,
                metadata=_metadata(
                    run_id="run-unhealthy",
                    template_id="vz_linux:bundle-b",
                ),
            ),
        ]
    )

    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator(
            [
                {"id": "sess-live", "vm_id": "vm-live", "template_id": "vz_linux:bundle-a"},
                {"id": "sess-unhealthy", "vm_id": "vm-unhealthy", "template_id": "vz_linux:bundle-c"},
            ]
        ),
        helper_client=helper,
    )

    items_by_session = {str(item["session_id"]): item for item in report["items"] if "session_id" in item}
    assert items_by_session["sess-live"]["persisted_template_id"] == "vz_linux:bundle-a"
    assert items_by_session["sess-live"]["helper_template_id"] == "vz_linux:bundle-a"
    assert items_by_session["sess-live"]["template_id_matches_persisted"] is True
    assert items_by_session["sess-unhealthy"]["persisted_template_id"] == "vz_linux:bundle-c"
    assert items_by_session["sess-unhealthy"]["helper_template_id"] == "vz_linux:bundle-b"
    assert items_by_session["sess-unhealthy"]["template_id_matches_persisted"] is False


def test_reconciliation_classifies_orphaned_vms_by_ownership_metadata():
    helper = _FakeHelper(
        [
            _vm("vm-owned", metadata=_metadata(run_id="run-owned")),
            _vm("vm-unknown"),
            _vm("vm-foreign-owner", metadata=_metadata(owner="other", run_id="run-foreign-owner")),
            _vm("vm-foreign-runtime", metadata=_metadata(runtime="vz_macos", run_id="run-foreign-runtime")),
            _vm("vm-missing-run", metadata=_metadata(run_id="")),
            _vm("vm-missing-created", metadata=_metadata(run_id="run-missing-created", created_at="")),
            _vm(
                "vm-missing-session",
                metadata=_metadata(run_id="run-missing-session", session_mode=True, session_id=""),
            ),
        ]
    )

    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([]),
        helper_client=helper,
    )

    assert report["orphaned_vm_ids"] == [
        "vm-foreign-owner",
        "vm-foreign-runtime",
        "vm-missing-created",
        "vm-missing-run",
        "vm-missing-session",
        "vm-owned",
        "vm-unknown",
    ]
    assert report["owned_orphaned_vm_ids"] == ["vm-owned"]
    assert report["unknown_orphaned_vm_ids"] == [
        "vm-missing-created",
        "vm-missing-run",
        "vm-missing-session",
        "vm-unknown",
    ]
    assert report["foreign_orphaned_vm_ids"] == ["vm-foreign-owner", "vm-foreign-runtime"]

    items_by_vm = {str(item["vm_id"]): item for item in report["items"] if "vm_id" in item}
    assert items_by_vm["vm-owned"]["status"] == "owned_orphaned_vm"
    assert items_by_vm["vm-owned"]["reason"] == "owned_orphan"
    assert items_by_vm["vm-owned"]["termination_eligible"] is True
    assert items_by_vm["vm-unknown"]["status"] == "unknown_orphaned_vm"
    assert items_by_vm["vm-unknown"]["reason"] == "unknown_ownership"
    assert items_by_vm["vm-unknown"]["termination_eligible"] is False
    assert items_by_vm["vm-missing-run"]["status"] == "unknown_orphaned_vm"
    assert items_by_vm["vm-missing-created"]["status"] == "unknown_orphaned_vm"
    assert items_by_vm["vm-missing-session"]["status"] == "unknown_orphaned_vm"
    assert items_by_vm["vm-foreign-owner"]["status"] == "foreign_orphaned_vm"
    assert items_by_vm["vm-foreign-owner"]["reason"] == "foreign_owner"
    assert items_by_vm["vm-foreign-owner"]["termination_eligible"] is False
    assert items_by_vm["vm-foreign-runtime"]["status"] == "foreign_orphaned_vm"


def test_reconciliation_downgrades_image_store_orphan_without_run_manifest(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "runs" / "run-owned" / "manifest.json"
    helper = _FakeHelper(
        [
            _vm(
                "vm-owned-image-store",
                metadata=_metadata(
                    run_id="run-owned",
                    template_id="vz_linux:bundle-owned",
                    run_manifest_path=str(missing_manifest),
                    planning_source="image_store",
                ),
            )
        ]
    )

    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([]),
        helper_client=helper,
    )

    assert report["owned_orphaned_vm_ids"] == []
    assert report["unknown_orphaned_vm_ids"] == ["vm-owned-image-store"]
    item = next(item for item in report["items"] if item.get("vm_id") == "vm-owned-image-store")
    assert item["status"] == "unknown_orphaned_vm"
    assert item["reason"] == "image_store_manifest_missing"
    assert item["termination_eligible"] is False
    assert item["planning_source"] == "image_store"
    assert item["template_id"] == "vz_linux:bundle-owned"
    assert item["run_manifest_path"] == str(missing_manifest)
    assert item["run_manifest_present"] is False


def test_reconciliation_treats_invalid_image_store_manifest_path_as_missing(monkeypatch) -> None:
    original_is_file = Path.is_file

    def _raising_is_file(path: Path) -> bool:
        if str(path) == "bad-manifest.json":
            raise ValueError("embedded null byte")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", _raising_is_file)
    helper = _FakeHelper(
        [
            _vm(
                "vm-owned-image-store",
                metadata=_metadata(
                    run_id="run-owned",
                    template_id="vz_linux:bundle-owned",
                    run_manifest_path="bad-manifest.json",
                    planning_source="image_store",
                ),
            )
        ]
    )

    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([]),
        helper_client=helper,
    )

    assert report["owned_orphaned_vm_ids"] == []
    assert report["unknown_orphaned_vm_ids"] == ["vm-owned-image-store"]
    item = next(item for item in report["items"] if item.get("vm_id") == "vm-owned-image-store")
    assert item["status"] == "unknown_orphaned_vm"
    assert item["reason"] == "image_store_manifest_missing"
    assert item["termination_eligible"] is False
    assert item["run_manifest_present"] is False


def test_reconciliation_reports_image_store_context_for_owned_orphan(tmp_path: Path) -> None:
    manifest_path = tmp_path / "runs" / "run-owned" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    helper = _FakeHelper(
        [
            _vm(
                "vm-owned-image-store",
                metadata=_metadata(
                    run_id="run-owned",
                    template_id="vz_linux:bundle-owned",
                    run_manifest_path=str(manifest_path),
                    planning_source="image_store",
                ),
            )
        ]
    )

    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([]),
        helper_client=helper,
    )

    assert report["owned_orphaned_vm_ids"] == ["vm-owned-image-store"]
    item = next(item for item in report["items"] if item.get("vm_id") == "vm-owned-image-store")
    assert item["status"] == "owned_orphaned_vm"
    assert item["termination_eligible"] is True
    assert item["planning_source"] == "image_store"
    assert item["template_id"] == "vz_linux:bundle-owned"
    assert item["run_id"] == "run-owned"
    assert item["run_manifest_path"] == str(manifest_path)
    assert item["run_manifest_present"] is True


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


def test_reconciliation_classifies_helper_failure():
    report = collect_vz_reconciliation(
        orchestrator=_FakeOrchestrator([{"id": "sess-live", "vm_id": "vm-live"}]),
        helper_client=_FailureHelper(),
    )

    assert report["computed"] is False
    assert report["reasons"] == ["macos_virtualization_helper_failure"]


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
        "owned_orphaned_vm_ids": [],
        "unknown_orphaned_vm_ids": [],
        "foreign_orphaned_vm_ids": [],
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
    assert report["unknown_orphaned_vm_ids"] == ["vm-orphan-a", "vm-orphan-b"]
    assert report["owned_orphaned_vm_ids"] == []
    assert report["foreign_orphaned_vm_ids"] == []
    assert report["items"] == sorted(
        report["items"],
        key=lambda item: (
            str(item.get("status") or ""),
            str(item.get("session_id") or ""),
            str(item.get("vm_id") or ""),
        ),
    )
