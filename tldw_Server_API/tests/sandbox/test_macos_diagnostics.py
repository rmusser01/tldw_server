from __future__ import annotations

import json

import tldw_Server_API.app.core.Sandbox.macos_diagnostics as diagnostics_module
import tldw_Server_API.app.core.Sandbox.vz_reconciliation as reconciliation_module
from tldw_Server_API.app.core.Sandbox.image_store import SandboxImageStore
from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperProtocolError,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperPingReply,
    HelperVMListReply,
    HelperVMMetadata,
    HelperVMStatusReply,
)
from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult


def _patch_macos_host(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostics_module,
        "vz_host_facts",
        lambda: {
            "os": "darwin",
            "arch": "arm64",
            "apple_silicon": True,
        },
    )
    monkeypatch.setattr(diagnostics_module.platform, "mac_ver", lambda: ("15.0", ("", "", ""), ""))


def _sample_diagnostics_payload() -> dict:
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
            "configured": True,
            "path": "/tmp/helper",
            "exists": True,
            "executable": True,
            "ready": True,
            "transport": "fake",
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "reasons": [],
        },
        "templates": {
            "vz_linux": {
                "configured": True,
                "ready": True,
                "source": "/tmp/vz-linux.img",
                "reasons": [],
            },
        },
        "runtimes": {
            "vz_linux": {
                "available": True,
                "supported_trust_levels": ["trusted", "standard", "untrusted"],
                "reasons": [],
                "execution_mode": "fake",
                "remediation": None,
            }
        },
        "reconciliation": {
            "computed": True,
            "persisted_sessions": 1,
            "live_vms": 1,
            "healthy_session_ids": ["sess-live"],
            "stale_session_ids": [],
            "unhealthy_session_ids": [],
            "skipped_active_session_ids": [],
            "orphaned_vm_ids": [],
            "owned_orphaned_vm_ids": [],
            "unknown_orphaned_vm_ids": [],
            "foreign_orphaned_vm_ids": [],
            "items": [
                {
                    "status": "healthy",
                    "session_id": "sess-live",
                    "vm_id": "vm-live",
                    "state": "running",
                    "healthy": True,
                }
            ],
            "reasons": [],
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


def test_collect_macos_diagnostics_reports_missing_helper_and_templates(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC", raising=False)

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["host"]["supported"] is True
    assert data["helper"]["configured"] is False
    assert data["helper"]["path"] is None
    assert data["helper"]["ready"] is False
    assert data["helper"]["protocol_version"] is None
    assert data["helper"]["helper_version"] is None
    assert data["templates"]["vz_linux"]["configured"] is False
    assert data["templates"]["vz_linux"]["source"] is None
    assert data["templates"]["vz_linux"]["ready"] is False
    assert "macos_helper_missing" in data["runtimes"]["vz_linux"]["reasons"]
    assert data["runtimes"]["vz_linux"]["execution_mode"] == "none"


def test_collect_macos_diagnostics_reports_real_vz_linux_execution_mode(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FakeHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            assert request["runtime"] == "vz_linux"
            return {
                "template_id": "vz_linux:ubuntu-24.04",
                "source": request["template"],
                "ready": True,
                "reasons": [],
            }

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=True,
                reasons=[],
                execution_mode="real",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["runtimes"]["vz_linux"]["available"] is True
    assert data["runtimes"]["vz_linux"]["execution_mode"] == "real"
    assert data["runtimes"]["vz_linux"]["reasons"] == []
    assert data["helper"]["ready"] is True
    assert data["helper"]["transport"] == "unix"
    assert data["helper"]["protocol_version"] == "1"
    assert data["helper"]["helper_version"] == "0.1.0"
    assert data["templates"]["vz_linux"]["ready"] is True
    assert data["templates"]["vz_linux"]["reasons"] == []


def test_collect_macos_diagnostics_separates_policy_from_host_readiness(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", raising=False)

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["runtimes"]["seatbelt"]["supported_trust_levels"] == ["trusted"]
    assert data["runtimes"]["seatbelt"]["available"] in (True, False)


def test_collect_macos_diagnostics_uses_optional_operator_metadata_env(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_PATH", "/tmp/macos-helper")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["helper"]["path"] == "/tmp/macos-helper"
    assert data["templates"]["vz_linux"]["source"] == "/tmp/vz-linux.img"


def test_collect_macos_diagnostics_reports_helper_validated_template_failure(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FailingTemplateHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": None,
                "source": request["template"],
                "ready": False,
                "reasons": ["template_invalid"],
            }

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FailingTemplateHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=["template_invalid"],
                execution_mode="none",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["templates"]["vz_linux"]["configured"] is True
    assert data["templates"]["vz_linux"]["ready"] is False
    assert data["templates"]["vz_linux"]["reasons"] == ["template_invalid"]


def test_collect_macos_diagnostics_does_not_trust_ready_env_without_reachable_helper(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_PATH", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", raising=False)

    class _FailingHelper:
        def ping(self):
            raise diagnostics_module.MacOSVirtualizationHelperUnavailable(
                "macos_virtualization_helper_unavailable"
            )

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FailingHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["helper"]["configured"] is False
    assert data["helper"]["ready"] is False
    assert "macos_virtualization_helper_unavailable" in data["helper"]["reasons"]


def test_service_macos_diagnostics_returns_probe_payload(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    expected = _sample_diagnostics_payload()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.service.collect_macos_diagnostics",
        lambda orchestrator=None: expected,
    )

    svc = SandboxService()

    assert svc.macos_diagnostics() == expected


def test_admin_schema_accepts_macos_diagnostics_payload() -> None:
    from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
        SandboxAdminMacOSDiagnosticsResponse,
    )

    model = SandboxAdminMacOSDiagnosticsResponse.model_validate(_sample_diagnostics_payload())

    assert model.host.supported is True
    assert model.runtimes["vz_linux"].execution_mode == "fake"
    assert model.reconciliation is not None
    assert model.reconciliation.computed is True
    assert model.reconciliation.healthy_session_ids == ["sess-live"]
    assert model.reconciliation.owned_orphaned_vm_ids == []
    assert model.reconciliation.items


def test_admin_schema_accepts_startup_warning_summary() -> None:
    from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
        SandboxAdminMacOSDiagnosticsResponse,
    )

    payload = _sample_diagnostics_payload()
    payload["startup_warning_summary"] = {
        "present": True,
        "blocking": False,
        "codes": [
            "vz_stale_session_controls_detected",
            "vz_orphaned_vms_detected",
        ],
    }

    model = SandboxAdminMacOSDiagnosticsResponse.model_validate(payload)

    assert model.startup_warning_summary is not None
    assert model.startup_warning_summary.present is True
    assert model.startup_warning_summary.blocking is False
    assert model.startup_warning_summary.codes == [
        "vz_stale_session_controls_detected",
        "vz_orphaned_vms_detected",
    ]


def test_collect_macos_diagnostics_classifies_helper_protocol_mismatch(monkeypatch) -> None:
    class _FakeHelper:
        def ping(self):
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    data = diagnostics_module.collect_macos_diagnostics()

    assert "macos_virtualization_helper_protocol_mismatch" in data["helper"]["reasons"]
    assert "macos_helper_missing" not in data["helper"]["reasons"]
    assert (
        diagnostics_module._remediation_for_reasons(["macos_virtualization_helper_protocol_mismatch"])
        == "Update the macOS virtualization helper and Python client to compatible protocol versions."
    )


def test_collect_macos_diagnostics_classifies_template_protocol_mismatch(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FakeHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=["macos_virtualization_helper_protocol_mismatch"],
                execution_mode="none",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert "macos_virtualization_helper_protocol_mismatch" in data["templates"]["vz_linux"]["reasons"]


def test_collect_macos_diagnostics_reports_reconciliation_mismatches(monkeypatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FakeHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": "vz_linux:ubuntu-24.04",
                "source": request["template"],
                "ready": True,
                "reasons": [],
            }

        def list_vms(self):
            return HelperVMListReply(
                protocol_version="1",
                helper_version="0.1.0",
                vms=[
                    HelperVMStatusReply(
                        protocol_version="1",
                        helper_version="0.1.0",
                        vm_id="vm-live",
                        state="running",
                        healthy=True,
                    )
                ],
            )

    class _FakeOrchestrator:
        def list_vz_session_controls(self):
            return [
                {"id": "sess-stale", "vm_id": "vm-stale"},
                {"id": "sess-live", "vm_id": "vm-live"},
            ]

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(reconciliation_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=True,
                reasons=[],
                execution_mode="real",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics(_FakeOrchestrator())

    assert data["reconciliation"]["computed"] is True
    assert data["reconciliation"]["persisted_sessions"] == 2
    assert data["reconciliation"]["live_vms"] == 1
    assert data["reconciliation"]["healthy_session_ids"] == ["sess-live"]
    assert data["reconciliation"]["stale_session_ids"] == ["sess-stale"]
    assert data["reconciliation"]["items"]
    assert data["reconciliation"]["orphaned_vm_ids"] == []


def test_collect_macos_diagnostics_reports_image_store_correlation(monkeypatch, tmp_path) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)

    store_root = tmp_path / "image-store"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "boot_mode": "linux_direct"}),
        encoding="utf-8",
    )
    store = SandboxImageStore(root_path=store_root)
    template_id = store.register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
    )
    live_manifest = store.prepare_run_clone(template_id=template_id, run_id="run-live")
    store.prepare_run_clone(template_id=template_id, run_id="run-manifest-only")
    store.prepare_run_clone(template_id=template_id, run_id="run-inactive")
    inactive_rootfs = store_root / "runs" / "run-inactive" / "rootfs.img"
    inactive_rootfs.write_bytes(b"clone")
    legacy_run = store_root / "runs" / "run-legacy"
    legacy_run.mkdir(parents=True)
    (legacy_run / "leftover.img").write_bytes(b"legacy")

    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", str(bundle))

    class _FakeHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": template_id,
                "source": request["template"],
                "ready": True,
                "reasons": [],
            }

        def list_vms(self):
            return HelperVMListReply(
                protocol_version="1",
                helper_version="0.1.0",
                vms=[
                    HelperVMStatusReply(
                        protocol_version="1",
                        helper_version="0.1.0",
                        vm_id="vm-live",
                        state="running",
                        healthy=True,
                        metadata=HelperVMMetadata(
                            owner="tldw",
                            runtime="vz_linux",
                            run_id="run-live",
                            session_id="sess-live",
                            session_mode=True,
                            template_id=template_id,
                            template_path=str(bundle),
                            run_manifest_path=str(store_root / "runs" / live_manifest.run_id / "manifest.json"),
                            planning_source="image_store",
                            workspace_path="/tmp/workspace",
                            created_at="2026-04-30T18:00:00Z",
                        ),
                    )
                ],
            )

    class _FakeOrchestrator:
        def list_vz_session_controls(self):
            return [{"id": "sess-live", "vm_id": "vm-live", "template_id": template_id}]

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(reconciliation_module, "MacOSVirtualizationHelperClient", _FakeHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=True,
                reasons=[],
                execution_mode="real",
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["macos_virtualization_helper_unavailable"],
                execution_mode="none",
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["seatbelt_unavailable"],
                execution_mode="none",
                supported_trust_levels=["trusted"],
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics(_FakeOrchestrator())

    image_store = data["image_store"]
    assert image_store["configured"] is True
    assert image_store["root_path"] == str(store_root)
    assert image_store["registered_templates"] == 1
    assert image_store["run_manifests"] == 3
    assert image_store["gc_candidates"] == 3
    items_by_run = {item["run_id"]: item for item in image_store["items"]}
    assert items_by_run["run-live"]["matched_vm_id"] == "vm-live"
    assert items_by_run["run-live"]["matched_reconciliation_status"] == "healthy"
    assert items_by_run["run-live"]["gc_reason"] is None
    assert items_by_run["run-manifest-only"]["gc_reason"] == "planning_only_run_manifest"
    assert items_by_run["run-manifest-only"]["run_manifest_present"] is True
    assert items_by_run["run-inactive"]["gc_reason"] == "inactive_run"
    assert items_by_run["run-legacy"]["gc_reason"] == "legacy_run_directory"
    assert items_by_run["run-legacy"]["run_manifest_present"] is False


def test_probe_image_store_does_not_create_missing_root(monkeypatch, tmp_path) -> None:
    store_root = tmp_path / "missing-image-store"
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))

    data = diagnostics_module.probe_image_store()

    assert data["configured"] is True
    assert data["root_path"] == str(store_root)
    assert data["registered_templates"] == 0
    assert data["run_manifests"] == 0
    assert data["gc_candidates"] == 0
    assert data["items"] == []
    assert "image_store_root_missing" in data["reasons"]
    assert not store_root.exists()
