from __future__ import annotations

import json
from pathlib import Path

import pytest

import tldw_Server_API.app.core.Sandbox.macos_diagnostics as diagnostics_module
import tldw_Server_API.app.core.Sandbox.vz_reconciliation as reconciliation_module
from tldw_Server_API.app.core.Sandbox.image_store import SandboxImageStore
from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
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
            "templates": [
                {
                    "template_id": "vz_linux:sample",
                    "runtime": "vz_linux",
                    "template_name": "sample",
                    "artifact_format": "tldw_bundle",
                    "source_path": "/tmp/vz-linux-bundle",
                    "artifact_count": 2,
                    "artifact_size_bytes": 1024,
                    "oci_image_ref": None,
                    "oci_platform": None,
                    "oci_manifest_digest": None,
                    "oci_config_digest": None,
                    "oci_layer_digests": [],
                    "registry": None,
                    "imported_at": None,
                    "provenance": {"suite": "bookworm"},
                }
            ],
            "items": [],
            "reasons": [],
        },
        "observability": {
            "configured": True,
            "serial_log_dir": "/tmp/vz-serial",
            "helper_log_dir": "/tmp/helper-logs",
            "helper_log_dir_source": "env",
            "helper_logs": {
                "stdout": {
                    "path": "/tmp/helper-logs/helper.stdout.log",
                    "exists": False,
                    "size_bytes": None,
                },
                "stderr": {
                    "path": "/tmp/helper-logs/helper.stderr.log",
                    "exists": False,
                    "size_bytes": None,
                },
            },
            "live_vms": 1,
            "vms": [
                {
                    "vm_id": "vm-live",
                    "state": "running",
                    "healthy": True,
                    "run_id": "run-live",
                    "session_id": "sess-live",
                    "session_mode": True,
                    "serial_log": {
                        "path": "/tmp/vz-serial/vm-live.serial.log",
                        "exists": False,
                        "size_bytes": None,
                    },
                    "guest": {
                        "version": "1.0.0",
                        "workspace_root": "/workspace",
                        "capabilities_known": True,
                        "capabilities": ["exec", "output_cap_v1"],
                        "compatibility": "compatible",
                        "reasons": [],
                        "expected_workspace_root": "/workspace",
                        "required_capabilities": ["exec"],
                        "missing_required_capabilities": [],
                    },
                    "resource_snapshot": {"cpu_time_sec": 1},
                }
            ],
            "reasons": [],
        },
        "recovery_summary": {
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
    assert model.image_store is not None
    assert model.image_store.templates[0].artifact_format == "tldw_bundle"
    assert model.image_store.templates[0].provenance == {"suite": "bookworm"}
    assert model.observability is not None
    assert model.observability.live_vms == 1
    assert model.observability.vms[0].serial_log.path == "/tmp/vz-serial/vm-live.serial.log"
    assert model.observability.vms[0].guest.capabilities == ["exec", "output_cap_v1"]
    assert model.observability.vms[0].guest.compatibility == "compatible"
    assert model.observability.vms[0].guest.reasons == []
    assert model.observability.vms[0].resource_snapshot == {"cpu_time_sec": 1}
    assert model.recovery_summary is not None
    assert model.recovery_summary.status == "healthy"
    assert model.recovery_summary.severity == "ok"
    assert model.recovery_summary.counts["healthy_session_controls"] == 1


def test_recovery_summary_reports_healthy_when_no_issues() -> None:
    summary = diagnostics_module.summarize_recovery(
        reconciliation={
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
            "items": [],
            "reasons": [],
        },
        image_store={"gc_candidates": 0, "reasons": []},
        observability={"live_vms": 1, "reasons": []},
    )

    assert summary["status"] == "healthy"
    assert summary["severity"] == "ok"
    assert summary["codes"] == []
    assert summary["counts"]["healthy_session_controls"] == 1
    assert summary["recommended_action"] == "No recovery action needed."
    assert summary["repair_endpoint"] is None
    assert summary["cleanup_plan_endpoint"] is None


def test_recovery_summary_keeps_observability_only_failure_healthy() -> None:
    summary = diagnostics_module.summarize_recovery(
        reconciliation={
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
            "items": [],
            "reasons": [],
        },
        image_store={"gc_candidates": 0, "reasons": []},
        observability={"live_vms": 1, "reasons": ["serial_log_dir_not_configured"]},
    )

    assert summary["status"] == "healthy"
    assert summary["severity"] == "ok"
    assert summary["codes"] == []
    assert summary["recommended_action"] == "No recovery action needed."
    assert summary["repair_endpoint"] is None
    assert summary["cleanup_plan_endpoint"] is None
    assert summary["notes"] == ["Observability reasons: serial_log_dir_not_configured."]


def test_recovery_summary_reports_unavailable_when_reconciliation_uncomputed() -> None:
    summary = diagnostics_module.summarize_recovery(
        reconciliation={
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
        image_store={"gc_candidates": 0, "reasons": []},
        observability=None,
    )

    assert summary["status"] == "unavailable"
    assert summary["severity"] == "error"
    assert "vz_recovery_unavailable" in summary["codes"]
    assert summary["repair_endpoint"] is None
    assert summary["cleanup_plan_endpoint"] is None


def test_recovery_summary_recommends_repair_for_stale_unhealthy_and_owned_orphans() -> None:
    summary = diagnostics_module.summarize_recovery(
        reconciliation={
            "computed": True,
            "persisted_sessions": 4,
            "live_vms": 3,
            "healthy_session_ids": ["sess-live"],
            "stale_session_ids": ["sess-stale"],
            "unhealthy_session_ids": ["sess-unhealthy"],
            "skipped_active_session_ids": [],
            "orphaned_vm_ids": ["vm-owned"],
            "owned_orphaned_vm_ids": ["vm-owned"],
            "unknown_orphaned_vm_ids": [],
            "foreign_orphaned_vm_ids": [],
            "items": [],
            "reasons": [],
        },
        image_store={"gc_candidates": 0, "reasons": []},
        observability={"live_vms": 3, "reasons": []},
    )

    assert summary["status"] == "action_recommended"
    assert summary["severity"] == "warning"
    assert "vz_stale_session_controls" in summary["codes"]
    assert "vz_unhealthy_session_controls" in summary["codes"]
    assert "vz_owned_orphaned_vms" in summary["codes"]
    assert summary["counts"]["stale_session_controls"] == 1
    assert summary["counts"]["unhealthy_session_controls"] == 1
    assert summary["counts"]["owned_orphaned_vms"] == 1
    assert summary["repair_endpoint"] == "/api/v1/sandbox/admin/macos-reconciliation/repair"


def test_recovery_summary_recommends_cleanup_plan_for_image_store_candidates() -> None:
    summary = diagnostics_module.summarize_recovery(
        reconciliation={
            "computed": True,
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
            "reasons": [],
        },
        image_store={"gc_candidates": 2, "reasons": []},
        observability={"live_vms": 0, "reasons": []},
    )

    assert summary["status"] == "action_recommended"
    assert summary["severity"] == "warning"
    assert summary["codes"] == ["vz_image_store_gc_candidates"]
    assert summary["counts"]["image_store_gc_candidates"] == 2
    assert summary["repair_endpoint"] is None
    assert summary["cleanup_plan_endpoint"] == "/api/v1/sandbox/admin/macos-image-store/cleanup-plan"


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
    templates = {template["template_id"]: template for template in image_store["templates"]}
    assert templates[template_id]["artifact_format"] == "tldw_bundle"
    assert templates[template_id]["runtime"] == "vz_linux"
    assert templates[template_id]["template_name"] == "debian-bookworm-arm64"
    assert templates[template_id]["artifact_count"] == 2
    assert templates[template_id]["artifact_size_bytes"] == len(b"kernel") + len(b"rootfs")
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


def test_probe_image_store_reports_oci_template_metadata(monkeypatch, tmp_path) -> None:
    store_root = tmp_path / "image-store"
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=store_root)
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="oci-backed",
        disk_paths=[str(disk)],
        artifact_format="oci_image",
        oci_image_ref="registry.example/tldw/sandbox:bookworm",
        oci_platform="linux/arm64",
        oci_manifest_digest="sha256:" + "a" * 64,
        oci_config_digest="sha256:" + "b" * 64,
        oci_layer_digests=["sha256:" + "c" * 64],
        registry="registry.example",
        imported_at="2026-05-02T00:00:00+00:00",
        provenance={"suite": "bookworm"},
    )
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))

    data = diagnostics_module.probe_image_store()

    templates = {template["template_id"]: template for template in data["templates"]}
    assert templates[template_id]["artifact_format"] == "oci_image"
    assert templates[template_id]["oci_image_ref"] == "registry.example/tldw/sandbox:bookworm"
    assert templates[template_id]["oci_platform"] == "linux/arm64"
    assert templates[template_id]["oci_manifest_digest"] == "sha256:" + "a" * 64
    assert templates[template_id]["oci_config_digest"] == "sha256:" + "b" * 64
    assert templates[template_id]["oci_layer_digests"] == ["sha256:" + "c" * 64]
    assert templates[template_id]["registry"] == "registry.example"
    assert templates[template_id]["imported_at"] == "2026-05-02T00:00:00+00:00"
    assert templates[template_id]["provenance"] == {"suite": "bookworm"}


def test_probe_vz_linux_observability_reports_log_pointers_and_vm_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper_log_dir = tmp_path / "helper-logs"
    serial_log_dir = helper_log_dir / "serial"
    helper_log_dir.mkdir()
    serial_log_dir.mkdir()
    (helper_log_dir / "helper.stdout.log").write_bytes(b"helper stdout")
    (helper_log_dir / "helper.stderr.log").write_bytes(b"helper stderr")
    (serial_log_dir / "vm_serial_log.serial.log").write_bytes(b"boot log")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_LOG_DIR", str(helper_log_dir))
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR", str(serial_log_dir))

    class _FakeHelper:
        def list_vms(self) -> HelperVMListReply:
            return HelperVMListReply(
                protocol_version="1",
                helper_version="0.1.0",
                vms=[
                    HelperVMStatusReply(
                        protocol_version="1",
                        helper_version="0.1.0",
                        vm_id="vm/serial log",
                        state="running",
                        healthy=True,
                        metadata=HelperVMMetadata(
                            owner="tldw",
                            runtime="vz_linux",
                            run_id="run-live",
                            session_id="sess-live",
                            session_mode=True,
                        ),
                        details={
                            "guest_version": "1.0.0",
                            "guest_workspace_root": "/workspace",
                            "guest_capabilities_known": "true",
                            "guest_capabilities": "exec,output_cap_v1",
                            "cpu_time_sec": "7",
                            "cpu_count": "2",
                            "memory_size_mb": "1024",
                            "peak_rss_mb": 128,
                            "disk_read_bytes": "2048",
                            "unexpected_detail": "not surfaced",
                        },
                    )
                ],
            )

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    data = diagnostics_module.probe_vz_linux_observability()

    assert data["configured"] is True
    assert data["serial_log_dir"] == str(serial_log_dir)
    assert data["helper_log_dir"] == str(helper_log_dir)
    assert data["helper_logs"]["stdout"]["path"] == str(helper_log_dir / "helper.stdout.log")
    assert data["helper_logs"]["stdout"]["exists"] is True
    assert data["helper_logs"]["stdout"]["size_bytes"] == len(b"helper stdout")
    assert data["helper_logs"]["stderr"]["path"] == str(helper_log_dir / "helper.stderr.log")
    assert data["helper_logs"]["stderr"]["exists"] is True
    assert data["live_vms"] == 1
    assert data["reasons"] == []

    vm = data["vms"][0]
    assert vm["vm_id"] == "vm/serial log"
    assert vm["run_id"] == "run-live"
    assert vm["session_id"] == "sess-live"
    assert vm["serial_log"]["path"] == str(serial_log_dir / "vm_serial_log.serial.log")
    assert vm["serial_log"]["exists"] is True
    assert vm["serial_log"]["size_bytes"] == len(b"boot log")
    assert vm["guest"] == {
        "version": "1.0.0",
        "workspace_root": "/workspace",
        "capabilities_known": True,
        "capabilities": ["exec", "output_cap_v1"],
        "compatibility": "compatible",
        "reasons": [],
        "expected_workspace_root": "/workspace",
        "required_capabilities": ["exec"],
        "missing_required_capabilities": [],
    }
    assert vm["resource_snapshot"] == {
        "cpu_time_sec": 7,
        "cpu_count": 2,
        "memory_size_mb": 1024,
        "peak_rss_mb": 128,
        "disk_read_bytes": 2048,
    }


def test_probe_vz_linux_observability_classifies_guest_agent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeHelper:
        def list_vms(self) -> HelperVMListReply:
            return HelperVMListReply(
                protocol_version="1",
                helper_version="0.1.0",
                vms=[
                    HelperVMStatusReply(
                        protocol_version="1",
                        helper_version="0.1.0",
                        vm_id="vm-mismatch",
                        state="running",
                        healthy=True,
                        details={
                            "guest_version": "0.9.0",
                            "guest_workspace_root": "/var/empty",
                            "guest_capabilities_known": "true",
                            "guest_capabilities": "output_cap_v1",
                        },
                    )
                ],
            )

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    data = diagnostics_module.probe_vz_linux_observability()

    guest = data["vms"][0]["guest"]
    assert guest["version"] == "0.9.0"
    assert guest["workspace_root"] == "/var/empty"
    assert guest["capabilities_known"] is True
    assert guest["capabilities"] == ["output_cap_v1"]
    assert guest["compatibility"] == "mismatch"
    assert guest["expected_workspace_root"] == "/workspace"
    assert guest["required_capabilities"] == ["exec"]
    assert guest["missing_required_capabilities"] == ["exec"]
    assert guest["reasons"] == [
        "vz_linux_guest_agent_workspace_mismatch",
        "vz_linux_guest_agent_required_capability_missing",
    ]


def test_probe_vz_linux_observability_handles_helper_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnavailableHelper:
        def list_vms(self) -> HelperVMListReply:
            raise MacOSVirtualizationHelperUnavailable("helper down")

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _UnavailableHelper)

    data = diagnostics_module.probe_vz_linux_observability()

    assert data["live_vms"] == 0
    assert data["vms"] == []
    assert data["reasons"] == ["macos_virtualization_helper_unavailable"]


def test_collect_macos_diagnostics_maps_unexpected_helper_list_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macos_host(monkeypatch)

    class _BuggyListHelper:
        def ping(self) -> HelperPingReply:
            return HelperPingReply(
                protocol_version="1",
                helper_version="0.1.0",
                status="ok",
                details={"transport": "unix"},
            )

        def list_vms(self) -> HelperVMListReply:
            raise KeyError("unexpected helper payload shape")

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _BuggyListHelper)
    monkeypatch.setattr(
        diagnostics_module,
        "collect_runtime_preflights",
        lambda network_policy="deny_all": {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=["vz_reconciliation_unavailable"],
                execution_mode="none",
            ),
        },
    )

    data = diagnostics_module.collect_macos_diagnostics()

    assert data["reconciliation"]["reasons"] == ["vz_reconciliation_unavailable"]
    assert data["observability"]["reasons"] == ["vz_linux_observability_unavailable"]
    assert data["recovery_summary"]["status"] == "unavailable"


def test_probe_vz_linux_observability_default_log_dir_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_LOG_DIR", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR", raising=False)

    class _FakeHelper:
        def list_vms(self) -> HelperVMListReply:
            return HelperVMListReply(protocol_version="1", helper_version="0.1.0", vms=[])

    monkeypatch.setattr(diagnostics_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    data = diagnostics_module.probe_vz_linux_observability()

    assert data["configured"] is False
    assert data["helper_log_dir_source"] == "default"
    assert data["live_vms"] == 0
    assert data["reasons"] == []


def test_collect_macos_diagnostics_fetches_live_vms_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_macos_host(monkeypatch)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")

    class _FakeHelper:
        list_calls = 0

        def ping(self) -> HelperPingReply:
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

        def list_vms(self) -> HelperVMListReply:
            _FakeHelper.list_calls += 1
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
        def list_vz_session_controls(self) -> list[dict[str, str]]:
            return [{"id": "sess-live", "vm_id": "vm-live"}]

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

    data = diagnostics_module.collect_macos_diagnostics(_FakeOrchestrator())

    assert _FakeHelper.list_calls == 1
    assert data["reconciliation"]["live_vms"] == 1
    assert data["observability"]["live_vms"] == 1


def test_observability_path_parser_rejects_embedded_nul() -> None:
    assert diagnostics_module._path_from_text("bad\x00path") is None
