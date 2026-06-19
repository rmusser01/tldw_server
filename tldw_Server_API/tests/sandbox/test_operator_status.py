import pytest

from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    SandboxAdminOperatorStatusResponse,
)
from tldw_Server_API.app.core.Sandbox.operator_status import build_operator_status
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _runtime_diagnostics(*, ready: int = 1) -> dict[str, object]:
    return {
        "source": "feature_discovery",
        "summary": {
            "total": 2,
            "ready": ready,
            "unavailable": 1,
            "host_gated": 1,
            "scaffold": 0,
            "host_local_warning_runtimes": [],
            "repair_supported_runtimes": ["vz_linux"],
        },
        "runtimes": [
            {
                "name": "docker",
                "available": ready > 0,
                "implementation_state": "supported",
                "readiness": "ready" if ready > 0 else "unavailable",
                "reasons": [],
                "normalized_reasons": [],
                "normalized_reason_details": [],
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
                "reasons": ["vz_linux_unavailable"],
                "normalized_reasons": ["runtime_unavailable"],
                "normalized_reason_details": [],
                "boundary_class": "vm_grade",
                "vm_grade_isolation": True,
                "untrusted_eligible": True,
                "isolation_warnings": [],
                "strict_deny_all_supported": False,
                "strict_allowlist_supported": False,
                "session_reuse_model": "warm_vm",
                "requires_live_health_check": True,
                "repair_supported": True,
                "recommended_action": "check_runtime_readiness",
            },
        ],
    }


def _macos_diagnostics_unconfigured() -> dict[str, object]:
    return {
        "helper": {"configured": False, "ready": False, "reasons": []},
        "templates": {},
        "reconciliation": None,
        "image_store": {
            "configured": False,
            "registered_templates": 0,
            "run_manifests": 0,
            "gc_candidates": 0,
            "reasons": [],
        },
        "recovery_summary": None,
        "startup_warning_summary": {"present": False, "blocking": False, "codes": []},
    }


def _macos_diagnostics_configured_broken() -> dict[str, object]:
    macos = _macos_diagnostics_unconfigured()
    macos["helper"] = {
        "configured": True,
        "ready": False,
        "reasons": ["macos_virtualization_helper_unavailable"],
    }
    macos["templates"] = {
        "vz_linux": {
            "configured": True,
            "ready": False,
            "reasons": ["vz_linux_template_missing"],
        }
    }
    macos["recovery_summary"] = {
        "status": "unavailable",
        "severity": "error",
        "codes": ["vz_recovery_unavailable"],
        "counts": {},
        "repair_endpoint": None,
        "cleanup_plan_endpoint": None,
        "notes": ["Reconciliation did not compute."],
    }
    return macos


def _evidence_ready(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "configured": True,
        "source": "host_smoke_evidence",
        "available": True,
        "valid": True,
        "evidence_dir": "/tmp/evidence",
        "schema_version": 1,
        "created_at": "2026-06-19T11:59:00+00:00",
        "age_seconds": 60,
        "stale": False,
        "smoke_run_id": "smoke-123",
        "final_exit_code": 0,
        "phases": {},
        "cleanup": {},
        "runtime_pointers": {},
        "expected_files": {},
        "skip_flags": {
            "skip_build": False,
            "skip_sign": False,
            "include_failure_drills": False,
        },
        "reasons": [],
    }
    payload.update(overrides)
    return payload


def test_operator_status_ready_when_runtime_ready_and_vz_unconfigured() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["source"] == "sandbox_operator_status"
    assert payload["overall_status"] == "ready"
    assert payload["overall_severity"] == "info"
    assert payload["sections"]["evidence"]["status"] == "not_configured"
    assert "generated_at" not in payload


def test_operator_status_projects_unconfigured_evidence_without_degrading() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary={
            "configured": False,
            "source": "host_smoke_evidence",
            "reasons": ["evidence_not_configured"],
        },
    )

    assert payload["overall_status"] == "ready"
    assert payload["sections"]["evidence"]["status"] == "not_configured"
    assert payload["sections"]["evidence"]["configured"] is False
    assert payload["recommended_actions"] == []


def test_operator_status_projects_successful_evidence_as_ready() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary=_evidence_ready(),
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert payload["overall_status"] == "ready"
    assert payload["sections"]["evidence"]["status"] == "ready"
    assert payload["sections"]["evidence"]["final_exit_code"] == 0
    assert payload["sections"]["evidence"]["evidence_dir"] == "/tmp/evidence"
    assert model.sections["evidence"].status == "ready"


def test_operator_status_degrades_for_unavailable_evidence() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary={
            "configured": True,
            "source": "host_smoke_evidence",
            "available": False,
            "valid": False,
            "reasons": ["evidence_directory_missing"],
        },
    )

    assert payload["overall_status"] == "degraded"
    assert payload["sections"]["evidence"]["status"] == "unavailable"
    assert payload["sections"]["evidence"]["severity"] == "warning"
    assert payload["recommended_actions"][0]["code"] == "inspect_host_gated_evidence"


def test_operator_status_requires_action_for_failed_evidence() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary=_evidence_ready(final_exit_code=2),
    )

    assert payload["overall_status"] == "action_required"
    assert payload["overall_severity"] == "error"
    assert payload["sections"]["evidence"]["status"] == "action_required"
    assert payload["sections"]["evidence"]["severity"] == "error"
    assert payload["recommended_actions"][0]["code"] == "run_host_gated_smoke"


def test_operator_status_degrades_for_stale_or_skipped_evidence() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary=_evidence_ready(
            stale=True,
            skip_flags={
                "skip_build": True,
                "skip_sign": False,
                "include_failure_drills": False,
            },
        ),
    )

    assert payload["overall_status"] == "degraded"
    assert payload["sections"]["evidence"]["status"] == "degraded"
    assert payload["recommended_actions"][0]["code"] == "review_expected_skips"


def test_operator_status_keeps_failure_drills_false_informational() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary=_evidence_ready(
            skip_flags={
                "skip_build": False,
                "skip_sign": False,
                "include_failure_drills": False,
            }
        ),
    )

    assert payload["overall_status"] == "ready"
    assert payload["sections"]["evidence"]["status"] == "ready"
    assert payload["recommended_actions"] == []


def test_operator_status_flags_configured_broken_vz_as_action_required() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_configured_broken(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert payload["overall_status"] == "action_required"
    assert payload["overall_severity"] == "error"
    assert payload["sections"]["macos_vz"]["status"] == "action_required"
    assert payload["sections"]["macos_vz"]["severity"] == "error"
    assert payload["sections"]["macos_vz"]["reasons"] == [
        "macos_virtualization_helper_unavailable"
    ]
    assert payload["recommended_actions"][0]["code"] == "restore_helper_readiness"
    assert model.sections["macos_vz"].status == "action_required"


def test_operator_status_payload_validates_against_schema() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert model.source == "sandbox_operator_status"
    assert model.overall_status == "ready"


def test_operator_status_points_reconciliation_to_dry_run_repair() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["recovery_summary"] = {
        "status": "action_recommended",
        "severity": "warning",
        "codes": ["vz_stale_session_controls"],
        "counts": {"stale_session_controls": 1},
        "repair_endpoint": "/api/v1/sandbox/admin/macos-reconciliation/repair",
        "cleanup_plan_endpoint": None,
        "notes": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert payload["overall_status"] == "action_required"
    assert payload["recommended_actions"][0]["code"] == "run_repair_dry_run"
    assert payload["recommended_actions"][0]["dry_run_required"] is True
    assert model.sections["reconciliation"].status == "action_recommended"


def test_operator_status_accepts_healthy_reconciliation_schema_status() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["recovery_summary"] = {
        "status": "healthy",
        "severity": "ok",
        "codes": [],
        "counts": {},
        "repair_endpoint": None,
        "cleanup_plan_endpoint": None,
        "notes": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert model.sections["reconciliation"].status == "healthy"


def test_operator_status_normalizes_invalid_reconciliation_values() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["recovery_summary"] = {
        "status": "unexpected_status",
        "severity": "unexpected_severity",
        "codes": ["vz_recovery_unknown"],
        "counts": {},
        "repair_endpoint": None,
        "cleanup_plan_endpoint": None,
        "notes": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert model.sections["reconciliation"].status == "unknown"
    assert model.sections["reconciliation"].severity == "warning"
    assert model.sections["reconciliation"].reasons == ["vz_recovery_unknown"]


def test_operator_status_keeps_runtime_section_when_macos_section_unavailable() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics={"_section_error": "macos_diagnostics_failed"},
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "ready"
    assert payload["sections"]["macos_vz"]["status"] == "unknown"
    assert payload["sections"]["macos_vz"]["reasons"] == ["macos_diagnostics_failed"]
    assert payload["sections"]["reconciliation"]["severity"] == "warning"
    assert payload["sections"]["reconciliation"]["reasons"] == [
        "macos_diagnostics_failed"
    ]
    assert payload["overall_status"] == "degraded"


def test_operator_status_reports_unknown_when_runtime_diagnostics_unavailable() -> None:
    payload = build_operator_status(
        runtime_diagnostics={"_section_error": "runtime_diagnostics_failed"},
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "unknown"
    assert payload["sections"]["runtime_readiness"]["reasons"] == [
        "runtime_diagnostics_failed"
    ]
    assert payload["overall_status"] == "unknown"


def test_operator_status_treats_configured_image_store_reasons_as_unavailable() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["helper"] = {"configured": True, "ready": True, "reasons": []}
    macos["image_store"] = {
        "configured": True,
        "registered_templates": 0,
        "run_manifests": 0,
        "gc_candidates": 0,
        "reasons": ["image_store_root_missing"],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert model.sections["image_store"].status == "unavailable"
    assert model.sections["image_store"].severity == "error"
    assert model.sections["image_store"].reasons == ["image_store_root_missing"]
    assert payload["overall_status"] == "action_required"
    assert payload["recommended_actions"][0]["code"] == "restore_image_store"


def test_operator_status_treats_string_booleans_as_unknown_not_actionable() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["helper"] = {"configured": False, "ready": "false", "reasons": []}
    macos["image_store"] = {
        "configured": "false",
        "registered_templates": 0,
        "run_manifests": 0,
        "gc_candidates": 0,
        "reasons": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": "false", "codes": []},
    )

    assert payload["sections"]["macos_vz"]["status"] == "unknown"
    assert payload["sections"]["image_store"]["status"] == "unknown"
    assert payload["sections"]["startup_warnings"]["status"] == "unknown"
    assert payload["recommended_actions"] == []
    assert payload["overall_status"] == "degraded"


def test_operator_status_malformed_boolean_payload_validates_against_schema() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["helper"] = {"configured": False, "ready": "false", "reasons": []}
    macos["image_store"] = {
        "configured": "false",
        "registered_templates": 0,
        "run_manifests": 0,
        "gc_candidates": 0,
        "reasons": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": "false", "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert model.sections["macos_vz"].status == "unknown"
    assert model.sections["image_store"].status == "unknown"
    assert model.sections["startup_warnings"].status == "unknown"


def test_operator_status_rejects_malformed_runtime_integer_values() -> None:
    runtime = _runtime_diagnostics()
    runtime["summary"]["ready"] = True
    runtime["summary"]["total"] = float("inf")

    payload = build_operator_status(
        runtime_diagnostics=runtime,
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "unavailable"
    assert payload["summary"]["runtime_ready"] == 0
    assert payload["summary"]["runtime_total"] == 0
    assert payload["overall_status"] == "unavailable"


def test_operator_status_clamps_negative_count_diagnostics() -> None:
    runtime = _runtime_diagnostics()
    runtime["summary"]["ready"] = -1
    runtime["summary"]["total"] = -2
    macos = _macos_diagnostics_unconfigured()
    macos["image_store"]["gc_candidates"] = -3

    payload = build_operator_status(
        runtime_diagnostics=runtime,
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "unavailable"
    assert payload["sections"]["runtime_readiness"]["ready"] == 0
    assert payload["sections"]["runtime_readiness"]["total"] == 0
    assert payload["sections"]["image_store"]["gc_candidates"] == 0
    assert payload["summary"]["runtime_ready"] == 0
    assert payload["summary"]["runtime_total"] == 0
    assert payload["overall_status"] == "unavailable"


def test_operator_status_points_image_store_to_cleanup_plan() -> None:
    macos = _macos_diagnostics_unconfigured()
    macos["recovery_summary"] = {
        "status": "cleanup_recommended",
        "severity": "warning",
        "codes": ["image_store_gc_candidates"],
        "counts": {"gc_candidates": 2},
        "repair_endpoint": None,
        "cleanup_plan_endpoint": "/api/v1/sandbox/admin/macos-image-store/cleanup-plan",
        "notes": [],
    }

    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=macos,
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert payload["overall_status"] == "degraded"
    assert payload["recommended_actions"][0]["code"] == (
        "inspect_image_store_cleanup_plan"
    )
    assert payload["recommended_actions"][0]["dry_run_required"] is False
    assert model.sections["reconciliation"].status == "cleanup_recommended"


def test_service_operator_status_uses_existing_diagnostics(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())
    monkeypatch.setattr(svc, "macos_diagnostics", lambda: _macos_diagnostics_unconfigured())

    payload = svc.operator_status(
        startup_warning_summary={"present": False, "blocking": False, "codes": []}
    )

    assert payload["source"] == "sandbox_operator_status"
    assert payload["overall_status"] == "ready"


def test_service_operator_status_isolates_macos_diagnostics_failure(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())

    def fail_macos() -> dict[str, object]:
        raise OSError("boom")

    monkeypatch.setattr(svc, "macos_diagnostics", fail_macos)

    payload = svc.operator_status(
        startup_warning_summary={"present": False, "blocking": False, "codes": []}
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "ready"
    assert payload["sections"]["macos_vz"]["status"] == "unknown"
    assert payload["sections"]["macos_vz"]["reasons"] == ["macos_diagnostics_failed"]


def test_service_operator_status_reports_unknown_for_runtime_diagnostics_failure(
    monkeypatch,
) -> None:
    svc = SandboxService()

    def fail_runtime() -> dict[str, object]:
        raise OSError("boom")

    monkeypatch.setattr(svc, "runtime_diagnostics_summary", fail_runtime)
    monkeypatch.setattr(svc, "macos_diagnostics", lambda: _macos_diagnostics_unconfigured())

    payload = svc.operator_status(
        startup_warning_summary={"present": False, "blocking": False, "codes": []}
    )

    assert payload["sections"]["runtime_readiness"]["status"] == "unknown"
    assert payload["sections"]["runtime_readiness"]["reasons"] == [
        "runtime_diagnostics_failed"
    ]
    assert payload["overall_status"] == "unknown"


def test_service_operator_status_propagates_macos_runtime_error(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())

    def fail_macos() -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "macos_diagnostics", fail_macos)

    with pytest.raises(RuntimeError, match="boom"):
        svc.operator_status(
            startup_warning_summary={"present": False, "blocking": False, "codes": []}
        )


def test_service_operator_status_propagates_runtime_diagnostics_runtime_error(
    monkeypatch,
) -> None:
    svc = SandboxService()

    def fail_runtime() -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "runtime_diagnostics_summary", fail_runtime)

    with pytest.raises(RuntimeError, match="boom"):
        svc.operator_status(
            startup_warning_summary={"present": False, "blocking": False, "codes": []}
        )
