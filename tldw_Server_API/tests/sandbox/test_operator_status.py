from tldw_Server_API.app.core.Sandbox.operator_status import build_operator_status


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
