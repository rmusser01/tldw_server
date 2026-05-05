from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox import runtime_capabilities as runtime_caps
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (
    RUNTIME_ISOLATION_METADATA,
    RuntimePreflightResult,
    normalize_runtime_reasons,
    runtime_isolation_metadata,
)
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def test_feature_discovery_covers_all_core_runtime_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    discovery = SandboxService().feature_discovery()

    discovered = {str(item.get("name")) for item in discovery}
    assert discovered == {runtime.value for runtime in RuntimeType}


def test_worktree_discovery_reports_host_local_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_WORKTREE_AVAILABLE", "1")

    discovery = SandboxService().feature_discovery()
    worktree = next(item for item in discovery if item["name"] == "worktree")

    assert worktree["supported_trust_levels"] == ["trusted", "standard"]
    assert worktree["strict_deny_all_supported"] is False
    assert worktree["strict_allowlist_supported"] is False
    assert worktree["egress_allowlist_supported"] is False
    assert worktree["interactive_supported"] is False
    assert "not VM-grade" in str(worktree["notes"])


def test_feature_discovery_reports_structured_isolation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    discovery = {
        str(item.get("name")): item
        for item in SandboxService().feature_discovery()
    }

    assert set(discovery) == {runtime.value for runtime in RuntimeType}
    for runtime in RuntimeType:
        info = discovery[runtime.value]
        assert "boundary_class" in info
        assert "vm_grade_isolation" in info
        assert "untrusted_eligible" in info

    assert discovery["docker"]["boundary_class"] == "container"
    assert discovery["docker"]["vm_grade_isolation"] is False
    assert discovery["docker"]["untrusted_eligible"] is True

    for vm_runtime in ("firecracker", "lima", "vz_linux"):
        assert discovery[vm_runtime]["boundary_class"] == "vm_grade"
        assert discovery[vm_runtime]["vm_grade_isolation"] is True
        assert discovery[vm_runtime]["untrusted_eligible"] is True

    assert discovery["vz_macos"]["boundary_class"] == "vm_grade_scaffold"
    assert discovery["vz_macos"]["vm_grade_isolation"] is False
    assert discovery["vz_macos"]["untrusted_eligible"] is False

    for host_local_runtime in ("seatbelt", "worktree"):
        assert discovery[host_local_runtime]["boundary_class"] == "host_local"
        assert discovery[host_local_runtime]["vm_grade_isolation"] is False
        assert discovery[host_local_runtime]["untrusted_eligible"] is False


def test_feature_discovery_reports_host_local_isolation_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    discovery = {
        str(item.get("name")): item
        for item in SandboxService().feature_discovery()
    }

    expected_host_local_warnings = [
        "host_local_boundary",
        "not_vm_grade_isolation",
        "not_untrusted_eligible",
    ]
    for host_local_runtime in ("seatbelt", "worktree"):
        assert (
            discovery[host_local_runtime]["isolation_warnings"]
            == expected_host_local_warnings
        )

    for runtime_name, runtime_info in discovery.items():
        if runtime_name in {"seatbelt", "worktree"}:
            continue
        assert (
            "host_local_boundary"
            not in runtime_info["isolation_warnings"]
        )


def test_runtime_isolation_metadata_contract_covers_runtime_enum() -> None:
    assert set(RUNTIME_ISOLATION_METADATA) == set(RuntimeType)


def test_runtime_isolation_metadata_rejects_unknown_runtime() -> None:
    with pytest.raises(ValueError, match="No isolation metadata configured"):
        runtime_isolation_metadata("future_runtime")  # type: ignore[arg-type]


def test_feature_discovery_reports_structured_network_policy_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    discovery = {
        str(item.get("name")): item
        for item in SandboxService().feature_discovery()
    }

    assert set(discovery) == {runtime.value for runtime in RuntimeType}
    for runtime in RuntimeType:
        contract = discovery[runtime.value]["network_policy_contract"]
        assert set(contract) == {"deny_all", "allowlist"}
        for mode in ("deny_all", "allowlist"):
            assert set(contract[mode]) == {
                "support_state",
                "strict_enforcement",
                "readiness_source",
            }

    for host_local_runtime in ("seatbelt", "worktree"):
        contract = discovery[host_local_runtime]["network_policy_contract"]
        assert contract["deny_all"] == {
            "support_state": "unsupported",
            "strict_enforcement": False,
            "readiness_source": "not_applicable",
        }
        assert contract["allowlist"] == {
            "support_state": "unsupported",
            "strict_enforcement": False,
            "readiness_source": "not_applicable",
        }

    vz_linux_contract = discovery["vz_linux"]["network_policy_contract"]
    assert vz_linux_contract["deny_all"] == {
        "support_state": "host_gated",
        "strict_enforcement": True,
        "readiness_source": "runtime_preflight",
    }
    assert vz_linux_contract["allowlist"] == {
        "support_state": "unsupported",
        "strict_enforcement": False,
        "readiness_source": "not_applicable",
    }


def test_feature_discovery_reports_effective_network_policy_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    discovery = {
        str(item.get("name")): item
        for item in SandboxService().feature_discovery()
    }

    for runtime in RuntimeType:
        info = discovery[runtime.value]
        effective_support = runtime_caps.runtime_network_policy_effective_support(
            runtime,
            info["enforcement_ready"],
        )
        assert info["strict_deny_all_supported"] is effective_support["deny_all"]
        assert info["strict_allowlist_supported"] is effective_support["allowlist"]
        assert info["egress_allowlist_supported"] is effective_support["allowlist"]


def test_docker_discovery_does_not_advertise_allowlist_for_coarse_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_AVAILABLE", "1")
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "1")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "0")

    discovery = {
        str(item.get("name")): item
        for item in SandboxService().feature_discovery()
    }
    docker = discovery["docker"]

    assert docker["enforcement_ready"] == {"deny_all": True, "allowlist": False}
    assert docker["strict_allowlist_supported"] is False
    assert docker["egress_allowlist_supported"] is False
    assert "fall back to deny-all" in str(docker["notes"])


def test_firecracker_discovery_does_not_advertise_scaffold_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_FIRECRACKER_AVAILABLE", "1")
    monkeypatch.setenv("SANDBOX_FIRECRACKER_EGRESS_ENFORCEMENT", "1")
    monkeypatch.setenv("SANDBOX_FIRECRACKER_EGRESS_GRANULAR_ENFORCEMENT", "1")

    discovery = {
        str(item.get("name")): item
        for item in SandboxService().feature_discovery()
    }
    firecracker = discovery["firecracker"]

    assert firecracker["enforcement_ready"] == {
        "deny_all": True,
        "allowlist": False,
    }
    assert firecracker["strict_allowlist_supported"] is False
    assert firecracker["egress_allowlist_supported"] is False
    assert "scaffold/planned" in str(firecracker["notes"])


def test_settings_flag_logs_read_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSettings:
        @property
        def SANDBOX_EGRESS_ENFORCEMENT(self) -> str:
            raise RuntimeError("settings unavailable")

    class CapturingLogger:
        def __init__(self) -> None:
            self.warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def warning(self, *args: object, **kwargs: object) -> None:
            self.warnings.append((args, kwargs))

    logger = CapturingLogger()
    monkeypatch.delenv("SANDBOX_EGRESS_ENFORCEMENT", raising=False)
    monkeypatch.setattr(runtime_caps, "app_settings", BrokenSettings())
    monkeypatch.setattr(runtime_caps, "logger", logger, raising=False)

    assert (  # noqa: SLF001
        runtime_caps._settings_flag("SANDBOX_EGRESS_ENFORCEMENT") is False
    )
    assert logger.warnings
    assert "Failed to read sandbox readiness flag" in str(logger.warnings[0][0][0])


def test_runtime_network_policy_metadata_contract_covers_runtime_enum() -> None:
    assert hasattr(runtime_caps, "RUNTIME_NETWORK_POLICY_METADATA")
    assert set(runtime_caps.RUNTIME_NETWORK_POLICY_METADATA) == set(RuntimeType)


def test_runtime_network_policy_metadata_rejects_unknown_runtime() -> None:
    assert hasattr(runtime_caps, "runtime_network_policy_metadata")
    with pytest.raises(ValueError, match="No network policy metadata configured"):
        runtime_caps.runtime_network_policy_metadata("future_runtime")


def test_runtime_reason_normalization_maps_raw_runtime_reasons() -> None:
    """Verify raw runtime-specific reasons collapse into stable discovery codes."""

    assert normalize_runtime_reasons(
        [
            "docker_unavailable",
            "macos_required",
            "apple_silicon_required",
            "macos_virtualization_helper_unavailable",
            "macos_virtualization_helper_protocol_mismatch",
            "macos_helper_missing",
            "vz_linux_template_missing",
            "template_unconfigured",
            "strict_allowlist_not_supported",
            "seatbelt_standard_disabled",
            "limactl_missing",
            "permission_denied_host_enforcement",
            "real_execution_not_implemented",
            "image_store_unavailable: bad root",
            "unexpected_future_reason",
        ]
    ) == [
        "runtime_unavailable",
        "unsupported_os",
        "unsupported_arch",
        "helper_unavailable",
        "helper_protocol_mismatch",
        "helper_missing",
        "template_missing",
        "template_unconfigured",
        "network_policy_unsupported",
        "trust_policy_denied",
        "host_prerequisite_missing",
        "host_permission_denied",
        "feature_not_implemented",
        "image_store_unavailable",
        "unknown",
    ]


def test_runtime_reason_normalization_treats_none_as_empty() -> None:
    """Defensively handle absent reason lists from discovery callers."""

    assert normalize_runtime_reasons(None) == []  # type: ignore[arg-type]


def test_runtime_reason_normalization_groups_helper_protocol_variants() -> None:
    """Helper protocol failures should share one stable client-facing code."""

    assert normalize_runtime_reasons(  # nosec B101
        [
            "macos_virtualization_helper_protocol_mismatch",
            "macos_virtualization_helper_protocol_error",
            "macos_virtualization_helper_empty_response",
            "macos_virtualization_helper_invalid_json",
        ]
    ) == ["helper_protocol_mismatch"]


def test_feature_discovery_preserves_raw_reasons_and_adds_normalized_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery should expose stable reason codes without replacing raw reasons."""

    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")

    def _preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        del self, network_policy
        return {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=False,
                reasons=["docker_unavailable"],
            ),
            RuntimeType.firecracker: RuntimePreflightResult(
                runtime=RuntimeType.firecracker,
                available=False,
                reasons=["firecracker_unavailable", "/dev/kvm_missing"],
            ),
            RuntimeType.lima: RuntimePreflightResult(
                runtime=RuntimeType.lima,
                available=False,
                reasons=["limactl_missing", "strict_deny_all_not_supported"],
            ),
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=False,
                reasons=[
                    "macos_virtualization_helper_unavailable",
                    "vz_linux_template_missing",
                ],
            ),
            RuntimeType.vz_macos: RuntimePreflightResult(
                runtime=RuntimeType.vz_macos,
                available=False,
                reasons=["real_execution_not_implemented"],
            ),
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=False,
                reasons=["macos_required", "sandbox_exec_missing"],
                supported_trust_levels=["trusted"],
            ),
            RuntimeType.worktree: RuntimePreflightResult(
                runtime=RuntimeType.worktree,
                available=False,
                reasons=["unsupported_platform"],
                supported_trust_levels=["trusted", "standard"],
            ),
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _preflights)

    discovery = {item["name"]: item for item in SandboxService().feature_discovery()}

    assert discovery["docker"]["reasons"] == ["docker_unavailable"]
    assert discovery["docker"]["normalized_reasons"] == ["runtime_unavailable"]
    assert discovery["firecracker"]["normalized_reasons"] == [
        "runtime_unavailable",
        "host_prerequisite_missing",
    ]
    assert discovery["lima"]["normalized_reasons"] == [
        "host_prerequisite_missing",
        "network_policy_unsupported",
    ]
    assert discovery["vz_linux"]["reasons"] == [
        "macos_virtualization_helper_unavailable",
        "vz_linux_template_missing",
    ]
    assert discovery["vz_linux"]["normalized_reasons"] == [
        "helper_unavailable",
        "template_missing",
    ]
    assert discovery["seatbelt"]["normalized_reasons"] == [
        "unsupported_os",
        "host_prerequisite_missing",
    ]
    assert discovery["worktree"]["normalized_reasons"] == ["unsupported_os"]
