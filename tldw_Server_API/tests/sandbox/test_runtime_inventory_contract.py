from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (
    RuntimePreflightResult,
    normalize_runtime_reasons,
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
