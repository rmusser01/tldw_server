from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .models import RuntimeType

RuntimeImplementationState = Literal[
    "supported",
    "unsupported",
    "scaffold",
    "host_gated",
    "not_applicable",
]
RuntimeReasonCode = Literal[
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

RUNTIME_IMPLEMENTATION_STATES: Mapping[RuntimeType, RuntimeImplementationState] = {
    RuntimeType.docker: "supported",
    RuntimeType.firecracker: "host_gated",
    RuntimeType.lima: "host_gated",
    RuntimeType.vz_linux: "host_gated",
    RuntimeType.vz_macos: "scaffold",
    RuntimeType.seatbelt: "host_gated",
    RuntimeType.worktree: "supported",
}

_RUNTIME_REASON_CODE_MAP: Mapping[str, RuntimeReasonCode] = {
    "/dev/kvm_missing": "host_prerequisite_missing",
    "apple_silicon_required": "unsupported_arch",
    "docker_unavailable": "runtime_unavailable",
    "firecracker_binary_missing": "host_prerequisite_missing",
    "firecracker_unavailable": "runtime_unavailable",
    "git_too_old_or_missing": "host_prerequisite_missing",
    "image_store_root_missing": "image_store_unavailable",
    "image_store_root_not_directory": "image_store_unavailable",
    "limactl_missing": "host_prerequisite_missing",
    "macos_helper_missing": "helper_missing",
    "macos_helper_not_executable": "helper_missing",
    "macos_helper_path_missing": "helper_missing",
    "macos_helper_path_unconfigured": "helper_missing",
    "macos_required": "unsupported_os",
    "macos_template_missing": "template_missing",
    "macos_virtualization_helper_protocol_mismatch": "helper_protocol_mismatch",
    "macos_virtualization_helper_unavailable": "helper_unavailable",
    "permission_denied_host_enforcement": "host_permission_denied",
    "real_execution_not_implemented": "feature_not_implemented",
    "sandbox_exec_missing": "host_prerequisite_missing",
    "seatbelt_standard_disabled": "trust_policy_denied",
    "seatbelt_unavailable": "runtime_unavailable",
    "strict_allowlist_not_supported": "network_policy_unsupported",
    "strict_deny_all_not_supported": "network_policy_unsupported",
    "template_missing": "template_missing",
    "template_unconfigured": "template_unconfigured",
    "trust_level_not_supported": "trust_policy_denied",
    "trust_level_requires_vm_runtime": "trust_policy_denied",
    "unshare_required_on_linux": "host_prerequisite_missing",
    "unsupported_network_policy": "network_policy_unsupported",
    "unsupported_platform": "unsupported_os",
    "virtiofsd_missing": "host_prerequisite_missing",
    "vz_linux_template_missing": "template_missing",
    "vz_linux_unavailable": "runtime_unavailable",
    "vz_macos_unavailable": "runtime_unavailable",
    "worktree_unavailable": "runtime_unavailable",
}


def runtime_implementation_state(runtime: RuntimeType) -> RuntimeImplementationState:
    """Return the roadmap maturity label for a runtime, independent of host availability."""
    return RUNTIME_IMPLEMENTATION_STATES.get(runtime, "unsupported")


def normalize_runtime_reason(reason: str) -> RuntimeReasonCode:
    """Return a stable client-facing code for a raw runtime preflight reason."""

    normalized = str(reason or "").strip()
    if not normalized:
        return "unknown"
    if normalized.startswith("macos_virtualization_helper_protocol_"):
        return "helper_protocol_mismatch"
    if normalized in {
        "macos_virtualization_helper_empty_response",
        "macos_virtualization_helper_invalid_json",
    }:
        return "helper_protocol_mismatch"
    if normalized.startswith("image_store_unavailable"):
        return "image_store_unavailable"
    return _RUNTIME_REASON_CODE_MAP.get(normalized, "unknown")


def normalize_runtime_reasons(
    reasons: list[str] | tuple[str, ...] | None,
) -> list[RuntimeReasonCode]:
    """Normalize raw runtime reasons while preserving first-seen order."""

    if reasons is None:
        return []

    normalized: list[RuntimeReasonCode] = []
    seen: set[RuntimeReasonCode] = set()
    for reason in reasons:
        code = normalize_runtime_reason(reason)
        if code in seen:
            continue
        seen.add(code)
        normalized.append(code)
    return normalized


@dataclass
class RuntimeCapabilities:
    """Capability flags advertised by a sandbox runtime provider."""

    supports_strict_deny_all: bool = False
    supports_strict_allowlist: bool = False
    supports_interactive: bool = False
    supports_port_mappings: bool = False
    supports_acp_session_mode: bool = False


@dataclass
class RuntimePreflightResult:
    """Host/runtime preflight status used by policy admission."""

    runtime: RuntimeType
    available: bool
    reasons: list[str] = field(default_factory=list)
    execution_mode: str = "none"
    supported_trust_levels: list[str] = field(
        default_factory=lambda: ["trusted", "standard", "untrusted"]
    )
    host: dict[str, Any] = field(default_factory=dict)
    enforcement_ready: dict[str, bool] = field(
        default_factory=lambda: {"deny_all": False, "allowlist": False}
    )


def collect_runtime_preflights(
    *,
    network_policy: str | None = None,
) -> dict[RuntimeType, RuntimePreflightResult]:
    """Collect a shared runtime preflight snapshot for policy admission."""

    from .runners.docker_runner import docker_available
    from .runners.firecracker_runner import firecracker_available
    from .runners.lima_runner import LimaRunner
    from .runners.seatbelt_runner import SeatbeltRunner
    from .runners.vz_linux_runner import VZLinuxRunner
    from .runners.vz_macos_runner import VZMacOSRunner
    from .runners.worktree_runner import WorktreeRunner

    requested_policy = str(network_policy or "deny_all").strip().lower() or "deny_all"

    docker_ok = bool(docker_available())
    firecracker_ok = bool(firecracker_available())

    return {
        RuntimeType.docker: RuntimePreflightResult(
            runtime=RuntimeType.docker,
            available=docker_ok,
            reasons=[] if docker_ok else ["docker_unavailable"],
        ),
        RuntimeType.firecracker: RuntimePreflightResult(
            runtime=RuntimeType.firecracker,
            available=firecracker_ok,
            reasons=[] if firecracker_ok else ["firecracker_unavailable"],
        ),
        RuntimeType.lima: LimaRunner().preflight(network_policy=requested_policy),
        RuntimeType.seatbelt: SeatbeltRunner().preflight(network_policy=requested_policy),
        RuntimeType.vz_linux: VZLinuxRunner().preflight(network_policy=requested_policy),
        RuntimeType.vz_macos: VZMacOSRunner().preflight(network_policy=requested_policy),
        RuntimeType.worktree: WorktreeRunner().preflight(network_policy=requested_policy),
    }
