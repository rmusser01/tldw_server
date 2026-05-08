from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, get_args

from loguru import logger

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

from .exceptions import SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS
from .models import RuntimeType

RuntimeImplementationState = Literal[
    "supported",
    "unsupported",
    "scaffold",
    "host_gated",
    "not_applicable",
]
RuntimeBoundaryClass = Literal[
    "container",
    "host_local",
    "vm_grade",
    "vm_grade_scaffold",
]
RuntimeIsolationWarningCode = Literal[
    "host_local_boundary",
    "not_vm_grade_isolation",
    "not_untrusted_eligible",
]
RuntimeNetworkPolicySupportState = Literal[
    "supported",
    "unsupported",
    "scaffold",
    "host_gated",
    "not_applicable",
]
RuntimeNetworkPolicyReadinessSource = Literal[
    "runtime_preflight",
    "config",
    "not_applicable",
]
RuntimeSessionReuseModel = Literal[
    "none",
    "workspace_only",
    "warm_vm",
    "scaffold",
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
RuntimeReasonCategory = Literal[
    "runtime",
    "platform",
    "helper",
    "template",
    "network",
    "policy",
    "host",
    "implementation",
    "image_store",
    "unknown",
]
RuntimeReasonSeverity = Literal["info", "warning", "error"]
RuntimeReasonOperatorAction = Literal[
    "none",
    "check_helper",
    "configure_template",
    "prepare_host",
    "adjust_request_policy",
    "use_different_runtime",
    "inspect_reasons",
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


@dataclass(frozen=True)
class RuntimeIsolationMetadata:
    """Machine-readable isolation posture for runtime discovery."""

    boundary_class: RuntimeBoundaryClass
    vm_grade_isolation: bool
    untrusted_eligible: bool


@dataclass(frozen=True)
class RuntimeNetworkPolicyModeMetadata:
    """Static network policy posture for one runtime policy mode."""

    support_state: RuntimeNetworkPolicySupportState
    strict_enforcement: bool
    readiness_source: RuntimeNetworkPolicyReadinessSource

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "support_state": self.support_state,
            "strict_enforcement": self.strict_enforcement,
            "readiness_source": self.readiness_source,
        }


@dataclass(frozen=True)
class RuntimeNetworkPolicyMetadata:
    """Machine-readable network policy contract for runtime discovery."""

    deny_all: RuntimeNetworkPolicyModeMetadata
    allowlist: RuntimeNetworkPolicyModeMetadata

    def as_dict(self) -> dict[str, dict[str, str | bool]]:
        return {
            "deny_all": self.deny_all.as_dict(),
            "allowlist": self.allowlist.as_dict(),
        }


@dataclass(frozen=True)
class RuntimeSessionContractMetadata:
    """Machine-readable session semantics for runtime discovery."""

    support_state: RuntimeImplementationState
    reuse_model: RuntimeSessionReuseModel
    requires_live_health_check: bool
    recovery_state: RuntimeImplementationState
    repair_state: RuntimeImplementationState

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "support_state": self.support_state,
            "reuse_model": self.reuse_model,
            "requires_live_health_check": self.requires_live_health_check,
            "recovery_state": self.recovery_state,
            "repair_state": self.repair_state,
        }


@dataclass(frozen=True)
class RuntimeReasonDetails:
    """Structured client/operator metadata for a normalized runtime reason."""

    code: RuntimeReasonCode
    category: RuntimeReasonCategory
    severity: RuntimeReasonSeverity
    availability_blocking: bool
    operator_action: RuntimeReasonOperatorAction
    user_message_key: str

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "code": self.code,
            "category": self.category,
            "severity": self.severity,
            "availability_blocking": self.availability_blocking,
            "operator_action": self.operator_action,
            "user_message_key": self.user_message_key,
        }


RUNTIME_ISOLATION_METADATA: Mapping[RuntimeType, RuntimeIsolationMetadata] = {
    RuntimeType.docker: RuntimeIsolationMetadata(
        boundary_class="container",
        vm_grade_isolation=False,
        untrusted_eligible=True,
    ),
    RuntimeType.firecracker: RuntimeIsolationMetadata(
        boundary_class="vm_grade",
        vm_grade_isolation=True,
        untrusted_eligible=True,
    ),
    RuntimeType.lima: RuntimeIsolationMetadata(
        boundary_class="vm_grade",
        vm_grade_isolation=True,
        untrusted_eligible=True,
    ),
    RuntimeType.vz_linux: RuntimeIsolationMetadata(
        boundary_class="vm_grade",
        vm_grade_isolation=True,
        untrusted_eligible=True,
    ),
    RuntimeType.vz_macos: RuntimeIsolationMetadata(
        boundary_class="vm_grade_scaffold",
        vm_grade_isolation=False,
        untrusted_eligible=False,
    ),
    RuntimeType.seatbelt: RuntimeIsolationMetadata(
        boundary_class="host_local",
        vm_grade_isolation=False,
        untrusted_eligible=False,
    ),
    RuntimeType.worktree: RuntimeIsolationMetadata(
        boundary_class="host_local",
        vm_grade_isolation=False,
        untrusted_eligible=False,
    ),
}


RUNTIME_SESSION_CONTRACT_METADATA: Mapping[
    RuntimeType,
    RuntimeSessionContractMetadata,
] = {
    RuntimeType.docker: RuntimeSessionContractMetadata(
        support_state="supported",
        reuse_model="workspace_only",
        requires_live_health_check=False,
        recovery_state="unsupported",
        repair_state="unsupported",
    ),
    RuntimeType.firecracker: RuntimeSessionContractMetadata(
        support_state="scaffold",
        reuse_model="scaffold",
        requires_live_health_check=False,
        recovery_state="unsupported",
        repair_state="unsupported",
    ),
    RuntimeType.lima: RuntimeSessionContractMetadata(
        support_state="scaffold",
        reuse_model="scaffold",
        requires_live_health_check=False,
        recovery_state="unsupported",
        repair_state="unsupported",
    ),
    RuntimeType.vz_linux: RuntimeSessionContractMetadata(
        support_state="host_gated",
        reuse_model="warm_vm",
        requires_live_health_check=True,
        recovery_state="host_gated",
        repair_state="host_gated",
    ),
    RuntimeType.vz_macos: RuntimeSessionContractMetadata(
        support_state="scaffold",
        reuse_model="scaffold",
        requires_live_health_check=False,
        recovery_state="scaffold",
        repair_state="scaffold",
    ),
    RuntimeType.seatbelt: RuntimeSessionContractMetadata(
        support_state="scaffold",
        reuse_model="workspace_only",
        requires_live_health_check=False,
        recovery_state="unsupported",
        repair_state="unsupported",
    ),
    RuntimeType.worktree: RuntimeSessionContractMetadata(
        support_state="scaffold",
        reuse_model="workspace_only",
        requires_live_health_check=False,
        recovery_state="unsupported",
        repair_state="unsupported",
    ),
}


RUNTIME_NETWORK_POLICY_METADATA: Mapping[
    RuntimeType,
    RuntimeNetworkPolicyMetadata,
] = {
    RuntimeType.docker: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="supported",
            strict_enforcement=True,
            readiness_source="config",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="host_gated",
            strict_enforcement=True,
            readiness_source="config",
        ),
    ),
    RuntimeType.firecracker: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="host_gated",
            strict_enforcement=True,
            readiness_source="runtime_preflight",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="scaffold",
            strict_enforcement=False,
            readiness_source="runtime_preflight",
        ),
    ),
    RuntimeType.lima: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="host_gated",
            strict_enforcement=True,
            readiness_source="runtime_preflight",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
    ),
    RuntimeType.vz_linux: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="host_gated",
            strict_enforcement=True,
            readiness_source="runtime_preflight",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
    ),
    RuntimeType.vz_macos: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="scaffold",
            strict_enforcement=False,
            readiness_source="runtime_preflight",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
    ),
    RuntimeType.seatbelt: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
    ),
    RuntimeType.worktree: RuntimeNetworkPolicyMetadata(
        deny_all=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
        allowlist=RuntimeNetworkPolicyModeMetadata(
            support_state="unsupported",
            strict_enforcement=False,
            readiness_source="not_applicable",
        ),
    ),
}


RUNTIME_REASON_METADATA: Mapping[str, RuntimeReasonDetails] = {
    "runtime_unavailable": RuntimeReasonDetails(
        code="runtime_unavailable",
        category="runtime",
        severity="error",
        availability_blocking=True,
        operator_action="use_different_runtime",
        user_message_key="sandbox.runtime.reason.runtime_unavailable",
    ),
    "unsupported_os": RuntimeReasonDetails(
        code="unsupported_os",
        category="platform",
        severity="error",
        availability_blocking=True,
        operator_action="prepare_host",
        user_message_key="sandbox.runtime.reason.unsupported_os",
    ),
    "unsupported_arch": RuntimeReasonDetails(
        code="unsupported_arch",
        category="platform",
        severity="error",
        availability_blocking=True,
        operator_action="prepare_host",
        user_message_key="sandbox.runtime.reason.unsupported_arch",
    ),
    "helper_unavailable": RuntimeReasonDetails(
        code="helper_unavailable",
        category="helper",
        severity="error",
        availability_blocking=True,
        operator_action="check_helper",
        user_message_key="sandbox.runtime.reason.helper_unavailable",
    ),
    "helper_protocol_mismatch": RuntimeReasonDetails(
        code="helper_protocol_mismatch",
        category="helper",
        severity="error",
        availability_blocking=True,
        operator_action="check_helper",
        user_message_key="sandbox.runtime.reason.helper_protocol_mismatch",
    ),
    "helper_missing": RuntimeReasonDetails(
        code="helper_missing",
        category="helper",
        severity="error",
        availability_blocking=True,
        operator_action="check_helper",
        user_message_key="sandbox.runtime.reason.helper_missing",
    ),
    "template_missing": RuntimeReasonDetails(
        code="template_missing",
        category="template",
        severity="error",
        availability_blocking=True,
        operator_action="configure_template",
        user_message_key="sandbox.runtime.reason.template_missing",
    ),
    "template_unconfigured": RuntimeReasonDetails(
        code="template_unconfigured",
        category="template",
        severity="error",
        availability_blocking=True,
        operator_action="configure_template",
        user_message_key="sandbox.runtime.reason.template_unconfigured",
    ),
    "network_policy_unsupported": RuntimeReasonDetails(
        code="network_policy_unsupported",
        category="network",
        severity="error",
        availability_blocking=True,
        operator_action="adjust_request_policy",
        user_message_key="sandbox.runtime.reason.network_policy_unsupported",
    ),
    "trust_policy_denied": RuntimeReasonDetails(
        code="trust_policy_denied",
        category="policy",
        severity="error",
        availability_blocking=True,
        operator_action="adjust_request_policy",
        user_message_key="sandbox.runtime.reason.trust_policy_denied",
    ),
    "host_prerequisite_missing": RuntimeReasonDetails(
        code="host_prerequisite_missing",
        category="host",
        severity="error",
        availability_blocking=True,
        operator_action="prepare_host",
        user_message_key="sandbox.runtime.reason.host_prerequisite_missing",
    ),
    "host_permission_denied": RuntimeReasonDetails(
        code="host_permission_denied",
        category="host",
        severity="error",
        availability_blocking=True,
        operator_action="prepare_host",
        user_message_key="sandbox.runtime.reason.host_permission_denied",
    ),
    "feature_not_implemented": RuntimeReasonDetails(
        code="feature_not_implemented",
        category="implementation",
        severity="error",
        availability_blocking=True,
        operator_action="use_different_runtime",
        user_message_key="sandbox.runtime.reason.feature_not_implemented",
    ),
    "image_store_unavailable": RuntimeReasonDetails(
        code="image_store_unavailable",
        category="image_store",
        severity="error",
        availability_blocking=True,
        operator_action="prepare_host",
        user_message_key="sandbox.runtime.reason.image_store_unavailable",
    ),
    "unknown": RuntimeReasonDetails(
        code="unknown",
        category="unknown",
        severity="warning",
        availability_blocking=True,
        operator_action="inspect_reasons",
        user_message_key="sandbox.runtime.reason.unknown",
    ),
}


def _validate_runtime_isolation_metadata_map() -> None:
    expected = set(RuntimeType)
    actual = set(RUNTIME_ISOLATION_METADATA)
    missing = sorted(runtime.value for runtime in expected - actual)
    extra = sorted(str(runtime) for runtime in actual - expected)
    if missing or extra:
        raise RuntimeError(
            "Runtime isolation metadata map is incomplete: "
            f"missing={missing}, extra={extra}"
        )


def _validate_runtime_network_policy_metadata_map() -> None:
    expected = set(RuntimeType)
    actual = set(RUNTIME_NETWORK_POLICY_METADATA)
    missing = sorted(runtime.value for runtime in expected - actual)
    extra = sorted(str(runtime) for runtime in actual - expected)
    if missing or extra:
        raise RuntimeError(
            "Runtime network policy metadata map is incomplete: "
            f"missing={missing}, extra={extra}"
        )


def _validate_runtime_session_contract_metadata_map() -> None:
    expected = set(RuntimeType)
    actual = set(RUNTIME_SESSION_CONTRACT_METADATA)
    missing = sorted(runtime.value for runtime in expected - actual)
    extra = sorted(str(runtime) for runtime in actual - expected)
    if missing or extra:
        raise RuntimeError(
            "Runtime session contract metadata map is incomplete: "
            f"missing={missing}, extra={extra}"
        )


def _validate_runtime_reason_metadata_map() -> None:
    expected = set(get_args(RuntimeReasonCode))
    actual = set(RUNTIME_REASON_METADATA)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            "Runtime reason metadata map is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    mismatched = sorted(
        (key, metadata.code)
        for key, metadata in RUNTIME_REASON_METADATA.items()
        if metadata.code != key
    )
    if mismatched:
        raise RuntimeError(
            "Runtime reason metadata code mismatch: "
            f"mismatched={mismatched}"
        )


_validate_runtime_isolation_metadata_map()
_validate_runtime_session_contract_metadata_map()
_validate_runtime_network_policy_metadata_map()
_validate_runtime_reason_metadata_map()


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


def runtime_isolation_metadata(runtime: RuntimeType | str) -> RuntimeIsolationMetadata:
    """Return stable isolation posture metadata, independent of host availability."""
    try:
        runtime_key = runtime if isinstance(runtime, RuntimeType) else RuntimeType(runtime)
    except ValueError as exc:
        raise ValueError(f"No isolation metadata configured for runtime {runtime!r}") from exc

    metadata = RUNTIME_ISOLATION_METADATA.get(runtime_key)
    if metadata is None:
        raise ValueError(f"No isolation metadata configured for runtime {runtime!r}")
    return metadata


def runtime_isolation_warnings(
    runtime: RuntimeType | str,
) -> list[RuntimeIsolationWarningCode]:
    """Return advisory discovery warnings derived from static isolation posture."""
    metadata = runtime_isolation_metadata(runtime)
    if metadata.boundary_class != "host_local":
        return []

    warnings: list[RuntimeIsolationWarningCode] = ["host_local_boundary"]
    if not metadata.vm_grade_isolation:
        warnings.append("not_vm_grade_isolation")
    if not metadata.untrusted_eligible:
        warnings.append("not_untrusted_eligible")
    return warnings


def runtime_network_policy_metadata(
    runtime: RuntimeType | str,
) -> RuntimeNetworkPolicyMetadata:
    """Return stable network policy posture metadata, independent of host availability."""
    try:
        runtime_key = runtime if isinstance(runtime, RuntimeType) else RuntimeType(runtime)
    except ValueError as exc:
        raise ValueError(
            f"No network policy metadata configured for runtime {runtime!r}"
        ) from exc

    metadata = RUNTIME_NETWORK_POLICY_METADATA.get(runtime_key)
    if metadata is None:
        raise ValueError(f"No network policy metadata configured for runtime {runtime!r}")
    return metadata


def runtime_session_contract_metadata(
    runtime: RuntimeType | str,
) -> RuntimeSessionContractMetadata:
    """Return stable session semantics metadata, independent of host availability."""
    try:
        runtime_key = runtime if isinstance(runtime, RuntimeType) else RuntimeType(runtime)
    except ValueError as exc:
        raise ValueError(
            f"No session contract metadata configured for runtime {runtime!r}"
        ) from exc

    metadata = RUNTIME_SESSION_CONTRACT_METADATA.get(runtime_key)
    if metadata is None:
        raise ValueError(
            f"No session contract metadata configured for runtime {runtime!r}"
        )
    return metadata


def _settings_flag(name: str) -> bool:
    """Read a boolean readiness flag, failing closed with operator-visible logs."""

    try:
        raw = os.getenv(name)
        if raw is None:
            raw = getattr(app_settings, name, "")
        return is_truthy(str(raw).strip().lower())
    except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Failed to read sandbox readiness flag {}: {}",
            name,
            exc,
        )
        return False


def docker_network_policy_readiness(docker_available: bool) -> dict[str, bool]:
    """Return Docker network readiness facts used by discovery and admission."""

    egress_enforced = _settings_flag("SANDBOX_EGRESS_ENFORCEMENT")
    granular_enforced = _settings_flag("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT")
    return {
        "deny_all": bool(docker_available),
        "allowlist": bool(docker_available and egress_enforced and granular_enforced),
    }


def runtime_network_policy_effective_support(
    runtime: RuntimeType | str,
    enforcement_ready: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """Return currently usable strict network policy support for a runtime."""

    contract = runtime_network_policy_metadata(runtime)
    ready = dict(enforcement_ready or {})

    def _supported(mode: RuntimeNetworkPolicyModeMetadata, key: str) -> bool:
        if mode.support_state not in {"supported", "host_gated"}:
            return False
        if not mode.strict_enforcement:
            return False
        return bool(ready.get(key))

    return {
        "deny_all": _supported(contract.deny_all, "deny_all"),
        "allowlist": _supported(contract.allowlist, "allowlist"),
    }


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


def runtime_reason_details(code: RuntimeReasonCode | str | None) -> RuntimeReasonDetails:
    """Return structured metadata for a normalized runtime reason code."""

    code_value = str(code or "unknown").strip()
    metadata = RUNTIME_REASON_METADATA.get(code_value)
    if metadata is None:
        return RUNTIME_REASON_METADATA["unknown"]
    return metadata


def runtime_reason_details_for_codes(
    codes: list[RuntimeReasonCode | str] | tuple[RuntimeReasonCode | str, ...] | None,
) -> list[RuntimeReasonDetails]:
    """Return reason metadata while preserving normalized reason order."""

    if codes is None:
        return []
    return [runtime_reason_details(code) for code in codes]


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
            enforcement_ready=docker_network_policy_readiness(docker_ok),
        ),
        RuntimeType.firecracker: RuntimePreflightResult(
            runtime=RuntimeType.firecracker,
            available=firecracker_ok,
            reasons=[] if firecracker_ok else ["firecracker_unavailable"],
            enforcement_ready={"deny_all": firecracker_ok, "allowlist": False},
        ),
        RuntimeType.lima: LimaRunner().preflight(network_policy=requested_policy),
        RuntimeType.seatbelt: SeatbeltRunner().preflight(network_policy=requested_policy),
        RuntimeType.vz_linux: VZLinuxRunner().preflight(network_policy=requested_policy),
        RuntimeType.vz_macos: VZMacOSRunner().preflight(network_policy=requested_policy),
        RuntimeType.worktree: WorktreeRunner().preflight(network_policy=requested_policy),
    }
