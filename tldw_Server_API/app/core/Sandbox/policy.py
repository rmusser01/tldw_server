from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

from .exceptions import SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS
from .models import RunSpec, RuntimeType, SessionSpec, TrustLevel
from .runtime_capabilities import (
    RuntimePreflightResult,
    runtime_network_policy_effective_support,
    runtime_network_policy_metadata,
)

# Trust-level presets define resource limits and security constraints
# based on the trust level of the code being executed.
TRUST_PROFILES: dict[TrustLevel, dict] = {
    TrustLevel.trusted: {
        "max_cpu": 8,
        "max_mem_mb": 16384,
        "timeout_sec": 600,
        "network_policy": "allowlist",  # Can access configured egress
        "workspace_cap_mb": 512,
        "pids_limit": 512,
        "ulimit_nofile": 4096,
    },
    TrustLevel.standard: {
        "max_cpu": 4,
        "max_mem_mb": 8192,
        "timeout_sec": 300,
        "network_policy": "deny_all",
        "workspace_cap_mb": 256,
        "pids_limit": 256,
        "ulimit_nofile": 1024,
    },
    TrustLevel.untrusted: {
        "max_cpu": 1,
        "max_mem_mb": 1024,
        "timeout_sec": 60,
        "network_policy": "deny_all",
        "workspace_cap_mb": 64,
        "pids_limit": 64,
        "ulimit_nofile": 256,
    },
}

_HOST_LOCAL_RUNTIMES: frozenset[RuntimeType] = frozenset(
    {RuntimeType.seatbelt, RuntimeType.worktree}
)


@dataclass
class SandboxPolicyConfig:
    default_runtime: RuntimeType = RuntimeType.docker
    network_default: str = "deny_all"  # deny_all | allowlist (allowlist controlled elsewhere)
    # Opt-in egress allowlist enforcement (runtime dependent; Docker only for now)
    egress_enforcement: bool = False
    egress_allowlist: list[str] = field(default_factory=list)
    artifact_ttl_hours: int = 24
    max_upload_mb: int = 64
    max_log_bytes: int = 10 * 1024 * 1024
    max_artifact_file_bytes: int = 64 * 1024 * 1024
    max_artifact_total_bytes: int = 256 * 1024 * 1024
    pids_limit: int = 256
    max_cpu: float = 4.0
    max_mem_mb: int = 8192
    workspace_cap_mb: int = 256
    supported_spec_versions: list[str] = field(default_factory=lambda: ["1.0"])

    @classmethod
    def from_settings(cls) -> SandboxPolicyConfig:
        try:
            rt_raw = str(getattr(app_settings, "SANDBOX_DEFAULT_RUNTIME", "docker")).strip().lower()
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            rt_raw = "docker"
        try:
            runtime = RuntimeType(rt_raw)
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            runtime = RuntimeType.docker
        try:
            network_default = str(getattr(app_settings, "SANDBOX_NETWORK_DEFAULT", "deny_all")).strip().lower()
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            network_default = "deny_all"
        def _get_int(key: str, dv: int) -> int:
            try:
                return int(getattr(app_settings, key))  # type: ignore[arg-type]
            except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
                return dv
        def _get_positive_int(key: str, dv: int) -> int:
            value = _get_int(key, dv)
            return value if value > 0 else dv
        def _get_float(key: str, dv: float) -> float:
            try:
                return float(getattr(app_settings, key))  # type: ignore[arg-type]
            except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
                return dv
        def _get_list(key: str, dv: list[str]) -> list[str]:
            try:
                v = getattr(app_settings, key)
                if isinstance(v, list):
                    return [str(x) for x in v]
                s = str(v)
                return [t.strip() for t in s.split(',') if t.strip()]
            except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
                return dv
        def _get_bool(key: str, dv: bool) -> bool:
            try:
                v = getattr(app_settings, key)
                if isinstance(v, bool):
                    return v
                s = str(v).strip().lower()
                return is_truthy(s)
            except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
                return dv
        return cls(
            default_runtime=runtime,
            network_default=network_default,
            egress_enforcement=_get_bool("SANDBOX_EGRESS_ENFORCEMENT", False),
            egress_allowlist=_get_list("SANDBOX_EGRESS_ALLOWLIST", []),
            artifact_ttl_hours=_get_int("SANDBOX_ARTIFACT_TTL_HOURS", 24),
            max_upload_mb=_get_int("SANDBOX_MAX_UPLOAD_MB", 64),
            max_log_bytes=_get_int("SANDBOX_MAX_LOG_BYTES", 10 * 1024 * 1024),
            max_artifact_file_bytes=_get_positive_int("SANDBOX_MAX_ARTIFACT_FILE_BYTES", cls.max_artifact_file_bytes),
            max_artifact_total_bytes=_get_positive_int("SANDBOX_MAX_ARTIFACT_TOTAL_BYTES", cls.max_artifact_total_bytes),
            pids_limit=_get_int("SANDBOX_PIDS_LIMIT", 256),
            max_cpu=_get_float("SANDBOX_MAX_CPU", 4.0),
            max_mem_mb=_get_int("SANDBOX_MAX_MEM_MB", 8192),
            workspace_cap_mb=_get_int("SANDBOX_WORKSPACE_CAP_MB", 256),
            supported_spec_versions=_get_list("SANDBOX_SUPPORTED_SPEC_VERSIONS", ["1.0"]),
        )


class SandboxPolicy:
    """Evaluates and normalizes sandbox requests against admin policy.

    This v0 policy is intentionally simple and conservative: deny-all network by default,
    and prefer Docker for broad compatibility unless Firecracker is explicitly requested
    and available.
    """

    def __init__(self, cfg: SandboxPolicyConfig | None = None) -> None:
        self.cfg = cfg or SandboxPolicyConfig.from_settings()

    class RuntimeUnavailable(Exception):
        def __init__(self, runtime: RuntimeType, reasons: list[str] | None = None) -> None:
            super().__init__(f"Requested runtime '{runtime.value}' is unavailable")
            self.runtime = runtime
            self.reasons = list(reasons or [])

    class PolicyUnsupported(Exception):
        def __init__(self, runtime: RuntimeType, requirement: str, reasons: list[str] | None = None) -> None:
            super().__init__(f"Runtime '{runtime.value}' does not satisfy requirement '{requirement}'")
            self.runtime = runtime
            self.requirement = str(requirement)
            self.reasons = list(reasons or [])

    @staticmethod
    def _runtime_preflight(
        runtime: RuntimeType,
        *,
        firecracker_available: bool,
        lima_available: bool,
        runtime_preflights: Mapping[RuntimeType, RuntimePreflightResult] | None = None,
    ) -> RuntimePreflightResult | None:
        if runtime_preflights is not None:
            preflight = runtime_preflights.get(runtime)
            if preflight is not None:
                return preflight
        if runtime == RuntimeType.firecracker:
            return RuntimePreflightResult(runtime=runtime, available=firecracker_available)
        if runtime == RuntimeType.lima:
            return RuntimePreflightResult(runtime=runtime, available=lima_available)
        return None

    def _require_runtime_available(
        self,
        runtime: RuntimeType,
        *,
        firecracker_available: bool,
        lima_available: bool,
        runtime_preflights: Mapping[RuntimeType, RuntimePreflightResult] | None = None,
    ) -> None:
        preflight = self._runtime_preflight(
            runtime,
            firecracker_available=firecracker_available,
            lima_available=lima_available,
            runtime_preflights=runtime_preflights,
        )
        if preflight is None or preflight.available:
            return
        raise SandboxPolicy.RuntimeUnavailable(runtime, reasons=list(preflight.reasons or []))

    @staticmethod
    def _require_trust_level_supported(
        runtime: RuntimeType,
        trust: TrustLevel,
        *,
        runtime_preflights: Mapping[RuntimeType, RuntimePreflightResult] | None = None,
    ) -> None:
        if runtime in _HOST_LOCAL_RUNTIMES and trust == TrustLevel.untrusted:
            raise SandboxPolicy.PolicyUnsupported(
                runtime,
                requirement="untrusted_requires_vm_runtime",
                reasons=["trust_level_requires_vm_runtime"],
            )

        if runtime_preflights is None:
            return

        preflight = runtime_preflights.get(runtime)
        if preflight is None:
            return

        supported = {
            str(level).strip().lower()
            for level in (preflight.supported_trust_levels or [])
            if str(level).strip()
        }
        if runtime == RuntimeType.seatbelt and trust == TrustLevel.standard and "standard" not in supported:
            raise SandboxPolicy.PolicyUnsupported(
                runtime,
                requirement="standard_not_enabled",
                reasons=["seatbelt_standard_disabled"],
            )
        if supported and trust.value not in supported:
            raise SandboxPolicy.PolicyUnsupported(
                runtime,
                requirement=f"trust_level:{trust.value}",
                reasons=["trust_level_not_supported"],
            )

    @staticmethod
    def _network_policy_unsupported_reason(network_policy: str) -> str:
        if network_policy == "allowlist":
            return "strict_allowlist_not_supported"
        return "strict_deny_all_not_supported"

    @staticmethod
    def _network_policy_static_support(
        runtime: RuntimeType,
        network_policy: str,
    ) -> bool:
        contract = runtime_network_policy_metadata(runtime)
        mode = (
            contract.deny_all
            if network_policy == "deny_all"
            else contract.allowlist
        )
        return mode.support_state == "supported" and mode.strict_enforcement

    @staticmethod
    def _require_network_policy_supported(
        runtime: RuntimeType,
        network_policy: str | None,
        runtime_preflight: RuntimePreflightResult | None = None,
    ) -> str:
        requested_policy = (
            str(network_policy or "deny_all").strip().lower() or "deny_all"
        )
        if requested_policy not in {"deny_all", "allowlist"}:
            raise SandboxPolicy.PolicyUnsupported(
                runtime,
                requirement=requested_policy,
                reasons=["unsupported_network_policy"],
            )

        if runtime_preflight is None:
            supported = SandboxPolicy._network_policy_static_support(
                runtime,
                requested_policy,
            )
        else:
            effective_support = runtime_network_policy_effective_support(
                runtime,
                runtime_preflight.enforcement_ready,
            )
            supported = effective_support.get(requested_policy, False)

        if not supported:
            raise SandboxPolicy.PolicyUnsupported(
                runtime,
                requirement=requested_policy,
                reasons=[
                    SandboxPolicy._network_policy_unsupported_reason(
                        requested_policy
                    )
                ],
            )
        return requested_policy

    def select_runtime(
        self,
        requested: RuntimeType | None,
        firecracker_available: bool,
        lima_available: bool = False,
        runtime_preflights: Mapping[RuntimeType, RuntimePreflightResult] | None = None,
    ) -> RuntimeType:
        if requested is not None:
            self._require_runtime_available(
                requested,
                firecracker_available=firecracker_available,
                lima_available=lima_available,
                runtime_preflights=runtime_preflights,
            )
            return requested
        # No explicit request: honor default, but still enforce availability
        self._require_runtime_available(
            self.cfg.default_runtime,
            firecracker_available=firecracker_available,
            lima_available=lima_available,
            runtime_preflights=runtime_preflights,
        )
        return self.cfg.default_runtime

    def apply_to_session(
        self,
        spec: SessionSpec,
        firecracker_available: bool,
        lima_available: bool = False,
        runtime_preflights: Mapping[RuntimeType, RuntimePreflightResult] | None = None,
    ) -> SessionSpec:
        spec.runtime = self.select_runtime(
            spec.runtime,
            firecracker_available,
            lima_available,
            runtime_preflights=runtime_preflights,
        )

        # Apply trust-level profile constraints
        trust = spec.trust_level or TrustLevel.standard
        self._require_trust_level_supported(
            spec.runtime,
            trust,
            runtime_preflights=runtime_preflights,
        )
        profile = TRUST_PROFILES.get(trust, TRUST_PROFILES[TrustLevel.standard])

        if not str(spec.network_policy or "").strip():
            spec.network_policy = profile.get(
                "network_policy",
                self.cfg.network_default,
            )
        spec.network_policy = self._require_network_policy_supported(
            spec.runtime,
            spec.network_policy,
            runtime_preflight=(
                runtime_preflights.get(spec.runtime)
                if runtime_preflights is not None
                else None
            ),
        )

        # Apply trust-level resource limits (more restrictive of trust profile and global policy)
        profile_max_cpu = float(profile.get("max_cpu", self.cfg.max_cpu))
        profile_max_mem = int(profile.get("max_mem_mb", self.cfg.max_mem_mb))
        profile_max_timeout = int(profile.get("timeout_sec", 300))

        # Effective max is the minimum of global policy and trust profile
        effective_max_cpu = min(self.cfg.max_cpu, profile_max_cpu)
        effective_max_mem = min(self.cfg.max_mem_mb, profile_max_mem)

        # Clamp resources to effective maxima
        try:
            if spec.cpu_limit is None:
                spec.cpu_limit = effective_max_cpu
            elif spec.cpu_limit > effective_max_cpu:
                spec.cpu_limit = float(effective_max_cpu)
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if spec.memory_mb is None:
                spec.memory_mb = effective_max_mem
            elif spec.memory_mb > effective_max_mem:
                spec.memory_mb = int(effective_max_mem)
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if spec.timeout_sec > profile_max_timeout:
                spec.timeout_sec = profile_max_timeout
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass

        return spec

    def apply_to_run(
        self,
        spec: RunSpec,
        firecracker_available: bool,
        lima_available: bool = False,
        runtime_preflights: Mapping[RuntimeType, RuntimePreflightResult] | None = None,
    ) -> RunSpec:
        # Always go through selection to enforce availability on defaults
        spec.runtime = self.select_runtime(
            spec.runtime,
            firecracker_available,
            lima_available,
            runtime_preflights=runtime_preflights,
        )

        # Apply trust-level profile constraints
        trust = spec.trust_level or TrustLevel.standard
        self._require_trust_level_supported(
            spec.runtime,
            trust,
            runtime_preflights=runtime_preflights,
        )
        profile = TRUST_PROFILES.get(trust, TRUST_PROFILES[TrustLevel.standard])

        if not str(spec.network_policy or "").strip():
            spec.network_policy = profile.get(
                "network_policy",
                self.cfg.network_default,
            )
        spec.network_policy = self._require_network_policy_supported(
            spec.runtime,
            spec.network_policy,
            runtime_preflight=(
                runtime_preflights.get(spec.runtime)
                if runtime_preflights is not None
                else None
            ),
        )

        # Apply trust-level resource limits (more restrictive of trust profile and global policy)
        profile_max_cpu = float(profile.get("max_cpu", self.cfg.max_cpu))
        profile_max_mem = int(profile.get("max_mem_mb", self.cfg.max_mem_mb))
        profile_max_timeout = int(profile.get("timeout_sec", 300))

        # Effective max is the minimum of global policy and trust profile
        effective_max_cpu = min(self.cfg.max_cpu, profile_max_cpu)
        effective_max_mem = min(self.cfg.max_mem_mb, profile_max_mem)

        # Clamp resources to effective maxima
        try:
            if spec.cpu is None:
                spec.cpu = effective_max_cpu
            elif spec.cpu > effective_max_cpu:
                spec.cpu = float(effective_max_cpu)
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if spec.memory_mb is None:
                spec.memory_mb = effective_max_mem
            elif spec.memory_mb > effective_max_mem:
                spec.memory_mb = int(effective_max_mem)
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if spec.timeout_sec > profile_max_timeout:
                spec.timeout_sec = profile_max_timeout
        except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
            pass

        return spec


def _canonical_policy_dict(cfg: SandboxPolicyConfig) -> dict:
    """Build a canonical, stable dict capturing policy-affecting settings.

    The material intentionally avoids environment-specific paths and
    includes only values that impact sandbox behavior and security.
    """
    try:
        # Runner security toggles (booleans for determinism)
        docker_seccomp_enabled = bool(getattr(app_settings, "SANDBOX_DOCKER_SECCOMP", None))
    except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
        docker_seccomp_enabled = False
    try:
        docker_apparmor_enabled = bool(getattr(app_settings, "SANDBOX_DOCKER_APPARMOR_PROFILE", None))
    except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
        docker_apparmor_enabled = False
    try:
        ul_nofile = int(getattr(app_settings, "SANDBOX_ULIMIT_NOFILE", 1024))
    except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
        ul_nofile = 1024
    try:
        ul_nproc = int(getattr(app_settings, "SANDBOX_ULIMIT_NPROC", 512))
    except SANDBOX_CONFIG_NONCRITICAL_EXCEPTIONS:
        ul_nproc = 512

    # Normalize supported spec versions list
    spec_versions = list(cfg.supported_spec_versions or ["1.0"])
    spec_versions = sorted(str(v) for v in spec_versions)

    material = {
        "default_runtime": cfg.default_runtime.value,
        "network_default": str(cfg.network_default),
        "egress": {
            "enforced": bool(cfg.egress_enforcement),
            "allowlist_count": int(len(cfg.egress_allowlist or [])),
        },
        "artifact_ttl_hours": int(cfg.artifact_ttl_hours),
        "max_upload_mb": int(cfg.max_upload_mb),
        "max_log_bytes": int(cfg.max_log_bytes),
        "max_artifact_file_bytes": int(cfg.max_artifact_file_bytes),
        "max_artifact_total_bytes": int(cfg.max_artifact_total_bytes),
        "pids_limit": int(cfg.pids_limit),
        "max_cpu": float(cfg.max_cpu),
        "max_mem_mb": int(cfg.max_mem_mb),
        "workspace_cap_mb": int(cfg.workspace_cap_mb),
        "supported_spec_versions": spec_versions,
        # Runner-level security primitives
        "security": {
            "docker_seccomp": bool(docker_seccomp_enabled),
            "docker_apparmor": bool(docker_apparmor_enabled),
            "ulimit_nofile": int(ul_nofile),
            "ulimit_nproc": int(ul_nproc),
        },
    }
    return material


def compute_policy_hash(cfg: SandboxPolicyConfig) -> str:
    """Return a short, reproducible hash for the canonical policy material.

    Uses sha256 of the canonical JSON (sorted keys, compact separators) and
    returns the first 16 hex chars for brevity, as used elsewhere.
    """
    mat = _canonical_policy_dict(cfg)
    canon = json.dumps(mat, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
