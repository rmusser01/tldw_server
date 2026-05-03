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

RUNTIME_IMPLEMENTATION_STATES: Mapping[RuntimeType, RuntimeImplementationState] = {
    RuntimeType.docker: "supported",
    RuntimeType.firecracker: "host_gated",
    RuntimeType.lima: "host_gated",
    RuntimeType.vz_linux: "host_gated",
    RuntimeType.vz_macos: "scaffold",
    RuntimeType.seatbelt: "host_gated",
    RuntimeType.worktree: "supported",
}


def runtime_implementation_state(runtime: RuntimeType) -> RuntimeImplementationState:
    """Return the roadmap maturity label for a runtime, independent of host availability."""
    return RUNTIME_IMPLEMENTATION_STATES.get(runtime, "unsupported")


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
