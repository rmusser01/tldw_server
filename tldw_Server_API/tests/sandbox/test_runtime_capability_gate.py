from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    SandboxRuntimesResponse,
)
from tldw_Server_API.app.core.Sandbox import runtime_capabilities as runtime_caps
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RuntimeType
from tldw_Server_API.app.core.Sandbox.run_status_taxonomy import (
    normalize_run_status_reason,
)
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (
    RuntimePreflightResult,
)
from tldw_Server_API.app.core.Sandbox.service import SandboxService

_EMITTED_UNAVAILABLE_REASONS: dict[RuntimeType, str] = {
    RuntimeType.docker: "docker_unavailable",
    RuntimeType.firecracker: "firecracker_unavailable",
    RuntimeType.lima: "limactl_missing",
    RuntimeType.vz_linux: "vz_linux_unavailable",
    RuntimeType.vz_macos: "vz_macos_unavailable",
    RuntimeType.seatbelt: "seatbelt_unavailable",
    RuntimeType.worktree: "worktree_unavailable",
}

_EMITTED_POLICY_FAILURE_MESSAGES = (
    "lima_policy_failed",
    "vz_linux_policy_failed",
    "vz_macos_policy_failed",
    "seatbelt_policy_failed",
    "worktree_policy_failed",
)

_EMITTED_RUNTIME_UNAVAILABLE_MESSAGES = (
    "docker_unavailable",
    "firecracker_unavailable",
    "vz_linux_unavailable",
    "vz_macos_unavailable",
    "seatbelt_unavailable",
    "worktree_unavailable",
)


def _synthetic_preflights() -> dict[RuntimeType, RuntimePreflightResult]:
    return {
        runtime: RuntimePreflightResult(
            runtime=runtime,
            available=False,
            reasons=[reason],
            execution_mode="none",
            supported_trust_levels=["trusted", "standard"],
            host={"portable_gate": True},
            enforcement_ready={"deny_all": False, "allowlist": False},
        )
        for runtime, reason in _EMITTED_UNAVAILABLE_REASONS.items()
    }


def test_portable_runtime_capability_gate_covers_all_runtime_discovery_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setattr(
        SandboxService,
        "_collect_runtime_preflights",
        lambda self, network_policy=None: _synthetic_preflights(),
    )

    rows = SandboxService().feature_discovery()
    response = SandboxRuntimesResponse(runtimes=rows)
    discovery = {row.name: row for row in response.runtimes}

    assert set(discovery) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_IMPLEMENTATION_STATES) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_ISOLATION_METADATA) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_NETWORK_POLICY_METADATA) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_SESSION_CONTRACT_METADATA) == set(RuntimeType)

    for runtime in RuntimeType:
        row = discovery[runtime]
        assert row.implementation_state in {
            "supported",
            "unsupported",
            "scaffold",
            "host_gated",
            "not_applicable",
        }
        assert row.boundary_class == runtime_caps.runtime_isolation_metadata(runtime).boundary_class
        assert row.network_policy_contract is not None
        assert row.session_contract is not None
        assert row.normalized_reasons
        assert "unknown" not in row.normalized_reasons

    for runtime in (RuntimeType.seatbelt, RuntimeType.worktree):
        row = discovery[runtime]
        assert row.boundary_class == "host_local"
        assert row.vm_grade_isolation is False
        assert row.untrusted_eligible is False
        assert row.session_contract.reuse_model == "workspace_only"
        assert row.session_contract.repair_state == "unsupported"

    assert discovery[RuntimeType.vz_linux].session_contract.requires_live_health_check is True
    assert discovery[RuntimeType.vz_linux].session_contract.repair_state == "host_gated"


def test_portable_runtime_capability_gate_covers_emitted_status_reason_aliases() -> None:
    for message in _EMITTED_POLICY_FAILURE_MESSAGES:
        assert normalize_run_status_reason(
            phase=RunPhase.failed,
            message=message,
            exit_code=None,
            resource_usage=None,
        ) == "policy_failed"

    for message in _EMITTED_RUNTIME_UNAVAILABLE_MESSAGES:
        assert normalize_run_status_reason(
            phase=RunPhase.failed,
            message=message,
            exit_code=None,
            resource_usage=None,
        ) == "runtime_unavailable"


def test_portable_runtime_capability_gate_is_documented_for_every_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    inventory = repo_root / "Docs" / "Sandbox" / "sandbox-runtime-capability-inventory.md"
    text = inventory.read_text(encoding="utf-8")

    assert "Portable Runtime Capability Gate" in text
    for runtime in RuntimeType:
        assert f"`{runtime.value}`" in text
