from __future__ import annotations

import re
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
    "limactl_missing",
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


def _runtime_name_is_documented(text: str, runtime: RuntimeType) -> bool:
    pattern = rf"(?<![\w`])`?{re.escape(runtime.value)}`?(?![\w`])"
    return re.search(pattern, text) is not None


def _markdown_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group("body") if match else ""


def _portable_gate_scope_is_documented(text: str) -> bool:
    section = _markdown_section(text, "Portable Runtime Capability Gate")
    has_host_gated_real_execution = re.search(
        r"\bhost[- ]gated\b.*\breal\s+runtime\s+execution\b",
        section,
        flags=re.IGNORECASE | re.DOTALL,
    )
    has_portable_capability_contract = re.search(
        r"\bportable\b.*\bcapability\s+contract",
        section,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return bool(has_host_gated_real_execution and has_portable_capability_contract)


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
    discovery = {
        row.name.value if isinstance(row.name, RuntimeType) else str(row.name): row
        for row in response.runtimes
    }

    assert set(discovery) == {runtime.value for runtime in RuntimeType}
    assert set(runtime_caps.RUNTIME_IMPLEMENTATION_STATES) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_ISOLATION_METADATA) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_NETWORK_POLICY_METADATA) == set(RuntimeType)
    assert set(runtime_caps.RUNTIME_SESSION_CONTRACT_METADATA) == set(RuntimeType)

    for runtime in RuntimeType:
        row = discovery[runtime.value]
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
        row = discovery[runtime.value]
        assert row.boundary_class == "host_local"
        assert row.vm_grade_isolation is False
        assert row.untrusted_eligible is False
        assert row.session_contract.reuse_model == "workspace_only"
        assert row.session_contract.repair_state == "unsupported"

    assert discovery[RuntimeType.vz_linux.value].session_contract.requires_live_health_check is True
    assert discovery[RuntimeType.vz_linux.value].session_contract.repair_state == "host_gated"


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


def test_portable_runtime_capability_gate_is_documented_for_every_runtime(
    pytestconfig: pytest.Config,
) -> None:
    repo_root = Path(pytestconfig.rootpath)
    inventory = repo_root / "Docs" / "Sandbox" / "sandbox-runtime-capability-inventory.md"
    text = inventory.read_text(encoding="utf-8")

    assert "Portable Runtime Capability Gate" in text
    for runtime in RuntimeType:
        assert _runtime_name_is_documented(text, runtime), (
            f"{runtime.value} missing from inventory"
        )


def test_portable_runtime_capability_gate_inventory_no_longer_lists_gate_as_missing(
    pytestconfig: pytest.Config,
) -> None:
    repo_root = Path(pytestconfig.rootpath)
    inventory = repo_root / "Docs" / "Sandbox" / "sandbox-runtime-capability-inventory.md"
    text = inventory.read_text(encoding="utf-8")

    assert "CI has no single cross-runtime capability gate" not in text
    assert _portable_gate_scope_is_documented(text)


def test_inventory_no_longer_lists_host_local_warning_ui_as_missing(
    pytestconfig: pytest.Config,
) -> None:
    """Guard against re-listing delivered host-local warning UI work as a gap."""
    repo_root = Path(pytestconfig.rootpath)
    inventory = repo_root / "Docs" / "Sandbox" / "sandbox-runtime-capability-inventory.md"
    text = inventory.read_text(encoding="utf-8")
    current_gaps = _markdown_section(text, "Current Gaps")
    diagnostics_section = _markdown_section(text, "Recovery And Diagnostics Support")

    assert "future UI/operator dashboards" not in current_gaps
    assert "Sandbox Runtime Isolation" in diagnostics_section
    assert re.search(
        r"\bhost-local isolation warnings\b",
        diagnostics_section,
        flags=re.IGNORECASE,
    )
