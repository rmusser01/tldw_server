from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import (
    RunSpec,
    RuntimeType,
    SessionSpec,
    TrustLevel,
)
from tldw_Server_API.app.core.Sandbox.policy import (
    SandboxPolicy,
    SandboxPolicyConfig,
)
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (
    RuntimePreflightResult,
)


def _available_preflight(
    runtime: RuntimeType,
    *,
    trust_levels: list[str] | None = None,
    enforcement_ready: dict[str, bool] | None = None,
) -> RuntimePreflightResult:
    return RuntimePreflightResult(
        runtime=runtime,
        available=True,
        reasons=[],
        supported_trust_levels=trust_levels
        or ["trusted", "standard", "untrusted"],
        enforcement_ready=enforcement_ready
        or {"deny_all": True, "allowlist": True},
    )


def _run_spec(
    runtime: RuntimeType | None,
    *,
    network_policy: str | None = None,
    trust_level: TrustLevel | None = None,
) -> RunSpec:
    return RunSpec(
        session_id=None,
        runtime=runtime,
        base_image=None,
        command=["echo", "ok"],
        network_policy=network_policy,
        trust_level=trust_level,
    )


def test_apply_to_run_rejects_worktree_deny_all_from_standard_default() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.worktree)

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=False,
            runtime_preflights={
                RuntimeType.worktree: _available_preflight(
                    RuntimeType.worktree,
                    trust_levels=["trusted", "standard"],
                )
            },
        )

    assert exc.value.runtime == RuntimeType.worktree
    assert exc.value.requirement == "deny_all"
    assert exc.value.reasons == ["strict_deny_all_not_supported"]


def test_apply_to_session_rejects_seatbelt_deny_all_from_standard_default() -> None:
    policy = SandboxPolicy()
    spec = SessionSpec(
        runtime=RuntimeType.seatbelt,
        trust_level=TrustLevel.standard,
        network_policy="",
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_session(
            spec,
            firecracker_available=False,
            runtime_preflights={
                RuntimeType.seatbelt: _available_preflight(
                    RuntimeType.seatbelt,
                    trust_levels=["trusted", "standard"],
                )
            },
        )

    assert exc.value.runtime == RuntimeType.seatbelt
    assert exc.value.requirement == "deny_all"
    assert exc.value.reasons == ["strict_deny_all_not_supported"]


@pytest.mark.parametrize("runtime", [RuntimeType.seatbelt, RuntimeType.worktree])
def test_apply_to_run_rejects_host_local_allowlist(runtime: RuntimeType) -> None:
    policy = SandboxPolicy()
    spec = _run_spec(runtime, network_policy="allowlist")

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=False,
            runtime_preflights={
                runtime: _available_preflight(
                    runtime,
                    trust_levels=["trusted", "standard"],
                )
            },
        )

    assert exc.value.runtime == runtime
    assert exc.value.requirement == "allowlist"
    assert exc.value.reasons == ["strict_allowlist_not_supported"]


def test_apply_to_run_rejects_invalid_network_policy() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.docker, network_policy="allow_all")

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=False,
            runtime_preflights={
                RuntimeType.docker: _available_preflight(RuntimeType.docker)
            },
        )

    assert exc.value.runtime == RuntimeType.docker
    assert exc.value.requirement == "allow_all"
    assert exc.value.reasons == ["unsupported_network_policy"]


def test_apply_to_run_canonicalizes_admitted_network_policy() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.docker, network_policy=" DeNy_AlL ")

    result = policy.apply_to_run(
        spec,
        firecracker_available=False,
        runtime_preflights={
            RuntimeType.docker: _available_preflight(RuntimeType.docker)
        },
    )

    assert result.network_policy == "deny_all"


def test_apply_to_session_canonicalizes_admitted_network_policy() -> None:
    policy = SandboxPolicy()
    spec = SessionSpec(
        runtime=RuntimeType.docker,
        trust_level=TrustLevel.standard,
        network_policy=" ALLOWLIST ",
    )

    result = policy.apply_to_session(
        spec,
        firecracker_available=False,
        runtime_preflights={
            RuntimeType.docker: _available_preflight(RuntimeType.docker)
        },
    )

    assert result.network_policy == "allowlist"


def test_apply_to_run_allows_default_docker_deny_all_without_preflights() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(None)

    result = policy.apply_to_run(spec, firecracker_available=False)

    assert result.runtime == RuntimeType.docker
    assert result.network_policy == "deny_all"


def test_apply_to_run_rejects_docker_allowlist_when_not_effectively_ready() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.docker, network_policy="allowlist")

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=False,
            runtime_preflights={
                RuntimeType.docker: _available_preflight(
                    RuntimeType.docker,
                    enforcement_ready={"deny_all": True, "allowlist": False},
                )
            },
        )

    assert exc.value.runtime == RuntimeType.docker
    assert exc.value.requirement == "allowlist"
    assert exc.value.reasons == ["strict_allowlist_not_supported"]


def test_apply_to_run_treats_whitespace_only_policy_as_missing() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.docker, network_policy="   ")

    result = policy.apply_to_run(
        spec,
        firecracker_available=False,
        runtime_preflights={
            RuntimeType.docker: _available_preflight(RuntimeType.docker)
        },
    )

    assert result.network_policy == "deny_all"


def test_apply_to_run_allows_vz_linux_deny_all_static_contract() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.vz_linux, network_policy="deny_all")

    result = policy.apply_to_run(
        spec,
        firecracker_available=False,
        runtime_preflights={
            RuntimeType.vz_linux: _available_preflight(RuntimeType.vz_linux)
        },
    )

    assert result.runtime == RuntimeType.vz_linux
    assert result.network_policy == "deny_all"


def test_apply_to_run_rejects_vz_linux_allowlist_static_contract() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.vz_linux, network_policy="allowlist")

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=False,
            runtime_preflights={
                RuntimeType.vz_linux: _available_preflight(RuntimeType.vz_linux)
            },
        )

    assert exc.value.runtime == RuntimeType.vz_linux
    assert exc.value.requirement == "allowlist"
    assert exc.value.reasons == ["strict_allowlist_not_supported"]


def test_apply_to_run_rejects_scaffold_network_policy_modes() -> None:
    policy = SandboxPolicy()
    spec = _run_spec(RuntimeType.firecracker, network_policy="allowlist")

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=True,
            runtime_preflights={
                RuntimeType.firecracker: _available_preflight(
                    RuntimeType.firecracker
                )
            },
        )

    assert exc.value.runtime == RuntimeType.firecracker
    assert exc.value.requirement == "allowlist"
    assert exc.value.reasons == ["strict_allowlist_not_supported"]


def test_apply_to_run_validates_default_runtime_after_profile_default() -> None:
    policy = SandboxPolicy(
        SandboxPolicyConfig(default_runtime=RuntimeType.worktree)
    )
    spec = _run_spec(runtime=None)

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=False,
            runtime_preflights={
                RuntimeType.worktree: _available_preflight(
                    RuntimeType.worktree,
                    trust_levels=["trusted", "standard"],
                )
            },
        )

    assert exc.value.runtime == RuntimeType.worktree
    assert exc.value.requirement == "deny_all"
    assert exc.value.reasons == ["strict_deny_all_not_supported"]
