from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.lima_enforcer import LimaSecurityEnforcer
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner


@pytest.mark.parametrize("override", [False, True])
def test_lima_native_host_cannot_claim_unimplemented_enforcement(monkeypatch, override) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(
        LimaSecurityEnforcer, "host_facts",
        lambda self: {"os": "linux", "arch": "x86_64", "variant": "native"},
    )
    for mode in ("DENY_ALL", "ALLOWLIST"):
        name = f"TLDW_SANDBOX_LIMA_ENFORCER_{mode}_READY"
        if override:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)

    result = LimaRunner().preflight(network_policy="deny_all")

    assert result.available is False
    assert result.enforcement_ready == {"deny_all": False, "allowlist": False}
    assert "strict_deny_all_not_supported" in result.reasons


def test_lima_enforcer_apply_and_verify_cannot_claim_unimplemented_enforcement(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_DENY_ALL_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", "1")
    enforcer = LimaSecurityEnforcer()

    assert enforcer.apply_deny_all({}) is False
    assert enforcer.apply_allowlist({}, ["example.com:443"]) is False
    assert enforcer.verify({}, "deny_all") is False
    assert enforcer.verify({}, "allowlist") is False


def test_lima_preflight_returns_unavailable_when_limactl_missing(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "0")
    result = LimaRunner().preflight(network_policy="deny_all")

    assert result.runtime == RuntimeType.lima
    assert result.available is False
    assert "limactl_missing" in result.reasons


def test_lima_preflight_wsl_fails_closed_with_permission_reason(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.delenv("TLDW_SANDBOX_LIMA_ENFORCER_DENY_ALL_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_LIMA_ENFORCER_PERMISSION_DENIED", raising=False)

    result = LimaRunner().preflight(network_policy="deny_all")

    assert result.runtime == RuntimeType.lima
    assert result.available is False
    assert result.host.get("variant") == "wsl"
    assert "permission_denied_host_enforcement" in result.reasons


def test_lima_preflight_permission_override_returns_permission_reason(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_PERMISSION_DENIED", "1")

    result = LimaRunner().preflight(network_policy="allowlist")

    assert result.runtime == RuntimeType.lima
    assert result.available is False
    assert result.enforcement_ready == {"deny_all": False, "allowlist": False}
    assert "permission_denied_host_enforcement" in result.reasons


def test_lima_preflight_wsl_overrides_cannot_bypass_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_DENY_ALL_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", "1")

    result = LimaRunner().preflight(network_policy="deny_all")

    assert result.runtime == RuntimeType.lima
    assert result.available is False
    assert result.enforcement_ready == {"deny_all": False, "allowlist": False}
    assert "permission_denied_host_enforcement" in result.reasons
