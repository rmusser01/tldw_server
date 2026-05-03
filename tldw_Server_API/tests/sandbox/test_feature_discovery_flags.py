from __future__ import annotations

import os
from fastapi.testclient import TestClient
from tldw_Server_API.app.core.config import clear_config_cache
from tldw_Server_API.app.main import app


def _client(monkeypatch) -> TestClient:


    monkeypatch.setenv("TEST_MODE", "1")
    # Advertise a specific store backend
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    clear_config_cache()
    return TestClient(app)


def test_runtimes_include_capability_flags_and_store_mode(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get("runtimes"), list) and data["runtimes"]
        for rt in data["runtimes"]:
            assert "interactive_supported" in rt
            assert "egress_allowlist_supported" in rt
            assert "store_mode" in rt
            assert rt["store_mode"] in {"memory", "sqlite", "cluster", "unknown"}


def test_egress_allowlist_supported_when_enforced(monkeypatch) -> None:


     # Ensure app is in test mode and config cache is fresh
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    clear_config_cache()
    # Flip enforcement on the live service instance to avoid re-importing app
    import tldw_Server_API.app.api.v1.endpoints.sandbox as sandbox_ep
    # Temporarily enforce egress via monkeypatch so state is restored after test
    monkeypatch.setattr(
        sandbox_ep._service.policy.cfg,  # type: ignore[attr-defined]
        "egress_enforcement",
        True,
        raising=False,
    )
    with TestClient(app) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        assert r.status_code == 200
        data = r.json()
        # Find docker entry and assert flag is true
        docker = next((rt for rt in data.get("runtimes", []) if rt.get("name") == "docker"), None)
        assert docker is not None
        assert docker.get("egress_allowlist_supported") is True


def test_runtimes_notes_reflect_granular_allowlist(monkeypatch) -> None:


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("SANDBOX_EGRESS_ENFORCEMENT", "true")
    monkeypatch.setenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "true")
    clear_config_cache()
    with TestClient(app) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        assert r.status_code == 200
        data = r.json()
        docker = next((rt for rt in data.get("runtimes", []) if rt.get("name") == "docker"), None)
        assert docker is not None
        assert isinstance(docker.get("notes"), str) and "Granular egress allowlist" in docker["notes"]


def test_firecracker_egress_supported_only_when_enforced(monkeypatch) -> None:


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    # Ensure enforcement not set: expect False
    monkeypatch.delenv("SANDBOX_FIRECRACKER_EGRESS_ENFORCEMENT", raising=False)
    clear_config_cache()
    with TestClient(app) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        data = r.json()
        fc = next((rt for rt in data.get("runtimes", []) if rt.get("name") == "firecracker"), None)
        assert fc is not None
        assert fc.get("egress_allowlist_supported") is False
    # Now flip enforcement on
    monkeypatch.setenv("SANDBOX_FIRECRACKER_EGRESS_ENFORCEMENT", "true")
    clear_config_cache()
    with TestClient(app) as client:
        r2 = client.get("/api/v1/sandbox/runtimes")
        d2 = r2.json()
        fc2 = next((rt for rt in d2.get("runtimes", []) if rt.get("name") == "firecracker"), None)
        assert fc2 is not None
        assert fc2.get("egress_allowlist_supported") is True


def test_firecracker_not_available_when_real_disabled(monkeypatch) -> None:


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("ROUTES_ENABLE", "sandbox")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("SANDBOX_FIRECRACKER_ENABLE_REAL", "0")
    monkeypatch.delenv("TLDW_SANDBOX_FIRECRACKER_AVAILABLE", raising=False)
    clear_config_cache()
    with TestClient(app) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        assert r.status_code == 200
        data = r.json()
        fc = next((rt for rt in data.get("runtimes", []) if rt.get("name") == "firecracker"), None)
        assert fc is not None
        assert fc.get("available") is False


def test_runtimes_discovery_includes_macos_runtime_capabilities(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_MACOS_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", raising=False)
    clear_config_cache()

    with TestClient(app) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        assert r.status_code == 200
        data = r.json()
        runtimes = {item["name"]: item for item in data["runtimes"]}
        assert "vz_linux" in runtimes
        assert "vz_macos" in runtimes
        assert "seatbelt" in runtimes
        assert "supported_trust_levels" in runtimes["vz_linux"]
        assert "standard" in runtimes["vz_linux"]["supported_trust_levels"]
        assert "supported_trust_levels" in runtimes["seatbelt"]
        assert runtimes["seatbelt"]["supported_trust_levels"] == ["trusted"]
        assert isinstance(runtimes["vz_macos"].get("host"), dict)


def test_runtimes_discovery_includes_implementation_state_labels(monkeypatch) -> None:
    """Verify discovery exposes roadmap maturity labels for every sandbox runtime."""
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    clear_config_cache()

    with TestClient(app) as client:
        response = client.get("/api/v1/sandbox/runtimes")

    assert response.status_code == 200
    runtimes = {item["name"]: item for item in response.json()["runtimes"]}
    assert runtimes["docker"]["implementation_state"] == "supported"
    assert runtimes["firecracker"]["implementation_state"] == "host_gated"
    assert runtimes["lima"]["implementation_state"] == "host_gated"
    assert runtimes["vz_linux"]["implementation_state"] == "host_gated"
    assert runtimes["vz_macos"]["implementation_state"] == "scaffold"
    assert runtimes["seatbelt"]["implementation_state"] == "host_gated"
    assert runtimes["worktree"]["implementation_state"] == "supported"


def test_runtimes_discovery_keeps_macos_diagnostics_summarized(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_PATH", "/tmp/macos-helper")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE", "/tmp/vz-linux.img")
    clear_config_cache()

    with TestClient(app) as client:
        data = client.get("/api/v1/sandbox/runtimes").json()
        vz_linux = next(item for item in data["runtimes"] if item["name"] == "vz_linux")

    assert "helper" not in vz_linux
    assert "templates" not in vz_linux
    assert "remediation" not in vz_linux
    assert "reconciliation" not in vz_linux
    assert "supported_trust_levels" in vz_linux
    assert isinstance(vz_linux.get("host"), dict)


def test_macos_diagnostics_runtime_reasons_align_with_feature_discovery(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "0")
    clear_config_cache()

    svc = SandboxService()
    diagnostics = svc.macos_diagnostics()
    discovery = {item["name"]: item for item in svc.feature_discovery()}

    assert diagnostics["runtimes"]["vz_linux"]["reasons"] == discovery["vz_linux"]["reasons"]
    assert diagnostics["runtimes"]["vz_macos"]["reasons"] == discovery["vz_macos"]["reasons"]
    assert diagnostics["runtimes"]["seatbelt"]["supported_trust_levels"] == discovery["seatbelt"]["supported_trust_levels"]
    assert "reconciliation" in diagnostics


def test_runtimes_discovery_keeps_seatbelt_network_claims_best_effort(monkeypatch) -> None:
    import tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner as seatbelt_module

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", raising=False)
    monkeypatch.setattr(seatbelt_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        seatbelt_module,
        "vz_host_facts",
        lambda: {"os": "darwin", "arch": "arm64", "apple_silicon": True},
    )
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)
    clear_config_cache()

    with TestClient(app) as client:
        data = client.get("/api/v1/sandbox/runtimes").json()
        seatbelt = next(item for item in data["runtimes"] if item["name"] == "seatbelt")

    assert seatbelt["available"] is True
    assert seatbelt["strict_deny_all_supported"] is False
    assert "best-effort" in seatbelt["notes"]
    assert "VM-grade" in seatbelt["notes"]
