from __future__ import annotations

from tldw_Server_API.app.core.config import clear_config_cache
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicyConfig, compute_policy_hash


def _pin_policy_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    # Pin some policy-related env for stability within this process
    monkeypatch.setenv("SANDBOX_DEFAULT_RUNTIME", "docker")
    monkeypatch.setenv("SANDBOX_NETWORK_DEFAULT", "deny_all")
    monkeypatch.setenv("SANDBOX_ARTIFACT_TTL_HOURS", "24")
    monkeypatch.setenv("SANDBOX_MAX_UPLOAD_MB", "64")
    monkeypatch.setenv("SANDBOX_MAX_LOG_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("SANDBOX_MAX_ARTIFACT_FILE_BYTES", str(64 * 1024 * 1024))
    monkeypatch.setenv("SANDBOX_MAX_ARTIFACT_TOTAL_BYTES", str(256 * 1024 * 1024))
    monkeypatch.setenv("SANDBOX_PIDS_LIMIT", "256")
    monkeypatch.setenv("SANDBOX_MAX_CPU", "4.0")
    monkeypatch.setenv("SANDBOX_MAX_MEM_MB", "8192")
    monkeypatch.setenv("SANDBOX_WORKSPACE_CAP_MB", "256")
    monkeypatch.setenv("SANDBOX_SUPPORTED_SPEC_VERSIONS", "1.0")
    # runner security knobs
    monkeypatch.delenv("SANDBOX_DOCKER_SECCOMP", raising=False)  # ensure absent
    monkeypatch.delenv("SANDBOX_DOCKER_APPARMOR_PROFILE", raising=False)
    monkeypatch.setenv("SANDBOX_ULIMIT_NOFILE", "1024")
    monkeypatch.setenv("SANDBOX_ULIMIT_NPROC", "512")


def test_policy_hash_is_deterministic_within_process(monkeypatch) -> None:
    _pin_policy_env(monkeypatch)
    clear_config_cache()

    cfg = SandboxPolicyConfig.from_settings()
    ph1 = compute_policy_hash(cfg)
    ph2 = compute_policy_hash(cfg)

    assert isinstance(ph1, str) and isinstance(ph2, str)
    assert ph1 == ph2


def test_policy_reads_artifact_capture_byte_caps(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MAX_ARTIFACT_FILE_BYTES", str(64 * 1024 * 1024))
    monkeypatch.setenv("SANDBOX_MAX_ARTIFACT_TOTAL_BYTES", str(256 * 1024 * 1024))
    clear_config_cache()

    cfg = SandboxPolicyConfig.from_settings()

    assert cfg.max_artifact_file_bytes == 64 * 1024 * 1024
    assert cfg.max_artifact_total_bytes == 256 * 1024 * 1024
