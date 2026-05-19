"""Validation tests for MCP local-safe default startup behavior."""

import pytest

from tldw_Server_API.app.core.MCP_unified.config import _is_local_only_safe_profile
from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.config import validate_config
from tldw_Server_API.app.core.MCP_unified.tests.support import SAFE_DEFAULT_ENV_VARS


@pytest.fixture(autouse=True)
def _clear_mcp_config_cache(monkeypatch):
    for name in SAFE_DEFAULT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None
    yield
    for name in SAFE_DEFAULT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None


def _set_non_test_runtime(monkeypatch):
    monkeypatch.setenv("MCP_DEBUG", "false")
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("MCP_WS_AUTH_REQUIRED", "true")
    # validate_config treats this env as test context when non-empty.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "")


def test_local_only_safe_profile_accepts_loopback_defaults(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_IPS", "127.0.0.1,::1")
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    monkeypatch.setenv("MCP_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    monkeypatch.setenv("MCP_TRUST_X_FORWARDED", "false")
    cfg = get_config()

    assert _is_local_only_safe_profile(cfg) is True  # nosec B101


def test_local_only_safe_profile_rejects_non_loopback_allowlist(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_IPS", "0.0.0.0/0")
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    monkeypatch.setenv("MCP_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    monkeypatch.setenv("MCP_TRUST_X_FORWARDED", "false")
    cfg = get_config()

    assert _is_local_only_safe_profile(cfg) is False  # nosec B101


def test_validate_config_allows_generated_secrets_for_local_safe_defaults(monkeypatch):
    _set_non_test_runtime(monkeypatch)

    monkeypatch.delenv("MCP_JWT_SECRET", raising=False)
    monkeypatch.delenv("MCP_API_KEY_SALT", raising=False)
    monkeypatch.setenv("MCP_ALLOWED_IPS", "127.0.0.1,::1")
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    monkeypatch.setenv("MCP_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    monkeypatch.setenv("MCP_TRUST_X_FORWARDED", "false")

    assert validate_config() is True  # nosec B101


def test_validate_config_rejects_generated_secrets_for_non_local_profile(monkeypatch):
    _set_non_test_runtime(monkeypatch)

    monkeypatch.delenv("MCP_JWT_SECRET", raising=False)
    monkeypatch.delenv("MCP_API_KEY_SALT", raising=False)
    # Empty allowlist means "allow all", which is not a local-only safe profile.
    monkeypatch.setenv("MCP_ALLOWED_IPS", "")
    monkeypatch.setenv("MCP_WS_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    monkeypatch.setenv("MCP_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    monkeypatch.setenv("MCP_TRUST_X_FORWARDED", "false")

    assert validate_config() is False  # nosec B101
