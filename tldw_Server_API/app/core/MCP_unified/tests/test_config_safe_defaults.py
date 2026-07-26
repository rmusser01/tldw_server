"""Validation tests for MCP local-safe default startup behavior."""

import pytest

from tldw_Server_API.app.core.MCP_unified import config as config_module
from tldw_Server_API.app.core.MCP_unified.config import _is_local_only_safe_profile, get_config, validate_config
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


def test_invalid_tool_category_map_records_sanitized_warning(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_CATEGORY_MAP", '{"media.search":')

    cfg = get_config()

    assert cfg.tool_category_map == {}
    warnings = config_module.get_config_warnings()
    assert warnings == [
        {
            "code": "invalid_tool_category_map",
            "message": "MCP_TOOL_CATEGORY_MAP must be a JSON object; using an empty category map.",
            "next_action": "Provide valid JSON such as {\"tool.name\":\"category\"} or remove MCP_TOOL_CATEGORY_MAP.",
        }
    ]
    assert '{"media.search":' not in repr(warnings)


def test_idempotency_policy_config_uses_bounded_safe_defaults() -> None:
    cfg = get_config()

    assert cfg.idempotency_ttl_seconds == 300
    assert cfg.idempotency_cache_size == 512
    assert cfg.idempotency_wait_seconds == 5
    assert cfg.idempotency_finalize_seconds == 5
    assert cfg.idempotency_result_max_bytes == 256_000


@pytest.mark.parametrize(
    ("environment_name", "invalid_value"),
    [
        ("MCP_IDEMPOTENCY_TTL_SECONDS", "604801"),
        ("MCP_IDEMPOTENCY_CACHE_SIZE", "100001"),
        ("MCP_IDEMPOTENCY_WAIT_SECONDS", "31"),
        ("MCP_IDEMPOTENCY_FINALIZE_SECONDS", "16"),
        ("MCP_IDEMPOTENCY_RESULT_MAX_BYTES", "1000001"),
    ],
)
def test_idempotency_policy_config_rejects_values_above_hard_limits(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    invalid_value: str,
) -> None:
    monkeypatch.setenv(environment_name, invalid_value)
    get_config.cache_clear()  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        get_config()
