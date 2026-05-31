# test_config_providers_endpoints.py
"""
Tests for GET /config/providers and POST /config/validate-provider endpoints.
"""
import asyncio
import configparser
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.api.v1.endpoints import config_info as config_info_mod
from tldw_Server_API.app.api.v1.endpoints.config_info import (
    _PROVIDER_VALIDATION_INFO,
    ProviderValidateRequest,
    _check_validate_rate_limit,
    _key_hint,
    _resolve_provider_key,
    _validate_call_log,
    get_quickstart_redirect,
    list_configured_providers,
    load_safe_config,
    validate_provider_key,
)

_VALIDATE_HTTP_TARGET = (
    "tldw_Server_API.app.api.v1.endpoints.config_info._validate_provider_http"
)


def test_setup_saved_moonshot_and_zai_fields_load_into_runtime_adapter_config(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    parser = configparser.ConfigParser()
    parser["API"] = {
        "moonshot_api_key": "moonshot-secret",
        "moonshot_model": "moonshot-v1-32k",
        "moonshot_api_base_url": "https://moonshot.example/v1",
        "zai_api_key": "zai-secret",
        "zai_model": "glm-4.5",
        "zai_api_base_url": "https://zai.example/api/paas/v4",
    }
    for env_var in ("MOONSHOT_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    data = cfg.load_and_log_configs()

    assert data["moonshot_api"]["api_key"] == "moonshot-secret"
    assert data["moonshot_api"]["model"] == "moonshot-v1-32k"
    assert data["moonshot_api"]["api_base_url"] == "https://moonshot.example/v1"
    assert data["zai_api"]["api_key"] == "zai-secret"
    assert data["zai_api"]["model"] == "glm-4.5"
    assert data["zai_api"]["api_base_url"] == "https://zai.example/api/paas/v4"


def _make_mock_request(client_host: str = "127.0.0.1") -> MagicMock:
    """Create a mock FastAPI Request with a client IP."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = client_host
    return req


@contextmanager
def _capture_config_info_logs() -> Iterator[list[str]]:
    messages: list[str] = []
    sink_id = config_info_mod.logger.add(
        lambda message: messages.append(str(message))
        if message.record["name"] == config_info_mod.__name__
        else None,
        level="DEBUG",
    )
    try:
        yield messages
    finally:
        config_info_mod.logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Unit tests for _key_hint
# ---------------------------------------------------------------------------


class TestKeyHint:
    def test_long_key(self):
        assert _key_hint("sk-1234567890abcdef") == "sk-...cdef"

    def test_short_key(self):
        assert _key_hint("abcd") == "****cd"

    def test_very_short_key(self):
        assert _key_hint("a") == "****"

    def test_exactly_8_chars(self):
        # len <= 8 triggers short path (last 2 chars)
        assert _key_hint("12345678") == "****78"

    def test_9_chars_uses_long_path(self):
        assert _key_hint("123456789") == "123...6789"


# ---------------------------------------------------------------------------
# Unit tests for _resolve_provider_key
# ---------------------------------------------------------------------------


class TestResolveProviderKey:
    def test_returns_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        result = _resolve_provider_key("openai")
        assert result == "sk-test-key-123"

    def test_returns_none_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Patch the function at the module where it's imported from
        _target = "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys"
        with patch(_target, return_value={"openai": ""}):
            result = _resolve_provider_key("openai")
            assert result is None

    def test_ignores_whitespace_only_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        _target = "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys"
        with patch(_target, return_value={"openai": ""}):
            result = _resolve_provider_key("openai")
            assert result is None

    def test_falls_back_to_get_api_keys(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _target = "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys"
        with patch(_target, return_value={"anthropic": "ant-key-from-config"}):
            result = _resolve_provider_key("anthropic")
            assert result == "ant-key-from-config"

    def test_get_api_keys_failure_log_is_sanitized(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        logger_stub = MagicMock()

        def _fail_get_api_keys():
            raise RuntimeError("provider key loader exploded at /private/config.txt")

        monkeypatch.setattr(config_info_mod, "logger", logger_stub)
        _target = "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys"
        with patch(_target, side_effect=_fail_get_api_keys):
            result = _resolve_provider_key("openai")

        assert result is None
        logger_stub.debug.assert_called_once_with("Failed to load API keys for provider")


class TestConfigInfoSanitizedLogs:
    def test_missing_config_path_log_is_sanitized(self, monkeypatch):
        monkeypatch.setenv("TLDW_CONFIG_PATH", "/private/missing-config.txt")

        with _capture_config_info_logs() as messages:
            config = load_safe_config()

        joined = "\n".join(messages)
        assert config["configured"] is False
        assert "Config file not found" in joined
        assert "/private/missing-config.txt" not in joined

    def test_capability_flag_failure_log_is_sanitized(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.txt"
        config_path.write_text(
            "[Authentication]\nauth_mode = single_user\n\n[Server]\nhost = 127.0.0.1\nport = 8000\n",
            encoding="utf-8",
        )

        def _failing_route_enabled(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError("capability config exploded at /private/config.txt")

        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
        monkeypatch.setattr(config_info_mod.config_mod, "route_enabled", _failing_route_enabled)

        with _capture_config_info_logs() as messages:
            config = load_safe_config()

        joined = "\n".join(messages)
        assert config["configured"] is True
        assert "Failed to derive safe capability flags" in joined
        assert "capability config exploded" not in joined
        assert "/private/config.txt" not in joined

    @pytest.mark.asyncio
    async def test_quickstart_config_read_failure_log_is_sanitized(self, monkeypatch):
        def _failing_load_config():
            raise RuntimeError("quickstart config exploded at /private/config.txt")

        monkeypatch.delenv("QUICKSTART_URL", raising=False)
        monkeypatch.setattr(config_info_mod.config_mod, "load_comprehensive_config", _failing_load_config)

        with _capture_config_info_logs() as messages:
            response = await get_quickstart_redirect()

        joined = "\n".join(messages)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"
        assert "Quickstart redirect: could not read config, using default" in joined
        assert "quickstart config exploded" not in joined
        assert "/private/config.txt" not in joined

    @pytest.mark.asyncio
    async def test_quickstart_outer_failure_log_is_sanitized(self, monkeypatch):
        def _failing_getenv(name: str, default=None):  # noqa: ARG001
            raise RuntimeError("quickstart env exploded at /private/env")

        monkeypatch.setattr(config_info_mod.os, "getenv", _failing_getenv)

        with _capture_config_info_logs() as messages:
            response = await get_quickstart_redirect()

        joined = "\n".join(messages)
        assert response.status_code == 200
        assert "Quickstart redirect failed" in joined
        assert "quickstart env exploded" not in joined
        assert "/private/env" not in joined


# ---------------------------------------------------------------------------
# Tests for GET /config/providers
# ---------------------------------------------------------------------------


class TestListConfiguredProviders:
    @pytest.mark.asyncio
    async def test_returns_provider_list_structure(self, monkeypatch):
        # Clear all provider env vars to get a clean state
        for env_var in [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "COHERE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        with patch(
            "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
            return_value={},
        ):
            response = await list_configured_providers()

        assert hasattr(response, "providers")
        assert hasattr(response, "any_configured")
        assert isinstance(response.providers, list)
        assert len(response.providers) > 0

        # Check structure of first item
        first = response.providers[0]
        assert hasattr(first, "name")
        assert hasattr(first, "configured")
        assert hasattr(first, "requires_api_key")

    @pytest.mark.asyncio
    async def test_detects_configured_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-12345678")
        with patch(
            "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
            return_value={"openai": "sk-test-12345678"},
        ):
            response = await list_configured_providers()

        openai_item = next(p for p in response.providers if p.name == "openai")
        assert openai_item.configured is True
        assert openai_item.key_hint is not None
        assert "sk-" in openai_item.key_hint
        assert openai_item.key_source == "env"
        assert response.any_configured is True

    @pytest.mark.asyncio
    async def test_local_providers_always_configured(self):
        # Local providers don't require API keys
        with patch(
            "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
            return_value={},
        ):
            response = await list_configured_providers()

        ollama = next(p for p in response.providers if p.name == "ollama")
        assert ollama.configured is True
        assert ollama.requires_api_key is False
        assert ollama.key_hint is None

    @pytest.mark.asyncio
    async def test_custom_openai_providers_do_not_require_keys_by_default(self, monkeypatch):
        monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_OPENAI2_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_OPENAI_API_KEY_99", raising=False)

        with patch(
            "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
            return_value={},
        ):
            response = await list_configured_providers()

        for provider_name in ("custom-openai-api", "custom-openai-api-2", "custom-openai-api-99"):
            provider = next(p for p in response.providers if p.name == provider_name)
            assert provider.configured is True
            assert provider.requires_api_key is False
            assert provider.key_hint is None

    @pytest.mark.asyncio
    async def test_no_cloud_providers_configured(self, monkeypatch):
        # Remove all env vars
        for env_var in [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "COHERE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
            "DEEPSEEK_API_KEY", "HUGGINGFACE_API_KEY", "OPENROUTER_API_KEY",
            "QWEN_API_KEY", "MOONSHOT_API_KEY", "ZAI_API_KEY",
            "NOVITA_API_KEY", "POE_API_KEY", "TOGETHER_API_KEY",
            "AWS_ACCESS_KEY_ID",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        with patch(
            "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
            return_value={},
        ):
            response = await list_configured_providers()

        assert response.any_configured is False

    @pytest.mark.asyncio
    async def test_key_hint_does_not_expose_full_key(self, monkeypatch):
        full_key = "sk-very-secret-key-that-should-not-leak"
        monkeypatch.setenv("OPENAI_API_KEY", full_key)

        with patch(
            "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
            return_value={"openai": full_key},
        ):
            response = await list_configured_providers()

        openai_item = next(p for p in response.providers if p.name == "openai")
        assert openai_item.key_hint is not None
        # The hint should NOT contain the full key
        assert full_key not in openai_item.key_hint
        # Should contain last 4 chars
        assert full_key[-4:] in openai_item.key_hint


# ---------------------------------------------------------------------------
# Tests for POST /config/validate-provider
# ---------------------------------------------------------------------------


class TestValidateProviderKey:
    @pytest.fixture(autouse=True)
    def _clear_rate_limit_state(self):
        """Reset the in-memory rate limiter between tests."""
        _validate_call_log.clear()
        yield
        _validate_call_log.clear()

    @pytest.mark.asyncio
    async def test_no_key_returns_invalid(self):
        """Omitting api_key should return an error requiring the caller to supply one."""
        body = ProviderValidateRequest(provider="openai")
        request = _make_mock_request()
        response = await validate_provider_key(body, request)

        assert response.valid is False
        assert "api_key is required" in response.error

    @pytest.mark.asyncio
    async def test_unknown_provider_with_key_returns_valid(self):
        """Providers without known validation URLs are assumed valid if key is present."""
        body = ProviderValidateRequest(
            provider="some-unknown-provider",
            api_key="test-key-123",
        )
        request = _make_mock_request()
        response = await validate_provider_key(body, request)
        assert response.valid is True
        assert response.error is None

    @pytest.mark.asyncio
    async def test_successful_validation(self):
        """Mock a successful HTTP validation call."""
        with patch(_VALIDATE_HTTP_TARGET, new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (True, None)
            body = ProviderValidateRequest(provider="openai", api_key="sk-valid-key-123")
            request = _make_mock_request()
            response = await validate_provider_key(body, request)

        assert response.valid is True
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_auth_failure_returns_invalid(self):
        """Mock a 401 response."""
        with patch(_VALIDATE_HTTP_TARGET, new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (False, "Authentication failed (HTTP 401)")
            body = ProviderValidateRequest(provider="openai", api_key="sk-invalid-key")
            request = _make_mock_request()
            response = await validate_provider_key(body, request)

        assert response.valid is False
        assert "Authentication failed" in response.error

    @pytest.mark.asyncio
    async def test_rate_limited_treated_as_valid(self):
        """429 means the key is valid but rate-limited."""
        with patch(_VALIDATE_HTTP_TARGET, new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (True, None)
            body = ProviderValidateRequest(provider="openai", api_key="sk-rate-limited-key")
            request = _make_mock_request()
            response = await validate_provider_key(body, request)

        assert response.valid is True

    @pytest.mark.asyncio
    async def test_anthropic_400_treated_as_valid(self):
        """Anthropic returns 400 for malformed requests even when auth succeeds."""
        with patch(_VALIDATE_HTTP_TARGET, new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (True, None)
            body = ProviderValidateRequest(provider="anthropic", api_key="ant-valid-key")
            request = _make_mock_request()
            response = await validate_provider_key(body, request)

        assert response.valid is True

    def test_google_uses_header_auth_not_query_string(self):
        """Google config must use x-goog-api-key header, not query parameter."""
        google_info = _PROVIDER_VALIDATION_INFO.get("google")
        assert google_info is not None, "Google provider must be in validation info"
        assert google_info.get("auth_header") == "x-goog-api-key"
        # No query_param or auth_style key should exist
        assert "auth_style" not in google_info
        assert "query_param" not in google_info

    @pytest.mark.asyncio
    async def test_timeout_handled_gracefully(self):
        """Timeouts should return a clear error."""
        with patch(_VALIDATE_HTTP_TARGET, new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = asyncio.TimeoutError()
            body = ProviderValidateRequest(provider="openai", api_key="sk-timeout-key")
            request = _make_mock_request()
            response = await validate_provider_key(body, request)

        assert response.valid is False
        assert "timed out" in response.error.lower()

    @pytest.mark.asyncio
    async def test_provider_validation_exception_log_is_sanitized(self):
        """Unexpected validation exceptions should not leak backend detail to logs."""
        with patch(_VALIDATE_HTTP_TARGET, new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = RuntimeError("provider backend exploded at /private/provider.key")
            body = ProviderValidateRequest(provider="openai", api_key="sk-test-key")
            request = _make_mock_request()

            with _capture_config_info_logs() as messages:
                response = await validate_provider_key(body, request)

        joined = "\n".join(messages)
        assert response.valid is False
        assert response.error == "Validation failed. The provider may be unreachable or the key may be invalid."
        assert "Provider validation failed" in joined
        assert "provider backend exploded" not in joined
        assert "/private/" not in joined

    @pytest.mark.asyncio
    async def test_no_fallback_to_configured_key(self, monkeypatch):
        """Even when a server key is configured, omitting api_key must fail."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-123")

        body = ProviderValidateRequest(provider="openai")
        request = _make_mock_request()
        response = await validate_provider_key(body, request)

        # Must NOT fall back to the server's configured key
        assert response.valid is False
        assert "api_key is required" in response.error


# ---------------------------------------------------------------------------
# Tests for rate limiting on validate-provider
# ---------------------------------------------------------------------------


class TestValidateProviderRateLimit:
    @pytest.fixture(autouse=True)
    def _clear_rate_limit_state(self):
        """Reset the in-memory rate limiter between tests."""
        _validate_call_log.clear()
        yield
        _validate_call_log.clear()

    def test_allows_up_to_limit(self):
        """5 calls from the same IP should succeed."""
        for _ in range(5):
            _check_validate_rate_limit("10.0.0.1")  # should not raise

    def test_rejects_over_limit(self):
        """6th call from the same IP should raise 429."""
        for _ in range(5):
            _check_validate_rate_limit("10.0.0.2")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _check_validate_rate_limit("10.0.0.2")
        assert exc_info.value.status_code == 429

    def test_different_ips_independent(self):
        """Each IP has its own counter."""
        for _ in range(5):
            _check_validate_rate_limit("10.0.0.3")
        # Different IP should still be allowed
        _check_validate_rate_limit("10.0.0.4")  # should not raise

    @pytest.mark.asyncio
    async def test_endpoint_returns_429_when_rate_limited(self):
        """The endpoint itself should return HTTP 429 when rate-limited."""
        # Exhaust the limit for this IP
        for _ in range(5):
            _check_validate_rate_limit("10.0.0.5")

        body = ProviderValidateRequest(provider="openai", api_key="sk-test")
        request = _make_mock_request(client_host="10.0.0.5")

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await validate_provider_key(body, request)
        assert exc_info.value.status_code == 429
