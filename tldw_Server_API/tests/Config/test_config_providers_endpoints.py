# test_config_providers_endpoints.py
"""
Tests for GET /config/providers and POST /config/validate-provider endpoints.
"""
import asyncio
import configparser
import copy
import logging
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import context as otel_context

from tldw_Server_API.app.api.v1.endpoints import config_info as config_info_mod
from tldw_Server_API.app.api.v1.endpoints.config_info import (
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
from tldw_Server_API.app.core.AuthNZ import byok_testing
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    PROVIDER_STREAM_ERROR_MESSAGES,
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


def test_setup_saved_kobold_and_tabby_fields_load_into_runtime_adapter_config(monkeypatch):
    from tldw_Server_API.app.core import config as cfg

    parser = configparser.ConfigParser()
    parser["Local-API"] = {
        "kobold_api_IP": "http://127.0.0.1:5001/api/v1/generate",
        "kobold_api_key": "kobold-secret",
        "tabby_api_IP": "http://127.0.0.1:5000/v1",
        "tabby_api_key": "tabby-secret",
        "tabby_model": "tabby-local-model",
    }
    monkeypatch.setattr(cfg, "load_comprehensive_config", lambda: parser)

    data = cfg.load_and_log_configs()

    assert data["kobold_api"]["api_ip"] == "http://127.0.0.1:5001/api/v1/generate"
    assert data["kobold_api"]["api_key"] == "kobold-secret"
    assert data["tabby_api"]["api_ip"] == "http://127.0.0.1:5000/v1"
    assert data["tabby_api"]["api_key"] == "tabby-secret"
    assert data["tabby_api"]["model"] == "tabby-local-model"


def _make_mock_request(client_host: str = "127.0.0.1") -> MagicMock:
    """Create a mock FastAPI Request with a client IP."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = client_host
    return req


class _RecordingValidationAdapter:
    """Record immutable adapter-boundary requests for provider validation tests."""

    async_chat_is_native = False

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        with self._lock:
            self.calls.append(copy.deepcopy(request))
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": "ok"}}]}


class _GatedValidationAdapter(_RecordingValidationAdapter):
    """Block admitted calls until released and expose actual worker lifetime."""

    def __init__(self, *, expected_calls: int = 1) -> None:
        super().__init__()
        self.expected_calls = expected_calls
        self.all_entered = threading.Event()
        self.release = threading.Event()
        self.drained = threading.Event()
        self.active_count = 0

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        with self._lock:
            self.calls.append(copy.deepcopy(request))
            self.active_count += 1
            if len(self.calls) >= self.expected_calls:
                self.all_entered.set()
        try:
            if not self.release.wait(timeout=5.0):
                raise AssertionError("Timed out waiting to release validation adapter")
            return {"choices": [{"message": {"content": "ok"}}]}
        finally:
            with self._lock:
                self.active_count -= 1
                if self.active_count == 0:
                    self.drained.set()


class _ProviderValidationHTTPResponse:
    """Complete successful response used below the real provider adapter."""

    status_code = 200
    headers: dict[str, str] = {}
    text = '{"choices":[{"message":{"content":"pong"}}]}'

    def __init__(self, url: str) -> None:
        self.url = url
        self.request = types.SimpleNamespace(url=url)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "pong"}}]}

    def close(self) -> None:
        return None


class _ProviderValidationHTTPClient:
    """Record only HTTP I/O that escaped the validation egress boundary."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    def __enter__(self) -> "_ProviderValidationHTTPClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _ProviderValidationHTTPResponse:
        self.calls.append(
            {
                "url": url,
                "headers": copy.deepcopy(headers),
                "json": copy.deepcopy(json),
            }
        )
        return _ProviderValidationHTTPResponse(url)


def _install_config_validation_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapters: dict[str, Any],
    snapshot_loader: Any,
    pool: BoundedDaemonPool | None = None,
    health_capacity: int = 3,
) -> BoundedDaemonPool:
    """Install the real configured adapter helper beneath the public endpoint."""
    boundary_pool = pool or BoundedDaemonPool(max(health_capacity + 1, 2))

    async def reject_legacy_http_probe(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Provider validation used the legacy HTTP probe")

    monkeypatch.setattr(
        config_info_mod,
        "_validate_provider_http",
        reject_legacy_http_probe,
        raising=False,
    )
    monkeypatch.setattr(
        config_info_mod,
        "load_server_config_snapshot",
        snapshot_loader,
        raising=False,
    )
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(
        byok_testing,
        "get_registry",
        lambda: types.SimpleNamespace(
            get_adapter=lambda provider: adapters.get(provider),
        ),
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", boundary_pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        threading.BoundedSemaphore(health_capacity),
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "1",
    )

    def _reject_live_model_fallback(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Provider validation consulted live model configuration")

    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        _reject_live_model_fallback,
    )
    return boundary_pool


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


@contextmanager
def _capture_all_provider_validation_logs() -> Iterator[list[str]]:
    """Capture cross-module logs so a denied endpoint cannot leak through adapters."""
    messages: list[str] = []
    sink_id = config_info_mod.logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )
    try:
        yield messages
    finally:
        config_info_mod.logger.remove(sink_id)


def _install_real_http_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str,
    endpoint: str,
    calls: list[dict[str, Any]],
) -> None:
    """Install a real OpenAI-compatible adapter with fake terminal network I/O."""
    from tldw_Server_API.app.core import http_client as http_client_mod
    from tldw_Server_API.app.core.LLM_Calls.providers import (
        custom_openai_adapter,
        openai_adapter,
    )

    if provider == "openai":
        adapter = openai_adapter.OpenAIAdapter()
        adapter_module = openai_adapter
        snapshot = {
            "openai_api": {
                "api_key": "sk-server-key-must-not-dispatch",
                "api_base_url": endpoint,
                "model": "gpt-validation",
            }
        }
    else:
        adapter = custom_openai_adapter.CustomOpenAIAdapter2()
        adapter_module = custom_openai_adapter
        snapshot = {
            "custom_openai_api_2": {
                "api_key": "sk-server-key-must-not-dispatch",
                "api_ip": endpoint,
                "model": "custom-validation",
            }
        }

    client = _ProviderValidationHTTPClient(calls)
    monkeypatch.setattr(
        adapter_module,
        "http_client_factory",
        lambda **_kwargs: client,
    )

    def fake_httpx_request_io(**kwargs: Any) -> _ProviderValidationHTTPResponse:
        calls.append(
            {
                "url": str(kwargs["url"]),
                "headers": copy.deepcopy(kwargs.get("headers") or {}),
                "json": copy.deepcopy(kwargs.get("json") or {}),
            }
        )
        return _ProviderValidationHTTPResponse(str(kwargs["url"]))

    monkeypatch.setattr(
        http_client_mod,
        "_httpx_request_io",
        fake_httpx_request_io,
    )
    monkeypatch.setenv("HTTP_CLIENT_BACKEND", "httpx")
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "permissive")
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443,8080")
    for name in (
        "EGRESS_ALLOWLIST",
        "EGRESS_DENYLIST",
        "WORKFLOWS_EGRESS_ALLOWLIST",
        "WORKFLOWS_EGRESS_DENYLIST",
    ):
        monkeypatch.delenv(name, raising=False)

    _install_config_validation_adapter_boundary(
        monkeypatch,
        adapters={provider: adapter},
        snapshot_loader=lambda: copy.deepcopy(snapshot),
    )


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

        monkeypatch.setattr(config_info_mod, "os", types.SimpleNamespace(getenv=_failing_getenv))

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
    async def test_unknown_provider_with_key_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unknown identities cannot become valid from key presence alone."""
        adapter = _RecordingValidationAdapter()
        load_calls = 0

        def load_snapshot() -> dict[str, Any]:
            nonlocal load_calls
            load_calls += 1
            return {
                "some_unknown_provider_api": {
                    "api_base_url": "https://unknown.example/v1",
                    "model": "unknown-model",
                }
            }

        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"some-unknown-provider": adapter},
            snapshot_loader=load_snapshot,
        )
        body = ProviderValidateRequest(
            provider="some-unknown-provider",
            api_key="test-key-123",
        )
        request = _make_mock_request()
        response = await validate_provider_key(body, request)
        assert response.valid is False
        assert response.error is not None
        assert load_calls == 0
        assert adapter.call_count == 0

    @pytest.mark.asyncio
    async def test_keyless_local_provider_cannot_trigger_validation_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _RecordingValidationAdapter()
        load_calls = 0

        def load_snapshot() -> dict[str, Any]:
            nonlocal load_calls
            load_calls += 1
            return {
                "ollama_api": {
                    "api_url": "http://127.0.0.1:11434",
                    "model": "local-model",
                }
            }

        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"ollama": adapter},
            snapshot_loader=load_snapshot,
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="ollama",
                api_key="caller-value-must-not-authorize-local-dispatch",
            ),
            _make_mock_request(),
        )

        assert response.provider == "ollama"
        assert response.valid is False
        assert response.error is not None
        assert load_calls == 0
        assert adapter.call_count == 0

    @pytest.mark.asyncio
    async def test_successful_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A supported provider succeeds through the shared adapter boundary."""
        adapter = _RecordingValidationAdapter()
        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"openai": adapter},
            snapshot_loader=lambda: {"openai_api": {"model": "gpt-validation"}},
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="openai",
                api_key="sk-valid-key-123",
            ),
            _make_mock_request(),
        )

        assert response.valid is True
        assert response.provider == "openai"
        assert adapter.call_count == 1
        assert adapter.calls[0]["api_key"] == "sk-valid-key-123"

    @pytest.mark.asyncio
    async def test_provider_validation_exception_log_is_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unexpected validation exceptions should not leak backend detail to logs."""
        adapter = _RecordingValidationAdapter(
            RuntimeError("provider backend exploded at /private/provider.key")
        )
        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"openai": adapter},
            snapshot_loader=lambda: {"openai_api": {"model": "gpt-validation"}},
        )

        with _capture_config_info_logs() as messages:
            response = await validate_provider_key(
                ProviderValidateRequest(provider="openai", api_key="sk-test-key"),
                _make_mock_request(),
            )

        joined = "\n".join(messages)
        assert response.valid is False
        assert response.error == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
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

    @pytest.mark.asyncio
    async def test_custom_provider_requires_configured_model_before_adapter_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _RecordingValidationAdapter()
        load_calls = 0

        def load_snapshot() -> dict[str, Any]:
            nonlocal load_calls
            load_calls += 1
            return {
                "custom_openai_api_2": {
                    "api_ip": "https://custom-two.example/v1",
                }
            }

        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"custom-openai-api-2": adapter},
            snapshot_loader=load_snapshot,
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="custom-openai-api-2",
                api_key="sk-caller-custom-two",
            ),
            _make_mock_request(),
        )

        assert response.provider == "custom-openai-api-2"
        assert response.valid is False
        assert response.error is not None
        assert load_calls == 1
        assert adapter.call_count == 0

    @pytest.mark.asyncio
    async def test_custom_provider_missing_endpoint_fails_before_http_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tldw_Server_API.app.core.LLM_Calls.providers import (
            custom_openai_adapter,
        )

        http_dispatches = 0

        def unexpected_http_dispatch(**_kwargs: Any) -> Any:
            nonlocal http_dispatches
            http_dispatches += 1
            raise AssertionError("Missing custom endpoint reached HTTP dispatch")

        monkeypatch.setattr(
            custom_openai_adapter,
            "http_client_factory",
            unexpected_http_dispatch,
        )
        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={
                "custom-openai-api-2": custom_openai_adapter.CustomOpenAIAdapter2(),
            },
            snapshot_loader=lambda: {
                "custom_openai_api_2": {"model": "custom-snapshot-model"},
            },
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="custom-openai-api-2",
                api_key="sk-caller-custom-two",
            ),
            _make_mock_request(),
        )

        assert response.provider == "custom-openai-api-2"
        assert response.valid is False
        assert response.error is not None
        assert http_dispatches == 0

    @pytest.mark.asyncio
    async def test_configured_custom_provider_uses_caller_key_and_one_frozen_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _RecordingValidationAdapter()
        server_key = "sk-server-key-that-must-not-dispatch"
        caller_key = "sk-caller-key-under-test"
        snapshot = {
            "custom_openai_api_2": {
                "api_key": server_key,
                "api_ip": "https://custom-two.example/v1",
                "model": "custom-snapshot-model",
                "org_id": "org-snapshot",
                "project_id": "project-snapshot",
            }
        }
        load_calls = 0

        def load_snapshot() -> dict[str, Any]:
            nonlocal load_calls
            load_calls += 1
            return copy.deepcopy(snapshot)

        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"custom-openai-api-2": adapter},
            snapshot_loader=load_snapshot,
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="custom-openai-api-2",
                api_key=caller_key,
            ),
            _make_mock_request(),
        )

        assert response.provider == "custom-openai-api-2"
        assert response.valid is True
        assert response.error is None
        assert load_calls == 1
        assert adapter.call_count == 1
        dispatched = adapter.calls[0]
        assert dispatched["api_key"] == caller_key
        assert dispatched["model"] == "custom-snapshot-model"
        assert dispatched["credentials_resolved"] is True
        assert dispatched["app_config"]["custom_openai_api_2"] == {
            "api_ip": "https://custom-two.example/v1",
            "model": "custom-snapshot-model",
            "org_id": "org-snapshot",
            "project_id": "project-snapshot",
        }
        assert server_key not in repr(dispatched)

    @pytest.mark.asyncio
    async def test_caller_bedrock_key_clears_default_chain_auth_marker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _RecordingValidationAdapter()
        snapshot = {
            "bedrock_api": {
                "runtime_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "region": "us-east-1",
                "model": "bedrock-snapshot-model",
            }
        }

        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"bedrock": adapter},
            snapshot_loader=lambda: copy.deepcopy(snapshot),
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="aws-bedrock",
                api_key="caller-bedrock-bearer-token",
            ),
            _make_mock_request(),
        )

        assert response.provider == "bedrock"
        assert response.valid is True
        assert adapter.call_count == 1
        dispatched = adapter.calls[0]
        assert dispatched["api_key"] == "caller-bedrock-bearer-token"
        assert dispatched["model"] == "bedrock-snapshot-model"
        assert (
            "_runtime_auth_source"
            not in dispatched["app_config"]["bedrock_api"]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error_type", "expected_code"),
        [
            (ChatAuthenticationError, "provider_authentication_failed"),
            (ChatBadRequestError, "provider_configuration_invalid"),
            (ChatRateLimitError, "provider_unavailable"),
        ],
    )
    async def test_validate_provider_sanitizes_real_adapter_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error_type: type[Exception],
        expected_code: str,
    ) -> None:
        sentinel = f"sk-hostile-{expected_code}-/private/provider-body.json"
        adapter = _RecordingValidationAdapter(
            error_type(
                message=f"hostile upstream response {sentinel}",
                provider="custom-openai-api-2",
            )
        )
        snapshot = {
            "custom_openai_api_2": {
                "api_ip": "https://custom-two.example/v1",
                "model": "custom-snapshot-model",
            }
        }
        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"custom-openai-api-2": adapter},
            snapshot_loader=lambda: copy.deepcopy(snapshot),
        )

        with _capture_config_info_logs() as messages:
            response = await validate_provider_key(
                ProviderValidateRequest(
                    provider="custom-openai-api-2",
                    api_key="sk-caller-custom-two",
                ),
                _make_mock_request(),
            )

        assert response.valid is False
        assert response.error == PROVIDER_STREAM_ERROR_MESSAGES[expected_code]
        assert adapter.call_count == 1
        assert sentinel not in response.error
        assert sentinel not in "\n".join(messages)

    @pytest.mark.asyncio
    async def test_concurrent_validate_provider_calls_do_not_mix_config_generations(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = _GatedValidationAdapter(expected_calls=2)
        snapshots = iter(
            [
                {
                    "custom_openai_api_2": {
                        "api_key": "sk-server-a",
                        "api_ip": "https://generation-a.example/v1",
                        "model": "model-a",
                        "org_id": "org-a",
                    }
                },
                {
                    "custom_openai_api_2": {
                        "api_key": "sk-server-b",
                        "api_ip": "https://generation-b.example/v1",
                        "model": "model-b",
                        "org_id": "org-b",
                    }
                },
            ]
        )
        load_calls = 0

        def load_snapshot() -> dict[str, Any]:
            nonlocal load_calls
            load_calls += 1
            return copy.deepcopy(next(snapshots))

        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"custom-openai-api-2": adapter},
            snapshot_loader=load_snapshot,
            health_capacity=3,
        )
        monkeypatch.setenv(
            "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
            "2",
        )

        tasks = [
            asyncio.create_task(
                validate_provider_key(
                    ProviderValidateRequest(
                        provider="custom-openai-api-2",
                        api_key=caller_key,
                    ),
                    _make_mock_request(client_host=f"127.0.0.{index}"),
                )
            )
            for index, caller_key in enumerate(("sk-caller-a", "sk-caller-b"), start=1)
        ]
        try:
            assert await asyncio.to_thread(adapter.all_entered.wait, 1.0)
        finally:
            adapter.release.set()
            responses = await asyncio.gather(*tasks)

        assert all(response.valid for response in responses)
        assert load_calls == 2
        assert {call["api_key"] for call in adapter.calls} == {
            "sk-caller-a",
            "sk-caller-b",
        }
        assert {
            (
                call["app_config"]["custom_openai_api_2"]["api_ip"],
                call["model"],
                call["app_config"]["custom_openai_api_2"]["org_id"],
            )
            for call in adapter.calls
        } == {
            (
                "https://generation-a.example/v1",
                "model-a",
                "org-a",
            ),
            (
                "https://generation-b.example/v1",
                "model-b",
                "org-b",
            ),
        }
        assert "sk-server-a" not in repr(adapter.calls)
        assert "sk-server-b" not in repr(adapter.calls)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("abandonment", ["timeout", "cancellation"])
    async def test_validate_provider_abandonment_retains_same_provider_lifetime_bound(
        self,
        monkeypatch: pytest.MonkeyPatch,
        abandonment: str,
    ) -> None:
        custom_adapter = _GatedValidationAdapter(expected_calls=1)
        anthropic_adapter = _RecordingValidationAdapter()
        snapshot = {
            "custom_openai_api_2": {
                "api_ip": "https://custom-two.example/v1",
                "model": "custom-snapshot-model",
            },
            "anthropic_api": {
                "api_base_url": "https://anthropic.example/v1",
                "model": "anthropic-snapshot-model",
            },
        }
        pool = _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={
                "custom-openai-api-2": custom_adapter,
                "anthropic": anthropic_adapter,
            },
            snapshot_loader=lambda: copy.deepcopy(snapshot),
            pool=BoundedDaemonPool(4),
            health_capacity=3,
        )
        monkeypatch.setattr(
            config_info_mod,
            "_VALIDATION_TIMEOUT_SECONDS",
            0.03 if abandonment == "timeout" else 1.0,
        )

        def request_for(provider: str, host: str) -> tuple[ProviderValidateRequest, MagicMock]:
            return (
                ProviderValidateRequest(
                    provider=provider,
                    api_key=f"sk-{provider}-caller",
                ),
                _make_mock_request(client_host=host),
            )

        first = asyncio.create_task(
            validate_provider_key(*request_for("custom-openai-api-2", "127.0.0.10"))
        )
        same_provider: asyncio.Task[Any] | None = None
        other_provider: asyncio.Task[Any] | None = None
        try:
            assert await asyncio.to_thread(custom_adapter.all_entered.wait, 1.0)
            if abandonment == "timeout":
                first_response = await first
                assert first_response.valid is False
                assert (
                    first_response.error
                    == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
                )
            else:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first

            monkeypatch.setattr(
                config_info_mod,
                "_VALIDATION_TIMEOUT_SECONDS",
                0.03,
            )
            same_provider = asyncio.create_task(
                validate_provider_key(*request_for("custom-openai-api-2", "127.0.0.11"))
            )
            other_provider = asyncio.create_task(
                validate_provider_key(*request_for("anthropic", "127.0.0.12"))
            )

            other_response = await other_provider
            same_response = await same_provider
            assert other_response.valid is True
            assert same_response.valid is False
            assert (
                same_response.error
                == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
            )
            assert custom_adapter.call_count == 1
            assert custom_adapter.active_count == 1
            assert anthropic_adapter.call_count == 1
        finally:
            custom_adapter.release.set()
            await asyncio.gather(
                first,
                *(task for task in (same_provider, other_provider) if task is not None),
                return_exceptions=True,
            )

        assert await asyncio.to_thread(custom_adapter.drained.wait, 1.0)
        for _ in range(1000):
            if pool.active_count == 0:
                break
            await asyncio.sleep(0.001)
        assert pool.active_count == 0

        monkeypatch.setattr(config_info_mod, "_VALIDATION_TIMEOUT_SECONDS", 0.2)
        recovered = await validate_provider_key(
            *request_for("custom-openai-api-2", "127.0.0.13")
        )
        assert recovered.valid is True
        assert custom_adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_provider_health_work_preserves_shared_chat_headroom(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        custom_adapter = _GatedValidationAdapter(expected_calls=1)
        anthropic_adapter = _GatedValidationAdapter(expected_calls=1)
        chat_adapter = _RecordingValidationAdapter()
        snapshot = {
            "custom_openai_api_2": {
                "api_ip": "https://custom-two.example/v1",
                "model": "custom-snapshot-model",
            },
            "anthropic_api": {
                "api_base_url": "https://anthropic.example/v1",
                "model": "anthropic-snapshot-model",
            },
        }
        pool = BoundedDaemonPool(3)
        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={
                "custom-openai-api-2": custom_adapter,
                "anthropic": anthropic_adapter,
            },
            snapshot_loader=lambda: copy.deepcopy(snapshot),
            pool=pool,
            health_capacity=2,
        )
        monkeypatch.setattr(config_info_mod, "_VALIDATION_TIMEOUT_SECONDS", 1.0)
        monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
        monkeypatch.setattr(
            chat_service,
            "_get_llm_registry",
            lambda: types.SimpleNamespace(get_adapter=lambda _provider: chat_adapter),
        )

        validations = [
            asyncio.create_task(
                validate_provider_key(
                    ProviderValidateRequest(
                        provider=provider,
                        api_key=f"sk-{provider}-caller",
                    ),
                    _make_mock_request(client_host=host),
                )
            )
            for provider, host in (
                ("custom-openai-api-2", "127.0.0.20"),
                ("anthropic", "127.0.0.21"),
            )
        ]
        try:
            entered = await asyncio.gather(
                asyncio.to_thread(custom_adapter.all_entered.wait, 1.0),
                asyncio.to_thread(anthropic_adapter.all_entered.wait, 1.0),
            )
            assert entered == [True, True]
            await chat_service.perform_chat_api_call_async(
                api_endpoint="groq",
                api_key="sk-foreground-chat",
                credentials_resolved=True,
                messages_payload=[{"role": "user", "content": "ping"}],
                model="foreground-chat-model",
                streaming=False,
            )
            assert chat_adapter.call_count == 1
            assert pool.active_count == 2
        finally:
            custom_adapter.release.set()
            anthropic_adapter.release.set()
            responses = await asyncio.gather(*validations)

        assert all(response.valid for response in responses)
        assert pool.active_count == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["openai", "custom-openai-api-2"])
    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://127.0.0.1:8080/validation-endpoint-secret",
            "http://169.254.169.254/validation-endpoint-secret",
        ],
        ids=["loopback", "link-local-metadata"],
    )
    async def test_unauthenticated_validation_denies_private_endpoint_before_http(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        endpoint: str,
    ) -> None:
        """The actual adapter request path must enforce egress before network I/O."""
        http_calls: list[dict[str, Any]] = []
        caller_key = "sk-caller-egress-secret"
        _install_real_http_validation_boundary(
            monkeypatch,
            provider=provider,
            endpoint=endpoint,
            calls=http_calls,
        )

        with _capture_all_provider_validation_logs() as messages:
            response = await validate_provider_key(
                ProviderValidateRequest(provider=provider, api_key=caller_key),
                _make_mock_request(client_host="127.0.0.90"),
            )

        assert response.provider == provider
        assert response.valid is False
        assert response.error in {
            PROVIDER_STREAM_ERROR_MESSAGES["provider_configuration_invalid"],
            PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
        }
        assert http_calls == []
        observed = f"{response!r}\n" + "\n".join(messages)
        assert caller_key not in observed
        assert "validation-endpoint-secret" not in observed
        assert "sk-server-key-must-not-dispatch" not in observed

    @pytest.mark.asyncio
    async def test_unauthenticated_custom_validation_enforces_egress_in_worker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An event-loop precheck cannot replace policy enforcement at dispatch."""
        from tldw_Server_API.app.core.Security import egress as egress_policy

        http_calls: list[dict[str, Any]] = []
        policy_threads: list[str] = []
        endpoint = "https://8.8.8.8/dispatch-policy-secret"
        _install_real_http_validation_boundary(
            monkeypatch,
            provider="custom-openai-api-2",
            endpoint=endpoint,
            calls=http_calls,
        )

        def dispatch_sensitive_policy(
            _url: str,
            **_kwargs: Any,
        ) -> types.SimpleNamespace:
            thread_name = threading.current_thread().name
            policy_threads.append(thread_name)
            in_validation_worker = thread_name.startswith("admin-provider-health-")
            return types.SimpleNamespace(
                allowed=not in_validation_worker,
                reason=("dispatch policy denied" if in_validation_worker else None),
                resolved_ips=(),
            )

        monkeypatch.setattr(
            egress_policy,
            "evaluate_url_policy",
            dispatch_sensitive_policy,
        )

        with _capture_all_provider_validation_logs() as messages:
            response = await validate_provider_key(
                ProviderValidateRequest(
                    provider="custom-openai-api-2",
                    api_key="sk-worker-policy-secret",
                ),
                _make_mock_request(client_host="127.0.0.92"),
            )

        assert response.valid is False
        assert http_calls == []
        assert any(
            name.startswith("admin-provider-health-") for name in policy_threads
        )
        observed = f"{response!r}\n" + "\n".join(messages)
        assert "dispatch-policy-secret" not in observed
        assert "sk-worker-policy-secret" not in observed

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", ["openai", "custom-openai-api-2"])
    async def test_unauthenticated_validation_allows_public_endpoint_control(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
    ) -> None:
        http_calls: list[dict[str, Any]] = []
        caller_key = f"sk-{provider}-public-control"
        endpoint = "https://8.8.8.8/validation-public-control"
        _install_real_http_validation_boundary(
            monkeypatch,
            provider=provider,
            endpoint=endpoint,
            calls=http_calls,
        )

        response = await validate_provider_key(
            ProviderValidateRequest(provider=provider, api_key=caller_key),
            _make_mock_request(client_host="127.0.0.91"),
        )

        assert response.provider == provider
        assert response.valid is True
        assert response.error is None
        assert len(http_calls) == 1
        assert http_calls[0]["url"].startswith(endpoint)
        assert http_calls[0]["headers"]["Authorization"] == f"Bearer {caller_key}"
        assert "sk-server-key-must-not-dispatch" not in repr(http_calls)

    @pytest.mark.asyncio
    async def test_allowed_validation_suppresses_transport_logs_and_auto_instrumentation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sensitive observability must cover the real worker adapter call."""
        from tldw_Server_API.app.core import http_client as http_client_mod
        from tldw_Server_API.app.core.LLM_Calls.providers import (
            custom_openai_adapter,
        )

        endpoint = (
            "https://8.8.8.8/provider-validation-log-secret"
            "?credential=url-query-secret"
        )
        caller_key = "sk-caller-transport-log-secret"
        http_calls: list[dict[str, Any]] = []
        suppression_states: list[object] = []
        auto_instrumented_urls: list[str] = []
        stdlib_messages: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                stdlib_messages.append(record.getMessage())

        class LoggingHTTPClient(_ProviderValidationHTTPClient):
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
            ) -> _ProviderValidationHTTPResponse:
                suppression = otel_context.get_value(
                    http_client_mod._OTEL_HTTP_SUPPRESSION_KEY
                )
                suppression_states.append(suppression)
                if suppression is not True:
                    auto_instrumented_urls.append(url)
                logging.getLogger("httpx").info(
                    "HTTP Request: POST %s headers=%r json=%r",
                    url,
                    headers,
                    json,
                )
                logging.getLogger("httpcore.connection").debug(
                    "request url=%s body=%r",
                    url,
                    json,
                )
                return super().post(url, headers=headers, json=json)

        _install_real_http_validation_boundary(
            monkeypatch,
            provider="custom-openai-api-2",
            endpoint=endpoint,
            calls=http_calls,
        )
        client = LoggingHTTPClient(http_calls)
        monkeypatch.setattr(
            custom_openai_adapter,
            "http_client_factory",
            lambda **_kwargs: client,
        )

        capture_handler = CaptureHandler()
        transport_loggers = [
            logging.getLogger("httpx"),
            logging.getLogger("httpcore.connection"),
        ]
        previous_logger_state = [
            (transport_logger.level, transport_logger.propagate)
            for transport_logger in transport_loggers
        ]
        for transport_logger in transport_loggers:
            transport_logger.addHandler(capture_handler)
            transport_logger.setLevel(logging.DEBUG)
            transport_logger.propagate = False

        try:
            with _capture_all_provider_validation_logs() as provider_logs:
                response = await validate_provider_key(
                    ProviderValidateRequest(
                        provider="custom-openai-api-2",
                        api_key=caller_key,
                    ),
                    _make_mock_request(client_host="127.0.0.95"),
                )
        finally:
            for transport_logger, (level, propagate) in zip(
                transport_loggers,
                previous_logger_state,
            ):
                transport_logger.removeHandler(capture_handler)
                transport_logger.setLevel(level)
                transport_logger.propagate = propagate

        assert response.valid is True
        assert response.error is None
        assert len(http_calls) == 1
        assert http_calls[0]["url"].startswith(endpoint)
        assert http_calls[0]["headers"]["Authorization"] == f"Bearer {caller_key}"
        assert http_calls[0]["json"]["messages"] == [
            {"role": "user", "content": "ping"}
        ]
        assert suppression_states == [True]
        assert auto_instrumented_urls == []
        assert otel_context.get_value(http_client_mod._OTEL_HTTP_SUPPRESSION_KEY) is None

        observability = repr(
            {
                "stdlib": stdlib_messages,
                "provider": provider_logs,
                "auto_instrumented": auto_instrumented_urls,
            }
        )
        for sensitive_fragment in (
            endpoint,
            "provider-validation-log-secret",
            "credential=url-query-secret",
            caller_key,
            "ping",
        ):
            assert sensitive_fragment not in observability

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("api_base_url", "irrelevant_runtime_endpoint", "expected_valid"),
        [
            (
                "http://169.254.169.254/openai-effective-private",
                "https://8.8.8.8/openai-irrelevant-public",
                False,
            ),
            (
                "https://8.8.8.8/openai-effective-public",
                "https://8.8.8.8/openai-irrelevant-public",
                True,
            ),
        ],
        ids=["conflicting-private-effective", "aligned-public-control"],
    )
    async def test_unauthenticated_openai_egress_uses_adapter_endpoint_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_base_url: str,
        irrelevant_runtime_endpoint: str,
        expected_valid: bool,
    ) -> None:
        """OpenAI policy must evaluate the base URL its adapter will dispatch."""
        http_calls: list[dict[str, Any]] = []
        _install_real_http_validation_boundary(
            monkeypatch,
            provider="openai",
            endpoint=api_base_url,
            calls=http_calls,
        )
        monkeypatch.setattr(
            config_info_mod,
            "load_server_config_snapshot",
            lambda: {
                "openai_api": {
                    "api_key": "sk-server-key-must-not-dispatch",
                    "api_base_url": api_base_url,
                    "runtime_endpoint": irrelevant_runtime_endpoint,
                    "model": "gpt-validation",
                }
            },
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="openai",
                api_key="sk-openai-precedence-candidate",
            ),
            _make_mock_request(client_host="127.0.0.93"),
        )

        assert response.valid is expected_valid
        if expected_valid:
            assert len(http_calls) == 1
            assert http_calls[0]["url"].startswith(api_base_url)
        else:
            assert http_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("router_base_url", "irrelevant_api_base_url", "expected_valid"),
        [
            (
                "http://169.254.169.254/hf-effective-private",
                "https://8.8.8.8/hf-irrelevant-public",
                False,
            ),
            (
                "https://8.8.8.8/hf-effective-public",
                "https://8.8.8.8/hf-effective-public",
                True,
            ),
        ],
        ids=["conflicting-private-effective", "aligned-public-control"],
    )
    async def test_unauthenticated_huggingface_egress_uses_router_endpoint_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        router_base_url: str,
        irrelevant_api_base_url: str,
        expected_valid: bool,
    ) -> None:
        """Router mode must validate the router base actually used for HTTP."""
        from tldw_Server_API.app.core.LLM_Calls.providers import (
            huggingface_adapter,
        )

        http_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            huggingface_adapter,
            "http_client_factory",
            lambda **_kwargs: _ProviderValidationHTTPClient(http_calls),
        )
        for name, value in (
            ("WORKFLOWS_EGRESS_PROFILE", "permissive"),
            ("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true"),
            ("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443,8080"),
        ):
            monkeypatch.setenv(name, value)
        for name in (
            "EGRESS_ALLOWLIST",
            "EGRESS_DENYLIST",
            "WORKFLOWS_EGRESS_ALLOWLIST",
            "WORKFLOWS_EGRESS_DENYLIST",
        ):
            monkeypatch.delenv(name, raising=False)
        _install_config_validation_adapter_boundary(
            monkeypatch,
            adapters={"huggingface": huggingface_adapter.HuggingFaceAdapter()},
            snapshot_loader=lambda: {
                "huggingface_api": {
                    "api_key": "hf-server-key-must-not-dispatch",
                    "api_base_url": irrelevant_api_base_url,
                    "router_base_url": router_base_url,
                    "use_router_url_format": "true",
                    "model": "hf-validation-model",
                }
            },
        )

        response = await validate_provider_key(
            ProviderValidateRequest(
                provider="huggingface",
                api_key="hf-precedence-candidate",
            ),
            _make_mock_request(client_host="127.0.0.94"),
        )

        assert response.valid is expected_valid
        if expected_valid:
            assert len(http_calls) == 1
            assert http_calls[0]["url"].startswith(router_base_url)
        else:
            assert http_calls == []


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
