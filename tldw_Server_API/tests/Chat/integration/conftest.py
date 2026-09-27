"""
Local conftest to make path-based runs of Chat integration tests work cleanly.

Some environments invoke pytest on this subpackage directly, which may bypass
global plugin registration configured in pyproject.toml. To ensure the shared
fixtures are always available, re-export the suite-wide plugin fixtures here.

Note: We intentionally avoid defining `pytest_plugins` here to sidestep
pytest>=8 constraints. Direct wildcard imports expose the fixtures without
double-registration.
"""

import copy

# Re-export shared fixtures used across Chat tests
try:
    from tldw_Server_API.tests._plugins.chat_fixtures import *  # noqa: F401,F403
except Exception:
    # Never break collection if plugin import fails; tests that need these
    # fixtures will naturally error with a clear missing-fixture message.
    _ = None

try:
    from tldw_Server_API.tests._plugins.authnz_fixtures import *  # noqa: F401,F403
except Exception:
    _ = None

try:
    from tldw_Server_API.tests._plugins.postgres import *  # noqa: F401,F403
except Exception:
    _ = None

# Also expose the isolated Chat fixtures used by several test modules
try:
    from tldw_Server_API.tests.Chat.integration.conftest_isolated import *  # noqa: F401,F403
except Exception:
    _ = None

# Ensure OpenAI-backed tests use the local mock server to keep
# Chat integration deterministic and avoid external dependencies.
try:
    import os

    import pytest

    @pytest.fixture(autouse=True)
    def _isolate_provider_override_cache():
        """Keep short-lived app refresh workers from poisoning the next test."""
        from tldw_Server_API.app.core.AuthNZ import (
            llm_provider_overrides as overrides_module,
        )

        with overrides_module._OVERRIDE_LOCK:
            original = copy.deepcopy(overrides_module._OVERRIDE_CACHE)
            original_healthy = overrides_module._OVERRIDE_CACHE_HEALTHY
            original_ttl_enabled = (
                not overrides_module._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
            )
        overrides_module.set_llm_provider_overrides_cache_for_tests({})
        try:
            yield
        finally:
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                original,
                healthy=original_healthy,
                ttl_enabled=original_ttl_enabled,
            )


    @pytest.fixture(autouse=True)
    def _auto_configure_openai_mock(request, monkeypatch):  # noqa: F811
        use_mock = os.getenv("USE_OPENAI_MOCK_SERVER", "").lower() in {"1", "true", "yes", "y", "on"}
        has_real_key = bool(os.getenv("OPENAI_API_KEY"))
        if not use_mock and has_real_key:
            yield
            return

        try:
            mock_openai_server = request.getfixturevalue("mock_openai_server")
        except Exception:
            # If the mock server cannot start (e.g., sandbox port binding), skip mock setup.
            yield
            return

        # Forced mock mode must not retain any real key or higher-priority
        # endpoint alias from the invoking environment.
        mock_endpoint = f"{mock_openai_server}/v1"
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-12345")
        for endpoint_alias in (
            "OPENAI_API_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "MOCK_OPENAI_BASE_URL",
        ):
            monkeypatch.setenv(endpoint_alias, mock_endpoint)

        # Patch schema-level API_KEYS so provider is considered configured
        try:
            import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as chat_schemas
            monkeypatch.setitem(
                chat_schemas.API_KEYS,
                "openai",
                os.environ["OPENAI_API_KEY"],
            )
            # Sync endpoint copy if present
            try:
                from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
                monkeypatch.setattr(
                    chat_endpoint,
                    "API_KEYS",
                    chat_schemas.API_KEYS,
                    raising=False,
                )
            except Exception:
                _ = None
        except Exception:
            _ = None

        # Optionally patch config to surface base URL for adapters that read it
        try:
            from tldw_Server_API.app.core.config import load_and_log_configs as _llc
            cfg = _llc()
            if "openai_api" not in cfg:
                cfg["openai_api"] = {}
            cfg["openai_api"]["api_key"] = os.environ["OPENAI_API_KEY"]
            cfg["openai_api"]["api_base_url"] = f"{mock_openai_server}/v1"
            # Replace loader functions to return patched cfg
            import tldw_Server_API.app.core.config as _config_mod
            monkeypatch.setattr(
                _config_mod,
                "load_and_log_configs",
                lambda *_args, **_kwargs: cfg,
            )
            import tldw_Server_API.app.core.LLM_Calls.chat_calls as _llm_calls_mod
            monkeypatch.setattr(
                _llm_calls_mod,
                "load_and_log_configs",
                _config_mod.load_and_log_configs,
            )
            # ProviderCredentialRuntime must keep its real request-time loader:
            # tests may configure another provider after fixture setup, and the
            # loader atomically overlays these mock OpenAI environment aliases.
        except Exception:
            _ = None

        yield
except Exception:
    _ = None
