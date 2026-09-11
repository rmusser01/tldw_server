"""Runtime credential and adapter-boundary regressions for admin provider tests."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_llm_providers
from tldw_Server_API.app.api.v1.schemas.admin_schemas import LLMProviderTestRequest
from tldw_Server_API.app.core.AuthNZ import (
    byok_runtime,
    byok_testing,
    llm_provider_overrides,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    bind_provider_call_credentials,
)
from tldw_Server_API.app.services import admin_llm_providers_service as service
from tldw_Server_API.tests.provider_credential_test_helpers import (
    resolved_request_fields_async,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_provider_override_cache() -> Iterator[None]:
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    yield
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})


async def _ready() -> None:
    return None


async def _noop_refresh(*, force: bool | None = None) -> None:
    del force


class _CapturingAdapter:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.timeouts: list[float | None] = []

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.requests.append(request)
        self.timeouts.append(timeout)
        return {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}


class _BlockingAdapter(_CapturingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.call_count = 0

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.requests.append(request)
        self.timeouts.append(timeout)
        self.call_count += 1
        if self.call_count == 1:
            self.entered.set()
            self.release.wait(timeout=2.0)
        return {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}


class _AlwaysBlockingAdapter(_CapturingAdapter):
    """Block every provider call and expose real worker concurrency to tests."""

    def __init__(self) -> None:
        super().__init__()
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release = threading.Event()
        self.drained = threading.Event()
        self._state_lock = threading.Lock()
        self.call_count = 0
        self.active_count = 0
        self.max_active = 0

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            self.requests.append(request)
            self.timeouts.append(timeout)
            self.call_count += 1
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)
            self.first_entered.set()
            if self.call_count >= 2:
                self.second_entered.set()
        try:
            if not self.release.wait(timeout=5.0):
                raise AssertionError("Timed out waiting to release provider adapter")
            return {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}
        finally:
            with self._state_lock:
                self.active_count -= 1
                if self.active_count == 0:
                    self.drained.set()


class _ProviderCancellingAdapter(_CapturingAdapter):
    def __init__(self, sentinel: str) -> None:
        super().__init__()
        self.sentinel = sentinel

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.requests.append(request)
        self.timeouts.append(timeout)
        raise asyncio.CancelledError(self.sentinel)


class _Registry:
    def __init__(self, adapters: dict[str, Any]) -> None:
        self.adapters = {
            provider: _CredentialBindingAdapter(provider, adapter)
            for provider, adapter in adapters.items()
        }

    def get_adapter(self, provider: str) -> Any | None:
        return self.adapters.get(provider)


class _CredentialBindingAdapter:
    """Mirror the credential-consumption boundary implemented by real adapters."""

    async_chat_is_native = False

    def __init__(self, provider: str, adapter: Any) -> None:
        self._provider = provider
        self._adapter = adapter

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        bound_request, _credentials = bind_provider_call_credentials(
            self._provider,
            request,
            consume=True,
        )
        return self._adapter.chat(bound_request, timeout=timeout)


def _install_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
    adapters: dict[str, Any],
) -> None:
    registry = _Registry(adapters)
    monkeypatch.setattr(byok_testing, "get_registry", lambda: registry)
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)


async def _wait_for_thread_state(
    predicate,
    *,
    timeout: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for blocking adapter state")
        await asyncio.sleep(0.005)


def _configure_server_runtime(
    monkeypatch: pytest.MonkeyPatch,
    app_config: dict[str, Any],
) -> None:
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", app_config)
    monkeypatch.setattr(
        service,
        "load_server_config_snapshot",
        lambda: copy.deepcopy(app_config),
    )
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(service, "get_llm_provider_override", lambda _provider: None)
    monkeypatch.setattr(
        service,
        "get_override_default_model",
        lambda _provider: None,
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "get_override_server_fallback",
        lambda _provider: None,
        raising=False,
    )


@pytest.mark.asyncio
async def test_admin_test_provider_dispatches_env_openai_with_server_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-env-admin-boundary-secret"
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {
            "openai_api": {
                "api_key": secret,
                "api_base_url": "https://configured.openai.example/v1",
                "model": "gpt-configured",
            }
        },
    )
    monkeypatch.setattr(service, "get_llm_provider_override", lambda _provider: None)
    monkeypatch.setattr(
        service,
        "get_override_server_fallback",
        lambda _provider: None,
        raising=False,
    )
    result = await service.test_provider(LLMProviderTestRequest(provider="openai"))

    assert result.model == "gpt-configured"
    assert len(adapter.requests) == 1
    assert adapter.requests[0]["api_key"] == secret
    assert adapter.requests[0]["app_config"]["openai_api"] == {
        "api_base_url": "https://configured.openai.example/v1",
        "model": "gpt-configured",
    }
    assert secret not in result.model_dump_json()


@pytest.mark.parametrize(
    ("initial_config", "expected_status"),
    (
        (
            {
                "openai_api": {
                    "api_key": "admin-static-key-a",
                    "api_base_url": "https://admin-static-a.example/v1",
                    "model": "gpt-admin-static-a",
                }
            },
            200,
        ),
        ({}, 400),
    ),
    ids=("a-to-b", "absent-to-b"),
)
@pytest.mark.asyncio
async def test_admin_test_provider_static_fallback_freezes_one_adapter_generation(
    monkeypatch: pytest.MonkeyPatch,
    initial_config: dict[str, Any],
    expected_status: int,
) -> None:
    """Admin provider tests cannot combine a static key with later adapter config."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    rotated_config = {
        "openai_api": {
            "api_key": "admin-static-key-b",
            "api_base_url": "https://admin-static-b.example/v1",
            "model": "gpt-admin-static-b",
        }
    }
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})

    def load_static_snapshot() -> dict[str, Any]:
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        return copy.deepcopy(initial_config)

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(byok_runtime, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(service, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", initial_config)
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(service, "get_llm_provider_override", lambda _provider: None)
    monkeypatch.setattr(service, "get_override_default_model", lambda _provider: None)
    monkeypatch.setattr(service, "get_override_server_fallback", lambda _provider: None)

    if expected_status == 200:
        result = await service.test_provider(LLMProviderTestRequest(provider="openai"))
        assert result.model == "gpt-admin-static-a"
        assert len(adapter.requests) == 1
        assert adapter.requests[0]["api_key"] == "admin-static-key-a"
        assert adapter.requests[0]["app_config"]["openai_api"] == {
            "api_base_url": "https://admin-static-a.example/v1",
            "model": "gpt-admin-static-a",
        }
    else:
        with pytest.raises(HTTPException) as exc_info:
            await service.test_provider(LLMProviderTestRequest(provider="openai"))
        assert exc_info.value.status_code == expected_status
        assert adapter.requests == []


@pytest.mark.asyncio
async def test_admin_test_provider_dispatches_keyless_ollama_with_server_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"ollama": adapter})
    _configure_server_runtime(
        monkeypatch,
        {
            "ollama_api": {
                "api_url": "http://127.0.0.1:11434/v1",
                "model": "qwen-local",
            }
        },
    )
    monkeypatch.setattr(service, "get_llm_provider_override", lambda _provider: None)
    monkeypatch.setattr(
        service,
        "get_override_server_fallback",
        lambda _provider: None,
        raising=False,
    )
    result = await service.test_provider(LLMProviderTestRequest(provider="ollama"))

    assert result.model == "qwen-local"
    assert len(adapter.requests) == 1
    assert adapter.requests[0]["api_key"] is None
    assert adapter.requests[0]["app_config"]["ollama_api"] == {
        "api_url": "http://127.0.0.1:11434/v1",
        "model": "qwen-local",
        "api_timeout": 10,
    }


@pytest.mark.asyncio
async def test_admin_test_provider_keeps_override_credentials_atomic_at_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-override-admin-boundary-secret"
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {
            "openai_api": {
                "api_base_url": "https://configured.openai.example/v1",
                "model": "gpt-base",
            }
        },
    )
    override = LLMProviderOverride(
        provider="openai",
        config={"default_model": "gpt-override-default"},
        api_key=secret,
        credential_fields={"org_id": "org-override"},
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": override}
    )
    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        lambda _provider, **_kwargs: None,
    )

    try:
        result = await service.test_provider(LLMProviderTestRequest(provider="openai"))
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert result.model == "gpt-override-default"
    assert len(adapter.requests) == 1
    assert adapter.requests[0]["api_key"] == secret
    assert adapter.requests[0]["app_config"]["openai_api"] == {
        "api_base_url": "https://configured.openai.example/v1",
        "model": "gpt-override-default",
        "org_id": "org-override",
    }
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
async def test_admin_test_provider_ignores_non_string_override_model_and_uses_server_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"openai_api": {"model": "gpt-server-fallback"}},
    )
    override = LLMProviderOverride(
        provider="openai",
        config={"default_model": {"secret": "not-a-model"}},
        api_key="sk-malformed-model-test",
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": override}
    )
    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        lambda _provider, **_kwargs: None,
    )

    try:
        result = await service.test_provider(LLMProviderTestRequest(provider="openai"))
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert result.model == "gpt-server-fallback"
    assert adapter.requests[0]["model"] == "gpt-server-fallback"


@pytest.mark.asyncio
async def test_admin_test_provider_use_override_false_keeps_server_model_source_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"openai_api": {"model": "gpt-server-only"}},
    )
    monkeypatch.setattr(
        service,
        "get_override_default_model",
        lambda _provider: pytest.fail("use_override=False must not read an override model"),
    )
    monkeypatch.setattr(
        service,
        "get_override_server_fallback",
        lambda _provider: pytest.fail("use_override=False must not read override credentials"),
    )
    result = await service.test_provider(
        LLMProviderTestRequest(
            provider="openai",
            api_key="sk-explicit-isolated",
            use_override=False,
        )
    )

    assert result.model == "gpt-server-only"
    assert adapter.requests[0]["api_key"] == "sk-explicit-isolated"
    assert adapter.requests[0]["model"] == "gpt-server-only"


@pytest.mark.asyncio
async def test_admin_test_provider_use_override_false_skips_override_store_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"openai_api": {"model": "gpt-server-only"}},
    )

    async def unexpected_refresh(*, force: bool = False) -> None:
        del force
        pytest.fail("use_override=False must not depend on the override store")

    monkeypatch.setattr(service, "refresh_llm_provider_overrides", unexpected_refresh)

    result = await service.test_provider(
        LLMProviderTestRequest(
            provider="openai",
            api_key="sk-explicit-isolated",
            use_override=False,
        )
    )

    assert result.model == "gpt-server-only"
    assert adapter.requests[0]["api_key"] == "sk-explicit-isolated"


@pytest.mark.asyncio
async def test_admin_test_provider_use_override_false_skips_override_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(monkeypatch, {"openai_api": {}})
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "gpt-env-server-default")
    monkeypatch.setattr(
        byok_testing,
        "resolve_provider_model",
        lambda _provider, _app_config: None,
    )
    monkeypatch.setattr(
        service,
        "get_override_default_model",
        lambda _provider: pytest.fail("use_override=False must not read an override model"),
    )
    monkeypatch.setattr(
        llm_provider_overrides,
        "get_override_default_model",
        lambda _provider: "gpt-hidden-override",
    )
    monkeypatch.setattr(
        llm_provider_overrides,
        "get_llm_provider_override",
        lambda _provider: pytest.fail("use_override=False must not read override models"),
    )

    result = await service.test_provider(
        LLMProviderTestRequest(
            provider="openai",
            api_key="sk-explicit-isolated",
            use_override=False,
        )
    )

    assert result.model == "gpt-env-server-default"
    assert adapter.requests[0]["model"] == "gpt-env-server-default"


@pytest.mark.asyncio
async def test_admin_test_provider_use_override_false_omitted_key_uses_server_sources_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_secret = "sk-server-only-boundary-secret"
    override_secret = "sk-conflicting-override-secret"
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {
            "openai_api": {
                "api_key": server_secret,
                "api_base_url": "https://server-only.example/v1",
                "model": "gpt-server-only",
                "org_id": "org-server-only",
            }
        },
    )
    override = LLMProviderOverride(
        provider="openai",
        config={"default_model": "gpt-override"},
        api_key=override_secret,
        credential_fields={
            "base_url": "https://override.example/v1",
            "org_id": "org-override",
        },
    )
    override_reads = 0

    def read_override(_provider: str) -> LLMProviderOverride:
        nonlocal override_reads
        override_reads += 1
        return override

    monkeypatch.setattr(
        llm_provider_overrides,
        "get_llm_provider_override",
        read_override,
    )
    monkeypatch.setattr(
        service,
        "get_override_default_model",
        lambda _provider: pytest.fail("use_override=False must not read override models"),
    )
    monkeypatch.setattr(
        service,
        "get_override_server_fallback",
        lambda _provider: pytest.fail("use_override=False must not read override credentials"),
    )

    result = await service.test_provider(
        LLMProviderTestRequest(provider="openai", use_override=False)
    )

    assert result.model == "gpt-server-only"
    assert override_reads == 0
    assert adapter.requests[0]["api_key"] == server_secret
    assert adapter.requests[0]["app_config"]["openai_api"] == {
        "api_base_url": "https://server-only.example/v1",
        "model": "gpt-server-only",
        "org_id": "org-server-only",
    }
    assert override_secret not in str(adapter.requests[0])


@pytest.mark.asyncio
async def test_admin_test_provider_uses_first_allowed_override_model_before_server_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"openai_api": {"model": "gpt-server"}},
    )
    override = LLMProviderOverride(
        provider="openai",
        allowed_models=("gpt-allowed-first", "gpt-allowed-second"),
        api_key="sk-allowed-model-secret",
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": override}
    )

    try:
        result = await service.test_provider(LLMProviderTestRequest(provider="openai"))
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert result.model == "gpt-allowed-first"
    assert adapter.requests[0]["model"] == "gpt-allowed-first"


@pytest.mark.asyncio
async def test_admin_test_provider_payload_model_beats_all_override_and_server_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"openai_api": {"model": "gpt-server"}},
    )
    override = LLMProviderOverride(
        provider="openai",
        config={"default_model": "gpt-override-default"},
        allowed_models=("gpt-allowed-first", "gpt-payload"),
        api_key="sk-payload-precedence-secret",
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": override}
    )

    try:
        result = await service.test_provider(
            LLMProviderTestRequest(provider="openai", model="gpt-payload")
        )
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert result.model == "gpt-payload"
    assert adapter.requests[0]["model"] == "gpt-payload"


@pytest.mark.asyncio
async def test_admin_health_uses_real_env_resolver_and_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-health-env-boundary-secret"
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"openai_api": {"api_key": secret, "model": "gpt-health-model"}},
    )

    async def configured_names() -> list[str]:
        return ["openai"]

    monkeypatch.setattr(
        service,
        "get_llm_provider_overrides_snapshot",
        lambda: {},
    )
    monkeypatch.setattr(
        service,
        "get_override_server_fallback",
        lambda _provider: None,
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "get_override_default_model",
        lambda _provider: None,
        raising=False,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        configured_names,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )

    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert result["healthy"] == 1
    assert result["providers"][0]["model"] == "gpt-health-model"
    assert adapter.requests[0]["api_key"] == secret
    assert secret not in str(result)


def _resolved_for_batch(provider: str) -> ResolvedByokCredentials:
    api_key = None if provider == "ollama" else f"sk-{provider}-batch-secret"
    section = "ollama_api" if provider == "ollama" else f"{provider}_api"
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config={section: {"model": f"{provider}-model"}},
        credential_fields={},
        source="server_default" if api_key else "none",
        allowlisted=False,
    )


def _install_batch_service_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refresh_calls: list[bool | None],
) -> None:
    async def refresh(*, force: bool | None = None) -> None:
        refresh_calls.append(force)

    async def resolve(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        return _resolved_for_batch(provider)

    async def configured_names() -> list[str]:
        return [
            "oai",
            "openai",
            "ollama",
            "voyage",
            "elevenlabs",
            "unknown-provider",
        ]

    override = LLMProviderOverride(
        provider="groq",
        api_key="sk-groq-override-secret",
        api_key_hint="sk-g...cret",
    )
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", refresh)
    monkeypatch.setattr(
        service,
        "get_override_default_model",
        lambda _provider: None,
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "get_llm_provider_overrides_snapshot",
        lambda: {"groq": override},
    )
    monkeypatch.setattr(
        service,
        "get_llm_provider_override",
        lambda provider: override if provider == "groq" else None,
    )
    monkeypatch.setattr(service, "resolve_byok_credentials", resolve, raising=False)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        configured_names,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )


def _install_real_health_chat_capacity_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    health_adapter: _AlwaysBlockingAdapter,
    chat_adapter: _CapturingAdapter,
    pool: BoundedDaemonPool,
    admission: threading.BoundedSemaphore,
    timeout_seconds: float,
) -> list[bool | None]:
    """Install real Admin health and Chat adapters over one shared pool."""
    refresh_calls: list[bool | None] = []
    _install_batch_service_boundary(monkeypatch, refresh_calls=refresh_calls)
    _install_adapter_boundary(
        monkeypatch,
        {
            "openai": health_adapter,
            "ollama": health_adapter,
            "groq": health_adapter,
        },
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        admission,
        raising=False,
    )
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: _Registry({"openai": chat_adapter}),
    )
    monkeypatch.setattr(chat_service, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        admin_llm_providers,
        "_provider_health_timeout_seconds",
        lambda: timeout_seconds,
    )
    return refresh_calls


async def _ordinary_chat_dispatch() -> dict[str, Any]:
    resolved_fields = await resolved_request_fields_async(
        "openai",
        api_key="sk-ordinary-chat",
        app_config={
            "openai_api": {
                "api_base_url": "https://ordinary-chat.example/v1",
            }
        },
        model="gpt-chat-headroom",
    )
    return await chat_service.perform_chat_api_call_async(
        api_endpoint="openai",
        messages_payload=[],
        model="gpt-chat-headroom",
        streaming=False,
        **resolved_fields,
    )


@pytest.mark.asyncio
async def test_admin_health_batch_shares_one_refresh_and_dispatches_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_calls: list[bool | None] = []
    _install_batch_service_boundary(monkeypatch, refresh_calls=refresh_calls)
    entered: set[str] = set()
    all_entered = asyncio.Event()
    release = asyncio.Event()

    async def test_credentials(**kwargs: Any) -> str:
        entered.add(kwargs["provider"])
        if entered == {"openai", "ollama", "groq"}:
            all_entered.set()
        await release.wait()
        return f"{kwargs['provider']}-model"

    monkeypatch.setattr(service, "test_provider_credentials", test_credentials)
    task = asyncio.create_task(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
    )
    try:
        await asyncio.wait_for(all_entered.wait(), timeout=1.0)
        assert not task.done()
        release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert refresh_calls == [False]
    assert entered == {"openai", "ollama", "groq"}
    assert result["total"] == 3
    assert result["healthy"] == 3


@pytest.mark.asyncio
async def test_admin_health_batch_cancellation_reaches_every_inflight_provider_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_calls: list[bool | None] = []
    _install_batch_service_boundary(monkeypatch, refresh_calls=refresh_calls)
    entered: set[str] = set()
    cancelled: set[str] = set()
    all_entered = asyncio.Event()
    block = asyncio.Event()

    async def test_credentials(**kwargs: Any) -> str:
        provider = kwargs["provider"]
        entered.add(provider)
        if entered == {"openai", "ollama", "groq"}:
            all_entered.set()
        try:
            await block.wait()
        except asyncio.CancelledError:
            cancelled.add(provider)
            raise
        return f"{provider}-model"

    monkeypatch.setattr(service, "test_provider_credentials", test_credentials)
    task = asyncio.create_task(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
    )
    await asyncio.wait_for(all_entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert refresh_calls == [False]
    assert cancelled == {"openai", "ollama", "groq"}


@pytest.mark.asyncio
async def test_admin_health_multi_provider_timeouts_retain_real_worker_admission_and_chat_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health_adapter = _AlwaysBlockingAdapter()
    chat_adapter = _CapturingAdapter()
    pool = BoundedDaemonPool(2)
    admission = threading.BoundedSemaphore(1)
    refresh_calls = _install_real_health_chat_capacity_boundary(
        monkeypatch,
        health_adapter=health_adapter,
        chat_adapter=chat_adapter,
        pool=pool,
        admission=admission,
        timeout_seconds=0.05,
    )

    task = asyncio.create_task(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
    )
    try:
        first = await asyncio.wait_for(task, timeout=1.0)
        assert first["total"] == 3
        assert first["healthy"] == 0
        assert first["unhealthy"] == 3
        assert health_adapter.call_count == 1
        assert health_adapter.max_active == 1
        assert pool.active_count == 1

        chat_result = await asyncio.wait_for(_ordinary_chat_dispatch(), timeout=1.0)
        assert chat_result["choices"][0]["message"]["content"] == "pong"
        assert len(chat_adapter.requests) == 1
        assert pool.active_count == 1
    finally:
        health_adapter.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _wait_for_thread_state(lambda: pool.active_count == 0)

    recovery = await asyncio.wait_for(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        ),
        timeout=1.0,
    )

    assert recovery["healthy"] == 3
    assert health_adapter.call_count == 4
    assert health_adapter.max_active == 1
    assert pool.active_count == 0
    assert refresh_calls == [False, False]


@pytest.mark.asyncio
async def test_cancelled_admin_health_retains_real_worker_admission_and_chat_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health_adapter = _AlwaysBlockingAdapter()
    chat_adapter = _CapturingAdapter()
    pool = BoundedDaemonPool(2)
    admission = threading.BoundedSemaphore(1)
    refresh_calls = _install_real_health_chat_capacity_boundary(
        monkeypatch,
        health_adapter=health_adapter,
        chat_adapter=chat_adapter,
        pool=pool,
        admission=admission,
        timeout_seconds=5.0,
    )
    task = asyncio.create_task(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
    )

    try:
        assert await asyncio.to_thread(health_adapter.first_entered.wait, 1.0)
        second_entered_before_cancel = await asyncio.to_thread(
            health_adapter.second_entered.wait,
            0.1,
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert second_entered_before_cancel is False
        assert health_adapter.call_count == 1
        assert health_adapter.max_active == 1
        assert pool.active_count == 1
        chat_result = await asyncio.wait_for(_ordinary_chat_dispatch(), timeout=1.0)
        assert chat_result["choices"][0]["message"]["content"] == "pong"
        assert len(chat_adapter.requests) == 1
        assert pool.active_count == 1
    finally:
        health_adapter.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _wait_for_thread_state(lambda: pool.active_count == 0)

    recovery = await asyncio.wait_for(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        ),
        timeout=1.0,
    )

    assert recovery["healthy"] == 3
    assert health_adapter.call_count == 4
    assert health_adapter.max_active == 1
    assert pool.active_count == 0
    assert refresh_calls == [False, False]


@pytest.mark.asyncio
async def test_individual_admin_provider_timeout_stays_prompt_but_retains_worker_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _AlwaysBlockingAdapter()
    pool = BoundedDaemonPool(2)
    admission = threading.BoundedSemaphore(1)
    _install_adapter_boundary(monkeypatch, {"ollama": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"ollama_api": {"api_url": "http://127.0.0.1:11434/v1", "model": "qwen-health"}},
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        admission,
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "PROVIDER_CREDENTIAL_VALIDATION_TIMEOUT_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    task = asyncio.create_task(
        admin_llm_providers.admin_test_llm_provider(
            LLMProviderTestRequest(provider="ollama"),
            admin_llm_providers_service=service,
        )
    )

    try:
        assert await asyncio.to_thread(adapter.first_entered.wait, 1.0)
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(task, timeout=1.0)
        assert exc_info.value.status_code == 502
        assert pool.active_count == 1

        admission_was_retained = not admission.acquire(blocking=False)
        if not admission_was_retained:
            admission.release()
        assert admission_was_retained is True
    finally:
        adapter.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _wait_for_thread_state(lambda: pool.active_count == 0)

    assert admission.acquire(blocking=False)
    admission.release()
    assert adapter.call_count == 1
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_admin_test_endpoint_detaches_secret_bearing_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-provider-error-must-not-escape"

    async def fail_provider_test(**_kwargs: Any) -> str:
        raise ChatProviderError(
            message=f"upstream rejected {secret} at /private/provider.json",
            status_code=418,
            provider="openai",
            details={"authorization": secret},
        )

    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_llm_providers.admin_test_llm_provider(
            LLMProviderTestRequest(
                provider="openai",
                model="gpt-explicit",
                api_key="sk-explicit-admin-test",
                use_override=False,
            ),
            admin_llm_providers_service=service,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in str(exc_info.value)
    assert "/private/provider.json" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_factory", "expected_status", "expected_detail"),
    [
        (
            lambda sentinel: ChatBadRequestError(
                message=sentinel,
                provider="openai",
            ),
            400,
            "The selected provider configuration is invalid.",
        ),
        (
            lambda sentinel: ChatAuthenticationError(
                message=sentinel,
                provider="openai",
                status_code=401,
            ),
            502,
            "The selected provider credentials could not be authenticated.",
        ),
        (
            lambda sentinel: ChatAuthenticationError(
                message=sentinel,
                provider="openai",
                status_code=403,
            ),
            502,
            "The selected provider credentials could not be authenticated.",
        ),
        (
            lambda sentinel: ChatRateLimitError(
                message=sentinel,
                provider="openai",
            ),
            429,
            "The chat service provider is currently unavailable.",
        ),
        (
            lambda sentinel: ChatConfigurationError(
                message=sentinel,
                provider="openai",
            ),
            500,
            "The selected provider configuration is invalid.",
        ),
        (
            lambda sentinel: ChatProviderError(
                message=sentinel,
                provider="openai",
                status_code=503,
            ),
            503,
            "The chat service provider is currently unavailable.",
        ),
        (
            lambda sentinel: ChatProviderError(
                message=sentinel,
                provider="openai",
                status_code=504,
            ),
            504,
            "The chat service provider is currently unavailable.",
        ),
    ],
)
async def test_admin_test_endpoint_maps_canonical_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_factory,
    expected_status: int,
    expected_detail: str,
) -> None:
    sentinel = f"sk-admin-endpoint-{expected_status}-/private/provider-{expected_status}.json"

    async def fail_provider_test(**_kwargs: Any) -> str:
        raise error_factory(sentinel)

    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_llm_providers.admin_test_llm_provider(
            LLMProviderTestRequest(
                provider="openai",
                model="gpt-explicit",
                api_key="sk-explicit-admin-test",
                use_override=False,
            ),
            admin_llm_providers_service=service,
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_concurrent_admin_provider_auth_failures_are_detached_upstream_502s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = {
        "model-auth-401": "admin-provider-401-secret-/private/admin-401.json",
        "model-auth-403": "admin-provider-403-secret-/private/admin-403.json",
    }
    statuses = {"model-auth-401": 401, "model-auth-403": 403}
    entered: set[str] = set()
    all_entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_provider_test(**kwargs: Any) -> str:
        model = kwargs["model"]
        entered.add(model)
        if entered == set(sentinels):
            all_entered.set()
        await release.wait()
        raise ChatAuthenticationError(
            message=sentinels[model],
            provider="openai",
            status_code=statuses[model],
        )

    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )

    async def invoke(model: str):
        return await admin_llm_providers.admin_test_llm_provider(
            LLMProviderTestRequest(
                provider="openai",
                model=model,
                api_key=f"sk-{model}",
                use_override=False,
            ),
            admin_llm_providers_service=service,
        )

    tasks = [asyncio.create_task(invoke(model)) for model in sentinels]
    try:
        await asyncio.wait_for(all_entered.wait(), timeout=1.0)
    finally:
        release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert entered == set(sentinels)
    for result in results:
        assert isinstance(result, HTTPException)
        assert result.status_code == 502
        assert result.detail == (
            "The selected provider credentials could not be authenticated."
        )
        assert result.__cause__ is None
        assert result.__context__ is None
        assert all(sentinel not in repr(result) for sentinel in sentinels.values())


@pytest.mark.asyncio
async def test_admin_health_sanitizes_detached_secret_bearing_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-health-error-must-not-escape"
    refresh_calls: list[bool | None] = []
    _install_batch_service_boundary(monkeypatch, refresh_calls=refresh_calls)

    async def fail_provider_test(**_kwargs: Any) -> str:
        raise ChatProviderError(
            message=f"provider response contained {secret}",
            status_code=502,
            provider="openai",
            details={"path": "/private/health-provider.json"},
        )

    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)

    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert refresh_calls == [False]
    assert result["healthy"] == 0
    assert result["unhealthy"] == 3
    assert all(item["error"] == "Provider health check failed" for item in result["providers"])
    assert secret not in str(result)
    assert "/private/health-provider.json" not in str(result)


@pytest.mark.asyncio
async def test_admin_health_provider_owned_cancellation_is_one_sanitized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "provider-health-cancel-secret-/private/adapter.key"
    adapter = _ProviderCancellingAdapter(sentinel)
    pool = BoundedDaemonPool(1)
    _install_adapter_boundary(monkeypatch, {"ollama": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"ollama_api": {"api_url": "http://127.0.0.1:11434/v1", "model": "qwen-health"}},
    )
    monkeypatch.setattr(
        byok_testing,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    monkeypatch.setattr(service, "get_override_server_fallback", lambda _provider: None)
    monkeypatch.setattr(service, "get_llm_provider_overrides_snapshot", lambda: {})

    async def configured_names() -> list[str]:
        return ["ollama"]

    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        configured_names,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_provider_health_timeout_seconds",
        lambda: 1.0,
        raising=False,
    )

    current_task = asyncio.current_task()
    assert current_task is not None
    assert current_task.cancelling() == 0
    with pytest.raises(HTTPException) as exc_info:
        await service.test_provider(
            LLMProviderTestRequest(provider="ollama"),
            refresh_overrides=False,
            timeout_seconds=1.0,
        )

    assert current_task.cancelling() == 0
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)

    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert current_task.cancelling() == 0
    assert result == {
        "providers": [
            {
                "provider": "ollama",
                "status": "error",
                "latency_ms": result["providers"][0]["latency_ms"],
                "error": "Provider health check failed",
            }
        ],
        "total": 1,
        "healthy": 0,
        "unhealthy": 1,
    }
    assert sentinel not in str(result)
    assert adapter.timeouts == [1.0, 1.0]
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _BlockingAdapter()
    pool = BoundedDaemonPool(1)
    _install_adapter_boundary(monkeypatch, {"ollama": adapter})
    _configure_server_runtime(
        monkeypatch,
        {
            "ollama_api": {
                "api_url": "http://127.0.0.1:11434/v1",
                "model": "qwen-health",
                "api_timeout": 300,
            }
        },
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(service, "get_override_server_fallback", lambda _provider: None)
    async def configured_names() -> list[str]:
        return ["ollama"]

    monkeypatch.setattr(service, "get_llm_provider_overrides_snapshot", lambda: {})
    monkeypatch.setattr(admin_llm_providers, "_get_configured_provider_names", configured_names)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_provider_health_timeout_seconds",
        lambda: 0.05,
        raising=False,
    )

    try:
        first = await admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
        assert first["providers"][0]["status"] == "error"
        assert pool.active_count == 1
        assert adapter.call_count == 1
        assert adapter.timeouts == [0.05]
        assert adapter.requests[0]["app_config"]["ollama_api"]["api_timeout"] == 1

        second = await admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
        assert second["providers"][0]["status"] == "error"
        assert adapter.call_count == 1

        adapter.release.set()
        await _wait_for_thread_state(lambda: pool.active_count == 0)
        third = await admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
        assert third["providers"][0]["status"] == "healthy"
        assert adapter.call_count == 2
    finally:
        adapter.release.set()


@pytest.mark.asyncio
async def test_admin_health_cancellation_detaches_blocking_adapter_and_recovers_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _BlockingAdapter()
    pool = BoundedDaemonPool(1)
    _install_adapter_boundary(monkeypatch, {"ollama": adapter})
    _configure_server_runtime(
        monkeypatch,
        {"ollama_api": {"api_url": "http://127.0.0.1:11434/v1", "model": "qwen-health"}},
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(service, "get_override_server_fallback", lambda _provider: None)
    async def configured_names() -> list[str]:
        return ["ollama"]

    monkeypatch.setattr(service, "get_llm_provider_overrides_snapshot", lambda: {})
    monkeypatch.setattr(admin_llm_providers, "_get_configured_provider_names", configured_names)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_provider_health_timeout_seconds",
        lambda: 5.0,
        raising=False,
    )

    task = asyncio.create_task(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
    )
    try:
        await _wait_for_thread_state(adapter.entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.active_count == 1

        adapter.release.set()
        await _wait_for_thread_state(lambda: pool.active_count == 0)
        result = await admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
        assert result["providers"][0]["status"] == "healthy"
        assert adapter.call_count == 2
    finally:
        adapter.release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_admin_test_provider_projects_real_override_snapshot_to_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admin probe uses the real frozen override/static adapter boundary."""
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(
        service,
        "load_server_config_snapshot",
        lambda: {
            "openai_api": {
                "api_key": "static-key-must-not-win",
                "api_base_url": "https://static-admin.example/v1",
                "model": "gpt-static-admin",
                "timeout": 23,
            }
        },
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["gpt-override-admin"],
                config={
                    "default_model": "gpt-override-admin",
                    "api_base_url": "https://override-admin.example/v1",
                },
                api_key="override-admin-key",
                credential_fields={"org_id": "org-override-admin"},
            )
        }
    )
    try:
        result = await service.test_provider(
            LLMProviderTestRequest(provider="openai")
        )

        assert result.model == "gpt-override-admin"
        assert adapter.requests == [
            {
                "messages": [{"role": "user", "content": "ping"}],
                "system_message": None,
                "model": "gpt-override-admin",
                "api_key": "override-admin-key",
                "temperature": 0.0,
                "max_tokens": 1,
                "app_config": {
                    "openai_api": {
                        "api_base_url": "https://override-admin.example/v1",
                        "model": "gpt-override-admin",
                        "timeout": 23,
                        "org_id": "org-override-admin",
                    }
                },
                "credentials_resolved": True,
            }
        ]
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
async def test_admin_test_provider_fails_closed_if_override_store_degrades_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured override cannot dispatch after its backing store turns unhealthy."""
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(
        service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"model": "gpt-static"}},
    )
    override = LLMProviderOverride(
        provider="openai",
        config={"default_model": "gpt-override"},
        api_key="override-key",
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": override}
    )
    original_resolve = service.resolve_byok_credentials

    async def degrade_after_resolution(*args: Any, **kwargs: Any):
        resolution = await original_resolve(*args, **kwargs)
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            {"openai": override},
            healthy=False,
        )
        return resolution

    monkeypatch.setattr(service, "resolve_byok_credentials", degrade_after_resolution)
    try:
        with pytest.raises(HTTPException) as exc_info:
            await service.test_provider(LLMProviderTestRequest(provider="openai"))

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == (
            "Provider credential storage is temporarily unavailable"
        )
        assert adapter.requests == []
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_admin_test_provider_keeps_model_policy_and_credentials_on_one_override_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotation between model selection and fallback cannot mix generations."""
    adapter = _CapturingAdapter()
    _install_adapter_boundary(monkeypatch, {"openai": adapter})
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(
        service,
        "load_server_config_snapshot",
        lambda: {
            "openai_api": {
                "api_key": "static-key",
                "model": "gpt-static",
            }
        },
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["gpt-generation-a"],
                config={
                    "default_model": "gpt-generation-a",
                    "api_base_url": "https://generation-a.example/v1",
                },
                api_key="generation-a-key",
            )
        }
    )
    model_selected = threading.Event()
    release_model = threading.Event()
    original_model_resolver = service._override_test_model

    def gated_model_resolver(*args: Any, **kwargs: Any) -> str | None:
        model = original_model_resolver(*args, **kwargs)
        model_selected.set()
        if not release_model.wait(10):
            raise TimeoutError("admin override rotation gate was not released")
        return model

    monkeypatch.setattr(service, "_override_test_model", gated_model_resolver)

    def invoke() -> Any:
        return asyncio.run(
            service.test_provider(LLMProviderTestRequest(provider="openai"))
        )

    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        assert await asyncio.to_thread(model_selected.wait, 10)
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            {
                "openai": LLMProviderOverride(
                    provider="openai",
                    allowed_models=["gpt-generation-b"],
                    config={
                        "default_model": "gpt-generation-b",
                        "api_base_url": "https://generation-b.example/v1",
                    },
                    api_key="generation-b-key",
                )
            }
        )
        release_model.set()
        result = await asyncio.wait_for(task, timeout=10)

        assert result.model == "gpt-generation-a"
        assert adapter.requests[0]["model"] == "gpt-generation-a"
        assert adapter.requests[0]["api_key"] == "generation-a-key"
        assert adapter.requests[0]["app_config"]["openai_api"]["api_base_url"] == (
            "https://generation-a.example/v1"
        )
    finally:
        release_model.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
