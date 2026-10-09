"""Commercial selections must be authorized by the resolved provider inventory."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatConfigurationError,
)
from tldw_Server_API.app.core.LLM_Calls import adapter_utils, provider_model_inventory
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import AuditedCallPolicyTransport
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter, openrouter_adapter
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


class RecordingAdapter(OpenAIAdapter):
    async_chat_is_native = True

    def __init__(self):
        self.calls = []

    def chat(self, request, **_kwargs):
        self.calls.append(("chat", request))
        return {"choices": [{"message": {"content": "accepted"}}]}

    def stream(self, request, **_kwargs):
        self.calls.append(("stream", request))
        return iter(["data: accepted\n\n"])

    async def achat(self, request, **_kwargs):
        self.calls.append(("achat", request))
        return {"choices": [{"message": {"content": "accepted"}}]}

    def astream(self, request, **_kwargs):
        self.calls.append(("astream", request))

        async def frames():
            yield "data: accepted\n\n"

        return frames()


@pytest.fixture
def boundary(monkeypatch):
    for name in (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    adapter = RecordingAdapter()
    registry = SimpleNamespace(
        get_adapter=lambda _provider: adapter,
        get_audited_call_policy_transport=lambda _provider: AuditedCallPolicyTransport(
            maximum_transport_attempts=1,
            enforces_configured_endpoint_scope=True,
            enforces_maximum_timeout=True,
        ),
    )
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    calls = []
    result = SimpleNamespace(status="ready", models=("current-model",))

    def discover(provider, api_key, *, base_url=None, force_refresh=False, fetch_fn=None, call_policy=None):
        calls.append((provider, api_key, base_url, force_refresh, threading.get_ident()))
        return result

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: ["retired-model"])
    monkeypatch.setattr(chat_service, "_configured_models_for_provider", lambda _provider: ("retired-model",))
    return SimpleNamespace(adapter=adapter, discovery=calls, result=result)


def args(**overrides):
    fields = {
        "api_provider": "openai",
        "model": "retired-model",
        "api_key": "fake-request-key",
        "app_config": {},
        "messages": [{"role": "user", "content": "hello"}],
    }
    fields.update(overrides)
    return fields


@pytest.mark.parametrize("stream", [False, True])
def test_sync_retired_selection_never_reaches_generation(boundary, stream):
    with pytest.raises(ChatBadRequestError):
        chat_service.perform_chat_api_call(**args(stream=stream))
    assert boundary.adapter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_retired_selection_never_reaches_generation(boundary, stream):
    with pytest.raises(ChatBadRequestError):
        await chat_service.perform_chat_api_call_async(**args(stream=stream))
    assert boundary.adapter.calls == []


@pytest.mark.parametrize("model", ["current-model-20260101", "CURRENT-MODEL", "vendor/current-model"])
def test_cloud_ids_are_not_dated_case_or_namespace_aliases(boundary, model):
    with pytest.raises(ChatBadRequestError):
        chat_service.perform_chat_api_call(**args(model=model))
    assert boundary.adapter.calls == []


@pytest.mark.parametrize("model", ["current-model", "vendor/current-model-20260101"])
def test_openrouter_rejects_aliases_and_pricing_only_ids(boundary, model):
    boundary.result.models = ("vendor/current-model",)
    with pytest.raises(ChatBadRequestError):
        chat_service.perform_chat_api_call(**args(api_provider="openrouter", model=model))
    assert boundary.adapter.calls == []


def test_openrouter_accepts_exact_current_namespaced_id(boundary):
    boundary.result.models = ("vendor/current-model",)
    chat_service.perform_chat_api_call(**args(api_provider="openrouter", model="vendor/current-model"))
    assert boundary.adapter.calls[0][1]["model"] == "vendor/current-model"


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_openrouter_retired_fallback_never_reaches_generation(boundary, async_call, stream):
    fields = args(
        api_provider="openrouter",
        model="current-model",
        stream=stream,
        extra_body={"models": ["current-model", "retired-model"]},
    )
    with pytest.raises(ChatBadRequestError):
        if async_call:
            asyncio.run(chat_service.perform_chat_api_call_async(**fields))
        else:
            chat_service.perform_chat_api_call(**fields)
    assert boundary.adapter.calls == []


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "models", [None, "current-model", [], [""], [" "], [None], [1], [{}], {"model": "current-model"}]
)
def test_openrouter_malformed_fallbacks_never_reach_generation(boundary, async_call, models):
    fields = args(api_provider="openrouter", model="current-model", extra_body={"models": models})
    with pytest.raises(ChatBadRequestError):
        if async_call:
            asyncio.run(chat_service.perform_chat_api_call_async(**fields))
        else:
            chat_service.perform_chat_api_call(**fields)
    assert boundary.adapter.calls == []


@pytest.mark.parametrize("async_call", [False, True])
def test_openrouter_current_fallbacks_match_actual_adapter_payload(boundary, monkeypatch, async_call):
    inventory_calls = []
    monkeypatch.setattr(chat_service, "discover_provider_models", provider_model_inventory.discover_provider_models)

    def fetch_inventory(**kwargs):
        inventory_calls.append(kwargs)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{"id": "current-model"}, {"id": "vendor/fallback-model"}]},
            close=lambda: None,
        )

    monkeypatch.setattr(provider_model_inventory, "_http_fetch", fetch_inventory)
    adapter = openrouter_adapter.OpenRouterAdapter()
    monkeypatch.setattr(
        chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: adapter)
    )
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENROUTER", "1")
    posts = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            posts.append((url, kwargs))
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "accepted"}}]},
            )

    monkeypatch.setattr(openrouter_adapter, "http_client_factory", lambda **_kwargs: Client())
    fields = args(
        api_provider="openrouter",
        model="current-model",
        api_key=f"fake-fallback-key-{async_call}",
        extra_body={"models": ["current-model", "vendor/fallback-model"], "provider": {"allow_fallbacks": True}},
    )
    if async_call:
        asyncio.run(chat_service.perform_chat_api_call_async(**fields))
    else:
        chat_service.perform_chat_api_call(**fields)
    assert len(inventory_calls) == 1
    assert inventory_calls[0]["url"] == "https://openrouter.ai/api/v1/models/user"
    assert inventory_calls[0]["headers"]["Authorization"] == f"Bearer fake-fallback-key-{async_call}"
    assert len(posts) == 1
    assert posts[0][1]["json"]["model"] == "current-model"
    assert posts[0][1]["json"]["models"] == ["current-model", "vendor/fallback-model"]
    assert posts[0][1]["json"]["provider"] == {"allow_fallbacks": True}


def test_openrouter_dispatch_snapshots_validated_fallback_candidates(boundary, monkeypatch):
    models = ["current-model"]

    def discover(_provider, _key, *, base_url):
        models.append("retired-model")
        return boundary.result

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    chat_service.perform_chat_api_call(
        **args(
            api_provider="openrouter",
            model="current-model",
            extra_body={"models": models},
        )
    )
    assert models == ["current-model", "retired-model"]
    assert boundary.adapter.calls[0][1]["extra_body"]["models"] == ["current-model"]


@pytest.mark.parametrize("adapter_type", [OpenAIAdapter, openrouter_adapter.OpenRouterAdapter])
@pytest.mark.parametrize("extra_model", ["retired-model", None, {"not": "a model"}])
def test_real_adapter_preserves_primary_model_and_bound_auth_with_shadowed_extras(
    boundary,
    monkeypatch,
    adapter_type,
    extra_model,
):
    adapter = adapter_type()
    provider = adapter.name
    monkeypatch.setattr(
        chat_service, "_get_llm_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: adapter)
    )
    monkeypatch.setenv(f"LLM_ADAPTERS_NATIVE_HTTP_{provider.upper()}", "1")
    posts = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            posts.append(kwargs)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "accepted"}}]},
            )

    module = openrouter_adapter if provider == "openrouter" else openai_adapter
    monkeypatch.setattr(module, "http_client_factory", lambda **_kwargs: Client())
    chat_service.perform_chat_api_call(
        **args(
            api_provider=provider,
            model="current-model",
            api_key="fake-bound-key",
            extra_body={"model": extra_model, "safe_extension": True},
            extra_headers={
                "authorization": "Bearer fake-attacker-key",
                "X-API-Key": "fake-attacker-key",
                "X-Goog-Api-Key": "fake-attacker-key",
                "Cookie": "fake-attacker-cookie",
                "X-Safe-Extension": "kept",
            },
        )
    )
    assert len(posts) == 1
    assert posts[0]["json"]["model"] == "current-model"
    assert posts[0]["json"]["safe_extension"] is True
    headers = {key.lower(): value for key, value in posts[0]["headers"].items()}
    assert headers["authorization"] == "Bearer fake-bound-key"
    assert not {"x-api-key", "x-goog-api-key", "cookie"}.intersection(headers)
    assert headers["x-safe-extension"] == "kept"


@pytest.mark.parametrize("status", ["unavailable", "unsupported", "authentication_failed"])
def test_inventory_failure_cannot_authorize_models(boundary, status):
    boundary.result.status = status
    with pytest.raises(ChatConfigurationError):
        chat_service.perform_chat_api_call(**args(model="current-model"))
    assert boundary.adapter.calls == []


def test_authoritative_empty_inventory_rejects_selection_as_bad_request(boundary):
    boundary.result.models = ()
    with pytest.raises(ChatBadRequestError):
        chat_service.perform_chat_api_call(**args(model="retired-model"))
    assert boundary.adapter.calls == []


def test_scoped_prevalidator_reports_unavailable_inventory(boundary):
    boundary.result.status = "unreachable"
    with pytest.raises(ChatConfigurationError):
        chat_service.is_model_known_for_provider(
            "openai",
            "current-model",
            api_key="fake-request-key",
            app_config={},
            credentials_resolved=True,
        )


def test_retired_config_default_is_rechecked(boundary):
    with pytest.raises(ChatBadRequestError):
        chat_service.perform_chat_api_call(**args(model=None, app_config={"openai_api": {"model": "retired-model"}}))
    assert boundary.adapter.calls == []


def test_exact_current_selection_reaches_adapter_after_inventory(boundary):
    response = chat_service.perform_chat_api_call(**args(model="current-model"))
    assert response["choices"][0]["message"]["content"] == "accepted"
    assert boundary.discovery[0][:4] == ("openai", "fake-request-key", "https://api.openai.com/v1", False)
    assert boundary.adapter.calls[0][1]["model"] == "current-model"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_async_inventory_is_off_event_loop(boundary, stream):
    loop_thread = threading.get_ident()
    await chat_service.perform_chat_api_call_async(**args(model="current-model", stream=stream))
    assert boundary.discovery[0][4] != loop_thread
    assert boundary.adapter.calls[0][0] == ("astream" if stream else "achat")


@pytest.mark.asyncio
async def test_byok_discovery_uses_runtime_key_and_endpoint_not_server_defaults(boundary, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://server.invalid/v1")

    def reject_server(*_args, **_kwargs):
        raise AssertionError("Resolved BYOK must not consult server credentials")

    monkeypatch.setattr(adapter_utils, "resolve_provider_api_key_from_config", reject_server)

    def scoped_config(config=None):
        if config is None:
            reject_server()
        return config

    monkeypatch.setattr(adapter_utils, "ensure_app_config", scoped_config)

    async def resolver(provider, **_kwargs):
        return ResolvedByokCredentials(
            provider=provider,
            api_key="fake-byok-key",
            app_config={"openai_api": {"api_base_url": "https://tenant.invalid/v1"}},
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
        )

    runtime = ProviderCredentialRuntime(
        user_id=17,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolver,
    )
    try:
        credentials = await runtime.resolve("openai")
        await chat_service.perform_chat_api_call_async(
            **args(
                api_provider="oai",
                model="current-model",
                api_key="fake-wrong-key",
                app_config={"openai_api": {"api_base_url": "https://wrong.invalid/v1"}},
                **{PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: credentials},
            )
        )
        assert boundary.discovery[0][:3] == ("openai", "fake-byok-key", "https://tenant.invalid/v1")
        assert boundary.adapter.calls[0][1]["api_key"] == "fake-byok-key"
    finally:
        await runtime.close()


@pytest.mark.parametrize("async_call", [False, True])
def test_audited_scope_mismatch_blocks_discovery(boundary, monkeypatch, async_call):
    monkeypatch.setattr(chat_service, "discover_provider_models", provider_model_inventory.discover_provider_models)
    http_calls = []

    def reject_http(**_kwargs):
        http_calls.append(_kwargs)
        raise AssertionError("Scope mismatch must be denied before network access")

    monkeypatch.setattr(provider_model_inventory, "_http_fetch", reject_http)
    policy = ProviderCallPolicy(
        max_transport_attempts=1,
        maximum_timeout_seconds=0.1,
        required_endpoint_scope=ConfiguredEndpointScope.from_url("https://tenant.invalid/v1"),
    )
    with pytest.raises(ChatConfigurationError):
        if async_call:
            asyncio.run(chat_service.perform_chat_api_call_async(**args(model="current-model", call_policy=policy)))
        else:
            chat_service.perform_chat_api_call(**args(model="current-model", call_policy=policy))
    assert boundary.discovery == []
    assert http_calls == []
    assert boundary.adapter.calls == []


@pytest.mark.parametrize("async_call", [False, True])
def test_audited_current_model_uses_checked_discovery_transport(boundary, monkeypatch, async_call):
    monkeypatch.setattr(chat_service, "discover_provider_models", provider_model_inventory.discover_provider_models)
    http_calls = []
    closed = []
    policy = ProviderCallPolicy(
        max_transport_attempts=1,
        maximum_timeout_seconds=3,
        required_endpoint_scope=ConfiguredEndpointScope.from_url("https://policy-positive.invalid/v1"),
        privacy_safe_errors=True,
    )

    def fetch(**kwargs):
        http_calls.append(kwargs)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{"id": "current-model"}]},
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr(provider_model_inventory, "_http_fetch", fetch)
    fields = args(
        model="current-model",
        api_key=f"fake-policy-key-{async_call}",
        call_policy=policy,
        app_config={"openai_api": {"api_base_url": "https://policy-positive.invalid/v1"}},
    )
    if async_call:
        asyncio.run(chat_service.perform_chat_api_call_async(**fields))
    else:
        chat_service.perform_chat_api_call(**fields)
    assert len(http_calls) == 1
    call = http_calls[0]
    assert call["url"] == "https://policy-positive.invalid/v1/models"
    assert call["headers"]["Authorization"] == f"Bearer fake-policy-key-{async_call}"
    assert call["configured_endpoint"] is policy.required_endpoint_scope
    assert call["retry"].attempts == 1
    assert call["allow_redirects"] is False
    assert call["sensitive_observability"] is True
    assert 0 < call["timeout"] <= 3
    assert closed == [True]
    assert len(boundary.adapter.calls) == 1
    assert 0 < boundary.adapter.calls[0][1]["call_policy"].maximum_timeout_seconds <= 3


@pytest.mark.parametrize("async_call", [False, True])
def test_audited_discovery_receives_policy_and_generation_shares_deadline(boundary, monkeypatch, async_call):
    clock = [100.0]
    monkeypatch.setattr(chat_service, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    policy = ProviderCallPolicy(
        max_transport_attempts=1,
        maximum_timeout_seconds=10,
        required_endpoint_scope=ConfiguredEndpointScope.from_url("https://api.openai.com/v1"),
        privacy_safe_errors=True,
    )
    policies = []

    def discover(_provider, _key, *, base_url, call_policy):
        policies.append(call_policy)
        clock[0] += 2
        return boundary.result

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    if async_call:
        asyncio.run(chat_service.perform_chat_api_call_async(**args(model="current-model", call_policy=policy)))
    else:
        chat_service.perform_chat_api_call(**args(model="current-model", call_policy=policy))
    assert policies == [policy]
    assert boundary.adapter.calls[0][1]["call_policy"].maximum_timeout_seconds == 8
    assert policy.maximum_timeout_seconds == 10


def test_audited_discovery_over_deadline_never_generates(boundary, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(chat_service, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    policy = ProviderCallPolicy(
        max_transport_attempts=1,
        maximum_timeout_seconds=0.5,
        required_endpoint_scope=ConfiguredEndpointScope.from_url("https://api.openai.com/v1"),
    )

    def discover(_provider, _key, *, base_url, call_policy):
        clock[0] += 1
        return boundary.result

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    with pytest.raises(ChatConfigurationError):
        chat_service.perform_chat_api_call(**args(model="current-model", call_policy=policy))
    assert boundary.adapter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("queued_elapsed, remaining", [(3, 6), (9, None), (10, None)])
async def test_async_audited_deadline_includes_discovery_dispatch_elapsed(
    boundary,
    monkeypatch,
    stream,
    queued_elapsed,
    remaining,
):
    clock = [100.0]
    monkeypatch.setattr(chat_service, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    policy = ProviderCallPolicy(
        max_transport_attempts=1,
        maximum_timeout_seconds=10,
        required_endpoint_scope=ConfiguredEndpointScope.from_url("https://api.openai.com/v1"),
    )
    dispatch = chat_service.await_bounded_sync_call
    discovery_threads = []
    loop_thread = threading.get_ident()

    async def queued_dispatch(call, **kwargs):
        clock[0] += queued_elapsed
        return await dispatch(call, **kwargs)

    def discover(_provider, _key, *, base_url, call_policy):
        discovery_threads.append(threading.get_ident())
        clock[0] += 1
        return boundary.result

    monkeypatch.setattr(chat_service, "await_bounded_sync_call", queued_dispatch)
    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    fields = args(model="current-model", stream=stream, call_policy=policy)
    if remaining is None:
        with pytest.raises(ChatConfigurationError):
            await chat_service.perform_chat_api_call_async(**fields)
        assert boundary.adapter.calls == []
    else:
        await chat_service.perform_chat_api_call_async(**fields)
        assert boundary.adapter.calls[0][0] == ("astream" if stream else "achat")
        assert boundary.adapter.calls[0][1]["call_policy"].maximum_timeout_seconds == remaining
    assert len(discovery_threads) == 1
    assert discovery_threads[0] != loop_thread
    assert policy.maximum_timeout_seconds == 10


@pytest.mark.parametrize("async_call", [False, True])
def test_attempt_only_policy_allows_current_model_generation(boundary, monkeypatch, async_call):
    policies = []
    policy = ProviderCallPolicy(max_transport_attempts=1)

    def discover(_provider, _key, *, base_url, call_policy):
        policies.append(call_policy)
        return boundary.result

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    if async_call:
        asyncio.run(chat_service.perform_chat_api_call_async(**args(model="current-model", call_policy=policy)))
    else:
        chat_service.perform_chat_api_call(**args(model="current-model", call_policy=policy))
    assert policies == [policy]
    assert boundary.adapter.calls[0][1]["call_policy"] is policy


def test_local_selection_does_not_require_cloud_inventory(boundary):
    chat_service.perform_chat_api_call(**args(api_provider="ollama", model="local/custom-model", api_key=None))
    assert boundary.discovery == []
    assert boundary.adapter.calls[0][1]["model"] == "local/custom-model"


def test_cloud_known_models_exclude_pricing_and_config(boundary, monkeypatch):
    assert chat_service.known_models_for_provider_cached(
        "openai",
        api_key="fake-request-key",
        app_config={},
        credentials_resolved=True,
    ) == ("current-model",)
    assert (
        chat_service.is_model_known_for_provider(
            "openai",
            "retired-model",
            api_key="fake-request-key",
            app_config={},
            credentials_resolved=True,
        )
        is False
    )


def test_unscoped_prevalidator_defers_without_using_server_inventory(boundary):
    assert chat_service.is_model_known_for_provider("openai", "tenant-model") is None
    assert boundary.discovery == []


def test_resolved_missing_key_never_falls_back_to_server_inventory(boundary, monkeypatch):
    monkeypatch.setattr(
        adapter_utils, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: "fake-server-key"
    )
    assert (
        chat_service.is_model_known_for_provider(
            "openai",
            "current-model",
            api_key=None,
            app_config={},
            credentials_resolved=True,
        )
        is False
    )
    assert boundary.discovery == []


def test_commercial_normalization_does_not_infer_aliases_from_pricing(boundary, monkeypatch):
    monkeypatch.setattr(chat_service, "_load_alias_overrides_cached", lambda: {})
    monkeypatch.setattr(chat_service, "_load_models_with_case_cached", lambda _provider: ("CURRENT-MODEL",))
    request = SimpleNamespace(api_provider="openai", model="current-model")
    assert chat_service.normalize_request_provider_and_model(request, "openai") == "openai"
    assert request.model == "current-model"


@pytest.mark.parametrize("provider, destination", [("openai", "current-model"), ("openrouter", "vendor/current-model")])
@pytest.mark.parametrize("available", [False, True])
def test_explicit_cloud_user_alias_destination_requires_exact_inventory(
    boundary,
    monkeypatch,
    provider,
    destination,
    available,
):
    monkeypatch.setenv("CHAT_MODEL_ALIAS_OVERRIDES", json.dumps({provider: {"friendly": destination}}))
    chat_service._load_alias_overrides_cached.cache_clear()
    try:
        request = SimpleNamespace(api_provider=provider, model="friendly")
        assert chat_service.normalize_request_provider_and_model(request, provider) == provider
        assert request.model == destination
        boundary.result.models = (destination,) if available else ("other-current-model",)
        fields = args(api_provider=provider, model=request.model)
        if available:
            chat_service.perform_chat_api_call(**fields)
            assert boundary.adapter.calls[0][1]["model"] == destination
        else:
            with pytest.raises(ChatBadRequestError):
                chat_service.perform_chat_api_call(**fields)
            assert boundary.adapter.calls == []
    finally:
        chat_service._load_alias_overrides_cached.cache_clear()


def test_inventory_endpoint_remains_generation_endpoint_after_env_changes(boundary, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://before.invalid/v1")

    def discover(_provider, _key, *, base_url):
        assert base_url == "https://before.invalid/v1"
        monkeypatch.setenv("OPENAI_BASE_URL", "https://after.invalid/v1")
        return boundary.result

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    chat_service.perform_chat_api_call(**args(model="current-model"))
    request = boundary.adapter.calls[0][1]
    assert boundary.adapter._resolve_base_url(request) == "https://before.invalid/v1"


@pytest.mark.parametrize("prevalidator", [False, True])
def test_unresolved_inventory_endpoint_never_falls_back(boundary, monkeypatch, prevalidator):
    monkeypatch.setattr(chat_service, "resolve_provider_models_base_url", lambda *_args, **_kwargs: None)
    with pytest.raises(ChatConfigurationError):
        if prevalidator:
            chat_service.is_model_known_for_provider(
                "openai",
                "current-model",
                api_key="fake-request-key",
                app_config={},
                credentials_resolved=True,
            )
        else:
            chat_service.perform_chat_api_call(**args(model="current-model"))
    assert boundary.discovery == []
    assert boundary.adapter.calls == []


def test_invalid_explicit_inventory_endpoint_is_rejected_before_discovery(boundary):
    with pytest.raises(ChatConfigurationError):
        chat_service._validate_provider_model_selection(
            "openai",
            {
                "api_key": "fake-request-key",
                "model": "current-model",
                "app_config": {},
                "base_url": "https://user:secret@invalid.example/v1",
            },
        )
    assert boundary.discovery == []
    assert boundary.adapter.calls == []


def test_bedrock_without_inventory_is_configuration_failure(boundary):
    with pytest.raises(ChatConfigurationError):
        chat_service.perform_chat_api_call(**args(api_provider="bedrock", model="retired-model"))
    assert boundary.discovery == []
    assert boundary.adapter.calls == []
