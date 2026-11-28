from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_config import merge_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatConfigurationError,
)
from tldw_Server_API.app.core.Evaluations import ms_g_eval
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.helpers.gateway_routing import route_gateway_adapters
from tldw_Server_API.app.core.LLM_Calls.providers import (
    bedrock_adapter,
    custom_openai_adapter,
    openai_adapter,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    evaluation_manager as prompt_evaluation_manager,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor,
    test_runner,
)

pytestmark = pytest.mark.unit


async def _issue_credentials(
    provider: str,
    *,
    api_key: str,
    app_config: dict[str, Any],
    credential_fields: dict[str, Any] | None = None,
    source: str = "user",
) -> ProviderCallCredentials:
    """Issue one authentic provider-call capability for boundary tests."""

    async def resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=app_config,
            credential_fields=dict(credential_fields or {}),
            source=source,
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
        )

    runtime = ProviderCredentialRuntime(
        user_id=73,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolver,
    )
    try:
        return await runtime.resolve(provider, model="test-model")
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "provider",
    ["custom-openai-api-2", "bedrock"],
)
async def test_concurrent_credential_endpoints_replace_stale_server_aliases_at_transport(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """A validated credential endpoint stays atomic with its key at real adapters."""

    section = "custom_openai_api_2" if provider == "custom-openai-api-2" else "bedrock_api"
    stale_field = "api_ip" if provider == "custom-openai-api-2" else "runtime_endpoint"
    handles = []
    for label in ("a", "b"):
        credential_endpoint = f"https://credential-{label}.example/v1"
        credential_fields = {"base_url": credential_endpoint}
        app_config = merge_app_config_overrides(
            {
                section: {
                    stale_field: f"https://stale-server-{label}.example/openai",
                    "model": "meta.llama3-8b-instruct",
                }
            },
            provider,
            credential_fields,
        )
        handles.append(
            await _issue_credentials(
                provider,
                api_key=f"credential-{label}-key",
                app_config=app_config,
                credential_fields=credential_fields,
            )
        )

    both_entered = threading.Event()
    release = threading.Event()
    capture_lock = threading.Lock()
    captured: list[tuple[str, str | None, float | None]] = []

    def record(
        url: str,
        headers: dict[str, str],
        timeout: float | None,
    ) -> httpx.Response:
        with capture_lock:
            captured.append((url, headers.get("Authorization"), timeout))
            if len(captured) == 2:
                both_entered.set()
        assert release.wait(timeout=5)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    if provider == "custom-openai-api-2":
        adapter = custom_openai_adapter.CustomOpenAIAdapter2()

        def fetch(**kwargs: Any) -> httpx.Response:
            scope = kwargs["configured_endpoint"]
            assert scope.matches(kwargs["url"])
            return record(kwargs["url"], kwargs["headers"], kwargs["timeout"])

        adapter.http_fetcher = fetch
    else:
        class RecordingClient:
            def __init__(self, timeout: float | None) -> None:
                self.timeout = timeout

            def __enter__(self) -> RecordingClient:
                return self

            def __exit__(self, *_args: Any) -> bool:
                return False

            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, Any],
            ) -> httpx.Response:
                del json
                return record(url, headers, self.timeout)

        monkeypatch.setattr(
            bedrock_adapter,
            "http_client_factory",
            lambda *, timeout=None: RecordingClient(timeout),
        )
        adapter = bedrock_adapter.BedrockAdapter()

    def invoke(handle: ProviderCallCredentials, timeout: float) -> dict[str, Any]:
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "meta.llama3-8b-instruct",
                "api_key": "loose-attacker-key",
                "app_config": {section: {stale_field: "https://loose-attacker.invalid"}},
                "credentials_resolved": True,
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
            },
            timeout=timeout,
        )

    calls = [(handles[0], 11.0), (handles[1], 23.0)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke, handle, timeout) for handle, timeout in calls]
        try:
            assert both_entered.wait(timeout=5)
        finally:
            release.set()
        for future in futures:
            future.result(timeout=5)

    assert set(captured) == {
        (
            "https://credential-a.example/v1/chat/completions",
            "Bearer credential-a-key",
            11.0,
        ),
        (
            "https://credential-b.example/v1/chat/completions",
            "Bearer credential-b-key",
            23.0,
        ),
    }


_ADAPTER_BOUNDARIES = (
    ("openai_adapter", "OpenAIAdapter"),
    ("anthropic_adapter", "AnthropicAdapter"),
    ("bedrock_adapter", "BedrockAdapter"),
    ("groq_adapter", "GroqAdapter"),
    ("openrouter_adapter", "OpenRouterAdapter"),
    ("mistral_adapter", "MistralAdapter"),
    ("qwen_adapter", "QwenAdapter"),
    ("google_adapter", "GoogleAdapter"),
    ("deepseek_adapter", "DeepSeekAdapter"),
    ("huggingface_adapter", "HuggingFaceAdapter"),
    ("cohere_adapter", "CohereAdapter"),
    ("moonshot_adapter", "MoonshotAdapter"),
    ("zai_adapter", "ZaiAdapter"),
    ("mlx_provider", "MLXChatAdapter"),
    ("custom_openai_adapter", "CustomOpenAIAdapter"),
    ("custom_openai_adapter", "CustomOpenAIAdapter2"),
    ("custom_openai_adapter", "NovitaAdapter"),
    ("custom_openai_adapter", "PoeAdapter"),
    ("custom_openai_adapter", "TogetherAdapter"),
    ("local_adapters", "KoboldAdapter"),
)


@pytest.mark.parametrize("mode", ["chat", "stream"])
@pytest.mark.parametrize(
    ("module_name", "adapter_name"),
    _ADAPTER_BOUNDARIES,
    ids=[adapter_name for _module_name, adapter_name in _ADAPTER_BOUNDARIES],
)
def test_every_adapter_boundary_rejects_bare_resolved_marker_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    adapter_name: str,
    mode: str,
) -> None:
    """A boolean marker alone must never grant server-resolved credential trust."""

    module = importlib.import_module(
        f"tldw_Server_API.app.core.LLM_Calls.providers.{module_name}"
    )
    adapter = getattr(module, adapter_name)()

    if module_name == "custom_openai_adapter":
        monkeypatch.setattr(
            adapter,
            "_resolve_transport_context",
            lambda *_args, **_kwargs: pytest.fail(
                "transport resolution ran before credential authentication"
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "validate_payload",
            lambda *_args, **_kwargs: pytest.fail(
                "payload validation ran before credential authentication"
            ),
        )

    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "meta.llama3-8b-instruct" if adapter_name == "BedrockAdapter" else "test-model",
        "api_key": "loose-untrusted-key",
        "app_config": {},
        "credentials_resolved": True,
    }

    with pytest.raises(ChatConfigurationError):
        result = getattr(adapter, mode)(request)
        if mode == "stream":
            list(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic", "bedrock"])
async def test_chat_translation_forwards_genuine_cloud_handle_and_overwrites_loose_fields(
    provider: str,
) -> None:
    """Cloud adapters receive the exact capability, never loose caller credentials."""

    section = f"{provider}_api"
    durable_config = {section: {"tenant": f"{provider}-tenant"}}
    handle = await _issue_credentials(
        provider,
        api_key=f"{provider}-runtime-key",
        app_config=durable_config,
    )

    translated_provider, request, _internal = (
        gateway._build_adapter_request_from_chat_args(
            {
                "api_provider": provider,
                "messages": [{"role": "user", "content": "hello"}],
                "model": (
                    "meta.llama3-8b-instruct"
                    if provider == "bedrock"
                    else "test-model"
                ),
                "api_key": "loose-untrusted-key",
                "app_config": {section: {"tenant": "attacker"}},
                "credentials_resolved": True,
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
            }
        )
    )

    assert translated_provider == provider
    assert request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
    assert request["api_key"] == f"{provider}-runtime-key"
    assert request["app_config"] == durable_config
    assert request["credentials_resolved"] is True


def test_chat_translation_rejects_bare_resolved_marker() -> None:
    """The compatibility marker cannot bypass legacy credential resolution alone."""

    with pytest.raises(ChatConfigurationError):
        gateway._build_adapter_request_from_chat_args(
            {
                "api_provider": "openai",
                "messages": [{"role": "user", "content": "hello"}],
                "model": "test-model",
                "api_key": "loose-untrusted-key",
                "app_config": {},
                "credentials_resolved": True,
            }
        )


def test_summary_translation_rejects_bare_resolved_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarization must reject a bare marker before registry dispatch."""

    monkeypatch.setattr(
        sgl,
        "get_registry",
        lambda: pytest.fail("registry dispatch must not run for a bare marker"),
    )

    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        sgl._summarize_via_adapter(
            api_name="openai",
            text_to_summarize="source",
            custom_prompt_arg=None,
            api_key="loose-untrusted-key",
            temp=0.2,
            system_message=None,
            streaming=False,
            model_override="test-model",
            app_config={},
            credentials_resolved=True,
            raise_on_error=True,
        )

    assert exc_info.value.code == "configuration"


@pytest.mark.asyncio
async def test_summary_translation_forwards_genuine_cloud_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGL must forward cloud capabilities just as it already does local ones."""

    handle = await _issue_credentials(
        "openai",
        api_key="openai-runtime-key",
        app_config={"openai_api": {"tenant": "runtime"}},
    )
    captured: dict[str, Any] = {}

    class RecordingAdapter:
        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            captured.update(request)
            return {"choices": [{"message": {"content": "summary"}}]}

    class Registry:
        def get_adapter(self, provider: str) -> RecordingAdapter:
            assert provider == "openai"
            return RecordingAdapter()

        def is_local_provider_name(self, provider: str) -> bool:
            del provider
            return False

    monkeypatch.setattr(sgl, "get_registry", lambda: Registry())

    result = sgl._summarize_via_adapter(
        api_name="openai",
        text_to_summarize="source",
        custom_prompt_arg=None,
        api_key="loose-untrusted-key",
        temp=0.2,
        system_message=None,
        streaming=False,
        model_override="test-model",
        app_config={"openai_api": {"tenant": "attacker"}},
        credentials_resolved=True,
        provider_credentials=handle,
        raise_on_error=True,
    )

    assert result == "summary"
    assert captured[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
    assert captured["api_key"] == "openai-runtime-key"
    assert captured["app_config"] == handle.app_config


_POLICY_DENIED = "Error: The provider or model is disabled by the administrator."
_STORE_DOWN = "Error: Provider credential storage is temporarily unavailable."
_OVERRIDE_MODULE = "tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides"


def _refused_line(reason: str) -> str:
    """The one warning a refused call writes (TASK-15571, TASK-15581)."""
    return f"Provider override for openai: {reason}; the call is refused"


async def _summary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: str | None = "user",
    overrides: dict[str, Any] | None = None,
    healthy: bool = True,
    env_model: str | None = "gpt-env-default",
    config_model: str | None = None,
    model_override: str | None = None,
    server_key: str | None = "server-key",
) -> tuple[str | None, list[tuple[str, str | None, bool]], list[str]]:
    """Run one summary through the shared adapter path (TASK-15496, TASK-15571).

    ``source`` is whose key the handle carries; None sends no handle (the server-key path). The server's
    config has the same provider section as the handle's and its key is ``server_key``. Returns the result,
    (model, key, handle in the request) for each adapter request, and the override module's warnings.
    """
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        set_llm_provider_overrides_cache_for_tests,
    )

    openai_section = {"model": config_model} if config_model else {}
    handle = None
    if source is not None:
        handle = await _issue_credentials(
            "openai", api_key="handle-key", app_config={"openai_api": openai_section}, source=source
        )
    calls: list[tuple[str, str | None, bool]] = []
    warnings: list[str] = []

    class RecordingAdapter:
        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            calls.append((request["model"], request["api_key"], PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY in request))
            return {"choices": [{"message": {"content": "summary"}}]}

    class Registry:
        def get_adapter(self, provider: str) -> RecordingAdapter:
            return RecordingAdapter()

    monkeypatch.setattr(sgl, "get_registry", lambda: Registry())
    monkeypatch.setattr(sgl, "load_and_log_configs", lambda: {"openai_api": dict(openai_section)})
    monkeypatch.setattr(sgl, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: server_key)
    if env_model:
        monkeypatch.setenv("DEFAULT_MODEL_OPENAI", env_model)
    else:
        monkeypatch.delenv("DEFAULT_MODEL_OPENAI", raising=False)
    set_llm_provider_overrides_cache_for_tests(overrides, healthy=healthy)
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING", filter=_OVERRIDE_MODULE
    )
    try:
        result = sgl._summarize_via_adapter(
            api_name="openai",
            text_to_summarize="source",
            custom_prompt_arg=None,
            api_key=None,
            temp=0.2,
            system_message=None,
            streaming=False,
            model_override=model_override,
            app_config=None,
            credentials_resolved=False,
            provider_credentials=handle,
            raise_on_error=False,
        )
    finally:
        logger.remove(sink)
        set_llm_provider_overrides_cache_for_tests(None)
    return result, calls, warnings


def _allowlist(*models: str) -> dict[str, Any]:
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride

    return {"openai": LLMProviderOverride(provider="openai", allowed_models=list(models))}


# Each source of the model: the handle's config, the DEFAULT_MODEL_OPENAI fallback, the request's model_override.
_MODEL_SOURCES = pytest.mark.parametrize(
    ("config_model", "model_override", "expected"),
    [("cfg-m", None, "cfg-m"), (None, None, "gpt-env-default"), (None, "ovr-m", "ovr-m")],
    ids=["configured", "default-model-env", "model-override"],
)


@pytest.mark.asyncio
async def test_own_key_summary_falls_back_to_the_default_model_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, warnings = await _summary(monkeypatch)

    assert result == "summary"
    assert calls == [("gpt-env-default", "handle-key", True)]
    assert warnings == []


# Each refusal on every key (TASK-15581, TASK-15582), and a policy that cannot be read.
_EVERY_KEY = [
    ({"denied_models": ["cfg-m", "gpt-env-default", "ovr-m"]}, True, "model_denied", _POLICY_DENIED),
    ({"is_enabled": False}, True, "provider_disabled", _POLICY_DENIED),
    ({}, False, "credential_store_unavailable", _STORE_DOWN),
]
_REFUSAL_FIELDS = ("override_kwargs", "healthy", "reason", "refused")
_REFUSALS = pytest.mark.parametrize(
    _REFUSAL_FIELDS, _EVERY_KEY, ids=["model-denied", "provider-disabled", "override-cache-unhealthy"]
)
# A model off allowed_models is refused on every key but the user's own (TASK-15582).
_OTHER_KEY_REFUSALS = pytest.mark.parametrize(
    _REFUSAL_FIELDS,
    [({"allowed_models": ["x"]}, True, "model_not_allowed", _POLICY_DENIED), *_EVERY_KEY],
    ids=["model-not-allowed", "model-denied", "provider-disabled", "override-cache-unhealthy"],
)


async def _refused_summary(
    monkeypatch: pytest.MonkeyPatch, override_kwargs: dict[str, Any], healthy: bool, **summary_kwargs: Any
) -> tuple[str | None, list[tuple[str, str | None, bool]], list[str]]:
    """``_summary`` under one admin override for openai."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride

    overrides = {"openai": LLMProviderOverride(provider="openai", **override_kwargs)}
    return await _summary(monkeypatch, overrides=overrides, healthy=healthy, **summary_kwargs)


@pytest.mark.asyncio
@_MODEL_SOURCES
@_REFUSALS
async def test_an_admin_disabled_provider_or_model_is_refused_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch,
    override_kwargs: dict[str, Any],
    healthy: bool,
    reason: str,
    refused: str,
    config_model: str | None,
    model_override: str | None,
    expected: str,
) -> None:
    """TASK-15581, the owner's rule: whatever the model's source, a provider or model the admin disabled is refused
    with 'configuration' (as TASK-15496 did), and the reason is logged. A policy that cannot be read refuses too, as
    the retryable 'provider_failure', so a job tries again once it loads."""
    result, calls, warnings = await _refused_summary(
        monkeypatch, override_kwargs, healthy, config_model=config_model, model_override=model_override
    )

    assert (result, calls, warnings) == (refused, [], [_refused_line(reason)])


@pytest.mark.asyncio
@_OTHER_KEY_REFUSALS
@pytest.mark.parametrize(
    "source", ["team", "org", "server_default", None], ids=["team", "org", "server-default-handle", "no-handle"]
)
async def test_an_admin_disabled_provider_or_model_is_refused_on_every_other_key(
    monkeypatch: pytest.MonkeyPatch,
    override_kwargs: dict[str, Any],
    healthy: bool,
    reason: str,
    refused: str,
    source: str | None,
) -> None:
    """TASK-15581: the same on a team's or an org's key and on the server's, with a handle or none."""
    result, calls, warnings = await _refused_summary(
        monkeypatch, override_kwargs, healthy, source=source, config_model="cfg-m"
    )

    assert (result, calls, warnings) == (refused, [], [_refused_line(reason)])


@pytest.mark.parametrize(
    ("error_code", "summary_code"),
    [("provider_disabled", "configuration"), ("credential_store_unavailable", "provider_failure")],
)
def test_a_refusal_at_the_provider_base_class_is_retryable_only_for_an_unreadable_store(error_code, summary_code):
    """TASK-15581 (I2): the audio worker keeps a transcript without its analysis for good on 'configuration', so
    a store blip must come back as the retryable 'provider_failure'."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError

    error = ChatConfigurationError("refused", provider="openai", error_code=error_code)

    assert sgl._summary_error_code(error) == summary_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_model", "model_override", "expected"),
    [("cfg-m", None, "cfg-m"), (None, "ovr-m", "ovr-m")],
    ids=["configured", "model-override"],
)
async def test_an_allowed_model_stays_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch, config_model: str | None, model_override: str | None, expected: str
) -> None:
    result, calls, warnings = await _summary(
        monkeypatch, overrides=_allowlist(expected), config_model=config_model, model_override=model_override
    )

    assert result == "summary"
    assert calls == [(expected, "handle-key", True)]
    assert warnings == []


@pytest.mark.asyncio
async def test_own_key_summary_with_an_allowlist_and_no_default_model_stays_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, warnings = await _summary(monkeypatch, overrides=_allowlist("gpt-allowed"), env_model=None)

    assert calls == []
    assert result == "Error: Model is required for provider 'openai'"
    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_event_gated_cloud_calls_keep_rotated_credentials_and_timeouts_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent same-provider calls retain their exact A/B runtime snapshots."""

    handles = [
        await _issue_credentials(
            "openai",
            api_key=f"rotation-{label}-key",
            app_config={"openai_api": {"rotation": label}},
        )
        for label in ("a", "b")
    ]
    start = threading.Event()
    both_entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    captured: list[tuple[dict[str, Any], float | None]] = []

    class RecordingAdapter:
        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            with lock:
                captured.append((request, timeout))
                if len(captured) == 2:
                    both_entered.set()
            assert release.wait(timeout=5)
            return {"choices": [{"message": {"content": request["api_key"]}}]}

    class Registry:
        def get_adapter(self, provider: str) -> RecordingAdapter:
            assert provider == "openai"
            return RecordingAdapter()

    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: Registry())

    def invoke(
        handle: ProviderCallCredentials,
        timeout: float,
    ) -> dict[str, Any]:
        assert start.wait(timeout=5)
        return chat_service.perform_chat_api_call(
            api_provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            api_key="loose-untrusted-key",
            app_config={"openai_api": {"rotation": "attacker"}},
            credentials_resolved=True,
            timeout=timeout,
            **{PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle},
        )

    calls = [(handles[0], 11.0), (handles[1], 23.0)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(invoke, handle, timeout)
            for handle, timeout in calls
        ]
        start.set()
        try:
            assert both_entered.wait(timeout=5)
        finally:
            release.set()
        results = [future.result(timeout=5) for future in futures]

    assert {result["choices"][0]["message"]["content"] for result in results} == {
        "rotation-a-key",
        "rotation-b-key",
    }
    assert {
        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
        for request, _timeout in captured
    } == set(handles)
    assert {
        request["api_key"]: timeout
        for request, timeout in captured
    } == {
        "rotation-a-key": 11.0,
        "rotation-b-key": 23.0,
    }
    for request, _timeout in captured:
        handle = request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
        assert request["api_key"] == handle.api_key
        assert request["app_config"] == handle.app_config
        assert "timeout" not in request


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "entrypoint",
    ["prompt-executor", "test-runner", "prompt-evaluation", "g-eval"],
)
async def test_shared_cloud_callers_reach_real_adapter_with_isolated_runtime_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    """Shared Prompt/Eval callers preserve each capability through the real adapter."""

    handles = [
        await _issue_credentials(
            "openai",
            api_key=f"shared-{label}-key",
            app_config={
                "openai_api": {
                    "api_base_url": f"https://runtime-{label}.example/v1",
                    "model": "test-model",
                }
            },
        )
        for label in ("a", "b")
    ]
    start = threading.Event()
    both_entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    captured: list[tuple[str, str, float | None]] = []

    class RecordingClient:
        def __init__(self, timeout: float | None) -> None:
            self.timeout = timeout

        def __enter__(self) -> RecordingClient:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            del json
            authorization = headers["Authorization"]
            with lock:
                captured.append((url, authorization, self.timeout))
                if len(captured) == 2:
                    both_entered.set()
            assert release.wait(timeout=5)
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {"message": {"content": authorization}}
                    ]
                },
            )

    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        openai_adapter,
        "http_client_factory",
        lambda *, timeout=None: RecordingClient(timeout),
    )
    adapter = openai_adapter.OpenAIAdapter()
    route_gateway_adapters(monkeypatch, lambda provider: adapter if provider == "openai" else None)
    monkeypatch.setattr(
        prompt_evaluation_manager,
        "is_test_mode",
        lambda: False,
    )
    monkeypatch.setattr(
        ms_g_eval,
        "get_adapter_or_raise",
        lambda provider: adapter if provider == "openai" else None,
    )

    def invoke(handle: ProviderCallCredentials, timeout: float) -> Any:
        assert start.wait(timeout=5)
        common = {
            "provider_credentials": handle,
            "credentials_resolved": True,
        }
        if entrypoint == "prompt-executor":
            request = prompt_executor.PromptExecutor(
                SimpleNamespace(client_id="boundary-test")
            )._build_adapter_request(
                provider="openai",
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                system_prompt=None,
                temperature=0.2,
                max_tokens=16,
                params={},
                app_config={"openai_api": {"api_base_url": "https://attacker.invalid"}},
                api_key_override="loose-attacker-key",
                **common,
            )
            return adapter.chat(request, timeout=timeout)
        if entrypoint == "test-runner":
            return test_runner.TestRunner(object())._call_adapter(
                provider="openai",
                model="test-model",
                messages_payload=[{"role": "user", "content": "hello"}],
                system_message=None,
                temperature=0.2,
                max_tokens=16,
                app_config={"openai_api": {"api_base_url": "https://attacker.invalid"}},
                api_key_override="loose-attacker-key",
                timeout_seconds=timeout,
                **common,
            )
        if entrypoint == "prompt-evaluation":
            return prompt_evaluation_manager.EvaluationManager._call_adapter_text(
                provider="openai",
                messages_payload=[{"role": "user", "content": "hello"}],
                temperature=0.2,
                max_tokens=16,
                api_key="loose-attacker-key",
                model="test-model",
                app_config={"openai_api": {"api_base_url": "https://attacker.invalid"}},
                timeout=timeout,
                **common,
            )
        return ms_g_eval._call_adapter_text(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            api_key="loose-attacker-key",
            model="test-model",
            app_config={"openai_api": {"api_base_url": "https://attacker.invalid"}},
            timeout=timeout,
            **common,
        )

    calls = [(handles[0], 11.0), (handles[1], 23.0)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(invoke, handle, timeout)
            for handle, timeout in calls
        ]
        start.set()
        if not both_entered.wait(timeout=2):
            release.set()
            for future in futures:
                future.result(timeout=5)
            pytest.fail("Both adapter transports were not reached")
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert len(results) == 2
    assert {
        authorization: (url, timeout)
        for url, authorization, timeout in captured
    } == {
        "Bearer shared-a-key": ("https://runtime-a.example/v1/chat/completions", 11.0),
        "Bearer shared-b-key": ("https://runtime-b.example/v1/chat/completions", 23.0),
    }


_UNSAFE_ADAPTERS = (
    ("openai_adapter", "OpenAIAdapter", "openai", "test-model"),
    ("anthropic_adapter", "AnthropicAdapter", "anthropic", "test-model"),
    ("bedrock_adapter", "BedrockAdapter", "bedrock", "meta.llama3-8b-instruct"),
    ("groq_adapter", "GroqAdapter", "groq", "test-model"),
    ("openrouter_adapter", "OpenRouterAdapter", "openrouter", "test-model"),
    ("mistral_adapter", "MistralAdapter", "mistral", "test-model"),
    ("qwen_adapter", "QwenAdapter", "qwen", "test-model"),
    ("google_adapter", "GoogleAdapter", "google", "test-model"),
    ("deepseek_adapter", "DeepSeekAdapter", "deepseek", "test-model"),
)


@pytest.mark.parametrize("mode", ["chat", "stream"])
@pytest.mark.parametrize(
    ("module_name", "adapter_name", "provider", "model"),
    _UNSAFE_ADAPTERS,
    ids=[adapter_name for _module, adapter_name, _provider, _model in _UNSAFE_ADAPTERS],
)
def test_cloud_adapter_error_matrix_detaches_upstream_body_url_cause_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    adapter_name: str,
    provider: str,
    model: str,
    mode: str,
) -> None:
    """Sync and streaming boundaries expose only typed, detached provider errors."""

    sentinel = f"raw-{provider}-{mode}-secret-token"

    def upstream_error() -> httpx.HTTPStatusError:
        request = httpx.Request(
            "POST",
            f"https://upstream.invalid/{sentinel}",
        )
        response = httpx.Response(
            500,
            request=request,
            json={"error": {"message": sentinel}},
        )
        return httpx.HTTPStatusError(
            f"{sentinel} at {request.url}",
            request=request,
            response=response,
        )

    class FailingClient:
        def __enter__(self) -> FailingClient:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def post(self, *_args: Any, **_kwargs: Any) -> Any:
            raise upstream_error()

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            raise upstream_error()

    module = importlib.import_module(
        f"tldw_Server_API.app.core.LLM_Calls.providers.{module_name}"
    )
    monkeypatch.setattr(
        module,
        "http_client_factory",
        lambda *_args, **_kwargs: FailingClient(),
    )
    adapter = getattr(module, adapter_name)()
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": model,
        "api_key": "legacy-test-key",
        "app_config": {},
    }
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with pytest.raises(ChatAPIError) as exc_info:
            result = getattr(adapter, mode)(request)
            if mode == "stream":
                list(result)
    finally:
        logger.remove(sink_id)

    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in "".join(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "stream"])
async def test_kobold_boundary_detaches_upstream_body_url_cause_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """The legacy Kobold handler obeys the same bounded error contract."""

    sentinel = f"raw-kobold-{mode}-secret-token"
    endpoint = "https://kobold.example/v1"
    handle = await _issue_credentials(
        "kobold",
        api_key="kobold-runtime-key",
        app_config={"kobold_api": {"api_ip": endpoint}},
    )

    def failing_fetch(**_kwargs: Any) -> Any:
        request = httpx.Request("POST", f"{endpoint}/{sentinel}")
        response = httpx.Response(
            500,
            request=request,
            json={"error": {"message": sentinel}},
        )
        raise httpx.HTTPStatusError(
            f"{sentinel} at {request.url}",
            request=request,
            response=response,
        )

    from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
        KoboldAdapter,
    )

    adapter = KoboldAdapter()
    adapter.http_fetcher = failing_fetch
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "api_key": "loose-untrusted-key",
        "app_config": {"kobold_api": {"api_ip": "https://attacker.invalid"}},
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with pytest.raises(ChatAPIError) as exc_info:
            result = getattr(adapter, mode)(request)
            if mode == "stream":
                list(result)
    finally:
        logger.remove(sink_id)

    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in "".join(logs)


@pytest.mark.asyncio
async def test_local_handler_cannot_mutate_reused_credential_snapshot() -> None:
    """Each local dispatch gives legacy handlers a detached config copy."""

    endpoint = "https://kobold.example/v1"
    handle = await _issue_credentials(
        "kobold",
        api_key="kobold-runtime-key",
        app_config={
            "kobold_api": {
                "api_ip": endpoint,
                "tenant": "runtime-tenant",
            }
        },
    )
    observed_tenants: list[str] = []

    def mutating_handler(**kwargs: Any) -> dict[str, Any]:
        app_config = kwargs["app_config"]
        observed_tenants.append(app_config["kobold_api"]["tenant"])
        app_config["kobold_api"]["tenant"] = "handler-mutated"
        return {"choices": [{"message": {"content": "ok"}}]}

    from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
        KoboldAdapter,
    )

    adapter = KoboldAdapter()
    adapter._handler = mutating_handler
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "api_key": "loose-untrusted-key",
        "app_config": {"kobold_api": {"api_ip": "https://attacker.invalid"}},
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }

    adapter.chat(request)
    adapter.chat(request)

    assert observed_tenants == ["runtime-tenant", "runtime-tenant"]
    assert handle.app_config == {
        "kobold_api": {
            "api_ip": endpoint,
            "tenant": "runtime-tenant",
        }
    }


@pytest.mark.parametrize("mode", ["chat", "stream"])
def test_openai_200_error_payload_is_rejected_without_body_leak(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """HTTP 200 error objects and SSE error events are still provider failures."""

    sentinel = f"openai-200-{mode}-secret-token"

    class ErrorResponse:
        status_code = 200

        def __enter__(self) -> ErrorResponse:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"error": {"message": sentinel}}

        def iter_lines(self):
            yield f'data: {{"error": {{"message": "{sentinel}"}}}}'
            yield "data: [DONE]"

    class ErrorClient:
        def __enter__(self) -> ErrorClient:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def post(self, *_args: Any, **_kwargs: Any) -> ErrorResponse:
            return ErrorResponse()

        def stream(self, *_args: Any, **_kwargs: Any) -> ErrorResponse:
            return ErrorResponse()

    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    monkeypatch.setattr(
        openai_adapter,
        "http_client_factory",
        lambda *_args, **_kwargs: ErrorClient(),
    )
    adapter = openai_adapter.OpenAIAdapter()
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "api_key": "legacy-test-key",
        "app_config": {},
    }
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        with pytest.raises(ChatAPIError) as exc_info:
            result = getattr(adapter, mode)(request)
            if mode == "stream":
                list(result)
    finally:
        logger.remove(sink_id)

    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in "".join(logs)


@pytest.mark.parametrize("mode", ["chat", "stream"])
@pytest.mark.parametrize(
    ("module_name", "adapter_name"),
    [
        ("cohere_adapter", "CohereAdapter"),
        ("moonshot_adapter", "MoonshotAdapter"),
        ("zai_adapter", "ZaiAdapter"),
    ],
    ids=["CohereAdapter", "MoonshotAdapter", "ZaiAdapter"],
)
def test_legacy_cloud_adapter_rejects_in_band_error_without_body_leak(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    adapter_name: str,
    mode: str,
) -> None:
    """Legacy requests-based cloud adapters replace HTTP-200/SSE errors."""

    sentinel = f"{module_name}-{mode}-in-band-secret"

    class ErrorResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"error": {"message": sentinel}}

        def iter_lines(self, decode_unicode: bool = False):
            del decode_unicode
            yield f'data: {{"error": {{"message": "{sentinel}"}}}}'

        def close(self) -> None:
            return None

    class ErrorSession:
        def post(self, *_args: Any, **_kwargs: Any) -> ErrorResponse:
            return ErrorResponse()

        def close(self) -> None:
            return None

    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: ErrorSession(),
    )
    module = importlib.import_module(
        f"tldw_Server_API.app.core.LLM_Calls.providers.{module_name}"
    )
    adapter = getattr(module, adapter_name)()
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "api_key": "legacy-test-key",
        "app_config": {},
    }

    if mode == "chat":
        with pytest.raises(ChatAPIError) as exc_info:
            adapter.chat(request)
        assert sentinel not in repr(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
    else:
        wire = "".join(adapter.stream(request))
        assert sentinel not in wire
        assert "provider_unavailable" in wire


@pytest.mark.parametrize("mode", ["chat", "stream"])
def test_huggingface_rejects_in_band_error_without_body_leak(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """HuggingFace rejects successful-status error envelopes in both modes."""

    sentinel = f"huggingface-{mode}-in-band-secret"

    class ErrorResponse:
        def __enter__(self) -> ErrorResponse:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"error": {"message": sentinel}}

        def iter_lines(self):
            yield f'data: {{"error": {{"message": "{sentinel}"}}}}'

    class ErrorClient:
        def __enter__(self) -> ErrorClient:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def post(self, *_args: Any, **_kwargs: Any) -> ErrorResponse:
            return ErrorResponse()

        def stream(self, *_args: Any, **_kwargs: Any) -> ErrorResponse:
            return ErrorResponse()

    from tldw_Server_API.app.core.LLM_Calls.providers import huggingface_adapter

    monkeypatch.setattr(
        huggingface_adapter,
        "http_client_factory",
        lambda *_args, **_kwargs: ErrorClient(),
    )
    adapter = huggingface_adapter.HuggingFaceAdapter()
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
        "api_key": "legacy-test-key",
        "app_config": {},
    }

    with pytest.raises(ChatAPIError) as exc_info:
        result = getattr(adapter, mode)(request)
        if mode == "stream":
            list(result)

    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
@_MODEL_SOURCES
async def test_a_model_off_the_allowlist_runs_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch, config_model: str | None, model_override: str | None, expected: str
) -> None:
    """TASK-15582: analyze on the user's own key is not held to the server's allowlist, whatever the model's source."""
    result, calls, warnings = await _summary(
        monkeypatch, overrides=_allowlist("x"), config_model=config_model, model_override=model_override
    )

    assert (result, calls, warnings) == ("summary", [(expected, "handle-key", True)], [])
