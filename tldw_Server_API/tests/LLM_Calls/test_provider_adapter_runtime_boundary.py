from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from loguru import logger

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
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl

pytestmark = pytest.mark.unit


async def _issue_credentials(
    provider: str,
    *,
    api_key: str,
    app_config: dict[str, Any],
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
            credential_fields={},
            source="user",
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
        chat_service._build_adapter_request_from_chat_args(
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
        chat_service._build_adapter_request_from_chat_args(
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


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_event_gated_cloud_calls_keep_rotated_credentials_isolated(
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
    captured: list[dict[str, Any]] = []

    class RecordingAdapter:
        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            with lock:
                captured.append(request)
                if len(captured) == 2:
                    both_entered.set()
            assert release.wait(timeout=5)
            return {"choices": [{"message": {"content": request["api_key"]}}]}

    class Registry:
        def get_adapter(self, provider: str) -> RecordingAdapter:
            assert provider == "openai"
            return RecordingAdapter()

    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: Registry())

    def invoke(handle: ProviderCallCredentials) -> dict[str, Any]:
        assert start.wait(timeout=5)
        return chat_service.perform_chat_api_call(
            api_provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            api_key="loose-untrusted-key",
            app_config={"openai_api": {"rotation": "attacker"}},
            credentials_resolved=True,
            **{PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke, handle) for handle in handles]
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
        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] for request in captured
    } == set(handles)
    for request in captured:
        handle = request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
        assert request["api_key"] == handle.api_key
        assert request["app_config"] == handle.app_config


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
