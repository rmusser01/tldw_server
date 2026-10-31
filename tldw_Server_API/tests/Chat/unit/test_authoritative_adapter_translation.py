from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
from tldw_Server_API.app.api.v1.schemas.anthropic_messages import (
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary_lib
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    bind_provider_call_credentials,
    resolve_provider_section,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload

_CHAT_PROVIDERS = (
    "anthropic",
    "openai",
    "deepseek",
    "google",
    "groq",
    "qwen",
    "mistral",
    "openrouter",
    "bedrock",
    "custom-openai-api",
    "custom-openai-api-2",
    "custom-openai-api-37",
    "novita",
    "poe",
    "together",
)


def _resolved_fields(provider: str) -> dict[str, Any]:
    """Issue one authentic capability for translation-boundary tests."""

    section = resolve_provider_section(provider)
    provider_config: dict[str, Any] = {}
    if provider.startswith("custom-openai-api"):
        provider_config["api_ip"] = "https://custom.example/v1"
    app_config = {section: provider_config} if section else {}

    async def issue():
        async def resolver(
            normalized_provider: str,
            **_kwargs: Any,
        ) -> ResolvedByokCredentials:
            return ResolvedByokCredentials(
                provider=normalized_provider,
                api_key="test-key",
                app_config=app_config,
                credential_fields={},
                source="user",
                allowlisted=True,
                status=ByokResolutionStatus.RESOLVED,
                auth_source="api_key",
            )

        runtime = ProviderCredentialRuntime(
            user_id=23,
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

    handle = asyncio.run(issue())
    return {
        "api_key": "test-key",
        "app_config": app_config,
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }


@pytest.mark.unit
@pytest.mark.parametrize("provider", _CHAT_PROVIDERS)
def test_chat_translation_and_validation_preserve_resolved_credentials_marker(
    provider: str,
) -> None:
    resolved = _resolved_fields(provider)
    translated_provider, request, _internal = (
        chat_service._build_adapter_request_from_chat_args(
            {
                "api_provider": provider,
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                **resolved,
            }
        )
    )

    bound, credentials = bind_provider_call_credentials(
        translated_provider,
        request,
        consume=True,
    )
    validated = validate_payload(translated_provider, bound)

    assert request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is resolved[
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
    ]
    assert credentials is resolved[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
    assert validated["credentials_resolved"] is True


@pytest.mark.unit
def test_summarization_adapter_request_preserves_resolved_credentials_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolved_fields("openai")
    captured: dict[str, Any] = {}

    class _Adapter:
        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            del timeout
            captured.update(request)
            return {"choices": [{"message": {"content": "summary"}}]}

    class _Registry:
        def get_adapter(self, provider: str) -> _Adapter:
            assert provider == "openai"
            return _Adapter()

    monkeypatch.setattr(summary_lib, "get_registry", lambda: _Registry())

    result = summary_lib._summarize_via_adapter(
        api_name="openai",
        text_to_summarize="source text",
        custom_prompt_arg=None,
        api_key=resolved["api_key"],
        temp=0.2,
        system_message=None,
        streaming=False,
        model_override="test-model",
        app_config=resolved["app_config"],
        credentials_resolved=True,
        provider_credentials=resolved[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY],
        raise_on_error=True,
    )

    assert result == "summary"
    assert captured["credentials_resolved"] is True

@pytest.mark.unit
def test_converted_messages_request_preserves_resolved_credentials_marker() -> None:
    resolved = _resolved_fields("openai")
    request_data = AnthropicMessagesRequest(
        model="test-model",
        messages=[AnthropicMessage(role="user", content="hello")],
    )

    call_params = messages_endpoint._build_openai_call_params(
        request_data=request_data,
        provider="openai",
        model="test-model",
        app_config=resolved["app_config"],
        api_key=resolved["api_key"],
        credentials=resolved[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY],
    )

    assert call_params["credentials_resolved"] is True
    assert call_params[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is resolved[
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
    ]


@pytest.mark.unit
@pytest.mark.parametrize("app_config", ({}, {"anthropic_api": {}}))
def test_native_messages_empty_snapshot_ignores_environment_added_later(
    monkeypatch: pytest.MonkeyPatch,
    app_config: dict[str, Any],
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://late-env.example/v1")

    resolved = messages_endpoint._resolve_messages_base_url("anthropic", app_config)

    assert resolved == "https://api.anthropic.com/v1"


@pytest.mark.unit
def test_native_messages_none_config_keeps_legacy_environment_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://legacy-env.example/v1")

    assert (
        messages_endpoint._resolve_messages_base_url("anthropic", None)
        == "https://legacy-env.example/v1"
    )


@pytest.mark.unit
def test_native_llamacpp_empty_snapshot_does_not_fall_through_to_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        messages_endpoint,
        "loaded_config_data",
        {"llama_api": {"api_ip": "http://ambient.example:8080"}},
    )

    with pytest.raises(messages_endpoint.HTTPException) as exc_info:
        messages_endpoint._resolve_messages_base_url("llama.cpp", {})

    assert exc_info.value.status_code == 503


@pytest.mark.unit
def test_native_llamacpp_empty_snapshot_does_not_adopt_global_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        messages_endpoint,
        "loaded_config_data",
        {"llama_api": {"api_key": "ambient-key"}},
    )

    assert messages_endpoint._resolve_llamacpp_api_key({}) is None
    assert messages_endpoint._resolve_llamacpp_api_key(None) == "ambient-key"


@pytest.mark.unit
def test_native_messages_empty_snapshot_does_not_adopt_global_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        messages_endpoint,
        "loaded_config_data",
        {"anthropic_api": {"api_timeout": 123}},
    )

    assert messages_endpoint._resolve_native_timeout("anthropic", {}) == 60.0
    assert messages_endpoint._resolve_native_timeout("anthropic", None) == 123.0
