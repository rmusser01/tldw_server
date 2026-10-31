from __future__ import annotations

import asyncio
from typing import Any

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
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls import adapter_utils, chat_calls
from tldw_Server_API.app.core.LLM_Calls.providers import moonshot_adapter


def _args(**overrides):
    args = {
        "api_provider": "openai",
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o-mini",
    }
    args.update(overrides)
    return args


def _resolved_fields(
    provider: str,
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build test call fields backed by an authentic runtime capability."""

    async def issue():
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
                auth_source=(
                    "aws_default_chain"
                    if provider == "bedrock" and api_key is None
                    else "api_key"
                ),
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
            return await runtime.resolve(provider)
        finally:
            await runtime.close()

    handle = asyncio.run(issue())
    return {
        "api_key": api_key,
        "app_config": app_config,
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }


def test_explicit_credentials_bypass_resolvers_and_copy_config(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("server credential resolver must not run")

    monkeypatch.setattr(adapter_utils, "ensure_app_config", fail)
    monkeypatch.setattr(adapter_utils, "resolve_provider_api_key_from_config", fail)
    supplied_config = {"openai_api": {"organization": "tenant-org"}}

    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            **_resolved_fields(
                "openai",
                api_key="explicit-key",
                app_config=supplied_config,
            ),
        )
    )

    assert provider == "openai"
    assert request["api_key"] == "explicit-key"
    assert request["app_config"] == supplied_config
    assert request["app_config"] is not supplied_config
    assert request["credentials_resolved"] is True


def test_explicit_missing_hosted_key_fails_safely(monkeypatch):
    monkeypatch.setattr(
        adapter_utils,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        chat_service._build_adapter_request_from_chat_args(
            _args(
                **_resolved_fields("openai", api_key="  ", app_config={})
            )
        )

    assert exc_info.value.provider == "openai"
    assert "key" in str(exc_info.value).lower()
    assert "explicit-key" not in repr(exc_info.value)


def test_explicit_local_provider_remains_keyless():
    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            api_provider="ollama",
            model="llama3",
            **_resolved_fields(
                "ollama",
                api_key=None,
                app_config={
                    "ollama_api": {"api_url": "http://127.0.0.1:11434"}
                },
            ),
        )
    )

    assert provider == "ollama"
    assert request["api_key"] is None


def test_legacy_request_without_marker_still_resolves_key(monkeypatch):
    calls = []

    def resolve(provider, app_config):
        calls.append((provider, app_config))
        return "server-key"

    monkeypatch.setattr(adapter_utils, "resolve_provider_api_key_from_config", resolve)

    _provider, request, _internal = chat_service._build_adapter_request_from_chat_args(_args(app_config={}))

    assert request["api_key"] == "server-key"
    assert calls == [("openai", {})]


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def close(self):
        return None


class _Session:
    def __init__(self):
        self.urls = []

    def post(self, url, **_kwargs):
        self.urls.append(url)
        return _Response()

    def close(self):
        return None


def test_explicit_absent_config_cannot_trigger_adapter_config_reload(monkeypatch):
    session = _Session()
    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda **_kwargs: session)
    monkeypatch.setattr(
        moonshot_adapter,
        "load_and_log_configs",
        lambda: (_ for _ in ()).throw(AssertionError("adapter must not reload server config")),
    )
    _provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            api_provider="moonshot",
            **_resolved_fields(
                "moonshot",
                api_key="explicit-key",
                app_config=None,
            ),
        )
    )
    adapter_request = {**request, "app_config": dict(request["app_config"])}

    result = moonshot_adapter.MoonshotAdapter().chat(adapter_request)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert session.urls == ["https://api.moonshot.cn/v1/chat/completions"]
    assert type(request["app_config"]) is dict
    assert request["app_config"] == {"moonshot_api": {}}
    assert request["credentials_resolved"] is True


def test_legacy_empty_config_still_allows_adapter_config_reload(monkeypatch):
    session = _Session()
    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda **_kwargs: session)
    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda _config: {})
    monkeypatch.setattr(
        moonshot_adapter,
        "load_and_log_configs",
        lambda: {"moonshot_api": {"api_base_url": "https://legacy.example/v1"}},
    )
    _provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(api_provider="moonshot", api_key="legacy-key")
    )

    moonshot_adapter.MoonshotAdapter().chat(request)

    assert session.urls == ["https://legacy.example/v1/chat/completions"]


@pytest.mark.parametrize(
    ("provider", "section", "api_key"),
    [
        ("cohere", "cohere_api", "key"),
        ("moonshot", "moonshot_api", "key"),
        ("zai", "zai_api", "key"),
        ("ollama", "ollama_api", None),
    ],
)
def test_explicit_absent_config_is_plain_provider_scoped_mapping(provider, section, api_key):
    _provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            api_provider=provider,
            **_resolved_fields(
                provider,
                api_key=api_key,
                app_config=None,
            ),
        )
    )

    assert type(request["app_config"]) is dict
    assert dict(request["app_config"]) == {section: {}}
    assert request["credentials_resolved"] is True


def test_bedrock_runtime_default_chain_satisfies_shared_auth_contract():
    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            api_provider="bedrock",
            model="meta.llama3-8b-instruct",
            **_resolved_fields(
                "bedrock",
                api_key=None,
                app_config={
                    "bedrock_api": {"_runtime_auth_source": "aws_default_chain"}
                },
            ),
        )
    )

    assert provider == "bedrock"
    assert request["api_key"] is None
    assert request["credentials_resolved"] is True


def test_bedrock_runtime_absent_auth_fails_before_server_key_resolution(monkeypatch):
    monkeypatch.setattr(
        adapter_utils,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    with pytest.raises(ChatConfigurationError, match="API key is required"):
        chat_service._build_adapter_request_from_chat_args(
            _args(
                api_provider="bedrock",
                model="meta.llama3-8b-instruct",
                **_resolved_fields(
                    "bedrock",
                    api_key=None,
                    app_config={"bedrock_api": {}},
                ),
            )
        )


def test_explicit_missing_model_does_not_use_default_model_environment(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_MOONSHOT", "server-env-model")

    with pytest.raises(ChatConfigurationError, match="Model is required"):
        chat_service._build_adapter_request_from_chat_args(
            {
                "api_provider": "moonshot",
                "messages": [{"role": "user", "content": "hello"}],
                **_resolved_fields(
                    "moonshot",
                    api_key="explicit-key",
                    app_config=None,
                ),
            }
        )
