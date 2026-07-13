from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls import adapter_utils


def _args(**overrides):
    args = {
        "api_provider": "openai",
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o-mini",
    }
    args.update(overrides)
    return args


def test_explicit_credentials_bypass_resolvers_and_copy_config(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("server credential resolver must not run")

    monkeypatch.setattr(adapter_utils, "ensure_app_config", fail)
    monkeypatch.setattr(adapter_utils, "resolve_provider_api_key_from_config", fail)
    supplied_config = {"openai_api": {"organization": "tenant-org"}}

    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            api_key="explicit-key",
            app_config=supplied_config,
            credentials_resolved=True,
        )
    )

    assert provider == "openai"
    assert request["api_key"] == "explicit-key"
    assert request["app_config"] == supplied_config
    assert request["app_config"] is not supplied_config
    assert "credentials_resolved" not in request


def test_explicit_missing_hosted_key_fails_safely(monkeypatch):
    monkeypatch.setattr(
        adapter_utils,
        "resolve_provider_api_key_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        chat_service._build_adapter_request_from_chat_args(
            _args(api_key="  ", app_config={}, credentials_resolved=True)
        )

    assert exc_info.value.provider == "openai"
    assert "key" in str(exc_info.value).lower()
    assert "explicit-key" not in repr(exc_info.value)


def test_explicit_local_provider_remains_keyless():
    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
        _args(
            api_provider="ollama",
            model="llama3",
            api_key=None,
            app_config={"ollama_api": {"api_url": "http://127.0.0.1:11434"}},
            credentials_resolved=True,
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
