from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)


def _credentials(
    source: str,
    *,
    base_url: str | None = None,
) -> ProviderCallCredentials:
    return ProviderCallCredentials(
        provider="custom-openai-api",
        api_key="key",
        app_config=None,
        auth_source=source,
        runtime_generation=0,
        runtime_identity=object(),
        credential_identity=object(),
        endpoint_provenance="byok" if base_url else "server_config",
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("user", "server_config"),
        ("team", "server_config"),
        ("org", "server_config"),
        ("server", "server_config"),
        ("fallback", "server_config"),
    ],
)
def test_endpoint_provenance_is_url_free_and_read_from_runtime_handle(
    source: str,
    expected: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.chat import _derive_endpoint_provenance

    value = _derive_endpoint_provenance(_credentials(source), request_override=False)

    assert value == expected
    assert "://" not in value


def test_byok_owned_endpoint_sets_byok_provenance() -> None:
    from tldw_Server_API.app.api.v1.endpoints.chat import _derive_endpoint_provenance

    value = _derive_endpoint_provenance(
        _credentials("user", base_url="http://user-owned:18080/v1"),
        request_override=False,
    )

    assert value == "byok"


def test_request_override_takes_provenance_precedence() -> None:
    from tldw_Server_API.app.api.v1.endpoints.chat import _derive_endpoint_provenance

    assert _derive_endpoint_provenance(_credentials("user"), request_override=True) == "request_override"


def test_chat_service_only_accepts_endpoint_owned_private_provenance() -> None:
    from tldw_Server_API.app.core.Chat import chat_service

    args = {
        "api_provider": "custom-openai-api",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "model",
        "api_key": "key",
        "_endpoint_provenance": "byok",
        "endpoint_provenance": "request_override",
        "configured_endpoint_scope": object(),
        "configured_endpoint_base_url": "http://attacker.invalid",
        "http_fetcher": object(),
        "http_streamer": object(),
    }

    _provider, request, internal = chat_service._build_adapter_request_from_chat_args(args)

    assert request["_endpoint_provenance"] == "byok"
    assert "endpoint_provenance" not in request
    assert "configured_endpoint_scope" not in request
    assert "configured_endpoint_base_url" not in request
    assert internal == {}


def test_non_custom_provider_filters_only_private_endpoint_provenance() -> None:
    from tldw_Server_API.app.core.Chat import chat_service

    args = {
        "api_provider": "huggingface",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/runtime-model",
        "api_key": "stored-hf-key",
        "credentials_resolved": True,
        "_endpoint_provenance": "server_config",
        "extra_headers": {"X-Provider-Extension": "allowed"},
        "extra_body": {"declared_extension": "kept"},
    }

    provider, request, internal = chat_service._build_adapter_request_from_chat_args(args)

    assert provider == "huggingface"
    assert "_endpoint_provenance" not in request
    assert request["credentials_resolved"] is True
    assert request["extra_headers"] == args["extra_headers"]
    assert request["extra_body"] == args["extra_body"]
    assert internal == {}


def test_untrusted_private_provenance_value_is_discarded() -> None:
    from tldw_Server_API.app.core.Chat import chat_service

    args = {
        "api_provider": "custom-openai-api",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "model",
        "api_key": "key",
        "_endpoint_provenance": "http://attacker.invalid",
    }

    _provider, request, _internal = chat_service._build_adapter_request_from_chat_args(args)

    assert "_endpoint_provenance" not in request


@pytest.mark.parametrize(
    ("source", "byok_base_url", "expected"),
    [
        ("user", None, "server_config"),
        ("team", None, "server_config"),
        ("org", None, "server_config"),
        ("user", "http://user-owned:18080/v1", "byok"),
        ("server", None, "server_config"),
    ],
)
def test_chat_endpoint_sets_post_parse_url_free_provenance(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    byok_base_url: str | None,
    expected: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    async def _resolve_byok(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=provider,
            api_key="key",
            app_config=None,
            credential_fields={"base_url": byok_base_url} if byok_base_url else {},
            source=source,
            allowlisted=True,
        )

    class _Runtime(ProviderCredentialRuntime):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("override_snapshot_resolver", None)
            super().__init__(
                resolver=_resolve_byok,
                server_config_snapshot={},
                **kwargs,
            )

    captured: list[dict] = []

    def _perform_chat_api_call(**kwargs):
        captured.append(kwargs)
        return {
            "id": "chatcmpl-provenance",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(chat_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", _perform_chat_api_call)
    body = {
        "api_provider": "custom-openai-api",
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
    }

    response = authenticated_client.post("/api/v1/chat/completions", json=body)

    assert response.status_code == 200
    marker = captured[-1]["_endpoint_provenance"]
    assert marker == expected
    assert "://" not in marker
