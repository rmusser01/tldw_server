from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials


def _credentials(source: str) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider="custom-openai-api",
        api_key="key",
        app_config=None,
        credential_fields={},
        source=source,
        allowlisted=True,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("user", "byok"),
        ("team", "byok"),
        ("org", "byok"),
        ("server", "server_config"),
        ("fallback", "server_config"),
    ],
)
def test_endpoint_provenance_is_url_free_and_derived_from_byok_source(
    source: str,
    expected: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.chat import _derive_endpoint_provenance

    value = _derive_endpoint_provenance(_credentials(source), request_override=False)

    assert value == expected
    assert "://" not in value


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
