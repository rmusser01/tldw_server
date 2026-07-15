from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ import byok_helpers
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError, ChatProviderError
from tldw_Server_API.app.core.exceptions import EgressPolicyError


def _base_args() -> dict:
    return {
        "api_provider": "openai",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "api_key": "test-key",
        "app_config": {},
    }


def test_base_url_override_allowed(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"openai"})
    monkeypatch.setattr(byok_helpers, "validate_base_url_override", lambda value: value)
    args = _base_args()
    args.update({"base_url": "https://example.com/v1", "trusted_base_url_override": True})
    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(args)
    assert provider == "openai"  # nosec B101
    assert request["base_url"] == "https://example.com/v1"  # nosec B101


def test_base_url_override_allowlist_accepts_provider_alias(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"oai"})
    monkeypatch.setattr(byok_helpers, "validate_base_url_override", lambda value: value)
    args = _base_args()
    args.update(
        {
            "api_provider": "oai",
            "base_url": "https://example.com/v1",
            "trusted_base_url_override": True,
        }
    )

    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(args)

    assert provider == "openai"  # nosec B101
    assert request["base_url"] == "https://example.com/v1"  # nosec B101


def test_base_url_override_rejected_when_untrusted(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"openai"})
    args = _base_args()
    args.update({"base_url": "https://example.com/v1"})
    with pytest.raises(ChatBadRequestError):
        chat_service._build_adapter_request_from_chat_args(args)


def test_base_url_override_rejected_when_not_allowlisted(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: set())
    args = _base_args()
    args.update({"base_url": "https://example.com/v1", "trusted_base_url_override": True})
    with pytest.raises(ChatBadRequestError):
        chat_service._build_adapter_request_from_chat_args(args)


@pytest.mark.parametrize(
    ("provider", "override_key", "expected_provider"),
    [
        ("llama.cpp", "api_url", "llama.cpp"),
        ("llama-cpp", "api_url", "llama.cpp"),
        ("tabby_api", "api_url", "tabbyapi"),
        ("vllm", "vllm_api_url", "vllm"),
        ("ollama", "ollama_api_url", "ollama"),
    ],
)
def test_local_provider_request_url_overrides_rejected(
    provider: str,
    override_key: str,
    expected_provider: str,
) -> None:
    args = _base_args()
    args.update(
        {
            "api_provider": provider,
            "model": "local-model",
            override_key: "http://127.0.0.1:1234/v1",
        }
    )

    with pytest.raises(ChatBadRequestError) as exc_info:
        chat_service._build_adapter_request_from_chat_args(args)

    assert exc_info.value.provider == expected_provider  # nosec B101
    assert override_key in exc_info.value.message  # nosec B101


def test_build_adapter_request_omits_internal_chat_metadata() -> None:
    args = _base_args()
    args.update(
        {
            "_chat_effective_tool_names": ["run", "notes.search"],
            "_chat_run_first_eligible": True,
            "_chat_run_first_ineligible_reason": "provider_not_in_rollout_allowlist",
            "_chat_run_first_presentation_variant": "chat_phase2b_v1",
            "_chat_run_first_cohort": "gated",
        }
    )

    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(args)

    assert provider == "openai"  # nosec B101
    assert not any(key.startswith("_chat_") for key in request)  # nosec B101


def test_chat_request_cannot_select_local_transport_or_scope() -> None:
    args = _base_args()
    args.update(
        {
            "api_provider": "llama.cpp",
            "configured_endpoint_base_url": "http://attacker.invalid:9000",
            "configured_endpoint_scope": object(),
            "http_client_factory": object(),
            "http_fetcher": object(),
            "http_streamer": object(),
        }
    )

    _provider, request, internal = chat_service._build_adapter_request_from_chat_args(args)

    assert not {
        "configured_endpoint_base_url",
        "configured_endpoint_scope",
        "http_client_factory",
        "http_fetcher",
        "http_streamer",
    }.intersection(request)
    assert internal == {}


def test_dns_unresolved_maps_to_sanitized_reachability_error() -> None:
    mapped = chat_service._map_provider_egress_error(
        "custom-openai-api",
        EgressPolicyError("secret host", reason_code="dns_unresolved"),
    )

    assert isinstance(mapped, ChatProviderError)
    assert mapped.status_code == 503
    assert "secret host" not in mapped.message


def test_policy_denial_maps_to_sanitized_configuration_error() -> None:
    mapped = chat_service._map_provider_egress_error(
        "custom-openai-api",
        EgressPolicyError("secret host", reason_code="origin_mismatch"),
    )

    assert isinstance(mapped, ChatConfigurationError)
    assert "secret host" not in mapped.message
