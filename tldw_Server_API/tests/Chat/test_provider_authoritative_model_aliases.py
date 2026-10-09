from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import messages
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory as inventory
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import AuditedCallPolicyTransport
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
from tldw_Server_API.tests.Chat.test_provider_authoritative_model_selection import RecordingAdapter

pytestmark = pytest.mark.unit


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> SimpleNamespace:
    """Record generation and fake all catalog/detail HTTP with isolated keys."""
    adapter = RecordingAdapter()
    state = SimpleNamespace(
        adapter=adapter,
        key="fake-" + request.node.name,
        models=("claude-current-20990101",),
        resolved="claude-current-20990101",
        alias_status=200,
        http=[],
    )

    def fetch(**kwargs: Any) -> SimpleNamespace:
        """Return a finite current catalog or configurable authoritative detail."""
        state.http.append(kwargs)
        url = kwargs["url"]
        if url.endswith("/models") or url.endswith("/models/user"):
            payload = {"data": [{"id": model} for model in state.models], "has_more": False}
            status = 200
        else:
            payload = {
                "id": state.resolved,
                "type": "model",
                "providers": [{"provider": "test-provider", "status": "live"}],
            }
            status = state.alias_status
        return SimpleNamespace(status_code=status, json=lambda: payload, close=lambda: None)

    registry = SimpleNamespace(
        get_adapter=lambda _provider: adapter,
        get_audited_call_policy_transport=lambda _provider: AuditedCallPolicyTransport(
            maximum_transport_attempts=1,
            enforces_configured_endpoint_scope=True,
            enforces_maximum_timeout=True,
        ),
    )
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(chat_service, "discover_provider_models", inventory.discover_provider_models)
    monkeypatch.setattr(inventory, "_http_fetch", fetch)
    return state


def _generate(
    state: SimpleNamespace, provider: str, model: str, *,
    async_call: bool = False, stream: bool = False, **extra: Any,
) -> Any:
    """Exercise the real dispatch guard with a recording generation adapter."""
    kwargs = dict(
        api_provider=provider,
        model=model,
        api_key=state.key,
        app_config=(
            {"huggingface_api": {"api_base_url": "https://router.huggingface.co/v1"}}
            if provider == "huggingface" else {}
        ),
        messages=[{"role": "user", "content": "hello"}],
        stream=stream,
        **extra,
    )
    if async_call:
        return asyncio.run(chat_service.perform_chat_api_call_async(**kwargs))
    return chat_service.perform_chat_api_call(**kwargs)


@pytest.mark.parametrize("async_call", (False, True))
@pytest.mark.parametrize("stream", (False, True))
def test_anthropic_api_confirmed_alias_reaches_generation(boundary: SimpleNamespace, async_call: bool, stream: bool) -> None:
    _generate(boundary, "anthropic", "claude-current-alias", async_call=async_call, stream=stream)
    assert len(boundary.adapter.calls) == 1
    assert boundary.adapter.calls[0][1]["model"] in {"claude-current-alias", boundary.resolved}
    alias_calls = [call for call in boundary.http if call["url"].endswith("/models/claude-current-alias")]
    assert len(alias_calls) == 1
    call = alias_calls[0]
    assert call["headers"]["x-api-key"] == boundary.key
    assert call["url"] == "https://api.anthropic.com/v1/models/claude-current-alias"
    assert call["sensitive_observability"] is True
    assert call["allow_redirects"] is False
    assert call["retry"].attempts == 1
    assert 0 < call["timeout"] <= 5


@pytest.mark.parametrize("resolved", ("claude-retired-20000101", "CLAUDE-CURRENT-20990101", ""))
def test_alias_cannot_resurrect_an_unlisted_destination(boundary: SimpleNamespace, resolved: str) -> None:
    boundary.resolved = resolved
    with pytest.raises(ChatBadRequestError):
        _generate(boundary, "anthropic", "claude-current-alias")
    assert boundary.adapter.calls == []


def test_anthropic_404_does_not_infer_a_date_stripped_alias(boundary: SimpleNamespace) -> None:
    boundary.alias_status = 404
    with pytest.raises(ChatBadRequestError):
        _generate(boundary, "anthropic", "claude-current")
    assert any(call["url"].endswith("/models/claude-current") for call in boundary.http)
    assert boundary.adapter.calls == []


@pytest.mark.parametrize("status", (401, 403, 429, 500))
def test_alias_lookup_failure_is_sanitized_and_never_generates(boundary: SimpleNamespace, status: int) -> None:
    boundary.alias_status = status
    with pytest.raises(ChatConfigurationError) as error:
        _generate(boundary, "anthropic", "claude-current-alias")
    assert boundary.key not in str(error.value)
    assert boundary.adapter.calls == []


def test_confirmed_alias_is_known_to_credential_scoped_prevalidation(boundary: SimpleNamespace) -> None:
    assert chat_service.is_model_known_for_provider(
        "anthropic", "claude-current-alias", api_key=boundary.key,
        app_config={}, credentials_resolved=True,
    ) is True


@pytest.mark.parametrize("suffix", ("nitro", "floor", "exacto", "online", "nitro:exacto"))
def test_openrouter_routing_suffix_preserves_original_selection(boundary: SimpleNamespace, suffix: str) -> None:
    boundary.models = ("vendor/current",)
    requested = "vendor/current:" + suffix
    _generate(boundary, "openrouter", requested, extra_body={"models": [requested]})
    selected = boundary.adapter.calls[0][1]
    assert selected["model"] == requested
    assert selected["extra_body"]["models"] == [requested]


@pytest.mark.parametrize("model", ("vendor/current:free", "vendor/current:free:nitro", "vendor/current:nitro:free"))
def test_openrouter_catalog_variant_is_never_replaced_by_paid_base(boundary: SimpleNamespace, model: str) -> None:
    boundary.models = ("vendor/current",)
    with pytest.raises(ChatBadRequestError):
        _generate(boundary, "openrouter", model)
    assert boundary.adapter.calls == []


def test_openrouter_listed_free_variant_with_routing_is_preserved(boundary: SimpleNamespace) -> None:
    boundary.models = ("vendor/current:free",)
    requested = "vendor/current:nitro:free:exacto"
    _generate(boundary, "openrouter", requested)
    assert boundary.adapter.calls[0][1]["model"] == requested


@pytest.mark.parametrize("suffix", ("cheapest", "fastest", "preferred", "test-provider"))
def test_huggingface_router_selector_preserves_original_selection(boundary: SimpleNamespace, suffix: str) -> None:
    boundary.models = ("vendor/current",)
    boundary.resolved = "vendor/current"
    requested = "vendor/current:" + suffix
    _generate(boundary, "huggingface", requested)
    assert boundary.adapter.calls[0][1]["model"] == requested


@pytest.mark.parametrize("provider,model", (("openrouter", "vendor/retired:nitro"), ("huggingface", "vendor/retired:fastest")))
def test_routing_suffix_never_resurrects_an_unlisted_base(boundary: SimpleNamespace, provider: str, model: str) -> None:
    boundary.models = ("vendor/current",)
    with pytest.raises(ChatBadRequestError):
        _generate(boundary, provider, model)
    assert boundary.adapter.calls == []


def test_alias_lookup_respects_audited_endpoint_and_deadline(boundary: SimpleNamespace) -> None:
    policy = ProviderCallPolicy(
        max_transport_attempts=1,
        maximum_timeout_seconds=0.5,
        required_endpoint_scope=ConfiguredEndpointScope.from_url("https://api.anthropic.com/v1/messages"),
        privacy_safe_errors=True,
    )
    _generate(boundary, "anthropic", "claude-current-alias", call_policy=policy)
    assert len(boundary.adapter.calls) == 1
    for call in boundary.http:
        assert call["configured_endpoint"] == policy.required_endpoint_scope
        assert 0 < call["timeout"] <= 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("messages", "count_tokens"))
@pytest.mark.parametrize("lookup_status,resolved,expected_status", (
    (200, "claude-current-20990101", None),
    (200, "claude-retired-20000101", 400),
    (503, "", 503),
))
async def test_native_messages_uses_same_authoritative_alias_membership(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, operation: str,
    lookup_status: int, resolved: str, expected_status: int | None,
) -> None:
    boundary.alias_status = lookup_status
    boundary.resolved = resolved
    credentials = SimpleNamespace(api_key=boundary.key, credentials_resolved=True, app_config={})

    async def resolve(provider: str, *, model: str) -> SimpleNamespace:
        """Return the same synthetic credential used by catalog and alias HTTP."""
        assert provider == "anthropic"
        assert model == "claude-current-alias"
        return credentials

    monkeypatch.setattr(messages, "provider_auth_is_resolved", lambda *_args, **_kwargs: True)
    runtime = SimpleNamespace(resolve=resolve)
    if expected_status is None:
        assert await messages._resolve_messages_credentials(
            runtime, "anthropic", "claude-current-alias", operation=operation,
        ) is credentials
    else:
        with pytest.raises(HTTPException) as error:
            await messages._resolve_messages_credentials(
                runtime, "anthropic", "claude-current-alias", operation=operation,
            )
        assert error.value.status_code == expected_status
    assert any(call["url"].endswith("/models/claude-current-alias") for call in boundary.http)
