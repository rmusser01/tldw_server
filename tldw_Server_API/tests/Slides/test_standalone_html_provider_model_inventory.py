from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace

import httpx
import pytest

from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory as inventory
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult
from tldw_Server_API.app.core.Slides import standalone_html_provider as provider
from tldw_Server_API.tests.Slides.test_standalone_html_provider import (
    DOCUMENT,
    PROVIDER_SECRET,
    SYSTEM_PROMPT,
    USER_CONTENT,
    _config,
    _response,
    _response_body,
    _target,
)

pytestmark = pytest.mark.unit

COMMERCIAL = ("openai_official_chat_v1", "anthropic_official_messages_v1")


@pytest.fixture
def boundary(monkeypatch):
    posts = []
    discoveries = []
    result = ModelDiscoveryResult("ready", ("Current-ID",))

    def discover(name, api_key, **kwargs):
        discoveries.append((name, api_key, kwargs, threading.get_ident()))
        return result

    def send(request):
        posts.append(request)
        name = "anthropic" if request.url.path.endswith("/messages") else "openai"
        return _response(_response_body(name))[0]

    monkeypatch.setattr(inventory, "discover_provider_models", discover)
    monkeypatch.setattr(
        provider,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(send), **kwargs),
    )
    return posts, discoveries


async def _generate(target, *, config=None, loader=None, api_key=PROVIDER_SECRET):
    config = config or _config(target)
    return await provider.generate_standalone_html(
        stored_target=target,
        system_prompt=SYSTEM_PROMPT,
        user_content=USER_CONTENT,
        provider_api_key=api_key,
        current_config_loader=loader or (lambda: config),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", COMMERCIAL)
@pytest.mark.parametrize("model", ("Retired-ID", "current-id"))
async def test_retired_or_case_mismatched_model_never_posts(boundary, adapter, model):
    posts, _ = boundary
    with pytest.raises(provider.StandaloneHtmlProviderError) as error:
        await _generate(_target(adapter, model=model))
    assert error.value.code == "standalone_html_model_not_allowed"
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", COMMERCIAL)
async def test_current_model_uses_same_credential_endpoint_and_scoped_policy(boundary, adapter):
    posts, discoveries = boundary
    target = _target(adapter, model="Current-ID")
    assert await _generate(target) == DOCUMENT.encode()
    assert len(posts) == len(discoveries) == 1
    name, key, options, worker_thread = discoveries[0]
    assert name == target.provider
    assert key == PROVIDER_SECRET
    suffix = "/messages" if name == "anthropic" else "/chat/completions"
    assert options["base_url"] == target.endpoint_identity.removesuffix(suffix)
    policy = options["call_policy"]
    assert policy.max_transport_attempts == 1
    assert policy.privacy_safe_errors is True
    assert 0 < policy.maximum_timeout_seconds <= 180
    assert policy.required_endpoint_scope.matches(target.endpoint_identity)
    assert not policy.required_endpoint_scope.matches("https://other.example/v1/models")
    assert worker_thread != threading.get_ident()
    assert json.loads(posts[0].content)["model"] == "Current-ID"
    header = "x-api-key" if name == "anthropic" else "authorization"
    assert posts[0].headers[header] == (PROVIDER_SECRET if name == "anthropic" else f"Bearer {PROVIDER_SECRET}")


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", COMMERCIAL)
@pytest.mark.parametrize("status", ("unsupported", "auth_failed", "unreachable", "server_error"))
async def test_unavailable_inventory_never_posts(boundary, monkeypatch, adapter, status):
    posts, _ = boundary
    monkeypatch.setattr(inventory, "discover_provider_models", lambda *_a, **_k: ModelDiscoveryResult(status))
    with pytest.raises(provider.StandaloneHtmlProviderError) as error:
        await _generate(_target(adapter, model="Current-ID"))
    assert error.value.code == "standalone_html_provider_unavailable"
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ("llamacpp_loopback_chat_v1_ipv4", "ollama_loopback_chat_v1_ipv4"))
async def test_local_targets_do_not_require_cloud_inventory(boundary, adapter):
    posts, discoveries = boundary
    assert await _generate(_target(adapter, model="Local-Model"), api_key=None) == DOCUMENT.encode()
    assert len(posts) == 1
    assert discoveries == []


@pytest.mark.asyncio
async def test_discovery_elapsed_exhausts_overall_deadline_without_post(boundary, monkeypatch):
    posts, _ = boundary
    target = _target(COMMERCIAL[0], model="Current-ID")
    ticked = threading.Event()

    def slow_discovery(*_args, **_kwargs):
        time.sleep(0.04)
        return ModelDiscoveryResult("ready", (target.model,))

    async def tick():
        await asyncio.sleep(0.002)
        ticked.set()

    monkeypatch.setattr(inventory, "discover_provider_models", slow_discovery)
    ticker = asyncio.create_task(tick())
    with pytest.raises(provider.StandaloneHtmlProviderError) as error:
        await _generate(target, config=_config(target, overall_timeout=0.01))
    await ticker
    assert error.value.code == "standalone_html_provider_timeout"
    assert ticked.is_set()
    assert posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", COMMERCIAL)
async def test_allowlist_revoked_during_discovery_never_posts(boundary, monkeypatch, adapter):
    posts, _ = boundary
    target = _target(adapter, model="Current-ID")
    current = _config(target)

    def revoke(*_args, **_kwargs):
        nonlocal current
        current = replace(current, egress_enabled=False)
        return ModelDiscoveryResult("ready", (target.model,))

    monkeypatch.setattr(inventory, "discover_provider_models", revoke)
    with pytest.raises(provider.StandaloneHtmlProviderError) as error:
        await _generate(target, loader=lambda: current)
    assert error.value.code == "standalone_html_egress_disabled"
    assert posts == []


@pytest.mark.asyncio
async def test_discovery_failure_is_sanitized_without_post(boundary, monkeypatch):
    posts, _ = boundary

    def fail(*_args, **_kwargs):
        raise RuntimeError(PROVIDER_SECRET + USER_CONTENT)

    monkeypatch.setattr(inventory, "discover_provider_models", fail)
    with pytest.raises(provider.StandaloneHtmlProviderError) as error:
        await _generate(_target(COMMERCIAL[0], model="Current-ID"))
    assert error.value.code == "standalone_html_provider_unavailable"
    assert error.value.__context__ is None
    assert PROVIDER_SECRET not in str(error.value)
    assert USER_CONTENT not in str(error.value)
    assert posts == []
