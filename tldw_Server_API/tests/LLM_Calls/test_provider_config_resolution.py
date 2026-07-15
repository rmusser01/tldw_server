from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


@pytest.mark.parametrize(
    ("provider", "section", "field", "url"),
    [
        ("local_llm", "local_llm", "api_ip", "http://local-box:18080/v1"),
        ("llama-cpp", "llama_api", "api_ip", "http://llama-box:18081/completion"),
        ("koboldcpp", "kobold_api", "api_ip", "http://kobold-box:15001/api/v1/generate"),
        ("oobabooga", "ooba_api", "api_ip", "http://ooba-box:15002/v1"),
        ("tabby-api", "tabby_api", "api_ip", "http://tabby-box:15003/v1"),
        ("vllm", "vllm_api", "api_ip", "http://vllm-box:18000/v1"),
        ("ollama", "ollama_api", "api_url", "http://ollama-box:11434/v1"),
        ("aphrodite", "aphrodite_api", "api_ip", "http://aphrodite-box:18082/v1"),
        ("custom_openai_api", "custom_openai_api", "api_ip", "http://custom-box:18083/v1"),
        ("custom-openai-api-37", "custom_openai_api_37", "api_ip", "http://slot37-box:18084/v1"),
    ],
)
def test_trusted_provider_endpoint_resolves_aliases_from_one_current_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    section: str,
    field: str,
    url: str,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_config_resolution as resolution

    calls = 0

    def _settings() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {section: {field: url}}

    monkeypatch.setattr(resolution, "load_settings", _settings)

    endpoint = resolution.resolve_trusted_provider_endpoint(provider)

    assert endpoint.base_url == url
    assert endpoint.scope == ConfiguredEndpointScope.from_url(url)
    assert calls == 1
    with pytest.raises(FrozenInstanceError):
        endpoint.base_url = "http://attacker.invalid"  # type: ignore[misc]


@pytest.mark.parametrize(
    "env_key",
    (
        "LOCAL_LLM_API_URL",
        "LOCAL_LLM_API_BASE",
        "LOCAL_LLM_API_IP",
        "LOCAL_LLM_BASE_URL",
    ),
)
def test_local_llm_trusted_endpoint_honors_all_environment_aliases(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_config_resolution as resolution

    for key in (
        "LOCAL_LLM_API_URL",
        "LOCAL_LLM_API_BASE",
        "LOCAL_LLM_API_IP",
        "LOCAL_LLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_key, "http://fresh-local:19000/v1")
    monkeypatch.setattr(
        resolution,
        "load_settings",
        lambda: {"local_llm": {"api_ip": "http://stale-config:19001/v1"}},
    )

    endpoint = resolution.resolve_trusted_provider_endpoint("local-llm")

    assert endpoint.base_url == "http://fresh-local:19000/v1"
    assert endpoint.scope.matches("http://fresh-local:19000/v1/chat/completions")


def test_trusted_endpoint_is_fresh_after_config_cache_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_config_resolution as resolution

    snapshots = iter(
        (
            {"llama_api": {"api_ip": "http://old-llama:8080/v1"}},
            {"llama_api": {"api_ip": "http://new-llama:18080/v1"}},
        )
    )
    monkeypatch.setattr(resolution, "load_settings", lambda: next(snapshots))

    first = resolution.resolve_trusted_provider_endpoint("llama.cpp")
    second = resolution.resolve_trusted_provider_endpoint("llama.cpp")

    assert first.base_url == "http://old-llama:8080/v1"
    assert second.base_url == "http://new-llama:18080/v1"
    assert second.scope.port == 18080


def test_public_custom_subclasses_are_not_configured_local_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_config_resolution as resolution

    monkeypatch.setattr(resolution, "load_settings", lambda: {})

    for provider in ("novita", "poe", "together", "openai"):
        assert resolution.resolve_trusted_provider_endpoint(provider) is None


def test_local_adapter_boundary_discards_request_context_and_pairs_fresh_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
    from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters

    trusted = TrustedProviderEndpoint(
        base_url="http://fresh-server:18080/v1",
        scope=ConfiguredEndpointScope.from_url("http://fresh-server:18080/v1"),
    )
    monkeypatch.setattr(local_adapters, "resolve_trusted_provider_endpoint", lambda _name: trusted)
    captured: dict[str, Any] = {}

    def _handler(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    adapter = local_adapters.LocalLLMAdapter()
    adapter._handler = _handler
    adapter.http_fetcher = object()
    adapter.http_streamer = object()
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "app_config": {"local_llm": {"api_ip": "http://stale-request:9999/v1"}},
            "configured_endpoint_base_url": "http://attacker:1/v1",
            "configured_endpoint_scope": ConfiguredEndpointScope.from_url("http://attacker:1/v1"),
            "http_fetcher": object(),
            "http_streamer": object(),
            "api_url": "http://attacker:2/v1",
        }
    )

    assert captured["configured_endpoint_base_url"] == trusted.base_url
    assert captured["configured_endpoint_scope"] is trusted.scope
    assert captured["http_fetcher"] is adapter.http_fetcher
    assert captured["http_streamer"] is adapter.http_streamer
    assert captured.get("api_url") is None


@pytest.mark.parametrize(
    ("provider", "section"),
    [
        ("local-llm", "local_llm"),
        ("llama.cpp", "llama_api"),
        ("kobold", "kobold_api"),
        ("ooba", "ooba_api"),
        ("tabbyapi", "tabby_api"),
        ("vllm", "vllm_api"),
        ("ollama", "ollama_api"),
        ("aphrodite", "aphrodite_api"),
    ],
)
def test_direct_registry_local_adapters_scope_sync_and_async_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    section: str,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
    from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters

    trusted = TrustedProviderEndpoint(
        base_url=f"http://{provider.replace('.', '-')}-lan:18080/v1",
        scope=ConfiguredEndpointScope.from_url(
            f"http://{provider.replace('.', '-')}-lan:18080/v1"
        ),
    )
    monkeypatch.setattr(local_adapters, "resolve_trusted_provider_endpoint", lambda _name: trusted)
    calls: list[dict[str, Any]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            if provider == "kobold":
                return {"results": [{"text": "ok"}]}
            return {"choices": [{"message": {"content": "ok"}}]}

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
            yield "data: [DONE]"

        def close(self) -> None:
            return None

    def _fetch(**kwargs: Any) -> _Response:
        calls.append(kwargs)
        return _Response()

    @contextmanager
    def _stream(**kwargs: Any):
        calls.append(kwargs)
        yield _Response()

    class _Client:
        def close(self) -> None:
            return None

    adapter = ChatProviderRegistry().get_adapter(provider)
    assert adapter is not None
    adapter.http_fetcher = _fetch
    adapter.http_streamer = _stream
    adapter.http_client_factory = lambda **_kwargs: _Client()
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "model",
        "app_config": {
            section: {
                "api_ip": "http://stale-request:9999/v1",
                "api_url": "http://stale-request:9999/v1",
                "model": "model",
            }
        },
    }

    adapter.chat(request)
    list(adapter.stream(request))
    asyncio.run(adapter.achat(request))

    async def _collect() -> list[str]:
        return [item async for item in adapter.astream(request)]

    asyncio.run(_collect())

    assert calls
    assert all(call["configured_endpoint"] is trusted.scope for call in calls)
    assert all(call["url"].startswith(trusted.base_url) for call in calls)


def test_direct_registry_custom_slot_37_scopes_all_dispatch_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import TrustedProviderEndpoint
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter

    trusted = TrustedProviderEndpoint(
        base_url="http://slot37-lan:18370/v1",
        scope=ConfiguredEndpointScope.from_url("http://slot37-lan:18370/v1"),
    )
    monkeypatch.setattr(
        custom_openai_adapter,
        "resolve_trusted_provider_endpoint",
        lambda _name: trusted,
    )
    calls: list[dict[str, Any]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

        def iter_lines(self):
            yield "data: [DONE]"

        def close(self) -> None:
            return None

    def _fetch(**kwargs: Any) -> _Response:
        calls.append(kwargs)
        return _Response()

    @contextmanager
    def _stream(**kwargs: Any):
        calls.append(kwargs)
        yield _Response()

    adapter = ChatProviderRegistry().get_adapter("custom-openai-api-37")
    assert adapter is not None
    adapter.http_fetcher = _fetch
    adapter.http_streamer = _stream
    request = {"messages": [{"role": "user", "content": "hi"}], "model": "model"}

    adapter.chat(request)
    list(adapter.stream(request))
    asyncio.run(adapter.achat(request))

    async def _collect() -> list[str]:
        return [item async for item in adapter.astream(request)]

    asyncio.run(_collect())

    assert len(calls) == 4
    assert all(call["configured_endpoint"] is trusted.scope for call in calls)
