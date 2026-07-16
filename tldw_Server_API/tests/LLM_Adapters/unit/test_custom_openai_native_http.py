from __future__ import annotations

from typing import Any, Dict, List

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.exceptions import EgressPolicyError


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: Dict[str, Any] | None = None, lines: List[str] | None = None):
        self.status_code = status_code
        self._json = json_obj or {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}
        self._lines = lines or [
            "data: chunk",
            "data: [DONE]",
        ]

    def raise_for_status(self):
        if 400 <= self.status_code:
            import httpx
            req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        for l in self._lines:
            yield l

    def close(self):
        return None


class _FakeStreamCtx:
    def __init__(self, r: _FakeResponse):
        self._r = r

    def __enter__(self):
        return self._r

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        assert "chat/completions" in url
        return _FakeResponse(200)

    def stream(self, method: str, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        return _FakeStreamCtx(_FakeResponse(200))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_CUSTOM_OPENAI", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def test_custom_openai_adapter_native_http_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter
    a = CustomOpenAIAdapter()
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://127.0.0.1:11434/v1")

    def _fetch(**kwargs):
        assert kwargs["configured_endpoint"].matches(kwargs["url"])
        return _FakeResponse(200)

    a.http_fetcher = _fetch
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "my-model",
        "api_key": "k",
        "app_config": {"custom_openai_api": {"api_ip": "http://127.0.0.1:11434/v1"}},
    }
    r = a.chat(request)
    assert r.get("object") == "chat.completion"


def test_custom_openai_adapter_native_http_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter2
    a = CustomOpenAIAdapter2()
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP_2", "http://127.0.0.1:11434")

    def _stream(**kwargs):
        assert kwargs["configured_endpoint"].matches(kwargs["url"])
        return _FakeStreamCtx(_FakeResponse(200))

    a.http_streamer = _stream
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "my-model",
        "api_key": "k",
        "app_config": {"custom_openai_api_2": {"api_ip": "http://127.0.0.1:11434"}},
        "stream": True,
    }
    chunks = list(a.stream(request))
    assert any(c.startswith("data: ") for c in chunks)
    assert sum(1 for c in chunks if "[DONE]" in c) == 1


def test_custom_openai_adapter_fallback_accepts_canonical_env(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://127.0.0.1:8000/v1")
    adapter = CustomOpenAIAdapter()

    assert adapter._resolve_base({"app_config": {}}) == "http://127.0.0.1:8000/v1"


def test_custom_openai_adapter_fallback_preserves_numbered_env_compatibility(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter2

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP_2", "http://127.0.0.1:8002/v1")
    adapter = CustomOpenAIAdapter2()

    assert adapter._resolve_base({"app_config": {}}) == "http://127.0.0.1:8002/v1"


def test_custom_openai_adapter_factory_supports_numbered_endpoint_env(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP_37", "http://127.0.0.1:8037/v1")

    registry = ChatProviderRegistry()
    adapter = registry.get_adapter("custom-openai-api-37")

    assert adapter is not None
    assert adapter.name == "custom-openai-api-37"
    assert adapter.config_section == "custom_openai_api_37"
    assert adapter._resolve_base({"app_config": {}}) == "http://127.0.0.1:8037/v1"


def test_numbered_custom_openai_adapters_require_explicit_base_url(monkeypatch):
    from tldw_Server_API.app.core.custom_openai_providers import custom_openai_endpoint_env_keys
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
        CustomOpenAIAdapter2,
        make_custom_openai_adapter_class,
    )

    for env_key in (*custom_openai_endpoint_env_keys(2), *custom_openai_endpoint_env_keys(37)):
        monkeypatch.delenv(env_key, raising=False)

    with pytest.raises(RuntimeError, match="requires an explicit base URL"):
        CustomOpenAIAdapter2()._resolve_base({"app_config": {}})

    with pytest.raises(RuntimeError, match="requires an explicit base URL"):
        make_custom_openai_adapter_class(37)()._resolve_base({"app_config": {}})


@pytest.mark.parametrize("provenance", ["byok", "request_override"])
def test_configured_custom_explicit_endpoint_uses_ordinary_checked_egress_without_scope(
    monkeypatch,
    provenance: str,
):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter

    captured: dict[str, Any] = {}

    def _fetch(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(200)

    adapter = CustomOpenAIAdapter()
    adapter.http_fetcher = _fetch
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "model",
            "base_url": "http://explicit-user-endpoint:18090/v1",
            "_endpoint_provenance": provenance,
            "configured_endpoint_scope": object(),
            "http_fetcher": object(),
        }
    )

    assert captured["url"] == "http://explicit-user-endpoint:18090/v1/chat/completions"
    assert captured["configured_endpoint"] is None


@pytest.mark.parametrize(
    "endpoint_key",
    [
        "base_url",
        "api_base_url",
        "api_base",
        "api_url",
        "api_ip",
        "custom_openai_api_ip",
        "custom_openai_api_base",
        "custom_openai_api_url",
        "custom_openai_api_base_url",
        "custom_openai_base_url",
        "custom_openai_api_ip_1",
    ],
)
def test_configured_custom_resolves_raw_endpoint_then_strips_reserved_context(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_key: str,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter

    validated: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    def _validate(_provider: str, request: dict[str, Any]) -> dict[str, Any]:
        validated.append(dict(request))
        return request

    def _fetch(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(200)

    monkeypatch.setattr(custom_openai_adapter, "validate_payload", _validate)
    adapter = custom_openai_adapter.CustomOpenAIAdapter()
    adapter.http_fetcher = _fetch
    reserved = {
        endpoint_key,
        "app_config",
        "_endpoint_provenance",
        "endpoint_provenance",
        "configured_endpoint_base_url",
        "configured_endpoint_scope",
        "configured_endpoint",
        "http_client_factory",
        "http_fetcher",
        "http_streamer",
        "trusted_base_url_override",
    }
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "model",
            endpoint_key: "http://explicit-user-endpoint:18092/v1",
            "app_config": {
                "custom_openai_api": {"api_ip": "http://stale-request-config:18093/v1"}
            },
            "_endpoint_provenance": "request_override",
            "endpoint_provenance": "server_config",
            "configured_endpoint_base_url": "http://attacker.invalid/v1",
            "configured_endpoint_scope": object(),
            "configured_endpoint": object(),
            "http_client_factory": object(),
            "http_fetcher": object(),
            "http_streamer": object(),
            "trusted_base_url_override": True,
        }
    )

    assert captured["url"] == "http://explicit-user-endpoint:18092/v1/chat/completions"
    assert captured["configured_endpoint"] is None
    assert reserved.isdisjoint(validated[0])
    assert reserved.isdisjoint(captured["json"])


def test_configured_custom_byok_app_config_is_used_then_stripped_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter

    validated: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    def _validate(_provider: str, request: dict[str, Any]) -> dict[str, Any]:
        validated.append(dict(request))
        return request

    def _fetch(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(200)

    monkeypatch.setattr(custom_openai_adapter, "validate_payload", _validate)
    adapter = custom_openai_adapter.CustomOpenAIAdapter()
    adapter.http_fetcher = _fetch
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "model",
            "app_config": {
                "custom_openai_api": {"api_base_url": "http://byok-endpoint:18094/v1"}
            },
            "_endpoint_provenance": "byok",
        }
    )

    assert captured["url"] == "http://byok-endpoint:18094/v1/chat/completions"
    assert captured["configured_endpoint"] is None
    assert "app_config" not in validated[0]
    assert "_endpoint_provenance" not in validated[0]
    assert "app_config" not in captured["json"]


@pytest.mark.parametrize("adapter_name", ["novita", "poe", "together"])
def test_public_custom_subclasses_never_use_configured_local_transport(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
):
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.providers import custom_openai_adapter

    monkeypatch.setattr(custom_openai_adapter, "http_client_factory", _FakeClient)

    adapter = ChatProviderRegistry().get_adapter(adapter_name)
    assert adapter is not None
    adapter.http_fetcher = lambda **_kwargs: pytest.fail("public provider used configured fetcher")
    result = adapter.chat({"messages": [{"role": "user", "content": "hi"}], "model": "model"})

    assert result["choices"][0]["message"]["content"] == "ok"


def test_public_custom_subclass_does_not_accept_configured_custom_endpoint_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import NovitaAdapter

    monkeypatch.delenv("NOVITA_BASE_URL", raising=False)
    monkeypatch.delenv("NOVITA_API_BASE_URL", raising=False)
    adapter = NovitaAdapter()
    with pytest.raises(ChatBadRequestError):
        adapter.chat(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "model",
                "novita_base_url": "http://attacker.invalid/v1",
            }
        )


@pytest.mark.parametrize("reason_code", ["origin_mismatch", "tls_pin_mismatch", "dns_unresolved"])
def test_configured_custom_preserves_egress_policy_error_sync_and_async(
    monkeypatch,
    reason_code: str,
):
    import asyncio
    from contextlib import contextmanager

    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://configured-custom:18091/v1")
    error = EgressPolicyError("sanitized", reason_code=reason_code)

    def _fetch(**_kwargs):
        raise error

    @contextmanager
    def _stream(**_kwargs):
        raise error
        yield  # pragma: no cover

    adapter = CustomOpenAIAdapter()
    adapter.http_fetcher = _fetch
    adapter.http_streamer = _stream
    request = {"messages": [{"role": "user", "content": "hi"}], "model": "model"}

    with pytest.raises(EgressPolicyError) as sync_exc:
        adapter.chat(request)
    assert sync_exc.value.reason_code == reason_code

    with pytest.raises(EgressPolicyError) as stream_exc:
        list(adapter.stream(request))
    assert stream_exc.value.reason_code == reason_code

    with pytest.raises(EgressPolicyError) as async_exc:
        asyncio.run(adapter.achat(request))
    assert async_exc.value.reason_code == reason_code

    async def _consume() -> None:
        async for _item in adapter.astream(request):
            pass

    with pytest.raises(EgressPolicyError) as astream_exc:
        asyncio.run(_consume())
    assert astream_exc.value.reason_code == reason_code
