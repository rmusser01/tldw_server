from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: dict[str, Any] | None = None, lines: list[str] | None = None):
        self.status_code = status_code
        self._json = json_obj or {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}
        self._lines = lines or [
            "data: chunk",
            "data: [DONE]",
        ]

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            req = httpx.Request("POST", "https://api-inference.huggingface.co/v1/chat/completions")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        yield from self._lines


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

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
        assert "/chat/completions" in url
        return _FakeResponse(200)

    def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
        return _FakeStreamCtx(_FakeResponse(200))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_HUGGINGFACE", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def test_huggingface_adapter_native_http_non_streaming(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter
    # Patch both the internal alias and the adapter's exposed factory
    monkeypatch.setattr(hf_mod, "_hc_create_client", lambda *a, **k: _FakeClient(*a, **k))
    monkeypatch.setattr(hf_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))
    a = HuggingFaceAdapter()
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "api_key": "k",
        "app_config": {"huggingface_api": {"api_base_url": "https://api-inference.huggingface.co/v1"}},
    }
    r = a.chat(request)
    assert r.get("object") == "chat.completion"


def test_huggingface_adapter_native_http_streaming(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter
    monkeypatch.setattr(hf_mod, "_hc_create_client", lambda *a, **k: _FakeClient(*a, **k))
    monkeypatch.setattr(hf_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))
    a = HuggingFaceAdapter()
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "api_key": "k",
        "app_config": {"huggingface_api": {"api_base_url": "https://api-inference.huggingface.co/v1"}},
        "stream": True,
    }
    chunks = list(a.stream(request))
    assert any(c.startswith("data: ") for c in chunks)
    assert sum(1 for c in chunks if "[DONE]" in c) == 1


@pytest.mark.parametrize(
    "configured_base_url",
    [None, "", "   "],
    ids=["missing", "empty", "whitespace"],
)
def test_huggingface_key_only_default_route_is_correct_for_chat_and_stream(
    monkeypatch,
    configured_base_url: str | None,
):
    """A key-only server credential must not duplicate the default /v1 path."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str, str]] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(("post", url, headers["Authorization"]))
            return _FakeResponse(200)

        def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
            assert method == "POST"
            calls.append(("stream", url, headers["Authorization"]))
            return _FakeStreamCtx(_FakeResponse(200))

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/runtime-model",
        "api_key": "stored-runtime-key",
    }
    if configured_base_url is not None:
        request["app_config"] = {
            "huggingface_api": {"api_base_url": configured_base_url}
        }
    adapter = HuggingFaceAdapter()

    assert adapter.chat(dict(request))["object"] == "chat.completion"
    chunks = list(adapter.stream(dict(request)))

    expected_url = "https://api-inference.huggingface.co/v1/chat/completions"
    assert calls == [
        ("post", expected_url, "Bearer stored-runtime-key"),
        ("stream", expected_url, "Bearer stored-runtime-key"),
    ]
    assert sum("[DONE]" in chunk for chunk in chunks) == 1


def test_huggingface_trusted_v1_override_does_not_duplicate_path_for_chat_or_stream(monkeypatch):
    """A trusted internal base ending in /v1 receives one /chat/completions suffix."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str]] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(("post", url))
            return _FakeResponse(200)

        def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
            assert method == "POST"
            calls.append(("stream", url))
            return _FakeStreamCtx(_FakeResponse(200))

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/runtime-model",
        "api_key": "stored-runtime-key",
        "base_url": "https://gateway.example/v1",
    }
    adapter = HuggingFaceAdapter()

    assert adapter.chat(dict(request))["object"] == "chat.completion"
    chunks = list(adapter.stream(dict(request)))

    expected_url = "https://gateway.example/v1/chat/completions"
    assert calls == [("post", expected_url), ("stream", expected_url)]
    assert sum("[DONE]" in chunk for chunk in chunks) == 1


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_huggingface_default_and_runtime_routes_stay_isolated(monkeypatch):
    """A default call cannot borrow a concurrent runtime endpoint or credential."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ.byok_config import runtime_base_url_override_provenance
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            with lock:
                calls.append((url, headers["Authorization"]))
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(5):
                raise TimeoutError("concurrent Hugging Face calls did not release")
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())
    adapter = HuggingFaceAdapter()
    requests = [
        {
            "messages": [{"role": "user", "content": "default"}],
            "model": "org/default-model",
            "api_key": "key-default",
        },
        {
            "messages": [{"role": "user", "content": "runtime"}],
            "model": "org/runtime-model",
            "api_key": "key-runtime",
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "api_base_url": "https://selected-runtime.example/hf-inference",
                    "_runtime_base_url_override": runtime_base_url_override_provenance(),
                    "router_base_url": "https://global-router.example/hf-inference",
                    "api_chat_path": "runtime/chat/completions",
                }
            },
        },
    ]
    tasks = [asyncio.create_task(adapter.achat(request)) for request in requests]
    try:
        assert await asyncio.to_thread(both_arrived.wait, 5)
        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert set(calls) == {
        (
            "https://api-inference.huggingface.co/v1/chat/completions",
            "Bearer key-default",
        ),
        (
            "https://selected-runtime.example/hf-inference/models/"
            "org/runtime-model/runtime/chat/completions",
            "Bearer key-runtime",
        ),
    }
    assert "global-router.example" not in repr(calls)


def test_huggingface_json_bool_marker_cannot_supersede_trusted_server_router(monkeypatch):
    """A JSON boolean cannot give api_base precedence over trusted router config."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str]] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append((url, headers["Authorization"]))
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/runtime-model",
        "api_key": "stored-runtime-key",
        "app_config": {
            "huggingface_api": {
                "use_router_url_format": "true",
                "api_base_url": "https://attacker.example/v1",
                "_runtime_base_url_override": True,
                "router_base_url": "https://trusted-router.example/hf-inference",
                "api_chat_path": "chat/completions",
            }
        },
    }

    assert HuggingFaceAdapter().chat(request)["object"] == "chat.completion"

    assert calls == [
        (
            "https://trusted-router.example/hf-inference/models/org/runtime-model/chat/completions",
            "Bearer stored-runtime-key",
        )
    ]
    assert "attacker.example" not in repr(calls)


def test_huggingface_runtime_projection_preserves_only_router_contract(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    captured: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            captured.update(url=url, json=json, headers=headers)
            calls.append({"kind": "post", "url": url, "headers": headers})
            return _FakeResponse(200)

        def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
            captured.update(method=method, url=url, json=json, headers=headers)
            calls.append({"kind": "stream", "url": url, "headers": headers})
            return _FakeStreamCtx(_FakeResponse(200))

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "huggingface_api": {
                "use_router_url_format": "true",
                "huggingface_use_router_url_format": "false",
                "api_base_url": "https://global-api-base.example/v1",
                "_runtime_base_url_override": True,
                "router_base_url": "https://router.runtime.example/hf-inference",
                "huggingface_router_base_url": "https://alias-router.runtime.example/hf-inference",
                "api_chat_path": "runtime/chat/completions",
                "huggingface_api_chat_path": "alias/chat/completions",
                "api_key": "config-secret",
                "unrelated_option": "unrelated-config",
            },
            "openai_api": {"api_key": "unrelated-secret"},
        },
    )

    app_config = byok_runtime._build_app_config("huggingface", {})
    assert app_config == {
        "huggingface_api": {
            "use_router_url_format": "true",
            "huggingface_use_router_url_format": "false",
            "api_base_url": "https://global-api-base.example/v1",
            "router_base_url": "https://router.runtime.example/hf-inference",
            "huggingface_router_base_url": "https://alias-router.runtime.example/hf-inference",
            "api_chat_path": "runtime/chat/completions",
            "huggingface_api_chat_path": "alias/chat/completions",
        }
    }
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/runtime-model",
        "api_key": "runtime-key",
        "app_config": app_config,
    }
    adapter = HuggingFaceAdapter()
    assert adapter.chat(dict(request))["object"] == "chat.completion"
    chunks = list(adapter.stream(dict(request)))

    expected_url = (
        "https://router.runtime.example/hf-inference/models/"
        "org/runtime-model/runtime/chat/completions"
    )
    assert [call["kind"] for call in calls] == ["post", "stream"]
    assert all(
        call["url"] == expected_url
        and call["headers"]["Authorization"] == "Bearer runtime-key"
        for call in calls
    )
    assert "config-secret" not in repr(captured)
    assert "unrelated-secret" not in repr(captured)
    assert "unrelated-config" not in repr(captured)
    assert "global-api-base.example" not in repr(calls)
    assert sum("[DONE]" in chunk for chunk in chunks) == 1


@pytest.mark.asyncio
async def test_huggingface_structured_fallback_base_wins_over_global_router(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_config import is_runtime_base_url_override
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        ProviderOverrideCallSnapshot,
    )
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[dict[str, Any]] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append({"kind": "post", "url": url, "headers": headers, "json": json})
            return _FakeResponse(200)

        def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append({"kind": "stream", "url": url, "headers": headers, "json": json})
            return _FakeStreamCtx(_FakeResponse(200))

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"huggingface"})
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://global-router.example/hf-inference",
                "api_chat_path": "runtime/chat/completions",
                "api_key": "global-config-secret",
                "unrelated_option": "unrelated-config-sentinel",
            },
            "openai_api": {"api_key": "unrelated-provider-secret"},
        },
    )
    runtime = ProviderCredentialRuntime(
        user_id=None,
        team_ids=None,
        org_ids=None,
        trusted_base_url_override=False,
        server_config_snapshot=byok_runtime.loaded_config_data,
        override_snapshot_resolver=lambda _provider: ProviderOverrideCallSnapshot(
            provider="huggingface",
            _override=LLMProviderOverride(
                provider="huggingface",
                api_key="selected-runtime-key",
                credential_fields={
                    "base_url": "https://selected-router.example/hf-inference"
                },
            ),
        ),
    )

    try:
        handle = await runtime.resolve("huggingface")
        provenance = handle.app_config["huggingface_api"]["_runtime_base_url_override"]
        assert is_runtime_base_url_override(provenance)
        assert not isinstance(provenance, (bool, str, int, float, dict, list, tuple))
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "org/selected-model",
            "api_key": handle.api_key,
            "app_config": handle.app_config,
        }
        adapter = HuggingFaceAdapter()
        assert adapter.chat(dict(request))["object"] == "chat.completion"
        chunks = list(adapter.stream(dict(request)))
    finally:
        await runtime.close()

    expected_url = (
        "https://selected-router.example/hf-inference/models/"
        "org/selected-model/runtime/chat/completions"
    )
    assert [call["kind"] for call in calls] == ["post", "stream"]
    assert all(
        call["url"] == expected_url
        and call["headers"]["Authorization"] == "Bearer selected-runtime-key"
        for call in calls
    )
    outbound_json = [json.dumps(call["json"], sort_keys=True) for call in calls]
    assert all("app_config" not in payload for payload in outbound_json)
    assert all("_runtime_base_url_override" not in payload for payload in outbound_json)
    captured = repr(calls)
    assert "global-router.example" not in captured
    assert "global-config-secret" not in captured
    assert "unrelated-config-sentinel" not in captured
    assert "unrelated-provider-secret" not in captured
    assert sum("[DONE]" in chunk for chunk in chunks) == 1


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_huggingface_astream_calls_keep_runtime_base_key_and_done_paired(monkeypatch):
    """Simultaneous async streams retain their own runtime endpoint and key."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        ProviderOverrideCallSnapshot,
    )
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
            assert method == "POST"
            with lock:
                calls.append((url, headers["Authorization"]))
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(5):
                raise TimeoutError("concurrent Hugging Face streams did not release")
            return _FakeStreamCtx(_FakeResponse(200))

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"huggingface"})
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://global-router.example/hf-inference",
                "api_chat_path": "runtime/chat/completions",
                "api_key": "global-config-secret",
                "unrelated_option": "global-sentinel",
            }
        },
    )

    def _runtime(label: str) -> ProviderCredentialRuntime:
        return ProviderCredentialRuntime(
            user_id=None,
            team_ids=None,
            org_ids=None,
            trusted_base_url_override=False,
            server_config_snapshot=byok_runtime.loaded_config_data,
            override_snapshot_resolver=lambda _provider: ProviderOverrideCallSnapshot(
                provider="huggingface",
                _override=LLMProviderOverride(
                    provider="huggingface",
                    api_key=f"key-{label}",
                    credential_fields={
                        "base_url": f"https://selected-{label}.example/hf-inference"
                    },
                ),
            ),
        )

    runtimes = {label: _runtime(label) for label in ("alpha", "beta")}
    tasks: list[asyncio.Task[list[str]]] = []

    async def _consume(label: str, handle) -> list[str]:
        adapter = HuggingFaceAdapter()
        return [
            chunk
            async for chunk in adapter.astream(
                {
                    "messages": [{"role": "user", "content": label}],
                    "model": f"org/model-{label}",
                    "api_key": handle.api_key,
                    "app_config": handle.app_config,
                }
            )
        ]

    try:
        handles = dict(
            zip(
                runtimes,
                await asyncio.gather(
                    *(runtime.resolve("huggingface") for runtime in runtimes.values())
                ),
                strict=True,
            )
        )
        tasks = [
            asyncio.create_task(_consume(label, handles[label]))
            for label in runtimes
        ]
        assert await asyncio.to_thread(both_arrived.wait, 5)
        release.set()
        chunks_by_request = await asyncio.gather(*tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(runtime.close() for runtime in runtimes.values()))

    assert set(calls) == {
        (
            "https://selected-alpha.example/hf-inference/models/"
            "org/model-alpha/runtime/chat/completions",
            "Bearer key-alpha",
        ),
        (
            "https://selected-beta.example/hf-inference/models/"
            "org/model-beta/runtime/chat/completions",
            "Bearer key-beta",
        ),
    }
    assert all(sum("[DONE]" in chunk for chunk in chunks) == 1 for chunks in chunks_by_request)
    captured = repr(calls)
    assert "global-router.example" not in captured
    assert "global-config-secret" not in captured
    assert "global-sentinel" not in captured


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_huggingface_runtime_configs_keep_router_and_key_paired(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            with lock:
                calls.append((url, headers["Authorization"]))
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(5):
                raise TimeoutError("concurrent Hugging Face calls did not release")
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())

    def _projected_config(label: str) -> dict[str, Any]:
        monkeypatch.setattr(
            byok_runtime,
            "loaded_config_data",
            {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": f"https://router-{label}.example/hf-inference",
                    "api_chat_path": f"{label}/chat/completions",
                    "api_key": f"must-not-use-{label}",
                }
            },
        )
        return byok_runtime._build_app_config("huggingface", {}) or {}

    configs = {label: _projected_config(label) for label in ("alpha", "beta")}
    adapter = HuggingFaceAdapter()
    tasks = [
        asyncio.create_task(
            adapter.achat(
                {
                    "messages": [{"role": "user", "content": label}],
                    "model": f"org/model-{label}",
                    "api_key": f"key-{label}",
                    "app_config": configs[label],
                }
            )
        )
        for label in ("alpha", "beta")
    ]
    assert await asyncio.to_thread(both_arrived.wait, 5)
    release.set()
    await asyncio.gather(*tasks)

    assert set(calls) == {
        (
            "https://router-alpha.example/hf-inference/models/org/model-alpha/alpha/chat/completions",
            "Bearer key-alpha",
        ),
        (
            "https://router-beta.example/hf-inference/models/org/model-beta/beta/chat/completions",
            "Bearer key-beta",
        ),
    }


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_huggingface_structured_fallbacks_keep_explicit_base_and_key_paired(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        LLMProviderOverride,
        ProviderOverrideCallSnapshot,
    )
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter

    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            with lock:
                calls.append((url, headers["Authorization"]))
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(5):
                raise TimeoutError("concurrent Hugging Face calls did not release")
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"huggingface"})
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://global-router.example/hf-inference",
                "api_chat_path": "runtime/chat/completions",
                "api_key": "global-config-secret",
                "unrelated_option": "unrelated-config-sentinel",
            },
            "openai_api": {"api_key": "unrelated-provider-secret"},
        },
    )

    def _runtime(label: str) -> ProviderCredentialRuntime:
        return ProviderCredentialRuntime(
            user_id=None,
            team_ids=None,
            org_ids=None,
            trusted_base_url_override=False,
            server_config_snapshot=byok_runtime.loaded_config_data,
            override_snapshot_resolver=lambda _provider: ProviderOverrideCallSnapshot(
                provider="huggingface",
                _override=LLMProviderOverride(
                    provider="huggingface",
                    api_key=f"key-{label}",
                    credential_fields={
                        "base_url": f"https://selected-{label}.example/hf-inference"
                    },
                ),
            ),
        )

    runtimes = {label: _runtime(label) for label in ("alpha", "beta")}
    tasks = []
    try:
        handles = dict(
            zip(
                runtimes,
                await asyncio.gather(
                    *(runtime.resolve("huggingface") for runtime in runtimes.values())
                ),
                strict=True,
            )
        )
        adapter = HuggingFaceAdapter()
        tasks = [
            asyncio.create_task(
                adapter.achat(
                    {
                        "messages": [{"role": "user", "content": label}],
                        "model": f"org/model-{label}",
                        "api_key": handles[label].api_key,
                        "app_config": handles[label].app_config,
                    }
                )
            )
            for label in runtimes
        ]
        assert await asyncio.to_thread(both_arrived.wait, 5)
        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(runtime.close() for runtime in runtimes.values()))

    assert len(calls) == 2
    assert set(calls) == {
        (
            "https://selected-alpha.example/hf-inference/models/"
            "org/model-alpha/runtime/chat/completions",
            "Bearer key-alpha",
        ),
        (
            "https://selected-beta.example/hf-inference/models/"
            "org/model-beta/runtime/chat/completions",
            "Bearer key-beta",
        ),
    }
    captured = repr(calls)
    assert "global-router.example" not in captured
    assert "global-config-secret" not in captured
    assert "unrelated-config-sentinel" not in captured
    assert "unrelated-provider-secret" not in captured


@pytest.mark.parametrize("operation", ["chat", "stream"])
@pytest.mark.parametrize(
    "model",
    [
        "../admin",
        "org/../../api/whoami-v2#",
        "org\\model",
        "org/%2e%2e/admin",
        "org/model?alt=admin",
        "org/model#fragment",
        "org/model\nforwarded",
        "//org/model",
    ],
)
def test_huggingface_router_rejects_unsafe_model_paths_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    model: str,
) -> None:
    """User model ids cannot redirect a credentialed router request."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    monkeypatch.setattr(
        hf_mod,
        "http_client_factory",
        lambda **_kwargs: pytest.fail("unsafe model must fail before HTTP dispatch"),
    )
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": model,
        "api_key": "trusted-key",
        "app_config": {
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://router.example/hf-inference",
            }
        },
    }

    with pytest.raises(ChatBadRequestError, match="model identifier") as exc_info:
        result = getattr(HuggingFaceAdapter(), operation)(request)
        if operation == "stream":
            list(result)

    assert model not in str(exc_info.value)
    assert "trusted-key" not in str(exc_info.value)


def test_huggingface_router_preserves_valid_namespaced_provider_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hugging Face's documented model:provider form remains routable."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[str] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(url)
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    HuggingFaceAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "ServiceNow-AI/Apriel-1.5-15b-Thinker:together",
            "api_key": "trusted-key",
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": "https://router.example/hf-inference",
                    "api_chat_path": "chat/completions",
                }
            },
        }
    )

    assert calls == [
        "https://router.example/hf-inference/models/"
        "ServiceNow-AI/Apriel-1.5-15b-Thinker:together/chat/completions"
    ]


def test_huggingface_router_accepts_default_style_single_leading_model_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repository's `/Qwen/...` default remains router-compatible."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[str] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(url)
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    HuggingFaceAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "/Qwen/Qwen3-235B-A22B",
            "api_key": "trusted-key",
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": "https://router.example/hf-inference",
                    "api_chat_path": "chat/completions",
                }
            },
        }
    )

    assert calls == [
        "https://router.example/hf-inference/models/"
        "Qwen/Qwen3-235B-A22B/chat/completions"
    ]


@pytest.mark.parametrize(
    "runtime_base",
    [
        None,
        "",
        "   ",
        "https://user:password@router.example/hf-inference",
        "https://router.example/hf-inference?tenant=attacker",
        "https://router.example/hf-inference#fragment",
    ],
    ids=["missing", "empty", "whitespace", "userinfo", "query", "fragment"],
)
def test_huggingface_runtime_provenance_with_missing_or_invalid_base_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    runtime_base: str | None,
) -> None:
    """A trusted runtime marker can never silently fall back to another endpoint."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ.byok_config import runtime_base_url_override_provenance
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    monkeypatch.setattr(
        hf_mod,
        "http_client_factory",
        lambda **_kwargs: pytest.fail("invalid runtime base must fail before HTTP dispatch"),
    )
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/model",
        "api_key": "trusted-key",
        "app_config": {
            "huggingface_api": {
                "use_router_url_format": "true",
                "api_base_url": runtime_base,
                "_runtime_base_url_override": runtime_base_url_override_provenance(),
            }
        },
    }

    with pytest.raises(ChatConfigurationError, match="endpoint configuration") as exc_info:
        HuggingFaceAdapter().chat(request)

    assert "user:password" not in str(exc_info.value)
    assert "tenant=attacker" not in str(exc_info.value)


def test_huggingface_infers_chat_path_from_final_selected_router_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ignored global api_base cannot influence a selected router path."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[str] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(url)
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    HuggingFaceAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "org/model",
            "api_key": "trusted-key",
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": " true ",
                    "api_base_url": "https://ignored.example/not-v1",
                    "router_base_url": "  https://selected.example/v1/  ",
                    "api_chat_path": "   ",
                    "huggingface_api_chat_path": "   ",
                }
            },
        }
    )

    assert calls == [
        "https://selected.example/v1/models/org/model/chat/completions"
    ]


@pytest.mark.concurrent
@pytest.mark.parametrize("boundary", ["model_path", "runtime_base"])
def test_concurrent_invalid_huggingface_request_cannot_redirect_legitimate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """A rejected request cannot mutate or add a credentialed in-flight dispatch."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.AuthNZ.byok_config import runtime_base_url_override_provenance
    from tldw_Server_API.app.core.Chat.Chat_Deps import (
        ChatBadRequestError,
        ChatConfigurationError,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[tuple[str, str, int]] = []
    lock = threading.Lock()
    arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            with lock:
                calls.append((url, headers["Authorization"], json["seed"]))
                arrived.set()
            if not release.wait(10):
                raise TimeoutError("legitimate Hugging Face call was not released")
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())
    legitimate = {
        "messages": [{"role": "user", "content": "legitimate"}],
        "model": "org/legitimate-model",
        "seed": 1907,
        "api_key": "legitimate-key",
        "app_config": {
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://legitimate.example/hf-inference",
                "api_chat_path": "chat/completions",
            }
        },
    }
    if boundary == "model_path":
        malicious = legitimate | {
            "model": "../../api/whoami-v2#",
            "seed": 7331,
            "api_key": "must-not-dispatch",
        }
        expected_error = ChatBadRequestError
    else:
        malicious = legitimate | {
            "model": "org/malicious-model",
            "seed": 7331,
            "api_key": "must-not-dispatch",
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "api_base_url": "https://attacker@malicious.example/hf-inference",
                    "_runtime_base_url_override": runtime_base_url_override_provenance(),
                }
            },
        }
        expected_error = ChatConfigurationError

    with ThreadPoolExecutor(max_workers=2) as executor:
        legitimate_future = executor.submit(HuggingFaceAdapter().chat, legitimate)
        try:
            assert arrived.wait(10)
            malicious_future = executor.submit(HuggingFaceAdapter().chat, malicious)
            with pytest.raises(expected_error):
                malicious_future.result(timeout=5)
        finally:
            release.set()
        assert legitimate_future.result(timeout=10)["object"] == "chat.completion"

    assert calls == [
        (
            "https://legitimate.example/hf-inference/models/"
            "org/legitimate-model/chat/completions",
            "Bearer legitimate-key",
            1907,
        )
    ]


@pytest.mark.concurrent
def test_concurrent_huggingface_extra_headers_remain_safe_and_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header filtering is deterministic while credentialed calls overlap."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[tuple[str, str, str, int]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            assert "Host" not in headers
            assert "Proxy-Authorization" not in headers
            assert "Content-Length" not in headers
            assert "X_API_KEY" not in headers
            assert "X_GOOG_API_KEY" not in headers
            with lock:
                calls.append(
                    (
                        url,
                        headers["Authorization"],
                        headers["X-Provider-Extension"],
                        json["seed"],
                    )
                )
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent Hugging Face calls were not released")
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())

    def _request(label: str, seed: int) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": label}],
            "model": f"org/model-{label}",
            "seed": seed,
            "api_key": f"key-{label}",
            "extra_headers": {
                "Host": f"attacker-{label}.example",
                "Proxy-Authorization": f"Bearer attacker-{label}",
                "Content-Length": "999",
                "X_API_KEY": f"attacker-api-key-{label}",
                "X_GOOG_API_KEY": f"attacker-google-key-{label}",
                "X-Provider-Extension": label,
            },
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": f"https://router-{label}.example/hf-inference",
                    "api_chat_path": "chat/completions",
                }
            },
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(HuggingFaceAdapter().chat, _request("alpha", 1907)),
            executor.submit(HuggingFaceAdapter().chat, _request("beta", 7331)),
        ]
        try:
            assert both_arrived.wait(10)
        finally:
            release.set()
        assert all(future.result(timeout=10)["object"] == "chat.completion" for future in futures)

    assert len(calls) == 2
    assert set(calls) == {
        (
            "https://router-alpha.example/hf-inference/models/org/model-alpha/chat/completions",
            "Bearer key-alpha",
            "alpha",
            1907,
        ),
        (
            "https://router-beta.example/hf-inference/models/org/model-beta/chat/completions",
            "Bearer key-beta",
            "beta",
            7331,
        ),
    }


def test_huggingface_adapter_drops_proxy_normalized_underscore_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Underscore aliases cannot become credential headers after proxy normalization."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[dict[str, str]] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            del url, json
            calls.append(dict(headers))
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    result = HuggingFaceAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "org/model",
            "api_key": "trusted-key",
            "extra_headers": {
                "X_API_KEY": "attacker-api-key",
                "X_GOOG_API_KEY": "attacker-google-key",
                "Proxy_Authorization": "attacker-proxy-key",
                "X-Provider-Extension": "kept",
            },
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": "https://router.example/hf-inference",
                    "api_chat_path": "chat/completions",
                }
            },
        }
    )

    assert result["object"] == "chat.completion"
    assert calls == [
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer trusted-key",
            "X-Provider-Extension": "kept",
        }
    ]


@pytest.mark.parametrize("operation", ["chat", "stream"])
def test_huggingface_debug_logs_never_include_header_values(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Header values stay secret even when adapter debug logging is enabled."""
    from loguru import logger

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[dict[str, str]] = []
    records: list[str] = []

    class _CaptureClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(dict(headers))
            return _FakeResponse(200)

        def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
            calls.append(dict(headers))
            return _FakeStreamCtx(_FakeResponse(200))

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _CaptureClient())
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "org/model",
        "api_key": "trusted-api-key-sentinel",
        "extra_headers": {"X-Provider-Extension": "provider-header-sentinel"},
        "app_config": {
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://router.example/hf-inference",
                "api_chat_path": "chat/completions",
            }
        },
    }

    try:
        result = getattr(HuggingFaceAdapter(), operation)(request)
        if operation == "stream":
            list(result)
    finally:
        logger.remove(sink_id)

    assert calls == [
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer trusted-api-key-sentinel",
            "X-Provider-Extension": "provider-header-sentinel",
        }
    ]
    header_logs = "\n".join(
        record for record in records if "HuggingFace headers:" in record
    )
    assert header_logs
    assert "Authorization" in header_logs
    assert "X-Provider-Extension" in header_logs
    for header_value in (
        "application/json",
        "trusted-api-key-sentinel",
        "provider-header-sentinel",
    ):
        assert header_value not in header_logs


@pytest.mark.concurrent
def test_concurrent_huggingface_debug_logs_never_cross_or_expose_header_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping adapter calls expose neither request's header values in logs."""
    from loguru import logger

    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
        HuggingFaceAdapter,
    )

    calls: list[tuple[str, str, str]] = []
    records: list[str] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    def _capture(message: Any) -> None:
        with lock:
            records.append(str(message))

    class _GatedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            with lock:
                calls.append(
                    (
                        url,
                        headers["Authorization"],
                        headers["X-Provider-Extension"],
                    )
                )
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent Hugging Face log calls were not released")
            return _FakeResponse(200)

    monkeypatch.setattr(hf_mod, "http_client_factory", lambda **_kwargs: _GatedClient())
    sink_id = logger.add(_capture, level="DEBUG", format="{message}")

    def _request(label: str) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": label}],
            "model": f"org/model-{label}",
            "api_key": f"secret-key-{label}",
            "extra_headers": {
                "X-Provider-Extension": f"secret-extension-{label}",
            },
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": f"https://router-{label}.example/hf-inference",
                    "api_chat_path": "chat/completions",
                }
            },
        }

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(HuggingFaceAdapter().chat, _request(label))
                for label in ("alpha", "beta")
            ]
            try:
                assert both_arrived.wait(10)
            finally:
                release.set()
            assert all(
                future.result(timeout=10)["object"] == "chat.completion"
                for future in futures
            )
    finally:
        release.set()
        logger.remove(sink_id)

    assert set(calls) == {
        (
            "https://router-alpha.example/hf-inference/models/org/model-alpha/chat/completions",
            "Bearer secret-key-alpha",
            "secret-extension-alpha",
        ),
        (
            "https://router-beta.example/hf-inference/models/org/model-beta/chat/completions",
            "Bearer secret-key-beta",
            "secret-extension-beta",
        ),
    }
    header_logs = "\n".join(
        record for record in records if "HuggingFace headers:" in record
    )
    assert header_logs.count("HuggingFace headers:") == 2
    for header_value in (
        "application/json",
        "secret-key-alpha",
        "secret-key-beta",
        "secret-extension-alpha",
        "secret-extension-beta",
    ):
        assert header_value not in header_logs
