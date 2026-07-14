from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List

import pytest


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
            req = httpx.Request("POST", "https://api-inference.huggingface.co/v1/chat/completions")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        for l in self._lines:
            yield l


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
        assert "/chat/completions" in url
        return _FakeResponse(200)

    def stream(self, method: str, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        return _FakeStreamCtx(_FakeResponse(200))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_HUGGINGFACE", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def test_huggingface_adapter_native_http_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
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
    from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import HuggingFaceAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_mod
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
        fallback_resolver=lambda _provider: byok_runtime.ServerFallbackCredentials(
            api_key="selected-runtime-key",
            credential_fields={"base_url": "https://selected-router.example/hf-inference"},
            auth_source="api_key",
        ),
    )

    try:
        handle = await runtime.resolve("huggingface")
        assert handle.app_config["huggingface_api"]["_runtime_base_url_override"] is True
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
    captured = repr(calls)
    assert "global-router.example" not in captured
    assert "global-config-secret" not in captured
    assert "unrelated-config-sentinel" not in captured
    assert "unrelated-provider-secret" not in captured
    assert sum("[DONE]" in chunk for chunk in chunks) == 1


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
            fallback_resolver=lambda _provider: byok_runtime.ServerFallbackCredentials(
                api_key=f"key-{label}",
                credential_fields={
                    "base_url": f"https://selected-{label}.example/hf-inference"
                },
                auth_source="api_key",
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
