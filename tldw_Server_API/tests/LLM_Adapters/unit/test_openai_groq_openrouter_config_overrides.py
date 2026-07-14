from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict

import pytest


class _FakeResponse:
    def __init__(self, json_obj: Dict[str, Any] | None = None):
        self.status_code = 200
        self._json = json_obj or {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        return iter(
            [
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                "data: [DONE]",
            ]
        )


class _FakeClient:
    def __init__(self, captured: Dict[str, Any]):
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any]):
        self._captured["url"] = url
        self._captured["headers"] = headers
        self._captured["json"] = json
        self._captured.setdefault("calls", []).append(
            {"kind": "post", "url": url, "headers": headers, "json": json}
        )
        return _FakeResponse()

    def stream(self, method: str, url: str, headers: Dict[str, str], json: Dict[str, Any]):
        self._captured["stream_method"] = method
        self._captured["url"] = url
        self._captured["headers"] = headers
        self._captured["json"] = json
        self._captured.setdefault("calls", []).append(
            {"kind": "stream", "url": url, "headers": headers, "json": json}
        )
        return _FakeResponse()


@pytest.fixture(autouse=True)
def _enable_adapters(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_GROQ", "1")
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENROUTER", "1")
    yield


def test_openai_app_config_base_url_and_timeout(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as mod

    captured: Dict[str, Any] = {}

    def _factory(*args, timeout: float | None = None, **kwargs):
        captured["timeout"] = timeout
        return _FakeClient(captured)

    monkeypatch.setattr(mod, "http_client_factory", _factory, raising=True)

    a = OpenAIAdapter()
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "api_key": "k",
        "app_config": {"openai_api": {"api_base_url": "https://mock.openai.local/v1", "api_timeout": 12}},
    }
    _ = a.chat(req)
    assert captured.get("timeout") == 12
    assert str(captured.get("url", "")).startswith("https://mock.openai.local/v1/chat/completions")


def test_openai_app_config_auth_headers_nonstream_and_stream(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as mod
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

    captured: dict[str, Any] = {}
    monkeypatch.setattr(mod, "http_client_factory", lambda **_kwargs: _FakeClient(captured), raising=True)
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "api_key": "runtime-key",
        "app_config": {
            "openai_api": {
                "api_base_url": "https://runtime.openai.example/v1",
                "org_id": "runtime-org",
                "project_id": "runtime-project",
            }
        },
    }

    adapter = OpenAIAdapter()
    assert adapter.chat(dict(request))["choices"][0]["message"]["content"] == "ok"
    chunks = list(adapter.stream(dict(request)))

    expected_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer runtime-key",
        "OpenAI-Organization": "runtime-org",
        "OpenAI-Project": "runtime-project",
    }
    assert [call["kind"] for call in captured["calls"]] == ["post", "stream"]
    assert all(
        call["url"] == "https://runtime.openai.example/v1/chat/completions"
        and call["headers"] == expected_headers
        for call in captured["calls"]
    )
    assert sum(chunk.strip() == "data: [DONE]" for chunk in chunks) == 1


@pytest.mark.asyncio
async def test_rag_runtime_openai_fallback_reaches_real_adapter_without_unrelated_secrets(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as mod
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.RAG.rag_service.generation import GenerationConfig, LLMGenerator

    captured: dict[str, Any] = {}
    monkeypatch.setattr(mod, "http_client_factory", lambda **_kwargs: _FakeClient(captured), raising=True)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "openai_api": {
                "model": "gpt-4o-mini",
                "api_key": "config-secret",
                "organization": "server-org-sentinel",
                "organization_id": "server-org-id-sentinel",
                "project": "server-project-sentinel",
            },
            "anthropic_api": {"api_key": "unrelated-secret"},
        },
    )
    runtime = ProviderCredentialRuntime(
        user_id=None,
        team_ids=None,
        org_ids=None,
        trusted_base_url_override=False,
        fallback_resolver=lambda _provider: byok_runtime.ServerFallbackCredentials(
            api_key="runtime-key",
            credential_fields={
                "base_url": "https://runtime.openai.example/v1",
                "org_id": "runtime-org",
                "project_id": "runtime-project",
            },
            auth_source="api_key",
        ),
    )
    generator = LLMGenerator(GenerationConfig(provider="openai", model="gpt-4o-mini"))

    try:
        response = await generator._call_llm("question", credential_runtime=runtime)
        stream = await generator._call_llm(
            "question",
            credential_runtime=runtime,
            streaming=True,
        )
        chunks = [chunk async for chunk in stream]
    finally:
        await runtime.close()

    assert response["choices"][0]["message"]["content"] == "ok"
    assert [call["kind"] for call in captured["calls"]] == ["post", "stream"]
    assert all(
        call["headers"]["Authorization"] == "Bearer runtime-key"
        and call["headers"]["OpenAI-Organization"] == "runtime-org"
        and call["headers"]["OpenAI-Project"] == "runtime-project"
        for call in captured["calls"]
    )
    assert "server-org-sentinel" not in repr(captured)
    assert "server-org-id-sentinel" not in repr(captured)
    assert "server-project-sentinel" not in repr(captured)
    assert "config-secret" not in repr(captured)
    assert "unrelated-secret" not in repr(captured)
    assert sum(chunk.strip() == "data: [DONE]" for chunk in chunks) == 1


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_rag_openai_fallbacks_keep_key_base_and_headers_paired(monkeypatch):
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as mod
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import ProviderCredentialRuntime
    from tldw_Server_API.app.core.RAG.rag_service.generation import GenerationConfig, LLMGenerator

    calls: list[tuple[str, str, str, str]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _GatedClient(_FakeClient):
        def post(self, url: str, headers: dict[str, str], json: dict[str, Any]):
            with lock:
                calls.append(
                    (
                        url,
                        headers["Authorization"],
                        headers["OpenAI-Organization"],
                        headers["OpenAI-Project"],
                    )
                )
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(5):
                raise TimeoutError("concurrent OpenAI calls did not release")
            return _FakeResponse()

    monkeypatch.setattr(mod, "http_client_factory", lambda **_kwargs: _GatedClient({}), raising=True)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", {})

    def _runtime(label: str) -> ProviderCredentialRuntime:
        return ProviderCredentialRuntime(
            user_id=None,
            team_ids=None,
            org_ids=None,
            trusted_base_url_override=False,
            fallback_resolver=lambda _provider: byok_runtime.ServerFallbackCredentials(
                api_key=f"key-{label}",
                credential_fields={
                    "base_url": f"https://{label}.openai.example/v1",
                    "org_id": f"org-{label}",
                    "project_id": f"project-{label}",
                },
                auth_source="api_key",
            ),
        )

    runtimes = [_runtime("alpha"), _runtime("beta")]
    generator = LLMGenerator(GenerationConfig(provider="openai", model="gpt-4o-mini"))
    tasks = [
        asyncio.create_task(generator._call_llm("question", credential_runtime=runtime))
        for runtime in runtimes
    ]
    assert await asyncio.to_thread(both_arrived.wait, 5)
    release.set()
    try:
        await asyncio.gather(*tasks)
    finally:
        await asyncio.gather(*(runtime.close() for runtime in runtimes))

    assert set(calls) == {
        (
            "https://alpha.openai.example/v1/chat/completions",
            "Bearer key-alpha",
            "org-alpha",
            "project-alpha",
        ),
        (
            "https://beta.openai.example/v1/chat/completions",
            "Bearer key-beta",
            "org-beta",
            "project-beta",
        ),
    }


@pytest.mark.parametrize("legacy_key", ["api_base", "base_url"])
def test_openai_app_config_legacy_base_url_keys(monkeypatch, legacy_key: str):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as mod

    captured: Dict[str, Any] = {}

    def _factory(*args, timeout: float | None = None, **kwargs):
        captured["timeout"] = timeout
        return _FakeClient(captured)

    monkeypatch.setattr(mod, "http_client_factory", _factory, raising=True)

    a = OpenAIAdapter()
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "api_key": "k",
        "app_config": {"openai_api": {legacy_key: "https://legacy.openai.local/v1", "api_timeout": 17}},
    }
    _ = a.chat(req)
    assert captured.get("timeout") == 17
    assert str(captured.get("url", "")).startswith("https://legacy.openai.local/v1/chat/completions")


def test_groq_app_config_base_url_and_timeout(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter import GroqAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter as mod

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(mod, "http_client_factory", lambda *a, timeout=None, **k: (captured.setdefault("timeout", timeout) or _FakeClient(captured)) and _FakeClient(captured), raising=True)

    a = GroqAdapter()
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "llama3-8b",
        "api_key": "k",
        "app_config": {"groq_api": {"api_base_url": "https://api.groq.test/openai/v1", "api_timeout": 22}},
    }
    _ = a.chat(req)
    assert captured.get("timeout") == 22
    assert str(captured.get("url", "")).startswith("https://api.groq.test/openai/v1/chat/completions")


def test_openrouter_app_config_base_url_and_timeout(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import OpenRouterAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as mod

    captured: Dict[str, Any] = {}

    def _factory(*args, timeout: float | None = None, **kwargs):
        captured["timeout"] = timeout
        return _FakeClient(captured)

    monkeypatch.setattr(mod, "http_client_factory", _factory, raising=True)

    a = OpenRouterAdapter()
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "meta-llama/llama-3-8b",
        "api_key": "k",
        "app_config": {"openrouter_api": {"api_base_url": "https://openrouter.mock/api/v1", "api_timeout": 44}},
    }
    _ = a.chat(req)
    assert captured.get("timeout") == 44
    assert str(captured.get("url", "")).startswith("https://openrouter.mock/api/v1/chat/completions")
