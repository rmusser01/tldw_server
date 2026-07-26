from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers import (
    cohere_adapter as cohere_adapter_module,
)
from tldw_Server_API.app.core.LLM_Calls.providers import (
    moonshot_adapter as moonshot_adapter_module,
)
from tldw_Server_API.app.core.LLM_Calls.providers import zai_adapter as zai_adapter_module
from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import (
    AnthropicAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter import (
    BedrockAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter import (
    DeepSeekAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter import GroqAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter import (
    MistralAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter import (
    MoonshotAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import (
    OpenRouterAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter import QwenAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.zai_adapter import ZaiAdapter
from tldw_Server_API.tests.provider_credential_test_helpers import (
    resolved_request_fields,
)

_SNAPSHOT_A = "https://snapshot-a.example/v1"
_SNAPSHOT_C = "https://snapshot-c.example/v1"
_LATE_ENV = "https://late-env.example/v1"


def _request(
    section: str,
    endpoint: str | None,
    *,
    include_section: bool = True,
) -> dict[str, Any]:
    provider_config = {"api_base_url": endpoint} if endpoint is not None else {}
    return {
        "credentials_resolved": True,
        "app_config": {section: provider_config} if include_section else {},
    }


def _anthropic(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._resolve_base_url(request)


def _openai(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._resolve_base_url(request)


def _deepseek(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._base_url(request["app_config"], request)


def _google(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._base_url(request)


def _groq(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._resolve_base_url(request)


def _qwen(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._base_url(request["app_config"], request)


def _mistral(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._resolve_base_url(request)


def _openrouter(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._resolve_base_url(request)


def _bedrock(adapter: Any, request: dict[str, Any]) -> str:
    return adapter._base_url(request)


_COMMERCIAL_CASES: tuple[
    tuple[
        type[Any],
        str,
        tuple[str, ...],
        str,
        Callable[[Any, dict[str, Any]], str],
    ],
    ...,
] = (
    (
        AnthropicAdapter,
        "anthropic_api",
        ("ANTHROPIC_BASE_URL",),
        "https://api.anthropic.com/v1",
        _anthropic,
    ),
    (
        OpenAIAdapter,
        "openai_api",
        (
            "OPENAI_API_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "MOCK_OPENAI_BASE_URL",
        ),
        "https://api.openai.com/v1",
        _openai,
    ),
    (
        DeepSeekAdapter,
        "deepseek_api",
        ("DEEPSEEK_BASE_URL",),
        "https://api.deepseek.com",
        _deepseek,
    ),
    (
        GoogleAdapter,
        "google_api",
        ("GOOGLE_GEMINI_BASE_URL",),
        "https://generativelanguage.googleapis.com/v1beta",
        _google,
    ),
    (
        GroqAdapter,
        "groq_api",
        ("GROQ_BASE_URL",),
        "https://api.groq.com/openai/v1",
        _groq,
    ),
    (
        QwenAdapter,
        "qwen_api",
        ("QWEN_BASE_URL", "QWEN_REGION"),
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        _qwen,
    ),
    (
        MistralAdapter,
        "mistral_api",
        ("MISTRAL_API_BASE",),
        "https://api.mistral.ai/v1",
        _mistral,
    ),
    (
        OpenRouterAdapter,
        "openrouter_api",
        ("OPENROUTER_BASE_URL",),
        "https://openrouter.ai/api/v1",
        _openrouter,
    ),
    (
        BedrockAdapter,
        "bedrock_api",
        (
            "BEDROCK_RUNTIME_ENDPOINT",
            "BEDROCK_API_BASE_URL",
            "BEDROCK_OPENAI_BASE_URL",
            "BEDROCK_REGION",
        ),
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai",
        _bedrock,
    ),
)


def _set_late_endpoint_env(monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(names[0], _LATE_ENV)


@pytest.mark.unit
@pytest.mark.parametrize(
    "include_section",
    (True, False),
    ids=("empty-provider-section", "empty-config-map"),
)
@pytest.mark.parametrize(
    "adapter_type,section,env_names,canonical_endpoint,resolver",
    _COMMERCIAL_CASES,
    ids=(
        "anthropic",
        "openai",
        "deepseek",
        "google",
        "groq",
        "qwen",
        "mistral",
        "openrouter",
        "bedrock",
    ),
)
def test_resolved_commercial_adapter_ignores_environment_added_after_empty_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    section: str,
    env_names: tuple[str, ...],
    canonical_endpoint: str,
    resolver: Callable[[Any, dict[str, Any]], str],
    include_section: bool,
) -> None:
    _set_late_endpoint_env(monkeypatch, env_names)

    resolved = resolver(
        adapter_type(),
        _request(section, None, include_section=include_section),
    )

    assert resolved == canonical_endpoint


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type,section,env_names,_canonical_endpoint,resolver",
    _COMMERCIAL_CASES,
    ids=(
        "anthropic",
        "openai",
        "deepseek",
        "google",
        "groq",
        "qwen",
        "mistral",
        "openrouter",
        "bedrock",
    ),
)
def test_resolved_commercial_adapter_keeps_snapshot_when_environment_rotates(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    section: str,
    env_names: tuple[str, ...],
    _canonical_endpoint: str,
    resolver: Callable[[Any, dict[str, Any]], str],
) -> None:
    _set_late_endpoint_env(monkeypatch, env_names)

    resolved = resolver(adapter_type(), _request(section, _SNAPSHOT_A))

    assert resolved == _SNAPSHOT_A


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "adapter_type,section,env_names,_canonical_endpoint,resolver",
    _COMMERCIAL_CASES,
    ids=(
        "anthropic",
        "openai",
        "deepseek",
        "google",
        "groq",
        "qwen",
        "mistral",
        "openrouter",
        "bedrock",
    ),
)
def test_concurrent_resolved_commercial_adapters_keep_request_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    section: str,
    env_names: tuple[str, ...],
    _canonical_endpoint: str,
    resolver: Callable[[Any, dict[str, Any]], str],
) -> None:
    _set_late_endpoint_env(monkeypatch, env_names)
    adapter = adapter_type()
    gate = threading.Barrier(2)

    def _resolve_after_both_started(request: dict[str, Any]) -> str:
        gate.wait(timeout=5)
        return resolver(adapter, request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_resolve_after_both_started, _request(section, _SNAPSHOT_A))
        second = executor.submit(_resolve_after_both_started, _request(section, _SNAPSHOT_C))

    assert {first.result(), second.result()} == {_SNAPSHOT_A, _SNAPSHOT_C}


@pytest.mark.unit
def test_resolved_bedrock_region_snapshot_controls_endpoint_and_signing_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BEDROCK_REGION", "us-west-1")
    request = {
        "credentials_resolved": True,
        "app_config": {"bedrock_api": {"region": "us-east-2"}},
    }
    adapter = BedrockAdapter()

    base_url = adapter._base_url(request)

    assert base_url == "https://bedrock-runtime.us-east-2.amazonaws.com/openai"
    assert adapter._resolve_region(request, "https://proxy.example/v1") == "us-east-2"


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type,section,env_names,_canonical_endpoint,resolver",
    _COMMERCIAL_CASES,
    ids=(
        "anthropic",
        "openai",
        "deepseek",
        "google",
        "groq",
        "qwen",
        "mistral",
        "openrouter",
        "bedrock",
    ),
)
def test_unmarked_commercial_adapter_keeps_legacy_environment_behavior(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    section: str,
    env_names: tuple[str, ...],
    _canonical_endpoint: str,
    resolver: Callable[[Any, dict[str, Any]], str],
) -> None:
    _set_late_endpoint_env(monkeypatch, env_names)
    request = {"app_config": {section: {}}}

    resolved = resolver(adapter_type(), request)

    assert "late-env.example" in resolved


class _RecordedResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload

    def close(self) -> None:
        return None


class _RecordingSession:
    def __init__(self, urls: list[str], payload: dict[str, Any]) -> None:
        self._urls = urls
        self._payload = payload

    def post(self, url: str, **_kwargs: Any) -> _RecordedResponse:
        self._urls.append(url)
        return _RecordedResponse(self._payload)

    def close(self) -> None:
        return None


def _cohere_response() -> dict[str, Any]:
    return {
        "generation_id": "cohere-snapshot-test",
        "text": "ok",
        "finish_reason": "COMPLETE",
    }


def _openai_compatible_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-snapshot-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


_DELEGATING_COMMERCIAL_CASES: tuple[
    tuple[type[Any], Any, str, str, dict[str, Any]],
    ...,
] = (
    (
        CohereAdapter,
        cohere_adapter_module,
        "cohere_api",
        "https://api.cohere.ai/v1/chat",
        _cohere_response(),
    ),
    (
        MoonshotAdapter,
        moonshot_adapter_module,
        "moonshot_api",
        "https://api.moonshot.cn/v1/chat/completions",
        _openai_compatible_response(),
    ),
    (
        ZaiAdapter,
        zai_adapter_module,
        "zai_api",
        "https://api.z.ai/api/paas/v4/chat/completions",
        _openai_compatible_response(),
    ),
)


def _delegating_request(
    section: str,
    endpoint: str | None,
    *,
    resolved: bool = True,
) -> dict[str, Any]:
    provider_config = {"api_base_url": endpoint} if endpoint is not None else {}
    app_config = {section: provider_config} if endpoint is not None else {}
    request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "snapshot-model",
        "api_key": "snapshot-api-key",
        "app_config": app_config,
    }
    if resolved:
        request.update(
            resolved_request_fields(
                section.removesuffix("_api"),
                api_key="snapshot-api-key",
                app_config=app_config,
                model="snapshot-model",
            )
        )
    return request


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type,module,section,canonical_url,response_payload",
    _DELEGATING_COMMERCIAL_CASES,
    ids=("cohere", "moonshot", "zai"),
)
def test_resolved_delegating_adapter_does_not_reload_after_empty_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    module: Any,
    section: str,
    canonical_url: str,
    response_payload: dict[str, Any],
) -> None:
    urls: list[str] = []
    monkeypatch.setattr(
        module,
        "load_and_log_configs",
        lambda: {
            section: {"api_base_url": "https://late-global.example/v1"}
        },
    )

    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _RecordingSession(urls, response_payload),
    )

    adapter_type().chat(_delegating_request(section, None))

    assert urls == [canonical_url]


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_type,module,section,_canonical_url,response_payload",
    _DELEGATING_COMMERCIAL_CASES,
    ids=("cohere", "moonshot", "zai"),
)
def test_unmarked_delegating_adapter_preserves_legacy_config_reload(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    module: Any,
    section: str,
    _canonical_url: str,
    response_payload: dict[str, Any],
) -> None:
    urls: list[str] = []
    legacy_base_url = "https://legacy-global.example"
    monkeypatch.setattr(
        module,
        "load_and_log_configs",
        lambda: {section: {"api_base_url": legacy_base_url}},
    )

    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _RecordingSession(urls, response_payload),
    )

    request = _delegating_request(section, None, resolved=False)
    adapter_type().chat(request)

    expected_suffix = "/v1/chat" if section == "cohere_api" else "/chat/completions"
    assert urls == [f"{legacy_base_url}{expected_suffix}"]


@pytest.mark.unit
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "adapter_type,module,section,_canonical_url,response_payload",
    _DELEGATING_COMMERCIAL_CASES,
    ids=("cohere", "moonshot", "zai"),
)
def test_concurrent_resolved_delegating_adapters_keep_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
    adapter_type: type[Any],
    module: Any,
    section: str,
    _canonical_url: str,
    response_payload: dict[str, Any],
) -> None:
    urls: list[str] = []
    gate = threading.Barrier(2)
    monkeypatch.setattr(
        module,
        "load_and_log_configs",
        lambda: pytest.fail("resolved adapter must not reload global config"),
    )

    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    def session_factory(**_kwargs: Any) -> _RecordingSession:
        gate.wait(timeout=5)
        return _RecordingSession(urls, response_payload)

    monkeypatch.setattr(chat_calls, "create_session_with_retries", session_factory)

    adapter = adapter_type()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            adapter.chat,
            _delegating_request(section, _SNAPSHOT_A),
        )
        second = executor.submit(
            adapter.chat,
            _delegating_request(section, _SNAPSHOT_C),
        )
        first.result(timeout=10)
        second.result(timeout=10)

    assert set(urls) == {
        f"{_SNAPSHOT_A.rstrip('/')}/v1/chat"
        if section == "cohere_api"
        else f"{_SNAPSHOT_A.rstrip('/')}/chat/completions",
        f"{_SNAPSHOT_C.rstrip('/')}/v1/chat"
        if section == "cohere_api"
        else f"{_SNAPSHOT_C.rstrip('/')}/chat/completions",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "handler",
    (
        cohere_adapter_module._cohere_request,
        moonshot_adapter_module._moonshot_request,
        zai_adapter_module._zai_request,
    ),
    ids=("cohere", "moonshot", "zai"),
)
def test_delegating_handler_marker_preserves_legacy_positional_binding(
    handler: Callable[..., Any],
) -> None:
    signature = inspect.signature(handler)
    legacy_names = tuple(
        name
        for name in signature.parameters
        if name != "credentials_resolved"
    )
    sentinels = tuple(object() for _name in legacy_names)

    bound = signature.bind(*sentinels)

    for name, sentinel in zip(legacy_names, sentinels):
        assert bound.arguments[name] is sentinel
    assert "credentials_resolved" not in bound.arguments
