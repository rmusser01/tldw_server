"""Transport-level coverage for the opt-in prompt-improvement call policy."""

from __future__ import annotations

import copy
import io
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import requests
from loguru import logger

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.LLM_Calls import capability_registry
from tldw_Server_API.app.core.LLM_Calls.providers import base as base_module
from tldw_Server_API.app.core.LLM_Calls.providers import (
    cohere_adapter as cohere_adapter_module,
)
from tldw_Server_API.app.core.LLM_Calls.providers import (
    local_adapters as local_adapters_module,
)
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter import (
    DeepSeekAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter import (
    HuggingFaceAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import (
    KoboldAdapter,
    LocalLLMAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.moonshot_adapter import (
    MoonshotAdapter,
)
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.LLM_Calls.providers.zai_adapter import ZaiAdapter

pytestmark = pytest.mark.unit

_PRIVATE_SENTINEL = "PRIVATE_PROMPT_IMPROVEMENT_DRAFT_SENTINEL"


def _prompt_policy():
    return capability_registry.ProviderCallPolicy(
        max_transport_attempts=1,
        allow_streaming=False,
        allow_tools=False,
        allow_stop=False,
        allow_response_format=False,
        candidate_count=1,
        temperature=0.2,
        top_p=0.95,
        privacy_safe_errors=True,
    )


def _hostile_config(section: str) -> dict[str, Any]:
    return {
        "api_retries": 9,
        "unrelated_provider_object": {
            "response_format": {"type": "json_object"},
            "nested": {"owner_setting": True},
        },
        section: {
            "api_retries": 4,
            "api_retry_delay": 0,
            "temperature": 0.99,
            "top_p": 0.11,
            "n": 7,
            "num_generations": 7,
            "tools": [{"type": "function", "function": {"name": "leak"}}],
            "functions": [{"name": "leak"}],
            "stop": ["hostile-stop"],
            "response_format": {"type": "json_object"},
        }
    }


def _hostile_extra_body() -> dict[str, Any]:
    return {
        "stream": True,
        "temperature": 0.88,
        "top_p": 0.12,
        "p": 0.12,
        "n": 8,
        "num_generations": 8,
        "tools": [{"type": "function", "function": {"name": "leak"}}],
        "tool_choice": "auto",
        "functions": [{"name": "leak"}],
        "function_call": "auto",
        "stop": ["hostile-extra-stop"],
        "response_format": {"type": "json_object"},
        "safe_extension": "preserved",
    }


class _SuccessResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body

    def close(self) -> None:
        return None


class _CapturingLogger:
    """Capture one logger method without mutating the shared Loguru singleton."""

    def __init__(self, delegate: Any, method: str, emitted: list[str]) -> None:
        self._delegate = delegate
        self._method = method
        self._emitted = emitted

    def __getattr__(self, name: str) -> Any:
        if name != self._method:
            return getattr(self._delegate, name)

        def capture(message: Any, *args: Any, **_kwargs: Any) -> None:
            rendered = str(message).format(*args) if args else str(message)
            self._emitted.append(rendered)

        return capture


def test_policy_is_immutable_opt_in_and_copies_hostile_configuration() -> None:
    policy = _prompt_policy()
    with pytest.raises(FrozenInstanceError):
        policy.temperature = 0.9

    app_config = _hostile_config("moonshot_api")
    extra_body = _hostile_extra_body()
    original_config = copy.deepcopy(app_config)
    original_extra = copy.deepcopy(extra_body)

    effective = capability_registry.validate_payload(
        "moonshot",
        {
            "messages": [{"role": "user", "content": "private"}],
            "model": "moonshot-v1-8k",
            "app_config": app_config,
            "extra_body": extra_body,
            "call_policy": policy,
        },
    )

    assert app_config == original_config
    assert extra_body == original_extra
    assert effective["app_config"] is not app_config
    assert effective["extra_body"] is not extra_body
    assert effective["stream"] is False
    assert effective["tools"] is None
    assert effective["tool_choice"] is None
    assert effective["stop"] is None
    assert effective["response_format"] is None
    assert effective["n"] == 1
    assert effective["temperature"] == 0.2
    assert effective["top_p"] == 0.95
    assert effective["app_config"]["moonshot_api"]["api_retries"] == 0
    assert effective["app_config"]["moonshot_api"]["tools"] is None
    assert effective["app_config"]["moonshot_api"]["stop"] is None
    assert effective["app_config"]["moonshot_api"]["response_format"] is None
    assert effective["app_config"]["unrelated_provider_object"] == {
        "response_format": {"type": "json_object"},
        "nested": {"owner_setting": True},
    }
    assert effective["extra_body"] == {"safe_extension": "preserved"}

    ordinary = capability_registry.validate_payload(
        "moonshot",
        {
            "messages": [{"role": "user", "content": "ordinary"}],
            "model": "moonshot-v1-8k",
            "app_config": app_config,
            "extra_body": extra_body,
        },
    )
    assert ordinary["app_config"] == original_config
    assert ordinary["extra_body"] == original_extra
    assert ordinary.get("stream") is None
    assert ordinary.get("n") is None


@pytest.mark.parametrize(
    ("adapter", "section", "success_body", "candidate_key", "top_p_key"),
    [
        (
            MoonshotAdapter(),
            "moonshot_api",
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            "n",
            "top_p",
        ),
        (
            ZaiAdapter(),
            "zai_api",
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            "n",
            "top_p",
        ),
        (
            CohereAdapter(),
            "cohere_api",
            {"text": "ok", "generation_id": "gen-1", "meta": {}},
            "num_generations",
            "p",
        ),
    ],
    ids=["moonshot", "zai", "cohere"],
)
def test_retrying_commercial_adapters_build_one_attempt_with_policy_payload(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    section: str,
    success_body: dict[str, Any],
    candidate_key: str | None,
    top_p_key: str,
) -> None:
    captured: dict[str, Any] = {"posts": 0}

    class FakeSession:
        def post(self, _url, **kwargs):
            captured["posts"] += 1
            captured["payload"] = kwargs["json"]
            return _SuccessResponse(success_body)

        def close(self) -> None:
            return None

    def session_factory(**kwargs):
        captured["retry"] = kwargs
        return FakeSession()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        session_factory,
    )
    app_config = _hostile_config(section)
    app_config[section]["api_base_url"] = "https://provider.invalid/v1"
    original_config = copy.deepcopy(app_config)
    adapter.chat(
        {
            "messages": [{"role": "user", "content": "private"}],
            "model": "model-test",
            "api_key": "test-key",
            "app_config": app_config,
            "extra_body": _hostile_extra_body(),
            "call_policy": _prompt_policy(),
        }
    )

    assert app_config == original_config
    assert captured["retry"]["total"] == 0
    assert captured["posts"] == 1
    payload = captured["payload"]
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload[top_p_key] == 0.95
    if candidate_key is not None:
        assert payload[candidate_key] == 1
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "functions" not in payload
    assert "function_call" not in payload
    assert "stop" not in payload
    assert "response_format" not in payload


def test_local_adapter_policy_reaches_retry_object_and_effective_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "")
    captured: dict[str, Any] = {"fetches": 0}

    class FakeClient:
        def close(self) -> None:
            return None

    def fake_fetch(**kwargs):
        captured["fetches"] += 1
        captured["payload"] = kwargs["json"]
        captured["retry"] = kwargs["retry"]
        return _SuccessResponse({"choices": []})

    app_config = _hostile_config("local_llm")
    app_config["local_llm"].update(
        {"api_ip": "http://127.0.0.1:18099", "api_key": "test-key"}
    )
    original_config = copy.deepcopy(app_config)
    adapter = LocalLLMAdapter()
    adapter.http_client_factory = lambda timeout: FakeClient()
    adapter.http_fetcher = fake_fetch
    result = adapter.chat(
        {
            "messages": [{"role": "user", "content": "private"}],
            "model": "local-test",
            "app_config": app_config,
            "extra_body": _hostile_extra_body(),
            "call_policy": _prompt_policy(),
        }
    )

    assert result == {"choices": []}
    assert app_config == original_config
    assert captured["fetches"] == 1
    assert captured["retry"].attempts == 1
    payload = captured["payload"]
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.95
    assert payload["n"] == 1
    assert "tools" not in payload
    assert "functions" not in payload
    assert "stop" not in payload
    assert "response_format" not in payload


def test_huggingface_policy_reaches_single_post_and_effective_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"posts": 0}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url, **kwargs):
            captured["posts"] += 1
            captured["payload"] = kwargs["json"]
            return _SuccessResponse({"choices": []})

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.http_client_factory",
        lambda **_kwargs: FakeClient(),
    )
    app_config = _hostile_config("huggingface_api")
    app_config["huggingface_api"]["api_base_url"] = "https://provider.invalid/v1"
    original_config = copy.deepcopy(app_config)
    HuggingFaceAdapter().chat(
        {
            "messages": [{"role": "user", "content": "private"}],
            "model": "hf-test",
            "api_key": "test-key",
            "app_config": app_config,
            "extra_body": _hostile_extra_body(),
            "call_policy": _prompt_policy(),
        }
    )

    assert app_config == original_config
    assert captured["posts"] == 1
    payload = captured["payload"]
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.95
    assert payload["n"] == 1
    assert "tools" not in payload
    assert "functions" not in payload
    assert "stop" not in payload
    assert "response_format" not in payload


def test_openai_policy_removes_inherited_controls_without_mutating_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url, **kwargs):
            captured["payload"] = kwargs["json"]
            return _SuccessResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda **_kwargs: FakeClient(),
    )
    app_config = _hostile_config("openai_api")
    app_config["openai_api"]["api_base_url"] = "https://provider.invalid/v1"
    original = copy.deepcopy(app_config)

    OpenAIAdapter().chat(
        {
            "messages": [{"role": "user", "content": "private"}],
            "model": "gpt-test",
            "api_key": "test-key",
            "app_config": app_config,
            "extra_body": _hostile_extra_body(),
            "call_policy": _prompt_policy(),
        }
    )

    assert app_config == original
    assert "stop" not in captured["payload"]
    assert "response_format" not in captured["payload"]


def test_openai_without_policy_preserves_inherited_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url, **kwargs):
            captured["payload"] = kwargs["json"]
            return _SuccessResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.http_client_factory",
        lambda **_kwargs: FakeClient(),
    )
    app_config = _hostile_config("openai_api")
    app_config["openai_api"]["api_base_url"] = "https://provider.invalid/v1"
    original = copy.deepcopy(app_config)

    OpenAIAdapter().chat(
        {
            "messages": [{"role": "user", "content": "ordinary"}],
            "model": "gpt-test",
            "api_key": "test-key",
            "app_config": app_config,
        }
    )

    assert app_config == original
    assert captured["payload"]["stop"] == ["hostile-stop"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_privacy_safe_policy_removes_echoed_body_from_logs_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 500
    response.url = "https://provider.invalid/v1/chat/completions"
    response._content = (
        '{"error":{"message":"upstream echoed '
        + _PRIVATE_SENTINEL
        + '"}}'
    ).encode()
    response.request = requests.Request("POST", response.url).prepare()

    class FakeSession:
        def post(self, *_args, **_kwargs):
            return response

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **_kwargs: FakeSession(),
    )
    adapter = MoonshotAdapter()
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )

    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(ChatAPIError) as captured:
            await chat_service.perform_chat_api_call_async(
                api_endpoint="moonshot",
                messages_payload=[{"role": "user", "content": _PRIVATE_SENTINEL}],
                model="moonshot-v1-8k",
                api_key="test-key",
                app_config=_hostile_config("moonshot_api"),
                call_policy=_prompt_policy(),
            )
    finally:
        logger.remove(sink_id)

    assert _PRIVATE_SENTINEL not in str(captured.value)
    assert _PRIVATE_SENTINEL not in sink.getvalue()


@pytest.mark.asyncio
async def test_deepseek_policy_uses_one_post_without_logging_echoed_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request(
        "POST", "https://provider.invalid/v1/chat/completions"
    )
    response = httpx.Response(
        500,
        request=request,
        json={"error": {"message": f"upstream echoed {_PRIVATE_SENTINEL}"}},
    )
    captured_transport: dict[str, Any] = {"posts": 0}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url, **kwargs):
            captured_transport["posts"] += 1
            captured_transport["payload"] = kwargs["json"]
            return response

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.http_client_factory",
        lambda **_kwargs: FakeClient(),
    )
    adapter = DeepSeekAdapter()
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )

    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(ChatAPIError) as captured_error:
            await chat_service.perform_chat_api_call_async(
                api_endpoint="deepseek",
                messages_payload=[{"role": "user", "content": _PRIVATE_SENTINEL}],
                model="deepseek-chat",
                api_key="test-key",
                app_config=_hostile_config("deepseek_api"),
                call_policy=_prompt_policy(),
            )
    finally:
        logger.remove(sink_id)

    assert captured_transport["posts"] == 1
    assert captured_transport["payload"]["stream"] is False
    assert captured_transport["payload"]["n"] == 1
    assert _PRIVATE_SENTINEL not in str(captured_error.value)
    assert _PRIVATE_SENTINEL not in sink.getvalue()


@pytest.mark.asyncio
async def test_cohere_policy_does_not_log_malformed_success_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []

    class FakeSession:
        def post(self, *_args, **_kwargs):
            return _SuccessResponse({"unexpected": _PRIVATE_SENTINEL})

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
        lambda **_kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: CohereAdapter()),
    )
    monkeypatch.setattr(
        cohere_adapter_module,
        "logging",
        _CapturingLogger(cohere_adapter_module.logging, "warning", emitted),
    )

    await chat_service.perform_chat_api_call_async(
        api_endpoint="cohere",
        messages_payload=[{"role": "user", "content": "private"}],
        model="command-r",
        api_key="test-key",
        app_config=_hostile_config("cohere_api"),
        call_policy=_prompt_policy(),
    )

    assert _PRIVATE_SENTINEL not in "\n".join(emitted)


@pytest.mark.asyncio
async def test_kobold_policy_does_not_log_or_embed_unexpected_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "")
    emitted: list[str] = []

    def fake_fetch(**_kwargs):
        return _SuccessResponse({"unexpected": _PRIVATE_SENTINEL})

    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.providers.local_adapters._hc_fetch",
        fake_fetch,
    )
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: KoboldAdapter()),
    )
    monkeypatch.setattr(
        local_adapters_module,
        "logging",
        _CapturingLogger(local_adapters_module.logging, "error", emitted),
    )
    app_config = _hostile_config("kobold_api")
    app_config["kobold_api"].update(
        {"api_ip": "http://127.0.0.1:18099", "api_retry_delay": 0}
    )

    with pytest.raises(ChatAPIError) as captured:
        await chat_service.perform_chat_api_call_async(
            api_endpoint="kobold",
            messages_payload=[{"role": "user", "content": "private"}],
            model="kobold-test",
            app_config=app_config,
            call_policy=_prompt_policy(),
        )

    assert _PRIVATE_SENTINEL not in str(captured.value)
    assert _PRIVATE_SENTINEL not in "\n".join(emitted)


class _BaseNormalizationAdapter(ChatProvider):
    name = "base-normalization-test"

    def capabilities(self) -> dict[str, Any]:
        return {}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None):
        try:
            raise RuntimeError(_PRIVATE_SENTINEL)
        except RuntimeError as exc:
            raise self.normalize_error(exc) from exc

    def stream(self, request: dict[str, Any], *, timeout: float | None = None):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_base_normalization_policy_does_not_log_or_embed_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _BaseNormalizationAdapter()
    emitted: list[str] = []

    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(
        base_module,
        "logger",
        _CapturingLogger(base_module.logger, "debug", emitted),
    )

    with pytest.raises(ChatAPIError) as captured:
        await chat_service.perform_chat_api_call_async(
            api_endpoint="base-normalization-test",
            messages_payload=[{"role": "user", "content": "private"}],
            model="base-test",
            call_policy=_prompt_policy(),
        )

    assert _PRIVATE_SENTINEL not in str(captured.value)
    assert _PRIVATE_SENTINEL not in "\n".join(emitted)


@pytest.mark.asyncio
async def test_base_normalization_without_policy_sanitizes_error_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _BaseNormalizationAdapter()
    emitted: list[str] = []
    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(
        base_module,
        "logger",
        _CapturingLogger(base_module.logger, "debug", emitted),
    )

    with pytest.raises(ChatAPIError) as captured:
        await chat_service.perform_chat_api_call_async(
            api_endpoint="base-normalization-test",
            messages_payload=[{"role": "user", "content": "ordinary"}],
            model="base-test",
        )

    assert _PRIVATE_SENTINEL not in str(captured.value)
    assert _PRIVATE_SENTINEL not in "\n".join(emitted)
