from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    capture_provider_override_call_snapshot,
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter
from tldw_Server_API.tests.provider_credential_test_helpers import (
    resolved_request_fields,
)

_BEHAVIOR_CASES = (
    (
        "openai",
        "openai_api",
        {
            "temperature": 0.19,
            "top_p": 0.79,
            "max_tokens": 319,
            "max_completion_tokens": 320,
            "n": 2,
            "seed": 10,
            "presence_penalty": 0.09,
            "frequency_penalty": 0.08,
            "response_format": {"type": "json_object"},
            "stop": ["OPENAI-END"],
        },
    ),
    (
        "cohere",
        "cohere_api",
        {
            "temperature": 0.2,
            "top_p": 0.3,
            "p": 0.31,
            "top_k": 7,
            "k": 8,
            "max_tokens": 321,
            "stop_sequences": ["END"],
            "seed": 11,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "num_generations": 2,
        },
    ),
    (
        "deepseek",
        "deepseek_api",
        {
            "temperature": 0.21,
            "top_p": 0.82,
            "max_tokens": 322,
            "seed": 12,
            "stop": ["DONE"],
            "presence_penalty": 0.11,
            "frequency_penalty": 0.12,
            "logprobs": True,
            "top_logprobs": 3,
            "response_format": {"type": "json_object"},
            "n": 2,
        },
    ),
    (
        "google",
        "google_api",
        {
            "temperature": 0.22,
            "top_p": 0.83,
            "topP": 0.831,
            "top_k": 9,
            "topK": 10,
            "max_output_tokens": 323,
            "max_tokens": 324,
            "stop_sequences": ["STOP"],
            "candidate_count": 2,
            "n": 3,
            "response_format": {"type": "application/json"},
        },
    ),
    (
        "mistral",
        "mistral_api",
        {
            "temperature": 0.23,
            "top_p": 0.84,
            "max_tokens": 324,
            "random_seed": 13,
            "top_k": 10,
            "safe_prompt": True,
            "response_format": {"type": "json_object"},
        },
    ),
    (
        "qwen",
        "qwen_api",
        {
            "temperature": 0.24,
            "top_p": 0.85,
            "max_tokens": 325,
            "seed": 14,
            "stop": ["QWEN-END"],
            "presence_penalty": 0.13,
            "frequency_penalty": 0.14,
            "logprobs": True,
            "top_logprobs": 4,
            "response_format": {"type": "json_object"},
            "n": 2,
        },
    ),
    (
        "moonshot",
        "moonshot_api",
        {
            "temperature": 0.25,
            "top_p": 0.86,
            "max_tokens": 326,
        },
    ),
    (
        "zai",
        "zai_api",
        {
            "temperature": 0.26,
            "top_p": 0.87,
            "max_tokens": 327,
        },
    ),
    (
        "local-llm",
        "local_llm",
        {
            "temperature": 0.265,
            "top_p": 0.875,
            "top_k": 10,
            "min_p": 0.035,
            "max_tokens": 327,
            "seed": 145,
            "stop": ["LOCAL-END"],
            "response_format": {"type": "json_object"},
            "n": 2,
            "presence_penalty": 0.145,
            "frequency_penalty": 0.146,
            "logprobs": True,
            "top_logprobs": 3,
            "streaming": False,
            "strict_openai_compat": False,
        },
    ),
    (
        "llama.cpp",
        "llama_api",
        {
            "temperature": 0.27,
            "top_p": 0.88,
            "top_k": 11,
            "min_p": 0.04,
            "max_tokens": 328,
            "n_predict": 329,
            "seed": 15,
            "stop": ["LLAMA-END"],
            "presence_penalty": 0.15,
            "frequency_penalty": 0.16,
            "n": 2,
            "n_probs": 3,
            "response_format": {"type": "json_object"},
        },
    ),
    (
        "kobold",
        "kobold_api",
        {
            "temperature": 0.28,
            "top_p": 0.89,
            "top_k": 12,
            "max_length": 330,
            "stop_sequence": ["KOBOLD-END"],
            "num_responses": 2,
            "seed": 16,
            "max_context_length": 4096,
            "rep_pen": 1.1,
        },
    ),
    (
        "ooba",
        "ooba_api",
        {
            "temperature": 0.281,
            "top_p": 0.891,
            "top_k": 13,
            "min_p": 0.051,
            "max_tokens": 331,
            "seed": 161,
            "stop": ["OOBA-END"],
            "response_format": {"type": "json_object"},
            "n": 2,
            "presence_penalty": 0.161,
            "frequency_penalty": 0.162,
        },
    ),
    (
        "tabbyapi",
        "tabby_api",
        {
            "temperature": 0.282,
            "temp": 0.283,
            "top_p": 0.892,
            "top_k": 14,
            "min_p": 0.052,
            "max_tokens": 332,
            "seed": 162,
            "stop": ["TABBY-END"],
            "response_format": {"type": "json_object"},
            "n": 2,
            "presence_penalty": 0.163,
            "frequency_penalty": 0.164,
            "logprobs": True,
            "top_logprobs": 4,
        },
    ),
    (
        "vllm",
        "vllm_api",
        {
            "temperature": 0.283,
            "top_p": 0.893,
            "top_k": 15,
            "min_p": 0.053,
            "max_tokens": 333,
            "seed": 163,
            "stop": ["VLLM-END"],
            "response_format": {"type": "json_object"},
            "n": 2,
            "presence_penalty": 0.165,
            "frequency_penalty": 0.166,
            "logprobs": True,
            "top_logprobs": 5,
        },
    ),
    (
        "aphrodite",
        "aphrodite_api",
        {
            "temperature": 0.284,
            "top_p": 0.894,
            "top_k": 16,
            "min_p": 0.054,
            "max_tokens": 334,
            "seed": 164,
            "stop": ["APHRODITE-END"],
            "response_format": {"type": "json_object"},
            "n": 2,
            "presence_penalty": 0.167,
            "frequency_penalty": 0.168,
            "logprobs": True,
        },
    ),
    (
        "ollama",
        "ollama_api",
        {
            "temperature": 0.29,
            "top_p": 0.9,
            "top_k": 13,
            "max_tokens": 331,
            "num_predict": 332,
            "seed": 17,
            "stop": ["OLLAMA-END"],
            "format": "json",
            "presence_penalty": 0.169,
            "frequency_penalty": 0.17,
            "logprobs": True,
            "top_logprobs": 6,
        },
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider,section,behavior_config",
    _BEHAVIOR_CASES,
    ids=[case[0] for case in _BEHAVIOR_CASES],
)
def test_static_snapshot_projects_only_known_provider_behavior_config(
    provider: str,
    section: str,
    behavior_config: dict[str, Any],
) -> None:
    source_behavior = copy.deepcopy(behavior_config)
    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        provider,
        {
            section: {
                **source_behavior,
                "api_key": "must-not-project",
                "access_token": "must-not-project",
                "client_secret": "must-not-project",
                "headers": {"X-Snapshot-Leak": "must-not-project"},
                "extra_headers": {"X-Snapshot-Leak": "must-not-project"},
                "extra_body": {"snapshot_leak": "must-not-project"},
                "tools": [{"snapshot_leak": "must-not-project"}],
                "tool_choice": "must-not-project",
                "user": "must-not-project",
                "user_identifier": "must-not-project",
                "logit_bias": {"must-not-project": 1},
                **(
                    {"streaming": "must-not-project"}
                    if provider != "local-llm"
                    else {}
                ),
                "unknown_behavior_canary": "must-not-project",
            }
        },
    )

    assert fallback.app_config == {section: behavior_config}
    for nested_key in ("stop", "stop_sequences", "stop_sequence", "response_format"):
        nested_value = fallback.app_config[section].get(nested_key)
        if isinstance(nested_value, list):
            nested_value.append("snapshot-mutation")
            assert source_behavior[nested_key] == behavior_config[nested_key]
        elif isinstance(nested_value, dict):
            nested_value["snapshot-mutation"] = True
            assert source_behavior[nested_key] == behavior_config[nested_key]
    fallback.app_config[section][next(iter(behavior_config))] = "mutated"
    assert source_behavior == behavior_config
    assert "must-not-project" not in repr(fallback.app_config)


class _CohereResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "generation_id": "behavior-snapshot",
            "text": "ok",
            "finish_reason": "COMPLETE",
        }


class _OpenAIResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "id": "behavior-snapshot",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }


class _OpenAIClient:
    def __init__(self, captured: dict[str, Any], timeout: float | None) -> None:
        self._captured = captured
        self._captured["timeout"] = timeout

    def __enter__(self) -> _OpenAIClient:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def post(self, url: str, **kwargs: Any) -> _OpenAIResponse:
        self._captured.update(
            {
                "url": url,
                "headers": copy.deepcopy(kwargs["headers"]),
                "json": copy.deepcopy(kwargs["json"]),
            }
        )
        return _OpenAIResponse()


class _CohereSession:
    def __init__(self, payloads: list[dict[str, Any]], lock: threading.Lock) -> None:
        self._payloads = payloads
        self._lock = lock

    def post(self, _url: str, **kwargs: Any) -> _CohereResponse:
        with self._lock:
            self._payloads.append(
                {
                    "authorization": kwargs["headers"]["Authorization"],
                    "json": copy.deepcopy(kwargs["json"]),
                }
            )
        return _CohereResponse()

    def close(self) -> None:
        return None


def _cohere_request_from_fallback(
    fallback: byok_runtime.ServerFallbackCredentials,
) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "command-r",
        **resolved_request_fields(
            "cohere",
            api_key=fallback.api_key,
            app_config=copy.deepcopy(dict(fallback.app_config or {})),
            model="command-r",
        ),
    }


@pytest.mark.unit
def test_openai_uses_snapshot_behavior_and_api_base_alias_at_http_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_module
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import (
        OpenAIAdapter,
    )

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        openai_module,
        "http_client_factory",
        lambda **kwargs: _OpenAIClient(captured, kwargs.get("timeout")),
    )
    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        "openai",
        {
            "openai_api": {
                "api_key": "static-secret",
                "api_base": "https://snapshot.openai.example/v1",
                "api_timeout": 17,
                "temperature": 0.17,
                "top_p": 0.61,
                "max_tokens": 444,
                "n": 2,
                "seed": 23,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.3,
                "response_format": {"type": "json_object"},
                "stop": ["STATIC-END"],
            }
        },
    )

    OpenAIAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "gpt-4o-mini",
            **resolved_request_fields(
                "openai",
                api_key=fallback.api_key,
                app_config=copy.deepcopy(dict(fallback.app_config or {})),
                model="gpt-4o-mini",
            ),
        }
    )

    assert captured == {
        "timeout": 17.0,
        "url": "https://snapshot.openai.example/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer static-secret",
        },
        "json": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.17,
            "top_p": 0.61,
            "max_tokens": 444,
            "n": 2,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.2,
            "response_format": {"type": "json_object"},
            "seed": 23,
            "stop": ["STATIC-END"],
            "stream": False,
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider,section",
    (("openai", "openai_api"), ("google", "google_api")),
)
def test_api_base_alias_survives_authoritative_projection(
    provider: str,
    section: str,
) -> None:
    endpoint = f"https://snapshot.{provider}.example/v1"
    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        provider,
        {section: {"api_key": "static-secret", "api_base": endpoint}},
    )
    request = {
        "credentials_resolved": True,
        "app_config": copy.deepcopy(dict(fallback.app_config or {})),
    }

    if provider == "openai":
        from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import (
            OpenAIAdapter,
        )

        resolved_endpoint = OpenAIAdapter()._resolve_base_url(request)
    else:
        from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import (
            GoogleAdapter,
        )

        resolved_endpoint = GoogleAdapter()._base_url(request)

    assert fallback.app_config == {section: {"api_base": endpoint}}
    assert resolved_endpoint == endpoint


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider,section",
    (
        ("anthropic", "anthropic_api"),
        ("groq", "groq_api"),
        ("openrouter", "openrouter_api"),
        ("huggingface", "huggingface_api"),
        ("bedrock", "bedrock_api"),
        ("custom-openai-api", "custom_openai_api"),
    ),
)
def test_snapshot_drops_generation_defaults_not_consumed_from_provider_config(
    provider: str,
    section: str,
) -> None:
    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        provider,
        {
            section: {
                "api_key": "static-secret",
                "temperature": 0.1,
                "top_p": 0.2,
                "max_tokens": 123,
                "stop": ["END"],
                "response_format": {"type": "json_object"},
            }
        },
    )

    assert fallback.app_config == {}


@pytest.mark.unit
def test_cohere_normalizes_loaded_max_p_when_request_omits_generation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.config import load_and_log_configs

    payloads: list[dict[str, Any]] = []
    lock = threading.Lock()
    loaded_config = load_and_log_configs()
    assert isinstance(loaded_config, dict)
    config_snapshot = copy.deepcopy(loaded_config)
    cohere_config = config_snapshot.get("cohere_api")
    assert isinstance(cohere_config, dict)
    assert "max_p" in cohere_config
    cohere_config.update(
        {
            "api_key": "static-secret",
            "temperature": 0.17,
            "max_p": 0.61,
            "top_k": 19,
            "max_tokens": 444,
            "stop_sequences": ["STATIC-END"],
            "seed": 23,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "num_generations": 2,
        }
    )
    cohere_config.pop("top_p", None)
    cohere_config.pop("p", None)
    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        "cohere",
        config_snapshot,
    )

    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _CohereSession(payloads, lock),
    )

    CohereAdapter().chat(_cohere_request_from_fallback(fallback))

    assert payloads == [
        {
            "authorization": "Bearer static-secret",
            "json": {
                "model": "command-r",
                "message": "hello",
                "temperature": 0.17,
                "p": 0.61,
                "k": 19,
                "max_tokens": 444,
                "stop_sequences": ["STATIC-END"],
                "seed": 23,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.3,
                "stream": False,
                "num_generations": 2,
            },
        }
    ]


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_cohere_static_and_override_generations_remain_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []
    payload_lock = threading.Lock()
    captured_a = threading.Event()
    release_a = threading.Event()
    source_a = {
        "cohere_api": {
            "api_key": "static-a",
            "temperature": 0.91,
            "top_p": 0.41,
            "top_k": 21,
            "max_tokens": 501,
            "stop_sequences": ["STATIC-A"],
        }
    }
    source_b = {
        "cohere_api": {
            "api_key": "static-b",
            "temperature": 0.92,
            "top_p": 0.42,
            "top_k": 22,
            "max_tokens": 502,
            "stop_sequences": ["STATIC-B"],
        }
    }

    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    monkeypatch.setattr(
        chat_calls,
        "create_session_with_retries",
        lambda **_kwargs: _CohereSession(payloads, payload_lock),
    )
    set_llm_provider_overrides_cache_for_tests(
        {
            "cohere": LLMProviderOverride(
                provider="cohere",
                api_key="override-a",
                config={"temperature": 0.11, "max_tokens": 601},
            )
        }
    )

    def call_a() -> None:
        static_fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
            "cohere",
            source_a,
        )
        override_snapshot = capture_provider_override_call_snapshot("cohere")
        captured_a.set()
        assert release_a.wait(timeout=5)
        fallback = override_snapshot.server_fallback(static_fallback)
        assert fallback is not None
        CohereAdapter().chat(_cohere_request_from_fallback(fallback))

    def call_b() -> None:
        static_fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
            "cohere",
            source_b,
        )
        fallback = capture_provider_override_call_snapshot("cohere").server_fallback(
            static_fallback
        )
        assert fallback is not None
        CohereAdapter().chat(_cohere_request_from_fallback(fallback))

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call_a)
            assert captured_a.wait(timeout=5)
            source_a["cohere_api"].update(source_b["cohere_api"])
            set_llm_provider_overrides_cache_for_tests(
                {
                    "cohere": LLMProviderOverride(
                        provider="cohere",
                        api_key="override-b",
                        config={"temperature": 0.12, "max_tokens": 602},
                    )
                }
            )
            second = executor.submit(call_b)
            release_a.set()
            first.result(timeout=10)
            second.result(timeout=10)
    finally:
        release_a.set()
        set_llm_provider_overrides_cache_for_tests({})

    by_auth = {entry["authorization"]: entry["json"] for entry in payloads}
    assert by_auth == {
        "Bearer override-a": {
            "model": "command-r",
            "message": "hello",
            "temperature": 0.11,
            "p": 0.41,
            "k": 21,
            "max_tokens": 601,
            "stop_sequences": ["STATIC-A"],
            "stream": False,
        },
        "Bearer override-b": {
            "model": "command-r",
            "message": "hello",
            "temperature": 0.12,
            "p": 0.42,
            "k": 22,
            "max_tokens": 602,
            "stop_sequences": ["STATIC-B"],
            "stream": False,
        },
    }
