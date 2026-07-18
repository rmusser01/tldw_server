"""Unit tests for chat_service call parameter construction."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    _is_server_managed_chat_request_field,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    build_call_params_from_request,
    perform_chat_api_call,
)
from tldw_Server_API.app.core.LLM_Calls import adapter_registry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("app_config", {"huggingface_api": {}}),
        ("base_url", "https://attacker.example"),
        ("api_base_url", "https://attacker.example"),
        ("router_base_url", "https://attacker.example"),
        ("api_url", "https://attacker.example"),
        ("huggingface_api_url", "https://attacker.example"),
        ("trusted_base_url_override", True),
        ("api_key", "attacker-key"),
        ("provider_api_key", "attacker-key"),
        ("aws_access_key_id", "attacker-access"),
        ("bedrock_aws_access_key_id", "attacker-access"),
        ("aws_secret_access_key", "attacker-secret"),
        ("bedrock_aws_secret_access_key", "attacker-secret"),
        ("aws_session_token", "attacker-session"),
        ("bedrock_aws_session_token", "attacker-session"),
        ("region", "attacker-region"),
        ("bedrock_region", "attacker-region"),
        ("provider_access_key_id", "attacker-access"),
        ("provider_secret_access_key", "attacker-secret"),
        ("provider_session_token", "attacker-session"),
        ("provider_endpoint", "https://attacker.example"),
        ("provider_endpoint_url", "https://attacker.example"),
        ("PROVIDER_API_KEY", "attacker-key"),
        ("apiKey", "attacker-key"),
        ("providerApiKey", "attacker-key"),
        ("baseUrl", "https://attacker.example"),
        ("providerEndpoint", "https://attacker.example"),
        ("credentialsResolved", True),
        ("trustedBaseUrlOverride", True),
        ("credentials_resolved", True),
        ("auth_source", "oauth"),
        ("request", {"state": "forged"}),
        ("caller_request", {"state": "forged"}),
        ("principal", {"roles": ["admin"]}),
        ("auth_user", {"role": "admin"}),
        ("http_client_factory", "forged"),
        ("http_fetcher", "forged"),
        ("api_endpoint", "huggingface"),
        ("provider", "huggingface"),
        ("target_api_provider", "huggingface"),
        ("messages_payload", [{"role": "user", "content": "forged"}]),
        ("system_message", "forged"),
        ("streaming", True),
        ("_future_server_control", True),
    ],
)
def test_chat_request_rejects_server_managed_top_level_fields(
    field_name: str,
    field_value: object,
) -> None:
    """Public JSON cannot supply routing, credentials, or internal controls."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        field_name: field_value,
    }

    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(payload)

    error_text = str(exc_info.value)
    assert field_name in error_text
    assert "extra_body" in error_text
    assert [error["input"] for error in exc_info.value.errors()] == [field_name]
    assert "attacker-key" not in error_text
    assert "attacker.example" not in error_text


@pytest.mark.parametrize(
    "header_name",
    [
        "Authorization",
        "Proxy-Authorization",
        "Host",
        ":authority",
        "Content-Length",
        "Content-Type",
        "Transfer-Encoding",
        "Connection",
        "Forwarded",
        "X-Forwarded-Host",
        "X-Original-URL",
        "X-HTTP-Method-Override",
        "Cookie",
        "X-API-Key",
        "X-Amz-Date",
        "X-Amz-Content-Sha256",
        "ApiKey",
        "X-ApiKey-Value",
        "providerApiKey",
        "baseUrl",
        "OpenAI-Organization",
        "OpenAI-Project",
        "x-goog-user-project",
        "X_API_KEY",
        "X_GOOG_API_KEY",
        "Proxy_Authorization",
        "X_Provider_Extension",
    ],
)
def test_chat_request_rejects_server_managed_extra_headers_without_echoing_values(
    header_name: str,
) -> None:
    """Public extension headers cannot alter auth, authority, routing, or framing."""
    secret_value = "attacker-header-secret"

    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest.model_validate(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {header_name: secret_value},
            }
        )

    assert exc_info.value.errors()[0]["loc"] == ("extra_headers", header_name)
    assert exc_info.value.errors()[0]["input"] == header_name
    assert secret_value not in str(exc_info.value)


def test_chat_request_keeps_safe_provider_extension_headers() -> None:
    """Documented Bedrock guardrails and ordinary provider extensions remain usable."""
    request = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_headers": {
                "X-Amzn-Bedrock-GuardrailIdentifier": "guardrail-123",
                "X-Provider-Extension": "enabled",
            },
        }
    )

    assert request.extra_headers == {
        "X-Amzn-Bedrock-GuardrailIdentifier": "guardrail-123",
        "X-Provider-Extension": "enabled",
    }


def test_reserved_request_names_remain_managed_if_declared_in_the_future() -> None:
    """A future schema declaration cannot silently dismantle the trust boundary."""
    assert _is_server_managed_chat_request_field("base_url")


@pytest.mark.parametrize(
    "field_name",
    [
        "reasoning_effort",
        "service_tier",
        "performance_config_latency",
        "provider_options",
        "cache_control",
        "safety_identifier",
    ],
)
def test_safe_provider_option_names_are_not_treated_as_server_controls(
    field_name: str,
) -> None:
    """Credential filtering must not become a generic extension denylist."""
    assert not _is_server_managed_chat_request_field(field_name)
    request = ChatCompletionRequest.model_validate(
        {
            "model": "meta.llama3-8b-instruct",
            "messages": [{"role": "user", "content": "hi"}],
            field_name: "kept",
        }
    )
    assert request.model_extra == {field_name: "kept"}


def test_build_call_params_uses_declared_extra_body_for_provider_extensions() -> None:
    """Only the declared extension channel reaches adapter-specific dispatch."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "future_provider_option": "kept-for-provider",
            "extra_body": {"declared_extension": "kept"},
        }
    )

    params = build_call_params_from_request(
        request_data=req,
        target_api_provider="openai",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        final_system_message=None,
        app_config=None,
    )

    assert req.model_extra == {"future_provider_option": "kept-for-provider"}
    assert "future_provider_option" not in params
    assert params["extra_body"] == {"declared_extension": "kept"}


def test_build_call_params_drops_undeclared_fields_when_validation_is_bypassed() -> None:
    """The dispatch boundary keeps declared extensions and drops all model extras."""
    forged_controls = {
        "app_config": {"huggingface_api": {"api_base_url": "https://attacker.example"}},
        "base_url": "https://attacker.example",
        "trusted_base_url_override": True,
        "auth_user": {"role": "admin"},
        "credentials_resolved": True,
        "_future_server_control": True,
    }
    req = ChatCompletionRequest.model_construct(
        model="org/model",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"declared_extension": "kept"},
        future_provider_option="kept-for-provider",
        **forged_controls,
    )

    params = build_call_params_from_request(
        request_data=req,
        target_api_provider="huggingface",
        provider_api_key="trusted-runtime-key",
        templated_llm_payload=[{"role": "user", "content": "trusted"}],
        final_system_message="trusted system",
        app_config=None,
    )

    assert set(forged_controls).isdisjoint(params)
    assert "future_provider_option" not in params
    assert params["api_endpoint"] == "huggingface"
    assert params["api_key"] == "trusted-runtime-key"
    assert params["messages_payload"] == [{"role": "user", "content": "trusted"}]
    assert params["system_message"] == "trusted system"
    assert params["extra_body"] == {"declared_extension": "kept"}


def test_model_construct_boundary_reaches_real_huggingface_adapter_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forged extras are removed before the real adapter sees trusted runtime state."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_module

    calls: list[dict[str, Any]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"object": "chat.completion", "choices": []}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": dict(headers), "json": dict(json)})
            return _Response()

    monkeypatch.setattr(hf_module, "http_client_factory", lambda **_kwargs: _Client())
    request = ChatCompletionRequest.model_construct(
        model="org/trusted-model",
        messages=[{"role": "user", "content": "public"}],
        seed=1907,
        extra_body={"declared_extension": "kept"},
        extra_headers={
            "Host": "attacker.example",
            "X-Provider-Extension": "kept",
        },
        app_config={"huggingface_api": {"api_base_url": "https://attacker.example"}},
        base_url="https://attacker.example",
        credentials_resolved=True,
        _future_server_control=True,
    )
    trusted_app_config = {
        "huggingface_api": {
            "use_router_url_format": "true",
            "router_base_url": "https://trusted.example/hf-inference",
            "api_chat_path": "chat/completions",
        }
    }

    params = build_call_params_from_request(
        request_data=request,
        target_api_provider="huggingface",
        provider_api_key="trusted-key",
        templated_llm_payload=[{"role": "user", "content": "trusted"}],
        final_system_message="trusted system",
        app_config=trusted_app_config,
    )
    result = perform_chat_api_call(**params)

    assert result["object"] == "chat.completion"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == (
        "https://trusted.example/hf-inference/models/"
        "org/trusted-model/chat/completions"
    )
    assert call["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer trusted-key",
        "X-Provider-Extension": "kept",
    }
    assert call["json"]["declared_extension"] == "kept"
    assert call["json"]["seed"] == 1907
    assert call["json"]["messages"] == [
        {"role": "system", "content": "trusted system"},
        {"role": "user", "content": "trusted"},
    ]
    captured = repr(call)
    assert "attacker.example" not in captured
    assert "app_config" not in call["json"]
    assert "_future_server_control" not in call["json"]


@pytest.mark.concurrent
def test_concurrent_model_construct_boundaries_keep_trusted_dispatch_state_paired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic extra stripping remains request-local while real adapter calls overlap."""
    import tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter as hf_module

    calls: list[tuple[str, str, str, int]] = []
    lock = threading.Lock()
    both_arrived = threading.Event()
    release = threading.Event()

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"object": "chat.completion", "choices": []}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            assert "Host" not in headers
            with lock:
                calls.append(
                    (
                        url,
                        headers["Authorization"],
                        json["request_label"],
                        json["seed"],
                    )
                )
                if len(calls) == 2:
                    both_arrived.set()
            if not release.wait(10):
                raise TimeoutError("concurrent full-chain calls were not released")
            return _Response()

    monkeypatch.setattr(hf_module, "http_client_factory", lambda **_kwargs: _Client())

    def _params(label: str, seed: int) -> dict[str, Any]:
        request = ChatCompletionRequest.model_construct(
            model=f"org/model-{label}",
            messages=[{"role": "user", "content": label}],
            seed=seed,
            extra_body={"request_label": label},
            extra_headers={
                "Host": f"attacker-{label}.example",
                "ApiKey": f"attacker-api-key-{label}",
                "X-ApiKey-Value": f"attacker-key-value-{label}",
                "providerApiKey": f"attacker-provider-key-{label}",
                "baseUrl": f"https://attacker-{label}.example",
            },
            base_url=f"https://attacker-{label}.example",
            app_config={"huggingface_api": {"api_key": f"attacker-{label}"}},
            credentials_resolved=True,
        )
        params = build_call_params_from_request(
            request_data=request,
            target_api_provider="huggingface",
            provider_api_key=f"key-{label}",
            templated_llm_payload=[{"role": "user", "content": label}],
            final_system_message=None,
            app_config={
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": f"https://router-{label}.example/hf-inference",
                    "api_chat_path": "chat/completions",
                }
            },
        )
        return params

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(perform_chat_api_call, **_params("alpha", 1907)),
            executor.submit(perform_chat_api_call, **_params("beta", 7331)),
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
    assert "attacker" not in repr(calls)


def test_public_bedrock_controls_fail_before_real_signer_while_safe_extensions_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public credentials/region never reach SigV4; safe extensions still dispatch."""
    import tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter as bedrock_module

    signer_calls: list[dict[str, Any]] = []
    http_calls: list[dict[str, Any]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"object": "chat.completion", "choices": []}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            http_calls.append({"url": url, "headers": dict(headers), "json": dict(json)})
            return _Response()

    def _capture_signer(
        *,
        url: str,
        payload: dict[str, Any],
        region: str,
        credentials: Any,
    ) -> dict[str, str]:
        signer_calls.append(
            {
                "url": url,
                "payload": dict(payload),
                "region": region,
                "access_key_id": credentials.access_key_id,
                "secret_access_key": credentials.secret_access_key,
                "session_token": credentials.session_token,
            }
        )
        return {"Authorization": "AWS4-HMAC-SHA256 trusted-test-signature"}

    monkeypatch.setattr(bedrock_module, "http_client_factory", lambda **_kwargs: _Client())
    monkeypatch.setattr(bedrock_module, "_build_sigv4_headers", _capture_signer)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "trusted-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "trusted-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "trusted-session")
    malicious_payload = {
        "model": "meta.llama3-8b-instruct",
        "messages": [{"role": "user", "content": "malicious"}],
        "aws_access_key_id": "attacker-access",
        "bedrock_aws_secret_access_key": "attacker-secret",
        "aws_session_token": "attacker-session",
        "region": "attacker-region",
        "provider_endpoint_url": "https://attacker.example",
    }

    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(malicious_payload)

    assert signer_calls == []
    assert http_calls == []

    request = ChatCompletionRequest.model_validate(
        {
            "model": "meta.llama3-8b-instruct",
            "messages": [{"role": "user", "content": "safe"}],
            "extra_body": {"performanceConfig": {"latency": "optimized"}},
            "extra_headers": {
                "X-Amzn-Bedrock-GuardrailIdentifier": "guardrail-123",
            },
        }
    )
    params = build_call_params_from_request(
        request_data=request,
        target_api_provider="bedrock",
        provider_api_key=None,
        templated_llm_payload=[{"role": "user", "content": "safe"}],
        final_system_message=None,
        app_config={
            "bedrock_api": {
                "api_base_url": "https://bedrock-runtime.us-west-2.amazonaws.com/openai",
                "_runtime_auth_source": "aws_default_chain",
            }
        },
    )
    params["credentials_resolved"] = True

    result = perform_chat_api_call(**params)

    assert result["object"] == "chat.completion"
    assert len(signer_calls) == 1
    signer_call = signer_calls[0]
    assert signer_call["url"] == (
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    )
    assert signer_call["payload"]["model"] == "meta.llama3-8b-instruct"
    assert signer_call["payload"]["messages"] == [
        {"role": "user", "content": "safe"}
    ]
    assert signer_call["payload"]["stream"] is False
    assert signer_call["payload"]["performanceConfig"] == {"latency": "optimized"}
    assert signer_call["region"] == "us-west-2"
    assert signer_call["access_key_id"] == "trusted-access"
    assert signer_call["secret_access_key"] == "trusted-secret"
    assert signer_call["session_token"] == "trusted-session"
    assert len(http_calls) == 1
    assert http_calls[0]["headers"]["X-Amzn-Bedrock-GuardrailIdentifier"] == "guardrail-123"
    captured = repr(signer_calls + http_calls)
    assert "attacker" not in captured


@pytest.mark.concurrent
@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("aws_access_key_id", "attacker-access"),
        ("bedrock_aws_access_key_id", "attacker-access"),
        ("aws_secret_access_key", "attacker-secret"),
        ("bedrock_aws_secret_access_key", "attacker-secret"),
        ("aws_session_token", "attacker-session"),
        ("bedrock_aws_session_token", "attacker-session"),
        ("region", "attacker-region"),
        ("bedrock_region", "attacker-region"),
        ("extra_headers", {"X-Amz-Date": "20990101T000000Z"}),
        ("extra_headers", {"X-Amz-Content-Sha256": "attacker-hash"}),
        ("extra_headers", {"Content-Type": "text/plain"}),
    ],
)
def test_concurrent_public_bedrock_control_cannot_reach_inflight_real_signer(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    field_value: Any,
) -> None:
    """Each public Bedrock signing sink is denied during a legitimate in-flight call."""
    import tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter as bedrock_module

    signer_calls: list[tuple[str, str, str]] = []
    http_calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    legitimate_arrived = threading.Event()
    release = threading.Event()

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"object": "chat.completion", "choices": []}

    class _GatedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            del exc_type, exc, traceback
            return False

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            with lock:
                http_calls.append((url, json["messages"][0]["content"]))
                legitimate_arrived.set()
            if not release.wait(10):
                raise TimeoutError("legitimate Bedrock call was not released")
            return _Response()

    def _capture_signer(
        *,
        url: str,
        payload: dict[str, Any],
        region: str,
        credentials: Any,
    ) -> dict[str, str]:
        del url, payload
        with lock:
            signer_calls.append(
                (region, credentials.access_key_id, credentials.session_token)
            )
        return {"Authorization": "AWS4-HMAC-SHA256 trusted-test-signature"}

    monkeypatch.setattr(
        bedrock_module,
        "http_client_factory",
        lambda **_kwargs: _GatedClient(),
    )
    monkeypatch.setattr(bedrock_module, "_build_sigv4_headers", _capture_signer)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "trusted-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "trusted-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "trusted-session")

    def _dispatch_public(payload: dict[str, Any]) -> dict[str, Any]:
        request = ChatCompletionRequest.model_validate(payload)
        params = build_call_params_from_request(
            request_data=request,
            target_api_provider="bedrock",
            provider_api_key=None,
            templated_llm_payload=list(payload["messages"]),
            final_system_message=None,
            app_config={
                "bedrock_api": {
                    "api_base_url": "https://bedrock-runtime.us-east-2.amazonaws.com/openai",
                    "_runtime_auth_source": "aws_default_chain",
                }
            },
        )
        params["credentials_resolved"] = True
        return perform_chat_api_call(**params)

    legitimate_payload = {
        "model": "meta.llama3-8b-instruct",
        "messages": [{"role": "user", "content": "legitimate"}],
        "extra_body": {"performanceConfig": {"latency": "standard"}},
        "extra_headers": {"X-Provider-Extension": "kept"},
    }
    malicious_payload = {
        "model": "meta.llama3-8b-instruct",
        "messages": [{"role": "user", "content": "malicious"}],
        field_name: field_value,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        legitimate_future = executor.submit(_dispatch_public, legitimate_payload)
        try:
            assert legitimate_arrived.wait(10)
            malicious_future = executor.submit(_dispatch_public, malicious_payload)
            with pytest.raises(ValidationError):
                malicious_future.result(timeout=5)
        finally:
            release.set()
        assert legitimate_future.result(timeout=10)["object"] == "chat.completion"

    assert signer_calls == [("us-east-2", "trusted-access", "trusted-session")]
    assert http_calls == [
        (
            "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions",
            "legitimate",
        )
    ]
    captured = repr(signer_calls + http_calls)
    assert "attacker" not in captured
    assert "malicious" not in captured


@pytest.mark.unit
def test_build_call_params_excludes_extension_fields() -> None:
    """Ensure extension-only fields are stripped from call params."""
    req = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        history_message_limit=5,
        history_message_order="desc",
        slash_command_injection_mode="preface",
    )

    params = build_call_params_from_request(
        request_data=req,
        target_api_provider="openai",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        final_system_message=None,
        app_config=None,
    )

    assert "history_message_limit" not in params
    assert "history_message_order" not in params
    assert "slash_command_injection_mode" not in params
    assert params["api_endpoint"] == "openai"
    assert params["api_key"] == "test-key"
    assert params["messages_payload"]


@pytest.mark.unit
def test_build_call_params_excludes_research_context() -> None:
    """Ensure attached research context stays out of raw provider call params."""
    req = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        research_context={
            "run_id": "run_123",
            "query": "battery recycling supply chain",
            "question": "What changed in the battery recycling market?",
            "outline": [{"title": "Overview"}],
            "key_claims": [{"text": "Claim one"}],
            "unresolved_questions": ["What changed in Europe?"],
            "verification_summary": {"unsupported_claim_count": 0},
            "source_trust_summary": {"high_trust_count": 3},
            "research_url": "/research?run=run_123",
        },
    )

    params = build_call_params_from_request(
        request_data=req,
        target_api_provider="openai",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        final_system_message=None,
        app_config=None,
    )

    assert "research_context" not in params


@pytest.mark.unit
def test_build_call_params_negotiates_structured_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure shared call-param construction downgrades unsupported json_schema requests."""

    class _JsonObjectOnlyAdapter:
        def capabilities(self):
            return {"response_format_types": ["json_object"]}

    class _Registry:
        def get_adapter(self, _provider: str):
            return _JsonObjectOnlyAdapter()

    monkeypatch.setattr(adapter_registry, "get_registry", lambda: _Registry())

    req = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[{"role": "user", "content": "return structured"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
    )

    params = build_call_params_from_request(
        request_data=req,
        target_api_provider="openai",
        provider_api_key="test-key",
        templated_llm_payload=[{"role": "user", "content": "return structured"}],
        final_system_message=None,
        app_config=None,
    )

    assert params["response_format"] == {"type": "json_object"}
    assert params["_structured_requested_response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }
