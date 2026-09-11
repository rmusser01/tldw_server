"""Public chat request trust-boundary regressions for provider credentials."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls.providers import huggingface_adapter as hf_module

pytestmark = pytest.mark.integration

_DEFAULT_HF_URL = "https://api-inference.huggingface.co/v1/chat/completions"


class _FakeHuggingFaceResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "id": "chatcmpl-hf-boundary",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "org/runtime-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class _RecordingHuggingFaceTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.arrived = threading.Event()
        self.release = threading.Event()
        self.gated = False

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _FakeHuggingFaceResponse:
        with self.lock:
            self.calls.append({"url": url, "headers": dict(headers), "json": dict(json)})
            self.arrived.set()
        if self.gated and not self.release.wait(15):
            raise TimeoutError("legitimate Hugging Face call was not released")
        return _FakeHuggingFaceResponse()


class _FakeHuggingFaceClient:
    def __init__(self, transport: _RecordingHuggingFaceTransport) -> None:
        self._transport = transport

    def __enter__(self) -> _FakeHuggingFaceClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _FakeHuggingFaceResponse:
        return self._transport.post(url, headers=headers, json=json)


async def _resolve_key_only_huggingface(
    provider: str,
    **_kwargs: Any,
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key="stored-hf-key",
        app_config=None,
        credential_fields={},
        source="user",
        allowlisted=True,
        status=ByokResolutionStatus.RESOLVED,
        auth_source="api_key",
    )


class _KeyOnlyCredentialRuntime(ProviderCredentialRuntime):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(resolver=_resolve_key_only_huggingface, **kwargs)


async def _resolve_router_huggingface(
    provider: str,
    **_kwargs: Any,
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key="stored-router-key",
        app_config={
            "huggingface_api": {
                "use_router_url_format": "true",
                "router_base_url": "https://trusted-router.example/hf-inference",
                "api_chat_path": "chat/completions",
            }
        },
        credential_fields={},
        source="user",
        allowlisted=True,
        status=ByokResolutionStatus.RESOLVED,
        auth_source="api_key",
    )


class _RouterCredentialRuntime(ProviderCredentialRuntime):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(resolver=_resolve_router_huggingface, **kwargs)


@pytest.fixture
def huggingface_boundary(monkeypatch: pytest.MonkeyPatch) -> _RecordingHuggingFaceTransport:
    transport = _RecordingHuggingFaceTransport()
    monkeypatch.setenv("CHAT_FORCE_MOCK", "false")
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_HUGGINGFACE", "1")
    monkeypatch.setattr(chat_endpoint, "ProviderCredentialRuntime", _KeyOnlyCredentialRuntime)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: None)
    monkeypatch.setattr(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False)
    monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False)
    monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(
        hf_module,
        "http_client_factory",
        lambda **_kwargs: _FakeHuggingFaceClient(transport),
    )
    return transport


@pytest.fixture
def huggingface_router_boundary(
    monkeypatch: pytest.MonkeyPatch,
    huggingface_boundary: _RecordingHuggingFaceTransport,
) -> _RecordingHuggingFaceTransport:
    monkeypatch.setattr(chat_endpoint, "ProviderCredentialRuntime", _RouterCredentialRuntime)
    return huggingface_boundary


def _legitimate_request() -> dict[str, Any]:
    return {
        "api_provider": "huggingface",
        "model": "org/runtime-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "save_to_db": False,
    }


def test_key_only_public_chat_uses_server_key_and_default_huggingface_route(
    authenticated_client,
    huggingface_boundary: _RecordingHuggingFaceTransport,
) -> None:
    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_legitimate_request(),
    )

    assert response.status_code == 200, response.text
    assert len(huggingface_boundary.calls) == 1
    call = huggingface_boundary.calls[0]
    assert call["url"] == _DEFAULT_HF_URL
    assert call["headers"]["Authorization"] == "Bearer stored-hf-key"


@pytest.mark.parametrize(
    "header_name",
    [
        "Authorization",
        "Host",
        "Proxy-Authorization",
        "Content-Length",
        "Forwarded",
        "X-API-Key",
    ],
)
def test_public_server_managed_extra_headers_are_rejected_before_dispatch(
    authenticated_client,
    huggingface_boundary: _RecordingHuggingFaceTransport,
    header_name: str,
) -> None:
    body = _legitimate_request() | {
        "extra_headers": {
            header_name: "attacker-header-secret",
        }
    }

    response = authenticated_client.post("/api/v1/chat/completions", json=body)

    assert response.status_code == 422, response.text
    assert header_name in response.text
    assert "attacker-header-secret" not in response.text
    assert huggingface_boundary.calls == []


def test_public_safe_extra_header_reaches_huggingface_adapter(
    authenticated_client,
    huggingface_boundary: _RecordingHuggingFaceTransport,
) -> None:
    body = _legitimate_request() | {
        "extra_headers": {"X-Provider-Extension": "allowed"}
    }

    response = authenticated_client.post("/api/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    assert len(huggingface_boundary.calls) == 1
    headers = huggingface_boundary.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer stored-hf-key"
    assert headers["X-Provider-Extension"] == "allowed"


def test_public_unknown_extra_is_ignored_while_extra_body_reaches_provider(
    authenticated_client,
    huggingface_boundary: _RecordingHuggingFaceTransport,
) -> None:
    body = _legitimate_request() | {
        "future_provider_option": "must-not-dispatch",
        "extra_body": {"declared_extension": "kept"},
    }

    response = authenticated_client.post("/api/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    assert len(huggingface_boundary.calls) == 1
    outbound = huggingface_boundary.calls[0]["json"]
    assert outbound["declared_extension"] == "kept"
    assert "future_provider_option" not in outbound


@pytest.mark.parametrize(
    "forged_controls",
    [
        {
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "api_base_url": "https://attacker.example/v1",
                    "_runtime_base_url_override": True,
                }
            }
        },
        {
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "false",
                    "api_base_url": "https://attacker.example/v1",
                }
            }
        },
        {
            "app_config": {
                "huggingface_api": {
                    "use_router_url_format": "true",
                    "router_base_url": "https://attacker.example/hf-inference",
                }
            }
        },
        {
            "base_url": "https://attacker.example/v1",
            "trusted_base_url_override": True,
        },
        {
            "base_url": "https://attacker.example/v1",
            "auth_user": {"role": "admin"},
        },
    ],
    ids=[
        "forged-app-config-provenance",
        "forged-app-config-direct-base",
        "forged-app-config-router-base",
        "forged-trust-flag",
        "forged-admin-user",
    ],
)
def test_public_chat_rejects_server_managed_controls_before_adapter_dispatch(
    authenticated_client,
    huggingface_boundary: _RecordingHuggingFaceTransport,
    forged_controls: dict[str, Any],
) -> None:
    body = _legitimate_request() | forged_controls

    response = authenticated_client.post("/api/v1/chat/completions", json=body)

    assert response.status_code == 422, response.text
    response_text = response.text
    assert any(field_name in response_text for field_name in forged_controls)
    assert "extra_body" in response_text
    assert "attacker.example" not in response_text
    assert huggingface_boundary.calls == []


@pytest.mark.concurrent
def test_concurrent_malicious_request_cannot_influence_legitimate_huggingface_call(
    authenticated_client,
    huggingface_boundary: _RecordingHuggingFaceTransport,
) -> None:
    huggingface_boundary.gated = True
    malicious_body = _legitimate_request() | {
        "app_config": {
            "huggingface_api": {
                "use_router_url_format": "true",
                "api_base_url": "https://attacker.example/v1",
                "_runtime_base_url_override": True,
            }
        }
    }

    legitimate_future = None
    malicious_future = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            legitimate_future = executor.submit(
                authenticated_client.post,
                "/api/v1/chat/completions",
                json=_legitimate_request(),
            )
            assert huggingface_boundary.arrived.wait(15)
            malicious_future = executor.submit(
                authenticated_client.post,
                "/api/v1/chat/completions",
                json=malicious_body,
            )
            malicious_response = malicious_future.result(timeout=15)
            assert malicious_response.status_code == 422, malicious_response.text
            assert "attacker.example" not in malicious_response.text
            assert len(huggingface_boundary.calls) == 1
            huggingface_boundary.release.set()
            legitimate_response = legitimate_future.result(timeout=15)
        finally:
            huggingface_boundary.release.set()
            if legitimate_future is not None:
                legitimate_future.result(timeout=15)
            if malicious_future is not None:
                malicious_future.result(timeout=15)

    assert legitimate_response.status_code == 200, legitimate_response.text
    assert len(huggingface_boundary.calls) == 1
    call = huggingface_boundary.calls[0]
    assert call["url"] == _DEFAULT_HF_URL
    assert call["headers"]["Authorization"] == "Bearer stored-hf-key"
    assert "attacker.example" not in repr(call)


def test_public_huggingface_router_rejects_unsafe_model_before_dispatch(
    authenticated_client,
    huggingface_router_boundary: _RecordingHuggingFaceTransport,
) -> None:
    malicious_model = "../../api/whoami-v2#"

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_legitimate_request() | {"model": malicious_model},
    )

    assert response.status_code == 400, response.text
    assert malicious_model not in response.text
    assert "stored-router-key" not in response.text
    assert huggingface_router_boundary.calls == []


@pytest.mark.concurrent
def test_concurrent_public_unsafe_model_cannot_redirect_legitimate_router_call(
    authenticated_client,
    huggingface_router_boundary: _RecordingHuggingFaceTransport,
) -> None:
    huggingface_router_boundary.gated = True
    legitimate_future = None
    malicious_future = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            legitimate_future = executor.submit(
                authenticated_client.post,
                "/api/v1/chat/completions",
                json=_legitimate_request() | {"seed": 1907},
            )
            assert huggingface_router_boundary.arrived.wait(15)
            malicious_future = executor.submit(
                authenticated_client.post,
                "/api/v1/chat/completions",
                json=_legitimate_request()
                | {"model": "../../api/whoami-v2#", "seed": 7331},
            )
            malicious_response = malicious_future.result(timeout=15)
            assert malicious_response.status_code == 400, malicious_response.text
            assert len(huggingface_router_boundary.calls) == 1
            huggingface_router_boundary.release.set()
            legitimate_response = legitimate_future.result(timeout=15)
        finally:
            huggingface_router_boundary.release.set()
            if legitimate_future is not None:
                legitimate_future.result(timeout=15)
            if malicious_future is not None:
                malicious_future.result(timeout=15)

    assert legitimate_response.status_code == 200, legitimate_response.text
    assert len(huggingface_router_boundary.calls) == 1
    call = huggingface_router_boundary.calls[0]
    assert call["url"] == (
        "https://trusted-router.example/hf-inference/models/"
        "org/runtime-model/chat/completions"
    )
    assert call["headers"]["Authorization"] == "Bearer stored-router-key"
    assert call["json"]["seed"] == 1907
