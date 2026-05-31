import httpx
import pytest

from tldw_Server_API.app.core.Setup import provider_validation
from tldw_Server_API.app.core.Setup.provider_validation import (
    HostedProviderValidationRequest,
    LocalEndpointValidationRequest,
    validate_hosted_provider_credentials,
    validate_local_openai_endpoint,
    validate_native_kobold_endpoint,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, json_error: Exception | None = None):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None

    async def get(self, url, headers=None):
        self.requests.append((url, headers or {}))
        if self._error is not None:
            raise self._error
        return self._response

    async def post(self, url, headers=None, json=None):
        self.requests.append((url, headers or {}, json or {}))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.mark.asyncio
async def test_unreachable_local_endpoint_maps_to_unreachable(monkeypatch):
    fake_client = _FakeAsyncClient(error=TimeoutError("raw timeout details"))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            api_key="secret-local-token",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"
    assert "secret-local-token" not in response.model_dump_json()
    assert "raw timeout" not in response.message


@pytest.mark.asyncio
async def test_openai_models_shape_maps_to_ready(monkeypatch):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            200,
            {"object": "list", "data": [{"id": "local-model", "object": "model"}]},
        )
    )
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="ollama",
            base_url="http://127.0.0.1:11434/v1/",
        )
    )

    assert response.status == "ready"
    assert response.models == ["local-model"]
    assert fake_client.requests == [("http://127.0.0.1:11434/v1/models", {})]


@pytest.mark.asyncio
async def test_auth_failure_maps_to_auth_failed(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(401, {"error": "nope"}))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://127.0.0.1:8000/v1",
            api_key="secret-local-token",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "auth_failed"
    assert fake_client.requests[0][1] == {"Authorization": "Bearer secret-local-token"}
    assert "secret-local-token" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_unsupported_api_shape_maps_to_unsupported_api_shape(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(200, {"models": ["local-model"]}))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="llamacpp",
            base_url="http://127.0.0.1:8080/v1",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "unsupported_api_shape"


@pytest.mark.asyncio
async def test_invalid_json_maps_to_unsupported_api_shape(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(200, json_error=ValueError("bad json")))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="vllm",
            base_url="http://127.0.0.1:8000/v1",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "unsupported_api_shape"


@pytest.mark.asyncio
async def test_kobold_native_shape_maps_to_ready_and_posts_supplied_url(monkeypatch):
    raw_token = "secret-local-token"
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(200, {"results": [{"text": "ok"}]})
    )
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_native_kobold_endpoint(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp",
            base_url="http://127.0.0.1:5001/api/v1/generate",
            api_key=raw_token,
        )
    )

    assert response.status == "ready"
    assert response.failure_category is None
    assert fake_client.requests == [
        (
            "http://127.0.0.1:5001/api/v1/generate",
            {
                "Content-Type": "application/json",
                "X-Api-Key": raw_token,
            },
            {
                "prompt": "ping",
                "max_context_length": 128,
                "max_length": 1,
                "temperature": 0.0,
            },
        )
    ]
    assert raw_token not in response.model_dump_json()


@pytest.mark.asyncio
async def test_kobold_native_auth_failure_maps_to_auth_failed(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(403, {"error": "nope"}))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_native_kobold_endpoint(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp",
            base_url="http://127.0.0.1:5001/api/v1/generate",
            api_key="secret-local-token",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "auth_failed"
    assert "secret-local-token" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_kobold_native_bad_shape_maps_to_unsupported_api_shape(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(200, {"data": [{"id": "model"}]}))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_native_kobold_endpoint(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp",
            base_url="http://127.0.0.1:5001/api/v1/generate",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "unsupported_api_shape"


@pytest.mark.asyncio
async def test_kobold_native_invalid_url_maps_to_unreachable(monkeypatch):
    raw_url = "http://[::1"
    fake_client = _FakeAsyncClient(error=httpx.InvalidURL(f"Invalid URL: {raw_url}"))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_native_kobold_endpoint(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp",
            base_url=raw_url,
            api_key="secret-local-token",
        )
    )

    body = response.model_dump_json()
    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"
    assert raw_url not in body
    assert "Invalid URL" not in body
    assert "secret-local-token" not in body


@pytest.mark.asyncio
async def test_malformed_local_endpoint_url_maps_to_typed_failure(monkeypatch):
    raw_url = "http://[::1"
    fake_client = _FakeAsyncClient(error=httpx.InvalidURL(f"Invalid URL: {raw_url}"))
    monkeypatch.setattr(provider_validation, "_create_validation_client", lambda: fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url=raw_url,
            api_key="secret-local-token",
        )
    )

    body = response.model_dump_json()
    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"
    assert raw_url not in body
    assert "Invalid URL" not in body
    assert "secret-local-token" not in body


def test_hosted_provider_validation_requires_api_key():
    response = validate_hosted_provider_credentials(
        HostedProviderValidationRequest(provider_key="openai", api_key="   ")
    )

    assert response.status == "failed"
    assert response.failure_category == "provider_api_key_required"
    assert "api key" in (response.message or "").lower()


def test_hosted_provider_validation_accepts_plausible_openai_key_without_echo():
    raw_key = "sk-abcdefghijklmnopqrstuvwxyz"

    response = validate_hosted_provider_credentials(
        HostedProviderValidationRequest(provider_key="openai", api_key=raw_key)
    )

    assert response.status == "accepted"
    assert response.failure_category is None
    assert raw_key not in response.model_dump_json()


def test_hosted_provider_validation_rejects_malformed_openai_key_without_echo():
    raw_key = "not-an-openai-key"

    response = validate_hosted_provider_credentials(
        HostedProviderValidationRequest(provider_key="openai", api_key=raw_key)
    )

    assert response.status == "failed"
    assert response.failure_category == "provider_api_key_invalid"
    assert raw_key not in response.model_dump_json()


def test_hosted_provider_validation_accepts_other_hosted_nonblank_key():
    response = validate_hosted_provider_credentials(
        HostedProviderValidationRequest(provider_key="anthropic", api_key="anthropic-secret")
    )

    assert response.status == "accepted"
    assert response.failure_category is None
