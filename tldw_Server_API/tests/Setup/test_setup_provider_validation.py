import pytest

from tldw_Server_API.app.core.Setup import provider_validation
from tldw_Server_API.app.core.Setup.provider_validation import (
    LocalEndpointValidationRequest,
    validate_local_openai_endpoint,
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
