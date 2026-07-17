from configparser import ConfigParser

import httpx
import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.api.v1.schemas.setup_schemas import SetupProviderSaveRequest
from tldw_Server_API.app.core.exceptions import EgressPolicyError
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
from tldw_Server_API.app.core.Setup import provider_validation
from tldw_Server_API.app.core.Setup.provider_validation import (
    HostedProviderValidationRequest,
    LocalEndpointValidationRequest,
    validate_hosted_provider_credentials,
)
from tldw_Server_API.app.core.Setup.provider_validation import (
    validate_local_openai_endpoint as _validate_local_openai_endpoint,
)
from tldw_Server_API.app.core.Setup.provider_validation import (
    validate_native_kobold_endpoint as _validate_native_kobold_endpoint,
)


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload=None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    def close(self) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
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


def _scope(url: str) -> ConfiguredEndpointScope:
    try:
        return ConfiguredEndpointScope.from_url(url)
    except ValueError:
        return ConfiguredEndpointScope(scheme="http", host="invalid", port=80)


async def validate_local_openai_endpoint(payload, *, configured_endpoint=None):
    return await _validate_local_openai_endpoint(
        payload,
        configured_endpoint=configured_endpoint or _scope(payload.base_url),
    )


async def validate_native_kobold_endpoint(payload, *, configured_endpoint=None):
    return await _validate_native_kobold_endpoint(
        payload,
        configured_endpoint=configured_endpoint or _scope(payload.base_url),
    )


def _patch_validation_fetch(monkeypatch, fake_client):
    async def fake_afetch(*, method, url, headers=None, json=None, **_kwargs):
        if method == "POST":
            return await fake_client.post(url, headers=headers, json=json)
        return await fake_client.get(url, headers=headers)

    monkeypatch.setattr(provider_validation, "afetch", fake_afetch)


def _patch_policy_denial(monkeypatch):
    async def deny(**_kwargs):
        raise EgressPolicyError("blocked", reason_code="address_forbidden")

    monkeypatch.setattr(provider_validation, "afetch", deny)


def _patch_dns_failure(monkeypatch):
    async def fail_dns(**_kwargs):
        raise EgressPolicyError(
            "DNS failed for http://secret-host.invalid/private",
            reason_code="dns_unresolved",
        )

    monkeypatch.setattr(provider_validation, "afetch", fail_dns)


def _config_with_openai_key(raw_key: str) -> ConfigParser:
    parser = ConfigParser()
    parser.optionxform = str
    parser.add_section("API")
    parser.set("API", "openai_api_key", raw_key)
    return parser


@pytest.mark.asyncio
async def test_openai_validation_uses_checked_afetch_with_route_scope(monkeypatch):
    scope = ConfiguredEndpointScope.from_url("http://10.0.0.5:18080/v1")
    calls = []

    async def fake_afetch(**kwargs):
        calls.append(kwargs)
        return _FakeResponse(200, {"data": [{"id": "lan-model"}]})

    monkeypatch.setattr(provider_validation, "afetch", fake_afetch)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="llamacpp",
            base_url="http://10.0.0.5:18080/v1",
        ),
        configured_endpoint=scope,
    )

    assert response.status == "ready"
    assert response.models == ["lan-model"]
    assert len(calls) == 1
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "http://10.0.0.5:18080/v1/models"
    assert calls[0]["headers"] == {}
    assert calls[0]["timeout"] == 5.0
    assert calls[0]["retry"].attempts == 1
    assert calls[0]["configured_endpoint"] is scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://host.docker.internal:18080/v1",
        "http://192.168.50.20:18080/v1",
        "http://[fd12:3456:789a::20]:18080/v1",
        "http://100.64.0.20:18080/v1",
    ],
)
async def test_openai_validation_forwards_scopes_for_local_network_classes(
    monkeypatch,
    base_url,
):
    seen_scopes = []

    async def fake_afetch(**kwargs):
        seen_scopes.append(kwargs["configured_endpoint"])
        return _FakeResponse(200, {"data": [{"id": "local-model"}]})

    monkeypatch.setattr(provider_validation, "afetch", fake_afetch)
    scope = ConfiguredEndpointScope.from_url(base_url)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(provider_key="llamacpp", base_url=base_url),
        configured_endpoint=scope,
    )

    assert response.status == "ready"
    assert seen_scopes == [scope]


@pytest.mark.asyncio
async def test_local_validator_requires_preconstructed_scope():
    with pytest.raises(TypeError):
        await _validate_local_openai_endpoint(  # type: ignore[call-arg]
            LocalEndpointValidationRequest(
                provider_key="llamacpp",
                base_url="http://127.0.0.1:8080/v1",
            )
        )


@pytest.mark.asyncio
async def test_openai_dns_unresolved_maps_to_reachability_without_details(monkeypatch):
    _patch_dns_failure(monkeypatch)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="llamacpp",
            base_url="http://secret-host.invalid:18080/v1",
            api_key="secret-local-token",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"
    assert "secret-host" not in response.model_dump_json()
    assert "secret-local-token" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_kobold_dns_unresolved_maps_to_reachability_without_details(monkeypatch):
    _patch_dns_failure(monkeypatch)

    response = await validate_native_kobold_endpoint(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp",
            base_url="http://secret-host.invalid:5001/api/v1/generate",
            api_key="secret-local-token",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"
    assert "secret-host" not in response.model_dump_json()
    assert "secret-local-token" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_unreachable_local_endpoint_maps_to_unreachable(monkeypatch):
    fake_client = _FakeAsyncClient(error=TimeoutError("raw timeout details"))
    _patch_validation_fetch(monkeypatch, fake_client)

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
@pytest.mark.parametrize(
    "validator",
    [validate_local_openai_endpoint, validate_native_kobold_endpoint],
)
async def test_unexpected_local_validation_error_is_logged_without_secrets(
    monkeypatch,
    validator,
):
    raw_secret = "secret-local-token"
    raw_url = "http://secret-host.invalid:18080/v1"
    fake_client = _FakeAsyncClient(error=RuntimeError(f"boom {raw_secret} {raw_url}"))
    _patch_validation_fetch(monkeypatch, fake_client)
    logged: list[tuple[dict, tuple]] = []

    monkeypatch.setattr(
        provider_validation.logger,
        "bind",
        lambda **context: type(
            "_CapturedLogger",
            (),
            {"error": lambda self, _message, *args: logged.append((context, args))},
        )(),
    )

    response = await validator(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp" if validator is validate_native_kobold_endpoint else "llamacpp",
            base_url=raw_url,
            api_key=raw_secret,
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"
    assert logged
    assert logged[0][0]["provider_key"] in {"koboldcpp", "llamacpp"}
    serialized_log = repr(logged)
    assert raw_secret not in serialized_log
    assert raw_url not in serialized_log


@pytest.mark.asyncio
async def test_openai_models_shape_maps_to_ready(monkeypatch):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            200,
            {"object": "list", "data": [{"id": "local-model", "object": "model"}]},
        )
    )
    _patch_validation_fetch(monkeypatch, fake_client)

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
async def test_openai_models_probe_appends_v1_when_base_url_omits_version(
    monkeypatch,
):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            200,
            {"object": "list", "data": [{"id": "local-model", "object": "model"}]},
        )
    )
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://127.0.0.1:8000/",
        )
    )

    assert response.status == "ready"
    assert response.models == ["local-model"]
    assert fake_client.requests == [("http://127.0.0.1:8000/v1/models", {})]


@pytest.mark.asyncio
async def test_openai_models_shape_ready_can_gate_first_chat(monkeypatch):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            200,
            {"object": "list", "data": [{"id": "local-model", "object": "model"}]},
        )
    )
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )
    )

    assert response.status == "ready"
    assert response.validation_level == "live_non_generative"
    assert response.can_gate_first_chat is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://203.0.113.10:8000/v1",
        "http://169.254.169.254/latest/meta-data",
        "ftp://127.0.0.1:11434/v1",
    ],
)
async def test_openai_local_validation_rejects_disallowed_targets_before_request(
    monkeypatch,
    base_url,
):
    _patch_policy_denial(monkeypatch)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url=base_url,
            api_key="secret-local-token",
        )
    )

    body = response.model_dump_json()
    assert response.status == "failed"
    assert response.failure_category == "local_provider_endpoint_not_allowed"
    assert base_url not in body
    assert "secret-local-token" not in body


@pytest.mark.asyncio
async def test_openai_local_validation_allows_private_ip_targets(monkeypatch):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            200,
            {"object": "list", "data": [{"id": "private-model", "object": "model"}]},
        )
    )
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://10.0.0.5:8000/v1",
        )
    )

    assert response.status == "ready"
    assert response.models == ["private-model"]
    assert fake_client.requests == [("http://10.0.0.5:8000/v1/models", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://ollama.local:11434/v1",
        "http://llama.lan:8080/v1",
        "http://workstation.home:5000/v1",
        "http://gpu-box.internal:8000/v1",
    ],
)
async def test_openai_local_validation_allows_common_local_domain_targets(
    monkeypatch,
    base_url,
):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(
            200,
            {"object": "list", "data": [{"id": "local-domain-model", "object": "model"}]},
        )
    )
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url=base_url,
        )
    )

    assert response.status == "ready"
    assert response.models == ["local-domain-model"]
    assert fake_client.requests == [(f"{base_url}/models", {})]


@pytest.mark.asyncio
async def test_auth_failure_maps_to_auth_failed(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(401, {"error": "nope"}))
    _patch_validation_fetch(monkeypatch, fake_client)

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
async def test_models_endpoint_unsupported_shape_accepts_manual_model_fallback(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(200, {"models": ["local-model"]}))
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="llamacpp",
            base_url="http://127.0.0.1:8080/v1",
        )
    )

    assert response.status == "accepted"
    assert response.models == []
    assert response.validation_level == "live_endpoint_shape"
    assert response.can_gate_first_chat is True
    assert response.failure_category == "model_discovery_unavailable"
    assert "manual" in (response.message or "").lower()


@pytest.mark.asyncio
async def test_models_endpoint_not_found_accepts_manual_model_fallback(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(404, {"error": "not found"}))
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://127.0.0.1:8000/v1",
            model="manual-local-model",
        )
    )

    assert response.status == "accepted"
    assert response.models == []
    assert response.validation_level == "live_endpoint_shape"
    assert response.can_gate_first_chat is True
    assert response.failure_category == "model_discovery_unavailable"
    assert "manual-local-model" not in response.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 500, 503])
async def test_models_endpoint_transient_failure_does_not_accept_manual_fallback(
    monkeypatch,
    status_code,
):
    fake_client = _FakeAsyncClient(
        response=_FakeResponse(status_code, {"error": "temporarily unavailable"})
    )
    _patch_validation_fetch(monkeypatch, fake_client)

    response = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://127.0.0.1:8000/v1",
            model="manual-local-model",
        )
    )

    assert response.status == "failed"
    assert response.failure_category == "local_provider_unreachable"


@pytest.mark.asyncio
async def test_invalid_json_maps_to_unsupported_api_shape(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(200, json_error=ValueError("bad json")))
    _patch_validation_fetch(monkeypatch, fake_client)

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
    fake_client = _FakeAsyncClient(response=_FakeResponse(200, {"results": [{"text": "ok"}]}))
    _patch_validation_fetch(monkeypatch, fake_client)

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
async def test_kobold_native_rejects_disallowed_target_before_request(monkeypatch):
    base_url = "http://169.254.169.254/api/v1/generate"
    _patch_policy_denial(monkeypatch)

    response = await validate_native_kobold_endpoint(
        LocalEndpointValidationRequest(
            provider_key="koboldcpp",
            base_url=base_url,
            api_key="secret-local-token",
        )
    )

    body = response.model_dump_json()
    assert response.status == "failed"
    assert response.failure_category == "local_provider_endpoint_not_allowed"
    assert base_url not in body
    assert "secret-local-token" not in body


@pytest.mark.asyncio
async def test_kobold_native_auth_failure_maps_to_auth_failed(monkeypatch):
    fake_client = _FakeAsyncClient(response=_FakeResponse(403, {"error": "nope"}))
    _patch_validation_fetch(monkeypatch, fake_client)

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
    _patch_validation_fetch(monkeypatch, fake_client)

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
    _patch_validation_fetch(monkeypatch, fake_client)

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
    _patch_validation_fetch(monkeypatch, fake_client)

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
    assert response.validation_level == "local_syntax"
    assert response.can_gate_first_chat is True
    assert raw_key not in response.model_dump_json()


def test_hosted_provider_validation_rejects_malformed_openai_key_without_echo():
    raw_key = "not-an-openai-key"

    response = validate_hosted_provider_credentials(
        HostedProviderValidationRequest(provider_key="openai", api_key=raw_key)
    )

    assert response.status == "failed"
    assert response.failure_category == "provider_api_key_invalid"
    assert response.can_gate_first_chat is False
    assert raw_key not in response.model_dump_json()


def test_hosted_provider_validation_accepts_other_hosted_nonblank_key():
    response = validate_hosted_provider_credentials(
        HostedProviderValidationRequest(provider_key="anthropic", api_key="anthropic-secret")
    )

    assert response.status == "accepted"
    assert response.failure_category is None


@pytest.mark.asyncio
async def test_hosted_provider_save_uses_existing_key_for_model_update_without_echo(
    monkeypatch,
):
    raw_existing_key = "configured-provider-key"
    updates_seen = []
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "_load_config_parser",
        lambda: _config_with_openai_key(raw_existing_key),
    )
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "update_config",
        lambda updates: updates_seen.append(updates),
    )
    monkeypatch.setattr(
        setup_endpoint,
        "_refresh_runtime_config_cache",
        lambda _context: True,
    )

    response = await setup_endpoint.save_first_run_provider(
        SetupProviderSaveRequest(
            provider_key="openai",
            model="gpt-4.1",
            make_default=True,
        )
    )

    assert response.status == "saved"
    assert response.credential_configured is True
    assert response.masked_api_key is None
    assert response.model == "gpt-4.1"
    assert raw_existing_key not in response.model_dump_json()
    assert updates_seen == [
        {
            "API": {
                "openai_model": "gpt-4.1",
                "default_api": "openai",
            }
        }
    ]


@pytest.mark.asyncio
async def test_hosted_provider_save_without_raw_key_rejects_missing_existing_key(
    monkeypatch,
):
    def fail_update_config(_updates):
        pytest.fail(
            "hosted provider save without configured credentials must not write config"
        )

    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "_load_config_parser",
        lambda: _config_with_openai_key("your_api_key_here"),
    )
    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", fail_update_config)

    with pytest.raises(HTTPException) as exc_info:
        await setup_endpoint.save_first_run_provider(
            SetupProviderSaveRequest(
                provider_key="openai",
                model="gpt-4.1",
                make_default=True,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "provider_api_key_required"
