from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
    EmbeddingProvider,
    EmbeddingsBatchRequest,
    _resolve_managed_vllm_embeddings_route,
    build_provider_config,
    create_embeddings_batch_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthorizationError, ChatBadRequestError
from tldw_Server_API.app.core.Chat.chat_service import build_call_params_from_request
from tldw_Server_API.app.core.VLLM_Management.resolver import ResolvedVLLMRoute


def _principal(*, is_admin: bool = False, subject: str | None = None, user_id: int = 1) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        api_key_id=None,
        subject=subject,
        token_type="access",
        jti=None,
        roles=["admin"] if is_admin else [],
        permissions=[],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def test_build_call_params_injects_resolved_vllm_route(monkeypatch):
    request_data = ChatCompletionRequest(
        api_provider="vllm",
        provider_instance_id="vision-id",
        model="legacy-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    def fake_resolver(**_: object) -> ResolvedVLLMRoute:
        return ResolvedVLLMRoute(
            instance_id="vision-id",
            base_url="http://10.0.0.9:8000/v1",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            api_key="managed-secret",
            effective_capabilities={"chat": True},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    cleaned_args = build_call_params_from_request(
        request_data=request_data,
        target_api_provider="vllm",
        provider_api_key=None,
        templated_llm_payload=[{"role": "user", "content": "hello"}],
        final_system_message=None,
        app_config=None,
        principal=_principal(is_admin=True),
    )

    assert cleaned_args["base_url"] == "http://10.0.0.9:8000/v1"
    assert cleaned_args["server_resolved_base_url_override"] is True
    assert cleaned_args["trusted_base_url_override"] is True
    assert cleaned_args["api_key"] == "managed-secret"
    assert cleaned_args["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_build_call_params_rejects_explicit_provider_instance_id_for_non_admin_principal():
    request_data = ChatCompletionRequest(
        api_provider="vllm",
        provider_instance_id="vision-id",
        model="legacy-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    with pytest.raises(ChatAuthorizationError, match="provider_instance_id requires an admin or single-user principal"):
        build_call_params_from_request(
            request_data=request_data,
            target_api_provider="vllm",
            provider_api_key=None,
            templated_llm_payload=[{"role": "user", "content": "hello"}],
            final_system_message=None,
            app_config=None,
            principal=_principal(user_id=999),
        )


def test_build_call_params_allows_explicit_provider_instance_id_for_admin_principal(monkeypatch):
    request_data = ChatCompletionRequest(
        api_provider="vllm",
        provider_instance_id="vision-id",
        model="legacy-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    def fake_resolver(**_: object) -> ResolvedVLLMRoute:
        return ResolvedVLLMRoute(
            instance_id="vision-id",
            base_url="http://10.0.0.9:8000/v1",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            api_key="managed-secret",
            effective_capabilities={"chat": True},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    cleaned_args = build_call_params_from_request(
        request_data=request_data,
        target_api_provider="vllm",
        provider_api_key=None,
        templated_llm_payload=[{"role": "user", "content": "hello"}],
        final_system_message=None,
        app_config=None,
        principal=_principal(is_admin=True),
    )

    assert cleaned_args["base_url"] == "http://10.0.0.9:8000/v1"


def test_build_call_params_requires_vision_for_image_messages(monkeypatch):
    request_data = ChatCompletionRequest(
        api_provider="vllm",
        provider_instance_id="vision-id",
        model="legacy-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
    )
    captured: dict[str, object] = {}

    def fake_resolver(**kwargs: object) -> ResolvedVLLMRoute:
        captured.update(kwargs)
        return ResolvedVLLMRoute(
            instance_id="vision-id",
            base_url="http://10.0.0.9:8000/v1",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            api_key="managed-secret",
            effective_capabilities={"chat": True, "vision": True},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    cleaned_args = build_call_params_from_request(
        request_data=request_data,
        target_api_provider="vllm",
        provider_api_key=None,
        templated_llm_payload=request_data.messages,
        final_system_message=None,
        app_config=None,
        principal=_principal(is_admin=True),
    )

    assert set(captured["required_capability"]) == {"chat", "vision"}
    assert cleaned_args["base_url"] == "http://10.0.0.9:8000/v1"


def test_build_call_params_surfaces_unhealthy_managed_vllm_route_as_bad_request(monkeypatch):
    request_data = ChatCompletionRequest(
        api_provider="vllm",
        provider_instance_id="vision-id",
        model="legacy-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    def fake_resolver(**_: object) -> ResolvedVLLMRoute:
        raise ValueError("Managed vLLM instance 'vision-id' is not healthy (observed_state='starting')")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    with pytest.raises(ChatBadRequestError, match="Managed vLLM instance 'vision-id' is not healthy"):
        build_call_params_from_request(
            request_data=request_data,
            target_api_provider="vllm",
            provider_api_key=None,
            templated_llm_payload=[{"role": "user", "content": "hello"}],
            final_system_message=None,
            app_config=None,
            principal=_principal(is_admin=True),
        )


def test_embeddings_helper_maps_managed_vllm_to_openai(monkeypatch):
    def fake_resolver(**_: object) -> ResolvedVLLMRoute:
        return ResolvedVLLMRoute(
            instance_id="embed-id",
            base_url="http://127.0.0.1:8010/v1",
            model="BAAI/bge-m3",
            api_key=None,
            effective_capabilities={"embeddings": True},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    managed_route, provider, model = _resolve_managed_vllm_embeddings_route(
        provider="vllm",
        provider_instance_id="embed-id",
        model="legacy-model",
        principal=_principal(is_admin=True),
    )

    assert managed_route is not None
    assert managed_route.base_url == "http://127.0.0.1:8010/v1"
    assert provider == "openai"
    assert model == "BAAI/bge-m3"


def test_embeddings_helper_surfaces_unhealthy_managed_vllm_route_as_http_400(monkeypatch):
    def fake_resolver(**_: object) -> ResolvedVLLMRoute:
        raise ValueError("Managed vLLM instance 'embed-id' is not healthy (observed_state='failed')")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    with pytest.raises(HTTPException) as exc_info:
        _resolve_managed_vllm_embeddings_route(
            provider="vllm",
            provider_instance_id="embed-id",
            model="legacy-model",
            principal=_principal(is_admin=True),
        )

    exc = exc_info.value
    assert exc.status_code == 400
    assert "Managed vLLM instance 'embed-id' is not healthy" in exc.detail


def test_embeddings_helper_rejects_explicit_provider_instance_id_for_non_admin_principal():
    with pytest.raises(HTTPException) as exc_info:
        _resolve_managed_vllm_embeddings_route(
            provider="vllm",
            provider_instance_id="embed-id",
            model="legacy-model",
            principal=_principal(user_id=999),
        )

    exc = exc_info.value
    assert exc.status_code == 403
    assert "provider_instance_id requires an admin or single-user principal" in exc.detail


def test_embeddings_helper_allows_explicit_provider_instance_id_for_single_user_principal(monkeypatch):
    def fake_resolver(**_: object) -> ResolvedVLLMRoute:
        return ResolvedVLLMRoute(
            instance_id="embed-id",
            base_url="http://127.0.0.1:8010/v1",
            model="BAAI/bge-m3",
            api_key="managed-secret",
            effective_capabilities={"embeddings": True},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.resolve_vllm_instance_for_request",
        fake_resolver,
    )

    managed_route, provider, model = _resolve_managed_vllm_embeddings_route(
        provider="vllm",
        provider_instance_id="embed-id",
        model="legacy-model",
        principal=_principal(subject="single_user"),
    )

    assert managed_route is not None
    assert provider == "openai"
    assert model == "BAAI/bge-m3"


def test_build_provider_config_preserves_openai_api_url_override():
    config = build_provider_config(
        EmbeddingProvider.OPENAI,
        "BAAI/bge-m3",
        api_key="managed-secret",
        api_url="http://127.0.0.1:8010/v1",
    )

    assert config["provider"] == "openai"
    assert config["api_url"] == "http://127.0.0.1:8010/v1"


@pytest.mark.asyncio
async def test_embeddings_batch_endpoint_routes_managed_vllm_over_openai_transport(monkeypatch):
    def fake_managed_route(**_: object):
        return (
            ResolvedVLLMRoute(
                instance_id="embed-id",
                base_url="http://127.0.0.1:8010/v1",
                model="BAAI/bge-m3",
                api_key="managed-secret",
                effective_capabilities={"embeddings": True},
            ),
            "openai",
            "BAAI/bge-m3",
        )

    async def fake_create_embeddings_batch_async(**kwargs: object):
        assert kwargs["provider"] == "openai"
        assert kwargs["api_url"] == "http://127.0.0.1:8010/v1"
        assert kwargs["api_key"] == "managed-secret"
        return [[0.1, 0.2, 0.3]]

    async def fake_check_backpressure_and_quotas(request: object, user: object):
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._resolve_managed_vllm_embeddings_route",
        fake_managed_route,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._should_enforce_policy_for_request",
        lambda request, user: True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._check_backpressure_and_quotas",
        fake_check_backpressure_and_quotas,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._build_user_metadata",
        lambda user: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.count_tokens",
        lambda text, model: 1,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._get_allowed_providers",
        lambda: {"vllm"},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._get_allowed_models",
        lambda: ["BAAI/*"],
    )

    payload = EmbeddingsBatchRequest(
        texts=["hello"],
        model="legacy-model",
        provider="vllm",
        provider_instance_id="embed-id",
    )
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/embeddings/batch", "headers": []})
    response = Response()
    current_user = SimpleNamespace(id=1)

    result = await create_embeddings_batch_endpoint(
        payload=payload,
        request=request,
        current_user=current_user,
        response=response,
    )

    assert result.provider == "vllm"
    assert result.model == "BAAI/bge-m3"
