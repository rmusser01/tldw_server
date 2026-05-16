from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
    EmbeddingsBatchRequest,
    create_embeddings_batch_endpoint,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceCreate
from tldw_Server_API.app.core.VLLM_Management.sqlite_repo import SqliteVLLMInstanceRepository


def _seed_embeddings_repo(tmp_path):
    repo = SqliteVLLMInstanceRepository(tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        VLLMInstanceCreate(
            name="embed-l4",
            execution_mode="local",
            transport_config={},
            launch_spec={
                "model": "BAAI/bge-m3",
                "served_model_name": "BAAI/bge-m3",
                "port": 8010,
                "api_key": "managed-secret",
            },
            routing_policy={},
            declared_capabilities={"embeddings": True},
        )
    )
    repo.update_instance_runtime(
        instance.instance_id,
        {
            "desired_state": "running",
            "observed_state": "healthy",
            "effective_capabilities": {"embeddings": True},
            "last_known_base_url": "http://127.0.0.1:8010/v1",
        },
    )
    return repo, instance


@pytest.mark.asyncio
async def test_embeddings_batch_endpoint_routes_managed_instance_without_provider(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.VLLM_Management.resolver as resolver_module

    repo, instance = _seed_embeddings_repo(tmp_path)

    async def fake_create_embeddings_batch_async(**kwargs):
        assert kwargs["provider"] == "openai"
        assert kwargs["api_url"] == "http://127.0.0.1:8010/v1"
        assert kwargs["api_key"] == "managed-secret"
        return [[0.1, 0.2, 0.3]]

    async def fake_check_backpressure_and_quotas(request, user):  # noqa: ARG001
        return None

    monkeypatch.setattr(resolver_module, "get_default_vllm_instance_repository", lambda: repo)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced._should_enforce_policy_for_request",
        lambda request, user: False,
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

    payload = EmbeddingsBatchRequest(
        texts=["hello"],
        model="text-embedding-3-small",
        provider_instance_id=instance.instance_id,
    )
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/embeddings/batch", "headers": []})
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=[],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
    )
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
