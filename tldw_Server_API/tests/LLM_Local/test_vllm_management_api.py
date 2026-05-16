from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import vllm_management as vm
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.VLLM_Management.sqlite_repo import SqliteVLLMInstanceRepository


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
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


def _make_app(repo: SqliteVLLMInstanceRepository) -> FastAPI:
    app = FastAPI()
    app.include_router(vm.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _admin_principal()
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        return principal

    async def _fake_check_rate_limit() -> None:
        return

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[vm.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[vm._resolve_vllm_repository] = lambda: repo
    return app


@pytest.mark.unit
def test_create_instance_returns_backend_metadata_and_persisted_record(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    app = _make_app(repo)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llm/providers/vllm/instances",
            json={
                "name": "embed-box",
                "execution_mode": "local",
                "launch_spec": {"model": "BAAI/bge-m3", "port": 8010},
                "declared_capabilities": {"embeddings": True},
            },
        )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["backend"] == "vllm"
    assert body["instance"]["name"] == "embed-box"
    assert body["instance"]["declared_capabilities"] == {"embeddings": True}
    assert repo.get_instance(body["instance"]["instance_id"]) is not None


@pytest.mark.unit
def test_admin_responses_redact_managed_vllm_secrets(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    app = _make_app(repo)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/llm/providers/vllm/instances",
            json={
                "name": "secure-box",
                "execution_mode": "ssh",
                "transport_config": {
                    "host": "gpu.internal",
                    "port": 22,
                    "user": "ubuntu",
                    "auth": {
                        "secret_ref": "VLLM_SSH_KEY_PATH",
                        "private_key_path": "/tmp/id_ed25519",
                    },
                    "probe_headers": {"X-Probe-Token": "probe-secret"},
                    "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
                },
                "launch_spec": {
                    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "port": 8001,
                    "api_key": "managed-secret",
                },
                "declared_capabilities": {"chat": True, "vision": True},
            },
        )
        instance_id = create_response.json()["instance"]["instance_id"]
        detail_response = client.get(f"/api/v1/llm/providers/vllm/instances/{instance_id}")
        list_response = client.get("/api/v1/llm/providers/vllm/instances")

    for body in (
        create_response.json(),
        detail_response.json(),
        list_response.json()["instances"][0],
    ):
        instance = body.get("instance", body)
        assert instance["launch_spec"]["api_key"] == "[REDACTED]"
        assert instance["transport_config"]["auth"]["secret_ref"] == "[REDACTED]"
        assert instance["transport_config"]["auth"]["private_key_path"] == "[REDACTED]"
        assert instance["transport_config"]["probe_headers"]["X-Probe-Token"] == "[REDACTED]"


@pytest.mark.unit
def test_admin_responses_redact_last_error_details(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="failed-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8050},
        ).to_domain()
    )
    repo.update_instance_runtime(
        instance.instance_id,
        {
            "desired_state": "running",
            "observed_state": "failed",
            "last_error": "ssh failed for /tmp/id_ed25519",
        },
    )
    app = _make_app(repo)

    with TestClient(app) as client:
        detail_response = client.get(f"/api/v1/llm/providers/vllm/instances/{instance.instance_id}")
        list_response = client.get("/api/v1/llm/providers/vllm/instances")

    assert detail_response.status_code == 200, detail_response.text
    assert list_response.status_code == 200, list_response.text
    assert detail_response.json()["instance"]["last_error"] == "[REDACTED]"
    assert list_response.json()["instances"][0]["last_error"] == "[REDACTED]"


@pytest.mark.unit
def test_create_instance_rejects_unimplemented_agent_mode(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    app = _make_app(repo)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llm/providers/vllm/instances",
            json={
                "name": "agent-box",
                "execution_mode": "agent",
                "launch_spec": {"model": "Qwen/Qwen2.5-7B-Instruct"},
            },
        )

    assert response.status_code == 422, response.text


@pytest.mark.unit
def test_patch_list_default_and_delete_flow(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    app = _make_app(repo)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/llm/providers/vllm/instances",
            json={
                "name": "vision-box",
                "execution_mode": "ssh",
                "transport_config": {
                    "host": "gpu.internal",
                    "port": 22,
                    "username": "ubuntu",
                    "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
                    "auth": {"secret_ref": "ssh-prod"},
                },
                "launch_spec": {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "port": 8001},
                "declared_capabilities": {"chat": True, "vision": True},
            },
        )
        instance_id = create_response.json()["instance"]["instance_id"]

        patch_response = client.patch(
            f"/api/v1/llm/providers/vllm/instances/{instance_id}",
            json={
                "name": "vision-box-v2",
                "routing_policy": {"is_default": True},
                "declared_capabilities": {"chat": True, "vision": True, "embeddings": False},
            },
        )
        default_response = client.post(
            "/api/v1/llm/providers/vllm/default",
            json={"instance_id": instance_id},
        )
        list_response = client.get("/api/v1/llm/providers/vllm/instances")
        detail_response = client.get(f"/api/v1/llm/providers/vllm/instances/{instance_id}")
        delete_response = client.delete(f"/api/v1/llm/providers/vllm/instances/{instance_id}")

    assert patch_response.status_code == 200, patch_response.text
    assert patch_response.json()["instance"]["name"] == "vision-box-v2"

    assert default_response.status_code == 200, default_response.text
    assert default_response.json()["default_instance_id"] == instance_id

    assert list_response.status_code == 200, list_response.text
    assert list_response.json()["default_instance_id"] == instance_id
    assert len(list_response.json()["instances"]) == 1

    assert detail_response.status_code == 200, detail_response.text
    assert detail_response.json()["instance"]["instance_id"] == instance_id

    assert delete_response.status_code == 200, delete_response.text
    assert delete_response.json()["deleted"] is True
    assert repo.get_instance(instance_id) is None
