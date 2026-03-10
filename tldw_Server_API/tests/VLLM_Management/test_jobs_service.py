from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import vllm_management as vm
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.VLLM_Management.service import VLLMManagementService
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


class _RecordingJobManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_job(self, **kwargs):  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return {
            "id": 42,
            "uuid": "job-uuid-42",
            "status": "queued",
            "job_type": kwargs["job_type"],
            "payload": kwargs["payload"],
        }


def _make_app(repo: SqliteVLLMInstanceRepository, jm: _RecordingJobManager) -> FastAPI:
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
    app.dependency_overrides[vm.get_job_manager] = lambda: jm
    return app


@pytest.mark.unit
def test_service_enqueue_start_creates_vllm_management_job(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="worker-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8012},
        ).to_domain()
    )
    jm = _RecordingJobManager()
    service = VLLMManagementService(repository=repo, job_manager=jm)

    job = service.enqueue_start(instance.instance_id, owner_user_id="1")

    assert job["id"] == 42
    assert jm.calls[0]["domain"] == "vllm_management"
    assert jm.calls[0]["job_type"] == "vllm_instance_start"
    assert jm.calls[0]["payload"] == {"instance_id": instance.instance_id, "action": "start"}


@pytest.mark.unit
def test_start_endpoint_returns_job_metadata_instead_of_blocking(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    created = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="seeded-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8013},
        ).to_domain()
    )
    jm = _RecordingJobManager()
    app = _make_app(repo, jm)

    with TestClient(app) as client:
        response = client.post(f"/api/v1/llm/providers/vllm/instances/{created.instance_id}/start", json={})

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] == 42
    assert body["requested_action"] == "start"
    assert body["instance_id"] == created.instance_id

