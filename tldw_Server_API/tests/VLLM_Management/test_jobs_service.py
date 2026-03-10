from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import vllm_management as vm
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.VLLM_Management.executors.base import ProbeResult
from tldw_Server_API.app.core.VLLM_Management.service import (
    VLLMManagementService,
    build_default_executor_map,
    build_probe_headers,
)
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


@pytest.mark.unit
def test_probe_instance_does_not_promote_probe_required_capabilities_without_probe_evidence(tmp_path):
    class _ReachabilityOnlyExecutor:
        def probe(self, instance):  # noqa: ANN001
            return ProbeResult(
                status="healthy",
                reachable=True,
                base_url="http://127.0.0.1:8016/v1",
                capabilities={},
            )

    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="embed-box",
            execution_mode="local",
            launch_spec={"model": "BAAI/bge-m3", "port": 8016},
            declared_capabilities={"chat": True, "embeddings": True, "vision": True},
        ).to_domain()
    )
    repo.update_instance_runtime(
        instance.instance_id,
        {
            "desired_state": "running",
            "observed_state": "starting",
        },
    )
    service = VLLMManagementService(repository=repo, executors={"local": _ReachabilityOnlyExecutor()})

    service.probe_instance(instance.instance_id)
    updated = repo.get_instance(instance.instance_id)

    assert updated is not None
    assert updated.observed_state == "healthy"
    assert updated.effective_capabilities["chat"] is True
    assert updated.effective_capabilities["embeddings"] is False
    assert updated.effective_capabilities["vision"] is False


@pytest.mark.unit
def test_build_probe_headers_supports_custom_api_key_header_shape():
    instance = vm.VLLMInstanceCreateRequest(
        name="proxy-box",
        execution_mode="local",
        transport_config={"probe_headers": {"X-Probe-Token": "probe-secret"}},
        launch_spec={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "api_key": "managed-secret",
            "api_key_header_name": "X-API-Key",
            "api_key_header_prefix": "Token",
        },
    ).to_domain()

    headers = build_probe_headers(instance)

    assert headers == {
        "X-Probe-Token": "probe-secret",
        "X-API-Key": "Token managed-secret",
    }


@pytest.mark.unit
def test_default_probe_uses_configured_auth_headers(monkeypatch):
    captured: dict[str, object] = {}

    class _DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _DummyResponse()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.VLLM_Management.service.urlopen",
        fake_urlopen,
    )

    repo_request = vm.VLLMInstanceCreateRequest(
        name="header-box",
        execution_mode="local",
        transport_config={"probe_headers": {"X-Probe-Token": "probe-secret"}},
        launch_spec={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "port": 8017,
            "api_key": "managed-secret",
            "api_key_header_name": "X-API-Key",
            "api_key_header_prefix": "Token",
        },
    ).to_domain()

    from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord

    instance = VLLMInstanceRecord(
        instance_id="header-box",
        name=repo_request.name,
        execution_mode=repo_request.execution_mode,
        transport_config=repo_request.transport_config,
        launch_spec=repo_request.launch_spec,
        routing_policy=repo_request.routing_policy,
        declared_capabilities=repo_request.declared_capabilities,
        desired_state="running",
        observed_state="starting",
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
    )

    probe = build_default_executor_map()["local"].probe(instance)

    assert probe.reachable is True
    assert captured["headers"] == {
        "X-probe-token": "probe-secret",
        "X-api-key": "Token managed-secret",
    }
    assert captured["timeout"] == 3
