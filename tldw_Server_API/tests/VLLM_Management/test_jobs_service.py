from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import vllm_management as vm
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.VLLM_Management.executors.base import LifecycleResult, ProbeResult
from tldw_Server_API.app.core.VLLM_Management.service import (
    VLLMManagementService,
    _probe_http_endpoint,
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


class _ScriptedExecutor:
    def __init__(
        self,
        *,
        start_result: LifecycleResult | Exception | None = None,
        probe_results: list[ProbeResult] | None = None,
    ) -> None:
        self._start_result = start_result or LifecycleResult(
            status="started",
            base_url="http://127.0.0.1:8018/v1",
            handle={"pid": 4321, "started_at": datetime.now(timezone.utc).isoformat()},
        )
        self._probe_results = list(probe_results or [])

    def start(self, instance):  # noqa: ANN001
        if isinstance(self._start_result, Exception):
            raise self._start_result
        return self._start_result

    def stop(self, instance, handle):  # noqa: ANN001
        raise AssertionError("stop should not be called in this test")

    def probe(self, instance):  # noqa: ANN001
        if not self._probe_results:
            raise AssertionError("probe called more times than expected")
        return self._probe_results.pop(0)


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


@pytest.mark.unit
def test_probe_http_endpoint_rejects_link_local_metadata_targets(monkeypatch):
    called = False

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        nonlocal called
        called = True
        raise AssertionError("urlopen should not run for blocked probe targets")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.VLLM_Management.service.urlopen",
        fake_urlopen,
    )

    probe = _probe_http_endpoint("http://169.254.169.254/v1")

    assert probe.reachable is False
    assert probe.status == "unhealthy"
    assert "blocked" in str(probe.detail or "").lower()
    assert called is False


@pytest.mark.unit
def test_start_instance_keeps_starting_state_on_initial_probe_miss(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="cold-boot-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8018},
        ).to_domain()
    )
    executor = _ScriptedExecutor(
        probe_results=[
            ProbeResult(
                status="unhealthy",
                reachable=False,
                base_url="http://127.0.0.1:8018/v1",
                detail="connection refused",
            )
        ]
    )
    service = VLLMManagementService(repository=repo, executors={"local": executor})

    result = service.start_instance(instance.instance_id)
    updated = repo.get_instance(instance.instance_id)

    assert result == {  # nosec B101
        "instance_id": instance.instance_id,
        "action": "start",
        "status": "starting",
    }
    assert updated is not None  # nosec B101
    assert updated.observed_state == "starting"  # nosec B101
    assert updated.desired_state == "running"  # nosec B101
    assert updated.last_error == "connection refused"  # nosec B101
    assert updated.executor_handle["pid"] == 4321  # nosec B101
    assert updated.executor_handle["started_at"] is not None  # nosec B101


@pytest.mark.unit
def test_follow_up_probe_can_promote_starting_instance_to_healthy(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="warming-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8019},
        ).to_domain()
    )
    executor = _ScriptedExecutor(
        probe_results=[
            ProbeResult(
                status="unhealthy",
                reachable=False,
                base_url="http://127.0.0.1:8019/v1",
                detail="connection refused",
            ),
            ProbeResult(
                status="healthy",
                reachable=True,
                base_url="http://127.0.0.1:8019/v1",
                capabilities={"chat": True},
            ),
        ]
    )
    service = VLLMManagementService(repository=repo, executors={"local": executor})

    service.start_instance(instance.instance_id)
    result = service.probe_instance(instance.instance_id)
    updated = repo.get_instance(instance.instance_id)

    assert result == {  # nosec B101
        "instance_id": instance.instance_id,
        "action": "probe",
        "status": "healthy",
    }
    assert updated is not None  # nosec B101
    assert updated.observed_state == "healthy"  # nosec B101
    assert updated.last_error is None  # nosec B101
    assert updated.probed_capabilities["chat"] is True  # nosec B101


@pytest.mark.unit
def test_startup_timeout_anchor_is_persisted_when_executor_omits_started_at(tmp_path, monkeypatch):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="anchor-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8020},
        ).to_domain()
    )
    monkeypatch.setenv("VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS", "300")
    executor = _ScriptedExecutor(
        start_result=LifecycleResult(
            status="started",
            base_url="http://127.0.0.1:8020/v1",
            handle={"pid": 7777},
        ),
        probe_results=[
            ProbeResult(
                status="unhealthy",
                reachable=False,
                base_url="http://127.0.0.1:8020/v1",
                detail="warming up",
            ),
            ProbeResult(
                status="unhealthy",
                reachable=False,
                base_url="http://127.0.0.1:8020/v1",
                detail="still warming up",
            ),
        ],
    )
    service = VLLMManagementService(repository=repo, executors={"local": executor})

    service.start_instance(instance.instance_id)
    first_update = repo.get_instance(instance.instance_id)
    assert first_update is not None  # nosec B101
    started_at = first_update.executor_handle.get("started_at")

    result = service.probe_instance(instance.instance_id)
    second_update = repo.get_instance(instance.instance_id)

    assert started_at is not None  # nosec B101
    assert result == {  # nosec B101
        "instance_id": instance.instance_id,
        "action": "probe",
        "status": "starting",
    }
    assert second_update is not None  # nosec B101
    assert second_update.executor_handle["started_at"] == started_at  # nosec B101
    assert second_update.observed_state == "starting"  # nosec B101


@pytest.mark.unit
def test_probe_instance_marks_starting_instance_unhealthy_after_startup_timeout(tmp_path, monkeypatch):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="stuck-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8021},
        ).to_domain()
    )
    started_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    repo.update_instance_runtime(
        instance.instance_id,
        {
            "desired_state": "running",
            "observed_state": "starting",
            "executor_handle": {"pid": 9999, "started_at": started_at},
        },
    )
    monkeypatch.setenv("VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS", "30")
    executor = _ScriptedExecutor(
        probe_results=[
            ProbeResult(
                status="unhealthy",
                reachable=False,
                base_url="http://127.0.0.1:8021/v1",
                detail="timed out waiting for server",
            )
        ]
    )
    service = VLLMManagementService(repository=repo, executors={"local": executor})

    result = service.probe_instance(instance.instance_id)
    updated = repo.get_instance(instance.instance_id)

    assert result == {  # nosec B101
        "instance_id": instance.instance_id,
        "action": "probe",
        "status": "unhealthy",
    }
    assert updated is not None  # nosec B101
    assert updated.observed_state == "unhealthy"  # nosec B101
    assert updated.last_error == "timed out waiting for server"  # nosec B101


@pytest.mark.unit
def test_start_instance_marks_failed_when_executor_start_raises(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    instance = repo.create_instance(
        vm.VLLMInstanceCreateRequest(
            name="broken-box",
            execution_mode="local",
            launch_spec={"model": "Qwen/Qwen2.5-7B-Instruct", "port": 8021},
        ).to_domain()
    )
    executor = _ScriptedExecutor(start_result=RuntimeError("spawn failed"))
    service = VLLMManagementService(repository=repo, executors={"local": executor})

    with pytest.raises(RuntimeError, match="spawn failed"):
        service.start_instance(instance.instance_id)

    updated = repo.get_instance(instance.instance_id)
    assert updated is not None  # nosec B101
    assert updated.observed_state == "failed"  # nosec B101
    assert updated.desired_state == "running"  # nosec B101
    assert updated.last_error == "spawn failed"  # nosec B101
