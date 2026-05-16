from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as lp
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import DEFAULT_PROFILE_ID
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)


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


class _Logger:
    def error(self, *args: Any, **kwargs: Any) -> None:
        return


class _SupervisorStub:
    def __init__(self) -> None:
        self.profiles: dict[str, LlamaCppProfile] = {
            "one": LlamaCppProfile(
                profile_id="one",
                name="One",
                model_id="gguf:one",
                host="127.0.0.1",
                port=8181,
            ),
            "two": LlamaCppProfile(
                profile_id="two",
                name="Two",
                model_id="gguf:two",
                host="127.0.0.1",
                port=8182,
            ),
        }
        self.runtimes: dict[str, LlamaCppRuntime] = {
            "one": LlamaCppRuntime(
                profile_id="one",
                state=LlamaCppRuntimeState.RUNNING,
                host="127.0.0.1",
                port=8181,
                endpoint="http://127.0.0.1:8181",
                model_id="gguf:one",
                model_path="/models/one.gguf",
                log_tail_available=True,
            ),
            "two": LlamaCppRuntime(profile_id="two", state=LlamaCppRuntimeState.STOPPED),
        }
        self.started_profile_ids: list[str] = []
        self.tail_requests: list[tuple[str, int]] = []

    def list_profiles(self) -> list[LlamaCppProfile]:
        return list(self.profiles.values())

    async def create_profile(self, request: Any) -> LlamaCppProfile:
        profile = LlamaCppProfile(
            profile_id=request.profile_id or "created",
            name=request.name,
            enabled=request.enabled,
            mode=request.mode,
            model_id=request.model_id,
            model_path=request.model_path,
            mmproj_model_id=request.mmproj_model_id,
            host=request.host,
            port=request.port,
            port_policy=request.port_policy,
            server_args=dict(request.server_args),
            autostart=request.autostart,
            restart_policy=dict(request.restart_policy),
            provider_alias=request.provider_alias,
            tags=list(request.tags),
        )
        self.profiles[profile.profile_id] = profile
        self.runtimes.setdefault(profile.profile_id, LlamaCppRuntime(profile_id=profile.profile_id, state="defined"))
        return profile

    async def update_profile(self, profile_id: str, request: Any) -> LlamaCppProfile:
        existing = self.profiles[profile_id]
        updates = {field: getattr(request, field) for field in request.model_fields_set}
        updated = LlamaCppProfile.model_validate(existing.model_dump(mode="python") | updates)
        self.profiles[profile_id] = updated
        return updated

    async def delete_profile(self, profile_id: str) -> bool:
        self.profiles.pop(profile_id)
        self.runtimes.pop(profile_id, None)
        return True

    async def start_profile(self, profile_id: str) -> LlamaCppRuntime:
        self.started_profile_ids.append(profile_id)
        runtime = LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.RUNNING, host="127.0.0.1", port=8188)
        self.runtimes[profile_id] = runtime
        return runtime

    async def stop_profile(self, profile_id: str, disable: bool = False) -> LlamaCppRuntime:
        _ = disable
        runtime = LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.STOPPED)
        self.runtimes[profile_id] = runtime
        return runtime

    async def pause_profile(self, profile_id: str) -> LlamaCppRuntime:
        runtime = LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.PAUSED)
        self.runtimes[profile_id] = runtime
        return runtime

    async def resume_profile(self, profile_id: str) -> LlamaCppRuntime:
        runtime = LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.STOPPED, message="Resumed")
        self.runtimes[profile_id] = runtime
        return runtime

    def list_runtimes(self) -> list[LlamaCppRuntime]:
        return list(self.runtimes.values())

    def get_runtime(self, profile_id: str) -> LlamaCppRuntime:
        return self.runtimes[profile_id]

    def tail_logs(self, profile_id: str, lines: int) -> dict[str, object]:
        self.tail_requests.append((profile_id, lines))
        return {"lines": ["first", "second"], "truncated": False, "warnings": []}

    async def start_default_by_model(self, model_id: str, server_args: dict[str, object]) -> LlamaCppRuntime:
        _ = server_args
        self.started_profile_ids.append(DEFAULT_PROFILE_ID)
        runtime = LlamaCppRuntime(
            profile_id=DEFAULT_PROFILE_ID,
            state=LlamaCppRuntimeState.RUNNING,
            host="127.0.0.1",
            port=8181,
            endpoint="http://127.0.0.1:8181",
            model_id=model_id,
            model_path="/models/default.gguf",
        )
        self.runtimes[DEFAULT_PROFILE_ID] = runtime
        return runtime

    async def stop_default(self) -> LlamaCppRuntime:
        return await self.stop_profile(DEFAULT_PROFILE_ID)

    def default_status_compat(self) -> dict[str, object]:
        runtime = self.runtimes.get(DEFAULT_PROFILE_ID) or LlamaCppRuntime(
            profile_id=DEFAULT_PROFILE_ID,
            state=LlamaCppRuntimeState.STOPPED,
        )
        return {
            "status": "running" if runtime.state == LlamaCppRuntimeState.RUNNING else runtime.state.value,
            "backend": "llamacpp",
            "model": runtime.model_path,
            "path": runtime.model_path,
            "host": runtime.host,
            "port": runtime.port,
            "pid": runtime.pid,
        }


class _ManagerStub:
    logger = _Logger()

    def __init__(self, supervisor: _SupervisorStub) -> None:
        self.llamacpp_supervisor = supervisor
        self.llamacpp = None


def _make_app_with_manager(manager: _ManagerStub) -> FastAPI:
    app = FastAPI()
    app.include_router(lp.router, prefix="/api/v1")
    app.state.llm_manager = manager

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _admin_principal()
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(principal=principal, ip=ip, user_agent=ua, request_id=request_id)
        return principal

    async def _fake_check_rate_limit() -> None:
        return

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[lp.check_rate_limit] = _fake_check_rate_limit
    return app


@pytest.mark.unit
def test_profiles_crud_returns_profiles():
    supervisor = _SupervisorStub()
    app = _make_app_with_manager(_ManagerStub(supervisor))

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/llamacpp/profiles",
            json={
                "profile_id": "qwen",
                "name": "Qwen fixed port",
                "model_id": "gguf:qwen",
                "host": "127.0.0.1",
                "port": 8190,
                "port_policy": "explicit",
                "server_args": {"ctx_size": 4096},
            },
        )
        listed = client.get("/api/v1/llamacpp/profiles")
        updated = client.put("/api/v1/llamacpp/profiles/qwen", json={"name": "Qwen updated"})
        deleted = client.delete("/api/v1/llamacpp/profiles/qwen")

    assert created.status_code == 200, created.text
    assert created.json()["name"] == "Qwen fixed port"
    assert any(profile["profile_id"] == "qwen" for profile in listed.json()["profiles"])
    assert updated.json()["name"] == "Qwen updated"
    assert deleted.json() == {"profile_id": "qwen", "deleted": True}


@pytest.mark.unit
def test_instances_and_lifecycle_actions_use_supervisor():
    supervisor = _SupervisorStub()
    app = _make_app_with_manager(_ManagerStub(supervisor))

    with TestClient(app) as client:
        instances = client.get("/api/v1/llamacpp/instances")
        one = client.get("/api/v1/llamacpp/instances/one")
        started = client.post("/api/v1/llamacpp/profiles/two/start")
        paused = client.post("/api/v1/llamacpp/profiles/two/pause")
        resumed = client.post("/api/v1/llamacpp/profiles/two/resume")
        stopped = client.post("/api/v1/llamacpp/profiles/two/stop")
        logs = client.get("/api/v1/llamacpp/instances/one/logs/tail?lines=2")

    assert instances.status_code == 200, instances.text
    assert len(instances.json()["runtimes"]) == 2
    assert one.json()["profile_id"] == "one"
    assert started.json()["action"] == "start"
    assert started.json()["state"] == "running"
    assert paused.json()["state"] == "paused"
    assert resumed.json()["state"] == "stopped"
    assert stopped.json()["state"] == "stopped"
    assert logs.json()["lines"] == ["first", "second"]
    assert supervisor.tail_requests == [("one", 2)]


@pytest.mark.unit
def test_v1_start_by_model_targets_default_profile_only():
    supervisor = _SupervisorStub()
    app = _make_app_with_manager(_ManagerStub(supervisor))

    with TestClient(app) as client:
        started = client.post(
            "/api/v1/llamacpp/start-by-model",
            json={"model_id": "gguf:abc", "server_args": {"port": 8181}},
        )
        status = client.get("/api/v1/llamacpp/status")
        logs = client.get("/api/v1/llamacpp/logs/tail?lines=3")
        stopped = client.post("/api/v1/llamacpp/stop_server", json={})

    assert started.status_code == 200, started.text
    assert started.json()["model_id"] == "gguf:abc"
    assert started.json()["status"] == "running"
    assert supervisor.started_profile_ids == [DEFAULT_PROFILE_ID]
    assert status.json()["backend"] == "llamacpp"
    assert status.json()["status"] == "running"
    assert logs.json()["lines"] == ["first", "second"]
    assert supervisor.tail_requests == [(DEFAULT_PROFILE_ID, 3)]
    assert stopped.json()["status"] == "stopped"


@pytest.mark.unit
def test_v1_use_in_chat_uses_supervisor_default_runtime(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, Any]] = []
    supervisor = _SupervisorStub()
    supervisor.runtimes[DEFAULT_PROFILE_ID] = LlamaCppRuntime(
        profile_id=DEFAULT_PROFILE_ID,
        state=LlamaCppRuntimeState.RUNNING,
        host="0.0.0.0",
        port=8181,
    )
    app = _make_app_with_manager(_ManagerStub(supervisor))

    class FakeLock:
        def __enter__(self) -> "FakeLock":
            calls.append(("lock_enter", None))
            return self

        def __exit__(self, *exc: Any) -> None:
            calls.append(("lock_exit", None))

    monkeypatch.setattr(
        lp.llamacpp_provider_service.setup_manager,
        "update_config",
        lambda updates: calls.append(("update_config", updates)),
    )
    monkeypatch.setattr(
        lp.llamacpp_provider_service,
        "refresh_config_cache",
        lambda: calls.append(("refresh_config_cache", None)),
    )
    monkeypatch.setattr(lp.llamacpp_provider_service, "llamacpp_config_write_lock", lambda: FakeLock())

    with TestClient(app) as client:
        response = client.post("/api/v1/llamacpp/use-in-chat")

    assert response.status_code == 200, response.text
    assert calls == [
        ("lock_enter", None),
        ("update_config", {"Local-API": {"llama_api_IP": "http://127.0.0.1:8181"}}),
        ("refresh_config_cache", None),
        ("lock_exit", None),
    ]
    body = response.json()
    assert body["provider"] == "llama"
    assert body["endpoint"] == "http://127.0.0.1:8181"
    assert body["updated"] is True
