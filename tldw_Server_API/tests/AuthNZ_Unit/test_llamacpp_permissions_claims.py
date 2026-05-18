from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as llamacpp_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


class _StubModelsHandler:
    config = type("Config", (), {"log_output_file": None})()
    _active_server_process = type("Process", (), {"pid": 1234, "returncode": None})()
    _active_server_host = "127.0.0.1"
    _active_server_port = 8080
    _active_server_model = "toy.gguf"

    async def get_server_status(self, **kwargs) -> dict[str, object]:  # noqa: ANN003
        _ = kwargs
        return {
            "status": "running",
            "host": self._active_server_host,
            "port": self._active_server_port,
            "model": self._active_server_model,
            "pid": self._active_server_process.pid,
        }

    async def list_models(self) -> list[str]:
        return ["toy.gguf"]

    def get_metrics(self) -> dict[str, int]:
        return {"starts": 1}


class _StubManager:
    class _Logger:
        def error(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return

    def __init__(self) -> None:
        self.logger = self._Logger()
        # Keep handler for `/llamacpp/models` and `/llamacpp/metrics`.
        # Start/stop/status resolve through manager compatibility methods.
        self.llamacpp = _StubModelsHandler()

    async def start_server(
        self,
        *,
        backend: str,
        model_name: str | None = None,
        server_args: dict | None = None,
    ) -> dict:
        return {
            "status": "started",
            "backend": backend,
            "model": model_name,
            "server_args": server_args or {},
        }

    async def stop_server(self, *, backend: str, pid: int | None = None, port: int | None = None) -> str:
        _ = (pid, port)
        return f"{backend} stopped"

    async def get_server_status(self, *, backend: str) -> dict:
        return {"status": "running", "backend": backend, "model": "toy.gguf"}

    async def list_local_models(self, *, backend: str) -> list[str]:
        _ = backend
        return ["toy.gguf"]


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(llamacpp_mod.router, prefix="/api/v1")
    app.state.llm_manager = _StubManager()

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None, "principal must be provided when fail_with_401 is False"
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
    app.dependency_overrides[llamacpp_mod.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[llamacpp_mod.get_job_manager] = lambda: object()

    return app


def _patch_provider_config_writes(monkeypatch) -> None:  # noqa: ANN001
    class FakeLock:
        def __enter__(self) -> "FakeLock":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(
        llamacpp_mod.llamacpp_provider_service.setup_manager,
        "update_config",
        lambda updates: None,
    )
    monkeypatch.setattr(llamacpp_mod.llamacpp_provider_service, "refresh_config_cache", lambda: None)
    monkeypatch.setattr(llamacpp_mod.llamacpp_provider_service, "llamacpp_config_write_lock", lambda: FakeLock())
    monkeypatch.setattr(
        llamacpp_mod.llamacpp_inventory_service,
        "preview_import_asset_folder",
        lambda path: {
            "folder": {
                "asset_id": "folder:authz",
                "kind": "folder",
                "identity_basis": "resolved_path",
                "path": str(path),
                "resolved_path": str(path),
                "display_name": "models",
                "source": "imported_folder",
                "metadata": {},
                "capabilities": ["asset_folder"],
                "mmproj_asset_ids": [],
                "base_model_asset_ids": [],
                "warnings": [],
            },
            "assets": [],
            "asset_counts": {},
            "warnings": [],
            "scan_limited": False,
            "will_persist": False,
        },
    )
    acquisition_job = {
        "job_id": "1",
        "status": "queued",
        "operation": "download",
        "queue": "acquisition",
        "source_label": "https://example.com/model.gguf",
        "destination_path": "/models/model.gguf",
        "asset_id": None,
        "progress": {},
        "warnings": [],
        "error_message": None,
    }
    monkeypatch.setattr(
        llamacpp_mod.llamacpp_acquisition_jobs,
        "create_download_job",
        lambda job_manager, payload, *, owner_user_id: acquisition_job,
    )
    monkeypatch.setattr(
        llamacpp_mod.llamacpp_acquisition_jobs,
        "get_download_job",
        lambda job_manager, job_id: acquisition_job,
    )
    monkeypatch.setattr(
        llamacpp_mod.llamacpp_acquisition_jobs,
        "list_download_jobs",
        lambda job_manager, *, limit=100: {"jobs": [acquisition_job]},
    )
    monkeypatch.setattr(
        llamacpp_mod.llamacpp_acquisition_jobs,
        "cancel_download_job",
        lambda job_manager, job_id: acquisition_job | {"status": "cancelled"},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llamacpp/start_server", {"model_filename": "toy.gguf", "server_args": {}}),
        ("post", "/api/v1/llamacpp/stop_server", {}),
        ("post", "/api/v1/llamacpp/use-in-chat", {}),
        ("get", "/api/v1/llamacpp/status", None),
        ("get", "/api/v1/llamacpp/models", None),
        ("get", "/api/v1/llamacpp/metrics", None),
        ("get", "/api/v1/llamacpp/logs/tail", None),
        ("get", "/api/v1/llamacpp/hardware", None),
        ("get", "/api/v1/llamacpp/profiles", None),
        ("post", "/api/v1/llamacpp/profiles", {"name": "Default", "model_path": "/models/model.gguf"}),
        ("get", "/api/v1/llamacpp/profiles/default", None),
        ("put", "/api/v1/llamacpp/profiles/default", {"name": "Updated"}),
        ("delete", "/api/v1/llamacpp/profiles/default", None),
        ("post", "/api/v1/llamacpp/profiles/default/start", None),
        ("post", "/api/v1/llamacpp/profiles/default/stop", None),
        ("post", "/api/v1/llamacpp/profiles/default/pause", None),
        ("post", "/api/v1/llamacpp/profiles/default/resume", None),
        ("post", "/api/v1/llamacpp/profiles/default/use-in-chat", None),
        ("get", "/api/v1/llamacpp/instances", None),
        ("get", "/api/v1/llamacpp/instances/default", None),
        ("get", "/api/v1/llamacpp/instances/default/logs/tail", None),
        ("post", "/api/v1/llamacpp/assets/import-folder/preview", {"path": "/models"}),
        ("post", "/api/v1/llamacpp/assets/downloads", {"url": "https://example.com/model.gguf"}),
        ("get", "/api/v1/llamacpp/assets/downloads", None),
        ("get", "/api/v1/llamacpp/assets/downloads/1", None),
        ("delete", "/api/v1/llamacpp/assets/downloads/1", None),
    ],
)
def test_llamacpp_lifecycle_401_when_principal_unavailable(
    monkeypatch,
    method: str,
    path: str,
    payload: dict | None,
):
    _patch_provider_config_writes(monkeypatch)
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        if method == "post":
            resp = client.post(path, json=payload or {})
        elif method == "put":
            resp = client.put(path, json=payload or {})
        elif method == "delete":
            resp = client.delete(path)
        else:
            resp = client.get(path)

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llamacpp/start_server", {"model_filename": "toy.gguf", "server_args": {}}),
        ("post", "/api/v1/llamacpp/stop_server", {}),
        ("post", "/api/v1/llamacpp/use-in-chat", {}),
        ("get", "/api/v1/llamacpp/status", None),
        ("get", "/api/v1/llamacpp/models", None),
        ("get", "/api/v1/llamacpp/metrics", None),
        ("get", "/api/v1/llamacpp/logs/tail", None),
        ("get", "/api/v1/llamacpp/hardware", None),
        ("get", "/api/v1/llamacpp/profiles", None),
        ("post", "/api/v1/llamacpp/profiles", {"name": "Default", "model_path": "/models/model.gguf"}),
        ("get", "/api/v1/llamacpp/profiles/default", None),
        ("put", "/api/v1/llamacpp/profiles/default", {"name": "Updated"}),
        ("delete", "/api/v1/llamacpp/profiles/default", None),
        ("post", "/api/v1/llamacpp/profiles/default/start", None),
        ("post", "/api/v1/llamacpp/profiles/default/stop", None),
        ("post", "/api/v1/llamacpp/profiles/default/pause", None),
        ("post", "/api/v1/llamacpp/profiles/default/resume", None),
        ("post", "/api/v1/llamacpp/profiles/default/use-in-chat", None),
        ("get", "/api/v1/llamacpp/instances", None),
        ("get", "/api/v1/llamacpp/instances/default", None),
        ("get", "/api/v1/llamacpp/instances/default/logs/tail", None),
        ("post", "/api/v1/llamacpp/assets/import-folder/preview", {"path": "/models"}),
        ("post", "/api/v1/llamacpp/assets/downloads", {"url": "https://example.com/model.gguf"}),
        ("get", "/api/v1/llamacpp/assets/downloads", None),
        ("get", "/api/v1/llamacpp/assets/downloads/1", None),
        ("delete", "/api/v1/llamacpp/assets/downloads/1", None),
    ],
)
def test_llamacpp_lifecycle_403_when_missing_admin_role(
    monkeypatch,
    method: str,
    path: str,
    payload: dict | None,
):
    _patch_provider_config_writes(monkeypatch)
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        if method == "post":
            resp = client.post(path, json=payload or {})
        elif method == "put":
            resp = client.put(path, json=payload or {})
        elif method == "delete":
            resp = client.delete(path)
        else:
            resp = client.get(path)

    assert resp.status_code == 403


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llamacpp/start_server", {"model_filename": "toy.gguf", "server_args": {}}),
        ("post", "/api/v1/llamacpp/stop_server", {}),
        ("post", "/api/v1/llamacpp/use-in-chat", {}),
        ("get", "/api/v1/llamacpp/status", None),
        ("get", "/api/v1/llamacpp/models", None),
        ("get", "/api/v1/llamacpp/metrics", None),
        ("get", "/api/v1/llamacpp/logs/tail", None),
        ("get", "/api/v1/llamacpp/hardware", None),
        ("post", "/api/v1/llamacpp/assets/import-folder/preview", {"path": "/models"}),
        ("post", "/api/v1/llamacpp/assets/downloads", {"url": "https://example.com/model.gguf"}),
        ("get", "/api/v1/llamacpp/assets/downloads", None),
        ("get", "/api/v1/llamacpp/assets/downloads/1", None),
        ("delete", "/api/v1/llamacpp/assets/downloads/1", None),
    ],
)
def test_llamacpp_lifecycle_200_for_admin_principal(
    monkeypatch,
    method: str,
    path: str,
    payload: dict | None,
):
    _patch_provider_config_writes(monkeypatch)
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        if method == "post":
            resp = client.post(path, json=payload or {})
        elif method == "delete":
            resp = client.delete(path)
        else:
            resp = client.get(path)

    assert resp.status_code == 200
