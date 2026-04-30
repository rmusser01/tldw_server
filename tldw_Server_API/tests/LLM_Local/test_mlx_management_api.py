from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import mlx as mlx_ep
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatProviderError


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(message.format(*args) if args else message)


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


class _RegistryStub:
    def __init__(
        self,
        *,
        load_result=None,
        unload_result=None,
        status_result=None,
        load_error: Exception | None = None,
        unload_error: Exception | None = None,
        status_error: Exception | None = None,
    ) -> None:
        self._load_result = load_result
        self._unload_result = unload_result
        self._status_result = status_result
        self._load_error = load_error
        self._unload_error = unload_error
        self._status_error = status_error
        self.last_model_path = None
        self.last_overrides = None

    def load(self, *, model_path=None, overrides=None):
        self.last_model_path = model_path
        self.last_overrides = dict(overrides or {})
        if self._load_error is not None:
            raise self._load_error
        if self._load_result is not None:
            return self._load_result
        return {"active": True, "model": model_path}

    def unload(self):
        if self._unload_error is not None:
            raise self._unload_error
        if self._unload_result is not None:
            return self._unload_result
        return {"status": "unloaded"}

    def status(self):
        if self._status_error is not None:
            raise self._status_error
        if self._status_result is not None:
            return self._status_result
        return {"active": True, "model": "stub-model"}


def _make_app_with_registry(registry: _RegistryStub) -> FastAPI:
    app = FastAPI()
    app.include_router(mlx_ep.router, prefix="/api/v1")

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
    app.dependency_overrides[mlx_ep.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[mlx_ep._resolve_mlx_registry] = lambda: registry
    return app


def _route_for_path(path: str) -> APIRoute:
    matches = [
        route
        for route in mlx_ep.router.routes
        if isinstance(route, APIRoute) and route.path == path
    ]
    assert len(matches) == 1
    return matches[0]


def _role_guard_dependencies(route: APIRoute) -> list[Callable[..., Any]]:
    expected_checker_code = auth_deps.RequireRole("admin").__code__
    return [
        dependency.dependency
        for dependency in route.dependencies
        if getattr(dependency.dependency, "__code__", None) is expected_checker_code
    ]


def _dependency_requires_roles(dependency: Callable[..., Any], roles: list[str]) -> bool:
    closure_values = [
        cell.cell_contents
        for cell in getattr(dependency, "__closure__", None) or ()
    ]
    return roles in closure_values


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "/llm/providers/mlx/load",
        "/llm/providers/mlx/unload",
        "/llm/providers/mlx/status",
    ],
)
def test_mlx_management_uses_standard_role_factory_alias(path: str) -> None:
    assert mlx_ep.RequireRole is auth_deps.RequireRole
    assert not hasattr(mlx_ep, "require_roles")
    role_guards = _role_guard_dependencies(_route_for_path(path))
    assert len(role_guards) == 1
    assert _dependency_requires_roles(role_guards[0], ["admin"])


@pytest.mark.unit
def test_mlx_load_uses_default_model_path_and_adds_backend(monkeypatch):
    registry = _RegistryStub(load_result={"active": True, "model": "stub-model"})
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "stub-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post("/api/v1/llm/providers/mlx/load", json={})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == "mlx"
    assert registry.last_model_path == "stub-model"
    assert registry.last_overrides == {}


@pytest.mark.unit
def test_mlx_load_preserves_explicit_model_path_and_overrides(monkeypatch):
    registry = _RegistryStub()
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "default-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llm/providers/mlx/load",
            json={"model_path": "explicit-model", "max_concurrent": 2},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == "mlx"
    assert registry.last_model_path == "explicit-model"
    assert registry.last_overrides == {"max_concurrent": 2}


@pytest.mark.unit
def test_mlx_load_trims_explicit_model_path(monkeypatch):
    registry = _RegistryStub()
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "default-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llm/providers/mlx/load",
            json={"model_path": "  explicit-model  "},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == "mlx"
    assert registry.last_model_path == "explicit-model"


@pytest.mark.unit
def test_mlx_load_blank_explicit_model_path_falls_back_to_default(monkeypatch):
    registry = _RegistryStub()
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "default-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llm/providers/mlx/load",
            json={"model_path": "   "},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == "mlx"
    assert registry.last_model_path == "default-model"


@pytest.mark.unit
def test_mlx_load_accepts_empty_post_body(monkeypatch):
    registry = _RegistryStub(load_result={"active": True, "model": "stub-model"})
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "stub-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post("/api/v1/llm/providers/mlx/load")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == "mlx"
    assert registry.last_model_path == "stub-model"
    assert registry.last_overrides == {}


@pytest.mark.unit
def test_mlx_unload_adds_backend():
    registry = _RegistryStub(unload_result={"status": "unloaded"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post("/api/v1/llm/providers/mlx/unload", json={})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "unloaded"
    assert body["backend"] == "mlx"


@pytest.mark.unit
def test_mlx_unload_accepts_empty_post_body():
    registry = _RegistryStub(unload_result={"status": "unloaded"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post("/api/v1/llm/providers/mlx/unload")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "unloaded"
    assert body["backend"] == "mlx"


@pytest.mark.unit
def test_mlx_status_adds_backend():
    registry = _RegistryStub(status_result={"active": False, "model": None})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.get("/api/v1/llm/providers/mlx/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active"] is False
    assert body["backend"] == "mlx"


@pytest.mark.unit
def test_mlx_load_maps_bad_request_error_to_400(monkeypatch):
    registry = _RegistryStub(load_error=ChatBadRequestError(provider="mlx", message="model_path is required"))
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "stub-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post("/api/v1/llm/providers/mlx/load", json={})

    assert response.status_code == 400
    assert "model_path is required" in response.json().get("detail", "")


@pytest.mark.unit
def test_mlx_load_maps_provider_error_to_500(monkeypatch):
    registry = _RegistryStub(load_error=ChatProviderError(provider="mlx", message="mlx-lm is not installed"))
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "stub-model"})
    app = _make_app_with_registry(registry)

    with TestClient(app) as client:
        response = client.post("/api/v1/llm/providers/mlx/load", json={})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to load MLX model"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "method", "registry_kwargs", "expected_detail", "expected_log"),
    [
        (
            "/api/v1/llm/providers/mlx/load",
            "post",
            {"load_error": RuntimeError("mlx backend exploded at /private/mlx")},
            "MLX load failed unexpectedly",
            "Unexpected MLX load failure",
        ),
        (
            "/api/v1/llm/providers/mlx/unload",
            "post",
            {"unload_error": RuntimeError("mlx backend exploded at /private/mlx")},
            "MLX unload failed unexpectedly",
            "Unexpected MLX unload failure",
        ),
        (
            "/api/v1/llm/providers/mlx/status",
            "get",
            {"status_error": RuntimeError("mlx backend exploded at /private/mlx")},
            "Failed to get MLX status",
            "Unexpected MLX status failure",
        ),
    ],
)
def test_mlx_generic_failure_logs_are_sanitized(
    monkeypatch,
    path: str,
    method: str,
    registry_kwargs: dict[str, Exception],
    expected_detail: str,
    expected_log: str,
):
    logger = _LoggerStub()
    monkeypatch.setattr(mlx_ep, "logger", logger)
    monkeypatch.setattr(mlx_ep, "_default_settings", lambda: {"model_path": "stub-model"})
    app = _make_app_with_registry(_RegistryStub(**registry_kwargs))

    with TestClient(app) as client:
        response = client.request(method, path, json={} if method == "post" else None)

    assert response.status_code == 500
    assert response.json()["detail"] == expected_detail
    assert logger.errors == [expected_log]
    logged = "\n".join(logger.errors)
    assert "mlx backend exploded" not in logged
    assert "/private/mlx" not in logged
