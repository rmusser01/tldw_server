from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as lp
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import InferenceError


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
    def error(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return


class _CapturingLogger:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(message.format(*args) if args else message)
        self.kwargs.append(kwargs)


class _ManagedStub:
    logger = _Logger()

    def __init__(self) -> None:
        self.llamacpp = self

    async def start_server(self, *, backend: str, model_name: str, server_args=None, **kwargs):  # noqa: ANN001
        _ = kwargs
        return {
            "status": "started",
            "backend": backend,
            "model": model_name,
            "server_args": server_args or {},
        }

    async def stop_server(self, *, backend: str, pid=None, port=None):  # noqa: ANN001
        _ = (backend, pid, port)
        return "Llama.cpp server stopped."

    async def get_server_status(self, backend: str):
        return {"status": "running", "model": "mock.gguf", "backend": backend}

    async def list_models(self):
        return ["mock.gguf", "other.gguf"]


class _FallbackModelsStub:
    logger = _Logger()
    llamacpp = None

    async def list_local_models(self, backend: str):
        return ["fallback.gguf"] if backend == "llamacpp" else []


class _StatusNoBackendStub(_ManagedStub):
    async def get_server_status(self, backend: str):
        _ = backend
        return {"status": "running", "model": "mock.gguf"}


class _StartNoStatusStub(_ManagedStub):
    async def start_server(self, *, backend: str, model_name: str, server_args=None, **kwargs):  # noqa: ANN001
        _ = (backend, server_args, kwargs)
        return {"model": model_name, "pid": 12345}


class _MetricsSyncStub(_ManagedStub):
    def get_metrics(self):
        return {"requests_total": 3}


class _MetricsAsyncStub(_ManagedStub):
    async def get_metrics(self):
        return {"requests_total": 7}


class _ExplodingManagedStub:
    def __init__(self, logger: _CapturingLogger) -> None:
        self.logger = logger
        self.llamacpp = self

    async def start_server(self, **kwargs: Any):
        raise RuntimeError("llamacpp backend exploded at /private/llama.cpp")

    async def stop_server(self, **kwargs: Any):
        raise RuntimeError("llamacpp backend exploded at /private/llama.cpp")

    async def get_server_status(self, **kwargs: Any):
        raise RuntimeError("llamacpp backend exploded at /private/llama.cpp")

    def get_metrics(self):
        raise RuntimeError("llamacpp backend exploded at /private/llama.cpp")

    async def list_models(self):
        raise RuntimeError("llamacpp backend exploded at /private/llama.cpp")


class _InferenceErrorManagedStub:
    logger = _Logger()

    def __init__(self) -> None:
        self.llamacpp = self

    async def start_server(self, **kwargs: Any):
        raise InferenceError("backend exploded at /private/llama.cpp with api_key=abc123")

    async def stop_server(self, **kwargs: Any):
        raise InferenceError("backend exploded at /private/llama.cpp with api_key=abc123")

    async def get_server_status(self, **kwargs: Any):
        raise InferenceError("backend exploded at /private/llama.cpp with api_key=abc123")

    def get_metrics(self):
        raise InferenceError("backend exploded at /private/llama.cpp with api_key=abc123")

    async def list_models(self):
        raise InferenceError("backend exploded at /private/llama.cpp with api_key=abc123")


class _ExplodingLlamafileMetricsStub:
    def __init__(self, logger: _CapturingLogger) -> None:
        self.logger = logger
        self.llamafile = self

    def get_metrics(self):
        raise RuntimeError("llamafile backend exploded at /private/llamafile.sock")


def _make_app_with_manager(manager) -> FastAPI:  # noqa: ANN001
    app = FastAPI()
    app.include_router(lp.router, prefix="/api/v1")
    app.state.llm_manager = manager

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
    app.dependency_overrides[lp.check_rate_limit] = _fake_check_rate_limit
    return app


@pytest.mark.unit
def test_llamacpp_management_uses_standard_role_factory_alias():
    assert lp.RequireRole is auth_deps.RequireRole
    assert not hasattr(lp, "require_roles")


@pytest.mark.unit
def test_llamacpp_start_server_happy_path():
    app = _make_app_with_manager(_ManagedStub())

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/llamacpp/start_server",
            json={"model_filename": "mock.gguf", "server_args": {"port": 8080}},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "started"
    assert body["model"] == "mock.gguf"


@pytest.mark.unit
def test_llamacpp_start_server_adds_status_when_missing():
    app = _make_app_with_manager(_StartNoStatusStub())

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/llamacpp/start_server",
            json={"model_filename": "mock.gguf", "server_args": {"port": 8080}},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "mock.gguf"
    assert body["pid"] == 12345
    assert body["status"] == "started"


@pytest.mark.unit
def test_llamacpp_stop_server_happy_path():
    app = _make_app_with_manager(_ManagedStub())

    with TestClient(app) as client:
        r = client.post("/api/v1/llamacpp/stop_server", json={})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "stopped"
    assert "message" in body
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
def test_llamacpp_status_happy_path():
    app = _make_app_with_manager(_ManagedStub())

    with TestClient(app) as client:
        r = client.get("/api/v1/llamacpp/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["model"] == "mock.gguf"
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
def test_llamacpp_status_adds_backend_when_missing():
    app = _make_app_with_manager(_StatusNoBackendStub())

    with TestClient(app) as client:
        r = client.get("/api/v1/llamacpp/status")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["model"] == "mock.gguf"
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
def test_llamacpp_models_happy_path():
    app = _make_app_with_manager(_ManagedStub())

    with TestClient(app) as client:
        r = client.get("/api/v1/llamacpp/models")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available_models"] == ["mock.gguf", "other.gguf"]
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
def test_llamacpp_models_fallback_to_manager_when_handler_missing():
    app = _make_app_with_manager(_FallbackModelsStub())

    with TestClient(app) as client:
        r = client.get("/api/v1/llamacpp/models")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available_models"] == ["fallback.gguf"]
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
def test_llamacpp_metrics_happy_path_sync():
    app = _make_app_with_manager(_MetricsSyncStub())

    with TestClient(app) as client:
        r = client.get("/api/v1/llamacpp/metrics")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requests_total"] == 3
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
def test_llamacpp_metrics_happy_path_async():
    app = _make_app_with_manager(_MetricsAsyncStub())

    with TestClient(app) as client:
        r = client.get("/api/v1/llamacpp/metrics")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requests_total"] == 7
    assert body["backend"] == "llamacpp"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "path", "payload", "expected_log"),
    [
        (
            "post",
            "/api/v1/llamacpp/start_server",
            {"model_filename": "mock.gguf", "server_args": {}},
            "Unexpected error starting Llama.cpp server",
        ),
        ("post", "/api/v1/llamacpp/stop_server", {}, "Unexpected error stopping Llama.cpp server"),
        ("get", "/api/v1/llamacpp/status", None, "Unexpected error getting Llama.cpp server status"),
        ("get", "/api/v1/llamacpp/metrics", None, "Unexpected error getting Llama.cpp metrics"),
        ("get", "/api/v1/llamacpp/models", None, "Unexpected error listing Llama.cpp models"),
    ],
)
def test_llamacpp_management_generic_failure_logs_are_sanitized(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    expected_log: str,
):
    logger = _CapturingLogger()
    app = _make_app_with_manager(_ExplodingManagedStub(logger))
    request_kwargs: dict[str, Any] = {}
    if payload is not None:
        request_kwargs["json"] = payload

    with TestClient(app) as client:
        response = client.request(method, path, **request_kwargs)

    assert response.status_code == 500
    assert response.json()["detail"] == "An unexpected error occurred."
    assert logger.errors == [expected_log]
    assert all(not kwargs.get("exc_info") for kwargs in logger.kwargs)
    logged = "\n".join(logger.errors)
    assert "llamacpp backend exploded" not in logged
    assert "/private/llama.cpp" not in logged


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/v1/llamacpp/start_server", {"model_filename": "mock.gguf", "server_args": {}}),
        ("post", "/api/v1/llamacpp/stop_server", {}),
        ("get", "/api/v1/llamacpp/status", None),
        ("get", "/api/v1/llamacpp/metrics", None),
        ("get", "/api/v1/llamacpp/models", None),
    ],
)
def test_llamacpp_management_inference_errors_return_safe_unavailable_detail(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
):
    app = _make_app_with_manager(_InferenceErrorManagedStub())
    request_kwargs: dict[str, Any] = {}
    if payload is not None:
        request_kwargs["json"] = payload

    with TestClient(app) as client:
        response = client.request(method, path, **request_kwargs)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "managed llama.cpp backend is not configured" in detail.lower()
    assert "backend exploded" not in detail
    assert "/private/llama.cpp" not in detail
    assert "api_key" not in detail.lower()


@pytest.mark.unit
def test_llamafile_metrics_generic_failure_log_is_sanitized():
    logger = _CapturingLogger()
    app = _make_app_with_manager(_ExplodingLlamafileMetricsStub(logger))

    with TestClient(app) as client:
        response = client.get("/api/v1/llamafile/metrics")

    assert response.status_code == 500
    assert response.json()["detail"] == "An unexpected error occurred."
    assert logger.errors == ["Unexpected error getting Llamafile metrics"]
    assert all(not kwargs.get("exc_info") for kwargs in logger.kwargs)
    logged = "\n".join(logger.errors)
    assert "llamafile backend exploded" not in logged
    assert "/private/llamafile.sock" not in logged
