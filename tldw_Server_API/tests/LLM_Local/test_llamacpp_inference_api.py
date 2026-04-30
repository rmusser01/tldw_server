from typing import Any, Tuple

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


class _DefaultMgr:
    logger = _Logger()
    llamacpp = True

    async def get_server_status(self, backend: str):
        return {"backend": backend, "model": "mock.gguf"}

    async def run_inference(self, backend: str, model_name_or_path: str, prompt=None, **kwargs):
        _ = prompt
        return {
            "model": model_name_or_path,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "kwargs": {"backend": backend, **kwargs},
        }


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


@pytest.fixture()
def llamacpp_client() -> Tuple[TestClient, dict]:
    app = _make_app_with_manager(_DefaultMgr())
    headers = {"Content-Type": "application/json"}
    client = TestClient(app)
    return client, headers


@pytest.mark.integration
def test_llamacpp_inference_happy_path(llamacpp_client, monkeypatch):
    client, headers = llamacpp_client

    # Patch llm_manager on the endpoint module
    class _Mgr:
        llamacpp = True
        logger = _Logger()

        async def get_server_status(self, backend: str):
            return {"backend": backend, "model": "mock.gguf"}

        async def run_inference(self, backend: str, model_name_or_path: str, prompt=None, **kwargs):
            # Echo a minimal OpenAI-style response
            return {
                "model": model_name_or_path,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "kwargs": kwargs,
            }

    import tldw_Server_API.app.api.v1.endpoints.llamacpp as lp

    stub = _Mgr()
    monkeypatch.setattr(lp, "llm_manager", stub, raising=False)
    # Ensure dependency resolver sees the stub instead of app.state.llm_manager.
    monkeypatch.setattr(client.app.state, "llm_manager", stub, raising=False)

    payload = {
        "model": "ignored-by-server",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
    }
    r = client.post("/api/v1/llamacpp/inference", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "mock.gguf"
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["backend"] == "llamacpp"


@pytest.mark.integration
def test_llamacpp_inference_falls_back_to_manager_when_handler_missing():
    class _MgrNoHandler:
        llamacpp = None
        logger = _Logger()

        async def get_server_status(self, backend: str):
            return {"backend": backend, "model": "mock.gguf"}

        async def run_inference(self, backend: str, model_name_or_path: str, prompt=None, **kwargs):
            _ = prompt
            return {
                "model": model_name_or_path,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "kwargs": {"backend": backend, **kwargs},
            }

    app = _make_app_with_manager(_MgrNoHandler())
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "ignored-by-server",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
    }

    with TestClient(app) as client:
        r = client.post("/api/v1/llamacpp/inference", json=payload, headers=headers)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "mock.gguf"
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["backend"] == "llamacpp"


@pytest.mark.integration
def test_llamacpp_inference_generic_failure_log_is_sanitized():
    class _ExplodingInferenceMgr:
        llamacpp = True

        def __init__(self, logger: _CapturingLogger) -> None:
            self.logger = logger

        async def get_server_status(self, backend: str):
            return {"backend": backend, "model": "mock.gguf"}

        async def run_inference(self, **kwargs: Any):
            raise RuntimeError("llamacpp inference exploded at /private/llama.cpp")

    logger = _CapturingLogger()
    app = _make_app_with_manager(_ExplodingInferenceMgr(logger))
    payload = {
        "model": "ignored-by-server",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/inference",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "An unexpected error occurred."
    assert logger.errors == ["Unexpected error during Llama.cpp inference"]
    assert all(not kwargs.get("exc_info") for kwargs in logger.kwargs)
    logged = "\n".join(logger.errors)
    assert "llamacpp inference exploded" not in logged
    assert "/private/llama.cpp" not in logged


@pytest.mark.integration
def test_llamacpp_inference_inference_error_returns_safe_unavailable_detail():
    class _InferenceErrorMgr:
        llamacpp = None
        logger = _Logger()

        async def get_server_status(self, backend: str):
            raise InferenceError("backend exploded at /private/llama.cpp with api_key=abc123")

        async def run_inference(self, **kwargs: Any):
            raise AssertionError("run_inference should not be reached")

    app = _make_app_with_manager(_InferenceErrorMgr())
    payload = {
        "model": "ignored-by-server",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/llamacpp/inference",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "managed llama.cpp backend is not configured" in detail.lower()
    assert "backend exploded" not in detail
    assert "/private/llama.cpp" not in detail
    assert "api_key" not in detail.lower()
