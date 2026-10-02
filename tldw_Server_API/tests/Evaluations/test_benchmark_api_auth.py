"""Benchmark API routes must require evals.read / evals.manage (TASK-13399).

The ratchet fix in route_auth_ratchet.py made these routes visible for the
first time: list/info/samples had no auth dependency at all, and run/
simpleqa-evaluate had only a rate limiter (not an authenticator). Mirrors the
auth test pattern already used for evaluations_datasets.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import benchmark_api as benchmark_module
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_MANAGE, EVALS_READ
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = pytest.mark.unit


def _make_app() -> FastAPI:
    """Return a standalone app mounting only the benchmarks router."""
    app = FastAPI()
    app.include_router(benchmark_module.router, prefix="/api/v1")
    return app


def _override_user(app: FastAPI, permissions: list[str]) -> None:
    """Override the app's eval auth deps with a fake user holding ``permissions``."""

    async def _verify_api_key_override() -> str:
        """Stand in for API-key verification; the test never checks the key itself."""
        return "user_1"

    async def _get_user_override() -> User:
        """Return a fake authenticated user whose permissions the test controls live."""
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            permissions=list(permissions),
            is_admin=False,
        )

    app.dependency_overrides[evaluations_auth.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[evaluations_auth.get_eval_request_user] = _get_user_override


def test_read_routes_reject_unauthenticated_requests() -> None:
    """GET list/info/samples must 401 without credentials."""
    app = _make_app()
    with TestClient(app) as client:
        assert client.get("/api/v1/benchmarks/list").status_code == 401
        assert client.get("/api/v1/benchmarks/demo/info").status_code == 401
        assert client.get("/api/v1/benchmarks/demo/samples").status_code == 401


def test_run_routes_reject_unauthenticated_requests() -> None:
    """POST run/simpleqa-evaluate must 401 without credentials."""
    app = _make_app()
    with TestClient(app) as client:
        assert client.post("/api/v1/benchmarks/demo/run", json={}).status_code == 401
        assert (
            client.post(
                "/api/v1/benchmarks/simpleqa/evaluate",
                params={"question": "2+2?"},
            ).status_code
            == 401
        )


def test_read_routes_reject_missing_evals_read_permission() -> None:
    """GET list must 403 without evals.read and 200 with it."""
    app = _make_app()
    current_permissions: list[str] = []
    _override_user(app, current_permissions)

    with TestClient(app) as client:
        current_permissions[:] = []
        forbidden = client.get("/api/v1/benchmarks/list")
        assert forbidden.status_code == 403

        current_permissions[:] = [EVALS_READ]
        allowed = client.get("/api/v1/benchmarks/list")
        assert allowed.status_code == 200, allowed.text


def test_evals_read_is_not_enough_to_run_a_benchmark() -> None:
    """evals.read alone must not satisfy the run/evaluate routes' evals.manage gate."""
    app = _make_app()
    current_permissions: list[str] = [EVALS_READ]
    _override_user(app, current_permissions)

    with TestClient(app) as client:
        run_forbidden = client.post("/api/v1/benchmarks/demo/run", json={})
        assert run_forbidden.status_code == 403

        evaluate_forbidden = client.post(
            "/api/v1/benchmarks/simpleqa/evaluate",
            params={"question": "2+2?"},
        )
        assert evaluate_forbidden.status_code == 403


def test_run_routes_accept_evals_manage_permission() -> None:
    """evals.manage must pass the auth layer for both run and simpleqa-evaluate."""
    app = _make_app()
    current_permissions: list[str] = [EVALS_MANAGE]
    _override_user(app, current_permissions)

    with TestClient(app) as client:
        # A nonexistent benchmark name clears the auth layer and 404s on lookup,
        # proving the permission check itself is not what's blocking the request.
        run_resp = client.post("/api/v1/benchmarks/does-not-exist/run", json={})
        assert run_resp.status_code == 404, run_resp.text

        evaluate_resp = client.post(
            "/api/v1/benchmarks/simpleqa/evaluate",
            params={"question": "2+2?"},
        )
        assert evaluate_resp.status_code == 200, evaluate_resp.text
