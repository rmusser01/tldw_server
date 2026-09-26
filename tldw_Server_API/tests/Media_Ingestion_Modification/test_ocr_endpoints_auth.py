"""The OCR router must not answer anonymously.

Both of its routes were public: `/ocr/backends` reports backend health, and
`/ocr/points/preload` loads a transformers model into the server's memory.
Unlike the llamacpp and flashcards routers, neither route here authenticated,
so there was no in-file precedent to read the omission against -- it was
confirmed as an oversight rather than a deliberate diagnostics surface.

The guard is applied at the router, so a route added later inherits it instead
of being public by omission. These tests assert that, not just today's two
routes.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps


def _app_without_a_caller() -> FastAPI:
    """Serve the OCR router with the rate limiter stubbed but no authenticated user."""
    from tldw_Server_API.app.api.v1.endpoints.ocr import router as ocr_router

    async def _fake_check_rate_limit() -> None:
        return

    app = FastAPI()
    app.include_router(ocr_router, prefix="/api/v1")
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    return app


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/api/v1/ocr/backends"), ("post", "/api/v1/ocr/points/preload")],
)
def test_ocr_routes_require_authentication(method: str, path: str) -> None:
    """Neither backend discovery nor a model preload should be free."""
    with TestClient(_app_without_a_caller()) as client:
        response = getattr(client, method)(path)

    assert response.status_code in (401, 403), response.text


@pytest.mark.unit
def test_every_ocr_route_carries_the_router_guard() -> None:
    """A route added to this router later must inherit the guard, not opt in.

    Asserted over the router's own routes so it keeps holding as routes are
    added, rather than pinning the two that exist today.
    """
    from tldw_Server_API.app.api.v1.endpoints.ocr import router as ocr_router

    assert ocr_router.routes, "the OCR router should expose routes"
    guards = {
        getattr(dependency.dependency, "__name__", "")
        for dependency in (ocr_router.dependencies or [])
    }
    assert "get_request_user" in guards, f"router guards were {guards}"
