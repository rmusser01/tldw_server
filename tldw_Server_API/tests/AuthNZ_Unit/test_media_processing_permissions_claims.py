from __future__ import annotations

import inspect
from collections.abc import Iterable

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints.media import (
    process_audios,
    process_code,
    process_documents,
    process_ebooks,
    process_emails,
    process_mediawiki,
    process_pdfs,
    process_videos,
    process_web_scraping,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

PROCESSING_ROUTES = (
    (process_videos.router, "/process-videos"),
    (process_audios.router, "/process-audios"),
    (process_pdfs.router, "/process-pdfs"),
    (process_documents.router, "/process-documents"),
    (process_ebooks.router, "/process-ebooks"),
    (process_code.router, "/process-code"),
    (process_emails.router, "/process-emails"),
    (process_web_scraping.router, "/process-web-scraping"),
    (process_mediawiki.router, "/mediawiki/ingest-dump"),
    (process_mediawiki.router, "/mediawiki/process-dump"),
)


def _route_for_path(router, path: str) -> APIRoute:
    for route in router.routes:
        if isinstance(route, APIRoute) and route.path == path and "POST" in route.methods:
            return route
    raise AssertionError(f"POST route {path} not found")


def _dependency_calls(route: APIRoute) -> Iterable[object]:
    dependant = getattr(route, "dependant", None)
    for dependency in getattr(dependant, "dependencies", []) or []:
        call = getattr(dependency, "call", None)
        if call is not None:
            yield call


def _required_permissions(route: APIRoute) -> set[str]:
    permissions: set[str] = set()
    for call in _dependency_calls(route):
        if not callable(call):
            continue
        try:
            closure_vars = inspect.getclosurevars(call)
        except TypeError:
            continue
        for permission in closure_vars.nonlocals.get("perms", []) or []:
            permissions.add(str(permission))
    return permissions


def _rate_limit_resources(route: APIRoute) -> set[str]:
    return {
        resource
        for call in _dependency_calls(route)
        if (resource := getattr(call, "_tldw_rate_limit_resource", None))
    }


@pytest.mark.parametrize(("router", "path"), PROCESSING_ROUTES)
def test_media_processing_routes_require_media_create(router, path: str) -> None:
    route = _route_for_path(router, path)

    assert MEDIA_CREATE in _required_permissions(route)
    assert "media.create" in _rate_limit_resources(route)


def _principal_without_media_create() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["user"],
        permissions=[],
        is_admin=False,
    )


def test_processing_endpoint_forbidden_without_media_create_claim(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_QUOTA_ENFORCEMENT", "0")
    principal = _principal_without_media_create()
    app = FastAPI()
    app.include_router(process_emails.router, prefix="/api/v1/media")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(principal=principal)
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    with TestClient(app) as client:
        response = client.post("/api/v1/media/process-emails", data={})

    assert response.status_code == 403
    assert MEDIA_CREATE in response.json().get("detail", "")
