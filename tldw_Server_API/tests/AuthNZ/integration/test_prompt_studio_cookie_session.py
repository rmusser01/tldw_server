"""Prompt Studio uses the canonical authenticated cookie and CSRF contracts."""

import pytest
import pytest_asyncio
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps
from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_projects
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    _mint,
)
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
PROJECTS_PATH = "/api/v1/prompt-studio/projects"


@pytest_asyncio.fixture
async def prompt_cookie_client(single_user_cookie_client, monkeypatch, tmp_path):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "prompt-users"))
    if not any(route.path == PROJECTS_PATH for route in iter_served_routes(client.app.routes)):
        client.app.include_router(prompt_studio_projects.router)
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    try:
        yield client, api_key
    finally:
        prompt_studio_deps.shutdown_prompt_studio_deps()


@pytest.mark.parametrize("suffix", ["", "/"])
def test_cookie_session_lists_prompt_projects(prompt_cookie_client, suffix):
    client, api_key = prompt_cookie_client
    assert _mint(client, api_key).status_code == 200  # nosec B101

    response = client.get(PROJECTS_PATH + suffix)

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json()["data"] == []  # nosec B101


@pytest.mark.parametrize("suffix", ["", "/"])
@pytest.mark.parametrize("credentials", ["absent", "invalid-cookie", "invalid-header"])
def test_prompt_projects_reject_missing_or_invalid_credentials(prompt_cookie_client, suffix, credentials):
    client, api_key = prompt_cookie_client
    headers = {}
    if credentials == "invalid-cookie":
        client.cookies.set("tldw_single_user_session", "invalid-cookie", path="/api")
    elif credentials == "invalid-header":
        assert _mint(client, api_key).status_code == 200  # nosec B101
        headers["X-API-KEY"] = "invalid-header-key"

    response = client.get(PROJECTS_PATH + suffix, headers=headers)

    assert response.status_code == 401, response.text  # nosec B101
    detail = "Invalid or missing API Key" if credentials == "invalid-header" else "Not authenticated (provide Bearer token or X-API-KEY)"
    assert response.json() == {"detail": detail}  # nosec B101 - Preserve credential precedence and omit private projects.


def test_cookie_project_write_requires_csrf_and_retains_owner(prompt_cookie_client):
    client, api_key = prompt_cookie_client
    assert _mint(client, api_key).status_code == 200  # nosec B101
    payload = {"name": "Cookie project"}

    rejected = client.post(PROJECTS_PATH, json=payload)
    assert rejected.status_code == 403, rejected.text  # nosec B101
    headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    created = client.post(PROJECTS_PATH, json=payload, headers=headers)
    assert created.status_code == 201, created.text  # nosec B101
    owner = client.get("/api/v1/auth/me").json()["id"]
    assert created.json()["user_id"] == str(owner)  # nosec B101
    projects = client.get(PROJECTS_PATH).json()["data"]
    assert [project["id"] for project in projects] == [created.json()["id"]]  # nosec B101


@pytest.mark.asyncio
async def test_non_admin_project_guards_retain_owner_fence(monkeypatch, tmp_path, request):
    """Single-user cookies represent the admin; test regular-owner guards directly."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "prompt-users"))
    request.addfinalizer(prompt_studio_deps.shutdown_prompt_studio_deps)
    context = {"user_id": "regular-owner", "client_id": "web", "is_admin": False}
    with prompt_studio_deps.managed_prompt_studio_db(context) as db:
        owned = db.create_project(name="Owned", user_id=context["user_id"])
        foreign = db.create_project(name="Foreign", user_id="another-owner")
        response = await prompt_studio_projects.list_projects(
            page=1,
            per_page=20,
            status_filter=None,
            include_deleted=False,
            search=None,
            user_context=context,
            db=db,
        )
        assert [project.id for project in response["data"]] == [owned["id"]]  # nosec B101
        assert await prompt_studio_deps.require_project_access(owned["id"], context, db)  # nosec B101
        with pytest.raises(HTTPException) as read_denied:
            await prompt_studio_deps.require_project_access(foreign["id"], context, db)
        assert read_denied.value.status_code == 403  # nosec B101
        with pytest.raises(HTTPException) as write_denied:
            await prompt_studio_deps.require_project_write_access(foreign["id"], context, db)
        assert write_denied.value.status_code == 403  # nosec B101
