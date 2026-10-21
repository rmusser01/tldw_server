"""Prompt ownership assertions bind routes to the loaded account scope."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps, prompt_studio_deps
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.endpoints import prompts
from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_projects,
    prompt_studio_prompts,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase

pytestmark = pytest.mark.integration


@pytest.fixture
def scoped_routes(isolated_db, tmp_path, monkeypatch):
    """Keep real route guards and stores, control only credential resolution."""
    library = PromptsDatabase(tmp_path / "bob-prompts.sqlite", client_id="scope-test")
    library.add_prompt(name="Bob private", author=None, details=None, user_prompt="BOB CONTENT")
    project = isolated_db.create_project(name="Bob project", user_id="3")
    isolated_db.create_prompt(
        project_id=project["id"], name="Bob private", user_prompt="BOB CONTENT", client_id="scope-test"
    )

    async def principal(_request):
        return AuthPrincipal(kind="user", user_id=3, roles=["user"])

    monkeypatch.setattr(auth_deps, "get_auth_principal", principal)
    app = FastAPI()
    app.include_router(prompt_studio_projects.router)
    app.include_router(prompt_studio_prompts.router)
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    app.dependency_overrides[prompt_studio_deps.get_prompt_studio_user] = lambda: {
        "user_id": "3",
        "client_id": "scope-test",
        "is_admin": False,
    }
    app.dependency_overrides[prompt_studio_deps.get_prompt_studio_db] = lambda: isolated_db
    app.dependency_overrides[prompt_studio_deps.get_security_config] = lambda: SecurityConfig(
        enable_rate_limiting=False
    )
    app.dependency_overrides[get_prompts_db_for_user] = lambda: library
    app.dependency_overrides[prompts.verify_prompts_user] = lambda: True
    with TestClient(app) as client:
        yield client, isolated_db, library, project["id"]
    library.close_connection()


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "/api/v1/prompt-studio/projects", None),
        ("GET", "/api/v1/prompt-studio/prompts/get/1", None),
        ("GET", "/api/v1/prompt-studio/prompts?project_id=1", None),
        ("POST", "/api/v1/prompt-studio/projects", {"name": "ALICE NEW"}),
        (
            "POST",
            "/api/v1/prompt-studio/prompts/create",
            {"project_id": 1, "name": "ALICE NEW", "user_prompt": "ALICE CONTENT"},
        ),
        ("PUT", "/api/v1/prompt-studio/prompts/update/1", {"name": "ALICE NEW", "change_description": "change"}),
        ("DELETE", "/api/v1/prompt-studio/projects/delete/1", None),
        ("GET", "/api/v1/prompts", None),
        ("GET", "/api/v1/prompts/1", None),
        ("POST", "/api/v1/prompts/search?search_query=Bob", None),
        ("GET", "/api/v1/prompts/export?export_format=csv", None),
        ("POST", "/api/v1/prompts", {"name": "ALICE NEW", "user_prompt": "ALICE CONTENT"}),
        ("PUT", "/api/v1/prompts/1", {"name": "ALICE NEW", "user_prompt": "ALICE CONTENT"}),
    ],
)
def test_stale_expected_account_rejects_reads_and_mutations(scoped_routes, method, path, payload):
    client, studio, library, project_id = scoped_routes
    before = studio.get_prompt(1)
    response = client.request(method, path, json=payload, headers={"X-TLDW-Expected-User-ID": "2"})
    assert response.status_code == 412, response.text  # nosec B101 # pytest verifies account scope admission and preservation.
    assert response.headers.get("Cache-Control") == "no-store"  # nosec B101 # pytest verifies account scope admission and preservation.
    assert studio.get_prompt(1) == before  # nosec B101 # pytest verifies account scope admission and preservation.
    assert studio.get_project(project_id)["name"] == "Bob project"  # nosec B101 # pytest verifies account scope admission and preservation.
    assert library.fetch_prompt_details(1)["user_prompt"] == "BOB CONTENT"  # nosec B101 # pytest verifies account scope admission and preservation.


@pytest.mark.parametrize("expected", [None, "3"])
@pytest.mark.parametrize(
    "path",
    ["/api/v1/prompt-studio/projects", "/api/v1/prompt-studio/prompts/get/1", "/api/v1/prompts", "/api/v1/prompts/1"],
)
def test_matching_and_absent_scope_headers_keep_existing_reads(scoped_routes, expected, path):
    headers = {} if expected is None else {"X-TLDW-Expected-User-ID": expected}
    response = scoped_routes[0].get(path, headers=headers)
    assert response.status_code == 200, response.text  # nosec B101 # pytest verifies account scope admission and preservation.
    payload = response.json()
    if path == "/api/v1/prompt-studio/projects":
        assert [row["name"] for row in payload["projects"]] == ["Bob project"]  # nosec B101 - Real account-owned projects.
    elif path == "/api/v1/prompt-studio/prompts/get/1":
        assert payload["data"]["user_prompt"] == "BOB CONTENT"  # nosec B101 - Real account-owned studio prompt.
    elif path == "/api/v1/prompts":
        assert [row["name"] for row in payload["items"]] == ["Bob private"]  # nosec B101 - Real account-owned library list.
    else:
        assert payload["user_prompt"] == "BOB CONTENT"  # nosec B101 - Real account-owned library prompt.
