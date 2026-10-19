"""Prompt-library routes accept canonical cookie owners and retain their guards."""

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps as prompts_deps
from tldw_Server_API.app.api.v1.endpoints import prompts
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
PROMPTS_PATH = "/api/v1/prompts"
LIST_PARAMS = {"page": 1, "per_page": 100, "sort_by": "id", "sort_order": "asc"}


@pytest_asyncio.fixture
async def prompt_library_client(single_user_cookie_client, monkeypatch, tmp_path):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "prompt-library-users"))
    if not any(route.path == PROMPTS_PATH + "/search" for route in iter_served_routes(client.app.routes)):
        client.app.include_router(prompts.router, prefix=PROMPTS_PATH)
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    try:
        yield client, api_key
    finally:
        client.portal.call(prompts_deps.close_all_cached_prompts_db_instances)
        client.portal.call(prompts_deps.stop_prompts_pending_close_worker)


@pytest.mark.parametrize("suffix", ["", "/"])
def test_cookie_prompt_library_lists_exact_composer_query(prompt_library_client, suffix):
    client, api_key = prompt_library_client
    assert _mint(client, api_key).status_code == 200  # nosec B101

    response = client.get(PROMPTS_PATH + suffix, params=LIST_PARAMS)

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json()["items"] == []  # nosec B101


def test_cookie_prompt_library_persistence_search_and_detail_keep_owner_and_csrf(prompt_library_client):
    client, api_key = prompt_library_client
    assert _mint(client, api_key).status_code == 200  # nosec B101
    payload = {"name": "Cookie library pirate", "system_prompt": "Speak like a pirate."}
    rejected = client.post(PROMPTS_PATH, json=payload)
    assert rejected.status_code == 403, rejected.text  # nosec B101
    headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    created = client.post(PROMPTS_PATH, json=payload, headers=headers)
    assert created.status_code == 201, created.text  # nosec B101
    identifier = created.json()["uuid"]
    listed = client.get(PROMPTS_PATH, params=LIST_PARAMS)
    assert listed.status_code == 200, listed.text  # nosec B101
    assert [item["uuid"] for item in listed.json()["items"]] == [identifier]  # nosec B101
    searched = client.post(PROMPTS_PATH + "/search", params={"search_query": "pirate"}, headers=headers)
    assert searched.status_code == 200, searched.text  # nosec B101
    assert [item["uuid"] for item in searched.json()["items"]] == [identifier]  # nosec B101
    detail = client.get(f"{PROMPTS_PATH}/{identifier}")
    assert detail.status_code == 200, detail.text  # nosec B101
    assert detail.json()["system_prompt"] == payload["system_prompt"]  # nosec B101
    owner = client.get("/api/v1/auth/me").json()["id"]
    request = Request({"type": "http", "app": client.app, "headers": []})
    owned_db = client.portal.call(prompts_deps.get_prompts_db_for_user, request, User(id=owner, username="owner"))
    assert client.portal.call(owned_db.fetch_prompt_details, identifier)["name"] == payload["name"]  # nosec B101
    foreign_db = client.portal.call(prompts_deps.get_prompts_db_for_user, request, User(id=owner + 1, username="other-owner"))
    assert client.portal.call(foreign_db.fetch_prompt_details, identifier) is None  # nosec B101


@pytest.mark.parametrize("credentials", ["absent", "invalid-cookie", "invalid-api-key", "invalid-bearer", "invalid-legacy"])
def test_prompt_library_rejects_missing_invalid_and_explicit_bad_credentials(prompt_library_client, credentials):
    client, api_key = prompt_library_client
    headers = {}
    if credentials == "invalid-cookie":
        client.cookies.set("tldw_single_user_session", "invalid-cookie", path="/api")
    elif credentials.startswith("invalid-"):
        assert _mint(client, api_key).status_code == 200  # nosec B101
        header = {"invalid-api-key": "X-API-KEY", "invalid-bearer": "Authorization", "invalid-legacy": "Token"}[credentials]
        headers[header] = "Bearer invalid-key" if header == "Authorization" else "invalid-key"

    response = client.get(PROMPTS_PATH, params=LIST_PARAMS, headers=headers)

    assert response.status_code == 401, response.text  # nosec B101


def test_prompt_library_rejects_revoked_cookie(prompt_library_client):
    client, api_key = prompt_library_client
    assert _mint(client, api_key).status_code == 200  # nosec B101
    cookie = client.cookies["tldw_single_user_session"]
    headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    assert client.delete("/api/v1/auth/single-user/session", headers=headers).status_code == 200  # nosec B101
    client.cookies.set("tldw_single_user_session", cookie, path="/api")

    assert client.get(PROMPTS_PATH, params=LIST_PARAMS).status_code == 401  # nosec B101


@pytest.mark.asyncio
async def test_prompt_library_headerless_canonical_failure_stays_fail_closed(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503, detail="Canonical auth unavailable")

    monkeypatch.setattr(prompts, "get_request_user", unavailable)
    request = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as rejected:
        await prompts._resolve_prompts_auth_user(request, Token=None, x_api_key=None, Authorization=None)
    assert rejected.value.status_code == 503  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("require_admin", [False, True])
async def test_prompt_library_headerless_owner_retains_claim_admin_gate(monkeypatch, require_admin):
    user = User(id=2, username="regular-owner", roles=["user"], permissions=["prompts.read"], is_admin=True)

    async def regular_owner(*args, **kwargs):
        return user

    monkeypatch.setattr(prompts, "get_request_user", regular_owner)
    monkeypatch.setenv("PROMPTS_REQUIRE_ADMIN", str(require_admin).lower())
    request = Request({"type": "http", "headers": []})
    if require_admin:
        with pytest.raises(HTTPException) as rejected:
            await prompts.verify_prompts_user(request, Token=None, x_api_key=None, Authorization=None)
        assert rejected.value.status_code == 403  # nosec B101
    else:
        assert await prompts.verify_prompts_user(request, Token=None, x_api_key=None, Authorization=None)  # nosec B101
    with pytest.raises(HTTPException) as admin_rejected:
        await prompts.verify_prompts_auth(request, Token=None, x_api_key=None, Authorization=None)
    assert admin_rejected.value.status_code == 403  # nosec B101
