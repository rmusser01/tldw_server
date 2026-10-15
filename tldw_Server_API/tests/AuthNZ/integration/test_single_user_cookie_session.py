"""HTTP coverage for persistent single-user cookie sessions."""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_auth_rate_limit
from tldw_Server_API.app.api.v1.endpoints import auth as auth_endpoints
from tldw_Server_API.app.api.v1.schemas.auth_schemas import (
    SingleUserSessionLogoutResponse,
    SingleUserSessionResponse,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import SessionError, SessionRevokedException
from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager, reset_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.services.registration_service import reset_registration_service
from tldw_Server_API.tests.helpers.app_main_state import reload_app_main

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def single_user_cookie_client(tmp_path, monkeypatch):
    db_path = tmp_path / "authnz_single_user_cookie.db"
    api_key = "test_single_user_cookie_api_key_123"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", api_key)
    monkeypatch.setenv("SESSION_ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("CSRF_ENABLED", "1")
    monkeypatch.setenv("DEFER_HEAVY_STARTUP", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", "1")

    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()

    await bootstrap_single_user_profile()
    app = reload_app_main().app
    # The minimal test app omits this optional router; retain its real dependencies.
    from tldw_Server_API.app.api.v1.endpoints import ingestion_sources

    if not any(route.path == "/api/v1/ingestion-sources/capabilities" for route in iter_served_routes(app.routes)):
        app.include_router(ingestion_sources.router, prefix="/api/v1")

    with TestClient(app) as client:
        yield client, get_settings().SINGLE_USER_API_KEY

    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()


def _mint(client: TestClient, api_key: str):
    return client.post(
        "/api/v1/auth/single-user/session",
        headers={"X-API-KEY": api_key},
    )


@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_single_user_session_routes_use_auth_rate_limit(method):
    route = next(
        route
        for route in auth_endpoints.router.routes
        if route.path.endswith("/single-user/session") and method in route.methods
    )

    assert any(dependency.dependency is check_auth_rate_limit for dependency in route.dependencies)


@pytest.mark.parametrize("method", ["POST", "DELETE"])
def test_single_user_session_routes_declare_response_models(method):
    route = next(
        route
        for route in auth_endpoints.router.routes
        if route.path.endswith("/single-user/session") and method in route.methods
    )

    expected = SingleUserSessionResponse if method == "POST" else SingleUserSessionLogoutResponse
    assert route.response_model is expected


def test_mint_returns_no_token_and_sets_exact_cookie(single_user_cookie_client):
    client, api_key = single_user_cookie_client

    response = _mint(client, api_key)

    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    assert set(response.json()) == {"authenticated", "expires_at"}
    cookie = response.headers["set-cookie"]
    assert "tldw_single_user_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/api" in cookie
    assert api_key not in response.text + cookie


def test_cookie_principal_authenticates_existing_http_dependencies(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200

    response = client.get("/api/v1/auth/sessions")

    assert response.status_code == 200


def test_cookie_mutation_requires_csrf_but_api_key_does_not(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    mint = _mint(client, api_key)
    assert mint.status_code == 200

    assert client.delete("/api/v1/auth/single-user/session").status_code == 403
    assert _mint(client, api_key).status_code == 200


def test_cookie_only_mint_is_rejected_after_csrf_validation(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200
    csrf_token = client.cookies["csrf_token"]

    response = client.post(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 401


def test_valid_cookie_session_is_reused_on_mint(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    first = _mint(client, api_key)
    assert first.status_code == 200
    cookie_before = client.cookies["tldw_single_user_session"]

    second = _mint(client, api_key)

    assert second.status_code == 200
    assert "no-store" in second.headers["cache-control"]
    assert second.json()["expires_at"] == first.json()["expires_at"]
    assert client.cookies["tldw_single_user_session"] == cookie_before


def test_revoked_cookie_is_replaced_on_mint(
    single_user_cookie_client,
    monkeypatch,
):
    client, api_key = single_user_cookie_client
    first = _mint(client, api_key)
    assert first.status_code == 200
    revoked_cookie = client.cookies["tldw_single_user_session"]

    async def revoked_session(request, manager, *, strict=False):
        raise SessionRevokedException()

    monkeypatch.setattr(
        auth_endpoints,
        "validate_single_user_session",
        revoked_session,
    )
    replacement = _mint(client, api_key)

    assert replacement.status_code == 200
    assert revoked_cookie not in replacement.headers["set-cookie"]


def test_mint_returns_no_store_503_when_strict_validation_fails(
    single_user_cookie_client,
    monkeypatch,
):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200
    cookie_before = client.cookies["tldw_single_user_session"]

    async def fail_only_when_strict(request, manager, *, strict=False):
        if strict:
            raise SessionError("session store unavailable")
        return None

    monkeypatch.setattr(
        auth_endpoints,
        "validate_single_user_session",
        fail_only_when_strict,
    )

    response = _mint(client, api_key)

    assert response.status_code == 503
    assert "no-store" in response.headers["cache-control"]
    assert "set-cookie" not in response.headers
    assert client.cookies["tldw_single_user_session"] == cookie_before


def test_csrf_disabled_refuses_cookie_mint(single_user_cookie_client, monkeypatch):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("CSRF_ENABLED", "0")

    response = _mint(client, api_key)

    assert response.status_code == 503


def test_logout_revokes_only_current_cookie_session(
    single_user_cookie_client,
    monkeypatch,
):
    client, api_key = single_user_cookie_client
    revoke_calls: list[int] = []
    original_revoke = SessionManager.revoke_session

    async def tracked_revoke(self, session_id, *args, **kwargs):
        revoke_calls.append(session_id)
        return await original_revoke(self, session_id, *args, **kwargs)

    monkeypatch.setattr(SessionManager, "revoke_session", tracked_revoke)
    first = _mint(client, api_key)
    assert first.status_code == 200
    first_cookie = client.cookies["tldw_single_user_session"]
    first_csrf = client.cookies["csrf_token"]

    client.cookies.clear()
    second = _mint(client, api_key)
    assert second.status_code == 200
    second_cookie = client.cookies["tldw_single_user_session"]
    assert second_cookie != first_cookie

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", first_cookie, path="/api")
    client.cookies.set("csrf_token", first_csrf, path="/")
    logout = client.delete(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": first_csrf},
    )
    assert logout.status_code == 200
    assert logout.json() == {"authenticated": False}
    assert "no-store" in logout.headers["cache-control"]
    assert "tldw_single_user_session=" in logout.headers["set-cookie"]
    assert "Max-Age=0" in logout.headers["set-cookie"]
    assert len(revoke_calls) == 1

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", first_cookie, path="/api")
    assert client.get("/api/v1/auth/sessions").status_code == 401

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", second_cookie, path="/api")
    assert client.get("/api/v1/auth/sessions").status_code == 200

    client.cookies.clear()
    client.get("/api/v1/health")
    unbound_csrf = client.cookies["csrf_token"]
    client.cookies.set("tldw_single_user_session", first_cookie, path="/api")
    stale_logout = client.delete(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": unbound_csrf},
    )

    assert stale_logout.status_code == 200
    assert stale_logout.json() == {"authenticated": False}
    assert "no-store" in stale_logout.headers["cache-control"]
    assert "Max-Age=0" in stale_logout.headers["set-cookie"]
    assert len(revoke_calls) == 1

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", second_cookie, path="/api")
    assert client.get("/api/v1/auth/sessions").status_code == 200


def test_logout_without_cookie_is_idempotent(single_user_cookie_client):
    client, _ = single_user_cookie_client
    client.cookies.clear()

    response = client.delete("/api/v1/auth/single-user/session")

    assert response.status_code == 200
    assert response.json() == {"authenticated": False}
    assert "no-store" in response.headers["cache-control"]
    assert "tldw_single_user_session=" in response.headers["set-cookie"]
    assert "Max-Age=0" in response.headers["set-cookie"]


def test_logout_with_invalid_cookie_is_idempotent(
    single_user_cookie_client,
    monkeypatch,
):
    client, _ = single_user_cookie_client
    revoke_calls: list[int] = []
    original_revoke = SessionManager.revoke_session

    async def tracked_revoke(self, session_id, *args, **kwargs):
        revoke_calls.append(session_id)
        return await original_revoke(self, session_id, *args, **kwargs)

    monkeypatch.setattr(SessionManager, "revoke_session", tracked_revoke)
    client.cookies.clear()
    client.get("/api/v1/health")
    csrf_token = client.cookies["csrf_token"]
    client.cookies.set("tldw_single_user_session", "invalid-cookie", path="/api")

    response = client.delete(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.json() == {"authenticated": False}
    assert "no-store" in response.headers["cache-control"]
    assert "Max-Age=0" in response.headers["set-cookie"]
    assert revoke_calls == []


def test_logout_preserves_cookie_when_strict_validation_fails(
    single_user_cookie_client,
    monkeypatch,
):
    client, api_key = single_user_cookie_client
    mint = _mint(client, api_key)
    assert mint.status_code == 200
    cookie_before = client.cookies["tldw_single_user_session"]
    csrf_token = client.cookies["csrf_token"]

    async def fail_only_when_strict(request, manager, *, strict=False):
        if strict:
            raise SessionError("session store unavailable")
        return None

    monkeypatch.setattr(
        auth_endpoints,
        "validate_single_user_session",
        fail_only_when_strict,
    )

    response = client.delete(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 503
    assert "no-store" in response.headers["cache-control"]
    assert "set-cookie" not in response.headers
    assert client.cookies["tldw_single_user_session"] == cookie_before


def test_single_user_session_logout_unavailable_in_multi_user_mode(
    single_user_cookie_client,
    monkeypatch,
):
    client, _ = single_user_cookie_client
    client.cookies.clear()
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()

    response = client.delete("/api/v1/auth/single-user/session")

    # CSRF middleware rejects the state-changing request before the
    # single-user-only route can conceal itself with a 404.
    assert response.status_code == 403


LEGACY_USER_PATHS = (
    "/api/v1/persona/profiles",
    "/api/v1/notifications",
    "/api/v1/ingestion-sources/capabilities",
)


@pytest.mark.parametrize("path", LEGACY_USER_PATHS)
def test_cookie_authenticates_legacy_user_routes(
    single_user_cookie_client,
    monkeypatch,
    tmp_path,
    path,
):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("PERSONA_ENABLED", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    assert _mint(client, api_key).status_code == 200

    response = client.get(path)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("session_state", ["absent", "invalid", "expired", "revoked", "wrong_type"])
def test_legacy_user_routes_reject_unusable_cookie_sessions(
    single_user_cookie_client,
    monkeypatch,
    session_state,
):
    client, api_key = single_user_cookie_client
    if session_state in {"expired", "wrong_type"}:
        original_create = SessionManager.create_session

        async def create_unusable_session(self, *args, **kwargs):
            if session_state == "expired":
                kwargs["expires_at_override"] = datetime.now(timezone.utc) - timedelta(days=1)
            else:
                kwargs["device_id"] = "ordinary-device"
            return await original_create(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, "create_session", create_unusable_session)

    if session_state in {"expired", "revoked", "wrong_type"}:
        assert _mint(client, api_key).status_code == 200
    if session_state == "revoked":
        cookie = client.cookies["tldw_single_user_session"]
        csrf = client.cookies["csrf_token"]
        assert (
            client.delete(
                "/api/v1/auth/single-user/session",
                headers={"X-CSRF-Token": csrf},
            ).status_code
            == 200
        )
        client.cookies.set("tldw_single_user_session", cookie, path="/api")
    elif session_state == "invalid":
        client.cookies.set("tldw_single_user_session", "invalid-cookie", path="/api")

    for path in LEGACY_USER_PATHS:
        response = client.get(path)
        assert response.status_code == 401, (path, response.text)


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer invalid-key"},
        {"Authorization": "Basic invalid"},
        {"Authorization": ""},
        {"X-API-KEY": "invalid-key"},
        {"X-API-KEY": ""},
    ],
)
def test_legacy_user_routes_do_not_fall_back_from_explicit_headers_to_cookie(
    single_user_cookie_client,
    headers,
):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200

    for path in LEGACY_USER_PATHS:
        response = client.get(path, headers=headers)
        assert response.status_code == 401, (path, response.text)


def test_legacy_persona_cookie_mutation_keeps_csrf_and_owner_scope(
    single_user_cookie_client,
    monkeypatch,
    tmp_path,
):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("PERSONA_ENABLED", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    assert _mint(client, api_key).status_code == 200
    payload = {"name": "Cookie-owned persona"}
    assert client.post("/api/v1/persona/profiles", json=payload).status_code == 403
    response = client.post(
        "/api/v1/persona/profiles",
        json=payload,
        headers={"X-CSRF-Token": client.cookies["csrf_token"]},
    )
    assert response.status_code == 201, response.text
    profile = response.json()
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id

    user_id = get_settings().SINGLE_USER_FIXED_ID
    db = client.portal.call(get_chacha_db_for_user_id, user_id)
    stored = db.get_persona_profile(profile["id"], user_id=str(user_id))
    assert stored["user_id"] == str(user_id)
    other_profile_id = db.create_persona_profile({"name": "Other owner", "user_id": "other-user"})
    assert client.get(f"/api/v1/persona/profiles/{other_profile_id}").status_code == 404
    listing = client.get("/api/v1/persona/profiles")
    assert listing.status_code == 200
    assert profile["id"] in {row["id"] for row in listing.json()}


def test_legacy_user_cookie_is_not_accepted_in_multi_user_mode(
    single_user_cookie_client,
    monkeypatch,
):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()

    for path in LEGACY_USER_PATHS:
        assert client.get(path).status_code == 401


@pytest.mark.parametrize("principal_first", [False, True])
def test_real_cookie_session_activates_content_scope(single_user_cookie_client, principal_first, monkeypatch):
    """A minted cookie establishes content authority in either dependency order."""
    from dataclasses import asdict

    from fastapi import Depends

    from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope

    client, api_key = single_user_cookie_client
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    dependencies = [get_auth_principal, get_request_user] if principal_first else [get_request_user]

    @client.app.get("/api/v1/authnz/cookie-content-scope", dependencies=[Depends(dep) for dep in dependencies])
    async def content_scope():
        return asdict(get_scope())

    assert get_scope() is None
    assert _mint(client, api_key).status_code == 200
    response = client.get("/api/v1/authnz/cookie-content-scope")
    assert response.status_code == 200
    assert response.json() == {
        "user_id": get_settings().SINGLE_USER_FIXED_ID,
        "org_ids": [], "team_ids": [], "active_org_id": None,
        "active_team_id": None, "is_admin": True, "session_role": None,
    }
    # A cookie cannot rescue an explicitly invalid header on this path.
    assert client.get("/api/v1/authnz/cookie-content-scope", headers={"X-API-KEY": ""}).status_code == 401
    assert get_scope() is None


@pytest.fixture
def cookie_scope_chat(single_user_cookie_client, tmp_path):
    """Serve real Chat guards and data with the fixture's real cookie authority."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        get_chacha_db_for_user,
    )
    from tldw_Server_API.app.api.v1.endpoints import chat
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
        create_character_conversation,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    client, api_key = single_user_cookie_client
    for router, prefix in (
        (chat.router, "/api/v1/chat"),
        (chat.conversations_alias_router, "/api/v1/chats"),
    ):
        path = prefix + "/conversations"
        if not any(route.path == path for route in iter_served_routes(client.app.routes)):
            client.app.include_router(router, prefix=prefix)
    db = CharactersRAGDB(str(tmp_path / "cookie-scope-chat.db"), client_id="1")
    character_id = db.add_character_card({"name": "Cookie session guide", "client_id": "1"})
    conversation_id = create_character_conversation(
        db, conversation_data={"character_id": character_id, "client_id": "1", "title": "Native cookie search"}
    )
    db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "A memorable zebrafinch",
            "client_id": "1",
        }
    )
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        yield client, api_key, conversation_id
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        db.close_all_connections()


@pytest.mark.parametrize("prefix", ["/api/v1/chat", "/api/v1/chats"])
@pytest.mark.parametrize("action", ["search", "share_links"])
def test_required_chat_scope_guard_accepts_real_cookie_owner(cookie_scope_chat, prefix, action):
    """A minted cookie reaches actual search/share guards without an API-key header."""
    client, api_key, conversation_id = cookie_scope_chat
    assert _mint(client, api_key).status_code == 200
    path = prefix + (
        "/conversations?query=zebrafinch&search_in=title,content"
        if action == "search"
        else f"/conversations/{conversation_id}/share-links"
    )
    response = client.get(path)
    assert response.status_code == 200, response.text
    if action == "search":
        assert [item["id"] for item in response.json()["items"]] == [conversation_id]
        assert response.json()["items"][0]["matched_in"] == ["content"]
    else:
        assert response.json() == {"conversation_id": conversation_id, "links": []}


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer invalid-key"},
        {"Authorization": "Basic invalid"},
        {"Authorization": ""},
        {"X-API-KEY": "invalid-key"},
        {"X-API-KEY": ""},
    ],
)
def test_required_chat_scope_guard_keeps_explicit_header_precedence(cookie_scope_chat, headers):
    """A valid cookie cannot rescue invalid or empty explicit credentials."""
    client, api_key, conversation_id = cookie_scope_chat
    assert _mint(client, api_key).status_code == 200
    for path in (
        "/api/v1/chats/conversations?query=zebrafinch&search_in=content",
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
    ):
        response = client.get(path, headers=headers)
        assert response.status_code == 401, (path, response.text)


@pytest.mark.parametrize("session_state", ["absent", "invalid", "expired", "revoked", "wrong_type"])
def test_required_chat_scope_guard_rejects_unusable_sessions(cookie_scope_chat, monkeypatch, session_state):
    """Only a live single-user session may satisfy a required scope guard."""
    client, api_key, conversation_id = cookie_scope_chat
    if session_state in {"expired", "wrong_type"}:
        original_create = SessionManager.create_session

        async def create_unusable_session(self, *args, **kwargs):
            if session_state == "expired":
                kwargs["expires_at_override"] = datetime.now(timezone.utc) - timedelta(days=1)
            else:
                kwargs["device_id"] = "ordinary-device"
            return await original_create(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, "create_session", create_unusable_session)
    if session_state in {"expired", "revoked", "wrong_type"}:
        assert _mint(client, api_key).status_code == 200
    if session_state == "revoked":
        cookie = client.cookies["tldw_single_user_session"]
        assert (
            client.delete(
                "/api/v1/auth/single-user/session",
                headers={
                    "X-CSRF-Token": client.cookies["csrf_token"],
                },
            ).status_code
            == 200
        )
        client.cookies.set("tldw_single_user_session", cookie, path="/api")
    elif session_state == "invalid":
        client.cookies.set("tldw_single_user_session", "invalid-cookie", path="/api")
    for path in (
        "/api/v1/chats/conversations?query=zebrafinch&search_in=content",
        f"/api/v1/chat/conversations/{conversation_id}/share-links",
    ):
        response = client.get(path)
        assert response.status_code == 401, (path, response.text)


def test_required_chat_scope_guard_rejects_cookie_in_multi_user_mode(cookie_scope_chat, monkeypatch):
    """Single-user sessions never become multi-user credentials."""
    client, api_key, _conversation_id = cookie_scope_chat
    assert _mint(client, api_key).status_code == 200
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "cookie-scope-wrong-mode-disposable-secret-13534")
    reset_settings()
    response = client.get("/api/v1/chats/conversations")
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("action", ["search", "share_links"])
def test_required_chat_scope_guard_fails_closed_when_session_store_fails(
    cookie_scope_chat,
    monkeypatch,
    action,
):
    """A cookie cannot authorize a Chat data read when its store is unavailable."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    client, api_key, conversation_id = cookie_scope_chat
    assert _mint(client, api_key).status_code == 200
    validation_attempts = []

    async def unavailable_session_store(self, *args, **kwargs):
        validation_attempts.append(True)
        raise SessionError("disposable session store unavailable")

    def forbidden_chat_read():
        pytest.fail("Chat data dependency ran before cookie authority was validated")

    monkeypatch.setattr(SessionManager, "validate_session", unavailable_session_store)
    client.app.dependency_overrides[get_chacha_db_for_user] = forbidden_chat_read
    path = (
        "/api/v1/chats/conversations?query=zebrafinch&search_in=content"
        if action == "search"
        else f"/api/v1/chat/conversations/{conversation_id}/share-links"
    )
    response = client.get(path)
    assert response.status_code == 401, response.text
    assert validation_attempts


def _attach_resume_docs_handlers(app):
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions, character_messages

    served = {route.path for route in iter_served_routes(app.routes)}
    for path, router, prefix in [
        ("/api/v1/chats/{chat_id}/roleplay/messages", character_messages.router, "/api/v1"),
        ("/api/v1/chats/{chat_id}/complete-v2", character_chat_sessions.router, "/api/v1/chats"),
    ]:
        if path not in served:
            app.include_router(router, prefix=prefix)


def test_docs_info_resume_authenticates_real_headerless_cookie_and_revocation(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    _attach_resume_docs_handlers(client.app)
    assert _mint(client, api_key).status_code == 200
    original_cookie = client.cookies["tldw_single_user_session"]
    csrf = client.cookies["csrf_token"]
    authenticated = client.get("/api/v1/config/docs-info")
    assert authenticated.status_code == 200
    assert authenticated.json()["capabilities"]["roleplay_resume_contract_version"] == 1
    assert authenticated.headers["cache-control"] == "no-store"
    logout = client.delete("/api/v1/auth/single-user/session", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 200
    client.cookies.set("tldw_single_user_session", original_cookie, path="/api")
    revoked = client.get("/api/v1/config/docs-info")
    assert revoked.status_code == 200
    assert "roleplay_resume_contract_version" not in revoked.json()["capabilities"]


@pytest.mark.parametrize("headers", [{"X-API-KEY": "invalid-key"}, {"Authorization": ""}])
def test_docs_info_resume_does_not_replace_invalid_headers_with_valid_cookie(single_user_cookie_client, headers):
    client, api_key = single_user_cookie_client
    _attach_resume_docs_handlers(client.app)
    assert _mint(client, api_key).status_code == 200
    assert client.get("/api/v1/config/docs-info").json()["capabilities"]["roleplay_resume_contract_version"] == 1
    assert client.get("/api/v1/config/docs-info", headers=headers).status_code == 401
