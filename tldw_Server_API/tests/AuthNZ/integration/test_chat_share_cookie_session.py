"""Real cookie authentication at shared token guards and chat owner boundaries."""

import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi import Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import TokenScopeGuard
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat
from tldw_Server_API.app.core.AuthNZ import User_DB_Handling
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager, reset_api_key_manager
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
VALIDATE_PATH = "/api/v1/chat/dictionaries/validate"
VALID_DICTIONARY = {
    "data": {"name": "Cookie dictionary", "entries": [
        {"type": "literal", "pattern": "hello", "replacement": "hi"},
    ]},
    "schema_version": 1,
}


@pytest_asyncio.fixture
async def chat_cookie_client(single_user_cookie_client, monkeypatch, tmp_path):
    """Retain real AuthNZ, session, RBAC, CSRF and handlers; isolate content SQLite."""
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "chat-users"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "shutdown-jobs.db"))
    for router, prefix in [(chat.router, "/api/v1/chat"),
                           (chat.conversations_alias_router, "/api/v1/chats")]:
        expected = prefix + "/conversations/{conversation_id}/share-links"
        if not any(route.path == expected for route in iter_served_routes(client.app.routes)):
            client.app.include_router(router, prefix=prefix)
    test_flags = {name: os.environ.get(name) for name in ("TEST_MODE", "TLDW_TEST_MODE")}
    for name in test_flags:
        monkeypatch.setenv(name, "false")
    owner = str(get_settings().SINGLE_USER_FIXED_ID)
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id=owner)
    conversation = db.add_conversation({"title": "Cookie owner", "client_id": owner})
    db.upsert_conversation_settings(conversation, {"temperature": 0.4})
    overrides = client.app.dependency_overrides
    previous = overrides.get(get_chacha_db_for_user)
    overrides[get_chacha_db_for_user] = lambda: db
    try:
        yield client, api_key, db, conversation, owner
    finally:
        client.portal.call(reset_api_key_manager)
        if previous is None:
            overrides.pop(get_chacha_db_for_user, None)
        else:
            overrides[get_chacha_db_for_user] = previous
        db.close_all_connections()
        # The enclosing fixture started the minimal app in test mode. Restore its
        # startup flags before its matching shutdown sequence runs.
        for name, value in test_flags.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)


def _share_path(conversation, prefix="/api/v1/chat"):
    return f"{prefix}/conversations/{conversation}/share-links"


def _authenticate(client, api_key):
    assert _mint(client, api_key).status_code == 200  # nosec B101
    return {"X-CSRF-Token": client.cookies["csrf_token"]}


@pytest.mark.parametrize("prefix", ["/api/v1/chat", "/api/v1/chats"])
def test_cookie_owner_lists_share_links(chat_cookie_client, prefix):
    client, api_key, db, conversation, _owner = chat_cookie_client
    _authenticate(client, api_key)
    before = db.get_conversation_settings(conversation)

    response = client.get(_share_path(conversation, prefix), params={"scope_type": "global"})

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"conversation_id": conversation, "links": []}  # nosec B101
    assert db.get_conversation_settings(conversation) == before  # nosec B101


def test_cookie_owner_validates_dictionary(chat_cookie_client):
    client, api_key, _db, _conversation, _owner = chat_cookie_client
    headers = _authenticate(client, api_key)

    response = client.post(VALIDATE_PATH, json=VALID_DICTIONARY, headers=headers)

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json()["ok"] is True  # nosec B101


def test_cookie_dictionary_validation_retains_validator_errors(chat_cookie_client):
    client, api_key, _db, _conversation, _owner = chat_cookie_client
    headers = _authenticate(client, api_key)
    payload = {"data": {"entries": [{"type": "regex", "pattern": "(", "replacement": "x"}]}}

    response = client.post(VALIDATE_PATH, json=payload, headers=headers)

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json()["ok"] is False  # nosec B101
    assert "regex_invalid" in {issue["code"] for issue in response.json()["errors"]}  # nosec B101


def test_cookie_dictionary_validation_requires_csrf(chat_cookie_client):
    client, api_key, _db, _conversation, _owner = chat_cookie_client
    _authenticate(client, api_key)

    response = client.post(VALIDATE_PATH, json=VALID_DICTIONARY)

    assert response.status_code == 403, response.text  # nosec B101
    assert response.json() == {"detail": "CSRF token validation failed"}  # nosec B101 - Verify the actual CSRF refusal.


@pytest.mark.parametrize("session_state", ["absent", "invalid", "expired", "revoked", "wrong_type"])
def test_share_guard_rejects_unusable_cookie_sessions(chat_cookie_client, monkeypatch, session_state):
    client, api_key, _db, conversation, _owner = chat_cookie_client
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
        headers = _authenticate(client, api_key)
    if session_state == "revoked":
        cookie_name = get_settings().SINGLE_USER_SESSION_COOKIE_NAME
        cookie = client.cookies[cookie_name]
        assert client.delete("/api/v1/auth/single-user/session", headers=headers).status_code == 200  # nosec B101
        client.cookies.set(cookie_name, cookie, path="/api")
    elif session_state == "invalid":
        client.cookies.set(get_settings().SINGLE_USER_SESSION_COOKIE_NAME, "invalid-cookie", path="/api")

    response = client.get(_share_path(conversation))

    assert response.status_code == 401, response.text  # nosec B101


@pytest.mark.parametrize("headers", [
    {"X-API-KEY": "invalid-key"}, {"Authorization": "Bearer invalid-key"},
    {"X-API-KEY": ""}, {"Authorization": ""}, {"Authorization": "Basic invalid"},
])
def test_share_guard_never_falls_back_from_explicit_headers(chat_cookie_client, headers):
    client, api_key, _db, conversation, _owner = chat_cookie_client
    _authenticate(client, api_key)

    response = client.get(_share_path(conversation), headers=headers)

    assert response.status_code == 401, response.text  # nosec B101
    detail = "Could not validate credentials" if (
        headers.get("X-API-KEY") == "invalid-key" or headers.get("Authorization") == "Bearer invalid-key"
    ) else "Not authenticated (provide Bearer token or X-API-KEY)"
    assert response.json() == {"detail": detail}  # nosec B101 - Explicit credentials cannot fall back to cookies.


@pytest.mark.parametrize("header", ["X-API-KEY", "Authorization"])
def test_share_route_retains_explicit_primary_key_authentication(chat_cookie_client, header):
    client, api_key, _db, conversation, _owner = chat_cookie_client
    _authenticate(client, api_key)
    value = f"Bearer {api_key}" if header == "Authorization" else api_key

    response = client.get(_share_path(conversation), headers={header: value})

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"conversation_id": conversation, "links": []}  # nosec B101 - Return only the authenticated owner conversation.


@pytest.mark.parametrize("claim", ["admin-role", "wildcard-permission", "boolean-only"])
def test_cookie_guard_accepts_canonical_owner_independently_of_bypass_claims(chat_cookie_client, monkeypatch, claim):
    client, api_key, _db, conversation, owner = chat_cookie_client
    _authenticate(client, api_key)
    user = User_DB_Handling.User(
        id=owner, username="single_user", role="user", is_admin=True,
        roles=["admin"] if claim == "admin-role" else ["user"],
        permissions=["*"] if claim == "wildcard-permission" else [],
    )
    monkeypatch.setattr(User_DB_Handling, "get_single_user_instance", lambda: user)

    response = client.get(_share_path(conversation))

    assert response.status_code == 200, response.text  # nosec B101 - A canonical cookie is ordinary identity, not a bypass.
    assert response.json() == {"conversation_id": conversation, "links": []}  # nosec B101 - Canonical owner identity retains storage ownership.


def test_cookie_guard_accepts_canonical_session_with_admin_bypass_disabled(chat_cookie_client):
    client, api_key, _db, _conversation, _owner = chat_cookie_client

    @client.app.get("/api/test/no-bypass", dependencies=[
        Depends(TokenScopeGuard("any", require_if_present=True, allow_admin_bypass=False)),
    ])
    async def no_bypass():
        return {"ok": True}

    _authenticate(client, api_key)
    response = client.get("/api/test/no-bypass")

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"ok": True}  # nosec B101 - Ordinary session auth remains separate from admin bypass.


@pytest.mark.parametrize("boundary", ["foreign-owner", "global-as-workspace", "workspace-as-global",
                                     "wrong-workspace", "exact-workspace"])
def test_cookie_share_routes_retain_owner_and_scope_fences(chat_cookie_client, boundary):
    client, api_key, db, conversation, owner = chat_cookie_client
    _authenticate(client, api_key)
    params = {"scope_type": "global"}
    expected = 404
    if boundary == "foreign-owner":
        conversation = db.add_conversation({"title": "Foreign", "client_id": "different-owner"})
        expected = 403
    elif boundary == "global-as-workspace":
        params = {"scope_type": "workspace", "workspace_id": "space-one"}
    else:
        db.upsert_workspace("space-one", "Cookie workspace")
        conversation = db.add_conversation({"title": "Workspace", "client_id": owner,
                                            "scope_type": "workspace", "workspace_id": "space-one"})
        if boundary in {"wrong-workspace", "exact-workspace"}:
            params = {"scope_type": "workspace", "workspace_id":
                      "space-one" if boundary == "exact-workspace" else "space-two"}
        if boundary == "exact-workspace":
            expected = 200

    response = client.get(_share_path(conversation), params=params)

    assert response.status_code == expected, response.text  # nosec B101
    if expected == 200:
        assert response.json() == {"conversation_id": conversation, "links": []}  # nosec B101 - Exact workspace ownership.
    else:
        assert response.json() == {"detail": "Forbidden for this conversation" if expected == 403 else "Conversation not found"}  # nosec B101 - Owner and scope refusals stay distinct.


def test_cookie_share_revocation_requires_csrf_and_preserves_settings(chat_cookie_client):
    client, api_key, db, conversation, owner = chat_cookie_client
    now = datetime.now(timezone.utc)
    settings = {"temperature": 0.4, "knowledge_qa_share_links": [{
        "id": "seeded-private-test-link", "permission": "view", "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=1)).isoformat(), "created_by_user_id": owner,
    }]}
    db.upsert_conversation_settings(conversation, settings)
    headers = _authenticate(client, api_key)
    path = _share_path(conversation) + "/seeded-private-test-link"

    rejected = client.delete(path)
    assert rejected.status_code == 403, rejected.text  # nosec B101
    assert db.get_conversation_settings(conversation)["settings"] == settings  # nosec B101
    accepted = client.delete(path, headers=headers)
    assert accepted.status_code == 200, accepted.text  # nosec B101
    stored = db.get_conversation_settings(conversation)["settings"]
    assert stored["temperature"] == 0.4  # nosec B101
    assert stored["knowledge_qa_share_links"][0]["revoked_at"]  # nosec B101
    listed = client.get(_share_path(conversation))
    assert listed.status_code == 200, listed.text  # nosec B101
    assert listed.json()["links"][0]["token"] is None  # nosec B101


def test_share_guard_uses_configured_cookie_name(chat_cookie_client, monkeypatch):
    client, api_key, _db, conversation, _owner = chat_cookie_client
    monkeypatch.setenv("SINGLE_USER_SESSION_COOKIE_NAME", "test_configured_owner_session")
    reset_settings()
    _authenticate(client, api_key)

    response = client.get(_share_path(conversation))

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"conversation_id": conversation, "links": []}  # nosec B101 - Return only the authenticated owner conversation.


@pytest.mark.parametrize("constraint", ["allowed", "scope", "endpoint", "method", "path", "revoked"])
def test_cookie_does_not_bypass_explicit_virtual_key_constraints(chat_cookie_client, constraint):
    """Exercise real key constraints at the guard, before legacy user auth.

    Single-user legacy user auth deliberately accepts only the primary key;
    the guarded probe isolates persisted virtual-key enforcement itself.
    """
    client, api_key, _db, _conversation, owner = chat_cookie_client
    _authenticate(client, api_key)
    path = "/api/test/virtual-key-guard"

    @client.app.get(path, dependencies=[Depends(TokenScopeGuard(
        "any", require_if_present=True, endpoint_id="chat.conversations.share_links",
    ))])
    async def virtual_key_guard():
        return {"ok": True}

    async def create_key():
        manager = await get_api_key_manager()
        key = await manager.create_virtual_key(
            user_id=int(owner), name="Cookie guard constraints",
            scope="write" if constraint == "scope" else "read",
            allowed_endpoints=["unrelated.endpoint" if constraint == "endpoint"
                               else "chat.conversations.share_links"],
            allowed_methods=["POST" if constraint == "method" else "GET"],
            allowed_paths=["/unrelated" if constraint == "path" else path],
        )
        if constraint == "revoked":
            assert await manager.revoke_api_key(key["id"], int(owner))  # nosec B101
        return key["key"]

    key = client.portal.call(create_key)
    response = client.get(path, headers={"X-API-KEY": key})

    expected = 200 if constraint == "allowed" else 401 if constraint == "revoked" else 403
    assert response.status_code == expected, response.text  # nosec B101
