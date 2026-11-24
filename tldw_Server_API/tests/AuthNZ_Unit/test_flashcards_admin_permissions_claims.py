from types import SimpleNamespace
from typing import Optional

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.permissions import FLASHCARDS_ADMIN


def _make_principal(
    *,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
    is_admin: bool = False,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=list(roles or []),
        permissions=list(permissions or []),
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def _build_app_with_flashcards(principal: AuthPrincipal) -> FastAPI:
    from tldw_Server_API.app.api.v1.endpoints import flashcards as flashcards_mod
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh

    app = FastAPI()
    app.include_router(flashcards_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request):
        ctx = AuthContext(
            principal=principal,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent") if request.headers else None,
            request_id=request.headers.get("X-Request-ID") if request.headers else None,
        )
        request.state.auth = ctx
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_get_request_user():
        return SimpleNamespace(
            id=principal.user_id,
            username="flash-user",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
        )

    app.dependency_overrides[udh.get_request_user] = _fake_get_request_user

    class _FakeDB(CharactersRAGDB):  # type: ignore[misc]
        flashcard_store = property(lambda self: self)  # the fake defines the flashcard methods itself
        def __init__(self):
            pass

        def list_decks(
            self,
            limit: int,
            offset: int,
            include_deleted: bool = False,
            workspace_id: Optional[str] = None,
            include_workspace_items: bool = False,
        ):
            return []

        def add_deck(self, name: str, description: Optional[str]):
            return 1

        def add_flashcard(self, data):
            return "uuid"

        def set_flashcard_tags(self, _uuid, _tags):
            return None

        def get_flashcard(self, _uuid):
            return {"id": "uuid", "deck_name": "D", "front": "F", "back": "B"}

    async def _fake_get_chacha_db_for_user():
        return _FakeDB()

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    app.dependency_overrides[get_chacha_db_for_user] = _fake_get_chacha_db_for_user

    return app


@pytest.mark.unit
def test_flashcards_import_overrides_forbidden_without_permission():
    principal = _make_principal(roles=["user"], permissions=[], is_admin=False)
    app = _build_app_with_flashcards(principal)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/flashcards/import",
            params={"max_lines": 10},
            json={"content": "Deck\tFront\tBack\tTags\tNotes\nD\tF\tB\tT\tN\n", "has_header": True},
        )
        assert resp.status_code == 403


@pytest.mark.unit
def test_flashcards_import_overrides_rejects_boolean_admin_without_claims():
    principal = _make_principal(roles=["user"], permissions=[], is_admin=True)
    app = _build_app_with_flashcards(principal)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/flashcards/import",
            params={"max_lines": 10},
            json={"content": "Deck\tFront\tBack\tTags\tNotes\nD\tF\tB\tT\tN\n", "has_header": True},
        )
        assert resp.status_code == 403


@pytest.mark.unit
def test_flashcards_import_overrides_allowed_with_flashcards_admin_permission():
    principal = _make_principal(roles=["user"], permissions=[FLASHCARDS_ADMIN], is_admin=False)
    app = _build_app_with_flashcards(principal)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/flashcards/import",
            params={"max_lines": 10},
            json={"content": "Deck\tFront\tBack\tTags\tNotes\nD\tF\tB\tT\tN\n", "has_header": True},
        )
        assert resp.status_code == 200


@pytest.mark.unit
def test_flashcards_import_no_overrides_does_not_require_special_permission():
    principal = _make_principal(roles=["user"], permissions=[], is_admin=False)
    app = _build_app_with_flashcards(principal)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/flashcards/import",
            json={"content": "Deck\tFront\tBack\tTags\tNotes\nD\tF\tB\tT\tN\n", "has_header": True},
        )
        # Base import remains allowed for regular users
        assert resp.status_code == 200


@pytest.mark.unit
def test_flashcards_import_json_overrides_forbidden_without_permission():
    principal = _make_principal(roles=["user"], permissions=[], is_admin=False)
    app = _build_app_with_flashcards(principal)

    payload = b'[{"deck":"D","front":"F","back":"B"}]'
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/flashcards/import/json",
            params={"max_items": 1},
            files={"file": ("cards.json", payload, "application/json")},
        )
        assert resp.status_code == 403


@pytest.mark.unit
def test_flashcards_import_json_overrides_allowed_with_flashcards_admin_permission():
    principal = _make_principal(roles=["user"], permissions=[FLASHCARDS_ADMIN], is_admin=False)
    app = _build_app_with_flashcards(principal)

    payload = b'[{"deck":"D","front":"F","back":"B"}]'
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/flashcards/import/json",
            params={"max_items": 1},
            files={"file": ("cards.json", payload, "application/json")},
        )
        assert resp.status_code == 200
