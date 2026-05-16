from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
from tldw_Server_API.app.main import app as fastapi_app


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")


@pytest.fixture()
def client_with_source_health_overrides(
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> Iterator[TestClient]:
    async def override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop() -> None:
        return None

    async def _fake_principal(request: Request) -> AuthPrincipal:  # noqa: ARG001
        return AuthPrincipal(
            kind="service",
            user_id=None,
            api_key_id=None,
            subject="service:rag-source-health-test",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.logs", "media.read"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    async def _no_rbac(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        return None

    class StubDB:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    async def _stub_media_db() -> StubDB:
        return StubDB("stub_media.db")

    async def _stub_chacha_db() -> StubDB:
        return StubDB("stub_chacha.db")

    async def _stub_prompts_db() -> StubDB:
        return StubDB("stub_prompts.db")

    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _no_rbac)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _fake_principal
    fastapi_app.dependency_overrides[check_rate_limit] = _noop

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user

    fastapi_app.dependency_overrides[get_media_db_for_user] = _stub_media_db
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = _stub_chacha_db
    fastapi_app.dependency_overrides[get_prompts_db_for_user] = _stub_prompts_db

    with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


def test_rag_source_health_returns_safe_canonical_sources(
    monkeypatch: pytest.MonkeyPatch,
    client_with_source_health_overrides: TestClient,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    captured: dict[str, Any] = {}

    class StubRetriever:
        def __init__(self, db_paths: dict[str, str], user_id: str = "0", **kwargs: Any) -> None:
            captured["db_paths"] = dict(db_paths)
            captured["user_id"] = user_id
            captured["kwargs"] = dict(kwargs)
            self.retrievers = {
                DataSource.MEDIA_DB: object(),
                DataSource.NOTES: object(),
                DataSource.CHAT_HISTORY: object(),
                DataSource.CHARACTER_CARDS: object(),
                DataSource.KANBAN: object(),
                DataSource.PROMPTS: object(),
                DataSource.WORLD_BOOKS: object(),
                DataSource.DICTIONARIES: object(),
            }

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:  # noqa: ARG002
            raise AssertionError("source health must not execute retrieval")

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", StubRetriever)
    monkeypatch.setattr(rag_ep, "_resolve_kanban_db_path", lambda *args, **kwargs: "stub_kanban.db")

    response = client_with_source_health_overrides.get("/api/v1/rag/source-health")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [source["source_id"] for source in payload["sources"]] == [
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    ]
    assert all(source["available"] is True for source in payload["sources"])
    assert all(source["searchable"] is True for source in payload["sources"])
    assert all("title" not in source for source in payload["sources"])
    assert all("metadata" not in source for source in payload["sources"])
    assert "stub_media.db" not in response.text
    assert "stub_chacha.db" not in response.text
    assert "stub_prompts.db" not in response.text
    assert captured["db_paths"] == {
        "media_db": "stub_media.db",
        "notes_db": "stub_chacha.db",
        "character_cards_db": "stub_chacha.db",
        "world_books_db": "stub_chacha.db",
        "chat_dictionaries_db": "stub_chacha.db",
        "prompts_db": "stub_prompts.db",
        "kanban_db": "stub_kanban.db",
    }
    assert "media_db_path" not in captured["db_paths"]
    assert "notes_db_path" not in captured["db_paths"]
    assert "character_db_path" not in captured["db_paths"]
    assert "prompts_db_path" not in captured["db_paths"]
