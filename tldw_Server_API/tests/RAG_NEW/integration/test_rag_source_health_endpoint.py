from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
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

    async def _fail_source_db_dependency() -> None:
        raise AssertionError("source health must not instantiate source databases")

    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _no_rbac)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _fake_principal
    fastapi_app.dependency_overrides[check_rate_limit] = _noop

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user

    fastapi_app.dependency_overrides[get_media_db_for_user] = _fail_source_db_dependency
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = _fail_source_db_dependency
    fastapi_app.dependency_overrides[get_prompts_db_for_user] = _fail_source_db_dependency

    with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


def test_rag_source_health_returns_safe_canonical_sources(
    monkeypatch: pytest.MonkeyPatch,
    client_with_source_health_overrides: TestClient,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    def fail_retriever_construction(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise AssertionError("source health must not instantiate retrievers")

    monkeypatch.setattr(rag_ep, "MultiDatabaseRetriever", fail_retriever_construction)
    monkeypatch.setattr(
        rag_ep,
        "_resolve_existing_source_db_paths",
        lambda *args, **kwargs: {
            "media_db": "stub_media.db",
            "chacha_db": "stub_chacha.db",
            "prompts_db": "stub_prompts.db",
            "kanban_db": "stub_kanban.db",
        },
    )

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
    assert "stub_kanban.db" not in response.text
