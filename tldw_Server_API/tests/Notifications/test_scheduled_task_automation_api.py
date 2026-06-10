from __future__ import annotations

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(*, permissions: list[str] | None = None) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=880,
        api_key_id=None,
        subject="scheduled-task-automation-api-test",
        token_type="access",  # nosec B106
        jti=None,
        roles=[],
        permissions=[TASKS_READ, TASKS_CONTROL] if permissions is None else list(permissions),
        is_admin=False,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
    )


def _override_auth(client, *, permissions: list[str] | None = None) -> None:
    principal = _make_principal(permissions=permissions)

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    async def _fake_get_request_user() -> User:
        return User(id=880, username="scheduled-user", email=None, is_active=True)

    client.app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal
    client.app.dependency_overrides[get_request_user] = _fake_get_request_user


@pytest.fixture()
def scheduled_tasks_client(client_user_only):
    _override_auth(client_user_only)
    yield client_user_only
    client_user_only.app.dependency_overrides.pop(get_auth_principal, None)
    client_user_only.app.dependency_overrides.pop(get_request_user, None)


def test_scheduled_task_static_child_routes_do_not_resolve_as_task_ids(scheduled_tasks_client, auth_headers):
    for path in (
        "/api/v1/scheduled-tasks/capabilities",
        "/api/v1/scheduled-tasks/previews",
        "/api/v1/scheduled-tasks/definitions",
    ):
        response = scheduled_tasks_client.get(path, headers=auth_headers)
        assert response.status_code != 404, response.text  # nosec B101
        assert response.text != "scheduled_task_not_found"  # nosec B101


def test_capabilities_report_definition_actions_but_no_execution(scheduled_tasks_client, auth_headers):
    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks/capabilities", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101
    body = response.json()
    families = {item["family"]: item for item in body["items"]}
    assert {"recurring_question", "agent_task"} <= set(families)  # nosec B101
    for family in ("recurring_question", "agent_task"):
        actions = families[family]["actions"]
        assert actions["preview"]["status"] == "available"  # nosec B101
        assert actions["create_definition"]["status"] == "available"  # nosec B101
        assert actions["execute"]["status"] == "unavailable"  # nosec B101
        assert actions["execute"]["reason"] == "execution_not_implemented"  # nosec B101
