from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileUpdateRequest,
)
from tldw_Server_API.app.api.v2.endpoints import user_profiles
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.unit


class _FakeCommandService:
    def __init__(self, result: LegacyProfileCommandResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def apply(self, command, *, db_conn, scope):
        self.calls.append({"command": command, "db_conn": db_conn, "scope": scope})
        return self.result


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        self.debugs.append(message)


def test_v2_single_update_has_no_skipped_field(auth_headers) -> None:
    with TestClient(app) as client:
        response = client.patch(
            "/api/v2/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "v2-contract"},
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["applied"] == ["preferences.ui.theme"]
    assert "skipped" not in payload


@pytest.mark.asyncio
async def test_v2_stale_version_precedes_invalid_value_rejection(monkeypatch) -> None:
    async def _active_user(_principal):
        return {"id": "7"}

    monkeypatch.setattr(
        user_profiles,
        "_require_principal_active_verified",
        _active_user,
    )
    stale_version = datetime(2000, 1, 1, tzinfo=timezone.utc)
    command_service = _FakeCommandService(
        LegacyProfileCommandResult(
            status_code=409,
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
            skipped=({"key": "profile_version", "message": "mismatch"},),
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await user_profiles.update_current_user_profile_v2(
            payload=UserProfileUpdateRequest(
                profile_version=stale_version,
                updates=[
                    {"key": "identity.email", "value": "not-an-email"},
                ],
            ),
            http_request=SimpleNamespace(),
            principal=object(),
            db=object(),
            command_service=command_service,
        )

    command = command_service.calls[0]["command"]
    assert command.expected_profile_version == stale_version
    assert command.updates == (("identity.email", "not-an-email"),)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "error_code": "profile_version_mismatch",
        "detail": "profile_version_mismatch",
        "errors": [{"key": "profile_version", "message": "mismatch"}],
    }


def test_v2_profile_update_route_has_rate_limit_dependency() -> None:
    route = next(
        route
        for route in user_profiles.router.routes
        if isinstance(route, APIRoute) and route.path == "/users/me/profile"
    )

    assert check_rate_limit in [dependency.call for dependency in route.dependant.dependencies]


@pytest.mark.asyncio
async def test_v2_profile_update_uses_injected_command_service_and_structured_errors(
    monkeypatch,
) -> None:
    async def _active_user(_principal):
        return {"id": "7"}

    monkeypatch.setattr(
        user_profiles,
        "_require_principal_active_verified",
        _active_user,
    )
    command_service = _FakeCommandService(
        LegacyProfileCommandResult(
            status_code=422,
            error_code="profile_update_invalid",
            detail="One or more updates failed validation",
            skipped=({"key": "identity.email", "message": "invalid_email"},),
        )
    )
    db_conn = object()
    payload = UserProfileUpdateRequest(
        updates=[{"key": "identity.email", "value": "not-an-email"}]
    )

    with pytest.raises(HTTPException) as exc_info:
        await user_profiles.update_current_user_profile_v2(
            payload=payload,
            http_request=SimpleNamespace(),
            principal=object(),
            db=db_conn,
            command_service=command_service,
        )

    assert command_service.calls[0]["db_conn"] is db_conn
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == {
        "error_code": "profile_update_invalid",
        "detail": "One or more updates failed validation",
        "errors": [{"key": "identity.email", "message": "invalid_email"}],
    }


@pytest.mark.asyncio
async def test_v2_forbidden_command_result_has_no_audit(monkeypatch) -> None:
    async def _active_user(_principal):
        return {"id": "7"}

    audit_calls: list[dict[str, Any]] = []

    async def _audit(*_args, **kwargs) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(
        user_profiles,
        "_require_principal_active_verified",
        _active_user,
    )
    monkeypatch.setattr(user_profiles, "_emit_user_profile_audit_event", _audit)
    command_service = _FakeCommandService(
        LegacyProfileCommandResult(
            status_code=403,
            error_code="profile_update_forbidden",
            detail="Caller cannot edit one or more fields",
            skipped=(
                {"key": "memberships.teams.role", "message": "forbidden"},
            ),
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await user_profiles.update_current_user_profile_v2(
            payload=UserProfileUpdateRequest(
                updates=[
                    {
                        "key": "memberships.teams.role",
                        "value": {"team_id": 4, "role": "admin"},
                    }
                ]
            ),
            http_request=SimpleNamespace(),
            principal=object(),
            db=object(),
            command_service=command_service,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == {
        "error_code": "profile_update_forbidden",
        "detail": "Caller cannot edit one or more fields",
        "errors": [{"key": "memberships.teams.role", "message": "forbidden"}],
    }
    assert audit_calls == []


@pytest.mark.asyncio
async def test_v2_profile_update_logs_audit_failure_without_failing(monkeypatch) -> None:
    async def _active_user(_principal):
        return {"id": "7"}

    async def _failing_audit(*_args, **_kwargs) -> None:
        raise AttributeError("audit backend exploded at /private/profile-audit.db")

    monkeypatch.setattr(
        user_profiles,
        "_require_principal_active_verified",
        _active_user,
    )
    monkeypatch.setattr(user_profiles, "_emit_user_profile_audit_event", _failing_audit)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(user_profiles, "logger", logger_stub)
    profile_version = datetime(2026, 1, 4, tzinfo=timezone.utc)
    command_service = _FakeCommandService(
        LegacyProfileCommandResult(
            profile_version=profile_version,
            applied=("preferences.ui.theme",),
        )
    )

    response = await user_profiles.update_current_user_profile_v2(
        payload=UserProfileUpdateRequest(
            updates=[{"key": "preferences.ui.theme", "value": "paper"}]
        ),
        http_request=SimpleNamespace(),
        principal=object(),
        db=object(),
        command_service=command_service,
    )

    assert response.profile_version == profile_version
    assert response.applied == ["preferences.ui.theme"]
    assert logger_stub.debugs == ["User profile v2 audit emission skipped"]
    assert "audit backend exploded" not in str(logger_stub.debugs)
    assert "/private/profile-audit.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_v2_dry_run_preserves_duplicate_applied_key_order_without_audit(
    monkeypatch,
) -> None:
    async def _active_user(_principal):
        return {"id": "7"}

    audit_calls: list[dict[str, Any]] = []

    async def _audit(*_args, **kwargs) -> None:
        audit_calls.append(kwargs)

    monkeypatch.setattr(
        user_profiles,
        "_require_principal_active_verified",
        _active_user,
    )
    monkeypatch.setattr(user_profiles, "_emit_user_profile_audit_event", _audit)
    profile_version = datetime(2026, 1, 5, tzinfo=timezone.utc)
    command_service = _FakeCommandService(
        LegacyProfileCommandResult(
            profile_version=profile_version,
            applied=("preferences.ui.theme", "preferences.ui.theme"),
        )
    )

    response = await user_profiles.update_current_user_profile_v2(
        payload=UserProfileUpdateRequest(
            dry_run=True,
            updates=[
                {"key": "preferences.ui.theme", "value": "paper"},
                {"key": "preferences.ui.theme", "value": "midnight"},
            ],
        ),
        http_request=SimpleNamespace(),
        principal=object(),
        db=object(),
        command_service=command_service,
    )

    assert command_service.calls[0]["command"].updates == (
        ("preferences.ui.theme", "paper"),
        ("preferences.ui.theme", "midnight"),
    )
    assert response.profile_version == profile_version
    assert response.applied == ["preferences.ui.theme", "preferences.ui.theme"]
    assert audit_calls == []
