from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileUpdateEntry,
    UserProfileUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileContractMode
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.services import admin_profiles_service


def _run_async(coro):
    return asyncio.run(coro)


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=5,
        username="admin",
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
        active_org_id=11,
        active_team_id=22,
    )


def test_admin_profile_update_delegates_to_command_service(monkeypatch) -> None:
    version = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_conn = object()
    captured: dict[str, object] = {}

    async def _allow_scope(*_args, **_kwargs) -> None:
        captured["scope_enforced"] = _args, _kwargs

    async def _fake_pool():
        return object()

    class _Repo:
        async def get_user_by_id(self, user_id: int) -> dict[str, object]:
            return {"id": user_id, "updated_at": version}

    async def _repo_from_pool():
        return _Repo()

    class _CommandService:
        def __init__(self, *, db_pool) -> None:
            captured["db_pool"] = db_pool

        async def apply(self, command, *, db_conn, scope):
            captured["command"] = command
            captured["db_conn"] = db_conn
            captured["scope"] = scope
            return LegacyProfileCommandResult(
                profile_version=version,
                applied=("preferences.ui.theme", "preferences.ui.theme"),
            )

    class _DirectUpdateServiceTrap:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("admin service must delegate to ProfileCommandService")

    monkeypatch.setattr(
        admin_profiles_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_profiles_service, "get_db_pool", _fake_pool)
    monkeypatch.setattr(admin_profiles_service.AuthnzUsersRepo, "from_pool", _repo_from_pool)
    monkeypatch.setattr(admin_profiles_service, "ProfileCommandService", _CommandService, raising=False)
    monkeypatch.setattr(admin_profiles_service, "UserProfileUpdateService", _DirectUpdateServiceTrap)

    principal = _admin_principal()
    payload = UserProfileUpdateRequest(
        profile_version=version,
        updates=[
            UserProfileUpdateEntry(key="preferences.ui.theme", value="paper"),
            UserProfileUpdateEntry(key="preferences.ui.theme", value="midnight"),
        ],
    )

    response, audit_info = _run_async(
        admin_profiles_service.update_user_profile(
            user_id=7,
            payload=payload,
            principal=principal,
            db=write_conn,
        )
    )

    command = captured["command"]
    assert command.actor_user_id == 5
    assert command.target_user_id == 7
    assert command.updates == (
        ("preferences.ui.theme", "paper"),
        ("preferences.ui.theme", "midnight"),
    )
    assert "admin" in command.roles
    assert command.dry_run is False
    assert command.expected_profile_version == version
    assert command.active_org_id == 11
    assert command.active_team_id == 22
    assert command.contract_mode == ProfileContractMode.LEGACY_V1
    assert captured["db_conn"] is write_conn

    scope = captured["scope"]
    assert scope.actor_user_id == 5
    assert scope.active_org_id == 11
    assert scope.active_team_id == 22

    assert response.profile_version == version
    assert response.applied == ["preferences.ui.theme", "preferences.ui.theme"]
    assert response.skipped == []
    assert audit_info["metadata"]["applied_count"] == 2
    assert audit_info["metadata"]["skipped_count"] == 0


def test_admin_profile_update_maps_command_error_to_legacy_error_response(monkeypatch) -> None:
    version = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _fake_pool():
        return object()

    class _Repo:
        async def get_user_by_id(self, user_id: int) -> dict[str, object]:
            return {"id": user_id, "updated_at": version}

    async def _repo_from_pool():
        return _Repo()

    class _CommandService:
        def __init__(self, *, db_pool) -> None:
            del db_pool

        async def apply(self, command, *, db_conn, scope):
            del command, db_conn, scope
            return LegacyProfileCommandResult(
                status_code=409,
                profile_version=version,
                error_code="profile_version_mismatch",
                detail="profile_version_mismatch",
                skipped=({"key": "profile_version", "message": "mismatch"},),
            )

    monkeypatch.setattr(
        admin_profiles_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(admin_profiles_service, "get_db_pool", _fake_pool)
    monkeypatch.setattr(admin_profiles_service.AuthnzUsersRepo, "from_pool", _repo_from_pool)
    monkeypatch.setattr(admin_profiles_service, "ProfileCommandService", _CommandService, raising=False)

    response, audit_info = _run_async(
        admin_profiles_service.update_user_profile(
            user_id=7,
            payload=UserProfileUpdateRequest(
                profile_version=version,
                updates=[UserProfileUpdateEntry(key="preferences.ui.theme", value="paper")],
            ),
            principal=_admin_principal(),
            db=object(),
        )
    )

    assert audit_info is None
    assert response.status_code == 409
    body = json.loads(response.body.decode("utf-8"))
    assert body == {
        "error_code": "profile_version_mismatch",
        "detail": "profile_version_mismatch",
        "errors": [{"key": "profile_version", "message": "mismatch"}],
    }


def test_admin_profile_update_rejects_empty_updates_before_scope_or_command(
    monkeypatch,
) -> None:
    async def _scope_trap(*_args, **_kwargs) -> None:
        raise AssertionError("empty update must be rejected before scope lookup")

    class _CommandTrap:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("empty update must not construct the command service")

    monkeypatch.setattr(
        admin_profiles_service.admin_scope_service,
        "enforce_admin_user_scope",
        _scope_trap,
    )
    monkeypatch.setattr(admin_profiles_service, "ProfileCommandService", _CommandTrap)

    response, audit_info = _run_async(
        admin_profiles_service.update_user_profile(
            user_id=7,
            payload=UserProfileUpdateRequest(updates=[]),
            principal=_admin_principal(),
            db=object(),
        )
    )

    assert audit_info is None
    assert response.status_code == 400
    assert json.loads(response.body.decode("utf-8")) == {
        "error_code": "profile_update_invalid",
        "detail": "No updates provided",
        "errors": [{"key": "updates", "message": "missing"}],
    }
