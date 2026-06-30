from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileUpdateEntry,
    UserProfileUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member,
    add_team_member,
    create_organization,
    create_team,
    list_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.UserProfiles import update_service as update_service_module
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileContractMode
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)


def _run_async(coro):
    return asyncio.run(coro)


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return resp.json()["user"]["id"]


def test_user_profile_update_preferences(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "paper"},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "preferences.ui.theme" in payload["applied"]

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        profile = profile_resp.json()

        effective_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "effective_config"},
            headers=auth_headers,
        )
        assert effective_resp.status_code == 200
        effective = effective_resp.json()

    assert profile.get("preferences", {}).get("preferences.ui.theme") == "paper"
    assert effective.get("effective_config", {}).get("preferences.ui.theme") == "paper"


@pytest.mark.asyncio
async def test_user_profile_override_writes_use_transaction_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_conn = object()
    calls: list[tuple[str, str, object | None]] = []

    class FakeOverridesRepo:
        def __init__(self, db_pool) -> None:
            self.db_pool = db_pool

        async def ensure_tables(self) -> None:
            return None

        async def upsert_override(
            self,
            *,
            user_id: int,
            key: str,
            value,
            updated_by: int | None,
            db_conn=None,
        ) -> None:
            calls.append(("upsert", key, db_conn))

        async def delete_override(
            self,
            *,
            user_id: int,
            key: str,
            db_conn=None,
        ) -> None:
            calls.append(("delete", key, db_conn))

    monkeypatch.setattr(
        update_service_module,
        "UserProfileOverridesRepo",
        FakeOverridesRepo,
    )

    result = await update_service_module.UserProfileUpdateService(
        db_pool=object(),
    ).apply_updates(
        user_id=7,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("preferences.chat.default_character_id", None),
        ),
        roles={"user"},
        dry_run=False,
        db_conn=db_conn,
        updated_by=7,
    )

    assert result.applied == [
        "preferences.ui.theme",
        "preferences.chat.default_character_id",
    ]
    assert result.skipped == []
    assert calls == [
        ("upsert", "preferences.ui.theme", db_conn),
        ("delete", "preferences.chat.default_character_id", db_conn),
    ]


def test_user_profile_preferences_include_sources(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "paper"},
                ]
            },
        )
        assert resp.status_code == 200

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences", "include_sources": "true"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        preferences = profile_resp.json().get("preferences", {})

    entry = preferences.get("preferences.ui.theme")
    assert entry.get("value") == "paper"
    assert entry.get("source") == "user"


def test_user_profile_update_default_character_preference_set_and_clear(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        set_resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "preferences.chat.default_character_id",
                        "value": "char-123",
                    }
                ]
            },
        )
        assert set_resp.status_code == 200
        assert (
            "preferences.chat.default_character_id"
            in set_resp.json().get("applied", [])
        )

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        preferences = profile_resp.json().get("preferences", {})
        assert preferences.get("preferences.chat.default_character_id") == "char-123"

        clear_resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "preferences.chat.default_character_id",
                        "value": None,
                    }
                ]
            },
        )
        assert clear_resp.status_code == 200
        assert (
            "preferences.chat.default_character_id"
            in clear_resp.json().get("applied", [])
        )

        profile_after_clear = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert profile_after_clear.status_code == 200
        cleared_preferences = profile_after_clear.json().get("preferences", {})

    assert "preferences.chat.default_character_id" not in cleared_preferences


def test_user_profile_update_default_character_preference_type_validation(
    auth_headers,
) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "preferences.chat.default_character_id",
                        "value": 123,
                    }
                ]
            },
        )
        assert resp.status_code == 422
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_invalid"
        assert payload.get("errors")


def test_admin_profile_update_storage_quota(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        warm_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert warm_resp.status_code == 200

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 4096},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "limits.storage_quota_mb" in payload["applied"]

        profile_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        profile = profile_resp.json()

    assert profile.get("quotas", {}).get("storage_quota_mb") == 4096


def test_user_profile_update_no_updates(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={"updates": []},
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_invalid"
        assert payload.get("errors")


def test_user_profile_update_version_conflict(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "profile_version": "2000-01-01T00:00:00Z",
                "updates": [
                    {"key": "preferences.ui.theme", "value": "midnight"},
                ],
            },
        )
        assert resp.status_code == 409
        payload = resp.json()
        assert payload.get("error_code") == "profile_version_mismatch"
        assert payload.get("errors")


def test_user_profile_update_unknown_key(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.unknown", "value": "oops"},
                ],
            },
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_unknown_key"
        assert payload.get("errors")


def test_user_profile_update_forbidden_key(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 1024},
                ],
            },
        )
        assert resp.status_code == 403
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_forbidden"
        assert payload.get("errors")


def test_user_profile_update_invalid_value(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": 123},
                ],
            },
        )
        assert resp.status_code == 422
        payload = resp.json()
        assert payload.get("error_code") == "profile_update_invalid"
        assert payload.get("errors")


def test_admin_profile_update_org_role(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Profile Org {suffix}", owner_user_id=None)
            await add_org_member(org_id=int(org["id"]), user_id=user_id, role="member")
            return int(org["id"])

        org_id = _run_async(_setup())

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "memberships.orgs.role", "value": {"org_id": org_id, "role": "admin"}}
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.orgs.role" in resp.json().get("applied", [])

        async def _fetch_roles():
            return await list_org_memberships_for_user(user_id)

        orgs = _run_async(_fetch_roles())
        target = next(item for item in orgs if int(item.get("org_id")) == org_id)
        assert target.get("role") == "admin"


def test_admin_profile_update_team_role(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Profile Team Org {suffix}", owner_user_id=None)
            await add_org_member(org_id=int(org["id"]), user_id=user_id, role="member")
            team = await create_team(org_id=int(org["id"]), name=f"Team {suffix}")
            await add_team_member(team_id=int(team["id"]), user_id=user_id, role="member")
            return int(team["id"])

        team_id = _run_async(_setup())

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "memberships.teams.role", "value": {"team_id": team_id, "role": "lead"}}
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.teams.role" in resp.json().get("applied", [])

        async def _fetch_team_roles():
            return await list_memberships_for_user(user_id)

        teams = _run_async(_fetch_team_roles())
        target = next(item for item in teams if int(item.get("team_id")) == team_id)
        assert target.get("role") == "lead"


def test_admin_profile_update_team_member_add_remove(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _setup():
            org = await create_organization(name=f"Profile Team Org 2 {suffix}", owner_user_id=None)
            await add_org_member(org_id=int(org["id"]), user_id=user_id, role="member")
            team_a = await create_team(org_id=int(org["id"]), name=f"Team A {suffix}")
            team_b = await create_team(org_id=int(org["id"]), name=f"Team B {suffix}")
            await add_team_member(team_id=int(team_a["id"]), user_id=user_id, role="member")
            return int(team_a["id"]), int(team_b["id"])

        team_a_id, team_b_id = _run_async(_setup())

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "memberships.teams.member",
                        "value": {"team_id": team_b_id, "action": "add", "role": "member"},
                    }
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.teams.member" in resp.json().get("applied", [])

        async def _list_memberships():
            return await list_memberships_for_user(user_id)

        memberships = _run_async(_list_memberships())
        team_ids = {int(item.get("team_id")) for item in memberships}
        assert team_b_id in team_ids

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {
                        "key": "memberships.teams.member",
                        "value": {"team_id": team_a_id, "action": "remove"},
                    }
                ]
            },
        )
        assert resp.status_code == 200
        assert "memberships.teams.member" in resp.json().get("applied", [])

        memberships = _run_async(_list_memberships())
        team_ids = {int(item.get("team_id")) for item in memberships}
        assert team_a_id not in team_ids


def test_user_profile_update_rejects_inactive_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "inactive-user",
            "email": "inactive@example.invalid",
            "role": "user",
            "is_active": False,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/users/me/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "preferences.ui.theme", "value": "paper"},
                ]
            },
        )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_user_profile_update_delegates_to_command_service(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    version = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_conn = object()
    captured: dict[str, object] = {}

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 7,
            "username": "delegated-user",
            "email": "delegated@example.invalid",
            "role": "user",
            "is_active": True,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": version,
            "last_login": None,
        }

    async def _fake_pool():
        return object()

    class _CommandService:
        def __init__(self, *, db_pool) -> None:
            captured["db_pool"] = db_pool

        async def apply(self, command, *, db_conn, scope):
            captured["command"] = command
            captured["db_conn"] = db_conn
            captured["scope"] = scope
            return LegacyProfileCommandResult(
                profile_version=version,
                applied=("preferences.ui.theme",),
            )

    class _DirectUpdateServiceTrap:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("route must delegate to ProfileCommandService")

    async def _audit(*_args, **_kwargs) -> None:
        captured["audit"] = _kwargs

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    monkeypatch.setattr(users_endpoints, "get_db_pool", _fake_pool)
    monkeypatch.setattr(users_endpoints, "ProfileCommandService", _CommandService, raising=False)
    monkeypatch.setattr(
        users_endpoints,
        "UserProfileUpdateService",
        _DirectUpdateServiceTrap,
        raising=False,
    )
    monkeypatch.setattr(users_endpoints, "_emit_user_profile_audit_event", _audit)

    principal = AuthPrincipal(
        kind="user",
        user_id=7,
        username="delegated-user",
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
    )
    payload = UserProfileUpdateRequest(
        profile_version=version,
        updates=[UserProfileUpdateEntry(key="preferences.ui.theme", value="paper")],
    )

    response = await users_endpoints.update_current_user_profile(
        payload,
        http_request=object(),
        principal=principal,
        db=write_conn,
    )

    command = captured["command"]
    assert command.actor_user_id == 7
    assert command.target_user_id == 7
    assert command.updates == (("preferences.ui.theme", "paper"),)
    assert command.roles == frozenset({"user"})
    assert command.dry_run is False
    assert command.expected_profile_version == version
    assert command.contract_mode == ProfileContractMode.LEGACY_V1
    assert captured["db_conn"] is write_conn
    assert captured["scope"] is None
    assert response.profile_version == version
    assert response.applied == ["preferences.ui.theme"]
    assert response.skipped == []
    assert captured["audit"]["applied_count"] == 1


def test_admin_profile_update_audio_limits(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.audio_daily_minutes", "value": 120},
                    {"key": "limits.audio_concurrent_jobs", "value": 4},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "limits.audio_daily_minutes" in payload.get("applied", [])
        assert "limits.audio_concurrent_jobs" in payload.get("applied", [])

        profile_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        quotas = profile_resp.json().get("quotas", {})
        audio = quotas.get("audio", {})
        assert audio.get("daily_minutes_limit") == 120
        assert audio.get("concurrent_jobs_limit") == 4


def test_admin_profile_update_evaluations_limits(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={
                "updates": [
                    {"key": "limits.evaluations_per_minute", "value": 42},
                    {"key": "limits.evaluations_per_day", "value": 900},
                ]
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert "limits.evaluations_per_minute" in payload.get("applied", [])
        assert "limits.evaluations_per_day" in payload.get("applied", [])

        profile_resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        quotas = profile_resp.json().get("quotas", {})
        evaluations = quotas.get("evaluations", {})
        limits = evaluations.get("limits", {})
        assert limits.get("per_minute", {}).get("evaluations") == 42
        assert limits.get("daily", {}).get("evaluations") == 900


def test_admin_profile_update_identity_locked(auth_headers) -> None:
    with TestClient(app) as client:
        profile_resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
        assert profile_resp.status_code == 200
        user = profile_resp.json().get("user", {})
        user_id = int(user.get("id"))
        username = user.get("username")
        assert username

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={"updates": [{"key": "identity.is_locked", "value": True}]},
        )
        assert resp.status_code == 200
        assert "identity.is_locked" in resp.json().get("applied", [])

        limiter = get_rate_limiter()
        is_locked, _ = _run_async(limiter.check_lockout(str(username), attempt_type="login"))
        assert is_locked is True

        resp = client.patch(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
            json={"updates": [{"key": "identity.is_locked", "value": False}]},
        )
        assert resp.status_code == 200
        assert "identity.is_locked" in resp.json().get("applied", [])

        is_locked, _ = _run_async(limiter.check_lockout(str(username), attempt_type="login"))
        assert is_locked is False
