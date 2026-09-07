from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.org_team_schemas import (
    OrganizationCreateRequest,
    OrganizationSTTSettingsUpdate,
    OrganizationWatchlistsSettingsUpdate,
    OrgMemberAddRequest,
    OrgMemberRoleUpdateRequest,
    TeamMemberAddRequest,
    TeamMemberRoleUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipAuthorizationError,
    MembershipPreflightChanged,
    MembershipScopeNotFound,
    MembershipTargetNotFound,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_e2e_support_service, admin_orgs_service


class _CursorStub:
    def __init__(self, *, row: Any = None) -> None:
        self._row = row

    async def fetchone(self) -> Any:
        return self._row


class _SQLiteTxConnWithFetchrowTrap:
    def __init__(self, *, metadata_json: str | None) -> None:
        self._is_sqlite = True
        self.metadata_json = metadata_json
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchrow_called = False

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - should never run
        self.fetchrow_called = True
        raise AssertionError("sqlite path should not use fetchrow()")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        q = str(query).lower()
        if "select metadata from organizations" in q:
            return _CursorStub(row=(self.metadata_json,))
        if "update organizations set metadata" in q:
            return _CursorStub(row=None)
        raise AssertionError(f"Unexpected query: {query!r}")


class _PostgresTxConnWithSqliteTrap:
    def __init__(self, *, metadata_json: str | None) -> None:
        self._is_sqlite = False
        self.metadata_json = metadata_json
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((str(query), tuple(args)))
        q = str(query).lower()
        if "select metadata from organizations" in q:
            return {"metadata": self.metadata_json}
        return None

    async def execute(self, query: str, *args: Any) -> Any:
        self.execute_calls.append((str(query), args))
        if "?" in str(query):
            raise AssertionError("postgres path should not use sqlite placeholders")
        return "OK"


class _ExplodingTeamsDb:
    _is_sqlite = True

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("teams backend exploded at /private/orgs.db")


async def _assert_orgs_service_log_sanitized(
    call: Callable[[], Awaitable[Any]],
    *,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = admin_orgs_service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(RuntimeError):
            await call()
    finally:
        admin_orgs_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
async def test_admin_e2e_org_seed_atomically_assigns_requested_owner() -> None:
    captured: dict[str, Any] = {}

    class _Repo:
        async def list_organizations(self, **_kwargs):
            return [], 0

        async def create_organization_with_owner_membership(self, **kwargs):
            captured.update(kwargs)
            return {"id": 55, "owner_user_id": kwargs["owner_user_id"]}

    organization = await admin_e2e_support_service._ensure_org(
        orgs_repo=_Repo(),  # type: ignore[arg-type]
        owner_user_id=23,
    )

    assert organization["owner_user_id"] == 23
    assert captured["owner_user_id"] == 23
    assert captured["context"] is admin_e2e_support_service._E2E_MEMBERSHIP_CONTEXT


@pytest.mark.asyncio
async def test_create_org_with_owner_uses_atomic_owner_membership_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _atomic_create(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": 55,
            "name": "Atomic Org",
            "slug": "atomic-org",
            "owner_user_id": 9,
            "is_active": True,
            "created_at": None,
            "updated_at": None,
        }

    monkeypatch.setattr(
        admin_orgs_service,
        "core_create_organization_with_owner_membership",
        _atomic_create,
        raising=False,
    )

    response = await admin_orgs_service.create_org(
        OrganizationCreateRequest(
            name="Atomic Org",
            slug="atomic-org",
            owner_user_id=9,
        ),
        _admin_principal(),
    )

    assert response.owner_user_id == 9
    assert captured["owner_user_id"] == 9
    assert captured["context"].actor_user_id == 1
    assert (
        captured["context"].required_authority
        is admin_orgs_service.MembershipAuthority.PLATFORM_ADMIN
    )


@pytest.mark.asyncio
async def test_create_org_without_owner_uses_actor_authorized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _authorized_create(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "id": 55,
            "name": "Ownerless Org",
            "slug": "ownerless-org",
            "owner_user_id": None,
            "is_active": True,
            "created_at": None,
            "updated_at": None,
        }

    async def _unexpected_unchecked_create(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("admin creation must reauthorize the persisted actor")

    monkeypatch.setattr(
        admin_orgs_service,
        "core_create_organization_as_actor",
        _authorized_create,
        raising=False,
    )
    monkeypatch.setattr(
        admin_orgs_service,
        "core_create_organization",
        _unexpected_unchecked_create,
        raising=False,
    )

    response = await admin_orgs_service.create_org(
        OrganizationCreateRequest(
            name="Ownerless Org",
            slug="ownerless-org",
        ),
        _admin_principal(),
    )

    assert response.owner_user_id is None
    assert captured["context"].actor_user_id == 1
    assert (
        captured["context"].required_authority
        is admin_orgs_service.MembershipAuthority.PLATFORM_ADMIN
    )


@pytest.mark.asyncio
async def test_create_org_with_owner_rejects_admin_without_persisted_actor_id() -> None:
    principal = AuthPrincipal(
        kind="service",
        user_id=None,
        roles=["admin"],
        is_admin=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_orgs_service.create_org(
            OrganizationCreateRequest(
                name="Atomic Org",
                slug="atomic-org",
                owner_user_id=9,
            ),
            principal,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Not authorized to manage memberships"


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (MembershipAuthorizationError(), 403),
        (MembershipScopeNotFound(), 404),
        (MembershipTargetNotFound(), 404),
        (MembershipPreflightChanged(), 409),
    ],
)
def test_membership_control_error_http_mapping(error: Exception, expected_status: int) -> None:
    mapped = admin_orgs_service._membership_control_http_error(error)

    assert mapped.status_code == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_name", "core_name"),
    [
        ("add_team_member", "core_add_team_member"),
        ("remove_team_member", "core_remove_team_member"),
        ("update_team_member_role", "core_update_team_member_role"),
        ("add_org_member", "core_add_org_member"),
        ("remove_org_member", "core_remove_org_member"),
        ("update_org_member_role", "core_update_org_member_role"),
    ],
)
async def test_admin_membership_writes_map_preflight_races_to_conflict(
    monkeypatch: pytest.MonkeyPatch,
    service_name: str,
    core_name: str,
) -> None:
    async def _allow_team(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"org_id": 5}

    async def _allow_org(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _preflight_changed(*_args: Any, **_kwargs: Any) -> Any:
        raise MembershipPreflightChanged()

    monkeypatch.setattr(
        admin_orgs_service.admin_scope_service,
        "get_scoped_team",
        _allow_team,
    )
    monkeypatch.setattr(
        admin_orgs_service.admin_scope_service,
        "enforce_admin_org_access",
        _allow_org,
    )
    monkeypatch.setattr(admin_orgs_service, core_name, _preflight_changed)

    service = getattr(admin_orgs_service, service_name)
    request = object()
    principal = _admin_principal()
    if service_name == "add_team_member":
        call = service(11, TeamMemberAddRequest(user_id=7), request, principal)
    elif service_name == "remove_team_member":
        call = service(11, 7, request, principal)
    elif service_name == "update_team_member_role":
        call = service(
            11,
            7,
            TeamMemberRoleUpdateRequest(role="lead"),
            request,
            principal,
        )
    elif service_name == "add_org_member":
        call = service(5, OrgMemberAddRequest(user_id=7), request, principal)
    elif service_name == "remove_org_member":
        call = service(5, 7, request, principal)
    else:
        call = service(
            5,
            7,
            OrgMemberRoleUpdateRequest(role="admin"),
            request,
            principal,
        )

    with pytest.raises(HTTPException) as exc_info:
        await call

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_list_teams_by_org_sanitizes_backend_failure_log() -> None:
    await _assert_orgs_service_log_sanitized(
        lambda: admin_orgs_service.list_teams_by_org(org_id=42, db=_ExplodingTeamsDb()),
        expected_log="Failed to list teams by organization",
        raw_marker="teams backend exploded",
    )


@pytest.mark.asyncio
async def test_update_org_watchlists_settings_sqlite_path_ignores_fetchrow_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_org_access(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(admin_orgs_service.admin_scope_service, "enforce_admin_org_access", _allow_org_access)

    db = _SQLiteTxConnWithFetchrowTrap(metadata_json="{}")
    payload = OrganizationWatchlistsSettingsUpdate(require_include_default=True)

    response = await admin_orgs_service.update_org_watchlists_settings(
        org_id=55,
        payload=payload,
        principal=_admin_principal(),
        db=db,
    )

    assert response.org_id == 55
    assert response.require_include_default is True
    assert db.fetchrow_called is False
    assert any("select metadata from organizations" in q.lower() for q, _ in db.execute_calls)
    update_calls = [call for call in db.execute_calls if "update organizations set metadata" in call[0].lower()]
    assert update_calls, "sqlite update path should use execute()"
    persisted_json = update_calls[0][1][0]
    persisted_meta = json.loads(persisted_json)
    assert persisted_meta["watchlists"]["require_include_default"] is True


@pytest.mark.asyncio
async def test_get_org_watchlists_settings_sqlite_path_ignores_fetchrow_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_org_access(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(admin_orgs_service.admin_scope_service, "enforce_admin_org_access", _allow_org_access)

    db = _SQLiteTxConnWithFetchrowTrap(
        metadata_json=json.dumps({"watchlists": {"require_include_default": True}})
    )

    response = await admin_orgs_service.get_org_watchlists_settings(
        org_id=99,
        principal=_admin_principal(),
        db=db,
    )

    assert response.org_id == 99
    assert response.require_include_default is True
    assert db.fetchrow_called is False
    assert db.execute_calls and "select metadata from organizations" in db.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_update_org_watchlists_settings_postgres_path_uses_fetchrow_and_pg_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_org_access(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(admin_orgs_service.admin_scope_service, "enforce_admin_org_access", _allow_org_access)

    db = _PostgresTxConnWithSqliteTrap(metadata_json="{}")
    payload = OrganizationWatchlistsSettingsUpdate(require_include_default=True)

    response = await admin_orgs_service.update_org_watchlists_settings(
        org_id=77,
        payload=payload,
        principal=_admin_principal(),
        db=db,
    )

    assert response.org_id == 77
    assert response.require_include_default is True
    assert db.fetchrow_calls and "$1" in db.fetchrow_calls[0][0]
    assert db.execute_calls and "$1" in db.execute_calls[0][0]
    assert all("?" not in q for q, _ in db.fetchrow_calls)


@pytest.mark.asyncio
async def test_get_org_watchlists_settings_postgres_path_uses_fetchrow_and_pg_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_org_access(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(admin_orgs_service.admin_scope_service, "enforce_admin_org_access", _allow_org_access)

    db = _PostgresTxConnWithSqliteTrap(
        metadata_json=json.dumps({"watchlists": {"require_include_default": True}})
    )

    response = await admin_orgs_service.get_org_watchlists_settings(
        org_id=88,
        principal=_admin_principal(),
        db=db,
    )

    assert response.org_id == 88
    assert response.require_include_default is True
    assert db.fetchrow_calls and "$1" in db.fetchrow_calls[0][0]
    assert not db.execute_calls


@pytest.mark.asyncio
async def test_update_org_stt_settings_uses_repo_and_returns_persisted_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_org_access(*args, **kwargs) -> None:
        return None

    async def _ensure_org_exists(*args, **kwargs) -> None:
        return None

    class _RepoStub:
        def __init__(self, db: Any) -> None:
            self.db = db
            self.ensure_tables_called = False

        async def ensure_tables(self) -> None:
            self.ensure_tables_called = True

        async def get_settings(self, org_id: int) -> dict[str, Any] | None:
            assert org_id == 44
            return None

        async def upsert_settings(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["org_id"] == 44
            assert self.ensure_tables_called is True
            return {
                "org_id": 44,
                "delete_audio_after_success": False,
                "audio_retention_hours": 6.0,
                "redact_pii": True,
                "allow_unredacted_partials": False,
                "redact_categories": ["email"],
            }

    monkeypatch.setattr(admin_orgs_service.admin_scope_service, "enforce_admin_org_access", _allow_org_access)
    monkeypatch.setattr(admin_orgs_service, "_ensure_org_exists", _ensure_org_exists)
    monkeypatch.setattr(admin_orgs_service, "AuthnzOrgSttSettingsRepo", _RepoStub)

    response = await admin_orgs_service.update_org_stt_settings(
        org_id=44,
        payload=OrganizationSTTSettingsUpdate(
            delete_audio_after_success=False,
            audio_retention_hours=6.0,
            redact_pii=True,
            allow_unredacted_partials=False,
            redact_categories=["email"],
        ),
        principal=_admin_principal(),
        db=object(),
    )

    assert response.org_id == 44
    assert response.delete_audio_after_success is False
    assert response.audio_retention_hours == 6.0
    assert response.redact_pii is True
    assert response.allow_unredacted_partials is False
    assert response.redact_categories == ["email"]


@pytest.mark.asyncio
async def test_get_org_stt_settings_falls_back_to_stt_defaults_when_org_row_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _allow_org_access(*args, **kwargs) -> None:
        return None

    async def _ensure_org_exists(*args, **kwargs) -> None:
        return None

    class _RepoStub:
        def __init__(self, db: Any) -> None:
            self.db = db
            self.ensure_tables_called = False

        async def ensure_tables(self) -> None:
            self.ensure_tables_called = True

        async def get_settings(self, org_id: int) -> dict[str, Any] | None:
            assert org_id == 51
            assert self.ensure_tables_called is True
            return None

    class _SttConfig:
        delete_audio_after_success = True
        audio_retention_hours = 0.0
        redact_pii = False
        allow_unredacted_partials = True
        redact_categories = ["email", "phone"]

    monkeypatch.setattr(admin_orgs_service.admin_scope_service, "enforce_admin_org_access", _allow_org_access)
    monkeypatch.setattr(admin_orgs_service, "_ensure_org_exists", _ensure_org_exists)
    monkeypatch.setattr(admin_orgs_service, "AuthnzOrgSttSettingsRepo", _RepoStub)
    monkeypatch.setattr(admin_orgs_service, "get_stt_config", lambda: _SttConfig())

    response = await admin_orgs_service.get_org_stt_settings(
        org_id=51,
        principal=_admin_principal(),
        db=object(),
    )

    assert response.org_id == 51
    assert response.delete_audio_after_success is True
    assert response.audio_retention_hours == 0.0
    assert response.redact_pii is False
    assert response.allow_unredacted_partials is True
    assert response.redact_categories == ["email", "phone"]


@pytest.mark.asyncio
async def test_org_stt_settings_repo_ensure_tables_uses_pg_connection_execute() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import AuthnzOrgSttSettingsRepo

    db = _PostgresTxConnWithSqliteTrap(metadata_json=None)

    repo = AuthnzOrgSttSettingsRepo(db)
    await repo.ensure_tables()

    assert db.execute_calls
    assert any("create table if not exists org_stt_settings" in q.lower() for q, _ in db.execute_calls)


@pytest.mark.asyncio
async def test_ensure_org_stt_settings_pg_returns_false_when_execute_fails() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_org_stt_settings_pg

    class _BrokenPgExecutor:
        async def execute(self, query: str, *args: Any) -> Any:
            raise RuntimeError(f"broken execute for {query.split()[0]}")

    assert await ensure_org_stt_settings_pg(_BrokenPgExecutor()) is False
