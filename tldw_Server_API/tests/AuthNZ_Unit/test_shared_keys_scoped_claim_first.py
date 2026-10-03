from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import shared_keys_scoped
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _principal(
    *,
    user_id: int | None,
    roles: list[str] | None = None,
    is_admin: bool = False,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        roles=roles or [],
        permissions=[],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
async def test_require_org_manager_allows_admin_role_claim_without_membership_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fail_if_called(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("admin principal should bypass org membership lookup")

    monkeypatch.setattr(shared_keys_scoped, "list_org_members", _fail_if_called)

    await shared_keys_scoped._require_org_manager(
        _principal(user_id=None, roles=["admin"], is_admin=False),
        org_id=11,
    )


@pytest.mark.asyncio
async def test_require_org_manager_rejects_boolean_admin_without_claims() -> None:
    with pytest.raises(HTTPException) as exc:
        await shared_keys_scoped._require_org_manager(
            _principal(user_id=None, roles=["user"], is_admin=True),
            org_id=11,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Org manager role required"


@pytest.mark.asyncio
async def test_require_org_manager_denies_missing_user_id_for_non_admin() -> None:
    with pytest.raises(HTTPException) as exc:
        await shared_keys_scoped._require_org_manager(
            _principal(user_id=None, roles=["user"], is_admin=False),
            org_id=7,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Org manager role required"


@pytest.mark.asyncio
async def test_require_team_manager_allows_lead_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_team_members(team_id: int):
        assert team_id == 25
        return [
            {"user_id": 5, "role": "member", "status": "active"},
            {"user_id": 42, "role": "lead", "status": "active"},
        ]

    monkeypatch.setattr(shared_keys_scoped, "list_team_members", _fake_team_members)

    await shared_keys_scoped._require_team_manager(
        _principal(user_id=42, roles=["user"], is_admin=False),
        team_id=25,
    )


@pytest.mark.asyncio
async def test_require_team_manager_denies_non_manager_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_team_members(team_id: int):
        assert team_id == 3
        return [{"user_id": 42, "role": "member", "status": "active"}]

    monkeypatch.setattr(shared_keys_scoped, "list_team_members", _fake_team_members)

    with pytest.raises(HTTPException) as exc:
        await shared_keys_scoped._require_team_manager(
            _principal(user_id=42, roles=["user"], is_admin=False),
            team_id=3,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Team manager role required"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [None, "inactive", "suspended"])
async def test_require_team_manager_denies_inactive_lead(
    monkeypatch: pytest.MonkeyPatch, status: str | None,
) -> None:
    async def team_members(_team_id: int) -> list[dict]:
        return [{"user_id": 42, "role": "lead", "status": status}]

    monkeypatch.setattr(shared_keys_scoped, "list_team_members", team_members)
    with pytest.raises(HTTPException) as exc:
        await shared_keys_scoped._require_team_manager(
            _principal(user_id=42, roles=["user"]),
            team_id=3,
        )
    assert exc.value.status_code == 403


def test_principal_user_id_rejects_invalid_values() -> None:
    with pytest.raises(HTTPException) as exc:
        shared_keys_scoped._principal_user_id(
            _principal(user_id=None, roles=["admin"], is_admin=True)
        )
    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid user context"
