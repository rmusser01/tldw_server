from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.federation import provisioning_service as provisioning_module
from tldw_Server_API.app.core.AuthNZ.federation.provisioning_service import (
    FederationProvisioningService,
)


@pytest.mark.asyncio
async def test_preview_mapped_grants_ignores_memberships_with_missing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubOrgsRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def get_org_member(self, org_id: int, user_id: int):
            if org_id == 12 and user_id == 7:
                return {"role": "member"}
            return None

        async def get_team_member(self, team_id: int, user_id: int):
            return None

        async def list_memberships_for_user(self, user_id: int):
            return [
                {"org_id": None, "team_id": None, "team_name": "Unexpected"},
                {"org_id": 12, "team_id": 99, "team_name": provisioning_module.DEFAULT_BASE_TEAM_NAME},
            ]

    class _StubUsersRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def has_role_assignment(self, user_id: int, role_name: str) -> bool:
            return False

    class _StubManagedGrantRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def ensure_tables(self) -> None:
            return None

        async def list_for_provider_user(self, *, identity_provider_id: int, user_id: int):
            return [{"grant_kind": "org", "target_ref": "12"}]

    monkeypatch.setattr(provisioning_module, "AuthnzOrgsTeamsRepo", _StubOrgsRepo)
    monkeypatch.setattr(provisioning_module, "AuthnzUsersRepo", _StubUsersRepo)
    monkeypatch.setattr(provisioning_module, "FederatedManagedGrantRepo", _StubManagedGrantRepo)

    service = FederationProvisioningService(db_pool=None)

    preview = await service.preview_mapped_grants(
        provider={"id": 5, "provisioning_policy": {"mode": "sync_managed_only"}},
        user_id=7,
        mapped_claims={"derived_org_ids": [], "derived_team_ids": [], "derived_roles": []},
    )

    assert preview["revoke_org_ids"] == [12]


@pytest.mark.asyncio
async def test_apply_mapped_grants_ignores_memberships_with_missing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubOrgsRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def get_org_member(self, org_id: int, user_id: int):
            if org_id == 12 and user_id == 7:
                return {"role": "member"}
            return None

        async def get_team_member(self, team_id: int, user_id: int):
            return None

        async def list_memberships_for_user(self, user_id: int):
            return [
                {"org_id": None, "team_name": None},
                {"org_id": 12, "team_name": provisioning_module.DEFAULT_BASE_TEAM_NAME},
            ]

        async def remove_org_member(self, *, org_id: int, user_id: int, context):
            assert context is provisioning_module._FEDERATION_MEMBERSHIP_CONTEXT
            return {"removed": True}

    class _StubUsersRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def has_role_assignment(self, user_id: int, role_name: str) -> bool:
            return False

        async def remove_role_if_present(self, user_id: int, role_name: str) -> bool:
            return False

    class _StubManagedGrantRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool
            self.deleted: list[tuple[str, str]] = []

        async def ensure_tables(self) -> None:
            return None

        async def list_for_provider_user(self, *, identity_provider_id: int, user_id: int):
            return [{"grant_kind": "org", "target_ref": "12"}]

        async def delete_grant(
            self,
            *,
            identity_provider_id: int,
            user_id: int,
            grant_kind: str,
            target_ref: str,
        ) -> None:
            self.deleted.append((grant_kind, target_ref))

        async def upsert_grant(self, **kwargs) -> None:  # noqa: ANN003
            return None

    monkeypatch.setattr(provisioning_module, "AuthnzOrgsTeamsRepo", _StubOrgsRepo)
    monkeypatch.setattr(provisioning_module, "AuthnzUsersRepo", _StubUsersRepo)
    monkeypatch.setattr(provisioning_module, "FederatedManagedGrantRepo", _StubManagedGrantRepo)

    service = FederationProvisioningService(db_pool=None)

    result = await service.apply_mapped_grants(
        provider={"id": 5, "provisioning_policy": {"mode": "sync_managed_only"}},
        user_id=7,
        mapped_claims={"derived_org_ids": [], "derived_team_ids": [], "derived_roles": []},
    )

    assert result["revoked_org_ids"] == [12]
