from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

import pytest

from tldw_Server_API.app.core.AuthNZ.federation import provisioning_service as provisioning_module
from tldw_Server_API.app.core.AuthNZ.federation.provisioning_service import (
    FederationProvisioningService,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import AnchorOwnership


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

        async def remove_org_member_on_connection(self, **kwargs):
            observed_removals.append(kwargs)
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

    class _ObservedPool:
        pool = None

        def __init__(self) -> None:
            self.transaction_count = 0
            self.conn = object()

        @asynccontextmanager
        async def transaction(self):
            self.transaction_count += 1
            yield self.conn

    observed_removals: list[dict] = []
    pool = _ObservedPool()
    monkeypatch.setattr(provisioning_module, "AuthnzOrgsTeamsRepo", _StubOrgsRepo)
    monkeypatch.setattr(provisioning_module, "AuthnzUsersRepo", _StubUsersRepo)
    monkeypatch.setattr(provisioning_module, "FederatedManagedGrantRepo", _StubManagedGrantRepo)

    service = FederationProvisioningService(db_pool=pool)

    result = await service.apply_mapped_grants(
        provider={"id": 5, "provisioning_policy": {"mode": "sync_managed_only"}},
        user_id=7,
        mapped_claims={"derived_org_ids": [], "derived_team_ids": [], "derived_roles": []},
    )

    assert result["revoked_org_ids"] == [12]
    assert pool.transaction_count == 1
    assert len(observed_removals) == 1
    assert observed_removals[0]["conn"] is pool.conn
    assert (
        observed_removals[0]["context"]
        is provisioning_module._FEDERATION_MEMBERSHIP_CONTEXT
    )
    assert (
        observed_removals[0]["anchor_ownership"]
        is AnchorOwnership.WRITER_OWNS_ANCHOR
    )
    assert isinstance(observed_removals[0]["operation_time"], datetime)


@pytest.mark.asyncio
async def test_federated_org_grant_uses_bootstrap_writer_on_caller_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_grants: list[dict] = []

    class _ObservedPool:
        pool = None

        def __init__(self) -> None:
            self.transaction_count = 0
            self.conn = object()

        @asynccontextmanager
        async def transaction(self):
            self.transaction_count += 1
            yield self.conn

    class _StubOrgsRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def get_org_member(self, org_id: int, user_id: int):
            return None

        async def provision_org_membership_on_connection(self, **kwargs):
            observed_grants.append(kwargs)
            return None

    class _StubUsersRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

    class _StubManagedGrantRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def ensure_tables(self) -> None:
            return None

        async def list_for_provider_user(self, **kwargs):
            return []

        async def upsert_grant(self, **kwargs) -> None:
            return None

    pool = _ObservedPool()
    monkeypatch.setattr(provisioning_module, "AuthnzOrgsTeamsRepo", _StubOrgsRepo)
    monkeypatch.setattr(provisioning_module, "AuthnzUsersRepo", _StubUsersRepo)
    monkeypatch.setattr(
        provisioning_module,
        "FederatedManagedGrantRepo",
        _StubManagedGrantRepo,
    )

    result = await FederationProvisioningService(db_pool=pool).apply_mapped_grants(
        provider={"id": 5, "provisioning_policy": {"mode": "jit_grant_only"}},
        user_id=7,
        mapped_claims={
            "derived_org_ids": [12],
            "derived_team_ids": [],
            "derived_roles": [],
        },
    )

    assert result["org_ids"] == [12]
    assert pool.transaction_count == 1
    assert len(observed_grants) == 1
    assert observed_grants[0]["conn"] is pool.conn
    assert (
        observed_grants[0]["context"]
        is provisioning_module._FEDERATION_MEMBERSHIP_CONTEXT
    )
    assert (
        observed_grants[0]["anchor_ownership"]
        is AnchorOwnership.WRITER_OWNS_ANCHOR
    )
    assert observed_grants[0]["team_id"] is None
    assert observed_grants[0]["team_role"] is None
    assert observed_grants[0]["team_failure_is_best_effort"] is False
    assert isinstance(observed_grants[0]["operation_time"], datetime)


@pytest.mark.parametrize(
    ("mode", "derived_roles", "managed_rows"),
    (
        ("jit_grant_only", ["member"], []),
        (
            "sync_managed_only",
            [],
            [{"grant_kind": "role", "target_ref": "member"}],
        ),
    ),
)
@pytest.mark.asyncio
async def test_role_grant_failures_do_not_render_backend_exception(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    derived_roles: list[str],
    managed_rows: list[dict[str, str]],
) -> None:
    secret = "secret SQL at /private/federation.db"

    class _StubOrgsRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

    class _StubUsersRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def has_role_assignment(self, **_kwargs) -> bool:
            raise RuntimeError(secret)

    class _StubManagedGrantRepo:
        def __init__(self, db_pool=None) -> None:
            self.db_pool = db_pool

        async def ensure_tables(self) -> None:
            return None

        async def list_for_provider_user(self, **_kwargs):
            return managed_rows

    class _Logger:
        def __init__(self) -> None:
            self.events: list[str] = []

        def bind(self, **values):
            self.events.append(repr(values))
            return self

        def warning(self, message: str, *args) -> None:
            self.events.append(message.format(*args))

    logger_stub = _Logger()
    monkeypatch.setattr(provisioning_module, "AuthnzOrgsTeamsRepo", _StubOrgsRepo)
    monkeypatch.setattr(provisioning_module, "AuthnzUsersRepo", _StubUsersRepo)
    monkeypatch.setattr(
        provisioning_module,
        "FederatedManagedGrantRepo",
        _StubManagedGrantRepo,
    )
    monkeypatch.setattr(provisioning_module, "logger", logger_stub)

    await FederationProvisioningService(db_pool=object()).apply_mapped_grants(
        provider={"id": 5, "provisioning_policy": {"mode": mode}},
        user_id=7,
        mapped_claims={
            "derived_org_ids": [],
            "derived_team_ids": [],
            "derived_roles": derived_roles,
        },
    )

    rendered = "\n".join(logger_stub.events)
    assert secret not in rendered
    assert "RuntimeError" in rendered
