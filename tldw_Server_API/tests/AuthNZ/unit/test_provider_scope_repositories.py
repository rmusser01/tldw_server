"""Single-snapshot repository contracts independent of the database driver."""

from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import ProviderCredentialAliasConflictError


class _DriverRow:
    """A keys/index row, like asyncpg.Record or sqlite3.Row."""

    def __init__(self, data):
        self.data = data

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data.values())


class _Pool:
    def __init__(self, postgres, rows):
        self.pool = object() if postgres else None
        self.fetchall = AsyncMock(return_value=rows)


async def _resolve(repo, kind, **overrides):
    if kind == "user":
        args = {"user_id": 7, "provider": "oai", **overrides}
    else:
        args = {
            "scope_type": kind,
            "scope_id": 11 if kind == "team" else 13,
            "user_id": 7,
            "provider": "oai",
            "active_team_id": 11,
            "active_organization_id": 13,
            **overrides,
        }
    return await repo.resolve_authorized_secret(**args)


def _repo(pool, kind):
    return AuthnzUserProviderSecretsRepo(pool) if kind == "user" else AuthnzOrgProviderSecretsRepo(pool)


@pytest.mark.asyncio
@pytest.mark.parametrize("postgres", [False, True], ids=["sqlite", "postgres"])
@pytest.mark.parametrize("kind", ["user", "team", "org"])
async def test_resolution_normalizes_driver_row_in_one_bound_query(postgres, kind):
    record = {"id": 1, "provider": "openai", "encrypted_blob": "private-blob", "revoked_at": None}
    pool = _Pool(postgres, [_DriverRow(record)])
    result = await _resolve(_repo(pool, kind), kind)
    assert result.status == "resolved"
    assert dict(result.record) == record
    pool.fetchall.assert_awaited_once()
    sql, *args = pool.fetchall.call_args.args
    assert "private-blob" not in sql
    assert "oai" not in sql
    assert "LEFT JOIN" in sql
    assert "users" in sql and "is_active" in sql
    if kind != "user":
        assert "team_members" in sql and "org_members" in sql
        assert "teams" in sql and "organizations" in sql and "org_id" in sql
    assert args
    assert ("$1" in sql) == postgres


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "team", "org"])
@pytest.mark.parametrize("rows,state", [([], "unauthorized"), ([{"id": None}], "authorized_absent")])
async def test_resolution_distinguishes_authorization_from_absence(kind, rows, state):
    result = await _resolve(_repo(_Pool(False, rows), kind), kind)
    assert result.status == state
    assert result.record is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "team", "org"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "7", None, [7]])
async def test_resolution_rejects_nonpositive_or_inexact_user_id_without_query(kind, value):
    pool = _Pool(False, [])
    with pytest.raises(ValueError, match="user_id"):
        await _resolve(_repo(pool, kind), kind, user_id=value)
    pool.fetchall.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["scope_id", "active_team_id", "active_organization_id"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "11", [11]])
async def test_shared_resolution_validates_all_scope_identifiers(field, value):
    pool = _Pool(False, [])
    with pytest.raises(ValueError, match=field):
        await _resolve(_repo(pool, "team"), "team", **{field: value})
    pool.fetchall.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("kind", ["user", "team", "org"])
async def test_store_error_returns_unavailable_without_disclosing_exception(postgres, kind):
    pool = _Pool(postgres, [])
    pool.fetchall.side_effect = RuntimeError("private-blob in database exception")
    messages = []
    sink = logger.add(messages.append, format="{message} {extra}")
    try:
        result = await _resolve(_repo(pool, kind), kind)
    finally:
        logger.remove(sink)
    assert result.status == "unavailable"
    assert result.record is None
    assert "private-blob" not in repr(result) + "".join(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "team", "org"])
async def test_legacy_alias_conflict_remains_explicit(kind):
    rows = [
        {"id": i, "provider": provider, "revoked_at": None}
        for i, provider in enumerate(("custom-openai", "openai-compatible"), start=1)
    ]
    repo = _repo(_Pool(False, rows), kind)
    with pytest.raises(ProviderCredentialAliasConflictError):
        await _resolve(repo, kind, provider="custom-openai-api")
