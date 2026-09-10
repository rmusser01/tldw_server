"""Regress array binding at repository -> DatabasePool -> driver boundaries."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, _flatten_params
from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
    OrgProfileOverridesRepo,
    TeamProfileOverridesRepo,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def postgres_pool(monkeypatch: pytest.MonkeyPatch):
    """Keep DatabasePool binding real and replace only the external driver."""
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL="sqlite:///:memory:"))
    pool.pool = object()
    connection = AsyncMock()

    @asynccontextmanager
    async def acquire():
        yield connection

    monkeypatch.setattr(pool, "acquire", acquire)
    return pool, connection


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repo_type", "scope"),
    [(OrgProfileOverridesRepo, "org"), (TeamProfileOverridesRepo, "team")],
)
@pytest.mark.parametrize("ids", [[], [7], [7, 11]])
async def test_override_lists_bind_one_array(postgres_pool, repo_type, scope, ids):
    """A one-element ID list must not become a scalar query argument."""
    pool, connection = postgres_pool
    updated_at = datetime(2026, 1, 2, 3, 4, 5)
    connection.fetch.return_value = [
        {
            f"{scope}_id": 7,
            "key": "theme",
            "value_json": '"dark"',
            "updated_at": updated_at,
            "updated_by": 3,
        }
    ]
    result = await getattr(repo_type(pool), f"list_overrides_for_{scope}s")(ids)

    if not ids:
        assert result == []
        connection.fetch.assert_not_awaited()
        return
    assert connection.fetch.await_args.args[1:] == (ids,)
    assert result == [
        {
            f"{scope}_id": 7,
            "key": "theme",
            "value": "dark",
            "updated_at": updated_at,
            "updated_by": 3,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repo_type", "scope"),
    [(OrgProfileOverridesRepo, "org"), (TeamProfileOverridesRepo, "team")],
)
@pytest.mark.parametrize("ids", [[], [7], [7, 11]])
async def test_override_update_versions_bind_one_array(postgres_pool, repo_type, scope, ids):
    """Profile version queries preserve arrays for one or many memberships."""
    pool, connection = postgres_pool
    updated_at = datetime(2026, 1, 2, 3, 4, 5)
    connection.fetchrow.return_value = {"updated_at": updated_at}

    result = await getattr(repo_type(pool), f"get_latest_update_for_{scope}s")(ids)

    if not ids:
        assert result is None
        connection.fetchrow.assert_not_awaited()
        return
    assert connection.fetchrow.await_args.args[1:] == (ids,)
    assert result == updated_at


@pytest.mark.asyncio
@pytest.mark.parametrize("include_revoked", [False, True])
@pytest.mark.parametrize("ids, expected_ids", [([], []), ([7], [7]), ([11, 7, 7, 0], [7, 11])])
async def test_managed_secret_refs_bind_one_normalized_array(postgres_pool, include_revoked, ids, expected_ids):
    """Bulk credential lookup must preserve its deduplicated ID array."""
    pool, connection = postgres_pool
    connection.fetch.return_value = [{"id": 7, "metadata_json": '{"region":"local"}'}]

    result = await ManagedSecretRefsRepo(pool).list_refs_by_ids(ids, include_revoked=include_revoked)

    if not expected_ids:
        assert result == {}
        connection.fetch.assert_not_awaited()
        return
    assert connection.fetch.await_args.args[1:] == (expected_ids,)
    assert result[7]["metadata"] == {"region": "local"}


@pytest.mark.asyncio
@pytest.mark.parametrize("repo_type", [AuthnzUsersRepo, AuthnzOrgsTeamsRepo])
@pytest.mark.parametrize("ids", [None, [], [7], [7, 11]])
async def test_scoped_listing_counts_preserve_an_array_as_the_only_filter(postgres_pool, repo_type, ids):
    """Pagination adds scalars to page queries, but their count still needs an array."""
    pool, connection = postgres_pool
    connection.fetch.return_value = []
    connection.fetchval.return_value = 2
    repo = repo_type(pool)

    if repo_type is AuthnzUsersRepo:
        rows, total = await repo.list_users(offset=0, limit=10, org_ids=ids)
    else:
        rows, total = await repo.list_organizations(offset=0, limit=10, org_ids=ids, with_total=True)

    assert rows == []
    if ids == []:
        assert total == 0
        connection.fetchval.assert_not_awaited()
        return
    assert total == 2
    assert connection.fetchval.await_args.args[1:] == (() if ids is None else (ids,))


@given(st.lists(st.integers(min_value=1, max_value=2**31 - 1), max_size=100))
def test_explicit_single_array_parameter_preserves_values(ids):
    """Nested parameter sequences distinguish an array from variadic arguments."""
    assert _flatten_params(((ids,),)) == (ids,)


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [(7, "active"), ([7, "active"],), ((7, "active"),)])
async def test_pool_retains_variadic_and_sequence_argument_compatibility(postgres_pool, args):
    """Document the existing convention that the caller fixes must preserve."""
    pool, connection = postgres_pool
    connection.fetchrow.return_value = {"id": 7}

    assert await pool.fetchone("SELECT id FROM sessions WHERE id = $1 AND status = $2", *args) == {"id": 7}
    assert connection.fetchrow.await_args.args[1:] == (7, "active")
