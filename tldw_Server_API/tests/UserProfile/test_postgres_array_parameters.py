"""Regress array-bound query results through repositories and the real DatabasePool."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
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


class _ArrayQueryDriver:
    """Select fixture rows while enforcing PostgreSQL's scalar/array argument boundary."""

    def __init__(self) -> None:
        """Start with an empty relation whose array filter selects the id column."""
        self.rows: list[dict[str, Any]] = []
        self.id_column = "id"

    @staticmethod
    def _integer_array(value: object) -> list[int]:
        """Accept integer sequences and reject a scalar bound to an integer array."""
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, int) for item in value):
            raise TypeError("An integer array query parameter must be an integer sequence")
        return list(value)

    async def fetch(self, query: str, *params: object) -> list[dict[str, Any]]:
        """Evaluate the array filters and pagination used by these repository queries."""
        parameter_count = max((int(index) for index in re.findall(r"\$(\d+)", query)), default=0)
        if len(params) != parameter_count:
            raise ValueError("Query parameter count does not match its placeholders")
        if query == "SELECT $1::int[] AS ids":
            return [{"ids": self._integer_array(params[0])}]

        rows = self.rows
        if "ANY(" in query:
            ids = self._integer_array(params[0])
            rows = [row for row in rows if row[self.id_column] in ids]
        elif "WHERE id = $1 AND status = $2" in query:
            rows = [row for row in rows if row["id"] == params[0] and row["status"] == params[1]]
        if "revoked_at IS NULL" in query:
            rows = [row for row in rows if row["revoked_at"] is None]
        if "LIMIT" in query:
            limit, offset = params[-2:]
            if not isinstance(limit, int) or not isinstance(offset, int):
                raise TypeError("Pagination parameters must be integers")
            rows = rows[offset : offset + limit]
        return rows

    async def fetchrow(self, query: str, *params: object) -> dict[str, Any] | None:
        """Return a selected row or the maximum activity among selected overrides."""
        rows = await self.fetch(query, *params)
        if "MAX(updated_at)" in query:
            return {"updated_at": max((row["updated_at"] for row in rows), default=None)}
        return rows[0] if rows else None

    async def fetchval(self, query: str, *params: object) -> int:
        """Count the rows selected by the repository's organization filter."""
        return len(await self.fetch(query, *params))


def _make_postgres_pool(monkeypatch: pytest.MonkeyPatch) -> tuple[DatabasePool, _ArrayQueryDriver]:
    """Keep DatabasePool binding real and replace only the external driver."""
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL="sqlite:///:memory:"))
    pool.pool = object()
    driver = _ArrayQueryDriver()

    @asynccontextmanager
    async def acquire() -> AsyncIterator[_ArrayQueryDriver]:
        """Provide the seeded driver for a public DatabasePool query."""
        yield driver

    monkeypatch.setattr(pool, "acquire", acquire)
    return pool, driver


@pytest.fixture
def postgres_pool(monkeypatch: pytest.MonkeyPatch) -> tuple[DatabasePool, _ArrayQueryDriver]:
    """Provide independent rows and a real pool for each repository test."""
    return _make_postgres_pool(monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repo_type", "scope"),
    [(OrgProfileOverridesRepo, "org"), (TeamProfileOverridesRepo, "team")],
)
@pytest.mark.parametrize("ids", [[], [7], [7, 11]])
async def test_override_lists_select_requested_memberships(
    postgres_pool: tuple[DatabasePool, _ArrayQueryDriver],
    repo_type: type[OrgProfileOverridesRepo] | type[TeamProfileOverridesRepo],
    scope: Literal["org", "team"],
    ids: list[int],
) -> None:
    """Empty, single, and multiple membership arrays select only their overrides."""
    pool, driver = postgres_pool
    driver.id_column = f"{scope}_id"
    updated_at = datetime(2026, 1, 2, 3, 4, 5)
    driver.rows = [
        {
            f"{scope}_id": member_id,
            "key": "theme",
            "value_json": '"dark"',
            "updated_at": updated_at,
            "updated_by": 3,
        }
        for member_id in [7, 11, 19]
    ]

    result = await getattr(repo_type(pool), f"list_overrides_for_{scope}s")(ids)

    assert result == [
        {
            f"{scope}_id": member_id,
            "key": "theme",
            "value": "dark",
            "updated_at": updated_at,
            "updated_by": 3,
        }
        for member_id in ids
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("repo_type", "scope"),
    [(OrgProfileOverridesRepo, "org"), (TeamProfileOverridesRepo, "team")],
)
@pytest.mark.parametrize(
    ("ids", "expected_update"),
    [([], None), ([7], datetime(2026, 1, 7)), ([7, 11], datetime(2026, 1, 11))],
)
async def test_override_versions_select_latest_requested_membership(
    postgres_pool: tuple[DatabasePool, _ArrayQueryDriver],
    repo_type: type[OrgProfileOverridesRepo] | type[TeamProfileOverridesRepo],
    scope: Literal["org", "team"],
    ids: list[int],
    expected_update: datetime | None,
) -> None:
    """The latest profile version excludes updates from unrelated memberships."""
    pool, driver = postgres_pool
    driver.id_column = f"{scope}_id"
    driver.rows = [
        {f"{scope}_id": 7, "updated_at": datetime(2026, 1, 7)},
        {f"{scope}_id": 11, "updated_at": datetime(2026, 1, 11)},
        {f"{scope}_id": 19, "updated_at": datetime(2026, 1, 19)},
    ]

    result = await getattr(repo_type(pool), f"get_latest_update_for_{scope}s")(ids)

    assert result == expected_update


@pytest.mark.asyncio
@pytest.mark.parametrize("include_revoked", [False, True])
@pytest.mark.parametrize("ids, expected_ids", [([], []), ([7], [7]), ([11, 7, 7, 0, -1], [7, 11])])
async def test_managed_secret_refs_select_normalized_identifiers(
    postgres_pool: tuple[DatabasePool, _ArrayQueryDriver],
    include_revoked: bool,
    ids: list[int],
    expected_ids: list[int],
) -> None:
    """Bulk lookup removes invalid IDs and respects revocation and metadata decoding."""
    pool, driver = postgres_pool
    driver.rows = [
        {"id": -1, "metadata_json": '{"region":"invalid"}', "revoked_at": None},
        {"id": 0, "metadata_json": '{"region":"invalid"}', "revoked_at": None},
        {"id": 7, "metadata_json": '{"region":"local"}', "revoked_at": None},
        {"id": 11, "metadata_json": '{"region":"remote"}', "revoked_at": datetime(2026, 1, 1)},
        {"id": 19, "metadata_json": '{"region":"unrelated"}', "revoked_at": None},
    ]

    result = await ManagedSecretRefsRepo(pool).list_refs_by_ids(ids, include_revoked=include_revoked)

    expected_metadata = {7: {"region": "local"}, 11: {"region": "remote"}}
    assert {ref_id: ref["metadata"] for ref_id, ref in result.items()} == {
        ref_id: expected_metadata[ref_id] for ref_id in expected_ids if include_revoked or ref_id != 11
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("repo_type", [AuthnzUsersRepo, AuthnzOrgsTeamsRepo])
@pytest.mark.parametrize(
    ("ids", "expected_ids"),
    [(None, [19, 11, 7]), ([], []), ([7], [7]), ([7, 11], [11, 7])],
)
async def test_scoped_listing_count_matches_selected_rows(
    postgres_pool: tuple[DatabasePool, _ArrayQueryDriver],
    repo_type: type[AuthnzUsersRepo] | type[AuthnzOrgsTeamsRepo],
    ids: list[int] | None,
    expected_ids: list[int],
) -> None:
    """The count and paginated rows apply the same optional organization array."""
    pool, driver = postgres_pool
    driver.rows = [{"id": member_id, "is_active": True} for member_id in [19, 11, 7]]
    repo = repo_type(pool)

    if isinstance(repo, AuthnzUsersRepo):
        rows, total = await repo.list_users(offset=0, limit=2, org_ids=ids)
    else:
        rows, total = await repo.list_organizations(offset=0, limit=2, org_ids=ids, with_total=True)

    assert ([row["id"] for row in rows], total) == (expected_ids[:2], len(expected_ids))


@pytest.mark.asyncio
# Allow Hypothesis's one-time scan of imported module constants in large suites.
@settings(deadline=1000)
@given(st.lists(st.integers(min_value=1, max_value=2**31 - 1), max_size=100))
async def test_public_fetchone_round_trips_generated_integer_arrays(ids: list[int]) -> None:
    """Explicit array parameters retain every value through the public pool API."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        pool, _driver = _make_postgres_pool(monkeypatch)

        assert await pool.fetchone("SELECT $1::int[] AS ids", (ids,)) == {"ids": ids}


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [(7, "active"), ([7, "active"],), ((7, "active"),)])
async def test_pool_retains_variadic_and_sequence_argument_compatibility(
    postgres_pool: tuple[DatabasePool, _ArrayQueryDriver],
    args: tuple[int | str | list[int | str] | tuple[int | str, ...], ...],
) -> None:
    """Variadic arguments and complete parameter sequences select the same session."""
    pool, driver = postgres_pool
    driver.rows = [{"id": 7, "status": "active"}, {"id": 11, "status": "inactive"}]

    assert await pool.fetchone("SELECT id, status FROM sessions WHERE id = $1 AND status = $2", *args) == {
        "id": 7,
        "status": "active",
    }
