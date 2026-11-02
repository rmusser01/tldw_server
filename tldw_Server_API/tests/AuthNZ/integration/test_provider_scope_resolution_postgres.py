"""Backend parity through the canonical isolated AuthNZ PostgreSQL fixture."""

import pytest

from tldw_Server_API.tests.AuthNZ_SQLite.test_provider_scope_resolution_sqlite import (
    exercise_scope_case,
    scope_cases,
    seed_scope_state,
    single_scope_cases,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def scope_postgres_pool(isolated_test_environment):
    """Own the test-loop pool independently of the fixture's app lifespan pool."""
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo

    _client, _db_name = isolated_test_environment
    pool = DatabasePool()
    await pool.initialize()
    try:
        assert pool.pool is not None
        await AuthnzUserProviderSecretsRepo(pool).ensure_tables()
        await AuthnzOrgProviderSecretsRepo(pool).ensure_tables()
        yield pool
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "team", "org"])
async def test_authoritative_scope_matrix_postgres(scope_postgres_pool, kind):
    pool = scope_postgres_pool
    for case, with_secret, expected in scope_cases(kind):
        state = await seed_scope_state(pool)
        await exercise_scope_case(state, kind, case, with_secret, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["team", "org"])
async def test_exact_single_scope_matrix_postgres(scope_postgres_pool, kind):
    pool = scope_postgres_pool
    for case, with_secret, expected in single_scope_cases(kind):
        state = await seed_scope_state(pool)
        await exercise_scope_case(state, kind, case, with_secret, expected, single_scope=True)
