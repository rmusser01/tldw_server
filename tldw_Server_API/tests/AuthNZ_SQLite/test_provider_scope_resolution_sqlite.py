"""Authoritative scope matrix, shared with the canonical PostgreSQL tests."""

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import dollar
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import ProviderCredentialAliasConflictError


def scope_cases(kind):
    """Use exactly the same authorization cases on both storage backends."""
    cases = [
        ("valid", True, "resolved"),
        ("absent", False, "authorized_absent"),
        ("revoked", True, "unauthorized"),
        ("canonical_precedence", True, "resolved"),
        ("revoked_canonical", True, "unauthorized"),
        ("legacy", True, "resolved"),
        ("legacy_conflict", True, "conflict"),
    ]
    denied = ["inactive_user", "missing_user"]
    if kind != "user":
        denied += [
            "inactive_org_member",
            "missing_org_member",
            "inactive_team_member",
            "missing_team_member",
            "inactive_org",
            "missing_org",
            "inactive_team",
            "missing_team",
            "mismatched_team_id",
            "mismatched_org_id",
            "wrong_relationship",
        ]
    return (
        cases
        + [(name, secret, "unauthorized") for name in denied for secret in (False, True)]
        + [("unavailable", False, "unavailable")]
    )


def single_scope_cases(kind):
    """An omitted counterpart never introduces an extra membership scope."""
    if kind == "team":
        independent = ["inactive_org_member", "missing_org_member"]
        denied = ["inactive_org", "missing_org"]
    else:
        independent = ["inactive_team_member", "missing_team_member", "inactive_team", "missing_team"]
        denied = []
    return [
        (case, secret, "resolved" if secret else "authorized_absent")
        for case in independent
        for secret in (False, True)
    ] + [(case, secret, "unauthorized") for case in denied for secret in (False, True)]


async def _execute(pool, sql, *params):
    if pool.pool is not None:
        return await pool.execute(dollar(sql), *params)
    return await pool.execute(sql, params)


class _UnavailablePool:
    """Inject a real driver read error without modifying the fixture schema."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.pool = delegate.pool

    async def fetchall(self, *_args):
        return await self.delegate.fetchall("SELECT provider_scope_unavailable_column FROM users")


async def _insert_id(pool, sql, *params):
    async with pool.transaction() as conn:
        if pool.pool is not None:
            row = await conn.fetchrow(dollar(sql), *params)
        else:
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
    return int(row["id"])


async def seed_scope_state(pool):
    """Seed fresh exact scopes using the fixture's already-migrated database."""
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    token = uuid.uuid4()
    user = await UsersDB(pool).create_user(
        username=f"scope-{token}",
        email=f"scope-{token}@example.test",
        password_hash="test-hash",
        is_active=True,
        is_verified=True,
        uuid_value=token,
    )
    user_id = int(user["id"])
    org_id = await _insert_id(pool, "INSERT INTO organizations (name) VALUES (?) RETURNING id", f"org-{token}")
    other_org_id = await _insert_id(pool, "INSERT INTO organizations (name) VALUES (?) RETURNING id", f"other-{token}")
    team_id = await _insert_id(
        pool, "INSERT INTO teams (org_id, name) VALUES (?, ?) RETURNING id", org_id, f"team-{token}"
    )
    other_team_id = await _insert_id(
        pool, "INSERT INTO teams (org_id, name) VALUES (?, ?) RETURNING id", other_org_id, f"other-team-{token}"
    )
    await _execute(pool, "INSERT INTO org_members (org_id, user_id, status) VALUES (?, ?, 'active')", org_id, user_id)
    await _execute(
        pool, "INSERT INTO team_members (team_id, user_id, status) VALUES (?, ?, 'active')", team_id, user_id
    )
    return {
        "pool": pool,
        "user": user_id,
        "org": org_id,
        "team": team_id,
        "other_org": other_org_id,
        "other_team": other_team_id,
    }


async def _insert_secret(state, kind, provider, *, revoked=False):
    pool = state["pool"]
    timestamp = datetime.now(timezone.utc)
    timestamp = timestamp.replace(tzinfo=None) if pool.pool is not None else timestamp.isoformat()
    if kind == "user":
        sql = """INSERT INTO user_provider_secrets
                 (user_id, provider, encrypted_blob, created_at, updated_at, revoked_at)
                 VALUES (?, ?, ?, ?, ?, ?)"""
        params = (state["user"], provider, f"private-{provider}", timestamp, timestamp, timestamp if revoked else None)
    else:
        sql = """INSERT INTO org_provider_secrets
                 (scope_type, scope_id, provider, encrypted_blob, created_at, updated_at, revoked_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        params = (
            kind,
            state[kind],
            provider,
            f"private-{provider}",
            timestamp,
            timestamp,
            timestamp if revoked else None,
        )
    await _execute(pool, sql, *params)


async def exercise_scope_case(state, kind, case, with_secret, expected, *, single_scope=False):
    """Apply one matrix mutation and assert the repository outcome."""
    pool = state["pool"]
    if with_secret:
        if case in {"legacy", "legacy_conflict"}:
            await _insert_secret(state, kind, "openai-compatible")
            if case == "legacy_conflict":
                await _insert_secret(state, kind, "custom-openai")
        else:
            await _insert_secret(state, kind, "custom-openai-api", revoked=case in {"revoked", "revoked_canonical"})
            if case in {"canonical_precedence", "revoked_canonical"}:
                await _insert_secret(state, kind, "openai-compatible")
    mutations = {
        "inactive_org_member": (
            "UPDATE org_members SET status = 'inactive' WHERE org_id = ? AND user_id = ?",
            (state["org"], state["user"]),
        ),
        "missing_org_member": (
            "DELETE FROM org_members WHERE org_id = ? AND user_id = ?",
            (state["org"], state["user"]),
        ),
        "inactive_team_member": (
            "UPDATE team_members SET status = 'inactive' WHERE team_id = ? AND user_id = ?",
            (state["team"], state["user"]),
        ),
        "missing_team_member": (
            "DELETE FROM team_members WHERE team_id = ? AND user_id = ?",
            (state["team"], state["user"]),
        ),
        "inactive_org": ("UPDATE organizations SET is_active = FALSE WHERE id = ?", (state["org"],)),
        "missing_org": ("DELETE FROM organizations WHERE id = ?", (state["org"],)),
        "inactive_team": ("UPDATE teams SET is_active = FALSE WHERE id = ?", (state["team"],)),
        "missing_team": ("DELETE FROM teams WHERE id = ?", (state["team"],)),
        "wrong_relationship": ("UPDATE teams SET org_id = ? WHERE id = ?", (state["other_org"], state["team"])),
    }
    if case in mutations:
        sql, params = mutations[case]
        await _execute(pool, sql, *params)
    if case == "unavailable":
        pool = _UnavailablePool(pool)
    if case == "inactive_user":
        from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

        await UsersDB(pool).update_user(state["user"], is_active=False)
    user_id = state["user"] + 999999999 if case == "missing_user" else state["user"]
    if kind == "user":
        repo = AuthnzUserProviderSecretsRepo(pool)
        args = {"user_id": user_id, "provider": "openai-compatible"}
    else:
        repo = AuthnzOrgProviderSecretsRepo(pool)
        args = {
            "scope_type": kind,
            "scope_id": state[kind],
            "user_id": user_id,
            "provider": "openai-compatible",
            "active_team_id": state["team"],
            "active_organization_id": state["org"],
        }
        if case == "mismatched_team_id":
            args["active_team_id"] = state["other_team"]
        if case == "mismatched_org_id":
            args["active_organization_id"] = state["other_org"]
        if single_scope:
            counterpart = "active_organization_id" if kind == "team" else "active_team_id"
            args.pop(counterpart)
    if expected == "conflict":
        with pytest.raises(ProviderCredentialAliasConflictError):
            await repo.resolve_authorized_secret(**args)
        return
    result = await repo.resolve_authorized_secret(**args)
    assert result.status == expected, case
    if expected == "resolved":
        provider = "openai-compatible" if case == "legacy" else "custom-openai-api"
        assert isinstance(result.record, Mapping)
        assert result.record["provider"] == provider
        assert result.record["encrypted_blob"] == f"private-{provider}"
        assert result.record["revoked_at"] is None
    else:
        assert result.record is None


@pytest.fixture
async def scope_sqlite_pool(tmp_path, monkeypatch):
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import _setup_byok_sqlite

    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    return state["pool"]


MATRIX = [(kind, *case) for kind in ("user", "team", "org") for case in scope_cases(kind)]
SINGLE_SCOPE_MATRIX = [(kind, *case) for kind in ("team", "org") for case in single_scope_cases(kind)]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,case,with_secret,expected", MATRIX)
async def test_authoritative_scope_matrix_sqlite(scope_sqlite_pool, kind, case, with_secret, expected):
    state = await seed_scope_state(scope_sqlite_pool)
    await exercise_scope_case(state, kind, case, with_secret, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,case,with_secret,expected", SINGLE_SCOPE_MATRIX)
async def test_exact_single_scope_matrix_sqlite(scope_sqlite_pool, kind, case, with_secret, expected):
    state = await seed_scope_state(scope_sqlite_pool)
    await exercise_scope_case(state, kind, case, with_secret, expected, single_scope=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["team", "org"])
async def test_shared_scope_without_optional_ids_does_not_infer_other_memberships(scope_sqlite_pool, kind):
    state = await seed_scope_state(scope_sqlite_pool)
    if kind == "team":
        await _execute(
            scope_sqlite_pool, "DELETE FROM org_members WHERE org_id = ? AND user_id = ?", state["org"], state["user"]
        )
    else:
        await _execute(
            scope_sqlite_pool,
            "DELETE FROM team_members WHERE team_id = ? AND user_id = ?",
            state["team"],
            state["user"],
        )
    result = await AuthnzOrgProviderSecretsRepo(scope_sqlite_pool).resolve_authorized_secret(
        kind, state[kind], state["user"], "openai"
    )
    assert result.status == "authorized_absent"
