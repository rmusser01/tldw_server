"""Storage admission after normal bootstrap on an official, restricted PG fixture."""

from __future__ import annotations

import secrets
from urllib.parse import quote
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import (
    _run_runtime,
    _runtime_env,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def restricted_database(pg_temp_db):
    """Own no database: give one official scratch DB a temporary restricted owner."""
    role = f"quota_test_{uuid4().hex}"
    password = secrets.token_urlsafe(32)
    with psycopg.connect(str(pg_temp_db["dsn"]), autocommit=True) as admin:
        admin.execute(sql.SQL(
            "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT "
            "NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD {}"
        ).format(sql.Identifier(role), sql.Literal(password)))
        try:
            admin.execute(sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(str(pg_temp_db["database"])), sql.Identifier(role),
            ))
            yield (
                f"postgresql://{quote(role)}:{quote(password)}@"
                f"{pg_temp_db['host']}:{pg_temp_db['port']}/{pg_temp_db['database']}"
            )
        finally:
            # Only this official scratch DB is connected; never DROP OWNED elsewhere.
            admin.execute(sql.SQL("REASSIGN OWNED BY {} TO {}").format(
                sql.Identifier(role), sql.Identifier(str(pg_temp_db["user"])),
            ))
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


RUNTIME_SCRIPT = r'''
import asyncio
import json
import os
from types import SimpleNamespace
import asyncpg
from fastapi import HTTPException, Request, Response
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.testing import is_test_mode, is_explicit_pytest_runtime

async def run():
    stage = 'runtime'
    try:
        assert not is_test_mode() and not is_explicit_pytest_runtime()
        stage = 'first_bootstrap'
        assert await setup_database()
        stage = 'repeated_bootstrap'
        assert await setup_database()
        stage = 'restricted_role_and_parent_seed'
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            flags = await conn.fetchrow("""
                SELECT current_user = session_user AS direct_login,
                    rolcanlogin, rolsuper, rolbypassrls, rolinherit, rolcreatedb,
                    rolcreaterole, rolreplication,
                    (SELECT count(*) FROM pg_auth_members WHERE member=r.oid) AS memberships
                FROM pg_roles r WHERE rolname=current_user
            """)
            assert flags['direct_login'] and flags['rolcanlogin']
            assert not any(flags[k] for k in (
                'rolsuper', 'rolbypassrls', 'rolinherit', 'rolcreatedb',
                'rolcreaterole', 'rolreplication', 'memberships'))
            version = await conn.fetchval("SELECT current_setting('server_version')")
            org = await conn.fetchval("INSERT INTO organizations (name) VALUES ('Quota fixture') RETURNING id")
            team = await conn.fetchval("INSERT INTO teams (org_id, name) VALUES ($1, 'Quota team') RETURNING id", org)
        repo = AuthnzStorageQuotasRepo(pool)
        stage = 'absent_quota_read'
        assert await repo.get_org_quota(org) is None
        assert await repo.get_team_quota(team) is None
        assert (await repo.check_quota_status(org_id=org))['has_quota'] is False
        if os.environ['QUOTA_SCENARIO'] == 'bootstrap':
            return {'ok': True, 'scenario': 'bootstrap', 'version': version}

        stage = 'quota_crud'
        first = await repo.upsert_org_quota(org)
        assert first['quota_mb'] == 10240 and first['used_mb'] == 0
        assert first['soft_limit_pct'] == 80 and first['hard_limit_pct'] == 100
        assert first['created_at'] is not None and first['updated_at'] is not None
        second = await repo.upsert_org_quota(org, quota_mb=10)
        assert second['id'] == first['id'] and second['quota_mb'] == 10
        first_team = await repo.upsert_team_quota(team)
        assert first_team['quota_mb'] == 5120
        second_team = await repo.upsert_team_quota(team, quota_mb=20)
        assert second_team['id'] == first_team['id'] and second_team['quota_mb'] == 20
        await repo.update_org_used_mb(org, 1.25)
        assert await repo.increment_org_used_mb(org, 0.125) == 1.375
        await repo.update_team_used_mb(team, 2.5)
        assert await repo.increment_team_used_mb(team, 0.125) == 2.625
        assert (await repo.get_org_quota(org))['used_mb'] == 1.375
        assert (await repo.get_team_quota(team))['used_mb'] == 2.625
        assert (await repo.check_quota_status(org_id=org))['remaining_mb'] == 8.625
        # Normal repeated bootstrap must preserve operator quotas and usage.
        assert await setup_database()
        assert (await repo.get_org_quota(org))['used_mb'] == 1.375
        assert (await repo.get_team_quota(team))['quota_mb'] == 20

        if os.environ['QUOTA_SCENARIO'] == 'constraints':
            stage = 'schema_constraints'
            async with pool.acquire() as conn:
                for query, args, expected in [
                    ('INSERT INTO storage_quotas (org_id, team_id) VALUES ($1, $2)', (org,team), asyncpg.CheckViolationError),
                    ('INSERT INTO storage_quotas (org_id) VALUES ($1)', (org,), asyncpg.UniqueViolationError),
                    ('INSERT INTO storage_quotas (team_id) VALUES ($1)', (team,), asyncpg.UniqueViolationError),
                    ('INSERT INTO storage_quotas (org_id) VALUES (-9999)', (), asyncpg.ForeignKeyViolationError),
                    ('INSERT INTO storage_quotas (team_id) VALUES (-9999)', (), asyncpg.ForeignKeyViolationError),
                ]:
                    try:
                        await conn.execute(query, *args)
                    except expected:
                        pass
                    else:
                        raise AssertionError('Missing storage quota constraint')
                # Preserve the existing SQLite schema's both-null compatibility.
                await conn.execute('INSERT INTO storage_quotas DEFAULT VALUES')
                await conn.execute('DELETE FROM organizations WHERE id=$1', org)
                assert await conn.fetchval('SELECT count(*) FROM storage_quotas WHERE org_id=$1 OR team_id=$2', org,team) == 0
                assert await conn.fetchval('SELECT count(*) FROM storage_quotas WHERE org_id IS NULL AND team_id IS NULL') == 1
            return {'ok': True, 'scenario': 'constraints', 'version': version}

        stage = 'actual_admission'
        async def admit(size=1024):
            request = Request({'type': 'http', 'method': 'POST', 'path': '/api/v1/media/ingest/jobs',
                'headers': [(b'content-length', str(size).encode())]})
            request.state.org_id = org
            response = Response()
            await guard_storage_quota(request, response, SimpleNamespace(id=2, active_org_id=org))
            return response

        await repo.delete_org_quota(org)
        await admit()  # Existing table + absent quota is truly unlimited.
        await repo.upsert_org_quota(org, quota_mb=10)
        await admit()  # Under quota.
        await repo.update_org_used_mb(org, 8.5)
        warning = await admit()
        assert 'soft limit' in warning.headers['X-Storage-Warning']
        for used, size in [(10.0, 1), (9.5, 1024*1024)]:
            await repo.update_org_used_mb(org, used)
            try:
                await admit(size)
            except HTTPException as exc:
                assert exc.status_code == 413 and exc.detail['error'] == 'storage_quota_exceeded'
                assert exc.detail['quota_mb'] == 10 and exc.detail['used_mb'] == used
            else:
                raise AssertionError('Over-limit admission allowed')
        await repo.delete_org_quota(org)
        await repo.delete_team_quota(team)
        assert await repo.get_org_quota(org) is None and await repo.get_team_quota(team) is None
        stage = 'actual_backend_failure'
        async with pool.acquire() as conn:
            await conn.execute('ALTER TABLE storage_quotas RENAME TO quota_fixture_unavailable')
        try:
            try:
                await admit()
            except HTTPException as exc:
                assert exc.status_code == 413
                assert exc.detail == {'error': 'storage_quota_exceeded', 'message': 'Quota check unavailable',
                    'used_mb': 0.0, 'quota_mb': None, 'remaining_mb': None}
            else:
                raise AssertionError('Unavailable quota backend admitted upload')
        finally:
            async with pool.acquire() as conn:
                await conn.execute('ALTER TABLE quota_fixture_unavailable RENAME TO storage_quotas')
        await admit()  # Recovery uses the same actual pool and guard.
        return {'ok': True, 'scenario': 'admission', 'version': version}
    except Exception as exc:
        # No raw startup output, SQL parameters, credentials or DSNs in assertions.
        import traceback
        return {'ok': False, 'stage': stage, 'error': type(exc).__name__,
            'frames': [(frame.name, frame.lineno) for frame in traceback.extract_tb(exc.__traceback__)[-6:]]}
    finally:
        await reset_db_pool()

print('RUNTIME_RESULT=' + json.dumps(asyncio.run(run())))
'''


@pytest.mark.parametrize('mode', ['multi_user', 'single_user'])
def test_normal_fresh_and_repeated_bootstrap_creates_storage_quotas(restricted_database, tmp_path, mode):
    """Normal initialization, with no test exemption or manually supplied quota DDL."""
    env = _runtime_env(tmp_path, restricted_database, backend='postgresql', mode=mode)
    env['QUOTA_SCENARIO'] = 'bootstrap'
    result = _run_runtime(tmp_path, env, RUNTIME_SCRIPT)
    assert result.get('ok') is True, result
    assert result['scenario'] == 'bootstrap'


@pytest.mark.parametrize('scenario', ['constraints', 'admission'])
def test_restricted_storage_quota_crud_constraints_and_admission(restricted_database, tmp_path, scenario):
    """Actual restricted login/repository/guard, with no mocked quota responses."""
    env = _runtime_env(tmp_path, restricted_database, backend='postgresql', mode='multi_user')
    env['QUOTA_SCENARIO'] = scenario
    result = _run_runtime(tmp_path, env, RUNTIME_SCRIPT)
    assert result.get('ok') is True, result
    assert result['scenario'] == scenario
