"""Verify source identity and bound quota locks with native AuthNZ repositories."""

from pathlib import Path

import pytest

from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import _run_runtime
from tldw_Server_API.tests.AuthNZ.integration.test_vn_generated_file_idempotency import _vn_runtime_env

IDENTITY_SCRIPT = r'''
import asyncio
import json
import uuid

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo


async def main() -> None:
    """Observe native owned lookup, live selection and registration identities."""
    try:
        assert await setup_database()
        pool = await get_db_pool()
        users = AuthnzUsersRepo(pool)
        owner, foreign = [await users.create_user(
            username=name, email=name + '@example.invalid', password_hash=str(uuid.uuid4()),
        ) for name in ('owner', 'foreign')]
        repo = AuthnzGeneratedFilesRepo(pool)

        async def register(user: int, feature: str, ref: str, name: str) -> dict[str, object]:
            """Register a distinct candidate through the public repository."""
            return await repo.create_file(
                user_id=user, filename=name + '.png', storage_path='images/' + name + '.png',
                file_category='image', source_feature=feature, source_ref=ref,
                file_size_bytes=7, tags=['original'],
            )

        async def lookup(user: int, feature: str, ref: str) -> dict[str, object] | None:
            """Read the committed public record for exactly the requested scope."""
            return await repo.get_file_by_source_ref(
                user_id=user, source_feature=feature, source_ref=ref,
            )

        ref = 'vn_asset_item:42'
        winning = await register(owner, 'vn_assets', ref, 'winning')
        foreign_row = await register(foreign, 'vn_assets', ref, 'foreign')
        feature_row = await register(owner, 'image_gen', ref, 'feature')
        owned = await lookup(owner, 'vn_assets', ref)
        result = {'isolated': (
            owned is not None and owned['id'] == winning['id']
            and owned['user_id'] == owner and owned['source_feature'] == 'vn_assets'
            and foreign_row['user_id'] == foreign and foreign_row['id'] != winning['id']
            and feature_row['user_id'] == owner and feature_row['source_feature'] == 'image_gen'
            and feature_row['id'] not in (winning['id'], foreign_row['id'])
            and (await lookup(foreign, 'vn_assets', ref))['id'] == foreign_row['id']
            and (await lookup(owner, 'image_gen', ref))['id'] == feature_row['id']
            and await lookup(owner, 'vn_assets', 'vn_asset_item:99') is None
        )}
        replay = await register(owner, 'vn_assets', ref, 'loser')
        result['canonical_replay'] = replay.get('_idempotent_replay') is True
        result['current_identity'] = all(replay[field] == winning[field] for field in (
            'id', 'uuid', 'user_id', 'source_feature', 'source_ref', 'filename',
            'storage_path', 'file_size_bytes', 'tags', 'is_deleted',
        ))
        assert await repo.soft_delete_file(winning['id'])
        replacement = await register(owner, 'vn_assets', ref, 'replacement')
        result['deleted_not_replayed'] = (
            replacement['id'] != winning['id'] and not replacement.get('_idempotent_replay')
            and (await lookup(owner, 'vn_assets', ref))['id'] == replacement['id']
        )

        older = await register(owner, 'vn_assets', 'historical-ref', 'older')
        newer = await register(owner, 'vn_assets', 'historical-ref', 'newer')
        result['newest_live'] = (await lookup(owner, 'vn_assets', 'historical-ref'))['id'] == newer['id']
        assert await repo.soft_delete_file(newer['id'])
        result['live_fallback'] = (await lookup(owner, 'vn_assets', 'historical-ref'))['id'] == older['id']

        result['noncanonical_distinct'] = True
        for feature, noncanonical in (
            ('vn_assets', 'vn_asset_item:'), ('vn_assets', 'vn_asset_item:abc'),
            ('vn_assets', 'vn_asset_item:42:variant'), ('vn_assets', 'vn_asset_item:0'),
            ('vn_assets', 'vn_asset_item:-1'), ('vn_assets', 'vn_asset_item:0042'),
            ('vn_assets', 'vn_asset_item:42\n'), ('image_gen', ref),
        ):
            first = await register(owner, feature, noncanonical, 'first')
            second = await register(owner, feature, noncanonical, 'second')
            stored_first = await repo.get_file_by_id(first['id'])
            stored_second = await repo.get_file_by_id(second['id'])
            result['noncanonical_distinct'] &= (
                first['id'] != second['id'] and not first.get('_idempotent_replay')
                and not second.get('_idempotent_replay')
                and stored_first['filename'] == 'first.png' and stored_second['filename'] == 'second.png'
                and (await lookup(owner, feature, noncanonical))['id'] == second['id']
            )
        print('RUNTIME_RESULT=' + json.dumps(result))
    finally:
        await reset_db_pool()

asyncio.run(main())
'''

LOCK_SCRIPT = r'''
import asyncio
import json
import os
import uuid

import asyncpg

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo


async def main() -> None:
    """Observe native blocking until the owning VN admission transaction commits."""
    try:
        assert await setup_database()
        pool = await get_db_pool()
        assert pool.backend_type == 'postgres'
        users = AuthnzUsersRepo(pool)
        owner, other = [await users.create_user(
            username=name, email=name + '@example.invalid', password_hash=str(uuid.uuid4()),
        ) for name in ('owner', 'other')]
        # Fixture data only; schema/cluster creation belongs to the official fixture.
        await pool.execute('INSERT INTO organizations (id, name) VALUES (?, ?)', 51, 'lock-org')
        await pool.execute('INSERT INTO teams (id, org_id, name) VALUES (?, ?, ?)', 52, 51, 'lock-team')
        quotas = AuthnzStorageQuotasRepo(pool)
        await quotas.upsert_org_quota(51, quota_mb=100)
        await quotas.upsert_team_quota(52, quota_mb=100)
        repo = AuthnzGeneratedFilesRepo(pool)
        scope = os.environ['TASK57_LOCK_SCOPE']
        target = {'user_id': owner if scope == 'user' else other,
                  'org_id': 51 if scope == 'org' else None,
                  'team_id': 52 if scope == 'team' else None}

        async def contend(
            arguments: dict[str, int | None], source_ref: str = 'vn_asset_item:42',
        ) -> bool:
            """Return whether independent native admission encounters a held lock."""
            blocked = False
            try:
                async with repo.vn_item_transaction(user_id=other, source_ref='vn_asset_item:99') as contender:
                    async with contender.db_pool.acquire() as conn:
                        await conn.execute("SET LOCAL lock_timeout = '150ms'")
                        try:
                            if scope == 'item':
                                async with contender.vn_item_transaction(
                                    user_id=arguments['user_id'], source_ref=source_ref,
                                ):
                                    pass
                            else:
                                await contender.lock_quota_scopes(**arguments)
                        except asyncpg.LockNotAvailableError:
                            blocked = True
                            raise
            except TransactionError:
                if not blocked:
                    raise
            return blocked

        async with repo.vn_item_transaction(user_id=owner, source_ref='vn_asset_item:42') as holder:
            await holder.lock_quota_scopes(user_id=owner, org_id=51, team_id=52)
            unrelated_allowed = not await contend({'user_id': other, 'org_id': None, 'team_id': None})
            if scope == 'item':
                target['user_id'] = owner
                unrelated_allowed &= not await contend(target, source_ref='vn_asset_item:43')
            blocked = await contend(target)
        released = not await contend(target)
        print('RUNTIME_RESULT=' + json.dumps({
            'blocked': blocked, 'unrelated_allowed': unrelated_allowed, 'released': released,
        }))
    finally:
        await reset_db_pool()

asyncio.run(main())
'''


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_native_source_identity_and_canonical_registration(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    """Preserve owner/feature isolation, live identity and canonical-only replay.

    Args:
        tmp_path: Private native-runtime directory provided by pytest.
        request: Obtain the official isolated PostgreSQL fixture when selected.
        backend: Native SQLite or PostgreSQL backend, never a SQL-parser fake.

    Returns:
        None. Assert committed identities, deleted-row exclusion, newest live
        selection and two persisted records for each noncanonical reference.
    """
    env = _vn_runtime_env(tmp_path, request, backend)
    env["AUTH_MODE"] = "multi_user"
    assert _run_runtime(tmp_path, env, IDENTITY_SCRIPT) == {
        "isolated": True, "canonical_replay": True, "current_identity": True,
        "deleted_not_replayed": True, "newest_live": True, "live_fallback": True,
        "noncanonical_distinct": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["user", "org", "team"])
def test_native_postgres_quota_locks_last_until_bound_transaction_commits(
    tmp_path: Path, request: pytest.FixtureRequest, scope: str,
) -> None:
    """Block competing admission in each quota scope but release after commit.

    Args:
        tmp_path: Private subprocess directory with no shared configuration.
        request: Borrow the official isolated_test_environment database only.
        scope: User, organization or team quota row held by native admission.

    Returns:
        None. Assert native lock-timeout rejection, unrelated-owner admission,
        and successful admission after the bound transaction releases its locks.
    """
    env = _vn_runtime_env(tmp_path, request, "postgres")
    env.update(AUTH_MODE="multi_user", TASK57_LOCK_SCOPE=scope)
    assert _run_runtime(tmp_path, env, LOCK_SCRIPT) == {
        "blocked": True, "unrelated_allowed": True, "released": True,
    }


@pytest.mark.integration
def test_native_postgres_item_lock_is_owned_source_scoped_until_commit(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """Serialize the same owned item while admitting foreign and sibling items.

    Args:
        tmp_path: Private runtime directory for the native transaction probe.
        request: Borrow the existing official isolated PostgreSQL test database.

    Returns:
        None. Assert same-owner/source lock-timeout rejection, unrelated owner
        and sibling-item admission, and release after the owning commit.
    """
    env = _vn_runtime_env(tmp_path, request, "postgres")
    env.update(AUTH_MODE="multi_user", TASK57_LOCK_SCOPE="item")
    assert _run_runtime(tmp_path, env, LOCK_SCRIPT) == {
        "blocked": True, "unrelated_allowed": True, "released": True,
    }
