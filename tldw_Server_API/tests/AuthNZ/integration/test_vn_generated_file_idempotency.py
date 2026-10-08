"""Exercise VN registration races against the real AuthNZ transaction backend."""

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import (
    _run_runtime,
    _runtime_env,
)

REGISTRATION_SCRIPT = r'''
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

async def main() -> None:
    """Exercise concurrent registration and close the runtime database pool."""
    try:
        assert await setup_database()
        await bootstrap_single_user_profile()
        pool = await get_db_pool()
        repo = AuthnzGeneratedFilesRepo(pool)
        owner = get_settings().SINGLE_USER_FIXED_ID
        async def register(index: int) -> dict[str, object]:
            """Register one candidate for the shared owned VN source reference."""
            return await repo.create_file(
                user_id=owner, filename=f'attempt-{index}.png',
                storage_path=f'vn_assets/attempt-{index}.png', file_category='image',
                source_feature='vn_assets', source_ref='vn_asset_item:42', file_size_bytes=3,
            )
        records = await asyncio.gather(*(register(index) for index in range(8)))
        count = await pool.fetchval(
            'SELECT COUNT(*) FROM generated_files WHERE user_id = ? AND source_ref = ? AND is_deleted = ?',
            owner, 'vn_asset_item:42', False,
        )
        print('RUNTIME_RESULT=' + json.dumps({
            'distinct_ids': len({record['id'] for record in records}),
            'live_records': count,
            'replays': sum(bool(record.get('_idempotent_replay')) for record in records),
        }))
    finally:
        await reset_db_pool()

asyncio.run(main())
'''


def _vn_runtime_env(tmp_path: Path, request: pytest.FixtureRequest, backend: str) -> dict[str, str]:
    """Borrow the shared per-test PostgreSQL database, retaining runtime isolation."""
    if backend == "postgres":
        _client, db_name = request.getfixturevalue("isolated_test_environment")
        database_url = os.environ["DATABASE_URL"]
        assert urlparse(database_url).path.lstrip("/") == db_name, "Use the fixture-owned database"
    else:
        database_url = f"sqlite:///{tmp_path / 'users.db'}"
    return _runtime_env(tmp_path, database_url, backend=backend)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_concurrent_vn_registration_converges_on_one_live_file(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    env = _vn_runtime_env(tmp_path, request, backend)

    assert _run_runtime(tmp_path, env, REGISTRATION_SCRIPT) == {
        "distinct_ids": 1, "live_records": 1, "replays": 7,
    }


STORAGE_SCRIPT = r'''
import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError, StorageError, TransactionError
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.Storage import generated_file_helpers as helpers
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
    OrgProfileOverridesRepo, TeamProfileOverridesRepo, reset_schema_verification_cache,
)
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

async def main() -> None:
    """Exercise the selected storage failure or race in an isolated runtime."""
    try:
        assert await setup_database()
        await bootstrap_single_user_profile()
        pool = await get_db_pool()
        owner = get_settings().SINGLE_USER_FIXED_ID
        service = StorageQuotaService(pool)
        await service.initialize()
        quota_mb = get_settings().DEFAULT_STORAGE_QUOTA_MB
        await service.set_user_quota(owner, quota_mb)
        repo = AuthnzGeneratedFilesRepo(pool)
        await pool.execute('INSERT INTO organizations (id, name) VALUES (?, ?)', 51, 'vn-storage-org')
        await pool.execute('INSERT INTO teams (id, org_id, name) VALUES (?, ?, ?)', 52, 51, 'vn-storage-team')
        quotas = AuthnzStorageQuotasRepo(pool)
        await quotas.upsert_org_quota(51, quota_mb=100)
        await quotas.upsert_team_quota(52, quota_mb=100)
        outputs = Path.cwd() / 'outputs'
        helpers.DatabasePaths.get_user_outputs_dir = staticmethod(lambda _user_id: outputs)
        async def get_service() -> StorageQuotaService:
            """Supply the initialized service for generated-image registration."""
            return service
        helpers.get_storage_service = get_service

        async def save(item: int = 42) -> dict[str, object]:
            """Save and register the owned VN image through the real helper."""
            return await helpers.save_and_register_vn_asset_image(
                user_id=owner, pack_id=7, item_id=item, asset_type='sprite',
                image_bytes=b'image', org_id=51, team_id=52,
            )

        async def snapshot() -> dict[str, object]:
            """Capture live files, quota usage and profile version for comparison."""
            user = await pool.fetchone('SELECT storage_used_mb, profile_version FROM users WHERE id = ?', owner)
            return {
                'live': await pool.fetchval('SELECT COUNT(*) FROM generated_files WHERE is_deleted = ?', False),
                'usage': [float(user['storage_used_mb']),
                          float((await quotas.get_org_quota(51))['used_mb']),
                          float((await quotas.get_team_quota(52))['used_mb'])],
                'version': str(user['profile_version']),
            }

        baseline = await snapshot()
        case = os.environ['VN_STORAGE_CASE']
        result = {}
        if case in {'user', 'org', 'team', 'cancel'}:
            method = {'user': 'update_usage', 'org': 'update_org_usage',
                      'team': 'update_team_usage', 'cancel': 'update_team_usage'}[case]
            original = getattr(StorageQuotaService, method)
            async def fail_after_update(self: StorageQuotaService, *args: Any, **kwargs: Any) -> None:
                """Inject failure after the real accounting update to test rollback."""
                await original(self, *args, **kwargs)
                if case == 'cancel':
                    raise asyncio.CancelledError()
                raise RuntimeError('injected accounting failure')
            with patch.object(StorageQuotaService, method, fail_after_update):
                try:
                    await save()
                except (Exception, asyncio.CancelledError):
                    result['failed'] = True
            failed = await snapshot()
            result['rolled_back'] = failed == baseline
            record = await save()
            final = await snapshot()
            result.update(live=final['live'], charged_once=final['usage'] == [5 / 1048576] * 3,
                          version_advanced=final['version'] > baseline['version'],
                          bytes_present=(outputs / record['storage_path']).is_file(),
                          files=len(list(outputs.rglob('*.png'))))
        elif case == 'capacity':
            record = await save()
            # Charge through the normal profile gateway, and fill the shared pools.
            await service.update_usage(owner, int(quota_mb * 1048576))
            await quotas.update_org_used_mb(51, 100)
            await quotas.update_team_used_mb(52, 100)
            before = await snapshot()
            for target in ('helper', 'service'):
                try:
                    replay = await save() if target == 'helper' else await service.register_generated_file(
                        user_id=owner, filename='replacement.png', storage_path='vn_assets/replacement.png',
                        file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                        file_size_bytes=5, org_id=51, team_id=52,
                    )
                    result[target] = replay['id'] == record['id']
                except Exception:
                    result[target] = False
            result['unchanged'] = await snapshot() == before
            result['files'] = len(list(outputs.rglob('*.png')))
        elif case == 'capacity_race':
            await service.update_usage(owner, int(quota_mb * 1048576) - 5)
            await quotas.update_org_used_mb(51, 100 - 5 / 1048576)
            await quotas.update_team_used_mb(52, 100 - 5 / 1048576)
            original = StorageQuotaService.get_vn_generated_file
            raced = False
            winner = None
            async def commit_after_empty_lookup(
                self: StorageQuotaService, **kwargs: Any,
            ) -> dict[str, object] | None:
                """Commit a competing registration after the initial empty lookup."""
                nonlocal raced, winner
                existing = await original(self, **kwargs)
                if self is service and existing is None and not raced:
                    raced = True
                    winner = await save()
                return existing
            with patch.object(StorageQuotaService, 'get_vn_generated_file', commit_after_empty_lookup):
                try:
                    replay = await save()
                    result['replayed'] = replay['id'] == winner['id']
                except Exception:
                    result['replayed'] = False
            result['live'] = (await snapshot())['live']
            result['files'] = len(list(outputs.rglob('*.png')))
        elif case == 'missing':
            record = await save()
            (outputs / record['storage_path']).unlink()
            before = await snapshot()
            for target in ('helper', 'service'):
                try:
                    if target == 'helper':
                        await save()
                    else:
                        await service.register_generated_file(
                            user_id=owner, filename='replacement.png', storage_path='vn_assets/replacement.png',
                            file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                            file_size_bytes=5,
                        )
                except Exception:
                    result[target] = 'rejected'
                else:
                    result[target] = 'accepted'
            result['unchanged'] = await snapshot() == before
        elif case == 'missing_race':
            original = StorageQuotaService.register_generated_file
            candidate: Path | None = None
            winner: dict[str, Any] | None = None
            winner_state: dict[str, object] | None = None
            async def register_missing_winner(
                self: StorageQuotaService, **kwargs: Any,
            ) -> dict[str, Any]:
                """Commit a missing winner at public registration after real candidate storage."""
                nonlocal candidate, winner, winner_state
                candidate = outputs / kwargs['storage_path']
                assert candidate.read_bytes() == b'image'
                winner = await original(
                    self, user_id=owner, filename='missing.png', storage_path='vn_assets/missing.png',
                    file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                    file_size_bytes=5, org_id=51, team_id=52,
                )
                winner_state = await snapshot()
                return await original(self, **kwargs)
            with patch.object(StorageQuotaService, 'register_generated_file', register_missing_winner):
                try:
                    await save()
                except StorageError:
                    result['rejected'] = True
                else:
                    result['rejected'] = False
            assert candidate is not None and winner is not None and winner_state is not None
            current = await repo.get_file_by_source_ref(
                user_id=owner, source_feature='vn_assets', source_ref='vn_asset_item:42',
            )
            final = await snapshot()
            try:
                await service.get_vn_generated_file(user_id=owner, source_ref='vn_asset_item:42')
            except StorageError:
                result['missing_winner_rejected'] = True
            else:
                result['missing_winner_rejected'] = False
            result.update(
                live=final['live'],
                winner_identity_preserved=(
                    current is not None and current['id'] == winner['id']
                    and current['user_id'] == owner and current['org_id'] == 51 and current['team_id'] == 52
                    and current['source_feature'] == 'vn_assets' and current['source_ref'] == 'vn_asset_item:42'
                    and current['filename'] == 'missing.png' and current['storage_path'] == 'vn_assets/missing.png'
                    and current['file_size_bytes'] == 5 and not current['is_deleted']
                ),
                winner_charged_once=winner_state['usage'] == [5 / 1048576] * 3,
                version_advanced=winner_state['version'] > baseline['version'],
                loser_rolled_back=final == winner_state,
                missing_winner_bytes=not (outputs / winner['storage_path']).exists(),
                candidate_unregistered=await repo.get_live_file_by_storage_path(
                    user_id=owner, storage_path=str(candidate.relative_to(outputs)),
                ) is None,
                replacement_preserved=candidate.is_file() and candidate.read_bytes() == b'image',
                files=len(list(outputs.rglob('*.png'))),
            )
        elif case == 'missing_record':
            original = AuthnzGeneratedFilesRepo.create_file
            async def lose_created_record(self: AuthnzGeneratedFilesRepo, **kwargs: Any) -> dict[str, object]:
                """Discard the created record response to test transaction rollback."""
                await original(self, **kwargs)
                return {}
            with patch.object(AuthnzGeneratedFilesRepo, 'create_file', lose_created_record):
                try:
                    await save()
                except Exception:
                    result['rejected'] = True
                else:
                    result['rejected'] = False
            result['rolled_back'] = await snapshot() == baseline
        elif case == 'postcommit':
            transaction = pool.transaction
            @contextlib.asynccontextmanager
            async def fail_after_commit(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                """Raise after the real transaction commits without deleting live bytes."""
                async with transaction(*args, **kwargs) as conn:
                    yield conn
                raise RuntimeError('injected response failure after commit')
            with patch.object(pool, 'transaction', fail_after_commit):
                try:
                    await save()
                except Exception:
                    result['failed'] = True
            record = await repo.get_file_by_source_ref(
                user_id=owner, source_feature='vn_assets', source_ref='vn_asset_item:42',
            )
            result['committed_bytes_preserved'] = (outputs / record['storage_path']).is_file()
        elif case == 'postcommit_cache':
            await service.check_quota(owner, 0)
            service.storage_cache[f'storage_calc:{owner}'] = {'total_mb': 0}
            transaction = pool.transaction
            cancelled = asyncio.CancelledError('after committed registration')
            @contextlib.asynccontextmanager
            async def cancel_after_commit(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                """Cancel the response only after the actual transaction commits."""
                async with transaction(*args, **kwargs) as conn:
                    yield conn
                raise cancelled
            with patch.object(pool, 'transaction', cancel_after_commit):
                try:
                    await save()
                except asyncio.CancelledError as exc:
                    result['cancellation_preserved'] = exc is cancelled
            committed = await snapshot()
            record = await repo.get_file_by_source_ref(
                user_id=owner, source_feature='vn_assets', source_ref='vn_asset_item:42',
            )
            result['cache_evicted_after_commit'] = (
                f'quota:{owner}' not in service.quota_cache
                and f'storage_calc:{owner}' not in service.storage_cache
            )
            await service.check_quota(owner, 0)
            result['fresh_usage'] = service.quota_cache[f'quota:{owner}'] == 5 / 1048576
            service.quota_cache[f'quota:{owner}'] = 0
            service.storage_cache[f'storage_calc:{owner}'] = {'total_mb': 0}
            replay = await service.register_generated_file(
                user_id=owner, filename='replacement.png', storage_path='vn_assets/replacement.png',
                file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                file_size_bytes=99, org_id=51, team_id=52,
            )
            result.update(
                replay_identity=replay['id'] == record['id'],
                cache_evicted_after_replay=(
                    f'quota:{owner}' not in service.quota_cache
                    and f'storage_calc:{owner}' not in service.storage_cache
                ),
                charged_once=committed['usage'] == [5 / 1048576] * 3,
                replay_unchanged=await snapshot() == committed,
                bytes_preserved=(outputs / record['storage_path']).read_bytes() == b'image',
            )
        elif case.startswith('unregister_'):
            record = await save()
            registered = await snapshot()
            kind = case.removeprefix('unregister_')
            method = {'user': 'update_usage', 'org': 'update_org_usage',
                      'team': 'update_team_usage', 'cancel': 'update_team_usage'}[kind]
            original = getattr(StorageQuotaService, method)
            cancelled = asyncio.CancelledError('during unregistration accounting')
            async def fail_removal(self: StorageQuotaService, *args: Any, **kwargs: Any) -> None:
                """Fail after a real decrement to verify record/accounting rollback."""
                await original(self, *args, **kwargs)
                if kind == 'cancel':
                    raise cancelled
                raise RuntimeError('injected removal accounting failure')
            with patch.object(StorageQuotaService, method, fail_removal):
                try:
                    await service.unregister_generated_file(record['id'], hard_delete=True)
                except asyncio.CancelledError as exc:
                    result['failed'] = exc is cancelled
                except Exception:
                    result['failed'] = kind != 'cancel'
            result['rolled_back'] = await snapshot() == registered
            result['retry_removed'] = await service.unregister_generated_file(record['id'], hard_delete=True)
            final = await snapshot()
            result.update(live=final['live'], usage_restored=final['usage'] == baseline['usage'])
        elif case.startswith('quota_policy_'):
            level = case.removeprefix('quota_policy_')
            if level == 'user':
                await service.update_usage(owner, int(quota_mb * 1048576) - 5)
            elif level == 'org':
                await quotas.update_org_used_mb(51, 100 - 5 / 1048576)
            else:
                await quotas.update_team_used_mb(52, 100 - 5 / 1048576)
            before = await snapshot()
            record = await save()
            published = await snapshot()
            replay = await save()
            replayed = await snapshot()
            quota_denied = False
            check_combined_quota = StorageQuotaService.check_combined_quota
            async def observe_quota_denial(self: StorageQuotaService, *args: Any, **kwargs: Any) -> Any:
                """Observe the real quota rejection before transaction error wrapping."""
                nonlocal quota_denied
                try:
                    return await check_combined_quota(self, *args, **kwargs)
                except QuotaExceededError:
                    quota_denied = True
                    raise
            with patch.object(StorageQuotaService, 'check_combined_quota', observe_quota_denial):
                try:
                    await save(43)
                except (QuotaExceededError, TransactionError):
                    assert quota_denied, 'Unexpected transaction failure is not quota denial'
                    distinct_admitted = False
                else:
                    distinct_admitted = True
            final = await snapshot()
            result = {
                'enabled': usage_quotas_enabled(),
                'first_charged_once': [after - prior for after, prior in zip(published['usage'], before['usage'])]
                                      == [5 / 1048576] * 3,
                'replay_unchanged': replayed == published,
                'identity_preserved': replay['id'] == record['id'] and replay['storage_path'] == record['storage_path'],
                'bytes_preserved': (outputs / record['storage_path']).read_bytes() == b'image',
                'distinct_admitted': distinct_admitted,
                'quota_denial_observed': quota_denied,
                'final_accounting': [after - prior for after, prior in zip(final['usage'], before['usage'])]
                                    == [(10 if distinct_admitted else 5) / 1048576] * 3,
                'live': final['live'], 'files': len(list(outputs.rglob('*.png'))),
            }
        elif case.startswith('distinct_'):
            level = case.removeprefix('distinct_')
            if level == 'user':
                await service.update_usage(owner, int(quota_mb * 1048576) - 5)
            elif level == 'org':
                await quotas.update_org_used_mb(51, 100 - 5 / 1048576)
            else:
                await quotas.update_team_used_mb(52, 100 - 5 / 1048576)
            before = await snapshot()
            outcomes = await asyncio.gather(*(save(item) for item in range(100, 108)), return_exceptions=True)
            final = await snapshot()
            result = {
                'accepted': sum(isinstance(outcome, dict) for outcome in outcomes),
                'live': final['live'], 'files': len(list(outputs.rglob('*.png'))),
                'charged_once': [after - before for after, before in zip(final['usage'], before['usage'])]
                                == [5 / 1048576] * 3,
            }
        elif case.startswith('resolver_'):
            kind = case.removeprefix('resolver_')
            limit = {'none': None, 'zero': 0, 'limited': 2, 'org': 2, 'team': 3, 'user': 4}[kind]
            if kind in {'org', 'team', 'user'}:
                memberships = AuthnzOrgsTeamsRepo(pool)
                await memberships.add_org_member(org_id=51, user_id=owner)
                await memberships.add_team_member(team_id=52, user_id=owner)
                await OrgProfileOverridesRepo(pool).upsert_override(
                    org_id=51, key='limits.storage_quota_mb', value=2, updated_by=owner,
                )
                if kind in {'team', 'user'}:
                    await TeamProfileOverridesRepo(pool).upsert_override(
                        team_id=52, key='limits.storage_quota_mb', value=3, updated_by=owner,
                    )
                quota_resolver.invalidate_all()
            await service.set_user_quota(owner, limit if kind not in {'org', 'team'} else None)
            loads = []
            quota_denied = False
            load_limits = quota_resolver._load_limits
            async def observe_load(user_id: int, **kwargs: Any) -> dict[str, int | float]:
                """Count successful real resolver reads, including schema readiness."""
                limits = await load_limits(user_id, **kwargs)
                loads.append(limits)
                return limits
            register = StorageQuotaService._register_and_account_generated_file
            async def cold_registration(self: StorageQuotaService, *args: Any, **kwargs: Any) -> Any:
                """Resolve cold and warm only after entering the real owning transaction."""
                nonlocal quota_denied
                assert self.db_pool is not pool
                quota_resolver.invalidate_user(owner)
                reset_schema_verification_cache(pool)
                reset_schema_verification_cache(self.db_pool)
                for _ in range(2):
                    actual = await asyncio.wait_for(
                        quota_resolver.user_quota(owner, 'limits.storage_quota_mb', db_pool=self.db_pool), timeout=5,
                    )
                    assert actual == limit
                assert len(loads) == 1, 'Cold lookup must succeed; warm lookup must use the cache'
                try:
                    return await register(self, *args, **kwargs)
                except QuotaExceededError:
                    quota_denied = True
                    raise
            with patch.object(quota_resolver, '_load_limits', observe_load), patch.object(
                StorageQuotaService, '_register_and_account_generated_file', cold_registration,
            ):
                if limit == 0:
                    try:
                        await service.register_generated_file(
                            user_id=owner, filename='blocked.png', storage_path='vn_assets/blocked.png',
                            file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                            file_size_bytes=5, org_id=51, team_id=52,
                        )
                    except (QuotaExceededError, TransactionError):
                        result['denied'] = True
                    else:
                        result['denied'] = False
                    result['unchanged'] = await snapshot() == baseline
                    result['files'] = len(list(outputs.rglob('*.png')))
                else:
                    record = await save()
                    before = await snapshot()
                    replay = await save()
                    result = {
                        'denied': False, 'charged_once': before['usage'] == [5 / 1048576] * 3,
                        'replay_unchanged': await snapshot() == before,
                        'identity_preserved': replay['id'] == record['id'],
                        'bytes_preserved': (outputs / record['storage_path']).read_bytes() == b'image',
                        'live': before['live'], 'files': len(list(outputs.rglob('*.png'))),
                    }
            result['successful_cold_loads'] = len(loads)
            result['quota_denial_observed'] = quota_denied
        elif case.startswith('lookup_error_'):
            from asyncpg.exceptions import InFailedSQLTransactionError, QueryCanceledError
            from tldw_Server_API.app.core.DB_Management._vn_quota_error_test_support import (
                cancel_postgres_quota_read, postgres_connection_usable,
            )
            from tldw_Server_API.app.core.UserProfiles import overrides_repo
            await service.set_user_quota(owner, 0)
            quota_resolver.invalidate_all()
            failures = 0
            fallbacks = []
            usable = []
            register = StorageQuotaService._register_and_account_generated_file
            async def register_with_failed_reads(self: StorageQuotaService, *args: Any, **kwargs: Any) -> Any:
                """Keep actual admission/accounting after failed reads on its owning connection."""
                async with self.db_pool.acquire() as conn:
                    async def canceled_read(*_args: Any, **_kwargs: Any) -> Any:
                        """Inject a real server cancellation at the selected read boundary."""
                        nonlocal failures
                        try:
                            await cancel_postgres_quota_read(conn)
                        except QueryCanceledError:
                            failures += 1
                            raise
                    target = overrides_repo if case == 'lookup_error_readiness' else self.db_pool
                    method = 'validate_postgres_profile_candidate_schema' if target is overrides_repo else 'fetchall'
                    with patch.object(target, method, canceled_read):
                        for _ in range(2):
                            fallbacks.append(await quota_resolver.user_quota(
                                owner, 'limits.storage_quota_mb', db_pool=self.db_pool,
                            ))
                        try:
                            usable.append(await postgres_connection_usable(conn))
                        except InFailedSQLTransactionError:
                            usable.append(False)
                        record = await register(self, *args, **kwargs)
                        usable.append(await postgres_connection_usable(conn))
                    return record
            admitted = False
            with patch.object(StorageQuotaService, '_register_and_account_generated_file', register_with_failed_reads):
                try:
                    await service.register_generated_file(
                        user_id=owner, filename='lookup-error.png', storage_path='vn_assets/lookup-error.png',
                        file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                        file_size_bytes=5, org_id=51, team_id=52,
                    )
                    admitted = True
                except TransactionError:
                    pass
            final = await snapshot()
            result = {
                'actual_cancellations': failures, 'fallbacks': fallbacks,
                'owning_connection_usable': usable, 'admitted': admitted,
                'healthy_limit_after_failures': await quota_resolver.user_quota(owner, 'limits.storage_quota_mb'),
                'live': final['live'], 'charged_once': final['usage'] == [5 / 1048576] * 3,
                'version_advanced': final['version'] > baseline['version'],
            }
        elif case == 'pool_saturation':
            await service.update_usage(owner, int(quota_mb * 1048576) - 5)
            before = await snapshot()
            quota_resolver.invalidate_all()
            reset_schema_verification_cache(pool)
            ready = asyncio.Event()
            arrivals = 0
            exhausted = False
            denials = 0
            loads = 0
            lock = AuthnzGeneratedFilesRepo.lock_quota_scopes
            register = StorageQuotaService._register_and_account_generated_file
            load_limits = quota_resolver._load_limits
            async def synchronize(self: AuthnzGeneratedFilesRepo, **kwargs: Any) -> None:
                """Let all real transactions lease connections before taking the user lock."""
                nonlocal arrivals, exhausted
                arrivals += 1
                if arrivals == 5:
                    exhausted = pool.pool.get_idle_size() == 0 and pool.pool.get_max_size() == 5
                    ready.set()
                await ready.wait()
                await lock(self, **kwargs)
            async def observe_load(*args: Any, **kwargs: Any) -> Any:
                """Count successful real override reads, never substitute a quota."""
                nonlocal loads
                result = await load_limits(*args, **kwargs)
                loads += 1
                return result
            async def observe_denial(self: StorageQuotaService, *args: Any, **kwargs: Any) -> Any:
                """Require a real quota exception, not a generic transaction failure."""
                nonlocal denials
                try:
                    return await register(self, *args, **kwargs)
                except QuotaExceededError:
                    denials += 1
                    raise
            async def candidate(index: int) -> bool:
                """Run real independent VN registrations against the same owner."""
                try:
                    await service.register_generated_file(
                        user_id=owner, filename=f'saturated-{index}.png',
                        storage_path=f'vn_assets/saturated-{index}.png', file_category='image',
                        source_feature='vn_assets', source_ref=f'vn_asset_item:{100 + index}',
                        file_size_bytes=5, org_id=51, team_id=52,
                    )
                    return True
                except (QuotaExceededError, TransactionError):
                    return False
            completed = False
            outcomes = []
            with patch.object(AuthnzGeneratedFilesRepo, 'lock_quota_scopes', synchronize), patch.object(
                StorageQuotaService, '_register_and_account_generated_file', observe_denial,
            ), patch.object(quota_resolver, '_load_limits', observe_load):
                try:
                    outcomes = await asyncio.wait_for(
                        asyncio.gather(*(candidate(index) for index in range(5))), timeout=10,
                    )
                    completed = True
                except TimeoutError:
                    pass
            after = await snapshot()
            result = {
                'completed': completed, 'arrivals': arrivals, 'exhausted': exhausted,
                'successful_loads': loads, 'admitted': sum(outcomes), 'denied': denials,
                'live': after['live'],
                'charged_once': [a - b for a, b in zip(after['usage'], before['usage'])]
                                == [5 / 1048576] * 3,
            }
        elif case == 'concurrent':
            records = await asyncio.gather(*(save() for _ in range(8)))
            final = await snapshot()
            result = {
                'distinct_ids': len({record['id'] for record in records}),
                'live': final['live'], 'charged_once': final['usage'] == [5 / 1048576] * 3,
                'version_advanced': final['version'] > baseline['version'],
                'files': len(list(outputs.rglob('*.png'))),
                'bytes_present': (outputs / records[0]['storage_path']).read_bytes() == b'image',
            }
        print('RUNTIME_RESULT=' + json.dumps(result))
    finally:
        await reset_db_pool()

asyncio.run(main())
'''


def _storage_result(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str, case: str,
    *, quota_policy: str | None = "on",
) -> dict[str, object]:
    """Run against the shared isolated PostgreSQL fixture or a private SQLite file."""
    env = _vn_runtime_env(tmp_path, request, backend)
    env.pop("USAGE_QUOTAS_ENABLED", None)
    env.pop("LIMIT_ENFORCEMENT_ENABLED", None)
    if quota_policy is not None:
        env["USAGE_QUOTAS_ENABLED"] = "true" if quota_policy == "on" else "false"
    env["VN_STORAGE_CASE"] = case
    return _run_runtime(tmp_path, env, STORAGE_SCRIPT)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("case", ["user", "org", "team", "cancel"])
def test_vn_accounting_failure_rolls_back_registration_and_retry_charges_once(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str, case: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, case) == {
        "failed": True, "rolled_back": True, "live": 1, "charged_once": True,
        "version_advanced": True, "bytes_present": True, "files": 1,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_replay_at_capacity_returns_original_without_charge_or_new_bytes(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "capacity") == {
        "helper": True, "service": True, "unchanged": True, "files": 1,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_replay_rejects_missing_registered_bytes(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "missing") == {
        "helper": "rejected", "service": "rejected", "unchanged": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_replay_wins_quota_race_after_an_initial_empty_lookup(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "capacity_race") == {
        "replayed": True, "live": 1, "files": 1,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_registration_error_never_unlinks_committed_live_bytes(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "postcommit") == {
        "failed": True, "committed_bytes_preserved": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_postcommit_cancellation_and_replay_evict_original_usage_cache(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    """Committed accounting remains authoritative despite response cancellation."""
    assert _storage_result(tmp_path, request, backend, "postcommit_cache") == {
        "cancellation_preserved": True, "cache_evicted_after_commit": True,
        "fresh_usage": True, "replay_identity": True, "cache_evicted_after_replay": True,
        "charged_once": True, "replay_unchanged": True, "bytes_preserved": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("case", ["user", "org", "team", "cancel"])
def test_vn_unregister_accounting_failure_retains_retryable_registration(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str, case: str,
) -> None:
    """Removal either commits all decrements or retains the full recovery record."""
    assert _storage_result(tmp_path, request, backend, f"unregister_{case}") == {
        "failed": True, "rolled_back": True, "retry_removed": True,
        "live": 0, "usage_restored": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_missing_replay_bytes_never_discard_a_replacement_write(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    """Reject a missing competing winner without charging or deleting candidate bytes."""
    assert _storage_result(tmp_path, request, backend, "missing_race") == {
        "rejected": True, "missing_winner_rejected": True, "live": 1,
        "winner_identity_preserved": True, "winner_charged_once": True,
        "version_advanced": True, "loser_rolled_back": True, "missing_winner_bytes": True,
        "candidate_unregistered": True, "replacement_preserved": True, "files": 1,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_vn_missing_created_record_cannot_commit_or_return_success(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "missing_record") == {
        "rejected": True, "rolled_back": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_concurrent_vn_helper_registration_charges_once_and_cleans_losers(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "concurrent") == {
        "distinct_ids": 1, "live": 1, "charged_once": True, "version_advanced": True,
        "files": 1, "bytes_present": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("level", ["user", "org", "team"])
def test_distinct_vn_refs_cannot_overallocate_any_quota_scope(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str, level: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, f"distinct_{level}") == {
        "accepted": 1, "live": 1, "files": 1, "charged_once": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("level", ["user", "org", "team"])
@pytest.mark.parametrize("quota_policy", [None, "off", "on"], ids=["default", "off", "on"])
def test_vn_quota_policy_preserves_publication_replay_and_accounting(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str, level: str,
    quota_policy: str | None,
) -> None:
    """Quota posture changes admission, never same-source bytes or accounting."""
    enabled = quota_policy == "on"
    assert _storage_result(
        tmp_path, request, backend, f"quota_policy_{level}", quota_policy=quota_policy,
    ) == {
        "enabled": enabled, "first_charged_once": True, "replay_unchanged": True,
        "identity_preserved": True, "bytes_preserved": True,
        "distinct_admitted": not enabled, "final_accounting": True,
        "quota_denial_observed": enabled,
        "live": 1 if enabled else 2, "files": 1 if enabled else 2,
    }


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("limit", ["none", "zero", "limited", "org", "team", "user"])
def test_vn_outer_transaction_resolves_cold_and_warm_storage_limits(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str, limit: str,
) -> None:
    """Real override reads must not deadlock or fail open inside VN registration."""
    expected = {"denied": True, "unchanged": True, "files": 0} if limit == "zero" else {
        "denied": False, "charged_once": True, "replay_unchanged": True,
        "identity_preserved": True, "bytes_preserved": True, "live": 1, "files": 1,
    }
    assert _storage_result(tmp_path, request, backend, f"resolver_{limit}") == {
        **expected, "successful_cold_loads": 1, "quota_denial_observed": limit == "zero",
    }


@pytest.mark.integration
def test_vn_cold_quota_registration_completes_with_a_saturated_postgres_pool(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """Five real VN transactions must admit one candidate without a second pool lease."""
    result = _storage_result(tmp_path, request, "postgres", "pool_saturation")
    assert result["arrivals"] == 5 and result["exhausted"] is True
    assert result == {
        "completed": True, "arrivals": 5, "exhausted": True, "successful_loads": 1,
        "admitted": 1, "denied": 4, "live": 1, "charged_once": True,
    }


@pytest.mark.integration
@pytest.mark.parametrize("phase", ["readiness", "overrides"])
def test_vn_postgres_quota_read_error_preserves_outer_transaction_and_is_not_cached(
    tmp_path: Path, request: pytest.FixtureRequest, phase: str,
) -> None:
    """Failed reads must fail open without aborting or committing owning accounting."""
    assert _storage_result(tmp_path, request, "postgres", f"lookup_error_{phase}") == {
        "actual_cancellations": 4, "fallbacks": [None, None],
        "owning_connection_usable": [True, True], "admitted": True,
        "healthy_limit_after_failures": 0, "live": 1, "charged_once": True,
        "version_advanced": True,
    }
