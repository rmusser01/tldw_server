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

async def main():
    try:
        assert await setup_database()
        await bootstrap_single_user_profile()
        pool = await get_db_pool()
        repo = AuthnzGeneratedFilesRepo(pool)
        owner = get_settings().SINGLE_USER_FIXED_ID
        async def register(index):
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
from pathlib import Path
from unittest.mock import patch
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Storage import generated_file_helpers as helpers
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

async def main():
    try:
        assert await setup_database()
        await bootstrap_single_user_profile()
        pool = await get_db_pool()
        owner = get_settings().SINGLE_USER_FIXED_ID
        service = StorageQuotaService(pool)
        await service.initialize()
        repo = AuthnzGeneratedFilesRepo(pool)
        await pool.execute('INSERT INTO organizations (id, name) VALUES (?, ?)', 51, 'vn-storage-org')
        await pool.execute('INSERT INTO teams (id, org_id, name) VALUES (?, ?, ?)', 52, 51, 'vn-storage-team')
        quotas = AuthnzStorageQuotasRepo(pool)
        await quotas.upsert_org_quota(51, quota_mb=100)
        await quotas.upsert_team_quota(52, quota_mb=100)
        outputs = Path.cwd() / 'outputs'
        helpers.DatabasePaths.get_user_outputs_dir = staticmethod(lambda _user_id: outputs)
        async def get_service():
            return service
        helpers.get_storage_service = get_service

        async def save(item=42):
            return await helpers.save_and_register_vn_asset_image(
                user_id=owner, pack_id=7, item_id=item, asset_type='sprite',
                image_bytes=b'image', org_id=51, team_id=52,
            )

        async def snapshot():
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
            async def fail_after_update(self, *args, **kwargs):
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
            await service.update_usage(owner, int(get_settings().DEFAULT_STORAGE_QUOTA_MB * 1048576))
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
            quota_mb = await pool.fetchval('SELECT storage_quota_mb FROM users WHERE id = ?', owner)
            await service.update_usage(owner, int(quota_mb * 1048576) - 5)
            await quotas.update_org_used_mb(51, 100 - 5 / 1048576)
            await quotas.update_team_used_mb(52, 100 - 5 / 1048576)
            original = StorageQuotaService.get_vn_generated_file
            raced = False
            winner = None
            async def commit_after_empty_lookup(self, **kwargs):
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
            original = helpers._save_file
            async def register_missing_winner(*args, **kwargs):
                candidate = await original(*args, **kwargs)
                await service.register_generated_file(
                    user_id=owner, filename='missing.png', storage_path='vn_assets/missing.png',
                    file_category='image', source_feature='vn_assets', source_ref='vn_asset_item:42',
                    file_size_bytes=5, org_id=51, team_id=52,
                )
                return candidate
            with patch.object(helpers, '_save_file', register_missing_winner):
                try:
                    await save()
                except Exception:
                    result['rejected'] = True
                else:
                    result['rejected'] = False
            result['replacement_preserved'] = len(list(outputs.rglob('*.png'))) == 1
        elif case == 'missing_record':
            original = AuthnzGeneratedFilesRepo.create_file
            async def lose_created_record(self, **kwargs):
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
            async def fail_after_commit(*args, **kwargs):
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
        elif case.startswith('distinct_'):
            level = case.removeprefix('distinct_')
            if level == 'user':
                quota_mb = await pool.fetchval('SELECT storage_quota_mb FROM users WHERE id = ?', owner)
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


def _storage_result(tmp_path: Path, request: pytest.FixtureRequest, backend: str, case: str) -> dict:
    """Run against the shared isolated PostgreSQL fixture or a private SQLite file."""
    env = _vn_runtime_env(tmp_path, request, backend)
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
def test_vn_missing_replay_bytes_never_discard_a_replacement_write(
    tmp_path: Path, request: pytest.FixtureRequest, backend: str,
) -> None:
    assert _storage_result(tmp_path, request, backend, "missing_race") == {
        "rejected": True, "replacement_preserved": True,
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
