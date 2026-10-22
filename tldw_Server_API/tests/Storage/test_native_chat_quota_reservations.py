"""Native retained bytes use the real AuthNZ authority, including cached readers."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tldw_Server_API.tests.AuthNZ.conftest import isolated_test_environment  # noqa: F401
from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import _run_runtime, _runtime_env

SCRIPT = r"""
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

async def main():
    try:
        assert await setup_database()
        case = os.environ['NATIVE_QUOTA_CASE']
        pool = await get_db_pool()
        if case == 'shared':
            from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
            repo = AuthnzUsersRepo(pool)
            members = [await repo.create_user(username=f'native-{i}', email=f'native-{i}@example.invalid', password_hash='fixture-hash') for i in range(2)]
            user = members[0]
        else:
            await bootstrap_single_user_profile()
            user = get_settings().SINGLE_USER_FIXED_ID
        service = StorageQuotaService(pool)
        await service.initialize()
        service.settings = service.settings.model_copy(update={'USER_DATA_BASE_PATH': Path.cwd() / 'scan-fixture', 'CHROMADB_BASE_PATH': None})
        from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaConflictError, NativeQuotaOwner
        owner = NativeQuotaOwner(user, 'native:fixture', str((await pool.fetchone('SELECT uuid FROM users WHERE id = ?', user))['uuid']))
        await service.set_user_quota(user, 10)
        if case == 'replay':
            await service.update_usage(user, 5 * 1048576)
            await service.check_quota(user, 0)
            first = await service.reserve_native_bytes(owner, 'candidate-one', 2 * 1048576)
            replay = await service.reserve_native_bytes(owner, 'candidate-one', 2 * 1048576)
            can, status = await service.check_quota(user, 4 * 1048576)
            mismatch = False
            try:
                await service.reserve_native_bytes(owner, 'candidate-one', 3 * 1048576)
            except NativeQuotaConflictError:
                mismatch = True
            released = await service.release_native_bytes(owner, 'candidate-one')
            again = await service.release_native_bytes(owner, 'candidate-one')
            terminal = await service.reserve_native_bytes(owner, 'candidate-one', 2 * 1048576)
            resolved = await service.resolve_native_reservation(owner, 'candidate-one')
            _, final = await service.check_quota(user, 0)
            print('RUNTIME_RESULT=' + json.dumps({
                'same': first == replay, 'charged': status['current_usage_mb'],
                'allowed': can, 'mismatch': mismatch, 'released': released == again == terminal == resolved,
                'state': terminal.state, 'final': final['current_usage_mb'],
                'legacy': (await pool.fetchone('SELECT storage_used_mb FROM users WHERE id = ?', user))['storage_used_mb'],
                'unknown': await service.release_native_bytes(owner, 'unknown') is None,
            }))
        elif case == 'race':
            async def reserve(key):
                try:
                    await service.reserve_native_bytes(owner, key, 8 * 1048576)
                    return 'admitted'
                except QuotaExceededError:
                    return 'denied'
            outcomes = await asyncio.gather(reserve('one'), reserve('two'))
            _, status = await service.check_quota(user, 0)
            print('RUNTIME_RESULT=' + json.dumps({'outcomes': sorted(outcomes), 'used': status['current_usage_mb']}))
        elif case == 'identity':
            await service.reserve_native_bytes(owner, 'one', 2 * 1048576)
            conflicts = []
            for changed in [replace(owner, user_uuid='different-account'),
                            replace(owner, owner_key='different-server-namespace'),
                            replace(owner, scope_type='workspace', workspace_id='11111111-1111-4111-8111-111111111111'),
                            replace(owner, org_id=51)]:
                try:
                    await service.reserve_native_bytes(changed, 'one', 2 * 1048576)
                    conflicts.append(False)
                except NativeQuotaConflictError:
                    conflicts.append(True)
            _, status = await service.check_quota(user, 0)
            print('RUNTIME_RESULT=' + json.dumps({'conflicts': conflicts, 'used': status['current_usage_mb']}))
        elif case == 'fresh':
            from tldw_Server_API.app.core.Usage import quota_resolver
            from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
            await service.check_quota(user, 0)
            old_cache = quota_resolver._cache[user]
            await UserProfileOverridesRepo(pool).upsert_override(user_id=user, key='limits.storage_quota_mb', value=1, updated_by=None)
            quota_resolver._cache[user] = old_cache
            denied = False
            try:
                await service.reserve_native_bytes(owner, 'one', 2 * 1048576)
            except QuotaExceededError:
                denied = True
            print('RUNTIME_RESULT=' + json.dumps({'denied': denied, 'native': (await service.native_usage(user))['bytes']}))
        elif case in {'crash', 'receipt', 'lease', 'bound', 'unknown'}:
            import hashlib
            from datetime import datetime, timedelta, timezone
            from unittest.mock import patch
            from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
            from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
            from tldw_Server_API.app.core.Chat.native_chat_assets import NativeChatAssetIO
            from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
            db = CharactersRAGDB(db_path=str(Path.cwd() / 'native.sqlite'), client_id=str(user), owner_user_id=str(user))
            native_owner = AuthorizedNativeOwner(str(user), owner.owner_key, NativeScopeV1(kind='global'))
            io = NativeChatAssetIO(db, user_id=user)
            DatabasePaths.get_user_base_directory(user)
            payload = b'quota-backed retained bytes'
            kind, operation, candidate = 'native_fork_v1', 'quota-fork', 'quota-candidate'
            with db.transaction() as conn:
                db.native_forks.reserve_operation(native_owner, kind, operation, 'sha256:quota', {'title': 'Quota'}, conn=conn)
                attempt = db.native_forks.claim_attempt(native_owner, kind, operation, 'sha256:quota', now=datetime.now(timezone.utc), conn=conn)
                db.native_assets.reserve_candidate(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                    candidate, 'a' * 32, 'quota-upload', hashlib.sha256(payload).hexdigest(), len(payload),
                    'application/octet-stream', 'document_original_v1', now=datetime.now(timezone.utc), conn=conn)
            if case == 'bound':
                class CallerRollback(NativeQuotaConflictError):
                    pass
                refused = False
                wrote = False
                try:
                    async with service._accounting_transaction() as bound:
                        try:
                            await io.reserve_quota(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                                candidate, quota_owner=owner, quota_service=bound)
                        except NativeQuotaConflictError:
                            refused = True
                        if not refused:
                            io.write_chunk(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                                candidate, chunk_index=0, payload=payload,
                                expected_hash='sha256:' + hashlib.sha256(payload).hexdigest())
                            wrote = True
                        raise CallerRollback()
                except CallerRollback:
                    pass
                facade_refusals = []
                async with service._accounting_transaction() as bound:
                    for call in (bound.reserve_native_bytes(owner, 'direct-bound', 1),
                                 bound.resolve_native_reservation(owner, 'direct-bound'),
                                 bound.release_native_bytes(owner, 'direct-bound')):
                        try:
                            await call
                            facade_refusals.append(False)
                        except NativeQuotaConflictError:
                            facade_refusals.append(True)
                print('RUNTIME_RESULT=' + json.dumps({'bridge_refused': refused, 'wrote': wrote,
                    'facade_refusals': facade_refusals, 'native': (await service.native_usage(user))['bytes']}))
                db.close_connection()
                return
            if case == 'lease':
                original_locked = db.native_assets._locked_candidate_for_operation
                def expire_at_candidate_lock(*args, **kwargs):
                    result = original_locked(*args, **kwargs)
                    kwargs['conn'].execute('UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?',
                        ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), operation))
                    return result
                intent_refused = False
                with patch.object(db.native_assets, '_locked_candidate_for_operation', expire_at_candidate_lock):
                    try:
                        await io.reserve_quota(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                            candidate, quota_owner=owner, quota_service=service)
                    except ValueError:
                        intent_refused = True
                before = (await service.native_usage(user))['bytes']
                with db.transaction() as conn:
                    conn.execute('UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?',
                        ((datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(), operation))
                    db.native_assets.record_quota_reservation_intent(native_owner, kind, operation, 'sha256:quota', candidate,
                        attempt.generation, owner, conn=conn)
                receipt = await service.reserve_native_bytes(owner, candidate, len(payload))
                ack_refused = False
                with patch.object(db.native_assets, '_locked_candidate_for_operation', expire_at_candidate_lock):
                    try:
                        with db.transaction() as conn:
                            db.native_assets.confirm_quota_reservation(native_owner, kind, operation, 'sha256:quota', candidate,
                                attempt.generation, owner, receipt, conn=conn)
                    except ValueError:
                        ack_refused = True
                with db.transaction() as conn:
                    state = conn.execute("SELECT state FROM native_chat_quota_intents WHERE candidate_id = ? AND intent_kind = 'reserve'", (candidate,)).fetchone()[0]
                print('RUNTIME_RESULT=' + json.dumps({'intent_refused': intent_refused, 'before': before,
                    'ack_refused': ack_refused, 'state': state}))
                db.close_connection()
                return
            if case == 'receipt':
                second = 'quota-candidate-second'
                with db.transaction() as conn:
                    db.native_assets.reserve_candidate(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                        second, 'b' * 32, 'quota-upload-second', hashlib.sha256(payload).hexdigest(), len(payload),
                        'application/octet-stream', 'document_original_v1', now=datetime.now(timezone.utc), conn=conn)
                    db.native_assets.record_quota_reservation_intent(native_owner, kind, operation, 'sha256:quota', second,
                        attempt.generation, owner, conn=conn)
                first_receipt = await service.reserve_native_bytes(owner, candidate, len(payload))
                await service.reserve_native_bytes(owner, second, len(payload))
                reserve_refused = False
                try:
                    with db.transaction() as conn:
                        db.native_assets.confirm_quota_reservation(native_owner, kind, operation, 'sha256:quota', second,
                            attempt.generation, owner, first_receipt, conn=conn)
                except ValueError:
                    reserve_refused = True
                with db.transaction() as conn:
                    conn.execute('UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?',
                                 ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), operation))
                from tldw_Server_API.app.core.DB_Management.chacha.native_chat_asset_store import NativeQuotaReleasePendingError
                try:
                    io.reclaim(native_owner, kind, operation, 'sha256:quota', second)
                except NativeQuotaReleasePendingError:
                    pass
                released_first = await service.release_native_bytes(owner, candidate)
                release_refused = False
                try:
                    with db.transaction() as conn:
                        db.native_assets.confirm_quota_release(native_owner, kind, operation, 'sha256:quota', second, released_first, conn=conn)
                except NativeQuotaReleasePendingError:
                    release_refused = True
                print('RUNTIME_RESULT=' + json.dumps({'reserve_refused': reserve_refused, 'release_refused': release_refused,
                    'second_charged': (await service.resolve_native_reservation(owner, second)).state == 'active'}))
                db.close_connection()
                return
            original_reserve = service.reserve_native_bytes
            async def lost_reserve(*args):
                assert not db._connection_state().conn.in_transaction
                await original_reserve(*args)
                raise TimeoutError('synthetic lost reservation response')
            with patch.object(service, 'reserve_native_bytes', lost_reserve):
                try:
                    await io.reserve_quota(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                        candidate, quota_owner=owner, quota_service=service)
                except TimeoutError:
                    pass
            retained_after_loss = (await service.native_usage(user))['bytes']
            await io.reserve_quota(native_owner, kind, operation, 'sha256:quota', attempt.generation,
                candidate, quota_owner=owner, quota_service=service)
            io.write_chunk(native_owner, kind, operation, 'sha256:quota', attempt.generation, candidate, chunk_index=0, payload=payload, expected_hash='sha256:' + hashlib.sha256(payload).hexdigest())
            io.publish(native_owner, kind, operation, 'sha256:quota', attempt.generation, candidate, chunk_indexes=[0])
            with db.transaction() as conn:
                conn.execute('UPDATE native_chat_operations SET lease_expires_at = ? WHERE operation_id = ?',
                             ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), operation))
            if case == 'unknown':
                from tldw_Server_API.app.core.DB_Management.chacha.native_chat_asset_store import NativeQuotaReleasePendingError
                async def unknown_release(*args):
                    assert not db._connection_state().conn.in_transaction
                    return None
                refused = False
                with patch.object(service, 'release_native_bytes', unknown_release):
                    try:
                        await io.reclaim_quota(native_owner, kind, operation, 'sha256:quota', candidate, quota_service=service)
                    except NativeQuotaReleasePendingError:
                        refused = True
                with db.transaction() as conn:
                    unresolved = conn.execute('SELECT state FROM native_chat_quota_intents WHERE candidate_id = ? AND intent_kind = ?', (candidate, 'release')).fetchone()[0]
                    reclaiming = conn.execute('SELECT state FROM native_chat_asset_candidates WHERE candidate_id = ?', (candidate,)).fetchone()[0]
                retained = (await service.native_usage(user))['bytes']
                await io.reclaim_quota(native_owner, kind, operation, 'sha256:quota', candidate, quota_service=service)
                print('RUNTIME_RESULT=' + json.dumps({'refused': refused, 'intent': unresolved, 'candidate': reclaiming,
                    'retained': retained == len(payload), 'final_native': (await service.native_usage(user))['bytes']}))
                db.close_connection()
                return
            original_release = service.release_native_bytes
            async def lost_release(*args):
                assert not db._connection_state().conn.in_transaction
                await original_release(*args)
                raise TimeoutError('synthetic lost release response')
            with patch.object(service, 'release_native_bytes', lost_release):
                try:
                    await io.reclaim_quota(native_owner, kind, operation, 'sha256:quota', candidate, quota_service=service)
                except TimeoutError:
                    pass
            with db.transaction() as conn:
                state_after_loss = conn.execute('SELECT state FROM native_chat_asset_candidates WHERE candidate_id = ?', (candidate,)).fetchone()[0]
            await io.reclaim_quota(native_owner, kind, operation, 'sha256:quota', candidate, quota_service=service)
            with db.transaction() as conn:
                state = conn.execute('SELECT state FROM native_chat_asset_candidates WHERE candidate_id = ?', (candidate,)).fetchone()[0]
            print('RUNTIME_RESULT=' + json.dumps({'charged_after_loss': retained_after_loss == len(payload),
                'reclaiming_after_release_loss': state_after_loss == 'reclaiming', 'discarded': state == 'discarded',
                'final_native': (await service.native_usage(user))['bytes']}))
            db.close_connection()
        elif case in {'registration-failure', 'org-failure', 'team-failure', 'delete-failure', 'bulk-failure', 'restore-failure', 'voice-failure', 'bulk-current', 'held-delete'}:
            from unittest.mock import patch
            from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
            from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
            from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaReservationStore, NativeQuotaUnavailableError
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
            await pool.execute('INSERT INTO organizations (id,name) VALUES (?, ?)', 51, 'native-ledger-failure-org')
            await pool.execute('INSERT INTO teams (id,org_id,name) VALUES (?, ?, ?)', 52, 51, 'native-ledger-failure-team')
            await pool.execute('INSERT INTO org_members (org_id,user_id) VALUES (?, ?)', 51, user)
            await pool.execute('INSERT INTO team_members (team_id,user_id) VALUES (?, ?)', 52, user)
            quotas = AuthnzStorageQuotasRepo(pool)
            await quotas.upsert_org_quota(51, quota_mb=10)
            await quotas.upsert_team_quota(52, quota_mb=10)
            await service.reserve_native_bytes(replace(owner, org_id=51, team_id=52), 'retained-lifecycle', 2 * 1048576)
            async def register():
                return await service.register_generated_file(user_id=user, filename='fixture.wav', storage_path='fixture.wav',
                    file_category='voice_clone' if case == 'voice-failure' else 'export', source_feature='export',
                    file_size_bytes=1048576, org_id=51, team_id=52)
            if case == 'bulk-current':
                first = await register()
                second = await service.register_generated_file(user_id=user, filename='second.txt', storage_path='second.txt',
                    file_category='export', source_feature='export', file_size_bytes=2 * 1048576, org_id=51, team_id=52)
                await service.unregister_generated_file(first['id'])
                changed = await service.unregister_generated_files([first, second, second])
                print('RUNTIME_RESULT=' + json.dumps({'changed': changed,
                    'legacy': (await pool.fetchone('SELECT storage_used_mb FROM users WHERE id = ?', user))['storage_used_mb'],
                    'pools': [(await quotas.get_org_quota(51))['used_mb'], (await quotas.get_team_quota(52))['used_mb']],
                    'native': (await service.native_usage(user))['bytes']}))
                return
            if case == 'held-delete':
                import copy
                from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo, _TransactionBoundPool
                first = await register()
                await service.register_generated_file(user_id=user, filename='second.txt', storage_path='second.txt',
                    file_category='export', source_feature='export', file_size_bytes=2 * 1048576, org_id=51, team_id=52)
                original_lock = AuthnzGeneratedFilesRepo.lock_quota_scopes
                reached_lock = asyncio.Event()
                worker = None
                async with pool.transaction() as held:
                    actor = copy.copy(service)
                    actor.db_pool = _TransactionBoundPool(held, postgres=True)
                    actor._quota_db_pool = actor.db_pool
                    await original_lock(AuthnzGeneratedFilesRepo(actor.db_pool), user_id=user, org_id=51, team_id=52)
                    xid = str(await held.fetchval('SELECT txid_current()'))
                    async def observed_lock(repo, **scope):
                        if repo.db_pool._conn is not held:
                            reached_lock.set()
                        await original_lock(repo, **scope)
                    with patch.object(AuthnzGeneratedFilesRepo, 'lock_quota_scopes', observed_lock):
                        worker = asyncio.create_task(service.unregister_generated_file(first['id'], hard_delete=True))
                        await asyncio.wait_for(reached_lock.wait(), 10)
                        async def blocked():
                            for _ in range(200):
                                if await pool.fetchval("SELECT COUNT(*) FROM pg_locks WHERE NOT granted AND locktype = 'transactionid' AND transactionid::text = ?", xid):
                                    return True
                                await asyncio.sleep(0.01)
                            return False
                        assert await blocked(), 'The genuine worker must wait on the held owner transaction'
                        await actor.unregister_generated_file(first['id'])
                assert await asyncio.wait_for(worker, 10)
                print('RUNTIME_RESULT=' + json.dumps({'legacy': (await pool.fetchone('SELECT storage_used_mb FROM users WHERE id = ?', user))['storage_used_mb'],
                    'pools': [(await quotas.get_org_quota(51))['used_mb'], (await quotas.get_team_quota(52))['used_mb']],
                    'native': (await service.native_usage(user))['bytes']}))
                return
            record = None
            voice = None
            if case in {'delete-failure', 'bulk-failure', 'restore-failure', 'voice-failure'}:
                record = await register()
                if case == 'restore-failure':
                    await service.unregister_generated_file(record['id'])
                if case == 'voice-failure':
                    voice = DatabasePaths.get_user_voices_dir(user) / 'fixture.wav'
                    voice.parent.mkdir(parents=True, exist_ok=True)
                    voice.write_bytes(b'voice fixture')
            async def snapshot():
                records = await pool.fetchall('SELECT id,is_deleted FROM generated_files ORDER BY id')
                return {'records': [(row['id'], bool(row['is_deleted'])) for row in records],
                    'legacy': [float((await pool.fetchone('SELECT storage_used_mb FROM users WHERE id = ?', user))['storage_used_mb']),
                        float((await quotas.get_org_quota(51))['used_mb']), float((await quotas.get_team_quota(52))['used_mb'])],
                    'voice_present': voice is None or voice.is_file()}
            before = await snapshot()
            original_usage = NativeQuotaReservationStore.usage
            async def failed_after_mutation(store, **scope):
                result = await original_usage(store, **scope)
                rows = await store.db_pool.fetchall('SELECT is_deleted FROM generated_files')
                trigger = False
                if case == 'registration-failure':
                    trigger = bool(rows)
                elif case in {'org-failure', 'team-failure'}:
                    column, identity = ('org_id', 51) if case == 'org-failure' else ('team_id', 52)
                    quota = await store.db_pool.fetchone(f'SELECT used_mb FROM storage_quotas WHERE {column} = ?', identity)
                    trigger = quota['used_mb'] > 0
                elif case in {'delete-failure', 'bulk-failure'}:
                    trigger = bool(rows) and bool(rows[0]['is_deleted'])
                elif case == 'restore-failure':
                    trigger = bool(rows) and not bool(rows[0]['is_deleted'])
                elif case == 'voice-failure':
                    trigger = not rows
                if trigger:
                    raise NativeQuotaUnavailableError('Native retained storage usage is unavailable')
                return result
            failed = False
            with patch.object(NativeQuotaReservationStore, 'usage', failed_after_mutation):
                try:
                    if case == 'registration-failure':
                        await register()
                    elif case == 'org-failure':
                        await service.update_org_usage(51, 1048576)
                    elif case == 'team-failure':
                        await service.update_team_usage(52, 1048576)
                    elif case in {'delete-failure', 'voice-failure'}:
                        await service.unregister_generated_file(record['id'], hard_delete=case == 'voice-failure')
                    elif case == 'bulk-failure':
                        await service.unregister_generated_files([record])
                    else:
                        await service.restore_generated_file(record['id'])
                except StorageError:
                    failed = True
            print('RUNTIME_RESULT=' + json.dumps({'failed': failed, 'rolled_back': await snapshot() == before,
                'native': (await service.native_usage(user))['bytes']}))
        elif case == 'failure':
            from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
            from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaUnavailableError
            from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer, UsageSummary
            import time
            await service.check_quota(user, 0)
            await service.reserve_native_bytes(owner, 'retained-before-read-failure', 2 * 1048576)
            original_fetchone = pool.fetchone
            async def failed_native_read(query, *args):
                if 'native_chat_quota_reservations' in query and 'SUM(' in query:
                    return await original_fetchone('SELECT COUNT(*) FROM missing_native_chat_ledger')
                return await original_fetchone(query, *args)
            pool.fetchone = failed_native_read
            write_failed = False
            try:
                await service.update_usage(user, 1048576)
            except StorageError:
                write_failed = True
            check_failed = False
            try:
                await service.check_quota(user, 0)
            except NativeQuotaUnavailableError:
                check_failed = True
            billing = BillingEnforcer()
            billing._usage_cache[51] = (UsageSummary(org_id=51, storage_bytes=3 * 1048576), time.time())
            billing_failed = False
            try:
                await billing.get_org_usage(51)
            except NativeQuotaUnavailableError:
                billing_failed = True
            print('RUNTIME_RESULT=' + json.dumps({'write_failed': write_failed, 'check_failed': check_failed,
                'billing_failed': billing_failed,
                'legacy': (await pool.fetchone('SELECT storage_used_mb FROM users WHERE id = ?', user))['storage_used_mb']}))
        elif case == 'admin':
            from fastapi import HTTPException
            from unittest.mock import patch
            from tldw_Server_API.app.services.admin_system_service import get_system_stats
            await service.reserve_native_bytes(owner, 'retained-admin', 2 * 1048576)
            async with pool.acquire() as conn:
                normal = await get_system_stats(conn)
                method = 'fetchrow' if pool.pool is not None else 'execute'
                original_query = getattr(type(conn), method)
                async def missing_sessions(connection, query, *args):
                    if connection is conn and 'FROM sessions' in query:
                        return await original_query(connection, 'SELECT COUNT(*) FROM missing_sessions')
                    return await original_query(connection, query, *args)
                unavailable = False
                with patch.object(type(conn), method, missing_sessions):
                    try:
                        await get_system_stats(conn)
                    except HTTPException as exc:
                        unavailable = exc.status_code == 503
            print('RUNTIME_RESULT=' + json.dumps({'used': normal.storage.total_used_mb, 'unavailable': unavailable}))
        elif case == 'policy':
            await service.set_user_quota(user, 0)
            denied = False
            try:
                await service.reserve_native_bytes(owner, 'zero', 1)
            except QuotaExceededError:
                denied = True
            await service.set_user_quota(user, None)
            await service.reserve_native_bytes(owner, 'unlimited', 2 * 1048576)
            await service.set_user_quota(user, 0)
            os.environ['USAGE_QUOTAS_ENABLED'] = '0'
            await service.reserve_native_bytes(owner, 'off', 2 * 1048576)
            print('RUNTIME_RESULT=' + json.dumps({'zero_denied': denied, 'charged': (await service.native_usage(user))['bytes']}))
        elif case == 'rollback':
            from unittest.mock import patch
            from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaReservationStore
            original = NativeQuotaReservationStore.insert
            async def cancelled(store, *args):
                await original(store, *args)
                raise asyncio.CancelledError()
            with patch.object(NativeQuotaReservationStore, 'insert', cancelled):
                try:
                    await service.reserve_native_bytes(owner, 'cancelled', 1048576)
                except asyncio.CancelledError:
                    pass
            print('RUNTIME_RESULT=' + json.dumps({'native': (await service.native_usage(user))['bytes'],
                'unknown': await service.resolve_native_reservation(owner, 'cancelled') is None}))
        elif case == 'shared':
            import time
            from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
            from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer, UsageSummary
            from tldw_Server_API.app.api.v1.endpoints.storage_helpers import _to_quota_status
            for member in members:
                await service.set_user_quota(member, 100)
            await pool.execute('INSERT INTO organizations (id,name) VALUES (?, ?)', 51, 'native-quota-org')
            await pool.execute('INSERT INTO teams (id,org_id,name) VALUES (?, ?, ?)', 52, 51, 'native-quota-team')
            for member in members:
                await pool.execute('INSERT INTO org_members (org_id,user_id) VALUES (?, ?)', 51, member)
                await pool.execute('INSERT INTO team_members (team_id,user_id) VALUES (?, ?)', 52, member)
            owners = [NativeQuotaOwner(member, 'native:fixture', str((await pool.fetchone('SELECT uuid FROM users WHERE id = ?', member))['uuid']), org_id=51, team_id=52) for member in members]
            await service.reserve_native_bytes(owners[0], 'before-pool', 2 * 1048576)
            quotas = AuthnzStorageQuotasRepo(pool)
            await quotas.upsert_org_quota(51, quota_mb=10)
            await quotas.upsert_team_quota(52, quota_mb=10)
            async def reserve(index):
                try:
                    await service.reserve_native_bytes(owners[index], f'shared-{index}', 6 * 1048576)
                    return 'admitted'
                except QuotaExceededError:
                    return 'denied'
            outcomes = await asyncio.gather(reserve(0), reserve(1))
            org = await quotas.check_quota_status(org_id=51)
            team = await quotas.check_quota_status(team_id=52)
            mapped = _to_quota_status(team)
            from types import SimpleNamespace
            from unittest.mock import patch, AsyncMock
            from tldw_Server_API.app.api.v1.endpoints import storage_admin_quotas
            from tldw_Server_API.app.api.v1.schemas.storage_schemas import SetUserQuotaRequest
            async def actual_service():
                return service
            with patch.object(storage_admin_quotas, '_get_storage_service', actual_service), patch.object(storage_admin_quotas, 'emit_storage_quota_audit_event', AsyncMock()):
                user_response = await storage_admin_quotas.set_user_quota(members[0], SetUserQuotaRequest(quota_mb=100), SimpleNamespace(user_id=members[0]))
            user_actual_native = (await service.native_usage(members[0]))['bytes'] / 1048576
            user_components = user_response.quota.legacy_used_mb == 0 and user_response.quota.native_used_mb == user_actual_native
            await service.reserve_native_bytes(replace(owners[1], org_id=None, team_id=None), 'global', 1048576)
            billing = BillingEnforcer()
            billing._usage_cache[51] = (UsageSummary(org_id=51, storage_bytes=3 * 1048576), time.time())
            summaries = [await billing.get_org_usage(51), await billing.get_org_usage(51)]
            await pool.execute('UPDATE org_members SET status = NULL WHERE org_id = ? AND user_id = ?', 51, members[1])
            native_first = (await service.native_usage(members[0]))['bytes']
            null_member_excluded = (await billing.get_org_usage(51)).storage_bytes == 3 * 1048576 + native_first
            await pool.execute("UPDATE org_members SET status = 'active' WHERE org_id = ?", 51)
            await pool.execute("UPDATE org_members SET status = 'inactive' WHERE org_id = ?", 51)
            await pool.execute("UPDATE team_members SET status = 'inactive' WHERE team_id = ?", 52)
            replay = await service.reserve_native_bytes(owners[0], 'before-pool', 2 * 1048576)
            denied = False
            try:
                await service.reserve_native_bytes(owners[0], 'after-loss', 1)
            except NativeQuotaConflictError:
                denied = True
            await service.release_native_bytes(owners[0], 'before-pool')
            remaining = await quotas.check_quota_status(team_id=52)
            print('RUNTIME_RESULT=' + json.dumps({'outcomes': sorted(outcomes), 'pool_totals': [org['used_mb'], team['used_mb']],
                'components': [mapped.legacy_used_mb, mapped.native_used_mb], 'user_components': user_components,
                'billing': [summary.storage_bytes // 1048576 for summary in summaries],
                'cached_legacy': billing._usage_cache[51][0].storage_bytes // 1048576, 'null_member_excluded': null_member_excluded,
                'same_key_cleanup': replay.state == 'active' and denied,
                'remaining': remaining['used_mb'], 'stored_pool_legacy': (await quotas.get_team_quota(52))['used_mb']}))
        elif case == 'recalculate':
            directory = Path(service.settings.USER_DATA_BASE_PATH) / str(user)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'legacy.bin').write_bytes(b'x' * 1048576)
            managed = directory / 'chat_assets'
            managed.mkdir()
            (managed / 'native.bin').write_bytes(b'y' * 2 * 1048576)
            before = await service.calculate_user_storage(user)
            await service.reserve_native_bytes(owner, 'one', 2 * 1048576)
            cached = await service.calculate_user_storage(user, update_database=False)
            recalculated = await service.calculate_user_storage(user)
            _, current = await service.check_quota(user, 0)
            status = await service.user_quota_status(user)
            generated = await service.get_user_generated_files_usage(user)
            users = await service.get_all_users_storage()
            breakdown = await service.get_storage_breakdown(user)
            await service.release_native_bytes(owner, 'one')
            released = await service.calculate_user_storage(user, update_database=False)
            print('RUNTIME_RESULT=' + json.dumps({
                'totals': [before['total_mb'], cached['total_mb'], recalculated['total_mb'], current['current_usage_mb'],
                           status['used_mb'], generated['quota_used_mb'], users[0]['storage_used_mb'], breakdown['current_usage_mb'], released['total_mb']],
                'native': recalculated['native_bytes'],
                'legacy': (await pool.fetchone('SELECT storage_used_mb FROM users WHERE id = ?', user))['storage_used_mb'],
                'breakdown_native': breakdown['native']['bytes'],
            }))
    finally:
        await reset_db_pool()

asyncio.run(main())
"""


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "replay",
            {
                "same": True,
                "charged": 7.0,
                "allowed": False,
                "mismatch": True,
                "released": True,
                "state": "released",
                "final": 5.0,
                "legacy": 5.0,
                "unknown": True,
            },
        ),
        ("bulk-current", {"changed": 1, "legacy": 0.0, "pools": [0.0, 0.0], "native": 2097152}),
        ("registration-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("org-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("team-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("delete-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("bulk-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("restore-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("voice-failure", {"failed": True, "rolled_back": True, "native": 2097152}),
        ("admin", {"used": 2.0, "unavailable": True}),
        ("lease", {"intent_refused": True, "before": 0, "ack_refused": True, "state": "pending"}),
        ("receipt", {"reserve_refused": True, "release_refused": True, "second_charged": True}),
        ("failure", {"write_failed": True, "check_failed": True, "billing_failed": True, "legacy": 0.0}),
        ("policy", {"zero_denied": True, "charged": 4194304}),
        ("rollback", {"native": 0, "unknown": True}),
        (
            "shared",
            {
                "outcomes": ["admitted", "denied"],
                "pool_totals": [8.0, 8.0],
                "components": [0.0, 8.0],
                "user_components": True,
                "billing": [12, 12],
                "cached_legacy": 3,
                "null_member_excluded": True,
                "same_key_cleanup": True,
                "remaining": 6.0,
                "stored_pool_legacy": 0.0,
            },
        ),
        ("race", {"outcomes": ["admitted", "denied"], "used": 8.0}),
        ("identity", {"conflicts": [True, True, True, True], "used": 2.0}),
        ("fresh", {"denied": True, "native": 0}),
        ("bound", {"bridge_refused": True, "wrote": False, "facade_refusals": [True, True, True], "native": 0}),
        (
            "unknown",
            {"refused": True, "intent": "pending", "candidate": "reclaiming", "retained": True, "final_native": 0},
        ),
        (
            "crash",
            {"charged_after_loss": True, "reclaiming_after_release_loss": True, "discarded": True, "final_native": 0},
        ),
        (
            "recalculate",
            {
                "totals": [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0],
                "native": 2097152,
                "legacy": 1.0,
                "breakdown_native": 2097152,
            },
        ),
    ],
)
def test_native_quota_authority_and_existing_readers(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    backend: str,
    case: str,
    expected: dict,
) -> None:
    """Use actual supported-runtime setup and official isolated PostgreSQL when available."""
    if backend == "postgres":
        _client, db_name = request.getfixturevalue("isolated_test_environment")
        database_url = os.environ["DATABASE_URL"]
        assert urlparse(database_url).path.lstrip("/") == db_name
    else:
        database_url = f"sqlite:///{tmp_path / 'users.db'}"
    env = _runtime_env(
        tmp_path, database_url, backend=backend, mode="multi_user" if case == "shared" else "single_user"
    )
    env.update(
        USAGE_QUOTAS_ENABLED="1",
        NATIVE_QUOTA_CASE=case,
        USER_DB_BASE_DIR=str(tmp_path / "owned-user-db"),
        USER_DATA_BASE_PATH=str(tmp_path / "user-data"),
    )
    assert _run_runtime(tmp_path, env, SCRIPT) == expected


@pytest.mark.integration
def test_native_accounting_postgres_held_single_delete_rereads_original_row(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A genuine quota-lock wait must not retain the pre-wait active state."""
    _client, db_name = request.getfixturevalue("isolated_test_environment")
    database_url = os.environ["DATABASE_URL"]
    assert urlparse(database_url).path.lstrip("/") == db_name
    env = _runtime_env(tmp_path, database_url, backend="postgres")
    env.update(
        USAGE_QUOTAS_ENABLED="1", NATIVE_QUOTA_CASE="held-delete", USER_DB_BASE_DIR=str(tmp_path / "owned-user-db")
    )
    assert _run_runtime(tmp_path, env, SCRIPT) == {"legacy": 2.0, "pools": [2.0, 2.0], "native": 2097152}


BOOTSTRAP_SCRIPT = r"""
import asyncio
import json
import os
import sqlite3
from pathlib import Path
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations
from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaOwner
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

async def main():
    prior = os.environ['NATIVE_QUOTA_BOOTSTRAP']
    previous = prior != 'fresh'
    if previous:
        path = Path.cwd() / 'users.db'
        apply_authnz_migrations(path, target_version=100)
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 100
            assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'native_chat_quota_reservations'").fetchone()[0] == 0
            conn.execute('INSERT INTO users (id,uuid,username,email,password_hash,storage_used_mb,is_active) VALUES (?,?,?,?,?,?,?)',
                         (41, '22222222222242228222222222222222', 'preserved', 'preserved@example.invalid', 'fixture-hash', 12, 0))
            conn.execute('DROP INDEX IF EXISTS idx_sessions_created_at')
            conn.execute('DROP INDEX IF EXISTS idx_org_members_org_user')
            if prior == 'preview101':
                from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import native_quota_schema_statements
                for statement in native_quota_schema_statements(postgres=False):
                    conn.execute(statement)
                conn.execute("INSERT INTO native_chat_quota_reservations (user_id,user_uuid,owner_key,candidate_key,scope_type,size_bytes,state) VALUES (?,?,?,?,?,?,?)",
                             (41, '22222222222242228222222222222222', 'native:preview', 'retained-preview-charge', 'global', 19, 'active'))
                conn.execute("INSERT INTO schema_migrations VALUES (101, 'Add native retained chat quota reservations', CURRENT_TIMESTAMP)")
        if prior == 'admin101':
            apply_authnz_migrations(path, target_version=101)
    try:
        assert await setup_database()
        assert await bootstrap_single_user_profile()
        pool = await get_db_pool()
        user = get_settings().SINGLE_USER_FIXED_ID
        before = dict(await pool.fetchone('SELECT uuid,storage_used_mb FROM users WHERE id = ?', user))
        owner = NativeQuotaOwner(user, 'native:bootstrap', str(before['uuid']))
        service = StorageQuotaService(pool)
        await service.reserve_native_bytes(owner, 'preserved-bootstrap-charge', 27)
        await reset_db_pool()
        outcomes = []
        for _ in range(2):
            assert await setup_database()
            assert await bootstrap_single_user_profile()
            pool = await get_db_pool()
            row = dict(await pool.fetchone('SELECT uuid,storage_used_mb FROM users WHERE id = ?', user))
            service = StorageQuotaService(pool)
            receipt = await service.resolve_native_reservation(owner, 'preserved-bootstrap-charge')
            outcomes.append(row == before and receipt.state == 'active' and receipt.size_bytes == 27
                and (await service.native_usage(user))['bytes'] == 27)
            if previous:
                sentinel = await pool.fetchone('SELECT uuid,storage_used_mb FROM users WHERE id = ?', 41)
                assert sentinel['uuid'] == '22222222222242228222222222222222' and sentinel['storage_used_mb'] == 12
            if os.environ['NATIVE_QUOTA_BACKEND'] == 'sqlite':
                assert (await pool.fetchone('SELECT MAX(version) AS version FROM schema_migrations'))['version'] == 103
                indexes = await pool.fetchall("SELECT name FROM sqlite_master WHERE type = 'index' AND name IN ('idx_sessions_created_at', 'idx_org_members_org_user')")
                assert {row['name'] for row in indexes} == {'idx_sessions_created_at', 'idx_org_members_org_user'}
            if prior == 'preview101':
                retained = await pool.fetchone("SELECT user_uuid,owner_key,size_bytes,state FROM native_chat_quota_reservations WHERE user_id = ? AND candidate_key = ?", 41, 'retained-preview-charge')
                assert dict(retained) == {'user_uuid': '22222222222242228222222222222222', 'owner_key': 'native:preview', 'size_bytes': 19, 'state': 'active'}
            await reset_db_pool()
        print('RUNTIME_RESULT=' + json.dumps({'preserved': outcomes, 'prior_user': previous}))
    finally:
        await reset_db_pool()

asyncio.run(main())
"""


@pytest.mark.integration
@pytest.mark.parametrize("backend, prior", [("sqlite", "fresh"), ("sqlite", "prior100"), ("sqlite", "admin101"), ("sqlite", "preview101"), ("postgres", "fresh")])
def test_native_quota_fresh_upgrade_and_repeated_bootstrap_preserve_charge(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    backend: str,
    prior: str,
) -> None:
    if backend == "postgres":
        _client, db_name = request.getfixturevalue("isolated_test_environment")
        database_url = os.environ["DATABASE_URL"]
        assert urlparse(database_url).path.lstrip("/") == db_name
    else:
        database_url = f"sqlite:///{tmp_path / 'users.db'}"
    env = _runtime_env(tmp_path, database_url, backend=backend)
    env.update(USAGE_QUOTAS_ENABLED="1", NATIVE_QUOTA_BOOTSTRAP=prior, NATIVE_QUOTA_BACKEND=backend)
    assert _run_runtime(tmp_path, env, BOOTSTRAP_SCRIPT) == {
        "preserved": [True, True],
        "prior_user": prior != "fresh",
    }
