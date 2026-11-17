"""Real AuthNZ accounting and house byte validation for private retention staging."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

from tldw_Server_API.tests.AuthNZ.conftest import isolated_test_environment  # noqa: F401
from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import _run_runtime, _runtime_env

SCRIPT = r"""
import asyncio, hashlib, io, json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from starlette.datastructures import UploadFile
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaOwner
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Chat.native_chat_assets import NativeChatAssetIO
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner, native_asset_retention_request_digest
from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1, NativeAssetRetentionRequestV1

async def main():
    db = None
    try:
        assert await setup_database()
        await bootstrap_single_user_profile()
        user = get_settings().SINGLE_USER_FIXED_ID
        pool = await get_db_pool()
        service = StorageQuotaService(pool)
        await service.initialize()
        await service.set_user_quota(user, 10)
        quota_owner = NativeQuotaOwner(user, 'native:retention', str((await pool.fetchone('SELECT uuid FROM users WHERE id = ?', user))['uuid']))
        owner = AuthorizedNativeOwner(str(user), quota_owner.owner_key, NativeScopeV1(kind='global'))
        DatabasePaths.get_user_base_directory(user)
        db = CharactersRAGDB(db_path=str(Path.cwd() / 'native.sqlite'), client_id=str(user), owner_user_id=str(user))
        adapter = NativeChatAssetIO(db, user_id=user)
        cid = db.add_conversation({'character_id': None, 'title': 'Original source'})
        with db.transaction() as conn:
            revision = db.native_forks.read_retention_source_revision(owner, cid, conn=conn)
        case = os.environ['RETENTION_UPLOAD_CASE']
        content = 'Résumé 😀'.encode()
        representation, mime, filename = 'document_text_v1', 'text/plain', 'reviewed.txt'
        if case == 'mime':
            representation, mime, filename = 'document_original_v1', 'application/pdf', 'source.pdf'
        value = NativeAssetRetentionRequestV1.model_validate({
            'operation_kind': 'native_asset_retention_v1',
            'operation_id': '11111111-1111-4111-8111-111111111111',
            'owner_key': owner.owner_key, 'scope': {'kind': 'global'},
            'source_conversation_id': cid, 'expected_asset_manifest_revision': revision,
            'asset_manifest': [{'reference_id': 'ref-a', 'reference_revision': 'original',
                'asset_id': 'source-asset', 'revision': 'selected', 'representation': representation,
                'role': 'document_context', 'context_enabled': True,
                'hash': hashlib.sha256(content).hexdigest(), 'fidelity_disposition': 'retained'}],
            'request_digest': '0' * 64,
        })
        value = value.model_copy(update={'request_digest': native_asset_retention_request_digest(value)})
        with db.transaction() as conn:
            attempt = db.native_forks.claim_retention_attempt(owner, value, conn=conn)
        kwargs = dict(reference_id='ref-a', candidate_id='candidate-one', namespace_id='a'*32,
            upload_id='upload-one', size_bytes=len(content), mime_type=mime,
            quota_owner=quota_owner, quota_service=service)
        if case == 'owner':
            from dataclasses import replace
            kwargs['quota_owner'] = replace(quota_owner, owner_key='foreign')
        async def call(upload):
            return await adapter.prepare_retention_upload(owner, value, attempt.generation, upload=upload, **kwargs)
        if case in {'promotion_resolve', 'promotion_reserve'}:
            target = 'resolve_native_reservation' if case == 'promotion_resolve' else 'reserve_native_bytes'
            original = getattr(service, target)
            promoted = False
            async def promote(*args, **kw):
                nonlocal promoted
                receipt = await original(*args, **kw)
                if not promoted:
                    promoted = True
                    second = UploadFile(file=io.BytesIO(content), filename=filename)
                    try:
                        assert (await call(second)).state == 'prepared'
                    finally:
                        await second.close()
                return receipt
            context = patch.object(service, target, promote)
        elif case == 'lock_order':
            from contextlib import ExitStack
            events = []
            original_workspace = db.native_forks.lock_open_workspace
            original_candidate = db.native_assets._candidate_row
            original_resolve = service.resolve_native_reservation
            original_reserve = service.reserve_native_bytes
            def workspace(*args, **kw):
                events.append('workspace')
                return original_workspace(*args, **kw)
            def candidate(*args, **kw):
                events.append('candidate')
                return original_candidate(*args, **kw)
            async def resolved(*args, **kw):
                receipt = await original_resolve(*args, **kw)
                events.clear()
                return receipt
            async def allocated(*args, **kw):
                assert not db._connection_state().conn.in_transaction
                assert events.index('workspace') < events.index('candidate')
                return await original_reserve(*args, **kw)
            context = ExitStack()
            context.enter_context(patch.object(db.native_forks, 'lock_open_workspace', workspace))
            context.enter_context(patch.object(db.native_assets, '_candidate_row', candidate))
            context.enter_context(patch.object(service, 'resolve_native_reservation', resolved))
            context.enter_context(patch.object(service, 'reserve_native_bytes', allocated))
        elif case in {'after_quota', 'after_upload'}:
            target = 'reserve_quota' if case == 'after_quota' else 'prepare_upload'
            original = getattr(adapter, target)
            async def changed(*args, **kw):
                result = await original(*args, **kw)
                assert not db._connection_state().conn.in_transaction
                db.add_message({'conversation_id': cid, 'sender': 'user', 'content': 'Changed'})
                return result
            context = patch.object(adapter, target, changed)
        else:
            from contextlib import nullcontext
            context = nullcontext()
        refused = False
        with context:
            upload = UploadFile(file=io.BytesIO(b'x' * len(content) if case == 'hash' else content), filename=filename)
            try:
                if case == 'nested':
                    with db.transaction():
                        await call(upload)
                else:
                    result = await call(upload)
            except ValueError as exc:
                refused = True
                if case == 'success':
                    import traceback
                    print('RUNTIME_RESULT=' + json.dumps({'unexpected_refusal': type(exc).__name__,
                        'source': ':'.join([Path(traceback.extract_tb(exc.__traceback__)[-1].filename).name, str(traceback.extract_tb(exc.__traceback__)[-1].lineno), traceback.extract_tb(exc.__traceback__)[-1].name])}))
                    return
            finally:
                await upload.close()
        row = db.execute_query('SELECT * FROM native_chat_asset_candidates WHERE candidate_id = ?', ('candidate-one',)).fetchone()
        charged = (await service.native_usage(user))['bytes']
        if case == 'after_quota':
            assert row['state'] == 'reserved' and row['storage_key'] is None
            root = DatabasePaths.resolve_user_base_directory(user) / 'chat_assets'
            assert not root.exists() or not any(root.rglob('*.part'))
        if case == 'after_upload':
            assert row['state'] == 'prepared'
            assert (DatabasePaths.resolve_user_base_directory(user) / 'chat_assets' / row['storage_key']).read_bytes() == content
        exact = False
        replay = False
        if case in {'success', 'missing_receipt', 'released', 'promotion_resolve', 'promotion_reserve', 'lock_order'}:
            assert not refused and result.state == 'prepared'
            exact = (DatabasePaths.resolve_user_base_directory(user) / 'chat_assets' / row['storage_key']).read_bytes() == content
            class ForbiddenUpload:
                filename = 'reviewed.txt'
                async def read(self, *args):
                    raise AssertionError('prepared replay consumed upload')
            before = dict(row)
            if case in {'missing_receipt', 'released'}:
                if case == 'released':
                    await service.release_native_bytes(quota_owner, 'candidate-one')
                    charged = (await service.native_usage(user))['bytes']
                from contextlib import nullcontext
                missing = patch.object(service, 'resolve_native_reservation', return_value=None) if case == 'missing_receipt' else nullcontext()
                with missing, patch.object(service, 'reserve_native_bytes', side_effect=AssertionError('prepared replay allocated missing authority')):
                    try:
                        await call(ForbiddenUpload())
                    except ValueError:
                        refused = True
                assert refused
                again = None
            else:
                again = await call(ForbiddenUpload())
            assert dict(db.execute_query('SELECT * FROM native_chat_asset_candidates WHERE candidate_id = ?', ('candidate-one',)).fetchone()) == before
            replay = again == result
            assert (await service.native_usage(user))['bytes'] == (0 if case == 'released' else len(content))
        claims = db.execute_query('SELECT COUNT(*) FROM native_chat_asset_claims').fetchone()[0]
        if row is not None:
            with db.transaction() as conn:
                conn.execute('UPDATE native_chat_operations SET lease_expires_at = ?', ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),))
            await adapter.reclaim_quota(owner, value.operation_kind, value.operation_id, value.request_digest, 'candidate-one', quota_service=service)
        final_charge = (await service.native_usage(user))['bytes']
        print('RUNTIME_RESULT=' + json.dumps({'refused': refused, 'exact': exact, 'replay': replay,
            'charged': charged == len(content), 'candidate': row is not None, 'claims': claims, 'clean': final_charge == 0}))
    finally:
        if db is not None:
            db.close_connection()
        await reset_db_pool()
try:
    asyncio.run(main())
except Exception as exc:
    import traceback
    print('RUNTIME_RESULT=' + json.dumps({'failure_type': type(exc).__name__,
        'source': ':'.join([Path(traceback.extract_tb(exc.__traceback__)[-1].filename).name, str(traceback.extract_tb(exc.__traceback__)[-1].lineno), traceback.extract_tb(exc.__traceback__)[-1].name])}))
"""


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    "case",
    [
        "success",
        "hash",
        "mime",
        "owner",
        "nested",
        "after_quota",
        "after_upload",
        "missing_receipt",
        "released",
        "promotion_resolve",
        "promotion_reserve",
        "lock_order",
    ],
)
def test_retention_upload_uses_real_authority_and_exact_reviewed_source(tmp_path, request, backend, case):
    if backend == "postgres":
        _client, name = request.getfixturevalue("isolated_test_environment")
        database_url = os.environ["DATABASE_URL"]
        assert urlparse(database_url).path.lstrip("/") == name
    else:
        database_url = f"sqlite:///{tmp_path / 'users.db'}"
    env = _runtime_env(tmp_path, database_url, backend=backend)
    env.update(
        USAGE_QUOTAS_ENABLED="1",
        RETENTION_UPLOAD_CASE=case,
        USER_DB_BASE_DIR=str(tmp_path / "owned-user-db"),
        USER_DATA_BASE_PATH=str(tmp_path / "user-data"),
    )
    success = case in {"success", "promotion_resolve", "promotion_reserve", "lock_order"}
    exact = success or case in {"missing_receipt", "released"}
    candidate = case not in {"owner", "nested"}
    assert _run_runtime(tmp_path, env, SCRIPT) == {
        "refused": not success,
        "exact": exact,
        "replay": success,
        "charged": candidate and case != "released",
        "candidate": candidate,
        "claims": 0,
        "clean": True,
    }
