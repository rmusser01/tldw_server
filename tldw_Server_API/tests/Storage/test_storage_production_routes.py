"""Catch storage routes missing from the real full app and its public contract."""

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import _run_runtime, _runtime_env


def test_full_content_catalog_reaches_existing_storage_router():
    specs = [spec for spec in iter_content_router_specs() if spec.route_key == "storage"]
    assert len(specs) == 1
    assert specs[0].prefix == "/api/v1"


SCRIPT = r"""
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaOwner
from tldw_Server_API.app.core.testing import is_test_mode, is_explicit_pytest_runtime
from tldw_Server_API.app.services.storage_quota_service import get_storage_service, reset_storage_service

async def main():
    assert not is_test_mode() and not is_explicit_pytest_runtime()
    try:
        assert await setup_database()
        pool = await get_db_pool()
        users = AuthnzUsersRepo(pool)
        member = await users.create_user(username='storage-member', email='member@example.invalid', password_hash='fixture-hash', is_verified=True)
        admin = await users.create_user(username='storage-admin', email='admin@example.invalid', password_hash='fixture-hash', role='admin', is_verified=True)
        await users.assign_role_if_missing(user_id=admin, role_name='admin')
        assert await users.has_role_assignment(user_id=admin, role_name='admin')
        manager = APIKeyManager()
        member_key = (await manager.create_api_key(member, scope=['read', 'write']))['key']
        read_key = (await manager.create_api_key(member, scope='read'))['key']
        write_key = (await manager.create_api_key(member, scope='write'))['key']
        admin_key = (await manager.create_api_key(admin, scope='admin'))['key']
        service = await get_storage_service()
        await service.set_user_quota(member, 10)
        await service.set_user_quota(admin, 10)
        member_file = await service.register_generated_file(user_id=member, filename='member.txt', storage_path='member.txt', file_category='export', source_feature='export', file_size_bytes=1048576)
        other_file = await service.register_generated_file(user_id=admin, filename='admin.txt', storage_path='admin.txt', file_category='export', source_feature='export', file_size_bytes=3145728)
        user_uuid = str((await pool.fetchone('SELECT uuid FROM users WHERE id = ?', member))['uuid'])
        await service.reserve_native_bytes(NativeQuotaOwner(member, 'native:production-route-fixture', user_uuid), 'retained-chat', 2097152)
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
        from tldw_Server_API.app.api.v1.router_registry import register_router_specs
        from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://fixture') as client:
            usage = await client.get('/api/v1/storage/usage', headers={'X-API-KEY': member_key})
            if usage.status_code != 200:
                print('RUNTIME_RESULT=' + json.dumps({'usage_status': usage.status_code}))
                return
            body = usage.json()
            write_only_info = await client.get('/api/v1/storage/files/' + str(member_file['id']), headers={'X-API-KEY': write_key})
            read_only_info = await client.get('/api/v1/storage/files/' + str(member_file['id']), headers={'X-API-KEY': read_key})
            if write_only_info.status_code != 403:
                print('RUNTIME_RESULT=' + json.dumps({'write_only_read_status': write_only_info.status_code, 'read_only_read_status': read_only_info.status_code}))
                return
            read_paths = ['/files', '/files/least-accessed', '/files/' + str(member_file['id']) + '/download', '/folders', '/usage', '/usage/breakdown', '/trash']
            rejected_reads = [write_only_info.status_code] + [
                (await client.get('/api/v1/storage' + path, headers={'X-API-KEY': write_key})).status_code
                for path in read_paths
            ]
            read_only_usage = await client.get('/api/v1/storage/usage', headers={'X-API-KEY': read_key})
            breakdown = await client.get('/api/v1/storage/usage/breakdown', headers={'X-API-KEY': member_key})
            anonymous = await client.get('/api/v1/storage/usage')
            foreign = await client.get('/api/v1/storage/files/' + str(other_file['id']), headers={'X-API-KEY': member_key})
            rejected_admin = await client.put('/api/v1/storage/admin/quotas/user/' + str(member), headers={'X-API-KEY': member_key}, json={'quota_mb': 99})
            accepted_admin = await client.put('/api/v1/storage/admin/quotas/user/' + str(member), headers={'X-API-KEY': admin_key}, json={'quota_mb': 10})
            rejected_delete = await client.delete('/api/v1/storage/files/' + str(member_file['id']), headers={'X-API-KEY': read_key})
            files_repo = await service.get_generated_files_repo()
            current_file = await files_repo.get_file_by_id(member_file['id'])
            current_usage = await service.get_user_generated_files_usage(member)
            if rejected_delete.status_code != 403:
                print('RUNTIME_RESULT=' + json.dumps({'read_only_delete_status': rejected_delete.status_code, 'deleted': bool(current_file['is_deleted']), 'legacy_after': current_usage['legacy_quota_used_mb']}))
                return
            mutations = [
                ('POST', '/files/bulk-delete', {'file_ids': [member_file['id']]}),
                ('POST', '/files/bulk-move', {'file_ids': [member_file['id']], 'folder_tag': 'blocked'}),
                ('PATCH', '/files/' + str(member_file['id']), {'tags': ['blocked']}),
                ('POST', '/folders', {'name': 'blocked'}),
                ('POST', '/trash/restore/' + str(member_file['id']), None),
                ('DELETE', '/trash/' + str(member_file['id']), None),
            ]
            rejected_mutations = [
                (await client.request(method, '/api/v1/storage' + path, headers={'X-API-KEY': read_key}, json=payload)).status_code
                for method, path, payload in mutations
            ]
            accepted_delete = await client.delete('/api/v1/storage/files/' + str(member_file['id']), headers={'X-API-KEY': member_key})
            final_usage = await service.get_user_generated_files_usage(member)
        counts = Counter(route.path for route in iter_served_routes(app.routes) if 'GET' in getattr(route, 'methods', set()))
        specs = [spec for spec in iter_content_router_specs() if spec.route_key == 'storage']
        repeat = register_router_specs(app, specs)
        from tldw_Server_API.app.core.config import refresh_config_cache
        (Path(os.environ['TLDW_CONFIG_DIR']) / 'config.txt').write_text('[API-Routes]\ndisable = storage\n', encoding='utf-8')
        refresh_config_cache()
        disabled = FastAPI()
        disabled_count = register_router_specs(disabled, specs)
        print('RUNTIME_RESULT=' + json.dumps({
            'usage_status': usage.status_code,
            'usage': [body['legacy_quota_used_mb'], body['native_bytes'], body['native_mb'], body['quota_used_mb'], body['available_mb'], body['usage']['total_bytes']],
            'breakdown_status': breakdown.status_code,
            'breakdown': [breakdown.json()['legacy_quota_used_mb'], breakdown.json()['native_mb'], breakdown.json()['quota_used_mb'], breakdown.json()['available_mb']],
            'anonymous': anonymous.status_code,
            'foreign': foreign.status_code,
            'member_admin': rejected_admin.status_code,
            'admin': accepted_admin.status_code,
            'admin_usage': [accepted_admin.json()['quota']['legacy_used_mb'], accepted_admin.json()['quota']['native_used_mb'], accepted_admin.json()['quota']['used_mb']],
            'write_only_reads': rejected_reads,
            'read_key_success': read_only_info.status_code == 200 and read_only_info.json()['file']['id'] == member_file['id'] and read_only_usage.status_code == 200 and read_only_usage.json()['quota_used_mb'] == 3.0,
            'read_only_delete': rejected_delete.status_code,
            'read_only_unchanged': not current_file['is_deleted'] and current_usage['legacy_quota_used_mb'] == 1.0 and current_usage['native_bytes'] == 2097152,
            'read_only_mutations': rejected_mutations,
            'write_delete': accepted_delete.status_code,
            'final_usage': [final_usage['legacy_quota_used_mb'], final_usage['native_mb'], final_usage['quota_used_mb']],
            'once': counts['/api/v1/storage/usage'] == counts['/api/v1/storage/usage/breakdown'] == 1 and repeat == 0,
            'disabled': disabled_count == 0 and '/api/v1/storage/usage' not in disabled.openapi()['paths'],
        }))
    finally:
        await shutdown_all_audit_services()
        await reset_storage_service()
        await reset_db_pool()

asyncio.run(main())
"""


@pytest.mark.integration
def test_normal_full_app_storage_usage_authorization_and_policy(tmp_path: Path):
    env = _runtime_env(tmp_path, f"sqlite:///{tmp_path / 'users.db'}", backend="sqlite", mode="multi_user")
    env.update(
        USAGE_QUOTAS_ENABLED="1",
        DISABLE_HEAVY_STARTUP="1",
        USER_DB_BASE_DIR=str(tmp_path / "owned-user-db"),
        USER_DATA_BASE_PATH=str(tmp_path / "user-data"),
    )
    assert _run_runtime(tmp_path, env, SCRIPT) == {
        "usage_status": 200,
        "usage": [1.0, 2097152, 2.0, 3.0, 7.0, 1048576],
        "breakdown_status": 200,
        "breakdown": [1.0, 2.0, 3.0, 7.0],
        "anonymous": 401,
        "foreign": 403,
        "member_admin": 403,
        "admin": 200,
        "admin_usage": [1.0, 2.0, 3.0],
        "write_only_reads": [403, 403, 403, 403, 403, 403, 403, 403],
        "read_key_success": True,
        "read_only_delete": 403,
        "read_only_unchanged": True,
        "read_only_mutations": [403, 403, 403, 403, 403, 403],
        "write_delete": 200,
        "final_usage": [0.0, 2.0, 2.0],
        "once": True,
        "disabled": True,
    }
