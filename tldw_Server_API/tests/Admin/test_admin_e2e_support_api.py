from __future__ import annotations

import base64
import importlib
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
from tldw_Server_API.app.services.admin_e2e_support_service import (
    _resolve_safe_backup_dir,
)
from tldw_Server_API.app.services.registration_service import reset_registration_service

pytestmark = pytest.mark.integration

ADMIN_E2E_SUPPORT_HEADER = 'X-TLDW-Admin-E2E-Key'
ADMIN_E2E_SUPPORT_KEY = 'playwright-admin-e2e-support-key'
ADMIN_E2E_WEBHOOK_KEY_ID = 'admin-e2e-primary'
ADMIN_E2E_WEBHOOK_KEY = base64.b64encode(b'w' * 32).decode('ascii')


def _set_fixture_password_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TLDW_ADMIN_E2E_ADMIN_PASSWORD', 'AdminPass123!')
    monkeypatch.setenv('TLDW_ADMIN_E2E_OWNER_PASSWORD', 'AdminPass123!')
    monkeypatch.setenv('TLDW_ADMIN_E2E_SUPER_ADMIN_PASSWORD', 'AdminPass123!')
    monkeypatch.setenv('TLDW_ADMIN_E2E_MEMBER_PASSWORD', 'MemberPass123!')
    monkeypatch.setenv('TLDW_ADMIN_E2E_REQUESTER_PASSWORD', 'RequesterPass123!')


def _set_webhook_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TLDW_ADMIN_WEBHOOKS_MODE', 'on')
    monkeypatch.setenv(
        'TLDW_ADMIN_WEBHOOK_KEYS_JSON',
        json.dumps({ADMIN_E2E_WEBHOOK_KEY_ID: ADMIN_E2E_WEBHOOK_KEY}),
    )
    monkeypatch.setenv(
        'TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID',
        ADMIN_E2E_WEBHOOK_KEY_ID,
    )


async def _reset_auth_runtime() -> None:
    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()


def _admin_e2e_headers(
    extra_headers: dict[str, str] | None = None,
    support_key: str = ADMIN_E2E_SUPPORT_KEY,
) -> dict[str, str]:
    headers = {ADMIN_E2E_SUPPORT_HEADER: support_key}
    if extra_headers:
        headers.update(extra_headers)
    return headers


@pytest_asyncio.fixture
async def client_without_e2e_support(tmp_path, monkeypatch):
    db_path = tmp_path / 'authnz_no_e2e.db'
    monkeypatch.setenv('AUTH_MODE', 'multi_user')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('JWT_SECRET_KEY', 'playwright-test-secret-1234567890')
    monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
    monkeypatch.setenv('DEFER_HEAVY_STARTUP', 'true')
    monkeypatch.setenv('TEST_MODE', 'true')
    monkeypatch.delenv('ENABLE_ADMIN_E2E_TEST_MODE', raising=False)

    await _reset_auth_runtime()

    import tldw_Server_API.app.main as app_main

    app = importlib.reload(app_main).app
    with TestClient(app) as client:
        yield client

    await _reset_auth_runtime()


@pytest_asyncio.fixture
async def e2e_client(tmp_path, monkeypatch):
    db_path = tmp_path / 'authnz_with_e2e.db'
    jobs_db_path = tmp_path / 'jobs_with_e2e.db'
    monitoring_db_path = tmp_path / 'monitoring_with_e2e.db'
    backup_root = tmp_path / 'backups'
    user_db_base_dir = tmp_path / 'user_dbs'
    monkeypatch.setenv('AUTH_MODE', 'multi_user')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('JOBS_DB_PATH', str(jobs_db_path))
    monkeypatch.setenv('JWT_SECRET_KEY', 'playwright-test-secret-1234567890')
    monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
    monkeypatch.setenv('DEFER_HEAVY_STARTUP', 'true')
    monkeypatch.setenv('TEST_MODE', 'true')
    monkeypatch.setenv('ENABLE_ADMIN_E2E_TEST_MODE', 'true')
    monkeypatch.setenv('TLDW_ADMIN_E2E_SUPPORT_KEY', ADMIN_E2E_SUPPORT_KEY)
    monkeypatch.setenv('TLDW_DB_BACKUP_PATH', str(backup_root))
    monkeypatch.setenv('USER_DB_BASE_DIR', str(user_db_base_dir))
    monkeypatch.setenv('MONITORING_ALERTS_DB', str(monitoring_db_path))
    _set_fixture_password_env(monkeypatch)

    await _reset_auth_runtime()

    import tldw_Server_API.app.main as app_main

    app = importlib.reload(app_main).app
    with TestClient(app) as client:
        yield client

    await _reset_auth_runtime()


@pytest_asyncio.fixture
async def e2e_client_without_support_key(tmp_path, monkeypatch):
    db_path = tmp_path / 'authnz_with_e2e_missing_support_key.db'
    jobs_db_path = tmp_path / 'jobs_with_e2e_missing_support_key.db'
    monitoring_db_path = tmp_path / 'monitoring_with_e2e_missing_support_key.db'
    backup_root = tmp_path / 'backups'
    user_db_base_dir = tmp_path / 'user_dbs'
    monkeypatch.setenv('AUTH_MODE', 'multi_user')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('JOBS_DB_PATH', str(jobs_db_path))
    monkeypatch.setenv('JWT_SECRET_KEY', 'playwright-test-secret-1234567890')
    monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
    monkeypatch.setenv('DEFER_HEAVY_STARTUP', 'true')
    monkeypatch.setenv('TEST_MODE', 'true')
    monkeypatch.setenv('ENABLE_ADMIN_E2E_TEST_MODE', 'true')
    monkeypatch.delenv('TLDW_ADMIN_E2E_SUPPORT_KEY', raising=False)
    monkeypatch.setenv('TLDW_DB_BACKUP_PATH', str(backup_root))
    monkeypatch.setenv('USER_DB_BASE_DIR', str(user_db_base_dir))
    monkeypatch.setenv('MONITORING_ALERTS_DB', str(monitoring_db_path))
    _set_fixture_password_env(monkeypatch)

    await _reset_auth_runtime()

    import tldw_Server_API.app.main as app_main

    app = importlib.reload(app_main).app
    with TestClient(app) as client:
        yield client

    await _reset_auth_runtime()


@pytest_asyncio.fixture
async def single_user_e2e_client(tmp_path, monkeypatch):
    db_path = tmp_path / 'authnz_single_user_e2e.db'
    jobs_db_path = tmp_path / 'jobs_single_user_e2e.db'
    monitoring_db_path = tmp_path / 'monitoring_single_user_e2e.db'
    backup_root = tmp_path / 'backups'
    user_db_base_dir = tmp_path / 'user_dbs'
    monkeypatch.setenv('AUTH_MODE', 'single_user')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('JOBS_DB_PATH', str(jobs_db_path))
    monkeypatch.setenv('JWT_SECRET_KEY', 'playwright-test-secret-1234567890')
    monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
    monkeypatch.setenv('DEFER_HEAVY_STARTUP', 'true')
    monkeypatch.setenv('TEST_MODE', 'true')
    monkeypatch.setenv('ENABLE_ADMIN_E2E_TEST_MODE', 'true')
    monkeypatch.setenv('TLDW_ADMIN_E2E_SUPPORT_KEY', ADMIN_E2E_SUPPORT_KEY)
    monkeypatch.setenv('TLDW_DB_BACKUP_PATH', str(backup_root))
    monkeypatch.setenv('USER_DB_BASE_DIR', str(user_db_base_dir))
    monkeypatch.setenv('MONITORING_ALERTS_DB', str(monitoring_db_path))
    monkeypatch.setenv('SINGLE_USER_API_KEY', 'single-user-admin-key')
    monkeypatch.setenv('SINGLE_USER_TEST_API_KEY', 'single-user-admin-key')
    _set_fixture_password_env(monkeypatch)

    await _reset_auth_runtime()

    import tldw_Server_API.app.main as app_main

    app = importlib.reload(app_main).app
    with TestClient(app) as client:
        yield client

    await _reset_auth_runtime()


@pytest_asyncio.fixture
async def e2e_client_without_fixture_passwords(tmp_path, monkeypatch):
    db_path = tmp_path / 'authnz_with_missing_fixture_passwords.db'
    jobs_db_path = tmp_path / 'jobs_with_missing_fixture_passwords.db'
    monitoring_db_path = tmp_path / 'monitoring_with_missing_fixture_passwords.db'
    backup_root = tmp_path / 'backups'
    user_db_base_dir = tmp_path / 'user_dbs'
    monkeypatch.setenv('AUTH_MODE', 'multi_user')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('JOBS_DB_PATH', str(jobs_db_path))
    monkeypatch.setenv('JWT_SECRET_KEY', 'playwright-test-secret-1234567890')
    monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
    monkeypatch.setenv('DEFER_HEAVY_STARTUP', 'true')
    monkeypatch.setenv('TEST_MODE', 'true')
    monkeypatch.setenv('ENABLE_ADMIN_E2E_TEST_MODE', 'true')
    monkeypatch.setenv('TLDW_ADMIN_E2E_SUPPORT_KEY', ADMIN_E2E_SUPPORT_KEY)
    monkeypatch.setenv('TLDW_DB_BACKUP_PATH', str(backup_root))
    monkeypatch.setenv('USER_DB_BASE_DIR', str(user_db_base_dir))
    monkeypatch.setenv('MONITORING_ALERTS_DB', str(monitoring_db_path))
    monkeypatch.delenv('TLDW_ADMIN_E2E_ADMIN_PASSWORD', raising=False)
    monkeypatch.delenv('TLDW_ADMIN_E2E_OWNER_PASSWORD', raising=False)
    monkeypatch.delenv('TLDW_ADMIN_E2E_SUPER_ADMIN_PASSWORD', raising=False)
    monkeypatch.delenv('TLDW_ADMIN_E2E_MEMBER_PASSWORD', raising=False)
    monkeypatch.delenv('TLDW_ADMIN_E2E_REQUESTER_PASSWORD', raising=False)

    await _reset_auth_runtime()

    import tldw_Server_API.app.main as app_main

    app = importlib.reload(app_main).app
    with TestClient(app) as client:
        yield client

    await _reset_auth_runtime()


@pytest_asyncio.fixture
async def e2e_client_with_unsafe_backup_path(tmp_path, monkeypatch):
    db_path = tmp_path / 'authnz_with_unsafe_backup_path.db'
    jobs_db_path = tmp_path / 'jobs_with_unsafe_backup_path.db'
    monitoring_db_path = tmp_path / 'monitoring_with_unsafe_backup_path.db'
    user_db_base_dir = tmp_path / 'user_dbs'
    unsafe_backup_root = Path.home() / 'unsafe-admin-e2e-backups'
    monkeypatch.setenv('AUTH_MODE', 'multi_user')
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_path}')
    monkeypatch.setenv('JOBS_DB_PATH', str(jobs_db_path))
    monkeypatch.setenv('JWT_SECRET_KEY', 'playwright-test-secret-1234567890')
    monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
    monkeypatch.setenv('DEFER_HEAVY_STARTUP', 'true')
    monkeypatch.setenv('TEST_MODE', 'true')
    monkeypatch.setenv('ENABLE_ADMIN_E2E_TEST_MODE', 'true')
    monkeypatch.setenv('TLDW_ADMIN_E2E_SUPPORT_KEY', ADMIN_E2E_SUPPORT_KEY)
    monkeypatch.setenv('TLDW_DB_BACKUP_PATH', str(unsafe_backup_root))
    monkeypatch.setenv('USER_DB_BASE_DIR', str(user_db_base_dir))
    monkeypatch.setenv('MONITORING_ALERTS_DB', str(monitoring_db_path))
    _set_fixture_password_env(monkeypatch)

    await _reset_auth_runtime()

    import tldw_Server_API.app.main as app_main

    app = importlib.reload(app_main).app
    with TestClient(app) as client:
        yield client

    await _reset_auth_runtime()


def test_admin_e2e_routes_are_unavailable_without_flag(client_without_e2e_support):
    response = client_without_e2e_support.post('/api/v1/test-support/admin-e2e/reset')
    assert response.status_code == 404


def test_admin_e2e_routes_require_support_header(e2e_client):
    response = e2e_client.post('/api/v1/test-support/admin-e2e/reset')

    assert response.status_code == 403
    assert response.json()['detail'] == 'admin_e2e_support_access_denied'


def test_admin_e2e_routes_reject_wrong_support_header(e2e_client):
    response = e2e_client.post(
        '/api/v1/test-support/admin-e2e/reset',
        headers={ADMIN_E2E_SUPPORT_HEADER: 'wrong-key'},
    )

    assert response.status_code == 403
    assert response.json()['detail'] == 'admin_e2e_support_access_denied'


def test_admin_e2e_routes_fail_closed_without_configured_support_key(
    e2e_client_without_support_key,
):
    response = e2e_client_without_support_key.post(
        '/api/v1/test-support/admin-e2e/reset',
        headers=_admin_e2e_headers(),
    )

    assert response.status_code == 500
    assert response.json()['detail'] == 'admin_e2e_support_key_not_configured'


def test_prepare_admin_webhooks_requires_on_mode(e2e_client):
    response = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )

    assert response.status_code == 409
    assert response.json()['detail'] == 'admin_e2e_webhook_mode_not_on'


def test_prepare_admin_webhooks_requires_valid_key_ring(e2e_client, monkeypatch):
    monkeypatch.setenv('TLDW_ADMIN_WEBHOOKS_MODE', 'on')
    monkeypatch.delenv('TLDW_ADMIN_WEBHOOK_KEYS_JSON', raising=False)
    monkeypatch.delenv('TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID', raising=False)

    response = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )

    assert response.status_code == 500
    assert response.json()['detail'] == 'admin_e2e_webhook_key_unavailable'


def test_prepare_admin_webhooks_completes_fresh_state_idempotently(
    e2e_client,
    monkeypatch,
):
    _set_webhook_runtime_env(monkeypatch)

    first = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )
    second = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json() == {
        'ok': True,
        'phase': 'complete',
        'active_primary_key_id': ADMIN_E2E_WEBHOOK_KEY_ID,
        'state_revision': 2,
    }
    assert second.json() == first.json()


def test_prepare_admin_webhooks_rejects_completed_key_mismatch(
    e2e_client,
    monkeypatch,
):
    _set_webhook_runtime_env(monkeypatch)
    prepared = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )
    assert prepared.status_code == 200, prepared.text

    replacement_id = 'admin-e2e-replacement'
    replacement_key = base64.b64encode(b'x' * 32).decode('ascii')
    monkeypatch.setenv(
        'TLDW_ADMIN_WEBHOOK_KEYS_JSON',
        json.dumps({replacement_id: replacement_key}),
    )
    monkeypatch.setenv('TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID', replacement_id)

    mismatch = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )

    assert mismatch.status_code == 409
    assert mismatch.json()['detail'] == 'admin_e2e_webhook_key_mismatch'


def test_prepare_admin_webhooks_rejects_unrelated_completed_state(
    e2e_client,
    monkeypatch,
    tmp_path,
):
    _set_webhook_runtime_env(monkeypatch)
    prepared = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )
    assert prepared.status_code == 200, prepared.text

    with sqlite3.connect(tmp_path / 'authnz_with_e2e.db') as connection:
        connection.execute(
            "UPDATE admin_webhook_migration_state "
            "SET import_operation_id = ? WHERE singleton_id = 1",
            ('whmig_' + ('d' * 32),),
        )

    mismatch = e2e_client.post(
        '/api/v1/test-support/admin-e2e/prepare-admin-webhooks',
        headers=_admin_e2e_headers(),
    )

    assert mismatch.status_code == 409
    assert mismatch.json()['detail'] == 'admin_e2e_webhook_state_mismatch'


def test_admin_e2e_seed_returns_stable_fixture_ids(e2e_client):
    response = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'dsr_jwt_admin'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['users']['admin']['id']
    assert payload['users']['admin']['key']
    assert payload['fixtures']['alerts'][0]['alert_id']
    assert payload['fixtures']['alerts'][0]['alert_identity'].startswith('alert:')
    assert payload['fixtures']['alerts'][0]['message'] == 'CPU high'


def test_admin_e2e_seed_returns_debug_role_principals(e2e_client):
    response = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'jwt_admin'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['users']['owner']['key'] == 'jwt_owner'
    assert payload['users']['super_admin']['key'] == 'jwt_super_admin'


def test_admin_e2e_seed_requires_explicit_fixture_passwords(e2e_client_without_fixture_passwords):
    response = e2e_client_without_fixture_passwords.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'jwt_admin'},
    )

    assert response.status_code == 500
    assert response.json()['detail'] == 'admin_e2e_fixture_secret_missing'


def test_admin_e2e_seed_returns_single_user_login_key(single_user_e2e_client):
    response = single_user_e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'single_user_admin'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['users']['admin']['username'] == 'single_user'
    assert payload['users']['admin']['key'] == 'single-user-admin-key'
    assert payload['fixtures']['alerts'][0]['alert_id']


def test_single_user_api_key_can_read_users_me(single_user_e2e_client):
    response = single_user_e2e_client.get(
        '/api/v1/users/me',
        headers={'X-API-KEY': 'single-user-admin-key'},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['username'] == 'single_user'


def test_admin_e2e_reset_rejects_unsafe_backup_paths(e2e_client_with_unsafe_backup_path):
    response = e2e_client_with_unsafe_backup_path.post(
        '/api/v1/test-support/admin-e2e/reset',
        headers=_admin_e2e_headers(),
    )

    assert response.status_code == 500
    assert response.json()['detail'] == 'admin_e2e_backup_path_must_be_temp_scoped'


def test_admin_e2e_backup_resolver_rejects_temp_root(monkeypatch):
    monkeypatch.setenv('TLDW_DB_BACKUP_PATH', tempfile.gettempdir())

    with pytest.raises(HTTPException) as exc_info:
        _resolve_safe_backup_dir()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == 'admin_e2e_backup_path_must_be_temp_scoped'


def test_admin_e2e_bootstrap_jwt_session_returns_cookie_payload(e2e_client):
    seed = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'jwt_admin'},
    ).json()
    response = e2e_client.post(
        '/api/v1/test-support/admin-e2e/bootstrap-jwt-session',
        headers=_admin_e2e_headers(),
        json={'principal_key': seed['users']['admin']['key']},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['cookies'][0]['name'] == 'access_token'


def test_admin_e2e_dsr_seed_supports_real_preview(e2e_client):
    seed = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'dsr_jwt_admin'},
    ).json()
    bootstrap = e2e_client.post(
        '/api/v1/test-support/admin-e2e/bootstrap-jwt-session',
        headers=_admin_e2e_headers(),
        json={'principal_key': seed['users']['admin']['key']},
    ).json()
    access_token = next(
        cookie['value']
        for cookie in bootstrap['cookies']
        if cookie['name'] == 'access_token'
    )

    response = e2e_client.post(
        '/api/v1/admin/data-subject-requests/preview',
        headers={'Authorization': f'Bearer {access_token}'},
        json={'requester_identifier': seed['users']['requester']['email'], 'request_type': 'access'},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['resolved_user_id'] == seed['users']['requester']['id']
    assert payload['counts']['media_records'] > 0
    assert payload['counts']['chat_messages'] > 0
    assert payload['counts']['notes'] > 0
    assert payload['counts']['audit_events'] > 0


def test_admin_e2e_reset_clears_backup_schedules(e2e_client):
    seed = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'dsr_jwt_admin'},
    ).json()
    bootstrap = e2e_client.post(
        '/api/v1/test-support/admin-e2e/bootstrap-jwt-session',
        headers=_admin_e2e_headers(),
        json={'principal_key': seed['users']['admin']['key']},
    ).json()
    access_token = next(
        cookie['value']
        for cookie in bootstrap['cookies']
        if cookie['name'] == 'access_token'
    )
    headers = {'Authorization': f'Bearer {access_token}'}

    create = e2e_client.post(
        '/api/v1/admin/backup-schedules',
        headers=headers,
        json={
            'dataset': 'media',
            'target_user_id': seed['users']['requester']['id'],
            'frequency': 'daily',
            'time_of_day': '02:00',
            'retention_count': 3,
        },
    )
    assert create.status_code == 200, create.text

    listed_before = e2e_client.get('/api/v1/admin/backup-schedules', headers=headers)
    assert listed_before.status_code == 200, listed_before.text
    assert listed_before.json()['total'] == 1

    reset = e2e_client.post(
        '/api/v1/test-support/admin-e2e/reset',
        headers=_admin_e2e_headers(),
    )
    assert reset.status_code == 200, reset.text

    listed_after = e2e_client.get('/api/v1/admin/backup-schedules', headers=headers)
    assert listed_after.status_code == 200, listed_after.text
    assert listed_after.json()['total'] == 0


def test_admin_e2e_reset_clears_monitoring_authority_state(e2e_client):
    seed = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'jwt_admin'},
    ).json()
    bootstrap = e2e_client.post(
        '/api/v1/test-support/admin-e2e/bootstrap-jwt-session',
        headers=_admin_e2e_headers(),
        json={'principal_key': seed['users']['admin']['key']},
    ).json()
    access_token = next(
        cookie['value']
        for cookie in bootstrap['cookies']
        if cookie['name'] == 'access_token'
    )
    headers = {'Authorization': f'Bearer {access_token}'}

    created_rule = e2e_client.post(
        '/api/v1/admin/monitoring/alert-rules',
        headers=headers,
        json={
            'metric': 'cpu',
            'operator': '>',
            'threshold': 91,
            'duration_minutes': 15,
            'severity': 'critical',
            'enabled': True,
        },
    )
    assert created_rule.status_code == 200, created_rule.text

    assign = e2e_client.post(
        f"/api/v1/admin/monitoring/alerts/{seed['fixtures']['alerts'][0]['alert_identity']}/assign",
        headers=headers,
        json={'assigned_to_user_id': seed['users']['admin']['id']},
    )
    assert assign.status_code == 200, assign.text

    history_before = e2e_client.get('/api/v1/admin/monitoring/alerts/history?limit=20', headers=headers)
    assert history_before.status_code == 200, history_before.text
    assert history_before.json()['items']

    reset = e2e_client.post(
        '/api/v1/test-support/admin-e2e/reset',
        headers=_admin_e2e_headers(),
    )
    assert reset.status_code == 200, reset.text

    rules_after = e2e_client.get('/api/v1/admin/monitoring/alert-rules', headers=headers)
    assert rules_after.status_code == 200, rules_after.text
    assert rules_after.json()['items'] == []

    history_after = e2e_client.get('/api/v1/admin/monitoring/alerts/history?limit=20', headers=headers)
    assert history_after.status_code == 200, history_after.text
    assert history_after.json()['items'] == []


def test_admin_e2e_run_due_backup_schedules_processes_scheduled_run(e2e_client):
    seed = e2e_client.post(
        '/api/v1/test-support/admin-e2e/seed',
        headers=_admin_e2e_headers(),
        json={'scenario': 'dsr_jwt_admin'},
    ).json()
    bootstrap = e2e_client.post(
        '/api/v1/test-support/admin-e2e/bootstrap-jwt-session',
        headers=_admin_e2e_headers(),
        json={'principal_key': seed['users']['admin']['key']},
    ).json()
    access_token = next(
        cookie['value']
        for cookie in bootstrap['cookies']
        if cookie['name'] == 'access_token'
    )
    headers = {'Authorization': f'Bearer {access_token}'}

    create = e2e_client.post(
        '/api/v1/admin/backup-schedules',
        headers=headers,
        json={
            'dataset': 'media',
            'target_user_id': seed['users']['requester']['id'],
            'frequency': 'daily',
            'time_of_day': '02:00',
            'retention_count': 3,
        },
    )
    assert create.status_code == 200, create.text

    trigger = e2e_client.post(
        '/api/v1/test-support/admin-e2e/run-due-backup-schedules',
        headers=_admin_e2e_headers(),
    )
    assert trigger.status_code == 200, trigger.text
    trigger_payload = trigger.json()
    assert trigger_payload['triggered_runs'] == 1

    listed = e2e_client.get('/api/v1/admin/backup-schedules', headers=headers)
    assert listed.status_code == 200, listed.text
    item = listed.json()['items'][0]
    assert item['last_status'] == 'succeeded'
    assert item['last_run_at']

    backups = e2e_client.get(
        '/api/v1/admin/backups',
        headers=headers,
        params={'dataset': 'media', 'user_id': seed['users']['requester']['id']},
    )
    assert backups.status_code == 200, backups.text
    assert backups.json()['items']
