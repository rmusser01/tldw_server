from __future__ import annotations

import os
from typing import Tuple

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.core.External_Sources import connectors_service as svc
from tldw_Server_API.app.core.External_Sources.sync_adapter import FileSyncWebhookSubscription


@pytest.fixture()
def connectors_client() -> Tuple[TestClient, dict]:
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("ROUTES_STABLE_ONLY", "false")
    os.environ["ROUTES_ENABLE"] = "connectors"
    os.environ.setdefault("AUTH_MODE", "single_user")
    os.environ.setdefault("TESTING", "true")

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.endpoints import connectors as connectors_router

    paths = {route.path for route in iter_served_routes(app.routes)}
    if "/api/v1/connectors/sources" not in paths:
        app.include_router(connectors_router.router, prefix="/api/v1", tags=["connectors"])

    api_key = get_settings().SINGLE_USER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    client = TestClient(app)
    return client, headers


@pytest.fixture
async def connectors_db(tmp_path):
    db = await aiosqlite.connect(tmp_path / "connectors.sqlite3")
    db.row_factory = aiosqlite.Row
    db._is_sqlite = True
    try:
        yield db
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_record_webhook_receipt_dedupes_by_provider_and_key(connectors_db):
    first = await svc.record_webhook_receipt(
        connectors_db,
        provider="onedrive",
        receipt_key="sub-1:event-1",
        source_id=77,
    )
    duplicate = await svc.record_webhook_receipt(
        connectors_db,
        provider="onedrive",
        receipt_key="sub-1:event-1",
        source_id=77,
    )
    different_provider = await svc.record_webhook_receipt(
        connectors_db,
        provider="drive",
        receipt_key="sub-1:event-1",
        source_id=77,
    )

    assert first is True
    assert duplicate is False
    assert different_provider is True


@pytest.mark.integration
def test_onedrive_webhook_validation_echoes_token(connectors_client):
    client, _headers = connectors_client

    response = client.get(
        "/api/v1/connectors/providers/onedrive/webhook",
        params={"validationToken": "token-123"},
    )

    assert response.status_code == 200
    assert response.text == "token-123"


@pytest.mark.integration
def test_onedrive_webhook_enqueues_incremental_sync_and_dedupes(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    seen_receipts: set[str] = set()
    queued_jobs: list[dict[str, object]] = []

    async def _fake_get_source_by_webhook_subscription(db, *, provider, subscription_id):
        assert provider == "onedrive"
        assert subscription_id == "sub-1"
        return {
            "id": 77,
            "provider": "onedrive",
            "user_id": 42,
            "enabled": True,
            "webhook_metadata": {"clientState": "state-123"},
        }

    async def _fake_record_webhook_receipt(db, *, provider, receipt_key, source_id=None, payload_hash=None):
        assert provider == "onedrive"
        if receipt_key in seen_receipts:
            return False
        seen_receipts.add(receipt_key)
        return True

    async def _fake_create_import_job(user_id, source_id, *, request_id=None, job_type="import"):
        queued_jobs.append(
            {
                "user_id": user_id,
                "source_id": source_id,
                "request_id": request_id,
                "job_type": job_type,
            }
        )
        return {
            "id": "job-1",
            "source_id": source_id,
            "type": job_type,
            "status": "queued",
            "progress_pct": 0,
            "counts": {"processed": 0, "skipped": 0, "failed": 0},
        }

    monkeypatch.setattr(ep, "get_source_by_webhook_subscription", _fake_get_source_by_webhook_subscription)
    monkeypatch.setattr(ep, "record_webhook_receipt", _fake_record_webhook_receipt)
    monkeypatch.setattr(ep, "create_import_job", _fake_create_import_job)

    payload = {"value": [{"subscriptionId": "sub-1", "changeType": "updated", "clientState": "state-123"}]}

    first = client.post(
        "/api/v1/connectors/providers/onedrive/webhook",
        json=payload,
        headers=headers,
    )
    second = client.post(
        "/api/v1/connectors/providers/onedrive/webhook",
        json=payload,
        headers=headers,
    )

    assert first.status_code == 202, first.text
    assert first.json()["status"] == "queued"
    assert first.json()["queued_jobs"] == 1
    assert first.json()["source_ids"] == [77]

    assert second.status_code == 202, second.text
    assert second.json()["status"] == "duplicate"
    assert second.json()["queued_jobs"] == 0
    assert second.json()["duplicate_notifications"] == 1

    assert len(queued_jobs) == 1
    assert queued_jobs[0]["user_id"] == 42
    assert queued_jobs[0]["source_id"] == 77
    assert queued_jobs[0]["job_type"] == "incremental_sync"


@pytest.mark.integration
def test_onedrive_webhook_ignores_invalid_client_state(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_get_source_by_webhook_subscription(db, *, provider, subscription_id):
        return {
            "id": 77,
            "provider": "onedrive",
            "user_id": 42,
            "enabled": True,
            "webhook_metadata": {"clientState": "expected-state"},
        }

    async def _unexpected_record_webhook_receipt(*args, **kwargs):
        raise AssertionError("invalid webhook notifications must be rejected before dedupe is recorded")

    async def _unexpected_create_import_job(*args, **kwargs):
        raise AssertionError("invalid webhook notifications must not enqueue sync jobs")

    monkeypatch.setattr(ep, "get_source_by_webhook_subscription", _fake_get_source_by_webhook_subscription)
    monkeypatch.setattr(ep, "record_webhook_receipt", _unexpected_record_webhook_receipt)
    monkeypatch.setattr(ep, "create_import_job", _unexpected_create_import_job)

    response = client.post(
        "/api/v1/connectors/providers/onedrive/webhook",
        json={"value": [{"subscriptionId": "sub-1", "changeType": "updated", "clientState": "wrong-state"}]},
        headers=headers,
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "ignored"
    assert response.json()["queued_jobs"] == 0
    assert response.json()["ignored_notifications"] == 1


@pytest.mark.integration
def test_drive_webhook_enqueues_incremental_sync_and_dedupes(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    seen_receipts: set[str] = set()
    queued_jobs: list[dict[str, object]] = []

    async def _fake_get_source_by_webhook_subscription(db, *, provider, subscription_id):
        assert provider == "drive"
        assert subscription_id == "drive-chan-1"
        return {
            "id": 88,
            "provider": "drive",
            "user_id": 24,
            "enabled": True,
            "webhook_metadata": {"clientState": "drive-state-123"},
        }

    async def _fake_record_webhook_receipt(db, *, provider, receipt_key, source_id=None, payload_hash=None):
        assert provider == "drive"
        if receipt_key in seen_receipts:
            return False
        seen_receipts.add(receipt_key)
        return True

    async def _fake_create_import_job(user_id, source_id, *, request_id=None, job_type="import"):
        queued_jobs.append(
            {
                "user_id": user_id,
                "source_id": source_id,
                "request_id": request_id,
                "job_type": job_type,
            }
        )
        return {
            "id": "job-drive-1",
            "source_id": source_id,
            "type": job_type,
            "status": "queued",
            "progress_pct": 0,
            "counts": {"processed": 0, "skipped": 0, "failed": 0},
        }

    monkeypatch.setattr(ep, "get_source_by_webhook_subscription", _fake_get_source_by_webhook_subscription)
    monkeypatch.setattr(ep, "record_webhook_receipt", _fake_record_webhook_receipt)
    monkeypatch.setattr(ep, "create_import_job", _fake_create_import_job)

    request_headers = dict(headers)
    request_headers.update(
        {
            "X-Goog-Channel-Id": "drive-chan-1",
            "X-Goog-Channel-Token": "drive-state-123",
            "X-Goog-Message-Number": "1",
            "X-Goog-Resource-State": "change",
            "X-Goog-Resource-Id": "drive-resource-1",
        }
    )

    first = client.post("/api/v1/connectors/providers/drive/webhook", headers=request_headers)
    second = client.post("/api/v1/connectors/providers/drive/webhook", headers=request_headers)

    assert first.status_code == 202, first.text
    assert first.json()["status"] == "queued"
    assert first.json()["queued_jobs"] == 1
    assert first.json()["source_ids"] == [88]

    assert second.status_code == 202, second.text
    assert second.json()["status"] == "duplicate"
    assert second.json()["queued_jobs"] == 0
    assert second.json()["duplicate_notifications"] == 1

    assert len(queued_jobs) == 1
    assert queued_jobs[0]["user_id"] == 24
    assert queued_jobs[0]["source_id"] == 88
    assert queued_jobs[0]["job_type"] == "incremental_sync"


@pytest.mark.integration
def test_drive_webhook_ignores_invalid_channel_token(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_get_source_by_webhook_subscription(db, *, provider, subscription_id):
        return {
            "id": 88,
            "provider": "drive",
            "user_id": 24,
            "enabled": True,
            "webhook_metadata": {"clientState": "expected-drive-state"},
        }

    async def _unexpected_record_webhook_receipt(*args, **kwargs):
        raise AssertionError("invalid drive notifications must be rejected before dedupe is recorded")

    async def _unexpected_create_import_job(*args, **kwargs):
        raise AssertionError("invalid drive notifications must not enqueue sync jobs")

    monkeypatch.setattr(ep, "get_source_by_webhook_subscription", _fake_get_source_by_webhook_subscription)
    monkeypatch.setattr(ep, "record_webhook_receipt", _unexpected_record_webhook_receipt)
    monkeypatch.setattr(ep, "create_import_job", _unexpected_create_import_job)

    request_headers = dict(headers)
    request_headers.update(
        {
            "X-Goog-Channel-Id": "drive-chan-1",
            "X-Goog-Channel-Token": "wrong-drive-state",
            "X-Goog-Message-Number": "1",
            "X-Goog-Resource-State": "change",
            "X-Goog-Resource-Id": "drive-resource-1",
        }
    )

    response = client.post("/api/v1/connectors/providers/drive/webhook", headers=request_headers)

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "ignored"
    assert response.json()["queued_jobs"] == 0
    assert response.json()["ignored_notifications"] == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_subscription_renewal_updates_sync_state(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *args, **kwargs):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    class FakeOneDriveConn:
        async def renew_webhook(self, account, *, subscription):
            assert account["tokens"]["access_token"] == "tok"
            assert subscription.subscription_id == "sub-1"
            assert subscription.expires_at == "2026-03-07T00:00:00Z"
            return FileSyncWebhookSubscription(
                subscription_id="sub-1",
                expires_at="2026-03-10T00:00:00Z",
                metadata={"expirationDateTime": "2026-03-10T00:00:00Z"},
            )

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "onedrive",
            "account_id": 123,
            "remote_id": "root",
            "type": "folder",
            "path": "/",
            "options": {},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_get_source_sync_state(db, *, source_id):
        return {
            "source_id": source_id,
            "webhook_subscription_id": "sub-1",
            "webhook_expires_at": "2026-03-07T00:00:00Z",
            "webhook_status": "active",
        }

    sync_state_updates: list[dict[str, object]] = []

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc_mod

    async def _fake_list_memberships_for_user(_user_id: int):
        return []

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeOneDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc_mod, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc_mod, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc_mod, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc_mod, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=501,
        lease_id="lease-renew",
        worker_id="worker-renew",
        source_id=99,
        user_id=42,
        job_type="subscription_renewal",
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert jm.completed["result"]["subscription_id"] == "sub-1"
    assert jm.completed["result"]["webhook_expires_at"] == "2026-03-10T00:00:00Z"
    assert sync_state_updates[-1]["webhook_status"] == "active"
    assert sync_state_updates[-1]["webhook_subscription_id"] == "sub-1"
    assert sync_state_updates[-1]["webhook_expires_at"] == "2026-03-10T00:00:00Z"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_subscription_renewal_round_trips_webhook_metadata(monkeypatch):
    import tldw_Server_API.app.services.connectors_worker as worker

    class FakeJM:
        def __init__(self):
            self.completed = None

        def renew_job_lease(self, *args, **kwargs):
            return None

        def complete_job(
            self,
            jid,
            result=None,
            worker_id=None,
            lease_id=None,
            completion_token=None,
        ):
            self.completed = {"jid": jid, "result": result}

    class FakeDriveConn:
        async def renew_webhook(self, account, *, subscription):
            assert account["tokens"]["access_token"] == "tok"
            assert subscription.subscription_id == "drive-chan-1"
            assert subscription.metadata["resourceId"] == "drive-resource-1"
            assert subscription.metadata["pageToken"] == "drive-cursor-1"
            assert subscription.metadata["callback_url"] == "https://example.com/api/v1/connectors/providers/drive/webhook"
            return FileSyncWebhookSubscription(
                subscription_id="drive-chan-2",
                expires_at="2026-03-10T00:00:00Z",
                metadata={
                    "resourceId": "drive-resource-2",
                    "pageToken": "drive-cursor-2",
                    "callback_url": "https://example.com/api/v1/connectors/providers/drive/webhook",
                    "clientState": "drive-state-1",
                },
            )

    class _DummyTx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyPool:
        def transaction(self):
            return _DummyTx()

    async def _fake_get_db_pool():
        return _DummyPool()

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "drive",
            "account_id": 123,
            "remote_id": "root",
            "type": "folder",
            "path": "/",
            "options": {},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_get_source_sync_state(db, *, source_id):
        return {
            "source_id": source_id,
            "webhook_subscription_id": "drive-chan-1",
            "webhook_expires_at": "2026-03-07T00:00:00Z",
            "webhook_status": "active",
            "webhook_metadata": {
                "resourceId": "drive-resource-1",
                "pageToken": "drive-cursor-1",
                "callback_url": "https://example.com/api/v1/connectors/providers/drive/webhook",
                "clientState": "drive-state-1",
            },
        }

    sync_state_updates: list[dict[str, object]] = []

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc_mod

    async def _fake_list_memberships_for_user(_user_id: int):
        return []

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc_mod, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc_mod, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc_mod, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc_mod, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(orgs, "list_memberships_for_user", _fake_list_memberships_for_user)

    jm = FakeJM()
    await worker._process_import_job(
        jm,
        jid=4321,
        lease_id="lease-1",
        worker_id="worker-1",
        source_id=99,
        user_id=42,
        job_type="subscription_renewal",
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert sync_state_updates[-1]["webhook_subscription_id"] == "drive-chan-2"
    assert sync_state_updates[-1]["webhook_expires_at"] == "2026-03-10T00:00:00Z"
    assert sync_state_updates[-1]["webhook_metadata"]["resourceId"] == "drive-resource-2"
    assert sync_state_updates[-1]["webhook_metadata"]["pageToken"] == "drive-cursor-2"
