from __future__ import annotations

import pytest

from tldw_Server_API.app.core.External_Sources.sync_adapter import FileSyncChange
from tldw_Server_API.app.core.External_Sources.sync_coordinator import SyncReconcileResult


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_incremental_sync_uses_delta_feed_and_advances_cursor(monkeypatch):
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
        list_changes_calls: list[str | None] = []
        download_calls: list[dict[str, object]] = []

        async def list_files(self, *args, **kwargs):
            raise AssertionError("delta sync should not fall back to recursive traversal")

        async def list_changes(
            self,
            account,
            *,
            cursor: str | None = None,
            page_size: int = 100,
        ):
            type(self).list_changes_calls.append(cursor)
            assert account["tokens"]["access_token"] == "tok"
            assert page_size == 100
            return (
                [
                    FileSyncChange(
                        event_type="content_updated",
                        remote_id="file-1",
                        remote_name="quarterly.txt",
                        remote_parent_id="folder-1",
                        remote_path="/finance/quarterly.txt",
                        remote_revision="rev-2",
                        remote_hash="hash-2",
                        metadata={
                            "mime_type": "text/plain",
                            "size": 128,
                            "modified_at": "2026-03-06T12:00:00Z",
                        },
                    )
                ],
                None,
                "cursor-2",
            )

        async def download_or_export(self, account, remote_id, *, metadata=None):
            type(self).download_calls.append(
                {
                    "account": dict(account),
                    "remote_id": remote_id,
                    "metadata": dict(metadata or {}),
                }
            )
            return b"Updated drive sync body"

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
        assert user_id == 42
        assert source_id == 99
        return {
            "id": source_id,
            "provider": "drive",
            "account_id": 123,
            "remote_id": "root",
            "type": "folder",
            "path": "/",
            "options": {"recursive": True},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        assert user_id == 42
        assert account_id == 123
        return {"access_token": "tok"}

    async def _fake_get_source_sync_state(db, *, source_id):
        assert source_id == 99
        return {"source_id": source_id, "cursor": "cursor-1", "cursor_kind": "drive_start_page_token"}

    sync_state_updates: list[dict[str, object]] = []

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    async def _fake_get_external_item_binding(db, *, source_id, provider, external_id):
        assert source_id == 99
        assert provider == "drive"
        assert external_id == "file-1"
        return {
            "id": 1,
            "source_id": source_id,
            "provider": provider,
            "external_id": external_id,
            "media_id": 77,
            "version": "rev-1",
            "hash": "hash-1",
            "sync_status": "active",
        }

    reconcile_calls: list[dict[str, object]] = []

    async def _fake_reconcile_file_change(
        connectors_db,
        media_db,
        *,
        source_id,
        provider,
        change,
        content=None,
        job_id=None,
    ):
        reconcile_calls.append(
            {
                "source_id": source_id,
                "provider": provider,
                "change": change,
                "content": content,
                "job_id": job_id,
                "media_db_type": type(media_db).__name__,
            }
        )
        return SyncReconcileResult(
            action="version_created",
            media_id=77,
            binding_id=1,
            current_version_number=2,
            sync_status="active",
        )

    created_mdb = {"db": None}

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.closed = False
            created_mdb["db"] = self

        def close_connection(self):
            self.closed = True

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    import tldw_Server_API.app.core.External_Sources.sync_coordinator as sync_coordinator

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(svc, "get_external_item_binding", _fake_get_external_item_binding)
    monkeypatch.setattr(sync_coordinator, "reconcile_file_change", _fake_reconcile_file_change)
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )
    monkeypatch.setattr(orgs, "list_memberships_for_user", lambda user_id: [])

    jm = FakeJM()

    await worker._process_import_job(
        jm,
        jid=1234,
        lease_id="lease-1",
        worker_id="worker-1",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert jm.completed["result"]["total"] == 1
    assert FakeDriveConn.list_changes_calls == ["cursor-1"]
    assert len(FakeDriveConn.download_calls) == 1
    assert FakeDriveConn.download_calls[0]["remote_id"] == "file-1"
    assert FakeDriveConn.download_calls[0]["metadata"] == {
        "mime_type": "text/plain",
        "size": 128,
        "modified_at": "2026-03-06T12:00:00Z",
    }
    assert len(reconcile_calls) == 1
    assert reconcile_calls[0]["source_id"] == 99
    assert reconcile_calls[0]["provider"] == "drive"
    assert reconcile_calls[0]["change"].remote_id == "file-1"
    assert reconcile_calls[0]["content"].text == "Updated drive sync body"
    assert reconcile_calls[0]["job_id"] == "1234"
    assert sync_state_updates[-1]["cursor"] == "cursor-2"
    assert sync_state_updates[-1]["cursor_kind"] == "drive_start_page_token"
    assert created_mdb["db"] is not None
    assert created_mdb["db"].closed is True


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("delta_event_type", ["created", "content_updated"])
async def test_worker_drive_created_delta_without_binding_still_reconciles(monkeypatch, delta_event_type):
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
        async def list_files(self, *args, **kwargs):
            raise AssertionError("delta sync should not fall back to recursive traversal")

        async def list_changes(
            self,
            account,
            *,
            cursor: str | None = None,
            page_size: int = 100,
        ):
            assert account["tokens"]["access_token"] == "tok"
            return (
                [
                    FileSyncChange(
                        event_type=delta_event_type,
                        remote_id="file-new",
                        remote_name="new-file.txt",
                        remote_parent_id="folder-1",
                        remote_path="/finance/new-file.txt",
                        remote_revision="rev-1",
                        remote_hash="hash-new",
                        metadata={
                            "mime_type": "text/plain",
                            "size": 64,
                            "modified_at": "2026-03-06T12:00:00Z",
                        },
                    )
                ],
                None,
                "cursor-2",
            )

        async def download_or_export(self, account, remote_id, *, metadata=None):
            assert remote_id == "file-new"
            return b"Brand new drive sync body"

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
            "options": {"recursive": True},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_get_source_sync_state(db, *, source_id):
        return {"source_id": source_id, "cursor": "cursor-1", "cursor_kind": "drive_start_page_token"}

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        return {"source_id": source_id, **updates}

    async def _fake_get_external_item_binding(db, *, source_id, provider, external_id):
        assert external_id == "file-new"
        return None

    reconcile_calls: list[dict[str, object]] = []

    async def _fake_reconcile_file_change(
        connectors_db,
        media_db,
        *,
        source_id,
        provider,
        change,
        content=None,
        job_id=None,
    ):
        reconcile_calls.append(
            {
                "source_id": source_id,
                "provider": provider,
                "change": change,
                "content": content,
                "job_id": job_id,
            }
        )
        return SyncReconcileResult(
            action="created",
            media_id=88,
            binding_id=5,
            current_version_number=1,
            sync_status="active",
        )

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    import tldw_Server_API.app.core.External_Sources.sync_coordinator as sync_coordinator

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(svc, "get_external_item_binding", _fake_get_external_item_binding)
    monkeypatch.setattr(sync_coordinator, "reconcile_file_change", _fake_reconcile_file_change)
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )
    monkeypatch.setattr(orgs, "list_memberships_for_user", lambda user_id: [])

    jm = FakeJM()

    await worker._process_import_job(
        jm,
        jid=5678,
        lease_id="lease-1",
        worker_id="worker-1",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert len(reconcile_calls) == 1
    assert reconcile_calls[0]["change"].event_type == "created"
    assert reconcile_calls[0]["content"].text == "Brand new drive sync body"
    assert reconcile_calls[0]["job_id"] == "5678"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_content_update_for_archived_binding_reconciles_as_restore(monkeypatch):
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
        download_calls: list[str] = []

        async def list_files(self, *args, **kwargs):
            raise AssertionError("delta sync should not fall back to recursive traversal")

        async def list_changes(
            self,
            account,
            *,
            cursor: str | None = None,
            page_size: int = 100,
        ):
            assert account["tokens"]["access_token"] == "tok"
            return (
                [
                    FileSyncChange(
                        event_type="content_updated",
                        remote_id="file-restore",
                        remote_name="restored-file.txt",
                        remote_parent_id="folder-1",
                        remote_path="/finance/restored-file.txt",
                        remote_revision="rev-3",
                        remote_hash="hash-restored",
                        metadata={
                            "mime_type": "text/plain",
                            "size": 64,
                            "modified_at": "2026-03-06T12:00:00Z",
                        },
                    )
                ],
                None,
                "cursor-2",
            )

        async def download_or_export(self, account, remote_id, *, metadata=None):
            type(self).download_calls.append(remote_id)
            raise AssertionError("restored bindings should be reconciled before content download")

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
            "options": {"recursive": True},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_get_source_sync_state(db, *, source_id):
        return {"source_id": source_id, "cursor": "cursor-1", "cursor_kind": "drive_start_page_token"}

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        return {"source_id": source_id, **updates}

    async def _fake_get_external_item_binding(db, *, source_id, provider, external_id):
        assert external_id == "file-restore"
        return {
            "id": 5,
            "source_id": source_id,
            "provider": provider,
            "external_id": external_id,
            "media_id": 88,
            "version": "rev-2",
            "hash": "hash-old",
            "sync_status": "archived_upstream_removed",
        }

    reconcile_calls: list[dict[str, object]] = []

    async def _fake_reconcile_file_change(
        connectors_db,
        media_db,
        *,
        source_id,
        provider,
        change,
        content=None,
        job_id=None,
    ):
        reconcile_calls.append(
            {
                "source_id": source_id,
                "provider": provider,
                "change": change,
                "content": content,
                "job_id": job_id,
            }
        )
        return SyncReconcileResult(
            action="restored",
            media_id=88,
            binding_id=5,
            current_version_number=1,
            sync_status="active",
        )

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    import tldw_Server_API.app.core.External_Sources.sync_coordinator as sync_coordinator

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(svc, "get_external_item_binding", _fake_get_external_item_binding)
    monkeypatch.setattr(sync_coordinator, "reconcile_file_change", _fake_reconcile_file_change)
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )
    monkeypatch.setattr(orgs, "list_memberships_for_user", lambda user_id: [])

    jm = FakeJM()

    await worker._process_import_job(
        jm,
        jid=5679,
        lease_id="lease-1",
        worker_id="worker-1",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 1
    assert FakeDriveConn.download_calls == []
    assert len(reconcile_calls) == 1
    assert reconcile_calls[0]["change"].event_type == "restored"
    assert reconcile_calls[0]["content"] is None
    assert reconcile_calls[0]["job_id"] == "5679"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_incremental_policy_block_marks_existing_binding_degraded(monkeypatch):
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
        async def list_files(self, *args, **kwargs):
            raise AssertionError("delta sync should not fall back to recursive traversal")

        async def list_changes(self, account, *, cursor: str | None = None, page_size: int = 100):
            return (
                [
                    FileSyncChange(
                        event_type="content_updated",
                        remote_id="file-1",
                        remote_name="blocked.exe",
                        remote_revision="rev-2",
                        remote_hash="hash-2",
                        metadata={
                            "mime_type": "application/octet-stream",
                            "size": 128,
                            "modified_at": "2026-03-06T12:00:00Z",
                        },
                    )
                ],
                None,
                "cursor-2",
            )

        async def download_or_export(self, account, remote_id, *, metadata=None):
            raise AssertionError("policy-blocked changes must not download content")

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
            "options": {"recursive": True},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_get_source_sync_state(db, *, source_id):
        return {"source_id": source_id, "cursor": "cursor-1", "cursor_kind": "drive_start_page_token"}

    sync_state_updates: list[dict[str, object]] = []

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    async def _fake_get_external_item_binding(db, *, source_id, provider, external_id):
        return {
            "id": 44,
            "source_id": source_id,
            "provider": provider,
            "external_id": external_id,
            "media_id": 77,
            "version": "rev-1",
            "hash": "hash-1",
            "sync_status": "active",
            "name": "blocked.exe",
            "mime": "application/octet-stream",
            "size": 32,
            "current_version_number": 1,
            "provider_metadata": {},
        }

    binding_updates: list[dict[str, object]] = []
    recorded_events: list[dict[str, object]] = []

    async def _fake_upsert_external_item_binding(db, *, source_id, provider, external_id, **updates):
        payload = {
            "id": 44,
            "source_id": source_id,
            "provider": provider,
            "external_id": external_id,
            **updates,
        }
        binding_updates.append(payload)
        return payload

    async def _fake_record_item_event(db, *, external_item_id, event_type, job_id=None, payload=None):
        recorded_events.append(
            {
                "external_item_id": external_item_id,
                "event_type": event_type,
                "job_id": job_id,
                "payload": dict(payload or {}),
            }
        )
        return recorded_events[-1]

    async def _unexpected_reconcile(*args, **kwargs):
        raise AssertionError("policy-blocked changes must not reconcile content")

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    import tldw_Server_API.app.core.External_Sources.policy as policy_mod
    import tldw_Server_API.app.core.External_Sources.sync_coordinator as sync_coordinator

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(svc, "get_external_item_binding", _fake_get_external_item_binding)
    monkeypatch.setattr(svc, "upsert_external_item_binding", _fake_upsert_external_item_binding)
    monkeypatch.setattr(svc, "record_item_event", _fake_record_item_event)
    monkeypatch.setattr(sync_coordinator, "reconcile_file_change", _unexpected_reconcile)
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )
    monkeypatch.setattr(orgs, "list_memberships_for_user", lambda user_id: [])
    monkeypatch.setattr(
        policy_mod,
        "get_default_policy_from_env",
        lambda org_id: {
            "allowed_export_formats": ["txt", "pdf"],
            "allowed_file_types": ["pdf", "txt", "text/plain"],
            "max_file_size_mb": 500,
        },
    )

    jm = FakeJM()

    await worker._process_import_job(
        jm,
        jid=6789,
        lease_id="lease-1",
        worker_id="worker-1",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 0
    assert jm.completed["result"]["failed"] == 1
    assert jm.completed["result"]["degraded"] == 1
    assert binding_updates[-1]["sync_status"] == "degraded"
    assert recorded_events[-1]["event_type"] == "ingest_failed"
    assert "policy" in recorded_events[-1]["payload"]["error"].lower()
    assert sync_state_updates[-1]["cursor"] == "cursor-2"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_drive_bootstrap_reconcile_failure_is_nonfatal(monkeypatch):
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
        async def list_files(self, account, parent_remote_id, *, page_size=50, cursor=None):
            assert account["tokens"]["access_token"] == "tok"
            assert parent_remote_id == "root"
            assert cursor is None
            return (
                [
                    {
                        "id": "file-bootstrap",
                        "name": "bootstrap.txt",
                        "mimeType": "text/plain",
                        "size": 32,
                    }
                ],
                None,
            )

        async def download_file(self, account, file_id, *, mime_type=None, export_mime=None):
            assert file_id == "file-bootstrap"
            return b"bootstrap body"

        async def get_start_page_token(self, account):
            return "cursor-bootstrap"

    class _DummyDb:
        async def execute(self, *args, **kwargs):
            return None

        async def fetchone(self, *args, **kwargs):
            return None

    class _DummyTx:
        async def __aenter__(self):
            return _DummyDb()

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
            "options": {"recursive": False},
            "email": "sync@example.com",
        }

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_should_ingest_item(db, **kwargs):
        return True

    async def _fake_get_source_sync_state(db, *, source_id):
        return {}

    sync_state_updates: list[dict[str, object]] = []

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    async def _fake_get_external_item_binding(db, *, source_id, provider, external_id):
        return None

    reconcile_calls: list[dict[str, object]] = []

    async def _fake_reconcile_file_change(
        connectors_db,
        media_db,
        *,
        source_id,
        provider,
        change,
        content=None,
        job_id=None,
    ):
        reconcile_calls.append(
            {
                "source_id": source_id,
                "provider": provider,
                "change": change,
                "content": content,
                "job_id": job_id,
            }
        )
        raise RuntimeError("ingest failed")

    class _FakeMDB:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    import tldw_Server_API.app.core.AuthNZ.database as dbmod
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs
    import tldw_Server_API.app.core.External_Sources as ext_pkg
    import tldw_Server_API.app.core.External_Sources.connectors_service as svc
    import tldw_Server_API.app.core.External_Sources.sync_coordinator as sync_coordinator

    monkeypatch.setattr(ext_pkg, "get_connector_by_name", lambda name: FakeDriveConn())
    monkeypatch.setattr(dbmod, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(svc, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(svc, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(svc, "should_ingest_item", _fake_should_ingest_item)
    monkeypatch.setattr(svc, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(svc, "upsert_source_sync_state", _fake_upsert_source_sync_state)
    monkeypatch.setattr(svc, "get_external_item_binding", _fake_get_external_item_binding)
    monkeypatch.setattr(sync_coordinator, "reconcile_file_change", _fake_reconcile_file_change)
    monkeypatch.setattr(
        worker,
        "create_media_database",
        lambda client_id, db_path=None: _FakeMDB(client_id, db_path=db_path),
        raising=False,
    )
    monkeypatch.setattr(orgs, "list_memberships_for_user", lambda user_id: [])

    jm = FakeJM()

    await worker._process_import_job(
        jm,
        jid=91011,
        lease_id="lease-1",
        worker_id="worker-1",
        source_id=99,
        user_id=42,
    )

    assert jm.completed is not None
    assert jm.completed["result"]["processed"] == 0
    assert jm.completed["result"]["failed"] == 1
    assert jm.completed["result"]["degraded"] == 0
    assert len(reconcile_calls) == 1
    assert reconcile_calls[0]["change"].event_type == "created"
    assert reconcile_calls[0]["job_id"] == "91011"
    assert sync_state_updates[0]["last_sync_started_at"] is not None
    assert sync_state_updates[-1]["cursor"] == "cursor-bootstrap"
    assert sync_state_updates[-1]["last_sync_succeeded_at"] is not None
