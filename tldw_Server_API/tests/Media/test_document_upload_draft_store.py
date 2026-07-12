"""Tests for shared chat document upload draft persistence."""

from __future__ import annotations

import threading
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_drafts import (
    DocumentUploadDraftQuotaError,
    DocumentUploadDraftStore,
)

pytestmark = pytest.mark.unit


def test_drafts_are_visible_across_store_instances(tmp_path) -> None:
    """Separate worker-local service instances must share persisted drafts."""
    db_path = tmp_path / "document-upload-drafts.db"
    now = datetime(2026, 7, 9, tzinfo=timezone.utc)
    first_store = DocumentUploadDraftStore(db_path=db_path, clock=lambda: now)
    second_store = DocumentUploadDraftStore(db_path=db_path, clock=lambda: now)

    created = first_store.create(owner="1", payload={"draft": "shared"})

    loaded = second_store.get(owner="1", draft_id=created.draft_id)
    assert loaded is not None
    assert loaded.payload == {"draft": "shared"}


def test_expired_drafts_are_removed_before_quota_is_enforced(tmp_path) -> None:
    """Expired rows must not consume owner quota."""
    current_time = [datetime(2026, 7, 9, tzinfo=timezone.utc)]
    store = DocumentUploadDraftStore(
        db_path=tmp_path / "document-upload-drafts.db",
        ttl_seconds=1,
        max_drafts_per_owner=1,
        clock=lambda: current_time[0],
    )
    store.create(owner="1", payload={"draft": "expired"})
    current_time[0] += timedelta(seconds=2)

    created = store.create(owner="1", payload={"draft": "replacement"})

    assert store.get(owner="1", draft_id=created.draft_id) is not None


def test_get_filters_expired_drafts_without_cleanup_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Reads must filter expired rows without issuing cleanup writes."""
    current_time = [datetime(2026, 7, 9, tzinfo=timezone.utc)]
    store = DocumentUploadDraftStore(
        db_path=tmp_path / "document-upload-drafts.db",
        ttl_seconds=1,
        clock=lambda: current_time[0],
    )
    created = store.create(owner="1", payload={"draft": "expired"})
    current_time[0] += timedelta(seconds=2)
    monkeypatch.setattr(
        store,
        "_cleanup_expired",
        lambda *_args: pytest.fail("get() performed a cleanup write"),
    )

    assert store.get(owner="1", draft_id=created.draft_id) is None


def test_store_operations_explicitly_close_connections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Every operation must close its SQLite connection deterministically."""
    store = DocumentUploadDraftStore(db_path=tmp_path / "document-upload-drafts.db")
    original_connect = store._connect
    opened = 0
    closed = 0

    class TrackedConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            nonlocal opened
            opened += 1
            self._connection = connection

        def __getattr__(self, name: str):
            return getattr(self._connection, name)

        def __enter__(self):
            self._connection.__enter__()
            return self

        def __exit__(self, *args: object):
            return self._connection.__exit__(*args)

        def close(self) -> None:
            nonlocal closed
            closed += 1
            self._connection.close()

    monkeypatch.setattr(store, "_connect", lambda: TrackedConnection(original_connect()))

    created = store.create(owner="1", payload={"draft": "tracked"})
    assert store.get(owner="1", draft_id=created.draft_id) is not None
    assert store.delete(owner="1", draft_id=created.draft_id)
    assert opened == closed == 3


def test_global_draft_quota_is_shared_by_all_owners(tmp_path) -> None:
    """The global quota must count drafts created by separate owners."""
    store = DocumentUploadDraftStore(
        db_path=tmp_path / "document-upload-drafts.db",
        max_drafts_total=1,
    )
    store.create(owner="1", payload={"draft": "first"})

    with pytest.raises(DocumentUploadDraftQuotaError, match="Too many active"):
        store.create(owner="2", payload={"draft": "second"})


def test_concurrent_creates_enforce_owner_quota_without_timing_sleeps(tmp_path) -> None:
    """The quota check and insert must be atomic across worker instances."""
    db_path = tmp_path / "document-upload-drafts.db"
    stores = [
        DocumentUploadDraftStore(db_path=db_path, max_drafts_per_owner=1),
        DocumentUploadDraftStore(db_path=db_path, max_drafts_per_owner=1),
    ]
    barrier = threading.Barrier(3, timeout=5)
    created_ids: list[str] = []
    quota_errors: list[DocumentUploadDraftQuotaError] = []
    unexpected_errors: list[BaseException] = []

    def create_draft(store: DocumentUploadDraftStore) -> None:
        barrier.wait()
        try:
            created_ids.append(store.create(owner="1", payload={"draft": "race"}).draft_id)
        except DocumentUploadDraftQuotaError as exc:
            quota_errors.append(exc)
        except BaseException as exc:  # pragma: no cover - asserted below
            unexpected_errors.append(exc)

    threads = [threading.Thread(target=create_draft, args=(store,), daemon=True) for store in stores]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert unexpected_errors == []
    assert len(created_ids) == 1
    assert len(quota_errors) == 1
