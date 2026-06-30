from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import SyncInvalidDomainError
from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate, SyncObjectState
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    db = SyncDatabase(sqlite_path=tmp_path / "sync_v2.db")
    store = SyncV2Store(db)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            domains=["notes.note", "chat.message"],
        )
    )
    return store


def test_object_state_upsert_and_fetch_is_keyed_by_dataset_domain_and_object(
    sync_store: SyncV2Store,
):
    note_state = sync_store.upsert_object_state(
        SyncObjectState(
            dataset_id="dataset-1",
            domain="notes.note",
            object_id="shared-id",
            object_revision=1,
            object_hash="sha256:note-v1",
            latest_server_cursor=11,
            deleted=False,
        )
    )
    message_state = sync_store.upsert_object_state(
        SyncObjectState(
            dataset_id="dataset-1",
            domain="chat.message",
            object_id="shared-id",
            object_revision=1,
            object_hash="sha256:message-v1",
            latest_server_cursor=12,
            deleted=False,
        )
    )

    assert sync_store.get_object_state("dataset-1", "notes.note", "shared-id") == note_state
    assert (
        sync_store.get_object_state("dataset-1", "chat.message", "shared-id")
        == message_state
    )


def test_object_state_updates_revision_hash_cursor_and_deleted_flag(
    sync_store: SyncV2Store,
):
    sync_store.upsert_object_state(
        SyncObjectState(
            dataset_id="dataset-1",
            domain="notes.note",
            object_id="note-1",
            object_revision=1,
            object_hash="sha256:note-v1",
            latest_server_cursor=11,
            deleted=False,
        )
    )

    updated = sync_store.upsert_object_state(
        SyncObjectState(
            dataset_id="dataset-1",
            domain="notes.note",
            object_id="note-1",
            object_revision=2,
            object_hash="sha256:note-v2",
            latest_server_cursor=19,
            deleted=True,
        )
    )

    assert updated.object_revision == 2
    assert updated.object_hash == "sha256:note-v2"
    assert updated.latest_server_cursor == 19
    assert updated.deleted is True
    assert sync_store.get_object_state("dataset-1", "notes.note", "note-1") == updated


def test_object_state_rejects_unenrolled_domain(sync_store: SyncV2Store):
    with pytest.raises(SyncInvalidDomainError):
        sync_store.upsert_object_state(
            SyncObjectState(
                dataset_id="dataset-1",
                domain="attachment.ref",
                object_id="attachment-1",
                object_revision=1,
                object_hash="sha256:attachment",
                latest_server_cursor=20,
                deleted=False,
            )
        )
