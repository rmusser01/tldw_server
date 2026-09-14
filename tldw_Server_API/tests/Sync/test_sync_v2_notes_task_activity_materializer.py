from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import materializers
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncDatasetCreate,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskActivityV1,
    notes_task_activity_object_hash,
    parse_notes_task_activity_tombstone_v1,
    parse_notes_task_activity_v1,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

OWNER_ID = "owner-user-1"
DATASET_ID = "dataset-1"
NOTE_ID = "22222222-2222-4222-8222-222222222222"
OTHER_NOTE_ID = "33333333-3333-4333-8333-333333333333"
TASK_ID = "11111111-1111-4111-8111-111111111111"
ACTIVITY_ID = "55555555-5555-4555-8555-555555555555"
NOW = "2026-08-13T10:00:00+00:00"
DELETED_AT = "2026-08-13T11:00:00+00:00"


def _materializer(note_db: CharactersRAGDB):
    materializer_type = getattr(materializers, "NotesTaskActivityMaterializer", None)
    assert materializer_type is not None, "NotesTaskActivityMaterializer is not implemented"
    return materializer_type(note_db)


def _activity(**overrides: object) -> NotesTaskActivityV1:
    raw: dict[str, object] = {
        "activity_id": ACTIVITY_ID,
        "note_id": NOTE_ID,
        "task_id": None,
        "event_type": "projection_drift",
        "actor_type": "system",
        "actor_id": None,
        "source_device_id": None,
        "client_occurred_at": NOW,
        "source_kind": "repair",
        "corrects_activity_id": None,
        "old_value": None,
        "new_value": {"reason_code": "missing_marker_base"},
        "metadata": {"repair_generation": 7},
    }
    raw.update(overrides)
    return parse_notes_task_activity_v1(
        raw,
        owner_user_id=OWNER_ID,
        bound_actor_type=str(raw["actor_type"]),
        bound_actor_id=raw["actor_id"],
        authenticated_device_id=None,
        trusted_server_origin=True,
    )


def _create(payload: NotesTaskActivityV1, *, suffix: str) -> SyncEnvelopeCreate:
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=f"activity-{suffix}",
        domain="notes.task_activity",
        operation="upsert",
        object_id=payload.activity_id,
        parent_id=payload.note_id,
        device_id="44444444-4444-4444-8444-444444444444",
        object_revision=1,
        entity_version=1,
        adapter_version=1,
        schema_version=1,
        payload=payload.model_dump(mode="json"),
        payload_hash=notes_task_activity_object_hash(
            payload,
            revision=1,
            deleted=False,
        ),
        created_at_client=NOW,
        routing_metadata={},
        status="accepted",
    )


def _tombstone(
    created: SyncEnvelope,
    original: NotesTaskActivityV1,
) -> SyncEnvelopeCreate:
    payload = parse_notes_task_activity_tombstone_v1(
        {
            "note_id": original.note_id,
            "task_id": original.task_id,
            "deleted_at": DELETED_AT,
            "delete_reason": "correction",
        },
        envelope_created_at_client=DELETED_AT,
        original_activity=original,
    )
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id="activity-tombstone",
        domain="notes.task_activity",
        operation="tombstone",
        object_id=original.activity_id,
        parent_id=original.note_id,
        device_id="44444444-4444-4444-8444-444444444444",
        base_server_cursor=created.server_cursor,
        base_object_revision=created.object_revision,
        base_object_hash=created.payload_hash,
        object_revision=2,
        entity_version=2,
        adapter_version=1,
        schema_version=1,
        payload=payload.model_dump(mode="json"),
        payload_hash=notes_task_activity_object_hash(
            payload,
            revision=2,
            deleted=True,
            activity_id=original.activity_id,
            original_create_hash=created.payload_hash,
        ),
        created_at_client=DELETED_AT,
        routing_metadata={},
        status="accepted",
    )


def _insert_internal(store: SyncV2Store, create: SyncEnvelopeCreate) -> SyncEnvelope:
    with store.db.backend.transaction() as conn:
        return store.db._insert_envelope_in_transaction(create, connection=conn)


@pytest.fixture()
def materialization_stack(tmp_path: Path):
    note_db = CharactersRAGDB(tmp_path / "notes-task-activity-product.db", client_id=OWNER_ID)
    note_db.note_store.add_note(NOTE_ID, "body", note_id=NOTE_ID)
    note_db.note_store.add_note(OTHER_NOTE_ID, "other", note_id=OTHER_NOTE_ID)
    note_db.bind_local_task_graph_to_dataset(
        owner_user_id=OWNER_ID,
        target_dataset_id=DATASET_ID,
    )
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "notes-task-activity-sync.db"))
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id=OWNER_ID,
            domains=["notes.note"],
        )
    )
    sync_store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
        (json.dumps(["notes.note", "notes.task_activity"]), DATASET_ID),
    )
    try:
        yield note_db, sync_store
    finally:
        note_db.close_connection()


def _get(note_db: CharactersRAGDB, activity_id: str = ACTIVITY_ID):
    return note_db.task_store.get_sync_task_activity(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        activity_id=activity_id,
    )


def _count(note_db: CharactersRAGDB) -> int:
    return int(
        note_db.execute_query(
            "SELECT COUNT(*) AS count FROM task_events "
            "WHERE owner_user_id = ? AND dataset_id = ?",
            (OWNER_ID, DATASET_ID),
        ).fetchone()["count"]
    )


def test_materializer_inserts_exact_activity_once_and_repairs_split_commit(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    payload = _activity()
    incoming = _insert_internal(sync_store, _create(payload, suffix="create"))

    class _FailObjectStateOnce:
        def __init__(self, wrapped: SyncV2Store) -> None:
            self.wrapped = wrapped
            self.failed = False

        def upsert_object_state(self, *_args, **_kwargs) -> None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("injected Sync status failure")
            self.wrapped.upsert_object_state(*_args, **_kwargs)

        def __getattr__(self, name: str):
            return getattr(self.wrapped, name)

    assert _materializer(note_db).apply(incoming, store=_FailObjectStateOnce(sync_store)).status == "failed"
    row = _get(note_db)
    assert row is not None
    assert row["note_id"] == NOTE_ID
    assert row["task_id"] is None
    assert row["event_type"] == "projection_drift"
    assert row["actor_type"] == "system"
    assert row["source_kind"] == "repair"
    assert row["client_occurred_at"] == NOW
    assert row["old_value_json"] is None
    assert row["new_value_json"] == {"reason_code": "missing_marker_base"}
    assert row["sync_revision"] == 1
    assert row["sync_object_hash"] == incoming.payload_hash
    assert row["sync_server_cursor"] == incoming.server_cursor
    assert not bool(row["deleted"])

    assert _materializer(note_db).apply(incoming, store=sync_store).status == "applied"
    assert _count(note_db) == 1
    state = sync_store.get_object_state(DATASET_ID, "notes.task_activity", ACTIVITY_ID)
    assert state is not None and state.latest_server_cursor == incoming.server_cursor


def test_materializer_rejects_changed_product_replay(materialization_stack) -> None:
    note_db, sync_store = materialization_stack
    incoming = _insert_internal(sync_store, _create(_activity(), suffix="changed"))
    assert _materializer(note_db).apply(incoming, store=sync_store).status == "applied"
    note_db.execute_query(
        "UPDATE task_events SET event_type = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        ("projection_linked", OWNER_ID, DATASET_ID, ACTIVITY_ID),
        commit=True,
    )

    result = _materializer(note_db).apply(incoming, store=sync_store)

    assert result.status == "conflict"
    assert result.conflict_type == "notes_task_activity_product_conflict"
    assert _count(note_db) == 1


def test_materializer_applies_one_way_revision_two_tombstone(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    original = _activity()
    created = _insert_internal(sync_store, _create(original, suffix="before-delete"))
    assert _materializer(note_db).apply(created, store=sync_store).status == "applied"
    deleted = _insert_internal(sync_store, _tombstone(created, original))

    assert _materializer(note_db).apply(deleted, store=sync_store).status == "applied"
    row = _get(note_db)
    assert row is not None and bool(row["deleted"])
    assert row["sync_revision"] == 2
    assert row["sync_object_hash"] == deleted.payload_hash
    assert row["sync_server_cursor"] == deleted.server_cursor
    assert row["deleted_at"] == DELETED_AT
    assert row["delete_reason"] == "correction"
    assert _materializer(note_db).apply(deleted, store=sync_store).status == "applied"
    assert _materializer(note_db).apply(created, store=sync_store).status == "conflict"
    assert _count(note_db) == 1


def test_activity_store_enforces_exact_note_task_and_dataset_scope(
    materialization_stack,
) -> None:
    note_db, _sync_store = materialization_stack
    task = note_db.task_store.create_task(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        note_id=NOTE_ID,
        text="Scoped task",
        task_id=TASK_ID,
        projection_status="unlinked",
    )
    payload = _activity(task_id=TASK_ID)
    canonical_hash = notes_task_activity_object_hash(payload, revision=1, deleted=False)

    with note_db.transaction() as conn:
        created = note_db.task_store.create_sync_task_activity(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            payload=payload,
            sync_object_hash=canonical_hash,
            sync_server_cursor=41,
            conn=conn,
        )
    assert created["task_id"] == task["id"]
    assert note_db.task_store.get_sync_task_activity(
        owner_user_id=OWNER_ID,
        dataset_id="other-dataset",
        activity_id=ACTIVITY_ID,
    ) is None

    wrong_parent = _activity(
        activity_id="66666666-6666-4666-8666-666666666666",
        note_id=OTHER_NOTE_ID,
        task_id=TASK_ID,
    )
    with pytest.raises(ConflictError):
        with note_db.transaction() as conn:
            note_db.task_store.create_sync_task_activity(
                owner_user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                payload=wrong_parent,
                sync_object_hash=notes_task_activity_object_hash(
                    wrong_parent,
                    revision=1,
                    deleted=False,
                ),
                sync_server_cursor=42,
                conn=conn,
            )


def test_activity_page_uses_cursor_id_keyset_and_caps_at_one_thousand(
    materialization_stack,
) -> None:
    note_db, _sync_store = materialization_stack
    with note_db.transaction() as conn:
        for ordinal in range(1, 1_003):
            activity_id = str(UUID(int=ordinal, version=4))
            payload = _activity(activity_id=activity_id)
            note_db.task_store.create_sync_task_activity(
                owner_user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                payload=payload,
                sync_object_hash=notes_task_activity_object_hash(
                    payload,
                    revision=1,
                    deleted=False,
                ),
                sync_server_cursor=50 if ordinal <= 2 else 50 + ordinal,
                conn=conn,
            )

    first = note_db.task_store.page_sync_task_activity(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        limit=5_000,
    )
    assert len(first) == 1_000
    assert [(row["sync_server_cursor"], row["id"]) for row in first[:3]] == [
        (50, str(UUID(int=1, version=4))),
        (50, str(UUID(int=2, version=4))),
        (53, str(UUID(int=3, version=4))),
    ]
    after = first[-1]
    second = note_db.task_store.page_sync_task_activity(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        after_server_cursor=after["sync_server_cursor"],
        after_activity_id=after["id"],
        limit=10,
    )
    assert len(second) == 2


def test_materializer_product_failure_rolls_back_insert(
    materialization_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, sync_store = materialization_stack
    incoming = _insert_internal(sync_store, _create(_activity(), suffix="rollback"))

    def fail_after_write(_stage: str) -> None:
        raise RuntimeError("injected product failure")

    monkeypatch.setattr(
        type(note_db.task_store),
        "_sync_task_activity_materialization_checkpoint",
        staticmethod(fail_after_write),
    )

    result = _materializer(note_db).apply(incoming, store=sync_store)

    assert result.status == "failed"
    assert _get(note_db) is None
