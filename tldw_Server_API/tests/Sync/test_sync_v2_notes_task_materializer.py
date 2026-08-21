from __future__ import annotations

import json
from pathlib import Path

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
    NotesTaskV1Payload,
    notes_task_object_hash,
    parse_notes_task_v1,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

OWNER_ID = "owner-user-1"
DATASET_ID = "dataset-1"
TASK_ID = "11111111-1111-4111-8111-111111111111"
NOTE_ID = "22222222-2222-4222-8222-222222222222"
MISSING_NOTE_ID = "33333333-3333-4333-8333-333333333333"
DEVICE_ID = "44444444-4444-4444-8444-444444444444"
NOW = "2026-08-13T10:00:00+00:00"


def _materializer(note_db: CharactersRAGDB):
    materializer_type = getattr(materializers, "NotesTaskMaterializer", None)
    assert materializer_type is not None, "NotesTaskMaterializer is not implemented"
    return materializer_type(note_db)


def _payload(**overrides: object) -> NotesTaskV1Payload:
    raw: dict[str, object] = {
        "task_id": TASK_ID,
        "note_id": NOTE_ID,
        "title": "Prepare launch notes",
        "description": "Confirm the final release checklist.",
        "status": "open",
        "completed_at": None,
        "priority": "high",
        "due_date": "2026-08-31",
        "estimate": "90m",
        "recurrence": {
            "frequency": "weekly",
            "interval": 2,
            "by_weekday": ["mo", "we", "fr"],
            "until": "2026-12-31",
            "state": "active",
            "occurrence_index": 7,
        },
        "assignee_id": OWNER_ID,
        "tags": ["alpha", "Zulu"],
        "custom": {"board.column": "Next"},
    }
    raw.update(overrides)
    return parse_notes_task_v1(raw, owner_user_id=OWNER_ID)


def _create(
    *,
    payload: NotesTaskV1Payload | None = None,
    operation: str = "upsert",
    base: SyncEnvelope | None = None,
    restore: bool = False,
    suffix: str,
) -> SyncEnvelopeCreate:
    canonical = payload or _payload()
    revision = 1 if base is None else int(base.object_revision or 0) + 1
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=f"task-{suffix}",
        domain="notes.task",
        operation=operation,
        object_id=canonical.task_id,
        parent_id=canonical.note_id,
        device_id=DEVICE_ID,
        base_server_cursor=base.server_cursor if base is not None else None,
        base_object_revision=base.object_revision if base is not None else None,
        base_object_hash=base.payload_hash if base is not None else None,
        object_revision=revision,
        entity_version=revision,
        adapter_version=1,
        schema_version=1,
        payload=canonical.model_dump(mode="json"),
        payload_hash=notes_task_object_hash(
            canonical,
            revision=revision,
            deleted=operation == "tombstone",
        ),
        created_at_client=NOW,
        routing_metadata={"restore_intent": True} if restore else {},
        status="accepted",
    )


def _insert_internal(store: SyncV2Store, create: SyncEnvelopeCreate) -> SyncEnvelope:
    with store.db.backend.transaction() as conn:
        return store.db._insert_envelope_in_transaction(create, connection=conn)


@pytest.fixture()
def materialization_stack(tmp_path: Path):
    note_db = CharactersRAGDB(tmp_path / "notes-task-product.db", client_id=OWNER_ID)
    note_db.note_store.add_note(NOTE_ID, "body", note_id=NOTE_ID)
    note_db.bind_local_task_graph_to_dataset(
        owner_user_id=OWNER_ID,
        target_dataset_id=DATASET_ID,
    )
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "notes-task-sync.db"))
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id=OWNER_ID,
            domains=["notes.note"],
        )
    )
    sync_store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
        (json.dumps(["notes.note", "notes.task"]), DATASET_ID),
    )
    try:
        yield note_db, sync_store
    finally:
        note_db.close_connection()


def _task(note_db: CharactersRAGDB) -> dict[str, object] | None:
    return note_db.task_store.get_task(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        task_id=TASK_ID,
        include_deleted=True,
    )


def _event_count(note_db: CharactersRAGDB) -> int:
    return int(
        note_db.execute_query(
            "SELECT COUNT(*) AS count FROM task_events "
            "WHERE owner_user_id = ? AND dataset_id = ?",
            (OWNER_ID, DATASET_ID),
        ).fetchone()["count"]
    )


def test_materializer_applies_full_task_lifecycle_without_activity(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = _materializer(note_db)

    created = _insert_internal(sync_store, _create(suffix="create"))
    assert materializer.apply(created, store=sync_store).status == "applied"
    task = _task(note_db)
    assert task is not None
    assert task["text"] == "Prepare launch notes"
    assert task["status"] == "open"
    assert task["projection_status"] == "unlinked"
    assert task["canonical_revision"] == 1
    assert task["canonical_hash"] == created.payload_hash
    assert task["metadata_json"] == {
        "assignee_id": OWNER_ID,
        "custom": {"board.column": "Next"},
        "description": "Confirm the final release checklist.",
        "due_date": "2026-08-31",
        "estimate": "90m",
        "priority": "high",
        "recurrence": {
            "by_weekday": ["mo", "we", "fr"],
            "frequency": "weekly",
            "interval": 2,
            "occurrence_index": 7,
            "state": "active",
            "until": "2026-12-31",
        },
        "tags": ["alpha", "Zulu"],
    }

    completed_payload = _payload(
        status="done",
        completed_at="2026-08-13T11:00:00+00:00",
    )
    completed = _insert_internal(
        sync_store,
        _create(payload=completed_payload, base=created, suffix="complete"),
    )
    assert materializer.apply(completed, store=sync_store).status == "applied"
    task = _task(note_db)
    assert task is not None
    assert task["status"] == "done"
    assert str(task["completed_at"]) == "2026-08-13T11:00:00+00:00"
    assert task["canonical_revision"] == 2

    reopened = _insert_internal(
        sync_store,
        _create(payload=_payload(), base=completed, suffix="reopen"),
    )
    assert materializer.apply(reopened, store=sync_store).status == "applied"
    assert _task(note_db)["completed_at"] is None

    tombstone = _insert_internal(
        sync_store,
        _create(operation="tombstone", base=reopened, suffix="delete"),
    )
    assert materializer.apply(tombstone, store=sync_store).status == "applied"
    task = _task(note_db)
    assert task is not None and bool(task["deleted"])
    assert task["projection_status"] == "deleted"

    restored_payload = _payload(title="Restored launch notes")
    restored = _insert_internal(
        sync_store,
        _create(
            payload=restored_payload,
            base=tombstone,
            restore=True,
            suffix="restore",
        ),
    )
    assert materializer.apply(restored, store=sync_store).status == "applied"
    task = _task(note_db)
    assert task is not None and not bool(task["deleted"])
    assert task["text"] == "Restored launch notes"
    assert task["projection_status"] == "unlinked"
    assert task["canonical_revision"] == 5
    assert _event_count(note_db) == 0


def test_materializer_preserves_separate_rest_version(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = _materializer(note_db)
    created = _insert_internal(sync_store, _create(suffix="version-create"))
    assert materializer.apply(created, store=sync_store).status == "applied"
    note_db.execute_query(
        "UPDATE note_tasks SET version = 9 WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        (OWNER_ID, DATASET_ID, TASK_ID),
        commit=True,
    )

    updated = _insert_internal(
        sync_store,
        _create(
            payload=_payload(title="Canonical update"),
            base=created,
            suffix="version-update",
        ),
    )
    assert materializer.apply(updated, store=sync_store).status == "applied"
    task = _task(note_db)
    assert task is not None
    assert task["version"] == 10
    assert task["canonical_revision"] == 2


def test_split_commit_replay_repairs_sync_state_without_rewriting_product(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = _materializer(note_db)
    created = _insert_internal(sync_store, _create(suffix="split-create"))

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

    failed = materializer.apply(created, store=_FailObjectStateOnce(sync_store))
    assert failed.status == "failed"
    product_after_failure = _task(note_db)
    assert product_after_failure is not None
    assert product_after_failure["version"] == 1

    replay = materializer.apply(created, store=sync_store)
    assert replay.status == "applied"
    assert _task(note_db)["version"] == 1
    state = sync_store.get_object_state(DATASET_ID, "notes.task", TASK_ID)
    assert state is not None
    assert state.latest_server_cursor == created.server_cursor
    assert state.object_hash == created.payload_hash


def test_materializer_conflicts_lower_cursor_and_divergent_product_state(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = _materializer(note_db)
    created = _insert_internal(sync_store, _create(suffix="conflict-create"))
    assert materializer.apply(created, store=sync_store).status == "applied"
    updated = _insert_internal(
        sync_store,
        _create(
            payload=_payload(title="Current title"),
            base=created,
            suffix="conflict-update",
        ),
    )
    assert materializer.apply(updated, store=sync_store).status == "applied"

    assert materializer.apply(created, store=sync_store).status == "conflict"

    note_db.execute_query(
        "UPDATE note_tasks SET canonical_hash = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        ("sha256:" + "f" * 64, OWNER_ID, DATASET_ID, TASK_ID),
        commit=True,
    )
    next_envelope = _insert_internal(
        sync_store,
        _create(
            payload=_payload(title="Rejected divergence"),
            base=updated,
            suffix="conflict-divergent",
        ),
    )
    result = materializer.apply(next_envelope, store=sync_store)
    assert result.status == "conflict"
    assert result.conflict_type == "notes_task_product_conflict"
    assert "title" not in (result.message or "")


def test_materializer_product_failure_rolls_back_without_activity(
    materialization_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = _materializer(note_db)
    incoming = _insert_internal(sync_store, _create(suffix="rollback"))

    task_store_type = type(note_db.task_store)

    def fail_after_write(_stage: str) -> None:
        raise RuntimeError("injected product failure")

    monkeypatch.setattr(
        task_store_type,
        "_sync_task_materialization_checkpoint",
        staticmethod(fail_after_write),
    )

    result = materializer.apply(incoming, store=sync_store)

    assert result.status == "failed"
    assert _task(note_db) is None
    assert _event_count(note_db) == 0


def test_task_store_sync_create_rejects_missing_parent_and_wrong_scope(
    materialization_stack,
) -> None:
    note_db, _sync_store = materialization_stack
    payload = _payload(note_id=MISSING_NOTE_ID)

    with pytest.raises(ConflictError):
        with note_db.transaction() as conn:
            note_db.task_store.apply_sync_task_create(
                owner_user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                payload=payload,
                canonical_revision=1,
                canonical_hash=notes_task_object_hash(
                    payload,
                    revision=1,
                    deleted=False,
                ),
                conn=conn,
            )

    with pytest.raises(ConflictError):
        with note_db.transaction() as conn:
            note_db.task_store.apply_sync_task_create(
                owner_user_id=OWNER_ID,
                dataset_id="other-dataset",
                payload=_payload(),
                canonical_revision=1,
                canonical_hash="sha256:" + "1" * 64,
                conn=conn,
            )

    assert _task(note_db) is None


def test_materializer_reports_existing_product_identity_as_conflict(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    note_db.task_store.create_task(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        note_id=NOTE_ID,
        text="Existing local task",
        task_id=TASK_ID,
        projection_status="unlinked",
    )
    incoming = _insert_internal(sync_store, _create(suffix="identity-collision"))

    result = _materializer(note_db).apply(incoming, store=sync_store)

    assert result.status == "conflict"
    assert result.conflict_type == "notes_task_product_conflict"
    assert _task(note_db)["text"] == "Existing local task"
    assert _event_count(note_db) == 0
