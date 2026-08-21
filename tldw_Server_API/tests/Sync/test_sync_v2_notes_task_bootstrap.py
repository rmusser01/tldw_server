from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters import NotesTaskDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    SYNC_V2_SUPPORTED_DOMAINS,
    SyncDeviceCursor,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

OWNER_ID = "owner-user-1"
NOTE_ID = "20000000-0000-4000-8000-000000000001"
TASK_IDS = (
    "10000000-0000-4000-8000-000000000001",
    "10000000-0000-4000-8000-000000000002",
    "10000000-0000-4000-8000-000000000003",
)


def _bootstrap_module():
    return importlib.import_module(
        "tldw_Server_API.app.core.Sync.v2.notes_task_bootstrap"
    )


@pytest.fixture()
def bootstrap_stack(tmp_path: Path):
    note_db = CharactersRAGDB(tmp_path / "tasks.db", client_id=OWNER_ID)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    dataset = sync_store.get_or_create_default_personal_dataset(OWNER_ID)
    note_db.note_store.add_note(NOTE_ID, "Task source", note_id=NOTE_ID)
    note_db.bind_local_task_graph_to_dataset(
        owner_user_id=OWNER_ID,
        target_dataset_id=dataset.dataset_id,
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry([NotesTaskDomainAdapter()]),
    )
    try:
        yield note_db, sync_store, service, dataset
    finally:
        note_db.close_connection()


def _create_tasks(
    note_db: CharactersRAGDB,
    dataset_id: str,
    task_ids: tuple[str, ...] = TASK_IDS,
) -> None:
    for index, task_id in enumerate(reversed(task_ids), start=1):
        note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id=dataset_id,
            note_id=NOTE_ID,
            text=f"Task {index}",
            metadata={"priority": "high" if index == 1 else None},
            task_id=task_id,
        )


def test_task_bootstrap_helpers_are_stable_and_domain_bound() -> None:
    module = _bootstrap_module()

    first = module._task_bootstrap_envelope_id(
        "bootstrap-1", TASK_IDS[0], "sha256:" + "a" * 64
    )
    assert first == module._task_bootstrap_envelope_id(
        "bootstrap-1", TASK_IDS[0], "sha256:" + "a" * 64
    )
    assert first != module._task_bootstrap_envelope_id(
        "bootstrap-1", TASK_IDS[1], "sha256:" + "a" * 64
    )
    assert module._task_bootstrap_routing("bootstrap-1") == {
        "bootstrap_capture": True,
        "bootstrap_id": "bootstrap-1",
    }
    assert module._task_bootstrap_fingerprint(
        None, TASK_IDS[0], "sha256:" + "a" * 64
    ) == module._task_bootstrap_fingerprint(
        None, TASK_IDS[0], "sha256:" + "a" * 64
    )


def test_task_store_bootstrap_page_is_bounded_keyset_ordered(
    bootstrap_stack,
) -> None:
    note_db, _sync_store, _service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id)

    first = note_db.task_store.page_tasks_for_sync_bootstrap(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        after_task_id=None,
        limit=2,
    )
    second = note_db.task_store.page_tasks_for_sync_bootstrap(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        after_task_id=first[-1]["id"],
        limit=500,
    )

    assert [row["id"] for row in first] == list(TASK_IDS[:2])
    assert [row["id"] for row in second] == [TASK_IDS[2]]
    with pytest.raises(ValueError, match="1..500"):
        note_db.task_store.page_tasks_for_sync_bootstrap(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            limit=501,
        )


def test_bootstrap_rejects_local_unbound_product_graph(tmp_path: Path) -> None:
    note_db = CharactersRAGDB(tmp_path / "unbound.db", client_id=OWNER_ID)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    dataset = sync_store.get_or_create_default_personal_dataset(OWNER_ID)
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry([NotesTaskDomainAdapter()]),
    )
    try:
        with pytest.raises(SyncStoreError, match="source_scope_invalid"):
            _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
                service=service,
                dataset=dataset,
            )
    finally:
        note_db.close_connection()


def test_bootstrap_captures_one_page_then_finishes_task_only_readiness(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id)
    bootstrapper = _bootstrap_module().NotesTaskBootstrapper(note_db, page_limit=2)

    partial = bootstrapper.bootstrap(service=service, dataset=dataset)
    task_state = partial.metadata["notes_task_v1"]
    assert task_state["state"] == "bootstrapping"
    assert task_state["source_count"] == 2
    assert task_state["source_cursor"] == TASK_IDS[1]

    ready = bootstrapper.bootstrap(service=service, dataset=partial)
    assert ready.metadata["notes_task_v1"]["state"] == "ready"
    assert ready.metadata["notes_task_v1"]["source_count"] == 3
    assert ready.metadata["notes_task_activity_v1"]["state"] == "enrolling"
    assert ready.metadata["task_activity_capture_enabled"] is True
    assert "notes.task" not in ready.domains
    assert "notes.task_activity" not in ready.domains
    assert "notes.task" not in SYNC_V2_SUPPORTED_DOMAINS

    envelopes = sync_store.list_envelopes_after(
        dataset.dataset_id,
        0,
        limit=10,
    )
    tasks = [item for item in envelopes if item.domain == "notes.task"]
    assert [item.object_id for item in tasks] == list(TASK_IDS)
    assert all(item.apply_status == "applied" for item in tasks)
    bootstrap_id = _bootstrap_module()._bootstrap_id(OWNER_ID, dataset.dataset_id)
    assert [item.client_envelope_id for item in tasks] == [
        _bootstrap_module()._task_bootstrap_envelope_id(
            bootstrap_id,
            str(item.object_id),
            str(item.payload_hash),
        )
        for item in tasks
    ]


def test_bootstrap_history_supports_internal_pull_cursor_and_acknowledgment(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:2])
    ready = _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )
    assert ready.metadata["notes_task_v1"]["state"] == "ready"
    sync_store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-task-bootstrap",
            user_id=OWNER_ID,
            display_name="Bootstrap test device",
            client_type="test",
            capabilities={"requested_domains": ["notes.note"]},
        )
    )

    first = sync_store.list_envelopes_after(
        dataset.dataset_id,
        0,
        limit=1,
        domains=["notes.task"],
    )
    assert len(first) == 1 and first[0].server_cursor is not None
    second = sync_store.list_envelopes_after(
        dataset.dataset_id,
        first[0].server_cursor,
        limit=1,
        domains=["notes.task"],
    )
    assert len(second) == 1 and second[0].server_cursor is not None
    assert sync_store.list_envelopes_after(
        dataset.dataset_id,
        second[0].server_cursor,
        limit=1,
        domains=["notes.task"],
    ) == []

    # Exercise the generic cursor/ack path under the future trusted activation
    # condition without adding a public task-domain enrollment seam in this PR.
    sync_store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
        (json.dumps([*dataset.domains, "notes.task"]), dataset.dataset_id),
    )
    sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id=dataset.dataset_id,
            device_id="device-task-bootstrap",
            domain="notes.task",
            last_pulled_sequence=second[0].server_cursor,
            max_delivered_sequence=second[0].server_cursor,
        )
    )
    summary = sync_store.acknowledge_device_state_atomic(
        dataset.dataset_id,
        "device-task-bootstrap",
        domain_acks=[
            SyncDeviceDomainAckCreate(
                dataset_id=dataset.dataset_id,
                device_id="device-task-bootstrap",
                domain="notes.task",
                through_server_sequence=second[0].server_cursor,
                applied_at=datetime.now(timezone.utc).isoformat(),
            )
        ],
    )
    cursor = sync_store.get_device_cursor(
        dataset.dataset_id,
        "device-task-bootstrap",
        "notes.task",
    )
    assert cursor is not None
    assert cursor.last_pulled_sequence == second[0].server_cursor
    assert summary.domain_acks["notes.task"].through_server_sequence == second[0].server_cursor


def test_bootstrap_empty_bound_graph_becomes_task_ready_only(bootstrap_stack) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack

    ready = _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )

    assert ready.metadata["notes_task_v1"]["state"] == "ready"
    assert ready.metadata["notes_task_v1"]["source_count"] == 0
    assert ready.metadata["notes_task_v1"]["source_cursor"] is None
    assert ready.metadata["notes_task_activity_v1"]["state"] == "enrolling"
    assert not sync_store.list_envelopes_after(dataset.dataset_id, 0, limit=10)


def test_bootstrap_preserves_tombstone_revision_and_hash(bootstrap_stack) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:1])
    updated = note_db.task_store.update_task_record(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        task_id=TASK_IDS[0],
        expected_version=1,
        text="Updated before bootstrap",
    )
    deleted = note_db.task_store.soft_delete_task(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        task_id=TASK_IDS[0],
        expected_version=int(updated["version"]),
        allow_record_only=True,
    )

    ready = _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )
    envelopes = sync_store.list_envelopes_after(dataset.dataset_id, 0, limit=10)
    tasks = [item for item in envelopes if item.domain == "notes.task"]

    assert ready.metadata["notes_task_v1"]["state"] == "ready"
    assert len(tasks) == 1
    assert tasks[0].operation == "tombstone"
    assert tasks[0].object_revision == deleted["canonical_revision"]
    assert tasks[0].payload_hash == deleted["canonical_hash"]


def test_bootstrap_resumes_exactly_after_append_before_progress_split(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:2])
    module = _bootstrap_module()

    def interrupt(_page_number: int) -> None:
        raise module.NotesTaskBootstrapInterrupted

    with pytest.raises(module.NotesTaskBootstrapInterrupted):
        module.NotesTaskBootstrapper(
            note_db,
            page_limit=2,
            after_page=interrupt,
        ).bootstrap(service=service, dataset=dataset)

    interrupted = sync_store.get_dataset(
        dataset.dataset_id,
        owner_user_id=OWNER_ID,
    )
    assert interrupted is not None
    assert interrupted.metadata["notes_task_v1"]["source_count"] == 0
    before = sync_store.list_envelopes_after(
        dataset.dataset_id,
        0,
        limit=10,
    )

    resumed = module.NotesTaskBootstrapper(note_db, page_limit=2).bootstrap(
        service=service,
        dataset=interrupted,
    )
    after = sync_store.list_envelopes_after(
        dataset.dataset_id,
        0,
        limit=10,
    )
    assert resumed.metadata["notes_task_v1"]["source_count"] == 2
    assert [item.client_envelope_id for item in after] == [
        item.client_envelope_id for item in before
    ]


def test_bootstrap_source_drift_blocks_without_exposing_payload(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id)
    bootstrapper = _bootstrap_module().NotesTaskBootstrapper(note_db, page_limit=2)
    partial = bootstrapper.bootstrap(service=service, dataset=dataset)

    note_db.task_store.update_task_record(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        task_id=TASK_IDS[0],
        text="Changed during bootstrap",
        expected_version=1,
    )
    blocked = bootstrapper.bootstrap(service=service, dataset=partial)

    state = blocked.metadata["notes_task_v1"]
    assert state["state"] == "blocked"
    assert state["reason_code"] == "notes_task_source_changed"
    assert "Changed during bootstrap" not in str(state)
    assert len(
        [
            item
            for item in sync_store.list_envelopes_after(
                dataset.dataset_id,
                0,
                limit=10,
            )
            if item.domain == "notes.task"
        ]
    ) == 2


def test_bootstrap_concurrent_source_change_after_append_blocks(bootstrap_stack) -> None:
    note_db, _sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:1])

    def mutate_source(_page_number: int) -> None:
        note_db.task_store.update_task_record(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            task_id=TASK_IDS[0],
            expected_version=1,
            text="Concurrent source change",
        )

    blocked = _bootstrap_module().NotesTaskBootstrapper(
        note_db,
        after_page=mutate_source,
    ).bootstrap(service=service, dataset=dataset)

    assert blocked.metadata["notes_task_v1"]["state"] == "blocked"
    assert (
        blocked.metadata["notes_task_v1"]["reason_code"]
        == "notes_task_source_changed"
    )


@pytest.mark.parametrize("field", ["source_count", "source_fingerprint"])
def test_bootstrap_stored_progress_mismatch_blocks(
    bootstrap_stack,
    field: str,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:2])
    partial = _bootstrap_module().NotesTaskBootstrapper(
        note_db,
        page_limit=1,
    ).bootstrap(service=service, dataset=dataset)
    metadata = dict(partial.metadata)
    readiness = dict(metadata["notes_task_v1"])
    readiness[field] = (
        int(readiness["source_count"]) + 1
        if field == "source_count"
        else "f" * 64
    )
    metadata["notes_task_v1"] = readiness
    sync_store.db.execute(
        "UPDATE sync_datasets SET metadata_json = ? WHERE dataset_id = ?",
        (json.dumps(metadata, sort_keys=True), dataset.dataset_id),
    )
    tampered = sync_store.get_dataset(
        dataset.dataset_id,
        owner_user_id=OWNER_ID,
    )
    assert tampered is not None

    blocked = _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=tampered,
    )

    assert blocked.metadata["notes_task_v1"]["state"] == "blocked"
    assert (
        blocked.metadata["notes_task_v1"]["reason_code"]
        == "notes_task_source_changed"
    )


def test_bootstrap_final_verification_rejects_wrong_operation_and_revision(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:2])
    bootstrapper = _bootstrap_module().NotesTaskBootstrapper(note_db, page_limit=1)
    partial = bootstrapper.bootstrap(service=service, dataset=dataset)
    sync_store.db.execute(
        "UPDATE sync_envelopes SET operation = ?, object_revision = ? "
        "WHERE dataset_id = ? AND domain = ? AND entity_id = ?",
        ("tombstone", 99, dataset.dataset_id, "notes.task", TASK_IDS[0]),
    )

    blocked = bootstrapper.bootstrap(service=service, dataset=partial)

    assert blocked.metadata["notes_task_v1"]["state"] == "blocked"
    assert (
        blocked.metadata["notes_task_v1"]["reason_code"]
        == "notes_task_source_changed"
    )


def test_bootstrap_invalid_legacy_task_blocks_before_append(bootstrap_stack) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:1])
    note_db.execute_query(
        "UPDATE note_tasks SET source_diagnostic_code = ?, source_diagnostic_hash = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        (
            "legacy_task_payload_invalid",
            "sha256:" + "b" * 64,
            OWNER_ID,
            dataset.dataset_id,
            TASK_IDS[0],
        ),
        commit=True,
    )

    blocked = _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )

    assert blocked.metadata["notes_task_v1"]["state"] == "blocked"
    assert blocked.metadata["notes_task_v1"]["reason_code"] == "notes_task_source_invalid"
    assert not [
        item
        for item in sync_store.list_envelopes_after(
            dataset.dataset_id,
            0,
            limit=10,
        )
        if item.domain == "notes.task"
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "description": None,
            "priority": "high",
            "due_date": None,
            "estimate": None,
            "recurrence": None,
            "assignee_id": None,
            "tags": [],
            "custom": {},
            "unknown": "ignored",
        },
        {"description": None, "priority": "high"},
    ],
)
def test_bootstrap_rejects_noncanonical_legacy_metadata_shape(
    bootstrap_stack,
    metadata: dict[str, object],
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:1])
    note_db.execute_query(
        "UPDATE note_tasks SET metadata_json = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        (
            json.dumps(metadata, sort_keys=True),
            OWNER_ID,
            dataset.dataset_id,
            TASK_IDS[0],
        ),
        commit=True,
    )

    blocked = _bootstrap_module().NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )

    assert blocked.metadata["notes_task_v1"]["state"] == "blocked"
    assert blocked.metadata["notes_task_v1"]["reason_code"] == "notes_task_source_invalid"
    assert not [
        item
        for item in sync_store.list_envelopes_after(dataset.dataset_id, 0, limit=10)
        if item.domain == "notes.task"
    ]


def test_bootstrap_resumes_blocked_source_after_operator_repairs_row(
    bootstrap_stack,
) -> None:
    note_db, _sync_store, service, dataset = bootstrap_stack
    _create_tasks(note_db, dataset.dataset_id, TASK_IDS[:1])
    invalid_metadata = {
        "description": None,
        "priority": "high",
        "due_date": None,
        "estimate": None,
        "recurrence": None,
        "assignee_id": None,
        "tags": [],
        "custom": {},
        "unknown": "ignored",
    }
    note_db.execute_query(
        "UPDATE note_tasks SET metadata_json = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        (
            json.dumps(invalid_metadata, sort_keys=True),
            OWNER_ID,
            dataset.dataset_id,
            TASK_IDS[0],
        ),
        commit=True,
    )
    bootstrapper = _bootstrap_module().NotesTaskBootstrapper(note_db)
    blocked = bootstrapper.bootstrap(service=service, dataset=dataset)
    assert blocked.metadata["notes_task_v1"]["state"] == "blocked"

    note_db.execute_query(
        "UPDATE note_tasks SET metadata_json = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        (
            json.dumps({"priority": "high"}, sort_keys=True),
            OWNER_ID,
            dataset.dataset_id,
            TASK_IDS[0],
        ),
        commit=True,
    )
    ready = bootstrapper.bootstrap(service=service, dataset=blocked)

    assert ready.metadata["notes_task_v1"]["state"] == "ready"
    assert ready.metadata["notes_task_v1"]["source_count"] == 1


def test_factory_registers_task_components_without_advertising_domain() -> None:
    from tldw_Server_API.app.core.Sync.v2 import factory
    from tldw_Server_API.app.core.Sync.v2.materializers import NotesTaskMaterializer

    registry = factory.default_sync_v2_registry()
    assert isinstance(registry.get("notes.task"), NotesTaskDomainAdapter)
    assert "notes.task" not in factory._sync_v2_settings_from_env().supported_domains
    assert hasattr(factory, "_validate_notes_task_components")
    assert NotesTaskMaterializer is not None
