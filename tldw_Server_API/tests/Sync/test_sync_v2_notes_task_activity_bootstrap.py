from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters import (
    NotesTaskActivityDomainAdapter,
    NotesTaskDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers import (
    NotesTaskActivityMaterializer,
    NotesTaskMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SYNC_V2_SUPPORTED_DOMAINS
from tldw_Server_API.app.core.Sync.v2.notes_task_bootstrap import NotesTaskBootstrapper
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

OWNER_ID = "owner-user-1"
NOTE_ID = "20000000-0000-4000-8000-000000000001"
TASK_ID = "10000000-0000-4000-8000-000000000001"
EVENT_IDS = (
    "30000000-0000-4000-8000-000000000001",
    "30000000-0000-4000-8000-000000000002",
    "30000000-0000-4000-8000-000000000003",
)
OCCURRED_AT = "2026-08-13T10:00:00+00:00"


def _bootstrap_module():
    return importlib.import_module(
        "tldw_Server_API.app.core.Sync.v2.notes_task_activity_bootstrap"
    )


@pytest.fixture()
def bootstrap_stack(tmp_path: Path):
    note_db = CharactersRAGDB(tmp_path / "activity.db", client_id=OWNER_ID)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    dataset = sync_store.get_or_create_default_personal_dataset(OWNER_ID)
    note_db.note_store.add_note(NOTE_ID, "Activity source", note_id=NOTE_ID)
    note_db.bind_local_task_graph_to_dataset(
        owner_user_id=OWNER_ID,
        target_dataset_id=dataset.dataset_id,
    )
    note_db.task_store.create_task(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        note_id=NOTE_ID,
        text="Source task",
        task_id=TASK_ID,
        projection_status="unlinked",
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [NotesTaskDomainAdapter(), NotesTaskActivityDomainAdapter()]
        ),
        materializers={
            "notes.task": NotesTaskMaterializer(note_db),
            "notes.task_activity": NotesTaskActivityMaterializer(note_db),
        },
    )
    dataset = NotesTaskBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )
    assert dataset.metadata["notes_task_v1"]["state"] == "ready"
    assert dataset.metadata["notes_task_activity_v1"]["state"] == "enrolling"
    try:
        yield note_db, sync_store, service, dataset
    finally:
        note_db.close_connection()


def _record_events(
    note_db: CharactersRAGDB,
    dataset_id: str,
    event_ids: tuple[str, ...] = EVENT_IDS,
) -> None:
    for event_id in reversed(event_ids):
        note_db.task_store.record_task_event(
            owner_user_id=OWNER_ID,
            dataset_id=dataset_id,
            event_id=event_id,
            task_id=TASK_ID,
            note_id=NOTE_ID,
            event_type="status_changed",
            actor_type="user",
            actor_id=OWNER_ID,
            old_value={"status": "open"},
            new_value={"status": "done"},
        )
    note_db.execute_query(
        "UPDATE task_events SET created_at = ?, client_occurred_at = ? "
        "WHERE owner_user_id = ? AND dataset_id = ?",
        (OCCURRED_AT, OCCURRED_AT, OWNER_ID, dataset_id),
        commit=True,
    )


def _activity_envelopes(sync_store: SyncV2Store, dataset_id: str):
    return [
        item
        for item in sync_store.list_envelopes_after(dataset_id, 0, limit=100)
        if item.domain == "notes.task_activity"
    ]


def test_legacy_activity_source_page_is_scoped_bounded_and_keyset_ordered(
    bootstrap_stack,
) -> None:
    note_db, _sync_store, _service, dataset = bootstrap_stack
    _record_events(note_db, dataset.dataset_id)

    first = note_db.task_store.page_legacy_events_for_sync_bootstrap(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        limit=2,
    )
    second = note_db.task_store.page_legacy_events_for_sync_bootstrap(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        after_created_at=first[-1]["created_at"],
        after_activity_id=first[-1]["id"],
        limit=10,
    )

    assert [row["id"] for row in first] == list(EVENT_IDS[:2])
    assert [row["id"] for row in second] == [EVENT_IDS[2]]
    assert first[0]["resolved_task_note_id"] == NOTE_ID
    assert note_db.task_store.page_legacy_events_for_sync_bootstrap(
        owner_user_id="other-owner",
        dataset_id=dataset.dataset_id,
    ) == []
    with pytest.raises(ValueError, match="1..1000"):
        note_db.task_store.page_legacy_events_for_sync_bootstrap(
            owner_user_id=OWNER_ID,
            dataset_id=dataset.dataset_id,
            limit=1_001,
        )


def test_activity_bootstrap_captures_pages_adopts_rows_and_ignores_read_state(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _record_events(note_db, dataset.dataset_id)
    note_db.task_store.mark_task_activity_read(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        event_id=EVENT_IDS[0],
        user_id=OWNER_ID,
    )
    module = _bootstrap_module()
    bootstrapper = module.NotesTaskActivityBootstrapper(note_db, page_limit=2)

    partial = bootstrapper.bootstrap(service=service, dataset=dataset)
    state = partial.metadata["notes_task_activity_v1"]
    assert state["state"] == "bootstrapping"
    assert state["source_count"] == 2
    assert state["source_cursor"] == f"{OCCURRED_AT}|{EVENT_IDS[1]}"

    ready = bootstrapper.bootstrap(service=service, dataset=partial)
    state = ready.metadata["notes_task_activity_v1"]
    assert state["state"] == "ready"
    assert state["source_count"] == 3
    assert "notes.task" not in ready.domains
    assert "notes.task_activity" not in ready.domains
    assert "notes.task_activity" not in SYNC_V2_SUPPORTED_DOMAINS

    envelopes = _activity_envelopes(sync_store, dataset.dataset_id)
    assert [item.object_id for item in envelopes] == list(EVENT_IDS)
    assert [item.created_at_client for item in envelopes] == [OCCURRED_AT] * 3
    assert all(item.apply_status == "applied" for item in envelopes)
    bootstrap_id = module._bootstrap_id(OWNER_ID, dataset.dataset_id)
    assert [item.client_envelope_id for item in envelopes] == [
        module._activity_bootstrap_envelope_id(
            bootstrap_id,
            item.object_id,
            item.payload_hash,
        )
        for item in envelopes
    ]
    rows = note_db.task_store.page_sync_task_activity(
        owner_user_id=OWNER_ID,
        dataset_id=dataset.dataset_id,
        limit=10,
    )
    assert [row["id"] for row in rows] == list(EVENT_IDS)
    assert all(row["event_type"] == "completed" for row in rows)
    assert all(row["source_kind"] == "trusted_bootstrap_v1" for row in rows)
    assert all(row["sync_server_cursor"] is not None for row in rows)
    assert len(envelopes) == 3


def test_activity_bootstrap_nonfinal_resume_scans_only_one_page(
    bootstrap_stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each non-final resume proportional to its configured page."""

    note_db, _sync_store, service, dataset = bootstrap_stack
    _record_events(note_db, dataset.dataset_id)
    page_limits: list[int] = []
    original_page = note_db.task_store.page_legacy_events_for_sync_bootstrap

    def tracked_page(**kwargs):
        """Record source-page limits without changing storage behavior."""

        page_limits.append(kwargs["limit"])
        return original_page(**kwargs)

    monkeypatch.setattr(
        note_db.task_store,
        "page_legacy_events_for_sync_bootstrap",
        tracked_page,
    )
    bootstrapper = _bootstrap_module().NotesTaskActivityBootstrapper(
        note_db,
        page_limit=1,
    )

    first = bootstrapper.bootstrap(service=service, dataset=dataset)
    assert first.metadata["notes_task_activity_v1"]["state"] == "bootstrapping"
    assert page_limits == [1]

    page_limits.clear()
    second = bootstrapper.bootstrap(service=service, dataset=first)
    assert second.metadata["notes_task_activity_v1"]["state"] == "bootstrapping"
    assert page_limits == [1]


def test_activity_bootstrap_resumes_after_append_before_progress_split(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _record_events(note_db, dataset.dataset_id, EVENT_IDS[:2])
    module = _bootstrap_module()

    def interrupt(_page_number: int) -> None:
        raise module.NotesTaskBootstrapInterrupted

    with pytest.raises(module.NotesTaskBootstrapInterrupted):
        module.NotesTaskActivityBootstrapper(
            note_db,
            page_limit=2,
            after_page=interrupt,
        ).bootstrap(service=service, dataset=dataset)
    interrupted = sync_store.get_dataset(
        dataset.dataset_id,
        owner_user_id=OWNER_ID,
    )
    assert interrupted is not None
    assert interrupted.metadata["notes_task_activity_v1"]["source_count"] == 0
    before = _activity_envelopes(sync_store, dataset.dataset_id)

    ready = module.NotesTaskActivityBootstrapper(note_db, page_limit=2).bootstrap(
        service=service,
        dataset=interrupted,
    )
    after = _activity_envelopes(sync_store, dataset.dataset_id)
    assert ready.metadata["notes_task_activity_v1"]["state"] == "ready"
    assert [item.client_envelope_id for item in after] == [
        item.client_envelope_id for item in before
    ]


def test_activity_bootstrap_source_drift_blocks_without_deleting_history(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _record_events(note_db, dataset.dataset_id, EVENT_IDS[:2])
    module = _bootstrap_module()
    bootstrapper = module.NotesTaskActivityBootstrapper(note_db, page_limit=1)
    partial = bootstrapper.bootstrap(service=service, dataset=dataset)
    note_db.execute_query(
        "UPDATE task_events SET sync_object_hash = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        ("sha256:" + "f" * 64, OWNER_ID, dataset.dataset_id, EVENT_IDS[0]),
        commit=True,
    )

    blocked = bootstrapper.bootstrap(service=service, dataset=partial)

    state = blocked.metadata["notes_task_activity_v1"]
    assert state["state"] == "blocked"
    assert state["reason_code"] == "notes_task_activity_source_changed"
    assert len(_activity_envelopes(sync_store, dataset.dataset_id)) == 1


def test_activity_bootstrap_malformed_legacy_row_blocks_before_append(
    bootstrap_stack,
) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack
    _record_events(note_db, dataset.dataset_id, EVENT_IDS[:1])
    note_db.execute_query(
        "UPDATE task_events SET event_type = ? "
        "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?",
        ("unknown", OWNER_ID, dataset.dataset_id, EVENT_IDS[0]),
        commit=True,
    )

    blocked = _bootstrap_module().NotesTaskActivityBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )

    state = blocked.metadata["notes_task_activity_v1"]
    assert state["state"] == "blocked"
    assert state["reason_code"] == "notes_task_activity_source_invalid"
    assert not _activity_envelopes(sync_store, dataset.dataset_id)


def test_activity_bootstrap_empty_source_becomes_ready(bootstrap_stack) -> None:
    note_db, sync_store, service, dataset = bootstrap_stack

    ready = _bootstrap_module().NotesTaskActivityBootstrapper(note_db).bootstrap(
        service=service,
        dataset=dataset,
    )

    state = ready.metadata["notes_task_activity_v1"]
    assert state["state"] == "ready"
    assert state["source_count"] == 0
    assert state["source_cursor"] is None
    assert not _activity_envelopes(sync_store, dataset.dataset_id)


def test_factory_wires_private_activity_components_without_advertising_domain() -> None:
    from tldw_Server_API.app.core.Sync.v2 import factory

    registry = factory.default_sync_v2_registry()
    assert isinstance(registry.get("notes.task_activity"), NotesTaskActivityDomainAdapter)
    assert "notes.task_activity" not in factory._sync_v2_settings_from_env().supported_domains
    assert hasattr(factory, "_validate_notes_task_components")
