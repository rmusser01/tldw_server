"""End-to-end SQLite convergence for the activated Notes task domains."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskService
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters import (
    NotesDomainAdapter,
    NotesTaskActivityDomainAdapter,
    NotesTaskDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers import (
    NotesMaterializer,
    NotesTaskActivityMaterializer,
    NotesTaskMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_TASK_SYNC_DOMAINS,
    SyncDeviceDomainAckCreate,
    SyncEnvelopeCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_activity_bootstrap import (
    NotesTaskActivityBootstrapper,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_bootstrap import NotesTaskBootstrapper
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    notes_task_object_hash,
    parse_notes_task_v1,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
    NotesTaskCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import canonical_payload_hash
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration

OWNER_ID = "end-to-end-owner"
DATASET_ID = "ds_personal_end-to-end-owner"
NOTE_ID = "10000000-0000-4000-8000-000000000001"
TASK_ID = "20000000-0000-4000-8000-000000000001"
DEVICE_IDS = (
    "30000000-0000-4000-8000-000000000001",
    "30000000-0000-4000-8000-000000000002",
)
NOW = "2026-08-24T12:00:00+00:00"


def _stack(tmp_path: Path) -> tuple[CharactersRAGDB, SyncV2Service]:
    note_db = CharactersRAGDB(tmp_path / "product.db", client_id=OWNER_ID)
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db")),
        adapters=SyncAdapterRegistry(
            [
                NotesDomainAdapter(),
                NotesTaskDomainAdapter(),
                NotesTaskActivityDomainAdapter(),
            ]
        ),
        materializers={
            "notes.note": NotesMaterializer(note_db),
            "notes.task": NotesTaskMaterializer(note_db),
            "notes.task_activity": NotesTaskActivityMaterializer(note_db),
        },
        notes_task_bootstrapper=NotesTaskBootstrapper(note_db),
        notes_task_activity_bootstrapper=NotesTaskActivityBootstrapper(note_db),
        clock=lambda: NOW,
        settings=SyncV2Settings(
            max_pull_page_size=1,
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
    )
    service.store.get_or_create_default_personal_dataset(OWNER_ID)
    return note_db, service


def _seed_note_head(service: SyncV2Service) -> None:
    payload = {"title": "Tasks", "content": "Body"}
    payload_hash, payload_size = canonical_payload_hash(payload)
    envelope = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id="end-to-end-note",
            device_id="server-origin",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            object_revision=1,
            payload=payload,
            payload_hash=payload_hash,
            payload_size_bytes=payload_size,
            created_at_client=NOW,
            status="accepted",
            apply_status="applied",
            applied_at=NOW,
        )
    )
    assert envelope.server_cursor is not None
    service.store.upsert_object_state(
        SyncObjectState(
            dataset_id=DATASET_ID,
            domain="notes.note",
            object_id=NOTE_ID,
            object_revision=1,
            object_hash=payload_hash,
            latest_server_cursor=envelope.server_cursor,
            deleted=False,
        )
    )


def _push_task_update(
    service: SyncV2Service,
    *,
    device_id: str,
    envelope_id: str,
    created_at: str,
    **payload_updates: object,
) -> None:
    head = service.store.get_current_head(DATASET_ID, "notes.task", TASK_ID)
    assert head is not None and head.server_cursor is not None
    revision = int(head.object_revision or 0) + 1
    payload = parse_notes_task_v1(
        {**dict(head.payload), **payload_updates},
        owner_user_id=OWNER_ID,
    )
    result = service.push(
        user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        device_id=device_id,
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=DATASET_ID,
                client_envelope_id=envelope_id,
                device_id=device_id,
                domain="notes.task",
                operation="upsert",
                object_id=TASK_ID,
                parent_id=NOTE_ID,
                base_server_cursor=head.server_cursor,
                base_object_revision=head.object_revision,
                base_object_hash=head.payload_hash,
                object_revision=revision,
                entity_version=revision,
                payload=payload.model_dump(mode="json"),
                payload_hash=notes_task_object_hash(
                    payload,
                    revision=revision,
                    deleted=False,
                ),
                created_at_client=created_at,
                encryption_metadata={"policy": "server_trusted_v1"},
            )
        ],
    )
    assert result.rejected == []
    assert result.conflicts == []
    assert len(result.accepted) == 1


def _pull_all(service: SyncV2Service, device_id: str):
    cursor = "0"
    envelopes = []
    while True:
        page = service.pull(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            device_id=device_id,
            cursor=cursor,
            domains=list(NOTES_TASK_SYNC_DOMAINS),
            page_size=1,
            include_own_changes=True,
        )
        envelopes.extend(page.envelopes)
        assert page.next_cursor is not None
        if not page.has_more:
            return envelopes
        assert page.next_cursor != cursor
        cursor = page.next_cursor


def test_two_devices_converge_on_server_and_client_task_transitions(
    tmp_path: Path,
) -> None:
    note_db, service = _stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="Converge task",
            projection_status="unlinked",
        )
        _seed_note_head(service)

        enrollment = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=[*M1_SYNC_DOMAINS, *NOTES_TASK_SYNC_DOMAINS],
        )
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(enrollment.dataset.domains)

        capabilities = {
            "requested_domains": list(NOTES_TASK_SYNC_DOMAINS),
            "supported_adapter_versions": dict.fromkeys(
                NOTES_TASK_SYNC_DOMAINS,
                [1],
            ),
        }
        for index, device_id in enumerate(DEVICE_IDS, start=1):
            service.register_device(
                user_id=OWNER_ID,
                device_id=device_id,
                display_name=f"Device {index}",
                client_type="chatbook",
                capabilities=capabilities,
            )

        coordinator = NotesTaskCoordinator(
            service=service,
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )
        task_service = NotesTaskService(
            task_coordinator_resolver=lambda **_kwargs: coordinator
        )
        task = note_db.task_store.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            task_id=TASK_ID,
        )
        assert task is not None
        task_service.update_task(
            db=note_db,
            task_id=TASK_ID,
            expected_task_version=int(task["version"]),
            expected_note_version=None,
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            metadata={"priority": "high"},
            record_only=True,
        )

        recurrence = {
            "frequency": "weekly",
            "interval": 2,
            "by_weekday": ["mo", "we", "fr"],
            "until": "2026-12-31",
            "state": "active",
            "occurrence_index": 7,
        }
        _push_task_update(
            service,
            device_id=DEVICE_IDS[0],
            envelope_id="client-recurrence",
            created_at="2026-08-24T12:01:00+00:00",
            recurrence=recurrence,
        )
        _push_task_update(
            service,
            device_id=DEVICE_IDS[1],
            envelope_id="client-complete",
            created_at="2026-08-24T12:02:00+00:00",
            status="done",
            completed_at="2026-08-24T12:02:00+00:00",
        )
        _push_task_update(
            service,
            device_id=DEVICE_IDS[0],
            envelope_id="client-reopen",
            created_at="2026-08-24T12:03:00+00:00",
            status="open",
            completed_at=None,
        )

        final_task = note_db.task_store.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            task_id=TASK_ID,
        )
        assert final_task is not None
        assert final_task["status"] == "open"
        assert final_task["metadata_json"]["recurrence"] == recurrence
        assert len(
            note_db.task_store.page_tasks_for_sync_bootstrap(
                owner_user_id=OWNER_ID,
                dataset_id=DATASET_ID,
            )
        ) == 1

        pulled = [_pull_all(service, device_id) for device_id in DEVICE_IDS]
        assert [item.server_cursor for item in pulled[0]] == [
            item.server_cursor for item in pulled[1]
        ]
        assert all(item.apply_status == "applied" for item in pulled[0])
        assert {item.domain for item in pulled[0]} == set(NOTES_TASK_SYNC_DOMAINS)
        assert len({item.server_cursor for item in pulled[0]}) == len(pulled[0])
        assert sum(item.domain == "notes.task" for item in pulled[0]) == 5
        assert sum(item.domain == "notes.task_activity" for item in pulled[0]) >= 4

        for device_id, delivered in zip(DEVICE_IDS, pulled, strict=True):
            acknowledgments = []
            for domain in NOTES_TASK_SYNC_DOMAINS:
                through = max(
                    int(item.server_cursor or 0)
                    for item in delivered
                    if item.domain == domain
                )
                acknowledgments.append(
                    SyncDeviceDomainAckCreate(
                        dataset_id=DATASET_ID,
                        device_id=device_id,
                        domain=domain,
                        adapter_version=1,
                        through_server_sequence=through,
                        applied_at=NOW,
                    )
                )
            service.acknowledge_device_state(
                user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                device_id=device_id,
                domain_acks=acknowledgments,
            )
            assert all(
                service.store.get_device_domain_ack(
                    DATASET_ID,
                    device_id,
                    domain,
                    adapter_version=1,
                )
                is not None
                for domain in NOTES_TASK_SYNC_DOMAINS
            )
    finally:
        note_db.close_connection()
