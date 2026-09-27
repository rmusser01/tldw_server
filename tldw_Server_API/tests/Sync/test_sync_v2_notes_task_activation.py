from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskService
from tldw_Server_API.app.core.Sync.v2.adapters import (
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters import (
    NotesDomainAdapter,
    NotesTaskActivityDomainAdapter,
    NotesTaskDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    NotesMaterializer,
    NotesTaskActivityMaterializer,
    NotesTaskMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_MOODBOARD_STUDIO_DOMAINS,
    NOTES_TASK_SYNC_DOMAINS,
    SyncEnvelopeCreate,
    normalize_supported_adapter_versions,
    normalize_sync_v2_requested_domains,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_activity_bootstrap import (
    NotesTaskActivityBootstrapper,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_bootstrap import NotesTaskBootstrapper
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    notes_task_activity_object_hash,
    parse_notes_task_activity_v1,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_coordinator import (
    NotesTaskCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import canonical_payload_hash
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    is_trusted_notes_task_coordinator_envelope,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

OWNER_ID = "activation-owner"
DATASET_ID = "ds_personal_activation-owner"
NOTE_ID = "10000000-0000-4000-8000-000000000001"
TASK_ID = "20000000-0000-4000-8000-000000000001"


def _activation_stack(
    tmp_path: Path,
    *,
    task_page_limit: int = NotesTaskBootstrapper.PAGE_LIMIT,
    activity_page_limit: int = NotesTaskActivityBootstrapper.PAGE_LIMIT,
    task_after_page: Callable[[int], None] | None = None,
    activity_after_page: Callable[[int], None] | None = None,
) -> tuple[CharactersRAGDB, SyncV2Service]:
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
        notes_task_bootstrapper=NotesTaskBootstrapper(
            note_db,
            page_limit=task_page_limit,
            after_page=task_after_page,
        ),
        notes_task_activity_bootstrapper=NotesTaskActivityBootstrapper(
            note_db,
            page_limit=activity_page_limit,
            after_page=activity_after_page,
        ),
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            )
        ),
    )
    service.store.get_or_create_default_personal_dataset(OWNER_ID)
    return note_db, service


def _requested_domains() -> list[str]:
    return [*M1_SYNC_DOMAINS, *NOTES_TASK_SYNC_DOMAINS]


def _client(service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = lambda: User(
        id=OWNER_ID,
        username=OWNER_ID,
    )
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    return TestClient(app)


def _seed_sync_note(
    service: SyncV2Service,
    *,
    envelope_id: str,
) -> None:
    note_payload = {"title": "Tasks", "content": "Body"}
    note_hash, note_size = canonical_payload_hash(note_payload)
    service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id=envelope_id,
            device_id="server-origin",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            object_revision=1,
            payload=note_payload,
            payload_hash=note_hash,
            payload_size_bytes=note_size,
            created_at_client="2026-08-24T12:00:00+00:00",
            status="accepted",
            apply_status="applied",
        )
    )


def test_unbound_capabilities_omit_coupled_task_domains(tmp_path: Path) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        capabilities = service.capabilities(user_id=OWNER_ID)

        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(capabilities.supported_domains)
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(
            capabilities.supported_adapter_versions
        )
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(
            capabilities.writable_adapter_versions
        )
    finally:
        note_db.close_connection()


def test_device_negotiation_accepts_only_the_complete_version_one_pair() -> None:
    requested = normalize_sync_v2_requested_domains(NOTES_TASK_SYNC_DOMAINS)

    assert requested == list(NOTES_TASK_SYNC_DOMAINS)
    assert normalize_supported_adapter_versions(
        dict.fromkeys(NOTES_TASK_SYNC_DOMAINS, [1]),
        requested_domains=requested,
    ) == dict.fromkeys(NOTES_TASK_SYNC_DOMAINS, [1])
    with pytest.raises(ValueError, match="both Notes task domains"):
        normalize_sync_v2_requested_domains(["notes.task"])


def test_http_enrollment_and_selected_capabilities_publish_pair_together(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        client = _client(service)

        enrolled = client.post(
            "/api/v1/sync/datasets/enroll",
            json={
                "dataset_id": DATASET_ID,
                "scope_type": "personal",
                "domains": _requested_domains(),
                "encryption_policy": "server_trusted_v1",
            },
        )
        capabilities = client.get(
            "/api/v1/sync/capabilities",
            params={"dataset_id": DATASET_ID},
        )

        assert enrolled.status_code == 200
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(enrolled.json()["domains"])
        assert capabilities.status_code == 200
        body = capabilities.json()
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(body["domains"])
        assert {
            domain: body["supported_adapter_versions"][domain]
            for domain in NOTES_TASK_SYNC_DOMAINS
        } == dict.fromkeys(NOTES_TASK_SYNC_DOMAINS, [1])
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(body["domain_schemas"])
    finally:
        note_db.close_connection()


def test_profile_bootstrap_can_explicitly_activate_complete_task_pair(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)

        profile = service.bootstrap_profile(
            user_id=OWNER_ID,
            mode="server_frontend",
            device_id="activation-device",
            device_name="Laptop",
            requested_domains=_requested_domains(),
        )

        assert profile.dataset is not None
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(profile.dataset.domains)
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(
            profile.capabilities.supported_domains
        )
    finally:
        note_db.close_connection()


@pytest.mark.parametrize("domains", [["notes.task"], ["notes.task_activity"]])
def test_enrollment_rejects_incomplete_task_domain_pair(
    tmp_path: Path,
    domains: list[str],
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        with pytest.raises(SyncStoreError, match="notes_task_sync_domains_incomplete"):
            service.enroll_dataset(
                user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                domains=[*M1_SYNC_DOMAINS, *domains],
            )
    finally:
        note_db.close_connection()


def test_explicit_enrollment_rekeys_bootstraps_and_advertises_pair(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="Ship activation",
            projection_status="unlinked",
        )

        enrollment = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        stored = service.store.get_dataset(DATASET_ID, owner_user_id=OWNER_ID)
        rebound = note_db.task_store.get_task(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            task_id=TASK_ID,
            include_deleted=True,
        )
        capabilities = service.capabilities(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )

        assert stored is not None
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(stored.domains)
        assert stored.metadata["notes_task_v1"]["state"] == "ready"
        assert stored.metadata["notes_task_activity_v1"]["state"] == "ready"
        assert stored.metadata["task_activity_capture_enabled"] is True
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(enrollment.dataset.metadata)
        assert rebound["dataset_id"] == DATASET_ID
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(capabilities.supported_domains)
        assert {
            domain: capabilities.supported_adapter_versions[domain]
            for domain in NOTES_TASK_SYNC_DOMAINS
        } == dict.fromkeys(NOTES_TASK_SYNC_DOMAINS, [1])
        assert {
            domain: capabilities.writable_adapter_versions[domain]
            for domain in NOTES_TASK_SYNC_DOMAINS
        } == dict.fromkeys(NOTES_TASK_SYNC_DOMAINS, [1])
    finally:
        note_db.close_connection()


def test_task_ready_public_capabilities_filter_dormant_private_schemas(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        core_capabilities = service.capabilities(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )
        public_capabilities = sync_endpoint._api_capabilities_from_core(
            core_capabilities
        )

        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(
            core_capabilities.domain_schemas
        )
        assert set(NOTES_MOODBOARD_STUDIO_DOMAINS).isdisjoint(
            core_capabilities.domain_schemas
        )
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(
            public_capabilities.domain_schemas
        )
        assert set(NOTES_MOODBOARD_STUDIO_DOMAINS).isdisjoint(
            public_capabilities.domain_schemas
        )
    finally:
        note_db.close_connection()


def test_active_reenrollment_is_idempotent_and_cannot_disable_pair(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        first = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        second = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        assert second.dataset.domains == first.dataset.domains
        with pytest.raises(SyncStoreError, match="notes_task_sync_disable_forbidden"):
            service.enroll_dataset(
                user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                domains=list(M1_SYNC_DOMAINS),
            )
    finally:
        note_db.close_connection()


def test_selected_dataset_capabilities_are_owner_scoped(tmp_path: Path) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        with pytest.raises(SyncStoreError, match="not found or is not accessible"):
            service.capabilities(user_id="other-owner", dataset_id=DATASET_ID)
    finally:
        note_db.close_connection()


def test_product_commit_sync_readiness_split_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="Survive split commit",
            projection_status="unlinked",
        )
        begin = service.store.begin_notes_task_activation

        def fail_after_product_commit(*_args: object, **_kwargs: object) -> object:
            raise SyncStoreError("injected_sync_readiness_failure")

        monkeypatch.setattr(
            service.store,
            "begin_notes_task_activation",
            fail_after_product_commit,
        )
        with pytest.raises(SyncStoreError, match="injected_sync_readiness_failure"):
            service.enroll_dataset(
                user_id=OWNER_ID,
                dataset_id=DATASET_ID,
                domains=_requested_domains(),
            )

        split = service.store.get_dataset(DATASET_ID, owner_user_id=OWNER_ID)
        assert split is not None
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(split.domains)
        assert note_db.task_store.resolve_task_compatibility_dataset_id(
            owner_user_id=OWNER_ID
        ) == DATASET_ID

        monkeypatch.setattr(service.store, "begin_notes_task_activation", begin)
        resumed = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(resumed.dataset.domains)
        task_envelopes = [
            item
            for item in service.store.list_envelopes_after(DATASET_ID, 0)
            if item.domain == "notes.task" and item.object_id == TASK_ID
        ]
        assert len(task_envelopes) == 1
    finally:
        note_db.close_connection()


def test_paged_activation_omits_pair_until_both_bootstraps_are_ready(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path, task_page_limit=1)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        for index in (1, 2):
            note_db.task_store.create_task(
                owner_user_id=OWNER_ID,
                dataset_id="local-unbound",
                task_id=f"20000000-0000-4000-8000-{index:012d}",
                note_id=NOTE_ID,
                text=f"Paged task {index}",
                projection_status="unlinked",
            )

        first = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        first_capabilities = service.capabilities(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )

        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(first.dataset.domains)
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(
            first_capabilities.supported_domains
        )

        second = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(second.dataset.domains)
        assert [
            item.object_id
            for item in service.store.list_envelopes_after(DATASET_ID, 0)
            if item.domain == "notes.task"
        ] == [
            "20000000-0000-4000-8000-000000000001",
            "20000000-0000-4000-8000-000000000002",
        ]
    finally:
        note_db.close_connection()


def test_task_mutation_is_captured_between_bootstrap_pages(tmp_path: Path) -> None:
    note_db, service = _activation_stack(tmp_path, task_page_limit=1)
    task_ids = [
        "20000000-0000-4000-8000-000000000001",
        "20000000-0000-4000-8000-000000000002",
    ]
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        tasks = [
            note_db.task_store.create_task(
                owner_user_id=OWNER_ID,
                dataset_id="local-unbound",
                task_id=task_id,
                note_id=NOTE_ID,
                text=f"Paged task {index}",
                projection_status="unlinked",
            )
            for index, task_id in enumerate(task_ids, start=1)
        ]
        _seed_sync_note(service, envelope_id="bootstrap-race-note")

        first = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(first.dataset.domains)
        internal = service.store.get_dataset(DATASET_ID, owner_user_id=OWNER_ID)
        assert internal is not None
        assert internal.metadata["task_activity_capture_enabled"] is True

        coordinator = NotesTaskCoordinator(
            service=service,
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )
        task_service = NotesTaskService(
            task_coordinator_resolver=lambda **_kwargs: coordinator
        )
        updated = task_service.update_task(
            db=note_db,
            task_id=task_ids[0],
            expected_task_version=int(tasks[0]["version"]),
            expected_note_version=None,
            actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
            metadata={"priority": "high"},
            record_only=True,
        )

        captured = service.store.get_current_head(
            DATASET_ID,
            "notes.task",
            task_ids[0],
        )
        assert captured is not None
        assert captured.object_revision == updated["canonical_revision"]
        assert captured.apply_status == "applied"
        assert captured.mutation_group_id is not None
        captured_group = service.store.list_mutation_group(
            DATASET_ID,
            captured.mutation_group_id,
        )
        activity = captured_group[1]
        internal = service.store.get_dataset(DATASET_ID, owner_user_id=OWNER_ID)
        assert internal is not None
        assert is_trusted_notes_task_coordinator_envelope(
            service=service,
            dataset=internal,
            envelope=activity,
        )
        activity_row = note_db.task_store.get_sync_task_activity(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            activity_id=activity.object_id,
        )
        assert activity_row is not None
        assert activity.created_at_client == activity.payload["client_occurred_at"]
        assert activity.payload_hash == activity_row["sync_object_hash"]
        parsed_activity = parse_notes_task_activity_v1(
            activity.payload,
            owner_user_id=OWNER_ID,
            bound_actor_type=str(activity.payload["actor_type"]),
            bound_actor_id=activity.payload.get("actor_id"),
            authenticated_device_id=None,
            trusted_server_origin=True,
        )
        assert activity.server_cursor is not None
        assert note_db.task_store.verify_sync_task_activity_postcondition(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            payload=parsed_activity,
            sync_revision=1,
            sync_object_hash=str(activity.payload_hash),
            sync_server_cursor=activity.server_cursor,
        )
        activity_rows = note_db.task_store.page_legacy_events_for_sync_bootstrap(
            owner_user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            after_created_at=None,
            after_activity_id=None,
            limit=100,
        )
        assert activity_rows == []
        assert activity.object_id == activity_row["id"]
        assert activity.parent_id == activity_row["note_id"]
        assert activity.operation == "upsert"
        assert activity.object_revision == 1
        assert activity.apply_status == "applied"

        resumed = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        final_internal = service.store.get_dataset(
            DATASET_ID,
            owner_user_id=OWNER_ID,
        )
        assert final_internal is not None
        assert final_internal.metadata["notes_task_v1"]["state"] == "ready", (
            final_internal.metadata["notes_task_v1"]
        )
        assert final_internal.metadata["notes_task_activity_v1"]["state"] == "ready", (
            final_internal.metadata["notes_task_activity_v1"]
        )
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(
            resumed.dataset.domains
        ), final_internal.metadata
        assert service.store.get_current_head(
            DATASET_ID,
            "notes.task",
            task_ids[0],
        ) == captured
    finally:
        note_db.close_connection()


def test_rest_mutation_before_task_scan_is_captured_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        task = note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="Before scan",
            projection_status="unlinked",
        )
        _seed_sync_note(service, envelope_id="before-task-scan-note")
        coordinator = NotesTaskCoordinator(
            service=service,
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )
        task_service = NotesTaskService(
            task_coordinator_resolver=lambda **_kwargs: coordinator
        )
        begin = service.store.begin_notes_task_activation
        calls = 0

        def begin_with_mutation(*args: object, **kwargs: object):
            nonlocal calls
            current = begin(*args, **kwargs)
            if calls == 0:
                calls += 1
                task_service.update_task(
                    db=note_db,
                    task_id=TASK_ID,
                    expected_task_version=int(task["version"]),
                    expected_note_version=None,
                    actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                    metadata={"priority": "high"},
                    record_only=True,
                )
            return current

        monkeypatch.setattr(
            service.store,
            "begin_notes_task_activation",
            begin_with_mutation,
        )

        enrolled = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        assert calls == 1
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(enrolled.dataset.domains)
        task_envelopes = [
            item
            for item in service.store.list_envelopes_after(DATASET_ID, 0)
            if item.domain == "notes.task" and item.object_id == TASK_ID
        ]
        activities = [
            item
            for item in service.store.list_envelopes_after(DATASET_ID, 0)
            if item.domain == "notes.task_activity"
        ]
        assert len(task_envelopes) == 1
        assert len(activities) == 1
        assert activities[0].payload["source_kind"] == "rest"
    finally:
        note_db.close_connection()


def test_mcp_mutation_after_task_scan_is_adopted_before_activity_scan(
    tmp_path: Path,
) -> None:
    hook: dict[str, Callable[[], None]] = {"run": lambda: None}
    note_db, service = _activation_stack(
        tmp_path,
        task_after_page=lambda _page: hook["run"](),
    )
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        task = note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="After task scan",
            projection_status="unlinked",
        )
        _seed_sync_note(service, envelope_id="after-task-scan-note")
        coordinator = NotesTaskCoordinator(
            service=service,
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )
        task_service = NotesTaskService(
            task_coordinator_resolver=lambda **_kwargs: coordinator
        )
        calls = 0

        def mutate() -> None:
            nonlocal calls
            if calls:
                return
            calls += 1
            task_service.update_task(
                db=note_db,
                task_id=TASK_ID,
                expected_task_version=int(task["version"]),
                expected_note_version=None,
                actor=TaskActor(
                    actor_type="user",
                    actor_id=OWNER_ID,
                    tool_name="notes.tasks.update",
                ),
                metadata={"priority": "high"},
                record_only=True,
            )

        hook["run"] = mutate
        enrolled = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        assert calls == 1
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(enrolled.dataset.domains)
        activities = [
            item
            for item in service.store.list_envelopes_after(DATASET_ID, 0)
            if item.domain == "notes.task_activity"
        ]
        assert len(activities) == 1
        assert activities[0].payload["source_kind"] == "mcp"
    finally:
        note_db.close_connection()


def test_rest_mutation_after_activity_scan_does_not_duplicate_activity(
    tmp_path: Path,
) -> None:
    hook: dict[str, Callable[[], None]] = {"run": lambda: None}
    note_db, service = _activation_stack(
        tmp_path,
        activity_after_page=lambda _page: hook["run"](),
    )
    legacy_activity_id = "30000000-0000-4000-8000-000000000001"
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        task = note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="After activity scan",
            projection_status="unlinked",
        )
        note_db.task_store.record_task_event(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            event_id=legacy_activity_id,
            task_id=TASK_ID,
            note_id=NOTE_ID,
            event_type="created",
            actor_type="user",
            actor_id=OWNER_ID,
            new_value={
                "text": "After activity scan",
                "status": "open",
                "metadata": {},
            },
        )
        _seed_sync_note(service, envelope_id="after-activity-scan-note")
        coordinator = NotesTaskCoordinator(
            service=service,
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )
        task_service = NotesTaskService(
            task_coordinator_resolver=lambda **_kwargs: coordinator
        )
        calls = 0

        def mutate() -> None:
            nonlocal calls
            if calls:
                return
            calls += 1
            task_service.update_task(
                db=note_db,
                task_id=TASK_ID,
                expected_task_version=int(task["version"]),
                expected_note_version=None,
                actor=TaskActor(actor_type="user", actor_id=OWNER_ID),
                metadata={"priority": "high"},
                record_only=True,
            )

        hook["run"] = mutate
        enrolled = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )

        assert calls == 1
        assert set(NOTES_TASK_SYNC_DOMAINS).issubset(enrolled.dataset.domains)
        activities = [
            item
            for item in service.store.list_envelopes_after(DATASET_ID, 0)
            if item.domain == "notes.task_activity"
        ]
        assert len(activities) == 2
        assert len({item.object_id for item in activities}) == 2
        assert sum(
            item.payload["source_kind"] == "rest" for item in activities
        ) == 1
    finally:
        note_db.close_connection()


def test_source_diagnostic_failure_remains_capture_on_but_not_advertised(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="Malformed source",
            projection_status="unlinked",
        )
        note_db.execute_query(
            "UPDATE note_tasks SET canonical_hash = ? WHERE id = ?",
            ("sha256:" + "f" * 64, TASK_ID),
        )

        enrollment = service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        stored = service.store.get_dataset(DATASET_ID, owner_user_id=OWNER_ID)
        capabilities = service.capabilities(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
        )

        assert stored is not None
        assert stored.metadata["notes_task_v1"]["state"] == "blocked"
        assert stored.metadata["notes_task_v1"]["reason_code"] == (
            "notes_task_source_invalid"
        )
        assert stored.metadata["task_activity_capture_enabled"] is True
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(enrollment.dataset.domains)
        assert set(NOTES_TASK_SYNC_DOMAINS).isdisjoint(
            capabilities.supported_domains
        )
    finally:
        note_db.close_connection()


def test_service_preflight_rejects_direct_lifecycle_activity_with_permissive_adapter(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        service.register_device(
            user_id=OWNER_ID,
            device_id="activation-device",
            display_name="Laptop",
            client_type="chatbook",
            capabilities={
                "requested_domains": list(NOTES_TASK_SYNC_DOMAINS),
                "supported_adapter_versions": dict.fromkeys(
                    NOTES_TASK_SYNC_DOMAINS,
                    [1],
                ),
            },
        )
        service.adapters.register(
            StaticSyncAdapter(
                domain="notes.task_activity",
                supported_adapter_versions={1},
            )
        )

        result = service.push(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            device_id="activation-device",
            envelopes=[
                SyncEnvelopeCreate(
                    dataset_id=DATASET_ID,
                    client_envelope_id="direct-lifecycle",
                    device_id="activation-device",
                    domain="notes.task_activity",
                    operation="upsert",
                    object_id="30000000-0000-4000-8000-000000000001",
                    parent_id=NOTE_ID,
                    object_revision=1,
                    entity_version=1,
                    payload={
                        "event_type": "completed",
                        "corrects_activity_id": None,
                    },
                    payload_hash="sha256:" + "a" * 64,
                    created_at_client="2026-08-24T12:00:00+00:00",
                )
            ],
        )

        assert result.accepted == []
        assert [item.error_code for item in result.rejected] == [
            "notes_task_activity_origin_invalid"
        ]
        assert service.store.list_envelopes_after(DATASET_ID, 0) == []
    finally:
        note_db.close_connection()


def test_direct_client_correction_requires_exact_same_scope_activity(
    tmp_path: Path,
) -> None:
    note_db, service = _activation_stack(tmp_path)
    target_activity_id = "30000000-0000-4000-8000-000000000001"
    device_id = "40000000-0000-4000-8000-000000000001"
    try:
        note_db.note_store.add_note("Tasks", "Body", note_id=NOTE_ID)
        note_db.task_store.create_task(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            task_id=TASK_ID,
            note_id=NOTE_ID,
            text="Correct activity",
            projection_status="unlinked",
        )
        note_db.task_store.record_task_event(
            owner_user_id=OWNER_ID,
            dataset_id="local-unbound",
            event_id=target_activity_id,
            task_id=TASK_ID,
            note_id=NOTE_ID,
            event_type="status_changed",
            actor_type="user",
            actor_id=OWNER_ID,
            old_value={"status": "open"},
            new_value={"status": "done"},
        )
        note_payload = {"title": "Tasks", "content": "Body"}
        note_hash, note_size = canonical_payload_hash(note_payload)
        service.store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=DATASET_ID,
                client_envelope_id="activation-note",
                device_id="server-origin",
                domain="notes.note",
                operation="upsert",
                object_id=NOTE_ID,
                object_revision=1,
                payload=note_payload,
                payload_hash=note_hash,
                payload_size_bytes=note_size,
                created_at_client="2026-08-24T12:00:00+00:00",
                status="accepted",
                apply_status="applied",
            )
        )
        service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            domains=_requested_domains(),
        )
        service.register_device(
            user_id=OWNER_ID,
            device_id=device_id,
            display_name="Laptop",
            client_type="chatbook",
            capabilities={
                "requested_domains": list(NOTES_TASK_SYNC_DOMAINS),
                "supported_adapter_versions": dict.fromkeys(
                    NOTES_TASK_SYNC_DOMAINS,
                    [1],
                ),
            },
        )

        def correction(
            *,
            activity_id: str,
            corrects_activity_id: str,
            envelope_id: str,
        ) -> SyncEnvelopeCreate:
            payload = parse_notes_task_activity_v1(
                {
                    "activity_id": activity_id,
                    "note_id": NOTE_ID,
                    "task_id": TASK_ID,
                    "event_type": "corrected",
                    "actor_type": "user",
                    "actor_id": OWNER_ID,
                    "source_device_id": device_id,
                    "client_occurred_at": "2026-08-24T12:01:00+00:00",
                    "source_kind": "client",
                    "corrects_activity_id": corrects_activity_id,
                    "old_value": {"status": "open"},
                    "new_value": {"status": "done"},
                    "metadata": {},
                },
                owner_user_id=OWNER_ID,
                bound_actor_type="user",
                bound_actor_id=OWNER_ID,
                authenticated_device_id=device_id,
                trusted_server_origin=False,
            )
            return SyncEnvelopeCreate(
                dataset_id=DATASET_ID,
                client_envelope_id=envelope_id,
                device_id=device_id,
                domain="notes.task_activity",
                operation="upsert",
                object_id=activity_id,
                parent_id=NOTE_ID,
                object_revision=1,
                entity_version=1,
                payload=payload.model_dump(mode="json"),
                payload_hash=notes_task_activity_object_hash(
                    payload,
                    revision=1,
                    deleted=False,
                ),
                created_at_client="2026-08-24T12:01:00+00:00",
            )

        accepted = service.push(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            device_id=device_id,
            envelopes=[
                correction(
                    activity_id="30000000-0000-4000-8000-000000000002",
                    corrects_activity_id=target_activity_id,
                    envelope_id="valid-correction",
                )
            ],
        )
        missing_target = "30000000-0000-4000-8000-000000000099"
        rejected = service.push(
            user_id=OWNER_ID,
            dataset_id=DATASET_ID,
            device_id=device_id,
            envelopes=[
                correction(
                    activity_id="30000000-0000-4000-8000-000000000003",
                    corrects_activity_id=missing_target,
                    envelope_id="missing-correction",
                )
            ],
        )

        assert len(accepted.accepted) == 1
        assert accepted.rejected == []
        assert rejected.accepted == []
        assert [item.error_code for item in rejected.rejected] == [
            "adapter_deferred"
        ]
        assert missing_target not in rejected.rejected[0].message
    finally:
        note_db.close_connection()
