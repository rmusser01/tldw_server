from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.materializers import MaterializationResult
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
    GuardedProductMutationIdentityError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncDataset,
    SyncDatasetCreate,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginBatchResult,
    ServerOriginMutationStep,
    SyncServerOriginBatchIdempotencyConflictError,
    SyncServerOriginBatchMaterializationError,
    capture_server_origin_mutation_batch,
    resume_server_origin_mutation_group,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


@dataclass(slots=True)
class _PlannedDependencyAdapter:
    domain: SyncDomain
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ):
        del dataset
        assert context is not None
        assert context.get_head is not None

        if envelope.parent_id is not None:
            parent = context.get_head(envelope.domain, envelope.parent_id)
            if parent is None or parent.operation == "tombstone":
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="planned_parent_missing",
                    message="planned parent must precede its child",
                )

        requirements = envelope.payload.get("requires", [])
        for requirement in requirements:
            required_domain = requirement["domain"]
            required_object_id = requirement["object_id"]
            head = context.get_head(required_domain, required_object_id)
            if head is None or head.operation == "tombstone":
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="planned_dependency_missing",
                    message="planned dependency must precede its consumer",
                )

        expected_prior_hash = envelope.payload.get("expected_prior_hash")
        if expected_prior_hash is not None:
            prior = next(
                (
                    item
                    for item in context.prior_envelopes
                    if item.object_id == envelope.object_id
                    and item.client_envelope_id != envelope.client_envelope_id
                ),
                None,
            )
            if (
                prior is None
                or prior.payload_hash != expected_prior_hash
                or envelope.base_object_hash != prior.payload_hash
                or envelope.base_object_revision != prior.object_revision
            ):
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="planned_head_missing",
                    message="later updates must use the earlier planned head",
                )

        if envelope.payload.get("adapter_conflict") is True:
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=envelope.domain,
                entity_id=envelope.object_id,
                conflict_type="planned_conflict",
            )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


@dataclass(slots=True)
class _RecordingMaterializer:
    domain: SyncDomain
    calls: list[int] = field(default_factory=list)
    projections: list[str] = field(default_factory=list)
    fail_once_at: int | None = None
    conflict_at: int | None = None

    def apply(self, envelope: SyncEnvelope, *, store: SyncV2Store) -> MaterializationResult:
        assert envelope.server_cursor is not None
        assert envelope.mutation_step is not None
        self.calls.append(envelope.mutation_step)
        if self.fail_once_at == envelope.mutation_step:
            self.fail_once_at = None
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="injected_projection_failure",
                apply_error_message="injected projection failure",
            )
            return MaterializationResult(
                status="failed",
                error_code="injected_projection_failure",
            )
        if self.conflict_at == envelope.mutation_step:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="injected_projection_conflict",
                apply_error_message="injected projection conflict",
            )
            return MaterializationResult(
                status="conflict",
                conflict_type="injected_projection_conflict",
            )
        self.projections.append(envelope.object_id)
        store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
        return MaterializationResult(status="applied")


@pytest.fixture()
def batch_service(tmp_path: Path) -> tuple[SyncV2Service, _RecordingMaterializer]:
    domains: list[SyncDomain] = ["chat.conversation", "chat.message", "notes.note"]
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            encryption_policy="server_trusted_v1",
            domains=domains,
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    materializer = _RecordingMaterializer(domain="chat.conversation")
    service = SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry(
            [_PlannedDependencyAdapter(domain=domain) for domain in domains]
        ),
        materializers=dict.fromkeys(domains, materializer),
        clock=lambda: "2026-08-08T20:00:00+00:00",
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="encrypted_volume",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            )
        ),
    )
    return service, materializer


def _step(
    object_id: str,
    *,
    domain: SyncDomain = "chat.conversation",
    payload: dict[str, object] | None = None,
    parent_id: str | None = None,
) -> ServerOriginMutationStep:
    return ServerOriginMutationStep(
        domain=domain,
        operation="append" if domain == "chat.message" else "upsert",
        object_id=object_id,
        payload=payload or {"name": object_id},
        parent_id=parent_id,
    )


def _capture(
    service: SyncV2Service,
    steps: list[ServerOriginMutationStep],
    *,
    key: str = "request-1",
):
    return capture_server_origin_mutation_batch(
        service=service,
        user_id="user-1",
        steps=steps,
        source="notes_api",
        idempotency_key=key,
    )


def test_trusted_bootstrap_id_requires_source_step_verifier(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service

    with pytest.raises(SyncStoreError, match="bootstrap.*verifier"):
        capture_server_origin_mutation_batch(
            service=service,
            user_id="user-1",
            steps=[_step("bootstrap-unverified")],
            source="notes-organization-bootstrap",
            idempotency_key="bootstrap-unverified",
            trusted_notes_organization_bootstrap_id="bootstrap-1",
        )

    assert materializer.calls == []
    assert service.store.list_envelopes_after("dataset-1", 0) == []


def test_guard_identity_mismatch_prevents_every_product_materializer_call(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service
    guard = GuardedProductMutation(
        expected_domain="notes.link",
        expected_object_id="11111111-1111-4111-8111-111111111111",
        before=lambda _conn: None,
        after=lambda _conn, _identity: None,
    )

    with pytest.raises(GuardedProductMutationIdentityError):
        capture_server_origin_mutation_batch(
            service=service,
            user_id="user-1",
            steps=[_step("conversation-1")],
            source="notes_api",
            idempotency_key="guard-mismatch",
            guarded_mutation=guard,
        )

    assert materializer.calls == []


def test_batch_rejects_explicit_nonpositive_object_revision(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service

    with pytest.raises(SyncStoreError, match="object revision must be positive"):
        _capture(
            service,
            [replace(_step("invalid-revision"), object_revision=0)],
            key="invalid-revision",
        )

    assert materializer.calls == []
    assert service.store.list_envelopes_after("dataset-1", 0) == []


def test_batch_evaluates_updates_parents_and_relationships_against_planned_heads(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service
    first_hash = "sha256:e25025fd2a8a992b9994062a80303f2561cbc09fde2b530aa0776fbcc2d69de7"
    steps = [
        _step("folder-1", payload={"name": "Drafts"}),
        _step(
            "folder-1",
            payload={"name": "Published", "expected_prior_hash": first_hash},
        ),
        _step("folder-child", parent_id="folder-1"),
        _step("note-1", domain="notes.note", payload={"title": "Note", "content": "Body"}),
        _step(
            "folder-1:note-1",
            domain="chat.message",
            payload={
                "folder_id": "folder-1",
                "note_id": "note-1",
                "requires": [
                    {"domain": "chat.conversation", "object_id": "folder-1"},
                    {"domain": "notes.note", "object_id": "note-1"},
                ],
            },
        ),
    ]

    result = _capture(service, steps)

    assert result.fully_applied is True
    assert [item.mutation_step for item in result.envelopes] == [0, 1, 2, 3, 4]
    assert materializer.calls == [0, 1, 2, 3, 4]


def test_evaluation_failure_appends_nothing_and_never_projects(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service
    steps = [
        _step(
            "folder-1:note-1",
            domain="chat.message",
            payload={
                "requires": [
                    {"domain": "chat.conversation", "object_id": "folder-1"},
                    {"domain": "notes.note", "object_id": "note-1"},
                ]
            },
        ),
        _step("folder-1"),
        _step("note-1", domain="notes.note"),
    ]

    with pytest.raises(SyncStoreError, match="planned dependency"):
        _capture(service, steps)

    assert service.store.list_envelopes_after("dataset-1", 0, limit=100) == []
    assert materializer.projections == []


def test_append_failure_appends_nothing_and_never_projects(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, materializer = batch_service

    def fail_append(envelopes):
        del envelopes
        raise SyncStoreError("injected append failure")

    monkeypatch.setattr(service.store, "insert_envelopes_atomic", fail_append)

    with pytest.raises(SyncStoreError, match="sync_server_origin_batch_append_failed"):
        _capture(service, [_step("folder-1"), _step("folder-2")])

    assert service.store.list_envelopes_after("dataset-1", 0, limit=100) == []
    assert materializer.projections == []


def test_failed_step_preserves_applied_prefix_and_resume_starts_at_failure(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service
    materializer.fail_once_at = 2

    with pytest.raises(SyncServerOriginBatchMaterializationError) as exc_info:
        _capture(service, [_step(f"folder-{index}") for index in range(4)])

    failed_result = exc_info.value.result
    assert failed_result.fully_applied is False
    assert [item.apply_status for item in failed_result.envelopes] == [
        "applied",
        "applied",
        "failed",
        "pending",
    ]
    assert materializer.calls == [0, 1, 2]

    resumed = resume_server_origin_mutation_group(
        service=service,
        dataset_id="dataset-1",
        mutation_group_id=failed_result.envelopes[0].mutation_group_id or "",
    )

    assert resumed.fully_applied is True
    assert [item.apply_status for item in resumed.envelopes] == ["applied"] * 4
    assert materializer.calls == [0, 1, 2, 2, 3]


def test_conflicted_step_blocks_itself_and_every_later_step(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, materializer = batch_service
    materializer.conflict_at = 1
    observed_connections: list[object | None] = []
    original_insert = service.store.db.insert_conflict

    def record_insert(conflict, *, connection=None):
        observed_connections.append(connection)
        return original_insert(conflict, connection=connection)

    monkeypatch.setattr(service.store.db, "insert_conflict", record_insert)

    with pytest.raises(SyncServerOriginBatchMaterializationError) as exc_info:
        _capture(service, [_step(f"folder-{index}") for index in range(3)])

    blocked = exc_info.value.result
    assert [item.apply_status for item in blocked.envelopes] == [
        "applied",
        "conflict",
        "pending",
    ]
    assert materializer.calls == [0, 1]
    assert len(service.store.list_conflicts("dataset-1")) == 1
    assert len(observed_connections) == 1
    assert observed_connections[0] is not None

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        resume_server_origin_mutation_group(
            service=service,
            dataset_id="dataset-1",
            mutation_group_id=blocked.envelopes[0].mutation_group_id or "",
        )
    assert materializer.calls == [0, 1]


@pytest.mark.parametrize(
    "statuses",
    [
        ("pending", "applied", "pending"),
        ("applied", "conflict", "applied"),
    ],
)
def test_resume_rejects_non_prefix_applied_status_before_materializing(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    statuses: tuple[str, str, str],
) -> None:
    service, materializer = batch_service
    result = _capture(service, [_step(f"folder-{index}") for index in range(3)])
    group_id = result.envelopes[0].mutation_group_id
    assert group_id is not None
    for step, status in enumerate(statuses):
        service.store.db.execute(
            "UPDATE sync_envelopes SET apply_status = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            (status, group_id, step),
        )
    materializer.calls.clear()

    with pytest.raises(SyncIdempotencyConflictError, match="non-prefix applied"):
        resume_server_origin_mutation_group(
            service=service,
            dataset_id="dataset-1",
            mutation_group_id=group_id,
        )

    assert materializer.calls == []


def test_same_idempotency_key_replays_group_and_changed_plan_conflicts(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, materializer = batch_service
    first = _capture(service, [_step("folder-1"), _step("folder-2")])

    replay = _capture(service, [_step("folder-1"), _step("folder-2")])

    assert [item.client_envelope_id for item in replay.envelopes] == [
        item.client_envelope_id for item in first.envelopes
    ]
    assert materializer.calls == [0, 1]

    with pytest.raises(SyncServerOriginBatchIdempotencyConflictError) as exc_info:
        _capture(service, [_step("folder-1"), _step("folder-changed")])
    assert exc_info.value.error_code == "sync_server_origin_batch_idempotency_conflict"


def test_concurrent_groups_cannot_append_from_the_same_canonical_head(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = batch_service
    _capture(service, [_step("folder-1", payload={"name": "Original"})], key="seed")
    append_barrier = Barrier(2)
    original_append = service.store.insert_envelopes_atomic

    def append_after_both_preflights(envelopes, **kwargs):
        append_barrier.wait()
        return original_append(envelopes, **kwargs)

    monkeypatch.setattr(
        service.store,
        "insert_envelopes_atomic",
        append_after_both_preflights,
    )

    def capture(name: str, key: str):
        try:
            return _capture(
                service,
                [_step("folder-1", payload={"name": name})],
                key=key,
            )
        except Exception as exc:  # noqa: BLE001 - the losing result is asserted below.
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: capture(*item),
                [("First", "concurrent-1"), ("Second", "concurrent-2")],
            )
        )

    successes = [result for result in results if isinstance(result, ServerOriginBatchResult)]
    failures = [result for result in results if isinstance(result, Exception)]
    history = service.store.list_envelopes_for_entity(
        "dataset-1",
        "chat.conversation",
        entity_id="folder-1",
        limit=10,
    )

    assert len(successes) == 1
    assert len(failures) == 1
    assert getattr(failures[0], "error_code", None) == "sync_server_origin_batch_append_failed"
    assert len([envelope for envelope in history if envelope.status == "accepted"]) == 2


def test_later_appended_group_waits_for_earlier_pending_projection(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = batch_service
    first_appended = Event()
    release_first = Event()
    original_append = service.store.insert_envelopes_atomic

    def pause_first_after_append(envelopes, **kwargs):
        inserted = original_append(envelopes, **kwargs)
        if inserted[0].payload["name"] == "First":
            first_appended.set()
            assert release_first.wait(5)
        return inserted

    monkeypatch.setattr(service.store, "insert_envelopes_atomic", pause_first_after_append)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            _capture,
            service,
            [_step("folder-first", payload={"name": "First"})],
            key="cursor-order-first",
        )
        assert first_appended.wait(5)
        try:
            second: object = _capture(
                service,
                [_step("folder-second", payload={"name": "Second"})],
                key="cursor-order-second",
            )
        except Exception as exc:  # noqa: BLE001 - retryable result asserted below.
            second = exc
        finally:
            release_first.set()
        first = first_future.result()

    assert first.fully_applied is True
    assert isinstance(second, SyncServerOriginBatchMaterializationError)
    assert second.retryable is True
    assert [item.apply_status for item in second.result.envelopes] == ["pending"]

    resumed = resume_server_origin_mutation_group(
        service=service,
        dataset_id="dataset-1",
        mutation_group_id=second.result.envelopes[0].mutation_group_id or "",
    )
    assert resumed.fully_applied is True


def test_sqlite_dataset_materialization_guard_serializes_nonoverlapping_groups(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, _ = batch_service
    stored = _capture(service, [_step("folder-1"), _step("folder-2")])
    second_group = tuple(
        replace(
            envelope,
            object_id=f"unrelated-{envelope.object_id}",
            mutation_group_id="second-logical-group",
        )
        for envelope in stored.envelopes
    )
    start = Barrier(2)
    state_lock = Lock()
    overlap = Event()
    active: set[str] = set()

    def hold_group(group_id: str, envelopes: tuple[SyncEnvelope, ...]) -> None:
        start.wait()
        with service.store.materialization_guard(envelopes):
            with state_lock:
                if active:
                    overlap.set()
                active.add(group_id)
            overlap.wait(0.2)
            with state_lock:
                active.remove(group_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(hold_group, "first", stored.envelopes),
            executor.submit(hold_group, "second", second_group),
        ]
        for future in futures:
            future.result()

    assert not overlap.is_set()
    assert service.store.db.execute(
        "SELECT COUNT(*) AS count FROM sync_materialization_locks WHERE dataset_id = ?",
        ("dataset-1",),
    ).rows[0]["count"] == 1


def test_materialization_does_not_advance_past_earlier_conflicted_envelope(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, _ = batch_service
    blocker = _capture(service, [_step("folder-blocker")], key="conflicted-blocker")
    blocker_cursor = blocker.envelopes[0].server_cursor
    assert blocker_cursor is not None
    service.store.mark_envelope_apply_status(blocker_cursor, apply_status="conflict")
    later = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="later-after-conflict",
            domain="chat.conversation",
            operation="upsert",
            object_id="unrelated-later-object",
            object_revision=1,
            payload={"name": "Later"},
            payload_hash="sha256:later",
            status="accepted",
        )
    )

    with pytest.raises(SyncStoreError, match="sync_projection_predecessor_unresolved"):
        with service.store.materialization_guard([later]):
            pass


def test_organization_domain_write_requires_ready_metadata_when_present(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, materializer = batch_service
    dataset = service.store.get_dataset("dataset-1")
    assert dataset is not None
    initializing = replace(
        dataset,
        domains=[*dataset.domains, "notes.folder"],
        metadata={
            **dataset.metadata,
            "notes_organization_v1": {"state": "initializing"},
        },
    )
    monkeypatch.setattr(service.store, "list_datasets_for_user", lambda user_id: [initializing])

    with pytest.raises(SyncStoreError, match="notes_organization_sync_not_ready"):
        _capture(service, [_step("folder-1", domain="notes.folder")])

    assert materializer.projections == []


def test_resume_revalidates_stored_plan_fingerprint(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
) -> None:
    service, _ = batch_service
    result = _capture(service, [_step("folder-1"), _step("folder-2")])
    group_id = result.envelopes[0].mutation_group_id
    assert group_id is not None
    service.store.db.execute(
        "UPDATE sync_envelopes SET payload_json = ? WHERE mutation_group_id = ? AND mutation_step = ?",
        ('{"name":"tampered"}', group_id, 1),
    )

    with pytest.raises(SyncIdempotencyConflictError, match="stored mutation group"):
        resume_server_origin_mutation_group(
            service=service,
            dataset_id="dataset-1",
            mutation_group_id=group_id,
        )


@pytest.mark.parametrize(
    ("tamper_sql", "tampered_value"),
    [
        (
            "UPDATE sync_envelopes SET client_envelope_id = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            "tampered-envelope-id",
        ),
        (
            "UPDATE sync_envelopes SET payload_hash = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "UPDATE sync_envelopes SET payload_size_bytes = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            999,
        ),
        (
            "UPDATE sync_envelopes SET object_revision = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            99,
        ),
        (
            "UPDATE sync_envelopes SET base_server_cursor = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            88,
        ),
        (
            "UPDATE sync_envelopes SET base_object_revision = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            88,
        ),
        (
            "UPDATE sync_envelopes SET base_object_hash = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        ),
        (
            "UPDATE sync_envelopes SET device_id = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            "tampered-device",
        ),
        (
            "UPDATE sync_envelopes SET schema_version = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            2,
        ),
        (
            "UPDATE sync_envelopes SET encryption_metadata_json = ? "
            "WHERE mutation_group_id = ? AND mutation_step = ?",
            '{"policy":"tampered"}',
        ),
    ],
)
def test_resume_rejects_materialization_fingerprint_tampering_before_apply(
    batch_service: tuple[SyncV2Service, _RecordingMaterializer],
    tamper_sql: str,
    tampered_value: object,
) -> None:
    service, materializer = batch_service
    result = _capture(service, [_step("folder-1"), _step("folder-1")])
    group_id = result.envelopes[0].mutation_group_id
    assert group_id is not None
    service.store.db.execute(tamper_sql, (tampered_value, group_id, 1))
    service.store.db.execute(
        "UPDATE sync_envelopes SET apply_status = ? "
        "WHERE mutation_group_id = ? AND mutation_step = ?",
        ("pending", group_id, 1),
    )
    materializer.calls.clear()

    with pytest.raises(SyncIdempotencyConflictError, match="fingerprint"):
        resume_server_origin_mutation_group(
            service=service,
            dataset_id="dataset-1",
            mutation_group_id=group_id,
        )

    assert materializer.calls == []
