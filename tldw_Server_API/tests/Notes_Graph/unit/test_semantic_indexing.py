"""Unit contracts for fenced Notes semantic indexing and observability."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticGenerationIntegrity,
    SemanticManifestPublication,
    SemanticWorkClaimState,
    SemanticWorkItem,
    SemanticWorkKind,
)
from tldw_Server_API.app.core.Notes_Graph import semantic_indexing
from tldw_Server_API.app.core.Notes_Graph.semantic_content import build_semantic_chunks
from tldw_Server_API.app.core.Notes_Graph.semantic_indexing import (
    NoteVersionRef,
    SemanticGenerationBuilder,
    VersionedNoteSnapshot,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_observability import (
    SemanticObservationError,
    build_semantic_audit_event,
    build_semantic_metric_event,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticAuthorityState,
    SemanticExecutionFence,
    SemanticIndexingError,
    SemanticPublicationService,
    revalidate_execution_fence,
    validate_execution_fence,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import SemanticVector

pytestmark = pytest.mark.unit
GENERATION_FENCE = "-".join(("job", "fence", "a"))
STALE_GENERATION_FENCE = "-".join(("stale", "fence"))


def _fence() -> SemanticExecutionFence:
    return SemanticExecutionFence(
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        generation_id="generation-a",
        generation_fencing_token=GENERATION_FENCE,
        configuration_revision=4,
        capability_revision="capability-a",
        disclosure_hash="disclosure-a",
        provider="openai",
        model="embedding-model-a",
        model_revision="revision-a",
        endpoint_origin="https://api.example.test",
        credential_source="server_default",
        endpoint_origin_revision="origin-a",
        compatibility_hash="compatibility-a",
        dimensions=2,
        vector_backend="chromadb",
    )


def _authority() -> SemanticAuthorityState:
    fence = _fence()
    return SemanticAuthorityState(
        user_exists=True,
        owner_authorized=True,
        semantic_manage_allowed=True,
        desired_enabled=True,
        owner_user_id=fence.owner_user_id,
        dataset_id=fence.dataset_id,
        generation_id=fence.generation_id,
        generation_fencing_token=fence.generation_fencing_token,
        configuration_revision=fence.configuration_revision,
        capability_revision=fence.capability_revision,
        disclosure_hash=fence.disclosure_hash,
        provider=fence.provider,
        model=fence.model,
        model_revision=fence.model_revision,
        endpoint_origin=fence.endpoint_origin,
        credential_source=fence.credential_source,
        endpoint_origin_revision=fence.endpoint_origin_revision,
        endpoint_policy_allowed=True,
        compatibility_hash=fence.compatibility_hash,
        dimensions=fence.dimensions,
        vector_backend=fence.vector_backend,
        vector_capable=True,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"user_exists": False}, "notes_semantic_user_missing"),
        ({"owner_authorized": False}, "notes_semantic_owner_authority_revoked"),
        (
            {"semantic_manage_allowed": False},
            "notes_semantic_manage_permission_revoked",
        ),
        ({"desired_enabled": False}, "notes_semantic_index_disabled"),
        ({"owner_user_id": "other-owner"}, "notes_semantic_owner_authority_revoked"),
        ({"dataset_id": "other-dataset"}, "notes_semantic_owner_authority_revoked"),
        ({"capability_revision": "drift"}, "notes_semantic_capability_drift"),
        ({"disclosure_hash": "drift"}, "notes_semantic_disclosure_drift"),
        ({"configuration_revision": 5}, "notes_semantic_configuration_drift"),
        (
            {"generation_fencing_token": STALE_GENERATION_FENCE},
            "notes_semantic_generation_fence_mismatch",
        ),
        ({"generation_id": "other-generation"}, "notes_semantic_generation_fence_mismatch"),
        ({"provider": "other"}, "notes_semantic_provider_model_drift"),
        ({"model": "other"}, "notes_semantic_provider_model_drift"),
        ({"model_revision": "other"}, "notes_semantic_model_revision_drift"),
        (
            {"endpoint_origin": "https://other.example.test"},
            "notes_semantic_endpoint_origin_drift",
        ),
        ({"credential_source": "user"}, "notes_semantic_credential_scope_drift"),
        ({"endpoint_policy_allowed": False}, "notes_semantic_endpoint_policy_denied"),
        ({"endpoint_origin_revision": "other"}, "notes_semantic_endpoint_drift"),
        ({"compatibility_hash": "other"}, "notes_semantic_compatibility_drift"),
        ({"dimensions": 3}, "notes_semantic_dimension_drift"),
        ({"vector_backend": "pgvector"}, "notes_semantic_vector_capability_drift"),
        ({"vector_capable": False}, "notes_semantic_vector_capability_drift"),
    ],
)
def test_complete_execution_fence_fails_closed_with_stable_codes(
    change: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(SemanticIndexingError) as exc_info:
        validate_execution_fence(_fence(), replace(_authority(), **change))

    assert exc_info.value.code == code
    assert str(exc_info.value) == code
    assert "owner-a" not in str(exc_info.value)
    assert "generation-a" not in str(exc_info.value)


def test_complete_execution_fence_accepts_the_exact_authoritative_identity() -> None:
    assert validate_execution_fence(_fence(), _authority()) == _authority()


def test_execution_identity_repr_is_fully_redacted() -> None:
    serialized = f"{_fence()!r}{_authority()!r}"

    for forbidden in (
        "owner-a",
        "dataset-a",
        "generation-a",
        "embedding-model-a",
        "chromadb",
        "https://api.example.test",
        "server_default",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_sync_authority_revalidator_does_not_block_event_loop() -> None:
    heartbeat_seen = asyncio.Event()

    def slow_revalidate(_fence_value: SemanticExecutionFence) -> SemanticAuthorityState:
        time.sleep(0.05)
        return _authority()

    async def heartbeat() -> None:
        await asyncio.sleep(0.005)
        heartbeat_seen.set()

    heartbeat_task = asyncio.create_task(heartbeat())
    result = await revalidate_execution_fence(slow_revalidate, _fence())
    await heartbeat_task

    assert result == _authority()
    assert heartbeat_seen.is_set()


def test_observability_exposes_only_allowlisted_low_cardinality_fields() -> None:
    metric = build_semantic_metric_event(
        operation="initial_build",
        status="degraded",
        backend="chromadb",
        error_code="note_failed",
        value=2,
    )
    audit = build_semantic_audit_event(
        event="generation_publication",
        status="degraded",
        reason="note_failed",
        counts={"indexed": 2, "excluded": 1, "failed": 1, "pending": 0},
    )

    assert metric.labels == {
        "operation": "initial_build",
        "status": "degraded",
        "backend": "chromadb",
        "error_code": "note_failed",
    }
    assert audit.fields == {
        "status": "degraded",
        "reason": "note_failed",
        "indexed": 2,
        "excluded": 1,
        "failed": 1,
        "pending": 0,
    }
    serialized = f"{metric!r}{audit!r}"
    for forbidden in (
        "owner-a",
        "dataset-a",
        "generation-a",
        "embedding-model-a",
        "https://",
        "collection",
        "table",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (build_semantic_metric_event, {"operation": "owner-a", "status": "success", "value": 1}),
        (
            build_semantic_audit_event,
            {"event": "generation_publication", "status": "success", "owner_id": "owner-a"},
        ),
    ],
)
def test_observability_rejects_unbounded_or_identifier_fields(builder, kwargs) -> None:
    with pytest.raises(SemanticObservationError):
        builder(**kwargs)


@pytest.mark.parametrize("value", [-1, 1.5, math.inf, -math.inf, math.nan, True])
def test_total_metric_requires_finite_non_negative_integer_increment(value) -> None:
    with pytest.raises(SemanticObservationError, match="notes_semantic_observation_value_invalid"):
        build_semantic_metric_event(
            operation="cleanup",
            status="success",
            backend="chromadb",
            value=value,
        )


def test_generation_cleanup_capacity_must_cover_the_run_chunk_cap() -> None:
    with pytest.raises(
        ValueError,
        match="max_chunks_per_run cannot exceed max_cleanup_vectors_per_run",
    ):
        SemanticIndexSettings(
            max_chunks_per_run=10_001,
            max_cleanup_vectors_per_run=10_000,
        )


class _NoopVectors:
    async def fetch(self, dataset_id, generation_id, vector_ids):
        return ()


def _empty_integrity() -> SemanticGenerationIntegrity:
    return SemanticGenerationIntegrity(
        generation_id="generation-a",
        generation_fencing_token=GENERATION_FENCE,
        expected_note_count=0,
        expected_chunk_count=0,
        published_note_count=0,
        published_chunk_count=0,
        terminal_note_count=0,
        indexed_note_count=0,
        excluded_note_count=0,
        failed_note_count=0,
        pending_note_count=0,
        tombstoned_note_count=0,
        eligible_note_count=0,
        waived_chunk_count=0,
        vector_ids=(),
        manifest_hash="sha256:" + "0" * 64,
        dimensions=2,
        compatibility_hash="compatibility-a",
        terminal_error_code=None,
    )


@pytest.mark.asyncio
async def test_cancelled_activation_drains_commit_and_returns_single_receipt() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStore:
        calls = 0

        def get_generation_integrity(self, dataset_id, generation_id):
            return _empty_integrity()

        def assert_generation_activatable(self, integrity):
            return None

        def activate_generation_verified(self, **kwargs):
            self.calls += 1
            started.set()
            assert release.wait(timeout=5)
            return SimpleNamespace(configuration_revision=5, semantic_index_revision=9)

    store = BlockingStore()
    service = SemanticPublicationService(
        store=store,
        vectors=_NoopVectors(),
        revalidate=lambda fence: _authority(),
        clock=lambda: None,
        receipt_factory=lambda: "receipt-once",
    )
    operation = asyncio.create_task(service.activate(_fence()))
    while not started.is_set():
        await asyncio.sleep(0)
    assert operation.cancel("first") is True
    await asyncio.sleep(0)
    assert operation.cancel("repeat") is True
    release.set()

    receipt = await operation

    assert receipt.receipt == "receipt-once"
    assert receipt.configuration_revision == 5
    assert store.calls == 1


@pytest.mark.asyncio
async def test_cancelled_uncommitted_activation_drains_then_preserves_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStore:
        calls = 0

        def get_generation_integrity(self, dataset_id, generation_id):
            return _empty_integrity()

        def assert_generation_activatable(self, integrity):
            return None

        def activate_generation_verified(self, **kwargs):
            self.calls += 1
            started.set()
            assert release.wait(timeout=5)
            return None

    store = BlockingStore()
    service = SemanticPublicationService(
        store=store,
        vectors=_NoopVectors(),
        revalidate=lambda fence: _authority(),
        clock=lambda: None,
        receipt_factory=lambda: "receipt-not-committed",
    )
    operation = asyncio.create_task(service.activate(_fence()))
    while not started.is_set():
        await asyncio.sleep(0)
    operation.cancel("before-commit")
    release.set()

    with pytest.raises(asyncio.CancelledError, match="before-commit"):
        await operation
    assert store.calls == 1


def _claimed_work(kind: SemanticWorkKind) -> SemanticWorkItem:
    return SemanticWorkItem(  # nosec B106 - fencing tokens are non-secret CAS identifiers
        id="work-a",
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        kind=kind,
        note_id="note-a",
        generation_id="generation-a",
        dirty_generation=1,
        fencing_token="work-fence-a",
        claim_state=SemanticWorkClaimState.CLAIMED,
        attempt_count=0,
        next_eligible_at="2026-08-30T12:00:00Z",
        claim_token="claim-a",
        claimed_at="2026-08-30T12:00:00Z",
        error_code=None,
        created_at="2026-08-30T12:00:00Z",
        updated_at="2026-08-30T12:00:00Z",
    )


@pytest.mark.parametrize("kind", [SemanticWorkKind.INDEX_NOTE, SemanticWorkKind.DELETE_NOTE_VECTORS])
@pytest.mark.parametrize("phase", ["before_commit", "during_commit", "after_commit"])
@pytest.mark.asyncio
async def test_cancelled_manifest_commit_is_drained_and_reports_truthful_result(
    kind: SemanticWorkKind,
    phase: str,
) -> None:
    started = threading.Event()
    committed = threading.Event()
    release = threading.Event()
    publication = SemanticManifestPublication(
        note_id="note-a",
        generation_id="generation-a",
        old_vector_ids=(),
        new_vector_ids=() if kind is SemanticWorkKind.DELETE_NOTE_VECTORS else ("vector-a",),
        dirty_generation=1,
        manifest_hash=None,
    )

    class BlockingStore:
        calls = 0

        def stage_obsolete_vector_cleanup(self, **kwargs):
            return len(kwargs["vector_ids"])

        def _commit(self):
            self.calls += 1
            started.set()
            if phase == "after_commit":
                committed.set()
            assert release.wait(timeout=5)
            if phase == "before_commit":
                return None
            committed.set()
            return publication

        def publish_indexed_manifest(self, **kwargs):
            return self._commit()

        def publish_note_tombstone(self, **kwargs):
            return self._commit()

    class PublicationVectors(_NoopVectors):
        async def upsert(self, dataset_id, generation_id, vectors):
            return len(vectors)

    store = BlockingStore()
    service = SemanticPublicationService(
        store=store,
        vectors=PublicationVectors(),
        revalidate=lambda fence: _authority(),
        clock=lambda: None,
        receipt_factory=lambda: "unused",
    )
    claim = _claimed_work(kind)
    if kind is SemanticWorkKind.INDEX_NOTE:
        chunks = build_semantic_chunks(
            generation_id="generation-a",
            note_id="note-a",
            title="Title",
            content="Body",
            content_version=1,
        )
        coroutine = service.publish_note(
            _fence(),
            claim,
            chunks,
            tuple(SemanticVector(chunk.vector_id, (1.0, 2.0)) for chunk in chunks),
        )
    else:
        coroutine = service.publish_tombstone(_fence(), claim)
    operation = asyncio.create_task(coroutine)
    try:
        while not started.is_set():
            await asyncio.sleep(0)
        if phase == "after_commit":
            while not committed.is_set():
                await asyncio.sleep(0)
        assert operation.cancel("first-cancel") is True
        await asyncio.sleep(0)
        assert operation.cancel("repeat-cancel") is True
        release.set()
        if phase == "before_commit":
            with pytest.raises(asyncio.CancelledError, match="first-cancel"):
                await operation
        else:
            assert await operation == publication
        assert store.calls == 1
    finally:
        release.set()


class _SnapshotReader:
    def __init__(self, snapshots: tuple[VersionedNoteSnapshot, ...]) -> None:
        self.snapshots = {snapshot.note_id: snapshot for snapshot in snapshots}

    async def list_note_versions(self, owner_user_id, dataset_id, *, limit):
        return tuple(
            NoteVersionRef(snapshot.note_id, snapshot.content_version)
            for snapshot in self.snapshots.values()
        )[:limit]

    async def read_note_version(
        self,
        owner_user_id,
        dataset_id,
        note_id,
        content_version,
    ):
        return self.snapshots.get(note_id)


def _snapshot_builder(
    reader: _SnapshotReader,
    *,
    settings: SemanticIndexSettings,
) -> SemanticGenerationBuilder:
    return SemanticGenerationBuilder(
        store=SimpleNamespace(),
        note_reader=reader,
        embedder=SimpleNamespace(),
        vectors=SimpleNamespace(),
        revalidate=lambda fence: _authority(),
        compatibility_hash_for_dimension=lambda resolved: "compatibility-a",
        settings=settings,
        clock=lambda: None,
        receipt_factory=lambda: "unused",
    )


@pytest.mark.asyncio
async def test_snapshot_cpu_work_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    heartbeat_seen = asyncio.Event()
    original = semantic_indexing._build_snapshot_note

    def slow_build(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(semantic_indexing, "_build_snapshot_note", slow_build)
    builder = _snapshot_builder(
        _SnapshotReader((VersionedNoteSnapshot("note-a", None, "Body", 1),)),
        settings=SemanticIndexSettings(),
    )

    async def heartbeat() -> None:
        await asyncio.sleep(0.005)
        heartbeat_seen.set()

    heartbeat_task = asyncio.create_task(heartbeat())
    await builder._read_snapshot(_fence())
    await heartbeat_task

    assert heartbeat_seen.is_set()


@pytest.mark.asyncio
async def test_snapshot_byte_batching_uses_one_cumulative_provider_request_budget() -> None:
    settings = SemanticIndexSettings(
        max_active_notes=2,
        max_chunk_code_points=4,
        max_chunks_per_note=4,
        max_chunks_per_run=4,
        max_provider_input_bytes=5,
        max_provider_batch_inputs=4,
        max_provider_batch_bytes=5,
        max_provider_bytes_per_run=64,
        max_provider_requests_per_run=3,
        max_query_vectors_per_call=2,
        max_cleanup_vectors_per_run=4,
    )
    builder = _snapshot_builder(
        _SnapshotReader(
            (
                VersionedNoteSnapshot("note-a", None, "abcdefgh", 1),
                VersionedNoteSnapshot("note-b", None, "ijklmnop", 1),
            )
        ),
        settings=settings,
    )

    with pytest.raises(SemanticIndexingError, match="notes_semantic_run_limit_exceeded"):
        await builder._read_snapshot(_fence())
