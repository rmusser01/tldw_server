"""Unit contracts for fenced Notes semantic indexing and observability."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticGenerationIntegrity,
    SemanticManifestPublication,
    SemanticWorkClaimState,
    SemanticWorkItem,
    SemanticWorkKind,
)
from tldw_Server_API.app.core.Notes_Graph import semantic_indexing, semantic_observability
from tldw_Server_API.app.core.Notes_Graph.semantic_content import build_semantic_chunks
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    PendingSemanticConfig,
    ResolvedDimension,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_indexing import (
    InitialGenerationRequest,
    NoteVersionRef,
    SemanticGenerationBuilder,
    VersionedNoteSnapshot,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_jobs import SemanticJobCancelled
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
NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


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


@pytest.mark.parametrize("value", [-1, math.inf, -math.inf, math.nan, True])
def test_total_metric_requires_finite_non_negative_value(value) -> None:
    with pytest.raises(SemanticObservationError, match="notes_semantic_observation_value_invalid"):
        build_semantic_metric_event(
            operation="cleanup",
            status="success",
            backend="chromadb",
            value=value,
        )


def test_total_metric_accepts_finite_non_negative_float() -> None:
    event = build_semantic_metric_event(
        operation="cleanup",
        status="success",
        backend="chromadb",
        value=1.5,
    )

    assert event.value == 1.5


def test_operational_metrics_use_only_closed_low_cardinality_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, dict[str, str]]] = []
    monkeypatch.setattr(
        semantic_observability,
        "_ensure_metrics_registered",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        semantic_observability,
        "increment_counter",
        lambda name, value=1, labels=None: calls.append((name, float(value), dict(labels or {}))),
        raising=False,
    )
    monkeypatch.setattr(
        semantic_observability,
        "observe_histogram",
        lambda name, value, labels=None: calls.append((name, float(value), dict(labels or {}))),
        raising=False,
    )
    monkeypatch.setattr(
        semantic_observability,
        "set_gauge",
        lambda name, value, labels=None: calls.append((name, float(value), dict(labels or {}))),
        raising=False,
    )

    semantic_observability.record_semantic_build_metrics(
        operation="rebuild",
        status="degraded",
        backend="chromadb",
        duration_seconds=1.25,
        counts={"indexed": 4, "excluded": 1, "failed": 1, "dirty": 2, "pending": 3},
    )
    semantic_observability.record_semantic_aggregate_metrics(
        snapshots=(
            SimpleNamespace(
                backend="chromadb",
                indexed_notes=4,
                excluded_notes=1,
                failed_notes=1,
                dirty_notes=2,
                pending_notes=3,
                stale_generations=1,
                cleanup_backlog=0,
                cleanup_retries=0,
                oldest_cleanup_created_at=None,
            ),
        ),
        now=NOW,
    )
    semantic_observability.record_semantic_query_metrics(
        status="success",
        backend="chromadb",
        duration_seconds=0.05,
        candidate_count=8,
        filtered_count=5,
        admitted_count=2,
        truncations=("semantic_candidates",),
    )
    semantic_observability.record_semantic_cleanup_metrics(
        status="failed",
        backend="pgvector",
        backlog=3,
        retries=1,
        oldest_age_seconds=30.0,
    )
    semantic_observability.record_semantic_denial("kill_switch")
    semantic_observability.record_semantic_cancellation("rebuild")
    semantic_observability.record_semantic_failure(
        component="provider",
        category="unavailable",
        backend="chromadb",
    )
    semantic_observability.record_semantic_dsr_metrics(
        status="success",
        backend="chromadb",
    )

    names = {name for name, _value, _labels in calls}
    assert {
        "notes_semantic_build_duration_seconds",
        "notes_semantic_note_count",
        "notes_semantic_coverage_ratio",
        "notes_semantic_stale_generations",
        "notes_semantic_query_duration_seconds",
        "notes_semantic_query_stage_total",
        "notes_semantic_truncations_total",
        "notes_semantic_cleanup_backlog",
        "notes_semantic_cleanup_retries_total",
        "notes_semantic_cleanup_oldest_age_seconds",
        "notes_semantic_denials_total",
        "notes_semantic_cancellations_total",
        "notes_semantic_failures_total",
        "notes_semantic_dsr_total",
    } <= names
    serialized = repr(calls)
    for forbidden in (
        "owner-a",
        "dataset-a",
        "generation-a",
        "run-a",
        "https://",
        "api_key",
    ):
        assert forbidden not in serialized


def test_build_completion_does_not_overwrite_global_health_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(semantic_observability, "_ensure_metrics_registered", lambda: None)
    monkeypatch.setattr(
        semantic_observability,
        "increment_counter",
        lambda name, value=1, labels=None: calls.append(name),
    )
    monkeypatch.setattr(
        semantic_observability,
        "observe_histogram",
        lambda name, value, labels=None: calls.append(name),
    )
    monkeypatch.setattr(
        semantic_observability,
        "set_gauge",
        lambda name, value, labels=None: calls.append(name),
    )

    semantic_observability.record_semantic_build_metrics(
        operation="build",
        status="success",
        backend="chromadb",
        duration_seconds=0.5,
        counts={"indexed": 20, "excluded": 0, "failed": 0, "dirty": 0, "pending": 0},
    )

    assert calls == [
        "notes_semantic_build_duration_seconds",
        "notes_semantic_builds_total",
    ]


def test_authoritative_health_aggregates_multi_dataset_state_cleanup_and_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, dict[str, str]]] = []
    monkeypatch.setattr(semantic_observability, "_ensure_metrics_registered", lambda: None)
    monkeypatch.setattr(
        semantic_observability,
        "increment_counter",
        lambda name, value=1, labels=None: calls.append((name, float(value), dict(labels or {}))),
    )
    monkeypatch.setattr(
        semantic_observability,
        "set_gauge",
        lambda name, value, labels=None: calls.append((name, float(value), dict(labels or {}))),
    )
    snapshots = (
        SimpleNamespace(
            backend="chromadb",
            indexed_notes=9,
            excluded_notes=0,
            failed_notes=0,
            dirty_notes=0,
            pending_notes=0,
            stale_generations=0,
            cleanup_backlog=0,
            cleanup_retries=0,
            oldest_cleanup_created_at=None,
        ),
        SimpleNamespace(
            backend="chromadb",
            indexed_notes=1,
            excluded_notes=1,
            failed_notes=1,
            dirty_notes=2,
            pending_notes=1,
            stale_generations=1,
            cleanup_backlog=2,
            cleanup_retries=3,
            oldest_cleanup_created_at=NOW - timedelta(seconds=120),
        ),
        SimpleNamespace(
            backend="pgvector",
            indexed_notes=2,
            excluded_notes=0,
            failed_notes=0,
            dirty_notes=0,
            pending_notes=0,
            stale_generations=0,
            cleanup_backlog=1,
            cleanup_retries=1,
            oldest_cleanup_created_at=(NOW - timedelta(seconds=60)).isoformat(),
        ),
    )

    semantic_observability.record_semantic_aggregate_metrics(
        snapshots=snapshots,
        now=NOW,
    )

    def gauge(name: str, labels: dict[str, str]) -> float:
        matches = [value for metric, value, actual in calls if metric == name and actual == labels]
        assert len(matches) == 1
        return matches[0]

    assert (
        gauge(
            "notes_semantic_note_count",
            {"state": "indexed", "backend": "chromadb"},
        )
        == 10
    )
    assert (
        gauge(
            "notes_semantic_note_count",
            {"state": "failed", "backend": "chromadb"},
        )
        == 1
    )
    assert gauge("notes_semantic_coverage_ratio", {"backend": "chromadb"}) == pytest.approx(10 / 13)
    assert gauge("notes_semantic_stale_generations", {"backend": "chromadb"}) == 1
    assert gauge("notes_semantic_cleanup_backlog", {"backend": "chromadb"}) == 2
    assert (
        gauge(
            "notes_semantic_cleanup_retries_total",
            {"status": "failed", "backend": "chromadb"},
        )
        == 3
    )
    assert gauge("notes_semantic_cleanup_oldest_age_seconds", {"backend": "chromadb"}) == 120
    assert gauge("notes_semantic_cleanup_backlog", {"backend": "pgvector"}) == 1
    assert gauge("notes_semantic_cleanup_oldest_age_seconds", {"backend": "pgvector"}) == 60
    assert gauge("notes_semantic_cleanup_backlog", {"backend": "unavailable"}) == 0


def test_metric_backend_failure_does_not_change_semantic_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("metrics unavailable")

    monkeypatch.setattr(semantic_observability, "_ensure_metrics_registered", lambda: None)
    monkeypatch.setattr(semantic_observability, "increment_counter", unavailable)
    monkeypatch.setattr(semantic_observability, "observe_histogram", unavailable)
    monkeypatch.setattr(semantic_observability, "set_gauge", unavailable)

    semantic_observability.record_semantic_build_metrics(
        operation="build",
        status="success",
        backend="chromadb",
        duration_seconds=0.5,
        counts={"indexed": 1, "excluded": 0, "failed": 0, "dirty": 0, "pending": 0},
    )


@pytest.mark.asyncio
async def test_semantic_audit_uses_unified_durable_flush() -> None:
    calls: list[dict[str, object]] = []
    flushes: list[bool] = []

    class AuditService:
        async def log_event(self, **kwargs: object) -> str:
            calls.append(dict(kwargs))
            return "event-a"

        async def flush(self, *, raise_on_failure: bool = False) -> bool:
            flushes.append(raise_on_failure)
            return True

    await semantic_observability.emit_semantic_audit_event(
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        event="generation_publication",
        status="degraded",
        reason="note_failed",
        generation_id="generation-a",
        run_id="run-a",
        counts={"indexed": 4, "failed": 1},
        audit_service=AuditService(),
    )

    assert len(calls) == 1
    assert calls[0]["action"] == "notes_semantic.generation_publication"
    assert calls[0]["resource_id"] == "dataset-a"
    assert calls[0]["metadata"] == {
        "semantic_event": "generation_publication",
        "status": "degraded",
        "reason": "note_failed",
        "indexed": 4,
        "failed": 1,
        "generation_id": "generation-a",
        "run_id": "run-a",
    }
    assert flushes == [True]


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

        def authorize_note_vector_upsert(self, **_kwargs):
            return True

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
            NoteVersionRef(snapshot.note_id, snapshot.content_version) for snapshot in self.snapshots.values()
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
async def test_snapshot_planning_does_not_charge_provider_budget_before_work_claim() -> None:
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

    plan = await builder._read_snapshot(_fence())

    assert len(plan.seeds) == 2
    assert sum(seed.planned_chunk_count for seed in plan.seeds) == 4


@pytest.mark.asyncio
async def test_dimension_provider_is_fenced_by_active_cancellation() -> None:
    provider_called = False
    fence = replace(
        _fence(),
        model_revision=None,
        compatibility_hash=None,
        dimensions=None,
    )
    authority = replace(
        _authority(),
        model_revision=None,
        compatibility_hash=None,
        dimensions=None,
    )

    class Embedder:
        async def resolve_dimensions(self, _config, *, user_id):
            nonlocal provider_called
            provider_called = True
            raise AssertionError("dimension provider must not run")

    builder = SemanticGenerationBuilder(
        store=SimpleNamespace(),
        note_reader=SimpleNamespace(),
        embedder=Embedder(),
        vectors=SimpleNamespace(),
        revalidate=lambda _fence_value: authority,
        compatibility_hash_for_dimension=lambda _resolved: "compatibility-a",
        settings=SemanticIndexSettings(),
        clock=lambda: None,
        receipt_factory=lambda: "unused",
    )

    async def cancel() -> None:
        raise SemanticJobCancelled()

    with pytest.raises(SemanticJobCancelled):
        await builder._resolve_generation(
            fence,
            PendingSemanticConfig(
                provider=fence.provider,
                model=fence.model,
                model_revision=fence.model_revision,
                endpoint_origin=fence.endpoint_origin,
                credential_source=fence.credential_source,
                consented=True,
                dimensions=None,
            ),
            before_side_effect=cancel,
        )

    assert provider_called is False


@pytest.mark.asyncio
async def test_dimension_resolution_rejects_preadmitted_model_revision_drift() -> None:
    store_reads = 0
    fence = replace(_fence(), compatibility_hash=None, dimensions=None)
    authority = replace(_authority(), compatibility_hash=None, dimensions=None)

    class Store:
        def get_configuration(self, _dataset_id):
            nonlocal store_reads
            store_reads += 1
            return None

    class Embedder:
        async def resolve_dimensions(self, _config, *, user_id):
            return ResolvedDimension(
                dimensions=2,
                provider=fence.provider,
                model=fence.model,
                model_revision="revision-b",
                endpoint_origin=fence.endpoint_origin,
                credential_source=fence.credential_source,
            )

    builder = SemanticGenerationBuilder(
        store=Store(),
        note_reader=SimpleNamespace(),
        embedder=Embedder(),
        vectors=SimpleNamespace(),
        revalidate=lambda _fence_value: authority,
        compatibility_hash_for_dimension=lambda _resolved: "compatibility-b",
        settings=SemanticIndexSettings(),
        clock=lambda: None,
        receipt_factory=lambda: "unused",
    )

    with pytest.raises(SemanticIndexingError) as drift:
        await builder._resolve_generation(
            fence,
            PendingSemanticConfig(
                provider=fence.provider,
                model=fence.model,
                model_revision=fence.model_revision,
                endpoint_origin=fence.endpoint_origin,
                credential_source=fence.credential_source,
                consented=True,
                dimensions=None,
            ),
        )

    assert drift.value.code == "notes_semantic_model_revision_drift"
    assert store_reads == 0


@pytest.mark.asyncio
async def test_build_preserves_active_cancellation_across_task6_boundary() -> None:
    provider_called = False
    fence = replace(
        _fence(),
        model_revision=None,
        compatibility_hash=None,
        dimensions=None,
    )
    authority = replace(
        _authority(),
        model_revision=None,
        compatibility_hash=None,
        dimensions=None,
    )

    class Store:
        def fail_generation(self, **_kwargs):
            return True

    class Embedder:
        async def resolve_dimensions(self, _config, *, user_id):
            nonlocal provider_called
            provider_called = True
            raise AssertionError("dimension provider must not run")

    builder = SemanticGenerationBuilder(
        store=Store(),
        note_reader=SimpleNamespace(),
        embedder=Embedder(),
        vectors=SimpleNamespace(),
        revalidate=lambda _fence_value: authority,
        compatibility_hash_for_dimension=lambda _resolved: "compatibility-a",
        settings=SemanticIndexSettings(),
        clock=lambda: None,
        receipt_factory=lambda: "unused",
    )

    async def cancel() -> None:
        raise SemanticJobCancelled()

    with pytest.raises(SemanticJobCancelled):
        await builder.build_initial_generation(
            InitialGenerationRequest(
                fence=fence,
                embedding_config=PendingSemanticConfig(
                    provider=fence.provider,
                    model=fence.model,
                    model_revision=fence.model_revision,
                    endpoint_origin=fence.endpoint_origin,
                    credential_source=fence.credential_source,
                    consented=True,
                    dimensions=None,
                ),
            ),
            before_side_effect=cancel,
        )

    assert provider_called is False


@pytest.mark.asyncio
async def test_note_publication_is_fenced_before_vector_or_database_side_effects() -> None:
    store_calls = 0
    vector_calls = 0

    class Store:
        def stage_obsolete_vector_cleanup(self, **_kwargs):
            nonlocal store_calls
            store_calls += 1
            return 1

    class Vectors:
        async def upsert(self, _dataset_id, _generation_id, _vectors):
            nonlocal vector_calls
            vector_calls += 1
            return 1

    service = SemanticPublicationService(
        store=Store(),
        vectors=Vectors(),
        revalidate=lambda _fence_value: _authority(),
        clock=lambda: None,
        receipt_factory=lambda: "unused",
    )
    chunks = build_semantic_chunks(
        generation_id="generation-a",
        note_id="note-a",
        title="Title",
        content="Body",
        content_version=1,
    )

    async def cancel() -> None:
        raise SemanticJobCancelled()

    with pytest.raises(SemanticJobCancelled):
        await service.publish_note(
            _fence(),
            _claimed_work(SemanticWorkKind.INDEX_NOTE),
            chunks,
            tuple(SemanticVector(chunk.vector_id, (1.0, 2.0)) for chunk in chunks),
            before_side_effect=cancel,
        )

    assert store_calls == 0
    assert vector_calls == 0


@pytest.mark.asyncio
async def test_activation_is_fenced_before_vector_read_or_publication_commit() -> None:
    vector_calls = 0
    activation_calls = 0

    class Store:
        def get_generation_integrity(self, _dataset_id, _generation_id):
            return replace(_empty_integrity(), vector_ids=("vector-a",))

        def assert_generation_activatable(self, _integrity):
            return None

        def activate_generation_verified(self, **_kwargs):
            nonlocal activation_calls
            activation_calls += 1
            return object()

    class Vectors:
        async def fetch(self, _dataset_id, _generation_id, _vector_ids):
            nonlocal vector_calls
            vector_calls += 1
            return ()

    service = SemanticPublicationService(
        store=Store(),
        vectors=Vectors(),
        revalidate=lambda _fence_value: _authority(),
        clock=lambda: None,
        receipt_factory=lambda: "receipt-a",
    )

    async def cancel() -> None:
        raise SemanticJobCancelled()

    with pytest.raises(SemanticJobCancelled):
        await service.activate(_fence(), before_side_effect=cancel)

    assert vector_calls == 0
    assert activation_calls == 0
