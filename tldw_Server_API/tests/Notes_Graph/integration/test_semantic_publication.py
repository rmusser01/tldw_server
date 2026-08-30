"""SQLite and PostgreSQL contracts for cross-store semantic publication."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticChunkRecord,
    SemanticDimensionState,
    SemanticGenerationState,
    SemanticSnapshotSeed,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph import semantic_indexing
from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    build_semantic_chunks,
    semantic_content_fingerprint,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    PendingSemanticConfig,
    ResolvedDimension,
    ResolvedSemanticConfig,
    SemanticEmbeddingBatch,
    SemanticEmbeddingSystemError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_indexing import (
    InitialGenerationRequest,
    NoteVersionRef,
    SemanticGenerationBuilder,
    SemanticNoteIndexingError,
    VersionedNoteSnapshot,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticAuthorityState,
    SemanticExecutionFence,
    SemanticIndexingError,
    SemanticPublicationService,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
    SemanticVectorCleanup,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
CLAIM_NOW = datetime(2026, 8, 30, 23, 0, tzinfo=timezone.utc)
OWNER_ID = "owner-a"
DATASET_ID = "dataset-a"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
GENERATION_FENCE = "-".join(("job", "fence", "v1"))
WRONG_GENERATION_FENCE = "-".join(("wrong", "fence"))


class MemoryVectors:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.values: dict[tuple[str, str], SemanticVector] = {}
        self.fetch_sizes: list[int] = []
        self.delete_id_sizes: list[int] = []

    async def create_generation_storage(self, dataset_id: str, generation_id: str) -> None:
        self.events.append("storage")

    async def upsert(
        self,
        dataset_id: str,
        generation_id: str,
        vectors: tuple[SemanticVector, ...],
    ) -> int:
        self.events.append("vector_upsert")
        for vector in vectors:
            self.values[(generation_id, vector.vector_id)] = vector
        return len(vectors)

    async def fetch(self, dataset_id: str, generation_id: str, vector_ids):
        self.events.append("vector_fetch")
        self.fetch_sizes.append(len(vector_ids))
        return tuple(
            self.values[(generation_id, vector_id)]
            for vector_id in vector_ids
            if (generation_id, vector_id) in self.values
        )

    async def delete_ids(self, dataset_id: str, generation_id: str, vector_ids):
        self.events.append("vector_cleanup")
        self.delete_id_sizes.append(len(vector_ids))
        for vector_id in vector_ids:
            self.values.pop((generation_id, vector_id), None)
        return SemanticVectorCleanup(confirmed_absent=True)

    async def delete_generation(self, dataset_id: str, generation_id: str):
        self.events.append("generation_cleanup")
        for key in tuple(self.values):
            if key[0] == generation_id:
                del self.values[key]
        return SemanticVectorCleanup(confirmed_absent=True)


class MemoryNotes:
    def __init__(self, snapshots: tuple[VersionedNoteSnapshot, ...], events: list[str]) -> None:
        self.snapshots = {snapshot.note_id: snapshot for snapshot in snapshots}
        self.events = events

    async def list_note_versions(self, owner_user_id: str, dataset_id: str, *, limit: int):
        self.events.append("list_versions")
        values = sorted(self.snapshots.values(), key=lambda item: item.note_id)
        return tuple(NoteVersionRef(item.note_id, item.content_version) for item in values[:limit])

    async def read_note_version(
        self,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        content_version: int,
    ):
        self.events.append("content_read")
        snapshot = self.snapshots.get(note_id)
        if snapshot is None or snapshot.content_version != content_version:
            return None
        return snapshot


class MemoryEmbedder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def resolve_dimensions(self, config: PendingSemanticConfig, *, user_id: str):
        self.events.append("dimension_probe")
        return ResolvedDimension(
            2,
            config.provider,
            config.model,
            config.model_revision,
            config.endpoint_origin,
            config.credential_source,
        )

    async def embed_chunks(self, chunks, config, *, user_id: str):
        self.events.append("embed")
        return SemanticEmbeddingBatch(
            vectors=tuple((1.0, float(index + 1)) for index, _ in enumerate(chunks)),
            provider=config.provider,
            model=config.model,
            model_revision=config.model_revision,
            dimensions=2,
            prompt_tokens=0,
            total_tokens=0,
            endpoint_origin=config.endpoint_origin,
            credential_source=config.credential_source,
            provider_request_count=1,
        )


def _create_configuration(db: CharactersRAGDB):
    return db.note_semantic_store.create_configuration(
        dataset_id=DATASET_ID,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="openai",
        model="embedding-model-v1",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="local semantic vectors",
        normalization_version="notes-semantic-normalization-v1",
        chunker_version="notes-semantic-chunker-v1",
        now=NOW,
    )


def _create_pending_generation(db: CharactersRAGDB):
    config = _create_configuration(db)
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    generation = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id=GENERATION_FENCE,
        now=NOW,
    )
    return enabled, generation


def _fence(config, generation) -> SemanticExecutionFence:
    return SemanticExecutionFence(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        generation_fencing_token=GENERATION_FENCE,
        configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="openai",
        model="embedding-model-v1",
        model_revision=None,
        endpoint_origin="https://api.example.test",
        credential_source="server_default",
        endpoint_origin_revision="origin-v1",
        compatibility_hash=config.compatibility_hash,
        dimensions=config.dimensions,
        vector_backend="chromadb",
    )


def _authority_from_store(db: CharactersRAGDB, fence: SemanticExecutionFence) -> SemanticAuthorityState:
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    generation = db.note_semantic_store.get_generation(DATASET_ID, fence.generation_id)
    assert config is not None and generation is not None
    return SemanticAuthorityState(
        user_exists=True,
        owner_authorized=True,
        semantic_manage_allowed=True,
        desired_enabled=config.desired_state.value == "enabled",
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        generation_fencing_token=generation.root_job_id or "",
        configuration_revision=config.configuration_revision,
        capability_revision=config.capability_revision or "",
        disclosure_hash=config.disclosure_hash or "",
        provider=config.provider or "",
        model=config.model or "",
        model_revision=fence.model_revision,
        endpoint_origin=fence.endpoint_origin,
        credential_source=fence.credential_source,
        endpoint_origin_revision=config.endpoint_origin_revision or "",
        endpoint_policy_allowed=True,
        compatibility_hash=config.compatibility_hash,
        dimensions=config.dimensions,
        vector_backend=config.vector_backend or "",
        vector_capable=True,
    )


def _prepare_ready_generation(db: CharactersRAGDB):
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="Body",
        content_version=1,
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    publication = db.note_semantic_store.publish_indexed_manifest(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        generation_fencing_token=GENERATION_FENCE,
        expected_configuration_revision=resolved.configuration_revision,
        work_id=claim.id,
        claim_token=claim.claim_token or "",
        work_fencing_token=claim.fencing_token,
        claimed_dirty_generation=claim.dirty_generation or 0,
        content_version=1,
        content_fingerprint=chunks[0].content_fingerprint,
        chunks=tuple(
            SemanticChunkRecord(
                chunk_id=chunk.vector_id,
                generation_id=chunk.generation_id,
                note_id=chunk.note_id,
                content_version=chunk.content_version,
                ordinal=chunk.ordinal,
                field=chunk.field,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                chunk_fingerprint=chunk.chunk_fingerprint,
                normalization_version=chunk.normalization_version,
                chunker_version=chunk.chunker_version,
            )
            for chunk in chunks
        ),
        now=NOW,
    )
    assert publication is not None
    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)
    return resolved, pending, integrity


@pytest.fixture()
def sqlite_db(tmp_path: Path):
    db = CharactersRAGDB(str(tmp_path / "semantic-publication.sqlite"), client_id=OWNER_ID)
    yield db
    db.close_all_connections()


@pytest.mark.asyncio
async def test_initial_build_probes_seeds_publishes_verifies_and_returns_receipt(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body text", note_id=NOTE_ID)
    enabled, generation = _create_pending_generation(db)
    initial_fence = _fence(enabled, generation)
    events: list[str] = []
    notes = MemoryNotes((VersionedNoteSnapshot(NOTE_ID, "Title", "Body text", 1),), events)
    vectors = MemoryVectors(events)
    embedder = MemoryEmbedder(events)
    revalidations: list[SemanticExecutionFence] = []

    async def revalidate(fence: SemanticExecutionFence) -> SemanticAuthorityState:
        events.append("revalidate")
        revalidations.append(fence)
        return _authority_from_store(db, fence)

    builder = SemanticGenerationBuilder(
        store=db.note_semantic_store,
        note_reader=notes,
        embedder=embedder,
        vectors=vectors,
        revalidate=revalidate,
        compatibility_hash_for_dimension=lambda _resolved: "compatibility-v1",
        settings=SemanticIndexSettings(max_provider_batch_inputs=1),
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-v1",
    )

    receipt = await builder.build_initial_generation(
        InitialGenerationRequest(
            fence=initial_fence,
            embedding_config=PendingSemanticConfig(
                provider="openai",
                model="embedding-model-v1",
                model_revision=None,
                endpoint_origin="https://api.example.test",
                credential_source="server_default",
                consented=True,
            ),
        )
    )

    active = db.note_semantic_store.get_generation(DATASET_ID, generation.id)
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert active is not None and active.state is SemanticGenerationState.ACTIVE
    assert active.expected_note_count == active.published_note_count == 1
    assert active.expected_chunk_count == active.published_chunk_count == 1
    assert active.publication_receipt == "receipt-v1"
    assert config is not None and config.active_generation_id == generation.id
    assert config.semantic_index_revision == 1
    assert receipt.receipt == "receipt-v1"
    assert receipt.indexed_notes == 1
    assert receipt.degraded is False
    assert events.index("dimension_probe") < events.index("content_read")
    assert events.index("storage") < events.index("vector_upsert")
    assert events.index("vector_upsert") < events.index("vector_fetch")
    assert len(revalidations) >= 4


@pytest.mark.parametrize(
    "embedding_override",
    [
        {"endpoint_origin": "https://wrong.example.test"},
        {"credential_source": "user"},
    ],
)
@pytest.mark.asyncio
async def test_initial_build_rejects_endpoint_or_credential_drift_before_note_transfer(
    sqlite_db: CharactersRAGDB,
    embedding_override: dict[str, str],
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, generation = _create_pending_generation(db)
    fence = _fence(enabled, generation)
    events: list[str] = []
    values = {
        "provider": "openai",
        "model": "embedding-model-v1",
        "model_revision": None,
        "endpoint_origin": "https://api.example.test",
        "credential_source": "server_default",
        "consented": True,
    }
    values.update(embedding_override)
    builder = SemanticGenerationBuilder(
        store=db.note_semantic_store,
        note_reader=MemoryNotes(
            (VersionedNoteSnapshot(NOTE_ID, "Title", "Body", 1),), events
        ),
        embedder=MemoryEmbedder(events),
        vectors=MemoryVectors(events),
        revalidate=lambda current: _authority_from_store(db, current),
        compatibility_hash_for_dimension=lambda _resolved: "compatibility-v1",
        clock=lambda: NOW,
        receipt_factory=lambda: "unreachable-receipt",
    )

    with pytest.raises(SemanticIndexingError, match="notes_semantic_execution_config_drift"):
        await builder.build_initial_generation(
            InitialGenerationRequest(
                fence=fence,
                embedding_config=PendingSemanticConfig(**values),
            )
        )
    assert "content_read" not in events


@pytest.mark.asyncio
async def test_initial_build_rejects_runtime_endpoint_drift_between_note_batches(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    second_note_id = "22222222-2222-4222-8222-222222222222"
    db.note_store.add_note("First", "Body", note_id=NOTE_ID)
    db.note_store.add_note("Second", "Body", note_id=second_note_id)
    enabled, generation = _create_pending_generation(db)
    events: list[str] = []

    class DriftingEmbedder(MemoryEmbedder):
        calls = 0

        async def embed_chunks(self, chunks, config, *, user_id: str):
            result = await super().embed_chunks(chunks, config, user_id=user_id)
            self.calls += 1
            if self.calls == 2:
                return replace(result, endpoint_origin="https://wrong.example.test")
            return result

    builder = SemanticGenerationBuilder(
        store=db.note_semantic_store,
        note_reader=MemoryNotes(
            (
                VersionedNoteSnapshot(NOTE_ID, "First", "Body", 1),
                VersionedNoteSnapshot(second_note_id, "Second", "Body", 1),
            ),
            events,
        ),
        embedder=DriftingEmbedder(events),
        vectors=MemoryVectors(events),
        revalidate=lambda current: _authority_from_store(db, current),
        compatibility_hash_for_dimension=lambda _resolved: "compatibility-v1",
        clock=lambda: NOW,
        receipt_factory=lambda: "unreachable-receipt",
    )

    with pytest.raises(
        SemanticIndexingError, match="notes_semantic_embedding_identity_mismatch"
    ):
        await builder.build_initial_generation(
            InitialGenerationRequest(
                fence=_fence(enabled, generation),
                embedding_config=PendingSemanticConfig(
                    provider="openai",
                    model="embedding-model-v1",
                    model_revision=None,
                    endpoint_origin="https://api.example.test",
                    credential_source="server_default",
                    consented=True,
                ),
            )
        )
    assert events.count("embed") == 2
    released = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        kind="index_note",
        limit=16,
        now=NOW,
    )
    assert len(released) == 1


def _seed_claim_batch(db: CharactersRAGDB):
    note_ids = (
        NOTE_ID,
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
    )
    for index, note_id in enumerate(note_ids):
        db.note_store.add_note(f"Title {index}", "Body", note_id=note_id)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks_by_note = {
        note_id: build_semantic_chunks(
            generation_id=pending.id,
            note_id=note_id,
            title=f"Title {index}",
            content="Body",
            content_version=1,
        )
        for index, note_id in enumerate(note_ids)
    }
    seeds = tuple(
        SemanticSnapshotSeed(
            note_id=note_id,
            content_version=1,
            content_fingerprint=semantic_content_fingerprint(
                f"Title {index}", "Body", 1
            ),
            state="pending",
            planned_chunk_count=len(chunks_by_note[note_id]),
            error_code=None,
        )
        for index, note_id in enumerate(note_ids)
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=seeds,
        now=NOW,
    )
    plan = semantic_indexing._SnapshotPlan(
        tuple(NoteVersionRef(note_id, 1) for note_id in note_ids),
        seeds,
        chunks_by_note,
    )
    config = ResolvedSemanticConfig(
        provider="openai",
        model="embedding-model-v1",
        model_revision=None,
        endpoint_origin="https://api.example.test",
        credential_source="server_default",
        dimensions=2,
    )
    return pending, resolved, plan, config


@pytest.mark.parametrize("failure_position", [0, 1, 2])
@pytest.mark.asyncio
async def test_systemic_error_releases_unprocessed_multi_claim_batch(
    sqlite_db: CharactersRAGDB,
    failure_position: int,
) -> None:
    db = sqlite_db
    pending, resolved, plan, config = _seed_claim_batch(db)
    events: list[str] = []

    class FailingEmbedder(MemoryEmbedder):
        calls = 0

        async def embed_chunks(self, chunks, config, *, user_id: str):
            position = self.calls
            self.calls += 1
            if position == failure_position:
                raise SemanticEmbeddingSystemError("provider_unavailable")
            return await super().embed_chunks(chunks, config, user_id=user_id)

    builder = SemanticGenerationBuilder(
        store=db.note_semantic_store,
        note_reader=MemoryNotes((), events),
        embedder=FailingEmbedder(events),
        vectors=MemoryVectors(events),
        revalidate=lambda fence: _authority_from_store(db, fence),
        compatibility_hash_for_dimension=lambda resolved_value: "compatibility-v1",
        settings=SemanticIndexSettings(
            max_active_notes=3,
            max_provider_batch_inputs=3,
        ),
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="provider_unavailable"):
        await builder._publish_claimed_notes(
            _fence(resolved, pending),
            config,
            plan,
        )

    released = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=3,
        now=NOW,
    )
    assert len(released) == 3 - failure_position


@pytest.mark.asyncio
async def test_cancellation_releases_entire_unprocessed_claim_batch(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    pending, resolved, plan, config = _seed_claim_batch(db)
    started = asyncio.Event()

    class BlockingEmbedder(MemoryEmbedder):
        async def embed_chunks(self, chunks, config, *, user_id: str):
            started.set()
            await asyncio.Event().wait()

    builder = SemanticGenerationBuilder(
        store=db.note_semantic_store,
        note_reader=MemoryNotes((), []),
        embedder=BlockingEmbedder([]),
        vectors=MemoryVectors([]),
        revalidate=lambda fence: _authority_from_store(db, fence),
        compatibility_hash_for_dimension=lambda resolved_value: "compatibility-v1",
        settings=SemanticIndexSettings(
            max_active_notes=3,
            max_provider_batch_inputs=3,
        ),
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )
    operation = asyncio.create_task(
        builder._publish_claimed_notes(_fence(resolved, pending), config, plan)
    )
    await started.wait()
    operation.cancel("cancel-claim-batch")

    with pytest.raises(asyncio.CancelledError, match="cancel-claim-batch"):
        await operation
    released = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=3,
        now=NOW,
    )
    assert len(released) == 3


@pytest.mark.asyncio
async def test_documented_note_error_degrades_without_downgrading_provider_failures(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    pending, resolved, plan, config = _seed_claim_batch(db)

    class NoteFailingEmbedder(MemoryEmbedder):
        calls = 0

        async def embed_chunks(self, chunks, config, *, user_id: str):
            self.calls += 1
            if self.calls == 1:
                raise SemanticNoteIndexingError("note_content_unavailable")
            return await super().embed_chunks(chunks, config, user_id=user_id)

    builder = SemanticGenerationBuilder(
        store=db.note_semantic_store,
        note_reader=MemoryNotes((), []),
        embedder=NoteFailingEmbedder([]),
        vectors=MemoryVectors([]),
        revalidate=lambda fence: _authority_from_store(db, fence),
        compatibility_hash_for_dimension=lambda resolved_value: "compatibility-v1",
        settings=SemanticIndexSettings(
            max_active_notes=3,
            max_provider_batch_inputs=3,
        ),
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )

    await builder._publish_claimed_notes(_fence(resolved, pending), config, plan)

    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)
    assert integrity.failed_note_count == 1
    assert integrity.indexed_note_count == 2
    assert integrity.degraded is True


def test_claim_lease_recovery_is_bounded_and_attempt_capped(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=1,
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    recovery_time = NOW + timedelta(hours=1)
    assert db.note_semantic_store.reclaim_expired_work_claims(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        expired_before=NOW + timedelta(minutes=30),
        limit=1,
        now=recovery_time,
    ) == 1
    retried = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=recovery_time,
    )[0]
    assert retried.id == claim.id
    assert retried.attempt_count == 1
    active_claim = retried
    for expected_attempt in range(2, 6):
        recovery_time += timedelta(hours=1)
        assert db.note_semantic_store.reclaim_expired_work_claims(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            kind="index_note",
            expired_before=recovery_time,
            limit=1,
            now=recovery_time,
        ) == 1
        if expected_attempt < 5:
            active_claim = db.note_semantic_store.claim_work_batch(
                dataset_id=DATASET_ID,
                generation_id=pending.id,
                kind="index_note",
                limit=1,
                now=recovery_time,
            )[0]
            assert active_claim.attempt_count == expected_attempt
    assert db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=recovery_time,
    ) == ()


def test_exact_claim_can_publish_per_note_failure_without_downgrading_systemic_failure(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=1,
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    assert db.note_semantic_store.fail_claimed_note(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        generation_fencing_token=GENERATION_FENCE,
        expected_configuration_revision=resolved.configuration_revision,
        work_id=claim.id,
        claim_token=claim.claim_token,
        work_fencing_token=claim.fencing_token,
        claimed_dirty_generation=claim.dirty_generation,
        note_id=NOTE_ID,
        error_code="note_content_unavailable",
        now=NOW,
    )
    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)
    assert integrity.failed_note_count == 1
    assert integrity.waived_chunk_count == 1
    assert integrity.degraded is True


@pytest.mark.parametrize(
    ("states", "eligible", "allowed", "degraded"),
    [
        ((), 0, True, False),
        (("excluded",), 0, True, True),
        (("failed",), 1, False, False),
        (("indexed", "failed"), 2, True, True),
        (("pending",), 1, False, False),
    ],
)
@pytest.mark.asyncio
async def test_activation_policy_requires_terminal_snapshot_and_indexed_eligible_coverage(
    sqlite_db: CharactersRAGDB,
    states: tuple[str, ...],
    eligible: int,
    allowed: bool,
    degraded: bool,
) -> None:
    db = sqlite_db
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    seeds = []
    chunks_by_note = {}
    for index, state in enumerate(states):
        note_id = f"00000000-0000-4000-8000-{index:012d}"
        db.note_store.add_note(f"Title {index}", "Body", note_id=note_id)
        chunks = build_semantic_chunks(
            generation_id=pending.id,
            note_id=note_id,
            title=f"Title {index}",
            content="Body",
            content_version=1,
        )
        seed_state = "pending" if state == "indexed" else state
        if state == "indexed":
            chunks_by_note[note_id] = chunks
        seeds.append(
            SemanticSnapshotSeed(
                note_id=note_id,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint(f"Title {index}", "Body", 1),
                state=seed_state,
                planned_chunk_count=len(chunks) if seed_state == "pending" else 0,
                error_code="note_failed" if state == "failed" else ("note_excluded" if state == "excluded" else None),
            )
        )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=tuple(seeds),
        now=NOW,
    )
    events: list[str] = []
    vectors = MemoryVectors(events)

    async def revalidate(fence: SemanticExecutionFence) -> SemanticAuthorityState:
        return _authority_from_store(db, fence)

    service = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-policy",
    )
    claims = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=16,
        now=NOW,
    )
    for claim in claims:
        if claim.note_id not in chunks_by_note:
            continue
        chunks = chunks_by_note[claim.note_id]
        await service.publish_note(
            _fence(resolved, pending),
            claim,
            chunks,
            tuple(
                SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
                for index, chunk in enumerate(chunks)
            ),
        )
    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)
    assert integrity.eligible_note_count == eligible
    if allowed:
        receipt = await service.activate(_fence(resolved, pending))
        assert receipt.receipt == "receipt-policy"
        assert integrity.degraded is degraded
    else:
        with pytest.raises(SemanticIndexingError):
            db.note_semantic_store.assert_generation_activatable(integrity)


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"generation_fencing_token": WRONG_GENERATION_FENCE}, "notes_semantic_activation_fence_lost"),
        ({"expected_manifest_hash": "sha256:" + "0" * 64}, "notes_semantic_manifest_hash_mismatch"),
        ({"expected_vector_ids": ("wrong-vector",)}, "notes_semantic_vector_ids_mismatch"),
        ({"expected_dimensions": 3}, "notes_semantic_generation_identity_mismatch"),
        ({"expected_compatibility_hash": "wrong-hash"}, "notes_semantic_generation_identity_mismatch"),
    ],
)
def test_activation_revalidates_exact_manifest_vector_dimension_hash_and_fence(
    sqlite_db: CharactersRAGDB,
    override: dict[str, object],
    code: str,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    seed = SemanticSnapshotSeed(
        note_id=NOTE_ID,
        content_version=1,
        content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
        state="excluded",
        planned_chunk_count=0,
        error_code="note_excluded",
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(seed,),
        now=NOW,
    )
    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)
    arguments: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "generation_id": pending.id,
        "expected_configuration_revision": resolved.configuration_revision,
        "generation_fencing_token": GENERATION_FENCE,
        "expected_manifest_hash": integrity.manifest_hash,
        "expected_vector_ids": integrity.vector_ids,
        "expected_dimensions": 2,
        "expected_compatibility_hash": "compatibility-v1",
        "publication_receipt": "receipt-corruption",
        "now": NOW,
    }
    arguments.update(override)

    if code == "notes_semantic_activation_fence_lost":
        assert db.note_semantic_store.activate_generation_verified(**arguments) is None
    else:
        with pytest.raises(SemanticIndexingError, match=code):
            db.note_semantic_store.activate_generation_verified(**arguments)


@pytest.mark.parametrize(
    ("corruption", "code"),
    [
        ("missing_id", "notes_semantic_vector_integrity_mismatch"),
        ("wrong_dimension", "notes_semantic_vector_dimension_mismatch"),
    ],
)
@pytest.mark.asyncio
async def test_activation_fetches_exact_physical_vector_ids_and_dimensions(
    sqlite_db: CharactersRAGDB,
    corruption: str,
    code: str,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="Body",
        content_version=1,
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=CLAIM_NOW,
    )[0]
    vectors = MemoryVectors([])

    async def revalidate(fence: SemanticExecutionFence) -> SemanticAuthorityState:
        return _authority_from_store(db, fence)

    service = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-vector-integrity",
    )
    fence = _fence(resolved, pending)
    publication = await service.publish_note(
        fence,
        claim,
        chunks,
        tuple(
            SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
            for index, chunk in enumerate(chunks)
        ),
    )
    vector_id = publication.new_vector_ids[0]
    if corruption == "missing_id":
        del vectors.values[(pending.id, vector_id)]
    else:
        vectors.values[(pending.id, vector_id)] = SemanticVector(
            vector_id,
            (1.0, 1.0, 1.0),
        )

    with pytest.raises(SemanticIndexingError, match=code):
        await service.activate(fence)


@pytest.mark.asyncio
async def test_activation_revision_supports_active_incremental_manifest_and_tombstone(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="Body",
        content_version=1,
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    vectors = MemoryVectors([])

    async def revalidate(fence: SemanticExecutionFence) -> SemanticAuthorityState:
        return _authority_from_store(db, fence)

    service = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-incremental",
    )
    build_fence = _fence(resolved, pending)
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    await service.publish_note(
        build_fence,
        claim,
        chunks,
        tuple(SemanticVector(chunk.vector_id, (1.0, 2.0)) for chunk in chunks),
    )
    receipt = await service.activate(build_fence)
    active = db.note_semantic_store.get_generation(DATASET_ID, pending.id)
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert active is not None and config is not None
    assert receipt.configuration_revision == config.configuration_revision
    assert active.configuration_revision == config.configuration_revision

    active_fence = replace(build_fence, configuration_revision=config.configuration_revision)
    assert db.note_store.update_note(
        NOTE_ID,
        {"content": "Body revised"},
        expected_version=1,
        semantic_dataset_id=DATASET_ID,
    )
    revised_chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="Body revised",
        content_version=2,
    )
    edit_claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=CLAIM_NOW,
    )[0]
    revised = await service.publish_note(
        active_fence,
        edit_claim,
        revised_chunks,
        tuple(SemanticVector(chunk.vector_id, (2.0, 3.0)) for chunk in revised_chunks),
    )
    assert db.note_semantic_store.list_visible_vector_ids(
        DATASET_ID, pending.id, NOTE_ID
    ) == revised.new_vector_ids

    assert db.note_store.soft_delete_note(
        NOTE_ID,
        expected_version=2,
        semantic_dataset_id=DATASET_ID,
    )
    delete_claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="delete_note_vectors",
        limit=1,
        now=CLAIM_NOW,
    )[0]
    tombstone = await service.publish_tombstone(active_fence, delete_claim)
    assert tombstone.old_vector_ids == revised.new_vector_ids
    assert db.note_semantic_store.list_visible_vector_ids(
        DATASET_ID, pending.id, NOTE_ID
    ) == ()


@pytest.mark.asyncio
async def test_activation_uses_seeded_planned_chunk_count_not_published_rows(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "abcdefghijklmnop", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="abcdefghijklmnop",
        content_version=1,
        settings=SemanticIndexSettings(max_chunk_code_points=4),
    )
    assert len(chunks) > 1
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint(
                    "Title", "abcdefghijklmnop", 1
                ),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    vectors = MemoryVectors([])
    service = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=lambda fence: _authority_from_store(db, fence),
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-missing-planned-vector",
    )
    await service.publish_note(
        _fence(resolved, pending),
        claim,
        chunks[:-1],
        tuple(SemanticVector(chunk.vector_id, (1.0, 2.0)) for chunk in chunks[:-1]),
    )
    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)
    assert integrity.expected_chunk_count == len(chunks)
    assert integrity.published_chunk_count == len(chunks) - 1
    with pytest.raises(SemanticIndexingError, match="notes_semantic_chunk_count_mismatch"):
        db.note_semantic_store.assert_generation_activatable(integrity)


@pytest.mark.asyncio
async def test_manifest_transaction_rechecks_configuration_after_authority_validation(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="Body",
        content_version=1,
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    calls = 0

    async def revalidate(fence: SemanticExecutionFence) -> SemanticAuthorityState:
        nonlocal calls
        calls += 1
        authority = _authority_from_store(db, fence)
        if calls == 2:
            disabled = db.note_semantic_store.disable_configuration(
                dataset_id=DATASET_ID,
                expected_configuration_revision=fence.configuration_revision,
                now=NOW,
            )
            assert disabled is not None
        return authority

    service = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=MemoryVectors([]),
        revalidate=revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-race",
    )

    with pytest.raises(SemanticIndexingError, match="notes_semantic_note_claim_stale"):
        await service.publish_note(
            _fence(resolved, pending),
            claim,
            chunks,
            tuple(
                SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
                for index, chunk in enumerate(chunks)
            ),
        )
    state = db.note_semantic_store.get_note_state(DATASET_ID, pending.id, NOTE_ID)
    assert state is not None and state.state.value == "pending"
    assert db.note_semantic_store.list_obsolete_vector_ids(
        DATASET_ID,
        pending.id,
        limit=16,
    ) == tuple(chunk.vector_id for chunk in chunks)


def test_obsolete_vector_ledger_claim_retry_and_hard_delete_survival(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    vector_ids = tuple(f"opaque-{index}" for index in range(3))
    with pytest.raises(
        ValueError,
        match="^notes_semantic_dirty_generation_invalid$",
    ):
        db.note_semantic_store.stage_obsolete_vector_cleanup(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            vector_ids=("invalid-zero-generation",),
            source_kind="unpublished",
            note_id=NOTE_ID,
            dirty_generation=0,
            now=NOW,
        )
    assert db.note_semantic_store.stage_obsolete_vector_cleanup(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        vector_ids=vector_ids,
        source_kind="unpublished",
        note_id=NOTE_ID,
        dirty_generation=1,
        now=NOW,
    ) == 3
    first = db.note_semantic_store.claim_obsolete_vector_cleanup_batch(
        dataset_id=DATASET_ID,
        limit=2,
        now=NOW,
    )
    assert first is not None
    assert first.vector_ids == vector_ids[:2]
    assert db.note_semantic_store.retry_obsolete_vector_cleanup(
        dataset_id=DATASET_ID,
        ledger_ids=first.ledger_ids,
        claim_token=first.claim_token,
        error_code="backend_unavailable",
        retry_at=NOW.replace(minute=1),
        now=NOW,
    )
    assert db.note_semantic_store.list_obsolete_vector_ids(
        DATASET_ID,
        pending.id,
        limit=16,
    ) == vector_ids
    db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
    assert db.note_store.delete_note(
        NOTE_ID,
        hard_delete=True,
        semantic_dataset_id=DATASET_ID,
    )
    assert db.note_semantic_store.list_obsolete_vector_ids(
        DATASET_ID,
        pending.id,
        limit=16,
    ) == vector_ids


def test_cleanup_ledger_never_claims_an_id_in_the_current_manifest(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    _resolved, pending, integrity = _prepare_ready_generation(db)
    assert integrity.vector_ids
    assert db.note_semantic_store.stage_obsolete_vector_cleanup(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        vector_ids=integrity.vector_ids,
        source_kind="unpublished",
        note_id=NOTE_ID,
        dirty_generation=1,
        now=NOW,
    ) == len(integrity.vector_ids)

    assert db.note_semantic_store.claim_obsolete_vector_cleanup_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        limit=16,
        now=NOW,
    ) is None
    assert db.note_semantic_store.list_obsolete_vector_ids(
        DATASET_ID,
        pending.id,
        limit=16,
    ) == integrity.vector_ids

def test_systemic_generation_failure_blocks_activation_but_note_failures_can_degrade(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    assert db.note_semantic_store.fail_generation(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        generation_fencing_token=GENERATION_FENCE,
        expected_configuration_revision=resolved.configuration_revision,
        error_code="provider_unavailable",
        now=NOW,
    )
    integrity = db.note_semantic_store.get_generation_integrity(DATASET_ID, pending.id)

    with pytest.raises(SemanticIndexingError, match="notes_semantic_systemic_failure"):
        db.note_semantic_store.assert_generation_activatable(integrity)


@pytest.mark.asyncio
async def test_activation_vector_integrity_fetches_are_bounded(
    sqlite_db: CharactersRAGDB,
) -> None:
    db = sqlite_db
    db.note_store.add_note("Title", "abcdefghijklmnop", note_id=NOTE_ID)
    enabled, pending = _create_pending_generation(db)
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=pending.id,
        note_id=NOTE_ID,
        title="Title",
        content="abcdefghijklmnop",
        content_version=1,
        settings=SemanticIndexSettings(
            max_chunk_code_points=4,
            max_chunks_per_note=16,
            max_query_vectors_per_call=16,
        ),
    )
    assert len(chunks) > 2
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=GENERATION_FENCE,
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint(
                    "Title", "abcdefghijklmnop", 1
                ),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    vectors = MemoryVectors([])
    service = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=lambda fence: _authority_from_store(db, fence),
        clock=lambda: NOW,
        receipt_factory=lambda: "receipt-bounded-fetch",
        max_cleanup_vectors=2,
        max_vectors_per_publication=16,
    )
    fence = _fence(resolved, pending)
    await service.publish_note(
        fence,
        claim,
        chunks,
        tuple(SemanticVector(chunk.vector_id, (1.0, 2.0)) for chunk in chunks),
    )

    await service.activate(fence)

    assert len(vectors.fetch_sizes) > 1
    assert max(vectors.fetch_sizes) <= 2


@pytest.mark.asyncio
async def test_store_publication_sql_has_live_postgres_parity(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    try:
        enabled, pending = _create_pending_generation(db)
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            expected_configuration_revision=enabled.configuration_revision,
            dimensions=2,
            compatibility_hash="compatibility-v1",
            now=NOW,
        )
        assert resolved is not None
        db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
        chunks = build_semantic_chunks(
            generation_id=pending.id,
            note_id=NOTE_ID,
            title="Title",
            content="Body",
            content_version=1,
        )
        assert db.note_semantic_store.seed_generation_snapshot(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            expected_configuration_revision=resolved.configuration_revision,
            generation_fencing_token=GENERATION_FENCE,
            seeds=(
                SemanticSnapshotSeed(
                    note_id=NOTE_ID,
                    content_version=1,
                    content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                    state="pending",
                    planned_chunk_count=len(chunks),
                    error_code=None,
                ),
            ),
            now=NOW,
        )
        claim = db.note_semantic_store.claim_work_batch(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            kind="index_note",
            limit=1,
            now=NOW,
        )[0]
        vectors = MemoryVectors([])

        async def revalidate(fence: SemanticExecutionFence) -> SemanticAuthorityState:
            return _authority_from_store(db, fence)

        service = SemanticPublicationService(
            store=db.note_semantic_store,
            vectors=vectors,
            revalidate=revalidate,
            clock=lambda: NOW,
            receipt_factory=lambda: "receipt-postgres",
        )
        fence = _fence(resolved, pending)
        publication = await service.publish_note(
            fence,
            claim,
            chunks,
            tuple(
                SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
                for index, chunk in enumerate(chunks)
            ),
        )
        assert db.note_semantic_store.list_visible_vector_ids(
            DATASET_ID,
            pending.id,
            NOTE_ID,
        ) == publication.new_vector_ids
        receipt = await service.activate(fence)
        activated = db.note_semantic_store.get_configuration(DATASET_ID)
        assert activated is not None and activated.active_generation_id == pending.id
        assert receipt.receipt == "receipt-postgres"

        active_fence = replace(
            fence,
            configuration_revision=activated.configuration_revision,
        )
        assert db.note_store.update_note(
            NOTE_ID,
            {"content": "Body revised"},
            expected_version=1,
            semantic_dataset_id=DATASET_ID,
        )
        revised_chunks = build_semantic_chunks(
            generation_id=pending.id,
            note_id=NOTE_ID,
            title="Title",
            content="Body revised",
            content_version=2,
        )
        edit_claim = db.note_semantic_store.claim_work_batch(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            kind="index_note",
            limit=1,
            now=CLAIM_NOW,
        )[0]
        revised = await service.publish_note(
            active_fence,
            edit_claim,
            revised_chunks,
            tuple(
                SemanticVector(chunk.vector_id, (2.0, 3.0))
                for chunk in revised_chunks
            ),
        )
        assert db.note_semantic_store.list_visible_vector_ids(
            DATASET_ID, pending.id, NOTE_ID
        ) == revised.new_vector_ids
        assert db.note_store.delete_note(
            NOTE_ID,
            expected_version=2,
            semantic_dataset_id=DATASET_ID,
        )
        delete_claim = db.note_semantic_store.claim_work_batch(
            dataset_id=DATASET_ID,
            generation_id=pending.id,
            kind="delete_note_vectors",
            limit=1,
            now=CLAIM_NOW,
        )[0]
        tombstone = await service.publish_tombstone(active_fence, delete_claim)
        assert tombstone.old_vector_ids == revised.new_vector_ids
        assert db.note_semantic_store.list_visible_vector_ids(
            DATASET_ID, pending.id, NOTE_ID
        ) == ()
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("ordering", ["mutation_first", "activation_first"])
def test_postgres_activation_and_note_mutation_are_serialized_without_lost_dirtiness(
    pg_database_config: DatabaseConfig,
    ordering: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    release = threading.Event()
    first_phase = threading.Event()
    second_started = threading.Event()
    errors: list[BaseException] = []
    activation_results: list[object] = []
    threads: list[threading.Thread] = []
    try:
        resolved, pending, integrity = _prepare_ready_generation(db)
        activation_kwargs = {
            "dataset_id": DATASET_ID,
            "generation_id": pending.id,
            "expected_configuration_revision": resolved.configuration_revision,
            "generation_fencing_token": GENERATION_FENCE,
            "expected_manifest_hash": integrity.manifest_hash,
            "expected_vector_ids": integrity.vector_ids,
            "expected_dimensions": 2,
            "expected_compatibility_hash": "compatibility-v1",
            "publication_receipt": f"receipt-race-{ordering}",
            "now": NOW,
        }

        def mutation_transaction() -> None:
            try:
                with db.transaction():
                    assert db.note_store.update_note(
                        NOTE_ID,
                        {"content": "Body revised"},
                        expected_version=1,
                        semantic_dataset_id=DATASET_ID,
                    )
                    first_phase.set()
                    assert release.wait(timeout=10)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
                errors.append(exc)
                first_phase.set()
                release.set()

        def activation_transaction() -> None:
            try:
                with db.transaction():
                    activation_results.append(
                        db.note_semantic_store.activate_generation_verified(
                            **activation_kwargs
                        )
                    )
                    first_phase.set()
                    assert release.wait(timeout=10)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
                errors.append(exc)
                first_phase.set()
                release.set()

        def activation_call() -> None:
            try:
                second_started.set()
                activation_results.append(
                    db.note_semantic_store.activate_generation_verified(**activation_kwargs)
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
                errors.append(exc)

        def mutation_call() -> None:
            try:
                second_started.set()
                assert db.note_store.update_note(
                    NOTE_ID,
                    {"content": "Body revised"},
                    expected_version=1,
                    semantic_dataset_id=DATASET_ID,
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
                errors.append(exc)

        if ordering == "mutation_first":
            first = threading.Thread(target=mutation_transaction, daemon=True)
            second = threading.Thread(target=activation_call, daemon=True)
        else:
            first = threading.Thread(target=activation_transaction, daemon=True)
            second = threading.Thread(target=mutation_call, daemon=True)
        threads.extend((first, second))
        first.start()
        assert first_phase.wait(timeout=10)
        second.start()
        assert second_started.wait(timeout=10)
        time.sleep(0.2)
        assert second.is_alive(), "the second transaction did not wait on the Note lock"
        release.set()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        state = db.note_semantic_store.get_note_state(DATASET_ID, pending.id, NOTE_ID)
        config = db.note_semantic_store.get_configuration(DATASET_ID)
        assert state is not None and config is not None
        assert state.content_version == 2
        assert state.state.value == "pending"
        if ordering == "mutation_first":
            assert activation_results == []
            assert len(errors) == 1
            assert isinstance(errors[0], SemanticIndexingError)
            assert errors[0].code == "notes_semantic_snapshot_incomplete"
            assert config.active_generation_id is None
        else:
            assert errors == []
            assert activation_results[0] is not None
            assert config.active_generation_id == pending.id
            assert state.dirty_generation >= 2
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=10)
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_cleanup_claims_are_concurrent_crash_durable_and_retryable(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    start = threading.Event()
    errors: list[BaseException] = []
    claims: list[object] = []
    threads: list[threading.Thread] = []
    try:
        vector_ids = tuple(f"opaque-pg-{index}" for index in range(4))
        assert db.note_semantic_store.stage_obsolete_vector_cleanup(
            dataset_id=DATASET_ID,
            generation_id="generation-cleanup-pg",
            vector_ids=vector_ids,
            source_kind="unpublished",
            note_id=NOTE_ID,
            dirty_generation=1,
            now=NOW,
        ) == 4

        def claim_page() -> None:
            try:
                assert start.wait(timeout=10)
                claim = db.note_semantic_store.claim_obsolete_vector_cleanup_batch(
                    dataset_id=DATASET_ID,
                    limit=2,
                    now=NOW,
                )
                assert claim is not None
                claims.append(claim)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
                errors.append(exc)

        threads = [
            threading.Thread(target=claim_page, daemon=True),
            threading.Thread(target=claim_page, daemon=True),
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert errors == []
        assert len(claims) == 2
        first_ids = set(claims[0].vector_ids)  # type: ignore[union-attr]
        second_ids = set(claims[1].vector_ids)  # type: ignore[union-attr]
        assert first_ids.isdisjoint(second_ids)
        concurrently_claimed = first_ids | second_ids
        assert 2 <= len(concurrently_claimed) <= 4

        recovery = NOW + timedelta(minutes=1)
        assert db.note_semantic_store.reclaim_expired_obsolete_vector_claims(
            dataset_id=DATASET_ID,
            expired_before=recovery,
            limit=4,
            now=recovery,
        ) == len(concurrently_claimed)
        assert db.note_semantic_store.list_obsolete_vector_ids(
            DATASET_ID,
            "generation-cleanup-pg",
            limit=8,
        ) == vector_ids
        retried = db.note_semantic_store.claim_obsolete_vector_cleanup_batch(
            dataset_id=DATASET_ID,
            limit=4,
            now=recovery,
        )
        assert retried is not None
        assert set(retried.vector_ids) == set(vector_ids)
        assert db.note_semantic_store.complete_obsolete_vector_claim(
            dataset_id=DATASET_ID,
            ledger_ids=retried.ledger_ids,
            claim_token=retried.claim_token,
        )
        assert db.note_semantic_store.list_obsolete_vector_ids(
            DATASET_ID,
            "generation-cleanup-pg",
            limit=8,
        ) == ()
    finally:
        start.set()
        for thread in threads:
            thread.join(timeout=10)
        db.close_all_connections()
        backend.get_pool().close_all()
