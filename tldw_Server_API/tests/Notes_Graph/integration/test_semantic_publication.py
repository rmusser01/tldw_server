"""SQLite and PostgreSQL contracts for cross-store semantic publication."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGenerationState,
    SemanticSnapshotSeed,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    build_semantic_chunks,
    semantic_content_fingerprint,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    PendingSemanticConfig,
    ResolvedDimension,
    SemanticEmbeddingBatch,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_indexing import (
    InitialGenerationRequest,
    NoteVersionRef,
    SemanticGenerationBuilder,
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
OWNER_ID = "owner-a"
DATASET_ID = "dataset-a"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
GENERATION_FENCE = "-".join(("job", "fence", "v1"))
WRONG_GENERATION_FENCE = "-".join(("wrong", "fence"))


class MemoryVectors:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.values: dict[tuple[str, str], SemanticVector] = {}

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
        return tuple(
            self.values[(generation_id, vector_id)]
            for vector_id in vector_ids
            if (generation_id, vector_id) in self.values
        )

    async def delete_ids(self, dataset_id: str, generation_id: str, vector_ids):
        self.events.append("vector_cleanup")
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
        return ResolvedDimension(2, config.provider, config.model, config.model_revision)

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
        endpoint_origin_revision=config.endpoint_origin_revision or "",
        endpoint_policy_allowed=True,
        compatibility_hash=config.compatibility_hash,
        dimensions=config.dimensions,
        vector_backend=config.vector_backend or "",
        vector_capable=True,
    )


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
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
