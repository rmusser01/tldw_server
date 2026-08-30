"""Generated lifecycle interleavings for semantic publication invariants."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticSnapshotSeed,
    SemanticWorkKind,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    build_semantic_chunks,
    semantic_content_fingerprint,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticAuthorityState,
    SemanticExecutionFence,
    SemanticIndexingError,
    SemanticPublicationService,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
    SemanticVectorCleanup,
)

pytestmark = pytest.mark.property

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
OWNER_ID = "owner-a"
DATASET_ID = "dataset-a"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
CASE_COUNTER = count()
GENERATION_FENCE_V1 = "-".join(("job", "fence", "v1"))
GENERATION_FENCE_V2 = "-".join(("job", "fence", "v2"))


class InterleavingVectors:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], SemanticVector] = {}
        self.after_upsert = None
        self.deleted_generations: list[str] = []

    async def upsert(self, dataset_id: str, generation_id: str, vectors) -> int:
        for vector in vectors:
            self.values[(generation_id, vector.vector_id)] = vector
        if self.after_upsert is not None:
            callback, self.after_upsert = self.after_upsert, None
            callback()
        return len(vectors)

    async def fetch(self, dataset_id: str, generation_id: str, vector_ids):
        return tuple(
            self.values[(generation_id, vector_id)]
            for vector_id in vector_ids
            if (generation_id, vector_id) in self.values
        )

    async def delete_ids(self, dataset_id: str, generation_id: str, vector_ids):
        for vector_id in vector_ids:
            self.values.pop((generation_id, vector_id), None)
        return SemanticVectorCleanup(confirmed_absent=True)

    async def delete_generation(self, dataset_id: str, generation_id: str):
        self.deleted_generations.append(generation_id)
        for key in tuple(self.values):
            if key[0] == generation_id:
                del self.values[key]
        return SemanticVectorCleanup(confirmed_absent=True)


def _db(path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(path), client_id=OWNER_ID)
    db.note_store._semantic_now = lambda: NOW
    return db


def _resolved_generation(db: CharactersRAGDB):
    config = db.note_semantic_store.create_configuration(
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
        root_job_id=GENERATION_FENCE_V1,
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    return resolved, generation


def _fence(config, generation_id: str, token: str = GENERATION_FENCE_V1):
    return SemanticExecutionFence(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        generation_id=generation_id,
        generation_fencing_token=token,
        configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="openai",
        model="embedding-model-v1",
        model_revision=None,
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=2,
        vector_backend="chromadb",
    )


def _authority(fence: SemanticExecutionFence):
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
        endpoint_origin_revision=fence.endpoint_origin_revision,
        endpoint_policy_allowed=True,
        compatibility_hash=fence.compatibility_hash,
        dimensions=fence.dimensions,
        vector_backend=fence.vector_backend,
        vector_capable=True,
    )


@given(edit_after_upsert=st.booleans(), tombstone_after_publish=st.booleans())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_stale_claims_never_clear_newer_dirtiness(
    tmp_path: Path,
    edit_after_upsert: bool,
    tombstone_after_publish: bool,
) -> None:
    db = _db(tmp_path / f"dirty-{next(CASE_COUNTER)}.sqlite")
    try:
        db.note_store.add_note("Title", "Body v1", note_id=NOTE_ID)
        config, generation = _resolved_generation(db)
        chunks = build_semantic_chunks(
            generation_id=generation.id,
            note_id=NOTE_ID,
            title="Title",
            content="Body v1",
            content_version=1,
        )
        seed = SemanticSnapshotSeed(
            note_id=NOTE_ID,
            content_version=1,
            content_fingerprint=semantic_content_fingerprint("Title", "Body v1", 1),
            state="pending",
            planned_chunk_count=len(chunks),
            error_code=None,
        )
        assert db.note_semantic_store.seed_generation_snapshot(
            dataset_id=DATASET_ID,
            generation_id=generation.id,
            expected_configuration_revision=config.configuration_revision,
            generation_fencing_token=GENERATION_FENCE_V1,
            seeds=(seed,),
            now=NOW,
        )
        claim = db.note_semantic_store.claim_work_batch(
            dataset_id=DATASET_ID,
            generation_id=generation.id,
            kind="index_note",
            limit=1,
            now=NOW,
        )[0]
        vectors = InterleavingVectors()
        fence = _fence(config, generation.id)

        async def revalidate(actual_fence):
            return _authority(actual_fence)

        service = SemanticPublicationService(
            store=db.note_semantic_store,
            vectors=vectors,
            revalidate=revalidate,
            clock=lambda: NOW,
            receipt_factory=lambda: "receipt-property",
        )
        if edit_after_upsert:
            vectors.after_upsert = lambda: db.note_store.update_note(
                NOTE_ID,
                {"content": "Body v2"},
                expected_version=1,
                semantic_dataset_id=DATASET_ID,
            )
        embeddings = tuple(
            SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
            for index, chunk in enumerate(chunks)
        )
        if edit_after_upsert:
            with pytest.raises(SemanticIndexingError, match="notes_semantic_note_claim_stale"):
                await service.publish_note(fence, claim, chunks, embeddings)
            state = db.note_semantic_store.get_note_state(DATASET_ID, generation.id, NOTE_ID)
            assert state is not None
            assert state.content_version == 2
            assert state.dirty_generation > (claim.dirty_generation or 0)
            assert state.state.value == "pending"
        else:
            publication = await service.publish_note(fence, claim, chunks, embeddings)
            assert publication.new_vector_ids == tuple(chunk.vector_id for chunk in chunks)
            assert db.note_semantic_store.list_visible_vector_ids(
                DATASET_ID, generation.id, NOTE_ID
            ) == publication.new_vector_ids
            if tombstone_after_publish:
                db.note_store.soft_delete_note(
                    NOTE_ID,
                    expected_version=1,
                    semantic_dataset_id=DATASET_ID,
                )
                tombstone_claim = db.note_semantic_store.claim_work_batch(
                    dataset_id=DATASET_ID,
                    generation_id=generation.id,
                    kind=SemanticWorkKind.DELETE_NOTE_VECTORS,
                    limit=1,
                    now=NOW,
                )[0]
                tombstone = await service.publish_tombstone(fence, tombstone_claim)
                assert tombstone.old_vector_ids == publication.new_vector_ids
                assert db.note_semantic_store.list_visible_vector_ids(
                    DATASET_ID, generation.id, NOTE_ID
                ) == ()
                assert any(key[0] == generation.id for key in vectors.values)
                assert await service.cleanup_obsolete(fence, tombstone) is True
                assert not any(key[0] == generation.id for key in vectors.values)
    finally:
        db.close_all_connections()


@given(cleanup_delay=st.integers(min_value=0, max_value=5))
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_manifest_visibility_precedes_cleanup_and_old_generation_cleanup_is_fenced(
    tmp_path: Path,
    cleanup_delay: int,
) -> None:
    db = _db(tmp_path / f"cleanup-{next(CASE_COUNTER)}.sqlite")
    try:
        db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
        config, old_generation = _resolved_generation(db)
        vectors = InterleavingVectors()

        async def revalidate(actual_fence):
            return _authority(actual_fence)

        service = SemanticPublicationService(
            store=db.note_semantic_store,
            vectors=vectors,
            revalidate=revalidate,
            clock=lambda: NOW,
            receipt_factory=lambda: "receipt-property",
        )

        async def publish_generation(generation, generation_config, token: str):
            chunks = build_semantic_chunks(
                generation_id=generation.id,
                note_id=NOTE_ID,
                title="Title",
                content="Body",
                content_version=1,
            )
            seed = SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            )
            assert db.note_semantic_store.seed_generation_snapshot(
                dataset_id=DATASET_ID,
                generation_id=generation.id,
                expected_configuration_revision=generation_config.configuration_revision,
                generation_fencing_token=token,
                seeds=(seed,),
                now=NOW,
            )
            claim = db.note_semantic_store.claim_work_batch(
                dataset_id=DATASET_ID,
                generation_id=generation.id,
                kind=SemanticWorkKind.INDEX_NOTE,
                limit=1,
                now=NOW,
            )[0]
            fence = _fence(generation_config, generation.id, token)
            publication = await service.publish_note(
                fence,
                claim,
                chunks,
                tuple(
                    SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
                    for index, chunk in enumerate(chunks)
                ),
            )
            receipt = await service.activate(fence)
            return publication, receipt

        old_publication, _ = await publish_generation(
            old_generation,
            config,
            GENERATION_FENCE_V1,
        )
        active = db.note_semantic_store.get_configuration(DATASET_ID)
        assert active is not None
        new_generation = db.note_semantic_store.create_generation(
            dataset_id=DATASET_ID,
            configuration_revision=active.configuration_revision,
            compatibility_hash="compatibility-v1",
            dimension_state=SemanticDimensionState.RESOLVED,
            dimensions=2,
            root_job_id=GENERATION_FENCE_V2,
            now=NOW,
        )
        new_publication, _ = await publish_generation(
            new_generation,
            active,
            GENERATION_FENCE_V2,
        )
        cleanup_claims = db.note_semantic_store.claim_work_batch(
            dataset_id=DATASET_ID,
            generation_id=old_generation.id,
            kind=SemanticWorkKind.DELETE_GENERATION,
            limit=1,
            now=NOW,
        )
        assert len(cleanup_claims) == 1
        cleanup_claim = cleanup_claims[0]

        current = db.note_semantic_store.get_configuration(DATASET_ID)
        assert current is not None and current.active_generation_id == new_generation.id
        assert all(
            (old_generation.id, vector_id) in vectors.values
            for vector_id in old_publication.new_vector_ids
        )
        assert all(
            (new_generation.id, vector_id) in vectors.values
            for vector_id in new_publication.new_vector_ids
        )
        for _ in range(cleanup_delay):
            assert current.active_generation_id == new_generation.id

        assert await service.cleanup_generation(cleanup_claim) is True
        assert vectors.deleted_generations == [old_generation.id]
        assert all(
            (old_generation.id, vector_id) not in vectors.values
            for vector_id in old_publication.new_vector_ids
        )
        assert all(
            (new_generation.id, vector_id) in vectors.values
            for vector_id in new_publication.new_vector_ids
        )
    finally:
        db.close_all_connections()
