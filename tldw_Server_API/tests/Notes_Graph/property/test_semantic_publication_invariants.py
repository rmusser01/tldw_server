"""Generated lifecycle interleavings for semantic publication invariants."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
        self.delete_id_calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail_next_delete_ids = False

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
        ids = tuple(vector_ids)
        self.delete_id_calls.append((generation_id, ids))
        if self.fail_next_delete_ids:
            self.fail_next_delete_ids = False
            return SemanticVectorCleanup(confirmed_absent=False)
        for vector_id in ids:
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
        endpoint_origin="https://api.example.test",
        credential_source="server_default",
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
        endpoint_origin=fence.endpoint_origin,
        credential_source=fence.credential_source,
        endpoint_origin_revision=fence.endpoint_origin_revision,
        endpoint_policy_allowed=True,
        compatibility_hash=fence.compatibility_hash,
        dimensions=fence.dimensions,
        vector_backend=fence.vector_backend,
        vector_capable=True,
    )


@pytest.mark.parametrize("seed", (3, 17, 101))
@given(
    operations=st.lists(
        st.sampled_from(("edit", "second_edit", "claim", "tombstone")),
        min_size=1,
        max_size=6,
    )
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_stale_claims_never_clear_newer_dirtiness(
    tmp_path: Path,
    seed: int,
    operations: list[str],
) -> None:
    db = _db(tmp_path / f"dirty-{seed}-{next(CASE_COUNTER)}.sqlite")
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
        mutation = {"version": 1, "changed": False, "deleted": False}

        def interleave_store_operations() -> None:
            for operation in operations:
                if operation == "claim":
                    db.note_semantic_store.claim_work_batch(
                        dataset_id=DATASET_ID,
                        generation_id=generation.id,
                        kind=SemanticWorkKind.INDEX_NOTE,
                        limit=1,
                        now=NOW,
                    )
                elif operation in {"edit", "second_edit"} and not mutation["deleted"]:
                    next_version = int(mutation["version"]) + 1
                    changed = db.note_store.update_note(
                        NOTE_ID,
                        {"content": f"Body v{next_version}"},
                        expected_version=int(mutation["version"]),
                        semantic_dataset_id=DATASET_ID,
                    )
                    if changed:
                        mutation.update(version=next_version, changed=True)
                elif operation == "tombstone" and not mutation["deleted"]:
                    deleted = db.note_store.soft_delete_note(
                        NOTE_ID,
                        expected_version=int(mutation["version"]),
                        semantic_dataset_id=DATASET_ID,
                    )
                    if deleted:
                        mutation.update(
                            version=int(mutation["version"]) + 1,
                            changed=True,
                            deleted=True,
                        )

        vectors.after_upsert = interleave_store_operations
        embeddings = tuple(
            SemanticVector(chunk.vector_id, (1.0, float(index + 1)))
            for index, chunk in enumerate(chunks)
        )
        try:
            publication = await service.publish_note(fence, claim, chunks, embeddings)
        except SemanticIndexingError as exc:
            assert exc.code == "notes_semantic_note_claim_stale"
            assert mutation["changed"]
            state = db.note_semantic_store.get_note_state(DATASET_ID, generation.id, NOTE_ID)
            assert state is not None
            assert state.content_version == mutation["version"]
            assert state.dirty_generation > (claim.dirty_generation or 0)
            assert state.state.value in {"pending", "tombstoned"}
            assert claim.claim_token is not None
            db.note_semantic_store.release_work_claim(
                dataset_id=DATASET_ID,
                work_id=claim.id,
                claim_token=claim.claim_token,
                fencing_token=claim.fencing_token,
                now=NOW,
            )
            after_release = db.note_semantic_store.get_note_state(
                DATASET_ID, generation.id, NOTE_ID
            )
            assert after_release is not None
            assert after_release.dirty_generation == state.dirty_generation
            assert after_release.state == state.state
        else:
            assert not mutation["changed"]
            assert publication.new_vector_ids == tuple(chunk.vector_id for chunk in chunks)
            assert db.note_semantic_store.list_visible_vector_ids(
                DATASET_ID, generation.id, NOTE_ID
            ) == publication.new_vector_ids
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("seed", (3, 17, 101))
@given(
    generated_operations=st.lists(
        st.sampled_from(
            (
                "edit",
                "tombstone",
                "hard_delete",
                "cleanup_retry",
                "cleanup_crash",
            )
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_manifest_visibility_precedes_cleanup_and_old_generation_cleanup_is_fenced(
    tmp_path: Path,
    seed: int,
    generated_operations: list[str],
) -> None:
    db = _db(tmp_path / f"cleanup-{seed}-{next(CASE_COUNTER)}.sqlite")
    try:
        db.note_store.add_note("Title", "Body", note_id=NOTE_ID)
        config, old_generation = _resolved_generation(db)
        vectors = InterleavingVectors()

        async def revalidate(actual_fence):
            return _authority(actual_fence)

        clock = [NOW]
        service = SemanticPublicationService(
            store=db.note_semantic_store,
            vectors=vectors,
            revalidate=revalidate,
            clock=lambda: clock[0],
            receipt_factory=lambda: "receipt-property",
            max_cleanup_vectors=1,
        )

        async def publish_generation(
            generation,
            generation_config,
            token: str,
            *,
            content: str,
            content_version: int,
        ):
            chunks = build_semantic_chunks(
                generation_id=generation.id,
                note_id=NOTE_ID,
                title="Title",
                content=content,
                content_version=content_version,
            )
            seed = SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=content_version,
                content_fingerprint=semantic_content_fingerprint(
                    "Title", content, content_version
                ),
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
            content="Body",
            content_version=1,
        )
        active = db.note_semantic_store.get_configuration(DATASET_ID)
        assert active is not None

        old_active_fence = replace(
            _fence(config, old_generation.id),
            configuration_revision=active.configuration_revision,
        )
        assert db.note_store.update_note(
            NOTE_ID,
            {"content": "Body v2"},
            expected_version=1,
            semantic_dataset_id=DATASET_ID,
        )
        revised_chunks = build_semantic_chunks(
            generation_id=old_generation.id,
            note_id=NOTE_ID,
            title="Title",
            content="Body v2",
            content_version=2,
        )
        revised_claim = db.note_semantic_store.claim_work_batch(
            dataset_id=DATASET_ID,
            generation_id=old_generation.id,
            kind=SemanticWorkKind.INDEX_NOTE,
            limit=1,
            now=NOW,
        )[0]
        revised_publication = await service.publish_note(
            old_active_fence,
            revised_claim,
            revised_chunks,
            tuple(
                SemanticVector(chunk.vector_id, (2.0, float(index + 1)))
                for index, chunk in enumerate(revised_chunks)
            ),
        )
        assert db.note_semantic_store.list_obsolete_vector_ids(
            DATASET_ID,
            old_generation.id,
            limit=16,
        ) == old_publication.new_vector_ids

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
            content="Body v2",
            content_version=2,
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
        assert db.note_semantic_store.list_visible_vector_ids(
            DATASET_ID, new_generation.id, NOTE_ID
        ) == new_publication.new_vector_ids

        prefixes = {
            3: ("cleanup_crash", "hard_delete"),
            17: ("hard_delete", "cleanup_crash"),
            101: ("tombstone", "cleanup_crash", "hard_delete"),
        }
        operations = prefixes[seed] + tuple(generated_operations)
        cleanup_complete = False
        cleanup_crashed = False
        note_deleted = False
        note_version = 2
        for operation in operations:
            if operation == "cleanup_crash" and not cleanup_crashed and not cleanup_complete:
                crashed_claim = (
                    db.note_semantic_store.claim_obsolete_vector_cleanup_batch(
                        dataset_id=DATASET_ID,
                        generation_id=old_generation.id,
                        limit=1,
                        now=clock[0],
                    )
                )
                assert crashed_claim is not None
                assert db.note_semantic_store.authorize_obsolete_vector_claim(
                    dataset_id=DATASET_ID,
                    ledger_ids=crashed_claim.ledger_ids,
                    claim_token=crashed_claim.claim_token,
                )
                deleted = await vectors.delete_ids(
                    DATASET_ID,
                    old_generation.id,
                    crashed_claim.vector_ids,
                )
                assert deleted.confirmed_absent
                assert all(
                    (old_generation.id, vector_id) not in vectors.values
                    for vector_id in crashed_claim.vector_ids
                )
                cleanup_crashed = True
                assert db.note_semantic_store.list_obsolete_vector_ids(
                    DATASET_ID,
                    old_generation.id,
                    limit=16,
                )
                recovery = clock[0] + timedelta(minutes=1)
                assert db.note_semantic_store.reclaim_expired_obsolete_vector_claims(
                    dataset_id=DATASET_ID,
                    expired_before=recovery,
                    limit=1,
                    now=recovery,
                ) == 1
                clock[0] = recovery
            elif operation == "cleanup_retry" and not cleanup_complete:
                cleanup_complete = await service.cleanup_generation(cleanup_claim)
            elif operation == "edit" and not note_deleted:
                next_version = note_version + 1
                if db.note_store.update_note(
                    NOTE_ID,
                    {"content": f"Body v{next_version}"},
                    expected_version=note_version,
                    semantic_dataset_id=DATASET_ID,
                ):
                    note_version = next_version
            elif operation == "tombstone" and not note_deleted:
                if db.note_store.soft_delete_note(
                    NOTE_ID,
                    expected_version=note_version,
                    semantic_dataset_id=DATASET_ID,
                ):
                    note_version += 1
                    note_deleted = True
            elif operation == "hard_delete":
                if db.note_store.delete_note(
                    NOTE_ID,
                    hard_delete=True,
                    semantic_dataset_id=DATASET_ID,
                ):
                    note_deleted = True

        for _ in range(8):
            if cleanup_complete:
                break
            cleanup_complete = await service.cleanup_generation(cleanup_claim)

        assert cleanup_complete is True
        assert vectors.deleted_generations == [old_generation.id]
        assert vectors.delete_id_calls
        assert all(
            generation_id == old_generation.id and len(vector_ids) <= 1
            for generation_id, vector_ids in vectors.delete_id_calls
        )
        assert all(
            (old_generation.id, vector_id) not in vectors.values
            for vector_id in old_publication.new_vector_ids
            + revised_publication.new_vector_ids
        )
        assert all(
            (new_generation.id, vector_id) in vectors.values
            for vector_id in new_publication.new_vector_ids
        )
        if note_deleted:
            assert set(new_publication.new_vector_ids) <= set(
                db.note_semantic_store.list_obsolete_vector_ids(
                    DATASET_ID,
                    new_generation.id,
                    limit=16,
                )
            )
    finally:
        db.close_all_connections()
