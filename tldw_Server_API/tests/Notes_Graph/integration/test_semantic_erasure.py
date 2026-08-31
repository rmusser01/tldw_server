"""Data-subject erasure contracts for Notes semantic projections."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticIndexingError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_erasure import (
    SemanticErasureCoordinator,
    SemanticErasureError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticPublicationService,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVectorCleanup,
    SemanticVectorError,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
OWNER_ID = "owner-a"
DATASET_ID = "dataset-a"
NOTE_ID = "11111111-1111-4111-8111-111111111111"


class RecordingVectors:
    def __init__(
        self,
        events: list[str],
        *,
        confirmed_absent: bool = True,
    ) -> None:
        self.events = events
        self.confirmed_absent = confirmed_absent
        self.deleted_generations: list[str] = []

    async def delete_ids(self, dataset_id: str, generation_id: str, vector_ids):
        del dataset_id
        self.events.append(f"delete_ids:{generation_id}")
        return SemanticVectorCleanup(confirmed_absent=self.confirmed_absent)

    async def delete_generation(self, dataset_id: str, generation_id: str):
        del dataset_id
        self.events.append(f"delete_generation:{generation_id}")
        self.deleted_generations.append(generation_id)
        return SemanticVectorCleanup(confirmed_absent=self.confirmed_absent)


class FailingVectors(RecordingVectors):
    async def delete_generation(self, dataset_id: str, generation_id: str):
        del dataset_id, generation_id
        raise SemanticVectorError("backend-specific-secret")


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        str(tmp_path / "semantic-erasure.sqlite"),
        client_id=OWNER_ID,
    )
    yield database
    database.close_all_connections()


def _create_resolved_generation(
    db: CharactersRAGDB,
    *,
    backend: str = "chromadb",
):
    config = db.note_semantic_store.create_configuration(
        dataset_id=DATASET_ID,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend=backend,
        storage_boundary="server_local",
        storage_label="semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision=config.capability_revision or "",
        now=NOW,
    )
    assert enabled is not None
    generation = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id="job-v1",
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=768,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    return resolved


def _insert_obsolete_vector(
    db: CharactersRAGDB,
    *,
    dataset_id: str,
    generation_id: str,
    vector_id: str,
) -> None:
    timestamp = NOW.isoformat()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_semantic_obsolete_vectors("
            "id,owner_user_id,dataset_id,generation_id,vector_id,source_kind,"
            "claim_state,attempt_count,next_eligible_at,created_at,updated_at"
            ") VALUES (?,?,?,?,?,'unpublished','pending',0,?,?,?)",
            (
                f"ledger-{dataset_id}-{vector_id}",
                OWNER_ID,
                dataset_id,
                generation_id,
                vector_id,
                timestamp,
                timestamp,
                timestamp,
            ),
        )


def _insert_receipt(db: CharactersRAGDB, *, dataset_id: str) -> None:
    timestamp = NOW.isoformat()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_semantic_operation_receipts("
            "owner_user_id,dataset_id,key_digest,action,request_fingerprint,"
            "expected_revision,state,expires_at,created_at,updated_at"
            ") VALUES (?,?,?,'enable',?,0,'completed',?,?,?)",
            (
                OWNER_ID,
                dataset_id,
                "a" * 64,
                "b" * 64,
                (NOW + timedelta(days=1)).isoformat(),
                timestamp,
                timestamp,
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("chromadb", "pgvector"))
async def test_erasure_fences_cleans_and_purges_only_semantic_state(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db, backend=backend)
    _insert_obsolete_vector(
        db,
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        vector_id="obsolete-vector",
    )
    _insert_receipt(db, dataset_id=DATASET_ID)
    events: list[str] = []
    vectors = RecordingVectors(events)

    original_disable = db.note_semantic_store.disable_and_schedule_cleanup
    original_purge = db.note_semantic_store.purge_semantic_dataset_for_erasure

    def _disable(**kwargs):
        events.append("fence")
        return original_disable(**kwargs)

    def _purge(**kwargs):
        events.append("purge")
        return original_purge(**kwargs)

    monkeypatch.setattr(db.note_semantic_store, "disable_and_schedule_cleanup", _disable)
    monkeypatch.setattr(
        db.note_semantic_store,
        "purge_semantic_dataset_for_erasure",
        _purge,
    )

    async def _vectors_for_backend(backend_name: str):
        events.append(f"backend:{backend_name}")
        return vectors

    result = await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=_vectors_for_backend,
        timeout_seconds=1,
    ).erase()

    assert result.datasets == 1
    assert result.cleaned_generations == 1
    assert events[0] == "fence"
    assert f"backend:{backend}" in events
    assert events.index(f"delete_ids:{generation.id}") < events.index("purge")
    assert events.index(f"delete_generation:{generation.id}") < events.index("purge")
    assert db.note_semantic_store.get_configuration(DATASET_ID) is None
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_semantic_obsolete_vectors"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM note_semantic_operation_receipts"
        ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_maintenance_catalog_discovers_v66_and_v67_only_state_for_purge(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    config = db.note_semantic_store.create_configuration(
        dataset_id=DATASET_ID,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    assert config.dataset_id == DATASET_ID
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM note_semantic_index_configs "
            "WHERE owner_user_id=? AND dataset_id=?",
            (OWNER_ID, DATASET_ID),
        )
    _insert_receipt(db, dataset_id=DATASET_ID)
    _insert_obsolete_vector(
        db,
        dataset_id=DATASET_ID,
        generation_id="already-removed-generation",
        vector_id="already-removed-vector",
    )

    async def _unexpected_vectors(_backend_name: str):
        raise AssertionError("orphan bookkeeping has no vector binding")

    result = await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=_unexpected_vectors,
        timeout_seconds=1,
    ).erase()

    assert result.datasets == 1
    assert db.note_store.get_note_by_id(NOTE_ID) is not None
    assert db.note_semantic_store.list_maintenance_dataset_ids(limit=100) == ()


def test_final_semantic_purge_rejects_pending_cleanup(
    db: CharactersRAGDB,
) -> None:
    generation = _create_resolved_generation(db)
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert config is not None
    disabled = db.note_semantic_store.disable_and_schedule_cleanup(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        now=NOW,
    )
    assert disabled is not None

    with pytest.raises(SemanticIndexingError) as exc_info:
        db.note_semantic_store.purge_semantic_dataset_for_erasure(
            dataset_id=DATASET_ID,
        )

    assert exc_info.value.code == "notes_semantic_erasure_finalization_fence_lost"
    assert db.note_semantic_store.get_generation(DATASET_ID, generation.id) is not None
    assert db.note_semantic_store.has_pending_cleanup(DATASET_ID) is True


@pytest.mark.asyncio
async def test_unconfirmed_vector_absence_fails_closed_and_retains_retry_identity(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    vectors = RecordingVectors([], confirmed_absent=False)

    async def _vectors_for_backend(_backend_name: str):
        return vectors

    with pytest.raises(SemanticErasureError) as exc_info:
        await SemanticErasureCoordinator(
            db=db,
            vector_store_factory=_vectors_for_backend,
            timeout_seconds=1,
        ).erase()

    assert exc_info.value.code == "notes_semantic_erasure_cleanup_unconfirmed"
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert config is not None
    assert config.desired_state.value == "disabled"
    assert db.note_semantic_store.get_generation(DATASET_ID, generation.id) is not None
    assert db.note_semantic_store.has_pending_cleanup(DATASET_ID) is True
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_vector_backend_failure_is_bounded_and_retains_retry_identity(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)

    async def _vectors_for_backend(_backend_name: str):
        return FailingVectors([])

    with pytest.raises(SemanticErasureError) as exc_info:
        await SemanticErasureCoordinator(
            db=db,
            vector_store_factory=_vectors_for_backend,
            timeout_seconds=1,
        ).erase()

    assert exc_info.value.code == "notes_semantic_erasure_backend_unavailable"
    assert "backend-specific-secret" not in str(exc_info.value)
    assert db.note_semantic_store.get_generation(DATASET_ID, generation.id) is not None
    assert db.note_semantic_store.has_pending_cleanup(DATASET_ID) is True
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_obsolete_ledger_requeues_an_already_deleted_generation(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert config is not None
    disabled = db.note_semantic_store.disable_and_schedule_cleanup(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        now=NOW,
    )
    assert disabled is not None
    claim = db.note_semantic_store.claim_generation_cleanup_batch(
        dataset_id=DATASET_ID,
        limit=1,
        now=NOW,
    )[0]
    first_vectors = RecordingVectors([])
    publication = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=first_vectors,
        revalidate=lambda _fence: None,
        clock=lambda: NOW,
        receipt_factory=lambda: "first-cleanup",
    )
    assert await publication.cleanup_generation(claim) is True
    _insert_obsolete_vector(
        db,
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        vector_id="late-obsolete-vector",
    )
    second_vectors = RecordingVectors([])

    async def _vectors_for_backend(_backend_name: str):
        return second_vectors

    result = await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=_vectors_for_backend,
        timeout_seconds=1,
        clock=lambda: NOW,
    ).erase()

    assert result.cleaned_generations == 1
    assert f"delete_ids:{generation.id}" in second_vectors.events
    assert second_vectors.deleted_generations == [generation.id]


@pytest.mark.asyncio
async def test_valid_in_flight_cleanup_claim_is_not_stolen_before_timeout(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert config is not None
    disabled = db.note_semantic_store.disable_and_schedule_cleanup(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        now=datetime.now(timezone.utc),
    )
    assert disabled is not None
    claimed = db.note_semantic_store.claim_generation_cleanup_batch(
        dataset_id=DATASET_ID,
        limit=1,
        now=datetime.now(timezone.utc),
    )[0]

    with pytest.raises(SemanticErasureError) as exc_info:
        await SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: RecordingVectors([]),
            timeout_seconds=0.03,
            poll_interval_seconds=0.005,
            lease_seconds=180,
        ).erase()

    assert exc_info.value.code == "notes_semantic_erasure_timeout"
    current = db.note_semantic_store.get_generation(DATASET_ID, generation.id)
    assert current is not None
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT claim_state,claim_token FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (OWNER_ID, DATASET_ID, claimed.id),
        ).fetchone()
    assert tuple(row) == ("claimed", claimed.claim_token)
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_erasure_fence_rejects_in_flight_generation_publication(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    publication_result: list[object] = []
    vectors = RecordingVectors([])

    async def _vectors_after_fence(_backend_name: str):
        publication_result.append(
            db.note_semantic_store.activate_generation(
                dataset_id=DATASET_ID,
                generation_id=generation.id,
                expected_configuration_revision=generation.configuration_revision,
                publication_receipt="late-publication",
                now=NOW + timedelta(seconds=1),
            )
        )
        return vectors

    await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=_vectors_after_fence,
        timeout_seconds=1,
    ).erase()

    assert publication_result == [None]


@pytest.mark.asyncio
async def test_delayed_old_generation_cleanup_cannot_target_new_generation(
    db: CharactersRAGDB,
) -> None:
    old_generation = _create_resolved_generation(db)
    active = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=old_generation.id,
        expected_configuration_revision=old_generation.configuration_revision,
        publication_receipt="old-publication",
        now=NOW,
    )
    assert active is not None
    disabled = db.note_semantic_store.disable_and_schedule_cleanup(
        dataset_id=DATASET_ID,
        expected_configuration_revision=active.configuration_revision,
        now=NOW,
    )
    assert disabled is not None
    reenabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=disabled.configuration_revision,
        capability_revision=disabled.capability_revision or "",
        now=NOW,
    )
    assert reenabled is not None
    new_generation = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=reenabled.configuration_revision,
        compatibility_hash=reenabled.compatibility_hash,
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=reenabled.dimensions,
        root_job_id="job-v2",
        model_revision=reenabled.model_revision,
        now=NOW,
    )
    active_new = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=new_generation.id,
        expected_configuration_revision=new_generation.configuration_revision,
        publication_receipt="new-publication",
        now=NOW,
    )
    assert active_new is not None
    claim = db.note_semantic_store.claim_generation_cleanup_batch(
        dataset_id=DATASET_ID,
        limit=1,
        now=NOW,
    )[0]
    assert claim.generation_id == old_generation.id
    vectors = RecordingVectors([])
    publication = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=lambda _fence: None,
        clock=lambda: NOW,
        receipt_factory=lambda: "cleanup-receipt",
    )

    assert await publication.cleanup_generation(claim) is True
    assert vectors.deleted_generations == [old_generation.id]
    assert db.note_semantic_store.get_configuration(DATASET_ID).active_generation_id == new_generation.id
