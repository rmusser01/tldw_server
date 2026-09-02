"""Data-subject erasure contracts for Notes semantic projections."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticIndexingError,
    SemanticSnapshotSeed,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_content import build_semantic_chunks
from tldw_Server_API.app.core.Notes_Graph.semantic_erasure import (
    SemanticErasureCoordinator,
    SemanticErasureError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticAuthorityState,
    SemanticExecutionFence,
    SemanticPublicationService,
    run_quiescent_operation,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
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


class MemoryVectors(RecordingVectors):
    def __init__(self, events: list[str], *, confirmed_absent: bool = True) -> None:
        super().__init__(events, confirmed_absent=confirmed_absent)
        self.values: dict[tuple[str, str], SemanticVector] = {}

    async def upsert(self, dataset_id: str, generation_id: str, vectors) -> int:
        del dataset_id
        self.events.append(f"upsert:{generation_id}")
        for vector in vectors:
            self.values[(generation_id, vector.vector_id)] = vector
        return len(vectors)

    async def delete_ids(self, dataset_id: str, generation_id: str, vector_ids):
        for vector_id in vector_ids:
            self.values.pop((generation_id, vector_id), None)
        return await super().delete_ids(dataset_id, generation_id, vector_ids)

    async def delete_generation(self, dataset_id: str, generation_id: str):
        for key in tuple(self.values):
            if key[0] == generation_id:
                del self.values[key]
        return await super().delete_generation(dataset_id, generation_id)


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


def _seed_and_claim_note(db: CharactersRAGDB, generation):
    chunks = build_semantic_chunks(
        generation_id=generation.id,
        note_id=NOTE_ID,
        title="Private",
        content="Body",
        content_version=1,
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=generation.configuration_revision,
        generation_fencing_token="job-v1",
        seeds=(
            SemanticSnapshotSeed(
                note_id=NOTE_ID,
                content_version=1,
                content_fingerprint=chunks[0].content_fingerprint,
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=NOW,
    )
    claim = db.note_semantic_store.claim_work_batch(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        kind="index_note",
        limit=1,
        now=NOW,
    )[0]
    return chunks, claim


def _publication_fence(generation) -> SemanticExecutionFence:
    return SemanticExecutionFence(
        owner_user_id=OWNER_ID,
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        generation_fencing_token="job-v1",
        configuration_revision=generation.configuration_revision,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision=None,
        endpoint_origin="https://api.example.test",
        credential_source="server_default",
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=768,
        vector_backend="chromadb",
    )


def _publication_authority(
    db: CharactersRAGDB,
    fence: SemanticExecutionFence,
) -> SemanticAuthorityState:
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
        model_revision=config.model_revision,
        endpoint_origin=fence.endpoint_origin,
        credential_source=fence.credential_source,
        endpoint_origin_revision=config.endpoint_origin_revision or "",
        endpoint_policy_allowed=True,
        compatibility_hash=config.compatibility_hash,
        dimensions=config.dimensions,
        vector_backend=config.vector_backend or "",
        vector_capable=True,
    )


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
    original_finalize = db.note_semantic_store.finalize_owner_erasure

    def _disable(**kwargs):
        events.append("fence")
        return original_disable(**kwargs)

    def _finalize(**kwargs):
        events.append("finalize")
        return original_finalize(**kwargs)

    monkeypatch.setattr(db.note_semantic_store, "disable_and_schedule_cleanup", _disable)
    monkeypatch.setattr(
        db.note_semantic_store,
        "finalize_owner_erasure",
        _finalize,
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
    assert result.deleted_notes == 1
    assert events[0] == "fence"
    assert f"backend:{backend}" in events
    assert events.index(f"delete_ids:{generation.id}") < events.index("finalize")
    assert events.index(f"delete_generation:{generation.id}") < events.index("finalize")
    assert db.note_semantic_store.get_configuration(DATASET_ID) is None
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM note_semantic_obsolete_vectors").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM note_semantic_operation_receipts").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_dsr_reclaims_emit_each_committed_retry_once_with_actual_backend(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db, backend="chromadb")
    _insert_obsolete_vector(
        db,
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        vector_id="obsolete-vector",
    )
    events: list[dict[str, object]] = []
    work_counts = iter((2, 0, 0))
    vector_counts = iter((3, 0, 0))
    monkeypatch.setattr(
        db.note_semantic_store,
        "reclaim_expired_dataset_work",
        lambda **_kwargs: next(work_counts, 0),
    )
    monkeypatch.setattr(
        db.note_semantic_store,
        "reclaim_expired_obsolete_vector_claims",
        lambda **_kwargs: next(vector_counts, 0),
    )
    from tldw_Server_API.app.core.Notes_Graph import semantic_erasure

    monkeypatch.setattr(
        semantic_erasure,
        "record_semantic_cleanup_retry",
        lambda **kwargs: events.append(dict(kwargs)),
        raising=False,
    )

    await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=lambda _backend: RecordingVectors([]),
        timeout_seconds=1,
    ).erase()

    assert events == [
        {"status": "failed", "backend": "chromadb", "count": 2},
        {"status": "failed", "backend": "chromadb", "count": 3},
    ]


@pytest.mark.asyncio
async def test_configless_obsolete_state_fails_closed_and_is_not_purged(
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
            "DELETE FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
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

    with pytest.raises(SemanticErasureError) as exc_info:
        await SemanticErasureCoordinator(
            db=db,
            vector_store_factory=_unexpected_vectors,
            timeout_seconds=1,
        ).erase()

    assert exc_info.value.code == "notes_semantic_erasure_cleanup_failed"
    assert db.note_store.get_note_by_id(NOTE_ID) is not None
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM note_semantic_obsolete_vectors").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM note_semantic_operation_receipts").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_sqlite_erasure_uses_database_file_as_note_owner_boundary(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE notes SET client_id=? WHERE id=?",
            ("capture-client", NOTE_ID),
        )

    result = await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=lambda _backend: pytest.fail("no semantic backend"),
        timeout_seconds=1,
    ).erase()

    assert result.deleted_notes == 1
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_sqlite_erasure_fails_closed_on_unscoped_semantic_owner_state(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    timestamp = NOW.isoformat()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_semantic_operation_receipts("
            "owner_user_id,dataset_id,key_digest,action,request_fingerprint,"
            "expected_revision,state,expires_at,created_at,updated_at"
            ") VALUES (?,?,?,'enable',?,0,'completed',?,?,?)",
            (
                "capture-client",
                "unscoped-dataset",
                "c" * 64,
                "d" * 64,
                (NOW + timedelta(days=1)).isoformat(),
                timestamp,
                timestamp,
            ),
        )

    with pytest.raises(SemanticErasureError):
        await SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: pytest.fail("no semantic backend"),
            timeout_seconds=1,
        ).erase()

    assert db.note_store.get_note_by_id(NOTE_ID) is not None
    with db.transaction() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM note_semantic_operation_receipts WHERE owner_user_id='capture-client'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.asyncio
async def test_configless_receipt_only_state_fails_closed(db: CharactersRAGDB) -> None:
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
            "DELETE FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
            (OWNER_ID, DATASET_ID),
        )
    _insert_receipt(db, dataset_id=DATASET_ID)

    with pytest.raises(SemanticErasureError) as exc_info:
        await SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: pytest.fail("no physical binding"),
            timeout_seconds=1,
        ).erase()

    assert exc_info.value.code == "notes_semantic_erasure_cleanup_failed"
    assert db.note_store.get_note_by_id(NOTE_ID) is not None
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM note_semantic_operation_receipts").fetchone()[0] == 1


def test_final_owner_erasure_rejects_pending_cleanup(
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
        db.note_semantic_store.finalize_owner_erasure(
            dataset_ids=(DATASET_ID,),
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
    with db.transaction() as conn:
        work = conn.execute(
            "SELECT claim_state,claim_token FROM note_semantic_work WHERE "
            "owner_user_id=? AND dataset_id=? AND kind='delete_generation'",
            (OWNER_ID, DATASET_ID),
        ).fetchone()
    assert tuple(work) == ("pending", None)


@pytest.mark.asyncio
async def test_vector_backend_failure_is_bounded_and_retains_retry_identity(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)

    observations: list[tuple[str, str, str]] = []
    audits: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notes_Graph.semantic_erasure._metric",
        lambda *, status, backend, error_code: observations.append((status, backend, error_code)),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notes_Graph.semantic_erasure._audit",
        lambda *, owner_user_id, status, reason: audits.append((status, reason)),
    )

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
    assert observations == [("failed", "chromadb", "backend_unavailable")]
    assert audits == [("failed", "backend_unavailable")]
    with db.transaction() as conn:
        work = conn.execute(
            "SELECT claim_state,claim_token FROM note_semantic_work WHERE "
            "owner_user_id=? AND dataset_id=? AND kind='delete_generation'",
            (OWNER_ID, DATASET_ID),
        ).fetchone()
    assert tuple(work) == ("pending", None)

    retry = await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=lambda _backend: RecordingVectors([]),
        timeout_seconds=1,
    ).erase()
    assert retry.deleted_notes == 1


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
            "SELECT claim_state,claim_token FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (OWNER_ID, DATASET_ID, claimed.id),
        ).fetchone()
    assert tuple(row) == ("claimed", claimed.claim_token)
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_generation_cleanup_claims_one_item_at_a_time(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    _create_resolved_generation(db)
    limits: list[int] = []
    original = db.note_semantic_store.claim_generation_cleanup_batch

    def _claim(**kwargs):
        limits.append(kwargs["limit"])
        return original(**kwargs)

    monkeypatch.setattr(db.note_semantic_store, "claim_generation_cleanup_batch", _claim)
    await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=lambda _backend: RecordingVectors([]),
        timeout_seconds=1,
    ).erase()

    assert limits and set(limits) == {1}


@pytest.mark.asyncio
async def test_dsr_disable_before_vector_authorization_prevents_upsert(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    chunks, claim = _seed_and_claim_note(db, generation)
    fence = _publication_fence(generation)
    vectors = MemoryVectors([])
    staged = asyncio.Event()
    allow_upsert = asyncio.Event()
    side_effect_count = 0

    async def _pause_before_upsert() -> None:
        nonlocal side_effect_count
        side_effect_count += 1
        if side_effect_count == 2:
            staged.set()
            await allow_upsert.wait()

    async def _revalidate(current_fence: SemanticExecutionFence):
        return _publication_authority(db, current_fence)

    publication = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=_revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )

    async def _writer() -> None:
        try:
            await publication.publish_note(
                fence,
                claim,
                chunks,
                tuple(SemanticVector(chunk.vector_id, tuple(1.0 for _ in range(768))) for chunk in chunks),
                before_side_effect=_pause_before_upsert,
            )
        finally:
            db.note_semantic_store.release_work_claim(
                dataset_id=DATASET_ID,
                work_id=claim.id,
                claim_token=claim.claim_token or "",
                fencing_token=claim.fencing_token,
                now=NOW + timedelta(seconds=1),
            )

    writer = asyncio.create_task(_writer())
    await staged.wait()
    erasure = asyncio.create_task(
        SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: vectors,
            timeout_seconds=1,
            poll_interval_seconds=0.005,
            clock=lambda: NOW,
        ).erase()
    )
    try:
        for _ in range(100):
            config = db.note_semantic_store.get_configuration(DATASET_ID)
            if config is not None and config.desired_state.value == "disabled":
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("erasure did not install the disabled fence")
        with db.transaction() as conn:
            live = conn.execute(
                "SELECT claim_state,claim_token FROM note_semantic_work WHERE id=?",
                (claim.id,),
            ).fetchone()
        assert tuple(live) == ("claimed", claim.claim_token)
        assert (
            db.note_semantic_store.claim_generation_cleanup_batch(
                dataset_id=DATASET_ID,
                limit=1,
                now=NOW,
            )
            == ()
        )
        allow_upsert.set()
        with pytest.raises(SemanticIndexingError):
            await writer
        result = await erasure
    finally:
        allow_upsert.set()
        if not writer.done():
            await writer
        if not erasure.done():
            await erasure

    assert result.deleted_notes == 1
    assert vectors.values == {}
    assert f"upsert:{generation.id}" not in vectors.events
    assert f"delete_generation:{generation.id}" in vectors.events


@pytest.mark.asyncio
async def test_expired_prewrite_claim_cannot_publish_after_dsr_finalizes(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    chunks, claim = _seed_and_claim_note(db, generation)
    fence = _publication_fence(generation)
    vectors = MemoryVectors([])
    staged = asyncio.Event()
    allow_authorization = asyncio.Event()
    side_effect_count = 0

    async def _pause_before_authorization() -> None:
        nonlocal side_effect_count
        side_effect_count += 1
        if side_effect_count == 2:
            staged.set()
            await allow_authorization.wait()

    async def _revalidate(current_fence: SemanticExecutionFence):
        return _publication_authority(db, current_fence)

    publication = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=_revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )

    async def _writer() -> None:
        try:
            await publication.publish_note(
                fence,
                claim,
                chunks,
                tuple(SemanticVector(chunk.vector_id, tuple(1.0 for _ in range(768))) for chunk in chunks),
                before_side_effect=_pause_before_authorization,
            )
        finally:
            db.note_semantic_store.release_work_claim(
                dataset_id=DATASET_ID,
                work_id=claim.id,
                claim_token=claim.claim_token or "",
                fencing_token=claim.fencing_token,
                now=NOW + timedelta(seconds=3),
            )

    writer = asyncio.create_task(_writer())
    await staged.wait()
    reclaimed = db.note_semantic_store.reclaim_expired_dataset_work(
        dataset_id=DATASET_ID,
        expired_before=NOW + timedelta(seconds=1),
        limit=10,
        now=NOW + timedelta(seconds=2),
    )
    assert reclaimed == 1

    result = await SemanticErasureCoordinator(
        db=db,
        vector_store_factory=lambda _backend: vectors,
        timeout_seconds=1,
        poll_interval_seconds=0.005,
    ).erase()
    allow_authorization.set()
    with pytest.raises(SemanticIndexingError):
        await writer

    assert result.deleted_notes == 1
    assert vectors.values == {}
    assert f"upsert:{generation.id}" not in vectors.events


@pytest.mark.asyncio
async def test_cancelled_vector_upsert_drains_before_claim_release_and_erasure(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    chunks, claim = _seed_and_claim_note(db, generation)
    fence = _publication_fence(generation)
    entered = asyncio.Event()
    allow_upsert = asyncio.Event()

    class _PausingUpsertVectors(MemoryVectors):
        async def upsert(self, dataset_id: str, generation_id: str, vectors) -> int:
            entered.set()
            await allow_upsert.wait()
            return await super().upsert(dataset_id, generation_id, vectors)

    vectors = _PausingUpsertVectors([])

    async def _revalidate(current_fence: SemanticExecutionFence):
        return _publication_authority(db, current_fence)

    publication = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=_revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )

    async def _writer() -> None:
        try:
            await publication.publish_note(
                fence,
                claim,
                chunks,
                tuple(SemanticVector(chunk.vector_id, tuple(1.0 for _ in range(768))) for chunk in chunks),
            )
        finally:
            db.note_semantic_store.release_work_claim(
                dataset_id=DATASET_ID,
                work_id=claim.id,
                claim_token=claim.claim_token or "",
                fencing_token=claim.fencing_token,
                now=NOW + timedelta(seconds=1),
            )

    writer = asyncio.create_task(_writer())
    await entered.wait()
    writer.cancel()
    erasure = asyncio.create_task(
        SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: vectors,
            timeout_seconds=1,
            poll_interval_seconds=0.005,
        ).erase()
    )
    await asyncio.sleep(0.03)
    assert not writer.done()
    assert not erasure.done()

    allow_upsert.set()
    with pytest.raises(asyncio.CancelledError):
        await writer
    result = await erasure

    assert result.deleted_notes == 1
    assert vectors.values == {}
    assert f"upsert:{generation.id}" in vectors.events
    assert f"delete_generation:{generation.id}" in vectors.events


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("update", "tombstone"))
async def test_note_mutation_cannot_remove_authorized_vector_write_fence(
    db: CharactersRAGDB,
    mutation: str,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    generation = _create_resolved_generation(db)
    chunks, claim = _seed_and_claim_note(db, generation)
    fence = _publication_fence(generation)
    entered = asyncio.Event()
    allow_upsert = asyncio.Event()

    class _PausingUpsertVectors(MemoryVectors):
        async def upsert(self, dataset_id: str, generation_id: str, vectors) -> int:
            entered.set()
            await allow_upsert.wait()
            return await super().upsert(dataset_id, generation_id, vectors)

    vectors = _PausingUpsertVectors([])

    async def _revalidate(current_fence: SemanticExecutionFence):
        return _publication_authority(db, current_fence)

    publication = SemanticPublicationService(
        store=db.note_semantic_store,
        vectors=vectors,
        revalidate=_revalidate,
        clock=lambda: NOW,
        receipt_factory=lambda: "unused",
    )

    async def _writer() -> None:
        try:
            await publication.publish_note(
                fence,
                claim,
                chunks,
                tuple(SemanticVector(chunk.vector_id, tuple(1.0 for _ in range(768))) for chunk in chunks),
            )
        finally:
            db.note_semantic_store.release_work_claim(
                dataset_id=DATASET_ID,
                work_id=claim.id,
                claim_token=claim.claim_token or "",
                fencing_token=claim.fencing_token,
                now=NOW + timedelta(seconds=1),
            )

    writer = asyncio.create_task(_writer())
    await entered.wait()
    if mutation == "update":
        assert db.note_store.update_note(
            NOTE_ID,
            {"title": "Revised"},
            expected_version=1,
            semantic_dataset_id=DATASET_ID,
        )
    else:
        assert db.note_store.soft_delete_note(
            NOTE_ID,
            expected_version=1,
            semantic_dataset_id=DATASET_ID,
        )
    with db.transaction() as conn:
        active = conn.execute(
            "SELECT claim_state,claim_token,error_code FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (OWNER_ID, DATASET_ID, claim.id),
        ).fetchone()
    assert tuple(active) == (
        "claimed",
        claim.claim_token,
        "vector_side_effect_in_progress",
    )

    erasure = asyncio.create_task(
        SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: vectors,
            timeout_seconds=1,
            poll_interval_seconds=0.005,
            clock=lambda: NOW,
        ).erase()
    )
    await asyncio.sleep(0.03)
    assert not erasure.done()
    allow_upsert.set()
    with pytest.raises(SemanticIndexingError):
        await writer
    result = await erasure

    assert result.deleted_notes == 1
    assert vectors.values == {}


@pytest.mark.asyncio
async def test_quiescent_operation_propagates_child_cancellation() -> None:
    async def _cancel_self() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_quiescent_operation(_cancel_self())


@pytest.mark.asyncio
async def test_committed_owner_finalizer_wins_over_parent_cancellation(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    entered = threading.Event()
    allow_finalize = threading.Event()
    original = db.note_semantic_store.finalize_owner_erasure

    def _finalize(**kwargs):
        entered.set()
        assert allow_finalize.wait(timeout=1)
        return original(**kwargs)

    monkeypatch.setattr(db.note_semantic_store, "finalize_owner_erasure", _finalize)
    task = asyncio.create_task(SemanticErasureCoordinator(db=db, timeout_seconds=1).erase())
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    allow_finalize.set()

    result = await task

    assert result.deleted_notes == 1
    assert db.note_store.get_note_by_id(NOTE_ID) is None


@pytest.mark.asyncio
async def test_timeout_waits_for_backend_quiescence_before_returning(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    _create_resolved_generation(db)

    class _SlowVectors(RecordingVectors):
        def __init__(self) -> None:
            super().__init__([])
            self.finished = False

        async def delete_generation(self, dataset_id: str, generation_id: str):
            await asyncio.sleep(0.25)
            self.finished = True
            return await super().delete_generation(dataset_id, generation_id)

    vectors = _SlowVectors()
    started = time.monotonic()
    returned_finished = False
    try:
        with pytest.raises(SemanticErasureError) as exc_info:
            await SemanticErasureCoordinator(
                db=db,
                vector_store_factory=lambda _backend: vectors,
                timeout_seconds=0.2,
            ).erase()
        returned_finished = vectors.finished
    finally:
        await asyncio.sleep(0.26)

    assert exc_info.value.code == "notes_semantic_erasure_timeout"
    assert returned_finished is True
    assert time.monotonic() - started >= 0.25
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_cancellation_waits_for_store_authorization_before_releasing_claim(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    _create_resolved_generation(db)
    entered = threading.Event()
    allow = threading.Event()
    original = db.note_semantic_store.authorize_generation_cleanup

    def _pausing_authorize(**kwargs):
        entered.set()
        assert allow.wait(timeout=1)
        return original(**kwargs)

    monkeypatch.setattr(
        db.note_semantic_store,
        "authorize_generation_cleanup",
        _pausing_authorize,
    )
    task = asyncio.create_task(
        SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: RecordingVectors([]),
            timeout_seconds=1,
        ).erase()
    )
    while not entered.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    with db.transaction() as conn:
        work = conn.execute(
            "SELECT claim_state,claim_token FROM note_semantic_work WHERE "
            "owner_user_id=? AND dataset_id=? AND kind='delete_generation'",
            (OWNER_ID, DATASET_ID),
        ).fetchone()
    assert tuple(work) == ("pending", None)
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_cancellation_waits_for_backend_and_releases_exact_cleanup_claim(
    db: CharactersRAGDB,
) -> None:
    db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
    _create_resolved_generation(db)
    entered = asyncio.Event()
    allow = asyncio.Event()

    class _PausingVectors(RecordingVectors):
        async def delete_generation(self, dataset_id: str, generation_id: str):
            entered.set()
            await allow.wait()
            return await super().delete_generation(dataset_id, generation_id)

    task = asyncio.create_task(
        SemanticErasureCoordinator(
            db=db,
            vector_store_factory=lambda _backend: _PausingVectors([]),
            timeout_seconds=1,
        ).erase()
    )
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    with db.transaction() as conn:
        work = conn.execute(
            "SELECT claim_state,claim_token FROM note_semantic_work WHERE "
            "owner_user_id=? AND dataset_id=? AND kind='delete_generation'",
            (OWNER_ID, DATASET_ID),
        ).fetchone()
    assert tuple(work) == ("pending", None)
    assert db.note_store.get_note_by_id(NOTE_ID) is not None


@pytest.mark.asyncio
async def test_postgres_finalizer_deletes_only_owner_and_keeps_shared_pool_alive(
    tmp_path: Path,
    pg_database_config,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner_db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    foreign_db = CharactersRAGDB(":memory:", client_id="owner-b", backend=backend)
    owner_note_2 = "22222222-2222-4222-8222-222222222222"
    foreign_note = "33333333-3333-4333-8333-333333333333"
    try:
        assert owner_db.backend_type == BackendType.POSTGRESQL
        owner_db.note_store.add_note("Private", "Body", note_id=NOTE_ID)
        owner_db.note_store.add_note("Private 2", "Body", note_id=owner_note_2)
        owner_db.create_manual_note_edge(
            user_id=OWNER_ID,
            from_note_id=NOTE_ID,
            to_note_id=owner_note_2,
            created_by=OWNER_ID,
        )
        foreign_db.note_store.add_note("Foreign", "Body", note_id=foreign_note)
        _create_resolved_generation(owner_db)

        result = await SemanticErasureCoordinator(
            db=owner_db,
            vector_store_factory=lambda _backend: RecordingVectors([]),
            timeout_seconds=2,
            close_database_on_exit=True,
        ).erase()

        assert result.deleted_notes == 2
        assert foreign_db.note_store.get_note_by_id(foreign_note) is not None
        assert foreign_db.note_store.add_note(
            "Foreign after erasure",
            "Body",
            note_id="44444444-4444-4444-8444-444444444444",
        )
    finally:
        owner_db.close_connection()
        foreign_db.close_connection()
        backend.get_pool().close_all()


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
