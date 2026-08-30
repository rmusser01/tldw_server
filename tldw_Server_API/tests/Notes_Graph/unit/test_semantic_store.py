"""Behavioral contracts for the owner-scoped Notes semantic persistence store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGenerationState,
)


pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
DATASET_ID = "dataset-a"


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "semantic-store.sqlite"), client_id="owner-a")
    yield database
    database.close_all_connections()


def _create_config(db: CharactersRAGDB):
    return db.note_semantic_store.create_configuration(
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
        storage_label="local semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )


def test_enable_disable_are_capability_and_revision_fenced(db: CharactersRAGDB) -> None:
    created = _create_config(db)

    assert db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=created.configuration_revision,
        capability_revision="different-capability",
        now=NOW,
    ) is None

    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=created.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    assert enabled.desired_state.value == "enabled"

    assert db.note_semantic_store.disable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=created.configuration_revision,
        now=NOW,
    ) is None
    disabled = db.note_semantic_store.disable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=enabled.configuration_revision,
        now=NOW,
    )
    assert disabled is not None
    assert disabled.desired_state.value == "disabled"


def test_switching_active_generation_increments_semantic_index_revision(db: CharactersRAGDB) -> None:
    config = _create_config(db)
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    first = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash="compatibility-v1",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=768,
        root_job_id="job-1",
        now=NOW,
    )
    switched = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=first.id,
        expected_configuration_revision=enabled.configuration_revision,
        publication_receipt="receipt-1",
        now=NOW,
    )
    assert switched is not None
    assert switched.semantic_index_revision == 1
    assert db.note_semantic_store.get_generation(DATASET_ID, first.id).state is SemanticGenerationState.ACTIVE


def test_manifest_publication_cannot_clear_a_newer_dirty_generation(db: CharactersRAGDB) -> None:
    config = _create_config(db)
    db.add_note("Note A", "content", note_id="note-a")
    generation = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=config.configuration_revision,
        compatibility_hash="compatibility-v1",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=768,
        root_job_id="job-1",
        now=NOW,
    )
    db.note_semantic_store.record_note_dirty(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=7,
        content_fingerprint="fingerprint-v7",
        now=NOW,
    )
    claimed = db.note_semantic_store.claim_dirty_note(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        dirty_generation=1,
        now=NOW,
    )
    assert claimed is not None
    db.note_semantic_store.record_note_dirty(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=8,
        content_fingerprint="fingerprint-v8",
        now=NOW,
    )

    published = db.note_semantic_store.publish_note_manifest(
        owner_user_id="owner-a",
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        claimed_dirty_generation=1,
        content_version=7,
        manifest={"chunk_count": 1, "manifest_hash": "manifest-v7"},
        now=NOW,
    )
    assert published is False


def test_tombstones_queue_coalesced_cleanup_with_bounded_retry(db: CharactersRAGDB) -> None:
    config = _create_config(db)
    db.add_note("Note A", "content", note_id="note-a")
    generation = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=config.configuration_revision,
        compatibility_hash="compatibility-v1",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=768,
        root_job_id="job-1",
        now=NOW,
    )
    db.note_semantic_store.record_note_dirty(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=1,
        content_fingerprint="fingerprint-v1",
        now=NOW,
    )
    tombstoned = db.note_semantic_store.tombstone_note(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=2,
        dirty_generation=2,
        now=NOW,
    )
    assert tombstoned is not None
    work = db.note_semantic_store.claim_work(dataset_id=DATASET_ID, now=NOW)
    assert work is not None
    assert work.kind.value == "delete_note_vectors"
    retried = db.note_semantic_store.retry_work(
        dataset_id=DATASET_ID,
        work_id=work.id,
        expected_claim_token=work.claim_token,
        error_code="provider_unavailable",
        retry_at=NOW + timedelta(minutes=1),
        now=NOW,
    )
    assert retried is not None
    assert retried.next_eligible_at == (NOW + timedelta(minutes=1)).isoformat()
    assert db.note_semantic_store.claim_work(dataset_id=DATASET_ID, now=NOW) is None


def test_owner_bound_store_hides_foreign_owner_rows(db: CharactersRAGDB) -> None:
    _create_config(db)
    foreign = CharactersRAGDB(db.db_path_str, client_id="owner-b")
    try:
        assert foreign.note_semantic_store.get_configuration(DATASET_ID) is None
    finally:
        foreign.close_all_connections()


def test_store_rejects_unsanitized_displays_and_error_codes(db: CharactersRAGDB) -> None:
    with pytest.raises(ValueError, match="notes_semantic_endpoint_origin_display_invalid"):
        db.note_semantic_store.create_configuration(
            dataset_id=DATASET_ID,
            capability_revision="capability-v1",
            disclosure_hash="disclosure-v1",
            provider="provider-a",
            model="model-a",
            endpoint_origin_revision="origin-v1",
            endpoint_origin_display="https://user:secret@example.test/path?token=secret",
            data_boundary="provider",
            vector_backend="chromadb",
            storage_boundary="server_local",
            storage_label="local semantic vectors",
            normalization_version="normalization-v1",
            chunker_version="chunker-v1",
            now=NOW,
        )
