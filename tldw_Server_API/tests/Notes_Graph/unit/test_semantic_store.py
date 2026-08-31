"""Behavioral contracts for the owner-scoped Notes semantic persistence store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGenerationState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
DATASET_ID = "dataset-a"
CONTENT_V1 = f"sha256:{'1' * 64}"
CONTENT_V7 = f"sha256:{'7' * 64}"
CONTENT_V8 = f"sha256:{'8' * 64}"


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


def _create_resolved_generation(
    db: CharactersRAGDB,
    *,
    root_job_id: str = "job-1",
    dimensions: int = 768,
    compatibility_hash: str = "compatibility-v1",
):
    config = _create_config(db)
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    pending = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id=root_job_id,
        now=NOW,
    )
    generation = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=dimensions,
        compatibility_hash=compatibility_hash,
        now=NOW,
    )
    assert generation is not None
    resolved_config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert resolved_config is not None
    return resolved_config, generation


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
    resolved_config, first = _create_resolved_generation(db)
    switched = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=first.id,
        expected_configuration_revision=resolved_config.configuration_revision,
        publication_receipt="receipt-1",
        now=NOW,
    )
    assert switched is not None
    assert switched.semantic_index_revision == 1
    assert db.note_semantic_store.get_generation(DATASET_ID, first.id).state is SemanticGenerationState.ACTIVE


def test_activation_rejects_generation_from_stale_configuration(db: CharactersRAGDB) -> None:
    resolved_config, stale = _create_resolved_generation(db, root_job_id="job-stale")
    disabled = db.note_semantic_store.disable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=resolved_config.configuration_revision,
        now=NOW,
    )
    assert disabled is not None
    reenabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=disabled.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert reenabled is not None

    assert db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=stale.id,
        expected_configuration_revision=reenabled.configuration_revision,
        publication_receipt="receipt-stale",
        now=NOW,
    ) is None
    assert db.note_semantic_store.get_generation(DATASET_ID, stale.id).state is SemanticGenerationState.STAGING
    assert db.note_semantic_store.get_configuration(DATASET_ID).active_generation_id is None


def test_replacement_activation_retires_and_queues_old_generation_cleanup(
    db: CharactersRAGDB,
) -> None:
    resolved_config, first = _create_resolved_generation(db)
    active = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=first.id,
        expected_configuration_revision=resolved_config.configuration_revision,
        publication_receipt="receipt-1",
        now=NOW,
    )
    assert active is not None
    replacement = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=active.configuration_revision,
        compatibility_hash="compatibility-v1",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=768,
        root_job_id="job-2",
        now=NOW,
    )

    switched = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=replacement.id,
        expected_configuration_revision=active.configuration_revision,
        publication_receipt="receipt-2",
        now=NOW,
    )

    assert switched is not None
    assert db.note_semantic_store.get_generation(DATASET_ID, first.id).state is SemanticGenerationState.RETIRED
    assert db.note_semantic_store.get_generation(DATASET_ID, replacement.id).state is SemanticGenerationState.ACTIVE
    with db.transaction() as conn:
        cleanup = conn.execute(
            "SELECT kind,note_id,generation_id,claim_state,attempt_count FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND kind='delete_generation'",
            ("owner-a", DATASET_ID),
        ).fetchall()
    assert [tuple(row) for row in cleanup] == [
        ("delete_generation", None, first.id, "pending", 0)
    ]


def test_pending_dimension_resolution_updates_config_and_generation_by_cas(
    db: CharactersRAGDB,
) -> None:
    config = _create_config(db)
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
        root_job_id="job-probe",
        now=NOW,
    )
    assert generation.compatibility_hash is None

    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=1536,
        compatibility_hash="compatibility-resolved",
        now=NOW,
    )

    assert resolved is not None
    assert resolved.dimension_state is SemanticDimensionState.RESOLVED
    assert resolved.dimensions == 1536
    assert resolved.compatibility_hash == "compatibility-resolved"
    assert resolved.configuration_revision == enabled.configuration_revision + 1
    updated_config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert updated_config.dimension_state is SemanticDimensionState.RESOLVED
    assert updated_config.dimensions == 1536
    assert updated_config.compatibility_hash == "compatibility-resolved"
    assert updated_config.configuration_revision == resolved.configuration_revision


def test_dimension_resolution_rejects_stale_config_or_generation_without_partial_update(
    db: CharactersRAGDB,
) -> None:
    config = _create_config(db)
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
        root_job_id="job-probe",
        now=NOW,
    )

    assert db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision + 1,
        dimensions=1536,
        compatibility_hash="compatibility-resolved",
        now=NOW,
    ) is None
    assert db.note_semantic_store.get_configuration(DATASET_ID).dimension_state is SemanticDimensionState.PENDING
    assert db.note_semantic_store.get_generation(DATASET_ID, generation.id).dimension_state is SemanticDimensionState.PENDING

    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_semantic_generations SET state='failed' "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            ("owner-a", DATASET_ID, generation.id),
        )
    assert db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=1536,
        compatibility_hash="compatibility-resolved",
        now=NOW,
    ) is None
    unchanged_config = db.note_semantic_store.get_configuration(DATASET_ID)
    unchanged_generation = db.note_semantic_store.get_generation(DATASET_ID, generation.id)
    assert unchanged_config.dimension_state is SemanticDimensionState.PENDING
    assert unchanged_config.compatibility_hash is None
    assert unchanged_generation.dimension_state is SemanticDimensionState.PENDING
    assert unchanged_generation.compatibility_hash is None


def test_pending_generation_rejects_final_compatibility_hash(db: CharactersRAGDB) -> None:
    config = _create_config(db)
    with pytest.raises(ValueError, match="notes_semantic_pending_compatibility_hash_invalid"):
        db.note_semantic_store.create_generation(
            dataset_id=DATASET_ID,
            configuration_revision=config.configuration_revision,
            compatibility_hash="compatibility-too-early",
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="job-probe",
            now=NOW,
        )


def test_pending_configuration_rejects_direct_resolved_generation(db: CharactersRAGDB) -> None:
    config = _create_config(db)

    with pytest.raises(ValueError, match="notes_semantic_generation_identity_mismatch"):
        db.note_semantic_store.create_generation(
            dataset_id=DATASET_ID,
            configuration_revision=config.configuration_revision,
            compatibility_hash="compatibility-v1",
            dimension_state=SemanticDimensionState.RESOLVED,
            dimensions=768,
            root_job_id="job-bypass",
            now=NOW,
        )


@pytest.mark.parametrize(
    ("dimension_state", "dimensions", "compatibility_hash"),
    (
        (SemanticDimensionState.PENDING, None, None),
        (SemanticDimensionState.RESOLVED, 384, "compatibility-v1"),
        (SemanticDimensionState.RESOLVED, 768, "compatibility-v2"),
    ),
)
def test_resolved_configuration_rejects_generation_identity_mismatch(
    db: CharactersRAGDB,
    dimension_state: SemanticDimensionState,
    dimensions: int | None,
    compatibility_hash: str | None,
) -> None:
    resolved_config, generation = _create_resolved_generation(db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_semantic_generations SET state='failed' "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            ("owner-a", DATASET_ID, generation.id),
        )

    with pytest.raises(ValueError, match="notes_semantic_generation_identity_mismatch"):
        db.note_semantic_store.create_generation(
            dataset_id=DATASET_ID,
            configuration_revision=resolved_config.configuration_revision,
            compatibility_hash=compatibility_hash,
            dimension_state=dimension_state,
            dimensions=dimensions,
            root_job_id="job-mismatch",
            now=NOW,
        )


def test_activation_rejects_resolved_generation_while_configuration_is_pending(
    db: CharactersRAGDB,
) -> None:
    config = _create_config(db)
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO note_semantic_generations(
                id,owner_user_id,dataset_id,configuration_revision,state,
                compatibility_hash,dimension_state,dimensions,created_at
            ) VALUES ('generation-bypass','owner-a',?,?, 'staging',
                      'compatibility-v1','resolved',768,CURRENT_TIMESTAMP)
            """,
            (DATASET_ID, enabled.configuration_revision),
        )

    assert db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id="generation-bypass",
        expected_configuration_revision=enabled.configuration_revision,
        publication_receipt="receipt-bypass",
        now=NOW,
    ) is None
    unchanged_config = db.note_semantic_store.get_configuration(DATASET_ID)
    unchanged_generation = db.note_semantic_store.get_generation(DATASET_ID, "generation-bypass")
    assert unchanged_config.configuration_revision == enabled.configuration_revision
    assert unchanged_config.semantic_index_revision == 0
    assert unchanged_config.active_generation_id is None
    assert unchanged_generation.state is SemanticGenerationState.STAGING
    with db.transaction() as conn:
        cleanup_count = conn.execute(
            "SELECT COUNT(*) FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=?",
            ("owner-a", DATASET_ID),
        ).fetchone()[0]
    assert cleanup_count == 0


@pytest.mark.parametrize(
    ("dimensions", "compatibility_hash"),
    (
        (384, "compatibility-v1"),
        (768, "compatibility-v2"),
    ),
)
def test_activation_rejects_generation_identity_mismatch_without_side_effects(
    db: CharactersRAGDB,
    dimensions: int,
    compatibility_hash: str,
) -> None:
    resolved_config, first = _create_resolved_generation(db)
    active = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=first.id,
        expected_configuration_revision=resolved_config.configuration_revision,
        publication_receipt="receipt-1",
        now=NOW,
    )
    assert active is not None
    replacement = db.note_semantic_store.create_generation(
        dataset_id=DATASET_ID,
        configuration_revision=active.configuration_revision,
        compatibility_hash="compatibility-v1",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=768,
        root_job_id="job-2",
        now=NOW,
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_semantic_generations SET dimensions=?, compatibility_hash=? "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (dimensions, compatibility_hash, "owner-a", DATASET_ID, replacement.id),
        )

    assert db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=replacement.id,
        expected_configuration_revision=active.configuration_revision,
        publication_receipt="receipt-2",
        now=NOW,
    ) is None
    unchanged_config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert unchanged_config.configuration_revision == active.configuration_revision
    assert unchanged_config.semantic_index_revision == active.semantic_index_revision
    assert unchanged_config.active_generation_id == first.id
    assert db.note_semantic_store.get_generation(DATASET_ID, first.id).state is SemanticGenerationState.ACTIVE
    assert db.note_semantic_store.get_generation(DATASET_ID, replacement.id).state is SemanticGenerationState.STAGING
    with db.transaction() as conn:
        cleanup_count = conn.execute(
            "SELECT COUNT(*) FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
            "AND kind='delete_generation'",
            ("owner-a", DATASET_ID),
        ).fetchone()[0]
    assert cleanup_count == 0


def test_manifest_publication_cannot_clear_a_newer_dirty_generation(db: CharactersRAGDB) -> None:
    _config, generation = _create_resolved_generation(db)
    db.add_note("Note A", "content", note_id="note-a")
    dirty = db.note_semantic_store.record_note_dirty(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=7,
        content_fingerprint=CONTENT_V7,
        now=NOW,
    )
    claimed = db.note_semantic_store.claim_dirty_note(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        dirty_generation=dirty.dirty_generation,
        now=NOW,
    )
    assert claimed is not None
    db.note_semantic_store.record_note_dirty(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=8,
        content_fingerprint=CONTENT_V8,
        now=NOW,
    )

    published = db.note_semantic_store.publish_note_manifest(
        owner_user_id="owner-a",
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        claimed_dirty_generation=dirty.dirty_generation,
        content_version=7,
        manifest={"chunk_count": 1, "manifest_hash": "manifest-v7"},
        now=NOW,
    )
    assert published is False


def test_tombstones_queue_coalesced_cleanup_with_bounded_retry(db: CharactersRAGDB) -> None:
    _config, generation = _create_resolved_generation(db)
    db.add_note("Note A", "content", note_id="note-a")
    dirty = db.note_semantic_store.record_note_dirty(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=1,
        content_fingerprint=CONTENT_V1,
        now=NOW,
    )
    tombstoned = db.note_semantic_store.tombstone_note(
        dataset_id=DATASET_ID,
        generation_id=generation.id,
        note_id="note-a",
        content_version=2,
        dirty_generation=dirty.dirty_generation + 1,
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


@pytest.mark.parametrize(
    "content_fingerprint",
    (
        "raw Note text must not be persisted",
        f"sha256:{'A' * 64}",
        "1" * 64,
    ),
)
def test_store_rejects_noncanonical_content_fingerprints(
    db: CharactersRAGDB,
    content_fingerprint: str,
) -> None:
    _config, generation = _create_resolved_generation(db)
    db.add_note("Note A", "content", note_id="note-a")

    with pytest.raises(ValueError, match="notes_semantic_content_fingerprint_invalid"):
        db.note_semantic_store.record_note_dirty(
            dataset_id=DATASET_ID,
            generation_id=generation.id,
            note_id="note-a",
            content_version=1,
            content_fingerprint=content_fingerprint,
            now=NOW,
        )


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
