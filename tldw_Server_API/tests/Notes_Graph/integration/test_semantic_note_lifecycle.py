"""Canonical NoteStore lifecycle fences for the semantic persistence ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_content import semantic_content_fingerprint

pytestmark = pytest.mark.integration

DATASET_ID = "dataset-a"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "semantic-lifecycle.sqlite"), client_id="owner-a")
    yield database
    database.close_all_connections()


def _activate_semantic_generation(db: CharactersRAGDB) -> str:
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
        storage_label="local semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
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
        root_job_id="job-1",
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=768,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    active = db.note_semantic_store.activate_generation(
        dataset_id=DATASET_ID,
        generation_id=pending.id,
        expected_configuration_revision=resolved.configuration_revision,
        publication_receipt="receipt-1",
        now=NOW,
    )
    assert active is not None
    return pending.id


def _state(db: CharactersRAGDB):
    with db.transaction() as conn:
        return conn.execute(
            "SELECT content_version,content_fingerprint,dirty_generation,state "
            "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND note_id=?",
            ("owner-a", DATASET_ID, NOTE_ID),
        ).fetchone()


def _work(db: CharactersRAGDB):
    with db.transaction() as conn:
        return conn.execute(
            "SELECT kind,generation_id,dirty_generation FROM note_semantic_work "
            "WHERE owner_user_id=? AND dataset_id=? AND note_id=? ORDER BY kind",
            ("owner-a", DATASET_ID, NOTE_ID),
        ).fetchall()


def test_enabled_note_lifecycle_records_transactional_semantic_work(db: CharactersRAGDB) -> None:
    generation_id = _activate_semantic_generation(db)

    db.note_store.add_note(
        "Original", "Body", note_id=NOTE_ID, semantic_dataset_id=DATASET_ID
    )
    assert tuple(_state(db)) == (
        1,
        semantic_content_fingerprint("Original", "Body", 1),
        1,
        "pending",
    )
    assert [tuple(row) for row in _work(db)] == [("index_note", generation_id, 1)]

    assert db.note_store.update_note(
        NOTE_ID,
        {"title": "Revised"},
        expected_version=1,
        semantic_dataset_id=DATASET_ID,
    )
    assert tuple(_state(db)) == (
        2,
        semantic_content_fingerprint("Revised", "Body", 2),
        2,
        "pending",
    )
    assert [tuple(row) for row in _work(db)] == [("index_note", generation_id, 2)]

    assert db.note_store.soft_delete_note(
        NOTE_ID, expected_version=2, semantic_dataset_id=DATASET_ID
    )
    assert tuple(_state(db)) == (
        3,
        semantic_content_fingerprint("Revised", "Body", 3),
        3,
        "tombstoned",
    )
    assert [tuple(row) for row in _work(db)] == [
        ("delete_note_vectors", generation_id, 3)
    ]

    assert db.note_store.restore_note(
        NOTE_ID, expected_version=3, semantic_dataset_id=DATASET_ID
    )
    assert tuple(_state(db)) == (
        4,
        semantic_content_fingerprint("Revised", "Body", 4),
        4,
        "pending",
    )
    assert [tuple(row) for row in _work(db)] == [("index_note", generation_id, 4)]

    assert db.note_store.delete_note(
        NOTE_ID,
        hard_delete=True,
        semantic_dataset_id=DATASET_ID,
    )
    assert _state(db) is None
    assert [tuple(row) for row in _work(db)] == [
        ("delete_note_vectors", generation_id, 5)
    ]
    assert db.note_semantic_store.get_configuration(DATASET_ID).semantic_index_revision == 3


def test_semantic_lifecycle_ignores_relationship_only_updates_and_rolls_back_with_note(
    db: CharactersRAGDB,
) -> None:
    _activate_semantic_generation(db)
    db.note_store.add_note(
        "Original", "Body", note_id=NOTE_ID, semantic_dataset_id=DATASET_ID
    )
    conversation_id = db.add_conversation({"id": "conversation-1", "title": "Source"})

    assert db.note_store.update_note(
        NOTE_ID,
        {"conversation_id": conversation_id},
        expected_version=1,
        semantic_dataset_id=DATASET_ID,
    )
    assert tuple(_state(db)) == (
        1,
        semantic_content_fingerprint("Original", "Body", 1),
        1,
        "pending",
    )

    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as conn:
            db.note_store.add_note(
                "Rolled back",
                "Body",
                note_id="22222222-2222-4222-8222-222222222222",
                semantic_dataset_id=DATASET_ID,
                conn=conn,
            )
            raise RuntimeError("rollback")

    assert db.get_note_by_id("22222222-2222-4222-8222-222222222222") is None
    with db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=?",
            ("owner-a", DATASET_ID),
        ).fetchone()[0] == 1


def test_disabled_or_unresolved_dataset_creates_no_semantic_state(db: CharactersRAGDB) -> None:
    db.note_store.add_note(
        "Local only", "Body", note_id=NOTE_ID, semantic_dataset_id=DATASET_ID
    )
    assert _state(db) is None
    assert _work(db) == []


def test_explicitly_disabled_active_generation_creates_no_semantic_work(
    db: CharactersRAGDB,
) -> None:
    generation_id = _activate_semantic_generation(db)
    enabled = db.note_semantic_store.get_configuration(DATASET_ID)
    assert enabled is not None
    disabled = db.note_semantic_store.disable_configuration(
        dataset_id=DATASET_ID,
        expected_configuration_revision=enabled.configuration_revision,
        now=NOW,
    )
    assert disabled is not None
    assert disabled.active_generation_id == generation_id
    baseline_semantic_revision = disabled.semantic_index_revision

    db.note_store.add_note(
        "Local only", "Body", note_id=NOTE_ID, semantic_dataset_id=DATASET_ID
    )
    assert db.note_store.update_note(
        NOTE_ID,
        {"title": "Still local"},
        expected_version=1,
        semantic_dataset_id=DATASET_ID,
    )
    assert db.note_store.soft_delete_note(
        NOTE_ID,
        expected_version=2,
        semantic_dataset_id=DATASET_ID,
    )
    assert db.note_store.restore_note(
        NOTE_ID,
        expected_version=3,
        semantic_dataset_id=DATASET_ID,
    )
    assert db.note_store.delete_note(
        NOTE_ID,
        hard_delete=True,
        semantic_dataset_id=DATASET_ID,
    )

    assert _state(db) is None
    assert _work(db) == []
    final_config = db.note_semantic_store.get_configuration(DATASET_ID)
    assert final_config is not None
    assert final_config.desired_state.value == "disabled"
    assert final_config.semantic_index_revision == baseline_semantic_revision
