"""Tests for managed Notes task projection markers and convergence."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import (
    TaskMarker,
    render_task_marker,
    task_marker_hash,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

TASK_ID = "11111111-1111-4111-8111-11111111111a"
TASK_HASH = "sha256:" + "a" * 64
NOTE_HASH = "sha256:" + "b" * 64


def test_managed_marker_round_trip() -> None:
    """A rendered marker preserves the exact canonical task base."""
    markers = import_module(
        "tldw_Server_API.app.core.Notes_Tasks.projection_markers"
    )

    marker = markers.render_task_marker(
        TASK_ID,
        revision=7,
        object_hash=TASK_HASH,
    )

    assert markers.parse_task_marker(marker) == markers.TaskMarker(
        task_id=TASK_ID,
        revision=7,
        object_hash=TASK_HASH,
    )


@pytest.mark.parametrize(
    ("task_id", "revision", "object_hash"),
    [
        (TASK_ID.upper(), 7, TASK_HASH),
        ("11111111-1111-1111-8111-111111111111", 7, TASK_HASH),
        (TASK_ID, 0, TASK_HASH),
        (TASK_ID, True, TASK_HASH),
        (TASK_ID, 7, TASK_HASH.upper()),
        (TASK_ID, 7, "a" * 63),
    ],
)
def test_managed_marker_render_rejects_noncanonical_values(
    task_id: str,
    revision: int,
    object_hash: str,
) -> None:
    """Only exact canonical task bases can become durable markers."""
    markers = import_module(
        "tldw_Server_API.app.core.Notes_Tasks.projection_markers"
    )

    with pytest.raises(ValueError):
        markers.render_task_marker(
            task_id,
            revision=revision,
            object_hash=object_hash,
        )


def test_extract_managed_marker_separates_visible_body_and_base() -> None:
    """A managed suffix is excluded from visible checklist task text."""
    markers = import_module(
        "tldw_Server_API.app.core.Notes_Tasks.projection_markers"
    )
    marker = markers.render_task_marker(
        TASK_ID,
        revision=7,
        object_hash=TASK_HASH,
    )

    result = markers.extract_task_marker(f"Review source {marker}")

    assert result == markers.TaskMarkerParseResult(
        body="Review source",
        marker=markers.TaskMarker(TASK_ID, 7, TASK_HASH),
        reason_code=None,
    )


def test_extract_managed_marker_reports_malformed_without_claiming_identity() -> None:
    """A malformed protected-looking suffix becomes reviewable drift."""
    markers = import_module(
        "tldw_Server_API.app.core.Notes_Tasks.projection_markers"
    )

    result = markers.extract_task_marker(
        "Review source <!-- tldw-task:v1:not-a-task:7:not-a-hash -->"
    )

    assert result == markers.TaskMarkerParseResult(
        body="Review source",
        marker=None,
        reason_code="malformed_marker",
    )


def test_extract_managed_marker_reports_duplicate_without_claiming_identity() -> None:
    """Multiple protected markers never select an arbitrary task identity."""
    markers = import_module(
        "tldw_Server_API.app.core.Notes_Tasks.projection_markers"
    )
    marker = markers.render_task_marker(
        TASK_ID,
        revision=7,
        object_hash=TASK_HASH,
    )

    result = markers.extract_task_marker(f"Review source {marker} {marker}")

    assert result == markers.TaskMarkerParseResult(
        body="Review source",
        marker=None,
        reason_code="duplicate_marker",
    )


def _projection_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "projection_version": 1,
        "task_id": TASK_ID,
        "task_envelope_id": "task-envelope-7",
        "task_revision": 7,
        "task_hash": TASK_HASH,
        "note_envelope_id": "note-envelope-9",
        "note_hash": NOTE_HASH,
        "linked": True,
        "marker_hash": "sha256:" + "c" * 64,
    }
    metadata.update(overrides)
    return metadata


def test_projection_group_metadata_is_closed_and_privacy_safe() -> None:
    """Durable group anchors contain only exact opaque projection evidence."""
    coordinator = import_module(
        "tldw_Server_API.app.core.Sync.v2.notes_task_coordinator"
    )

    anchor = coordinator._validate_task_projection_group_metadata(  # noqa: SLF001
        _projection_metadata()
    )

    assert anchor.task_id == TASK_ID
    assert anchor.task_revision == 7
    assert anchor.task_hash == TASK_HASH
    assert anchor.linked is True
    assert not hasattr(anchor, "markdown")
    assert not hasattr(anchor, "title")


@pytest.mark.parametrize(
    "overrides",
    [
        {"projection_version": 2},
        {"task_id": TASK_ID.upper()},
        {"task_envelope_id": ""},
        {"task_revision": True},
        {"task_hash": TASK_HASH.upper()},
        {"note_envelope_id": "x" * 129},
        {"linked": 1},
        {"marker_hash": None},
        {"markdown": "- [ ] secret"},
    ],
)
def test_projection_group_metadata_rejects_malformed_or_extra_fields(
    overrides: dict[str, object],
) -> None:
    """Malformed, oversized, or content-bearing anchor metadata fails closed."""
    coordinator = import_module(
        "tldw_Server_API.app.core.Sync.v2.notes_task_coordinator"
    )

    with pytest.raises(ValueError):
        coordinator._validate_task_projection_group_metadata(  # noqa: SLF001
            _projection_metadata(**overrides)
        )


def _sync_store_with_task_domain(tmp_path: Path) -> SyncV2Store:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "projection-sync.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="owner-1",
            domains=["notes.note"],
        )
    )
    store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
        (json.dumps(["notes.note", "notes.task"]), "dataset-1"),
    )
    return store


def _insert_task_envelope(store: SyncV2Store) -> object:
    create = SyncEnvelopeCreate(
        dataset_id="dataset-1",
        client_envelope_id="task-envelope-7",
        domain="notes.task",
        operation="upsert",
        object_id=TASK_ID,
        parent_id="22222222-2222-4222-8222-222222222222",
        object_revision=7,
        entity_version=7,
        payload={"task_id": TASK_ID},
        payload_hash=TASK_HASH,
        apply_status="applied",
        applied_at="2026-08-24T10:00:00+00:00",
        created_at_client="2026-08-24T10:00:00+00:00",
        status="accepted",
    )
    with store.db.backend.transaction() as connection:
        return store.db._insert_envelope_in_transaction(  # noqa: SLF001
            create,
            connection=connection,
        )


def test_historical_task_envelope_lookup_requires_every_anchor_claim(
    tmp_path: Path,
) -> None:
    """A projection base resolves only through its exact immutable task envelope."""
    store = _sync_store_with_task_domain(tmp_path)
    inserted = _insert_task_envelope(store)

    resolved = store.get_historical_task_envelope(
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        task_id=TASK_ID,
        envelope_id="task-envelope-7",
        object_revision=7,
        object_hash=TASK_HASH,
    )

    assert resolved == inserted
    for changed in (
        {"owner_user_id": "owner-2"},
        {"task_id": "33333333-3333-4333-8333-333333333333"},
        {"envelope_id": "task-envelope-8"},
        {"object_revision": 8},
        {"object_hash": "sha256:" + "d" * 64},
    ):
        claims = {
            "owner_user_id": "owner-1",
            "dataset_id": "dataset-1",
            "task_id": TASK_ID,
            "envelope_id": "task-envelope-7",
            "object_revision": 7,
            "object_hash": TASK_HASH,
            **changed,
        }
        assert store.get_historical_task_envelope(**claims) is None


def test_projection_cache_rebuild_requires_marker_group_and_immutable_envelopes(
    tmp_path: Path,
) -> None:
    """Deleting a locator cache does not lose durable managed projection authority."""
    note_db = CharactersRAGDB(tmp_path / "projection-product.db", client_id="owner-1")
    note_db.note_store.add_note("Projection note", "body", note_id="note-1")
    note_db.bind_local_task_graph_to_dataset(
        owner_user_id="owner-1",
        target_dataset_id="dataset-1",
    )
    task = note_db.create_task(
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        task_id=TASK_ID,
        note_id="note-1",
        text="Review source",
    )
    marker = TaskMarker(
        task_id=TASK_ID,
        revision=task["canonical_revision"],
        object_hash=task["canonical_hash"],
    )
    marker_text = render_task_marker(
        marker.task_id,
        revision=marker.revision,
        object_hash=marker.object_hash,
    )
    note_db.update_note(
        note_id="note-1",
        update_data={"content": f"- [ ] Review source {marker_text}\n"},
        expected_version=1,
    )
    note = note_db.get_note_by_id("note-1")
    parsed = parse_note_checklists(
        note_id="note-1",
        note_version=note["version"],
        content=note["content"],
    )
    item = parsed.items[0]
    note_db.set_task_projection(
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        task_id=TASK_ID,
        note_id="note-1",
        note_version=item.locator.note_version,
        line_number=item.locator.line_number,
        start_offset=item.locator.start_offset,
        end_offset=item.locator.end_offset,
        normalized_text_hash=item.locator.normalized_text_hash,
        occurrence_index=item.locator.occurrence_index,
        block_fingerprint=item.locator.block_fingerprint,
        raw_line=item.raw_line,
        has_child_content=item.has_child_content,
    )
    note_db.execute_query(
        "DELETE FROM task_note_projections WHERE owner_user_id = ? "
        "AND dataset_id = ? AND task_id = ?",
        ("owner-1", "dataset-1", TASK_ID),
    )

    sync_store = _sync_store_with_task_domain(tmp_path)
    anchor = _projection_metadata(
        task_revision=marker.revision,
        task_hash=marker.object_hash,
        note_envelope_id="note-envelope-9",
        note_hash=NOTE_HASH,
        marker_hash=task_marker_hash(marker),
    )
    routing = {"task_projection": anchor}
    plan_hash = "d" * 64
    envelopes = (
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="note-envelope-9",
            domain="notes.note",
            operation="upsert",
            object_id="note-1",
            object_revision=9,
            entity_version=9,
            payload={"note_id": "note-1"},
            payload_hash=NOTE_HASH,
            apply_status="applied",
            applied_at="2026-08-24T10:00:00+00:00",
            created_at_client="2026-08-24T10:00:00+00:00",
            routing_metadata=routing,
            mutation_group_id="projection-group-1",
            mutation_step=0,
            mutation_step_count=2,
            mutation_plan_hash=plan_hash,
        ),
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="task-envelope-7",
            domain="notes.task",
            operation="upsert",
            object_id=TASK_ID,
            parent_id="note-1",
            object_revision=marker.revision,
            entity_version=marker.revision,
            payload={"task_id": TASK_ID},
            payload_hash=marker.object_hash,
            apply_status="applied",
            applied_at="2026-08-24T10:00:00+00:00",
            created_at_client="2026-08-24T10:00:00+00:00",
            routing_metadata=routing,
            mutation_group_id="projection-group-1",
            mutation_step=1,
            mutation_step_count=2,
            mutation_plan_hash=plan_hash,
        ),
    )
    with sync_store.db.backend.transaction() as connection:
        for envelope in envelopes:
            sync_store.db._insert_envelope_in_transaction(  # noqa: SLF001
                envelope,
                connection=connection,
            )

    coordinator = import_module(
        "tldw_Server_API.app.core.Sync.v2.notes_task_coordinator"
    )
    result = coordinator.rebuild_task_projection_cache(
        task_store=note_db.task_store,
        sync_store=sync_store,
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        note_id="note-1",
        item=item,
    )

    assert result.reason_code is None
    assert result.projection is not None
    assert result.projection["task_id"] == TASK_ID
    assert result.projection["raw_line"].endswith(marker_text)
    note_db.close_connection()
