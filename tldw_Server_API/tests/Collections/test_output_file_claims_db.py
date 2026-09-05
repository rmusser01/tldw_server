"""Cross-writer row/path reservations on real Collections databases."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ReadingArtifactOwnershipConflict
from tldw_Server_API.tests.Collections.test_output_file_operations_db import insert_binding, insert_operation
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import make_reading as create_reading

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def make_reading(db):
    # Compare persisted state, not the upsert result's transient is_new flags.
    return db.get_content_item(create_reading(db).id)


def output(db, name="source.md"):
    return db.create_output_artifact(
        type_="reading_archive",
        title=name,
        format_="md",
        storage_path=name,
        retention_until="2000-01-01T00:00:00",
    )


def reserved(db):
    insert_binding(db)
    original = output(db)
    operation = db.prepare_output_file_operation(
        "test-volume",
        kind="replace",
        output_id=original.id,
        destination_path="destination.md",
        intended={"title": "Destination"},
        lease_seconds=120,
        reserved_bytes=512,
    )
    return original, operation


@pytest.mark.parametrize("action", ["title", "metadata", "media", "soft_delete", "hard_delete", "purge"])
def test_pending_operation_blocks_generic_row_mutations(db, action):
    original, operation = reserved(db)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        if action == "title":
            db.update_output_artifact(original.id, title="Changed")
        elif action == "metadata":
            db.update_output_artifact_metadata(original.id, metadata_json='{"changed":true}')
        elif action == "media":
            db.update_output_media_item_id(original.id, 99)
        elif action == "soft_delete":
            db.delete_output_artifact(original.id)
        elif action == "hard_delete":
            db.delete_output_artifact(original.id, hard=True)
        else:
            db.delete_output_artifact_record(original.id, hard=True, purge_before="2030-01-01T00:00:00")
    assert db.get_output_artifact(original.id) == original
    assert db.get_output_file_operation(operation["token"], "test-volume") == operation


@pytest.mark.parametrize("column", ["source_path", "stage_path", "destination_path"])
@pytest.mark.parametrize("action", ["create", "retarget"])
def test_all_generic_path_claims_block_new_attachments(db, column, action):
    original, operation = reserved(db)
    other = output(db, "other.md")
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        if action == "create":
            output(db, operation[column].upper())
        else:
            db.update_output_artifact(other.id, storage_path=operation[column].upper())
    assert db.get_output_artifact(other.id) == other
    assert db.get_output_artifact(original.id) == original


def test_pending_row_cannot_acquire_reading_ownership(db):
    original, operation = reserved(db)
    parent = make_reading(db)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.register_reading_output_ownership(
            parent.id, original.id, expected_revision=parent.revision, storage_namespace_id="test-volume"
        )
    assert db.get_content_item(parent.id) == parent
    assert db.get_output_file_operation(operation["token"], "test-volume") == operation


def test_foreign_user_is_not_blocked_by_same_filename(db):
    original, _ = reserved(db)
    other = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    foreign = output(other, original.storage_path)
    assert foreign.user_id == "781"


@pytest.mark.parametrize("db", ["sqlite"], indirect=True)
def test_output_id_cannot_be_reused_until_file_completion(db):
    original, operation = reserved(db)
    # Simulate an already-committed remove whose file cleanup is still pending.
    with db.transaction() as conn:
        db.backend.execute("UPDATE output_file_operations SET phase = 'committed'", (), connection=conn)
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (original.id,), connection=conn)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        output(db, "unrelated.md")
    assert db.backend.execute("SELECT COUNT(*) FROM outputs").scalar == 0
    assert db.finish_output_file_operation(operation["token"], "test-volume")
    assert output(db, "unrelated.md").id == original.id


@pytest.mark.parametrize("column", ["source_path", "destination_path"])
def test_generic_prepare_cannot_claim_another_operations_path(db, column):
    _, first = reserved(db)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.prepare_output_file_operation(
            "test-volume", kind="create", destination_path=first[column].upper(), lease_seconds=120
        )
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 1


@pytest.mark.parametrize(
    "alias", ["destination.md", "DESTINATION.md", "/legacy/DESTINATION.md", r"C:\legacy\DESTINATION.md"]
)
def test_prepare_rejects_occupied_destination_alias(db, alias):
    insert_binding(db)
    existing = output(db, "existing.md")
    db.backend.execute("UPDATE outputs SET storage_path = ? WHERE id = ?", (alias, existing.id))
    with pytest.raises(RuntimeError, match="^output_path_conflict$"):
        db.prepare_output_file_operation(
            "test-volume", kind="create", destination_path="destination.md", lease_seconds=120
        )
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


def test_prepare_rejects_unowned_alias_of_managed_source(db):
    insert_binding(db)
    original = output(db)
    managed = output(db, "managed.md")
    db.backend.execute("UPDATE outputs SET storage_path = ? WHERE id = ?", ("/legacy/SOURCE.md", managed.id))
    parent = make_reading(db)
    db.register_reading_output_ownership(
        parent.id, managed.id, expected_revision=parent.revision, storage_namespace_id="test-volume"
    )
    with pytest.raises(RuntimeError, match="^output_path_conflict$"):
        db.prepare_output_file_operation("test-volume", kind="remove", output_id=original.id, lease_seconds=120)


def test_shared_unowned_source_can_be_reserved(db):
    insert_binding(db)
    original = output(db)
    shared = output(db, "shared.md")
    db.backend.execute("UPDATE outputs SET storage_path = ? WHERE id = ?", (original.storage_path, shared.id))
    operation = db.prepare_output_file_operation(
        "test-volume",
        kind="replace",
        output_id=original.id,
        destination_path="new.md",
        lease_seconds=120,
    )
    assert operation["source_path"] == original.storage_path
    assert db.get_output_artifact(shared.id).storage_path == original.storage_path


def test_generic_prepare_respects_reading_staging(db):
    insert_binding(db)
    parent = make_reading(db)
    staging = db.reserve_reading_artifact(
        parent.id, expected_revision=parent.revision, storage_namespace_id="test-volume", lease_until=2**31
    )
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.prepare_output_file_operation(
            "test-volume", kind="create", destination_path=staging["storage_path"].upper(), lease_seconds=120
        )


def test_reading_adoption_rejects_conflicting_generic_claim(db):
    insert_binding(db)
    parent = make_reading(db)
    staging = db.reserve_reading_artifact(
        parent.id, expected_revision=parent.revision, storage_namespace_id="test-volume", lease_until=2**31
    )
    # Simulate inconsistent pre-rollout journal state, not a permitted prepare.
    insert_operation(db, source_path=staging["storage_path"], source_key=staging["storage_path"])
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.adopt_reading_artifact(staging["token"], "test-volume", title="Archive", now=1)
    assert db.get_content_item(parent.id) == parent
    assert db.get_reading_artifact(staging["token"], "test-volume") == staging


def test_reading_delete_rejects_conflicting_generic_claim(db):
    insert_binding(db)
    original = output(db)
    parent = make_reading(db)
    db.register_reading_output_ownership(
        parent.id, original.id, expected_revision=parent.revision, storage_namespace_id="test-volume"
    )
    parent = db.get_content_item(parent.id)
    insert_operation(db, output_id=original.id, source_path=original.storage_path, source_key=original.storage_path)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.hard_delete_reading_item(parent.id, expected_revision=parent.revision)
    assert db.get_content_item(parent.id) == parent
    assert db.get_output_artifact(original.id) == original


def test_generated_private_stage_cannot_collide_with_existing_claim(db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Collections_DB as module

    _, first = reserved(db)
    monkeypatch.setattr(module, "uuid4", lambda: UUID(first["stage_path"].removeprefix(".output-stage-")))
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.prepare_output_file_operation("test-volume", kind="create", destination_path="other.md", lease_seconds=120)


def test_reading_reservation_checks_generic_paths(db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Collections_DB as module

    insert_binding(db)
    fixed = UUID("12345678123456781234567812345678")
    parent = make_reading(db)
    db.prepare_output_file_operation(
        "test-volume", kind="create", destination_path=f"reading_archive_{fixed.hex}.md", lease_seconds=120
    )
    monkeypatch.setattr(module, "uuid4", lambda: fixed)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.reserve_reading_artifact(
            parent.id, expected_revision=parent.revision, storage_namespace_id="test-volume", lease_until=2**31
        )


@pytest.mark.parametrize("action", ["prepare_cleanup", "finish_cleanup"])
def test_reading_cleanup_cannot_discard_conflicting_generic_claim(db, action):
    insert_binding(db)
    parent = make_reading(db)
    staging = db.reserve_reading_artifact(
        parent.id, expected_revision=parent.revision, storage_namespace_id="test-volume", lease_until=2**31
    )
    db.cancel_reading_artifact(staging["token"], "test-volume")
    insert_operation(db, source_path=staging["storage_path"], source_key=staging["storage_path"])
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        if action == "prepare_cleanup":
            db.prepare_reading_artifact_cleanup("test-volume", now=2**31)
        else:
            db.finish_reading_artifact_cleanup(staging["token"], "test-volume")
    assert db.get_reading_artifact(staging["token"], "test-volume")["state"] == "pending"


def test_other_reading_namespace_does_not_block_generic_destination(db):
    insert_binding(db)
    existing = output(db)
    parent = make_reading(db)
    db.register_reading_output_ownership(
        parent.id, existing.id, expected_revision=parent.revision, storage_namespace_id="old-volume"
    )
    operation = db.prepare_output_file_operation(
        "test-volume", kind="create", destination_path=existing.storage_path, lease_seconds=120
    )
    assert operation["destination_path"] == existing.storage_path
    assert db.get_output_artifact(existing.id) == existing


def test_revalidation_does_not_ignore_a_new_conflicting_operation(db):
    _, operation = reserved(db)
    insert_operation(db, source_path="destination.md", source_key="destination.md")
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.validate_output_file_operation(operation["token"], "test-volume")


@pytest.mark.parametrize("first", ["reserve", "writer"])
@pytest.mark.parametrize("writer", ["ownership", "destination", "metadata", "delete"])
def test_reservation_and_writer_serialize_in_both_orders(db, monkeypatch, first, writer):
    insert_binding(db)
    original = output(db)
    parent = make_reading(db)
    first_db = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    acquired, release, second_started = Event(), Event(), Event()
    lock = first_db._lock_reading_revision_clock

    def held_lock(conn):
        lock(conn)
        acquired.set()
        assert release.wait(timeout=15)

    monkeypatch.setattr(first_db, "_lock_reading_revision_clock", held_lock)

    def run(adapter, action):
        try:
            if action == "reserve":
                adapter.prepare_output_file_operation(
                    "test-volume",
                    kind="replace",
                    output_id=original.id,
                    destination_path="destination.md",
                    lease_seconds=120,
                )
            elif writer == "ownership":
                adapter.register_reading_output_ownership(
                    parent.id, original.id, expected_revision=parent.revision, storage_namespace_id="test-volume"
                )
            elif writer == "destination":
                output(adapter, "destination.md")
            elif writer == "metadata":
                adapter.update_output_artifact(original.id, title="Changed")
            else:
                adapter.delete_output_artifact(original.id, hard=True)
        except (RuntimeError, KeyError) as exc:
            return exc.args[0]
        return "ok"

    def second():
        second_started.set()
        return run(db, "writer" if first == "reserve" else "reserve")

    with ThreadPoolExecutor(max_workers=2) as workers:
        one = workers.submit(run, first_db, first)
        try:
            assert acquired.wait(timeout=15)
            two = workers.submit(second)
            assert second_started.wait(timeout=15)
        finally:
            release.set()
        assert one.result(timeout=15) == "ok"
        expected = (
            "output_file_busy"
            if first == "reserve"
            else {
                "ownership": "output_operation_conflict",
                "destination": "output_path_conflict",
                "metadata": "ok",
                "delete": "output_not_found",
            }[writer]
        )
        assert two.result(timeout=15) == expected


@pytest.mark.parametrize("action", ["attach", "prepare", "revalidate"])
def test_legacy_absolute_reading_intent_blocks_relative_alias(db, action):
    insert_binding(db)
    parent = make_reading(db)
    staging = db.reserve_reading_artifact(
        parent.id, expected_revision=parent.revision, storage_namespace_id="test-volume", lease_until=2**31
    )
    if action == "revalidate":
        operation = db.prepare_output_file_operation(
            "test-volume", kind="create", destination_path="source.md", lease_seconds=120
        )
    # Existing disposal can retain a legacy absolute name. Simulate that persisted intent.
    db.backend.execute(
        "UPDATE reading_artifact_paths SET storage_path = ?, state = 'pending' WHERE token = ?",
        ("/legacy/SOURCE.md", staging["token"]),
    )
    with pytest.raises(ReadingArtifactOwnershipConflict):
        if action == "attach":
            output(db)
        elif action == "prepare":
            db.prepare_output_file_operation(
                "test-volume", kind="create", destination_path="source.md", lease_seconds=120
            )
        else:
            db.validate_output_file_operation(operation["token"], "test-volume")


@pytest.mark.parametrize("action", ["metadata", "delete"])
def test_known_other_volume_owned_row_is_not_blocked_by_path_only_claim(db, action):
    insert_binding(db)
    existing = output(db)
    parent = make_reading(db)
    db.register_reading_output_ownership(
        parent.id, existing.id, expected_revision=parent.revision, storage_namespace_id="old-volume"
    )
    operation = db.prepare_output_file_operation(
        "test-volume", kind="create", destination_path=existing.storage_path, lease_seconds=120
    )
    if action == "metadata":
        assert db.update_managed_reading_output(existing.id, title="Updated").title == "Updated"
    else:
        assert db.delete_output_artifact(existing.id, hard=True)
    assert db.get_output_file_operation(operation["token"], "test-volume") == operation


def test_file_completion_releases_claims_while_history_is_pending(db):
    insert_binding(db)
    original = output(db)
    operation = db.prepare_output_file_operation("test-volume", kind="remove", output_id=original.id, lease_seconds=120)
    with db.commit_output_file_operation(operation["token"], "test-volume", dispose_history=True) as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (original.id,), connection=conn)
    db.finish_output_file_operation(operation["token"], "test-volume")
    assert db.get_output_file_operation(operation["token"], "test-volume")["effects_pending"] == 1
    replacement = output(db)
    replacement = db.update_output_artifact(replacement.id, title="New instance")
    assert db.ack_output_file_effect(operation["token"], "test-volume", "dispose_history")
    assert db.get_output_artifact(replacement.id) == replacement
