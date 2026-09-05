"""Inert generic output journal schema; no filesystem authority is activated."""

from __future__ import annotations

from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.tests.Collections.test_reading_artifact_adoption import adopt
from tldw_Server_API.tests.Collections.test_reading_artifact_cleanup import reserve
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import make_archive_output

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def test_schema_bootstrap_is_inert_and_repeatable(db, tmp_path):
    assert db.backend.table_exists("output_storage_bindings")
    assert db.backend.table_exists("output_file_operations")
    before_paths = set(tmp_path.rglob("*"))
    db._ensure_reading_revision_schema()
    db._ensure_reading_revision_schema()
    other = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    assert other.backend.execute("SELECT COUNT(*) FROM output_storage_bindings").scalar == 0
    assert other.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0
    assert set(tmp_path.rglob("*")) == before_paths


def incarnation(db, output_id):
    return db.backend.execute(
        "SELECT file_incarnation FROM outputs WHERE user_id = ? AND id = ?",
        (db.user_id, output_id),
    ).scalar


def test_new_output_has_internal_stable_incarnation(db):
    assert "file_incarnation" in db._table_columns("outputs")
    first = make_archive_output(db)
    token = incarnation(db, first.id)
    assert UUID(token).hex == token
    db.update_output_artifact(first.id, title="Display title")
    db._ensure_reading_revision_schema()
    assert incarnation(db, first.id) == token
    assert incarnation(db, make_archive_output(db).id) != token
    assert "file_incarnation" not in vars(db.get_output_artifact(first.id))


def test_reading_adoption_allocates_internal_output_incarnation(db, tmp_path):
    assert "file_incarnation" in db._table_columns("outputs")
    _, namespace, reservation = reserve(db, tmp_path)
    output = adopt(db, namespace, reservation, tmp_path)
    token = incarnation(db, output.id)
    assert UUID(token).hex == token


def test_incarnation_backfill_is_explicit_scoped_and_idempotent(db):
    assert "file_incarnation" in db._table_columns("outputs")
    first = make_archive_output(db)
    other = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    foreign = make_archive_output(other)
    db.backend.execute("UPDATE outputs SET file_incarnation = NULL", ())
    db._ensure_reading_revision_schema()
    assert incarnation(db, first.id) is None
    assert db.backfill_output_file_incarnations() == 1
    token = incarnation(db, first.id)
    assert UUID(token).hex == token
    assert incarnation(other, foreign.id) is None
    assert db.backfill_output_file_incarnations() == 0
    assert incarnation(db, first.id) == token


def test_existing_output_schema_upgrade_preserves_rows_without_activating(db):
    output = make_archive_output(db)
    db.backend.execute("DROP INDEX ux_output_file_incarnation", ())
    db.backend.execute("ALTER TABLE outputs DROP COLUMN file_incarnation", ())
    db._ensure_reading_revision_schema()
    db._ensure_reading_revision_schema()
    assert db.get_output_artifact(output.id) == output
    assert incarnation(db, output.id) is None
    assert db.backend.execute("SELECT COUNT(*) FROM output_storage_bindings").scalar == 0


def test_output_idempotency_replay_keeps_original_incarnation(db):
    fields = {
        "type_": "report",
        "title": "One report",
        "format_": "md",
        "storage_path": "one.md",
        "idempotency_key": "one",
    }
    output = db.create_output_artifact(**fields)
    token = incarnation(db, output.id)
    assert db.create_output_artifact(**fields) == output
    assert incarnation(db, output.id) == token


def test_output_identity_is_not_recycled_with_numeric_id(db):
    assert "file_incarnation" in db._table_columns("outputs")
    first = make_archive_output(db)
    old = incarnation(db, first.id)
    db.delete_output_artifact_record(first.id, hard=True)
    replacement = make_archive_output(db)
    assert incarnation(db, replacement.id) != old
    # PostgreSQL sequences do not normally reuse IDs; exercise reuse explicitly.
    if replacement.id != first.id:
        db.backend.execute("UPDATE outputs SET id = ? WHERE id = ?", (first.id, replacement.id))
    assert incarnation(db, first.id) != old


def insert_binding(db, **changes):
    values = {
        "user_id": db.user_id,
        "storage_namespace_id": "test-volume",
        "protocol_version": 1,
        "operation_bytes": 1024,
        "user_pending_bytes": 4096,
        "active_operations": 4,
        "text_input_bytes": 512,
        "text_output_bytes": 1024,
        "free_space_margin_bytes": 1024,
    }
    values.update(changes)
    with db.transaction() as conn:
        db.backend.execute(
            f"INSERT INTO output_storage_bindings ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
            tuple(values.values()),
            connection=conn,
        )


def insert_operation(db, **changes):
    values = {
        "token": "operation-one",
        "user_id": db.user_id,
        "storage_namespace_id": "test-volume",
        "kind": "remove",
        "source_path": "capture.md",
        "source_key": "capture.md",
        "lease_until": 100,
    }
    values.update(changes)
    with db.transaction() as conn:
        db.backend.execute(
            f"INSERT INTO output_file_operations ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
            tuple(values.values()),
            connection=conn,
        )


def test_binding_rejects_missing_identity_and_nonpositive_resource_policy(db):
    assert db.backend.table_exists("output_storage_bindings")
    for patch in (
        {"user_id": None},
        {"storage_namespace_id": " "},
        {"protocol_version": 0},
        {"operation_bytes": 0},
        {"user_pending_bytes": -1},
        {"active_operations": 0},
        {"text_input_bytes": 0},
        {"text_output_bytes": None},
        {"free_space_margin_bytes": 0},
    ):
        with pytest.raises(DatabaseError):
            insert_binding(db, **patch)
    assert db.backend.execute("SELECT COUNT(*) FROM output_storage_bindings").scalar == 0
    insert_binding(db)
    with pytest.raises(DatabaseError):
        insert_binding(db, storage_namespace_id="another-volume")


def test_journal_constraints_reject_invalid_authority_and_unbounded_payloads(db):
    assert db.backend.table_exists("output_file_operations")
    insert_binding(db)
    for patch in (
        {"token": None},
        {"token": " "},
        {"user_id": "781"},
        {"storage_namespace_id": "other-volume"},
        {"kind": "arbitrary-job"},
        {"phase": "unknown"},
        {"fs_done": 1},
        {"fs_done": 2, "phase": "committed"},
        {"lease_until": -1},
        {"attempts": -1},
        {"retry_after": -1},
        {"reserved_bytes": -1},
        {"effects_pending": -1},
        {"original_json": "x" * 32769},
        {"intended_json": "x" * 32769},
        {"effects_json": "x" * 16385},
        {"source_path": None},
        {"source_key": None},
        {"source_key": "different.md"},
        {"stage_path": "not-a-remove.md", "stage_key": "not-a-remove.md"},
    ):
        with pytest.raises(DatabaseError):
            insert_operation(db, **patch)
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


def test_journal_survives_output_deletion_and_schema_reinitialization(db):
    assert db.backend.table_exists("output_file_operations")
    output = make_archive_output(db)
    insert_binding(db)
    insert_operation(db, output_id=output.id, phase="committed", effects_pending=1)
    # Raw SQL isolates the no-cascade schema invariant from later writer guards.
    db.backend.execute("DELETE FROM outputs WHERE user_id = ? AND id = ?", (db.user_id, output.id))
    db._ensure_reading_revision_schema()
    row = db.backend.execute("SELECT * FROM output_file_operations WHERE token = ?", ("operation-one",)).first
    assert row["output_id"] == output.id
    assert row["phase"] == "committed"
    assert row["effects_pending"] == 1


def test_journal_accepts_each_bounded_kind_and_completed_delivery_state(db):
    insert_binding(db)
    insert_operation(db, token="remove", phase="aborting", fs_done=1)
    insert_operation(
        db,
        token="create",
        kind="create",
        source_path=None,
        source_key=None,
        stage_path="private.tmp",
        stage_key="private.tmp",
        destination_path="Report.md",
        destination_key="report.md",
        phase="committed",
        fs_done=1,
        effects_pending=1,
    )
    insert_operation(
        db,
        token="replace",
        kind="replace",
        stage_path="another.tmp",
        stage_key="another.tmp",
        destination_path="new.md",
        destination_key="new.md",
    )
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 3


def test_incarnation_backfill_rolls_back_all_rows_on_failure(db, monkeypatch):
    assert "file_incarnation" in db._table_columns("outputs")
    first = make_archive_output(db)
    db.update_output_artifact(first.id, title="Distinct first title")
    make_archive_output(db)
    db.backend.execute("UPDATE outputs SET file_incarnation = NULL", ())
    execute = db.backend.execute
    writes = []

    def fail_second_update(query, params=None, *, connection=None, **kwargs):
        if query.startswith("UPDATE outputs SET file_incarnation"):
            writes.append(connection)
            if len(writes) == 2:
                raise RuntimeError("backfill interrupted")
        return execute(query, params, connection=connection, **kwargs)

    monkeypatch.setattr(db.backend, "execute", fail_second_update)
    with pytest.raises(RuntimeError, match="backfill interrupted"):
        db.backfill_output_file_incarnations()
    assert len(writes) == 2 and writes[0] is writes[1] and writes[0] is not None
    assert db.backend.execute("SELECT COUNT(*) FROM outputs WHERE file_incarnation IS NULL").scalar == 2
