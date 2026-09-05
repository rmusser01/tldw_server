"""Inert generic output journal schema; no filesystem authority is activated."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
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


def prepared(db, *, kind="remove", **changes):
    assert hasattr(db, "prepare_output_file_operation"), "journal transitions are not implemented"
    insert_binding(db)
    output = make_archive_output(db)
    fields = {"output_id": output.id, "kind": kind, "lease_seconds": 120, "reserved_bytes": 512}
    if kind == "replace":
        fields.update(destination_path="new.md", intended={"title": "New"})
    fields.update(changes)
    return output, db.prepare_output_file_operation("test-volume", **fields)


def test_prepare_captures_bounded_snapshot_without_changing_output(db):
    output, operation = prepared(db, kind="replace")
    assert UUID(operation["token"]).hex == operation["token"]
    assert operation["phase"] == "prepared" and operation["fs_done"] == 0
    assert operation["source_path"] == output.storage_path
    assert operation["stage_path"] not in {output.storage_path, "new.md"}
    assert json.loads(operation["intended_json"]) == {"title": "New"}
    assert output.metadata_json not in operation["original_json"]
    assert db.get_output_artifact(output.id) == output
    assert db.validate_output_file_operation(operation["token"], "test-volume") == operation


def test_prepare_create_has_no_original_row(db):
    assert hasattr(db, "prepare_output_file_operation")
    insert_binding(db)
    operation = db.prepare_output_file_operation(
        "test-volume",
        kind="create",
        destination_path="new.md",
        intended={"title": "New", "format": "md"},
        lease_seconds=120,
        reserved_bytes=512,
    )
    assert operation["source_path"] is None and operation["output_id"] is None
    assert json.loads(operation["original_json"]) == {}
    assert db.backend.execute("SELECT COUNT(*) FROM outputs").scalar == 0


def test_prepare_rejects_invalid_inputs_without_a_journal_row(db):
    assert hasattr(db, "prepare_output_file_operation")
    insert_binding(db)
    output = make_archive_output(db)
    for patch in (
        {"kind": "other"},
        {"lease_seconds": 0},
        {"lease_seconds": True},
        {"reserved_bytes": -1},
        {"reserved_bytes": 0.5},
        {"destination_path": "../outside.md"},
        {"destination_path": ".."},
        {"destination_path": ".reading-storage.lock"},
        {"intended": {"body": "private body"}},
        {"intended": {"title": "x" * 32769}},
        {"intended": {"title": None}},
    ):
        fields = {
            "kind": "replace",
            "output_id": output.id,
            "destination_path": "new.md",
            "lease_seconds": 120,
            "reserved_bytes": 512,
        }
        fields.update(patch)
        with pytest.raises(ValueError, match="^output_operation_invalid$"):
            db.prepare_output_file_operation("test-volume", **fields)
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


def test_prepare_requires_bound_namespace_and_original_output_identity(db):
    assert hasattr(db, "prepare_output_file_operation")
    output = make_archive_output(db)
    with pytest.raises(RuntimeError, match="^output_storage_unavailable$"):
        db.prepare_output_file_operation("test-volume", kind="remove", output_id=output.id, lease_seconds=120)
    insert_binding(db)
    db.backend.execute("UPDATE outputs SET file_incarnation = NULL WHERE id = ?", (output.id,))
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        db.prepare_output_file_operation("test-volume", kind="remove", output_id=output.id, lease_seconds=120)
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0


def test_journal_lookups_and_mutations_are_user_and_namespace_scoped(db):
    _, operation = prepared(db)
    other = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    for adapter, namespace in ((other, "test-volume"), (db, "wrong-volume")):
        with pytest.raises(KeyError, match="output_operation_not_found"):
            adapter.get_output_file_operation(operation["token"], namespace)
        assert not adapter.abort_output_file_operation(operation["token"], namespace)
        assert not adapter.finish_output_file_operation(operation["token"], namespace)
        assert not adapter.ack_output_file_effect(operation["token"], namespace, "dispose_history")
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "prepared"


def test_expired_operation_cannot_validate_or_commit_but_can_abort(db, monkeypatch):
    _, operation = prepared(db)
    db.backend.execute("UPDATE output_file_operations SET lease_until = 1", ())
    for action in ("validate", "commit"):
        with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
            if action == "validate":
                db.validate_output_file_operation(operation["token"], "test-volume")
            else:
                with db.commit_output_file_operation(operation["token"], "test-volume"):
                    pytest.fail("expired operation entered commit")
    assert db.abort_output_file_operation(operation["token"], "test-volume")
    assert not db.abort_output_file_operation(operation["token"], "test-volume")


def test_original_snapshot_rechecked_before_commit(db):
    output, operation = prepared(db)
    db.update_output_artifact_metadata(output.id, metadata_json='{"private": "changed"}')
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        with db.commit_output_file_operation(operation["token"], "test-volume"):
            pytest.fail("stale snapshot entered commit")
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "prepared"


def test_commit_and_output_write_share_transaction_and_rollback(db):
    output, operation = prepared(db, kind="replace")
    with pytest.raises(RuntimeError, match="rollback checkpoint"):
        with db.transaction() as outer:
            with db.commit_output_file_operation(operation["token"], "test-volume", connection=outer) as conn:
                assert conn is outer
                db.backend.execute("UPDATE outputs SET title = ? WHERE id = ?", ("New", output.id), connection=conn)
            assert (
                db.get_output_file_operation(operation["token"], "test-volume", connection=outer)["phase"]
                == "committed"
            )
            raise RuntimeError("rollback checkpoint")
    assert db.get_output_artifact(output.id) == output
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "prepared"


def test_exception_in_commit_body_preserves_prepared_state(db):
    _, operation = prepared(db)
    with pytest.raises(RuntimeError, match="write failed"):
        with db.commit_output_file_operation(operation["token"], "test-volume"):
            raise RuntimeError("write failed")
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "prepared"


def test_committed_phase_wins_over_abort_and_cannot_commit_twice(db):
    output, operation = prepared(db)
    with db.commit_output_file_operation(operation["token"], "test-volume") as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=conn)
    assert not db.abort_output_file_operation(operation["token"], "test-volume")
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        with db.commit_output_file_operation(operation["token"], "test-volume"):
            pytest.fail("committed work was replayed")
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "committed"


def test_filesystem_completion_keeps_history_until_idempotent_ack(db):
    output, operation = prepared(db)
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        db.finish_output_file_operation(operation["token"], "test-volume")
    with db.commit_output_file_operation(operation["token"], "test-volume", dispose_history=True) as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=conn)
    assert not db.ack_output_file_effect(operation["token"], "test-volume", "dispose_history")
    assert db.finish_output_file_operation(operation["token"], "test-volume")
    row = db.get_output_file_operation(operation["token"], "test-volume")
    assert row["fs_done"] == 1 and row["reserved_bytes"] == 0 and row["effects_pending"] == 1
    assert json.loads(row["effects_json"])[0]["incarnation"] == json.loads(operation["original_json"])["incarnation"]
    assert not db.finish_output_file_operation(operation["token"], "test-volume")
    assert not db.ack_output_file_effect(operation["token"], "test-volume", "unknown")
    assert db.ack_output_file_effect(operation["token"], "test-volume", "dispose_history")
    assert not db.ack_output_file_effect(operation["token"], "test-volume", "dispose_history")
    with pytest.raises(KeyError):
        db.get_output_file_operation(operation["token"], "test-volume")


def test_abort_cleanup_retires_without_touching_output(db):
    output, operation = prepared(db)
    assert db.abort_output_file_operation(operation["token"], "test-volume")
    assert db.finish_output_file_operation(operation["token"], "test-volume")
    assert not db.finish_output_file_operation(operation["token"], "test-volume")
    assert db.get_output_artifact(output.id) == output


def test_remove_can_commit_a_soft_deleted_output(db):
    insert_binding(db)
    output = make_archive_output(db)
    db.backend.execute("UPDATE outputs SET deleted = 1 WHERE id = ?", (output.id,))
    operation = db.prepare_output_file_operation("test-volume", kind="remove", output_id=output.id, lease_seconds=120)
    with db.commit_output_file_operation(operation["token"], "test-volume") as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=conn)
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "committed"
    assert db.backend.execute("SELECT COUNT(*) FROM outputs WHERE id = ?", (output.id,)).scalar == 0


def test_remove_rechecks_original_deletion_state(db):
    insert_binding(db)
    output = make_archive_output(db)
    db.backend.execute("UPDATE outputs SET deleted = 1 WHERE id = ?", (output.id,))
    operation = db.prepare_output_file_operation("test-volume", kind="remove", output_id=output.id, lease_seconds=120)
    db.backend.execute("UPDATE outputs SET deleted = 0 WHERE id = ?", (output.id,))
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        db.validate_output_file_operation(operation["token"], "test-volume")


def test_replace_cannot_target_a_soft_deleted_output(db):
    insert_binding(db)
    output = make_archive_output(db)
    db.backend.execute("UPDATE outputs SET deleted = 1 WHERE id = ?", (output.id,))
    with pytest.raises(KeyError, match="output_not_found"):
        db.prepare_output_file_operation(
            "test-volume", kind="replace", output_id=output.id, destination_path="new.md", lease_seconds=120
        )


def test_validation_rejects_binding_protocol_changed_after_prepare(db):
    _, operation = prepared(db)
    db.backend.execute("UPDATE output_storage_bindings SET protocol_version = 2", ())
    with pytest.raises(RuntimeError, match="^output_storage_unavailable$"):
        with db.commit_output_file_operation(operation["token"], "test-volume"):
            pytest.fail("unknown protocol entered commit")
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "prepared"


def test_lease_is_checked_after_waiting_for_revision_fence(db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management import Collections_DB as module

    _, operation = prepared(db)
    lock = db._lock_reading_revision_clock

    class AfterLease(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(operation["lease_until"], timezone.utc) + timedelta(seconds=1)

    def wait_then_lock(conn):
        lock(conn)
        monkeypatch.setattr(module, "datetime", AfterLease)

    monkeypatch.setattr(db, "_lock_reading_revision_clock", wait_then_lock)
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        with db.commit_output_file_operation(operation["token"], "test-volume"):
            pytest.fail("lease expired during lock wait")


@pytest.mark.parametrize("first", ["commit", "abort"])
def test_concurrent_commit_and_abort_have_one_winner(db, first):
    output, operation = prepared(db)
    locked, contender_started, release = Event(), Event(), Event()

    def transition(action, conn=None):
        if action == "abort":
            return db.abort_output_file_operation(operation["token"], "test-volume", connection=conn)
        try:
            with db.commit_output_file_operation(operation["token"], "test-volume", connection=conn) as active:
                db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=active)
        except RuntimeError as exc:
            assert str(exc) == "output_operation_conflict"
            return False
        return True

    def winner():
        with db.transaction() as conn:
            db._lock_reading_revision_clock(conn)
            locked.set()
            assert release.wait(timeout=15)
            return transition(first, conn)

    def contender():
        contender_started.set()
        return transition("abort" if first == "commit" else "commit")

    with ThreadPoolExecutor(max_workers=2) as workers:
        winning = workers.submit(winner)
        try:
            assert locked.wait(timeout=15)
            losing = workers.submit(contender)
            assert contender_started.wait(timeout=15)
        finally:
            release.set()
        assert winning.result(timeout=15)
        assert not losing.result(timeout=15)
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == (
        "committed" if first == "commit" else "aborting"
    )
    assert db.backend.execute("SELECT COUNT(*) FROM outputs WHERE id = ?", (output.id,)).scalar == (
        0 if first == "commit" else 1
    )
