"""Inert generic output journal schema; no filesystem authority is activated."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event
from types import SimpleNamespace
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
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        db.update_output_artifact_metadata(output.id, metadata_json='{"private": "changed"}')
    # An incompatible old writer or offline SQL bypass still cannot commit a stale snapshot.
    db.backend.execute("UPDATE outputs SET metadata_json = ? WHERE id = ?", ('{"private": "changed"}', output.id))
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


def prepare_create(db, *, name="new.md", budget=512, connection=None):
    return db.prepare_output_file_operation(
        "test-volume",
        kind="create",
        destination_path=name,
        lease_seconds=120,
        reserved_bytes=budget,
        connection=connection,
    )


@pytest.mark.parametrize("limit", ["operation_bytes", "user_pending_bytes", "active_operations"])
def test_admission_rejects_capacity_before_inserting_operation(db, limit):
    insert_binding(db, **{limit: 1})
    if limit == "active_operations":
        prepare_create(db, name="first.md")
    before = db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare_create(db)
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == before
    assert db.backend.execute("SELECT COUNT(*) FROM outputs").scalar == 0


@pytest.mark.parametrize("phase", ["prepared", "committed", "aborting"])
def test_unfinished_files_keep_capacity_even_when_expired_or_blocked(db, phase):
    insert_binding(db, user_pending_bytes=1024)
    insert_operation(db, reserved_bytes=1024, phase=phase, last_error="identity_unconfirmed", lease_until=1)
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare_create(db, budget=1)
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 1


def test_admission_accepts_exact_byte_and_count_boundaries(db):
    insert_binding(db, operation_bytes=512, user_pending_bytes=1024, active_operations=2)
    prepare_create(db, name="first.md")
    prepare_create(db, name="second.md")
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare_create(db, budget=0)
    assert db.backend.execute("SELECT SUM(reserved_bytes) FROM output_file_operations").scalar == 1024


@pytest.mark.parametrize("limit", ["user_pending_bytes", "active_operations"])
def test_concurrent_admission_cannot_overspend(db, limit):
    insert_binding(db, **{limit: 512 if limit == "user_pending_bytes" else 1})
    ready = Barrier(2)

    def reserve_budget(name):
        ready.wait(timeout=15)
        try:
            prepare_create(db, name=name)
        except RuntimeError as exc:
            assert str(exc) == "output_storage_capacity"
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(reserve_budget, name) for name in ("first.md", "second.md")]
        assert sorted(future.result(timeout=15) for future in futures) == [False, True]
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 1


@pytest.mark.parametrize("db", ["sqlite"], indirect=True)
def test_admission_rejects_fractional_persisted_policy(db):
    # SQLite BIGINT affinity accepts fractions despite the positive-range CHECK.
    for field in (
        "operation_bytes",
        "user_pending_bytes",
        "active_operations",
        "text_input_bytes",
        "text_output_bytes",
        "free_space_margin_bytes",
    ):
        insert_binding(db, **{field: 2.5})
        with pytest.raises(RuntimeError, match="^output_storage_unavailable$"):
            prepare_create(db, budget=1)
        assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0
        db.backend.execute("DELETE FROM output_storage_bindings", ())


def test_admission_capacity_is_user_scoped(db):
    insert_binding(db, active_operations=1, user_pending_bytes=512)
    prepare_create(db)
    other = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    insert_binding(other, active_operations=1, user_pending_bytes=512)
    foreign = prepare_create(other)
    assert foreign["user_id"] == "781" and foreign["reserved_bytes"] == 512


def test_rolled_back_admission_does_not_spend_capacity(db):
    insert_binding(db, active_operations=1, user_pending_bytes=512)
    with pytest.raises(RuntimeError, match="rollback admission"):
        with db.transaction() as conn:
            prepare_create(db, connection=conn)
            raise RuntimeError("rollback admission")
    assert prepare_create(db)["reserved_bytes"] == 512
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 1


def test_file_completion_releases_capacity_without_history_ack(db):
    output, operation = prepared(db)
    db.backend.execute("UPDATE output_storage_bindings SET active_operations = 1, user_pending_bytes = 512", ())
    with db.commit_output_file_operation(operation["token"], "test-volume", dispose_history=True) as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=conn)
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare_create(db)
    assert db.finish_output_file_operation(operation["token"], "test-volume")
    replacement = prepare_create(db)
    assert db.get_output_file_operation(operation["token"], "test-volume")["effects_pending"] == 1
    assert db.ack_output_file_effect(operation["token"], "test-volume", "dispose_history")
    assert db.get_output_file_operation(replacement["token"], "test-volume") == replacement


def test_abort_keeps_capacity_until_file_completion(db):
    insert_binding(db, active_operations=1)
    operation = prepare_create(db)
    assert db.abort_output_file_operation(operation["token"], "test-volume")
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare_create(db, name="second.md")
    assert db.finish_output_file_operation(operation["token"], "test-volume")
    assert prepare_create(db, name="second.md")["reserved_bytes"] == 512


@pytest.mark.parametrize("method,value", [("set_audiobook_output_usage", 80), ("update_audiobook_output_usage", -20)])
def test_output_and_quota_rollback_on_same_connection(db, method, value):
    output, operation = prepared(db)
    db.set_audiobook_output_usage(100)
    with pytest.raises(RuntimeError, match="rollback accounting"):
        with db.commit_output_file_operation(operation["token"], "test-volume") as conn:
            db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=conn)
            assert getattr(db, method)(value, connection=conn) == 80
            assert db.get_audiobook_output_usage(connection=conn) == 80
            raise RuntimeError("rollback accounting")
    assert db.get_output_artifact(output.id) == output
    assert db.get_audiobook_output_usage() == 100
    assert db.get_output_file_operation(operation["token"], "test-volume")["phase"] == "prepared"


def test_committed_output_cannot_replay_quota_delta(db):
    output, operation = prepared(db)
    db.set_audiobook_output_usage(100)
    with db.commit_output_file_operation(operation["token"], "test-volume") as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,), connection=conn)
        assert db.update_audiobook_output_usage(-20, connection=conn) == 80
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        with db.commit_output_file_operation(operation["token"], "test-volume") as conn:
            db.update_audiobook_output_usage(-20, connection=conn)
    assert db.get_audiobook_output_usage() == 80


def prepared_recorded_mutation(db, *, kind="replace", audiobook=False, deleted=False):
    insert_binding(db)
    output = None
    if kind != "create":
        output = db.create_output_artifact(
            type_="audiobook_mp3" if audiobook else "report",
            title="Original",
            format_="md",
            storage_path="source.md",
            metadata_json='{"byte_size":8,"keep":"unchanged"}',
        )
        if deleted:
            db.delete_output_artifact(output.id, hard=False)
    intended = {} if kind == "remove" else {"title": "Recorded title", "format": "html", "retention_until": None}
    if kind == "create":
        intended["type"] = "audiobook_mp3" if audiobook else "report"
    row = db.prepare_output_file_operation(
        "test-volume",
        kind=kind,
        output_id=output.id if output else None,
        destination_path=None if kind == "remove" else "destination.html",
        intended=intended,
        reserved_bytes=32,
        lease_seconds=60,
    )
    source = {"dev": 1, "ino": 2, "mode": 32768, "nlink": 1, "size": 8, "mtime_ns": 1, "ctime_ns": 1}
    stage = {"dev": 1, "ino": 3, "mode": 32768, "nlink": 1}
    db.record_output_file_progress(
        row["token"],
        "test-volume",
        expected_offset=0,
        written_bytes=0,
        source_identity=source if kind != "create" else None,
        stage_identity=stage if kind != "remove" else None,
    )
    if kind != "remove":
        db.record_output_file_progress(row["token"], "test-volume", expected_offset=0, written_bytes=4)
    return output, row, {**stage, "nlink": 2}


def recorded_commit(db, row, publication):
    assert hasattr(db, "apply_output_file_operation"), "DB-owned recorded output mutation is missing"
    return db.apply_output_file_operation(row["token"], "test-volume", publication_identity=publication)


@pytest.mark.parametrize("kind", ["create", "replace", "remove"])
def test_recorded_mutation_commits_only_intended_changes_and_phase(db, kind):
    output, row, publication = prepared_recorded_mutation(db, kind=kind)
    result = recorded_commit(db, row, publication if kind != "remove" else None)
    journal = db.get_output_file_operation(row["token"], "test-volume")
    assert journal["phase"] == "committed" and not journal["fs_done"]
    if kind == "remove":
        assert result is None
        with pytest.raises(KeyError):
            db.get_output_artifact(output.id)
        effects = json.loads(journal["effects_json"])
        assert effects[0]["incarnation"] == json.loads(row["original_json"])["incarnation"]
        assert journal["effects_pending"] == 1
    else:
        assert result.title == "Recorded title" and result.format == "html"
        assert result.storage_path == "destination.html"
        assert journal["output_id"] == result.id
        assert json.loads(journal["publication_identity_json"]) == publication
        if kind == "replace":
            assert result.id == output.id
            assert result.created_at == output.created_at and result.type == output.type
            assert json.loads(result.metadata_json)["keep"] == "unchanged"
        with pytest.raises(RuntimeError, match="^output_file_busy$"):
            db.update_output_artifact(result.id, title="Too soon")


@pytest.mark.parametrize(
    "problem", ["missing_publication", "wrong_inode", "wrong_links", "missing_stage", "missing_source"]
)
def test_recorded_mutation_rejects_unproved_publication(db, problem):
    output, row, publication = prepared_recorded_mutation(db)
    if problem == "missing_publication":
        publication = None
    elif problem == "wrong_inode":
        publication["ino"] += 1
    elif problem == "wrong_links":
        publication["nlink"] = 1
    else:
        column = "stage_identity_json" if problem == "missing_stage" else "source_identity_json"
        db.backend.execute(f"UPDATE output_file_operations SET {column} = NULL WHERE token = ?", (row["token"],))
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        recorded_commit(db, row, publication)
    assert db.get_output_artifact(output.id) == output
    assert db.get_output_file_operation(row["token"], "test-volume")["phase"] == "prepared"


@pytest.mark.parametrize(
    "kind,deleted,expected",
    [("create", False, 24), ("replace", False, 16), ("remove", False, 12), ("remove", True, 20)],
)
def test_recorded_audiobook_usage_joins_commit_once(db, kind, deleted, expected):
    output, row, publication = prepared_recorded_mutation(db, kind=kind, audiobook=True, deleted=deleted)
    db.set_audiobook_output_usage(20)
    result = recorded_commit(db, row, publication if kind != "remove" else None)
    assert db.get_audiobook_output_usage() == expected
    if result:
        assert json.loads(result.metadata_json)["byte_size"] == 4
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        recorded_commit(db, row, publication if kind != "remove" else None)
    assert db.get_audiobook_output_usage() == expected


def test_recorded_mutation_rolls_back_output_publication_and_usage_together(db, monkeypatch):
    output, row, publication = prepared_recorded_mutation(db, audiobook=True)
    db.set_audiobook_output_usage(20)
    update_usage = db.update_audiobook_output_usage

    def fail_after_accounting(delta, *, connection=None):
        assert connection is not None
        update_usage(delta, connection=connection)
        raise RuntimeError("simulated_commit_failure")

    monkeypatch.setattr(db, "update_audiobook_output_usage", fail_after_accounting)
    with pytest.raises(RuntimeError, match="^simulated_commit_failure$"):
        recorded_commit(db, row, publication)
    assert db.get_output_artifact(output.id) == output
    assert db.get_audiobook_output_usage() == 20
    journal = db.get_output_file_operation(row["token"], "test-volume")
    assert journal["phase"] == "prepared" and journal["publication_identity_json"] is None


def test_recorded_mutation_does_not_guess_missing_audiobook_usage(db):
    output, row, publication = prepared_recorded_mutation(db, audiobook=True)
    with pytest.raises(RuntimeError, match="^output_accounting_unavailable$"):
        recorded_commit(db, row, publication)
    assert db.get_audiobook_output_usage() is None
    assert db.get_output_artifact(output.id) == output


@pytest.mark.parametrize("problem", ["foreign_user", "wrong_volume", "expired", "aborted", "stale_snapshot"])
def test_recorded_commit_requires_current_scoped_authority(db, problem):
    output, row, publication = prepared_recorded_mutation(db)
    target, namespace = db, "test-volume"
    if problem == "foreign_user":
        target = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    elif problem == "wrong_volume":
        namespace = "other-volume"
    elif problem == "expired":
        db.backend.execute("UPDATE output_file_operations SET lease_until = 0 WHERE token = ?", (row["token"],))
    elif problem == "aborted":
        db.abort_output_file_operation(row["token"], namespace)
    else:
        # Incompatible/offline writer: normal APIs are already fenced.
        db.backend.execute("UPDATE outputs SET title = ? WHERE id = ?", ("Changed elsewhere", output.id))
    before = db.get_output_artifact(output.id)
    with pytest.raises((KeyError, RuntimeError)):
        target.apply_output_file_operation(row["token"], namespace, publication_identity=publication)
    assert db.get_output_artifact(output.id) == before
    assert db.get_output_file_operation(row["token"], "test-volume")["phase"] != "committed"


def test_recorded_create_rolls_back_if_allocated_sqlite_id_is_still_claimed(db):
    if db.backend.backend_type.value != "sqlite":
        pytest.skip("SQLite recycled rowid boundary")
    _, row, publication = prepared_recorded_mutation(db, kind="create")
    abandoned = db.create_output_artifact(type_="report", title="Old", format_="md", storage_path="old.md")
    removal = db.prepare_output_file_operation("test-volume", kind="remove", output_id=abandoned.id, lease_seconds=60)
    with db.commit_output_file_operation(removal["token"], "test-volume") as conn:
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (abandoned.id,), connection=conn)
    with pytest.raises(RuntimeError, match="^output_file_busy$"):
        recorded_commit(db, row, publication)
    assert db.backend.execute("SELECT COUNT(*) FROM outputs").scalar == 0
    assert db.get_output_file_operation(row["token"], "test-volume")["output_id"] is None


@pytest.mark.parametrize("usage", [2, 2**63 - 1])
def test_recorded_accounting_rejects_underflow_and_overflow(db, usage):
    kind = "replace" if usage == 2 else "create"
    output, row, publication = prepared_recorded_mutation(db, kind=kind, audiobook=True)
    db.set_audiobook_output_usage(usage)
    with pytest.raises(RuntimeError, match="^output_accounting_unavailable$"):
        recorded_commit(db, row, publication)
    assert db.get_audiobook_output_usage() == usage
    assert db.get_output_file_operation(row["token"], "test-volume")["phase"] == "prepared"


def test_recorded_removal_accounting_falls_back_to_recorded_size_not_file_io(db, monkeypatch):
    import tldw_Server_API.app.core.DB_Management.Collections_DB as module

    insert_binding(db)
    output = db.create_output_artifact(type_="audiobook_mp3", title="Old", format_="mp3", storage_path="old.mp3")
    row = db.prepare_output_file_operation(
        "test-volume", kind="remove", output_id=output.id, reserved_bytes=8, lease_seconds=60
    )
    db.record_output_file_progress(
        row["token"],
        "test-volume",
        expected_offset=0,
        written_bytes=0,
        source_identity={"dev": 1, "ino": 2, "mode": 32768, "nlink": 1, "size": 8, "mtime_ns": 1, "ctime_ns": 1},
    )
    db.set_audiobook_output_usage(8)
    monkeypatch.setattr(
        module, "_resolve_output_size_bytes", lambda *args: pytest.fail("filesystem access inside commit")
    )
    recorded_commit(db, row, None)
    assert db.get_audiobook_output_usage() == 0


def test_recorded_concurrent_commits_apply_one_accounting_delta(db):
    output, row, publication = prepared_recorded_mutation(db, audiobook=True)
    db.set_audiobook_output_usage(20)
    barrier = Barrier(2)

    def commit():
        adapter = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
        barrier.wait(timeout=10)
        try:
            adapter.apply_output_file_operation(row["token"], "test-volume", publication_identity=publication)
            return "committed"
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: commit(), range(2)))
    assert sorted(results) == ["committed", "output_operation_conflict"]
    assert db.get_audiobook_output_usage() == 16


@pytest.mark.parametrize("kind", ["replace", "remove"])
def test_recorded_commit_rejects_a_database_suppressed_mutation(db, monkeypatch, kind):
    output, row, publication = prepared_recorded_mutation(db, kind=kind)
    execute = db.backend.execute

    def suppress_output_write(query, params=(), **kwargs):
        # Model a database policy/trigger suppressing the row write without an error.
        if query.startswith("UPDATE outputs SET title") or query.startswith("DELETE FROM outputs WHERE user_id"):
            return SimpleNamespace(rowcount=0)
        return execute(query, params, **kwargs)

    monkeypatch.setattr(db.backend, "execute", suppress_output_write)
    with pytest.raises(DatabaseError, match="^output_mutation_failed$"):
        recorded_commit(db, row, publication if kind != "remove" else None)
    assert db.get_output_artifact(output.id) == output
    journal = db.get_output_file_operation(row["token"], "test-volume")
    assert journal["phase"] == "prepared" and journal["effects_pending"] == 0
