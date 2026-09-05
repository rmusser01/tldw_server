"""Restart recovery against real files and both production DB adapters."""

from __future__ import annotations

import asyncio
import errno
import os
from threading import Event
from unittest.mock import patch

import pytest

from tldw_Server_API.tests.Collections.test_output_file_operations_storage import (
    db as db,
)
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import (
    prepare,
    run,
    service,
)
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import (
    storage as storage,
)

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "posix", reason="POSIX file protocol")]
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def publish_before_interruption(writer, token):
    """Leave a real committed mutation at the pre-cleanup crash boundary."""
    real_apply = writer.db.apply_output_file_operation
    outputs = []

    class Interrupted(BaseException):
        pass

    def commit_then_interrupt(*args, **kwargs):
        outputs.append(real_apply(*args, **kwargs))
        raise Interrupted()

    with patch.object(writer.db, "apply_output_file_operation", commit_then_interrupt):
        with pytest.raises(Interrupted):
            run(writer.publish_and_commit, token)
    return outputs[0]


def aborted_publication(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    run(writer.write_chunk, row["token"], b"replacement", expected_offset=0)
    os.link(root / row["stage_path"], root / row["destination_path"])
    db.abort_output_file_operation(row["token"], namespace)
    return writer, db.get_output_file_operation(row["token"], namespace)


def test_recovery_abort_syncs_destination_before_unlinking_witness(db, storage, monkeypatch):
    root, namespace, original = storage
    writer, row = aborted_publication(db, storage)
    real_unlink, real_fsync = os.unlink, os.fsync
    operations = []

    def unlink(name, **kwargs):
        assert kwargs["dir_fd"] is not None
        operations.append(name)
        return real_unlink(name, **kwargs)

    def fsync(fd):
        if os.fstat(fd).st_ino == root.stat().st_ino:
            operations.append("sync")
        return real_fsync(fd)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "fsync", fsync)
    assert run(writer.recover_due)["finished"] == 1
    assert operations[:4] == [row["destination_path"], "sync", row["stage_path"], "sync"]
    assert (root / "source.md").read_bytes() == b"original"
    assert db.get_output_artifact(original.id) == original
    with pytest.raises(KeyError):
        db.get_output_file_operation(row["token"], namespace)


def test_recovery_commit_preserves_destination_and_cleans_old_source(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    run(writer.write_chunk, row["token"], b"replacement", expected_offset=0)
    committed = publish_before_interruption(writer, row["token"])
    assert run(writer.recover_due)["finished"] == 1
    assert (root / "destination.md").read_bytes() == b"replacement"
    assert (root / "destination.md").stat().st_nlink == 1
    assert not (root / "source.md").exists() and not (root / row["stage_path"]).exists()
    assert db.get_output_artifact(original.id) == committed
    with pytest.raises(KeyError):
        db.get_output_file_operation(row["token"], namespace)
    assert run(writer.recover_due)["finished"] == 0


def test_recovery_live_lease_untouched_then_expired_producer_cannot_resume(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    assert run(writer.recover_due)["finished"] == 0
    assert db.get_output_file_operation(row["token"], namespace)["phase"] == "prepared"
    db.backend.execute("UPDATE output_file_operations SET lease_until = 0 WHERE token = ?", (row["token"],))
    assert run(writer.recover_due)["finished"] == 1
    with pytest.raises(KeyError):
        run(writer.write_chunk, row["token"], b"late", expected_offset=0)
    assert not (root / row["stage_path"]).exists()
    assert (root / "source.md").read_bytes() == b"original"


@pytest.mark.parametrize(
    "problem", ["preidentity", "foreign_destination", "missing_witness", "stage_replaced", "extra_bytes", "extra_link"]
)
def test_recovery_blocks_unproved_abort_files_without_deletion(db, storage, problem):
    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)
    stage, destination = root / row["stage_path"], root / row["destination_path"]
    if problem == "preidentity":
        db.backend.execute(
            "UPDATE output_file_operations SET stage_identity_json = NULL WHERE token = ?", (row["token"],)
        )
    elif problem == "foreign_destination":
        destination.unlink()
        destination.write_bytes(b"foreign")
    elif problem == "missing_witness":
        stage.unlink()
    elif problem == "stage_replaced":
        stage.unlink()
        stage.write_bytes(b"replacement")
    elif problem == "extra_bytes":
        stage.write_bytes(b"unacknowledged")
    else:
        os.link(stage, root / "extra")
    before = {p.name: p.lstat() for p in root.iterdir()}
    assert run(writer.recover_due)["blocked"] == 1
    assert {p.name: p.lstat() for p in root.iterdir()} == before
    pending = db.get_output_file_operation(row["token"], namespace)
    assert not pending["fs_done"] and pending["reserved_bytes"]
    assert pending["last_error"] == "output_identity_unconfirmed"
    assert run(writer.recover_due)["blocked"] == 0  # Blocked work is not age-swept.


@pytest.mark.parametrize("alias", ["source.md", "SOURCE.md", "/legacy/path/source.md", r"C:\legacy\source.md"])
def test_recovery_committed_source_with_surviving_reference_is_preserved(db, storage, alias):
    root, _, original = storage
    shared = db.create_output_artifact(type_="report", title="Shared", format_="md", storage_path="shared.md")
    # Historical rows predate today's normalized-path insertion boundary.
    db.backend.execute("UPDATE outputs SET storage_path = ? WHERE id = ?", (alias, shared.id))
    shared = db.get_output_artifact(shared.id)
    writer = service(db, storage)
    row = prepare(writer, original)
    publish_before_interruption(writer, row["token"])
    assert run(writer.recover_due)["finished"] == 1
    assert (root / "source.md").read_bytes() == b"original"
    assert db.get_output_artifact(shared.id) == shared


def test_recovery_remove_releases_filesystem_claim_but_keeps_history_effect(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = run(writer.prepare, kind="remove", output_id=original.id, max_output_bytes=0)
    publish_before_interruption(writer, row["token"])
    assert run(writer.recover_due)["finished"] == 1
    pending = db.get_output_file_operation(row["token"], namespace)
    assert pending["fs_done"] and pending["reserved_bytes"] == 0 and pending["effects_pending"] == 1
    assert not (root / "source.md").exists()
    (root / "source.md").write_bytes(b"reused")
    assert run(writer.recover_due)["finished"] == 0
    assert (root / "source.md").read_bytes() == b"reused"


def test_recovery_changed_committed_source_is_never_unlinked(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    publish_before_interruption(writer, row["token"])
    (root / "source.md").write_bytes(b"different")
    assert run(writer.recover_due)["blocked"] == 1
    assert (root / "source.md").read_bytes() == b"different"
    assert (root / "destination.md").exists()
    assert not db.get_output_file_operation(row["token"], namespace)["fs_done"]


def test_recovery_sync_failure_retains_witness_and_retry_authority(db, storage, monkeypatch):
    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)
    real_fsync = os.fsync

    def failed_sync(fd):
        raise OSError(errno.EIO, "private storage details")

    monkeypatch.setattr(os, "fsync", failed_sync)
    assert run(writer.recover_due)["retry"] == 1
    assert not (root / row["destination_path"]).exists()
    assert (root / row["stage_path"]).exists()
    pending = db.get_output_file_operation(row["token"], namespace)
    assert pending["last_error"] == "output_storage_unavailable" and pending["attempts"] == 1
    assert pending["retry_after"] > 0 and not pending["fs_done"]
    monkeypatch.setattr(os, "fsync", real_fsync)
    db.backend.execute("UPDATE output_file_operations SET retry_after = 0 WHERE token = ?", (row["token"],))
    assert run(writer.recover_due)["finished"] == 1


def test_recovery_missing_volume_is_not_absence_success(db, storage):
    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)
    root.rename(root.with_name("detached"))
    assert run(writer.recover_due)["retry"] == 1
    pending = db.get_output_file_operation(row["token"], namespace)
    assert not pending["fs_done"] and pending["reserved_bytes"]
    assert not root.exists()


@pytest.mark.parametrize("kind", ["aborting", "committed"])
@pytest.mark.parametrize("boundary", ["first_unlink", "second_unlink", "finish_before", "finish_after"])
def test_recovery_restart_after_cleanup_interruptions(db, storage, monkeypatch, kind, boundary):
    root, namespace, original = storage
    if kind == "aborting":
        writer, row = aborted_publication(db, storage)
    else:
        writer = service(db, storage)
        row = prepare(writer, original)
        publish_before_interruption(writer, row["token"])
    real_unlink, real_finish = os.unlink, db.finish_output_file_operation
    calls = []

    class Crash(BaseException):
        pass

    def unlink(*args, **kwargs):
        real_unlink(*args, **kwargs)
        calls.append(args[0])
        if len(calls) == (1 if boundary == "first_unlink" else 2) and boundary.endswith("unlink"):
            raise Crash()

    def finish(*args, **kwargs):
        if boundary == "finish_before":
            raise Crash()
        result = real_finish(*args, **kwargs)
        if boundary == "finish_after":
            raise Crash()
        return result

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(db, "finish_output_file_operation", finish)
    with pytest.raises(Crash):
        run(writer.recover_due)
    monkeypatch.setattr(os, "unlink", real_unlink)
    monkeypatch.setattr(db, "finish_output_file_operation", real_finish)
    run(service(db, storage).recover_due)
    with pytest.raises(KeyError):
        db.get_output_file_operation(row["token"], namespace)
    assert not (root / row["stage_path"]).exists()
    if kind == "aborting":
        assert (root / "source.md").read_bytes() == b"original"
        assert not (root / "destination.md").exists()
    else:
        assert (root / "destination.md").stat().st_nlink == 1
        assert not (root / "source.md").exists()


@pytest.mark.parametrize("changed", ["missing", "replaced", "symlink", "hardlink"])
def test_recovery_unproved_committed_destination_keeps_witness_and_source(db, storage, changed):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    publish_before_interruption(writer, row["token"])
    dest = root / "destination.md"
    if changed != "hardlink":
        dest.unlink()
    if changed == "replaced":
        dest.write_bytes(b"foreign")
    elif changed == "symlink":
        dest.symlink_to(root / "source.md")
    elif changed == "hardlink":
        os.link(dest, root / "extra")
    assert run(writer.recover_due)["blocked"] == 1
    assert (root / row["stage_path"]).exists()
    assert (root / "source.md").read_bytes() == b"original"
    assert db.get_output_file_operation(row["token"], namespace)["reserved_bytes"]


def test_recovery_batch_limit_and_post_selection_due_recheck(db, storage, monkeypatch):
    root, namespace, original = storage
    writer, first = aborted_publication(db, storage)
    second = run(
        writer.prepare,
        kind="create",
        destination_path="second.md",
        max_output_bytes=0,
        intended={"title": "Second", "type": "report", "format": "md"},
    )
    db.abort_output_file_operation(second["token"], namespace)
    assert run(writer.recover_due, limit=1)["finished"] == 1
    remaining = db.list_due_output_file_operations(namespace)
    assert len(remaining) == 1
    real_list = db.list_due_output_file_operations

    def defer_after_selection(*args, **kwargs):
        result = real_list(*args, **kwargs)
        db.backend.execute("UPDATE output_file_operations SET retry_after = ? WHERE token = ?", (2**63 - 1, result[0]))
        return result

    monkeypatch.setattr(db, "list_due_output_file_operations", defer_after_selection)
    assert run(writer.recover_due)["skipped"] == 1
    assert not db.get_output_file_operation(remaining[0], namespace)["fs_done"]
    assert db.get_output_artifact(original.id) == original
    assert (root / "source.md").read_bytes() == b"original"


def test_recovery_empty_and_history_only_batches_never_open_volume(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)
    row = run(writer.prepare, kind="remove", output_id=original.id, max_output_bytes=0)
    publish_before_interruption(writer, row["token"])
    run(writer.recover_due)

    def forbidden(*args, **kwargs):
        pytest.fail("history-only batch inspected filesystem")

    monkeypatch.setattr(os, "open", forbidden)
    assert run(writer.recover_due) == {"finished": 0, "blocked": 0, "retry": 0, "skipped": 0}
    assert db.get_output_file_operation(row["token"], namespace)["effects_pending"] == 1


def test_recovery_busy_lock_retains_prepared_lease(db, storage):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import reading_storage_lock

    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)
    with reading_storage_lock(root, storage_namespace_id=namespace):
        assert run(writer.recover_due)["retry"] == 1
    pending = db.get_output_file_operation(row["token"], namespace)
    assert pending["last_error"] == "output_storage_busy" and not pending["fs_done"]
    assert (root / row["stage_path"]).exists()


def test_recovery_cancellation_drains_unlink_without_aborting_other_work(db, storage, monkeypatch):
    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)
    entered, release = Event(), Event()
    real_unlink = os.unlink

    def delayed_unlink(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_unlink(*args, **kwargs)

    monkeypatch.setattr(os, "unlink", delayed_unlink)

    async def exercise():
        task = asyncio.create_task(writer.recover_due())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    run(exercise)
    assert not (root / row["stage_path"]).exists()
    with pytest.raises(KeyError):
        db.get_output_file_operation(row["token"], namespace)


@pytest.mark.parametrize("boundary", ["select", "begin", "finish"])
def test_recovery_database_failure_is_unconfirmed_without_releasing_claims(db, storage, monkeypatch, boundary):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import DatabaseError

    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)

    def failed(*args, **kwargs):
        raise DatabaseError("private database details")

    method = {
        "select": "list_due_output_file_operations",
        "begin": "begin_output_file_recovery",
        "finish": "finish_output_file_operation",
    }[boundary]
    monkeypatch.setattr(db, method, failed)
    with pytest.raises(RuntimeError, match="^output_update_unconfirmed$"):
        run(writer.recover_due)
    assert not db.get_output_file_operation(row["token"], namespace)["fs_done"]
    assert (root / "source.md").read_bytes() == b"original"


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_recovery_rejects_unbounded_batch_before_file_access(db, storage, limit):
    writer, row = aborted_publication(db, storage)
    with pytest.raises(ValueError, match="^output_operation_invalid$"):
        run(writer.recover_due, limit=limit)
    assert (storage[0] / row["stage_path"]).exists()


@pytest.mark.parametrize("late_category", ["output_storage_busy", "output_storage_unavailable"])
def test_recovery_delayed_retry_cannot_downgrade_identity_block(db, storage, late_category):
    root, namespace, _ = storage
    writer, row = aborted_publication(db, storage)
    (root / row["stage_path"]).write_bytes(b"ambiguous unacknowledged write")
    assert run(writer.recover_due)["blocked"] == 1
    blocked = db.get_output_file_operation(row["token"], namespace)
    # A second worker selected this token before the first acquired exclusion,
    # then its busy/IO result arrived after the identity block was persisted.
    db.record_output_file_recovery_failure(row["token"], namespace, late_category)
    assert db.get_output_file_operation(row["token"], namespace) == blocked
