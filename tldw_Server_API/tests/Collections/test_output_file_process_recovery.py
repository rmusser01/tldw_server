"""Actual process death at output protocol boundaries, using fresh DB connections."""

from __future__ import annotations

import multiprocessing
import os
import signal
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.services.output_file_operations import OutputFileOperations
from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
    ReadingStorageBusy,
    reading_storage_lock,
)
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import (
    db as db,
)
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import (
    run,
    service,
)
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import (
    storage as storage,
)

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.name != "posix", reason="POSIX kill/lock protocol")]
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def _crash_boundary_worker(config, user_id, root, namespace, output_id, boundary, control):
    """Spawn, never fork a live DB connection; pause only after the real operation."""
    backend = DatabaseBackendFactory.create_backend(config)
    db = CollectionsDatabase.from_backend(user_id=user_id, backend=backend)
    writer = OutputFileOperations(db, output_root=Path(root), storage_namespace_id=namespace)
    state = {}

    def pause(at):
        if boundary == at:
            control.send((at, state["token"]))
            control.recv()  # Parent kills this process; no semaphore is reused afterward.
            raise AssertionError("crash boundary unexpectedly resumed")

    real_prepare = db.prepare_output_file_operation
    real_progress = db.record_output_file_progress
    real_apply = db.apply_output_file_operation
    real_finish = db.finish_output_file_operation
    real_open, real_fsync, real_link, real_unlink = os.open, os.fsync, os.link, os.unlink

    def prepare(*args, **kwargs):
        row = real_prepare(*args, **kwargs)
        state.update(row)
        pause("reserved")
        return row

    def opened(name, flags, *args, **kwargs):
        fd = real_open(name, flags, *args, **kwargs)
        if name == state.get("stage_path") and flags & os.O_CREAT:
            pause("stage_created")
        return fd

    def progress(*args, **kwargs):
        row = real_progress(*args, **kwargs)
        if row["stage_identity_json"] is not None and row["written_bytes"] == 0:
            pause("stage_recorded")
        return row

    def synced(fd):
        real_fsync(fd)
        if os.fstat(fd).st_size == len(b"replacement"):
            pause("write_synced")

    def linked(*args, **kwargs):
        real_link(*args, **kwargs)
        pause("linked")

    def committed(*args, **kwargs):
        result = real_apply(*args, **kwargs)
        pause("committed")
        return result

    def unlinked(name, **kwargs):
        real_unlink(name, **kwargs)
        if name == state["stage_path"]:
            pause("abort_witness" if boundary.startswith("abort") else "commit_witness")
        elif name == "destination.md":
            pause("abort_destination")
        elif name == "source.md":
            pause("source_unlinked")

    def finished(*args, **kwargs):
        pause("finish_before")
        result = real_finish(*args, **kwargs)
        pause("finish_after")
        return result

    with pytest.MonkeyPatch.context() as patch:
        for name, function in (
            ("prepare_output_file_operation", prepare),
            ("record_output_file_progress", progress),
            ("apply_output_file_operation", committed),
            ("finish_output_file_operation", finished),
        ):
            patch.setattr(db, name, function)
        for name, function in (("open", opened), ("fsync", synced), ("link", linked), ("unlink", unlinked)):
            patch.setattr(os, name, function)
        row = run(
            writer.prepare,
            kind="replace",
            output_id=output_id,
            destination_path="destination.md",
            max_output_bytes=1024,
            intended={"title": "Replacement"},
        )
        run(writer.write_chunk, row["token"], b"replacement", expected_offset=0)
        if boundary.startswith("abort"):
            with reading_storage_lock(Path(root), storage_namespace_id=namespace):
                real_link(Path(root) / row["stage_path"], Path(root) / "destination.md")
                db.abort_output_file_operation(row["token"], namespace)
        else:
            run(writer.publish_and_commit, row["token"])
        run(writer.recover_due)
    raise AssertionError(f"boundary {boundary} was not reached")


@contextmanager
def spawned_worker(db, storage, target, *args):
    """Bounded spawn and cleanup; never signal a killed process's synchronization primitive."""
    root, namespace, original = storage
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=target, args=(db.backend.config, db.user_id, str(root), namespace, original.id, *args, child)
    )
    process.start()
    child.close()
    try:
        yield process, parent
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        parent.close()
        assert not process.is_alive()
        process.close()


@pytest.mark.parametrize(
    "boundary",
    [
        "reserved",
        "stage_created",
        "stage_recorded",
        "write_synced",
        "linked",
        "committed",
        "abort_destination",
        "abort_witness",
        "commit_witness",
        "source_unlinked",
        "finish_before",
        "finish_after",
    ],
)
def test_killed_output_process_recovers_only_proved_files(db, storage, boundary):
    root, namespace, original = storage
    lock_inode = (root / ".reading-storage.lock").stat().st_ino
    with spawned_worker(db, storage, _crash_boundary_worker, boundary) as (process, control):
        assert control.poll(30), f"worker did not reach {boundary}; exit={process.exitcode}"
        reached, token = control.recv()
        assert reached == boundary
        with pytest.raises(ReadingStorageBusy):
            with reading_storage_lock(root, storage_namespace_id=namespace):
                pytest.fail("another process entered the active file interval")
        process.kill()
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
        with reading_storage_lock(root, storage_namespace_id=namespace):
            assert (root / ".reading-storage.lock").stat().st_ino == lock_inode
        try:
            row = db.get_output_file_operation(token, namespace)
        except KeyError:
            assert boundary == "finish_after"
            row = None
        if row and row["phase"] == "prepared":
            assert row["written_bytes"] == (11 if boundary == "linked" else 0)
            if boundary == "reserved":
                assert not (root / row["stage_path"]).exists()
            elif boundary == "stage_created":
                assert row["stage_identity_json"] is None
            elif boundary == "write_synced":
                assert row["stage_identity_json"] is not None
                assert (root / row["stage_path"]).read_bytes() == b"replacement"
            db.backend.execute("UPDATE output_file_operations SET lease_until = 0 WHERE token = ?", (token,))
        result = run(service(db, storage).recover_due)
        if boundary in {"stage_created", "write_synced"}:
            assert result["blocked"] == 1
            blocked = db.get_output_file_operation(token, namespace)
            assert blocked["reserved_bytes"] and not blocked["fs_done"]
            assert (root / blocked["stage_path"]).exists()
        else:
            with pytest.raises(KeyError):
                db.get_output_file_operation(token, namespace)
            assert not list(root.glob(".output-stage-*")), "retired operation left an orphan private file"
        committed = boundary in {"committed", "commit_witness", "source_unlinked", "finish_before", "finish_after"}
        if committed:
            assert (root / "destination.md").read_bytes() == b"replacement"
            assert (root / "destination.md").stat().st_nlink == 1
            assert not (root / "source.md").exists()
            assert db.get_output_artifact(original.id).storage_path == "destination.md"
        else:
            assert (root / "source.md").read_bytes() == b"original"
            assert db.get_output_artifact(original.id) == original
            assert not (root / "destination.md").exists()


def _late_producer(config, user_id, root, namespace, output_id, token, control):
    backend = DatabaseBackendFactory.create_backend(config)
    db = CollectionsDatabase.from_backend(user_id=user_id, backend=backend)
    writer = OutputFileOperations(db, output_root=Path(root), storage_namespace_id=namespace)
    control.send("ready")
    assert control.recv() == "resume"
    try:
        run(writer.write_chunk, token, b"late", expected_offset=0)
    except (KeyError, RuntimeError) as exc:
        control.send(("rejected", exc.args[0]))
    else:
        control.send(("wrote", None))
    backend.get_pool().close_all()


@pytest.mark.parametrize("retire", [False, True])
def test_other_process_producer_cannot_resume_after_abort_or_retirement(db, storage, retire):
    root, namespace, original = storage
    writer = service(db, storage)
    row = run(writer.prepare, kind="replace", output_id=original.id, destination_path="late.md", max_output_bytes=16)
    with spawned_worker(db, storage, _late_producer, row["token"]) as (process, control):
        assert control.poll(30)
        assert control.recv() == "ready"
        if retire:
            db.backend.execute("UPDATE output_file_operations SET lease_until = 0 WHERE token = ?", (row["token"],))
            assert run(writer.recover_due)["finished"] == 1
        else:
            db.abort_output_file_operation(row["token"], namespace)
        control.send("resume")
        assert control.poll(30)
        assert control.recv() == ("rejected", "output_operation_not_found" if retire else "output_operation_conflict")
        process.join(10)
        assert process.exitcode == 0
        if not retire:
            assert run(writer.recover_due)["finished"] == 1
        assert not (root / row["stage_path"]).exists()
        assert not (root / "late.md").exists()
        assert (root / "source.md").read_bytes() == b"original"


def _cooperative_copy(config, user_id, root, namespace, output_id, control):
    import tracemalloc

    backend = DatabaseBackendFactory.create_backend(config)
    db = CollectionsDatabase.from_backend(user_id=user_id, backend=backend)
    writer = OutputFileOperations(db, output_root=Path(root), storage_namespace_id=namespace)
    size = (Path(root) / "source.md").stat().st_size
    row = run(writer.prepare, kind="replace", output_id=output_id, destination_path="copied.md", max_output_bytes=size)
    real_write = writer.write_chunk

    async def write(token, data, *, expected_offset):
        assert len(data) <= 1024 * 1024
        offset = await real_write(token, data, expected_offset=expected_offset)
        control.send(("chunk", row["stage_path"], offset))
        assert control.recv() == "continue"  # Outside storage exclusion and writable-FD lifetime.
        return offset

    writer.write_chunk = write
    tracemalloc.start()
    written = run(writer.copy_source, row["token"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    control.send(("finished", written, peak))
    backend.get_pool().close_all()


def test_large_copy_has_bounded_memory_and_cross_process_reader_progress(db, storage):
    root, namespace, original = storage
    size = 12 * 1024 * 1024
    with (root / "source.md").open("wb") as source:
        chunk = b"x" * (1024 * 1024)
        for _ in range(12):
            source.write(chunk)
    db.backend.execute(
        "UPDATE output_storage_bindings SET operation_bytes = ?, user_pending_bytes = ? WHERE user_id = ?",
        (32 * 1024 * 1024, 64 * 1024 * 1024, db.user_id),
    )
    offsets = []
    with spawned_worker(db, storage, _cooperative_copy) as (process, control):
        while True:
            assert control.poll(30), f"copy child stopped progressing; exit={process.exitcode}"
            message = control.recv()
            if message[0] == "finished":
                assert message[1] == size
                assert message[2] < 8 * 1024 * 1024, f"copy allocation peak was {message[2]}"
                break
            _, stage, offset = message
            with reading_storage_lock(root, storage_namespace_id=namespace):
                assert (root / stage).stat().st_size == offset
                with (root / "source.md").open("rb") as source:
                    assert source.read(16) == b"x" * 16
            if not offsets:
                # Exclusion is available between chunks, but durable row/path
                # claims must still reject a competing process's mutation.
                with pytest.raises(RuntimeError, match="^output_file_busy$"):
                    run(
                        service(db, storage).prepare,
                        kind="replace",
                        output_id=original.id,
                        destination_path="competing.md",
                        max_output_bytes=16,
                    )
                assert not (root / "competing.md").exists()
                peer = service(db, storage)
                operation = run(
                    peer.prepare,
                    kind="create",
                    destination_path="peer.md",
                    max_output_bytes=4,
                    intended={"title": "Peer", "type": "report", "format": "md"},
                )
                run(peer.write_chunk, operation["token"], b"peer", expected_offset=0)
                published = run(peer.publish_and_commit, operation["token"])
                assert run(peer.recover_due)["finished"] == 1
                assert db.get_output_artifact(published.id).storage_path == "peer.md"
                assert (root / "peer.md").read_bytes() == b"peer"
            offsets.append(offset)
            control.send("continue")
        process.join(10)
        assert process.exitcode == 0
    assert offsets == list(range(1024 * 1024, size + 1, 1024 * 1024))
    assert not (root / "copied.md").exists()
