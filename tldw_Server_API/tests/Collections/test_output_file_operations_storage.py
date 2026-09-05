"""Real-volume staging and journal evidence, without runtime activation."""

from __future__ import annotations

import asyncio
import errno
import importlib
import json
import os
from functools import partial
from threading import Event
from types import SimpleNamespace

import anyio
import pytest

from tldw_Server_API.app.services.reading_artifact_cleanup_service import provision_reading_storage_namespace
from tldw_Server_API.tests.Collections.test_output_file_operations_db import insert_binding
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "posix", reason="POSIX file protocol")]
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture
def storage(db, tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    namespace = provision_reading_storage_namespace(root)
    insert_binding(
        db,
        storage_namespace_id=namespace,
        operation_bytes=16 * 1024 * 1024,
        user_pending_bytes=32 * 1024 * 1024,
        free_space_margin_bytes=1,
    )
    original = db.create_output_artifact(type_="report", title="Original", format_="md", storage_path="source.md")
    (root / "source.md").write_bytes(b"original")
    return root, namespace, original


def service(db, storage):
    name = "tldw_Server_API.app.services.output_file_operations"
    assert importlib.util.find_spec(name) is not None, "bounded output storage service is missing"
    return importlib.import_module(name).OutputFileOperations(
        db, output_root=storage[0], storage_namespace_id=storage[1]
    )


def run(method, *args, **kwargs):
    return anyio.run(partial(method, *args, **kwargs))


def prepare(writer, original, **kwargs):
    return run(
        writer.prepare,
        kind="replace",
        output_id=original.id,
        destination_path="destination.md",
        max_output_bytes=1024 * 1024,
        intended={"title": "Destination"},
        **kwargs,
    )


def test_prepare_reserves_before_create_and_persists_file_evidence(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)
    real_open = os.open

    def checked_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            row = db.backend.execute("SELECT * FROM output_file_operations WHERE user_id = ?", (db.user_id,)).first
            assert row and row["stage_path"] == path
            assert json.loads(row["source_identity_json"])["size"] == 8
            assert flags & os.O_EXCL and flags & os.O_NOFOLLOW
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", checked_open)
    operation = prepare(writer, original)
    row = db.get_output_file_operation(operation["token"], namespace)
    assert json.loads(row["stage_identity_json"])["ino"] == (root / row["stage_path"]).stat().st_ino
    assert row["reserved_bytes"] == 1024 * 1024 + 8
    assert (root / "source.md").read_bytes() == b"original"
    assert not (root / "destination.md").exists()
    assert db.get_output_artifact(original.id) == original


@pytest.mark.parametrize("problem", ["missing", "symlink", "fifo", "hardlink", "destination"])
def test_prepare_rejects_unsafe_files_without_changing_output(db, storage, problem):
    root, _, original = storage
    writer = service(db, storage)
    source = root / "source.md"
    if problem in {"missing", "symlink", "fifo"}:
        source.unlink()
    if problem == "symlink":
        (root / "elsewhere").write_bytes(b"elsewhere")
        source.symlink_to(root / "elsewhere")
    elif problem == "fifo":
        os.mkfifo(source)
    elif problem == "hardlink":
        os.link(source, root / "alias.md")
    elif problem == "destination":
        (root / "destination.md").write_bytes(b"occupied")
    with pytest.raises(RuntimeError, match="^output_(source_unavailable|path_conflict)$"):
        prepare(writer, original)
    assert not list(root.glob(".output-stage-*"))
    assert db.get_output_artifact(original.id) == original
    if problem == "destination":
        assert (root / "destination.md").read_bytes() == b"occupied"


def test_bounded_write_persists_offset_and_preserves_source(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    operation = prepare(writer, original)
    assert run(writer.write_chunk, operation["token"], b"new", expected_offset=0) == 3
    assert run(writer.write_chunk, operation["token"], b" body", expected_offset=3) == 8
    row = db.get_output_file_operation(operation["token"], namespace)
    assert row["written_bytes"] == 8
    assert (root / row["stage_path"]).read_bytes() == b"new body"
    assert (root / "source.md").read_bytes() == b"original"
    assert db.get_output_artifact(original.id) == original


@pytest.mark.parametrize("problem", ["source_changed", "stage_changed", "offset", "expired", "aborted", "oversize"])
def test_resumed_write_rejects_lost_authority(db, storage, problem):
    root, namespace, original = storage
    writer = service(db, storage)
    operation = prepare(writer, original)
    token = operation["token"]
    stage = root / operation["stage_path"]
    data, offset = b"new", 0
    if problem == "source_changed":
        (root / "source.md").write_bytes(b"changed!")
    elif problem == "stage_changed":
        stage.rename(root / "old-stage")
        stage.write_bytes(b"")
    elif problem == "offset":
        offset = 1
    elif problem == "expired":
        db.backend.execute("UPDATE output_file_operations SET lease_until = 0 WHERE token = ?", (token,))
    elif problem == "aborted":
        db.abort_output_file_operation(token, namespace)
    else:
        data = b"x" * (1024 * 1024 + 1)
    with pytest.raises((RuntimeError, ValueError)):
        run(writer.write_chunk, token, data, expected_offset=offset)
    assert stage.read_bytes() == b""
    assert db.get_output_artifact(original.id) == original


def test_other_volume_cannot_resume_same_database_operation(db, storage, tmp_path):
    root, namespace, original = storage
    writer = service(db, storage)
    operation = prepare(writer, original)
    other = tmp_path / "other"
    other.mkdir()
    provision_reading_storage_namespace(other)
    other_writer = type(writer)(db, output_root=other, storage_namespace_id=namespace)
    with pytest.raises(RuntimeError, match="storage_unavailable"):
        run(other_writer.write_chunk, operation["token"], b"wrong", expected_offset=0)
    assert (root / operation["stage_path"]).read_bytes() == b""


def test_journal_evidence_is_immutable_and_offset_is_compare_and_set(db, storage):
    _, namespace, original = storage
    assert hasattr(db, "record_output_file_progress"), "guarded file evidence transition is missing"
    operation = db.prepare_output_file_operation(
        namespace,
        kind="replace",
        output_id=original.id,
        destination_path="destination.md",
        reserved_bytes=20,
        lease_seconds=60,
    )
    token = operation["token"]
    source = {"dev": 1, "ino": 2, "mode": 32768, "nlink": 1, "size": 8, "mtime_ns": 1, "ctime_ns": 1}
    stage = {"dev": 1, "ino": 3, "mode": 32768, "nlink": 1}
    db.record_output_file_progress(
        token, namespace, source_identity=source, stage_identity=stage, expected_offset=0, written_bytes=0
    )
    for changes in (
        {"source_identity": {**source, "ino": 4}},
        {"stage_identity": {**stage, "ino": 4}},
        {"expected_offset": 1},
        {"written_bytes": 13},
    ):
        with pytest.raises((RuntimeError, ValueError)):
            db.record_output_file_progress(token, namespace, **({"expected_offset": 0, "written_bytes": 0} | changes))
    assert db.get_output_file_operation(token, namespace)["written_bytes"] == 0


def test_prepare_sync_failure_is_sanitized_and_retains_preidentity_claim(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)

    def fail_sync(fd):
        raise OSError(errno.ENOSPC, "sensitive mount path")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare(writer, original)
    row = dict(db.backend.execute("SELECT * FROM output_file_operations").first)
    assert row["phase"] == "aborting" and not row["fs_done"]
    assert row["stage_identity_json"] is None
    assert (root / row["stage_path"]).exists()
    assert (root / "source.md").read_bytes() == b"original"


def test_capacity_check_precedes_reservation_or_file_creation(db, storage, monkeypatch):
    root, _, original = storage
    writer = service(db, storage)
    monkeypatch.setattr(os, "fstatvfs", lambda fd: SimpleNamespace(f_bavail=1, f_frsize=1))
    with pytest.raises(RuntimeError, match="^output_storage_capacity$"):
        prepare(writer, original)
    assert db.backend.execute("SELECT COUNT(*) FROM output_file_operations").scalar == 0
    assert not list(root.glob(".output-stage-*"))


def test_cumulative_write_limit_counts_source_separately(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = run(
        writer.prepare, kind="replace", output_id=original.id, destination_path="destination.md", max_output_bytes=4
    )
    run(writer.write_chunk, row["token"], b"123", expected_offset=0)
    with pytest.raises(RuntimeError, match="^output_size_limit$"):
        run(writer.write_chunk, row["token"], b"45", expected_offset=3)
    assert (root / row["stage_path"]).read_bytes() == b"123"
    assert db.get_output_file_operation(row["token"], namespace)["written_bytes"] == 3


@pytest.mark.parametrize("fault", ["partial_write", "fsync", "offset_ack", "identity_ack"])
def test_failed_write_keeps_source_and_durable_claims(db, storage, monkeypatch, fault):
    root, namespace, original = storage
    writer = service(db, storage)
    record = db.record_output_file_progress

    def fail_record(*args, **kwargs):
        if (fault == "identity_ack" and kwargs.get("stage_identity") is not None) or (
            fault == "offset_ack" and kwargs.get("written_bytes", 0) > 0
        ):
            raise RuntimeError("simulated_database_unavailable")
        return record(*args, **kwargs)

    monkeypatch.setattr(db, "record_output_file_progress", fail_record)
    if fault == "identity_ack":
        with pytest.raises(RuntimeError):
            prepare(writer, original)
        row = dict(db.backend.execute("SELECT * FROM output_file_operations").first)
    else:
        row = prepare(writer, original)
        real_write = os.write

        def fail_write(fd, data):
            real_write(fd, data[:1])
            raise OSError(errno.ENOSPC, "private directory")

        def fail_sync(fd):
            raise OSError(errno.EIO, "private directory")

        if fault == "partial_write":
            monkeypatch.setattr(os, "write", fail_write)
        elif fault == "fsync":
            monkeypatch.setattr(os, "fsync", fail_sync)
        with pytest.raises(RuntimeError):
            run(writer.write_chunk, row["token"], b"new bytes", expected_offset=0)
    current = db.get_output_file_operation(row["token"], namespace)
    assert current["phase"] == "aborting" and not current["fs_done"]
    assert current["written_bytes"] == 0
    assert (root / row["stage_path"]).exists()
    assert (root / "source.md").read_bytes() == b"original"
    assert db.get_output_artifact(original.id) == original


def test_unacknowledged_bytes_cannot_be_truncated_or_resumed(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    (root / row["stage_path"]).write_bytes(b"unacknowledged")
    with pytest.raises(RuntimeError, match="^output_operation_conflict$"):
        run(writer.write_chunk, row["token"], b"new", expected_offset=0)
    assert (root / row["stage_path"]).read_bytes() == b"unacknowledged"
    assert not db.get_output_file_operation(row["token"], namespace)["fs_done"]


def test_cancel_waits_for_offloaded_write_to_close_before_returning(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    entered, release = Event(), Event()
    real_write = os.write

    def paused_write(fd, data):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test release timed out")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", paused_write)

    async def scenario():
        task = asyncio.create_task(writer.write_chunk(row["token"], b"new", expected_offset=0))
        try:
            assert await anyio.to_thread.run_sync(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done(), "cancellation detached a live writable descriptor"
        finally:
            release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass

    anyio.run(scenario)
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import reading_storage_lock

    with reading_storage_lock(root, storage_namespace_id=namespace):
        assert (root / row["stage_path"]).read_bytes() == b"new"
    assert db.get_output_file_operation(row["token"], namespace)["phase"] == "aborting"


@pytest.mark.parametrize("kind", ["create", "remove"])
def test_create_and_remove_preparation_have_only_their_recorded_files(db, storage, kind):
    root, namespace, original = storage
    writer = service(db, storage)
    kwargs = {"destination_path": "new.md"} if kind == "create" else {"output_id": original.id}
    row = run(writer.prepare, kind=kind, max_output_bytes=4 if kind == "create" else 0, **kwargs)
    if kind == "create":
        run(writer.write_chunk, row["token"], b"body", expected_offset=0)
        assert (root / row["stage_path"]).read_bytes() == b"body"
        assert row["source_identity_json"] is None
    else:
        assert row["stage_path"] is None
        assert json.loads(row["source_identity_json"])["size"] == 8
    assert not (root / "new.md").exists()
    assert (root / "source.md").read_bytes() == b"original"


def test_removal_preparation_includes_soft_deleted_outputs(db, storage):
    root, namespace, original = storage
    writer = service(db, storage)
    db.delete_output_artifact(original.id, hard=False)
    row = run(writer.prepare, kind="remove", output_id=original.id, max_output_bytes=0)
    assert row["kind"] == "remove"
    assert (root / "source.md").read_bytes() == b"original"
    assert db.get_output_artifact(original.id, include_deleted=True) == original
    with pytest.raises(KeyError, match="output_not_found"):
        db.get_output_artifact(original.id)


def test_anyio_cancellation_aborts_only_after_write_closes(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    entered, release = Event(), Event()
    real_write = os.write

    def paused_write(fd, data):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test release timed out")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", paused_write)

    async def scenario():
        async with anyio.create_task_group() as group:
            group.start_soon(partial(writer.write_chunk, row["token"], b"new", expected_offset=0))
            try:
                assert await anyio.to_thread.run_sync(entered.wait, 5)
                group.cancel_scope.cancel()
            finally:
                release.set()

    anyio.run(scenario)
    assert db.get_output_file_operation(row["token"], namespace)["phase"] == "aborting"
    assert (root / row["stage_path"]).read_bytes() == b"new"


def test_root_replacement_after_validation_stays_on_held_descriptor(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    validate = db.validate_output_file_operation
    moved = root.with_name("original-volume")

    def replace_root(*args, **kwargs):
        result = validate(*args, **kwargs)
        root.rename(moved)
        root.mkdir()
        (root / row["stage_path"]).write_bytes(b"replacement")
        return result

    monkeypatch.setattr(db, "validate_output_file_operation", replace_root)
    run(writer.write_chunk, row["token"], b"new", expected_offset=0)
    assert (root / row["stage_path"]).read_bytes() == b"replacement"
    assert (moved / row["stage_path"]).read_bytes() == b"new"


@pytest.mark.parametrize("bad_identity", [{}, {"dev": True}, {"dev": -1}, {"mode": 0}, {"nlink": 2}])
def test_journal_rejects_malformed_file_evidence(db, storage, bad_identity):
    _, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    identity = json.loads(row["stage_identity_json"])
    identity = identity | bad_identity if bad_identity else {}
    with pytest.raises(ValueError, match="^output_operation_invalid$"):
        db.record_output_file_progress(
            row["token"], namespace, stage_identity=identity, expected_offset=0, written_bytes=0
        )
    assert db.get_output_file_operation(row["token"], namespace)["stage_identity_json"] == row["stage_identity_json"]


@pytest.mark.parametrize("phase", ["prepare", "write"])
def test_free_space_probe_errors_do_not_expose_paths(db, storage, monkeypatch, phase):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original) if phase == "write" else None

    def unavailable(fd):
        raise OSError(errno.EIO, "sensitive directory")

    monkeypatch.setattr(os, "fstatvfs", unavailable)
    with pytest.raises(RuntimeError, match="^output_storage_unavailable$"):
        if row:
            run(writer.write_chunk, row["token"], b"new", expected_offset=0)
        else:
            prepare(writer, original)
    assert (root / "source.md").read_bytes() == b"original"
