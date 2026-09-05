"""Immediate post-commit cleanup shares the verified publication interval."""

from __future__ import annotations

import asyncio
import errno
import os
from threading import Event

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.services.reading_artifact_cleanup_service import ReadingStorageBusy, reading_storage_lock
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


@pytest.mark.parametrize("kind", ["create", "replace", "remove"])
def test_confirmed_commit_finishes_files_before_releasing_exclusion(db, storage, monkeypatch, kind):
    root, namespace, original = storage
    writer = service(db, storage)
    row = run(
        writer.prepare,
        kind=kind,
        output_id=original.id if kind != "create" else None,
        destination_path="destination.md" if kind != "remove" else None,
        max_output_bytes=0,
        intended={"title": "New", "type": "report", "format": "md"} if kind == "create" else None,
    )
    real_unlink = os.unlink
    removed = []

    def unlink(name, **kwargs):
        with pytest.raises(ReadingStorageBusy):
            with reading_storage_lock(root, storage_namespace_id=namespace):
                pytest.fail("cleanup released publication exclusion")
        assert db.get_output_file_operation(row["token"], namespace)["phase"] == "committed"
        removed.append(name)
        return real_unlink(name, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    result = run(writer.publish_and_commit, row["token"])
    assert removed
    assert not list(root.glob(".output-stage-*"))
    if kind != "create":
        assert not (root / "source.md").exists()
    if kind == "remove":
        assert result is None
        pending = db.get_output_file_operation(row["token"], namespace)
        assert pending["fs_done"] and pending["effects_pending"] == 1 and pending["reserved_bytes"] == 0
    else:
        assert result.storage_path == "destination.md"
        assert (root / "destination.md").stat().st_nlink == 1
        with pytest.raises(KeyError):
            db.get_output_file_operation(row["token"], namespace)


@pytest.mark.parametrize("fault", ["unlink", "sync", "begin_db", "finish_db", "finish_ack", "source_identity"])
def test_cleanup_failure_does_not_reject_confirmed_commit(db, storage, monkeypatch, fault):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    real_apply, real_finish = db.apply_output_file_operation, db.finish_output_file_operation
    faults = []

    def failure(*args, **kwargs):
        faults.append(fault)
        if fault in {"begin_db", "finish_db"}:
            raise DatabaseError("private database details")
        raise OSError(errno.EIO, "private filesystem details")

    def lost_finish(*args, **kwargs):
        real_finish(*args, **kwargs)
        faults.append(fault)
        raise DatabaseError("private commit acknowledgement")

    def apply(*args, **kwargs):
        result = real_apply(*args, **kwargs)
        if fault == "unlink":
            monkeypatch.setattr(os, "unlink", failure)
        elif fault == "sync":
            monkeypatch.setattr(os, "fsync", failure)
        elif fault == "begin_db":
            monkeypatch.setattr(db, "begin_output_file_recovery", failure)
        elif fault == "finish_db":
            monkeypatch.setattr(db, "finish_output_file_operation", failure)
        elif fault == "finish_ack":
            monkeypatch.setattr(db, "finish_output_file_operation", lost_finish)
        else:
            faults.append(fault)
            (root / "source.md").write_bytes(b"changed after commit")
        return result

    monkeypatch.setattr(db, "apply_output_file_operation", apply)
    result = run(writer.publish_and_commit, row["token"])
    assert faults, "the post-commit cleanup boundary was not reached"
    assert result.storage_path == "destination.md"
    assert db.get_output_artifact(original.id) == result
    assert (root / "destination.md").exists()
    if fault == "finish_ack":
        with pytest.raises(KeyError):
            db.get_output_file_operation(row["token"], namespace)
    else:
        pending = db.get_output_file_operation(row["token"], namespace)
        assert pending["phase"] == "committed" and not pending["fs_done"] and pending["reserved_bytes"]
        if fault == "source_identity":
            assert pending["last_error"] == "output_identity_unconfirmed"
            assert (root / "source.md").read_bytes() == b"changed after commit"
        elif fault in {"unlink", "sync"}:
            assert pending["last_error"] == "output_storage_unavailable"


@pytest.mark.parametrize("failure_kind", ["status_write", "unexpected"])
def test_cleanup_reporting_failure_is_sanitized_and_recoverable(db, storage, monkeypatch, failure_kind):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    real_apply = db.apply_output_file_operation
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))

    def unlink(*args, **kwargs):
        if failure_kind == "unexpected":
            raise ValueError("private filesystem details")
        raise OSError(errno.EIO, "private filesystem details")

    def report(*args, **kwargs):
        raise DatabaseError("private database details")

    try:
        with monkeypatch.context() as patch:

            def apply(*args, **kwargs):
                result = real_apply(*args, **kwargs)
                patch.setattr(os, "unlink", unlink)
                patch.setattr(db, "record_output_file_recovery_failure", report)
                return result

            patch.setattr(db, "apply_output_file_operation", apply)
            result = run(writer.publish_and_commit, row["token"])
    finally:
        logger.remove(sink)
    assert result == db.get_output_artifact(original.id)
    cleanup_messages = [message for message in messages if "Output post-commit cleanup" in message]
    assert cleanup_messages and all("private" not in message for message in cleanup_messages)
    pending = db.get_output_file_operation(row["token"], namespace)
    assert pending["phase"] == "committed" and not pending["fs_done"] and pending["reserved_bytes"]
    assert run(writer.recover_due)["finished"] == 1
    assert not (root / "source.md").exists() and not (root / row["stage_path"]).exists()
    assert (root / "destination.md").stat().st_nlink == 1


def test_cancellation_during_post_commit_cleanup_drains_exclusion(db, storage, monkeypatch):
    root, namespace, original = storage
    writer = service(db, storage)
    row = prepare(writer, original)
    entered, release = Event(), Event()
    real_unlink = os.unlink

    def unlink(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_unlink(*args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)

    async def exercise():
        task = asyncio.create_task(writer.publish_and_commit(row["token"]))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            with pytest.raises(ReadingStorageBusy):
                with reading_storage_lock(root, storage_namespace_id=namespace):
                    pytest.fail("cancelled cleanup released exclusion early")
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    run(exercise)
    assert db.get_output_artifact(original.id).storage_path == "destination.md"
    assert not (root / "source.md").exists() and not (root / row["stage_path"]).exists()
    with pytest.raises(KeyError):
        db.get_output_file_operation(row["token"], namespace)
    with reading_storage_lock(root, storage_namespace_id=namespace):
        assert (root / "destination.md").stat().st_nlink == 1
