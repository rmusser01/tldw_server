"""Durable unadopted Reading artifact lifecycle against real database backends."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.services import reading_artifact_cleanup_service as service
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import make_reading

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


def reserve(db, tmp_path):
    namespace = service.provision_reading_storage_namespace(tmp_path)
    item = make_reading(db)
    reservation = db.reserve_reading_artifact(
        item.id, expected_revision=item.revision, storage_namespace_id=namespace, lease_until=int(time.time()) + 300
    )
    return item, namespace, reservation


def test_staging_reservation_survives_restart_without_advancing_parent(db, tmp_path):
    item, namespace, reservation = reserve(db, tmp_path)
    assert not (tmp_path / reservation["storage_path"]).exists()
    again = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    again.ensure_schema()
    assert again.get_reading_artifact(reservation["token"], namespace) == reservation
    assert again.get_content_item(item.id).revision == item.revision


def test_staging_write_and_failed_unlink_retry_survive_restart(db, tmp_path, monkeypatch):
    _, namespace, reservation = reserve(db, tmp_path)
    service.write_staged_reading_artifact(
        db, reservation["token"], output_root=tmp_path, storage_namespace_id=namespace, body="archive"
    )
    path = tmp_path / reservation["storage_path"]
    assert path.read_text() == "archive"
    db.cancel_reading_artifact(reservation["token"], namespace)
    unlink = os.unlink

    def fail(target, *args, **kwargs):
        if target == path.name and kwargs.get("dir_fd") is not None:
            raise PermissionError("private path and content")
        return unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail)
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    pending = db.get_reading_artifact(reservation["token"], namespace)
    assert pending["state"] == "pending"
    assert pending["last_error"] == "permission"
    assert pending["attempts"] == 1
    assert path.exists()
    monkeypatch.setattr(os, "unlink", unlink)
    again = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    monkeypatch.setattr(service.time, "time", lambda: pending["retry_after"] + 1)
    assert service.drain_reading_artifact_cleanup(again, output_root=tmp_path, storage_namespace_id=namespace) == 1
    assert not path.exists()
    with pytest.raises(KeyError):
        again.get_reading_artifact(reservation["token"], namespace)
    assert service.drain_reading_artifact_cleanup(again, output_root=tmp_path, storage_namespace_id=namespace) == 0


def test_delayed_writer_cannot_create_file_after_expired_reservation_retired(db, tmp_path, monkeypatch):
    _, namespace, reservation = reserve(db, tmp_path)
    waiting, resume = Event(), Event()
    lock = service._validated_storage_directory
    armed = True

    @contextmanager
    def delayed_lock(*args, **kwargs):
        nonlocal armed
        if armed:
            armed = False
            waiting.set()
            assert resume.wait(10)
        with lock(*args, **kwargs) as root:
            yield root

    monkeypatch.setattr(service, "_validated_storage_directory", delayed_lock)
    with ThreadPoolExecutor(max_workers=1) as workers:
        writer = workers.submit(
            service.write_staged_reading_artifact,
            db,
            reservation["token"],
            output_root=tmp_path,
            storage_namespace_id=namespace,
            body="late",
        )
        try:
            assert waiting.wait(10)
            db.backend.execute(
                "UPDATE reading_artifact_paths SET lease_until = 0 WHERE token = ?", (reservation["token"],)
            )
            assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1
        finally:
            resume.set()
        with pytest.raises(KeyError):
            writer.result(timeout=10)
    assert not (tmp_path / reservation["storage_path"]).exists()


def test_changed_parent_prevents_staged_file_creation(db, tmp_path):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import ReadingRevisionConflict

    item, namespace, reservation = reserve(db, tmp_path)
    db.update_content_item(item.id, title="newer")
    with pytest.raises(ReadingRevisionConflict):
        service.write_staged_reading_artifact(
            db, reservation["token"], output_root=tmp_path, storage_namespace_id=namespace, body="stale"
        )
    assert not (tmp_path / reservation["storage_path"]).exists()


@pytest.mark.parametrize("uppercase", [False, True])
def test_generic_output_create_and_move_cannot_attach_to_reserved_path(db, tmp_path, uppercase):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import ReadingArtifactOwnershipConflict

    _, namespace, reservation = reserve(db, tmp_path)
    kwargs = {"type_": "summary", "title": "other", "format_": "md"}
    path = reservation["storage_path"].upper() if uppercase else reservation["storage_path"]
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.create_output_artifact(storage_path=path, **kwargs)
    other = db.create_output_artifact(storage_path="other.md", **kwargs)
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.rename_output_artifact(other.id, "collision", path)
    assert db.get_output_artifact(other.id) == other
    db.cancel_reading_artifact(reservation["token"], namespace)
    with pytest.raises(ReadingArtifactOwnershipConflict):
        db.create_output_artifact(storage_path=path, **kwargs)


@pytest.mark.parametrize("uppercase", [False, True])
def test_cleanup_preserves_shared_output_reference(db, tmp_path, uppercase):
    _, namespace, reservation = reserve(db, tmp_path)
    service.write_staged_reading_artifact(
        db, reservation["token"], output_root=tmp_path, storage_namespace_id=namespace, body="shared"
    )
    output = db.create_output_artifact(type_="summary", title="shared", format_="md", storage_path="different.md")
    # Deliberately bypass new-writer guard to represent an existing/legacy reference.
    alias = reservation["storage_path"].upper() if uppercase else reservation["storage_path"]
    db.backend.execute("UPDATE outputs SET storage_path = ? WHERE id = ?", (alias, output.id))
    db.cancel_reading_artifact(reservation["token"], namespace)
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    assert (tmp_path / reservation["storage_path"]).read_text() == "shared"
    assert db.get_reading_artifact(reservation["token"], namespace)["last_error"] == "shared_output"


def test_cleanup_wrong_volume_and_missing_marker_leave_intents_unchanged(db, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _, namespace, reservation = reserve(db, first)
    other_namespace = service.provision_reading_storage_namespace(second)
    db.cancel_reading_artifact(reservation["token"], namespace)
    before = db.get_reading_artifact(reservation["token"], namespace)
    assert service.drain_reading_artifact_cleanup(db, output_root=second, storage_namespace_id=other_namespace) == 0
    with pytest.raises(service.ReadingStorageUnavailable):
        service.drain_reading_artifact_cleanup(db, output_root=second, storage_namespace_id=namespace)
    (first / ".reading-storage-namespace").unlink()
    with pytest.raises(service.ReadingStorageUnavailable):
        service.drain_reading_artifact_cleanup(db, output_root=first, storage_namespace_id=namespace)
    assert db.get_reading_artifact(reservation["token"], namespace) == before


def test_staged_filename_collision_is_blocked_not_deleted(db, tmp_path):
    _, namespace, reservation = reserve(db, tmp_path)
    path = tmp_path / reservation["storage_path"]
    path.write_text("not ours")
    with pytest.raises(FileExistsError):
        service.write_staged_reading_artifact(
            db, reservation["token"], output_root=tmp_path, storage_namespace_id=namespace, body="new"
        )
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    assert path.read_text() == "not ours"
    assert db.get_reading_artifact(reservation["token"], namespace)["last_error"] == "path_collision"


def test_cleanup_cannot_retire_a_writer_with_an_open_file(db, tmp_path, monkeypatch):
    _, namespace, reservation = reserve(db, tmp_path)
    path = tmp_path / reservation["storage_path"]
    opened, resume = Event(), Event()
    original_open = os.open

    def paused_open(target, *args, **kwargs):
        fd = original_open(target, *args, **kwargs)
        if target == path.name:
            opened.set()
            assert resume.wait(10)
        return fd

    monkeypatch.setattr(os, "open", paused_open)
    with ThreadPoolExecutor(max_workers=1) as workers:
        writer = workers.submit(
            service.write_staged_reading_artifact,
            db,
            reservation["token"],
            output_root=tmp_path,
            storage_namespace_id=namespace,
            body="archive",
        )
        try:
            assert opened.wait(10)
            db.backend.execute(
                "UPDATE reading_artifact_paths SET lease_until = 0 WHERE token = ?", (reservation["token"],)
            )
            with pytest.raises(service.ReadingStorageBusy):
                service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace)
            assert db.get_reading_artifact(reservation["token"], namespace)["state"] == "staged"
        finally:
            resume.set()
        writer.result(timeout=10)
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1


@pytest.mark.parametrize(
    "invalid_path",
    ["../outside.md", ".reading-storage-namespace", ".READING-STORAGE-NAMESPACE", ".READING-STORAGE.LOCK"],
)
def test_cleanup_blocks_invalid_paths(db, tmp_path, invalid_path):
    _, namespace, reservation = reserve(db, tmp_path)
    db.backend.execute(
        "UPDATE reading_artifact_paths SET storage_path = ? WHERE token = ?", (invalid_path, reservation["token"])
    )
    db.cancel_reading_artifact(reservation["token"], namespace)
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    assert db.get_reading_artifact(reservation["token"], namespace)["last_error"] == "invalid_path"
    assert service.provision_reading_storage_namespace(tmp_path) == namespace


def test_cleanup_sync_failure_keeps_intent_and_does_no_io_in_db_transaction(db, tmp_path, monkeypatch):
    _, namespace, reservation = reserve(db, tmp_path)
    service.write_staged_reading_artifact(
        db, reservation["token"], output_root=tmp_path, storage_namespace_id=namespace, body="archive"
    )
    db.cancel_reading_artifact(reservation["token"], namespace)
    transaction = db.transaction
    in_transaction = False

    @contextmanager
    def track_transaction():
        nonlocal in_transaction
        with transaction() as conn:
            in_transaction = True
            try:
                yield conn
            finally:
                in_transaction = False

    sync = service._sync_directory

    def fail_sync(root):
        assert not in_transaction
        raise OSError("not durable")

    monkeypatch.setattr(db, "transaction", track_transaction)
    monkeypatch.setattr(service, "_sync_directory", fail_sync)
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 0
    assert not (tmp_path / reservation["storage_path"]).exists()
    pending = db.get_reading_artifact(reservation["token"], namespace)
    assert pending["last_error"] == "io"
    monkeypatch.setattr(service, "_sync_directory", sync)
    monkeypatch.setattr(service.time, "time", lambda: pending["retry_after"] + 1)
    assert service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace) == 1


def test_staging_authorization_and_stale_revision_do_not_reserve_paths(db, tmp_path):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import ReadingRevisionConflict

    item, namespace, reservation = reserve(db, tmp_path)
    foreign = CollectionsDatabase.from_backend(user_id="781", backend=db.backend)
    with pytest.raises(KeyError):
        foreign.get_reading_artifact(reservation["token"], namespace)
    with pytest.raises(KeyError):
        foreign.reserve_reading_artifact(
            item.id, expected_revision=item.revision, storage_namespace_id=namespace, lease_until=1
        )
    db.update_content_item(item.id, title="changed")
    with pytest.raises(ReadingRevisionConflict):
        db.reserve_reading_artifact(
            item.id, expected_revision=item.revision, storage_namespace_id=namespace, lease_until=1
        )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_artifact_paths", ()).scalar == 1


@pytest.mark.parametrize("operation", ["write", "cleanup"])
def test_artifact_io_remains_on_locked_volume_when_root_path_is_replaced(db, tmp_path, monkeypatch, operation):
    root = tmp_path / "volume"
    root.mkdir()
    _, namespace, reservation = reserve(db, root)
    old_root = tmp_path / "original-volume"
    name = reservation["storage_path"]
    method = "validate_reading_artifact_for_write" if operation == "write" else "prepare_reading_artifact_cleanup"
    original = getattr(db, method)
    if operation == "cleanup":
        (root / name).write_text("ours")
        db.cancel_reading_artifact(reservation["token"], namespace)

    def replace_after_db_check(*args, **kwargs):
        result = original(*args, **kwargs)
        root.rename(old_root)
        root.mkdir()
        (root / name).write_text("replacement volume must survive")
        return result

    monkeypatch.setattr(db, method, replace_after_db_check)
    if operation == "write":
        service.write_staged_reading_artifact(
            db, reservation["token"], output_root=root, storage_namespace_id=namespace, body="ours"
        )
        assert (old_root / name).read_text() == "ours"
    else:
        assert service.drain_reading_artifact_cleanup(db, output_root=root, storage_namespace_id=namespace) == 1
        assert not (old_root / name).exists()
    assert (root / name).read_text() == "replacement volume must survive"


def test_cleanup_is_bounded_and_reservation_failure_rolls_back(db, tmp_path, monkeypatch):
    item, namespace, first = reserve(db, tmp_path)
    execute = db.backend.execute

    def abort_reservation(query, *args, **kwargs):
        result = execute(query, *args, **kwargs)
        if query.startswith("INSERT INTO reading_artifact_paths"):
            raise RuntimeError("abort reservation")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(db.backend, "execute", abort_reservation)
        with pytest.raises(RuntimeError, match="abort reservation"):
            db.reserve_reading_artifact(
                item.id, expected_revision=item.revision, storage_namespace_id=namespace, lease_until=1
            )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_artifact_paths", ()).scalar == 1
    db.cancel_reading_artifact(first["token"], namespace)
    db.reserve_reading_artifact(item.id, expected_revision=item.revision, storage_namespace_id=namespace, lease_until=1)
    assert (
        service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace, limit=1) == 1
    )
    assert db.backend.execute("SELECT COUNT(*) FROM reading_artifact_paths", ()).scalar == 1
    assert (
        service.drain_reading_artifact_cleanup(db, output_root=tmp_path, storage_namespace_id=namespace, limit=1) == 1
    )
