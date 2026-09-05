"""Real-filesystem exclusion prerequisites for durable Reading cleanup."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(os.name != "posix", reason="POSIX storage-lock contract; unsupported support is fault-injected"),
]


def _hold_storage_lock(root, namespace, ready, release):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import reading_storage_lock

    with reading_storage_lock(Path(root), storage_namespace_id=namespace):
        ready.set()
        if not release.wait(15):
            raise RuntimeError("test release timed out")


def test_storage_provisioning_is_explicit_idempotent_and_distinguishes_roots(tmp_path):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
        provision_reading_storage_namespace,
        reading_storage_lock,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    namespace = provision_reading_storage_namespace(first)
    assert namespace == provision_reading_storage_namespace(first)
    assert namespace != provision_reading_storage_namespace(second)
    with reading_storage_lock(first, storage_namespace_id=namespace) as root:
        assert root == first
    assert sorted(path.name for path in first.iterdir()) == [".reading-storage-namespace", ".reading-storage.lock"]
    for path in first.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("problem", ["root", "marker", "lock", "mismatch", "invalid_marker"])
def test_storage_runtime_fails_closed_without_recreating_missing_state(tmp_path, problem):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
        ReadingStorageUnavailable,
        provision_reading_storage_namespace,
        reading_storage_lock,
    )

    namespace = provision_reading_storage_namespace(tmp_path)
    root = tmp_path
    if problem == "root":
        root = tmp_path / "missing"
    elif problem in ("marker", "lock"):
        name = ".reading-storage-namespace" if problem == "marker" else ".reading-storage.lock"
        (root / name).unlink()
    elif problem == "mismatch":
        namespace = "0" * 32
    else:
        (root / ".reading-storage-namespace").write_text("malformed private data")
    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(ReadingStorageUnavailable, match="^reading_storage_unavailable$"):
        with reading_storage_lock(root, storage_namespace_id=namespace):
            pytest.fail("unavailable storage was accepted")
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_provisioning_will_not_replace_a_missing_lock_on_an_initialized_volume(tmp_path):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
        ReadingStorageUnavailable,
        provision_reading_storage_namespace,
    )

    provision_reading_storage_namespace(tmp_path)
    lock = tmp_path / ".reading-storage.lock"
    lock.unlink()
    with pytest.raises(ReadingStorageUnavailable):
        provision_reading_storage_namespace(tmp_path)
    assert not lock.exists()


@pytest.mark.parametrize("name", [".reading-storage.lock", ".reading-storage-namespace"])
@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "hardlink"])
def test_storage_rejects_nonprivate_regular_files(tmp_path, name, kind):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
        ReadingStorageUnavailable,
        provision_reading_storage_namespace,
        reading_storage_lock,
    )

    namespace = provision_reading_storage_namespace(tmp_path)
    target = tmp_path / name
    backup = tmp_path / "private-backup"
    target.rename(backup)
    if kind == "symlink":
        target.symlink_to(backup)
    elif kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        os.link(backup, target)
    with pytest.raises(ReadingStorageUnavailable):
        with reading_storage_lock(tmp_path, storage_namespace_id=namespace):
            pytest.fail("unsafe marker/lock accepted")


@pytest.mark.parametrize("crash", [False, True])
def test_storage_lock_excludes_other_processes_and_survives_process_exit(tmp_path, crash):
    from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
        ReadingStorageBusy,
        provision_reading_storage_namespace,
        reading_storage_lock,
    )

    namespace = provision_reading_storage_namespace(tmp_path)
    inode = (tmp_path / ".reading-storage.lock").stat().st_ino
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    holder = context.Process(target=_hold_storage_lock, args=(str(tmp_path), namespace, ready, release))
    holder.start()
    try:
        assert ready.wait(10)
        with pytest.raises(ReadingStorageBusy):
            with reading_storage_lock(tmp_path, storage_namespace_id=namespace):
                pytest.fail("two processes acquired exclusive storage lock")
        if crash:
            holder.terminate()
        else:
            release.set()
        holder.join(10)
        assert not holder.is_alive()
        with reading_storage_lock(tmp_path, storage_namespace_id=namespace):
            assert (tmp_path / ".reading-storage.lock").stat().st_ino == inode
    finally:
        if holder.is_alive():
            holder.terminate()
        holder.join(10)


def test_storage_rechecks_marker_after_lock_acquisition(tmp_path, monkeypatch):
    from tldw_Server_API.app.services import reading_artifact_cleanup_service as service

    namespace = service.provision_reading_storage_namespace(tmp_path)
    flock = service.fcntl.flock

    def remove_marker_on_acquire(fd, flags):
        flock(fd, flags)
        if flags & service.fcntl.LOCK_EX:
            (tmp_path / ".reading-storage-namespace").unlink()

    monkeypatch.setattr(service.fcntl, "flock", remove_marker_on_acquire)
    with pytest.raises(service.ReadingStorageUnavailable):
        with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
            pytest.fail("removed marker accepted after acquisition")


def test_storage_rejects_replaced_lock_inode(tmp_path, monkeypatch):
    from tldw_Server_API.app.services import reading_artifact_cleanup_service as service

    namespace = service.provision_reading_storage_namespace(tmp_path)
    flock = service.fcntl.flock

    def replace_lock_on_acquire(fd, flags):
        flock(fd, flags)
        if flags & service.fcntl.LOCK_EX:
            lock = tmp_path / ".reading-storage.lock"
            lock.rename(tmp_path / "old-lock")
            lock.touch(mode=0o600)

    monkeypatch.setattr(service.fcntl, "flock", replace_lock_on_acquire)
    with pytest.raises(service.ReadingStorageUnavailable):
        with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
            pytest.fail("replacement lock accepted")


def test_storage_without_os_lock_support_is_unavailable(tmp_path, monkeypatch):
    from tldw_Server_API.app.services import reading_artifact_cleanup_service as service

    monkeypatch.setattr(service, "fcntl", None)
    with pytest.raises(service.ReadingStorageUnavailable):
        service.provision_reading_storage_namespace(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_provisioning_retry_cannot_bypass_failed_durability(tmp_path, monkeypatch):
    from tldw_Server_API.app.services import reading_artifact_cleanup_service as service

    fsync = os.fsync

    def fail_sync(fd):
        raise OSError("sensitive mount diagnostic")

    monkeypatch.setattr(os, "fsync", fail_sync)
    for _ in range(2):
        with pytest.raises(service.ReadingStorageUnavailable, match="^reading_storage_unavailable$"):
            service.provision_reading_storage_namespace(tmp_path)
    marker = (tmp_path / ".reading-storage-namespace").read_text()
    monkeypatch.setattr(os, "fsync", fsync)
    assert service.provision_reading_storage_namespace(tmp_path) == marker.strip()


def test_storage_missing_marker_recovers_only_with_the_original_identity(tmp_path):
    from tldw_Server_API.app.services import reading_artifact_cleanup_service as service

    namespace = service.provision_reading_storage_namespace(tmp_path)
    marker = tmp_path / ".reading-storage-namespace"
    backup = tmp_path / "unmounted-marker"
    marker.rename(backup)
    with pytest.raises(service.ReadingStorageUnavailable):
        with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
            pytest.fail("missing volume marker accepted")
    backup.rename(marker)
    with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
        assert marker.read_text().strip() == namespace


def test_storage_lock_releases_on_caller_error_without_rewriting_it(tmp_path):
    from tldw_Server_API.app.services import reading_artifact_cleanup_service as service

    namespace = service.provision_reading_storage_namespace(tmp_path)
    failure = OSError("caller operation failed")
    with pytest.raises(OSError) as exc:
        with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
            raise failure
    assert exc.value is failure
    with service.reading_storage_lock(tmp_path, storage_namespace_id=namespace):
        pass
