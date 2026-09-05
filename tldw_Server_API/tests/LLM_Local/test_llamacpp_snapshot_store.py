import errno
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import (
    Fingerprint,
    OperationReceipt,
    SnapshotMetadata,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_store import (
    SnapshotCorruptError,
    SnapshotStorageUnavailableError,
    SnapshotStore,
    SnapshotStoreError,
)


def fingerprint() -> Fingerprint:
    return Fingerprint(
        model_sha256="a" * 64,
        executable_sha256="b" * 64,
        effective_options_sha256="c" * 64,
        adapters_sha256="d" * 64,
    )


def metadata(payload: bytes, snapshot_id: str = "snap_1", sequence: int = 1) -> SnapshotMetadata:
    return SnapshotMetadata(
        profile_id="profile_1",
        snapshot_id=snapshot_id,
        source_slot=0,
        created_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        commit_sequence=sequence,
        byte_count=len(payload),
        token_count=42,
        sha256=hashlib.sha256(payload).hexdigest(),
        fingerprint=fingerprint(),
        actor_id="admin_1",
    )


def test_commit_publishes_verified_file_and_manifest_last(tmp_path: Path):
    payload = b"known slot cache"
    staged = tmp_path / "staged.bin"
    staged.write_bytes(payload)
    with SnapshotStore(tmp_path / "private") as store:
        result = store.commit("profile_1", staged, metadata(payload))

        assert result.sha256 == hashlib.sha256(payload).hexdigest()
        assert store.list("profile_1") == [result]
        restored = store.stage_restore("profile_1", "snap_1", tmp_path / "working")
        assert restored.read_bytes() == payload
        assert restored.name != "snap_1"
        assert restored.stat().st_mode & 0o077 == 0


def test_restore_rejects_corruption_and_removes_partial_output(tmp_path: Path):
    payload = b"cache"
    staged = tmp_path / "staged.bin"
    staged.write_bytes(payload)
    with SnapshotStore(tmp_path / "private") as store:
        store.commit("profile_1", staged, metadata(payload))
        (tmp_path / "private/profile_1/snapshots/snap_1.bin").write_bytes(b"wrong")
        working = tmp_path / "working"

        with pytest.raises(SnapshotCorruptError):
            store.stage_restore("profile_1", "snap_1", working)

        assert list(working.iterdir()) == []


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", ".", "", "two words"])
def test_ids_cannot_traverse_storage(tmp_path: Path, bad_id: str):
    with SnapshotStore(tmp_path / "private") as store:
        with pytest.raises((SnapshotStoreError, ValueError)):
            store.list(bad_id)


def test_symlink_root_and_staged_file_fail_closed(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SnapshotStoreError):
        SnapshotStore(linked_root)

    real = tmp_path / "real.bin"
    real.write_bytes(b"cache")
    staged = tmp_path / "staged.bin"
    staged.symlink_to(real)
    with SnapshotStore(tmp_path / "private") as store:
        with pytest.raises(SnapshotStoreError):
            store.commit("profile_1", staged, metadata(b"cache"))
    assert real.read_bytes() == b"cache"


@pytest.mark.parametrize("target", ["root", "staged", "working"])
def test_symlink_ancestors_fail_closed_without_touching_outside(tmp_path: Path, target: str):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    marker = outside / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    ancestor = tmp_path / "ancestor"
    ancestor.symlink_to(outside, target_is_directory=True)

    if target == "root":
        with pytest.raises(SnapshotStoreError):
            SnapshotStore(ancestor / "snapshots")
    else:
        with SnapshotStore(tmp_path / "private") as store:
            payload = b"cache"
            staged = tmp_path / "staged.bin"
            staged.write_bytes(payload)
            if target == "staged":
                (outside / "staged.bin").write_bytes(payload)
                with pytest.raises(SnapshotStoreError):
                    store.commit("profile_1", ancestor / "staged.bin", metadata(payload))
            else:
                store.commit("profile_1", staged, metadata(payload))
                with pytest.raises(SnapshotStoreError):
                    store.stage_restore("profile_1", "snap_1", ancestor / "working")

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_ancestor_swap_before_directory_creation_never_writes_outside(tmp_path: Path, monkeypatch):
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir(mode=0o700)
    displaced = tmp_path / "displaced"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = ancestor / "store"
    original_mkdir = os.mkdir
    swapped = False

    def swap_then_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and Path(path).name == "store":
            ancestor.rename(displaced)
            ancestor.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", swap_then_mkdir)
    with pytest.raises(SnapshotStoreError):
        SnapshotStore(root)

    assert swapped is True
    assert not (outside / "store").exists()


def test_oversized_and_incomplete_manifests_are_not_catalog_entries(tmp_path: Path):
    with SnapshotStore(tmp_path / "private") as store:
        profile = tmp_path / "private/profile_1"
        manifests = profile / "manifests"
        snapshots = profile / "snapshots"
        manifests.mkdir(parents=True, mode=0o700)
        snapshots.mkdir(mode=0o700)
        (snapshots / "orphan.bin").write_bytes(b"orphan")
        (manifests / "huge.json").write_bytes(b"x" * (1024 * 1024 + 1))
        (manifests / "partial.json").write_text("{", encoding="utf-8")
        for path in manifests.iterdir():
            path.chmod(0o600)

        assert store.list("profile_1") == []


def test_catalog_and_receipt_propagate_operational_read_errors(tmp_path: Path, monkeypatch):
    payload = b"cache"
    staged = tmp_path / "staged.bin"
    staged.write_bytes(payload)
    receipt = OperationReceipt(
        profile_id="profile_1",
        operation_id="operation_1",
        launch_generation="launch_1",
        request_digest="e" * 64,
        kind="save",
        state="validating",
    )
    with SnapshotStore(tmp_path / "private") as store:
        store.commit("profile_1", staged, metadata(payload))
        store.write_receipt(receipt)
        original_read = os.read

        def fail_read(fd: int, size: int) -> bytes:
            if size <= 1024 * 1024:
                raise OSError(errno.EIO, "I/O error")
            return original_read(fd, size)

        monkeypatch.setattr(os, "read", fail_read)
        with pytest.raises(SnapshotStorageUnavailableError):
            store.list("profile_1")
        with pytest.raises(SnapshotStorageUnavailableError):
            store.read_receipt("profile_1", "operation_1")


def test_disk_full_before_publication_preserves_previous_snapshot(tmp_path: Path, monkeypatch):
    first = b"first"
    staged = tmp_path / "first.bin"
    staged.write_bytes(first)
    with SnapshotStore(tmp_path / "private") as store:
        store.commit("profile_1", staged, metadata(first, "snap_1", 1))
        second = tmp_path / "second.bin"
        second.write_bytes(b"second")
        monkeypatch.setattr(store, "_write_chunk", lambda *_: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))

        with pytest.raises(OSError):
            store.commit("profile_1", second, metadata(b"second", "snap_2", 2))

        assert [item.snapshot_id for item in store.list("profile_1")] == ["snap_1"]


@pytest.mark.parametrize("failure_boundary", ["binary_rename", "manifest_publish"])
def test_interrupted_publication_never_lists_incomplete_snapshot(tmp_path: Path, monkeypatch, failure_boundary: str):
    first = b"first"
    staged = tmp_path / "first.bin"
    staged.write_bytes(first)
    with SnapshotStore(tmp_path / "private") as store:
        store.commit("profile_1", staged, metadata(first, "snap_1", 1))
        second = tmp_path / "second.bin"
        second.write_bytes(b"second")
        if failure_boundary == "binary_rename":
            monkeypatch.setattr(os, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupt")))
        else:
            monkeypatch.setattr(
                store,
                "_publish_json",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupt")),
            )

        with pytest.raises(OSError, match="interrupt"):
            store.commit("profile_1", second, metadata(b"second", "snap_2", 2))

        assert [item.snapshot_id for item in store.list("profile_1")] == ["snap_1"]
        assert (tmp_path / "private/profile_1/snapshots/snap_1.bin").read_bytes() == first


@pytest.mark.parametrize(
    ("failure_boundary", "expected_ids"),
    [
        ("copy", ["snap_1"]),
        ("file_fsync", ["snap_1"]),
        ("binary_rename", ["snap_1"]),
        ("directory_fsync", ["snap_1"]),
        ("manifest_write", ["snap_1"]),
        ("manifest_fsync", ["snap_1"]),
        ("manifest_rename", ["snap_1"]),
        ("manifest_directory_fsync", ["snap_2", "snap_1"]),
    ],
)
def test_every_commit_boundary_preserves_previous_catalog_entry(
    tmp_path: Path, monkeypatch, failure_boundary: str, expected_ids: list[str]
):
    first = b"first"
    staged = tmp_path / "first.bin"
    staged.write_bytes(first)
    with SnapshotStore(tmp_path / "private") as store:
        store.commit("profile_1", staged, metadata(first, "snap_1", 1))
        second = tmp_path / "second.bin"
        second.write_bytes(b"second")

        def interrupt(boundary: str) -> None:
            if boundary == failure_boundary:
                raise OSError("interrupted")

        monkeypatch.setattr(store, "_checkpoint", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            store.commit("profile_1", second, metadata(b"second", "snap_2", 2))

        assert [item.snapshot_id for item in store.list("profile_1")] == expected_ids
        assert (tmp_path / "private/profile_1/snapshots/snap_1.bin").read_bytes() == first


def test_prune_uses_sequence_not_rollback_timestamp_and_reports_failures(tmp_path: Path, monkeypatch):
    with SnapshotStore(tmp_path / "private") as store:
        for snapshot_id, sequence in (("new_clock", 1), ("old_clock", 2), ("newest", 3)):
            staged = tmp_path / f"{snapshot_id}.bin"
            staged.write_bytes(snapshot_id.encode())
            item = metadata(snapshot_id.encode(), snapshot_id, sequence)
            if snapshot_id == "old_clock":
                item = item.model_copy(update={"created_at": datetime(2020, 1, 1, tzinfo=timezone.utc)})
            store.commit("profile_1", staged, item)
        original_delete = store.delete

        def fail_one(profile_id: str, snapshot_id: str) -> None:
            if snapshot_id == "new_clock":
                raise OSError("busy")
            original_delete(profile_id, snapshot_id)

        monkeypatch.setattr(store, "delete", fail_one)
        assert store.prune("profile_1", 1) == ["new_clock"]
        assert {item.snapshot_id for item in store.list("profile_1")} == {"new_clock", "newest"}


def test_receipt_round_trip_is_private_and_path_free(tmp_path: Path):
    receipt = OperationReceipt(
        profile_id="profile_1",
        operation_id="operation_1",
        launch_generation="launch_1",
        request_digest="e" * 64,
        kind="save",
        state="validating",
    )
    with SnapshotStore(tmp_path / "private") as store:
        store.write_receipt(receipt)
        assert store.read_receipt("profile_1", "operation_1") == receipt
        assert (tmp_path / "private").stat().st_mode & 0o077 == 0


def test_second_process_owner_is_rejected(tmp_path: Path):
    first = SnapshotStore(tmp_path / "private")
    try:
        with pytest.raises(SnapshotStoreError, match="owned"):
            SnapshotStore(tmp_path / "private")
    finally:
        first.close()


def test_closed_store_rejects_operations_after_new_owner_acquires_root(tmp_path: Path):
    first = SnapshotStore(tmp_path / "private")
    first.close()
    with SnapshotStore(tmp_path / "private") as replacement:
        with pytest.raises(SnapshotStoreError, match="closed"):
            first.list("profile_1")
        assert replacement.list("profile_1") == []
