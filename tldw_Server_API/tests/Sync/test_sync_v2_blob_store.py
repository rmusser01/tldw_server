from __future__ import annotations

import hashlib
import inspect
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import tldw_Server_API.app.core.Sync.v2.blob_store as blob_store_module
from tldw_Server_API.app.core.Sync.v2.blob_store import (
    LocalSyncBlobStore,
    SyncBlobStoreError,
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_local_blob_store_commits_verified_chunks_atomically(tmp_path: Path):
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    first = b"a" * 8
    second = b"b" * 8

    first_key = store.write_upload_chunk(
        upload_id="upload-1",
        chunk_index=0,
        payload=first,
        expected_hash=_sha256(first),
    )
    second_key = store.write_upload_chunk(
        upload_id="upload-1",
        chunk_index=1,
        payload=second,
        expected_hash=_sha256(second),
    )
    final_key = store.commit_upload(
        upload_id="upload-1",
        payload_hash=_sha256(first + second),
        chunk_indexes=[0, 1],
    )

    assert first_key.endswith("0.part")
    assert second_key.endswith("1.part")
    assert store.read_blob(final_key) == first + second
    assert (tmp_path / "sync_blobs" / "_uploads" / "upload-1").exists()
    store.discard_upload("upload-1")
    assert not (tmp_path / "sync_blobs" / "_uploads" / "upload-1").exists()


def test_local_blob_store_streams_commit_and_read_without_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    first = b"a" * 8
    second = b"b" * 8
    store.write_upload_chunk(
        upload_id="upload-1",
        chunk_index=0,
        payload=first,
        expected_hash=_sha256(first),
    )
    store.write_upload_chunk(
        upload_id="upload-1",
        chunk_index=1,
        payload=second,
        expected_hash=_sha256(second),
    )

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("blob store should stream files instead of using read_bytes")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    final_key = store.commit_upload(
        upload_id="upload-1",
        payload_hash=_sha256(first + second),
        chunk_indexes=[0, 1],
    )

    assert b"".join(store.iter_blob(final_key, chunk_size=3)) == first + second


def test_local_blob_store_uses_unique_commit_temp_paths_for_same_payload_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"same shared payload"
    payload_hash = _sha256(payload)
    for upload_id in ("upload-1", "upload-2"):
        store.write_upload_chunk(
            upload_id=upload_id,
            chunk_index=0,
            payload=payload,
            expected_hash=payload_hash,
        )

    original_open = Path.open
    commit_write_paths: list[Path] = []

    def track_commit_write_path(self: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "w" in mode and self.name.endswith(".tmp"):
            commit_write_paths.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_commit_write_path)

    first_key = store.commit_upload(
        upload_id="upload-1",
        payload_hash=payload_hash,
        chunk_indexes=[0],
    )
    second_key = store.commit_upload(
        upload_id="upload-2",
        payload_hash=payload_hash,
        chunk_indexes=[0],
    )

    assert first_key == second_key
    assert len(commit_write_paths) == 2
    assert len(set(commit_write_paths)) == 2


def test_local_blob_store_does_not_overwrite_chunk_after_publish_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    existing_payload = b"winner"
    losing_payload = b"losing"
    upload_id = "upload-1"
    storage_key = store.write_upload_chunk(
        upload_id=upload_id,
        chunk_index=0,
        payload=existing_payload,
        expected_hash=_sha256(existing_payload),
    )
    target = store.resolve_storage_key(storage_key)
    original_link = os.link

    def publish_race(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if Path(dst) == target:
            raise FileExistsError
        original_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", publish_race)

    with pytest.raises(SyncBlobStoreError, match="different content"):
        store.write_upload_chunk(
            upload_id=upload_id,
            chunk_index=0,
            payload=losing_payload,
            expected_hash=_sha256(losing_payload),
        )

    assert target.read_bytes() == existing_payload


def test_local_blob_store_commit_cleanup_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"payload"
    store.write_upload_chunk(
        upload_id="upload-1",
        chunk_index=0,
        payload=payload,
        expected_hash=_sha256(payload),
    )

    def fail_rmtree(_path: Path, *args, **kwargs) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(shutil, "rmtree", fail_rmtree)

    final_key = store.commit_upload(
        upload_id="upload-1",
        payload_hash=_sha256(payload),
        chunk_indexes=[0],
    )

    assert store.read_blob(final_key) == payload


def test_local_blob_store_rejects_bad_hashes_and_path_escape(tmp_path: Path):
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")

    with pytest.raises(SyncBlobStoreError):
        store.write_upload_chunk(
            upload_id="upload-1",
            chunk_index=0,
            payload=b"not-this-hash",
            expected_hash="sha256:" + "0" * 64,
        )

    with pytest.raises(SyncBlobStoreError):
        store.resolve_storage_key("../outside")


def test_storage_namespace_key_contains_only_opaque_namespace_and_digest(
    tmp_path: Path,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"namespaced payload"
    payload_hash = _sha256(payload)
    namespace = "1" * 32
    store.write_upload_chunk(
        upload_id="upload-namespace",
        chunk_index=0,
        payload=payload,
        expected_hash=payload_hash,
    )

    storage_key = store.commit_upload(
        upload_id="upload-namespace",
        payload_hash=payload_hash,
        chunk_indexes=[0],
        storage_namespace_id=namespace,
    )

    assert storage_key == f"blobs/v2/{namespace}/{payload_hash[7:]}.blob"
    assert store.read_blob(storage_key) == payload
    assert "dataset" not in storage_key
    assert "attachment" not in storage_key
    with pytest.raises(SyncBlobStoreError, match="storage_namespace_id"):
        store.namespace_storage_key("../owner-dataset", payload_hash)
    with pytest.raises(SyncBlobStoreError, match="lowercase"):
        store.namespace_storage_key(namespace.upper().replace("1", "A"), payload_hash)
    with pytest.raises(SyncBlobStoreError, match="lowercase"):
        store.namespace_storage_key(namespace, "sha256:" + "A" * 64)


def _legacy_blob(store: LocalSyncBlobStore, payload: bytes) -> tuple[str, str]:
    payload_hash = _sha256(payload)
    store.write_upload_chunk(
        upload_id="legacy-upload",
        chunk_index=0,
        payload=payload,
        expected_hash=payload_hash,
    )
    legacy_key = store.commit_upload(
        upload_id="legacy-upload",
        payload_hash=payload_hash,
        chunk_indexes=[0],
    )
    return legacy_key, payload_hash


def test_legacy_blob_relocation_verifies_copies_reverifies_and_keeps_global_key(
    tmp_path: Path,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy shared bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "2" * 32

    relocated_key = store.relocate_legacy_blob(
        legacy_storage_key=legacy_key,
        storage_namespace_id=namespace,
        payload_hash=payload_hash,
        expected_size=len(payload),
    )

    assert relocated_key == store.namespace_storage_key(namespace, payload_hash)
    assert store.read_blob(relocated_key) == payload
    assert store.read_blob(legacy_key) == payload
    assert store.resolve_storage_key(legacy_key).exists()
    assert (store.root / relocated_key).stat().st_nlink == 1
    assert not any(
        path.name.endswith(".relocating")
        for path in (store.root / relocated_key).parent.iterdir()
    )


def test_repeated_relocation_copy_failure_leaves_only_canonical_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy bounded failed copies"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "9" * 32
    target_key = store.namespace_storage_key(namespace, payload_hash)
    target = store.root / target_key
    original_verify = blob_store_module._verify_open_blob
    verify_calls = 0

    def fail_after_each_copy(handle: Any, **kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(handle, **kwargs)
        if verify_calls % 2 == 0:
            raise SyncBlobStoreError("injected post-copy failure")

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", fail_after_each_copy)
    for _attempt in range(3):
        with pytest.raises(SyncBlobStoreError, match="injected"):
            store.relocate_legacy_blob(
                legacy_storage_key=legacy_key,
                storage_namespace_id=namespace,
                payload_hash=payload_hash,
                expected_size=len(payload),
            )

    assert target.read_bytes() == payload
    assert not any(
        path.name.endswith(".relocating") for path in target.parent.iterdir()
    )
    assert store.read_blob(legacy_key) == payload

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", original_verify)
    assert (
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )
        == target_key
    )
    assert store.read_blob(target_key) == payload


def test_legacy_blob_relocation_is_idempotent_and_concurrent_safe(
    tmp_path: Path,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy concurrent bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "3" * 32
    target_key = store.namespace_storage_key(namespace, payload_hash)

    def relocate() -> str:
        return store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        keys = list(pool.map(lambda _index: relocate(), range(2)))

    assert keys == [target_key, target_key]
    assert store.read_blob(target_key) == payload
    assert store.read_blob(legacy_key) == payload


@pytest.mark.parametrize("failure", ["corrupt", "symlink"])
def test_legacy_blob_relocation_fails_closed_for_invalid_existing_target(
    tmp_path: Path,
    failure: str,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy target bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "5" * 32
    target_key = store.namespace_storage_key(namespace, payload_hash)
    target = store.root / target_key
    target.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-target.blob"
    outside.write_bytes(b"outside must remain unchanged")
    if failure == "corrupt":
        target.write_bytes(b"corrupt target")
    else:
        target.symlink_to(outside)

    with pytest.raises(SyncBlobStoreError):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert store.read_blob(legacy_key) == payload
    assert outside.read_bytes() == b"outside must remain unchanged"
    if failure == "corrupt":
        assert target.read_bytes() == b"corrupt target"


def test_legacy_blob_relocation_rejects_source_symlink_swap_after_path_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy race bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    legacy_path = store.resolve_storage_key(legacy_key)
    outside = tmp_path / "outside-race.blob"
    outside.write_bytes(payload)
    original_open = os.open
    swapped = False

    def swap_source_after_resolution(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == legacy_path.name and dir_fd is not None and not swapped:
            swapped = True
            os.unlink(path, dir_fd=dir_fd)
            os.symlink(outside, path, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_source_after_resolution)

    with pytest.raises(SyncBlobStoreError, match="relocation"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id="6" * 32,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    target_key = store.namespace_storage_key("6" * 32, payload_hash)
    assert not (store.root / target_key).exists()
    assert swapped is True
    assert outside.read_bytes() == payload


def test_legacy_blob_relocation_rejects_target_symlink_swap_during_reverification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy target race bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "7" * 32
    target_key = store.namespace_storage_key(namespace, payload_hash)
    target = store.root / target_key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    outside = tmp_path / "outside-target-race.blob"
    outside.write_bytes(payload)
    target_parent = target.parent.stat()
    original_open = os.open
    swapped = False

    def swap_target_before_hash(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        parent = None if dir_fd is None else os.fstat(dir_fd)
        if (
            path == target.name
            and parent is not None
            and parent.st_dev == target_parent.st_dev
            and parent.st_ino == target_parent.st_ino
            and not swapped
        ):
            swapped = True
            os.unlink(path, dir_fd=dir_fd)
            os.symlink(outside, path, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_target_before_hash)

    with pytest.raises(SyncBlobStoreError):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert store.read_blob(legacy_key) == payload
    assert swapped is True
    assert outside.read_bytes() == payload


def test_legacy_blob_relocation_rejects_post_verify_target_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy post-verify target race"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "d" * 32
    target = store.root / store.namespace_storage_key(namespace, payload_hash)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    displaced = tmp_path / "displaced-target.blob"
    original_verify = blob_store_module._verify_open_blob
    calls = 0

    def swap_name_after_verify(handle: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original_verify(handle, **kwargs)
        if calls == 2:
            target.rename(displaced)
            target.write_bytes(payload)

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", swap_name_after_verify)

    with pytest.raises(SyncBlobStoreError, match="identity"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert calls == 2
    assert target.read_bytes() == payload
    assert displaced.read_bytes() == payload


def test_relocation_cleanup_never_unlinks_swapped_target_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy cleanup target race"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "e" * 32
    target = store.root / store.namespace_storage_key(namespace, payload_hash)
    displaced = tmp_path / "published-target.blob"
    replacement = b"attacker target must remain"
    original_verify = blob_store_module._verify_open_blob
    calls = 0

    def fail_after_target_swap(handle: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original_verify(handle, **kwargs)
        if calls == 2:
            target.rename(displaced)
            target.write_bytes(replacement)
            raise SyncBlobStoreError("injected post-publish failure")

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", fail_after_target_swap)

    with pytest.raises(SyncBlobStoreError, match="injected"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert target.read_bytes() == replacement
    assert displaced.read_bytes() == payload


def test_relocation_failure_never_attempts_racy_target_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy cleanup unlink race"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "1" * 32
    target = store.root / store.namespace_storage_key(namespace, payload_hash)
    displaced = tmp_path / "displaced-cleanup-target.blob"
    replacement = tmp_path / "replacement-cleanup-target.blob"
    replacement_payload = b"replacement must survive"
    replacement.write_bytes(replacement_payload)
    original_verify = blob_store_module._verify_open_blob
    original_unlink = os.unlink
    verify_calls = 0
    cleanup_swap_attempted = False

    def fail_after_target_verify(handle: Any, **kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(handle, **kwargs)
        if verify_calls == 2:
            raise SyncBlobStoreError("injected post-publish failure")

    def swap_at_unlink(
        path: Any,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal cleanup_swap_attempted
        if os.fspath(path) == target.name and dir_fd is not None:
            cleanup_swap_attempted = True
            target.rename(displaced)
            replacement.rename(target)
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", fail_after_target_verify)
    monkeypatch.setattr(os, "unlink", swap_at_unlink)

    with pytest.raises(SyncBlobStoreError, match="injected"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert cleanup_swap_attempted is False
    assert replacement.read_bytes() == replacement_payload
    assert target.read_bytes() == payload
    assert store.read_blob(legacy_key) == payload

    relocated_key = store.relocate_legacy_blob(
        legacy_storage_key=legacy_key,
        storage_namespace_id=namespace,
        payload_hash=payload_hash,
        expected_size=len(payload),
    )
    assert store.read_blob(relocated_key) == payload
    assert store.read_blob(legacy_key) == payload


def test_relocation_write_failure_leaves_one_partial_canonical_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy partial canonical target"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "f" * 32
    target = store.root / store.namespace_storage_key(namespace, payload_hash)
    original_fdopen = os.fdopen

    class PartialWriter:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> PartialWriter:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.handle.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

        def write(self, data: bytes) -> int:
            self.handle.write(data[:4])
            raise OSError("injected target write failure")

    def fail_target_write(descriptor: int, mode: str, **kwargs: Any):
        handle = original_fdopen(descriptor, mode, **kwargs)
        return PartialWriter(handle) if mode == "r+b" else handle

    monkeypatch.setattr(os, "fdopen", fail_target_write)
    with pytest.raises(SyncBlobStoreError, match="relocation failed"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert target.read_bytes() == payload[:4]
    assert list(target.parent.iterdir()) == [target]
    assert store.read_blob(legacy_key) == payload

    monkeypatch.setattr(os, "fdopen", original_fdopen)
    with pytest.raises(SyncBlobStoreError):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )
    assert target.read_bytes() == payload[:4]


def test_legacy_relocation_fails_closed_when_secure_dirfd_capabilities_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy unsupported platform"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    monkeypatch.setattr(os, "supports_dir_fd", set())

    with pytest.raises(SyncBlobStoreError, match="unsupported platform"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id="1" * 32,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )


def test_legacy_relocation_wraps_runtime_unsupported_dir_relative_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy runtime unsupported link"
    legacy_key, payload_hash = _legacy_blob(store, payload)

    original_stat = os.stat

    def unsupported_stat(path: Any, *args: Any, **kwargs: Any):
        if os.fspath(path).endswith(".blob") and kwargs.get("dir_fd") is not None:
            raise NotImplementedError("dir-relative stat unavailable")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", unsupported_stat)
    with pytest.raises(SyncBlobStoreError, match="unsupported platform"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id="3" * 32,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )


def test_verify_blob_uses_descriptor_anchored_path() -> None:
    source = inspect.getsource(LocalSyncBlobStore.verify_blob)
    assert "_open_contained_regular_file" in source
    assert ".is_file()" not in source
    assert ".stat()" not in source


def test_verify_blob_rejects_post_verify_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"anchored replay name race"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    key = store.relocate_legacy_blob(
        legacy_storage_key=legacy_key,
        storage_namespace_id="4" * 32,
        payload_hash=payload_hash,
        expected_size=len(payload),
    )
    target = store.root / key
    displaced = tmp_path / "replay-displaced.blob"
    original_verify = blob_store_module._verify_open_blob

    def swap_after_verify(handle: Any, **kwargs: Any) -> None:
        original_verify(handle, **kwargs)
        target.rename(displaced)
        target.write_bytes(payload)

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", swap_after_verify)
    with pytest.raises(SyncBlobStoreError, match="identity"):
        store.verify_blob(key, payload_hash=payload_hash, expected_size=len(payload))

    assert target.read_bytes() == payload
    assert displaced.read_bytes() == payload


def test_relocation_has_no_race_prone_path_cleanup() -> None:
    source = inspect.getsource(blob_store_module._relocate_open_blob)
    assert "os.unlink" not in source
    assert "_unlink_name_if_inode_matches" not in source
    assert "logger.warning" not in source


def test_legacy_blob_relocation_never_reopens_temporary_file_by_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy anchored temp bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    original_open = Path.open

    def reject_relocation_temp_reopen(self: Path, *args: Any, **kwargs: Any):
        if self.name.endswith(".relocating"):
            raise AssertionError("relocation temp must remain descriptor-anchored")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_relocation_temp_reopen)

    relocated_key = store.relocate_legacy_blob(
        legacy_storage_key=legacy_key,
        storage_namespace_id="8" * 32,
        payload_hash=payload_hash,
        expected_size=len(payload),
    )

    assert store.read_blob(relocated_key) == payload
    assert store.read_blob(legacy_key) == payload


def test_legacy_blob_relocation_rejects_new_target_name_swap_after_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy new target inode race bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "a" * 32
    target_key = store.namespace_storage_key(namespace, payload_hash)
    target = store.root / target_key
    displaced = tmp_path / "displaced-new-target.blob"
    outside = tmp_path / "outside-new-target-race.blob"
    outside.write_bytes(payload)
    original_verify = blob_store_module._verify_open_blob
    calls = 0
    attacked = False

    def swap_new_target_after_verify(
        handle: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal attacked, calls
        calls += 1
        original_verify(handle, **kwargs)
        if calls == 2:
            attacked = True
            target.rename(displaced)
            target.symlink_to(outside)

    monkeypatch.setattr(blob_store_module, "_verify_open_blob", swap_new_target_after_verify)

    with pytest.raises(SyncBlobStoreError, match="identity"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert attacked is True
    assert target.is_symlink()
    assert target.resolve() == outside
    assert displaced.read_bytes() == payload
    assert outside.read_bytes() == payload
    assert store.read_blob(legacy_key) == payload


def test_legacy_blob_relocation_fails_closed_on_intermediate_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy intermediate race bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "9" * 32
    outside = tmp_path / "outside-namespace"
    outside.mkdir()
    original_open = os.open
    attacked = False

    def swap_namespace_before_anchored_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if (
            path == namespace
            and dir_fd is not None
            and flags & getattr(os, "O_DIRECTORY", 0)
            and not attacked
        ):
            attacked = True
            os.rename(
                namespace,
                namespace + ".moved",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.symlink(outside, namespace, target_is_directory=True, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_namespace_before_anchored_open)

    with pytest.raises(SyncBlobStoreError):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert attacked is True
    assert list(outside.iterdir()) == []
    assert store.read_blob(legacy_key) == payload


def test_legacy_blob_relocation_rejects_intermediate_lock_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy lock directory race bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    lock_parent = store.root / "_locks"
    lock_directory = lock_parent / "legacy-relocation"
    lock_directory.mkdir(parents=True)
    lock_parent_identity = lock_parent.stat()
    outside = tmp_path / "outside-lock-directory"
    outside.mkdir()
    original_open = os.open
    attacked = False

    def swap_lock_directory_before_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        parent = None if dir_fd is None else os.fstat(dir_fd)
        if (
            path == "legacy-relocation"
            and parent is not None
            and parent.st_dev == lock_parent_identity.st_dev
            and parent.st_ino == lock_parent_identity.st_ino
            and not attacked
        ):
            attacked = True
            os.rename(
                "legacy-relocation",
                "legacy-relocation.moved",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.symlink(
                outside,
                "legacy-relocation",
                target_is_directory=True,
                dir_fd=dir_fd,
            )
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_lock_directory_before_open)

    with pytest.raises(SyncBlobStoreError):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id="b" * 32,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert attacked is True
    assert list(outside.iterdir()) == []
    assert store.read_blob(legacy_key) == payload


def test_legacy_blob_relocation_rejects_final_lock_file_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy lock file race bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    namespace = "c" * 32
    lock_directory = store.root / "_locks" / "legacy-relocation"
    lock_directory.mkdir(parents=True)
    lock_name = f"{namespace}.{payload_hash[7:]}.lock"
    lock_path = lock_directory / lock_name
    lock_path.write_bytes(b"")
    lock_directory_identity = lock_directory.stat()
    outside = tmp_path / "outside-lock-file"
    outside.write_bytes(b"outside lock bytes")
    original_open = os.open
    attacked = False

    def swap_lock_file_before_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        parent = None if dir_fd is None else os.fstat(dir_fd)
        if (
            path == lock_name
            and parent is not None
            and parent.st_dev == lock_directory_identity.st_dev
            and parent.st_ino == lock_directory_identity.st_ino
            and not attacked
        ):
            attacked = True
            os.unlink(lock_name, dir_fd=dir_fd)
            os.symlink(outside, lock_name, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_lock_file_before_open)

    with pytest.raises(SyncBlobStoreError, match="lock"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert attacked is True
    assert outside.read_bytes() == b"outside lock bytes"
    assert store.read_blob(legacy_key) == payload


@pytest.mark.parametrize("failure", ["missing", "corrupt", "symlink"])
def test_legacy_blob_relocation_fails_closed_for_invalid_legacy_source(
    tmp_path: Path,
    failure: str,
) -> None:
    store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    payload = b"legacy source bytes"
    legacy_key, payload_hash = _legacy_blob(store, payload)
    legacy_path = store.resolve_storage_key(legacy_key)
    if failure == "missing":
        legacy_path.unlink()
    elif failure == "corrupt":
        legacy_path.write_bytes(b"corrupt")
    else:
        outside = tmp_path / "outside.blob"
        outside.write_bytes(payload)
        legacy_path.unlink()
        legacy_path.symlink_to(outside)

    with pytest.raises(SyncBlobStoreError, match="legacy"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id="4" * 32,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    target_key = store.namespace_storage_key("4" * 32, payload_hash)
    assert not store.resolve_storage_key(target_key).exists()
