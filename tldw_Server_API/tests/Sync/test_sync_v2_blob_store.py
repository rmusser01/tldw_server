from __future__ import annotations

import hashlib
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
    original_temp_path = blob_store_module._relocation_temp_path

    def swap_source_after_resolution(
        target: Path,
        *,
        digest: str,
        namespace: str,
    ) -> Path:
        temp_path = original_temp_path(target, digest=digest, namespace=namespace)
        legacy_path.unlink()
        legacy_path.symlink_to(outside)
        return temp_path

    monkeypatch.setattr(
        blob_store_module,
        "_relocation_temp_path",
        swap_source_after_resolution,
    )

    with pytest.raises(SyncBlobStoreError, match="relocation"):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id="6" * 32,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    target_key = store.namespace_storage_key("6" * 32, payload_hash)
    assert not (store.root / target_key).exists()
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
    original_hash = blob_store_module._sha256_file
    swapped = False

    def swap_target_before_hash(path: Path) -> str:
        nonlocal swapped
        if path == target and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(outside)
        return original_hash(path)

    monkeypatch.setattr(blob_store_module, "_sha256_file", swap_target_before_hash)

    with pytest.raises(SyncBlobStoreError):
        store.relocate_legacy_blob(
            legacy_storage_key=legacy_key,
            storage_namespace_id=namespace,
            payload_hash=payload_hash,
            expected_size=len(payload),
        )

    assert store.read_blob(legacy_key) == payload
    assert outside.read_bytes() == payload


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
