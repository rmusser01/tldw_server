from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

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
