from __future__ import annotations

import hashlib
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
    assert not (tmp_path / "sync_blobs" / "_uploads" / "upload-1").exists()


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
