"""Existing Chroma 1.5.5 disk-profile validation, before any SDK initialization.

SQL belongs here, not in Claims or Embeddings. Unknown SDK/schema profiles are
unsupported, never missing embeddings. This does not migrate or repair storage.
"""

from __future__ import annotations

import stat
import struct
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.chroma_read_validation import (
    ChromaPreflightLimits as ChromaPreflightLimits,
)
from tldw_Server_API.app.core.DB_Management.chroma_read_validation import (
    ReadBudget,
    read_metadata,
)
from tldw_Server_API.app.core.DB_Management.chroma_storage_errors import ChromaStorageError

# Complete sqlite_master signature from the real Rust 1.5.5 fresh-store profile,
# including constraints, FTS shadow tables and indexes, not just table presence.
_SCHEMA_155 = "cf1476ebfbafc67b9ed8f699490d5d77a3bb23fcd59b4548e236703efc9606a8"
_MIGRATIONS_155 = {
    "sysdb": (
        "38352d725ad1c16074fac420b22b4633",
        "2913cb6a503055a95f625448037e8912",
        "42d22d0574d31d419c2a0e7f625c93aa",
        "048867ce8fcdefe4023c7110e4433591",
        "b1367c826b8fba5f96f27befdc1d42d2",
        "4eea7468935bf25d4604a0fed2366116",
        "1c7e63bba346a42a18b6ab7f1c989bed",
        "0a0e7e93111a01789addf64961c6127c",
        "054355aef9e63702bf54ea29e61563f1",
        "5c3a5ac4b79df76799b4721827ed5e1d",
    ),
    "metadb": (
        "2b4cf52c4bb2676e21d6860a4409f856",
        "12a570f7121b3a8ce750a2a7c36da20f",
        "f97ad6334aeaa8f419f01110b648b97a",
        "fb36603a45ee2cd0254cef3ef86585e8",
        "0e9de46758761b373ce682925edcc326",
        "e026f01ea92c1baa1493f4ad5ca7cfe7",
    ),
    "embeddings_queue": ("d3755dfd232be8e8301f4d7fcfb3a486", "8fbfe4ffb3e57f1d8bfdc58510a82e85"),
}
_INDEX_FILES = ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle")


def _invalid() -> ChromaStorageError:
    return ChromaStorageError("vector_corruption", retryable=False)


def _regular_file(path: Path) -> int:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise _invalid()
    return info.st_size


def _validate_index(path: Path, *, dimension: int, budget: ReadBudget | None = None, persisted: bool = True) -> None:
    """Reject absent/partial files before the SDK can recreate or overwrite them."""
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise _invalid()
    budget = budget or ReadBudget(ChromaPreflightLimits())
    names = _INDEX_FILES if persisted else _INDEX_FILES[:-1]
    sizes = {name: _regular_file(path / name) for name in names}
    if sizes["header.bin"] != 100:
        raise _invalid()
    # Persisted local HNSW v1 uses little-endian u32 + twelve 64-bit fields.
    # Check the complete file extents before a loader can reinterpret them.
    header = struct.unpack("<I12Q", (path / "header.bin").read_bytes())
    version, offset, capacity, count, stride, label_offset, data_offset = header[:7]
    if (version, offset) != (1, 0) or not 0 <= count <= capacity or capacity <= 0:
        raise _invalid()
    budget.charge("elements", count)
    if data_offset != 4 + 4 * header[9] or label_offset != data_offset + 4 * dimension or stride != label_offset + 8:
        raise _invalid()
    # Resizing reserves memory capacity, but persistent files may cover only
    # the written slots. Their paired allocation must agree and cover count.
    allocated = sizes["data_level0.bin"] // stride
    if (
        not 0 < header[8] == header[10] <= header[9]
        or sizes["data_level0.bin"] % stride
        or sizes["length.bin"] != allocated * 4
        or not 0 < allocated <= capacity
        or allocated < count
    ):
        raise _invalid()
    upper_stride = 4 + 4 * header[8]
    remaining = sizes["link_lists.bin"]
    with (path / "link_lists.bin").open("rb") as links:
        for _ in range(count):
            prefix = links.read(4)
            if len(prefix) != 4:
                raise _invalid()
            (length,) = struct.unpack("<I", prefix)
            remaining -= 4 + length
            if remaining < 0 or length % upper_stride:
                raise _invalid()
            links.seek(length, 1)
        if remaining:
            raise _invalid()
    if persisted:
        read_metadata(
            path / "index_metadata.pickle",
            size=sizes["index_metadata.pickle"],
            count=count,
            dimension=dimension,
            budget=budget,
        )
    elif count != 0 or (path / "index_metadata.pickle").exists():
        raise _invalid()


def validate_existing_chroma(path: Path, *, collection_name: str | None = None) -> str | None:
    """Validate a complete, ordinary local disk store while caller holds its lock.

    None proves collection absence only after complete schema/default validation.
    A missing SQLite file in an existing directory is corruption, not absence.
    All persisted vector segments are checked: SDK startup owns the whole store.
    """
    from tldw_Server_API.app.core.DB_Management.chroma_existing_state import preflight_existing_chroma

    snapshot = preflight_existing_chroma(path, limits=ChromaPreflightLimits())
    state = snapshot.collections.get(collection_name)
    return state.collection_id if state is not None else None
