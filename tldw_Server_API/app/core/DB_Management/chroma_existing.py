"""Existing Chroma 1.5.5 disk-profile validation, before any SDK initialization.

SQL belongs here, not in Claims or Embeddings. Unknown SDK/schema profiles are
unsupported, never missing embeddings. This does not migrate or repair storage.
"""

from __future__ import annotations

import hashlib
import json

# Only genops is used: it inspects opcodes without loading/executing a pickle.
import pickletools  # nosec B403
import sqlite3
import stat
import struct
from contextlib import closing
from pathlib import Path
from uuid import UUID

from tldw_Server_API.app.core.Embeddings.chroma_storage import ChromaStorageError

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
_DATA_PICKLE_OPS = frozenset(
    (
        "PROTO",
        "EMPTY_DICT",
        "MARK",
        "BINUNICODE",
        "NONE",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG1",
        "SETITEM",
        "SETITEMS",
        "STOP",
    )
)


def _invalid() -> ChromaStorageError:
    return ChromaStorageError("vector_corruption", retryable=False)


def _regular_file(path: Path) -> int:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise _invalid()
    return info.st_size


def _validate_index(path: Path, *, dimension: int) -> None:
    """Reject absent/partial files before the SDK can recreate or overwrite them."""
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise _invalid()
    sizes = {name: _regular_file(path / name) for name in _INDEX_FILES}
    if sizes["header.bin"] != 100 or any(size <= 0 for size in sizes.values()):
        raise _invalid()
    # Persisted local HNSW v1 uses little-endian u32 + twelve 64-bit fields.
    # Check the complete file extents before a loader can reinterpret them.
    header = struct.unpack("<I12Q", (path / "header.bin").read_bytes())
    version, offset, capacity, count, stride, label_offset, data_offset = header[:7]
    if (version, offset) != (1, 0) or not 0 < count <= capacity:
        raise _invalid()
    if data_offset != 4 + 4 * header[9] or label_offset != data_offset + 4 * dimension or stride != label_offset + 8:
        raise _invalid()
    if (
        not 0 < header[8] == header[10] <= header[9]
        or sizes["data_level0.bin"] != capacity * stride
        or sizes["length.bin"] != capacity * 4
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
    if sizes["index_metadata.pickle"] > 33_554_432:
        raise _invalid()
    # Inspect data-only Rust metadata without executing pickle globals/reducers.
    data = (path / "index_metadata.pickle").read_bytes()
    operations = list(pickletools.genops(data))
    if not operations or operations[-1][0].name != "STOP" or operations[-1][2] != len(data) - 1:
        raise _invalid()
    if any(op.name not in _DATA_PICKLE_OPS for op, _, _ in operations):
        raise _invalid()
    keys = {arg for op, arg, _ in operations if op.name == "BINUNICODE"}
    if not {"dimensionality", "total_elements_added", "id_to_label", "label_to_id", "id_to_seq_id"} <= keys:
        raise _invalid()


def validate_existing_chroma(path: Path, *, collection_name: str | None = None) -> str | None:
    """Validate a complete, ordinary local disk store while caller holds its lock.

    None proves collection absence only after complete schema/default validation.
    A missing SQLite file in an existing directory is corruption, not absence.
    All persisted vector segments are checked: SDK startup owns the whole store.
    """
    import chromadb

    if chromadb.__version__ != "1.5.5":
        raise ChromaStorageError("vector_unsupported_sdk", retryable=False)
    try:
        _regular_file(path / "chroma.sqlite3")
        # A quiescent supported store has no recovery/WAL sidecars. Immutable
        # validation avoids even SQLite's read-only WAL shared-memory creation.
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                (path / f"chroma.sqlite3{suffix}").lstat()
            except FileNotFoundError:
                continue
            raise _invalid()
        with closing(
            sqlite3.connect((path / "chroma.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        ) as db:
            db.execute("PRAGMA query_only=ON")
            shape = db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name").fetchall()
            signature = hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()
            if signature != _SCHEMA_155:
                raise _invalid()
            migrations = db.execute("SELECT dir,version,sql,hash FROM migrations ORDER BY dir,version").fetchall()
            expected = {
                (directory, i + 1): digest
                for directory, hashes in _MIGRATIONS_155.items()
                for i, digest in enumerate(hashes)
            }
            if len(migrations) != len(expected):
                raise _invalid()
            for directory, version, sql, digest in migrations:
                md5 = hashlib.md5(sql.encode(), usedforsecurity=False).hexdigest()
                if expected.get((directory, version)) != md5 or digest != md5:
                    raise _invalid()
            if db.execute("SELECT id FROM tenants WHERE id = 'default_tenant'").fetchall() != [("default_tenant",)]:
                raise _invalid()
            if db.execute(
                "SELECT id FROM databases WHERE name = 'default_database' AND tenant_id = 'default_tenant'"
            ).fetchall() != [("00000000-0000-0000-0000-000000000000",)]:
                raise _invalid()
            # Rust appends an acquire_write row on each SDK startup; this is
            # not a singleton default record and must survive repeated opens.
            if (
                not db.execute("SELECT 1 FROM acquire_write LIMIT 1").fetchone()
                or db.execute("SELECT 1 FROM acquire_write WHERE id <= 0 OR lock_status != 1 LIMIT 1").fetchone()
            ):
                raise _invalid()
            configs = db.execute("SELECT id,config_json_str FROM embeddings_queue_config").fetchall()
            if len(configs) != 1 or configs[0][0] != 1:
                raise _invalid()
            config = json.loads(configs[0][1])
            if (
                not isinstance(config, dict)
                or type(config.get("automatically_purge")) is not bool
                or config.get("_type") != "EmbeddingsQueueConfigurationInternal"
            ):
                raise _invalid()
            for collection_id, dimension, database_id in db.execute("SELECT id,dimension,database_id FROM collections"):
                if str(UUID(collection_id)) != collection_id or (
                    dimension is not None and (type(dimension) is not int or dimension <= 0)
                ):
                    raise _invalid()
                if not db.execute(
                    "SELECT 1 FROM databases d JOIN tenants t ON t.id = d.tenant_id WHERE d.id = ?", (database_id,)
                ).fetchone():
                    raise _invalid()
                if db.execute(
                    "SELECT scope FROM segments WHERE collection = ? ORDER BY scope", (collection_id,)
                ).fetchall() != [("METADATA",), ("VECTOR",)]:
                    raise _invalid()
            for segment_id, segment_type, scope, collection_id in db.execute(
                "SELECT id,type,scope,collection FROM segments"
            ):
                if (
                    str(UUID(segment_id)) != segment_id
                    or not db.execute("SELECT 1 FROM collections WHERE id = ?", (collection_id,)).fetchone()
                ):
                    raise _invalid()
                if scope == "VECTOR" and segment_type == "urn:chroma:segment/vector/hnsw-local-persisted":
                    (dimension,) = db.execute(
                        "SELECT dimension FROM collections WHERE id = ?", (collection_id,)
                    ).fetchone()
                    if dimension is None:
                        raise _invalid()
                    _validate_index(path / segment_id, dimension=dimension)
                elif (scope, segment_type) != ("METADATA", "urn:chroma:segment/metadata/sqlite"):
                    raise _invalid()
            if collection_name is None:
                return None
            rows = db.execute(
                "SELECT id,dimension FROM collections WHERE name = ? AND database_id = ?",
                (collection_name, "00000000-0000-0000-0000-000000000000"),
            ).fetchall()
            if not rows:
                return None
            collection_id, dimension = rows[0]
            if dimension is not None and dimension > 4096:
                raise ChromaStorageError("resource_limit", retryable=False)
            if str(UUID(collection_id)) != collection_id or (
                dimension is not None and (type(dimension) is not int or dimension <= 0)
            ):
                raise _invalid()
            segments = db.execute(
                "SELECT scope FROM segments WHERE collection = ? ORDER BY scope", (collection_id,)
            ).fetchall()
            if segments != [("METADATA",), ("VECTOR",)]:
                raise _invalid()
            return collection_id
    except (FileNotFoundError, ValueError, TypeError, sqlite3.DatabaseError) as exc:
        if isinstance(exc, sqlite3.OperationalError) and ("locked" in str(exc) or "unable to open" in str(exc)):
            raise ChromaStorageError("vector_access", retryable=True) from exc
        raise _invalid() from exc
    except OSError as exc:
        raise ChromaStorageError("vector_access", retryable=True) from exc
