"""Bounded existing local Chroma catalog and never-persisted logical log reads.

No SDK, recovery, migrations, writes, or new persisted backend. A materialized
target still uses Chroma after preflight. A never-persisted target is replayed
from the existing SQLite log only with checkpoint and metadata coverage proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.chroma_existing import _MIGRATIONS_155, _SCHEMA_155, _validate_index
from tldw_Server_API.app.core.DB_Management.chroma_read_validation import (
    ChromaPreflightLimits,
    ReadBudget,
    existing_sqlite,
    invalid,
    resource_limit,
)
from tldw_Server_API.app.core.DB_Management.chroma_storage_errors import ChromaStorageError


def _identity(path: Path) -> tuple[int, ...]:
    root = path.lstat()
    if not stat.S_ISDIR(root.st_mode):
        raise invalid()
    database = (path / "chroma.sqlite3").lstat()
    if not stat.S_ISREG(database.st_mode):
        raise invalid()
    return root.st_dev, root.st_ino, database.st_dev, database.st_ino


@dataclass(frozen=True)
class CollectionState:
    collection_id: str
    dimension: int | None
    vector_segment: str
    metadata_segment: str
    vector_checkpoint: int
    metadata_checkpoint: int
    kind: str


@dataclass
class ExistingSnapshot:
    path: Path
    identity: tuple[int, ...]
    collections: dict[str, CollectionState]
    budget: ReadBudget
    logical_vectors: dict[str, dict[str, tuple[float, ...]]] = field(default_factory=dict)

    def check_identity(self) -> None:
        try:
            if _identity(self.path) != self.identity:
                raise ChromaStorageError("vector_source_changed", retryable=True)
        except FileNotFoundError as exc:
            raise ChromaStorageError("vector_source_changed", retryable=True) from exc


def _uuid(value: str) -> None:
    if str(UUID(value)) != value:
        raise invalid()


def preflight_existing_chroma(path: Path, *, limits: ChromaPreflightLimits) -> ExistingSnapshot:
    """Validate at most the aggregate ceilings, without provisioning anything."""
    import chromadb

    if chromadb.__version__ != "1.5.5":
        raise ChromaStorageError("vector_unsupported_sdk", retryable=False)
    budget = ReadBudget(limits)
    try:
        identity = _identity(path)
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                (path / f"chroma.sqlite3{suffix}").lstat()
            except FileNotFoundError:
                continue
            raise invalid()
        with existing_sqlite(path, budget) as db:
            rows = budget.rows
            shape = rows(db, "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name", 128)
            if hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest() != _SCHEMA_155:
                raise invalid()
            expected = {
                (directory, i + 1): digest
                for directory, hashes in _MIGRATIONS_155.items()
                for i, digest in enumerate(hashes)
            }
            migrations = rows(db, "SELECT dir,version,sql,hash FROM migrations ORDER BY dir,version", len(expected))
            if len(migrations) != len(expected):
                raise invalid()
            for directory, version, sql, digest in migrations:
                if (
                    type(directory) is not str
                    or type(version) is not int
                    or type(sql) is not str
                    or type(digest) is not str
                ):
                    raise invalid()
                md5 = hashlib.md5(sql.encode(), usedforsecurity=False).hexdigest()
                if expected.get((directory, version)) != md5 or digest != md5:
                    raise invalid()
            tenants = {row[0] for row in rows(db, "SELECT id FROM tenants", limits.collections)}
            databases = {
                key: (name, tenant)
                for key, name, tenant in rows(db, "SELECT id,name,tenant_id FROM databases", limits.collections)
            }
            default_database = "00000000-0000-0000-0000-000000000000"
            if "default_tenant" not in tenants or databases.get(default_database) != (
                "default_database",
                "default_tenant",
            ):
                raise invalid()
            if any(tenant not in tenants for _, tenant in databases.values()):
                raise invalid()
            # Only the latest startup marker affects SDK initialization. Avoid
            # scanning all historical acquire_write bookkeeping on every read.
            markers = rows(
                db, "SELECT id,lock_status FROM acquire_write WHERE id = (SELECT MAX(id) FROM acquire_write)", 1
            )
            if not markers or type(markers[0][0]) is not int or markers[0][0] <= 0 or markers[0][1] != 1:
                raise invalid()
            configs = rows(db, "SELECT id,config_json_str FROM embeddings_queue_config", 1)
            if len(configs) != 1 or configs[0][0] != 1:
                raise invalid()
            config = json.loads(configs[0][1])
            if (
                not isinstance(config, dict)
                or type(config.get("automatically_purge")) is not bool
                or config.get("_type") != "EmbeddingsQueueConfigurationInternal"
            ):
                raise invalid()
            collections = rows(db, "SELECT id,name,dimension,database_id FROM collections", limits.collections)
            budget.charge("collections", len(collections))
            catalog = {row[0]: row for row in collections}
            segments = rows(db, "SELECT id,type,scope,collection FROM segments", limits.segments)
            budget.charge("segments", len(segments))
            grouped: dict[str, dict[str, str]] = {key: {} for key in catalog}
            for segment, segment_type, scope, collection in segments:
                _uuid(segment)
                if collection not in grouped or scope in grouped[collection]:
                    raise invalid()
                if (scope, segment_type) not in (
                    ("VECTOR", "urn:chroma:segment/vector/hnsw-local-persisted"),
                    ("METADATA", "urn:chroma:segment/metadata/sqlite"),
                ):
                    raise invalid()
                grouped[collection][scope] = segment
            checkpoints = dict(rows(db, "SELECT segment_id,seq_id FROM max_seq_id", limits.segments))
            segment_ids = {row[0] for row in segments}
            if any(
                segment not in segment_ids or type(value) is not int or value < 0
                for segment, value in checkpoints.items()
            ):
                raise invalid()
            result = {}
            for collection_id, name, dimension, database in collections:
                _uuid(collection_id)
                if (
                    type(name) is not str
                    or not name
                    or database not in databases
                    or (dimension is not None and (type(dimension) is not int or dimension <= 0))
                ):
                    raise invalid()
                group = grouped[collection_id]
                if set(group) != {"VECTOR", "METADATA"}:
                    raise invalid()
                vector_seq = checkpoints.get(group["VECTOR"], 0)
                metadata_seq = checkpoints.get(group["METADATA"], 0)
                index = path / group["VECTOR"]
                if dimension is None:
                    if vector_seq or metadata_seq or index.exists() or index.is_symlink():
                        raise invalid()
                    kind = "empty"
                else:
                    kind = "materialized" if vector_seq else "buffered"
                    _validate_index(index, dimension=dimension, budget=budget, persisted=bool(vector_seq))
                if database == default_database:
                    result[name] = CollectionState(
                        collection_id, dimension, group["VECTOR"], group["METADATA"], vector_seq, metadata_seq, kind
                    )
        return ExistingSnapshot(path, identity, result, budget)
    except (FileNotFoundError, ValueError, TypeError, struct.error, RecursionError) as exc:
        raise invalid() from exc
    except OSError as exc:
        raise ChromaStorageError("vector_access", retryable=True) from exc


def read_logical_vectors(snapshot: ExistingSnapshot, state: CollectionState) -> dict[str, tuple[float, ...]]:
    """Read a complete never-persisted target, not a tail of a purged index.

    Qualified Rust purge uses min(metadata,HNSW checkpoint); no HNSW checkpoint
    means zero and no records can have been purged by that protocol. Reconcile
    every surviving ID AND its effective sequence against metadata, and require
    the log's end equals that metadata checkpoint. Never infer a base from a
    truncated materialized tail or merely from the current purge setting.
    """
    if state.vector_checkpoint or state.kind == "materialized":
        raise invalid()
    if state.collection_id in snapshot.logical_vectors:
        return snapshot.logical_vectors[state.collection_id]
    snapshot.check_identity()
    budget = snapshot.budget
    live: dict[str, tuple[tuple[float, ...], int]] = {}
    last = 0
    try:
        with existing_sqlite(snapshot.path, budget) as db:
            # Stream records: never fetchall or materialize a history before
            # checking component/row ceilings, including ignored ADD records.
            cursor = db.execute(
                "SELECT seq_id,id,operation,vector,encoding FROM embeddings_queue WHERE topic = ? ORDER BY seq_id LIMIT ?",
                (f"persistent://default/default/{state.collection_id}", budget.limits.queue_records + 1),
            )
            for sequence, key, operation, raw, encoding in cursor:
                budget.charge("queue_records")
                if (
                    type(sequence) is not int
                    or sequence <= last
                    or type(key) is not str
                    or type(operation) is not int
                    or operation not in (0, 1, 2, 3)
                ):
                    raise invalid()
                budget.charge("catalog_bytes", len(key.encode("utf-8")))
                last = sequence
                vector = None
                if raw is not None:
                    if encoding != "FLOAT32" or state.dimension is None or len(raw) != 4 * state.dimension:
                        raise invalid()
                    if state.dimension > budget.limits.vector_width:
                        raise resource_limit()
                    budget.charge("queue_components", state.dimension)
                    vector = struct.unpack(f"<{state.dimension}f", raw)
                    if not all(math.isfinite(value) for value in vector):
                        raise invalid()
                elif encoding is not None:
                    raise invalid()
                if operation == 3:
                    if raw is not None:
                        raise invalid()
                    live.pop(key, None)
                elif operation == 0 and key in live or operation == 1 and key not in live:
                    continue
                else:
                    if key not in live:
                        budget.charge("elements")
                    value = vector if vector is not None else live.get(key, (None, 0))[0]
                    if value is None:
                        raise invalid()
                    live[key] = (value, sequence)
            if last != state.metadata_checkpoint and not (state.kind == "empty" and state.metadata_checkpoint == 0):
                raise ChromaStorageError("vector_log_coverage", retryable=False)
            metadata = budget.rows(
                db,
                "SELECT embedding_id,seq_id FROM embeddings WHERE segment_id = ?",
                budget.limits.elements,
                (state.metadata_segment,),
            )
            budget.charge("elements", len(metadata))
            if len(metadata) != len(live) or any(
                type(sequence) is not int or key not in live or live[key][1] != sequence for key, sequence in metadata
            ):
                raise ChromaStorageError("vector_log_coverage", retryable=False)
        result = {key: vector for key, (vector, _) in live.items()}
        snapshot.logical_vectors[state.collection_id] = result
        return result
    except (ValueError, TypeError, struct.error) as exc:
        raise invalid() from exc
