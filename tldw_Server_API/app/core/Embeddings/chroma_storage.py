"""Cooperating local Chroma access: lock through SDK destruction, not lookup.

Only inert handles are cached. SDK instances/collections never escape a session.
The sibling lock file is coordination metadata, outside the owner's directory;
never unlink it, including on owner deletion. No silent thread-only fallback.
"""

from __future__ import annotations

import errno
import math
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection
from chromadb.api.types import validate_ids
from chromadb.errors import DuplicateIDError, NotFoundError

from tldw_Server_API.app.core.Infrastructure.distributed_lock import (
    _acquire_platform_file_lock,
    _release_platform_file_lock,
)

# A failed SDK stop cannot prove quiescence. Retain its SDK AND native lock
# until process exit; dropping a manager must not unlock a possibly live SDK.
_FAILED_SHUTDOWNS: dict[str, StorageSession] = {}


class ChromaStorageError(RuntimeError):
    """Source-free storage failure; absence is a separate, proven condition."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class StorageSession:
    """Reentrant per-path thread/process exclusion with bounded acquisition.

    Opening and stopping an SDK on each outer session is deliberate: retaining a
    live SDK outside the file lock would leave its background/index work unlocked.
    Unclosed managers and escaped collection handles retain no SDK or OS lock.
    """

    def __init__(self, path: Path, *, timeout: float = 5.0) -> None:
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("Chroma storage lock timeout must be finite in (0, 60]")
        self.path = path
        self.timeout = timeout
        self.lock_path = path.parent.parent / f".chroma-{path.parent.name}.lock"
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._fd: int | None = None
        self.sdk: Any = None
        self._stop = None
        self._mode: bool | None = None
        self._shutdown_failed = False

    def __enter__(self) -> StorageSession:
        deadline = time.monotonic() + self.timeout
        if not self._thread_lock.acquire(timeout=self.timeout):
            raise ChromaStorageError("vector_busy", retryable=True)
        if self._shutdown_failed:
            self._thread_lock.release()
            raise ChromaStorageError("vector_shutdown", retryable=True)
        if self._depth:
            self._depth += 1
            return self
        try:
            # The parent must already exist; readers never mkdir, even for locks.
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            self._fd = os.open(self.lock_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(self._fd).st_mode):
                raise ChromaStorageError("vector_access", retryable=False)
            while True:
                try:
                    _acquire_platform_file_lock(self._fd)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise ChromaStorageError("vector_lock_unsupported", retryable=False) from exc
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ChromaStorageError("vector_busy", retryable=True) from exc
                    time.sleep(min(0.02, remaining))
            self._depth = 1
            return self
        except (OSError, ChromaStorageError):
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._thread_lock.release()
            raise

    def __exit__(self, *exc: Any) -> None:
        self._depth -= 1
        try:
            if not self._depth:
                stopped = False
                try:
                    if self._stop is not None:
                        self._stop()
                    # RustBindingsAPI.stop() drops Bindings; raw Bindings uses
                    # that same destructor. Drop it before unlocking the store.
                    self.sdk = None
                    stopped = True
                finally:
                    if not stopped:
                        self._shutdown_failed = True
                        _FAILED_SHUTDOWNS[str(self.path)] = self
                    else:
                        self._stop = None
                        self._mode = None
                        if self._fd is not None:
                            try:
                                _release_platform_file_lock(self._fd)
                            finally:
                                os.close(self._fd)
                                self._fd = None
        finally:
            self._thread_lock.release()

    def client(self, factory, *, existing_only: bool):
        """Lazily open an SDK only inside the coordinated lifetime."""
        if not self._depth:
            raise RuntimeError("Chroma SDK access requires a storage session")
        if self.sdk is not None and self._mode != existing_only:
            raise ChromaStorageError("vector_mixed_access", retryable=False)
        if self.sdk is None:
            self.sdk = factory() if existing_only else self._open_persistent(factory)
            self._mode = existing_only
            if not existing_only:
                self._stop = lambda: _stop_client(self.sdk)
        return self.sdk

    def _open_persistent(self, factory):
        """Retain partial SDK startup for shutdown before leaving the lock.

        Chroma registers its System before starting it. A failing constructor
        can therefore retain a runtime even though it returned no client.
        """
        from chromadb.api.shared_system_client import SharedSystemClient

        cache = SharedSystemClient._identifier_to_system
        identifier = str(self.path)
        previous = cache.get(identifier)
        opened = False
        try:
            client = factory()
            opened = True
            return client
        finally:
            if not opened:
                system = cache.get(identifier)
                if system is not None and system is not previous:
                    self.sdk = system
                    self._stop = lambda: _stop_system(system, identifier, cache)


def _stop_client(client) -> None:
    """Use the SDK's shutdown protocol and evict its stopped system cache."""
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    _stop_system(client._system, client._identifier, type(client)._identifier_to_system)


def _stop_system(system, identifier, cache) -> None:
    """Evict only a system whose shutdown actually completed."""
    system.stop()
    if cache.get(identifier) is system:
        cache.pop(identifier, None)


class GuardedClient:
    """Public manager client facade; neither methods nor results leak the SDK."""

    def __init__(self, session: StorageSession, factory, *, existing_only: bool = False) -> None:
        self.session = session
        self.factory = factory
        self.existing_only = existing_only
        self.closed = False

    def _sdk(self):
        if self.closed:
            raise ChromaStorageError("vector_closed", retryable=True)
        return self.session.client(self.factory, existing_only=self.existing_only)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def invoke(*args, **kwargs):
            if self.existing_only and name not in ("get_collection", "list_collections", "count_collections"):
                raise ChromaStorageError("vector_read_only", retryable=False)
            with self.session:
                result = getattr(self._sdk(), name)(*args, **kwargs)
                if isinstance(result, Collection):
                    return GuardedCollection(self, result)
                if isinstance(result, list):
                    return [GuardedCollection(self, c) if isinstance(c, Collection) else c for c in result]
                return result

        return invoke


class GuardedCollection:
    """Collection identity that resolves anew under the complete SDK lifetime."""

    def __init__(self, client: GuardedClient, collection: Collection) -> None:
        self._guard = client
        self.name = collection.name
        self.id = str(collection.id)
        # Keep user embedding/data adapters, but never retain the SDK collection
        # (which owns a live API/system). Explicit embedding writes still work.
        self._embedding_function = collection._embedding_function
        self._data_loader = collection._data_loader

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in ("metadata", "configuration", "schema", "tenant", "database"):
            with self._guard.session:
                return getattr(self._resolve(), name)

        def invoke(*args, **kwargs):
            if self._guard.existing_only and name not in ("get", "count", "peek", "query"):
                raise ChromaStorageError("vector_read_only", retryable=False)
            with self._guard.session:
                result = getattr(self._resolve(), name)(*args, **kwargs)
                if name == "modify":
                    new_name = kwargs.get("name") or (args[0] if args else None)
                    if new_name:
                        self.name = new_name
                return result

        return invoke

    def _resolve(self):
        collection = self._guard._sdk().get_collection(
            name=self.name, embedding_function=self._embedding_function, data_loader=self._data_loader
        )
        if str(collection.id) != self.id:
            raise NotFoundError("Collection handle no longer refers to the same collection")
        return collection


class ExistingClient:
    """Narrow no-create ABI adapter, never a fallback to a writer or stub."""

    def __init__(self, session: StorageSession) -> None:
        self.session = session
        self.closed = False

    def validate(self, name: str | None = None) -> str | None:
        from tldw_Server_API.app.core.DB_Management.chroma_existing import validate_existing_chroma

        if self.closed:
            raise ChromaStorageError("vector_closed", retryable=True)
        return validate_existing_chroma(self.session.path, collection_name=name)

    def _factory(self):
        import inspect

        import chromadb_rust_bindings as bindings

        required = {"allow_reset", "sqlite_db_config", "hnsw_cache_size", "persist_path"}
        if set(inspect.signature(bindings.Bindings).parameters) != required:
            raise ChromaStorageError("vector_unsupported_sdk", retryable=False)
        return bindings.Bindings(
            allow_reset=False,
            sqlite_db_config=bindings.SqliteDBConfig(
                hash_type=bindings.MigrationHash.MD5,
                migration_mode=bindings.MigrationMode.Validate,
                url=(self.session.path / "chroma.sqlite3").as_uri() + "?mode=rw",
            ),
            hnsw_cache_size=5,
            persist_path=str(self.session.path),
        )

    def get_collection(self, name: str):
        with self.session:
            collection_id = self.validate(name)
            if collection_id is None:
                raise NotFoundError("Collection is absent from validated existing storage")
            return ExistingCollection(self, name, collection_id)


class ExistingCollection:
    """Strict lookup identity, validated under the same lock as each vector read."""

    def __init__(self, client: ExistingClient, name: str, collection_id: str) -> None:
        self._guard = client
        self.name = name
        self.id = collection_id

    def get(self, *, ids, include):
        try:
            validate_ids(list(ids))
        except (DuplicateIDError, ValueError) as exc:
            raise ChromaStorageError("vector_invalid_request", retryable=False) from exc
        with self._guard.session:
            if self._guard.validate(self.name) != self.id:
                raise ChromaStorageError("vector_source_changed", retryable=True)
            client = self._guard.session.client(self._guard._factory, existing_only=True)
            collection = client.get_collection(self.name, "default_tenant", "default_database")
            if str(collection.id) != self.id:
                raise ChromaStorageError("vector_source_changed", retryable=True)
            response = client.get(
                self.id, ids=list(ids), include=include, tenant="default_tenant", database="default_database"
            )
            return {"ids": response.ids, "embeddings": response.embeddings}
