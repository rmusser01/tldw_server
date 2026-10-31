"""Connection lifecycle helpers owned by the package-native Media DB runtime."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


def _get_txn_conn(self: Any):
    return self._txn_conn_var.get()


def _set_txn_conn(self: Any, conn) -> None:
    self._txn_conn_var.set(conn)


def _get_tx_depth(self: Any) -> int:
    return int(self._tx_depth_var.get() or 0)


def _set_tx_depth(self: Any, depth: int) -> None:
    self._tx_depth_var.set(int(depth))


def _inc_tx_depth(self: Any) -> int:
    depth: int = self._get_tx_depth() + 1
    self._set_tx_depth(depth)
    return depth


def _dec_tx_depth(self: Any) -> int:
    depth: int = self._get_tx_depth() - 1
    if depth < 0:
        depth = 0
    self._set_tx_depth(depth)
    return depth


def _get_persistent_conn(self: Any):
    if self.backend_type == BackendType.POSTGRESQL:
        return self._persistent_conn_var.get()
    return self._persistent_conn


def _set_persistent_conn(self: Any, conn) -> None:
    if self.backend_type == BackendType.POSTGRESQL:
        self._persistent_conn_var.set(conn)
    else:
        self._persistent_conn = conn


def _release_persistent_conn(self: Any) -> None:
    try:
        conn = self._get_persistent_conn()
        if conn is not None:
            self.backend.get_pool().return_connection(conn)
    finally:
        self._set_persistent_conn(None)


def get_connection(self: Any):
    """Compatibility shim to return a usable connection."""
    txn_conn = self._get_txn_conn()
    if txn_conn is not None:
        return txn_conn
    if self.backend_type == BackendType.SQLITE:
        conn = self._get_persistent_conn()
        if self.is_memory_db and conn is not None:
            return conn
        return self.backend.get_pool().get_connection()
    conn = self._get_persistent_conn()
    if conn is None:
        conn = self.backend.get_pool().get_connection()
        self._set_persistent_conn(conn)
    try:
        self.backend.apply_scope(conn)
    except BaseException:
        # apply_scope discarded the connection; it may carry another tenant's GUCs.
        self._set_persistent_conn(None)
        raise
    return conn


def close_connection(self: Any):
    """Release persistent non-transaction connection if present."""
    if self._get_txn_conn() is not None:
        return
    _release_persistent_conn(self)


def release_context_connection(self: Any) -> None:
    """Return context-scoped Postgres connection to the pool (no-op for SQLite)."""
    if self.backend_type != BackendType.POSTGRESQL:
        return
    if self._get_txn_conn() is not None:
        return
    _release_persistent_conn(self)


__all__ = [
    "_dec_tx_depth",
    "_get_persistent_conn",
    "_get_tx_depth",
    "_get_txn_conn",
    "_inc_tx_depth",
    "_set_persistent_conn",
    "_set_tx_depth",
    "_set_txn_conn",
    "close_connection",
    "get_connection",
    "release_context_connection",
]
