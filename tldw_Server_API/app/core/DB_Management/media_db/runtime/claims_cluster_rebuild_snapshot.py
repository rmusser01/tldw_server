"""Bounded owner snapshots and transaction-internal cluster integrity locks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.claims_analytics_export_contract import is_routable_claims_owner_id_text
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
    ClusterRebuildStorageError,
    ClusterSourceRow,
    ClusterSourceSnapshot,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

_PAGE_SIZE = 256


def _validate_limit(value: int, maximum: int) -> None:
    """Keep caller-selected limits within the approved hard ceilings."""
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"limit must be an integer in 1..{maximum}")


def _validate_owner(owner_user_id: str) -> None:
    """Do not coerce invalid owners into a storage scope."""
    if not is_routable_claims_owner_id_text(owner_user_id):
        raise ValueError("owner_user_id must be a routable Claims owner")


def _check_bound(size: int, limit: int) -> None:
    """Fail before any destructive operation when a bounded set overflows."""
    if size > limit:
        raise ClusterRebuildStorageError("claims_cluster_resource_limit", retryable=False)


def _lock_suffix(db: MediaDatabase) -> str:
    return " FOR UPDATE" if db.backend_type == BackendType.POSTGRESQL else ""


def _id_pages(ids: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Bound each parameter list below both backends' statement limits."""
    for offset in range(0, len(ids), _PAGE_SIZE):
        yield ids[offset : offset + _PAGE_SIZE]


def _placeholders(ids: tuple[int, ...]) -> str:
    """Interpolate only generated parameter markers, never identifiers or values."""
    return ",".join("?" for _ in ids)


def _assert_reference_visibility(db: MediaDatabase, conn: Any) -> None:
    """Require full reference enumeration without bypassing RLS.

    Ordinary SELECT relation locks pin these base-table definitions through commit.
    Any deployed RLS on reference tables is unsupported: even permissive-looking
    policies are not proof that another reference cannot be hidden.
    """
    if db.backend_type != BackendType.POSTGRESQL:
        return
    db._execute_with_connection(conn, "SELECT id FROM Claims LIMIT 0")
    db._execute_with_connection(conn, "SELECT claim_id FROM claim_cluster_membership LIMIT 0")
    db._execute_with_connection(conn, "SELECT id FROM claim_clusters LIMIT 0")
    rows = db._fetchall_with_connection(
        conn,
        "SELECT relkind, relrowsecurity FROM pg_class WHERE oid IN "
        "(to_regclass('claims'), to_regclass('claim_cluster_membership'), to_regclass('claim_clusters'))",
    )
    if len(rows) != 3 or any(row["relkind"] != "r" or row["relrowsecurity"] for row in rows):
        raise ClusterRebuildStorageError("claims_cluster_corruption", retryable=False)


def lock_claim_rows(
    db: MediaDatabase,
    conn: Any,
    claim_ids: tuple[int, ...],
    *,
    owner_user_id: str | None,
) -> None:
    """Resolve references, prove/lock Media ownership, then lock Claims by ID.

    The optional owner is used only by the existing membership writer, whose
    visibility contract is unchanged. Cleanup and publication always supply it.
    No foreign Claims lock is taken until *all* referenced Media ownership is proven.
    """
    if conn is None:
        raise ValueError("an active caller connection is required")
    media_by_claim: dict[int, int] = {}
    ids = tuple(sorted(set(claim_ids)))
    for page in _id_pages(ids):
        query = "SELECT id, media_id FROM Claims WHERE id IN ({ids}) ORDER BY id"
        rows = db._fetchall_with_connection(conn, query.replace("{ids}", _placeholders(page)), page)
        if len(rows) != len(page):
            raise ClusterRebuildStorageError("claims_cluster_corruption", retryable=False)
        media_by_claim.update((int(row["id"]), int(row["media_id"])) for row in rows)
    for page in _id_pages(tuple(sorted(set(media_by_claim.values())))):
        query = "SELECT id FROM Media WHERE id IN ({ids})"
        params = page
        if owner_user_id is not None:
            query += " AND COALESCE(CAST(owner_user_id AS TEXT), client_id) = ?"
            params = (*page, owner_user_id)
        query += " ORDER BY id" + _lock_suffix(db)
        rows = db._fetchall_with_connection(conn, query.replace("{ids}", _placeholders(page)), params)
        if len(rows) != len(page):
            # Hidden, deleted or missing Media is NOT proof of absent references.
            raise ClusterRebuildStorageError("claims_cluster_foreign_reference", retryable=False)
    for page in _id_pages(ids):
        media_ids = tuple(sorted({media_by_claim[claim_id] for claim_id in page}))
        query = "SELECT id, media_id FROM Claims WHERE id IN ({ids}) AND media_id IN ({media_ids}) ORDER BY id"
        if db.backend_type == BackendType.POSTGRESQL:
            query = "SELECT id, media_id FROM Claims WHERE id IN ({ids}) AND media_id IN ({media_ids}) ORDER BY id FOR UPDATE"
        query = query.replace("{ids}", _placeholders(page)).replace("{media_ids}", _placeholders(media_ids))
        rows = db._fetchall_with_connection(conn, query, (*page, *media_ids))
        if len(rows) != len(page) or any(int(row["media_id"]) != media_by_claim[int(row["id"])] for row in rows):
            raise ClusterRebuildStorageError("claims_cluster_source_changed", retryable=True)


def _read_rows(
    db: MediaDatabase,
    conn: Any,
    *,
    owner_user_id: str,
    claim_limit: int,
    text_byte_limit: int,
) -> ClusterSourceSnapshot:
    """Keyset-page one consistent view; hash fields with unsigned length prefixes."""
    rows: list[ClusterSourceRow] = []
    digest = hashlib.sha256()
    digest.update(b"claims-cluster-source-v1\x00")
    owner_bytes = owner_user_id.encode("utf-8")
    digest.update(len(owner_bytes).to_bytes(8, "big"))
    digest.update(owner_bytes)
    after = 0
    inspected = 0
    text_bytes = 0
    query = (
        "SELECT c.id, c.media_id, c.chunk_index, length(CAST(c.claim_text AS BLOB)) AS claim_bytes, "
        "CASE WHEN SUM(length(CAST(c.claim_text AS BLOB))) OVER (ORDER BY c.id ROWS UNBOUNDED PRECEDING) <= ? "
        "THEN c.claim_text ELSE NULL END AS claim_text FROM Claims c "
        "JOIN Media m ON m.id = c.media_id WHERE c.id > ? AND c.deleted = 0 AND m.deleted = 0 "
        "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ? ORDER BY c.id LIMIT ?"
    )
    if db.backend_type == BackendType.POSTGRESQL:
        query = query.replace("length(CAST(c.claim_text AS BLOB))", "octet_length(c.claim_text)")
    while True:
        page = db._fetchall_with_connection(
            conn,
            query,
            (text_byte_limit - text_bytes, after, owner_user_id, min(_PAGE_SIZE, claim_limit - inspected + 1)),
        )
        if not page:
            break
        for raw in page:
            after = int(raw["id"])
            inspected += 1
            _check_bound(inspected, claim_limit)
            text_bytes += int(raw["claim_bytes"] or 0)
            _check_bound(text_bytes, text_byte_limit)
            text = raw["claim_text"]
            if not isinstance(text, str) or not text.strip():
                continue
            _check_bound(len(rows) + 1, claim_limit)
            row = ClusterSourceRow(after, int(raw["media_id"]), int(raw["chunk_index"]), text)
            rows.append(row)
            for field in (row.claim_id, row.media_id, row.chunk_index, row.claim_text):
                encoded = str(field).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return ClusterSourceSnapshot(owner_user_id, tuple(rows), digest.hexdigest())


def read_claim_cluster_source_on_connection(
    db: MediaDatabase,
    conn: Any,
    *,
    owner_user_id: str,
    claim_limit: int,
    text_byte_limit: int,
    lock: bool,
) -> ClusterSourceSnapshot:
    """Use only the caller's transaction; never commit or create storage here.

    A publisher must acquire old-cluster cleanup locks before calling with lock=True.
    Its source digest comparison and all writes belong to that same transaction.
    """
    _validate_owner(owner_user_id)
    _validate_limit(claim_limit, 50_000)
    _validate_limit(text_byte_limit, 33_554_432)
    if conn is None:
        raise ValueError("an active caller connection is required")
    result = _read_rows(db, conn, owner_user_id=owner_user_id, claim_limit=claim_limit, text_byte_limit=text_byte_limit)
    if lock:
        _assert_reference_visibility(db, conn)
        lock_claim_rows(db, conn, tuple(row.claim_id for row in result.rows), owner_user_id=owner_user_id)
        current = _read_rows(
            db,
            conn,
            owner_user_id=owner_user_id,
            claim_limit=claim_limit,
            text_byte_limit=text_byte_limit,
        )
        if current.digest != result.digest:
            raise ClusterRebuildStorageError("claims_cluster_source_changed", retryable=True)
    return result


def read_claim_cluster_source(
    self: MediaDatabase,
    *,
    owner_user_id: str,
    claim_limit: int,
    text_byte_limit: int,
) -> ClusterSourceSnapshot:
    """Read without a writer reservation (SQLite) or snapshot drift (PostgreSQL)."""
    if self._get_txn_conn() is not None:
        raise ValueError("use the connection-taking snapshot helper inside a transaction")
    if self.backend_type == BackendType.POSTGRESQL:
        pool = self.backend.get_pool()
        conn = pool.get_connection()
        try:
            # Pool checkout installs scope with a SELECT. Close that implicit
            # transaction before beginning the source snapshot, as clone snapshots do.
            conn.rollback()
            self.backend.apply_and_verify_scope(conn)
            with self.backend.transaction(conn):
                self._execute_with_connection(conn, "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
                return read_claim_cluster_source_on_connection(
                    self,
                    conn,
                    owner_user_id=owner_user_id,
                    claim_limit=claim_limit,
                    text_byte_limit=text_byte_limit,
                    lock=False,
                )
        finally:
            pool.return_connection(conn)
    conn = self.get_connection()
    began = False
    try:
        if conn.in_transaction:
            raise ValueError("snapshot requires an idle SQLite connection")
        conn.execute("BEGIN DEFERRED")
        began = True
        return read_claim_cluster_source_on_connection(
            self,
            conn,
            owner_user_id=owner_user_id,
            claim_limit=claim_limit,
            text_byte_limit=text_byte_limit,
            lock=False,
        )
    finally:
        if began:
            conn.rollback()
        if conn is not self._get_persistent_conn():
            self.backend.get_pool().return_connection(conn)


def lock_claim_cluster_cleanup(
    db: MediaDatabase,
    conn: Any,
    *,
    owner_user_id: str,
    limit: int,
) -> tuple[int, ...]:
    """Lock bounded old clusters and prove every reference before destructive writes."""
    _validate_owner(owner_user_id)
    _validate_limit(limit, 50_000)
    if conn is None:
        raise ValueError("an active caller connection is required")
    _assert_reference_visibility(db, conn)
    query = "SELECT id, representative_claim_id FROM claim_clusters WHERE user_id = ? ORDER BY id LIMIT ?"
    if db.backend_type == BackendType.POSTGRESQL:
        query = (
            "SELECT id, representative_claim_id FROM claim_clusters WHERE user_id = ? ORDER BY id LIMIT ? FOR UPDATE"
        )
    clusters = db._fetchall_with_connection(conn, query, (owner_user_id, limit + 1))
    _check_bound(len(clusters), limit)
    claim_ids = {int(row["representative_claim_id"]) for row in clusters if row["representative_claim_id"] is not None}
    _check_bound(len(claim_ids), limit)
    # Enumerate references directly: no Media join may hide transferred references.
    for query in (
        "SELECT c.id AS claim_id FROM Claims c JOIN claim_clusters old ON old.id = c.claim_cluster_id "
        "WHERE old.user_id = ? ORDER BY c.id LIMIT ?",
        "SELECT cm.claim_id FROM claim_cluster_membership cm JOIN claim_clusters old ON old.id = cm.cluster_id "
        "WHERE old.user_id = ? ORDER BY cm.cluster_id, cm.claim_id LIMIT ?",
    ):
        references = db._fetchall_with_connection(conn, query, (owner_user_id, limit + 1))
        _check_bound(len(references), limit)
        claim_ids.update(int(row["claim_id"]) for row in references)
        _check_bound(len(claim_ids), limit)
    ids = tuple(sorted(claim_ids))
    lock_claim_rows(db, conn, ids, owner_user_id=owner_user_id)
    return ids
