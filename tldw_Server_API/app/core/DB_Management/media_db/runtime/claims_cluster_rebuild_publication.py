"""Atomic owner output replacement and lifetime receipts, never Jobs lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
    ClusterCalculation,
    ClusterPublicationIdentity,
    ClusterPublicationResult,
    ClusterRebuildStorageError,
    ClusterSourceSnapshot,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_rebuild_snapshot import (
    _check_bound,
    _id_pages,
    _placeholders,
    _validate_owner,
    lock_claim_cluster_cleanup,
    read_claim_cluster_source_on_connection,
)
from tldw_Server_API.app.core.Utils.claims_cluster_rebuild_contract import (
    ClusterRebuildLimits,
    ClusterRebuildSettings,
    canonical_cluster_settings,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

_MAX_INT = 2**63 - 1
_RESULT_FIELDS = (
    "clusters_created",
    "claims_assigned",
    "claims_skipped",
    "method",
    "similarity_threshold",
    "outcome",
    "reason",
)


def _corruption() -> ClusterRebuildStorageError:
    return ClusterRebuildStorageError("claims_cluster_corruption", retryable=False)


def _integer(value: object, minimum: int = 1, maximum: int = _MAX_INT) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _uuid(value: object) -> bool:
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _validate_identity(identity: ClusterPublicationIdentity) -> None:
    """Schema lengths alone cannot validate opaque persisted resource metadata."""
    _validate_owner(identity.owner_user_id)
    if not (
        _uuid(identity.job_uuid)
        and _uuid(identity.authority_uuid)
        and _integer(identity.generation)
        and _digest(identity.settings_fingerprint)
    ):
        raise ValueError("invalid canonical cluster publication identity")


def _receipt(db: MediaDatabase, conn: Any, identity: ClusterPublicationIdentity) -> ClusterPublicationResult | None:
    """Read and verify the immutable receipt, including its exact binding."""
    row = db._fetchone_with_connection(
        conn,
        "SELECT * FROM claims_cluster_publication_receipts WHERE owner_user_id = ? AND job_uuid = ?",
        (identity.owner_user_id, identity.job_uuid),
    )
    if row is None:
        return None
    if any(
        row[field] != getattr(identity, field)
        for field in (
            "owner_user_id",
            "job_uuid",
            "authority_uuid",
            "generation",
            "settings_fingerprint",
        )
    ):
        raise _corruption()
    threshold = row["similarity_threshold"]
    valid_method = (
        row["method"] == "exact"
        and threshold is None
        or row["method"] == "embeddings"
        and type(threshold) in (int, float)
        and math.isfinite(threshold)
        and 0 <= threshold <= 1
    )
    valid_outcome = (
        row["outcome"] == "ok"
        and row["reason"] is None
        or row["outcome"] == "skipped"
        and row["reason"] == "no_embeddings"
        and row["method"] == "embeddings"
        and row["clusters_created"] == 0
        and row["claims_assigned"] == 0
    )
    if not valid_method or not valid_outcome or any(not _integer(row[key], 0, 50_000) for key in _RESULT_FIELDS[:3]):
        raise _corruption()
    head = db._fetchone_with_connection(
        conn,
        "SELECT authority_uuid, generation, job_uuid FROM claims_cluster_publication_heads WHERE owner_user_id = ?",
        (identity.owner_user_id,),
    )
    if (
        head is None
        or not _integer(head["generation"])
        or head["generation"] < identity.generation
        or head["authority_uuid"] != identity.authority_uuid
        or not _uuid(head["job_uuid"])
        or head["generation"] == identity.generation
        and head["job_uuid"] != identity.job_uuid
    ):
        raise _corruption()
    return ClusterPublicationResult({key: row[key] for key in _RESULT_FIELDS}, False)


def load_claim_cluster_publication(
    self: MediaDatabase,
    *,
    identity: ClusterPublicationIdentity,
) -> ClusterPublicationResult | None:
    """Replay a Job's immutable domain result without consulting/pruning Jobs."""
    _validate_identity(identity)
    with self.transaction() as conn:
        return _receipt(self, conn, identity)


def _validate_calculation(
    identity: ClusterPublicationIdentity,
    settings: ClusterRebuildSettings,
    snapshot: ClusterSourceSnapshot,
    calculation: ClusterCalculation,
    limits: ClusterRebuildLimits,
) -> None:
    """Bound and validate untrusted calculated IDs before any destructive write."""
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical_cluster_settings(settings),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        fingerprint != identity.settings_fingerprint
        or snapshot.owner_user_id != identity.owner_user_id
        or not _digest(snapshot.digest)
    ):
        raise _corruption()
    count_limit = limits.exact_claims if settings.method == "exact" else limits.embedding_claims
    _check_bound(len(snapshot.rows), count_limit)
    source_ids: set[int] = set()
    previous = 0
    text_bytes = 0
    for row in snapshot.rows:
        if not (
            _integer(row.claim_id)
            and row.claim_id > previous
            and _integer(row.media_id)
            and _integer(row.chunk_index, -_MAX_INT - 1)
            and isinstance(row.claim_text, str)
            and row.claim_text.strip()
        ):
            raise _corruption()
        previous = row.claim_id
        source_ids.add(row.claim_id)
        text_bytes += len(row.claim_text.encode("utf-8"))
        _check_bound(text_bytes, limits.text_bytes)
    if not _integer(calculation.claims_skipped, 0, len(source_ids)):
        raise _corruption()
    if calculation.outcome == "skipped":
        if settings.method != "embeddings" or calculation.reason != "no_embeddings" or calculation.assignments:
            raise _corruption()
    elif calculation.outcome != "ok" or calculation.reason is not None:
        raise _corruption()
    _check_bound(len(calculation.assignments), count_limit)
    assigned: set[int] = set()
    output_bytes = 0
    for cluster in calculation.assignments:
        if not isinstance(cluster.canonical_claim_text, str) or not cluster.canonical_claim_text.strip():
            raise _corruption()
        output_bytes += len(cluster.canonical_claim_text.encode("utf-8"))
        _check_bound(output_bytes, limits.text_bytes)
        if not _integer(cluster.representative_claim_id) or cluster.representative_claim_id not in source_ids:
            raise _corruption()
        _check_bound(len(cluster.members), count_limit)
        if len(cluster.members) < settings.min_size:
            raise _corruption()
        members = set()
        for member in cluster.members:
            if not (
                _integer(member.claim_id)
                and member.claim_id in source_ids
                and member.claim_id not in assigned
                and type(member.similarity) in (int, float)
                and math.isfinite(member.similarity)
            ):
                raise _corruption()
            members.add(member.claim_id)
            assigned.add(member.claim_id)
        if cluster.representative_claim_id not in members:
            raise _corruption()
    if len(assigned) + calculation.claims_skipped > len(source_ids):
        raise _corruption()


def _lock_head(db: MediaDatabase, conn: Any, identity: ClusterPublicationIdentity) -> Any:
    """An insert conflict also serializes the initially absent owner head."""
    db._execute_with_connection(
        conn,
        "INSERT INTO claims_cluster_publication_heads (owner_user_id) VALUES (?) ON CONFLICT DO NOTHING",
        (identity.owner_user_id,),
    )
    query = "SELECT * FROM claims_cluster_publication_heads WHERE owner_user_id = ?"
    if db.backend_type == BackendType.POSTGRESQL:
        query = "SELECT * FROM claims_cluster_publication_heads WHERE owner_user_id = ? FOR UPDATE"
    head = db._fetchone_with_connection(conn, query, (identity.owner_user_id,))
    if head is None or not _integer(head["generation"], 0):
        raise _corruption()
    if head["generation"] == 0:
        if head["authority_uuid"] is not None or head["job_uuid"] is not None:
            raise _corruption()
    elif not _uuid(head["authority_uuid"]) or not _uuid(head["job_uuid"]):
        raise _corruption()
    return head


def _replace_output(
    db: MediaDatabase,
    conn: Any,
    *,
    owner: str,
    cluster_ids: tuple[int, ...],
    claim_ids: tuple[int, ...],
    calculation: ClusterCalculation,
) -> None:
    """Only write the locked/proven ID sets, never a broad owner Claims update."""
    # These templates substitute generated parameter markers only, never values.
    membership_delete = (
        "DELETE FROM claim_cluster_membership WHERE cluster_id IN (SELECT id FROM claim_clusters "
        "WHERE user_id = ? AND id IN ({ids}))"
    )
    assignment_clear = (
        "UPDATE Claims SET claim_cluster_id = NULL WHERE id IN ({ids}) AND media_id IN "
        "(SELECT id FROM Media WHERE COALESCE(CAST(owner_user_id AS TEXT), client_id) = ?)"
    )
    cluster_delete = "DELETE FROM claim_clusters WHERE user_id = ? AND id IN ({ids})"
    for page in _id_pages(cluster_ids):
        db._execute_with_connection(
            conn,
            membership_delete.replace("{ids}", _placeholders(page)),
            (owner, *page),
        )
    for page in _id_pages(claim_ids):
        cursor = db._execute_with_connection(
            conn,
            assignment_clear.replace("{ids}", _placeholders(page)),
            (*page, owner),
        )
        if cursor.rowcount != len(page):
            raise _corruption()
    for page in _id_pages(cluster_ids):
        db._execute_with_connection(
            conn,
            cluster_delete.replace("{ids}", _placeholders(page)),
            (owner, *page),
        )
    now = db._get_current_utc_timestamp_str()
    for cluster in calculation.assignments:
        sql = (
            "INSERT INTO claim_clusters (user_id, canonical_claim_text, representative_claim_id, "
            "cluster_version, watchlist_count, created_at, updated_at) VALUES (?, ?, ?, 1, 0, ?, ?)"
        )
        if db.backend_type == BackendType.POSTGRESQL:
            sql += " RETURNING id"
        cursor = db._execute_with_connection(
            conn,
            sql,
            (owner, cluster.canonical_claim_text, cluster.representative_claim_id, now, now),
        )
        cluster_id = cursor.fetchone()["id"] if db.backend_type == BackendType.POSTGRESQL else cursor.lastrowid
        if not cluster_id:
            raise _corruption()
        for member in cluster.members:
            cursor = db._execute_with_connection(
                conn,
                "UPDATE Claims SET claim_cluster_id = ? WHERE id = ? AND media_id IN "
                "(SELECT id FROM Media WHERE COALESCE(CAST(owner_user_id AS TEXT), client_id) = ?)",
                (cluster_id, member.claim_id, owner),
            )
            if cursor.rowcount != 1:
                raise _corruption()
            db._execute_with_connection(
                conn,
                "INSERT INTO claim_cluster_membership (cluster_id, claim_id, similarity_score, cluster_joined_at) "
                "SELECT ?, c.id, ?, ? FROM Claims c JOIN Media m ON m.id = c.media_id "
                "WHERE c.id = ? AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?",
                (cluster_id, member.similarity, now, member.claim_id, owner),
            )


def publish_claim_cluster_rebuild(
    self: MediaDatabase,
    *,
    identity: ClusterPublicationIdentity,
    settings: ClusterRebuildSettings,
    snapshot: ClusterSourceSnapshot,
    calculation: ClusterCalculation,
    limits: ClusterRebuildLimits,
) -> ClusterPublicationResult:
    """Serialize head -> receipt -> old clusters -> Media -> Claims through commit.

    The locked source re-read is the input point; later arrivals may wait for the
    next rebuild. An earlier receipt always replays without restoring old output.
    """
    _validate_identity(identity)
    # The publisher must own the commit that its newly_published decision describes.
    if self._get_txn_conn() is not None:
        raise ValueError("publication requires its own transaction")
    with self.transaction() as conn:
        head = _lock_head(self, conn, identity)
        replay = _receipt(self, conn, identity)
        if replay is not None:
            return replay
        if head["generation"] and (
            head["authority_uuid"] != identity.authority_uuid or head["generation"] == identity.generation
        ):
            raise _corruption()
        # Equal-generation UUID collisions are corruption even behind a newer head.
        collision = self._fetchone_with_connection(
            conn,
            "SELECT job_uuid FROM claims_cluster_publication_receipts "
            "WHERE owner_user_id = ? AND authority_uuid = ? AND generation = ?",
            (identity.owner_user_id, identity.authority_uuid, identity.generation),
        )
        if collision is not None:
            raise _corruption()
        _validate_calculation(identity, settings, snapshot, calculation, limits)
        result = {
            "clusters_created": len(calculation.assignments),
            "claims_assigned": sum(len(cluster.members) for cluster in calculation.assignments),
            "claims_skipped": calculation.claims_skipped,
            "method": settings.method,
            "similarity_threshold": settings.similarity_threshold,
            "outcome": calculation.outcome,
            "reason": calculation.reason,
        }
        if head["generation"] > identity.generation:
            return ClusterPublicationResult(
                {
                    **result,
                    "clusters_created": 0,
                    "claims_assigned": 0,
                    "claims_skipped": 0,
                    "outcome": "skipped",
                    "reason": "superseded",
                },
                False,
            )
        query = "SELECT id FROM claim_clusters WHERE user_id = ? ORDER BY id LIMIT ?"
        if self.backend_type == BackendType.POSTGRESQL:
            query = "SELECT id FROM claim_clusters WHERE user_id = ? ORDER BY id LIMIT ? FOR UPDATE"
        clusters = self._fetchall_with_connection(conn, query, (identity.owner_user_id, limits.cleanup_references + 1))
        _check_bound(len(clusters), limits.cleanup_references)
        cleanup_ids = lock_claim_cluster_cleanup(
            self,
            conn,
            owner_user_id=identity.owner_user_id,
            limit=limits.cleanup_references,
        )
        current = read_claim_cluster_source_on_connection(
            self,
            conn,
            owner_user_id=identity.owner_user_id,
            claim_limit=limits.exact_claims if settings.method == "exact" else limits.embedding_claims,
            text_byte_limit=limits.text_bytes,
            lock=True,
        )
        if current != snapshot:
            raise ClusterRebuildStorageError("claims_cluster_source_changed", retryable=True)
        # Source and cleanup locks must be held before validating the deletion
        # set: a new cluster can otherwise assign a source row without digest drift.
        # Do not acquire new cluster locks after Claims locks; retry on arrival.
        checked_clusters = self._fetchall_with_connection(
            conn,
            "SELECT id FROM claim_clusters WHERE user_id = ? ORDER BY id LIMIT ?",
            (identity.owner_user_id, limits.cleanup_references + 1),
        )
        cluster_ids = tuple(int(row["id"]) for row in clusters)
        if tuple(int(row["id"]) for row in checked_clusters) != cluster_ids:
            raise ClusterRebuildStorageError("claims_cluster_source_changed", retryable=True)
        if calculation.outcome == "ok":
            _replace_output(
                self,
                conn,
                owner=identity.owner_user_id,
                cluster_ids=cluster_ids,
                claim_ids=tuple(sorted(set(cleanup_ids) | {row.claim_id for row in current.rows})),
                calculation=calculation,
            )
        self._execute_with_connection(
            conn,
            "UPDATE claims_cluster_publication_heads SET authority_uuid = ?, generation = ?, job_uuid = ? "
            "WHERE owner_user_id = ?",
            (identity.authority_uuid, identity.generation, identity.job_uuid, identity.owner_user_id),
        )
        self._execute_with_connection(
            conn,
            "INSERT INTO claims_cluster_publication_receipts "
            "(owner_user_id, job_uuid, authority_uuid, generation, settings_fingerprint, "
            "clusters_created, claims_assigned, claims_skipped, method, similarity_threshold, outcome, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                identity.owner_user_id,
                identity.job_uuid,
                identity.authority_uuid,
                identity.generation,
                identity.settings_fingerprint,
                *(result[key] for key in _RESULT_FIELDS),
            ),
        )
        return ClusterPublicationResult(result, True)
