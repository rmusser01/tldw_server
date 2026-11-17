"""Bounded, pure version-1 cluster calculation; never publishes domain data."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import ClusterRebuildLimits, ClusterRebuildSettings
from tldw_Server_API.app.core.Claims_Extraction.claims_embeddings import claim_embedding_id
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
    ClusterAssignment,
    ClusterCalculation,
    ClusterMember,
    ClusterRebuildStorageError,
    ClusterSourceRow,
    ClusterSourceSnapshot,
)

VectorReader = Callable[[tuple[str, ...]], Mapping[str, Sequence[float]]]


def _failure(reason: str = "corruption") -> ClusterRebuildStorageError:
    """Keep errors source-free at the calculation boundary."""
    return ClusterRebuildStorageError(f"claims_cluster_{reason}", retryable=False)


def _norm(vector: Sequence[float]) -> float:
    """Match legacy arithmetic, rejecting overflowing/underflowing norms."""
    value = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(value) or value <= 0:
        raise _failure()
    return value


@dataclass
class _Group:
    members: list[tuple[ClusterSourceRow, list[float], float]]
    total: list[float]
    centroid: list[float]
    norm: float


def calculate_claim_cluster_rebuild(
    snapshot: ClusterSourceSnapshot,
    *,
    settings: ClusterRebuildSettings,
    limits: ClusterRebuildLimits,
    read_vectors: VectorReader | None,
) -> ClusterCalculation:
    """Preserve legacy grouping within explicit input and component-work ceilings.

    Text and vector limits bound retained input, not process RSS or wall time.
    Working normalized strings and centroids add only a constant-factor copy.
    Missing vectors skip publication; malformed vectors/storage fail closed.
    """
    rows = snapshot.rows
    count_limit = limits.exact_claims if settings.method == "exact" else limits.embedding_claims
    if len(rows) > count_limit:
        raise _failure("resource_limit")
    text_bytes = 0
    previous_id = 0
    for row in rows:
        if row.claim_id <= previous_id or not isinstance(row.claim_text, str):
            raise _failure()
        previous_id = row.claim_id
        text_bytes += len(row.claim_text.encode("utf-8"))
        if text_bytes > limits.text_bytes:
            raise _failure("resource_limit")

    if settings.method == "exact":
        groups: dict[str, list[ClusterSourceRow]] = {}
        normalized_bytes = 0
        for row in rows:
            text = row.claim_text.strip()
            if not text:
                continue
            key = " ".join(text.lower().split())
            if key not in groups:
                normalized_bytes += len(key.encode("utf-8"))
                if normalized_bytes > limits.text_bytes:
                    raise _failure("resource_limit")
            groups.setdefault(key, []).append(row)
        assignments = tuple(
            ClusterAssignment(
                group[0].claim_text.strip(), group[0].claim_id, tuple(ClusterMember(row.claim_id, 1.0) for row in group)
            )
            for group in groups.values()
            if len(group) >= settings.min_size
        )
        return ClusterCalculation(assignments, "ok", None, 0)

    if read_vectors is None:
        return ClusterCalculation((), "skipped", "no_embeddings", len(rows))
    ids = tuple(claim_embedding_id(row.media_id, row.chunk_index, row.claim_text) for row in rows)
    # Legacy IDs can collide: the last source row wins, not every matching row.
    by_id = dict(zip(ids, rows, strict=True))
    ids = tuple(by_id)
    embeddings: list[tuple[ClusterSourceRow, list[float], float]] = []
    components = 0
    width = None
    missing = len(rows) - len(ids)
    for start in range(0, len(ids), 200):
        batch = ids[start : start + 200]
        try:
            result = read_vectors(batch)
        except ClusterRebuildStorageError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise ClusterRebuildStorageError("claims_cluster_vector_access", retryable=True) from exc
        if not isinstance(result, Mapping) or len(result) > len(batch) or any(key not in batch for key in result):
            raise _failure()
        missing += len(batch) - len(result)
        for embed_id, raw in result.items():
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise _failure()
            if len(raw) > limits.vector_width:
                raise _failure("resource_limit")
            components += len(raw)
            if components > limits.embedding_components:
                raise _failure("resource_limit")
            try:
                vector = [float(value) for value in raw]
            except (TypeError, ValueError, OverflowError) as exc:
                raise _failure() from exc
            if any(isinstance(value, bool) for value in raw) or not all(math.isfinite(v) for v in vector):
                raise _failure()
            if width is not None and len(vector) != width:
                raise _failure()
            width = len(vector)
            embeddings.append((by_id[embed_id], vector, _norm(vector)))
    if not embeddings:
        return ClusterCalculation((), "skipped", "no_embeddings", missing)

    comparisons = 0

    def similarity(a: Sequence[float], norm_a: float, b: Sequence[float], norm_b: float) -> float:
        nonlocal comparisons
        comparisons += len(a)
        if comparisons > limits.similarity_component_comparisons:
            raise _failure("resource_limit")
        dot = 0.0
        for x, y in zip(a, b, strict=True):
            dot += x * y
        value = dot / (norm_a * norm_b)
        if not math.isfinite(value):
            raise _failure()
        return value

    clusters: list[_Group] = []
    for entry in sorted(embeddings, key=lambda entry: entry[0].claim_id):
        _, vector, norm = entry
        best = None
        best_sim = settings.similarity_threshold
        for group in clusters:
            sim = similarity(vector, norm, group.centroid, group.norm)
            if sim > best_sim:
                best, best_sim = group, sim
        if best is None:
            clusters.append(_Group([entry], list(vector), list(vector), norm))
        else:
            best.members.append(entry)
            for i, value in enumerate(vector):
                best.total[i] += value
            best.centroid = [value / len(best.members) for value in best.total]
            best.norm = _norm(best.centroid)
    assignments = tuple(
        ClusterAssignment(
            group.members[0][0].claim_text,
            group.members[0][0].claim_id,
            tuple(
                ClusterMember(row.claim_id, similarity(vector, norm, group.centroid, group.norm))
                for row, vector, norm in group.members
            ),
        )
        for group in clusters
        if len(group.members) >= settings.min_size
    )
    return ClusterCalculation(assignments, "ok", None, missing)
