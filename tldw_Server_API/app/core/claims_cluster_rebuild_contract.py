"""Dependency-neutral Claims cluster rebuild settings and payload validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

from .claims_analytics_export_contract import is_routable_claims_owner_id_text

CLAIMS_REBUILD_CLUSTERS_JOB_TYPE = "claims_rebuild_clusters"
_LIMIT_MAXIMA = (50_000, 5_000, 33_554_432, 4_096, 2_000_000, 50_000_000, 50_000)


def _integer(value: object, minimum: int, maximum: int, name: str) -> None:
    """Reject booleans and coercions at the persisted contract boundary."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")


@dataclass(frozen=True)
class ClusterRebuildSettings:
    """Versioned effective settings independent of Jobs and domain services."""

    version: int
    method: Literal["exact", "embeddings"]
    min_size: int
    similarity_threshold: float | None
    algorithm_version: int

    def __post_init__(self) -> None:
        """Validate supported versions and normalize semantically ignored values."""
        _integer(self.version, 1, 1, "version")
        _integer(self.algorithm_version, 1, 1, "algorithm_version")
        _integer(self.min_size, 1, 1000, "min_size")
        if self.method not in ("exact", "embeddings"):
            raise ValueError("method must be exact or embeddings")
        if self.method == "exact":
            object.__setattr__(self, "similarity_threshold", None)
        else:
            value = self.similarity_threshold
            if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError("similarity_threshold must be finite in 0..1")
            object.__setattr__(self, "similarity_threshold", 0.0 if value == 0 else float(value))


@dataclass(frozen=True)
class ClusterRebuildLimits:
    """Operator-selected limits bounded by the approved hard ceilings."""

    exact_claims: int
    embedding_claims: int
    text_bytes: int
    vector_width: int
    embedding_components: int
    similarity_component_comparisons: int
    cleanup_references: int

    def __post_init__(self) -> None:
        """Validate all limits without accepting bool as an integer."""
        for field, maximum in zip(fields(self), _LIMIT_MAXIMA, strict=True):
            _integer(getattr(self, field.name), 1, maximum, field.name)


def canonical_cluster_settings(settings: ClusterRebuildSettings) -> dict[str, object]:
    """Return semantic settings; exact-mode thresholds never affect identity."""
    result: dict[str, object] = {
        "version": settings.version,
        "method": settings.method,
        "min_size": settings.min_size,
        "algorithm_version": settings.algorithm_version,
    }
    if settings.method == "embeddings":
        result["similarity_threshold"] = settings.similarity_threshold
    return result


def validate_cluster_rebuild_payload(payload: object) -> dict[str, object]:
    """Validate the sole supported bounded payload, with strict routable owner."""
    expected = {field.name for field in fields(ClusterRebuildSettings)} | {"owner_user_id"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("cluster rebuild payload must contain only the contract fields")
    if not is_routable_claims_owner_id_text(payload["owner_user_id"]):
        raise ValueError("owner_user_id must be a routable Claims owner")
    settings = ClusterRebuildSettings(**{key: payload[key] for key in expected if key != "owner_user_id"})
    return {
        **canonical_cluster_settings(settings),
        "owner_user_id": payload["owner_user_id"],
        "similarity_threshold": settings.similarity_threshold,
    }
