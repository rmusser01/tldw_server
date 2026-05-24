from __future__ import annotations

"""Contracts for Sync v2 materialized projection writers."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..models import SyncDomain, SyncEnvelope
from ..store import SyncV2Store


MaterializationStatus = Literal["applied", "conflict", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Result returned after attempting to project one accepted envelope."""

    status: MaterializationStatus
    conflict_type: str | None = None
    error_code: str | None = None
    message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class SyncMaterializer(Protocol):
    """Protocol implemented by domain-specific projection writers."""

    domain: SyncDomain

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Apply one accepted envelope to the live server projection."""


__all__ = ["MaterializationResult", "MaterializationStatus", "SyncMaterializer"]
