"""Media-owned cluster calculation/publication values; no Jobs lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


class ClusterRebuildStorageError(RuntimeError):
    """Sanitized storage failure suitable for translation at the Jobs boundary.

    Codes include claims_cluster_source_changed, claims_cluster_foreign_reference,
    claims_cluster_corruption and claims_cluster_resource_limit. Never attach rows.
    """

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ClusterSourceRow:
    """One clustering-relevant source row in stable claim-ID order."""

    claim_id: int
    media_id: int
    chunk_index: int
    claim_text: str


@dataclass(frozen=True)
class ClusterSourceSnapshot:
    """Bounded owner input and its length-delimited SHA-256 digest."""

    owner_user_id: str
    rows: tuple[ClusterSourceRow, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class ClusterMember:
    """A calculated member and its similarity to the representative."""

    claim_id: int
    similarity: float


@dataclass(frozen=True)
class ClusterAssignment:
    """Calculated output, not yet persisted."""

    canonical_claim_text: str
    representative_claim_id: int
    members: tuple[ClusterMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))


@dataclass(frozen=True)
class ClusterCalculation:
    """Pure calculation output, including legitimate embedding skips."""

    assignments: tuple[ClusterAssignment, ...]
    outcome: Literal["ok", "skipped"]
    reason: str | None
    claims_skipped: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))


@dataclass(frozen=True)
class ClusterPublicationIdentity:
    """Immutable authority supplied by the caller, not a domain lease."""

    owner_user_id: str
    job_uuid: str
    authority_uuid: str
    generation: int
    settings_fingerprint: str


@dataclass(frozen=True)
class ClusterPublicationResult:
    """Defensively frozen, source-free receipt result and side-effect decision."""

    result: dict[str, int | float | str | None]
    newly_published: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", MappingProxyType(dict(self.result)))
