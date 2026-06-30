"""Evidence contracts for retrieval and post-retrieval coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedEvidence:
    """Canonical evidence returned directly from retrieval."""

    documents: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DerivedEvidence:
    """Evidence derived from the authoritative retrieved set."""

    retrieved: RetrievedEvidence
    documents: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    verification_report: dict[str, Any] | None = None
    derived_from_document_ids: tuple[str, ...] = field(default_factory=tuple)
