"""Shared types for deterministic study suggestion topic resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EvidenceClass = Literal["grounded", "weakly_grounded", "derived"]
RankReason = Literal["weakness", "adjacent", "exploratory", "candidate"]


@dataclass(slots=True)
class SuggestionContext:
    service: str
    activity_type: str
    anchor_type: str
    anchor_id: int
    workspace_id: str | None
    summary_metrics: dict[str, Any]
    performance_signals: dict[str, Any]
    source_bundle: list[dict[str, Any]]


@dataclass(slots=True)
class NormalizedTopicLabel:
    canonical_label: str
    raw_labels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResolvedTopicLabel:
    namespace: str
    canonical_slug: str
    canonical_label: str
    semantic_label: str
    topic_key: str
    normalization_version: str
    raw_labels: list[str] = field(default_factory=list)
    evidence_reasons: list[str] = field(default_factory=list)
    source_count: int = 0


@dataclass(slots=True)
class TopicCandidate:
    canonical_label: str
    semantic_label: str
    raw_labels: list[str]
    evidence_class: EvidenceClass
    topic_key: str | None = None
    normalization_version: str | None = None
    source_count: int = 0
    evidence_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RankedTopic:
    canonical_label: str
    semantic_label: str
    raw_labels: list[str]
    evidence_class: EvidenceClass
    rank_reason: RankReason
    adjacency_is_source_aware: bool
    topic_key: str | None = None
    normalization_version: str | None = None
    source_count: int = 0
    evidence_reasons: list[str] = field(default_factory=list)
