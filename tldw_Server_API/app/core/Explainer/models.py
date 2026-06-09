"""Core models for persisted Explainer workspace state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExplainerMode(str, Enum):
    GOAL = "goal"
    SOURCES = "sources"


class ExplainerSessionStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    ERROR = "error"


class ExplainerOutputIntent(str, Enum):
    EXPLAIN = "explain"
    PLAN = "plan"
    BOTH = "both"


class ExplainerGrounding(str, Enum):
    SOURCE_ONLY = "source_only"
    SOURCE_LED = "source_led"
    OPEN = "open"


class ExplainerDepthPreset(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ExplainerNodeKind(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    EXPLANATION = "explanation"
    STEP = "step"
    SUMMARY = "summary"


class ExplainerNodeStatus(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    GENERATING = "generating"
    ERROR = "error"
    COMPLETE = "complete"


class ExplainerEvidenceState(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNCITED = "uncited"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ExplainerSelectedSource:
    source_id: str
    source_type: str
    title: str
    added_at: str
    snapshot_version: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExplainerCitation:
    id: str
    source_id: str
    source_type: str
    title: str
    excerpt: str
    location_label: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    url: str | None = None
    snapshot_hash: str | None = None


@dataclass
class ExplainerNode:
    id: str
    session_id: str
    parent_id: str | None
    ordinal: int
    title: str
    body: str | None
    kind: str
    intent: str
    status: str
    evidence_state: str
    outside_knowledge_used: bool
    citations: list[ExplainerCitation] = field(default_factory=list)
    question_options: list[dict[str, Any]] | None = None
    selected_option_id: str | None = None
    selected_custom_answer: str | None = None
    generation_metadata: dict[str, Any] | None = None
    child_node_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str | None = None


@dataclass
class ExplainerSession:
    id: str
    owner_user_id: str
    title: str
    mode: str
    status: str
    output_intent: str
    grounding: str
    depth_preset: str
    selected_sources: list[ExplainerSelectedSource]
    root_node_ids: list[str]
    nodes: dict[str, ExplainerNode]
    created_at: str
    updated_at: str
    archived_at: str | None = None


@dataclass(frozen=True)
class ExplainerSessionSummary:
    id: str
    owner_user_id: str
    title: str
    mode: str
    status: str
    output_intent: str
    grounding: str
    depth_preset: str
    node_count: int
    selected_source_count: int
    created_at: str
    updated_at: str
    archived_at: str | None = None
