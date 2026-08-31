# app/api/v1/schemas/notes_graph.py
#
# Schemas for Notes Graph API (MVP)
# Aligns with Docs/Design/Graphing-Notes-PRD.md

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from tldw_Server_API.app.core.Notes_Graph.semantic_settings import (
    DEFAULT_SEMANTIC_INDEX_SETTINGS,
)


class EdgeType(str, Enum):
    manual = "manual"
    wikilink = "wikilink"
    backlink = "backlink"
    tag_membership = "tag_membership"
    source_membership = "source_membership"
    semantic = "semantic"


LEGACY_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {
        EdgeType.manual,
        EdgeType.wikilink,
        EdgeType.backlink,
        EdgeType.tag_membership,
        EdgeType.source_membership,
    }
)
_SEMANTIC_MAX_TOP_K = DEFAULT_SEMANTIC_INDEX_SETTINGS.max_query_neighbors
_SEMANTIC_MAX_EVIDENCE_PAIRS = 3
_SEMANTIC_MAX_EXCERPT_CODE_POINTS = 480
_SEMANTIC_MAX_EDGE_EVIDENCE_CODE_POINTS = 2_880
_SEMANTIC_MAX_RESPONSE_EVIDENCE_BYTES = 256 * 1024


def _canonical_edge_types(values: list[EdgeType]) -> tuple[EdgeType, ...]:
    return tuple(sorted(set(values), key=lambda edge_type: edge_type.value))


class GraphFormat(str, Enum):
    default = "default"
    cytoscape = "cytoscape"


class TimeRange(BaseModel):
    start: datetime | None = Field(None, description="Start timestamp (inclusive) in ISO-8601")
    end: datetime | None = Field(None, description="End timestamp (inclusive) in ISO-8601")


class GraphNode(BaseModel):
    id: str = Field(..., description="Opaque node identifier (e.g., note UUID or typed id)")
    type: Literal["note", "tag", "source"] = Field(..., description="Node entity type")
    label: str = Field(..., description="Human-readable label for rendering")
    created_at: datetime | None = Field(None, description="Creation timestamp (where applicable)")
    deleted: bool | None = Field(
        None, description="Soft-deleted status (applies to notes; clients should dim/mark)"
    )
    degree: int | None = Field(None, ge=0, description="Degree in the returned subgraph")
    tag_count: int | None = Field(None, ge=0, description="Number of tags on a note (if computed)")
    primary_source_id: str | None = Field(
        None, description="Primary source id for notes (when available)"
    )

    model_config = ConfigDict(from_attributes=True)


class _StrictSemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticExcerpt(_StrictSemanticModel):
    """One current canonical Note excerpt with field-relative offsets."""

    field: Literal["title", "content"]
    start_code_point: int = Field(ge=0)
    end_code_point: int = Field(gt=0)
    text: str = Field(max_length=_SEMANTIC_MAX_EXCERPT_CODE_POINTS)

    @model_validator(mode="after")
    def _validate_offsets(self) -> SemanticExcerpt:
        if self.end_code_point <= self.start_code_point:
            raise ValueError("semantic excerpt offsets must define a non-empty range")
        if len(self.text) != self.end_code_point - self.start_code_point:
            raise ValueError("semantic excerpt text must match its code-point range")
        return self


class SemanticExcerptPair(_StrictSemanticModel):
    source: SemanticExcerpt
    target: SemanticExcerpt


class SemanticEdgeEvidence(_StrictSemanticModel):
    """Stable provenance and bounded current excerpts for one semantic edge."""

    similarity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    qualitative_band: Literal["low", "moderate", "high", "very_high"]
    source_note_id: str = Field(min_length=1)
    target_note_id: str = Field(min_length=1)
    source_content_version: int = Field(ge=1)
    target_content_version: int = Field(ge=1)
    generation_id: str = Field(min_length=1)
    semantic_index_revision: int = Field(ge=0)
    configuration_revision: int = Field(ge=0)
    normalization_version: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)
    provider_label: str = Field(min_length=1)
    model_label: str = Field(min_length=1)
    model_revision: str | None = None
    excerpt_pairs: list[SemanticExcerptPair] = Field(
        default_factory=list,
        max_length=_SEMANTIC_MAX_EVIDENCE_PAIRS,
    )


SemanticTruncationReason = Literal[
    "semantic_candidates",
    "semantic_nodes",
    "semantic_edges",
    "semantic_evidence_bytes",
]


class SemanticGraphStatus(_StrictSemanticModel):
    """Fresh semantic state projected beside a stable graph response."""

    available: bool
    state: Literal[
        "off",
        "preparing",
        "ready",
        "updating",
        "needs_attention",
        "unavailable",
        "focus_required",
    ]
    detail_reason: str | None = None
    generation_id: str | None = None
    semantic_index_revision: int | None = Field(default=None, ge=0)
    configuration_revision: int | None = Field(default=None, ge=0)
    active_notes: int = Field(default=0, ge=0)
    indexed_notes: int = Field(default=0, ge=0)
    dirty_notes: int = Field(default=0, ge=0)
    excluded_notes: int = Field(default=0, ge=0)
    failed_notes: int = Field(default=0, ge=0)
    effective_top_k: int | None = Field(default=None, ge=1, le=_SEMANTIC_MAX_TOP_K)
    effective_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    max_top_k: int = Field(ge=1, le=_SEMANTIC_MAX_TOP_K)
    max_admission_nodes: int = Field(ge=0, le=_SEMANTIC_MAX_TOP_K)
    max_admission_edges: int = Field(ge=0, le=_SEMANTIC_MAX_TOP_K)
    max_evidence_pairs: int = Field(ge=0, le=_SEMANTIC_MAX_EVIDENCE_PAIRS)
    max_excerpt_code_points: int = Field(
        ge=0,
        le=_SEMANTIC_MAX_EXCERPT_CODE_POINTS,
    )
    max_edge_evidence_code_points: int = Field(
        ge=0,
        le=_SEMANTIC_MAX_EDGE_EVIDENCE_CODE_POINTS,
    )
    max_response_evidence_bytes: int = Field(
        ge=0,
        le=_SEMANTIC_MAX_RESPONSE_EVIDENCE_BYTES,
    )
    truncated_by: list[SemanticTruncationReason] = Field(default_factory=list)


class GraphEdge(BaseModel):
    id: str = Field(..., description="Opaque edge id")
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    type: EdgeType = Field(..., description="Edge type")
    directed: bool = Field(..., description="Whether the edge is directed")
    weight: float | None = Field(1.0, ge=0.0, description="Optional weight; defaults to 1.0")
    label: str | None = Field(None, description="Optional label for the edge")
    evidence: SemanticEdgeEvidence | None = None
    evidence_omitted: Literal["response_byte_cap"] | None = None

    @model_validator(mode="after")
    def _validate_semantic_evidence(self) -> GraphEdge:
        if self.type == EdgeType.semantic:
            if self.directed:
                raise ValueError("semantic edges must be undirected")
            if self.evidence is None and self.evidence_omitted is None:
                raise ValueError("semantic edges require typed evidence")
            if self.evidence is not None and self.evidence_omitted is not None:
                raise ValueError("semantic evidence conflicts with its omission marker")
            if self.evidence is not None and self.evidence.source_note_id != self.source:
                raise ValueError("semantic evidence source does not match the edge")
            if self.evidence is not None and self.evidence.target_note_id != self.target:
                raise ValueError("semantic evidence target does not match the edge")
        elif self.evidence is not None or self.evidence_omitted is not None:
            raise ValueError("semantic evidence is only valid on semantic edges")
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_empty_evidence(self, handler):
        data = handler(self)
        if self.evidence is None:
            data.pop("evidence", None)
        if self.evidence_omitted is None:
            data.pop("evidence_omitted", None)
        return data

    model_config = ConfigDict(from_attributes=True)


class GraphLimits(BaseModel):
    max_nodes: int = Field(..., ge=1)
    max_edges: int = Field(..., ge=0)
    max_degree: int = Field(..., ge=1)


class NoteGraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    truncated_by: list[str] = Field(default_factory=list)
    has_more: bool = False
    cursor: str | None = None
    limits: GraphLimits
    radius_cap_applied: bool = False
    active_note_count: int = Field(..., ge=0)
    all_notes_note_cap: int = Field(..., ge=1)
    all_notes_eligible: bool
    suggestions_authorized: bool = False
    semantic_status: SemanticGraphStatus | None = None

    @model_serializer(mode="wrap")
    def _serialize_without_empty_semantic_status(self, handler):
        data = handler(self)
        if self.semantic_status is None:
            data.pop("semantic_status", None)
        return data

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nodes": [
                    {
                        "id": "note:123",
                        "type": "note",
                        "label": "My Note",
                        "created_at": "2025-01-01T12:00:00Z",
                        "degree": 2,
                        "tag_count": 3,
                        "primary_source_id": "source:yt:abcd",
                    },
                    {"id": "tag:ml", "type": "tag", "label": "ml"},
                    {
                        "id": "source:yt:abcd",
                        "type": "source",
                        "label": "YouTube: abcd",
                    },
                ],
                "edges": [
                    {
                        "id": "e:1",
                        "source": "note:123",
                        "target": "note:456",
                        "type": "manual",
                        "directed": False,
                        "weight": 1.0,
                    },
                    {
                        "id": "e:2",
                        "source": "note:123",
                        "target": "tag:ml",
                        "type": "tag_membership",
                        "directed": False,
                    },
                ],
                "truncated": False,
                "truncated_by": [],
                "has_more": False,
                "cursor": None,
                "limits": {"max_nodes": 300, "max_edges": 1200, "max_degree": 40},
                "active_note_count": 2,
                "all_notes_note_cap": 100,
                "all_notes_eligible": True,
                "suggestions_authorized": False,
            }
        }
    )


class SemanticConversionContext(_StrictSemanticModel):
    """Immutable semantic authority supplied when converting a derived edge."""

    generation_id: str = Field(min_length=1, max_length=256)


class NoteLinkCreate(BaseModel):
    to_note_id: str = Field(..., min_length=1, description="Target note id to link to")
    directed: bool = Field(False, description="Whether the link is directed; defaults to false")
    weight: float | None = Field(1.0, ge=0.0, description="Optional weight of the link")
    label: str | None = Field(default=None, max_length=256)
    properties: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata to attach to the link"
    )
    dataset_id: str | None = Field(default=None, min_length=1, max_length=256)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    semantic_conversion: SemanticConversionContext | None = None

    @field_validator("dataset_id", "idempotency_key", mode="before")
    @classmethod
    def _normalize_authority_fields(cls, value: object) -> object:
        return _normalize_optional_nonblank(value)

    @model_validator(mode="after")
    def _normalize_legacy_metadata(self) -> NoteLinkCreate:
        metadata = dict(self.metadata or {})
        legacy_label = metadata.pop("label", None)
        if legacy_label is not None and not isinstance(legacy_label, str):
            raise ValueError("metadata.label must be a string or null")
        if self.label is not None and legacy_label is not None and self.label != legacy_label:
            raise ValueError("label conflicts with metadata.label")
        overlap = set(metadata).intersection(self.properties or {})
        if any(metadata[key] != (self.properties or {})[key] for key in overlap):
            raise ValueError("properties conflict with legacy metadata")
        self.label = self.label if self.label is not None else legacy_label
        self.properties = {**metadata, **dict(self.properties or {})}
        self.metadata = None
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "to_note_id": "note:456",
                "directed": False,
                "weight": 1.0,
                "metadata": {"label": "related"},
            }
        }
    )


class NoteLinkUpdate(BaseModel):
    """Mutable presentation update for one explicit link."""

    expected_version: int | None = Field(default=None, ge=1)
    weight: float | None = Field(default=None, ge=0.0)
    label: str | None = Field(default=None, max_length=256)
    properties: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Legacy full properties map. Supplying metadata replaces the existing properties map; "
            "omitted properties are removed."
        ),
    )
    dataset_id: str | None = Field(default=None, min_length=1, max_length=256)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("dataset_id", "idempotency_key", mode="before")
    @classmethod
    def _normalize_authority_fields(cls, value: object) -> object:
        return _normalize_optional_nonblank(value)

    @model_validator(mode="after")
    def _normalize_update(self) -> NoteLinkUpdate:
        if not self.model_fields_set.intersection({"weight", "label", "properties", "metadata"}):
            raise ValueError("at least one mutable link field is required")
        metadata = dict(self.metadata or {})
        if "label" in metadata:
            legacy_label = metadata.pop("label")
            if legacy_label is not None and not isinstance(legacy_label, str):
                raise ValueError("metadata.label must be a string or null")
            if "label" in self.model_fields_set and self.label != legacy_label:
                raise ValueError("label conflicts with metadata.label")
            self.label = legacy_label
            self.__pydantic_fields_set__.add("label")
        if metadata or "metadata" in self.model_fields_set:
            overlap = set(metadata).intersection(self.properties or {})
            if any(metadata[key] != (self.properties or {})[key] for key in overlap):
                raise ValueError("properties conflict with legacy metadata")
            self.properties = {**metadata, **dict(self.properties or {})}
            self.__pydantic_fields_set__.add("properties")
        self.metadata = None
        return self


class NoteLinkRestore(BaseModel):
    """Explicit restore request for one tombstoned link."""

    expected_version: int | None = Field(default=None, ge=1)
    dataset_id: str | None = Field(default=None, min_length=1, max_length=256)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("dataset_id", "idempotency_key", mode="before")
    @classmethod
    def _normalize_authority_fields(cls, value: object) -> object:
        return _normalize_optional_nonblank(value)


def _normalize_optional_nonblank(value: object) -> object:
    """Trim optional string authority fields and reject blank values."""

    if value is None or not isinstance(value, str):
        return value
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be blank")
    return normalized


class NoteGraphRequest(BaseModel):
    dataset_id: str | None = Field(
        default=None,
        description="Canonical default-personal Notes dataset; omitted for automatic resolution",
    )
    center_note_id: str | None = Field(
        default=None, description="Focal note id for ego expansion"
    )
    radius: int = Field(1, ge=1, le=2, description="Expansion radius; 1 by default, 2 allowed with caps")
    edge_types: list[EdgeType] | None = Field(
        default=None,
        description="Edge types to include; accepts repeated values or CSV",
    )
    tag: str | None = Field(default=None, description="Filter to notes with a specific tag id")
    source: str | None = Field(default=None, description="Filter to notes with a specific source id")
    time_range: TimeRange | None = None
    time_range_field: Literal["created_at", "updated_at"] = Field(
        "updated_at", description="Which timestamp field time_range applies to"
    )
    max_nodes: int | None = Field(default=None, ge=1)
    max_edges: int | None = Field(default=None, ge=0)
    max_degree: int | None = Field(default=None, ge=1)
    semantic_top_k: int | None = Field(default=None, ge=1, le=_SEMANTIC_MAX_TOP_K)
    semantic_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    format: GraphFormat = GraphFormat.default
    cursor: str | None = None
    allow_heavy: bool = False

    @field_validator("dataset_id", mode="before")
    @classmethod
    def _normalize_dataset_id(cls, value: object) -> object:
        return _normalize_optional_nonblank(value)

    @field_validator("edge_types", mode="before")
    @classmethod
    def _split_csv_edge_types(cls, v):
        if v is None:
            return v
        # Accept CSV string or repeated values that arrive as list[str]
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return list(_canonical_edge_types([EdgeType(p) for p in parts]))
        if isinstance(v, list):
            out: list[EdgeType] = []
            for item in v:
                if isinstance(item, EdgeType):
                    out.append(item)
                elif isinstance(item, str):
                    out.extend(
                        EdgeType(part)
                        for part in (value.strip() for value in item.split(","))
                        if part
                    )
            return list(_canonical_edge_types(out))
        return v

    @property
    def resolved_edge_types(self) -> tuple[EdgeType, ...]:
        if not self.edge_types:
            return _canonical_edge_types(list(LEGACY_EDGE_TYPES))
        return _canonical_edge_types(self.edge_types)

    @property
    def semantic_requested(self) -> bool:
        return EdgeType.semantic in self.resolved_edge_types and bool(self.edge_types)

    @model_validator(mode="after")
    def _validate_semantic_controls(self) -> NoteGraphRequest:
        if (
            self.semantic_top_k is not None or self.semantic_threshold is not None
        ) and not self.semantic_requested:
            raise ValueError("semantic controls require semantic in edge_types")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Ego graph for a note (default)",
                    "value": {
                        "center_note_id": "note:123",
                        "radius": 1,
                        "edge_types": [
                            "manual",
                            "wikilink",
                            "backlink",
                            "tag_membership",
                            "source_membership",
                        ],
                        "format": "default",
                        "max_nodes": 300,
                        "max_edges": 1200,
                        "max_degree": 40,
                    },
                }
            ]
        }
    )
