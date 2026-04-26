"""Pydantic schemas for manuscript management endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import json

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class ManuscriptProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Project title")
    subtitle: str | None = Field(None, max_length=500, description="Project subtitle")
    author: str | None = Field(None, max_length=255, description="Author name")
    genre: str | None = Field(None, max_length=100, description="Genre")
    status: Literal["draft", "outlining", "writing", "revising", "complete", "archived"] = Field(
        "draft", description="Project status"
    )
    synopsis: str | None = Field(None, description="Project synopsis")
    target_word_count: int | None = Field(None, ge=0, description="Target word count")
    settings: dict[str, Any] | None = Field(None, description="Project settings JSON")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptProjectUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Project title")
    subtitle: str | None = Field(None, max_length=500, description="Project subtitle")
    author: str | None = Field(None, max_length=255, description="Author name")
    genre: str | None = Field(None, max_length=100, description="Genre")
    status: Literal["draft", "outlining", "writing", "revising", "complete", "archived"] | None = Field(
        None, description="Project status"
    )
    synopsis: str | None = Field(None, description="Project synopsis")
    target_word_count: int | None = Field(None, ge=0, description="Target word count")
    settings: dict[str, Any] | None = Field(None, description="Project settings JSON")


class ManuscriptProjectSettings(BaseModel):
    """Structured manuscript project settings with forward-compatible keys."""

    model_config = ConfigDict(extra="allow")


class ManuscriptProjectResponse(BaseModel):
    id: str
    title: str
    subtitle: str | None = None
    author: str | None = None
    genre: str | None = None
    status: str
    synopsis: str | None = None
    target_word_count: int | None = None
    settings: ManuscriptProjectSettings = Field(default_factory=ManuscriptProjectSettings)
    word_count: int = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


class ManuscriptProjectListResponse(BaseModel):
    projects: list[ManuscriptProjectResponse]
    total: int


# ---------------------------------------------------------------------------
# Part
# ---------------------------------------------------------------------------


class ManuscriptPartCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Part title")
    sort_order: float = Field(0, description="Sort order within the project")
    synopsis: str | None = Field(None, description="Part synopsis")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptPartUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Part title")
    sort_order: float | None = Field(None, description="Sort order within the project")
    synopsis: str | None = Field(None, description="Part synopsis")


class ManuscriptPartResponse(BaseModel):
    id: str
    project_id: str
    title: str
    sort_order: float
    synopsis: str | None = None
    word_count: int = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Chapter
# ---------------------------------------------------------------------------


class ManuscriptChapterCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Chapter title")
    part_id: str | None = Field(None, description="Optional parent part ID")
    sort_order: float = Field(0, description="Sort order")
    synopsis: str | None = Field(None, description="Chapter synopsis")
    status: Literal["outline", "draft", "revising", "final"] = Field("draft", description="Chapter status")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptChapterUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Chapter title")
    part_id: str | None = Field(None, description="Parent part ID")
    sort_order: float | None = Field(None, description="Sort order")
    synopsis: str | None = Field(None, description="Chapter synopsis")
    status: Literal["outline", "draft", "revising", "final"] | None = Field(None, description="Chapter status")


class ManuscriptChapterResponse(BaseModel):
    id: str
    project_id: str
    part_id: str | None = None
    title: str
    sort_order: float
    synopsis: str | None = None
    pov_character_id: str | None = None
    word_count: int = 0
    status: str = "draft"
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class ManuscriptSceneCreate(BaseModel):
    title: str = Field("Untitled Scene", min_length=1, max_length=500, description="Scene title")
    content: dict[str, Any] | None = Field(None, description="TipTap JSON content")
    content_plain: str = Field("", description="Plain-text content for word counting and search")
    synopsis: str | None = Field(None, description="Scene synopsis")
    sort_order: float = Field(0, description="Sort order within the chapter")
    status: Literal["outline", "draft", "revising", "final"] = Field("draft", description="Scene status")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptSceneUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Scene title")
    content: dict[str, Any] | None = Field(None, description="TipTap JSON content")
    content_plain: str | None = Field(None, description="Plain-text content")
    synopsis: str | None = Field(None, description="Scene synopsis")
    sort_order: float | None = Field(None, description="Sort order")
    status: Literal["outline", "draft", "revising", "final"] | None = Field(None, description="Scene status")


class ManuscriptSceneResponse(BaseModel):
    id: str
    chapter_id: str
    project_id: str
    title: str
    sort_order: float
    content_json: str | None = None
    content_plain: str | None = None
    synopsis: str | None = None
    word_count: int = 0
    pov_character_id: str | None = None
    status: str = "draft"
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content(self) -> dict[str, Any] | None:
        """Parsed TipTap JSON content (mirrors the ``content`` field on create/update)."""
        if self.content_json is None:
            return None
        try:
            return json.loads(self.content_json)
        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Structure tree
# ---------------------------------------------------------------------------


class SceneSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int = 0
    status: str = "draft"
    version: int = 1


class ChapterSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    part_id: str | None = None
    word_count: int = 0
    status: str = "draft"
    version: int = 1
    scenes: list[SceneSummary] = Field(default_factory=list)


class PartSummary(BaseModel):
    id: str
    title: str
    sort_order: float
    word_count: int = 0
    version: int = 1
    chapters: list[ChapterSummary] = Field(default_factory=list)


class ManuscriptStructureResponse(BaseModel):
    project_id: str
    parts: list[PartSummary] = Field(default_factory=list)
    unassigned_chapters: list[ChapterSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reorder
# ---------------------------------------------------------------------------


class ReorderItem(BaseModel):
    id: str = Field(..., description="Entity ID")
    sort_order: float = Field(..., description="New sort order")
    version: int | None = Field(
        None,
        description="Expected version for optimistic locking; optional for backward-compatible reorders",
    )
    new_parent_id: str | None = Field(None, description="Optional new parent ID (for reparenting chapters)")

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_null_version(cls, data: Any) -> Any:
        """Permit omitted versions for legacy clients, but reject explicit null values."""
        if isinstance(data, dict) and "version" in data and data["version"] is None:
            raise ValueError("version may be omitted, but explicit null is invalid")
        return data


class ReorderRequest(BaseModel):
    entity_type: Literal["parts", "chapters", "scenes"] = Field(
        ..., description="Type of entities being reordered"
    )
    items: list[ReorderItem] = Field(..., min_length=1, description="Items with new sort orders")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class ManuscriptSearchResult(BaseModel):
    id: str
    title: str
    chapter_id: str
    word_count: int = 0
    status: str = "draft"
    snippet: str | None = None


class ManuscriptSearchResponse(BaseModel):
    query: str
    results: list[ManuscriptSearchResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------

CHARACTER_ROLES = Literal["protagonist", "antagonist", "supporting", "minor", "mentioned"]
_CHARACTER_ROLES = CHARACTER_ROLES


class ManuscriptCharacterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Character name")
    role: _CHARACTER_ROLES = Field("supporting", description="Character role")
    cast_group: str | None = Field(None, description="Grouping label (e.g. 'heroes', 'villains')")
    full_name: str | None = Field(None, description="Full / legal name")
    age: str | None = Field(None, description="Age or age range")
    gender: str | None = Field(None, description="Gender")
    appearance: str | None = Field(None, description="Physical description")
    personality: str | None = Field(None, description="Personality traits")
    backstory: str | None = Field(None, description="Character backstory")
    motivation: str | None = Field(None, description="Primary motivation")
    arc_summary: str | None = Field(None, description="Summary of character arc")
    notes: str | None = Field(None, description="Free-form notes")
    custom_fields: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value data")
    sort_order: float = Field(0, description="Sort order")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptCharacterUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Character name")
    role: _CHARACTER_ROLES | None = Field(None, description="Character role")
    cast_group: str | None = Field(None, description="Grouping label")
    full_name: str | None = Field(None, description="Full / legal name")
    age: str | None = Field(None, description="Age or age range")
    gender: str | None = Field(None, description="Gender")
    appearance: str | None = Field(None, description="Physical description")
    personality: str | None = Field(None, description="Personality traits")
    backstory: str | None = Field(None, description="Character backstory")
    motivation: str | None = Field(None, description="Primary motivation")
    arc_summary: str | None = Field(None, description="Summary of character arc")
    notes: str | None = Field(None, description="Free-form notes")
    custom_fields: dict[str, Any] | None = Field(None, description="Arbitrary key-value data")
    sort_order: float | None = Field(None, description="Sort order")


class ManuscriptCharacterResponse(BaseModel):
    id: str
    project_id: str
    name: str
    role: str
    cast_group: str | None = None
    full_name: str | None = None
    age: str | None = None
    gender: str | None = None
    appearance: str | None = None
    personality: str | None = None
    backstory: str | None = None
    motivation: str | None = None
    arc_summary: str | None = None
    notes: str | None = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    sort_order: float = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Character Relationship
# ---------------------------------------------------------------------------


class ManuscriptRelationshipCreate(BaseModel):
    from_character_id: str = Field(..., description="Source character ID")
    to_character_id: str = Field(..., description="Target character ID")
    relationship_type: str = Field(..., min_length=1, description="Relationship type (e.g. 'sibling')")
    description: str | None = Field(None, description="Details about the relationship")
    bidirectional: bool = Field(True, description="Whether the relationship is mutual")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptRelationshipResponse(BaseModel):
    id: str
    project_id: str
    from_character_id: str
    to_character_id: str
    relationship_type: str
    description: str | None = None
    bidirectional: bool = True
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# World Info
# ---------------------------------------------------------------------------

WORLD_INFO_KINDS = Literal["location", "item", "faction", "concept", "event", "custom"]
_WORLD_INFO_KINDS = WORLD_INFO_KINDS


class ManuscriptWorldInfoCreate(BaseModel):
    kind: _WORLD_INFO_KINDS = Field(..., description="Category of world-info entry")
    name: str = Field(..., min_length=1, max_length=255, description="Entry name")
    description: str | None = Field(None, description="Entry description")
    parent_id: str | None = Field(None, description="Parent world-info entry ID")
    properties: dict[str, Any] = Field(default_factory=dict, description="Arbitrary properties")
    tags: list[str] = Field(default_factory=list, description="Tags")
    sort_order: float = Field(0, description="Sort order")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptWorldInfoUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Entry name")
    description: str | None = Field(None, description="Entry description")
    parent_id: str | None = Field(None, description="Parent world-info entry ID")
    properties: dict[str, Any] | None = Field(None, description="Arbitrary properties")
    tags: list[str] | None = Field(None, description="Tags")
    sort_order: float | None = Field(None, description="Sort order")


class ManuscriptWorldInfoResponse(BaseModel):
    id: str
    project_id: str
    kind: str
    name: str
    description: str | None = None
    parent_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    sort_order: float = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Plot Line
# ---------------------------------------------------------------------------

_PLOT_LINE_STATUSES = Literal["active", "resolved", "abandoned", "dormant"]


class ManuscriptPlotLineCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Plot line title")
    description: str | None = Field(None, description="Plot line description")
    status: _PLOT_LINE_STATUSES = Field("active", description="Plot line status")
    color: str | None = Field(None, description="Display colour hex code")
    sort_order: float = Field(0, description="Sort order")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptPlotLineUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Plot line title")
    description: str | None = Field(None, description="Plot line description")
    status: _PLOT_LINE_STATUSES | None = Field(None, description="Plot line status")
    color: str | None = Field(None, description="Display colour hex code")
    sort_order: float | None = Field(None, description="Sort order")


class ManuscriptPlotLineResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str | None = None
    status: str = "active"
    color: str | None = None
    sort_order: float = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Plot Event
# ---------------------------------------------------------------------------

_PLOT_EVENT_TYPES = Literal["setup", "conflict", "action", "emotional", "plot", "resolution"]


class ManuscriptPlotEventCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Event title")
    description: str | None = Field(None, description="Event description")
    scene_id: str | None = Field(None, description="Associated scene ID")
    chapter_id: str | None = Field(None, description="Associated chapter ID")
    event_type: _PLOT_EVENT_TYPES = Field("plot", description="Event type")
    sort_order: float = Field(0, description="Sort order")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptPlotEventUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Event title")
    description: str | None = Field(None, description="Event description")
    scene_id: str | None = Field(None, description="Associated scene ID")
    chapter_id: str | None = Field(None, description="Associated chapter ID")
    event_type: _PLOT_EVENT_TYPES | None = Field(None, description="Event type")
    sort_order: float | None = Field(None, description="Sort order")


class ManuscriptPlotEventResponse(BaseModel):
    id: str
    project_id: str
    plot_line_id: str
    title: str
    description: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    event_type: str = "plot"
    sort_order: float = 0
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Plot Hole
# ---------------------------------------------------------------------------

_PLOT_HOLE_SEVERITIES = Literal["low", "medium", "high", "critical"]
_PLOT_HOLE_STATUSES = Literal["open", "investigating", "resolved", "wontfix"]
_PLOT_HOLE_DETECTED_BY = Literal["manual", "ai"]


class ManuscriptPlotHoleCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500, description="Plot hole title")
    description: str | None = Field(None, description="Plot hole description")
    severity: _PLOT_HOLE_SEVERITIES = Field("medium", description="Severity level")
    scene_id: str | None = Field(None, description="Associated scene ID")
    chapter_id: str | None = Field(None, description="Associated chapter ID")
    plot_line_id: str | None = Field(None, description="Associated plot line ID")
    detected_by: _PLOT_HOLE_DETECTED_BY = Field("manual", description="Detection method")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptPlotHoleUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=500, description="Plot hole title")
    description: str | None = Field(None, description="Plot hole description")
    severity: _PLOT_HOLE_SEVERITIES | None = Field(None, description="Severity level")
    status: _PLOT_HOLE_STATUSES | None = Field(None, description="Plot hole status")
    resolution: str | None = Field(None, description="Resolution description")
    scene_id: str | None = Field(None, description="Associated scene ID")
    chapter_id: str | None = Field(None, description="Associated chapter ID")
    plot_line_id: str | None = Field(None, description="Associated plot line ID")
    detected_by: _PLOT_HOLE_DETECTED_BY | None = Field(None, description="Detection method")


class ManuscriptPlotHoleResponse(BaseModel):
    id: str
    project_id: str
    title: str
    description: str | None = None
    severity: str = "medium"
    status: str = "open"
    resolution: str | None = None
    scene_id: str | None = None
    chapter_id: str | None = None
    plot_line_id: str | None = None
    detected_by: str = "manual"
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class ManuscriptCitationCreate(BaseModel):
    source_type: str = Field(..., min_length=1, description="Source type (e.g. 'rag', 'web', 'manual')")
    source_id: str | None = Field(None, description="Source item ID (e.g. media item ID)")
    source_title: str | None = Field(None, description="Human-readable source title")
    excerpt: str | None = Field(None, description="Quoted excerpt from source")
    query_used: str | None = Field(None, description="RAG query that found this source")
    anchor_offset: int | None = Field(None, description="Character offset in scene content")
    id: str | None = Field(None, description="Optional client-provided UUID")


class ManuscriptCitationResponse(BaseModel):
    id: str
    project_id: str
    scene_id: str
    source_type: str
    source_id: str | None = None
    source_title: str | None = None
    excerpt: str | None = None
    query_used: str | None = None
    anchor_offset: int | None = None
    created_at: datetime
    last_modified: datetime
    deleted: bool = False
    client_id: str
    version: int


# ---------------------------------------------------------------------------
# Scene linking
# ---------------------------------------------------------------------------


class SceneCharacterLink(BaseModel):
    character_id: str = Field(..., description="Character ID to link")
    is_pov: bool = Field(False, description="Whether this character is the POV character")


class SceneCharacterLinkResponse(BaseModel):
    scene_id: str
    character_id: str
    is_pov: bool = False
    name: str
    role: str


class SceneWorldInfoLink(BaseModel):
    world_info_id: str = Field(..., description="World-info entry ID to link")


class SceneWorldInfoLinkResponse(BaseModel):
    scene_id: str
    world_info_id: str
    name: str
    kind: str


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


class ManuscriptResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research query")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of results to return")


class ManuscriptResearchResult(BaseModel):
    """A single research result item."""
    source_id: str | None = None
    title: str
    excerpt: str | None = None
    source_type: str | None = None
    relevance_score: float | None = None


class ManuscriptResearchResponse(BaseModel):
    query: str
    results: list[ManuscriptResearchResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AI Analysis
# ---------------------------------------------------------------------------


class ManuscriptAnalysisRequest(BaseModel):
    analysis_types: list[Literal["pacing", "plot_holes", "consistency"]] = Field(
        default_factory=lambda: ["pacing"],
        min_length=1,
        description="Analysis types to run: pacing, plot_holes, consistency",
    )
    provider: str | None = Field(None, description="LLM provider override")
    model: str | None = Field(None, description="Model override")

    @field_validator("analysis_types")
    @classmethod
    def validate_analysis_types(cls, v: list[str]) -> list[str]:
        allowed = {"pacing", "plot_holes", "consistency"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(
                f"Unsupported analysis types: {', '.join(sorted(invalid))}. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )
        return v


class ManuscriptAnalysisResponse(BaseModel):
    id: str
    project_id: str
    scope_type: str
    scope_id: str
    analysis_type: str
    result: dict[str, Any]
    score: float | None = None
    stale: bool = False
    provider: str | None = None
    model: str | None = None
    created_at: datetime
    last_modified: datetime
    version: int


class ManuscriptAnalysisListResponse(BaseModel):
    analyses: list[ManuscriptAnalysisResponse]
    total: int
