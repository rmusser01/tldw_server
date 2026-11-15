"""Schemas and contract helpers for Media navigation endpoints."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator

MEDIA_NAVIGATION_FORMAT_VALUES = ("auto", "plain", "markdown", "html")
MediaNavigationFormat = Literal["auto", "plain", "markdown", "html"]

MEDIA_NAVIGATION_TARGET_TYPE_VALUES = ("page", "char_range", "time_range", "href")
MediaNavigationTargetType = Literal["page", "char_range", "time_range", "href"]


def coerce_media_navigation_format(
    value: Any,
    *,
    default: MediaNavigationFormat = "auto",
) -> MediaNavigationFormat:
    """Normalize user input to the canonical navigation content-format enum."""

    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized not in MEDIA_NAVIGATION_FORMAT_VALUES:
        raise ValueError(
            f"Unsupported media navigation format '{value}'. " f"Expected one of {MEDIA_NAVIGATION_FORMAT_VALUES}."
        )  # noqa: TRY003
    return cast(MediaNavigationFormat, normalized)


def coerce_media_navigation_target_type(value: Any) -> MediaNavigationTargetType:
    """Normalize input to the canonical target-type enum."""

    normalized = str(value or "").strip().lower()
    if normalized not in MEDIA_NAVIGATION_TARGET_TYPE_VALUES:
        raise ValueError(
            f"Unsupported media navigation target_type '{value}'. "
            f"Expected one of {MEDIA_NAVIGATION_TARGET_TYPE_VALUES}."
        )  # noqa: TRY003
    return cast(MediaNavigationTargetType, normalized)


class MediaNavigationQueryParams(BaseModel):
    include_generated_fallback: bool = Field(
        False,
        description="Allow generated section fallback when native structure sources are unavailable.",
    )
    max_depth: int = Field(
        4,
        ge=1,
        le=8,
        description="Maximum hierarchy depth to return.",
    )
    max_nodes: int = Field(
        500,
        ge=1,
        le=2000,
        description="Maximum number of nodes to return before truncation.",
    )
    parent_id: str | None = Field(
        None,
        description="Optional parent node id for lazy child loading.",
    )


class MediaNavigationContentQueryParams(BaseModel):
    navigation_version: str | None = Field(None, min_length=1, max_length=200)
    format: MediaNavigationFormat = Field(
        "auto",
        description="Requested content format (canonical values only).",
    )
    include_alternates: bool = Field(
        False,
        description="When true, response MAY include alternate representations.",
    )

    @field_validator("format", mode="before")
    @classmethod
    def _coerce_format(cls, value: Any) -> MediaNavigationFormat:
        return coerce_media_navigation_format(value)


class MediaNavigationNode(BaseModel):
    id: str = Field(..., min_length=1, description="Node identifier.")
    parent_id: str | None = Field(None, description="Parent node id for hierarchy.")
    level: int = Field(..., ge=0, description="Hierarchy level.")
    title: str = Field(..., min_length=1, description="Display title.")
    order: int = Field(..., ge=0, description="Sibling display order.")
    path_label: str | None = Field(
        None,
        description="Optional numeric path label (for example: 12.5).",
    )
    target_type: MediaNavigationTargetType = Field(
        ...,
        description="Target coordinate type.",
    )
    target_start: float | None = Field(
        None,
        description="Type-specific target start coordinate.",
    )
    target_end: float | None = Field(
        None,
        description="Type-specific target end coordinate.",
    )
    target_href: str | None = Field(
        None,
        description="Internal href target (required for href target_type).",
    )
    source: str = Field(..., min_length=1, description="Provenance source identifier.")
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Extraction confidence when available.",
    )

    @field_validator("target_type", mode="before")
    @classmethod
    def _coerce_target_type(cls, value: Any) -> MediaNavigationTargetType:
        return coerce_media_navigation_target_type(value)

    @model_validator(mode="after")
    def _validate_target_shape(self) -> MediaNavigationNode:
        if self.target_type == "char_range":
            if self.target_start is None or self.target_end is None:
                raise ValueError("char_range nodes require target_start and target_end")  # noqa: TRY003
            if not float(self.target_start).is_integer() or not float(self.target_end).is_integer():
                raise ValueError("char_range coordinates must be integers")  # noqa: TRY003
            if int(self.target_start) < 0 or int(self.target_end) <= int(self.target_start):
                raise ValueError("char_range requires 0 <= start < end")  # noqa: TRY003
            if self.target_href is not None:
                raise ValueError("char_range nodes must not include target_href")  # noqa: TRY003
            return self

        if self.target_type == "page":
            if self.target_start is None:
                raise ValueError("page nodes require target_start")  # noqa: TRY003
            if not float(self.target_start).is_integer() or int(self.target_start) < 1:
                raise ValueError("page target_start must be a 1-indexed integer >= 1")  # noqa: TRY003
            if self.target_end is not None:
                raise ValueError("page nodes must not include target_end")  # noqa: TRY003
            if self.target_href is not None:
                raise ValueError("page nodes must not include target_href")  # noqa: TRY003
            return self

        if self.target_type == "time_range":
            if self.target_start is None:
                raise ValueError("time_range nodes require target_start")  # noqa: TRY003
            if self.target_start < 0:
                raise ValueError("time_range target_start must be >= 0")  # noqa: TRY003
            if self.target_end is not None and self.target_end <= self.target_start:
                raise ValueError("time_range requires target_end > target_start when provided")  # noqa: TRY003
            if self.target_href is not None:
                raise ValueError("time_range nodes must not include target_href")  # noqa: TRY003
            return self

        if not self.target_href:
            raise ValueError("href nodes require target_href")  # noqa: TRY003
        if not str(self.target_href).startswith("#"):
            raise ValueError("href nodes only allow internal anchors")  # noqa: TRY003
        if self.target_start is not None or self.target_end is not None:
            raise ValueError("href nodes must not include target_start/target_end")  # noqa: TRY003
        return self


class MediaNavigationStats(BaseModel):
    returned_node_count: int = Field(..., ge=0)
    node_count: int = Field(..., ge=0)
    max_depth: int = Field(..., ge=0)
    truncated: bool = False


class MediaNavigationResponse(BaseModel):
    media_id: int = Field(..., ge=1)
    available: bool = True
    navigation_version: str = Field(..., min_length=1)
    source_order_used: list[str] = Field(default_factory=list)
    nodes: list[MediaNavigationNode] = Field(default_factory=list)
    stats: MediaNavigationStats


class MediaNavigationTarget(BaseModel):
    target_type: MediaNavigationTargetType
    target_start: float | None = None
    target_end: float | None = None
    target_href: str | None = None

    @field_validator("target_type", mode="before")
    @classmethod
    def _coerce_target_type(cls, value: Any) -> MediaNavigationTargetType:
        return coerce_media_navigation_target_type(value)


class MediaNavigationContentResponse(BaseModel):
    media_id: int = Field(..., ge=1)
    node_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content_format: MediaNavigationFormat
    available_formats: list[MediaNavigationFormat] = Field(default_factory=list)
    content: str
    alternate_content: dict[MediaNavigationFormat, str] | None = None
    target: MediaNavigationTarget
    navigation_version: str | None = None
    resolved_start_char: int | None = Field(None, ge=0)
    resolved_end_char: int | None = Field(None, ge=0)
    page_spans: list[dict[str, Any]] = Field(default_factory=list)
    coverage_status: Literal["exact", "unavailable"] = "unavailable"

    @field_validator("content_format", mode="before")
    @classmethod
    def _coerce_content_format(cls, value: Any) -> MediaNavigationFormat:
        return coerce_media_navigation_format(value)

    @field_validator("available_formats", mode="before")
    @classmethod
    def _coerce_available_formats(
        cls,
        value: Any,
    ) -> list[MediaNavigationFormat]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("available_formats must be a list")  # noqa: TRY003
        normalized: list[MediaNavigationFormat] = []
        for item in value:
            parsed = coerce_media_navigation_format(item)
            if parsed not in normalized:
                normalized.append(parsed)
        return normalized
