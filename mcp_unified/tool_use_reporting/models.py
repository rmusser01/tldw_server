"""Metadata-only event models for MCP tool-use reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from mcp_unified.tool_use_reporting.sanitization import (
    sanitize_reason_code,
    sanitize_safe_id,
)

RuntimeSurface = Literal["protocol", "gateway"]
ExecutionOrigin = Literal[
    "executed",
    "cached",
    "denied",
    "unavailable",
    "failed_before_execution",
]
ToolUseStatus = Literal[
    "success",
    "error",
    "denied",
    "approval_required",
    "unavailable",
    "invalid_params",
    "rate_limited",
]
SourceKind = Literal["local", "external", "federated", "bridge"]
ToolUseEventExportFormat = Literal["json", "jsonl"]
ToolUseReportGroupBy = Literal["profile", "tool_prompt", "model", "tool"]

MAX_EVENT_QUERY_LIMIT = 10_000
MAX_REPORT_EVENT_LIMIT = 5_000
MAX_REPORT_GROUP_LIMIT = 500
MAX_REPORT_REASON_CODE_LIMIT = 25

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for event defaults."""

    return datetime.now(timezone.utc)


def _normalize_utc(value: datetime) -> datetime:
    """Normalize an aware datetime to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _epoch_microseconds(value: datetime) -> int:
    """Return integer microseconds since Unix epoch without float rounding."""

    delta = value - _EPOCH
    return (((delta.days * 86_400) + delta.seconds) * 1_000_000) + delta.microseconds


def _safe_required_id(value: Any, *, field: str) -> str:
    """Return a safe id or the stable unknown sentinel."""

    return sanitize_safe_id(value, field=field) or "unknown"


def _safe_optional_id(value: Any, *, field: str) -> str | None:
    """Return a safe optional id."""

    return sanitize_safe_id(value, field=field)


class ToolUseEvent(BaseModel):
    """Immutable metadata-only record for one attempted MCP tool call."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at_utc: datetime = Field(
        default_factory=_utc_now,
        validation_alias=AliasChoices("created_at_utc", "created_at"),
    )
    created_at_epoch_us: int = 0
    runtime_surface: RuntimeSurface
    execution_origin: ExecutionOrigin = "executed"
    nested: bool = False
    correlation_id: str | None = None
    requested_tool_name: str
    effective_tool_name: str | None = None
    module_id: str | None = None
    category: str | None = None
    read_only: bool | None = None
    is_write: bool | None = None
    source_kind: SourceKind | None = None
    profile_id: str | None = None
    mode_id: str | None = None
    model_id: str | None = None
    tool_prompt_id: str | None = None
    tool_prompt_version: str | None = None
    prompt_variant: str | None = None
    action_family: str | None = None
    result_kind: str | None = None
    status: ToolUseStatus
    reason_code: str | None = None
    duration_ms: float | None = None
    latency_bucket: str | None = None
    truncated: bool | None = None
    path_filter_used: bool | None = None
    grant_outcome: str | None = None
    approval_outcome: str | None = None
    installation_status: str | None = None
    runtime_availability: str | None = None
    idempotency_replay: bool = False
    capture_ref: str | None = None

    @field_validator("event_id", mode="before")
    @classmethod
    def _sanitize_event_id(cls, value: Any) -> str:
        """Sanitize caller-provided ids, replacing unsafe values."""

        return sanitize_safe_id(value, field="event_id") or uuid4().hex

    @field_validator("created_at_utc", mode="after")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        """Normalize created_at aliases to UTC."""

        return _normalize_utc(value)

    @field_validator("requested_tool_name", mode="before")
    @classmethod
    def _sanitize_requested_tool_name(cls, value: Any) -> str:
        """Sanitize the required requested tool dimension."""

        return _safe_required_id(value, field="requested_tool_name")

    @field_validator(
        "correlation_id",
        "effective_tool_name",
        "module_id",
        "category",
        "profile_id",
        "mode_id",
        "model_id",
        "tool_prompt_id",
        "tool_prompt_version",
        "prompt_variant",
        "action_family",
        "result_kind",
        "latency_bucket",
        "grant_outcome",
        "approval_outcome",
        "installation_status",
        "runtime_availability",
        "capture_ref",
        mode="before",
    )
    @classmethod
    def _sanitize_optional_id_fields(cls, value: Any, info: Any) -> str | None:
        """Sanitize optional scalar dimensions."""

        return _safe_optional_id(value, field=str(info.field_name))

    @field_validator("reason_code", mode="before")
    @classmethod
    def _sanitize_reason_code(cls, value: Any) -> str | None:
        """Sanitize reason codes without preserving raw error text."""

        return sanitize_reason_code(value)

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _normalize_duration_ms(cls, value: Any) -> float | None:
        """Return a non-negative duration when provided."""

        if value is None:
            return None
        try:
            duration = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, duration)

    @model_validator(mode="after")
    def _derive_event_fields(self) -> ToolUseEvent:
        """Derive dependent fields after validation."""

        object.__setattr__(
            self,
            "created_at_epoch_us",
            _epoch_microseconds(self.created_at_utc),
        )
        if self.effective_tool_name is None:
            object.__setattr__(
                self,
                "effective_tool_name",
                self.requested_tool_name,
            )
        return self


class ToolUseEventQuery(BaseModel):
    """Bounded query filters for metadata-only tool-use events."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    limit: int = MAX_EVENT_QUERY_LIMIT
    cursor: str | None = None
    runtime_surface: RuntimeSurface | None = None
    requested_tool_name: str | None = None
    effective_tool_name: str | None = None
    profile_id: str | None = None
    mode_id: str | None = None
    model_id: str | None = None
    tool_prompt_id: str | None = None
    status: ToolUseStatus | None = None
    created_at_epoch_us_gte: int | None = None
    created_at_epoch_us_lt: int | None = None

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, value: Any) -> int:
        """Clamp query limits to a bounded, positive range."""

        try:
            limit = int(value)
        except (TypeError, ValueError):
            return MAX_EVENT_QUERY_LIMIT
        return min(max(1, limit), MAX_EVENT_QUERY_LIMIT)

    @field_validator(
        "cursor",
        "requested_tool_name",
        "effective_tool_name",
        "profile_id",
        "mode_id",
        "model_id",
        "tool_prompt_id",
        mode="before",
    )
    @classmethod
    def _sanitize_query_ids(cls, value: Any, info: Any) -> str | None:
        """Sanitize optional query dimensions."""

        return _safe_optional_id(value, field=str(info.field_name))


class ToolUseReportQuery(BaseModel):
    """Bounded aggregate report request for tool-use events."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    group_by: ToolUseReportGroupBy
    event_limit: int = 1_000
    group_limit: int = 100
    top_reason_code_limit: int = 5
    runtime_surface: RuntimeSurface | None = None
    requested_tool_name: str | None = None
    effective_tool_name: str | None = None
    profile_id: str | None = None
    mode_id: str | None = None
    model_id: str | None = None
    tool_prompt_id: str | None = None
    status: ToolUseStatus | None = None
    created_at_epoch_us_gte: int | None = None
    created_at_epoch_us_lt: int | None = None

    @field_validator("event_limit", mode="before")
    @classmethod
    def _clamp_event_limit(cls, value: Any) -> int:
        """Clamp report scan limits."""

        try:
            limit = int(value)
        except (TypeError, ValueError):
            return 1_000
        return min(max(1, limit), MAX_REPORT_EVENT_LIMIT)

    @field_validator("group_limit", mode="before")
    @classmethod
    def _clamp_group_limit(cls, value: Any) -> int:
        """Clamp report group limits."""

        try:
            limit = int(value)
        except (TypeError, ValueError):
            return 100
        return min(max(1, limit), MAX_REPORT_GROUP_LIMIT)

    @field_validator("top_reason_code_limit", mode="before")
    @classmethod
    def _clamp_reason_code_limit(cls, value: Any) -> int:
        """Clamp report reason-code limits."""

        try:
            limit = int(value)
        except (TypeError, ValueError):
            return 5
        return min(max(1, limit), MAX_REPORT_REASON_CODE_LIMIT)

    @field_validator(
        "requested_tool_name",
        "effective_tool_name",
        "profile_id",
        "mode_id",
        "model_id",
        "tool_prompt_id",
        mode="before",
    )
    @classmethod
    def _sanitize_report_ids(cls, value: Any, info: Any) -> str | None:
        """Sanitize optional report filters."""

        return _safe_optional_id(value, field=str(info.field_name))

    def to_event_query(self, *, limit: int) -> ToolUseEventQuery:
        """Convert report filters to an event query."""

        return ToolUseEventQuery(
            limit=limit,
            runtime_surface=self.runtime_surface,
            requested_tool_name=self.requested_tool_name,
            effective_tool_name=self.effective_tool_name,
            profile_id=self.profile_id,
            mode_id=self.mode_id,
            model_id=self.model_id,
            tool_prompt_id=self.tool_prompt_id,
            status=self.status,
            created_at_epoch_us_gte=self.created_at_epoch_us_gte,
            created_at_epoch_us_lt=self.created_at_epoch_us_lt,
        )


class ToolUseReportRow(BaseModel):
    """One aggregate report row for a bounded event set."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    group_key: str
    call_count: int
    tool_call_success_rate: float
    top_reason_codes: list[dict[str, int | str]] = Field(default_factory=list)
    p50_duration_ms: float | None = None
    p95_duration_ms: float | None = None


class ToolUseReport(BaseModel):
    """Aggregate report payload with bounded disclosure metadata."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    rows: list[ToolUseReportRow] = Field(default_factory=list)
    events_scanned: int
    event_limit: int
    truncated: bool
