"""Dataclasses used by pure VN asset pack core helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.VN_Assets.constants import (
    ITEM_REVIEW_STATUS_DRAFT,
    PACK_STATUS_DRAFT,
    SLOT_STATUS_PLANNED,
)


@dataclass(slots=True, kw_only=True)
class VNAssetPack:
    """Minimal pack record shape consumed by core helpers."""

    id: int
    owner_user_id: int
    title: str
    primary_character_id: int
    description: str | None = None
    status: str = PACK_STATUS_DRAFT
    content_rating: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class VNAssetSlot:
    """Minimal slot record shape consumed by matrix, state, and manifest helpers."""

    asset_type: str
    slot_key: str
    id: int | None = None
    pack_id: int | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    prompt_template: str | None = None
    negative_prompt_template: str | None = None
    variant_count: int = 1
    width: int | None = None
    height: int | None = None
    requires_review: bool = True
    required_for_runtime: bool = True
    depends_on_slot_id: int | None = None
    depends_on_slot_key: str | None = None
    status: str = SLOT_STATUS_PLANNED
    last_error: str | None = None


@dataclass(slots=True, kw_only=True)
class VNAssetItem:
    """Minimal item record shape consumed by manifest helpers."""

    id: int
    pack_id: int
    slot_id: int
    variant_index: int = 0
    review_status: str = ITEM_REVIEW_STATUS_DRAFT
    generated_file_id: int | None = None
    file_artifact_id: str | None = None
    storage_ref: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    preferred: bool = False
    source: str = "generated"
    depth_kind: str | None = None
    parent_item_id: int | None = None
    has_alpha: bool | None = None
    crop_box: dict[str, Any] | None = None
    anchor: dict[str, float] | None = None
    scale_hint: float | None = None
    trim_status: str = "unknown"
    quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SlotReadiness:
    """Derived readiness for a single slot."""

    slot_id: int
    status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class PackReadiness:
    """Derived readiness for a pack."""

    ready: bool
    status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
