"""Renderer capability registry for Persona/Buddy visual packs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


_DEFAULT_ALLOWED_MIME_TYPES = (
    "image/png",
    "image/jpeg",
    "image/webp",
)
_DEFAULT_ALLOWED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)
_DEFAULT_MAX_FILE_COUNT = 256
_DEFAULT_MAX_TOTAL_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_TEXTURE_SIZE = 4096


@dataclass(frozen=True)
class PersonaVisualRendererCapability:
    """Server-supported renderer behavior for Persona Visual Pack manifests."""

    renderer_type: str
    display_name: str
    manifest_versions: tuple[int, ...]
    can_validate: bool
    can_activate: bool
    buddy_runtime_supported: bool
    import_supported: bool
    export_supported: bool
    disabled_reason: str | None = None
    renderer_contract_versions: tuple[int, ...] = ()
    supported_asset_roles: tuple[str, ...] = ()
    required_role_categories: tuple[str, ...] = ()
    role_category_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    allowed_mime_types: tuple[str, ...] = _DEFAULT_ALLOWED_MIME_TYPES
    allowed_extensions: tuple[str, ...] = _DEFAULT_ALLOWED_EXTENSIONS
    max_file_count: int | None = _DEFAULT_MAX_FILE_COUNT
    max_total_bytes: int | None = _DEFAULT_MAX_TOTAL_BYTES
    max_texture_width: int | None = _DEFAULT_MAX_TEXTURE_SIZE
    max_texture_height: int | None = _DEFAULT_MAX_TEXTURE_SIZE
    feature_flag: str | None = None
    setup_status: str = "supported"
    setup_blockers: tuple[str, ...] = ()
    requires_static_fallback: bool = False
    requires_license_ack: bool = False


_SPRITE_FRAMES = PersonaVisualRendererCapability(
    renderer_type="sprite_frames",
    display_name="Sprite frames",
    manifest_versions=(1,),
    can_validate=True,
    can_activate=True,
    buddy_runtime_supported=True,
    import_supported=True,
    export_supported=True,
    renderer_contract_versions=(1,),
    supported_asset_roles=(
        "frame",
        "still_pose",
        "sprite_sheet",
        "preview",
        "generated_candidate",
    ),
    role_category_map={
        "frame": ("frame",),
        "sprite_sheet": ("sprite_sheet",),
        "preview": ("preview",),
    },
)

_LIVE2D = PersonaVisualRendererCapability(
    renderer_type="live2d",
    display_name="Live2D",
    manifest_versions=(2,),
    can_validate=False,
    can_activate=False,
    buddy_runtime_supported=False,
    import_supported=False,
    export_supported=False,
    disabled_reason="runtime_adapter_not_implemented",
    renderer_contract_versions=(1,),
    supported_asset_roles=(
        "fallback_preview",
        "source_manifest",
        "license_notice",
        "live2d_model_manifest",
        "live2d_moc",
        "live2d_texture",
        "live2d_motion",
        "live2d_expression",
        "live2d_physics",
        "live2d_pose",
        "live2d_userdata",
    ),
    required_role_categories=("fallback_preview", "source_manifest"),
    role_category_map={
        "fallback_preview": ("fallback_preview",),
        "source_manifest": ("live2d_model_manifest",),
        "license_notice": ("license_notice",),
    },
    allowed_mime_types=(
        *_DEFAULT_ALLOWED_MIME_TYPES,
        "application/json",
        "application/octet-stream",
        "text/plain",
    ),
    allowed_extensions=(
        *_DEFAULT_ALLOWED_EXTENSIONS,
        ".model3.json",
        ".moc3",
        ".motion3.json",
        ".exp3.json",
        ".physics3.json",
        ".pose3.json",
        ".userdata3.json",
        ".txt",
        ".md",
    ),
    feature_flag="persona_visual_live2d",
    setup_status="unsupported_renderer",
    setup_blockers=("runtime_adapter_not_implemented",),
    requires_static_fallback=True,
)

_CAPABILITIES: dict[str, PersonaVisualRendererCapability] = {
    _SPRITE_FRAMES.renderer_type: _SPRITE_FRAMES,
    _LIVE2D.renderer_type: _LIVE2D,
}


def list_persona_visual_renderer_capabilities() -> tuple[PersonaVisualRendererCapability, ...]:
    """Return known renderer capabilities exposed by this server."""

    return tuple(_CAPABILITIES.values())


def get_persona_visual_renderer_capability(
    renderer_type: str,
) -> PersonaVisualRendererCapability | None:
    """Return the known capability for a renderer type, if registered."""

    return _CAPABILITIES.get(str(renderer_type or ""))
