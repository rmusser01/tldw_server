"""Renderer capability registry for Persona/Buddy visual packs."""

from __future__ import annotations

from dataclasses import dataclass


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


_SPRITE_FRAMES = PersonaVisualRendererCapability(
    renderer_type="sprite_frames",
    display_name="Sprite frames",
    manifest_versions=(1,),
    can_validate=True,
    can_activate=True,
    buddy_runtime_supported=True,
    import_supported=True,
    export_supported=True,
)

_CAPABILITIES: dict[str, PersonaVisualRendererCapability] = {
    _SPRITE_FRAMES.renderer_type: _SPRITE_FRAMES,
}


def list_persona_visual_renderer_capabilities() -> tuple[PersonaVisualRendererCapability, ...]:
    """Return enabled renderer capabilities exposed by this server."""

    return tuple(_CAPABILITIES.values())


def get_persona_visual_renderer_capability(
    renderer_type: str,
) -> PersonaVisualRendererCapability | None:
    """Return the enabled capability for a renderer type, if supported."""

    return _CAPABILITIES.get(str(renderer_type or ""))
