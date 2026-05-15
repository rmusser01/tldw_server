"""Bundled Persona Visual starter-pack fixtures.

The fixtures in this module are immutable server-owned source material. Runtime
code must copy their assets and manifests into normal user-owned draft visual
packs before a persona can use them.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any


DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID = "research-buddy-starter"


@dataclass(frozen=True)
class PersonaVisualStarterAsset:
    """One immutable asset included in a bundled Persona Visual starter pack."""

    asset_key: str
    filename: str
    mime_type: str
    content: bytes
    asset_role: str = "frame"


@dataclass(frozen=True)
class PersonaVisualStarterPack:
    """Immutable bundled starter-pack definition using local fixture asset keys."""

    id: str
    title: str
    description: str
    renderer_type: str
    manifest: dict[str, Any]
    assets: tuple[PersonaVisualStarterAsset, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)
    license_label: str = "bundled"


_STARTER_IDLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGM0SFjwn4GBgYEJRIAwACC4AjN5lYLvAAAAAElFTkSuQmCC"
)


def _research_buddy_manifest() -> dict[str, Any]:
    states = {
        state: {"animation_id": "idle"}
        for state in ("idle", "listening", "thinking", "speaking", "error")
    }
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": states,
        "animations": {
            "idle": {
                "frames": [{"asset_id": "starter_idle", "duration_ms": 250}],
                "frame_rate": 1,
                "preview_asset_id": "starter_idle",
            }
        },
    }


DEFAULT_PERSONA_VISUAL_STARTER_PACKS: tuple[PersonaVisualStarterPack, ...] = (
    PersonaVisualStarterPack(
        id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
        title="Research Buddy Starter",
        description="A minimal bundled sprite-frame visual for first-run Buddy setup.",
        renderer_type="sprite_frames",
        manifest=_research_buddy_manifest(),
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="research-buddy-idle.png",
                mime_type="image/png",
                content=_STARTER_IDLE_PNG,
                asset_role="frame",
            ),
        ),
        tags=("starter", "sprite_frames"),
    ),
)


__all__ = [
    "DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID",
    "DEFAULT_PERSONA_VISUAL_STARTER_PACKS",
    "PersonaVisualStarterAsset",
    "PersonaVisualStarterPack",
]
