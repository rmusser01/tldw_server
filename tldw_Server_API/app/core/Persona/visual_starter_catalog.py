"""Bundled Persona Visual starter-pack catalog.

Starter packs are immutable server-bundled inputs. Runtime use copies their
assets into user-owned draft storage before any activation can happen.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersonaVisualStarterAsset:
    """One immutable asset bundled with a starter pack."""

    id: str
    filename: str
    mime_type: str
    asset_role: str
    content: bytes


@dataclass(frozen=True)
class PersonaVisualStarterPack:
    """Immutable bundled Persona Visual pack definition."""

    id: str
    title: str
    description: str
    renderer_type: str
    manifest: dict[str, Any]
    assets: tuple[PersonaVisualStarterAsset, ...]

    @property
    def manifest_version(self) -> int:
        return int(self.manifest.get("manifest_version") or 1)


_RESEARCH_BUDDY_IDLE_ASSET_ID = "research-buddy-idle-frame"
_RESEARCH_BUDDY_IDLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M/wHwAEAQH/"
    "cetH5QAAAABJRU5ErkJggg=="
)

_RESEARCH_BUDDY_MANIFEST: dict[str, Any] = {
    "manifest_version": 1,
    "renderer_type": "sprite_frames",
    "states": {
        "idle": {"animation_id": "idle"},
        "listening": {"animation_id": "idle"},
        "thinking": {"animation_id": "idle"},
        "speaking": {"animation_id": "idle"},
        "error": {"animation_id": "idle"},
    },
    "animations": {
        "idle": {
            "frames": [
                {
                    "asset_id": _RESEARCH_BUDDY_IDLE_ASSET_ID,
                    "duration_ms": 1000,
                }
            ],
            "frame_rate": 1,
            "loop": True,
        }
    },
}

_STARTER_PACKS: tuple[PersonaVisualStarterPack, ...] = (
    PersonaVisualStarterPack(
        id="research-buddy-sprite-frames-v1",
        title="Research Buddy starter",
        description="A minimal sprite_frames starter pack for first-run Persona Buddy setup.",
        renderer_type="sprite_frames",
        manifest=_RESEARCH_BUDDY_MANIFEST,
        assets=(
            PersonaVisualStarterAsset(
                id=_RESEARCH_BUDDY_IDLE_ASSET_ID,
                filename="research-buddy-idle.png",
                mime_type="image/png",
                asset_role="frame",
                content=_RESEARCH_BUDDY_IDLE_PNG,
            ),
        ),
    ),
)


def list_persona_visual_starter_packs() -> list[PersonaVisualStarterPack]:
    """Return immutable bundled Persona Visual starter packs."""
    return list(_STARTER_PACKS)


def get_persona_visual_starter_pack(starter_pack_id: str) -> PersonaVisualStarterPack | None:
    """Return one bundled starter pack by stable ID."""
    normalized_id = str(starter_pack_id or "").strip()
    for starter_pack in _STARTER_PACKS:
        if starter_pack.id == normalized_id:
            return starter_pack
    return None


__all__ = [
    "PersonaVisualStarterAsset",
    "PersonaVisualStarterPack",
    "get_persona_visual_starter_pack",
    "list_persona_visual_starter_packs",
]
