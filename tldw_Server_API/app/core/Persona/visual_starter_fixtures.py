"""Bundled Persona Visual starter-pack fixtures.

The fixtures in this module are immutable server-owned source material. Runtime
code must copy their assets and manifests into normal user-owned draft visual
packs before a persona can use them. The fixture artwork is intentionally
lightweight; it establishes catalog contracts and copy semantics rather than a
production image-generation pipeline. These are catalog scaffolds, not claims
that final default buddy art or character animation packs have been authored.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Any


DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID = "research-buddy-basic"
LEGACY_PERSONA_VISUAL_STARTER_PACK_ID = "research-buddy-starter"
DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS: tuple[str, ...] = (
    "research-buddy-basic",
    "migu-marker-basic",
    "minimal-helper-basic",
    "study-desk-intermediate",
    "tool-helper-intermediate",
    "object-creature-intermediate",
    "lofi-study-intricate",
    "action-guide-intricate",
    "elaborate-persona-intricate",
)

_REQUIRED_STATE_IDS = ("idle", "listening", "thinking", "speaking", "error")


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


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk with length and CRC fields."""
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _solid_png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    """Return deterministic RGBA PNG bytes without requiring Pillow at runtime."""
    pixel = bytes(rgba)
    scanline = b"\x00" + pixel * width
    raw = scanline * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _asset(
    starter_id: str,
    key: str,
    rgba: tuple[int, int, int, int],
    *,
    size: tuple[int, int] = (4, 4),
    role: str = "frame",
) -> PersonaVisualStarterAsset:
    """Create one deterministic raster fixture asset for a starter pack."""
    return PersonaVisualStarterAsset(
        asset_key=key,
        filename=f"{starter_id}-{key}.png",
        mime_type="image/png",
        content=_solid_png(size[0], size[1], rgba),
        asset_role=role,
    )


def _animation(asset_key: str, *, duration_ms: int = 250) -> dict[str, Any]:
    """Create single-frame sprite animation metadata for a fixture asset key."""
    return {
        "frames": [{"asset_id": asset_key, "duration_ms": duration_ms}],
        "frame_rate": 1,
        "preview_asset_id": asset_key,
    }


def _atlas_animation(
    asset_key: str,
    *,
    y: int,
    duration_ms: int = 160,
) -> dict[str, Any]:
    """Create two-frame atlas animation metadata with bounded sheet regions."""
    return {
        "frames": [
            {
                "asset_id": asset_key,
                "region": {"x": 0, "y": y, "width": 2, "height": 2},
                "duration_ms": duration_ms,
            },
            {
                "asset_id": asset_key,
                "region": {"x": 2, "y": y, "width": 2, "height": 2},
                "duration_ms": duration_ms,
            },
        ],
        "frame_rate": 8,
        "preview_frame": 0,
    }


def _sprite_manifest(
    *,
    base_asset_key: str,
    state_asset_keys: dict[str, str] | None = None,
    custom_states: dict[str, dict[str, Any]] | None = None,
    custom_state_assets: dict[str, str] | None = None,
    authored_triggers: list[dict[str, Any]] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Create a sprite_frames manifest with required and optional custom states."""
    state_asset_keys = state_asset_keys or {}
    animations: dict[str, Any] = {}
    states: dict[str, dict[str, str]] = {}
    for state in _REQUIRED_STATE_IDS:
        asset_key = state_asset_keys.get(state, base_asset_key)
        animation_id = f"{state}-loop"
        animations[animation_id] = _animation(asset_key)
        states[state] = {"animation_id": animation_id}

    for state, asset_key in (custom_state_assets or {}).items():
        animation_id = state.replace(".", "-").replace(":", "-") + "-loop"
        animations[animation_id] = _animation(asset_key, duration_ms=180)
        states[state] = {"animation_id": animation_id}

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": states,
        "animations": animations,
    }
    if custom_states:
        manifest["state_catalog"] = custom_states
    if authored_triggers:
        manifest["authored_triggers"] = authored_triggers
    if fallbacks:
        manifest["fallbacks"] = fallbacks
    return manifest


def _atlas_manifest(asset_key: str) -> dict[str, Any]:
    """Create a sprite_frames manifest that demonstrates atlas-backed frames."""
    states = {
        "idle": {"animation_id": "idle-loop"},
        "listening": {"animation_id": "listening-loop"},
        "thinking": {"animation_id": "thinking-loop"},
        "speaking": {"animation_id": "speaking-loop"},
        "error": {"animation_id": "error-loop"},
        "tool.research": {"animation_id": "tool-research-loop"},
    }
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "state_catalog": {
            "tool.research": {
                "label": "Researching",
                "kind": "tool_variant",
                "description": "Used while research or retrieval tools are running.",
                "tags": ["tool", "search"],
            }
        },
        "states": states,
        "fallbacks": {"tool.research": ["tool_running", "thinking", "idle"]},
        "authored_triggers": [
            {
                "id": "research-tool-category",
                "source": "tool_category",
                "match": "search",
                "state": "tool.research",
                "duration_ms": 2400,
                "priority": 70,
            }
        ],
        "animations": {
            "idle-loop": _atlas_animation(asset_key, y=0),
            "listening-loop": _atlas_animation(asset_key, y=0),
            "thinking-loop": _atlas_animation(asset_key, y=2),
            "speaking-loop": _atlas_animation(asset_key, y=2, duration_ms=120),
            "error-loop": _atlas_animation(asset_key, y=0, duration_ms=300),
            "tool-research-loop": _atlas_animation(asset_key, y=2, duration_ms=140),
        },
    }


def _basic_pack(
    *,
    starter_id: str,
    title: str,
    description: str,
    rgba: tuple[int, int, int, int],
    tags: tuple[str, ...],
) -> PersonaVisualStarterPack:
    """Create a low-complexity starter whose states reuse one neutral asset."""
    asset = _asset(starter_id, "neutral", rgba)
    return PersonaVisualStarterPack(
        id=starter_id,
        title=title,
        description=(
            "Catalog scaffold fixture, not final character art or animation: "
            f"{description}"
        ),
        renderer_type="sprite_frames",
        manifest=_sprite_manifest(base_asset_key=asset.asset_key),
        assets=(asset,),
        tags=("starter", "sprite_frames", "catalog:scaffold", "tier:basic", *tags),
    )


def _multi_asset_pack(
    *,
    starter_id: str,
    title: str,
    description: str,
    palette: tuple[tuple[int, int, int, int], ...],
    tags: tuple[str, ...],
    custom_states: dict[str, dict[str, Any]] | None = None,
    authored_triggers: list[dict[str, Any]] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> PersonaVisualStarterPack:
    """Create an intermediate starter with separate required-state assets."""
    asset_keys = (
        "idle",
        "listening",
        "thinking",
        "speaking",
        "error",
        *(("variant",) if custom_states else ()),
    )
    assets = tuple(
        _asset(starter_id, key, palette[index % len(palette)])
        for index, key in enumerate(asset_keys)
    )
    state_asset_keys = {state: state for state in _REQUIRED_STATE_IDS}
    custom_state_assets = {
        state: "variant"
        for state in (custom_states or {})
    }
    return PersonaVisualStarterPack(
        id=starter_id,
        title=title,
        description=(
            "Catalog scaffold fixture, not final character art or animation: "
            f"{description}"
        ),
        renderer_type="sprite_frames",
        manifest=_sprite_manifest(
            base_asset_key="idle",
            state_asset_keys=state_asset_keys,
            custom_states=custom_states,
            custom_state_assets=custom_state_assets,
            authored_triggers=authored_triggers,
            fallbacks=fallbacks,
        ),
        assets=assets,
        tags=("starter", "sprite_frames", "catalog:scaffold", *tags),
    )


def _atlas_pack(
    *,
    starter_id: str,
    title: str,
    description: str,
    rgba: tuple[int, int, int, int],
    tags: tuple[str, ...],
) -> PersonaVisualStarterPack:
    """Create an intricate starter that uses one sprite-sheet fixture asset."""
    asset = _asset(starter_id, "atlas", rgba, size=(4, 4), role="sprite_sheet")
    return PersonaVisualStarterPack(
        id=starter_id,
        title=title,
        description=(
            "Catalog scaffold fixture, not final character art or animation: "
            f"{description}"
        ),
        renderer_type="sprite_frames",
        manifest=_atlas_manifest(asset.asset_key),
        assets=(asset,),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:scaffold",
            "tier:intricate",
            "atlas",
            *tags,
        ),
    )


DEFAULT_PERSONA_VISUAL_STARTER_PACKS: tuple[PersonaVisualStarterPack, ...] = (
    _basic_pack(
        starter_id="research-buddy-basic",
        title="Research Buddy Basic",
        description=(
            "Clean assistant mascot starter with readable required-state coverage."
        ),
        rgba=(48, 96, 160, 255),
        tags=("research", "mascot"),
    ),
    _basic_pack(
        starter_id="migu-marker-basic",
        title="Migu Marker Basic",
        description=(
            "Rough marker-line inspired starter that keeps a playful user-art feel."
        ),
        rgba=(12, 190, 200, 255),
        tags=("user-art", "marker"),
    ),
    _basic_pack(
        starter_id="minimal-helper-basic",
        title="Minimal Helper Basic",
        description="Geometric low-complexity starter for quick custom Buddy setup.",
        rgba=(92, 118, 48, 255),
        tags=("minimal", "geometric"),
    ),
    _multi_asset_pack(
        starter_id="study-desk-intermediate",
        title="Study Desk Intermediate",
        description=(
            "Calm study companion starter with separate required-state frame assets."
        ),
        palette=(
            (76, 106, 146, 255),
            (95, 128, 176, 255),
            (126, 111, 168, 255),
        ),
        tags=("tier:intermediate", "study", "desk"),
    ),
    _multi_asset_pack(
        starter_id="tool-helper-intermediate",
        title="Tool Helper Intermediate",
        description=(
            "Utility-themed starter with a declared exact tool animation variant."
        ),
        palette=(
            (44, 132, 116, 255),
            (72, 150, 132, 255),
            (96, 166, 148, 255),
        ),
        tags=("tier:intermediate", "tool", "variant"),
        custom_states={
            "tool.notes_search": {
                "label": "Searching notes",
                "kind": "tool_variant",
                "description": "Used when a notes search tool is running.",
                "tags": ["tool", "notes"],
            }
        },
        authored_triggers=[
            {
                "id": "notes-search-tool",
                "source": "tool_name",
                "match": "notes.search",
                "state": "tool.notes_search",
                "duration_ms": 2400,
                "priority": 80,
            }
        ],
        fallbacks={"tool.notes_search": ["tool_running", "thinking", "idle"]},
    ),
    _multi_asset_pack(
        starter_id="object-creature-intermediate",
        title="Object Creature Intermediate",
        description=(
            "Non-human expressive object starter to show the format is not humanoid-only."
        ),
        palette=(
            (144, 92, 72, 255),
            (168, 118, 88, 255),
            (190, 142, 108, 255),
        ),
        tags=("tier:intermediate", "object", "creature"),
        custom_states={
            "reaction.success": {
                "label": "Success reaction",
                "kind": "reaction",
                "description": "Small celebratory reaction for completed work.",
                "tags": ["reaction"],
            }
        },
        fallbacks={"reaction.success": ["speaking", "idle"]},
    ),
    _atlas_pack(
        starter_id="lofi-study-intricate",
        title="Lo-fi Study Intricate",
        description=(
            "Original study companion starter with atlas-backed loops and tool variant metadata."
        ),
        rgba=(108, 84, 164, 255),
        tags=("lofi", "study"),
    ),
    _multi_asset_pack(
        starter_id="action-guide-intricate",
        title="Action Guide Intricate",
        description=(
            "Energetic guide starter with reaction beats and richer state metadata."
        ),
        palette=(
            (186, 72, 76, 255),
            (208, 96, 88, 255),
            (232, 132, 92, 255),
        ),
        tags=("tier:intricate", "action", "reaction"),
        custom_states={
            "reaction.anticipation": {
                "label": "Anticipation",
                "kind": "reaction",
                "description": "Short anticipation beat before high-energy responses.",
                "tags": ["reaction", "motion"],
            },
            "reaction.success": {
                "label": "Success",
                "kind": "reaction",
                "description": "Celebratory response after successful tool completion.",
                "tags": ["reaction"],
            },
        },
        fallbacks={
            "reaction.anticipation": ["thinking", "idle"],
            "reaction.success": ["speaking", "idle"],
        },
    ),
    _multi_asset_pack(
        starter_id="elaborate-persona-intricate",
        title="Elaborate Persona Intricate",
        description=(
            "High-detail fantasy/sci-fi starter demonstrating multiple custom-state rows."
        ),
        palette=(
            (92, 78, 142, 255),
            (116, 96, 166, 255),
            (148, 124, 192, 255),
        ),
        tags=("tier:intricate", "fantasy", "sci-fi"),
        custom_states={
            "mood.focused": {
                "label": "Focused mood",
                "kind": "mood",
                "description": "A concentrated state for deep work sessions.",
                "tags": ["mood"],
            },
            "tool.media_import": {
                "label": "Importing media",
                "kind": "tool_variant",
                "description": "Used while media import or ingestion tools are running.",
                "tags": ["tool", "media"],
            },
        },
        authored_triggers=[
            {
                "id": "media-import-category",
                "source": "tool_category",
                "match": "ingestion",
                "state": "tool.media_import",
                "duration_ms": 2400,
                "priority": 75,
            }
        ],
        fallbacks={
            "mood.focused": ["thinking", "idle"],
            "tool.media_import": ["tool_running", "thinking", "idle"],
        },
    ),
)


__all__ = [
    "DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID",
    "DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS",
    "DEFAULT_PERSONA_VISUAL_STARTER_PACKS",
    "LEGACY_PERSONA_VISUAL_STARTER_PACK_ID",
    "PersonaVisualStarterAsset",
    "PersonaVisualStarterPack",
]
