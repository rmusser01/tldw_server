"""Bundled Persona Visual starter-pack fixtures.

The fixtures in this module are immutable server-owned source material. Runtime
code must copy their assets and manifests into normal user-owned draft visual
packs before a persona can use them. Bundled artwork includes PNG sequences
and region-based sprite atlases.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from tldw_Server_API.app.core.Persona.visual_artwork import ARTWORK_MANIFEST_KEY

DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID = "search-lens-basic"
LEGACY_PERSONA_VISUAL_STARTER_PACK_ID = "research-buddy-starter"
DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS: tuple[str, ...] = (
    "search-lens-basic",
    "index-card-basic",
    "archive-cube-basic",
    "paperclip-basic",
    "terminal-tile-basic",
    "migu-marker-basic",
    "pixel-migu",
    "study-desk-intermediate",
    "tool-helper-intermediate",
    "object-creature-intermediate",
    "lofi-study-intricate",
    "action-guide-intricate",
    "elaborate-persona-intricate",
)

_REQUIRED_STATE_IDS = ("idle", "listening", "thinking", "speaking", "error")
_BUILTIN_FALLBACKS = {
    "wake_armed": ["listening"],
    "tool_running": ["thinking"],
    "approval_needed": ["thinking"],
    "offline": ["idle"],
}
_AUTHORED_ARTWORK_LICENSE = "AGPL-3.0-only"
_BASIC_EXPECTED_ASSET_GROUPS = (
    "identity_brief",
    "neutral_anchor",
    "preview_image",
    "required_state_loops",
)
_INTERMEDIATE_EXPECTED_ASSET_GROUPS = (
    "identity_brief",
    "neutral_anchor",
    "static_talking_sheet",
    "static_reaction_sheet",
    "required_state_loops",
    "custom_state_variants",
)
_INTRICATE_EXPECTED_ASSET_GROUPS = (
    "identity_brief",
    "neutral_anchor",
    "model_sheet",
    "static_talking_sheet",
    "static_reaction_sheet",
    "required_state_loops",
    "animation_strips",
    "animation_atlas",
    "custom_state_variants",
)
_BASIC_ANIMATION_NOTES = ("Reviewed bundled basic default with neutral-anchor-derived required-state loops.",)
_RECIPE_REVIEW_CHECKS = (
    "neutral_identity_consistency",
    "transparent_background",
    "one_subject_per_frame",
    "state_manifest_alignment",
)


@dataclass(frozen=True)
class PersonaVisualStarterProductionRecipe:
    """Authored-asset handoff recipe for one bundled starter scaffold."""

    identity_brief: str
    neutral_anchor: str
    static_sheet: str
    animation_outputs: tuple[str, ...]
    review_checks: tuple[str, ...] = _RECIPE_REVIEW_CHECKS


def _production_recipe(
    *,
    identity_brief: str,
    neutral_anchor: str,
    static_sheet: str,
    animation_outputs: tuple[str, ...],
    review_checks: tuple[str, ...] = _RECIPE_REVIEW_CHECKS,
) -> PersonaVisualStarterProductionRecipe:
    """Create one immutable starter production-recipe metadata block."""
    return PersonaVisualStarterProductionRecipe(
        identity_brief=identity_brief,
        neutral_anchor=neutral_anchor,
        static_sheet=static_sheet,
        animation_outputs=animation_outputs,
        review_checks=review_checks,
    )


def _default_basic_production_recipe() -> PersonaVisualStarterProductionRecipe:
    """Return the default low-complexity starter production recipe."""
    return _production_recipe(
        identity_brief="Simple readable buddy with one strong silhouette and limited props.",
        neutral_anchor="Create one front-facing neutral pose used as the identity anchor.",
        static_sheet=("Optional small mouth or expression sheet; keep it separate from timed loops."),
        animation_outputs=("required_state_loops",),
    )


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
    companion_behavior: dict[str, Any] = field(
        default_factory=lambda: {"schema_version": 1, "entries": []}
    )
    tags: tuple[str, ...] = field(default_factory=tuple)
    license_label: str = "bundled"
    complexity_tier: str = "basic"
    production_status: str = "scaffold"
    neutral_anchor_required: bool = True
    expected_asset_groups: tuple[str, ...] = _BASIC_EXPECTED_ASSET_GROUPS
    animation_coverage_notes: tuple[str, ...] = _BASIC_ANIMATION_NOTES
    production_recipe: PersonaVisualStarterProductionRecipe = field(default_factory=_default_basic_production_recipe)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk with length and CRC fields."""
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _png_from_rgba(width: int, height: int, pixels: bytes | bytearray) -> bytes:
    """Return deterministic RGBA PNG bytes without requiring Pillow at runtime."""
    if len(pixels) != width * height * 4:
        raise ValueError("RGBA pixel buffer size does not match image dimensions")
    raw = b"".join(
        b"\x00" + bytes(pixels[row_start : row_start + width * 4]) for row_start in range(0, len(pixels), width * 4)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _blank_canvas(width: int = 96, height: int = 96) -> bytearray:
    """Create a transparent RGBA canvas."""
    return bytearray(width * height * 4)


def _blend_pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int, int],
) -> None:
    """Alpha-blend one pixel into a flat RGBA buffer."""
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    src_alpha = color[3]
    if src_alpha <= 0:
        return
    index = (y * width + x) * 4
    if src_alpha >= 255 or pixels[index + 3] == 0:
        pixels[index : index + 4] = bytes(color)
        return

    dst_alpha = pixels[index + 3]
    out_alpha = src_alpha + (dst_alpha * (255 - src_alpha) // 255)
    if out_alpha <= 0:
        return
    for channel in range(3):
        src_value = color[channel]
        dst_value = pixels[index + channel]
        pixels[index + channel] = (
            src_value * src_alpha + dst_value * dst_alpha * (255 - src_alpha) // 255
        ) // out_alpha
    pixels[index + 3] = out_alpha


def _draw_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a filled rectangle."""
    for y in range(max(0, y0), min(height, y1)):
        for x in range(max(0, x0), min(width, x1)):
            _blend_pixel(pixels, width, height, x, y, color)


def _draw_ellipse(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a filled ellipse."""
    if rx <= 0 or ry <= 0:
        return
    for y in range(cy - ry, cy + ry + 1):
        for x in range(cx - rx, cx + rx + 1):
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            if dx * dx + dy * dy <= 1:
                _blend_pixel(pixels, width, height, x, y, color)


def _draw_ellipse_outline(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    *,
    thickness: int = 2,
) -> None:
    """Draw a simple filled ellipse with an outline."""
    _draw_ellipse(pixels, width, height, cx, cy, rx, ry, outline)
    _draw_ellipse(
        pixels,
        width,
        height,
        cx,
        cy,
        max(1, rx - thickness),
        max(1, ry - thickness),
        fill,
    )


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int, int],
    *,
    thickness: int = 1,
) -> None:
    """Draw a thick anti-aliased-enough line for small sprite art."""
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    radius = max(0, thickness // 2)
    for step in range(steps + 1):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        if radius <= 0:
            _blend_pixel(pixels, width, height, x, y, color)
        else:
            _draw_ellipse(pixels, width, height, x, y, radius, radius, color)


def _draw_round_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a filled rounded rectangle."""
    _draw_rect(pixels, width, height, x0 + radius, y0, x1 - radius, y1, color)
    _draw_rect(pixels, width, height, x0, y0 + radius, x1, y1 - radius, color)
    _draw_ellipse(pixels, width, height, x0 + radius, y0 + radius, radius, radius, color)
    _draw_ellipse(pixels, width, height, x1 - radius - 1, y0 + radius, radius, radius, color)
    _draw_ellipse(pixels, width, height, x0 + radius, y1 - radius - 1, radius, radius, color)
    _draw_ellipse(
        pixels,
        width,
        height,
        x1 - radius - 1,
        y1 - radius - 1,
        radius,
        radius,
        color,
    )


def _draw_round_rect_outline(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    *,
    thickness: int = 2,
) -> None:
    """Draw a filled rounded rectangle with an outline."""
    _draw_round_rect(pixels, width, height, x0, y0, x1, y1, radius, outline)
    _draw_round_rect(
        pixels,
        width,
        height,
        x0 + thickness,
        y0 + thickness,
        x1 - thickness,
        y1 - thickness,
        max(1, radius - thickness),
        fill,
    )


def _draw_diamond(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a filled diamond."""
    for y in range(cy - ry, cy + ry + 1):
        for x in range(cx - rx, cx + rx + 1):
            if abs(x - cx) / max(1, rx) + abs(y - cy) / max(1, ry) <= 1:
                _blend_pixel(pixels, width, height, x, y, color)


def _draw_diamond_outline(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    *,
    thickness: int = 3,
) -> None:
    """Draw a filled diamond with a simple outline."""
    _draw_diamond(pixels, width, height, cx, cy, rx, ry, outline)
    _draw_diamond(
        pixels,
        width,
        height,
        cx,
        cy,
        max(1, rx - thickness),
        max(1, ry - thickness),
        fill,
    )


def _state_loop_animation(
    state: str,
    *,
    duration_ms: int = 220,
) -> dict[str, Any]:
    """Create a two-frame basic-tier animation loop."""
    frame_keys = (f"{state}-1", f"{state}-2")
    return {
        "frames": [
            {"asset_id": frame_keys[0], "duration_ms": duration_ms},
            {"asset_id": frame_keys[1], "duration_ms": duration_ms},
        ],
        "frame_rate": 4,
        "preview_asset_id": frame_keys[0],
    }


def _basic_art_manifest() -> dict[str, Any]:
    """Create a reviewed basic-tier sprite manifest with required-state loops."""
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {state: {"animation_id": f"{state}-loop"} for state in _REQUIRED_STATE_IDS},
        "animations": {f"{state}-loop": _state_loop_animation(state) for state in _REQUIRED_STATE_IDS},
        "fallbacks": {state: targets.copy() for state, targets in _BUILTIN_FALLBACKS.items()},
    }


def _success_reaction_art_manifest(success_description: str) -> dict[str, Any]:
    """Create a basic-tier manifest with required loops and success reaction."""
    manifest = _basic_art_manifest()
    manifest["state_catalog"] = {
        "reaction.success": {
            "label": "Success reaction",
            "kind": "reaction",
            "description": success_description,
            "tags": ["reaction"],
        }
    }
    manifest["states"]["reaction.success"] = {"animation_id": "reaction-success-loop"}
    manifest["fallbacks"]["reaction.success"] = ["speaking", "idle"]
    manifest["animations"]["reaction-success-loop"] = {
        "frames": [{"asset_id": "reaction-success-1", "duration_ms": 700}],
        "frame_rate": 1,
        "preview_asset_id": "reaction-success-1",
    }
    return manifest


def _search_lens_art_manifest() -> dict[str, Any]:
    """Create the Search Lens manifest with required loops and success reaction."""
    return _success_reaction_art_manifest("Single celebratory pose derived from the Search Lens source sheet.")


def _buddy_state_variant(state: str, frame_index: int) -> dict[str, bool | int | str]:
    """Return deterministic pose/expression controls for one basic state frame."""
    bounce = 1 if frame_index % 2 else 0
    if state == "idle":
        return {"bounce": bounce, "mouth": "smile", "eyes": "neutral", "accent": "none"}
    if state == "listening":
        return {
            "bounce": bounce,
            "mouth": "small",
            "eyes": "wide",
            "accent": "listen",
        }
    if state == "thinking":
        return {
            "bounce": bounce,
            "mouth": "flat",
            "eyes": "focused",
            "accent": "thought",
        }
    if state == "speaking":
        return {
            "bounce": bounce,
            "mouth": "open" if frame_index % 2 else "smile",
            "eyes": "happy",
            "accent": "speak",
        }
    return {
        "bounce": bounce,
        "mouth": "worry",
        "eyes": "worried",
        "accent": "error",
    }


def _draw_research_buddy(
    pixels: bytearray,
    width: int,
    height: int,
    state: str,
    frame_index: int,
) -> None:
    """Draw the clean assistant mascot basic default."""
    variant = _buddy_state_variant(state, frame_index)
    y_offset = int(variant["bounce"])
    outline = (31, 42, 66, 255)
    shell = (232, 241, 255, 255)
    blue = (56, 116, 202, 255)
    accent = (97, 217, 214, 255)
    red = (218, 62, 72, 255)

    _draw_line(pixels, width, height, 31, 75 + y_offset, 25, 87, outline, thickness=3)
    _draw_line(pixels, width, height, 65, 75 + y_offset, 71, 87, outline, thickness=3)
    _draw_round_rect_outline(
        pixels,
        width,
        height,
        35,
        62 + y_offset,
        61,
        82 + y_offset,
        7,
        outline,
        blue,
        thickness=2,
    )
    _draw_round_rect_outline(
        pixels,
        width,
        height,
        24,
        22 + y_offset,
        72,
        62 + y_offset,
        12,
        outline,
        shell,
        thickness=3,
    )
    _draw_rect(pixels, width, height, 32, 34 + y_offset, 64, 52 + y_offset, (198, 228, 246, 255))
    _draw_ellipse(pixels, width, height, 38, 43 + y_offset, 3, 4, outline)
    _draw_ellipse(pixels, width, height, 58, 43 + y_offset, 3, 4, outline)

    mouth = str(variant["mouth"])
    if mouth == "open":
        _draw_ellipse(pixels, width, height, 48, 51 + y_offset, 5, 4, outline)
        _draw_ellipse(pixels, width, height, 48, 50 + y_offset, 3, 2, (255, 255, 255, 255))
    elif mouth == "flat":
        _draw_line(pixels, width, height, 43, 51 + y_offset, 53, 51 + y_offset, outline, thickness=2)
    elif mouth == "worry":
        _draw_line(pixels, width, height, 43, 52 + y_offset, 48, 50 + y_offset, outline, thickness=2)
        _draw_line(pixels, width, height, 48, 50 + y_offset, 53, 52 + y_offset, outline, thickness=2)
    else:
        _draw_line(pixels, width, height, 42, 50 + y_offset, 46, 53 + y_offset, outline, thickness=2)
        _draw_line(pixels, width, height, 46, 53 + y_offset, 54, 49 + y_offset, outline, thickness=2)

    _draw_line(pixels, width, height, 48, 22 + y_offset, 48, 13 + y_offset, outline, thickness=2)
    _draw_ellipse(pixels, width, height, 48, 11 + y_offset, 4, 4, accent)
    _draw_ellipse(pixels, width, height, 48, 11 + y_offset, 2, 2, (255, 255, 255, 255))

    accent_name = str(variant["accent"])
    if accent_name == "listen":
        _draw_line(pixels, width, height, 17, 33, 10, 28, accent, thickness=2)
        _draw_line(pixels, width, height, 17, 44, 9, 47, accent, thickness=2)
        _draw_line(pixels, width, height, 79, 33, 86, 28, accent, thickness=2)
        _draw_line(pixels, width, height, 79, 44, 87, 47, accent, thickness=2)
    elif accent_name == "thought":
        _draw_ellipse_outline(pixels, width, height, 76, 21, 5, 4, outline, (255, 255, 255, 255))
        _draw_ellipse_outline(pixels, width, height, 84, 15, 3, 3, outline, (255, 255, 255, 255))
    elif accent_name == "speak":
        _draw_line(pixels, width, height, 75, 39, 84, 35, blue, thickness=2)
        _draw_line(pixels, width, height, 76, 47, 86, 50, blue, thickness=2)
    elif accent_name == "error":
        _draw_line(pixels, width, height, 78, 22, 83, 31, red, thickness=3)
        _draw_ellipse(pixels, width, height, 85, 35, 2, 2, red)


def _draw_migu_marker(
    pixels: bytearray,
    width: int,
    height: int,
    state: str,
    frame_index: int,
) -> None:
    """Draw the marker-line Migu basic default."""
    variant = _buddy_state_variant(state, frame_index)
    y_offset = int(variant["bounce"])
    black = (24, 28, 30, 255)
    teal = (18, 198, 206, 255)
    magenta = (226, 42, 178, 255)
    grey = (214, 218, 214, 255)
    red = (220, 50, 60, 255)

    _draw_line(pixels, width, height, 31, 64 + y_offset, 20, 82, black, thickness=3)
    _draw_line(pixels, width, height, 61, 64 + y_offset, 73, 82, black, thickness=3)
    _draw_line(pixels, width, height, 35, 80, 31, 91, black, thickness=3)
    _draw_line(pixels, width, height, 55, 80, 62, 91, black, thickness=3)
    _draw_round_rect_outline(
        pixels,
        width,
        height,
        31,
        51 + y_offset,
        62,
        78 + y_offset,
        6,
        black,
        grey,
        thickness=2,
    )

    _draw_line(pixels, width, height, 31, 30 + y_offset, 12, 45 + y_offset, teal, thickness=5)
    _draw_line(pixels, width, height, 14, 45 + y_offset, 7, 62 + y_offset, teal, thickness=5)
    _draw_line(pixels, width, height, 61, 30 + y_offset, 82, 46 + y_offset, teal, thickness=5)
    _draw_line(pixels, width, height, 80, 46 + y_offset, 88, 63 + y_offset, teal, thickness=5)
    _draw_line(pixels, width, height, 28, 27 + y_offset, 65, 27 + y_offset, teal, thickness=7)
    _draw_line(pixels, width, height, 31, 33 + y_offset, 62, 31 + y_offset, teal, thickness=6)
    _draw_line(pixels, width, height, 27, 31 + y_offset, 24, 39 + y_offset, black, thickness=2)
    _draw_line(pixels, width, height, 66, 31 + y_offset, 69, 39 + y_offset, black, thickness=2)
    _draw_ellipse(pixels, width, height, 27, 29 + y_offset, 5, 4, magenta)
    _draw_ellipse(pixels, width, height, 66, 29 + y_offset, 5, 4, magenta)

    _draw_ellipse_outline(
        pixels,
        width,
        height,
        47,
        42 + y_offset,
        19,
        16,
        black,
        (252, 249, 239, 255),
        thickness=2,
    )
    _draw_ellipse(pixels, width, height, 40, 41 + y_offset, 2, 3, black)
    _draw_ellipse(pixels, width, height, 55, 41 + y_offset, 2, 3, black)

    mouth = str(variant["mouth"])
    if mouth == "open":
        _draw_ellipse(pixels, width, height, 48, 50 + y_offset, 4, 3, black)
    elif mouth == "worry":
        _draw_line(pixels, width, height, 43, 50 + y_offset, 48, 48 + y_offset, black, thickness=2)
        _draw_line(pixels, width, height, 48, 48 + y_offset, 53, 50 + y_offset, black, thickness=2)
    elif mouth == "flat":
        _draw_line(pixels, width, height, 43, 50 + y_offset, 53, 50 + y_offset, black, thickness=2)
    else:
        _draw_line(pixels, width, height, 43, 48 + y_offset, 47, 51 + y_offset, black, thickness=2)
        _draw_line(pixels, width, height, 47, 51 + y_offset, 55, 47 + y_offset, black, thickness=2)

    accent_name = str(variant["accent"])
    if accent_name == "listen":
        _draw_line(pixels, width, height, 16, 24, 8, 20, teal, thickness=3)
        _draw_line(pixels, width, height, 80, 24, 88, 20, teal, thickness=3)
    elif accent_name == "thought":
        _draw_ellipse_outline(pixels, width, height, 77, 22, 4, 4, black, (255, 255, 255, 255))
        _draw_ellipse_outline(pixels, width, height, 84, 16, 3, 3, black, (255, 255, 255, 255))
    elif accent_name == "speak":
        _draw_line(pixels, width, height, 70, 44, 82, 39, teal, thickness=3)
        _draw_line(pixels, width, height, 71, 51, 84, 55, teal, thickness=3)
    elif accent_name == "error":
        _draw_line(pixels, width, height, 75, 21, 82, 31, red, thickness=4)
        _draw_ellipse(pixels, width, height, 84, 35, 2, 2, red)


def _draw_minimal_helper(
    pixels: bytearray,
    width: int,
    height: int,
    state: str,
    frame_index: int,
) -> None:
    """Draw the geometric minimal-helper basic default."""
    variant = _buddy_state_variant(state, frame_index)
    y_offset = int(variant["bounce"])
    outline = (36, 51, 43, 255)
    green = (106, 148, 76, 255)
    lime = (190, 223, 126, 255)
    blue = (74, 137, 191, 255)
    red = (214, 72, 72, 255)

    _draw_line(pixels, width, height, 33, 63 + y_offset, 22, 72 + y_offset, outline, thickness=4)
    _draw_line(pixels, width, height, 63, 63 + y_offset, 74, 72 + y_offset, outline, thickness=4)
    _draw_line(pixels, width, height, 41, 74 + y_offset, 35, 86, outline, thickness=4)
    _draw_line(pixels, width, height, 55, 74 + y_offset, 61, 86, outline, thickness=4)
    _draw_diamond_outline(
        pixels,
        width,
        height,
        48,
        48 + y_offset,
        29,
        31,
        outline,
        green,
        thickness=4,
    )
    _draw_diamond(pixels, width, height, 48, 44 + y_offset, 17, 17, lime)
    _draw_ellipse(pixels, width, height, 39, 45 + y_offset, 3, 4, outline)
    _draw_ellipse(pixels, width, height, 57, 45 + y_offset, 3, 4, outline)

    mouth = str(variant["mouth"])
    if mouth == "open":
        _draw_rect(pixels, width, height, 44, 55 + y_offset, 53, 60 + y_offset, outline)
    elif mouth == "flat":
        _draw_line(pixels, width, height, 43, 56 + y_offset, 53, 56 + y_offset, outline, thickness=2)
    elif mouth == "worry":
        _draw_line(pixels, width, height, 43, 57 + y_offset, 48, 55 + y_offset, outline, thickness=2)
        _draw_line(pixels, width, height, 48, 55 + y_offset, 53, 57 + y_offset, outline, thickness=2)
    else:
        _draw_line(pixels, width, height, 42, 54 + y_offset, 47, 57 + y_offset, outline, thickness=2)
        _draw_line(pixels, width, height, 47, 57 + y_offset, 55, 53 + y_offset, outline, thickness=2)

    accent_name = str(variant["accent"])
    if accent_name == "listen":
        _draw_ellipse_outline(pixels, width, height, 22, 34, 6, 8, outline, blue, thickness=2)
        _draw_ellipse_outline(pixels, width, height, 74, 34, 6, 8, outline, blue, thickness=2)
    elif accent_name == "thought":
        _draw_diamond_outline(pixels, width, height, 76, 25, 7, 7, outline, (255, 255, 255, 255))
        _draw_diamond_outline(pixels, width, height, 84, 17, 4, 4, outline, (255, 255, 255, 255))
    elif accent_name == "speak":
        _draw_line(pixels, width, height, 73, 43, 85, 38, blue, thickness=3)
        _draw_line(pixels, width, height, 73, 51, 85, 56, blue, thickness=3)
    elif accent_name == "error":
        _draw_line(pixels, width, height, 75, 22, 83, 32, red, thickness=4)
        _draw_ellipse(pixels, width, height, 84, 37, 2, 2, red)


_BASIC_ART_RENDERERS = {
    "research": _draw_research_buddy,
    "migu": _draw_migu_marker,
    "minimal": _draw_minimal_helper,
}


def _basic_art_png(style: str, state: str, frame_index: int) -> bytes:
    """Render one deterministic reviewed basic-tier Buddy frame."""
    pixels = _blank_canvas()
    renderer = _BASIC_ART_RENDERERS[style]
    renderer(pixels, 96, 96, state, frame_index)
    return _png_from_rgba(96, 96, pixels)


def _basic_art_asset(
    starter_id: str,
    style: str,
    key: str,
    *,
    state: str,
    frame_index: int,
    role: str = "frame",
) -> PersonaVisualStarterAsset:
    """Create one reviewed basic-tier raster asset."""
    return PersonaVisualStarterAsset(
        asset_key=key,
        filename=f"{starter_id}-{key}.png",
        mime_type="image/png",
        content=_basic_art_png(style, state, frame_index),
        asset_role=role,
    )


_SEARCH_LENS_FRAME_FILES: tuple[tuple[str, str, str], ...] = (
    ("neutral-anchor", "00_neutral_anchor.png", "still_pose"),
    ("preview", "00_neutral_anchor.png", "preview"),
    ("idle-1", "01_idle_a.png", "frame"),
    ("idle-2", "02_idle_b.png", "frame"),
    ("listening-1", "03_listening_a.png", "frame"),
    ("listening-2", "04_listening_b.png", "frame"),
    ("thinking-1", "05_thinking_a.png", "frame"),
    ("thinking-2", "06_thinking_b.png", "frame"),
    ("speaking-1", "07_speaking_a.png", "frame"),
    ("speaking-2", "08_speaking_b.png", "frame"),
    ("reaction-success-1", "09_success.png", "frame"),
    ("error-1", "10_error_a.png", "frame"),
    ("error-2", "11_error_b.png", "frame"),
)

_INDEX_CARD_FRAME_FILES: tuple[tuple[str, str, str], ...] = (
    ("neutral-anchor", "00_neutral_anchor.png", "still_pose"),
    ("preview", "00_neutral_anchor.png", "preview"),
    ("idle-1", "01_idle_a.png", "frame"),
    ("idle-2", "02_idle_b.png", "frame"),
    ("listening-1", "03_listening_a.png", "frame"),
    ("listening-2", "04_listening_b.png", "frame"),
    ("thinking-1", "05_thinking_a.png", "frame"),
    ("thinking-2", "06_thinking_b.png", "frame"),
    ("speaking-1", "07_speaking_a.png", "frame"),
    ("speaking-2", "08_speaking_b.png", "frame"),
    ("reaction-success-1", "09_success.png", "frame"),
    ("error-1", "10_error_a.png", "frame"),
    ("error-2", "11_error_b.png", "frame"),
)

_ARCHIVE_CUBE_FRAME_FILES: tuple[tuple[str, str, str], ...] = (
    ("neutral-anchor", "00_neutral_anchor.png", "still_pose"),
    ("preview", "00_neutral_anchor.png", "preview"),
    ("idle-1", "01_idle_a.png", "frame"),
    ("idle-2", "02_idle_b.png", "frame"),
    ("listening-1", "03_listening_a.png", "frame"),
    ("listening-2", "04_listening_b.png", "frame"),
    ("thinking-1", "05_thinking_a.png", "frame"),
    ("thinking-2", "06_thinking_b.png", "frame"),
    ("speaking-1", "07_speaking_a.png", "frame"),
    ("speaking-2", "08_speaking_b.png", "frame"),
    ("reaction-success-1", "09_success.png", "frame"),
    ("error-1", "10_error_a.png", "frame"),
    ("error-2", "11_error_b.png", "frame"),
)

_PAPERCLIP_FRAME_FILES: tuple[tuple[str, str, str], ...] = (
    ("neutral-anchor", "00_neutral_anchor.png", "still_pose"),
    ("preview", "00_neutral_anchor.png", "preview"),
    ("idle-1", "01_idle_a.png", "frame"),
    ("idle-2", "02_idle_b.png", "frame"),
    ("listening-1", "03_listening_a.png", "frame"),
    ("listening-2", "04_listening_b.png", "frame"),
    ("thinking-1", "05_thinking_a.png", "frame"),
    ("thinking-2", "06_thinking_b.png", "frame"),
    ("speaking-1", "07_speaking_a.png", "frame"),
    ("speaking-2", "08_speaking_b.png", "frame"),
    ("reaction-success-1", "09_success.png", "frame"),
    ("error-1", "10_error_a.png", "frame"),
    ("error-2", "11_error_b.png", "frame"),
)

_TERMINAL_TILE_FRAME_FILES: tuple[tuple[str, str, str], ...] = (
    ("neutral-anchor", "00_neutral_anchor.png", "still_pose"),
    ("preview", "00_neutral_anchor.png", "preview"),
    ("idle-1", "01_idle_a.png", "frame"),
    ("idle-2", "02_idle_b.png", "frame"),
    ("listening-1", "03_listening_a.png", "frame"),
    ("listening-2", "04_listening_b.png", "frame"),
    ("thinking-1", "05_thinking_a.png", "frame"),
    ("thinking-2", "06_thinking_b.png", "frame"),
    ("speaking-1", "07_speaking_a.png", "frame"),
    ("speaking-2", "08_speaking_b.png", "frame"),
    ("reaction-success-1", "09_success.png", "frame"),
    ("error-1", "10_error_a.png", "frame"),
    ("error-2", "11_error_b.png", "frame"),
)

_MIGU_MARKER_FRAME_FILES: tuple[tuple[str, str, str], ...] = (
    ("neutral-anchor", "00_neutral_anchor.png", "still_pose"),
    ("preview", "00_neutral_anchor.png", "preview"),
    ("idle-1", "01_idle_a.png", "frame"),
    ("idle-2", "02_idle_b.png", "frame"),
    ("listening-1", "03_listening_a.png", "frame"),
    ("listening-2", "04_listening_b.png", "frame"),
    ("thinking-1", "05_thinking_a.png", "frame"),
    ("thinking-2", "06_thinking_b.png", "frame"),
    ("speaking-1", "07_speaking_a.png", "frame"),
    ("speaking-2", "08_speaking_b.png", "frame"),
    ("reaction-success-1", "09_success.png", "frame"),
    ("error-1", "10_error_a.png", "frame"),
    ("error-2", "11_error_b.png", "frame"),
)


def _starter_resource_bytes(starter_id: str, filename: str) -> bytes:
    """Load one bundled starter PNG from package resources."""
    return resources.files(__package__).joinpath("assets", "starter_packs", starter_id, "frames", filename).read_bytes()


def _resource_png_asset(
    starter_id: str,
    key: str,
    filename: str,
    *,
    role: str,
) -> PersonaVisualStarterAsset:
    """Create one starter asset backed by a reviewed package PNG resource."""
    return PersonaVisualStarterAsset(
        asset_key=key,
        filename=f"{starter_id}-{key}.png",
        mime_type="image/png",
        content=_starter_resource_bytes(starter_id, filename),
        asset_role=role,
    )


def _search_lens_basic_pack() -> PersonaVisualStarterPack:
    """Create the reviewed Search Lens low-complexity starter."""
    starter_id = "search-lens-basic"
    return PersonaVisualStarterPack(
        id=starter_id,
        title="Search Lens Basic",
        description=(
            "Bundled art-ready basic Buddy default: friendly magnifying-glass "
            "research helper with approved 3x4 source-sheet frames."
        ),
        renderer_type="sprite_frames",
        manifest=_search_lens_art_manifest(),
        assets=tuple(
            _resource_png_asset(starter_id, key, filename, role=role)
            for key, filename, role in _SEARCH_LENS_FRAME_FILES
        ),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:art-ready",
            "tier:basic",
            "search",
            "research",
            "object-buddy",
        ),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=(
                "Friendly magnifying-glass buddy with a teal-blue lens face, "
                "charcoal rim, short handle body, tiny arms, and two short legs."
            ),
            neutral_anchor=(
                "Neutral anchor: mostly front-facing round lens, charcoal rim, "
                "handle body, dot eyes, smile, arms at sides, and two short legs."
            ),
            static_sheet=(
                "Recreate state loops from the neutral lens pose by changing lens "
                "tilt, arm gesture, mouth shape, and accents while preserving the same legs."
            ),
            animation_outputs=("required_state_loops",),
        ),
    )


def _index_card_basic_pack() -> PersonaVisualStarterPack:
    """Create the reviewed Index Card low-complexity starter."""
    starter_id = "index-card-basic"
    return PersonaVisualStarterPack(
        id=starter_id,
        title="Index Card Basic",
        description=(
            "Bundled art-ready basic Buddy default: animated tabbed index-card "
            "research helper with approved 3x4 source-sheet frames."
        ),
        renderer_type="sprite_frames",
        manifest=_success_reaction_art_manifest("Single celebratory pose derived from the Index Card source sheet."),
        assets=tuple(
            _resource_png_asset(starter_id, key, filename, role=role) for key, filename, role in _INDEX_CARD_FRAME_FILES
        ),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:art-ready",
            "tier:basic",
            "index-card",
            "notes",
            "paper-buddy",
        ),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=(
                "Friendly index card buddy with a cream ruled-card body, tabbed top, "
                "paper lines, dot eyes, smile, tiny arms, and two short legs."
            ),
            neutral_anchor=(
                "Neutral anchor: front-facing ruled card with tabbed top, dot eyes, "
                "small smile, tiny arms, and two short legs on a transparent canvas."
            ),
            static_sheet=(
                "Recreate state loops from the neutral card by changing page tilt, "
                "mouth shape, paper lines, tab motion, and small accent marks."
            ),
            animation_outputs=("required_state_loops",),
        ),
    )


def _archive_cube_basic_pack() -> PersonaVisualStarterPack:
    """Create the reviewed Archive Cube low-complexity starter."""
    starter_id = "archive-cube-basic"
    return PersonaVisualStarterPack(
        id=starter_id,
        title="Archive Cube Basic",
        description=(
            "Bundled art-ready basic Buddy default: compact archive cube "
            "helper with approved 3x4 source-sheet frames."
        ),
        renderer_type="sprite_frames",
        manifest=_success_reaction_art_manifest("Single celebratory pose derived from the Archive Cube source sheet."),
        assets=tuple(
            _resource_png_asset(starter_id, key, filename, role=role)
            for key, filename, role in _ARCHIVE_CUBE_FRAME_FILES
        ),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:art-ready",
            "tier:basic",
            "archive",
            "storage",
            "object-buddy",
        ),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=(
                "Friendly archive cube buddy with muted teal-blue sides, a cream "
                "drawer panel, tab slot, dot eyes, simple mouth, tiny arms, and two short legs."
            ),
            neutral_anchor=(
                "Neutral anchor: front-facing cube body with cream drawer panel, "
                "small label slot, dot eyes, relaxed arms, and two short legs."
            ),
            static_sheet=(
                "Recreate state loops from the neutral cube by changing cube tilt, "
                "mouth shape, drawer-panel emphasis, arm gesture, and alert corner marks."
            ),
            animation_outputs=("required_state_loops",),
        ),
    )


def _paperclip_basic_pack() -> PersonaVisualStarterPack:
    """Create the reviewed Paperclip low-complexity starter."""
    starter_id = "paperclip-basic"
    return PersonaVisualStarterPack(
        id=starter_id,
        title="Paperclip Basic",
        description=(
            "Bundled art-ready basic Buddy default: expressive paperclip "
            "helper with approved 3x4 source-sheet frames."
        ),
        renderer_type="sprite_frames",
        manifest=_success_reaction_art_manifest("Single celebratory pose derived from the Paperclip source sheet."),
        assets=tuple(
            _resource_png_asset(starter_id, key, filename, role=role) for key, filename, role in _PAPERCLIP_FRAME_FILES
        ),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:art-ready",
            "tier:basic",
            "paperclip",
            "office",
            "object-buddy",
        ),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=(
                "Friendly paperclip buddy with a looped wire body, pale metal linework, "
                "small face, tiny arms, and two short legs."
            ),
            neutral_anchor=(
                "Neutral anchor: front-facing looped clip silhouette, simple face, "
                "relaxed arms, and two short legs on a transparent canvas."
            ),
            static_sheet=(
                "Recreate state loops from the neutral clip by changing clip tilt, "
                "wire loop squash, mouth shape, arm gesture, and small alert marks."
            ),
            animation_outputs=("required_state_loops",),
        ),
    )


def _terminal_tile_basic_pack() -> PersonaVisualStarterPack:
    """Create the reviewed Terminal Tile low-complexity starter."""
    starter_id = "terminal-tile-basic"
    return PersonaVisualStarterPack(
        id=starter_id,
        title="Terminal Tile Basic",
        description=(
            "Bundled art-ready basic Buddy default: compact terminal-window "
            "tile helper with approved 3x4 source-sheet frames."
        ),
        renderer_type="sprite_frames",
        manifest=_success_reaction_art_manifest("Single celebratory pose derived from the Terminal Tile source sheet."),
        assets=tuple(
            _resource_png_asset(starter_id, key, filename, role=role)
            for key, filename, role in _TERMINAL_TILE_FRAME_FILES
        ),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:art-ready",
            "tier:basic",
            "terminal",
            "cli",
            "object-buddy",
        ),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=(
                "Friendly terminal-window tile buddy with a charcoal rounded-square "
                "body, subtle title bar, mint cursor-face accents, tiny arms, and two short legs."
            ),
            neutral_anchor=(
                "Neutral anchor: front-facing charcoal tile with a subtle "
                "top bar, mint cursor-like face, relaxed arms, and two short legs."
            ),
            static_sheet=(
                "Recreate state loops from the neutral terminal tile by changing "
                "cursor face expression, arm gesture, tiny bounce, and mint accent marks "
                "while preserving the same tile body and legs."
            ),
            animation_outputs=("required_state_loops",),
        ),
    )


def _migu_marker_basic_pack() -> PersonaVisualStarterPack:
    """Create the reviewed Migu Marker low-complexity starter."""
    starter_id = "migu-marker-basic"
    return PersonaVisualStarterPack(
        id=starter_id,
        title="Migu Marker Basic",
        description=(
            "Bundled art-ready basic Buddy default: playful marker-line Migu "
            "helper with approved 3x4 source-sheet frames."
        ),
        renderer_type="sprite_frames",
        manifest=_success_reaction_art_manifest("Single celebratory pose derived from the Migu Marker source sheet."),
        assets=tuple(
            _resource_png_asset(starter_id, key, filename, role=role)
            for key, filename, role in _MIGU_MARKER_FRAME_FILES
        ),
        tags=(
            "starter",
            "sprite_frames",
            "catalog:art-ready",
            "tier:basic",
            "user-art",
            "marker",
        ),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=(
                "Playful marker-line Migu with tiny gray body, cream oval face, "
                "cyan twin tails, magenta ties, headset mic, split shirt line, and sketchy charm."
            ),
            neutral_anchor=(
                "Neutral anchor: front-facing chibi pose with cream oval face, gray body, "
                "black marker limbs, headset mic, split shirt line, and wide twin-tail silhouette."
            ),
            static_sheet=(
                "Recreate state loops from neutral by preserving rough asymmetry; hair bob, "
                "mouth changes, headset position, and teal marks carry animation."
            ),
            animation_outputs=("required_state_loops",),
        ),
    )


def _pixel_migu_pack() -> PersonaVisualStarterPack:
    """Load the user-supplied pixel-art Buddy without changing the default."""
    root = resources.files("tldw_Server_API.app.core.Persona").joinpath("assets", "starter_packs", "pixel-migu")
    manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
    asset_keys = sorted(
        {frame["asset_id"] for animation in manifest["animations"].values() for frame in animation["frames"]}
    )
    return PersonaVisualStarterPack(
        id="pixel-migu",
        title="pixel-migu",
        description="Turquoise pixel-art robot companion with twin tails, state loops and expression poses.",
        renderer_type="sprite_frames",
        manifest=manifest,
        assets=tuple(
            _resource_png_asset("pixel-migu", key, f"{key}.png", role="preview" if key == "pose-neutral" else "frame")
            for key in asset_keys
        ),
        tags=("starter", "sprite_frames", "catalog:art-ready", "tier:basic", "pixel-art", "user-art"),
        license_label="LicenseRef-User-Supplied",
        production_status="art_ready",
        animation_coverage_notes=("Reviewed user-supplied art: twelve four-frame sequences and sixteen static poses.",),
        production_recipe=_production_recipe(
            identity_brief="Turquoise pixel-art robot with rectangular face, black outlines and long twin tails.",
            neutral_anchor="Use the supplied neutral pose; retain the pixel grid, silhouette and turquoise palette.",
            static_sheet="Sixteen supplied expression poses complement the timed state sequences.",
            animation_outputs=("required_state_loops",),
        ),
    )


def _basic_pack(
    *,
    starter_id: str,
    title: str,
    description: str,
    style: str,
    tags: tuple[str, ...],
    identity_brief: str,
    neutral_anchor: str,
    static_sheet: str,
) -> PersonaVisualStarterPack:
    """Create a reviewed low-complexity starter with required-state loops."""
    state_assets = tuple(
        _basic_art_asset(
            starter_id,
            style,
            f"{state}-{frame_index}",
            state=state,
            frame_index=frame_index,
        )
        for state in _REQUIRED_STATE_IDS
        for frame_index in (1, 2)
    )
    assets = (
        _basic_art_asset(
            starter_id,
            style,
            "neutral-anchor",
            state="idle",
            frame_index=1,
            role="still_pose",
        ),
        _basic_art_asset(
            starter_id,
            style,
            "preview",
            state="idle",
            frame_index=1,
            role="preview",
        ),
        *state_assets,
    )
    return PersonaVisualStarterPack(
        id=starter_id,
        title=title,
        description=f"Bundled art-ready basic Buddy default: {description}",
        renderer_type="sprite_frames",
        manifest=_basic_art_manifest(),
        assets=assets,
        tags=("starter", "sprite_frames", "catalog:art-ready", "tier:basic", *tags),
        complexity_tier="basic",
        production_status="art_ready",
        neutral_anchor_required=True,
        expected_asset_groups=_BASIC_EXPECTED_ASSET_GROUPS,
        animation_coverage_notes=_BASIC_ANIMATION_NOTES,
        production_recipe=_production_recipe(
            identity_brief=identity_brief,
            neutral_anchor=neutral_anchor,
            static_sheet=static_sheet,
            animation_outputs=("required_state_loops",),
        ),
    )


def _authored_sprite_pack(
    *,
    starter_id: str,
    title: str,
    identity: str,
    tags: tuple[str, ...],
    complexity_tier: str = "intermediate",
    custom_states: dict[str, dict[str, Any]] | None = None,
    authored_triggers: list[dict[str, Any]] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> PersonaVisualStarterPack:
    """Bind reviewed package artwork to the established starter states."""
    uses_atlas = complexity_tier == "intricate"
    states: dict[str, dict[str, str]] = {}
    animations: dict[str, Any] = {}
    assets: list[PersonaVisualStarterAsset] = []
    if uses_atlas:
        assets.append(_resource_png_asset(starter_id, "atlas", "atlas.png", role="sprite_sheet"))

    durations = {
        "idle": (1500, 150),
        "listening": (360, 360),
        "thinking": (480, 480),
        "speaking": (140, 140),
        "error": (430, 430),
    }
    for row, state in enumerate((*_REQUIRED_STATE_IDS, *(custom_states or {}))):
        key = state.replace(".", "-")
        animation_id = f"{key}-loop"
        states[state] = {"animation_id": animation_id}
        frames: list[dict[str, Any]] = []
        for index, duration in enumerate(durations.get(state, (380, 420))):
            if uses_atlas:
                frames.append(
                    {
                        "asset_id": "atlas",
                        "duration_ms": duration,
                        "region": {"x": index * 128, "y": row * 128, "width": 128, "height": 128},
                    }
                )
            else:
                asset_key = f"{key}-{index + 1}"
                assets.append(_resource_png_asset(starter_id, asset_key, f"{asset_key}.png", role="frame"))
                frames.append({"asset_id": asset_key, "duration_ms": duration})
        animations[animation_id] = {
            "frames": frames,
            **({"preview_frame": 0} if uses_atlas else {"preview_asset_id": frames[0]["asset_id"]}),
        }

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": states,
        "animations": animations,
        "fallbacks": {
            **{state: targets.copy() for state, targets in _BUILTIN_FALLBACKS.items()},
            **(fallbacks or {}),
        },
        ARTWORK_MANIFEST_KEY: {
            "version": 1,
            "creator": "tldw-project",
            "license": _AUTHORED_ARTWORK_LICENSE,
            "source_url": "https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/Buddy_Starter_Artwork.md",
            "notices": "Original AI-assisted Buddy artwork prepared by tldw-project.",
        },
    }
    if custom_states:
        manifest["state_catalog"] = custom_states
    if authored_triggers:
        manifest["authored_triggers"] = authored_triggers

    outputs = ["required_state_loops"]
    if uses_atlas:
        outputs.extend(("animation_strips", "animation_atlas"))
    # Recipes can generate custom states even when the starter ships without any.
    outputs.append("custom_state_variants")
    return PersonaVisualStarterPack(
        id=starter_id,
        title=title,
        description=identity,
        renderer_type="sprite_frames",
        manifest=manifest,
        assets=tuple(assets),
        tags=("starter", "sprite_frames", "catalog:art-ready", f"tier:{complexity_tier}", *tags),
        license_label=_AUTHORED_ARTWORK_LICENSE,
        complexity_tier=complexity_tier,
        production_status="art_ready",
        expected_asset_groups=(
            _INTRICATE_EXPECTED_ASSET_GROUPS if uses_atlas else _INTERMEDIATE_EXPECTED_ASSET_GROUPS
        ),
        animation_coverage_notes=("Reviewed two-pose loops aligned to a fixed neutral anchor.",),
        production_recipe=_production_recipe(
            identity_brief=identity,
            neutral_anchor="The neutral pose fixes the position and scale on the 128 × 128 canvas.",
            static_sheet="Separate static mouth and reaction pose sheets provide the source for the timed loops.",
            animation_outputs=tuple(outputs),
        ),
    )


DEFAULT_PERSONA_VISUAL_STARTER_PACKS: tuple[PersonaVisualStarterPack, ...] = (
    _search_lens_basic_pack(),
    _index_card_basic_pack(),
    _archive_cube_basic_pack(),
    _paperclip_basic_pack(),
    _terminal_tile_basic_pack(),
    _migu_marker_basic_pack(),
    _pixel_migu_pack(),
    _authored_sprite_pack(
        starter_id="study-desk-intermediate",
        title="Study Desk Intermediate",
        identity="Bespectacled study companion with an open book on a walnut desk.",
        tags=("study", "desk"),
    ),
    _authored_sprite_pack(
        starter_id="tool-helper-intermediate",
        title="Tool Helper Intermediate",
        identity="Teal utility toolbox with two rubber feet and a notes-search magnifier.",
        tags=("tool", "variant"),
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
    _authored_sprite_pack(
        starter_id="object-creature-intermediate",
        title="Object Creature Intermediate",
        identity="Coral enamel teapot with a cream lid and a small celebration pose.",
        tags=("object", "creature"),
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
    _authored_sprite_pack(
        starter_id="lofi-study-intricate",
        title="Lo-fi Study Intricate",
        identity="Amber fox student in violet headphones and an indigo hoodie.",
        tags=("atlas", "lofi", "study"),
        complexity_tier="intricate",
        custom_states={
            "tool.research": {
                "label": "Researching",
                "kind": "tool_variant",
                "description": "Used while research or retrieval tools are running.",
                "tags": ["tool", "search"],
            }
        },
        authored_triggers=[
            {
                "id": "research-tool-category",
                "source": "tool_category",
                "match": "search",
                "state": "tool.research",
                "duration_ms": 2400,
                "priority": 70,
            }
        ],
        fallbacks={"tool.research": ["tool_running", "thinking", "idle"]},
    ),
    _authored_sprite_pack(
        starter_id="action-guide-intricate",
        title="Action Guide Intricate",
        identity="Moss-green lizard guide in an orange neckerchief with anticipation and success poses.",
        tags=("atlas", "action", "reaction"),
        complexity_tier="intricate",
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
    _authored_sprite_pack(
        starter_id="elaborate-persona-intricate",
        title="Elaborate Persona Intricate",
        identity="Cloaked stargazer with a gold star clasp, focused expression and floating media cards.",
        tags=("atlas", "fantasy", "sci-fi"),
        complexity_tier="intricate",
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
    "PersonaVisualStarterProductionRecipe",
]
