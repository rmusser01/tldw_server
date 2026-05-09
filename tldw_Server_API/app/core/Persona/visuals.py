from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


VISUAL_STATE_IDS = {
    "idle",
    "wake_armed",
    "listening",
    "thinking",
    "speaking",
    "tool_running",
    "approval_needed",
    "error",
    "offline",
}
REQUIRED_VISUAL_STATES = {"idle", "listening", "thinking", "speaking", "error"}
SUPPORTED_RENDERER_TYPES = {"sprite_frames"}
SUPPORTED_TRIGGER_SOURCES = {"live_state", "tool_category", "mcp_runtime"}

MAX_FRAMES_PER_ANIMATION = 240
MIN_FRAME_DURATION_MS = 16
MAX_FRAME_DURATION_MS = 30_000
MIN_TRIGGER_DURATION_MS = 100
MAX_TRIGGER_DURATION_MS = 30_000


class PersonaVisualManifestError(ValueError):
    """Raised when a persona visual-pack manifest is invalid."""


@dataclass(frozen=True)
class PersonaVisualManifestValidation:
    manifest: dict[str, Any]
    resolved_required_states: dict[str, str]


def validate_visual_manifest(
    manifest: dict[str, Any],
    *,
    available_asset_ids: set[str],
    require_activatable: bool,
    available_asset_dimensions: dict[str, tuple[int, int]] | None = None,
) -> PersonaVisualManifestValidation:
    normalized = _normalize_manifest_shape(manifest)
    dimensions = available_asset_dimensions or {}

    for animation_id, animation in normalized["animations"].items():
        _validate_animation(
            animation_id,
            animation,
            available_asset_ids=available_asset_ids,
            available_asset_dimensions=dimensions,
        )

    _validate_state_references(normalized)
    _validate_fallbacks(normalized.get("fallbacks", {}))
    _validate_authored_triggers(normalized.get("authored_triggers", []))

    resolved_required = {
        state: _resolve_state(state, normalized)
        for state in REQUIRED_VISUAL_STATES
    }
    if require_activatable:
        missing = sorted(
            state for state, animation_id in resolved_required.items() if not animation_id
        )
        if missing:
            raise PersonaVisualManifestError(
                "Required visual states do not resolve: " + ", ".join(missing)
            )

    return PersonaVisualManifestValidation(
        manifest=normalized,
        resolved_required_states={
            state: animation_id
            for state, animation_id in resolved_required.items()
            if animation_id
        },
    )


def _normalize_manifest_shape(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise PersonaVisualManifestError("Manifest must be an object")

    normalized = deepcopy(manifest)
    if normalized.get("manifest_version") != 1:
        raise PersonaVisualManifestError("manifest_version must be 1")
    if normalized.get("renderer_type") not in SUPPORTED_RENDERER_TYPES:
        raise PersonaVisualManifestError("renderer_type must be sprite_frames")

    states = normalized.setdefault("states", {})
    animations = normalized.setdefault("animations", {})
    fallbacks = normalized.setdefault("fallbacks", {})
    authored_triggers = normalized.setdefault("authored_triggers", [])

    if not isinstance(states, dict):
        raise PersonaVisualManifestError("states must be an object")
    if not isinstance(animations, dict):
        raise PersonaVisualManifestError("animations must be an object")
    if not isinstance(fallbacks, dict):
        raise PersonaVisualManifestError("fallbacks must be an object")
    if not isinstance(authored_triggers, list):
        raise PersonaVisualManifestError("authored_triggers must be a list")

    for animation_id, animation in animations.items():
        if not isinstance(animation_id, str) or not animation_id:
            raise PersonaVisualManifestError("Animation ids must be non-empty strings")
        if not isinstance(animation, dict):
            raise PersonaVisualManifestError(f"Animation {animation_id} must be an object")
        animation["frames"] = _normalize_animation_frames(animation_id, animation)

    return normalized


def _normalize_animation_frames(
    animation_id: str,
    animation: dict[str, Any],
) -> list[dict[str, Any]]:
    frames = animation.get("frames")
    asset_ids = animation.get("asset_ids")
    if frames is None:
        if asset_ids is None:
            raise PersonaVisualManifestError(
                f"Animation {animation_id} must define frames or asset_ids"
            )
        if not isinstance(asset_ids, list):
            raise PersonaVisualManifestError(
                f"Animation {animation_id} asset_ids must be a list"
            )
        frames = [{"asset_id": asset_id} for asset_id in asset_ids]
    elif not isinstance(frames, list):
        raise PersonaVisualManifestError(f"Animation {animation_id} frames must be a list")

    normalized_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise PersonaVisualManifestError(
                f"Animation {animation_id} frame {index} must be an object"
            )
        normalized_frames.append(deepcopy(frame))
    return normalized_frames


def _validate_animation(
    animation_id: str,
    animation: dict[str, Any],
    *,
    available_asset_ids: set[str],
    available_asset_dimensions: dict[str, tuple[int, int]],
) -> None:
    frame_rate = animation.get("frame_rate", 1)
    if not isinstance(frame_rate, (int, float)) or not 1 <= frame_rate <= 60:
        raise PersonaVisualManifestError(
            f"Animation {animation_id} frame_rate must be between 1 and 60"
        )

    alignment = animation.get("alignment")
    if alignment is not None:
        _validate_alignment(animation_id, alignment)

    frames = animation["frames"]
    if not frames:
        raise PersonaVisualManifestError(f"Animation {animation_id} must contain frames")
    if len(frames) > MAX_FRAMES_PER_ANIMATION:
        raise PersonaVisualManifestError(
            f"Animation {animation_id} may reference at most {MAX_FRAMES_PER_ANIMATION} frames"
        )

    frame_asset_ids: set[str] = set()
    for index, frame in enumerate(frames):
        asset_id = frame.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise PersonaVisualManifestError(
                f"Animation {animation_id} frame {index} asset_id is required"
            )
        if asset_id not in available_asset_ids:
            raise PersonaVisualManifestError(
                f"Animation {animation_id} references unknown asset_id {asset_id}"
            )
        frame_asset_ids.add(asset_id)
        _validate_frame_duration(animation_id, index, frame)
        _validate_frame_region(
            animation_id,
            index,
            frame,
            asset_id=asset_id,
            available_asset_dimensions=available_asset_dimensions,
        )

    preview_frame = animation.get("preview_frame")
    if preview_frame is not None:
        if not isinstance(preview_frame, int) or not 0 <= preview_frame < len(frames):
            raise PersonaVisualManifestError(
                f"Animation {animation_id} preview_frame must be a valid frame index"
            )

    preview_asset_id = animation.get("preview_asset_id")
    if preview_asset_id is not None:
        if preview_asset_id not in frame_asset_ids:
            raise PersonaVisualManifestError(
                f"Animation {animation_id} preview_asset_id must reference an animation asset"
            )


def _validate_alignment(animation_id: str, alignment: Any) -> None:
    if not isinstance(alignment, dict):
        raise PersonaVisualManifestError(f"Animation {animation_id} alignment must be an object")
    for axis in ("x", "y"):
        value = alignment.get(axis)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise PersonaVisualManifestError(
                f"Animation {animation_id} alignment.{axis} must be between 0 and 1"
            )


def _validate_frame_duration(
    animation_id: str,
    index: int,
    frame: dict[str, Any],
) -> None:
    duration_ms = frame.get("duration_ms")
    if duration_ms is None:
        return
    if (
        not isinstance(duration_ms, int)
        or not MIN_FRAME_DURATION_MS <= duration_ms <= MAX_FRAME_DURATION_MS
    ):
        raise PersonaVisualManifestError(
            f"Animation {animation_id} frame {index} duration_ms must be between "
            f"{MIN_FRAME_DURATION_MS} and {MAX_FRAME_DURATION_MS}"
        )


def _validate_frame_region(
    animation_id: str,
    index: int,
    frame: dict[str, Any],
    *,
    asset_id: str,
    available_asset_dimensions: dict[str, tuple[int, int]],
) -> None:
    region = frame.get("region")
    if region is None:
        return
    if not isinstance(region, dict):
        raise PersonaVisualManifestError(
            f"Animation {animation_id} frame {index} region must be an object"
        )

    for key in ("x", "y", "width", "height"):
        value = region.get(key)
        if not isinstance(value, int):
            raise PersonaVisualManifestError(
                f"Animation {animation_id} frame {index} region.{key} must be an integer"
            )

    if region["x"] < 0 or region["y"] < 0 or region["width"] <= 0 or region["height"] <= 0:
        raise PersonaVisualManifestError(
            f"Animation {animation_id} frame {index} region has invalid bounds"
        )

    dimensions = available_asset_dimensions.get(asset_id)
    if not dimensions:
        return
    width, height = dimensions
    if region["x"] + region["width"] > width or region["y"] + region["height"] > height:
        raise PersonaVisualManifestError(
            f"Animation {animation_id} frame {index} region exceeds asset bounds"
        )


def _validate_state_references(manifest: dict[str, Any]) -> None:
    animations = manifest["animations"]
    for state, mapping in manifest["states"].items():
        if state not in VISUAL_STATE_IDS:
            raise PersonaVisualManifestError(f"Unknown visual state {state}")
        if not isinstance(mapping, dict):
            raise PersonaVisualManifestError(f"State {state} mapping must be an object")
        animation_id = mapping.get("animation_id")
        if not isinstance(animation_id, str) or not animation_id:
            raise PersonaVisualManifestError(f"State {state} animation_id is required")
        if animation_id not in animations:
            raise PersonaVisualManifestError(
                f"State {state} references unknown animation_id {animation_id}"
            )


def _validate_fallbacks(fallbacks: dict[str, Any]) -> None:
    for state, fallback_chain in fallbacks.items():
        if state not in VISUAL_STATE_IDS:
            raise PersonaVisualManifestError(f"Unknown fallback state {state}")
        if not isinstance(fallback_chain, list):
            raise PersonaVisualManifestError(f"Fallback {state} must be a list")
        for fallback_state in fallback_chain:
            if fallback_state not in VISUAL_STATE_IDS:
                raise PersonaVisualManifestError(
                    f"Fallback {state} references unknown state {fallback_state}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state: str, path: tuple[str, ...]) -> None:
        if state in visiting:
            raise PersonaVisualManifestError(
                "Fallback cycle detected: " + " -> ".join((*path, state))
            )
        if state in visited:
            return
        visiting.add(state)
        for next_state in fallbacks.get(state, []):
            visit(next_state, (*path, state))
        visiting.remove(state)
        visited.add(state)

    for state in fallbacks:
        visit(state, ())


def _validate_authored_triggers(triggers: list[Any]) -> None:
    for index, trigger in enumerate(triggers):
        if not isinstance(trigger, dict):
            raise PersonaVisualManifestError(f"authored_triggers[{index}] must be an object")
        trigger_id = trigger.get("id")
        if not isinstance(trigger_id, str) or not trigger_id:
            raise PersonaVisualManifestError(f"authored_triggers[{index}].id is required")
        source = trigger.get("source")
        if source not in SUPPORTED_TRIGGER_SOURCES:
            raise PersonaVisualManifestError(
                f"authored_triggers[{index}].source must be one of "
                f"{sorted(SUPPORTED_TRIGGER_SOURCES)}"
            )
        match = trigger.get("match")
        if not isinstance(match, str) or not match:
            raise PersonaVisualManifestError(f"authored_triggers[{index}].match is required")
        state = trigger.get("state")
        if state not in VISUAL_STATE_IDS:
            raise PersonaVisualManifestError(
                f"authored_triggers[{index}].state must be a known visual state"
            )
        duration_ms = trigger.get("duration_ms")
        if (
            not isinstance(duration_ms, int)
            or not MIN_TRIGGER_DURATION_MS <= duration_ms <= MAX_TRIGGER_DURATION_MS
        ):
            raise PersonaVisualManifestError(
                f"authored_triggers[{index}].duration_ms must be between "
                f"{MIN_TRIGGER_DURATION_MS} and {MAX_TRIGGER_DURATION_MS}"
            )
        priority = trigger.get("priority")
        if not isinstance(priority, int) or not 0 <= priority <= 100:
            raise PersonaVisualManifestError(
                f"authored_triggers[{index}].priority must be between 0 and 100"
            )


def _resolve_state(
    state: str,
    manifest: dict[str, Any],
    *,
    seen: set[str] | None = None,
) -> str:
    seen = seen or set()
    if state in seen:
        return ""
    seen.add(state)

    state_mapping = manifest["states"].get(state)
    if state_mapping:
        animation_id = state_mapping.get("animation_id")
        if animation_id in manifest["animations"]:
            return animation_id

    for fallback_state in manifest.get("fallbacks", {}).get(state, []):
        animation_id = _resolve_state(fallback_state, manifest, seen=seen)
        if animation_id:
            return animation_id
    return ""


__all__ = [
    "MAX_FRAMES_PER_ANIMATION",
    "REQUIRED_VISUAL_STATES",
    "SUPPORTED_RENDERER_TYPES",
    "VISUAL_STATE_IDS",
    "PersonaVisualManifestError",
    "PersonaVisualManifestValidation",
    "validate_visual_manifest",
]
