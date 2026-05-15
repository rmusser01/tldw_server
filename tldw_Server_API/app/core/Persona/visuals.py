from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    get_persona_visual_renderer_capability,
)


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
SUPPORTED_TRIGGER_SOURCES = {"live_state", "tool_category", "mcp_runtime", "tool_name"}
SUPPORTED_STATE_CATALOG_KINDS = {
    "tool_variant",
    "reaction",
    "live_variant",
    "mcp_runtime",
    "mood",
    "pack_private",
}

MAX_FRAMES_PER_ANIMATION = 240
MAX_CUSTOM_VISUAL_STATES = 256
MAX_AUTHORED_TRIGGERS = 512
MAX_FALLBACK_DEPTH = 8
MIN_FRAME_DURATION_MS = 16
MAX_FRAME_DURATION_MS = 30_000
MIN_TRIGGER_DURATION_MS = 100
MAX_TRIGGER_DURATION_MS = 30_000
MAX_RENDERER_TYPE_ERROR_LENGTH = 100
MAX_STATE_CATALOG_LABEL_LENGTH = 80
MAX_STATE_CATALOG_DESCRIPTION_LENGTH = 280
MAX_STATE_CATALOG_TAGS = 16
MAX_STATE_CATALOG_TAG_LENGTH = 32
CUSTOM_VISUAL_STATE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,95}$")
UNSAFE_CUSTOM_STATE_MARKERS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
)
UNSAFE_CUSTOM_STATE_PREFIXES = (
    "env:",
    "file:",
    "ftp:",
    "http:",
    "https:",
    "proc:",
    "ssh:",
)


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
    normalized = _normalize_manifest_shape(
        manifest,
        require_activatable=require_activatable,
    )
    dimensions = available_asset_dimensions or {}

    for animation_id, animation in normalized["animations"].items():
        _validate_animation(
            animation_id,
            animation,
            available_asset_ids=available_asset_ids,
            available_asset_dimensions=dimensions,
        )

    allowed_states = _allowed_visual_state_ids(normalized)
    _validate_state_references(normalized, allowed_states=allowed_states)
    _validate_fallbacks(normalized.get("fallbacks", {}), allowed_states=allowed_states)
    _validate_authored_triggers(
        normalized.get("authored_triggers", []),
        allowed_states=allowed_states,
    )

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


def _normalize_manifest_shape(
    manifest: dict[str, Any],
    *,
    require_activatable: bool,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise PersonaVisualManifestError("Manifest must be an object")

    normalized = deepcopy(manifest)
    renderer_type = normalized.get("renderer_type")
    renderer_type_for_error = _format_renderer_type_for_error(renderer_type)
    capability = get_persona_visual_renderer_capability(str(renderer_type or ""))
    if capability is None or not capability.can_validate:
        raise PersonaVisualManifestError(
            f"unsupported renderer_type: {renderer_type_for_error}"
        )
    if require_activatable and not capability.can_activate:
        raise PersonaVisualManifestError(
            f"unsupported renderer_type for activation: {renderer_type_for_error}"
        )
    if normalized.get("manifest_version") not in capability.manifest_versions:
        raise PersonaVisualManifestError(
            "manifest_version must be one of "
            f"{', '.join(str(version) for version in capability.manifest_versions)}"
        )

    states = normalized.setdefault("states", {})
    animations = normalized.setdefault("animations", {})
    fallbacks = normalized.setdefault("fallbacks", {})
    authored_triggers = normalized.setdefault("authored_triggers", [])
    state_catalog = normalized.setdefault("state_catalog", {})

    if not isinstance(states, dict):
        raise PersonaVisualManifestError("states must be an object")
    if not isinstance(animations, dict):
        raise PersonaVisualManifestError("animations must be an object")
    if not isinstance(fallbacks, dict):
        raise PersonaVisualManifestError("fallbacks must be an object")
    if not isinstance(authored_triggers, list):
        raise PersonaVisualManifestError("authored_triggers must be a list")
    if not isinstance(state_catalog, dict):
        raise PersonaVisualManifestError("state_catalog must be an object")

    for animation_id, animation in animations.items():
        if not isinstance(animation_id, str) or not animation_id:
            raise PersonaVisualManifestError("Animation ids must be non-empty strings")
        if not isinstance(animation, dict):
            raise PersonaVisualManifestError(f"Animation {animation_id} must be an object")
        animation["frames"] = _normalize_animation_frames(animation_id, animation)

    _validate_state_catalog(state_catalog)

    return normalized


def _format_renderer_type_for_error(renderer_type: Any) -> str:
    value = str(renderer_type or "")
    value = (
        value.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    if len(value) > MAX_RENDERER_TYPE_ERROR_LENGTH:
        return value[:MAX_RENDERER_TYPE_ERROR_LENGTH] + "..."
    return value or "<empty>"


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


def _validate_state_catalog(state_catalog: dict[str, Any]) -> None:
    if len(state_catalog) > MAX_CUSTOM_VISUAL_STATES:
        raise PersonaVisualManifestError(
            f"state_catalog may define at most {MAX_CUSTOM_VISUAL_STATES} custom states"
        )

    for state_id, entry in state_catalog.items():
        if not isinstance(state_id, str) or not _is_safe_custom_state_id(state_id):
            raise PersonaVisualManifestError(
                "state_catalog custom state ids must match "
                f"{CUSTOM_VISUAL_STATE_ID_PATTERN.pattern}"
            )
        if state_id in VISUAL_STATE_IDS:
            raise PersonaVisualManifestError(
                f"state_catalog custom state {state_id} is reserved"
            )
        if not isinstance(entry, dict):
            raise PersonaVisualManifestError(
                f"state_catalog[{state_id}] must be an object"
            )

        label = entry.get("label")
        if (
            not isinstance(label, str)
            or not label.strip()
            or len(label) > MAX_STATE_CATALOG_LABEL_LENGTH
            or _contains_control_character(label)
        ):
            raise PersonaVisualManifestError(
                f"state_catalog[{state_id}].label must be a non-empty string up to "
                f"{MAX_STATE_CATALOG_LABEL_LENGTH} characters"
            )

        kind = entry.get("kind")
        if kind not in SUPPORTED_STATE_CATALOG_KINDS:
            raise PersonaVisualManifestError(
                f"state_catalog[{state_id}].kind must be one of "
                f"{sorted(SUPPORTED_STATE_CATALOG_KINDS)}"
            )

        description = entry.get("description")
        if description is not None and (
            not isinstance(description, str)
            or len(description) > MAX_STATE_CATALOG_DESCRIPTION_LENGTH
            or _contains_control_character(description)
        ):
            raise PersonaVisualManifestError(
                f"state_catalog[{state_id}].description must be a string up to "
                f"{MAX_STATE_CATALOG_DESCRIPTION_LENGTH} characters"
            )

        tags = entry.get("tags")
        if tags is not None:
            _validate_state_catalog_tags(state_id, tags)


def _validate_state_catalog_tags(state_id: str, tags: Any) -> None:
    if not isinstance(tags, list) or len(tags) > MAX_STATE_CATALOG_TAGS:
        raise PersonaVisualManifestError(
            f"state_catalog[{state_id}].tags must be a list of at most "
            f"{MAX_STATE_CATALOG_TAGS} strings"
        )
    for index, tag in enumerate(tags):
        if (
            not isinstance(tag, str)
            or not tag.strip()
            or len(tag) > MAX_STATE_CATALOG_TAG_LENGTH
            or _contains_control_character(tag)
        ):
            raise PersonaVisualManifestError(
                f"state_catalog[{state_id}].tags[{index}] must be a non-empty string "
                f"up to {MAX_STATE_CATALOG_TAG_LENGTH} characters"
            )


def _allowed_visual_state_ids(manifest: dict[str, Any]) -> set[str]:
    return set(VISUAL_STATE_IDS) | set(manifest.get("state_catalog", {}))


def _is_safe_custom_state_id(state_id: str) -> bool:
    if not CUSTOM_VISUAL_STATE_ID_PATTERN.fullmatch(state_id):
        return False
    lowered = state_id.lower()
    if lowered.startswith(UNSAFE_CUSTOM_STATE_PREFIXES):
        return False
    compact = re.sub(r"[._:-]+", "_", lowered)
    return not any(marker in compact for marker in UNSAFE_CUSTOM_STATE_MARKERS)


def _contains_control_character(value: str) -> bool:
    return any(character in value for character in ("\r", "\n", "\t"))


def _validate_state_references(
    manifest: dict[str, Any],
    *,
    allowed_states: set[str],
) -> None:
    animations = manifest["animations"]
    for state, mapping in manifest["states"].items():
        if state not in allowed_states:
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


def _validate_fallbacks(
    fallbacks: dict[str, Any],
    *,
    allowed_states: set[str],
) -> None:
    for state, fallback_chain in fallbacks.items():
        if state not in allowed_states:
            raise PersonaVisualManifestError(f"Unknown fallback state {state}")
        if not isinstance(fallback_chain, list):
            raise PersonaVisualManifestError(f"Fallback {state} must be a list")
        for fallback_state in fallback_chain:
            if fallback_state not in allowed_states:
                raise PersonaVisualManifestError(
                    f"Fallback {state} references unknown state {fallback_state}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state: str, path: tuple[str, ...]) -> None:
        if len(path) >= MAX_FALLBACK_DEPTH:
            raise PersonaVisualManifestError(
                f"Fallback chain depth exceeds {MAX_FALLBACK_DEPTH}: "
                + " -> ".join((*path, state))
            )
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


def _validate_authored_triggers(
    triggers: list[Any],
    *,
    allowed_states: set[str],
) -> None:
    if len(triggers) > MAX_AUTHORED_TRIGGERS:
        raise PersonaVisualManifestError(
            f"authored_triggers may define at most {MAX_AUTHORED_TRIGGERS} triggers"
        )
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
        if state not in allowed_states:
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
    "VISUAL_STATE_IDS",
    "PersonaVisualManifestError",
    "PersonaVisualManifestValidation",
    "validate_visual_manifest",
]
