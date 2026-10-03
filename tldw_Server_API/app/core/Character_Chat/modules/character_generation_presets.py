"""
Character generation preset helpers.

Resolves per-character generation metadata and returns normalized
sampling settings for character chat completions.
"""

from __future__ import annotations

import json
import re
from math import isfinite
from typing import Any


def supports_saved_character_extensions(extensions: Any) -> bool:
    """Accept only prompt presets and recognized generation metadata in saved context.

    Generation values are carried for provenance; normalized snapshot sampling
    remains authoritative and is never recomputed from this metadata.
    """
    if not isinstance(extensions, dict) or set(extensions) - {"prompt_preset", "promptPreset", "tldw"}:
        return False
    tldw = extensions.get("tldw", {})
    if not isinstance(tldw, dict) or set(tldw) - {"prompt_preset", "promptPreset", "generation"}:
        return False
    if "generation" not in tldw:
        return True
    generation = tldw["generation"]
    return isinstance(generation, dict) and not set(generation) - {
        "temperature",
        "top_p",
        "topP",
        "repetition_penalty",
        "repetitionPenalty",
        "stop",
        "stop_strings",
        "stopStrings",
        "stop_sequences",
        "stopSequences",
    }


def _coerce_float(value: Any, *, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    if parsed < minimum or parsed > maximum:
        return None
    return parsed


def _normalize_stop(value: Any) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            value = parsed if isinstance(parsed, list) else text
        except json.JSONDecodeError:
            value = text

    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"\r?\n|;", value) if p.strip()]
        return parts or None

    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    parts.append(cleaned)
        return parts or None

    return None


def _parse_extensions(character: dict[str, Any]) -> dict[str, Any]:
    extensions: Any = character.get("extensions")
    if isinstance(extensions, str):
        try:
            extensions = json.loads(extensions)
        except json.JSONDecodeError:
            return {}
    return extensions if isinstance(extensions, dict) else {}


def _resolve_from_containers(
    containers: list[dict[str, Any]],
    keys: tuple[str, ...],
    parser,
) -> Any:
    for container in containers:
        for key in keys:
            if key not in container:
                continue
            parsed = parser(container.get(key))
            if parsed is not None:
                return parsed
    return None


def resolve_character_generation_settings(
    character: dict[str, Any],
) -> dict[str, Any]:
    """Resolve normalized generation settings from character metadata.

    Resolution order:
    1) `extensions.tldw.generation`
    2) `extensions` top-level keys
    3) direct character keys

    Returned keys (when valid):
    - `temperature` in [0.0, 2.0]
    - `top_p` in [0.0, 1.0]
    - `repetition_penalty` in [0.0, 3.0]
    - `stop` as list[str]
    """
    if not isinstance(character, dict):
        return {}

    extensions = _parse_extensions(character)
    tldw = extensions.get("tldw") if isinstance(extensions, dict) else None
    generation = tldw.get("generation") if isinstance(tldw, dict) else None

    containers: list[dict[str, Any]] = []
    if isinstance(generation, dict):
        containers.append(generation)
    if isinstance(extensions, dict):
        containers.append(extensions)
    containers.append(character)

    resolved: dict[str, Any] = {}

    temperature = _resolve_from_containers(
        containers,
        ("temperature",),
        lambda value: _coerce_float(value, minimum=0.0, maximum=2.0),
    )
    if temperature is not None:
        resolved["temperature"] = temperature

    top_p = _resolve_from_containers(
        containers,
        ("top_p", "topP"),
        lambda value: _coerce_float(value, minimum=0.0, maximum=1.0),
    )
    if top_p is not None:
        resolved["top_p"] = top_p

    repetition_penalty = _resolve_from_containers(
        containers,
        ("repetition_penalty", "repetitionPenalty"),
        lambda value: _coerce_float(value, minimum=0.0, maximum=3.0),
    )
    if repetition_penalty is not None:
        resolved["repetition_penalty"] = repetition_penalty

    stop = _resolve_from_containers(
        containers,
        ("stop", "stop_strings", "stopStrings", "stop_sequences", "stopSequences"),
        _normalize_stop,
    )
    if stop is not None:
        resolved["stop"] = stop

    return resolved
