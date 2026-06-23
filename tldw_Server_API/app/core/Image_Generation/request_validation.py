"""Shared request validation helpers for image-generation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Image_Generation.config import (
    DEFAULT_INLINE_MAX_BYTES,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_WIDTH,
    get_image_generation_config,
)


@dataclass(frozen=True)
class ImageGenerationValidationIssue:
    code: str
    message: str
    path: str


def effective_inline_max_bytes(config: Any | None = None) -> int:
    """Return a positive byte cap for inline/remote image payloads."""

    if config is None:
        config = get_image_generation_config()
    raw = getattr(config, "inline_max_bytes", None)
    if raw is None:
        return DEFAULT_INLINE_MAX_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_INLINE_MAX_BYTES
    return value if value > 0 else DEFAULT_INLINE_MAX_BYTES


def allowed_extra_params_for_backend(backend: str, config: Any) -> set[str]:
    """Return configured passthrough allowlist keys for an image backend."""

    backend_name = str(backend or "").strip().lower()
    attr_by_backend = {
        "stable_diffusion_cpp": "sd_cpp_allowed_extra_params",
        "swarmui": "swarmui_allowed_extra_params",
        "openrouter": "openrouter_image_allowed_extra_params",
        "novita": "novita_image_allowed_extra_params",
        "together": "together_image_allowed_extra_params",
        "modelstudio": "modelstudio_image_allowed_extra_params",
    }
    attr = attr_by_backend.get(backend_name)
    if not attr:
        return set()
    return {str(item).strip() for item in getattr(config, attr, []) or [] if str(item).strip()}


def validate_image_generation_request(
    structured: dict[str, Any],
    *,
    config: Any | None = None,
) -> list[ImageGenerationValidationIssue]:
    """Validate shared image-generation bounds and backend passthrough controls."""

    if config is None:
        config = get_image_generation_config()

    issues: list[ImageGenerationValidationIssue] = []
    prompt = structured.get("prompt")
    max_prompt_length = _positive_int_attr(config, "max_prompt_length", DEFAULT_MAX_PROMPT_LENGTH)
    if isinstance(prompt, str) and len(prompt) > max_prompt_length:
        issues.append(_issue("prompt exceeds max length", "prompt"))

    width = structured.get("width")
    height = structured.get("height")
    max_width = _positive_int_attr(config, "max_width", DEFAULT_MAX_WIDTH)
    max_height = _positive_int_attr(config, "max_height", DEFAULT_MAX_HEIGHT)
    max_pixels = _positive_int_attr(config, "max_pixels", DEFAULT_MAX_PIXELS)

    width_ok = _validate_int_bound(issues, width, path="width", max_value=max_width)
    height_ok = _validate_int_bound(issues, height, path="height", max_value=max_height)
    if width_ok and height_ok and isinstance(width, int) and isinstance(height, int) and width * height > max_pixels:
        issues.append(_issue("image dimensions exceed max pixels", "width,height"))

    steps = structured.get("steps")
    max_steps = _positive_int_attr(config, "max_steps", DEFAULT_MAX_STEPS)
    _validate_int_bound(issues, steps, path="steps", max_value=max_steps)

    _validate_extra_params(structured, config, issues)
    return issues


def _issue(message: str, path: str) -> ImageGenerationValidationIssue:
    return ImageGenerationValidationIssue(
        code="image_params_invalid",
        message=message,
        path=path,
    )


def _positive_int_attr(config: Any, attr: str, default: int) -> int:
    try:
        value = int(getattr(config, attr, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _validate_int_bound(
    issues: list[ImageGenerationValidationIssue],
    value: Any,
    *,
    path: str,
    max_value: int,
) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(_issue(f"{path} must be an integer", path))
        return False
    if value <= 0 or value > max_value:
        issues.append(_issue(f"{path} out of range", path))
        return False
    return True


def _validate_extra_params(
    structured: dict[str, Any],
    config: Any,
    issues: list[ImageGenerationValidationIssue],
) -> None:
    extra_params = structured.get("extra_params") or {}
    if not extra_params:
        return
    if not isinstance(extra_params, dict):
        issues.append(_issue("extra_params must be an object", "extra_params"))
        return

    backend = str(structured.get("backend") or "").strip().lower()
    allowlist = allowed_extra_params_for_backend(backend, config)
    exempt_control_keys: set[str] = set()
    if backend == "modelstudio":
        exempt_control_keys.add("mode")

    keys_to_validate = [key for key in extra_params if key not in exempt_control_keys]
    if not keys_to_validate:
        return
    for key in keys_to_validate:
        if key not in allowlist:
            issues.append(_issue("extra_params key not allowlisted", f"extra_params.{key}"))

    if "cli_args" in extra_params and "cli_args" in allowlist:
        cli_args = extra_params.get("cli_args")
        if not isinstance(cli_args, (list, tuple)):
            issues.append(_issue("cli_args must be a list", "extra_params.cli_args"))
