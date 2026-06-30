"""Configuration helpers for image generation backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.config import get_config_section

DEFAULT_BACKEND = "stable_diffusion_cpp"
DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1024
DEFAULT_MAX_PIXELS = 1024 * 1024
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_PROMPT_LENGTH = 1000
DEFAULT_INLINE_MAX_BYTES = 4_000_000

DEFAULT_SD_CPP_STEPS = 25
DEFAULT_SD_CPP_CFG_SCALE = 7.5
DEFAULT_SD_CPP_SAMPLER = "euler_a"
DEFAULT_SD_CPP_DEVICE = "auto"
DEFAULT_SD_CPP_TIMEOUT_SECONDS = 120
DEFAULT_SWARMUI_BASE_URL = "http://127.0.0.1:7801"
DEFAULT_SWARMUI_TIMEOUT_SECONDS = 120
DEFAULT_OPENROUTER_IMAGE_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_IMAGE_MODEL = "openai/gpt-image-1"
DEFAULT_OPENROUTER_IMAGE_TIMEOUT_SECONDS = 120
DEFAULT_NOVITA_IMAGE_BASE_URL = "https://api.novita.ai"
DEFAULT_NOVITA_IMAGE_MODEL = "sd_xl_base_1.0.safetensors"
DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS = 180
DEFAULT_NOVITA_IMAGE_POLL_INTERVAL_SECONDS = 2
DEFAULT_TOGETHER_IMAGE_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell-Free"
DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS = 120
DEFAULT_MODELSTUDIO_IMAGE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_MODELSTUDIO_IMAGE_MODEL = "qwen-image"
DEFAULT_MODELSTUDIO_IMAGE_REGION = "sg"
DEFAULT_MODELSTUDIO_IMAGE_MODE = "auto"
DEFAULT_MODELSTUDIO_IMAGE_POLL_INTERVAL_SECONDS = 2
DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ImageGenerationConfig:
    default_backend: str | None
    enabled_backends: list[str]
    max_width: int
    max_height: int
    max_pixels: int
    max_steps: int
    max_prompt_length: int
    inline_max_bytes: int | None
    sd_cpp_diffusion_model_path: str | None
    sd_cpp_llm_path: str | None
    sd_cpp_binary_path: str | None
    sd_cpp_model_path: str | None
    sd_cpp_vae_path: str | None
    sd_cpp_lora_paths: list[str]
    sd_cpp_allowed_extra_params: list[str]
    sd_cpp_default_steps: int
    sd_cpp_default_cfg_scale: float
    sd_cpp_default_sampler: str
    sd_cpp_device: str
    sd_cpp_timeout_seconds: int
    swarmui_base_url: str | None
    swarmui_default_model: str | None
    swarmui_swarm_token: str | None
    swarmui_allowed_extra_params: list[str]
    swarmui_timeout_seconds: int
    openrouter_image_base_url: str | None
    openrouter_image_api_key: str | None
    openrouter_image_default_model: str | None
    openrouter_image_allowed_extra_params: list[str]
    openrouter_image_timeout_seconds: int
    novita_image_base_url: str | None
    novita_image_api_key: str | None
    novita_image_default_model: str | None
    novita_image_allowed_extra_params: list[str]
    novita_image_timeout_seconds: int
    novita_image_poll_interval_seconds: int
    together_image_base_url: str | None
    together_image_api_key: str | None
    together_image_default_model: str | None
    together_image_allowed_extra_params: list[str]
    together_image_timeout_seconds: int
    modelstudio_image_base_url: str | None
    modelstudio_image_api_key: str | None
    modelstudio_image_default_model: str | None
    modelstudio_image_region: str
    modelstudio_image_mode: str
    modelstudio_image_poll_interval_seconds: int
    modelstudio_image_timeout_seconds: int
    modelstudio_image_allowed_extra_params: list[str]
    reference_image_supported_models: dict[str, list[str]] = field(default_factory=dict)


_config_cache: ImageGenerationConfig | None = None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_choice(
    value: Any,
    *,
    default: str,
    allowed: set[str],
) -> str:
    """Normalize a string choice to lowercase and return `default` when invalid."""
    raw = str(value or "").strip().lower()
    if raw in allowed:
        return raw
    return default


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_mapping_of_lists(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}

    result: dict[str, list[str]] = {}
    for raw_key, raw_values in parsed.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if isinstance(raw_values, list):
            items = [str(item).strip() for item in raw_values if str(item).strip()]
        elif raw_values is None:
            items = []
        else:
            raw = str(raw_values).strip()
            if not raw:
                items = []
            else:
                try:
                    candidate = json.loads(raw)
                except Exception:
                    candidate = None
                if isinstance(candidate, list):
                    items = [str(item).strip() for item in candidate if str(item).strip()]
                else:
                    items = [item.strip() for item in raw.split(",") if item.strip()]
        result[key] = items
    return result


def _get_config_value(section: dict[str, str], key: str) -> str | None:
    raw = section.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def get_image_generation_config(*, reload: bool = False) -> ImageGenerationConfig:
    global _config_cache
    if _config_cache is not None and not reload:
        return _config_cache

    section = get_config_section("Image-Generation", reload=reload)

    default_backend = _get_config_value(section, "default_backend") or DEFAULT_BACKEND
    enabled_backends = _parse_list(section.get("enabled_backends"))
    if not enabled_backends:
        enabled_backends = []

    inline_max_bytes_raw = _get_config_value(section, "inline_max_bytes")
    inline_max_bytes = DEFAULT_INLINE_MAX_BYTES
    if inline_max_bytes_raw is not None:
        inline_max_bytes = max(1, _coerce_int(inline_max_bytes_raw, DEFAULT_INLINE_MAX_BYTES))

    config = ImageGenerationConfig(
        default_backend=default_backend,
        enabled_backends=enabled_backends,
        max_width=_coerce_int(section.get("max_width"), DEFAULT_MAX_WIDTH),
        max_height=_coerce_int(section.get("max_height"), DEFAULT_MAX_HEIGHT),
        max_pixels=_coerce_int(section.get("max_pixels"), DEFAULT_MAX_PIXELS),
        max_steps=_coerce_int(section.get("max_steps"), DEFAULT_MAX_STEPS),
        max_prompt_length=_coerce_int(section.get("max_prompt_length"), DEFAULT_MAX_PROMPT_LENGTH),
        inline_max_bytes=inline_max_bytes,
        sd_cpp_diffusion_model_path=_get_config_value(section, "sd_cpp_diffusion_model_path"),
        sd_cpp_llm_path=_get_config_value(section, "sd_cpp_llm_path"),
        sd_cpp_binary_path=_get_config_value(section, "sd_cpp_binary_path"),
        sd_cpp_model_path=_get_config_value(section, "sd_cpp_model_path"),
        sd_cpp_vae_path=_get_config_value(section, "sd_cpp_vae_path"),
        sd_cpp_lora_paths=_parse_list(section.get("sd_cpp_lora_paths")),
        sd_cpp_allowed_extra_params=_parse_list(section.get("sd_cpp_allowed_extra_params")),
        sd_cpp_default_steps=_coerce_int(section.get("sd_cpp_default_steps"), DEFAULT_SD_CPP_STEPS),
        sd_cpp_default_cfg_scale=_coerce_float(section.get("sd_cpp_default_cfg_scale"), DEFAULT_SD_CPP_CFG_SCALE),
        sd_cpp_default_sampler=_get_config_value(section, "sd_cpp_default_sampler") or DEFAULT_SD_CPP_SAMPLER,
        sd_cpp_device=_get_config_value(section, "sd_cpp_device") or DEFAULT_SD_CPP_DEVICE,
        sd_cpp_timeout_seconds=_coerce_int(section.get("sd_cpp_timeout_seconds"), DEFAULT_SD_CPP_TIMEOUT_SECONDS),
        swarmui_base_url=_get_config_value(section, "swarmui_base_url") or DEFAULT_SWARMUI_BASE_URL,
        swarmui_default_model=_get_config_value(section, "swarmui_default_model"),
        swarmui_swarm_token=_get_config_value(section, "swarmui_swarm_token"),
        swarmui_allowed_extra_params=_parse_list(section.get("swarmui_allowed_extra_params")),
        swarmui_timeout_seconds=_coerce_int(section.get("swarmui_timeout_seconds"), DEFAULT_SWARMUI_TIMEOUT_SECONDS),
        openrouter_image_base_url=_get_config_value(section, "openrouter_image_base_url")
        or DEFAULT_OPENROUTER_IMAGE_BASE_URL,
        openrouter_image_api_key=_get_config_value(section, "openrouter_image_api_key"),
        openrouter_image_default_model=_get_config_value(section, "openrouter_image_default_model")
        or DEFAULT_OPENROUTER_IMAGE_MODEL,
        openrouter_image_allowed_extra_params=_parse_list(section.get("openrouter_image_allowed_extra_params")),
        openrouter_image_timeout_seconds=_coerce_int(
            section.get("openrouter_image_timeout_seconds"),
            DEFAULT_OPENROUTER_IMAGE_TIMEOUT_SECONDS,
        ),
        novita_image_base_url=_get_config_value(section, "novita_image_base_url")
        or DEFAULT_NOVITA_IMAGE_BASE_URL,
        novita_image_api_key=_get_config_value(section, "novita_image_api_key"),
        novita_image_default_model=_get_config_value(section, "novita_image_default_model")
        or DEFAULT_NOVITA_IMAGE_MODEL,
        novita_image_allowed_extra_params=_parse_list(section.get("novita_image_allowed_extra_params")),
        novita_image_timeout_seconds=_coerce_int(
            section.get("novita_image_timeout_seconds"),
            DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS,
        ),
        novita_image_poll_interval_seconds=max(
            1,
            _coerce_int(
                section.get("novita_image_poll_interval_seconds"),
                DEFAULT_NOVITA_IMAGE_POLL_INTERVAL_SECONDS,
            ),
        ),
        together_image_base_url=_get_config_value(section, "together_image_base_url")
        or DEFAULT_TOGETHER_IMAGE_BASE_URL,
        together_image_api_key=_get_config_value(section, "together_image_api_key"),
        together_image_default_model=_get_config_value(section, "together_image_default_model")
        or DEFAULT_TOGETHER_IMAGE_MODEL,
        together_image_allowed_extra_params=_parse_list(section.get("together_image_allowed_extra_params")),
        together_image_timeout_seconds=_coerce_int(
            section.get("together_image_timeout_seconds"),
            DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS,
        ),
        modelstudio_image_base_url=_get_config_value(section, "modelstudio_image_base_url"),
        modelstudio_image_api_key=_get_config_value(section, "modelstudio_image_api_key"),
        modelstudio_image_default_model=_get_config_value(section, "modelstudio_image_default_model")
        or DEFAULT_MODELSTUDIO_IMAGE_MODEL,
        modelstudio_image_region=_coerce_choice(
            _get_config_value(section, "modelstudio_image_region"),
            default=DEFAULT_MODELSTUDIO_IMAGE_REGION,
            allowed={"sg", "cn", "us"},
        ),
        modelstudio_image_mode=_coerce_choice(
            _get_config_value(section, "modelstudio_image_mode"),
            default=DEFAULT_MODELSTUDIO_IMAGE_MODE,
            allowed={"sync", "async", "auto"},
        ),
        modelstudio_image_poll_interval_seconds=max(
            1,
            _coerce_int(
                section.get("modelstudio_image_poll_interval_seconds"),
                DEFAULT_MODELSTUDIO_IMAGE_POLL_INTERVAL_SECONDS,
            ),
        ),
        modelstudio_image_timeout_seconds=_coerce_int(
            section.get("modelstudio_image_timeout_seconds"),
            DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
        ),
        modelstudio_image_allowed_extra_params=_parse_list(section.get("modelstudio_image_allowed_extra_params")),
        reference_image_supported_models=_parse_mapping_of_lists(section.get("reference_image_supported_models")),
    )

    _config_cache = config
    return config


def reset_image_generation_config_cache() -> None:
    global _config_cache
    _config_cache = None
