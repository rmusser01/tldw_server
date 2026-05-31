from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class ChatConfig:
    enable_provider_fallback: bool
    max_base64_image_size_mb: int
    max_text_length_per_message: int
    max_messages_per_request: int
    max_images_per_request: int
    chat_stream_channel_maxsize: int
    chat_stream_include_metadata: bool
    chat_save_default: bool
    allow_autoswitch_to_openai: bool
    rate_limit_per_minute: int
    rate_limit_per_conversation_per_minute: int
    rate_limit_tokens_per_minute: int
    run_first_rollout_mode: str
    run_first_provider_allowlist: list[str]
    run_first_presentation_variant: str


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    *,
    env_keys: tuple[str, ...] = (),
    option: str,
    fallback_option: str | None = None,
    default: str,
) -> str:
    for env_key in env_keys:
        env_value = env_map.get(env_key)
        if env_value is not None and str(env_value).strip() != "":
            return str(env_value)

    raw = config_parser.get("Chat-Module", option, fallback=None)
    text = str(raw).strip() if raw is not None else ""
    if text:
        return text

    if fallback_option is not None:
        fallback_raw = config_parser.get("Chat-Module", fallback_option, fallback=None)
        fallback_text = str(fallback_raw).strip() if fallback_raw is not None else ""
        if fallback_text:
            return fallback_text

    return default


def _parse_bool(raw: object, default: bool) -> bool:
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _parse_int(raw: object, default: int, *, minimum: int | None = None) -> int:
    text = str(raw).strip()
    if not text:
        return default
    try:
        value = int(text)
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


def _parse_csv(raw: object) -> list[str]:
    text = str(raw).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def load_chat_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ChatConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    enable_provider_fallback = _parse_bool(
        _get_raw(
            config_parser,
            env_map,
            option="enable_provider_fallback",
            default="false",
        ),
        False,
    )
    max_base64_image_size_mb = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_IMAGE_MAX_MB",),
            option="max_base64_image_size_mb",
            default="3",
        ),
        3,
        minimum=1,
    )
    max_text_length_per_message = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            option="max_text_length_per_message",
            default="400000",
        ),
        400000,
        minimum=1,
    )
    max_messages_per_request = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            option="max_messages_per_request",
            default="1000",
        ),
        1000,
        minimum=1,
    )
    max_images_per_request = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            option="max_images_per_request",
            default="10",
        ),
        10,
        minimum=0,
    )
    chat_stream_channel_maxsize = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_STREAM_CHANNEL_MAXSIZE",),
            option="chat_stream_channel_maxsize",
            default="100",
        ),
        100,
        minimum=1,
    )
    chat_stream_include_metadata = _parse_bool(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_STREAM_INCLUDE_METADATA",),
            option="chat_stream_include_metadata",
            default="true",
        ),
        True,
    )
    chat_save_default = _parse_bool(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_SAVE_DEFAULT", "DEFAULT_CHAT_SAVE"),
            option="chat_save_default",
            fallback_option="default_save_to_db",
            default="false",
        ),
        False,
    )
    allow_autoswitch_to_openai = _parse_bool(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("ALLOW_AUTOSWITCH_TO_OPENAI",),
            option="allow_autoswitch_to_openai",
            default="false",
        ),
        False,
    )
    rate_limit_per_minute = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            option="rate_limit_per_minute",
            default="60",
        ),
        60,
        minimum=1,
    )
    rate_limit_per_conversation_per_minute = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            option="rate_limit_per_conversation_per_minute",
            default="20",
        ),
        20,
        minimum=1,
    )
    rate_limit_tokens_per_minute = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            option="rate_limit_tokens_per_minute",
            default="100000",
        ),
        100000,
        minimum=1,
    )
    run_first_rollout_mode = str(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_RUN_FIRST_ROLLOUT_MODE",),
            option="run_first_rollout_mode",
            default="default_on",
        )
    ).strip() or "default_on"
    run_first_provider_allowlist = _parse_csv(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_RUN_FIRST_PROVIDER_ALLOWLIST",),
            option="run_first_provider_allowlist",
            default="",
        )
    )
    run_first_presentation_variant = str(
        _get_raw(
            config_parser,
            env_map,
            env_keys=("CHAT_RUN_FIRST_PRESENTATION_VARIANT",),
            option="run_first_presentation_variant",
            default="chat_phase2b_v1",
        )
    ).strip() or "chat_phase2b_v1"

    return ChatConfig(
        enable_provider_fallback=enable_provider_fallback,
        max_base64_image_size_mb=max_base64_image_size_mb,
        max_text_length_per_message=max_text_length_per_message,
        max_messages_per_request=max_messages_per_request,
        max_images_per_request=max_images_per_request,
        chat_stream_channel_maxsize=chat_stream_channel_maxsize,
        chat_stream_include_metadata=chat_stream_include_metadata,
        chat_save_default=chat_save_default,
        allow_autoswitch_to_openai=allow_autoswitch_to_openai,
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_per_conversation_per_minute=rate_limit_per_conversation_per_minute,
        rate_limit_tokens_per_minute=rate_limit_tokens_per_minute,
        run_first_rollout_mode=run_first_rollout_mode,
        run_first_provider_allowlist=run_first_provider_allowlist,
        run_first_presentation_variant=run_first_presentation_variant,
    )
