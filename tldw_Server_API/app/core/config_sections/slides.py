from __future__ import annotations

import math
import os
from collections.abc import Mapping
from configparser import NoSectionError
from dataclasses import dataclass

from .types import ConfigParserLike

_SECTION = "SlidesStandaloneHtml"
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True, slots=True)
class SlidesConfig:
    """Raw typed settings for the standalone HTML generation boundary."""

    enabled: bool
    egress_enabled: bool
    default_provider: str
    default_model: str
    default_adapter_id: str
    allowed_targets_json: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    overall_timeout_seconds: float
    max_output_tokens: int
    max_source_chars: int
    max_source_tokens: int
    max_provider_response_bytes: int


def _invalid() -> ValueError:
    return ValueError("standalone_html_config_invalid")


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    *,
    env_key: str,
    option: str,
    default: str,
    strip: bool = True,
) -> str:
    if env_key in env_map:
        text = str(env_map[env_key])
        return text.strip() if strip else text
    try:
        text = str(config_parser.get(_SECTION, option, fallback=default))
        return text.strip() if strip else text
    except NoSectionError:
        return default


def _parse_bool(raw: str) -> bool:
    value = raw.casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise _invalid()


def _parse_positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _invalid() from None
    if value <= 0:
        raise _invalid()
    return value


def _parse_positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise _invalid() from None
    if not math.isfinite(value) or value <= 0:
        raise _invalid()
    return value


def load_slides_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> SlidesConfig:
    """Load strict standalone Slides settings with env-over-file precedence."""

    env_map: Mapping[str, str] = env if env is not None else os.environ

    def value(option: str, default: str, *, strip: bool = True) -> str:
        return _get_raw(
            config_parser,
            env_map,
            env_key=f"SLIDES_STANDALONE_{option.upper()}",
            option=option,
            default=default,
            strip=strip,
        )

    return SlidesConfig(
        enabled=_parse_bool(value("enabled", "false")),
        egress_enabled=_parse_bool(value("egress_enabled", "false")),
        default_provider=value("default_provider", ""),
        default_model=value("default_model", ""),
        default_adapter_id=value("default_adapter_id", ""),
        allowed_targets_json=value("allowed_targets_json", "[]", strip=False),
        connect_timeout_seconds=_parse_positive_float(value("connect_timeout_seconds", "10")),
        read_timeout_seconds=_parse_positive_float(value("read_timeout_seconds", "120")),
        overall_timeout_seconds=_parse_positive_float(value("overall_timeout_seconds", "180")),
        max_output_tokens=_parse_positive_int(value("max_output_tokens", "16384")),
        max_source_chars=_parse_positive_int(value("max_source_chars", "200000")),
        max_source_tokens=_parse_positive_int(value("max_source_tokens", "50000")),
        max_provider_response_bytes=_parse_positive_int(value("max_provider_response_bytes", "8388608")),
    )


__all__ = ["SlidesConfig", "load_slides_config"]
