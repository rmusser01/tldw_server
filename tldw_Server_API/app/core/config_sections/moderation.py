from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.core.Moderation.surfaces import KNOWN_SURFACES

from .precedence import resolve
from .types import ConfigParserLike

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class ModerationConfig:
    enabled: bool
    input_enabled: bool
    output_enabled: bool
    input_action: str
    output_action: str
    redact_replacement: str
    per_user_overrides: bool
    blocklist_file: str
    user_overrides_file: str
    runtime_overrides_file: str
    max_scan_chars: int
    max_replacements_per_pattern: int
    match_window_chars: int
    max_fallback_scan_chars: int
    blocklist_write_debounce_ms: int
    categories_enabled: list[str]
    pii_enabled: bool


_SURFACE_PREFIX = "surface."
_warned_unknown_surfaces: set[str] = set()


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    *,
    env_key: str | None,
    option: str,
    default: str,
) -> str:
    env_keys = () if env_key is None else (env_key,)
    resolved = resolve(env_keys, (("Moderation", option),), config_parser=config_parser, env=env_map, default=default)
    text = str(resolved.value)
    return text if resolved.source == "env" else (text.strip() or default)


def _parse_bool(raw: object, *, option: str, default: bool) -> bool:
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"Invalid Moderation.{option} value {raw!r}; expected one of: {accepted}")


def parse_surface_enabled(options: Mapping[str, object]) -> dict[str, bool]:
    """Read ``surface.<id> = true|false`` entries; an unknown id warns once and is ignored."""
    out: dict[str, bool] = {}
    for key, raw in options.items():
        name = str(key).strip().lower()
        if not name.startswith(_SURFACE_PREFIX):
            continue
        surface = name[len(_SURFACE_PREFIX):]
        if surface not in KNOWN_SURFACES:
            if surface not in _warned_unknown_surfaces:
                _warned_unknown_surfaces.add(surface)
                logger.warning("Ignoring unknown moderation surface key {!r} in [Moderation]", key)
            continue
        out[surface] = _parse_bool(raw, option=name, default=True)
    return out


def _parse_int(raw: object, *, option: str, default: int) -> int:
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Moderation.{option} integer value {raw!r}") from exc


def _parse_string_list(raw: object) -> list[str]:
    if raw is None:
        return []

    items: list[str]
    if isinstance(raw, (list, tuple, set)):
        items = [str(item) for item in raw]
    else:
        text = str(raw).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list):
                items = [str(item) for item in parsed]
            else:
                items = text.split(",")
        else:
            items = text.split(",")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def load_moderation_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ModerationConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    enabled = _parse_bool(
        _get_raw(config_parser, env_map, env_key=None, option="enabled", default="false"),
        option="enabled",
        default=False,
    )
    input_enabled = _parse_bool(
        _get_raw(config_parser, env_map, env_key=None, option="input_enabled", default="true"),
        option="input_enabled",
        default=True,
    )
    output_enabled = _parse_bool(
        _get_raw(config_parser, env_map, env_key=None, option="output_enabled", default="true"),
        option="output_enabled",
        default=True,
    )
    input_action = _get_raw(
        config_parser,
        env_map,
        env_key=None,
        option="input_action",
        default="block",
    ).lower()
    output_action = _get_raw(
        config_parser,
        env_map,
        env_key=None,
        option="output_action",
        default="redact",
    ).lower()
    redact_replacement = _get_raw(
        config_parser,
        env_map,
        env_key=None,
        option="redact_replacement",
        default="[REDACTED]",
    )
    per_user_overrides = _parse_bool(
        _get_raw(config_parser, env_map, env_key=None, option="per_user_overrides", default="true"),
        option="per_user_overrides",
        default=True,
    )
    blocklist_file = _get_raw(
        config_parser,
        env_map,
        env_key="MODERATION_BLOCKLIST_FILE",
        option="blocklist_file",
        default="tldw_Server_API/Config_Files/moderation_blocklist.txt",
    )
    user_overrides_file = _get_raw(
        config_parser,
        env_map,
        env_key="MODERATION_USER_OVERRIDES_FILE",
        option="user_overrides_file",
        default="tldw_Server_API/Config_Files/moderation_user_overrides.json",
    )
    runtime_overrides_file = _get_raw(
        config_parser,
        env_map,
        env_key="MODERATION_RUNTIME_OVERRIDES_FILE",
        option="runtime_overrides_file",
        default="tldw_Server_API/Config_Files/moderation_runtime_overrides.json",
    )
    max_scan_chars = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_MAX_SCAN_CHARS",
            option="max_scan_chars",
            default="200000",
        ),
        option="max_scan_chars",
        default=200000,
    )
    max_replacements_per_pattern = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_MAX_REPLACEMENTS_PER_PATTERN",
            option="max_replacements_per_pattern",
            default="1000",
        ),
        option="max_replacements_per_pattern",
        default=1000,
    )
    match_window_chars = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_MATCH_WINDOW_CHARS",
            option="match_window_chars",
            default="4096",
        ),
        option="match_window_chars",
        default=4096,
    )
    max_fallback_scan_chars = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_MAX_FALLBACK_SCAN_CHARS",
            option="max_fallback_scan_chars",
            default="800000",
        ),
        option="max_fallback_scan_chars",
        default=800000,
    )
    blocklist_write_debounce_ms = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_BLOCKLIST_WRITE_DEBOUNCE_MS",
            option="blocklist_write_debounce_ms",
            default="0",
        ),
        option="blocklist_write_debounce_ms",
        default=0,
    )
    categories_enabled = _parse_string_list(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_CATEGORIES_ENABLED",
            option="categories_enabled",
            default="",
        )
    )
    pii_enabled = _parse_bool(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_PII_ENABLED",
            option="pii_enabled",
            default="false",
        ),
        option="pii_enabled",
        default=False,
    )

    return ModerationConfig(
        enabled=enabled,
        input_enabled=input_enabled,
        output_enabled=output_enabled,
        input_action=input_action,
        output_action=output_action,
        redact_replacement=redact_replacement,
        per_user_overrides=per_user_overrides,
        blocklist_file=blocklist_file,
        user_overrides_file=user_overrides_file,
        runtime_overrides_file=runtime_overrides_file,
        max_scan_chars=max_scan_chars,
        max_replacements_per_pattern=max_replacements_per_pattern,
        match_window_chars=match_window_chars,
        max_fallback_scan_chars=max_fallback_scan_chars,
        blocklist_write_debounce_ms=blocklist_write_debounce_ms,
        categories_enabled=categories_enabled,
        pii_enabled=pii_enabled,
    )


@dataclass(frozen=True)
class ModerationSurfaceConfig:
    stream_buffer_chars: int
    surface_enabled: dict[str, bool]


def load_moderation_surface_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ModerationSurfaceConfig:
    """Per-surface keys and the output stream buffer size.

    Kept apart from ``load_moderation_config`` so that loader's characterisation golden is unchanged.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    stream_buffer_chars = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="MODERATION_STREAM_BUFFER_CHARS",
            option="stream_buffer_chars",
            default="1024",
        ),
        option="stream_buffer_chars",
        default=1024,
    )
    if stream_buffer_chars <= 0:
        raise ValueError(f"Invalid Moderation.stream_buffer_chars value {stream_buffer_chars!r}; must be positive")
    surface_options: dict[str, object] = {}
    if config_parser.has_section("Moderation"):  # type: ignore[attr-defined]
        surface_options = {
            name: config_parser.get("Moderation", name, fallback="")
            for name in config_parser.options("Moderation")  # type: ignore[attr-defined]
            if name.lower().startswith(_SURFACE_PREFIX)
        }

    return ModerationSurfaceConfig(
        stream_buffer_chars=stream_buffer_chars,
        surface_enabled=parse_surface_enabled(surface_options),
    )
