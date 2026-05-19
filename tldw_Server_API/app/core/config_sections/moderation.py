from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

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


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    *,
    env_key: str | None,
    option: str,
    default: str,
) -> str:
    if env_key is not None:
        env_value = env_map.get(env_key)
        if env_value is not None and str(env_value).strip() != "":
            return str(env_value)

    raw = config_parser.get("Moderation", option, fallback=default)
    text = str(raw).strip()
    return text or default


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
