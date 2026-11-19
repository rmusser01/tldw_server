from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from .precedence import resolve
from .types import ConfigParserLike

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}

_ENV_KEYS = {
    "chunking_method": ("CHUNKING_METHOD", "CHUNKING_CHUNKING_METHOD"),
    "chunk_max_size": ("CHUNKING_MAX_SIZE", "CHUNK_MAX_SIZE", "CHUNKING_CHUNK_MAX_SIZE"),
    "chunk_overlap": ("CHUNKING_OVERLAP", "CHUNK_OVERLAP", "CHUNKING_CHUNK_OVERLAP"),
    "adaptive_chunking": ("CHUNKING_ADAPTIVE", "ADAPTIVE_CHUNKING", "CHUNKING_ADAPTIVE_CHUNKING"),
    "chunking_multi_level": ("CHUNKING_MULTI_LEVEL", "CHUNKING_CHUNKING_MULTI_LEVEL"),
    "chunk_language": ("CHUNKING_LANGUAGE", "CHUNK_LANGUAGE"),
}


@dataclass(frozen=True)
class ChunkingConfig:
    method: str
    max_size: int
    overlap: int
    adaptive: bool
    multi_level: bool
    language: str


def _parse_int(raw: object, default: int) -> int:
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _parse_bool(raw: object, default: bool) -> bool:
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in _TRUE_VALUES:
        return True
    return False if text else default


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    option: str,
    default: str,
) -> str:
    """Return a chunking value with env -> parser -> default precedence."""
    env_keys = _ENV_KEYS.get(option, (option.upper(),))
    resolved = resolve(env_keys, (("Chunking", option),), config_parser=config_parser, env=env_map, default=default)
    return str(resolved.value).strip() or default


def load_chunking_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ChunkingConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    return ChunkingConfig(
        method=_get_raw(config_parser, env_map, "chunking_method", "words"),
        max_size=_parse_int(_get_raw(config_parser, env_map, "chunk_max_size", "400"), 400),
        overlap=_parse_int(_get_raw(config_parser, env_map, "chunk_overlap", "200"), 200),
        adaptive=_parse_bool(_get_raw(config_parser, env_map, "adaptive_chunking", "false"), False),
        multi_level=_parse_bool(
            _get_raw(config_parser, env_map, "chunking_multi_level", "false"),
            False,
        ),
        language=_get_raw(config_parser, env_map, "chunk_language", "en"),
    )
