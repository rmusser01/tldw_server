from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


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


def load_chunking_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ChunkingConfig:
    del env  # Reserved for future overrides; current runtime reads config defaults directly.

    return ChunkingConfig(
        method=str(config_parser.get("Chunking", "chunking_method", fallback="words")).strip() or "words",
        max_size=_parse_int(config_parser.get("Chunking", "chunk_max_size", fallback="400"), 400),
        overlap=_parse_int(config_parser.get("Chunking", "chunk_overlap", fallback="200"), 200),
        adaptive=_parse_bool(config_parser.get("Chunking", "adaptive_chunking", fallback="false"), False),
        multi_level=_parse_bool(
            config_parser.get("Chunking", "chunking_multi_level", fallback="false"),
            False,
        ),
        language=str(config_parser.get("Chunking", "chunk_language", fallback="en")).strip() or "en",
    )
