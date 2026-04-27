from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.testing import is_truthy

from .types import ConfigParserLike

_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class EmbeddingsConfig:
    embedding_provider: str
    embedding_model: str
    onnx_model_path: str
    model_dir: str
    embedding_api_url: str
    chunk_size: int
    overlap: int
    enable_contextual_chunking: bool
    contextual_llm_model: str


def _env_keys_for(option: str) -> tuple[str, ...]:
    """Return canonical and legacy environment keys for an embeddings option."""
    env_key = option.upper()
    if env_key.startswith("EMBEDDING_"):
        return (env_key, f"EMBEDDING_{env_key}")
    return (f"EMBEDDING_{env_key}",)


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    option: str,
    default: str,
) -> str:
    """Return the first non-empty env override, parser value, or default."""
    for env_key in _env_keys_for(option):
        raw = env_map.get(env_key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()

    raw = config_parser.get("Embeddings", option, fallback=default)
    text = str(raw).strip()
    return text or default


def _parse_int(raw: object, default: int) -> int:
    """Parse an integer value, returning the default for malformed input."""
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _parse_bool(raw: object, default: bool) -> bool:
    """Parse a boolean value with project-standard truthy tokens."""
    text = str(raw).strip().lower()
    if not text:
        return default
    if is_truthy(text):
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def load_embeddings_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> EmbeddingsConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    return EmbeddingsConfig(
        embedding_provider=_get_raw(config_parser, env_map, "embedding_provider", "huggingface"),
        embedding_model=_get_raw(config_parser, env_map, "embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        onnx_model_path=_get_raw(
            config_parser,
            env_map,
            "onnx_model_path",
            "./App_Function_Libraries/models/onnx_models/",
        ),
        model_dir=_get_raw(config_parser, env_map, "model_dir", "./App_Function_Libraries/models/embedding_models"),
        embedding_api_url=_get_raw(
            config_parser,
            env_map,
            "embedding_api_url",
            "http://localhost:8080/v1/embeddings",
        ),
        chunk_size=_parse_int(_get_raw(config_parser, env_map, "chunk_size", "400"), 400),
        overlap=_parse_int(_get_raw(config_parser, env_map, "overlap", "200"), 200),
        enable_contextual_chunking=_parse_bool(
            _get_raw(config_parser, env_map, "enable_contextual_chunking", "false"),
            False,
        ),
        contextual_llm_model=_get_raw(config_parser, env_map, "contextual_llm_model", "gpt-3.5-turbo"),
    )
