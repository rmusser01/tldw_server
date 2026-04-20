from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike


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


def load_embeddings_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> EmbeddingsConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ
    g = lambda key, default: str(
        env_map.get(f"EMBEDDING_{key.upper()}", "")
        or config_parser.get("Embeddings", key, fallback=default)
    ).strip() or default

    return EmbeddingsConfig(
        embedding_provider=env_map.get("EMBEDDING_PROVIDER", "") or g("embedding_provider", "huggingface"),
        embedding_model=env_map.get("EMBEDDING_MODEL", "") or g("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        onnx_model_path=g("onnx_model_path", "./App_Function_Libraries/models/onnx_models/"),
        model_dir=g("model_dir", "./App_Function_Libraries/models/embedding_models"),
        embedding_api_url=g("embedding_api_url", "http://localhost:8080/v1/embeddings"),
        chunk_size=int(g("chunk_size", "400")),
        overlap=int(g("overlap", "200")),
        enable_contextual_chunking=g("enable_contextual_chunking", "false").lower() in ("true", "1", "yes"),
        contextual_llm_model=g("contextual_llm_model", "gpt-3.5-turbo"),
    )
