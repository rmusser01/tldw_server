from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .precedence import resolve
from .types import ConfigParserLike


@dataclass(frozen=True)
class RAGConfig:
    vector_store_type: str
    default_llm_provider: str
    default_llm_model: str


def load_rag_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> RAGConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    vector_store_type = str(
        resolve(("RAG_VECTOR_STORE_TYPE",), env=env_map, blank_env=True).value
        or config_parser.get("RAG", "vector_store_type", fallback="chromadb")
    ).strip() or "chromadb"

    default_llm_provider = str(
        resolve(("RAG_DEFAULT_LLM_PROVIDER",), env=env_map, blank_env=True).value
        or config_parser.get("RAG", "rag_default_llm_provider", fallback="openai")
    ).strip() or "openai"

    default_llm_model = str(
        resolve(("RAG_DEFAULT_LLM_MODEL",), env=env_map, blank_env=True).value
        or config_parser.get("RAG", "rag_default_llm_model", fallback="gpt-4o-mini")
    ).strip() or "gpt-4o-mini"

    return RAGConfig(
        vector_store_type=vector_store_type,
        default_llm_provider=default_llm_provider,
        default_llm_model=default_llm_model,
    )
