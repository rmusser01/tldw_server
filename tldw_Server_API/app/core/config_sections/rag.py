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


def default_fts_level(
    config_parser: ConfigParserLike | None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Default RAG FTS level, lower-cased and not validated: env (blank included), else ``[RAG] default_fts_level``."""
    env_value = resolve(("RAG_DEFAULT_FTS_LEVEL",), env=env, blank_env=True).value
    if env_value is not None:
        return str(env_value).lower()
    configured = resolve((), (("RAG", "default_fts_level"),), config_parser=config_parser, blank_config=True).value
    return str(configured if configured is not None else "media").lower()


@dataclass(frozen=True)
class ContextualDefaults:
    include_parent_document: bool = False
    parent_max_tokens: int = 1200
    include_sibling_chunks: bool = False
    sibling_window: int = 1


def contextual_defaults(
    config_parser: ConfigParserLike | None,
    env: Mapping[str, str] | None = None,
) -> ContextualDefaults:
    """Contextual-retrieval request defaults: env first, then ``[RAG]``, then the literals.

    A non-integer value raises ``ValueError``.
    """
    from tldw_Server_API.app.core.config import _int_env_or_cfg

    def _env(key: str) -> str | None:
        return resolve((key,), env=env, blank_env=True).value

    def _cfg(option: str) -> str | None:
        return resolve((), (("RAG", option),), config_parser=config_parser, blank_config=True).value

    def _flag(key: str, option: str) -> bool:
        env_value = _env(key)
        if env_value is not None:
            return env_value.lower() == "true"
        return str(_cfg(option) or "false").lower() == "true"

    return ContextualDefaults(
        include_parent_document=_flag("RAG_INCLUDE_PARENT_DOCUMENT", "include_parent_document"),
        parent_max_tokens=_int_env_or_cfg(_env("RAG_PARENT_MAX_TOKENS"), _cfg("parent_max_tokens"), 1200),
        include_sibling_chunks=_flag("RAG_INCLUDE_SIBLING_CHUNKS", "include_sibling_chunks"),
        sibling_window=_int_env_or_cfg(_env("RAG_SIBLING_WINDOW"), _cfg("sibling_window"), 1),
    )
