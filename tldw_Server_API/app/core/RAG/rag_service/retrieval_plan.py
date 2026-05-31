"""Build normalized retrieval policy objects from resolved RAG requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .request_resolution import ResolvedRAGRequest

_SOURCE_ALIASES = {
    "character": "characters",
    "characters": "characters",
    "character_cards": "characters",
    "character_cards_db": "characters",
    "chat": "chats",
    "chats": "chats",
    "chat_history": "chats",
    "chat_history_db": "chats",
    "conversation": "chats",
    "conversations": "chats",
    "notes_db": "notes",
    "media": "media_db",
    "media_db_path": "media_db",
    "kanban_db": "kanban",
    "task_board": "kanban",
    "task_boards": "kanban",
    "tasks": "kanban",
    "prompt": "prompts",
    "prompts_db": "prompts",
    "worldbook": "world_books",
    "worldbooks": "world_books",
    "world_book": "world_books",
    "world_books_db": "world_books",
    "dictionary": "dictionaries",
    "chat_dictionary": "dictionaries",
    "chat_dictionaries": "dictionaries",
    "chat_dictionaries_db": "dictionaries",
}


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Normalized retrieval policy for a resolved RAG request."""

    query: str
    sources: tuple[str, ...]
    search_mode: str
    top_k: int
    min_score: float
    index_namespace: str | None
    collection_names: dict[str, str] = field(default_factory=dict)


def _normalize_sources(raw_sources: Any) -> tuple[str, ...]:
    if raw_sources is None:
        return ("media_db",)

    if isinstance(raw_sources, str):
        values = [raw_sources]
    elif isinstance(raw_sources, (list, tuple, set)):
        values = list(raw_sources)
    else:
        values = [raw_sources]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        try:
            source = str(value).strip().lower()
        except (TypeError, ValueError):
            continue
        source = _SOURCE_ALIASES.get(source, source)
        if not source or source in seen:
            continue
        seen.add(source)
        normalized.append(source)

    return tuple(normalized or ["media_db"])


def build_retrieval_plan(resolved: ResolvedRAGRequest) -> RetrievalPlan:
    """Derive a stable retrieval plan from a resolved request."""

    payload = resolved.payload or {}
    sources = _normalize_sources(payload.get("sources"))
    search_mode = str(payload.get("search_mode", "hybrid") or "hybrid").strip().lower() or "hybrid"
    try:
        top_k = int(payload.get("top_k", 10) or 10)
    except (TypeError, ValueError):
        top_k = 10
    try:
        min_score = float(payload.get("min_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        min_score = 0.0

    user_key = str(resolved.user_id or resolved.feedback_user_id or "0").strip() or "0"
    collection_names: dict[str, str] = {
        "media_db": f"user_{user_key}_media_embeddings",
        "notes": f"user_{user_key}_notes_embeddings",
    }
    if "characters" in sources:
        collection_names["character_cards"] = f"user_{user_key}_character_embeddings"
    if "kanban" in sources:
        collection_names["kanban"] = f"user_{user_key}_kanban_embeddings"

    logger.debug(
        "Built retrieval plan for query={!r}: sources={}, namespace={}",
        resolved.query,
        sources,
        resolved.index_namespace,
    )
    return RetrievalPlan(
        query=resolved.query,
        sources=sources,
        search_mode=search_mode,
        top_k=top_k,
        min_score=min_score,
        index_namespace=resolved.index_namespace,
        collection_names=collection_names,
    )
