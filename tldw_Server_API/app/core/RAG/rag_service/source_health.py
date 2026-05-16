"""Safe source readiness summaries for Knowledge QA."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    KnowledgeSourceHealthEntry,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

CANONICAL_KNOWLEDGE_SOURCE_IDS: tuple[str, ...] = (
    "media_db",
    "notes",
    "chats",
    "characters",
    "kanban",
    "prompts",
    "world_books",
    "dictionaries",
)

_SOURCE_LABELS: dict[str, str] = {
    "media_db": "Documents & Media",
    "notes": "Notes",
    "chats": "Chats",
    "characters": "Characters",
    "kanban": "Task Boards",
    "prompts": "Prompts",
    "world_books": "World Books",
    "dictionaries": "Dictionaries",
}

_SOURCE_TO_DATASOURCE: dict[str, DataSource] = {
    "media_db": DataSource.MEDIA_DB,
    "notes": DataSource.NOTES,
    "chats": DataSource.CHAT_HISTORY,
    "characters": DataSource.CHARACTER_CARDS,
    "kanban": DataSource.KANBAN,
    "prompts": DataSource.PROMPTS,
    "world_books": DataSource.WORLD_BOOKS,
    "dictionaries": DataSource.DICTIONARIES,
}

_ALIASES_TO_DATASOURCE: dict[str, DataSource] = {
    source_id: data_source
    for source_id, data_source in _SOURCE_TO_DATASOURCE.items()
}
_ALIASES_TO_DATASOURCE.update(
    {
        data_source.value: data_source
        for data_source in _SOURCE_TO_DATASOURCE.values()
    }
)


def _normalize_configured_sources(
    configured_sources: Iterable[DataSource | str] | Mapping[Any, Any],
) -> set[DataSource]:
    """Normalize configured source identifiers without reading metadata values."""
    raw_sources: Iterable[Any]
    if isinstance(configured_sources, Mapping):
        raw_sources = configured_sources.keys()
    else:
        raw_sources = configured_sources

    normalized: set[DataSource] = set()
    for source in raw_sources:
        if isinstance(source, DataSource):
            normalized.add(source)
            continue
        if isinstance(source, str):
            data_source = _ALIASES_TO_DATASOURCE.get(source)
            if data_source is not None:
                normalized.add(data_source)
    return normalized


def build_source_health_entries(
    *,
    configured_sources: Iterable[DataSource | str] | Mapping[Any, Any],
    unsafe_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[KnowledgeSourceHealthEntry]:
    """Build safe pre-query source readiness entries for Knowledge QA."""
    del unsafe_metadata
    configured = _normalize_configured_sources(configured_sources)

    entries: list[KnowledgeSourceHealthEntry] = []
    for source_id in CANONICAL_KNOWLEDGE_SOURCE_IDS:
        data_source = _SOURCE_TO_DATASOURCE[source_id]
        is_configured = data_source in configured
        entries.append(
            KnowledgeSourceHealthEntry(
                source_id=source_id,
                label=_SOURCE_LABELS[source_id],
                available=is_configured,
                searchable=is_configured,
                index_status="ready" if is_configured else "unavailable",
                embedding_status="unknown" if is_configured else "unavailable",
                disabled_reason=None if is_configured else "no_retriever_configured",
            )
        )
    return entries
