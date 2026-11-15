"""Bounded canonical context for retrieved media excerpts."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_document_version,
    get_media_by_id,
    get_unvectorized_chunks_in_range,
    managed_media_database,
)
from tldw_Server_API.app.core.Media.page_provenance import page_map_for_content, page_spans_for_range

from .types import DataSource, Document

_PAGE_FIELDS = {
    "page",
    "page_no",
    "page_number",
    "page_start",
    "page_end",
    "page_spans",
    "printed_page_label",
    "page_map",
    "page_map_content_hash",
    "page_count",
}


def _integer(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _without_pages(metadata: dict[str, Any]) -> dict[str, Any]:
    """Discard old locations before deriving locations for the canonical slice."""
    cleaned = {
        key: value
        for key, value in metadata.items()
        if key not in _PAGE_FIELDS and key.removeprefix("citation.") not in _PAGE_FIELDS
    }
    if isinstance(cleaned.get("citation"), dict):
        cleaned["citation"] = {key: value for key, value in cleaned["citation"].items() if key not in _PAGE_FIELDS}
    return cleaned


async def expand_media_context(
    documents: list[Document],
    *,
    media_db: Any = None,
    media_db_path: str | None = None,
    enable_parent_expansion: bool,
    parent_context_size: int,
    include_sibling_chunks: bool,
    sibling_window: int,
) -> tuple[list[Document], dict[str, int]]:
    """Expand exact media seeds in place in ranking, with a per-side character cap.

    Neighbor chunks extend the same seed rather than becoming separately ranked
    results. Hidden parents are removed; outdated excerpts remain unexpanded.
    Cancellation stops the worker before its next database read.
    """
    stats = {
        "context_expanded_count": 0,
        "siblings_added_count": 0,
        "context_skipped_count": 0,
        "context_hidden_count": 0,
        "context_unmapped_count": 0,
    }
    if not documents or not (enable_parent_expansion or include_sibling_chunks):
        return documents, stats
    if not any(doc.source == DataSource.MEDIA_DB for doc in documents):
        return documents, stats
    if media_db is None and not media_db_path:
        stats["context_skipped_count"] = sum(doc.source == DataSource.MEDIA_DB for doc in documents)
        return documents, stats
    budget = min(2000, max(0, _integer(parent_context_size) or 0))
    window = min(50, max(0, _integer(sibling_window) or 0))
    cancelled = threading.Event()

    def run() -> tuple[list[Document], dict[str, int]]:
        output = []
        parents = {}
        incorporated = set()
        session = (
            nullcontext(media_db)
            if media_db is not None
            else managed_media_database(
                "rag_context_expansion", db_path=media_db_path, initialize=False, existing_only=True
            )
        )
        with session as db:
            for doc in documents:
                if cancelled.is_set():
                    break
                if doc.source != DataSource.MEDIA_DB:
                    output.append(doc)
                    continue
                metadata = doc.metadata or {}
                media_id = _integer(metadata.get("media_id") or metadata.get("source_id"))
                if media_id is None or media_id < 1:
                    output.append(doc)
                    stats["context_skipped_count"] += 1
                    continue
                if media_id not in parents:
                    parent = get_media_by_id(db, media_id, respect_scope=True)
                    if cancelled.is_set():
                        break
                    if (
                        not parent
                        or parent.get("deleted")
                        or parent.get("is_trash")
                        or parent.get("system_operation_id")
                    ):
                        parents[media_id] = None
                    else:
                        version = get_document_version(db, media_id, include_content=True)
                        if cancelled.is_set():
                            break
                        canonical = version or parent
                        content = canonical.get("content")
                        safe_metadata = canonical.get("safe_metadata") or {}
                        if isinstance(safe_metadata, str):
                            try:
                                safe_metadata = json.loads(safe_metadata)
                            except (ValueError, TypeError):
                                safe_metadata = {}
                        parents[media_id] = (content, page_map_for_content(safe_metadata, content))
                parent = parents[media_id]
                if parent is None:
                    stats["context_hidden_count"] += 1
                    continue
                content, page_map = parent
                start = _integer(doc.start_char if doc.start_char is not None else metadata.get("start_char"))
                end = _integer(doc.end_char if doc.end_char is not None else metadata.get("end_char"))
                cleaned = _without_pages(metadata)
                if (
                    not isinstance(content, str)
                    or start is None
                    or end is None
                    or not 0 <= start < end <= len(content)
                    or content[start:end] != doc.content
                ):
                    output.append(replace(doc, metadata=cleaned, page_number=None))
                    stats["context_skipped_count"] += 1
                    continue
                lower, upper = max(0, start - budget), min(len(content), end + budget)
                section_start, section_end = (
                    _integer(metadata.get("section_start")),
                    _integer(metadata.get("section_end")),
                )
                if (
                    section_start is not None
                    and section_end is not None
                    and 0 <= section_start <= start < end <= section_end <= len(content)
                ):
                    lower, upper = max(lower, section_start), min(upper, section_end)
                expanded_start, expanded_end = (lower, upper) if enable_parent_expansion else (start, end)
                chunk_index = _integer(doc.chunk_index if doc.chunk_index is not None else metadata.get("chunk_index"))
                if include_sibling_chunks and window and chunk_index is not None:
                    if cancelled.is_set():
                        break
                    neighbors = get_unvectorized_chunks_in_range(
                        db, media_id, max(0, chunk_index - window), chunk_index + window
                    )
                    for neighbor in neighbors:
                        neighbor_start, neighbor_end = (
                            _integer(neighbor.get("start_char")),
                            _integer(neighbor.get("end_char")),
                        )
                        if (
                            neighbor_start is None
                            or neighbor_end is None
                            or not 0 <= neighbor_start < neighbor_end <= len(content)
                            or content[neighbor_start:neighbor_end] != neighbor.get("chunk_text")
                            or neighbor_start >= upper
                            or neighbor_end <= lower
                        ):
                            continue
                        expanded_start = min(expanded_start, max(lower, neighbor_start))
                        expanded_end = max(expanded_end, min(upper, neighbor_end))
                        if neighbor_start < start or neighbor_end > end:
                            incorporated.add((media_id, neighbor_start, neighbor_end))
                if cancelled.is_set():
                    break
                spans = page_spans_for_range(page_map, expanded_start, expanded_end)
                cleaned.update(start_char=expanded_start, end_char=expanded_end)
                if spans:
                    cleaned.update(page_spans=spans, page_number=spans[0]["page_number"])
                else:
                    stats["context_unmapped_count"] += 1
                if (expanded_start, expanded_end) != (start, end):
                    cleaned.update(matched_start_char=start, matched_end_char=end)
                    stats["context_expanded_count"] += 1
                output.append(
                    replace(
                        doc,
                        content=content[expanded_start:expanded_end],
                        metadata=cleaned,
                        start_char=expanded_start,
                        end_char=expanded_end,
                        page_number=spans[0]["page_number"] if spans else None,
                    )
                )
        stats["siblings_added_count"] = len(incorporated)
        return output, stats

    try:
        return await asyncio.to_thread(run)
    except asyncio.CancelledError:
        cancelled.set()
        raise
