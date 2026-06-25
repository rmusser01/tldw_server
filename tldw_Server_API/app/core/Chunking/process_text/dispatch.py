from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from tldw_Server_API.app.core.Chunking.base import ChunkMetadata
from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.exceptions import ChunkingError
from tldw_Server_API.app.core.Chunking.process_text.models import (
    NormalizedChunk,
    ProcessTextContext,
    ResolvedProcessOptions,
)


def dispatch_chunks(
    context: ProcessTextContext,
    processed_text: str,
    resolved: ResolvedProcessOptions,
) -> list[NormalizedChunk]:
    if resolved.hierarchical or resolved.hier_template:
        return _dispatch_hierarchical(context, processed_text, resolved)
    if resolved.multi_level:
        return _dispatch_multi_level(context, processed_text, resolved)
    return _dispatch_normal(context, processed_text, resolved)


def _dispatch_hierarchical(
    context: ProcessTextContext,
    processed_text: str,
    resolved: ResolvedProcessOptions,
) -> list[NormalizedChunk]:
    raw_chunks = context.chunk_text_hierarchical_flat(
        text=processed_text,
        method=resolved.method,
        max_size=resolved.max_size,
        overlap=resolved.overlap,
        language=resolved.language,
        template=resolved.hier_template,
        method_options=resolved.method_options_for_chunk,
    )
    return [
        NormalizedChunk(
            text=item.get("text", "") if isinstance(item, dict) else str(item),
            metadata=_metadata_to_dict(item.get("metadata") if isinstance(item, dict) else None),
        )
        for item in (raw_chunks or [])
    ]


def _dispatch_multi_level(
    context: ProcessTextContext,
    processed_text: str,
    resolved: ResolvedProcessOptions,
) -> list[NormalizedChunk]:
    method_options_for_chunk = dict(resolved.method_options_for_chunk)
    norm_chunks: list[NormalizedChunk] = []
    spans = context._compute_paragraph_spans(processed_text, template=None)
    pidx = 0
    for start, end, kind in spans:
        if kind == "blank":
            continue
        segment = processed_text[start:end]
        if not segment:
            continue
        try:
            base_results = context.chunk_text_with_metadata(
                segment,
                method=resolved.method,
                max_size=resolved.max_size,
                overlap=resolved.overlap,
                language=resolved.language,
                align_text_to_source=resolved.align_text_to_source,
                **method_options_for_chunk,
            )
            use_metadata = True
        except ChunkingError:
            base_results = context.chunk_text(
                segment,
                method=resolved.method,
                max_size=resolved.max_size,
                overlap=resolved.overlap,
                language=resolved.language,
                **method_options_for_chunk,
            )
            use_metadata = False

        if use_metadata:
            _append_metadata_results(
                norm_chunks,
                base_results,
                start=start,
                kind=kind,
                paragraph_index=pidx,
                resolved=resolved,
            )
        else:
            _append_fallback_results(
                norm_chunks,
                base_results,
                processed_text=processed_text,
                start=start,
                end=end,
                kind=kind,
                paragraph_index=pidx,
                resolved=resolved,
            )
        pidx += 1
    return norm_chunks


def _append_metadata_results(
    norm_chunks: list[NormalizedChunk],
    base_results: list[Any] | None,
    *,
    start: int,
    kind: str,
    paragraph_index: int,
    resolved: ResolvedProcessOptions,
) -> None:
    for res in base_results or []:
        chunk_text = getattr(res, "text", "")
        metadata_obj = getattr(res, "metadata", None)
        md = _metadata_to_dict(metadata_obj)

        local_start = md.get("start_char")
        local_end = md.get("end_char")
        global_start = start + local_start if isinstance(local_start, int) else start
        if isinstance(local_end, int):
            global_end = start + local_end
        else:
            global_end = global_start + len(chunk_text)

        md["start_char"] = global_start
        md["end_char"] = global_end
        md["start_offset"] = global_start
        md["end_offset"] = global_end
        md["method"] = resolved.method
        md["language"] = resolved.language
        md["paragraph_index"] = paragraph_index
        md["paragraph_kind"] = kind
        md["multi_level"] = True

        norm_chunks.append(NormalizedChunk(text=chunk_text, metadata=md))


def _append_fallback_results(
    norm_chunks: list[NormalizedChunk],
    base_results: list[Any] | None,
    *,
    processed_text: str,
    start: int,
    end: int,
    kind: str,
    paragraph_index: int,
    resolved: ResolvedProcessOptions,
) -> None:
    cursor = start
    for chunk in base_results or []:
        txt = chunk if isinstance(chunk, str) else (chunk.get("text") if isinstance(chunk, dict) else str(chunk))
        pos = processed_text.find(txt, cursor, end)
        if pos == -1:
            pos = cursor
        if pos < start:
            pos = start
        elif pos > end:
            pos = end
        end_pos = pos + len(txt)
        if end_pos > end:
            end_pos = end
        if end_pos < pos:
            end_pos = pos
        md = {}
        if isinstance(chunk, dict):
            md.update(_metadata_to_dict(chunk.get("metadata")))
        md.update(
            {
                "method": resolved.method,
                "start_offset": pos,
                "end_offset": end_pos,
                "language": resolved.language,
                "paragraph_index": paragraph_index,
                "paragraph_kind": kind,
                "multi_level": True,
            }
        )
        norm_chunks.append(NormalizedChunk(text=txt, metadata=md))
        cursor = min(end, end_pos)


def _dispatch_normal(
    context: ProcessTextContext,
    processed_text: str,
    resolved: ResolvedProcessOptions,
) -> list[NormalizedChunk]:
    base_chunks = context.chunk_text(
        processed_text,
        method=resolved.method,
        max_size=resolved.max_size,
        overlap=resolved.overlap,
        language=resolved.language,
        **resolved.method_options_for_chunk,
    )
    norm_chunks: list[NormalizedChunk] = []
    for chunk in base_chunks or []:
        if isinstance(chunk, dict) and "json" in chunk and "metadata" in chunk:
            try:
                txt = json.dumps(chunk["json"], ensure_ascii=False)
            except CHUNKER_NONCRITICAL_EXCEPTIONS:
                txt = str(chunk["json"])
            norm_chunks.append(NormalizedChunk(text=txt, metadata=_metadata_to_dict(chunk.get("metadata"))))
        elif isinstance(chunk, dict) and "text" in chunk:
            norm_chunks.append(
                NormalizedChunk(
                    text=chunk["text"],
                    metadata=_metadata_to_dict(chunk.get("metadata")),
                )
            )
        elif isinstance(chunk, str):
            norm_chunks.append(NormalizedChunk(text=chunk, metadata={}))
        else:
            norm_chunks.append(NormalizedChunk(text=str(chunk), metadata={}))
    return norm_chunks


def _metadata_to_dict(metadata_obj: Any) -> dict[str, Any]:
    if isinstance(metadata_obj, ChunkMetadata):
        return asdict(metadata_obj)
    if isinstance(metadata_obj, Mapping):
        return dict(metadata_obj)
    try:
        return dict(metadata_obj)
    except (TypeError, ValueError, KeyError):
        pass
    return {}
