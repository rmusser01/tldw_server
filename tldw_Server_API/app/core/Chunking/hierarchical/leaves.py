"""Leaf construction for hierarchical chunking."""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..base import ChunkingMethod
from ..error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS as _CHUNKER_NONCRITICAL_EXCEPTIONS
from .models import HierarchyTextViews, LeafChunkingContext, ResolvedHierarchyOptions

_REWRITE_METHODS = {
    ChunkingMethod.SEMANTIC.value,
    ChunkingMethod.PROPOSITIONS.value,
    ChunkingMethod.JSON.value,
    ChunkingMethod.XML.value,
    ChunkingMethod.EBOOK_CHAPTERS.value,
    ChunkingMethod.ROLLING_SUMMARIZE.value,
    ChunkingMethod.CODE.value,
    "code_ast",
}


def build_leaf_block(
    context: LeafChunkingContext,
    texts: HierarchyTextViews,
    span: tuple[int, int, str],
    options: ResolvedHierarchyOptions,
) -> dict[str, Any] | None:
    """Build one hierarchical leaf block for a source span."""
    start, end, kind = span
    if start >= end:
        return None

    segment_raw = texts.original[start:end]
    segment_clean = texts.sanitized[start:end]
    output_text = texts.output if options.sanitize_output else texts.original
    method = options.method
    max_size = options.max_size
    overlap = options.overlap
    language = options.language
    method_opts = options.method_options
    chunks = None
    out_chunks: list[dict[str, Any]] = []
    if method in _REWRITE_METHODS:
        # Compute chunks using selected method (may rewrite text, offsets invalid)
        chunks = context.chunk_text(
            segment_raw,
            method=method,
            max_size=max_size,
            overlap=overlap,
            language=language,
            **method_opts,
        )
        for ch in chunks:
            ch_text = ch if isinstance(ch, str) else str(ch)
            out_chunks.append(
                {
                    "type": "text",
                    "text": ch_text,
                    "metadata": {
                        "method": method,
                        "start_offset": None,
                        "end_offset": None,
                        "language": language,
                        "paragraph_kind": kind,
                        "offsets_valid": False,
                    },
                }
            )
        return {
            "kind": kind,
            "start_offset": start,
            "end_offset": end,
            "chunks": out_chunks,
            "children": [],
        }

    # Method-aware offset mapping to avoid misplacing spans on repeated content
    try:
        if method in ("words", "sentences"):
            # Use chunk_text_with_metadata to keep offsets aligned with overlap clamping
            try:
                meta_results = context.chunk_text_with_metadata(
                    segment_raw,
                    method=method,
                    max_size=max_size,
                    overlap=overlap,
                    language=language,
                    **method_opts,
                )
                for res in meta_results or []:
                    local_start = getattr(res.metadata, "start_char", None)
                    local_end = getattr(res.metadata, "end_char", None)
                    if not isinstance(local_start, int) or not isinstance(local_end, int):
                        continue
                    _gstart = start + local_start
                    _gend = start + local_end
                    exact_text = output_text[_gstart:_gend]
                    out_chunks.append(
                        {
                            "type": "text",
                            "text": exact_text,
                            "metadata": {
                                "method": method,
                                "start_offset": _gstart,
                                "end_offset": _gend,
                                "language": language,
                                "paragraph_kind": kind,
                            },
                        }
                    )
            except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"{method} metadata mapping failed, using fallback: {e}")
                # Fallback: bound search within the segment using a rolling cursor
                if chunks is None:
                    chunks = context.chunk_text(
                        segment_raw,
                        method=method,
                        max_size=max_size,
                        overlap=overlap,
                        language=language,
                        **method_opts,
                    )
                cursor = 0
                for ch in chunks or []:
                    ch_text = ch if isinstance(ch, str) else str(ch)
                    idx = segment_clean.find(ch_text, cursor)
                    if idx == -1:
                        idx = cursor
                    _gstart = min(start + idx, end)
                    _gend = min(start + idx + len(ch_text), end)
                    if _gend < _gstart:
                        _gend = _gstart
                    exact_text = output_text[_gstart:_gend]
                    out_chunks.append(
                        {
                            "type": "text",
                            "text": exact_text,
                            "metadata": {
                                "method": method,
                                "start_offset": _gstart,
                                "end_offset": _gend,
                                "language": language,
                                "paragraph_kind": kind,
                            },
                        }
                    )
                    cursor = idx + len(ch_text)
        elif method == "tokens":
            # Prefer precise offsets from token strategy metadata
            try:
                meta_results = context.chunk_text_with_metadata(
                    segment_raw,
                    method=ChunkingMethod.TOKENS.value,
                    max_size=max_size,
                    overlap=overlap,
                    language=language,
                    **method_opts,
                )
                for res in meta_results or []:
                    local_start = getattr(res.metadata, "start_char", None)
                    local_end = getattr(res.metadata, "end_char", None)
                    if not isinstance(local_start, int) or not isinstance(local_end, int):
                        continue
                    global_start = start + local_start
                    global_end = start + local_end
                    # Emit exact source slice to guarantee fidelity
                    exact_text = output_text[global_start:global_end]
                    out_chunks.append(
                        {
                            "type": "text",
                            "text": exact_text,
                            "metadata": {
                                "method": method,
                                "start_offset": global_start,
                                "end_offset": global_end,
                                "language": language,
                                "paragraph_kind": kind,
                            },
                        }
                    )
            except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Token metadata mapping failed, using fallback: {e}")
                # Fallback to naive mapping below
                if chunks is None:
                    chunks = context.chunk_text(
                        segment_raw,
                        method=method,
                        max_size=max_size,
                        overlap=overlap,
                        language=language,
                        **method_opts,
                    )
                cursor = 0
                for ch in chunks or []:
                    ch_text = ch if isinstance(ch, str) else str(ch)
                    idx = segment_clean.find(ch_text, cursor)
                    if idx == -1:
                        idx = cursor
                    _gstart = min(start + idx, end)
                    _gend = min(start + idx + len(ch_text), end)
                    if _gend < _gstart:
                        _gend = _gstart
                    exact_text = output_text[_gstart:_gend]
                    out_chunks.append(
                        {
                            "type": "text",
                            "text": exact_text,
                            "metadata": {
                                "method": method,
                                "start_offset": _gstart,
                                "end_offset": _gend,
                                "language": language,
                                "paragraph_kind": kind,
                            },
                        }
                    )
                    cursor = idx + len(ch_text)
        elif method == "structure_aware":
            # Carry block span directly as a precise chunk for structure-aware mode
            exact_text = output_text[start:end]
            out_chunks.append(
                {
                    "type": "text",
                    "text": exact_text,
                    "metadata": {
                        "method": method,
                        "start_offset": start,
                        "end_offset": end,
                        "language": language,
                        "paragraph_kind": kind,
                    },
                }
            )
        else:
            # Fallback: bound search within the segment using a rolling cursor
            if chunks is None:
                chunks = context.chunk_text(
                    segment_raw,
                    method=method,
                    max_size=max_size,
                    overlap=overlap,
                    language=language,
                    **method_opts,
                )
            cursor = 0
            for ch in chunks or []:
                ch_text = ch if isinstance(ch, str) else str(ch)
                idx = segment_clean.find(ch_text, cursor)
                if idx == -1:
                    # If not found, place at cursor to keep monotonicity
                    idx = cursor
                _gstart = min(start + idx, end)
                _gend = min(start + idx + len(ch_text), end)
                if _gend < _gstart:
                    _gend = _gstart
                exact_text = output_text[_gstart:_gend]
                out_chunks.append(
                    {
                        "type": "text",
                        "text": exact_text,
                        "metadata": {
                            "method": method,
                            "start_offset": _gstart,
                            "end_offset": _gend,
                            "language": language,
                            "paragraph_kind": kind,
                        },
                    }
                )
                cursor = idx + len(ch_text)
    except _CHUNKER_NONCRITICAL_EXCEPTIONS as e:
        # As a last resort, return chunks with naive offsets bounded to this block
        logger.warning(f"Offset mapping failed for method={method}: {e}; using naive offsets")
        if chunks is None:
            chunks = context.chunk_text(
                segment_raw,
                method=method,
                max_size=max_size,
                overlap=overlap,
                language=language,
                **method_opts,
            )
        cursor = 0
        for ch in chunks or []:
            ch_text = ch if isinstance(ch, str) else str(ch)
            local_end = min(len(segment_clean), cursor + len(ch_text))
            _gstart = min(start + cursor, end)
            _gend = min(start + local_end, end)
            if _gend < _gstart:
                _gend = _gstart
            exact_text = output_text[_gstart:_gend]
            out_chunks.append(
                {
                    "type": "text",
                    "text": exact_text,
                    "metadata": {
                        "method": method,
                        "start_offset": _gstart,
                        "end_offset": _gend,
                        "language": language,
                        "paragraph_kind": kind,
                    },
                }
            )
            cursor += len(ch_text)

    return {
        "kind": kind,
        "start_offset": start,
        "end_offset": end,
        "chunks": out_chunks,
        "children": [],
    }
