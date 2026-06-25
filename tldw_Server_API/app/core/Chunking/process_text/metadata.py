from __future__ import annotations

import hashlib
from contextlib import suppress
from typing import Any

from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.process_text.models import (
    NormalizedChunk,
    PreparedText,
    ResolvedProcessOptions,
)


def finalize_chunks(
    *,
    original_text: str,
    chunks: list[NormalizedChunk],
    prepared: PreparedText,
    resolved: ResolvedProcessOptions,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(chunks)
    time_segments = _parse_time_segments(prepared.options.get("timecode_map"))

    for index, chunk in enumerate(chunks):
        txt = chunk.text
        md = dict(chunk.metadata)
        _restore_prefix_offsets(md, prepared.prefix_offset)

        md.setdefault("chunk_index", index + 1)
        md.setdefault("total_chunks", total)
        md.setdefault("chunk_method", resolved.method)
        md.setdefault("max_size_setting", resolved.max_size)
        md.setdefault("overlap_setting", resolved.overlap)
        md.setdefault("max_size", resolved.max_size)
        md.setdefault("overlap", resolved.overlap)
        md.setdefault("language", resolved.language)
        md.setdefault("adaptive_chunking_used", resolved.adaptive)
        if resolved.method_lower in ("code", "code_ast"):
            effective_code_mode = resolved.code_mode_for_method
            if effective_code_mode is None:
                effective_code_mode = "ast" if resolved.method_lower == "code_ast" else "auto"
            md.setdefault("code_mode_used", effective_code_mode)

        rel = _relative_position(
            metadata=md,
            index=index,
            total=total,
            original_text=original_text,
            time_segments=time_segments,
        )
        md.setdefault("relative_position", rel)

        if prepared.json_meta:
            md.setdefault("initial_document_json_metadata", prepared.json_meta)
        if prepared.header_text:
            md.setdefault("initial_document_header_text", prepared.header_text)

        with suppress(CHUNKER_NONCRITICAL_EXCEPTIONS):
            md.setdefault(
                "chunk_content_hash",
                hashlib.md5(txt.encode("utf-8"), usedforsecurity=False).hexdigest(),
            )

        md.setdefault("origin", "unified_chunker")

        out.append({"text": txt, "metadata": md})
    return out


def _restore_prefix_offsets(metadata: dict[str, Any], prefix_offset: int) -> None:
    if not prefix_offset:
        return
    for key in ("start_offset", "end_offset", "start_char", "end_char"):
        value = metadata.get(key)
        if isinstance(value, int):
            metadata[key] = value + prefix_offset


def _parse_time_segments(segs: Any) -> list[tuple[int, int, float, float]] | None:
    try:
        if not isinstance(segs, list):
            return None
        time_segments: list[tuple[int, int, float, float]] = []
        for segment in segs:
            if not isinstance(segment, dict):
                continue
            start_offset = segment.get("start_offset")
            end_offset = segment.get("end_offset")
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")
            if (
                isinstance(start_offset, int)
                and isinstance(end_offset, int)
                and isinstance(start_time, (int, float))
                and isinstance(end_time, (int, float))
            ):
                time_segments.append(
                    (start_offset, end_offset, float(start_time), float(end_time))
                )
        if not time_segments:
            return None
        return sorted(time_segments, key=lambda item: item[0])
    except CHUNKER_NONCRITICAL_EXCEPTIONS:
        return None


def _relative_position(
    *,
    metadata: dict[str, Any],
    index: int,
    total: int,
    original_text: str,
    time_segments: list[tuple[int, int, float, float]] | None,
) -> float:
    try:
        start = metadata.get("start_offset")
        end = metadata.get("end_offset")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            mid = 0.5 * (float(start) + float(end))
            rel = mid / max(1.0, float(len(original_text)))
            if time_segments is not None and (
                "start_time" not in metadata or "end_time" not in metadata
            ):
                _apply_time_segments(metadata, start, end, time_segments)
        else:
            rel = _fallback_relative_position(index, total)
    except CHUNKER_NONCRITICAL_EXCEPTIONS:
        rel = _fallback_relative_position(index, total)
    return rel


def _fallback_relative_position(index: int, total: int) -> float:
    return (index + 1) / total if total > 0 else 0.0


def _apply_time_segments(
    metadata: dict[str, Any],
    chunk_start: int,
    chunk_end: int,
    time_segments: list[tuple[int, int, float, float]],
) -> None:
    try:
        chunk_start_time = None
        chunk_end_time = None
        for start_offset, end_offset, start_time, end_time in time_segments:
            if chunk_end <= start_offset:
                break
            if chunk_start >= end_offset:
                continue
            overlap_start = max(chunk_start, start_offset)
            overlap_end = min(chunk_end, end_offset)
            if overlap_end <= overlap_start:
                continue
            seg_len = max(1.0, float(end_offset - start_offset))
            seg_duration = float(end_time - start_time)
            frac_start = (overlap_start - start_offset) / seg_len
            frac_end = (overlap_end - start_offset) / seg_len
            mapped_start = start_time + frac_start * seg_duration
            mapped_end = start_time + frac_end * seg_duration
            if chunk_start_time is None:
                chunk_start_time = mapped_start
            chunk_end_time = mapped_end
            if overlap_end >= chunk_end:
                break
        if chunk_start_time is not None and "start_time" not in metadata:
            metadata["start_time"] = round(chunk_start_time, 3)
        if chunk_end_time is not None and "end_time" not in metadata:
            metadata["end_time"] = round(chunk_end_time, 3)
    except CHUNKER_NONCRITICAL_EXCEPTIONS:
        pass
