"""Alignment utilities for audiobook subtitle generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audiobook_schemas import AlignmentPayload, AlignmentWord


@dataclass(frozen=True)
class AlignmentAnchor:
    offset: int
    time_ms: int


def apply_alignment_anchors(
    words: Sequence[AlignmentWord],
    anchors: Sequence[AlignmentAnchor],
) -> list[AlignmentWord]:
    if not anchors:
        return list(words)
    if not words:
        return []

    ordered = sorted(anchors, key=lambda a: a.offset)
    valid_pairs: list[tuple[AlignmentAnchor, int]] = []
    for anchor in ordered:
        idx = _find_anchor_word_index(words, anchor.offset)
        if idx is None:
            logger.warning("alignment anchor ignored: no word at offset {}", anchor.offset)
            continue
        valid_pairs.append((anchor, idx))

    adjusted: list[AlignmentWord] = []
    active_delta = 0
    last_adjusted_end: int | None = None
    next_anchor_idx = 0
    next_word_boundary = len(words)
    if valid_pairs:
        next_word_boundary = valid_pairs[0][1]

    for i, word in enumerate(words):
        while next_anchor_idx < len(valid_pairs) and i == next_word_boundary:
            anchor, word_index = valid_pairs[next_anchor_idx]
            next_anchor_idx += 1
            next_word_boundary = valid_pairs[next_anchor_idx][1] if next_anchor_idx < len(valid_pairs) else len(words)
            if last_adjusted_end is not None and anchor.time_ms < last_adjusted_end:
                logger.warning("alignment anchor ignored: non-monotonic at {}", anchor.offset)
                continue
            active_delta = anchor.time_ms - word.start_ms

        new_start = word.start_ms + active_delta
        new_end = word.end_ms + active_delta
        if new_start < 0 or new_end < 0:
            logger.warning("alignment anchor produced negative timestamp at offset {}", word.char_start)
            new_start = max(0, new_start)
            new_end = max(new_start, new_end)
        adjusted.append(word.model_copy(update={"start_ms": new_start, "end_ms": new_end}))
        last_adjusted_end = new_end

    return adjusted


def apply_alignment_anchors_to_payload(
    payload: AlignmentPayload,
    anchors: Iterable[AlignmentAnchor],
) -> AlignmentPayload:
    adjusted_words = apply_alignment_anchors(payload.words, list(anchors))
    return payload.model_copy(update={"words": adjusted_words})


def scale_alignment_payload(payload: AlignmentPayload, speed_ratio: float) -> AlignmentPayload:
    if speed_ratio <= 0:
        raise ValueError("speed_ratio must be positive")
    if speed_ratio == 1.0:
        return payload
    scale = 1.0 / speed_ratio
    words: list[AlignmentWord] = []
    for word in payload.words:
        start_ms = int(round(word.start_ms * scale))
        end_ms = int(round(word.end_ms * scale))
        if end_ms < start_ms:
            end_ms = start_ms
        words.append(word.model_copy(update={"start_ms": start_ms, "end_ms": end_ms}))
    return payload.model_copy(update={"words": words})


def stitch_alignment_payloads(
    payloads: Sequence[AlignmentPayload],
    *,
    segment_offsets_ms: Sequence[int],
    segment_offsets_chars: Sequence[int] | None = None,
) -> AlignmentPayload:
    if not payloads:
        raise ValueError("payloads must not be empty")
    if len(payloads) != len(segment_offsets_ms):
        raise ValueError("segment_offsets_ms must align with payloads")
    if segment_offsets_chars is not None and len(segment_offsets_chars) != len(payloads):
        raise ValueError("segment_offsets_chars must align with payloads")
    engine = payloads[0].engine
    sample_rate = payloads[0].sample_rate
    words: list[AlignmentWord] = []
    for idx, payload in enumerate(payloads):
        if payload.engine != engine or payload.sample_rate != sample_rate:
            logger.warning("alignment stitch: mismatched engine or sample_rate in segment {}", idx)
        time_offset = int(segment_offsets_ms[idx])
        char_offset = int(segment_offsets_chars[idx]) if segment_offsets_chars is not None else 0
        for word in payload.words:
            start_ms = word.start_ms + time_offset
            end_ms = word.end_ms + time_offset
            char_start = word.char_start + char_offset if word.char_start is not None else None
            char_end = word.char_end + char_offset if word.char_end is not None else None
            if end_ms < start_ms:
                end_ms = start_ms
            words.append(
                word.model_copy(
                    update={
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )
            )
    return AlignmentPayload(engine=engine, sample_rate=sample_rate, words=words)


def _find_anchor_word_index(words: Sequence[AlignmentWord], offset: int) -> int | None:
    for i, word in enumerate(words):
        if word.char_start is None:
            continue
        if word.char_start >= offset:
            return i
    return None
