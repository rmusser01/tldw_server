"""
Utility helpers for TTS modules.

Currently includes:
- parse_bool: robust conversion of common string/numeric values to boolean.
- estimate_max_new_tokens: heuristic sizing for TTS generation.
- resolve_qwen3_runtime_name: shared Qwen3 runtime selection logic.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import platform
import re
import unicodedata
from collections.abc import Callable, Iterator
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy

FALSY_STRINGS = {"0", "false", "no", "n", "off", "none", "null", ""}
_TTS_TEXT_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>[\s\S]*?</(?:think|thinking|reasoning)>\s*",
    flags=re.IGNORECASE,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|thinking|reasoning)>", flags=re.IGNORECASE)


def parse_bool(value: Any, default: bool | None = False) -> bool:
    """Parse a value into a boolean in a tolerant, explicit way.

    Behavior:
    - bool -> returned as-is
    - int/float -> 0 is False, non-zero True
    - str -> case-insensitive check against common truthy/falsy tokens;
             unknown strings return `default`
    - None -> returns `default` (defaults to False)
    - other types -> bool(value) if default is None else default
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default) if default is not None else False
    if isinstance(value, (int, float)):
        try:
            return int(value) != 0
        except (ValueError, TypeError, OverflowError):
            return bool(default) if default is not None else False
    if isinstance(value, str):
        s = value.strip().lower()
        if is_truthy(s):
            return True
        if s in FALSY_STRINGS:
            return False
        # Unknown string token
        return bool(default) if default is not None else False
    # Fallback for other types
    return bool(value) if default is None else bool(default)


async def run_tts_blocking_call(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous local TTS runtime call without blocking the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def run_tts_blocking_next(iterator: Iterator[Any], sentinel: Any) -> Any:
    """Fetch the next item from a synchronous iterator without blocking the event loop."""
    def _next_item() -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    return await asyncio.to_thread(_next_item)


def estimate_max_new_tokens(
    text: str,
    tokens_per_char: float = 2.5,
    safety: float = 1.3,
    min_tokens: int = 256,
    max_cap: int = 4096,
) -> int:
    """Estimate a safe max_new_tokens based on text length."""
    try:
        length = len(text or "")
    except (TypeError, ValueError):
        length = 0
    try:
        est = math.ceil(length * float(tokens_per_char) * float(safety))
    except (OverflowError, TypeError, ValueError):
        est = min_tokens
    try:
        min_tokens = int(min_tokens)
    except (OverflowError, TypeError, ValueError):
        min_tokens = 0
    try:
        max_cap = int(max_cap)
    except (OverflowError, TypeError, ValueError):
        max_cap = 4096
    if max_cap <= 0:
        max_cap = 4096
    if min_tokens < 0:
        min_tokens = 0
    return max(min_tokens, min(est, max_cap))


def resolve_qwen3_runtime_name(configured_runtime: Any) -> str:
    """Resolve the effective Qwen3 runtime for the current host."""
    normalized = str(configured_runtime or "auto").strip().lower()
    if normalized in {"upstream", "mlx", "remote"}:
        return normalized
    if platform.system() == "Darwin" and platform.machine().lower() == "arm64":
        return "mlx"
    return "upstream"


def clean_text_for_tts(text: str | None) -> str:
    """Normalize text for more natural TTS output.

    This helper applies conservative speech-focused cleanup:
    - remove common reasoning blocks/tags
    - flatten line breaks
    - replace symbols that are often read awkwardly
    - collapse excess whitespace
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    cleaned = _REASONING_BLOCK_RE.sub("", text)
    cleaned = _REASONING_TAG_RE.sub("", cleaned)
    # Some upstream adapters may pass escaped newlines as literal "\\n".
    cleaned = cleaned.replace("\\r\\n", " ").replace("\\r", " ").replace("\\n", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("+", " plus ")
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("\u2014", ", ")
    cleaned = cleaned.replace("\u2013", ", ")
    cleaned = _TTS_TEXT_WS_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_tts_history_text(text: str | None) -> str:
    """Normalize text for stable TTS history hashing.

    Steps:
    - Unicode NFKC normalization
    - Normalize newlines to LF
    - Trim leading/trailing whitespace
    - Collapse internal whitespace to single spaces
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    normalized = _TTS_TEXT_WS_RE.sub(" ", normalized).strip()
    return normalized


def tts_history_text_length(text: str | None) -> int:
    """Return length of the normalized text (Unicode codepoints)."""
    return len(normalize_tts_history_text(text))


def compute_tts_history_text_hash(text: str | None, secret: str | None = None) -> str:
    """Return HMAC-SHA256 of normalized text for TTS history hashing."""
    key = secret or os.getenv("TTS_HISTORY_HASH_KEY")
    if not key:
        raise ValueError("TTS_HISTORY_HASH_KEY is required to compute TTS history text hash")
    normalized = normalize_tts_history_text(text)
    return hmac.new(key.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def build_tts_segments_payload(
    raw_segments: Any,
    *,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any] | None:
    """Build normalized segments payload with summary + truncation policy."""
    if not raw_segments or not isinstance(raw_segments, list):
        return None
    segments: list[dict[str, Any]] = [s for s in raw_segments if isinstance(s, dict)]
    if not segments:
        return None

    success = sum(1 for seg in segments if seg.get("status") == "success")
    failed = sum(1 for seg in segments if seg.get("status") == "failed")
    attempts_vals = [
        int(seg.get("attempts"))
        for seg in segments
        if isinstance(seg.get("attempts"), (int, float))
    ]
    max_attempts = max(attempts_vals) if attempts_vals else None
    durations = [
        int(seg.get("duration_ms"))
        for seg in segments
        if isinstance(seg.get("duration_ms"), (int, float))
    ]
    total_duration_ms = sum(durations) if durations else None

    payload: dict[str, Any] = {
        "segments": segments,
        "summary": {
            "total": len(segments),
            "success": success,
            "failed": failed,
            "total_duration_ms": total_duration_ms,
            "max_attempts": max_attempts,
        },
        "truncated": False,
    }
    return _truncate_segments_payload(payload, max_bytes=max_bytes)


def _truncate_segments_payload(payload: dict[str, Any], *, max_bytes: int) -> dict[str, Any]:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return payload

    def _size_bytes(obj: dict[str, Any]) -> int:
        try:
            return len(json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        except (OverflowError, TypeError, UnicodeError, ValueError):
            return max_bytes + 1

    if _size_bytes(payload) <= max_bytes:
        return payload

    indices = list(range(len(segments)))
    failed_indices = [i for i in indices if segments[i].get("status") == "failed"]
    keep_indices = failed_indices[-256:]

    def _build(indices_to_keep: list[int]) -> dict[str, Any]:
        kept = [segments[i] for i in indices_to_keep]
        return {
            "segments": kept,
            "summary": payload.get("summary"),
            "truncated": True,
        }

    keep_indices = sorted(keep_indices)
    candidate = _build(keep_indices)
    if _size_bytes(candidate) > max_bytes:
        while keep_indices and _size_bytes(candidate) > max_bytes:
            keep_indices.pop(0)
            candidate = _build(keep_indices)
        return candidate

    success_indices = [i for i in indices if segments[i].get("status") != "failed"]
    for idx in reversed(success_indices):
        if idx in keep_indices:
            continue
        trial_indices = sorted(keep_indices + [idx])
        trial = _build(trial_indices)
        if _size_bytes(trial) <= max_bytes:
            keep_indices = trial_indices
            candidate = trial
    return candidate
