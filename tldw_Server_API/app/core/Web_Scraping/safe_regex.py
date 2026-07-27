"""Bounded execution for untrusted regular expressions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import regex

_MAX_PATTERN_CHARS = 4_096
_MAX_INPUT_CHARS = 1_000_000
_MAX_TIMEOUT_S = 0.100
_REGEX_ERRORS = (regex.error, TypeError, ValueError, OverflowError, RecursionError)


@dataclass(frozen=True, slots=True)
class SafeRegexLimits:
    max_pattern_chars: int = 4_096
    max_input_chars: int = 8_192
    timeout_s: float = 0.100


@dataclass(frozen=True, slots=True)
class SafeRegexResult:
    matched: bool
    match: Any | None = None
    code: str | None = None


_FLAG_MAP = (
    (int(re.IGNORECASE), int(regex.IGNORECASE)),
    (int(re.LOCALE), int(regex.LOCALE)),
    (int(re.MULTILINE), int(regex.MULTILINE)),
    (int(re.DOTALL), int(regex.DOTALL)),
    (int(re.UNICODE), int(regex.UNICODE)),
    (int(re.VERBOSE), int(regex.VERBOSE)),
    (int(re.ASCII), int(regex.ASCII)),
)
_SUPPORTED_STDLIB_FLAGS = 0
for _stdlib_flag, _engine_flag in _FLAG_MAP:
    _SUPPORTED_STDLIB_FLAGS |= _stdlib_flag


def _normalize_flags(flags: int) -> int | None:
    if isinstance(flags, bool) or not isinstance(flags, int):
        return None
    raw_flags = int(flags)
    if raw_flags < 0 or raw_flags & ~_SUPPORTED_STDLIB_FLAGS:
        return None

    normalized = 0
    for stdlib_flag, engine_flag in _FLAG_MAP:
        if raw_flags & stdlib_flag:
            normalized |= engine_flag
    return normalized


def _validated_limits(limits: SafeRegexLimits) -> tuple[int, int, float] | None:
    if not isinstance(limits, SafeRegexLimits):
        return None
    if (
        isinstance(limits.max_pattern_chars, bool)
        or not isinstance(limits.max_pattern_chars, int)
        or limits.max_pattern_chars <= 0
    ):
        return None
    if (
        isinstance(limits.max_input_chars, bool)
        or not isinstance(limits.max_input_chars, int)
        or limits.max_input_chars <= 0
    ):
        return None
    if (
        isinstance(limits.timeout_s, bool)
        or not isinstance(limits.timeout_s, (int, float))
        or not math.isfinite(limits.timeout_s)
        or limits.timeout_s <= 0
    ):
        return None
    return (
        min(limits.max_pattern_chars, _MAX_PATTERN_CHARS),
        min(limits.max_input_chars, _MAX_INPUT_CHARS),
        min(float(limits.timeout_s), _MAX_TIMEOUT_S),
    )


def _compile_pattern(pattern: str, flags: int) -> Any:
    return regex.compile(pattern, flags)


def search_untrusted(
    pattern: str,
    value: str,
    *,
    flags: int = 0,
    limits: SafeRegexLimits = SafeRegexLimits(),
) -> SafeRegexResult:
    """Search with fixed size and execution bounds, returning stable error codes."""
    validated_limits = _validated_limits(limits)
    if validated_limits is None or not isinstance(pattern, str) or not isinstance(value, str):
        return SafeRegexResult(matched=False, code="regex_invalid")

    max_pattern_chars, max_input_chars, timeout_s = validated_limits
    if len(pattern) > max_pattern_chars or len(value) > max_input_chars:
        return SafeRegexResult(matched=False, code="regex_too_large")

    normalized_flags = _normalize_flags(flags)
    if normalized_flags is None:
        return SafeRegexResult(matched=False, code="regex_invalid")

    try:
        compiled = _compile_pattern(pattern, normalized_flags)
    except _REGEX_ERRORS:
        return SafeRegexResult(matched=False, code="regex_invalid")

    try:
        match = compiled.search(value, timeout=timeout_s)
    except TimeoutError:
        return SafeRegexResult(matched=False, code="regex_timeout")
    except _REGEX_ERRORS:
        return SafeRegexResult(matched=False, code="regex_invalid")

    return SafeRegexResult(matched=match is not None, match=match)


__all__ = ["SafeRegexLimits", "SafeRegexResult", "search_untrusted"]
