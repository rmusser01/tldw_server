"""Bounded execution for untrusted regular expressions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import regex

_MAX_PATTERN_CHARS = 4_096
_MAX_INPUT_CHARS = 1_000_000
_MAX_REPLACEMENT_CHARS = 4_096
_MAX_SUB_OUTPUT_CHARS = 1_000_000
_MAX_TIMEOUT_S = 0.100
_REGEX_ERRORS = (
    IndexError,
    re.error,
    regex.error,
    TypeError,
    ValueError,
    OverflowError,
    RecursionError,
)


class _RegexOutputTooLarge(Exception):
    pass


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


@dataclass(frozen=True, slots=True)
class SafeRegexSubResult:
    value: str | None = None
    code: str | None = None


_FLAG_MAP = (
    (int(re.IGNORECASE), int(regex.IGNORECASE)),
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


def _prepare_untrusted_pattern(
    pattern: str,
    value: str,
    flags: int,
    limits: SafeRegexLimits,
) -> tuple[Any, float] | str:
    validated_limits = _validated_limits(limits)
    if validated_limits is None or not isinstance(pattern, str) or not isinstance(value, str):
        return "regex_invalid"

    max_pattern_chars, max_input_chars, timeout_s = validated_limits
    if len(pattern) > max_pattern_chars or len(value) > max_input_chars:
        return "regex_too_large"

    normalized_flags = _normalize_flags(flags)
    if normalized_flags is None:
        return "regex_invalid"

    try:
        re.compile(pattern, int(flags))
    except _REGEX_ERRORS:
        return "regex_invalid"

    try:
        compiled = _compile_pattern(pattern, normalized_flags)
    except _REGEX_ERRORS:
        return "regex_invalid"
    return compiled, timeout_s


def search_untrusted(
    pattern: str,
    value: str,
    *,
    flags: int = 0,
    limits: SafeRegexLimits = SafeRegexLimits(),
) -> SafeRegexResult:
    """Search with fixed size and execution bounds, returning stable error codes."""
    prepared = _prepare_untrusted_pattern(pattern, value, flags, limits)
    if isinstance(prepared, str):
        return SafeRegexResult(matched=False, code=prepared)
    compiled, timeout_s = prepared

    try:
        match = compiled.search(value, timeout=timeout_s)
    except TimeoutError:
        return SafeRegexResult(matched=False, code="regex_timeout")
    except _REGEX_ERRORS:
        return SafeRegexResult(matched=False, code="regex_invalid")

    return SafeRegexResult(matched=match is not None, match=match)


def _parse_replacement_template(
    pattern: str,
    repl: str,
    flags: int,
) -> tuple[int, tuple[int, ...]] | None:
    try:
        compiled = re.compile(pattern, int(flags))
    except _REGEX_ERRORS:
        return None

    group_count = compiled.groups
    group_names = compiled.groupindex
    literal_chars = 0
    references: list[int] = []
    index = 0
    while index < len(repl):
        if repl[index] != "\\":
            literal_chars += 1
            index += 1
            continue
        if index + 1 >= len(repl):
            return None

        escaped = repl[index + 1]
        index += 2
        if escaped == "g":
            if index >= len(repl) or repl[index] != "<":
                return None
            closing = repl.find(">", index + 1)
            if closing < 0:
                return None
            reference = repl[index + 1 : closing]
            index = closing + 1
            if reference.isascii() and reference.isdecimal():
                try:
                    group_index = int(reference)
                except ValueError:
                    return None
                if group_index > group_count:
                    return None
            elif reference.isidentifier() and reference in group_names:
                group_index = group_names[reference]
            else:
                return None
            references.append(group_index)
            continue
        if escaped == "0":
            for _offset in range(2):
                if index < len(repl) and repl[index] in "01234567":
                    index += 1
                else:
                    break
            literal_chars += 1
            continue
        if escaped in "123456789":
            digits = escaped
            if index < len(repl) and repl[index] in "0123456789":
                digits += repl[index]
                index += 1
                if (
                    escaped in "01234567"
                    and digits[1] in "01234567"
                    and index < len(repl)
                    and repl[index] in "01234567"
                ):
                    digits += repl[index]
                    index += 1
                    if int(digits, 8) > 0o377:
                        return None
                    literal_chars += 1
                    continue
            group_index = int(digits)
            if group_index > group_count:
                return None
            references.append(group_index)
            continue
        if escaped in "abfnrtv\\":
            literal_chars += 1
            continue
        if escaped.isascii() and escaped.isalpha():
            return None
        literal_chars += 2
    return literal_chars, tuple(references)


def sub_untrusted(
    pattern: str,
    repl: str,
    value: str,
    *,
    flags: int = 0,
    limits: SafeRegexLimits = SafeRegexLimits(),
) -> SafeRegexSubResult:
    """Substitute globally with fixed execution, input, and output bounds."""
    if not isinstance(repl, str):
        return SafeRegexSubResult(code="regex_invalid")
    if len(repl) > _MAX_REPLACEMENT_CHARS:
        return SafeRegexSubResult(code="regex_too_large")
    prepared = _prepare_untrusted_pattern(pattern, value, flags, limits)
    if isinstance(prepared, str):
        return SafeRegexSubResult(code=prepared)
    compiled, timeout_s = prepared
    replacement_template = _parse_replacement_template(pattern, repl, flags)
    if replacement_template is None:
        return SafeRegexSubResult(code="regex_invalid")
    literal_chars, references = replacement_template

    written = 0
    last_end = 0

    def _bounded_replacement(match: Any) -> str:
        nonlocal last_end, written
        start, end = match.span()
        unmatched_chars = max(0, start - last_end)
        referenced_chars = 0
        for group_index in references:
            group_start, group_end = match.span(group_index)
            if group_start >= 0:
                referenced_chars += group_end - group_start
        expansion_chars = literal_chars + referenced_chars
        if written + unmatched_chars + expansion_chars > _MAX_SUB_OUTPUT_CHARS:
            raise _RegexOutputTooLarge

        expanded = match.expand(repl)
        written += unmatched_chars + len(expanded)
        if written > _MAX_SUB_OUTPUT_CHARS:
            raise _RegexOutputTooLarge
        last_end = end
        return expanded

    try:
        replaced = compiled.sub(_bounded_replacement, value, timeout=timeout_s)
    except _RegexOutputTooLarge:
        return SafeRegexSubResult(code="regex_too_large")
    except TimeoutError:
        return SafeRegexSubResult(code="regex_timeout")
    except _REGEX_ERRORS:
        return SafeRegexSubResult(code="regex_invalid")
    if len(replaced) > _MAX_SUB_OUTPUT_CHARS:
        return SafeRegexSubResult(code="regex_too_large")
    return SafeRegexSubResult(value=replaced)


__all__ = [
    "SafeRegexLimits",
    "SafeRegexResult",
    "SafeRegexSubResult",
    "search_untrusted",
    "sub_untrusted",
]
