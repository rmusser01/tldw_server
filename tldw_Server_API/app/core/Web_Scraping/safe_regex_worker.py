"""Isolated stdlib-regex worker and shared replacement-template parser."""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

MAX_PATTERN_CHARS = 4_096
MAX_INPUT_CHARS = 1_000_000
MAX_REPLACEMENT_CHARS = 4_096
MAX_SUB_OUTPUT_CHARS = 1_000_000
MAX_REQUEST_BYTES = 12 * (MAX_PATTERN_CHARS + MAX_INPUT_CHARS + MAX_REPLACEMENT_CHARS) + 512
MAX_SEARCH_RESPONSE_BYTES = 131_072
MAX_SUB_RESPONSE_BYTES = 12 * MAX_SUB_OUTPUT_CHARS + 128
MAX_RESPONSE_BYTES = max(MAX_SEARCH_RESPONSE_BYTES, MAX_SUB_RESPONSE_BYTES)
MAX_WORKER_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
MAX_WORKER_CPU_SECONDS = 1
SUPPORTED_FLAGS = int(re.IGNORECASE | re.MULTILINE | re.DOTALL | re.UNICODE | re.VERBOSE | re.ASCII)

_RegexDialect = Literal["stdlib", "regex"]
_RESOURCE_LIMIT_ERROR = "worker resource limits could not be applied"


class _OutputTooLarge(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ParsedReplacement:
    literal_chars: int
    references: tuple[int, ...]
    execution_repl: str


def parse_replacement_template(
    repl: str,
    *,
    group_count: int,
    group_names: Mapping[str, int],
    dialect: _RegexDialect,
    legacy_numeric_group_ids: bool = False,
) -> ParsedReplacement | None:
    """Parse replacement grammar without expanding attacker-controlled matches."""
    literal_chars = 0
    references: list[int] = []
    normalizations: list[tuple[int, int, str]] = []
    index = 0
    while index < len(repl):
        if repl[index] != "\\":
            literal_chars += 1
            index += 1
            continue
        if index + 1 >= len(repl):
            return None

        escape_start = index
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
            numeric_reference = False
            if dialect == "stdlib" and legacy_numeric_group_ids:
                try:
                    group_index = int(reference)
                except ValueError:
                    pass
                else:
                    if group_index < 0:
                        return None
                    numeric_reference = True
            elif reference.isascii() and reference.isdecimal():
                group_index = int(reference)
                numeric_reference = True

            if numeric_reference:
                if group_index > group_count:
                    return None
                if not (reference.isascii() and reference.isdecimal()):
                    normalizations.append((escape_start, index, "\\g<" + str(group_index) + ">"))
            elif reference.isidentifier() and reference in group_names:
                group_index = group_names[reference]
            else:
                return None
            references.append(group_index)
            continue
        if dialect == "regex" and escaped in {"x", "u", "U"}:
            width = {"x": 2, "u": 4, "U": 8}[escaped]
            digits = repl[index : index + width]
            if len(digits) != width or any(char not in "0123456789abcdefABCDEF" for char in digits):
                return None
            try:
                chr(int(digits, 16))
            except ValueError:
                return None
            literal_chars += 1
            index += width
            continue
        if dialect == "regex" and escaped == "N":
            if index >= len(repl) or repl[index] != "{":
                return None
            closing = index + 1
            while closing < len(repl) and repl[closing].isascii() and (repl[closing].isalpha() or repl[closing] == " "):
                closing += 1
            if closing >= len(repl) or repl[closing] != "}":
                return None
            try:
                named_char = unicodedata.lookup(repl[index + 1 : closing])
            except KeyError:
                return None
            if len(named_char) != 1:
                return None
            literal_chars += 1
            index = closing + 1
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
                    maximum = 0o777 if dialect == "regex" else 0o377
                    if int(digits, 8) > maximum:
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

    execution_repl = repl
    for start, end, normalized in reversed(normalizations):
        execution_repl = execution_repl[:start] + normalized + execution_repl[end:]
    return ParsedReplacement(
        literal_chars=literal_chars,
        references=tuple(references),
        execution_repl=execution_repl,
    )


def _apply_resource_limits() -> dict[str, bool]:
    applied = {"address_space": False, "cpu": False}
    if os.name != "posix":
        return applied
    try:
        import resource
    except ImportError:
        raise RuntimeError(_RESOURCE_LIMIT_ERROR) from None

    limits = (
        ("address_space", getattr(resource, "RLIMIT_AS", None), MAX_WORKER_ADDRESS_SPACE_BYTES),
        ("cpu", getattr(resource, "RLIMIT_CPU", None), MAX_WORKER_CPU_SECONDS),
    )
    for name, resource_id, desired_soft in limits:
        if resource_id is None:
            if name == "address_space" and sys.platform == "darwin":
                continue
            raise RuntimeError(_RESOURCE_LIMIT_ERROR)
        try:
            current_soft, current_hard = resource.getrlimit(resource_id)
            infinity = getattr(resource, "RLIM_INFINITY", -1)
            hard_allows = current_hard == infinity or current_hard >= desired_soft
            if not hard_allows:
                desired_soft = current_hard
            if current_soft == infinity or current_soft > desired_soft:
                resource.setrlimit(resource_id, (desired_soft, current_hard))
            verified_soft, _verified_hard = resource.getrlimit(resource_id)
            if verified_soft == infinity or verified_soft > desired_soft:
                raise RuntimeError(_RESOURCE_LIMIT_ERROR)
        except (OSError, ValueError, RuntimeError):
            if name == "address_space" and sys.platform == "darwin":
                continue
            raise RuntimeError(_RESOURCE_LIMIT_ERROR) from None
        applied[name] = True
    return applied


def _emit(payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = b'{"status":"invalid"}\n'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _validated_request() -> dict[str, object]:
    raw_request = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if not raw_request.endswith(b"\n") or len(raw_request) > MAX_REQUEST_BYTES:
        raise ValueError
    request = json.loads(raw_request)
    if not isinstance(request, dict):
        raise ValueError
    operation = request.get("operation", "search")
    expected_keys = (
        {"pattern", "value", "flags"}
        if operation == "search"
        else {
            "operation",
            "pattern",
            "repl",
            "value",
            "flags",
            "legacy_numeric_group_ids",
            "max_output_chars",
        }
    )
    if operation not in {"search", "sub"} or set(request) != expected_keys:
        raise ValueError
    pattern = request["pattern"]
    value = request["value"]
    flags = request["flags"]
    if (
        not isinstance(pattern, str)
        or not isinstance(value, str)
        or isinstance(flags, bool)
        or not isinstance(flags, int)
        or flags < 0
        or flags & ~SUPPORTED_FLAGS
        or len(pattern) > MAX_PATTERN_CHARS
        or len(value) > MAX_INPUT_CHARS
    ):
        raise ValueError
    return request


def _run_search(request: dict[str, object]) -> None:
    pattern = request["pattern"]
    value = request["value"]
    flags = request["flags"]
    if not isinstance(pattern, str) or not isinstance(value, str) or not isinstance(flags, int):
        raise ValueError
    match = re.search(pattern, value, flags)
    if match is None:
        _emit({"status": "no_match"})
        return
    _emit(
        {
            "status": "match",
            "spans": [match.span(index) for index in range(match.re.groups + 1)],
        }
    )


def _run_substitution(request: dict[str, object]) -> None:
    pattern = request["pattern"]
    repl = request["repl"]
    value = request["value"]
    flags = request["flags"]
    legacy_numeric_group_ids = request["legacy_numeric_group_ids"]
    max_output_chars = request["max_output_chars"]
    if (
        not isinstance(pattern, str)
        or not isinstance(repl, str)
        or not isinstance(value, str)
        or isinstance(flags, bool)
        or not isinstance(flags, int)
        or not isinstance(legacy_numeric_group_ids, bool)
        or isinstance(max_output_chars, bool)
        or not isinstance(max_output_chars, int)
        or len(repl) > MAX_REPLACEMENT_CHARS
        or max_output_chars < 0
        or max_output_chars > MAX_SUB_OUTPUT_CHARS
    ):
        raise ValueError

    compiled = re.compile(pattern, flags)
    parsed = parse_replacement_template(
        repl,
        group_count=compiled.groups,
        group_names=compiled.groupindex,
        dialect="stdlib",
        legacy_numeric_group_ids=legacy_numeric_group_ids,
    )
    if parsed is None:
        raise ValueError

    written = 0
    last_end = 0

    def bounded_replacement(match: re.Match[str]) -> str:
        nonlocal last_end, written
        start, end = match.span()
        unmatched_chars = max(0, start - last_end)
        referenced_chars = 0
        for group_index in parsed.references:
            group_start, group_end = match.span(group_index)
            if group_start >= 0:
                referenced_chars += group_end - group_start
        if written + unmatched_chars + parsed.literal_chars + referenced_chars > max_output_chars:
            raise _OutputTooLarge

        expanded = match.expand(parsed.execution_repl)
        written += unmatched_chars + len(expanded)
        if written > max_output_chars:
            raise _OutputTooLarge
        last_end = end
        return expanded

    replaced = compiled.sub(bounded_replacement, value)
    if len(replaced) > max_output_chars:
        raise _OutputTooLarge
    _emit({"status": "sub", "value": replaced})


def main() -> int:
    resource_limits = _apply_resource_limits()
    _emit({"status": "ready", "resource_limits": resource_limits})
    try:
        request = _validated_request()
        if request.get("operation", "search") == "search":
            _run_search(request)
        else:
            _run_substitution(request)
    except _OutputTooLarge:
        _emit({"status": "too_large"})
    except Exception:  # noqa: BLE001 - the process protocol must fail closed.
        _emit({"status": "invalid"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
