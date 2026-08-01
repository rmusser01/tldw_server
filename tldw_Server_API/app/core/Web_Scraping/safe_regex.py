"""Bounded execution for untrusted regular expressions."""

from __future__ import annotations

import json
import math
import operator
import queue
import re
import subprocess  # nosec B404
import sys
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import regex

_MAX_PATTERN_CHARS = 4_096
_MAX_INPUT_CHARS = 1_000_000
_MAX_REPLACEMENT_CHARS = 4_096
_MAX_SUB_OUTPUT_CHARS = 1_000_000
_MAX_TIMEOUT_S = 0.100
_STDLIB_WORKER_STARTUP_TIMEOUT_S = 1.0
_STDLIB_WORKER_REAP_TIMEOUT_S = 0.500
_MAX_WORKER_HANDSHAKE_BYTES = 64
_MAX_WORKER_REQUEST_BYTES = 12 * (_MAX_PATTERN_CHARS + _MAX_INPUT_CHARS + _MAX_REPLACEMENT_CHARS) + 512
_MAX_WORKER_SEARCH_RESPONSE_BYTES = 131_072
_MAX_WORKER_SUB_RESPONSE_BYTES = 12 * _MAX_SUB_OUTPUT_CHARS + 128
_MAX_WORKER_RESPONSE_BYTES = max(_MAX_WORKER_SEARCH_RESPONSE_BYTES, _MAX_WORKER_SUB_RESPONSE_BYTES)
_REGEX_ERRORS = (
    IndexError,
    re.error,
    regex.error,
    TypeError,
    ValueError,
    OverflowError,
    RecursionError,
)
_RegexDialect = Literal["stdlib", "regex"]
_STDLIB_LEGACY_NUMERIC_GROUP_IDS = sys.version_info < (3, 12)

_STDLIB_WORKER_CODE = f"""
import json
import re
import sys

MAX_REQUEST_BYTES = {_MAX_WORKER_REQUEST_BYTES}
MAX_RESPONSE_BYTES = {_MAX_WORKER_RESPONSE_BYTES}
MAX_PATTERN_CHARS = {_MAX_PATTERN_CHARS}
MAX_INPUT_CHARS = {_MAX_INPUT_CHARS}
MAX_REPLACEMENT_CHARS = {_MAX_REPLACEMENT_CHARS}
MAX_SUB_OUTPUT_CHARS = {_MAX_SUB_OUTPUT_CHARS}
SUPPORTED_FLAGS = {int(re.IGNORECASE | re.MULTILINE | re.DOTALL | re.UNICODE | re.VERBOSE | re.ASCII)}


class OutputTooLarge(Exception):
    pass


def emit(payload):
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\\n"
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = b'{{"status":"invalid"}}\\n'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


emit({{"status": "ready"}})
try:
    raw_request = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if (
        not raw_request.endswith(b"\\n")
        or len(raw_request) > MAX_REQUEST_BYTES
    ):
        raise ValueError
    request = json.loads(raw_request)
    operation = request.get("operation", "search")
    expected_keys = (
        {{"pattern", "value", "flags"}}
        if operation == "search"
        else {{
            "operation",
            "pattern",
            "repl",
            "value",
            "flags",
            "legacy_numeric_group_ids",
            "max_output_chars",
        }}
    )
    if operation not in {{"search", "sub"}} or set(request) != expected_keys:
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

    if operation == "search":
        match = re.search(pattern, value, flags)
        if match is None:
            emit({{"status": "no_match"}})
        else:
            emit({{
                "status": "match",
                "spans": [match.span(index) for index in range(match.re.groups + 1)],
            }})
    else:
        repl = request["repl"]
        legacy_numeric_group_ids = request["legacy_numeric_group_ids"]
        max_output_chars = request["max_output_chars"]
        if (
            not isinstance(repl, str)
            or len(repl) > MAX_REPLACEMENT_CHARS
            or not isinstance(legacy_numeric_group_ids, bool)
            or isinstance(max_output_chars, bool)
            or not isinstance(max_output_chars, int)
            or max_output_chars < 0
            or max_output_chars > MAX_SUB_OUTPUT_CHARS
        ):
            raise ValueError

        compiled = re.compile(pattern, flags)
        group_count = compiled.groups
        group_names = compiled.groupindex
        literal_chars = 0
        references = []
        normalizations = []
        index = 0
        while index < len(repl):
            if repl[index] != "\\\\":
                literal_chars += 1
                index += 1
                continue
            if index + 1 >= len(repl):
                raise ValueError

            escape_start = index
            escaped = repl[index + 1]
            index += 2
            if escaped == "g":
                if index >= len(repl) or repl[index] != "<":
                    raise ValueError
                closing = repl.find(">", index + 1)
                if closing < 0:
                    raise ValueError
                reference = repl[index + 1 : closing]
                index = closing + 1
                numeric_reference = False
                if legacy_numeric_group_ids:
                    try:
                        group_index = int(reference)
                    except ValueError:
                        pass
                    else:
                        if group_index < 0:
                            raise ValueError
                        numeric_reference = True
                elif reference.isascii() and reference.isdecimal():
                    group_index = int(reference)
                    numeric_reference = True

                if numeric_reference:
                    if group_index > group_count:
                        raise ValueError
                    if not (reference.isascii() and reference.isdecimal()):
                        normalizations.append(
                            (escape_start, index, "\\\\g<" + str(group_index) + ">")
                        )
                elif reference.isidentifier() and reference in group_names:
                    group_index = group_names[reference]
                else:
                    raise ValueError
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
                            raise ValueError
                        literal_chars += 1
                        continue
                group_index = int(digits)
                if group_index > group_count:
                    raise ValueError
                references.append(group_index)
                continue
            if escaped in "abfnrtv\\\\":
                literal_chars += 1
                continue
            if escaped.isascii() and escaped.isalpha():
                raise ValueError
            literal_chars += 2

        written = 0
        last_end = 0
        execution_repl = repl
        for start, end, normalized in reversed(normalizations):
            execution_repl = execution_repl[:start] + normalized + execution_repl[end:]

        def bounded_replacement(match):
            global written, last_end
            start, end = match.span()
            unmatched_chars = max(0, start - last_end)
            referenced_chars = 0
            for group_index in references:
                group_start, group_end = match.span(group_index)
                if group_start >= 0:
                    referenced_chars += group_end - group_start
            if written + unmatched_chars + literal_chars + referenced_chars > max_output_chars:
                raise OutputTooLarge

            expanded = match.expand(execution_repl)
            written += unmatched_chars + len(expanded)
            if written > max_output_chars:
                raise OutputTooLarge
            last_end = end
            return expanded

        replaced = re.sub(pattern, bounded_replacement, value, flags=flags)
        if len(replaced) > max_output_chars:
            raise OutputTooLarge
        emit({{"status": "sub", "value": replaced}})
except OutputTooLarge:
    emit({{"status": "too_large"}})
except BaseException:
    emit({{"status": "invalid"}})
"""


class _RegexOutputTooLarge(Exception):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class _MatchSnapshot:
    _value: str
    _spans: tuple[tuple[int, int], ...]
    _groupindex: Mapping[str, int]

    def __repr__(self) -> str:
        return "<safe regex match>"

    def __bool__(self) -> bool:
        return True

    def _resolve_group(self, group: int | str) -> int:
        if isinstance(group, str):
            try:
                return self._groupindex[group]
            except KeyError:
                raise IndexError("no such group") from None
        try:
            index = operator.index(group)
        except TypeError:
            raise IndexError("no such group") from None
        if index < 0 or index >= len(self._spans):
            raise IndexError("no such group")
        return index

    def _group_value(self, group: int | str) -> str | None:
        start, end = self._spans[self._resolve_group(group)]
        if start < 0:
            return None
        return self._value[start:end]

    def group(self, *groups: int | str) -> Any:
        if not groups:
            return self._group_value(0)
        values = tuple(self._group_value(group) for group in groups)
        return values[0] if len(values) == 1 else values

    def groups(self, default: Any = None) -> tuple[Any, ...]:
        return tuple(
            default if value is None else value
            for value in (self._group_value(index) for index in range(1, len(self._spans)))
        )

    def groupdict(self, default: Any = None) -> dict[str, Any]:
        return {
            name: default if (value := self._group_value(index)) is None else value
            for name, index in self._groupindex.items()
        }

    def span(self, group: int | str = 0) -> tuple[int, int]:
        return self._spans[self._resolve_group(group)]

    def start(self, group: int | str = 0) -> int:
        return self.span(group)[0]

    def end(self, group: int | str = 0) -> int:
        return self.span(group)[1]


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


def _normalize_stdlib_flags(flags: int) -> int | None:
    if isinstance(flags, bool) or not isinstance(flags, int):
        return None
    raw_flags = int(flags)
    if raw_flags < 0 or raw_flags & ~_SUPPORTED_STDLIB_FLAGS:
        return None

    return raw_flags


def _normalize_flags(flags: int) -> int | None:
    raw_flags = _normalize_stdlib_flags(flags)
    if raw_flags is None:
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


def _stdlib_fast_path_is_compatible(pattern: str) -> bool:
    if not pattern.isascii():
        return False
    if "[[" in pattern:
        return False

    index = 0
    while index < len(pattern):
        if pattern[index] != "\\":
            index += 1
            continue
        if index + 1 >= len(pattern):
            return True

        escaped = pattern[index + 1]
        if escaped == "\\":
            index += 2
            continue
        if escaped == "N":
            return False
        if escaped in {"u", "U"}:
            width = 4 if escaped == "u" else 8
            digits = pattern[index + 2 : index + 2 + width]
            if len(digits) == width and all(char in "0123456789abcdefABCDEF" for char in digits):
                if int(digits, 16) > 0x7F:
                    return False
                index += width + 2
                continue
        if escaped == "x":
            digits = pattern[index + 2 : index + 4]
            if len(digits) == 2 and all(char in "0123456789abcdefABCDEF" for char in digits):
                if int(digits, 16) > 0x7F:
                    return False
                index += 4
                continue
        if escaped in "01234567":
            digits = escaped
            offset = index + 2
            while offset < len(pattern) and len(digits) < 3 and pattern[offset] in "01234567":
                digits += pattern[offset]
                offset += 1
            if len(digits) == 3 and int(digits, 8) > 0x7F:
                return False
            index = offset
            continue
        index += 2
    return True


def _stdlib_search_needs_worker(pattern: str, value: str) -> bool:
    return not value.isascii() or not _stdlib_fast_path_is_compatible(pattern)


def _decode_worker_message(raw: bytes, max_bytes: int) -> dict[str, Any] | None:
    if not raw or len(raw) > max_bytes or not raw.endswith(b"\n"):
        return None
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return None
    return message if isinstance(message, dict) else None


def _read_worker_handshake(stdout: Any, result: queue.Queue[bytes]) -> None:
    try:
        line = stdout.readline(_MAX_WORKER_HANDSHAKE_BYTES + 1)
    except (OSError, ValueError):
        line = b""
    result.put(line)


def _close_worker_pipes(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout):
        if stream is None:
            continue
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _terminate_and_reap_worker(
    process: subprocess.Popen[bytes],
    *,
    startup_reader: threading.Thread | None = None,
) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass

    if startup_reader is not None:
        startup_reader.join(_STDLIB_WORKER_REAP_TIMEOUT_S)

    try:
        process.communicate(timeout=_STDLIB_WORKER_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_STDLIB_WORKER_REAP_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass
    except (OSError, ValueError):
        try:
            process.wait(timeout=_STDLIB_WORKER_REAP_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass
    finally:
        _close_worker_pipes(process)


def _validated_worker_spans(
    raw_spans: Any,
    *,
    expected_count: int,
    value_length: int,
) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(raw_spans, list) or len(raw_spans) != expected_count:
        return None

    spans: list[tuple[int, int]] = []
    for index, raw_span in enumerate(raw_spans):
        if not isinstance(raw_span, list) or len(raw_span) != 2:
            return None
        start, end = raw_span
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            return None
        if (start, end) == (-1, -1) and index > 0:
            spans.append((start, end))
            continue
        if start < 0 or end < start or end > value_length:
            return None
        spans.append((start, end))
    return tuple(spans)


def _run_stdlib_worker(
    request_payload: dict[str, Any],
    *,
    timeout_s: float,
    max_response_bytes: int,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        request = (
            json.dumps(
                request_payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError):
        return None, "regex_invalid"
    if len(request) > _MAX_WORKER_REQUEST_BYTES:
        return None, "regex_too_large"

    try:
        # Untrusted values are sent only over the bounded stdin protocol.
        process = subprocess.Popen(  # nosec B603
            [sys.executable, "-I", "-S", "-u", "-c", _STDLIB_WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, "regex_invalid"

    if process.stdin is None or process.stdout is None:
        _terminate_and_reap_worker(process)
        return None, "regex_invalid"

    startup_result: queue.Queue[bytes] = queue.Queue(maxsize=1)
    startup_reader = threading.Thread(
        target=_read_worker_handshake,
        args=(process.stdout, startup_result),
        name="safe-regex-startup-reader",
        daemon=True,
    )
    startup_reader.start()
    try:
        raw_ready = startup_result.get(timeout=_STDLIB_WORKER_STARTUP_TIMEOUT_S)
    except queue.Empty:
        _terminate_and_reap_worker(process, startup_reader=startup_reader)
        return None, "regex_timeout"

    startup_reader.join(_STDLIB_WORKER_REAP_TIMEOUT_S)
    ready = _decode_worker_message(raw_ready, _MAX_WORKER_HANDSHAKE_BYTES)
    if startup_reader.is_alive() or ready != {"status": "ready"}:
        _terminate_and_reap_worker(process, startup_reader=startup_reader)
        return None, "regex_invalid"

    try:
        raw_response, _stderr = process.communicate(input=request, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _terminate_and_reap_worker(process)
        return None, "regex_timeout"
    except (OSError, ValueError, subprocess.SubprocessError):
        _terminate_and_reap_worker(process)
        return None, "regex_invalid"

    response = _decode_worker_message(raw_response, max_response_bytes)
    if process.returncode != 0 or response is None:
        return None, "regex_invalid"
    return response, None


def _search_stdlib_in_worker(
    pattern: str,
    value: str,
    flags: int,
    timeout_s: float,
    stdlib_compiled: Any,
) -> SafeRegexResult:
    response, code = _run_stdlib_worker(
        {"flags": flags, "pattern": pattern, "value": value},
        timeout_s=timeout_s,
        max_response_bytes=_MAX_WORKER_SEARCH_RESPONSE_BYTES,
    )
    if code is not None or response is None:
        return SafeRegexResult(matched=False, code=code or "regex_invalid")
    if response == {"status": "no_match"}:
        return SafeRegexResult(matched=False)
    if set(response) != {"status", "spans"} or response.get("status") != "match":
        return SafeRegexResult(matched=False, code="regex_invalid")

    spans = _validated_worker_spans(
        response["spans"],
        expected_count=stdlib_compiled.groups + 1,
        value_length=len(value),
    )
    if spans is None:
        return SafeRegexResult(matched=False, code="regex_invalid")
    match = _MatchSnapshot(
        _value=value,
        _spans=spans,
        _groupindex=MappingProxyType(dict(stdlib_compiled.groupindex)),
    )
    return SafeRegexResult(matched=True, match=match)


def _prepare_stdlib_substitution(
    pattern: str,
    value: str,
    flags: int,
    limits: SafeRegexLimits,
) -> tuple[int, float] | str:
    validated_limits = _validated_limits(limits)
    if validated_limits is None or not isinstance(pattern, str) or not isinstance(value, str):
        return "regex_invalid"

    max_pattern_chars, max_input_chars, timeout_s = validated_limits
    if len(pattern) > max_pattern_chars or len(value) > max_input_chars:
        return "regex_too_large"

    normalized_flags = _normalize_stdlib_flags(flags)
    if normalized_flags is None:
        return "regex_invalid"
    return normalized_flags, timeout_s


def _sub_stdlib_in_worker(
    pattern: str,
    repl: str,
    value: str,
    flags: int,
    timeout_s: float,
    output_limit: int,
) -> SafeRegexSubResult:
    response, code = _run_stdlib_worker(
        {
            "flags": flags,
            "legacy_numeric_group_ids": _STDLIB_LEGACY_NUMERIC_GROUP_IDS,
            "max_output_chars": output_limit,
            "operation": "sub",
            "pattern": pattern,
            "repl": repl,
            "value": value,
        },
        timeout_s=timeout_s,
        max_response_bytes=_MAX_WORKER_SUB_RESPONSE_BYTES,
    )
    if code is not None or response is None:
        return SafeRegexSubResult(code=code or "regex_invalid")
    if response == {"status": "too_large"}:
        return SafeRegexSubResult(code="regex_too_large")
    if set(response) != {"status", "value"} or response.get("status") != "sub":
        return SafeRegexSubResult(code="regex_invalid")

    replaced = response.get("value")
    if not isinstance(replaced, str) or len(replaced) > output_limit:
        return SafeRegexSubResult(code="regex_invalid")
    return SafeRegexSubResult(value=replaced)


def _prepare_untrusted_pattern(
    pattern: str,
    value: str,
    flags: int,
    limits: SafeRegexLimits,
    dialect: _RegexDialect,
    *,
    defer_stdlib_search: bool = False,
) -> tuple[Any, float] | str:
    validated_limits = _validated_limits(limits)
    if (
        validated_limits is None
        or not isinstance(pattern, str)
        or not isinstance(value, str)
        or not isinstance(dialect, str)
        or dialect not in {"stdlib", "regex"}
    ):
        return "regex_invalid"

    max_pattern_chars, max_input_chars, timeout_s = validated_limits
    if len(pattern) > max_pattern_chars or len(value) > max_input_chars:
        return "regex_too_large"

    normalized_flags = _normalize_flags(flags)
    if normalized_flags is None:
        return "regex_invalid"

    stdlib_compiled = None
    if dialect == "stdlib":
        try:
            stdlib_compiled = re.compile(pattern, int(flags))
        except _REGEX_ERRORS:
            return "regex_invalid"
        if defer_stdlib_search and _stdlib_search_needs_worker(pattern, value):
            return stdlib_compiled, timeout_s

    try:
        engine_flags = normalized_flags
        if dialect == "stdlib":
            engine_flags |= int(regex.VERSION0)
        compiled = _compile_pattern(pattern, engine_flags)
    except _REGEX_ERRORS:
        return "regex_invalid"
    return compiled, timeout_s


def search_untrusted(
    pattern: str,
    value: str,
    *,
    flags: int = 0,
    limits: SafeRegexLimits = SafeRegexLimits(),
    dialect: _RegexDialect = "stdlib",
) -> SafeRegexResult:
    """Search with fixed size and execution bounds, returning stable error codes."""
    prepared = _prepare_untrusted_pattern(
        pattern,
        value,
        flags,
        limits,
        dialect,
        defer_stdlib_search=True,
    )
    if isinstance(prepared, str):
        return SafeRegexResult(matched=False, code=prepared)
    compiled, timeout_s = prepared

    if dialect == "stdlib" and _stdlib_search_needs_worker(pattern, value):
        return _search_stdlib_in_worker(pattern, value, int(flags), timeout_s, compiled)

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
    dialect: _RegexDialect,
) -> tuple[int, tuple[int, ...]] | None:
    try:
        if dialect == "stdlib":
            compiled = re.compile(pattern, int(flags))
        else:
            normalized_flags = _normalize_flags(flags)
            if normalized_flags is None:
                return None
            compiled = regex.compile(pattern, normalized_flags)
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
                    if int(digits, 8) > (0o777 if dialect == "regex" else 0o377):
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
    max_output_chars: int = _MAX_SUB_OUTPUT_CHARS,
    dialect: _RegexDialect = "stdlib",
) -> SafeRegexSubResult:
    """Substitute globally with fixed execution, input, and output bounds."""
    if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int) or max_output_chars < 0:
        return SafeRegexSubResult(code="regex_invalid")
    output_limit = min(max_output_chars, _MAX_SUB_OUTPUT_CHARS)
    if not isinstance(repl, str):
        return SafeRegexSubResult(code="regex_invalid")
    if len(repl) > _MAX_REPLACEMENT_CHARS:
        return SafeRegexSubResult(code="regex_too_large")

    if dialect == "stdlib":
        prepared_stdlib = _prepare_stdlib_substitution(pattern, value, flags, limits)
        if isinstance(prepared_stdlib, str):
            return SafeRegexSubResult(code=prepared_stdlib)
        stdlib_flags, timeout_s = prepared_stdlib
        return _sub_stdlib_in_worker(
            pattern,
            repl,
            value,
            stdlib_flags,
            timeout_s,
            output_limit,
        )

    prepared = _prepare_untrusted_pattern(pattern, value, flags, limits, dialect)
    if isinstance(prepared, str):
        return SafeRegexSubResult(code=prepared)
    compiled, timeout_s = prepared
    replacement_template = _parse_replacement_template(pattern, repl, flags, dialect)
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
        if written + unmatched_chars + expansion_chars > output_limit:
            raise _RegexOutputTooLarge

        expanded = match.expand(repl)
        written += unmatched_chars + len(expanded)
        if written > output_limit:
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
    if len(replaced) > output_limit:
        return SafeRegexSubResult(code="regex_too_large")
    return SafeRegexSubResult(value=replaced)


__all__ = [
    "SafeRegexLimits",
    "SafeRegexResult",
    "SafeRegexSubResult",
    "search_untrusted",
    "sub_untrusted",
]
