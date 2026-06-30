from __future__ import annotations

"""Chat dictionary validator (library + CLI).

Validates a dictionary JSON structure and returns a structured report with
errors, warnings, and basic stats. Also provides a CLI entrypoint:

  python -m tldw_Server_API.app.core.Chat.validate_dictionary --file path.json [--strict]

Supports .json/.yaml/.yml; optionally .md (key: value or multi-line blocks)
via the existing markdown parser, converting each entry to a literal pattern.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    import yaml  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    yaml = None  # type: ignore

import contextlib

from jinja2 import StrictUndefined, TemplateError, nodes
from jinja2.sandbox import SandboxedEnvironment
from loguru import logger

from tldw_Server_API.app.core.Chat.chat_dictionary import parse_user_dict_markdown_file
from tldw_Server_API.app.core.Chunking.regex_safety import check_pattern as check_regex_pattern
from tldw_Server_API.app.core.Chunking.regex_safety import warn_ambiguity
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
from tldw_Server_API.app.core.testing import is_truthy

_VALIDATE_DICT_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_YAML_LOAD_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if yaml is not None:  # pragma: no cover
    _yaml_error = getattr(yaml, "YAMLError", None)
    if isinstance(_yaml_error, type) and issubclass(_yaml_error, BaseException):
        _YAML_LOAD_EXCEPTIONS = (_yaml_error,)
_CLI_LOAD_EXCEPTIONS = _VALIDATE_DICT_NONCRITICAL_EXCEPTIONS + _YAML_LOAD_EXCEPTIONS

# -----------------------------
# Schema and limits
# -----------------------------


ALLOWED_ENTRY_FIELDS = {
    "type",
    "pattern",
    "replacement",
    "probability",
    "group",
    "timed_effects",
    "max_replacements",
    "enabled",
    "case_sensitive",
    "enable_templates",
}


def _as_list(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    return []


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {}


def _truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return is_truthy(v)


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return default


DEFAULT_DICTIONARY_MAX_ENTRIES = 10000
DEFAULT_ENTRY_MAX_CHARS = 20000
DEFAULT_VALIDATE_TIMEOUT_MS = 500
DEFAULT_VALIDATE_MAX_ENTRIES = 1000

# Fatal codes are used by strict CLI mode and server-side workflows that opt
# into fatal-only rejection semantics.
FATAL_ERROR_CODES = {
    "schema_invalid",
    "regex_invalid",
    "regex_unsafe",
    "regex_timeout",
    "template_parse_error",
    "template_forbidden_construct",
    "template_output_too_large",
    "dictionary_entry_too_large",
    "dictionary_too_large",
}

# In strict mode, keep warnings visible but promote select codes into `errors`.
# Not all strict errors are fatal.
STRICT_WARNING_TO_ERROR_CODES = {
    "dictionary_entry_too_large",
    "dictionary_too_large",
    "template_unknown_function",
    "template_undefined_name",
}


# -----------------------------
# Template checks (expression-only)
# -----------------------------


_TPL_ENV = SandboxedEnvironment(autoescape=False, undefined=StrictUndefined)

_DISALLOWED_NODE_TYPES = {
    nodes.For,
    nodes.If,
    nodes.Macro,
    nodes.Block,
    nodes.Import,
    nodes.FromImport,
    nodes.Include,
    nodes.Assign,
    nodes.AssignBlock,
    nodes.CallBlock,
    nodes.FilterBlock,
    nodes.Extends,
    nodes.With,
    nodes.ScopedEvalContextModifier,
    nodes.Mul,
    nodes.Pow,
}

# Functions available in templates (Stage 1)
_ALLOWED_FUNCS = {
    "now",
    "today",
    "iso_now",
    "now_tz",
    "upper",
    "lower",
    "title",
    "slugify",
    # Gated random helpers are off by default; keep them as known names to downgrade to warnings
    "randint",
    "choice",
    # Not a function but an allowed variable name
    "match",
    "matched_text",
    "env",
    "user",
}

_ALLOWED_METHOD_CALLS_BY_ROOT = {
    "match": frozenset({"end", "group", "groupdict", "groups", "start"}),
}


def _template_ast_checks(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (errors, warnings) for a template string.

    Detect forbidden constructs and unknown functions. Unknown functions are
    reported as warnings; some special cases (e.g., 'weather') are flagged as
    external-calls-disabled.
    """
    errs: list[dict[str, Any]] = []
    warns: list[dict[str, Any]] = []
    if not isinstance(text, str) or ("{{" not in text and "{%" not in text):
        return errs, warns
    try:
        ast = _TPL_ENV.parse(text)
    except (TemplateError, TypeError, ValueError) as e:
        errs.append({
            "code": "template_parse_error",
            "field": "replacement",
            "message": str(e),
        })
        return errs, warns

    def _root_name(n: nodes.Node) -> str | None:
        if isinstance(n, nodes.Name):
            return n.name
        if isinstance(n, nodes.Getattr):
            return _root_name(n.node)
        if isinstance(n, nodes.Getitem):
            return _root_name(n.node)
        return None

    def _walk(n: nodes.Node) -> None:
        if isinstance(n, tuple(_DISALLOWED_NODE_TYPES)):
            errs.append({
                "code": "template_forbidden_construct",
                "field": "replacement",
                "message": f"Forbidden construct: {type(n).__name__}",
            })
        # Detect Name nodes used as callable (function calls)
        if isinstance(n, nodes.Call):
            target = n.node
            if isinstance(target, nodes.Name):
                func_name = target.name
                if func_name not in _ALLOWED_FUNCS:
                    if func_name == "weather":
                        errs.append({
                            "code": "template_external_calls_disabled",
                            "field": "replacement",
                            "message": "Function weather() requires external calls and is disabled",
                        })
                    else:
                        warns.append({
                            "code": "template_unknown_function",
                            "field": "replacement",
                            "message": f"Unknown function: {func_name}",
                        })
            elif isinstance(target, nodes.Getattr):
                root = _root_name(target.node)
                allowed_methods = _ALLOWED_METHOD_CALLS_BY_ROOT.get(str(root or ""))
                if allowed_methods is None or target.attr not in allowed_methods:
                    errs.append({
                        "code": "template_forbidden_construct",
                        "field": "replacement",
                        "message": f"Forbidden method call: {target.attr}",
                    })
            else:
                errs.append({
                    "code": "template_forbidden_construct",
                    "field": "replacement",
                    "message": f"Forbidden callable: {type(target).__name__}",
                })
        for child in n.iter_child_nodes():
            _walk(child)

    _walk(ast)
    return errs, warns


# -----------------------------
# Validation API
# -----------------------------


@dataclass
class ValidationResult:
    ok: bool
    schema_version: int
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    entry_stats: dict[str, int]
    suggested_fixes: list[str]
    partial: bool = False
    partial_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "schema_version": self.schema_version,
            "errors": self.errors,
            "warnings": self.warnings,
            "entry_stats": self.entry_stats,
            "suggested_fixes": self.suggested_fixes,
            "partial": self.partial,
            "partial_reason": self.partial_reason,
        }


def _promote_warnings_for_strict(
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for issue in warnings:
        code = str(issue.get("code", "unknown"))
        if code in STRICT_WARNING_TO_ERROR_CODES:
            promoted.append(issue)
        else:
            remaining.append(issue)
    return promoted, remaining


def has_fatal_errors(issues: list[dict[str, Any]]) -> bool:
    """Return True if any issue code is classified as fatal."""
    for issue in issues:
        code = str(issue.get("code", "unknown"))
        if code in FATAL_ERROR_CODES:
            return True
    return False


def _validate_entries(
    entries: list[dict[str, Any]],
    *,
    dictionary_max_entries: int,
    max_entry_chars: int,
    validate_max_entries: int,
    deadline: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[str], bool, str | None]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    fixes: list[str] = []

    total = len(entries)
    n_regex = 0
    n_lit = 0

    seen = set()
    partial = False
    partial_reason: str | None = None

    if total > dictionary_max_entries:
        warnings.append({
            "code": "dictionary_too_large",
            "field": "entries",
            "message": f"Dictionary has {total} entries (max {dictionary_max_entries})",
        })

    entries_to_validate = entries
    if validate_max_entries > 0 and total > validate_max_entries:
        entries_to_validate = entries[:validate_max_entries]
        partial = True
        partial_reason = "max_entries"

    for idx, e in enumerate(entries_to_validate):
        if deadline is not None and time.perf_counter() >= deadline:
            partial = True
            partial_reason = partial_reason or "timeout"
            break

        path = f"entries[{idx}]"
        if not isinstance(e, dict):
            errors.append({"code": "schema_invalid", "field": path, "message": "Entry must be an object"})
            continue

        etype = str(e.get("type", "literal") or "literal").lower()
        if etype not in {"literal", "regex"}:
            errors.append({"code": "schema_invalid", "field": f"{path}.type", "message": "Must be 'literal' or 'regex'"})
            continue
        n_regex += int(etype == "regex")
        n_lit += int(etype == "literal")

        pattern = e.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            errors.append({"code": "empty_pattern", "field": f"{path}.pattern", "message": "Pattern must be non-empty"})
            continue

        replacement = e.get("replacement", "")
        if not isinstance(replacement, str):
            errors.append({"code": "schema_invalid", "field": f"{path}.replacement", "message": "Replacement must be a string"})
            replacement = ""

        if len(replacement) > max_entry_chars:
            warnings.append({
                "code": "dictionary_entry_too_large",
                "field": f"{path}.replacement",
                "message": f"Replacement length {len(replacement)} exceeds {max_entry_chars}",
            })

        # Probability
        prob = e.get("probability", 1.0)
        try:
            pf = float(prob)
        except (TypeError, ValueError):
            errors.append({"code": "schema_invalid", "field": f"{path}.probability", "message": "Must be a number"})
            pf = 1.0
        if pf < 0.0 or pf > 1.0:
            errors.append({"code": "probability_out_of_range", "field": f"{path}.probability", "message": "Must be within [0.0, 1.0]"})

        # max_replacements
        mr = e.get("max_replacements", 0)
        try:
            mi = int(mr)
        except (TypeError, ValueError):
            errors.append({"code": "schema_invalid", "field": f"{path}.max_replacements", "message": "Must be an integer"})
            mi = 0
        if mi < 0:
            errors.append({"code": "max_replacements_invalid", "field": f"{path}.max_replacements", "message": "Must be >= 0"})

        # Unknown fields
        for k in e:
            if k not in ALLOWED_ENTRY_FIELDS:
                warnings.append({
                    "code": "unknown_field",
                    "field": f"{path}.{k}",
                    "message": f"Unknown field '{k}' will be ignored",
                })

        # Regex/literal specific checks
        if etype == "regex":
            msg = check_regex_pattern(pattern, max_len=256)
            if msg:
                if msg.startswith("Invalid regex:"):
                    errors.append({"code": "regex_invalid", "field": f"{path}.pattern", "message": msg})
                elif "dangerous" in msg:
                    errors.append({"code": "regex_unsafe", "field": f"{path}.pattern", "message": msg})
                else:
                    errors.append({"code": "regex_invalid", "field": f"{path}.pattern", "message": msg})
            warn = warn_ambiguity(pattern)
            if warn:
                warnings.append({"code": "regex_ambiguous", "field": f"{path}.pattern", "message": warn})
            key = ("regex", pattern)
        else:
            key = ("literal", pattern.lower())

        # Duplicate detection
        if key in seen:
            errors.append({"code": "duplicate_pattern", "field": f"{path}.pattern", "message": "Duplicate pattern for type"})
        else:
            seen.add(key)

        # Template checks (in replacement)
        t_errs, t_warns = _template_ast_checks(replacement)
        # If unknown function warnings exist and external calls disabled, escalate 'weather' to error via code used above
        errors.extend(t_errs)
        warnings.extend(t_warns)

    stats = {"total": total, "regex": n_regex, "literal": n_lit}
    return errors, warnings, stats, fixes, partial, partial_reason


def validate_dictionary(data: dict[str, Any], schema_version: int = 1, strict: bool = False) -> ValidationResult:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    _start_t = None
    try:
        _start_t = time.perf_counter()  # type: ignore[name-defined]
    except _VALIDATE_DICT_NONCRITICAL_EXCEPTIONS:
        pass
    with contextlib.suppress(_VALIDATE_DICT_NONCRITICAL_EXCEPTIONS):
        increment_counter("chat_dictionary_validate_requests_total", labels={"strict": str(bool(strict)).lower()})

    # Validation controls (read per-call so tests and runtime toggles are honored)
    dictionary_max_entries = max(1, _int_env("CHAT_DICT_MAX_ENTRIES", DEFAULT_DICTIONARY_MAX_ENTRIES))
    max_entry_chars = max(1, _int_env("CHAT_DICT_MAX_ENTRY_CHARS", DEFAULT_ENTRY_MAX_CHARS))
    validate_timeout_ms = max(0, _int_env("CHAT_DICT_VALIDATE_TIMEOUT_MS", DEFAULT_VALIDATE_TIMEOUT_MS))
    validate_max_entries = _int_env("CHAT_DICT_VALIDATE_MAX_ENTRIES", DEFAULT_VALIDATE_MAX_ENTRIES)
    deadline = (time.perf_counter() + (validate_timeout_ms / 1000.0)) if validate_timeout_ms > 0 else None
    partial = False
    partial_reason: str | None = None

    if not isinstance(data, dict):
        errors.append({"code": "schema_invalid", "field": "root", "message": "Payload must be an object"})
        return ValidationResult(
            False,
            schema_version,
            errors,
            warnings,
            {"total": 0, "regex": 0, "literal": 0},
            [],
            partial=False,
            partial_reason=None,
        )

    try:
        normalized_schema_version = int(schema_version)
    except (TypeError, ValueError):
        normalized_schema_version = -1

    if normalized_schema_version != 1:
        errors.append({
            "code": "schema_invalid",
            "field": "schema_version",
            "message": f"Unsupported schema_version: {schema_version}. Supported: 1",
        })
        return ValidationResult(
            False,
            schema_version,
            errors,
            warnings,
            {"total": 0, "regex": 0, "literal": 0},
            [],
            partial=False,
            partial_reason=None,
        )

    entries = _as_list(data.get("entries"))
    if not entries:
        # Some inputs may use markdown; we permit empty entries here
        warnings.append({"code": "schema_invalid", "field": "entries", "message": "No entries found"})

    entry_errors, entry_warnings, stats, fixes, partial, partial_reason = _validate_entries(
        entries,
        dictionary_max_entries=dictionary_max_entries,
        max_entry_chars=max_entry_chars,
        validate_max_entries=validate_max_entries,
        deadline=deadline,
    )
    errors.extend(entry_errors)
    warnings.extend(entry_warnings)

    if strict:
        promoted, warnings = _promote_warnings_for_strict(warnings)
        errors.extend(promoted)

    # Emit per-code counters
    try:
        for e in errors:
            code = str(e.get("code", "unknown"))
            increment_counter("chat_dictionary_validate_errors_total", labels={"code": code})
        for w in warnings:
            code = str(w.get("code", "unknown"))
            increment_counter("chat_dictionary_validate_warnings_total", labels={"code": code})
    except _VALIDATE_DICT_NONCRITICAL_EXCEPTIONS:
        pass

    ok = len(errors) == 0
    try:
        if _start_t is not None:
            observe_histogram(
                "chat_dictionary_validate_duration_seconds",
                value=float(time.perf_counter() - _start_t),  # type: ignore[name-defined]
                labels={"strict": str(bool(strict)).lower()},
            )
    except _VALIDATE_DICT_NONCRITICAL_EXCEPTIONS:
        pass
    return ValidationResult(
        ok,
        schema_version,
        errors,
        warnings,
        stats,
        fixes,
        partial=partial,
        partial_reason=partial_reason,
    )


# -----------------------------
# CLI
# -----------------------------


def _load_file(path: str) -> dict[str, Any]:
    p = str(path)
    if p.endswith(".json"):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if (p.endswith(".yaml") or p.endswith(".yml")) and yaml is not None:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    if p.endswith(".md"):
        kv = parse_user_dict_markdown_file(p)
        # Convert to a basic literal entries format
        entries = [{
            "type": "literal",
            "pattern": k,
            "replacement": v,
            "probability": 1.0,
            "max_replacements": 0,
        } for k, v in kv.items()]
        return {"name": os.path.basename(p), "entries": entries}
    # Fallback: try JSON
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a chat dictionary JSON/YAML/MD file")
    parser.add_argument("--file", "-f", required=True, help="Path to dictionary file")
    parser.add_argument("--schema-version", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on validation errors")
    args = parser.parse_args(argv)

    try:
        payload = _load_file(args.file)
    except _CLI_LOAD_EXCEPTIONS as e:
        logger.error(f"Failed to read {args.file}: {e}")
        print(json.dumps({
            "ok": False,
            "schema_version": args.schema_version,
            "errors": [{"code": "schema_invalid", "field": "root", "message": f"Failed to read file: {e}"}],
            "warnings": [],
            "entry_stats": {"total": 0, "regex": 0, "literal": 0},
            "suggested_fixes": [],
        }, ensure_ascii=False))
        return 2

    try:
        result = validate_dictionary(payload, schema_version=args.schema_version, strict=args.strict)
    except _VALIDATE_DICT_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Validation failed internally: {e}")
        print(json.dumps({
            "ok": False,
            "schema_version": args.schema_version,
            "errors": [{"code": "schema_invalid", "field": "root", "message": f"Internal validation error: {e}"}],
            "warnings": [],
            "entry_stats": {"total": 0, "regex": 0, "literal": 0},
            "suggested_fixes": [],
            "partial": False,
            "partial_reason": None,
        }, ensure_ascii=False))
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False))
    if args.strict and has_fatal_errors(result.errors):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
