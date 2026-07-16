#!/usr/bin/env python3
"""Apply the authoritative JSON Schema gate to research inventory documents."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

INVENTORY_FORMAT_CHECKER = FormatChecker()
_RFC3339_DATETIME = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"[Tt](?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-](?P<offset_hour>[0-9]{2}):(?P<offset_minute>[0-9]{2}))\Z"
)
_URI_PCHAR = r"(?:[A-Za-z0-9\-._~!$&'()*+,;=:@]|%[0-9A-Fa-f]{2})"
_URI_AUTHORITY = re.compile(r"(?:[A-Za-z0-9\-._~!$&'()*+,;=%]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]+)?\Z")
_URI_PATH = re.compile(rf"(?:{_URI_PCHAR}|/)*\Z")
_URI_QUERY_OR_FRAGMENT = re.compile(rf"(?:{_URI_PCHAR}|[/?])*\Z")
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


@INVENTORY_FORMAT_CHECKER.checks("date-time", raises=(TypeError, ValueError))
def _valid_datetime(value: object) -> bool:
    """Validate a strict RFC 3339 timestamp without optional format extras."""
    if not isinstance(value, str):
        return False
    match = _RFC3339_DATETIME.fullmatch(value)
    if match is None:
        return False
    parts = {name: int(raw) for name, raw in match.groupdict(default="0").items()}
    try:
        date(parts["year"], parts["month"], parts["day"])
    except ValueError:
        return False
    if parts["hour"] > 23 or parts["minute"] > 59 or parts["second"] > 60:
        return False
    return parts["offset_hour"] <= 23 and parts["offset_minute"] <= 59


@INVENTORY_FORMAT_CHECKER.checks("uri", raises=(TypeError, ValueError))
def _valid_uri(value: object) -> bool:
    """Require an absolute URI with a scheme and network location."""
    if (
        not isinstance(value, str)
        or not value.isascii()
        or any(ord(character) <= 32 or ord(character) == 127 for character in value)
        or _BAD_PERCENT_ESCAPE.search(value) is not None
    ):
        return False
    parsed = urlsplit(value)
    port = parsed.port
    return (
        bool(parsed.scheme and parsed.netloc and parsed.hostname)
        and _URI_AUTHORITY.fullmatch(parsed.netloc) is not None
        and _URI_PATH.fullmatch(parsed.path) is not None
        and _URI_QUERY_OR_FRAGMENT.fullmatch(parsed.query) is not None
        and _URI_QUERY_OR_FRAGMENT.fullmatch(parsed.fragment) is not None
        and (port is None or 0 <= port <= 65535)
    )


def _leaf_errors(error: ValidationError) -> list[ValidationError]:
    """Expand combinator errors so callers receive actionable field paths."""
    if not error.context:
        return [error]
    leaves: list[ValidationError] = []
    for child in error.context:
        leaves.extend(_leaf_errors(child))
    return leaves


def validate_document(schema: dict[str, Any], document: Any, label: str) -> list[str]:
    """Return deterministic, path-qualified schema errors for one document."""
    validator = Draft202012Validator(
        schema,
        format_checker=INVENTORY_FORMAT_CHECKER,
    )
    errors = sorted(
        (leaf for error in validator.iter_errors(document) for leaf in _leaf_errors(error)),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    rendered: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        rendered.append(f"{label}:{location}: {error.message}")
    return rendered


def _load_json(path: Path) -> Any:
    """Load one UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    """Parse schema and labelled document paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument(
        "--document",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    """Validate all requested documents and emit a machine-readable result."""
    args = _parse_args()
    schema = _load_json(args.schema)
    Draft202012Validator.check_schema(schema)
    errors: list[str] = []
    for label, raw_path in args.document:
        errors.extend(validate_document(schema, _load_json(Path(raw_path)), label))
    print(json.dumps({"errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
