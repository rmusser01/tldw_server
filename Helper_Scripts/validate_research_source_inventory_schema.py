#!/usr/bin/env python3
"""Apply the authoritative JSON Schema gate to research inventory documents."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

INVENTORY_FORMAT_CHECKER = FormatChecker()


@INVENTORY_FORMAT_CHECKER.checks("date-time", raises=(TypeError, ValueError))
def _valid_datetime(value: object) -> bool:
    """Validate an RFC 3339-like timestamp even without optional format extras."""
    if not isinstance(value, str) or "T" not in value:
        return False
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.tzinfo is not None


@INVENTORY_FORMAT_CHECKER.checks("uri", raises=(TypeError, ValueError))
def _valid_uri(value: object) -> bool:
    """Require an absolute URI with a scheme and network location."""
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return bool(parsed.scheme and parsed.netloc)


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
