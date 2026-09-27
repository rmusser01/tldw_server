"""Strict JSON difference collection and generic fixture contract models."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class DifferenceToken(Enum):
    ANY_PATH = "<ANY_PATH>"
    MISSING = "<MISSING>"


ANY_PATH = DifferenceToken.ANY_PATH
MISSING = DifferenceToken.MISSING
PathPart = str | int
PathPatternPart = PathPart | DifferenceToken


@dataclass(frozen=True)
class Difference:
    path: tuple[PathPart, ...]
    actual: object
    expected: object
    issue: str | None = None


@dataclass(frozen=True)
class DifferenceRule:
    identifier: str
    path: tuple[PathPatternPart, ...]
    description: str
    validator: Callable[[Difference], bool]
    minimum_count: int = 0
    maximum_count: int = 1


@dataclass(frozen=True)
class DifferenceContract:
    behavior_change: int
    rules: tuple[DifferenceRule, ...]
    allow_predecessor_equality: bool = False
    profile_validator: Callable[[object, object], None] | None = None


def validate_json_value(
    value: object,
    *,
    label: str,
    path: tuple[PathPart, ...] = (),
) -> None:
    """Require a recursively valid JSON value made only from exact built-in types."""
    value_type = type(value)
    location = format_path(path)
    if value_type is dict:
        assert isinstance(value, dict)
        for key in value:
            if type(key) is not str:
                raise AssertionError(
                    f"Strict JSON validation failed for {label} at {location}: "
                    f"object key type must be exact str, got {type(key).__name__}"
                )
        for key in sorted(value):
            validate_json_value(
                value[key],
                label=label,
                path=(*path, key),
            )
        return
    if value_type is list:
        assert isinstance(value, list)
        for index, item in enumerate(value):
            validate_json_value(
                item,
                label=label,
                path=(*path, index),
            )
        return
    if value_type is float:
        assert isinstance(value, float)
        if not math.isfinite(value):
            raise AssertionError(f"Strict JSON validation failed for {label} at {location}: " "float must be finite")
        return
    if value_type in {str, int, bool, type(None)}:
        return
    raise AssertionError(
        f"Strict JSON validation failed for {label} at {location}: "
        f"expected an exact built-in JSON type, got {value_type.__name__}"
    )


def collect_differences(
    actual: object,
    expected: object,
    path: tuple[PathPart, ...] = (),
) -> list[Difference]:
    if actual is MISSING or expected is MISSING:
        return [Difference(path=path, actual=actual, expected=expected)]
    actual_type = type(actual)
    expected_type = type(expected)
    if actual_type is not expected_type:
        return [
            Difference(
                path=path,
                actual=actual,
                expected=expected,
                issue=(f"JSON type mismatch: {actual_type.__name__} is not " f"{expected_type.__name__}"),
            )
        ]
    if actual_type is dict:
        assert isinstance(actual, dict) and isinstance(expected, dict)
        if any(type(key) is not str for key in (*actual, *expected)):
            return [
                Difference(
                    path=path,
                    actual=actual,
                    expected=expected,
                    issue="JSON object keys must be strings",
                )
            ]
        differences: list[Difference] = []
        for key in sorted(set(actual) | set(expected), key=str):
            differences.extend(
                collect_differences(
                    actual.get(key, MISSING),
                    expected.get(key, MISSING),
                    (*path, key),
                )
            )
        return differences
    if actual_type is list:
        assert isinstance(actual, list) and isinstance(expected, list)
        differences = []
        for index in range(max(len(actual), len(expected))):
            differences.extend(
                collect_differences(
                    actual[index] if index < len(actual) else MISSING,
                    expected[index] if index < len(expected) else MISSING,
                    (*path, index),
                )
            )
        return differences
    if actual_type not in {str, int, float, bool, type(None)}:
        return [
            Difference(
                path=path,
                actual=actual,
                expected=expected,
                issue=f"non-JSON value type: {actual_type.__name__}",
            )
        ]
    if actual_type is float:
        assert isinstance(actual, float) and isinstance(expected, float)
        if not math.isfinite(actual) or not math.isfinite(expected):
            return [
                Difference(
                    path=path,
                    actual=actual,
                    expected=expected,
                    issue="non-finite floats are not valid JSON values",
                )
            ]
    if actual != expected:
        return [Difference(path=path, actual=actual, expected=expected)]
    return []


def path_matches(pattern: tuple[PathPatternPart, ...], path: tuple[PathPart, ...]) -> bool:
    return len(pattern) == len(path) and all(
        expected_part is ANY_PATH or expected_part == actual_part for expected_part, actual_part in zip(pattern, path)
    )


def format_path(path: tuple[PathPart, ...]) -> str:
    formatted = "$"
    for part in path:
        formatted += f"[{part}]" if isinstance(part, int) else f".{part}"
    return formatted
