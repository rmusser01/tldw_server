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
