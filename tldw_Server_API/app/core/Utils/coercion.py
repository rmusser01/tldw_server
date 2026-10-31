"""Scalar and environment-variable coercion with one boolean vocabulary.

Design of record: Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md

``parse_bool`` is three-way: a recognised truthy token is ``True``, a recognised
falsy token is ``False``, and anything else returns the caller's ``default``.
Unrecognised input never silently becomes ``True``, so a switch that gates
egress or subprocess execution fails closed when the caller passes
``default=False``.

Strings are stripped and lower-cased before matching, so ``" Yes "`` from a
docker-compose ``environment:`` list is still ``True``.

Kept stdlib-only at import time: ``core/testing.py`` re-exports from here.
"""

from __future__ import annotations

import os
from typing import Any

TRUTHY = frozenset({"1", "true", "yes", "y", "on", "enabled"})
FALSY = frozenset({"0", "false", "no", "n", "off", "disabled", "none", "null", ""})


def parse_bool(value: Any, *, default: bool, key: str | None = None) -> bool:
    """Coerce ``value`` to bool.

    - bool: returned as-is
    - int/float: 0 is False, anything else True
    - str: stripped, case-insensitive match against TRUTHY then FALSY
    - None, unrecognised strings and other types: ``default``

    When ``key`` is given, an unrecognised string is logged at WARNING with the
    key name only (never the value, which may be a secret).
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in TRUTHY:
            return True
        if token in FALSY:
            return False
        if key:
            from loguru import logger

            logger.warning("Unrecognised boolean value for {}; using default {}", key, default)
    return default


def env_bool(key: str, *, default: bool) -> bool:
    """Read ``key`` from the environment and parse it with :func:`parse_bool`.

    An unset variable returns ``default``; an empty one is falsy, matching an
    operator who wrote ``KEY=`` to switch a feature off.
    """
    return parse_bool(os.getenv(key), default=default, key=key)
