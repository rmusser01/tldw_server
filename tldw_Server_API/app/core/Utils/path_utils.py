from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def resolve_path(path: Path) -> Path:
    """Expand and resolve a path without requiring it to exist."""
    expanded = path.expanduser()
    return expanded.resolve(strict=False)


def safe_join(
    base_dir: str,
    name: str,
    *,
    error_factory: Callable[[Exception | None], Exception] | None = None,
) -> str | None:
    """
    Safely join a base directory and relative name, preventing traversal or symlink escapes.

    Returns the normalized, real path on success. When ``error_factory`` is provided,
    raises that exception on failure; otherwise returns None.
    """
    def _fail(exc: Exception | None = None) -> str | None:
        if error_factory:
            if exc is not None:
                raise error_factory(exc) from exc
            raise error_factory(None)
        return None

    if not name or os.path.isabs(name):
        return _fail()

    base_dir_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base_dir_abs, name))
    if os.path.islink(candidate):
        return _fail()
    base_real = os.path.realpath(base_dir_abs)
    candidate_real = os.path.realpath(candidate)
    # Compare using platform case rules, but preserve canonical spelling for
    # filesystem access (Windows also supports case-sensitive directories).
    base_compare = os.path.normcase(base_real)
    candidate_compare = os.path.normcase(candidate_real)
    # The separator keeps sibling names from matching; join also handles
    # filesystem, drive and UNC roots that already end with a separator.
    if candidate_compare != base_compare and not candidate_compare.startswith(os.path.join(base_compare, "")):
        return _fail()
    try:
        relative = os.path.relpath(candidate, base_dir_abs)
    except ValueError as exc:
        return _fail(exc)
    if relative.startswith(os.pardir + os.sep) or relative == os.pardir:
        return _fail()
    current = base_dir_abs
    for part in relative.split(os.sep):
        if part in ("", "."):
            continue
        current = os.path.join(current, part)
        if os.path.islink(current):
            return _fail()
    return candidate_real
