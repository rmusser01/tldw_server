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

    Canonical root spelling is compared exactly, including on case-sensitive
    Windows directories. Root-escaping case-only aliases are rejected before
    filesystem probes, even if the filesystem might canonicalize them back.

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
    # Reject lexical escapes before probing any candidate path. Keep the
    # caller-configured base distinct from the untrusted child candidate.
    if candidate == base_dir_abs:
        if os.path.islink(base_dir_abs):
            return _fail()
        lexical_path = base_dir_abs
    elif candidate.startswith(os.path.join(base_dir_abs, "")):
        lexical_path = candidate
    else:
        return _fail()
    try:
        relative = os.path.relpath(lexical_path, base_dir_abs)
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
    # Parents are checked before children, so resolving the candidate cannot
    # traverse a pre-existing directory link beneath the trusted base.
    base_real = os.path.realpath(base_dir_abs)
    candidate_real = os.path.realpath(lexical_path)
    # Compare canonical spelling: case-only siblings can be distinct on Windows.
    if candidate_real == base_real:
        validated_path = base_real
    elif candidate_real.startswith(os.path.join(base_real, "")):
        validated_path = candidate_real
    else:
        return _fail()
    return validated_path
