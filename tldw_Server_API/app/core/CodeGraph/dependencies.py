from __future__ import annotations

import importlib.util
from dataclasses import dataclass

_REQUIRED_MODULES = (
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_java",
    "tree_sitter_kotlin",
)


@dataclass(frozen=True)
class DependencyHealth:
    """Availability of optional CodeGraph parser dependencies."""

    available: bool
    missing: tuple[str, ...]
    present: tuple[str, ...]


def probe_codegraph_dependencies() -> DependencyHealth:
    """Probe optional parser packages without importing them."""

    present: list[str] = []
    missing: list[str] = []
    for module_name in _REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
        else:
            present.append(module_name)

    return DependencyHealth(
        available=not missing,
        missing=tuple(missing),
        present=tuple(present),
    )
