from __future__ import annotations

import importlib.util
from dataclasses import dataclass

_CORE_MODULES = (
    "tree_sitter",
)
_OPTIONAL_LANGUAGE_MODULES = (
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_java",
    "tree_sitter_kotlin",
    "tree_sitter_c_sharp",
    "tree_sitter_c",
    "tree_sitter_cpp",
)
_PROBED_MODULES = (*_CORE_MODULES, *_OPTIONAL_LANGUAGE_MODULES)


@dataclass(frozen=True)
class DependencyHealth:
    """Availability of optional CodeGraph parser dependencies."""

    available: bool
    missing: tuple[str, ...]
    present: tuple[str, ...]

    @property
    def all_optional_available(self) -> bool:
        """Return whether every optional language parser package is installed."""
        return not any(module_name in self.missing for module_name in _OPTIONAL_LANGUAGE_MODULES)


def probe_codegraph_dependencies() -> DependencyHealth:
    """Probe optional parser packages without importing them."""

    present: list[str] = []
    missing: list[str] = []
    for module_name in _PROBED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
        else:
            present.append(module_name)

    return DependencyHealth(
        available=not any(module_name in missing for module_name in _CORE_MODULES),
        missing=tuple(missing),
        present=tuple(present),
    )
