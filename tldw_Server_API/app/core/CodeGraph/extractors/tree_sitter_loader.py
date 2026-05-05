"""Optional Tree-sitter parser loading for CodeGraph extractors."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParserLoadResult:
    """Result of loading an optional Tree-sitter parser."""

    parser: Any | None = None
    missing: tuple[str, ...] = ()
    error: str | None = None

    @property
    def available(self) -> bool:
        """Return whether a parser was built successfully."""
        return self.parser is not None and not self.missing and self.error is None


_LANGUAGE_MODULES: dict[str, tuple[str, str]] = {
    "javascript": ("tree_sitter_javascript", "language"),
    "jsx": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
}


def load_parser(language_id: str) -> ParserLoadResult:
    """Dynamically load a Tree-sitter parser for a supported language."""
    parser_package = _LANGUAGE_MODULES.get(language_id)
    if parser_package is None:
        return ParserLoadResult(error=f"Unsupported Tree-sitter language: {language_id}")

    tree_sitter = _import_optional("tree_sitter")
    if tree_sitter.missing:
        return ParserLoadResult(missing=tree_sitter.missing)
    if tree_sitter.error:
        return ParserLoadResult(error=tree_sitter.error)

    module_name, language_function = parser_package
    language_module = _import_optional(module_name)
    if language_module.missing:
        return ParserLoadResult(missing=language_module.missing)
    if language_module.error:
        return ParserLoadResult(error=language_module.error)

    try:
        language = tree_sitter.module.Language(getattr(language_module.module, language_function)())
        parser = tree_sitter.module.Parser(language)
    except (AttributeError, TypeError, ValueError) as exc:  # pragma: no cover - optional native package boundary.
        return ParserLoadResult(error=str(exc))

    return ParserLoadResult(parser=parser)


@dataclass(frozen=True)
class _ImportResult:
    """Internal dynamic import result."""

    module: Any | None = None
    missing: tuple[str, ...] = ()
    error: str | None = None


def _import_optional(module_name: str) -> _ImportResult:
    """Import an optional module and report optional dependency failures without raising."""
    try:
        return _ImportResult(module=importlib.import_module(module_name))
    except ModuleNotFoundError as exc:
        if exc.name in {None, module_name}:
            return _ImportResult(missing=(module_name,))
        return _ImportResult(error=f"Failed to import {module_name}: {exc}")
    except (ImportError, OSError) as exc:
        return _ImportResult(error=f"Failed to import {module_name}: {exc}")


__all__ = ["ParserLoadResult", "load_parser"]
