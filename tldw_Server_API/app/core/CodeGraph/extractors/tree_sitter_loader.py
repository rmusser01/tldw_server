"""Optional Tree-sitter parser loading for CodeGraph JS-family extractors."""

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
}


def load_parser(language_id: str) -> ParserLoadResult:
    """Dynamically load a Tree-sitter parser for a supported JS-family language."""
    parser_package = _LANGUAGE_MODULES.get(language_id)
    if parser_package is None:
        return ParserLoadResult(error=f"Unsupported Tree-sitter language: {language_id}")

    tree_sitter = _import_optional("tree_sitter")
    if tree_sitter.missing:
        return ParserLoadResult(missing=tree_sitter.missing)

    module_name, language_function = parser_package
    language_module = _import_optional(module_name)
    if language_module.missing:
        return ParserLoadResult(missing=language_module.missing)

    try:
        language = tree_sitter.module.Language(getattr(language_module.module, language_function)())
        parser = tree_sitter.module.Parser(language)
    except Exception as exc:  # pragma: no cover - defensive boundary for optional native packages.
        return ParserLoadResult(error=str(exc))

    return ParserLoadResult(parser=parser)


@dataclass(frozen=True)
class _ImportResult:
    """Internal dynamic import result."""

    module: Any | None = None
    missing: tuple[str, ...] = ()


def _import_optional(module_name: str) -> _ImportResult:
    """Import an optional module and report missing modules without raising."""
    try:
        return _ImportResult(module=importlib.import_module(module_name))
    except ModuleNotFoundError:
        return _ImportResult(missing=(module_name,))


__all__ = ["ParserLoadResult", "load_parser"]
