from __future__ import annotations

from pathlib import Path

from .dependencies import DependencyHealth, probe_codegraph_dependencies
from .models import LanguageInfo


class CodeGraphLanguageRegistry:
    """Language metadata and extension lookup for the foundation indexer."""

    def __init__(self, dependency_health: DependencyHealth | None = None) -> None:
        self.dependency_health = dependency_health or probe_codegraph_dependencies()
        self._languages = _foundation_languages(self.dependency_health)
        self._by_extension = {
            extension: language
            for language in self._languages
            for extension in language.extensions
        }

    def list_languages(self) -> tuple[LanguageInfo, ...]:
        return self._languages

    def language_for_path(self, path: str | Path) -> LanguageInfo | None:
        suffix = Path(path).suffix.lower()
        return self._by_extension.get(suffix)

    def known_language_ids(self) -> set[str]:
        return {language.language_id for language in self._languages}

    def foundation_language_ids(self) -> set[str]:
        return {
            language.language_id
            for language in self._languages
            if language.stage == "foundation"
        }


def _foundation_languages(dependency_health: DependencyHealth) -> tuple[LanguageInfo, ...]:
    missing = set(dependency_health.missing)
    javascript_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_javascript"))
    typescript_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_typescript"))
    java_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_java"))
    kotlin_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_kotlin"))
    csharp_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_c_sharp"))
    c_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_c"))
    cpp_missing = _missing_dependencies(missing, ("tree_sitter", "tree_sitter_cpp"))
    return (
        LanguageInfo(
            language_id="python",
            display_name="Python",
            extensions=(".py", ".pyi"),
            stage="foundation",
            symbol_extraction=True,
        ),
        LanguageInfo(
            language_id="javascript",
            display_name="JavaScript",
            extensions=(".js", ".jsx", ".mjs", ".cjs"),
            stage="foundation",
            dependency_missing=javascript_missing,
            symbol_extraction=not javascript_missing,
        ),
        LanguageInfo(
            language_id="typescript",
            display_name="TypeScript",
            extensions=(".ts", ".tsx"),
            stage="foundation",
            dependency_missing=typescript_missing,
            symbol_extraction=not typescript_missing,
        ),
        LanguageInfo(
            language_id="java",
            display_name="Java",
            extensions=(".java",),
            stage="foundation",
            dependency_missing=java_missing,
            symbol_extraction=not java_missing,
        ),
        LanguageInfo(
            language_id="kotlin",
            display_name="Kotlin",
            extensions=(".kt", ".kts"),
            stage="foundation",
            dependency_missing=kotlin_missing,
            symbol_extraction=not kotlin_missing,
        ),
        LanguageInfo(
            language_id="csharp",
            display_name="C#",
            extensions=(".cs",),
            stage="foundation",
            dependency_missing=csharp_missing,
            symbol_extraction=not csharp_missing,
        ),
        LanguageInfo(
            language_id="c",
            display_name="C",
            extensions=(".c", ".h"),
            stage="foundation",
            dependency_missing=c_missing,
            symbol_extraction=not c_missing,
        ),
        LanguageInfo(
            language_id="cpp",
            display_name="C++",
            extensions=(".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"),
            stage="foundation",
            dependency_missing=cpp_missing,
            symbol_extraction=not cpp_missing,
        ),
    )


def _missing_dependencies(missing: set[str], dependencies: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dependency for dependency in dependencies if dependency in missing)
