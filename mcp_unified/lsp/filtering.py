"""Path filtering helpers for LSP result payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .errors import LspToolError
from .models import (
    LspCodeActionsResult,
    LspDiagnosticsResult,
    LspLocationsResult,
    LspPreview,
    LspSymbolsResult,
)

PathAllowPredicate = Callable[[str], bool]


def filter_lsp_result_paths(result: object, *, is_path_allowed: PathAllowPredicate) -> object:
    """Filter or reject LSP result paths according to the provided predicate."""

    if isinstance(result, LspDiagnosticsResult):
        diagnostics = tuple(diagnostic for diagnostic in result.diagnostics if is_path_allowed(diagnostic.path))
        return replace(
            result,
            diagnostics=diagnostics,
            filtered_count=result.filtered_count + len(result.diagnostics) - len(diagnostics),
        )
    if isinstance(result, LspSymbolsResult):
        symbols = tuple(symbol for symbol in result.symbols if is_path_allowed(symbol.location.path))
        return replace(
            result,
            symbols=symbols,
            filtered_count=result.filtered_count + len(result.symbols) - len(symbols),
        )
    if isinstance(result, LspLocationsResult):
        locations = tuple(location for location in result.locations if is_path_allowed(location.path))
        return replace(
            result,
            locations=locations,
            filtered_count=result.filtered_count + len(result.locations) - len(locations),
        )
    if isinstance(result, LspPreview):
        _require_allowed_path(result.path, is_path_allowed=is_path_allowed)
        return result
    if isinstance(result, LspCodeActionsResult):
        if result.path is not None or _has_text_edits(result):
            _require_allowed_path(result.path, is_path_allowed=is_path_allowed)
        return result
    return result


def _require_allowed_path(path: str | None, *, is_path_allowed: PathAllowPredicate) -> None:
    if path is None or not is_path_allowed(path):
        raise LspToolError("path_denied", "LSP result includes a path outside the active grant")


def _has_text_edits(result: LspCodeActionsResult) -> bool:
    return any(action.edits for action in result.actions)
