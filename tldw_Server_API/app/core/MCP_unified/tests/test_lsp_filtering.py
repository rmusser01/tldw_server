import pytest
from mcp_unified.lsp import (
    LspCodeAction,
    LspCodeActionsResult,
    LspDiagnostic,
    LspDiagnosticsResult,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspPreview,
    LspRange,
    LspSymbol,
    LspSymbolsResult,
    LspTextEdit,
    LspToolError,
)
from mcp_unified.lsp.filtering import filter_lsp_result_paths


def _sample_range() -> LspRange:
    return LspRange(start=LspPosition(0, 0), end=LspPosition(0, 4))


def _is_src_path(path: str) -> bool:
    return path.startswith("src/")


def test_filter_locations_removes_denied_paths_and_counts_them():
    result = LspLocationsResult(
        locations=(
            LspLocation(path="src/allowed.py", range=_sample_range()),
            LspLocation(path="private/secret.py", range=_sample_range()),
        )
    )

    filtered = filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert isinstance(filtered, LspLocationsResult)
    assert [location.path for location in filtered.locations] == ["src/allowed.py"]
    assert filtered.filtered_count == 1
    assert filtered.to_dict()["filtered_count"] == 1


def test_filter_diagnostics_removes_denied_paths_and_counts_them():
    result = LspDiagnosticsResult(
        diagnostics=(
            LspDiagnostic(path="src/allowed.py", range=_sample_range(), message="unused import"),
            LspDiagnostic(path="private/secret.py", range=_sample_range(), message="syntax error"),
        )
    )

    filtered = filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert isinstance(filtered, LspDiagnosticsResult)
    assert [diagnostic.path for diagnostic in filtered.diagnostics] == ["src/allowed.py"]
    assert filtered.filtered_count == 1


def test_filter_symbols_removes_denied_paths_and_counts_them():
    result = LspSymbolsResult(
        symbols=(
            LspSymbol(
                name="allowed",
                kind="function",
                location=LspLocation(path="src/allowed.py", range=_sample_range()),
            ),
            LspSymbol(
                name="secret",
                kind="function",
                location=LspLocation(path="private/secret.py", range=_sample_range()),
            ),
        )
    )

    filtered = filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert isinstance(filtered, LspSymbolsResult)
    assert [symbol.name for symbol in filtered.symbols] == ["allowed"]
    assert filtered.filtered_count == 1


def test_filter_preview_rejects_denied_path():
    result = LspPreview(
        path="private/secret.py",
        text_edits=(LspTextEdit(range=_sample_range(), new_text="x"),),
        preview="--- private/secret.py\n+++ private/secret.py\n",
    )

    with pytest.raises(LspToolError) as exc:
        filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert exc.value.reason_code == "path_denied"


def test_filter_code_actions_rejects_denied_path():
    result = LspCodeActionsResult(
        path="private/secret.py",
        actions=(LspCodeAction(title="fix", edits=(LspTextEdit(range=_sample_range(), new_text="x"),)),),
    )

    with pytest.raises(LspToolError) as exc:
        filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert exc.value.reason_code == "path_denied"


def test_filter_code_actions_rejects_missing_path_when_edits_exist():
    result = LspCodeActionsResult(
        actions=(LspCodeAction(title="fix", edits=(LspTextEdit(range=_sample_range(), new_text="x"),)),),
    )

    with pytest.raises(LspToolError) as exc:
        filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert exc.value.reason_code == "path_denied"


def test_filter_code_actions_preserves_allowed_path():
    result = LspCodeActionsResult(
        path="src/allowed.py",
        actions=(LspCodeAction(title="fix", edits=(LspTextEdit(range=_sample_range(), new_text="x"),)),),
    )

    filtered = filter_lsp_result_paths(result, is_path_allowed=_is_src_path)

    assert filtered is result
