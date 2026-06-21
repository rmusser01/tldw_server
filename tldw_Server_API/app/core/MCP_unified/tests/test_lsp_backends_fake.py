import pytest
from mcp_unified.lsp import LspPosition, LspRange, LspToolError
from mcp_unified.lsp.backends import (
    CodeActionsRequest,
    DiagnosticsRequest,
    DocumentSymbolsRequest,
    FakeLspBackend,
    FormatPreviewRequest,
    PositionRequest,
    ReferencesRequest,
    WorkspaceSymbolsRequest,
)


def _range() -> LspRange:
    return LspRange(start=LspPosition(1, 2), end=LspPosition(1, 5))


async def test_fake_backend_returns_deterministic_diagnostics():
    backend = FakeLspBackend("ruff")

    result = await backend.diagnostics(DiagnosticsRequest(file_path="pkg/app.py"))

    payload = result.to_dict()
    assert payload["diagnostics"][0]["path"] == "pkg/app.py"
    assert payload["diagnostics"][0]["source"] == "ruff"
    assert backend.calls == [("diagnostics", DiagnosticsRequest(file_path="pkg/app.py"))]


async def test_fake_backend_returns_symbol_and_location_results():
    backend = FakeLspBackend("pylsp")

    document_symbols = await backend.document_symbols(DocumentSymbolsRequest(file_path="pkg/app.py"))
    workspace_symbols = await backend.workspace_symbols(WorkspaceSymbolsRequest(query="app"))
    definition = await backend.definition(PositionRequest(file_path="pkg/app.py", position=LspPosition(0, 1)))
    references = await backend.references(
        ReferencesRequest(file_path="pkg/app.py", position=LspPosition(0, 1), include_declaration=True)
    )

    assert document_symbols.symbols[0].name == "app_symbol"
    assert workspace_symbols.symbols[0].name == "app"
    assert definition.locations[0].path == "pkg/app.py"
    assert references.locations[0].path == "pkg/app.py"


async def test_fake_backend_returns_hover_and_signature_help():
    backend = FakeLspBackend("pylsp")

    hover = await backend.hover(PositionRequest(file_path="pkg/app.py", position=LspPosition(0, 1)))
    signature_help = await backend.signature_help(PositionRequest(file_path="pkg/app.py", position=LspPosition(0, 1)))

    assert hover.contents == "pylsp hover for pkg/app.py"
    assert signature_help.signatures == ["func(arg: str) -> None"]


async def test_fake_format_preview_hides_text_edits_until_requested():
    backend = FakeLspBackend("ruff")

    hidden = await backend.format_preview(FormatPreviewRequest(file_path="pkg/app.py"))
    included = await backend.format_preview(FormatPreviewRequest(file_path="pkg/app.py", include_text_edits=True))

    assert hidden.preview and "--- pkg/app.py" in hidden.preview
    assert hidden.text_edits == ()
    assert included.text_edits


async def test_fake_code_actions_return_explicit_text_edits():
    backend = FakeLspBackend("ruff")

    result = await backend.code_actions(CodeActionsRequest(file_path="pkg/app.py", range=_range()))

    assert result.actions[0].title == "Apply Ruff fix"
    assert result.actions[0].edits[0].new_text == "fixed"


async def test_fake_code_actions_can_raise_unsupported_action_shape():
    backend = FakeLspBackend("ruff", unsupported_code_actions=True)

    with pytest.raises(LspToolError) as exc:
        await backend.code_actions(CodeActionsRequest(file_path="pkg/app.py", range=_range()))

    assert exc.value.reason_code == "unsupported_action_shape"


async def test_fake_backend_can_report_truncated_results():
    backend = FakeLspBackend("pylsp", truncated_methods={"references"})

    result = await backend.references(ReferencesRequest(file_path="pkg/app.py", position=LspPosition(0, 1)))

    assert result.truncated is True


async def test_fake_backend_can_report_unhealthy_status():
    backend = FakeLspBackend("ruff", healthy=False, detail="/tmp/ruff failed")

    status = await backend.status()

    assert status.healthy is False
    assert status.detail == "/tmp/ruff failed"


async def test_fake_backend_can_simulate_backend_crash():
    backend = FakeLspBackend("pylsp", crash_methods={"hover"})

    with pytest.raises(RuntimeError, match="fake backend crash"):
        await backend.hover(PositionRequest(file_path="pkg/app.py", position=LspPosition(0, 1)))
