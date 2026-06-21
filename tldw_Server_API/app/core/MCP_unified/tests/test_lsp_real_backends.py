import os
import shutil
import stat
from pathlib import Path

import pytest
from mcp_unified.lsp import LspPosition, LspRange, LspToolError
from mcp_unified.lsp.backends import (
    CodeActionsRequest,
    DiagnosticsRequest,
    DocumentSymbolsRequest,
    FormatPreviewRequest,
    PositionRequest,
)
from mcp_unified.lsp.executables import LspExecutableResolver
from mcp_unified.lsp.pylsp import PylspLspBackend, _read_workspace_text, _uri_for_path
from mcp_unified.lsp.ruff import RuffLspBackend


def _write_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_resolver_prefers_explicit_backend_command(tmp_path: Path):
    ruff = _write_executable(tmp_path / "ruff")
    resolver = LspExecutableResolver(workspace_root=tmp_path, explicit_commands={"ruff": [str(ruff), "server"]})

    resolved = resolver.resolve("ruff")

    assert resolved.available is True
    assert resolved.source == "explicit"
    assert resolved.argv == (str(ruff), "server")


def test_resolver_discovers_project_virtualenv_executable(tmp_path: Path):
    bin_dir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    pylsp = _write_executable(bin_dir / "pylsp")
    resolver = LspExecutableResolver(workspace_root=tmp_path, path_env="")

    resolved = resolver.resolve("pylsp")

    assert resolved.available is True
    assert resolved.source == "venv"
    assert resolved.argv == (str(pylsp),)


def test_resolver_discovers_path_executable(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ruff = _write_executable(bin_dir / "ruff")
    resolver = LspExecutableResolver(workspace_root=tmp_path, path_env=str(bin_dir))

    resolved = resolver.resolve("ruff")

    assert resolved.available is True
    assert resolved.source == "path"
    assert resolved.argv == (str(ruff), "server")


def test_resolver_reports_missing_backend_without_raising(tmp_path: Path):
    resolver = LspExecutableResolver(workspace_root=tmp_path, path_env="")

    resolved = resolver.resolve("ruff")

    assert resolved.available is False
    assert resolved.reason_code == "backend_missing"
    assert "pip install" in resolved.install_hint


@pytest.mark.parametrize(
    "command",
    [
        "ruff server",
        ["npx", "ruff"],
        ["docker", "exec", "ruff"],
        ["devbox", "run", "ruff"],
    ],
)
def test_resolver_rejects_shell_strings_and_wrappers(tmp_path: Path, command: object):
    resolver = LspExecutableResolver(workspace_root=tmp_path, explicit_commands={"ruff": command})

    with pytest.raises(LspToolError) as exc:
        resolver.resolve("ruff")

    assert exc.value.reason_code == "config_error"


def test_resolver_rejects_non_executable_paths(tmp_path: Path):
    ruff = tmp_path / "ruff"
    ruff.write_text("#!/bin/sh\n", encoding="utf-8")
    resolver = LspExecutableResolver(workspace_root=tmp_path, explicit_commands={"ruff": [str(ruff), "server"]})

    with pytest.raises(LspToolError) as exc:
        resolver.resolve("ruff")

    assert exc.value.reason_code == "config_error"


def test_lsp_workspace_file_read_rejects_path_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("SECRET = True\n", encoding="utf-8")

    with pytest.raises(LspToolError) as exc:
        _read_workspace_text(tmp_path, "../outside.py")

    assert exc.value.reason_code == "invalid_path"


def test_lsp_workspace_uri_rejects_absolute_path_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("SECRET = True\n", encoding="utf-8")

    with pytest.raises(LspToolError) as exc:
        _uri_for_path(tmp_path, str(outside))

    assert exc.value.reason_code == "invalid_path"


def _real_backend_argv(backend_id: str, tmp_path: Path) -> tuple[str, ...]:
    if os.getenv("TLDW_MCP_LSP_REAL_BACKENDS") != "1":
        pytest.skip("set TLDW_MCP_LSP_REAL_BACKENDS=1 to run real LSP backend tests")
    resolved = LspExecutableResolver(workspace_root=tmp_path).resolve(backend_id)
    if not resolved.available:
        pytest.skip(f"{backend_id} executable is not available")
    return resolved.argv


async def test_real_ruff_diagnostics_detects_lint_issue(tmp_path: Path):
    argv = _real_backend_argv("ruff", tmp_path)
    source = tmp_path / "bad.py"
    source.write_text("import os\n\nprint('x')\n", encoding="utf-8")
    backend = RuffLspBackend(workspace_root=tmp_path, argv=argv)
    try:
        result = await backend.diagnostics(DiagnosticsRequest(file_path="bad.py"))
    finally:
        await backend.close()

    assert any(diagnostic.code in {"F401", "unused-import"} for diagnostic in result.diagnostics)


async def test_real_ruff_format_preview_returns_unified_diff_without_default_text_edits(tmp_path: Path):
    argv = _real_backend_argv("ruff", tmp_path)
    source = tmp_path / "format_me.py"
    source.write_text("x=1\n", encoding="utf-8")
    backend = RuffLspBackend(workspace_root=tmp_path, argv=argv)
    try:
        result = await backend.format_preview(FormatPreviewRequest(file_path="format_me.py"))
        with_edits = await backend.format_preview(
            FormatPreviewRequest(file_path="format_me.py", include_text_edits=True)
        )
    finally:
        await backend.close()

    assert result.preview and "--- format_me.py" in result.preview
    assert result.text_edits == ()
    assert with_edits.text_edits


async def test_real_ruff_code_actions_reject_opaque_command_actions(tmp_path: Path):
    argv = _real_backend_argv("ruff", tmp_path)
    source = tmp_path / "action.py"
    source.write_text("import os\n", encoding="utf-8")
    backend = RuffLspBackend(workspace_root=tmp_path, argv=argv)
    try:
        try:
            result = await backend.code_actions(
                CodeActionsRequest(
                    file_path="action.py",
                    range=LspRange(start=LspPosition(0, 0), end=LspPosition(0, 9)),
                )
            )
        except LspToolError as exc:
            assert exc.reason_code == "unsupported_action_shape"
        else:
            assert result.actions
    finally:
        await backend.close()


async def test_real_pylsp_document_symbols_returns_python_symbols(tmp_path: Path):
    if shutil.which("pylsp") is None:
        pytest.skip("pylsp executable is not available")
    argv = _real_backend_argv("pylsp", tmp_path)
    source = tmp_path / "symbols.py"
    source.write_text("class Widget:\n    pass\n\ndef build():\n    return Widget()\n", encoding="utf-8")
    backend = PylspLspBackend(workspace_root=tmp_path, argv=argv)
    try:
        result = await backend.document_symbols(DocumentSymbolsRequest(file_path="symbols.py"))
    finally:
        await backend.close()

    assert {symbol.name for symbol in result.symbols} >= {"Widget", "build"}


async def test_real_pylsp_definition_resolves_local_function(tmp_path: Path):
    if shutil.which("pylsp") is None:
        pytest.skip("pylsp executable is not available")
    argv = _real_backend_argv("pylsp", tmp_path)
    source = tmp_path / "defs.py"
    source.write_text("def target():\n    return 1\n\nvalue = target()\n", encoding="utf-8")
    backend = PylspLspBackend(workspace_root=tmp_path, argv=argv)
    try:
        result = await backend.definition(PositionRequest(file_path="defs.py", position=LspPosition(3, 9)))
    finally:
        await backend.close()

    assert any(location.path == "defs.py" for location in result.locations)
