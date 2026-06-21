"""Ruff LSP backend for diagnostics and edit previews."""

from __future__ import annotations

import difflib
from pathlib import Path

from .backends import RUFF_TOOLS, CodeActionsRequest, DiagnosticsRequest, FormatPreviewRequest
from .config import DEFAULT_LSP_CONFIG, LspRuntimeConfig
from .errors import LspToolError
from .jsonrpc import LspJsonRpcClient
from .models import (
    LspBackendStatus,
    LspCodeAction,
    LspCodeActionsResult,
    LspDiagnostic,
    LspDiagnosticsResult,
    LspPreview,
    LspRange,
    LspTextEdit,
)
from .pylsp import _position_to_lsp, _range_from_lsp, _read_workspace_text, _uri_for_path


class RuffLspBackend:
    """Ruff-backed diagnostics, formatting, and code-action preview backend."""

    name = "ruff"
    capabilities = RUFF_TOOLS

    def __init__(
        self,
        *,
        workspace_root: Path,
        argv: tuple[str, ...],
        config: LspRuntimeConfig = DEFAULT_LSP_CONFIG,
    ):
        self.workspace_root = workspace_root.resolve()
        self.argv = argv
        self.config = config
        self._client: LspJsonRpcClient | None = None
        self._initialized = False
        self._opened_versions: dict[str, int] = {}

    async def status(self) -> LspBackendStatus:
        return LspBackendStatus(name=self.name, healthy=True, capabilities=sorted(self.capabilities))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._initialized = False

    async def diagnostics(self, request: DiagnosticsRequest) -> LspDiagnosticsResult:
        await self._open_document(request.file_path)
        client = self._require_client()
        notification = await client.wait_for_notification("textDocument/publishDiagnostics")
        params = notification.get("params") if isinstance(notification.get("params"), dict) else {}
        diagnostics_payload = params.get("diagnostics", []) if isinstance(params, dict) else []
        diagnostics = _diagnostics(diagnostics_payload, path=request.file_path)
        truncated = len(diagnostics) > self.config.max_diagnostics
        return LspDiagnosticsResult(
            diagnostics=tuple(diagnostics[: self.config.max_diagnostics]),
            truncated=truncated,
        )

    async def format_preview(self, request: FormatPreviewRequest) -> LspPreview:
        uri = await self._open_document(request.file_path)
        original = _read_workspace_text(self.workspace_root, request.file_path)
        edits_payload = await self._request(
            "textDocument/formatting",
            {
                "textDocument": {"uri": uri},
                "options": {"tabSize": 4, "insertSpaces": True},
            },
        )
        text_edits = _text_edits(edits_payload)
        formatted = _apply_text_edits(original, text_edits)
        preview = _unified_diff(request.file_path, original, formatted)
        truncated = len(preview.encode("utf-8")) > self.config.max_preview_bytes
        if truncated:
            preview = preview.encode("utf-8")[: self.config.max_preview_bytes].decode("utf-8", errors="ignore")
        return LspPreview(
            path=request.file_path,
            text_edits=tuple(text_edits) if request.include_text_edits else (),
            preview=preview,
            truncated=truncated,
        )

    async def code_actions(self, request: CodeActionsRequest) -> LspCodeActionsResult:
        uri = await self._open_document(request.file_path)
        range_payload = _range_to_lsp(request.range) if request.range else _whole_file_range()
        response = await self._request(
            "textDocument/codeAction",
            {
                "textDocument": {"uri": uri},
                "range": range_payload,
                "context": {"diagnostics": [_diagnostic_to_lsp(diagnostic) for diagnostic in request.diagnostics]},
            },
        )
        actions, saw_opaque = _code_actions(response, uri=uri)
        if not actions and saw_opaque:
            raise LspToolError("unsupported_action_shape", "Ruff returned only opaque command-shaped actions")
        return LspCodeActionsResult(actions=tuple(actions), truncated=False)

    async def _request(self, method: str, params: object | None = None) -> object:
        await self._ensure_initialized()
        return await self._require_client().request(method, params)

    async def _open_document(self, file_path: str) -> str:
        await self._ensure_initialized()
        uri = _uri_for_path(self.workspace_root, file_path)
        text = _read_workspace_text(self.workspace_root, file_path)
        version = self._opened_versions.get(uri, 0) + 1
        self._opened_versions[uri] = version
        method = "textDocument/didOpen" if version == 1 else "textDocument/didChange"
        params: dict[str, object]
        if version == 1:
            params = {"textDocument": {"uri": uri, "languageId": "python", "version": version, "text": text}}
        else:
            params = {"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]}
        await self._require_client().notify(method, params)
        return uri

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._client = LspJsonRpcClient(argv=self.argv, workspace_root=self.workspace_root, config=self.config)
        await self._client.start()
        await self._client.request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.workspace_root.as_uri(),
                "capabilities": {},
                "workspaceFolders": [{"uri": self.workspace_root.as_uri(), "name": self.workspace_root.name}],
            },
        )
        await self._client.notify("initialized", {})
        self._initialized = True

    def _require_client(self) -> LspJsonRpcClient:
        if self._client is None:
            raise LspToolError("backend_unhealthy", "Ruff client is not initialized")
        return self._client


def _diagnostics(payload: object, *, path: str) -> list[LspDiagnostic]:
    if not isinstance(payload, list):
        return []
    diagnostics: list[LspDiagnostic] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        lsp_range = item.get("range")
        if not isinstance(lsp_range, dict):
            continue
        diagnostics.append(
            LspDiagnostic(
                path=path,
                range=_range_from_lsp(lsp_range),
                message=str(item.get("message", "")),
                severity=_severity(item.get("severity")),
                code=item.get("code") if isinstance(item.get("code"), (str, int)) else None,
                source=str(item.get("source", "ruff")),
            )
        )
    return diagnostics


def _severity(value: object) -> str | None:
    return {1: "error", 2: "warning", 3: "information", 4: "hint"}.get(value if isinstance(value, int) else None)


def _text_edits(payload: object) -> list[LspTextEdit]:
    if not isinstance(payload, list):
        return []
    edits: list[LspTextEdit] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        lsp_range = item.get("range")
        if not isinstance(lsp_range, dict):
            continue
        edits.append(LspTextEdit(range=_range_from_lsp(lsp_range), new_text=str(item.get("newText", ""))))
    return edits


def _apply_text_edits(text: str, edits: list[LspTextEdit]) -> str:
    result = text
    for edit in sorted(edits, key=lambda item: _offset_for_position(result, item.range.start), reverse=True):
        start = _offset_for_position(result, edit.range.start)
        end = _offset_for_position(result, edit.range.end)
        result = result[:start] + edit.new_text + result[end:]
    return result


def _offset_for_position(text: str, position: object) -> int:
    line = getattr(position, "line", 0)
    character = getattr(position, "character", 0)
    lines = text.splitlines(keepends=True)
    if line >= len(lines):
        return len(text)
    return sum(len(part) for part in lines[:line]) + min(character, len(lines[line]))


def _unified_diff(path: str, original: str, formatted: str) -> str:
    diff_lines = difflib.unified_diff(
        original.splitlines(),
        formatted.splitlines(),
        fromfile=path,
        tofile=path,
        lineterm="",
    )
    diff = "\n".join(diff_lines)
    return f"{diff}\n" if diff else ""


def _range_to_lsp(value: LspRange) -> dict[str, dict[str, int]]:
    return {"start": _position_to_lsp(value.start), "end": _position_to_lsp(value.end)}


def _whole_file_range() -> dict[str, dict[str, int]]:
    return {"start": {"line": 0, "character": 0}, "end": {"line": 999_999, "character": 0}}


def _diagnostic_to_lsp(diagnostic: LspDiagnostic) -> dict[str, object]:
    payload: dict[str, object] = {
        "range": _range_to_lsp(diagnostic.range),
        "message": diagnostic.message,
    }
    if diagnostic.code is not None:
        payload["code"] = diagnostic.code
    if diagnostic.source is not None:
        payload["source"] = diagnostic.source
    return payload


def _code_actions(payload: object, *, uri: str) -> tuple[list[LspCodeAction], bool]:
    if not isinstance(payload, list):
        return [], False
    actions: list[LspCodeAction] = []
    saw_opaque = False
    for item in payload:
        if not isinstance(item, dict):
            continue
        edits = _workspace_edit_text_edits(item.get("edit"), uri=uri)
        if edits:
            actions.append(
                LspCodeAction(
                    title=str(item.get("title", "code action")),
                    kind=item.get("kind") if isinstance(item.get("kind"), str) else None,
                    edits=tuple(edits),
                )
            )
        elif item.get("command") is not None:
            saw_opaque = True
    return actions, saw_opaque


def _workspace_edit_text_edits(payload: object, *, uri: str) -> list[LspTextEdit]:
    if not isinstance(payload, dict):
        return []
    edits: list[LspTextEdit] = []
    changes = payload.get("changes")
    if isinstance(changes, dict):
        edits.extend(_text_edits(changes.get(uri)))
    document_changes = payload.get("documentChanges")
    if isinstance(document_changes, list):
        for change in document_changes:
            if not isinstance(change, dict):
                continue
            text_document = change.get("textDocument")
            if isinstance(text_document, dict) and text_document.get("uri") == uri:
                edits.extend(_text_edits(change.get("edits")))
    return edits
