"""Backend protocol and deterministic fakes for LSP code intelligence."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from .errors import LspToolError
from .models import (
    LspBackendStatus,
    LspCodeAction,
    LspCodeActionsResult,
    LspDiagnostic,
    LspDiagnosticsResult,
    LspHover,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspPreview,
    LspRange,
    LspSignatureHelp,
    LspSymbol,
    LspSymbolsResult,
    LspTextEdit,
)

LSP_STATUS_TOOL = "lsp.status"
LSP_DIAGNOSTICS_TOOL = "lsp.diagnostics"
LSP_FORMAT_PREVIEW_TOOL = "lsp.format_preview"
LSP_CODE_ACTIONS_TOOL = "lsp.code_actions"
LSP_DOCUMENT_SYMBOLS_TOOL = "lsp.document_symbols"
LSP_WORKSPACE_SYMBOLS_TOOL = "lsp.workspace_symbols"
LSP_DEFINITION_TOOL = "lsp.definition"
LSP_REFERENCES_TOOL = "lsp.references"
LSP_HOVER_TOOL = "lsp.hover"
LSP_SIGNATURE_HELP_TOOL = "lsp.signature_help"

RUFF_TOOLS = frozenset({LSP_DIAGNOSTICS_TOOL, LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL})
PYLSP_TOOLS = frozenset(
    {
        LSP_DOCUMENT_SYMBOLS_TOOL,
        LSP_WORKSPACE_SYMBOLS_TOOL,
        LSP_DEFINITION_TOOL,
        LSP_REFERENCES_TOOL,
        LSP_HOVER_TOOL,
        LSP_SIGNATURE_HELP_TOOL,
    }
)
LSP_OPERATION_TOOLS = RUFF_TOOLS | PYLSP_TOOLS
LSP_TOOL_NAMES = frozenset({LSP_STATUS_TOOL}) | LSP_OPERATION_TOOLS


@dataclass(frozen=True, slots=True)
class DiagnosticsRequest:
    file_path: str


@dataclass(frozen=True, slots=True)
class DocumentSymbolsRequest:
    file_path: str


@dataclass(frozen=True, slots=True)
class WorkspaceSymbolsRequest:
    query: str
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class PositionRequest:
    file_path: str
    position: LspPosition


@dataclass(frozen=True, slots=True)
class ReferencesRequest(PositionRequest):
    include_declaration: bool = False
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class FormatPreviewRequest:
    file_path: str
    include_text_edits: bool = False


@dataclass(frozen=True, slots=True)
class CodeActionsRequest:
    file_path: str
    range: LspRange | None = None
    diagnostics: Sequence[LspDiagnostic] = field(default_factory=tuple)
    include_text_edits: bool = False


@runtime_checkable
class LspBackend(Protocol):
    """Protocol implemented by concrete LSP backends."""

    name: str
    capabilities: frozenset[str]

    async def status(self) -> LspBackendStatus:
        """Return backend health and capability metadata."""

    async def diagnostics(self, request: DiagnosticsRequest) -> LspDiagnosticsResult:
        """Return file diagnostics."""

    async def document_symbols(self, request: DocumentSymbolsRequest) -> LspSymbolsResult:
        """Return document symbols."""

    async def workspace_symbols(self, request: WorkspaceSymbolsRequest) -> LspSymbolsResult:
        """Return workspace symbols."""

    async def definition(self, request: PositionRequest) -> LspLocationsResult:
        """Return definition locations."""

    async def references(self, request: ReferencesRequest) -> LspLocationsResult:
        """Return reference locations."""

    async def hover(self, request: PositionRequest) -> LspHover:
        """Return hover information."""

    async def signature_help(self, request: PositionRequest) -> LspSignatureHelp:
        """Return signature help."""

    async def format_preview(self, request: FormatPreviewRequest) -> LspPreview:
        """Return a formatting preview without mutating files."""

    async def code_actions(self, request: CodeActionsRequest) -> LspCodeActionsResult:
        """Return explicit code-action edit previews."""


class FakeLspBackend:
    """Deterministic in-memory backend used by router and service tests."""

    def __init__(
        self,
        name: str,
        *,
        capabilities: Iterable[str] | None = None,
        healthy: bool = True,
        version: str | None = None,
        detail: str | None = None,
        truncated_methods: Iterable[str] = (),
        crash_methods: Iterable[str] = (),
        unsupported_code_actions: bool = False,
    ):
        self.name = name
        self.capabilities = frozenset(capabilities) if capabilities is not None else _default_capabilities(name)
        self.healthy = healthy
        self.version = version
        self.detail = detail
        self.truncated_methods = frozenset(truncated_methods)
        self.crash_methods = frozenset(crash_methods)
        self.unsupported_code_actions = unsupported_code_actions
        self.calls: list[tuple[str, object | None]] = []

    async def status(self) -> LspBackendStatus:
        self.calls.append(("status", None))
        self._maybe_crash("status")
        return LspBackendStatus(
            name=self.name,
            healthy=self.healthy,
            capabilities=sorted(self.capabilities),
            version=self.version,
            detail=self.detail,
        )

    async def diagnostics(self, request: DiagnosticsRequest) -> LspDiagnosticsResult:
        self._record("diagnostics", request)
        return LspDiagnosticsResult(
            diagnostics=(
                LspDiagnostic(
                    path=request.file_path,
                    range=_sample_range(),
                    message=f"{self.name} diagnostic for {request.file_path}",
                    severity="warning",
                    code="F401",
                    source=self.name,
                ),
            ),
            truncated=self._is_truncated("diagnostics"),
        )

    async def document_symbols(self, request: DocumentSymbolsRequest) -> LspSymbolsResult:
        self._record("document_symbols", request)
        location = LspLocation(path=request.file_path, range=_sample_range())
        return LspSymbolsResult(
            symbols=(
                LspSymbol(name=f"{PurePosixPath(request.file_path).stem}_symbol", kind="function", location=location),
            ),
            truncated=self._is_truncated("document_symbols"),
        )

    async def workspace_symbols(self, request: WorkspaceSymbolsRequest) -> LspSymbolsResult:
        self._record("workspace_symbols", request)
        location = LspLocation(path="pkg/app.py", range=_sample_range())
        return LspSymbolsResult(
            symbols=(LspSymbol(name=request.query or "workspace_symbol", kind="function", location=location),),
            truncated=self._is_truncated("workspace_symbols"),
        )

    async def definition(self, request: PositionRequest) -> LspLocationsResult:
        self._record("definition", request)
        return LspLocationsResult(
            locations=(LspLocation(path=request.file_path, range=_sample_range()),),
            truncated=self._is_truncated("definition"),
        )

    async def references(self, request: ReferencesRequest) -> LspLocationsResult:
        self._record("references", request)
        return LspLocationsResult(
            locations=(LspLocation(path=request.file_path, range=_sample_range()),),
            truncated=self._is_truncated("references"),
        )

    async def hover(self, request: PositionRequest) -> LspHover:
        self._record("hover", request)
        return LspHover(contents=f"{self.name} hover for {request.file_path}", range=_sample_range())

    async def signature_help(self, request: PositionRequest) -> LspSignatureHelp:
        self._record("signature_help", request)
        return LspSignatureHelp(signatures=["func(arg: str) -> None"], active_signature=0, active_parameter=0)

    async def format_preview(self, request: FormatPreviewRequest) -> LspPreview:
        self._record("format_preview", request)
        edits: tuple[LspTextEdit, ...] = ()
        if request.include_text_edits:
            edits = (LspTextEdit(range=_sample_range(), new_text="formatted"),)
        return LspPreview(
            path=request.file_path,
            text_edits=edits,
            preview=f"--- {request.file_path}\n+++ {request.file_path}\n@@\n-unformatted\n+formatted\n",
            truncated=self._is_truncated("format_preview"),
        )

    async def code_actions(self, request: CodeActionsRequest) -> LspCodeActionsResult:
        self._record("code_actions", request)
        if self.unsupported_code_actions:
            raise LspToolError(
                "unsupported_action_shape",
                "code action requires an unsupported opaque command",
                detail=f"{self.name}: workspace/executeCommand",
            )
        return LspCodeActionsResult(
            actions=(
                LspCodeAction(
                    title="Apply Ruff fix",
                    kind="quickfix",
                    diagnostics=tuple(request.diagnostics),
                    edits=(LspTextEdit(range=request.range or _sample_range(), new_text="fixed"),),
                ),
            ),
            path=request.file_path,
            truncated=self._is_truncated("code_actions"),
        )

    def _record(self, method: str, request: object) -> None:
        self.calls.append((method, request))
        self._maybe_crash(method)

    def _maybe_crash(self, method: str) -> None:
        if method in self.crash_methods:
            raise RuntimeError(f"fake backend crash: {method}")

    def _is_truncated(self, method: str) -> bool:
        return method in self.truncated_methods


def _default_capabilities(name: str) -> frozenset[str]:
    if name == "ruff":
        return RUFF_TOOLS
    if name == "pylsp":
        return PYLSP_TOOLS
    return LSP_OPERATION_TOOLS


def _sample_range() -> LspRange:
    return LspRange(start=LspPosition(1, 2), end=LspPosition(1, 5))
