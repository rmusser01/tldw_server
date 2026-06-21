"""Capability router for Python-first LSP tooling."""

from __future__ import annotations

from collections.abc import Callable

from .backends import (
    LSP_CODE_ACTIONS_TOOL,
    LSP_DEFINITION_TOOL,
    LSP_DIAGNOSTICS_TOOL,
    LSP_DOCUMENT_SYMBOLS_TOOL,
    LSP_FORMAT_PREVIEW_TOOL,
    LSP_HOVER_TOOL,
    LSP_OPERATION_TOOLS,
    LSP_REFERENCES_TOOL,
    LSP_SIGNATURE_HELP_TOOL,
    LSP_STATUS_TOOL,
    LSP_TOOL_NAMES,
    LSP_WORKSPACE_SYMBOLS_TOOL,
    PYLSP_TOOLS,
    RUFF_TOOLS,
    CodeActionsRequest,
    DiagnosticsRequest,
    DocumentSymbolsRequest,
    FormatPreviewRequest,
    LspBackend,
    PositionRequest,
    ReferencesRequest,
    WorkspaceSymbolsRequest,
)
from .errors import LspToolError
from .models import LspDiagnostic, LspPosition, LspRange

ToolPayload = dict[str, object]


class LspCapabilityRouter:
    """Route stable `lsp.*` tool calls to the preferred backend."""

    def __init__(self, *, ruff: LspBackend | None = None, pylsp: LspBackend | None = None):
        self._backends: dict[str, LspBackend | None] = {"ruff": ruff, "pylsp": pylsp}

    @property
    def backends(self) -> dict[str, LspBackend | None]:
        """Return a shallow copy of configured backend references."""

        return dict(self._backends)

    async def diagnostics(self, *, file_path: str) -> ToolPayload:
        return await self._call(
            LSP_DIAGNOSTICS_TOOL,
            "diagnostics",
            DiagnosticsRequest(file_path=file_path),
        )

    async def document_symbols(self, *, file_path: str) -> ToolPayload:
        return await self._call(
            LSP_DOCUMENT_SYMBOLS_TOOL,
            "document_symbols",
            DocumentSymbolsRequest(file_path=file_path),
        )

    async def workspace_symbols(self, *, query: str, limit: int | None = None) -> ToolPayload:
        return await self._call(
            LSP_WORKSPACE_SYMBOLS_TOOL,
            "workspace_symbols",
            WorkspaceSymbolsRequest(query=query, limit=limit),
        )

    async def definition(self, *, file_path: str, position: LspPosition) -> ToolPayload:
        return await self._call(
            LSP_DEFINITION_TOOL,
            "definition",
            PositionRequest(file_path=file_path, position=position),
        )

    async def references(
        self,
        *,
        file_path: str,
        position: LspPosition,
        include_declaration: bool = False,
        limit: int | None = None,
    ) -> ToolPayload:
        return await self._call(
            LSP_REFERENCES_TOOL,
            "references",
            ReferencesRequest(
                file_path=file_path,
                position=position,
                include_declaration=include_declaration,
                limit=limit,
            ),
        )

    async def hover(self, *, file_path: str, position: LspPosition) -> ToolPayload:
        return await self._call(
            LSP_HOVER_TOOL,
            "hover",
            PositionRequest(file_path=file_path, position=position),
        )

    async def signature_help(self, *, file_path: str, position: LspPosition) -> ToolPayload:
        return await self._call(
            LSP_SIGNATURE_HELP_TOOL,
            "signature_help",
            PositionRequest(file_path=file_path, position=position),
        )

    async def format_preview(self, *, file_path: str, include_text_edits: bool = False) -> ToolPayload:
        return await self._call(
            LSP_FORMAT_PREVIEW_TOOL,
            "format_preview",
            FormatPreviewRequest(file_path=file_path, include_text_edits=include_text_edits),
        )

    async def code_actions(
        self,
        *,
        file_path: str,
        range: LspRange | None = None,
        diagnostics: tuple[LspDiagnostic, ...] = (),
        include_text_edits: bool = False,
    ) -> ToolPayload:
        return await self._call(
            LSP_CODE_ACTIONS_TOOL,
            "code_actions",
            CodeActionsRequest(
                file_path=file_path,
                range=range,
                diagnostics=diagnostics,
                include_text_edits=include_text_edits,
            ),
        )

    async def _call(self, tool_name: str, method_name: str, request: object) -> ToolPayload:
        backend = await self._select_backend(tool_name)
        operation = getattr(backend, method_name)
        try:
            result = await operation(request)
        except LspToolError:
            raise
        except Exception as exc:
            raise LspToolError(
                "backend_unhealthy",
                f"{backend.name} backend failed while handling {tool_name}",
                detail=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        return _result_envelope(tool_name=tool_name, backend_name=backend.name, result=result)

    async def _select_backend(self, tool_name: str) -> LspBackend:
        backend_name = _preferred_backend_name(tool_name)
        backend = self._backends.get(backend_name)
        if backend is None:
            raise LspToolError("backend_missing", f"{backend_name} backend is not configured")
        if tool_name not in backend.capabilities:
            raise LspToolError(
                "capability_unavailable",
                f"{backend.name} backend does not provide {tool_name}",
                detail=f"available capabilities: {', '.join(sorted(backend.capabilities))}",
            )
        try:
            status = await backend.status()
        except LspToolError:
            raise
        except Exception as exc:
            raise LspToolError(
                "backend_unhealthy",
                f"{backend.name} backend status check failed",
                detail=f"{exc.__class__.__name__}: {exc}",
            ) from exc
        if not status.healthy:
            raise LspToolError(
                "backend_unhealthy",
                f"{backend.name} backend is unhealthy",
                detail=status.detail,
            )
        return backend


def _preferred_backend_name(tool_name: str) -> str:
    if tool_name in RUFF_TOOLS:
        return "ruff"
    if tool_name in PYLSP_TOOLS:
        return "pylsp"
    raise LspToolError("capability_unavailable", f"unknown LSP tool: {tool_name}")


def _result_envelope(*, tool_name: str, backend_name: str, result: object) -> ToolPayload:
    payload = _to_payload(result)
    envelope: ToolPayload = {"tool": tool_name, "backend": backend_name, "result": payload}
    truncated = getattr(result, "truncated", None)
    if isinstance(truncated, bool):
        envelope["truncated"] = truncated
    return envelope


def _to_payload(result: object) -> object:
    to_dict: Callable[[], object] | None = getattr(result, "to_dict", None)
    if to_dict is not None:
        return to_dict()
    return result


__all__ = [
    "LSP_CODE_ACTIONS_TOOL",
    "LSP_DEFINITION_TOOL",
    "LSP_DIAGNOSTICS_TOOL",
    "LSP_DOCUMENT_SYMBOLS_TOOL",
    "LSP_FORMAT_PREVIEW_TOOL",
    "LSP_HOVER_TOOL",
    "LSP_OPERATION_TOOLS",
    "LSP_REFERENCES_TOOL",
    "LSP_SIGNATURE_HELP_TOOL",
    "LSP_STATUS_TOOL",
    "LSP_TOOL_NAMES",
    "LSP_WORKSPACE_SYMBOLS_TOOL",
    "PYLSP_TOOLS",
    "RUFF_TOOLS",
    "LspCapabilityRouter",
]
