"""Public model and error contracts for MCP LSP tooling."""

from .config import DEFAULT_LSP_CONFIG, LspRuntimeConfig
from .errors import LSP_REASON_CODES, LspToolError, redact_lsp_detail
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

__all__ = [
    "DEFAULT_LSP_CONFIG",
    "LSP_REASON_CODES",
    "LspBackendStatus",
    "LspCodeAction",
    "LspCodeActionsResult",
    "LspDiagnostic",
    "LspDiagnosticsResult",
    "LspHover",
    "LspLocation",
    "LspLocationsResult",
    "LspPosition",
    "LspPreview",
    "LspRange",
    "LspRuntimeConfig",
    "LspSignatureHelp",
    "LspSymbol",
    "LspSymbolsResult",
    "LspTextEdit",
    "LspToolError",
    "redact_lsp_detail",
]
