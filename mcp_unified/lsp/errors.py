"""Structured error payloads for LSP MCP tools."""

from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_LSP_CONFIG


LSP_REASON_CODES = frozenset(
    {
        "tool_not_granted",
        "path_denied",
        "invalid_path",
        "invalid_position",
        "backend_missing",
        "backend_unhealthy",
        "backend_timeout",
        "capability_unavailable",
        "response_truncated",
        "preview_too_large",
        "unsupported_action_shape",
        "unsupported_language",
        "workspace_not_supported",
        "config_error",
    }
)


def redact_lsp_detail(
    detail: str | None,
    *,
    workspace_root: Path | None = None,
    max_length: int = DEFAULT_LSP_CONFIG.max_stderr_bytes,
) -> str | None:
    """Redact workspace paths from an LSP error detail and bound its size."""

    if detail is None:
        return None

    safe_detail = detail
    if workspace_root is not None:
        workspace_text = str(workspace_root)
        if workspace_text:
            safe_detail = safe_detail.replace(workspace_text, "<workspace>")

    if len(safe_detail) > max_length:
        suffix = "..."
        if max_length <= len(suffix):
            return suffix[:max_length]
        return safe_detail[: max_length - len(suffix)] + suffix
    return safe_detail


class LspToolError(RuntimeError):
    """Exception raised for user-facing LSP tool failures."""

    def __init__(
        self,
        reason_code: str,
        message: str | None = None,
        *,
        detail: str | None = None,
    ):
        super().__init__(message or reason_code)
        if reason_code not in LSP_REASON_CODES:
            raise ValueError(f"unknown LSP reason_code: {reason_code}")
        self.reason_code = reason_code
        self.detail = detail

    def to_payload(self, *, workspace_root: Path | None = None) -> dict[str, object]:
        """Return a deterministic JSON-serializable error payload."""

        safe_detail = redact_lsp_detail(self.detail, workspace_root=workspace_root)
        safe_message = redact_lsp_detail(str(self), workspace_root=workspace_root)
        return {"reason_code": self.reason_code, "message": safe_message, "detail": safe_detail}
