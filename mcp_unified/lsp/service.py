"""High-level LSP code intelligence service facade."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .backends import LSP_OPERATION_TOOLS, LspBackend
from .errors import redact_lsp_detail
from .models import LspDiagnostic, LspPosition, LspRange
from .router import LspCapabilityRouter, ToolPayload

BackendName = Literal["ruff", "pylsp"]


class LspCodeIntelligenceService:
    """Host-neutral facade consumed by MCP modules and gateway runtimes."""

    def __init__(self, router: LspCapabilityRouter):
        self.router = router

    @classmethod
    def from_backends(
        cls,
        *,
        ruff: LspBackend | None = None,
        pylsp: LspBackend | None = None,
    ) -> LspCodeIntelligenceService:
        """Build a service from explicit backend instances."""

        return cls(LspCapabilityRouter(ruff=ruff, pylsp=pylsp))

    async def status(self, *, workspace_root: Path | None = None) -> dict[str, object]:
        """Return safe backend health and capability metadata."""

        backends: dict[str, dict[str, object]] = {}
        available_capabilities: set[str] = set()
        for backend_name in ("ruff", "pylsp"):
            backend = self.router.backends[backend_name]
            if backend is None:
                backends[backend_name] = _missing_backend_status(backend_name)
                continue
            backend_status = await _safe_backend_status(backend_name, backend, workspace_root=workspace_root)
            backends[backend_name] = backend_status
            if backend_status.get("healthy") is True:
                available_capabilities.update(
                    capability
                    for capability in backend_status.get("capabilities", [])
                    if isinstance(capability, str) and capability in LSP_OPERATION_TOOLS
                )

        return {
            "supported_languages": ["python"],
            "backends": backends,
            "capabilities": {
                "available": sorted(available_capabilities),
                "missing": sorted(LSP_OPERATION_TOOLS - available_capabilities),
            },
        }

    async def diagnostics(self, *, file_path: str) -> ToolPayload:
        return await self.router.diagnostics(file_path=file_path)

    async def document_symbols(self, *, file_path: str) -> ToolPayload:
        return await self.router.document_symbols(file_path=file_path)

    async def workspace_symbols(self, *, query: str, limit: int | None = None) -> ToolPayload:
        return await self.router.workspace_symbols(query=query, limit=limit)

    async def definition(self, *, file_path: str, position: LspPosition) -> ToolPayload:
        return await self.router.definition(file_path=file_path, position=position)

    async def references(
        self,
        *,
        file_path: str,
        position: LspPosition,
        include_declaration: bool = False,
        limit: int | None = None,
    ) -> ToolPayload:
        return await self.router.references(
            file_path=file_path,
            position=position,
            include_declaration=include_declaration,
            limit=limit,
        )

    async def hover(self, *, file_path: str, position: LspPosition) -> ToolPayload:
        return await self.router.hover(file_path=file_path, position=position)

    async def signature_help(self, *, file_path: str, position: LspPosition) -> ToolPayload:
        return await self.router.signature_help(file_path=file_path, position=position)

    async def format_preview(self, *, file_path: str, include_text_edits: bool = False) -> ToolPayload:
        return await self.router.format_preview(file_path=file_path, include_text_edits=include_text_edits)

    async def code_actions(
        self,
        *,
        file_path: str,
        range: LspRange | None = None,
        diagnostics: tuple[LspDiagnostic, ...] = (),
        include_text_edits: bool = False,
    ) -> ToolPayload:
        return await self.router.code_actions(
            file_path=file_path,
            range=range,
            diagnostics=diagnostics,
            include_text_edits=include_text_edits,
        )


async def _safe_backend_status(
    backend_name: str,
    backend: LspBackend,
    *,
    workspace_root: Path | None,
) -> dict[str, object]:
    try:
        status = await backend.status()
    except Exception as exc:  # noqa: BLE001
        # Status is a degradation surface: one crashed optional backend must not hide the rest.
        return {
            "name": backend_name,
            "available": True,
            "healthy": False,
            "capabilities": [],
            "reason_code": "backend_unhealthy",
            "detail": redact_lsp_detail(f"{exc.__class__.__name__}: {exc}", workspace_root=workspace_root),
        }

    payload = status.to_dict()
    payload["available"] = True
    payload["reason_code"] = None if status.healthy else "backend_unhealthy"
    for key in ("name", "version", "detail"):
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = redact_lsp_detail(value, workspace_root=workspace_root)
    return payload


def _missing_backend_status(backend_name: str) -> dict[str, object]:
    return {
        "name": backend_name,
        "available": False,
        "healthy": False,
        "capabilities": [],
        "reason_code": "backend_missing",
        "install_hint": f"Run `pip install mcp-unified[lsp]` or configure the {backend_name} executable.",
    }
