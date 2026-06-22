"""python-lsp-server backend for semantic LSP tools."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from .backends import (
    PYLSP_TOOLS,
    DocumentSymbolsRequest,
    PositionRequest,
    ReferencesRequest,
    WorkspaceSymbolsRequest,
)
from .config import DEFAULT_LSP_CONFIG, LspRuntimeConfig
from .errors import LspToolError
from .jsonrpc import LspJsonRpcClient
from .models import (
    LspBackendStatus,
    LspHover,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspRange,
    LspSignatureHelp,
    LspSymbol,
    LspSymbolsResult,
)

_SYMBOL_KIND_NAMES = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum_member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type_parameter",
}


class PylspLspBackend:
    """pylsp-backed semantic navigation backend."""

    name = "pylsp"
    capabilities = PYLSP_TOOLS

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

    async def document_symbols(self, request: DocumentSymbolsRequest) -> LspSymbolsResult:
        uri = await self._open_document(request.file_path)
        payload = await self._request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
        )
        symbols = _document_symbols(payload, self.workspace_root, fallback_path=request.file_path)
        truncated = len(symbols) > self.config.max_symbols
        return LspSymbolsResult(symbols=tuple(symbols[: self.config.max_symbols]), truncated=truncated)

    async def workspace_symbols(self, request: WorkspaceSymbolsRequest) -> LspSymbolsResult:
        payload = await self._request("workspace/symbol", {"query": request.query})
        symbols = _workspace_symbols(payload, self.workspace_root)
        limit = min(request.limit or self.config.max_symbols, self.config.max_symbols)
        truncated = len(symbols) > limit
        return LspSymbolsResult(symbols=tuple(symbols[:limit]), truncated=truncated)

    async def definition(self, request: PositionRequest) -> LspLocationsResult:
        uri = await self._open_document(request.file_path)
        payload = await self._request(
            "textDocument/definition",
            {"textDocument": {"uri": uri}, "position": _position_to_lsp(request.position)},
        )
        locations = _locations(payload, self.workspace_root)
        truncated = len(locations) > self.config.max_references
        return LspLocationsResult(locations=tuple(locations[: self.config.max_references]), truncated=truncated)

    async def references(self, request: ReferencesRequest) -> LspLocationsResult:
        uri = await self._open_document(request.file_path)
        payload = await self._request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": _position_to_lsp(request.position),
                "context": {"includeDeclaration": request.include_declaration},
            },
        )
        locations = _locations(payload, self.workspace_root)
        limit = min(request.limit or self.config.max_references, self.config.max_references)
        truncated = len(locations) > limit
        return LspLocationsResult(locations=tuple(locations[:limit]), truncated=truncated)

    async def hover(self, request: PositionRequest) -> LspHover:
        uri = await self._open_document(request.file_path)
        payload = await self._request(
            "textDocument/hover",
            {"textDocument": {"uri": uri}, "position": _position_to_lsp(request.position)},
        )
        if not isinstance(payload, dict):
            return LspHover(contents="")
        contents = _hover_text(payload.get("contents"))
        truncated = len(contents.encode("utf-8")) > self.config.max_hover_bytes
        if truncated:
            contents = contents.encode("utf-8")[: self.config.max_hover_bytes].decode("utf-8", errors="ignore")
        lsp_range = payload.get("range")
        return LspHover(
            contents=contents,
            range=_range_from_lsp(lsp_range) if isinstance(lsp_range, dict) else None,
            truncated=truncated,
        )

    async def signature_help(self, request: PositionRequest) -> LspSignatureHelp:
        uri = await self._open_document(request.file_path)
        payload = await self._request(
            "textDocument/signatureHelp",
            {"textDocument": {"uri": uri}, "position": _position_to_lsp(request.position)},
        )
        if not isinstance(payload, dict):
            return LspSignatureHelp(signatures=())
        signatures = [
            str(signature.get("label", ""))
            for signature in payload.get("signatures", [])
            if isinstance(signature, dict) and signature.get("label")
        ]
        return LspSignatureHelp(
            signatures=tuple(signatures),
            active_signature=payload.get("activeSignature"),
            active_parameter=payload.get("activeParameter"),
        )

    async def _request(self, method: str, params: object | None = None) -> object:
        await self._ensure_initialized()
        if self._client is None:
            raise LspToolError("backend_unhealthy", "pylsp client is not initialized")
        return await self._client.request(method, params)

    async def _open_document(self, file_path: str) -> str:
        await self._ensure_initialized()
        if self._client is None:
            raise LspToolError("backend_unhealthy", "pylsp client is not initialized")
        uri = _uri_for_path(self.workspace_root, file_path)
        text = _read_workspace_text(self.workspace_root, file_path)
        version = self._opened_versions.get(uri, 0) + 1
        self._opened_versions[uri] = version
        await self._client.notify(
            "textDocument/didOpen" if version == 1 else "textDocument/didChange",
            _open_or_change_params(uri=uri, text=text, version=version),
        )
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


def _open_or_change_params(*, uri: str, text: str, version: int) -> dict[str, object]:
    if version == 1:
        return {"textDocument": {"uri": uri, "languageId": "python", "version": version, "text": text}}
    return {"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]}


def _uri_for_path(workspace_root: Path, file_path: str) -> str:
    return _workspace_path(workspace_root, file_path).as_uri()


def _read_workspace_text(workspace_root: Path, file_path: str) -> str:
    return _workspace_path(workspace_root, file_path).read_text(encoding="utf-8")


def _workspace_path(workspace_root: Path, file_path: str) -> Path:
    root = workspace_root.resolve(strict=False)
    requested = Path(file_path)
    absolute_requested = requested if requested.is_absolute() else root / requested
    normalized_requested = Path(os.path.normpath(str(absolute_requested)))
    resolved_requested = absolute_requested.resolve(strict=False)
    for candidate in (normalized_requested, resolved_requested):
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise LspToolError("invalid_path", "LSP file path is outside the workspace") from exc
    return resolved_requested


def _position_to_lsp(position: LspPosition) -> dict[str, int]:
    return {"line": position.line, "character": position.character}


def _position_from_lsp(payload: dict[str, object]) -> LspPosition:
    return LspPosition(line=int(payload.get("line", 0)), character=int(payload.get("character", 0)))


def _range_from_lsp(payload: dict[str, object]) -> LspRange:
    start = payload.get("start")
    end = payload.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return LspRange(start=LspPosition(0, 0), end=LspPosition(0, 0))
    return LspRange(start=_position_from_lsp(start), end=_position_from_lsp(end))


def _document_symbols(payload: object, workspace_root: Path, *, fallback_path: str) -> list[LspSymbol]:
    if not isinstance(payload, list):
        return []
    symbols: list[LspSymbol] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "location" in item:
            symbol = _symbol_information(item, workspace_root)
            if symbol is not None:
                symbols.append(symbol)
            continue
        symbols.extend(_document_symbol(item, fallback_path=fallback_path, container_name=None))
    return symbols


def _document_symbol(
    payload: dict[str, object],
    *,
    fallback_path: str,
    container_name: str | None,
) -> list[LspSymbol]:
    name = str(payload.get("name", ""))
    lsp_range = payload.get("selectionRange") or payload.get("range") or {}
    if not name or not isinstance(lsp_range, dict):
        return []
    symbol = LspSymbol(
        name=name,
        kind=_symbol_kind(payload.get("kind")),
        location=LspLocation(path=fallback_path, range=_range_from_lsp(lsp_range)),
        container_name=container_name,
    )
    symbols = [symbol]
    children = payload.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                symbols.extend(_document_symbol(child, fallback_path=fallback_path, container_name=name))
    return symbols


def _workspace_symbols(payload: object, workspace_root: Path) -> list[LspSymbol]:
    if not isinstance(payload, list):
        return []
    return [symbol for item in payload if isinstance(item, dict) for symbol in [_symbol_information(item, workspace_root)] if symbol]


def _symbol_information(payload: dict[str, object], workspace_root: Path) -> LspSymbol | None:
    location_payload = payload.get("location")
    if not isinstance(location_payload, dict):
        return None
    location = _location_from_lsp(location_payload, workspace_root)
    if location is None:
        return None
    name = str(payload.get("name", ""))
    if not name:
        return None
    return LspSymbol(
        name=name,
        kind=_symbol_kind(payload.get("kind")),
        location=location,
        container_name=payload.get("containerName") if isinstance(payload.get("containerName"), str) else None,
    )


def _locations(payload: object, workspace_root: Path) -> list[LspLocation]:
    if payload is None:
        return []
    items = payload if isinstance(payload, list) else [payload]
    locations: list[LspLocation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        location = _location_from_lsp(item, workspace_root)
        if location is not None:
            locations.append(location)
    return locations


def _location_from_lsp(payload: dict[str, object], workspace_root: Path) -> LspLocation | None:
    uri = payload.get("uri") or payload.get("targetUri")
    range_payload = payload.get("range") or payload.get("targetRange")
    if not isinstance(uri, str) or not isinstance(range_payload, dict):
        return None
    return LspLocation(path=_relative_path_from_uri(uri, workspace_root), range=_range_from_lsp(range_payload))


def _relative_path_from_uri(uri: str, workspace_root: Path) -> str:
    parsed = urlparse(uri)
    path = Path(unquote(parsed.path)).resolve(strict=False)
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return os.path.basename(path)


def _symbol_kind(value: object) -> str:
    try:
        return _SYMBOL_KIND_NAMES.get(int(value), str(value))
    except (TypeError, ValueError):
        return "unknown"


def _hover_text(contents: object) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        value = contents.get("value")
        return str(value) if value is not None else ""
    if isinstance(contents, list):
        return "\n".join(_hover_text(item) for item in contents)
    return ""
