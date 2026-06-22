"""Standalone gateway runtime for LSP-only MCP deployments."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext

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
)
from .config import DEFAULT_LSP_CONFIG, LspRuntimeConfig
from .errors import LspToolError
from .executables import LspExecutableResolver
from .filtering import PathAllowPredicate, filter_lsp_result_paths
from .models import LspDiagnostic, LspPosition, LspRange
from .pylsp import PylspLspBackend
from .ruff import RuffLspBackend
from .service import LspCodeIntelligenceService

_FILE_SCOPED_TOOLS = frozenset(
    {
        LSP_DIAGNOSTICS_TOOL,
        LSP_DOCUMENT_SYMBOLS_TOOL,
        LSP_DEFINITION_TOOL,
        LSP_REFERENCES_TOOL,
        LSP_HOVER_TOOL,
        LSP_SIGNATURE_HELP_TOOL,
        LSP_FORMAT_PREVIEW_TOOL,
        LSP_CODE_ACTIONS_TOOL,
    }
)
_POSITION_TOOLS = frozenset(
    {LSP_DEFINITION_TOOL, LSP_REFERENCES_TOOL, LSP_HOVER_TOOL, LSP_SIGNATURE_HELP_TOOL}
)
_ANALYSIS_TOOLS = frozenset(
    {LSP_DIAGNOSTICS_TOOL, LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL}
)
_ALL_TOOLS = (
    LSP_STATUS_TOOL,
    LSP_DIAGNOSTICS_TOOL,
    LSP_DOCUMENT_SYMBOLS_TOOL,
    LSP_WORKSPACE_SYMBOLS_TOOL,
    LSP_DEFINITION_TOOL,
    LSP_REFERENCES_TOOL,
    LSP_HOVER_TOOL,
    LSP_SIGNATURE_HELP_TOOL,
    LSP_FORMAT_PREVIEW_TOOL,
    LSP_CODE_ACTIONS_TOOL,
)
_MAX_LIMIT = 500
_TOOL_ERROR_RESULT_REASON_CODES = frozenset(
    {
        "backend_missing",
        "backend_unhealthy",
        "backend_timeout",
        "capability_unavailable",
        "preview_too_large",
        "unsupported_action_shape",
        "unsupported_language",
        "config_error",
        "response_truncated",
    }
)


class LspGatewayRuntime:
    """Expose LSP code intelligence tools through the standalone gateway runtime."""

    name = "mcp-unified-lsp-gateway"
    version = "0.1.0"

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        service: Any | None = None,
        config: LspRuntimeConfig | None = None,
        path_allow_predicate: PathAllowPredicate | None = None,
        explicit_commands: Mapping[str, object] | None = None,
    ) -> None:
        self.workspace_root = _coerce_workspace_root(workspace_root)
        self._service = service
        self._config = config or DEFAULT_LSP_CONFIG
        self._path_allow_predicate = path_allow_predicate
        self._explicit_commands = dict(explicit_commands or {})

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return the LSP tool catalog visible to this standalone runtime."""

        del context
        return [_tool_definition(name) for name in _ALL_TOOLS]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute one LSP tool and return an MCP tool result."""

        del context
        _validate_tool_arguments(name, arguments)
        workspace_root = self._require_workspace_root()
        service = self._service_for_workspace(workspace_root)
        try:
            result = await self._dispatch(service, name, arguments, workspace_root=workspace_root)
            payload = _serialize_result(result)
            if name == LSP_STATUS_TOOL and isinstance(payload, dict):
                payload = _status_payload(payload)
            elif name != LSP_STATUS_TOOL:
                payload = self._filter_and_serialize(name, payload, result)
        except LspToolError as exc:
            if exc.reason_code in {"path_denied", "invalid_path"}:
                raise GatewayPolicyDenied(
                    exc.reason_code,
                    reason_code=exc.reason_code,
                    provenance={"tool": name},
                ) from exc
            if exc.reason_code in _TOOL_ERROR_RESULT_REASON_CODES:
                return _tool_error_result(name, exc, workspace_root=workspace_root)
            raise
        return _tool_result(name, payload)

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return this standalone runtime's single logical module."""

        del context
        return [
            {
                "id": "lsp",
                "name": "LSP Code Intelligence",
                "version": self.version,
                "enabled": True,
            }
        ]

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Return runtime health metadata without requiring a tool call."""

        result = await self.call_tool(LSP_STATUS_TOOL, {}, context)
        structured = result.get("structuredContent")
        return {"lsp": structured if isinstance(structured, dict) else result}

    def _require_workspace_root(self) -> Path:
        if self.workspace_root is None:
            raise GatewayPolicyDenied(
                "workspace_not_supported",
                reason_code="workspace_not_supported",
                provenance={"module": "lsp"},
            )
        return self.workspace_root

    def _build_service(self, workspace_root: Path) -> LspCodeIntelligenceService:
        resolver = LspExecutableResolver(
            workspace_root=workspace_root,
            explicit_commands=self._explicit_commands,
        )
        ruff_resolution = resolver.resolve("ruff")
        pylsp_resolution = resolver.resolve("pylsp")
        ruff = (
            RuffLspBackend(
                workspace_root=workspace_root,
                argv=ruff_resolution.argv,
                config=self._config,
            )
            if ruff_resolution.available
            else None
        )
        pylsp = (
            PylspLspBackend(
                workspace_root=workspace_root,
                argv=pylsp_resolution.argv,
                config=self._config,
            )
            if pylsp_resolution.available
            else None
        )
        return LspCodeIntelligenceService.from_backends(ruff=ruff, pylsp=pylsp)

    def _service_for_workspace(self, workspace_root: Path) -> Any:
        if self._service is None:
            self._service = self._build_service(workspace_root)
        return self._service

    async def _dispatch(
        self,
        service: Any,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        workspace_root: Path,
    ) -> Any:
        if tool_name == LSP_STATUS_TOOL:
            return await service.status(workspace_root=workspace_root)
        if tool_name == LSP_DIAGNOSTICS_TOOL:
            return await service.diagnostics(file_path=str(arguments["path"]))
        if tool_name == LSP_DOCUMENT_SYMBOLS_TOOL:
            return await service.document_symbols(file_path=str(arguments["path"]))
        if tool_name == LSP_WORKSPACE_SYMBOLS_TOOL:
            return await service.workspace_symbols(
                query=str(arguments["query"]),
                limit=arguments.get("limit"),
            )
        if tool_name == LSP_DEFINITION_TOOL:
            return await service.definition(
                file_path=str(arguments["path"]),
                position=_position_from_arguments(arguments),
            )
        if tool_name == LSP_REFERENCES_TOOL:
            return await service.references(
                file_path=str(arguments["path"]),
                position=_position_from_arguments(arguments),
                include_declaration=bool(arguments.get("include_declaration", False)),
                limit=arguments.get("limit"),
            )
        if tool_name == LSP_HOVER_TOOL:
            return await service.hover(
                file_path=str(arguments["path"]),
                position=_position_from_arguments(arguments),
            )
        if tool_name == LSP_SIGNATURE_HELP_TOOL:
            return await service.signature_help(
                file_path=str(arguments["path"]),
                position=_position_from_arguments(arguments),
            )
        if tool_name == LSP_FORMAT_PREVIEW_TOOL:
            return await service.format_preview(
                file_path=str(arguments["path"]),
                include_text_edits=bool(arguments.get("include_text_edits", False)),
            )
        if tool_name == LSP_CODE_ACTIONS_TOOL:
            return await service.code_actions(
                file_path=str(arguments["path"]),
                range=_range_from_payload(arguments["range"])
                if arguments.get("range") is not None
                else None,
                diagnostics=tuple(_diagnostics_from_payload(arguments.get("diagnostics"))),
                include_text_edits=bool(arguments.get("include_text_edits", False)),
            )
        raise NotImplementedError(tool_name)

    def _filter_and_serialize(
        self,
        tool_name: str,
        payload: object,
        original_result: object,
    ) -> object:
        predicate = self._path_allow_predicate or _default_path_allow_predicate(
            self._require_workspace_root()
        )
        filtered = filter_lsp_result_paths(original_result, is_path_allowed=predicate)
        if filtered is not original_result or hasattr(filtered, "to_dict"):
            return _serialize_result(filtered)
        if isinstance(payload, dict):
            return _filter_serialized_result(tool_name, payload, is_path_allowed=predicate)
        return _serialize_result(filtered)


def _tool_definition(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": _TOOL_DESCRIPTIONS[name],
        "inputSchema": {
            "type": "object",
            "properties": _TOOL_PARAMETERS[name]["properties"],
            "required": _TOOL_PARAMETERS[name].get("required", []),
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
        "metadata": {
            "category": "analysis" if name in _ANALYSIS_TOOLS else "retrieval",
            "uses_filesystem": True,
            "path_boundable": True,
            "path_scope_candidate_source": "runtime",
            "path_scope_action": "read",
            "capabilities": ["lsp.read"],
        },
    }


_POSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "line": {"type": "integer", "minimum": 0},
        "character": {"type": "integer", "minimum": 0},
    },
    "required": ["line", "character"],
}
_RANGE_SCHEMA = {
    "type": "object",
    "properties": {"start": _POSITION_SCHEMA, "end": _POSITION_SCHEMA},
    "required": ["start", "end"],
}
_TOOL_PARAMETERS = {
    LSP_STATUS_TOOL: {"properties": {}},
    LSP_DIAGNOSTICS_TOOL: {
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    LSP_DOCUMENT_SYMBOLS_TOOL: {
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    LSP_WORKSPACE_SYMBOLS_TOOL: {
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
    },
    LSP_DEFINITION_TOOL: {
        "properties": {"path": {"type": "string"}, "position": _POSITION_SCHEMA},
        "required": ["path", "position"],
    },
    LSP_REFERENCES_TOOL: {
        "properties": {
            "path": {"type": "string"},
            "position": _POSITION_SCHEMA,
            "include_declaration": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["path", "position"],
    },
    LSP_HOVER_TOOL: {
        "properties": {"path": {"type": "string"}, "position": _POSITION_SCHEMA},
        "required": ["path", "position"],
    },
    LSP_SIGNATURE_HELP_TOOL: {
        "properties": {"path": {"type": "string"}, "position": _POSITION_SCHEMA},
        "required": ["path", "position"],
    },
    LSP_FORMAT_PREVIEW_TOOL: {
        "properties": {
            "path": {"type": "string"},
            "include_text_edits": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    },
    LSP_CODE_ACTIONS_TOOL: {
        "properties": {
            "path": {"type": "string"},
            "range": _RANGE_SCHEMA,
            "diagnostics": {"type": "array"},
            "include_text_edits": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    },
}
_TOOL_DESCRIPTIONS = {
    LSP_STATUS_TOOL: "Inspect LSP backend availability for the active workspace.",
    LSP_DIAGNOSTICS_TOOL: "Return bounded Python diagnostics for one workspace file.",
    LSP_DOCUMENT_SYMBOLS_TOOL: "Return symbols declared in one Python workspace file.",
    LSP_WORKSPACE_SYMBOLS_TOOL: "Search Python symbols across the active workspace.",
    LSP_DEFINITION_TOOL: "Resolve the definition location for a Python symbol position.",
    LSP_REFERENCES_TOOL: "List reference locations for a Python symbol position.",
    LSP_HOVER_TOOL: "Return hover/type information for a Python symbol position.",
    LSP_SIGNATURE_HELP_TOOL: "Return function signature help at a Python position.",
    LSP_FORMAT_PREVIEW_TOOL: "Preview Python formatting edits without mutating files.",
    LSP_CODE_ACTIONS_TOOL: "Preview explicit Python LSP code-action edits without mutating files.",
}


def _coerce_workspace_root(workspace_root: str | Path | None) -> Path | None:
    if workspace_root is None:
        return None
    return Path(workspace_root).expanduser().resolve(strict=False)


def _validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    if tool_name not in LSP_TOOL_NAMES:
        raise NotImplementedError(tool_name)
    _reject_unknown(arguments, allowed=set(_TOOL_PARAMETERS[tool_name]["properties"]))
    if tool_name in _FILE_SCOPED_TOOLS:
        path = _required_path(arguments)
        _validate_python_path(path)
    if tool_name in _POSITION_TOOLS:
        _position_from_arguments(arguments)
    if tool_name == LSP_WORKSPACE_SYMBOLS_TOOL:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        _validate_limit(arguments)
    if tool_name == LSP_REFERENCES_TOOL:
        _validate_bool_argument(arguments, "include_declaration")
        _validate_limit(arguments)
    if tool_name in {LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL}:
        _validate_bool_argument(arguments, "include_text_edits")
    if tool_name == LSP_CODE_ACTIONS_TOOL:
        if arguments.get("range") is not None:
            _range_from_payload(arguments["range"])
        diagnostics = arguments.get("diagnostics")
        if diagnostics is not None and not isinstance(diagnostics, list):
            raise ValueError("diagnostics must be a list")


def _reject_unknown(arguments: dict[str, Any], *, allowed: set[str]) -> None:
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ValueError(f"unknown arguments: {', '.join(unknown)}")


def _required_path(arguments: dict[str, Any]) -> str:
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    return path.strip()


def _validate_python_path(path: str) -> None:
    cleaned = path.strip()
    windows_path = PureWindowsPath(cleaned)
    if PurePath(cleaned).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("absolute paths are not allowed")
    normalized = cleaned.replace("\\", "/")
    if normalized.endswith("/") or "/../" in f"/{normalized}/":
        raise ValueError("path traversal is not allowed")
    if not normalized.endswith(".py"):
        raise ValueError("only python .py paths are supported")


def _position_from_arguments(arguments: dict[str, Any]) -> LspPosition:
    position = arguments.get("position")
    if not isinstance(position, dict):
        raise ValueError("position is required")
    return _position_from_payload(position)


def _position_from_payload(payload: object) -> LspPosition:
    if not isinstance(payload, dict):
        raise ValueError("position must be an object")
    line = payload.get("line")
    character = payload.get("character")
    if not isinstance(line, int) or isinstance(line, bool):
        raise ValueError("line must be a non-negative integer")
    if not isinstance(character, int) or isinstance(character, bool):
        raise ValueError("character must be a non-negative integer")
    return LspPosition(line=line, character=character)


def _range_from_payload(payload: object) -> LspRange:
    if not isinstance(payload, dict):
        raise ValueError("range must be an object")
    return LspRange(
        start=_position_from_payload(payload.get("start")),
        end=_position_from_payload(payload.get("end")),
    )


def _diagnostics_from_payload(payload: object) -> list[LspDiagnostic]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("diagnostics must be a list")
    diagnostics: list[LspDiagnostic] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("diagnostics must contain objects")
        path = item.get("path")
        message = item.get("message")
        if not isinstance(path, str) or not isinstance(message, str):
            raise ValueError("diagnostics require path and message")
        diagnostics.append(
            LspDiagnostic(
                path=path,
                range=_range_from_payload(item.get("range")),
                message=message,
                severity=item.get("severity") if isinstance(item.get("severity"), str) else None,
                code=item.get("code") if isinstance(item.get("code"), (str, int)) else None,
                source=item.get("source") if isinstance(item.get("source"), str) else None,
            )
        )
    return diagnostics


def _validate_bool_argument(arguments: dict[str, Any], key: str) -> None:
    value = arguments.get(key)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")


def _validate_limit(arguments: dict[str, Any]) -> None:
    value = arguments.get("limit")
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError("limit must be a positive integer")
    if isinstance(value, int) and value > _MAX_LIMIT:
        raise ValueError(f"limit exceeds maximum ({_MAX_LIMIT})")


def _default_path_allow_predicate(workspace_root: Path) -> Callable[[str], bool]:
    root = workspace_root.resolve(strict=False)

    def _is_allowed(path: str) -> bool:
        if not isinstance(path, str) or not path.strip():
            return False
        cleaned = path.strip().replace("\\", "/")
        windows_path = PureWindowsPath(cleaned)
        if PurePath(cleaned).is_absolute() or windows_path.is_absolute() or windows_path.drive:
            return False
        if cleaned == ".":
            return True
        if cleaned.endswith("/") or "/../" in f"/{cleaned}/":
            return False
        candidate = (root / cleaned).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return True

    return _is_allowed


def _filter_serialized_result(
    tool_name: str,
    payload: dict[str, Any],
    *,
    is_path_allowed: PathAllowPredicate,
) -> dict[str, Any]:
    inner = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    filtered = dict(inner)
    if tool_name == LSP_DIAGNOSTICS_TOOL and isinstance(inner.get("diagnostics"), list):
        diagnostics = [
            item
            for item in inner["diagnostics"]
            if isinstance(item, dict) and is_path_allowed(item.get("path", ""))
        ]
        filtered["diagnostics"] = diagnostics
        filtered["filtered_count"] = int(inner.get("filtered_count") or 0) + len(inner["diagnostics"]) - len(diagnostics)
    elif tool_name in {LSP_DEFINITION_TOOL, LSP_REFERENCES_TOOL} and isinstance(inner.get("locations"), list):
        locations = [
            item
            for item in inner["locations"]
            if isinstance(item, dict) and is_path_allowed(item.get("path", ""))
        ]
        filtered["locations"] = locations
        filtered["filtered_count"] = int(inner.get("filtered_count") or 0) + len(inner["locations"]) - len(locations)
    elif tool_name in {LSP_DOCUMENT_SYMBOLS_TOOL, LSP_WORKSPACE_SYMBOLS_TOOL} and isinstance(inner.get("symbols"), list):
        symbols = [
            item
            for item in inner["symbols"]
            if isinstance(item, dict)
            and isinstance(item.get("location"), dict)
            and is_path_allowed(item["location"].get("path", ""))
        ]
        filtered["symbols"] = symbols
        filtered["filtered_count"] = int(inner.get("filtered_count") or 0) + len(inner["symbols"]) - len(symbols)
    elif tool_name in {LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL}:
        path = inner.get("path")
        if not isinstance(path, str) or not is_path_allowed(path):
            raise LspToolError("path_denied", "LSP result includes a path outside the active grant")
    return filtered


def _status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    available = set()
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, dict) and isinstance(capabilities.get("available"), list):
        available = {item for item in capabilities["available"] if isinstance(item, str)}
    status = "healthy" if LSP_OPERATION_TOOLS.issubset(available) else "degraded"
    return {"status": status, **payload}


def _tool_result(tool_name: str, payload: object) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": _tool_result_text(tool_name, payload)}],
        "structuredContent": payload,
        "isError": False,
    }


def _tool_error_result(
    tool_name: str,
    exc: LspToolError,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    payload = exc.to_payload(workspace_root=workspace_root)
    return {
        "content": [{"type": "text", "text": f"{tool_name}: {exc.reason_code}"}],
        "structuredContent": payload,
        "isError": True,
    }


def _tool_result_text(tool_name: str, payload: object) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("status"), str):
        return f"{tool_name}: {payload['status']}"
    return f"{tool_name}: ok"


def _serialize_result(result: object) -> Any:
    to_dict = getattr(result, "to_dict", None)
    if to_dict is not None:
        return to_dict()
    try:
        json.dumps(result)
    except TypeError:
        return {"type": result.__class__.__name__}
    return result


__all__ = ["LspGatewayRuntime"]
