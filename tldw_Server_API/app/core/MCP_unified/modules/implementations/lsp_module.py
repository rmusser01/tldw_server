"""tldw-hosted MCP module for LSP code intelligence tools."""

from __future__ import annotations

from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from loguru import logger
from mcp_unified.interfaces.path_scope import PathScopeCandidate
from mcp_unified.lsp import (
    LspCodeIntelligenceService,
    LspDiagnostic,
    LspExecutableResolver,
    LspPosition,
    LspRange,
    LspRuntimeConfig,
    LspToolError,
    PylspLspBackend,
    RuffLspBackend,
    filter_lsp_result_paths,
)
from mcp_unified.lsp.backends import (
    LSP_CODE_ACTIONS_TOOL,
    LSP_DEFINITION_TOOL,
    LSP_DIAGNOSTICS_TOOL,
    LSP_DOCUMENT_SYMBOLS_TOOL,
    LSP_FORMAT_PREVIEW_TOOL,
    LSP_HOVER_TOOL,
    LSP_REFERENCES_TOOL,
    LSP_SIGNATURE_HELP_TOOL,
    LSP_STATUS_TOOL,
    LSP_WORKSPACE_SYMBOLS_TOOL,
)

from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)

from ..base import BaseModule, ModuleConfig, create_tool_definition

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
_POSITION_TOOLS = frozenset({LSP_DEFINITION_TOOL, LSP_REFERENCES_TOOL, LSP_HOVER_TOOL, LSP_SIGNATURE_HELP_TOOL})
_WORKSPACE_SCOPED_TOOLS = frozenset({LSP_STATUS_TOOL, LSP_WORKSPACE_SYMBOLS_TOOL})
_ANALYSIS_TOOLS = frozenset({LSP_DIAGNOSTICS_TOOL, LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL})
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
_FLOAT_SETTING_NAMES = frozenset({"request_timeout_seconds", "startup_timeout_seconds"})
_INT_SETTING_NAMES = frozenset(
    {
        "idle_ttl_seconds",
        "max_diagnostics",
        "max_symbols",
        "max_references",
        "max_hover_bytes",
        "max_preview_bytes",
        "max_stderr_bytes",
    }
)


class LSPModule(BaseModule):
    """Expose Python LSP diagnostics and navigation through tldw MCP."""

    def __init__(
        self,
        config: ModuleConfig,
        service: Any | None = None,
        workspace_root_resolver: Any | None = None,
    ) -> None:
        super().__init__(config)
        self._service = service
        self._workspace_root_resolver = workspace_root_resolver or McpHubWorkspaceRootResolver()
        self._runtime_config = LspRuntimeConfig.from_mapping(_runtime_settings(config.settings))

    async def on_initialize(self) -> None:
        logger.info(f"Initializing LSP module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down LSP module: {self.name}")
        close = getattr(self._service, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True, "workspace_root_resolver": self._workspace_root_resolver is not None}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [_tool_definition(name) for name in _ALL_TOOLS]

    async def extract_path_scope_candidates(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> list[PathScopeCandidate]:
        del context
        if tool_name in _FILE_SCOPED_TOOLS:
            path = _required_path(arguments)
            return [
                PathScopeCandidate(
                    path=path,
                    action="read",
                    source=tool_name,
                    requires_existing_file=True,
                )
            ]
        if tool_name in _WORKSPACE_SCOPED_TOOLS:
            return [PathScopeCandidate(path=".", action="read", source=tool_name)]
        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        _reject_unknown(arguments, allowed=_allowed_arguments(tool_name))
        if tool_name in _FILE_SCOPED_TOOLS:
            path = _required_path(arguments)
            _validate_python_path(path)
        if tool_name in _POSITION_TOOLS:
            _position_from_arguments(arguments)
        if tool_name == LSP_WORKSPACE_SYMBOLS_TOOL:
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query is required")
        if tool_name == LSP_REFERENCES_TOOL:
            _validate_bool_argument(arguments, "include_declaration")
            _validate_limit(arguments)
        if tool_name == LSP_WORKSPACE_SYMBOLS_TOOL:
            _validate_limit(arguments)
        if tool_name in {LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL}:
            _validate_bool_argument(arguments, "include_text_edits")
        if tool_name == LSP_CODE_ACTIONS_TOOL:
            if arguments.get("range") is not None:
                _range_from_payload(arguments["range"])
            diagnostics = arguments.get("diagnostics")
            if diagnostics is not None and not isinstance(diagnostics, list):
                raise ValueError("diagnostics must be a list")
        if tool_name not in _ALL_TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        self.validate_tool_arguments(tool_name, arguments)
        workspace_root = await self._resolve_workspace_root(context)
        service = self._service or self._build_service(workspace_root)

        try:
            result = await self._dispatch(service, tool_name, arguments, workspace_root=workspace_root)
            if isinstance(result, LspToolError):
                raise result
            return self._filter_and_serialize(tool_name, arguments, result, context=context)
        except LspToolError as exc:
            _raise_safe_module_error(exc)  # noqa: B904

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
            return await service.workspace_symbols(query=str(arguments["query"]), limit=arguments.get("limit"))
        if tool_name == LSP_DEFINITION_TOOL:
            return await service.definition(file_path=str(arguments["path"]), position=_position_from_arguments(arguments))
        if tool_name == LSP_REFERENCES_TOOL:
            return await service.references(
                file_path=str(arguments["path"]),
                position=_position_from_arguments(arguments),
                include_declaration=bool(arguments.get("include_declaration", False)),
                limit=arguments.get("limit"),
            )
        if tool_name == LSP_HOVER_TOOL:
            return await service.hover(file_path=str(arguments["path"]), position=_position_from_arguments(arguments))
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
                range=_range_from_payload(arguments["range"]) if arguments.get("range") is not None else None,
                diagnostics=tuple(_diagnostics_from_payload(arguments.get("diagnostics"))),
                include_text_edits=bool(arguments.get("include_text_edits", False)),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _resolve_workspace_root(self, context: Any | None) -> Path:
        resolution = await self._workspace_root_resolver.resolve_for_context(
            context=context,
            user_id=getattr(context, "user_id", None),
            metadata=getattr(context, "metadata", None),
        )
        workspace_root = resolution.get("workspace_root") if isinstance(resolution, dict) else None
        if not isinstance(workspace_root, str) or not workspace_root.strip():
            raise PermissionError("workspace_root_unresolved")
        return Path(workspace_root).resolve(strict=False)

    def _build_service(self, workspace_root: Path) -> LspCodeIntelligenceService:
        explicit_commands = _explicit_lsp_commands(self.config.settings)
        resolver = LspExecutableResolver(workspace_root=workspace_root, explicit_commands=explicit_commands)
        ruff_resolution = resolver.resolve("ruff")
        pylsp_resolution = resolver.resolve("pylsp")
        ruff = (
            RuffLspBackend(workspace_root=workspace_root, argv=ruff_resolution.argv, config=self._runtime_config)
            if ruff_resolution.available
            else None
        )
        pylsp = (
            PylspLspBackend(workspace_root=workspace_root, argv=pylsp_resolution.argv, config=self._runtime_config)
            if pylsp_resolution.available
            else None
        )
        return LspCodeIntelligenceService.from_backends(ruff=ruff, pylsp=pylsp)

    def _filter_and_serialize(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: object,
        *,
        context: Any | None,
    ) -> Any:
        if tool_name == LSP_STATUS_TOOL:
            return _serialize_result(result)
        predicate = _path_allow_predicate(tool_name, arguments, context)
        filtered = filter_lsp_result_paths(result, is_path_allowed=predicate)
        if filtered is not result or hasattr(filtered, "to_dict"):
            return _serialize_result(filtered)
        if isinstance(result, dict):
            return _filter_serialized_result(tool_name, result, is_path_allowed=predicate)
        return _serialize_result(filtered)


def _tool_definition(name: str) -> dict[str, Any]:
    metadata = {
        "category": "analysis" if name in _ANALYSIS_TOOLS else "retrieval",
        "readOnlyHint": True,
        "uses_filesystem": True,
        "path_boundable": True,
        "path_scope_candidate_source": "module",
        "path_scope_action": "read",
        "capabilities": ["lsp.read"],
    }
    tool = create_tool_definition(
        name=name,
        description=_TOOL_DESCRIPTIONS[name],
        parameters=_TOOL_PARAMETERS[name],
        metadata=metadata,
    )
    tool["inputSchema"]["additionalProperties"] = False
    return tool


_POSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "line": {"type": "integer", "minimum": 0},
        "character": {"type": "integer", "minimum": 0},
    },
    "required": ["line", "character"],
}
_RANGE_SCHEMA = {"type": "object", "properties": {"start": _POSITION_SCHEMA, "end": _POSITION_SCHEMA}}
_TOOL_PARAMETERS = {
    LSP_STATUS_TOOL: {"properties": {}},
    LSP_DIAGNOSTICS_TOOL: {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    LSP_DOCUMENT_SYMBOLS_TOOL: {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    LSP_WORKSPACE_SYMBOLS_TOOL: {
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
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
        "properties": {"path": {"type": "string"}, "include_text_edits": {"type": "boolean", "default": False}},
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


def _allowed_arguments(tool_name: str) -> set[str]:
    if tool_name not in _TOOL_PARAMETERS:
        raise ValueError(f"Unknown tool: {tool_name}")
    return set(_TOOL_PARAMETERS[tool_name]["properties"])


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
    return LspRange(start=_position_from_payload(payload.get("start")), end=_position_from_payload(payload.get("end")))


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
        range_payload = item.get("range")
        if not isinstance(path, str) or not isinstance(message, str):
            raise ValueError("diagnostics require path and message")
        diagnostics.append(
            LspDiagnostic(
                path=path,
                range=_range_from_payload(range_payload),
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


def _path_allow_predicate(tool_name: str, arguments: dict[str, Any], context: Any | None):
    metadata = getattr(context, "metadata", {}) if context is not None else {}
    extra_allowed = _metadata_allowed_paths(metadata if isinstance(metadata, dict) else {})
    if tool_name in _FILE_SCOPED_TOOLS:
        request_path = str(arguments.get("path") or "").strip()
        allowed_paths = {request_path, *extra_allowed}
        return lambda path: path in allowed_paths
    if "." in extra_allowed:
        return lambda _path: True
    return lambda path: path in extra_allowed or tool_name in _WORKSPACE_SCOPED_TOOLS


def _metadata_allowed_paths(metadata: dict[str, Any]) -> set[str]:
    raw = metadata.get("lsp_allowed_paths") or metadata.get("path_scope_allowed_paths")
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {item for item in raw if isinstance(item, str) and item}
    return set()


def _filter_serialized_result(tool_name: str, payload: dict[str, Any], *, is_path_allowed: Any) -> dict[str, Any]:
    inner = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    filtered = dict(inner)
    if tool_name == LSP_DIAGNOSTICS_TOOL and isinstance(inner.get("diagnostics"), list):
        diagnostics = [item for item in inner["diagnostics"] if isinstance(item, dict) and is_path_allowed(item.get("path"))]
        filtered["diagnostics"] = diagnostics
        filtered["filtered_count"] = int(inner.get("filtered_count") or 0) + len(inner["diagnostics"]) - len(diagnostics)
    elif tool_name in {LSP_DEFINITION_TOOL, LSP_REFERENCES_TOOL} and isinstance(inner.get("locations"), list):
        locations = [item for item in inner["locations"] if isinstance(item, dict) and is_path_allowed(item.get("path"))]
        filtered["locations"] = locations
        filtered["filtered_count"] = int(inner.get("filtered_count") or 0) + len(inner["locations"]) - len(locations)
    elif tool_name in {LSP_DOCUMENT_SYMBOLS_TOOL, LSP_WORKSPACE_SYMBOLS_TOOL} and isinstance(inner.get("symbols"), list):
        symbols = [
            item
            for item in inner["symbols"]
            if isinstance(item, dict)
            and isinstance(item.get("location"), dict)
            and is_path_allowed(item["location"].get("path"))
        ]
        filtered["symbols"] = symbols
        filtered["filtered_count"] = int(inner.get("filtered_count") or 0) + len(inner["symbols"]) - len(symbols)
    elif tool_name in {LSP_FORMAT_PREVIEW_TOOL, LSP_CODE_ACTIONS_TOOL}:
        path = inner.get("path")
        if not isinstance(path, str) or not is_path_allowed(path):
            raise PermissionError("path_denied")
    return filtered


def _serialize_result(result: object) -> Any:
    to_dict = getattr(result, "to_dict", None)
    if to_dict is not None:
        return to_dict()
    return result


def _explicit_lsp_commands(settings: dict[str, Any]) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for backend_id, setting_name in (("ruff", "ruff_command"), ("pylsp", "pylsp_command")):
        value = settings.get(setting_name)
        if isinstance(value, list) and value:
            commands[backend_id] = value
        elif isinstance(value, str) and value.strip():
            commands[backend_id] = [value.strip()]
    return commands


def _runtime_settings(settings: dict[str, Any]) -> dict[str, Any]:
    values = dict(settings)
    for name in _FLOAT_SETTING_NAMES:
        value = values.get(name)
        if isinstance(value, str) and value.strip():
            values[name] = float(value)
    for name in _INT_SETTING_NAMES:
        value = values.get(name)
        if isinstance(value, str) and value.strip():
            values[name] = int(value)
    return values


def _raise_safe_module_error(exc: LspToolError) -> None:
    if exc.reason_code in {"path_denied", "invalid_path"}:
        raise PermissionError(exc.reason_code) from exc
    raise RuntimeError(exc.reason_code) from exc
