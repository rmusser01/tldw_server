"""
Workspace-bounded native CodeGraph MCP module.

This module exposes bounded foreground indexing, file inventory, Python symbol
search, and same-file call relationship queries for trusted workspace roots.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.dependencies import probe_codegraph_dependencies
from tldw_Server_API.app.core.CodeGraph.indexer import CodeGraphIndexer
from tldw_Server_API.app.core.CodeGraph.language_registry import CodeGraphLanguageRegistry
from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode, IndexedFile, IndexRunSummary, WorkspaceResolution
from tldw_Server_API.app.core.CodeGraph.workspace import CodeGraphWorkspaceResolver, WorkspaceRootResolver
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository

from ..base import BaseModule, ModuleConfig, create_tool_definition

_EMPTY_COUNTS = {"files": 0, "nodes": 0, "edges": 0, "unresolved_refs": 0}
_FOREGROUND_MODE = "foreground"


class CodeGraphModule(BaseModule):
    """Workspace-scoped CodeGraph index tools for MCP."""

    def __init__(
        self,
        config: ModuleConfig,
        workspace_root_resolver: WorkspaceRootResolver | None = None,
    ) -> None:
        """Create module-scoped settings, dependency probes, and workspace resolver."""
        super().__init__(config)
        self._settings = CodeGraphSettings.from_mapping(config.settings)
        self._dependency_health = probe_codegraph_dependencies()
        self._language_registry = CodeGraphLanguageRegistry(self._dependency_health)
        self._workspace = CodeGraphWorkspaceResolver(workspace_root_resolver, self._settings)

    async def on_initialize(self) -> None:
        """Log CodeGraph module initialization."""
        logger.info(f"Initializing CodeGraph module: {self.name}")

    async def on_shutdown(self) -> None:
        """Log CodeGraph module shutdown."""
        logger.info(f"Shutting down CodeGraph module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        """Return lightweight module health flags for MCP status reporting."""
        return {
            "initialized": True,
            "workspace_root_resolver": self._workspace is not None,
            "dependencies_available": self._dependency_health.available,
        }

    async def get_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions for native CodeGraph operations."""
        shared_metadata = {
            "uses_filesystem": True,
            "path_boundable": True,
        }
        workspace_metadata = {
            **shared_metadata,
            "path_argument_hints": [],
        }
        file_listing_metadata = {
            **shared_metadata,
            "path_argument_hints": ["path"],
        }

        status_tool = create_tool_definition(
            name="codegraph.status",
            description="Inspect CodeGraph index availability and language support for the active workspace.",
            parameters={"properties": {}},
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["codegraph.read"],
                **workspace_metadata,
            },
        )
        status_tool["inputSchema"]["additionalProperties"] = False

        index_tool = create_tool_definition(
            name="codegraph.index",
            description="Run bounded foreground CodeGraph file indexing for the active workspace.",
            parameters={
                "properties": {
                    "mode": {"type": "string", "description": "Only foreground is supported in Stage 1"},
                    "force": {"type": "boolean"},
                    "languages": {"type": "array"},
                    "max_files": {"type": "integer"},
                },
            },
            metadata={
                "category": "management",
                "capabilities": ["codegraph.write"],
                **workspace_metadata,
            },
        )
        index_tool["inputSchema"]["additionalProperties"] = False

        sync_tool = create_tool_definition(
            name="codegraph.sync",
            description="Run bounded foreground CodeGraph file sync for the active workspace.",
            parameters={
                "properties": {
                    "mode": {"type": "string", "description": "Only foreground is supported in Stage 1"},
                    "languages": {"type": "array"},
                    "max_files": {"type": "integer"},
                },
            },
            metadata={
                "category": "management",
                "capabilities": ["codegraph.write"],
                **workspace_metadata,
            },
        )
        sync_tool["inputSchema"]["additionalProperties"] = False

        files_tool = create_tool_definition(
            name="codegraph.files",
            description="List files currently recorded in the active workspace CodeGraph index.",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "Optional workspace-relative path prefix"},
                    "pattern": {"type": "string", "description": "Optional shell-style path pattern"},
                    "format": {"type": "string", "description": "flat, tree, or grouped"},
                    "include_metadata": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["codegraph.read"],
                **file_listing_metadata,
            },
        )
        files_tool["inputSchema"]["additionalProperties"] = False

        search_tool = create_tool_definition(
            name="codegraph.search",
            description="Search indexed CodeGraph symbols for the active workspace.",
            parameters={
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string"},
                    "language": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["codegraph.read"],
                **workspace_metadata,
            },
        )
        search_tool["inputSchema"]["additionalProperties"] = False

        node_tool = create_tool_definition(
            name="codegraph.node",
            description="Fetch one indexed CodeGraph symbol by node id or exact symbol name.",
            parameters={
                "properties": {
                    "node_id": {"type": "string"},
                    "symbol": {"type": "string"},
                    "include_code": {"type": "boolean"},
                },
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["codegraph.read"],
                **workspace_metadata,
            },
        )
        node_tool["inputSchema"]["additionalProperties"] = False

        callers_tool = create_tool_definition(
            name="codegraph.callers",
            description="List indexed call relationships that target a symbol.",
            parameters={
                "properties": {
                    "node_id": {"type": "string"},
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["codegraph.read"],
                **workspace_metadata,
            },
        )
        callers_tool["inputSchema"]["additionalProperties"] = False

        callees_tool = create_tool_definition(
            name="codegraph.callees",
            description="List indexed call relationships emitted by a symbol.",
            parameters={
                "properties": {
                    "node_id": {"type": "string"},
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["codegraph.read"],
                **workspace_metadata,
            },
        )
        callees_tool["inputSchema"]["additionalProperties"] = False

        return [status_tool, index_tool, sync_tool, files_tool, search_tool, node_tool, callers_tool, callees_tool]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        """Validate and dispatch one CodeGraph MCP tool invocation."""
        args = self.sanitize_input(arguments or {})
        self.validate_tool_arguments(tool_name, args)

        resolution = await self._workspace.resolve(context)

        if tool_name == "codegraph.status":
            return await self._status(resolution)

        if tool_name == "codegraph.index":
            return await asyncio.to_thread(
                self._run_index,
                resolution,
                bool(args.get("force", False)),
                args.get("languages"),
                args.get("max_files"),
            )

        if tool_name == "codegraph.sync":
            return await asyncio.to_thread(
                self._run_sync,
                resolution,
                args.get("languages"),
                args.get("max_files"),
            )

        if tool_name == "codegraph.files":
            return await asyncio.to_thread(
                self._list_files,
                resolution,
                args.get("path"),
                args.get("pattern"),
                str(args.get("format") or "flat"),
                bool(args.get("include_metadata", True)),
                args.get("limit"),
            )

        if tool_name == "codegraph.search":
            return await asyncio.to_thread(
                self._search_nodes,
                resolution,
                str(args["query"]),
                args.get("kind"),
                args.get("language"),
                args.get("limit"),
            )

        if tool_name == "codegraph.node":
            return await asyncio.to_thread(
                self._get_node,
                resolution,
                args.get("node_id"),
                args.get("symbol"),
                bool(args.get("include_code", False)),
            )

        if tool_name == "codegraph.callers":
            return await asyncio.to_thread(
                self._list_relationships,
                resolution,
                "callers",
                args.get("node_id"),
                args.get("symbol"),
                args.get("limit"),
            )

        if tool_name == "codegraph.callees":
            return await asyncio.to_thread(
                self._list_relationships,
                resolution,
                "callees",
                args.get("node_id"),
                args.get("symbol"),
                args.get("limit"),
            )

        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validate the input shape for a supported CodeGraph tool."""
        if tool_name == "codegraph.status":
            self._reject_unknown(arguments, allowed=set())
            return

        if tool_name == "codegraph.index":
            self._reject_unknown(arguments, allowed={"mode", "force", "languages", "max_files"})
            self._validate_foreground_mode(arguments.get("mode"))
            force = arguments.get("force")
            if force is not None and not isinstance(force, bool):
                raise ValueError("force must be a boolean")
            self._validate_languages(arguments.get("languages"))
            self._validate_positive_int(arguments.get("max_files"), "max_files")
            return

        if tool_name == "codegraph.sync":
            self._reject_unknown(arguments, allowed={"mode", "languages", "max_files"})
            self._validate_foreground_mode(arguments.get("mode"))
            self._validate_languages(arguments.get("languages"))
            self._validate_positive_int(arguments.get("max_files"), "max_files")
            return

        if tool_name == "codegraph.files":
            self._reject_unknown(arguments, allowed={"path", "pattern", "format", "include_metadata", "limit"})
            path = arguments.get("path")
            pattern = arguments.get("pattern")
            file_format = arguments.get("format")
            include_metadata = arguments.get("include_metadata")
            if path is not None and not isinstance(path, str):
                raise ValueError("path must be a string")
            if pattern is not None and not isinstance(pattern, str):
                raise ValueError("pattern must be a string")
            if file_format is not None and file_format not in {"flat", "tree", "grouped"}:
                raise ValueError("format must be flat, tree, or grouped")
            if include_metadata is not None and not isinstance(include_metadata, bool):
                raise ValueError("include_metadata must be a boolean")
            self._validate_positive_int(arguments.get("limit"), "limit")
            return

        if tool_name == "codegraph.search":
            self._reject_unknown(arguments, allowed={"query", "kind", "language", "limit"})
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            arguments["query"] = query.strip()
            kind = arguments.get("kind")
            language = arguments.get("language")
            if kind is not None and (not isinstance(kind, str) or not kind.strip()):
                raise ValueError("kind must be a non-empty string")
            if isinstance(kind, str):
                arguments["kind"] = kind.strip()
            if language is not None:
                if not isinstance(language, str) or language not in self._language_registry.known_language_ids():
                    raise ValueError("language must be a known language id")
            self._validate_positive_int(arguments.get("limit"), "limit")
            return

        if tool_name == "codegraph.node":
            self._reject_unknown(arguments, allowed={"node_id", "symbol", "include_code"})
            self._validate_node_selector(arguments)
            include_code = arguments.get("include_code")
            if include_code is not None and not isinstance(include_code, bool):
                raise ValueError("include_code must be a boolean")
            return

        if tool_name in {"codegraph.callers", "codegraph.callees"}:
            self._reject_unknown(arguments, allowed={"node_id", "symbol", "limit"})
            self._validate_node_selector(arguments)
            self._validate_positive_int(arguments.get("limit"), "limit")
            return

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _status(self, resolution: WorkspaceResolution) -> dict[str, Any]:
        """Build a read-only status payload without creating the index database."""
        if resolution.index_db_path.exists():
            counts, last_run = await asyncio.to_thread(self._read_status_from_repository, resolution)
            index_present = True
        else:
            counts = dict(_EMPTY_COUNTS)
            last_run = None
            index_present = False

        return {
            "dependency_available": self._dependency_health.available,
            "dependency_missing": list(self._dependency_health.missing),
            "dependency_present": list(self._dependency_health.present),
            "languages": [asdict(language) for language in self._language_registry.list_languages()],
            "workspace_key": resolution.workspace_key,
            "workspace_id": resolution.workspace_id,
            "workspace_source": resolution.source,
            "index_present": index_present,
            "index_db_path": str(resolution.index_db_path),
            "counts": counts,
            "last_index_run": _index_run_to_dict(last_run),
        }

    def _read_status_from_repository(
        self,
        resolution: WorkspaceResolution,
    ) -> tuple[dict[str, int], IndexRunSummary | None]:
        """Read status details from an existing index repository."""
        repository = CodeGraphRepository(resolution.index_db_path)
        return repository.counts(), repository.last_index_run()

    def _run_index(
        self,
        resolution: WorkspaceResolution,
        force: bool,
        languages: list[str] | None,
        max_files: int | None,
    ) -> dict[str, Any]:
        """Run foreground indexing and serialize the result."""
        repository = CodeGraphRepository(resolution.index_db_path)
        indexer = self._new_indexer()
        result = indexer.index_workspace(
            resolution.workspace_root,
            resolution.workspace_key,
            repository,
            force=force,
            languages=languages,
            max_files=max_files,
        )
        return _index_result_to_dict(result, resolution)

    def _run_sync(
        self,
        resolution: WorkspaceResolution,
        languages: list[str] | None,
        max_files: int | None,
    ) -> dict[str, Any]:
        """Run foreground sync and serialize the result."""
        repository = CodeGraphRepository(resolution.index_db_path)
        indexer = self._new_indexer()
        result = indexer.sync_workspace(
            resolution.workspace_root,
            resolution.workspace_key,
            repository,
            languages=languages,
            max_files=max_files,
        )
        return _index_result_to_dict(result, resolution)

    def _list_files(
        self,
        resolution: WorkspaceResolution,
        path: str | None,
        pattern: str | None,
        file_format: str,
        include_metadata: bool,
        limit: int | None,
    ) -> dict[str, Any]:
        """List indexed files without mutating absent index storage."""
        if not resolution.index_db_path.exists():
            return {
                "workspace_key": resolution.workspace_key,
                "index_present": False,
                "format": file_format,
                "files": [],
                "truncated": False,
            }

        repository = CodeGraphRepository(resolution.index_db_path)
        effective_limit = self._bounded_limit(limit, default=100)
        path_prefix = _normalize_path_prefix(path)
        rows = repository.list_files(
            limit=effective_limit + 1,
            path_prefix=path_prefix,
            path_pattern=pattern,
        )

        truncated = len(rows) > effective_limit
        visible_rows = rows[:effective_limit]
        return {
            "workspace_key": resolution.workspace_key,
            "index_present": True,
            "format": file_format,
            "files": [_indexed_file_to_dict(row, include_metadata=include_metadata) for row in visible_rows],
            "truncated": truncated,
        }

    def _search_nodes(
        self,
        resolution: WorkspaceResolution,
        query: str,
        kind: str | None,
        language: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        """Search indexed graph nodes for one workspace."""
        effective_limit = self._bounded_limit(limit)
        if not resolution.index_db_path.exists():
            return {
                "workspace_key": resolution.workspace_key,
                "index_present": False,
                "results": [],
                "truncated": False,
            }
        repository = CodeGraphRepository(resolution.index_db_path)
        rows = repository.search_nodes(query, limit=effective_limit + 1, kind=kind, language=language)
        truncated = len(rows) > effective_limit
        return {
            "workspace_key": resolution.workspace_key,
            "index_present": True,
            "query": query,
            "results": [_node_to_dict(row) for row in rows[:effective_limit]],
            "truncated": truncated,
        }

    def _get_node(
        self,
        resolution: WorkspaceResolution,
        node_id: str | None,
        symbol: str | None,
        include_code: bool,
    ) -> dict[str, Any]:
        """Fetch one indexed graph node by id or symbol selector."""
        if not resolution.index_db_path.exists():
            return {"workspace_key": resolution.workspace_key, "index_present": False, "node": None}
        repository = CodeGraphRepository(resolution.index_db_path)
        node = repository.get_node(node_id) if node_id else repository.find_node_by_symbol(str(symbol))
        node_dict = _node_to_dict(node) if node is not None else None
        if node_dict is not None and include_code:
            node_dict["code_available"] = False
        return {
            "workspace_key": resolution.workspace_key,
            "index_present": True,
            "node": node_dict,
        }

    def _list_relationships(
        self,
        resolution: WorkspaceResolution,
        direction: str,
        node_id: str | None,
        symbol: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        """List callers or callees for one indexed graph node selector."""
        effective_limit = self._bounded_limit(limit)
        if not resolution.index_db_path.exists():
            return {
                "workspace_key": resolution.workspace_key,
                "index_present": False,
                "node": None,
                "relationships": [],
                "truncated": False,
            }
        repository = CodeGraphRepository(resolution.index_db_path)
        node = repository.get_node(node_id) if node_id else repository.find_node_by_symbol(str(symbol))
        if node is None:
            return {
                "workspace_key": resolution.workspace_key,
                "index_present": True,
                "node": None,
                "relationships": [],
                "truncated": False,
            }
        if direction == "callers":
            rows = repository.list_callers(node.id, limit=effective_limit + 1)
        else:
            rows = repository.list_callees(node.id, limit=effective_limit + 1)
        truncated = len(rows) > effective_limit
        return {
            "workspace_key": resolution.workspace_key,
            "index_present": True,
            "node": _node_to_dict(node),
            "relationships": rows[:effective_limit],
            "truncated": truncated,
        }

    def _new_indexer(self) -> CodeGraphIndexer:
        """Create a fresh indexer for a blocking foreground operation."""
        return CodeGraphIndexer(settings=self._settings, registry=self._language_registry)

    @staticmethod
    def _reject_unknown(arguments: dict[str, Any], *, allowed: set[str]) -> None:
        """Reject argument names outside a tool-specific allowlist."""
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")

    def _validate_languages(self, languages: Any) -> None:
        """Validate optional language filters against the registry."""
        if languages is None:
            return
        if not isinstance(languages, list) or not all(isinstance(item, str) for item in languages):
            raise ValueError("languages must be an array of strings")
        unknown = sorted(set(languages) - self._language_registry.known_language_ids())
        if unknown:
            raise ValueError(f"unknown languages: {', '.join(unknown)}")

    @staticmethod
    def _validate_foreground_mode(mode: Any) -> None:
        """Ensure Stage 1 callers only request foreground execution."""
        if mode is not None and mode != _FOREGROUND_MODE:
            raise ValueError("only foreground mode is supported")

    @staticmethod
    def _validate_positive_int(value: Any, field_name: str) -> None:
        """Validate optional positive integer tool arguments."""
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")

    @staticmethod
    def _validate_node_selector(arguments: dict[str, Any]) -> None:
        """Require a non-empty node id or symbol selector."""
        node_id = arguments.get("node_id")
        symbol = arguments.get("symbol")
        if node_id is None and symbol is None:
            raise ValueError("node_id or symbol is required")
        if node_id is not None and (not isinstance(node_id, str) or not node_id.strip()):
            raise ValueError("node_id must be a non-empty string")
        if symbol is not None and (not isinstance(symbol, str) or not symbol.strip()):
            raise ValueError("symbol must be a non-empty string")
        if node_id is not None and symbol is not None:
            raise ValueError("node_id and symbol are mutually exclusive")
        if isinstance(node_id, str):
            arguments["node_id"] = node_id.strip()
        if isinstance(symbol, str):
            arguments["symbol"] = symbol.strip()

    def _bounded_limit(self, limit: int | None, *, default: int = 10) -> int:
        """Clamp user-provided result limits to configured maximums."""
        return min(max(1, int(limit or default)), self._settings.max_search_results)


def _normalize_path_prefix(path: str | None) -> str | None:
    """Normalize an optional workspace-relative path prefix."""
    if path is None or not path.strip() or path.strip() == ".":
        return None
    raw = Path(path)
    if raw.is_absolute() or ".." in raw.parts:
        raise PermissionError("path is outside workspace scope")
    return raw.as_posix().strip("/") or None


def _indexed_file_to_dict(file: IndexedFile, *, include_metadata: bool) -> dict[str, Any]:
    """Serialize an indexed file row for MCP responses."""
    result: dict[str, Any] = {
        "path": file.path,
        "language": file.language,
        "status": file.status,
        "node_count": file.node_count,
    }
    if include_metadata:
        result.update(
            {
                "size": file.size,
                "content_hash": file.content_hash,
                "modified_at": file.modified_at,
                "indexed_at": file.indexed_at,
                "errors": list(file.errors),
            }
        )
    return result


def _node_to_dict(node: CodeGraphNode) -> dict[str, Any]:
    """Serialize a graph node for MCP responses."""
    return {
        "id": node.id,
        "kind": node.kind,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "file_path": node.file_path,
        "language": node.language,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "start_column": node.start_column,
        "end_column": node.end_column,
        "signature": node.signature,
        "docstring": node.docstring,
        "visibility": node.visibility,
        "flags": list(node.flags),
        "metadata": dict(node.metadata),
    }


def _index_run_to_dict(run: IndexRunSummary | None) -> dict[str, Any] | None:
    """Serialize an index run summary or preserve a missing run as None."""
    if run is None:
        return None
    return {
        "run_id": run.run_id,
        "workspace_key": run.workspace_key,
        "mode": run.mode,
        "status": run.status,
        "counters": dict(run.counters),
        "error_summary": list(run.error_summary),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }


def _index_result_to_dict(result: Any, resolution: WorkspaceResolution) -> dict[str, Any]:
    """Serialize an indexer result with workspace identity and database path."""
    return {
        "workspace_key": resolution.workspace_key,
        "index_db_path": str(resolution.index_db_path),
        "status": result.status,
        "counters": dict(result.counters),
        "errors": list(result.errors),
    }
