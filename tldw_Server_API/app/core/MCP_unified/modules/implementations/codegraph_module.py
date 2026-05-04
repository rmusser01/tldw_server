"""
Workspace-bounded native CodeGraph MCP module.

Stage 1 exposes bounded foreground indexing and file inventory only. Symbol
extraction and graph queries are intentionally deferred.
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
from tldw_Server_API.app.core.CodeGraph.models import IndexedFile, IndexRunSummary, WorkspaceResolution
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
        super().__init__(config)
        self._settings = CodeGraphSettings.from_mapping(config.settings)
        self._dependency_health = probe_codegraph_dependencies()
        self._language_registry = CodeGraphLanguageRegistry(self._dependency_health)
        self._workspace = CodeGraphWorkspaceResolver(workspace_root_resolver, self._settings)

    async def on_initialize(self) -> None:
        logger.info(f"Initializing CodeGraph module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down CodeGraph module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        return {
            "initialized": True,
            "workspace_root_resolver": self._workspace is not None,
            "dependencies_available": self._dependency_health.available,
        }

    async def get_tools(self) -> list[dict[str, Any]]:
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

        return [status_tool, index_tool, sync_tool, files_tool]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
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

        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
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

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _status(self, resolution: WorkspaceResolution) -> dict[str, Any]:
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
        repository = CodeGraphRepository(resolution.index_db_path)
        return repository.counts(), repository.last_index_run()

    def _run_index(
        self,
        resolution: WorkspaceResolution,
        force: bool,
        languages: list[str] | None,
        max_files: int | None,
    ) -> dict[str, Any]:
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
        if not resolution.index_db_path.exists():
            return {
                "workspace_key": resolution.workspace_key,
                "index_present": False,
                "format": file_format,
                "files": [],
                "truncated": False,
            }

        repository = CodeGraphRepository(resolution.index_db_path)
        effective_limit = max(1, int(limit or 100))
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

    def _new_indexer(self) -> CodeGraphIndexer:
        return CodeGraphIndexer(settings=self._settings, registry=self._language_registry)

    @staticmethod
    def _reject_unknown(arguments: dict[str, Any], *, allowed: set[str]) -> None:
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")

    def _validate_languages(self, languages: Any) -> None:
        if languages is None:
            return
        if not isinstance(languages, list) or not all(isinstance(item, str) for item in languages):
            raise ValueError("languages must be an array of strings")
        unknown = sorted(set(languages) - self._language_registry.known_language_ids())
        if unknown:
            raise ValueError(f"unknown languages: {', '.join(unknown)}")

    @staticmethod
    def _validate_foreground_mode(mode: Any) -> None:
        if mode is not None and mode != _FOREGROUND_MODE:
            raise ValueError("only foreground mode is supported")

    @staticmethod
    def _validate_positive_int(value: Any, field_name: str) -> None:
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")


def _normalize_path_prefix(path: str | None) -> str | None:
    if path is None or not path.strip() or path.strip() == ".":
        return None
    raw = Path(path)
    if raw.is_absolute() or ".." in raw.parts:
        raise PermissionError("path is outside workspace scope")
    return raw.as_posix().strip("/") or None


def _indexed_file_to_dict(file: IndexedFile, *, include_metadata: bool) -> dict[str, Any]:
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


def _index_run_to_dict(run: IndexRunSummary | None) -> dict[str, Any] | None:
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
    return {
        "workspace_key": resolution.workspace_key,
        "index_db_path": str(resolution.index_db_path),
        "status": result.status,
        "counters": dict(result.counters),
        "errors": list(result.errors),
    }
