"""
Workspace-bounded filesystem MCP module.

Exposes:
- fs.list
- fs.read_text
- fs.write_text
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)

from ..base import BaseModule, ModuleConfig, create_tool_definition


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


class FilesystemModule(BaseModule):
    """Workspace-scoped text filesystem primitives."""

    _DEFAULT_MAX_READ_BYTES = 1_000_000

    def __init__(
        self,
        config: ModuleConfig,
        workspace_root_resolver: McpHubWorkspaceRootResolver | Any | None = None,
    ) -> None:
        super().__init__(config)
        self._workspace_root_resolver = workspace_root_resolver or McpHubWorkspaceRootResolver()

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Filesystem module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Filesystem module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True, "workspace_root_resolver": self._workspace_root_resolver is not None}

    async def get_tools(self) -> list[dict[str, Any]]:
        shared_fs_metadata = {
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
        }
        list_tool = create_tool_definition(
            name="fs.list",
            description="List directory entries under the active trusted workspace root.",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative or absolute path"},
                },
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["filesystem.read"],
                **shared_fs_metadata,
            },
        )
        list_tool["inputSchema"]["additionalProperties"] = False

        read_text_tool = create_tool_definition(
            name="fs.read_text",
            description="Read a UTF-8 text file under the active trusted workspace root.",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative or absolute file path"},
                },
                "required": ["path"],
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["filesystem.read"],
                **shared_fs_metadata,
            },
        )
        read_text_tool["inputSchema"]["additionalProperties"] = False

        write_text_tool = create_tool_definition(
            name="fs.write_text",
            description="Write UTF-8 text content to a file under the active trusted workspace root.",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative or absolute file path"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            metadata={
                "category": "management",
                "capabilities": ["filesystem.write"],
                **shared_fs_metadata,
            },
        )
        write_text_tool["inputSchema"]["additionalProperties"] = False

        return [list_tool, read_text_tool, write_text_tool]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments or {})
        self.validate_tool_arguments(tool_name, args)

        workspace_root = await self._resolve_workspace_root(context)

        if tool_name == "fs.list":
            target = self._resolve_workspace_path(workspace_root, str(args.get("path") or "."))
            return await asyncio.to_thread(
                self._list_directory,
                workspace_root,
                target,
                self._list_entry_limit(),
            )

        if tool_name == "fs.read_text":
            target = self._resolve_workspace_path(workspace_root, str(args.get("path")))
            read_result = await asyncio.to_thread(self._read_text_file, target, self._max_read_bytes())
            return {
                "path": self._to_workspace_relative_path(workspace_root, target),
                "text": read_result["text"],
            }

        if tool_name == "fs.write_text":
            target = self._resolve_workspace_path(workspace_root, str(args.get("path")))
            content = args.get("content")
            write_result = await asyncio.to_thread(self._write_text_file, target, str(content))
            return {
                "path": self._to_workspace_relative_path(workspace_root, target),
                "bytes_written": write_result["bytes_written"],
            }

        raise ValueError(f"Unknown tool: {tool_name}")

    def _list_entry_limit(self) -> int:
        raw_limit = self.config.settings.get("list_entry_limit", 1000)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 1000
        return max(1, limit)

    def _max_read_bytes(self) -> int:
        raw_limit = self.config.settings.get("max_read_bytes", self._DEFAULT_MAX_READ_BYTES)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = self._DEFAULT_MAX_READ_BYTES
        return max(1, limit)

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "fs.list":
            unknown = sorted({key for key in arguments.keys()} - {"path"})
            if unknown:
                raise ValueError(f"unknown arguments: {', '.join(unknown)}")
            path = arguments.get("path")
            if path is not None and not isinstance(path, str):
                raise ValueError("path must be a string")
            return

        if tool_name == "fs.read_text":
            unknown = sorted({key for key in arguments.keys()} - {"path"})
            if unknown:
                raise ValueError(f"unknown arguments: {', '.join(unknown)}")
            path = arguments.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("path is required")
            return

        if tool_name == "fs.write_text":
            unknown = sorted({key for key in arguments.keys()} - {"path", "content"})
            if unknown:
                raise ValueError(f"unknown arguments: {', '.join(unknown)}")
            path = arguments.get("path")
            content = arguments.get("content")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("path is required")
            if not isinstance(content, str):
                raise ValueError("content must be a string")
            return

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _resolve_workspace_root(self, context: Any | None) -> Path:
        metadata = getattr(context, "metadata", None)
        metadata_map = dict(metadata) if isinstance(metadata, dict) else {}
        session_id = _first_nonempty(
            getattr(context, "session_id", None),
            metadata_map.get("session_id"),
        )
        user_id = _first_nonempty(
            getattr(context, "user_id", None),
            metadata_map.get("user_id"),
        )
        workspace_trust_source = _first_nonempty(
            metadata_map.get("workspace_trust_source"),
            metadata_map.get("selected_workspace_trust_source"),
        )
        if session_id and not user_id and workspace_trust_source != "shared_registry":
            raise PermissionError("workspace_root_unavailable")

        resolution = await self._workspace_root_resolver.resolve_for_context(
            session_id=session_id,
            user_id=user_id,
            workspace_id=_first_nonempty(metadata_map.get("workspace_id")),
            workspace_trust_source=workspace_trust_source,
            owner_scope_type=_first_nonempty(
                metadata_map.get("owner_scope_type"),
                metadata_map.get("selected_workspace_scope_type"),
            ),
            owner_scope_id=metadata_map.get("owner_scope_id", metadata_map.get("selected_workspace_scope_id")),
        )
        workspace_root_raw = str(resolution.get("workspace_root") or "").strip()
        if not workspace_root_raw:
            reason = str(resolution.get("reason") or "workspace_root_unavailable")
            raise PermissionError(reason)
        return Path(workspace_root_raw).expanduser().resolve(strict=False)

    @staticmethod
    def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved != workspace_root and workspace_root not in resolved.parents:
            raise PermissionError("path is outside workspace scope")
        return resolved

    @staticmethod
    def _list_directory(workspace_root: Path, target: Path, entry_limit: int) -> dict[str, Any]:
        if not target.exists():
            raise FileNotFoundError(f"path not found: {target}")
        if not target.is_dir():
            raise NotADirectoryError(f"path is not a directory: {target}")

        entries: list[dict[str, Any]] = []
        remaining_count = 0
        with os.scandir(target) as iterator:
            for entry in iterator:
                if len(entries) >= entry_limit:
                    remaining_count += 1
                    continue
                if entry.is_symlink():
                    entry_type = "symlink"
                elif entry.is_dir():
                    entry_type = "directory"
                else:
                    entry_type = "file"
                entry_record = {
                    "name": entry.name,
                    "path": FilesystemModule._to_workspace_relative_path(workspace_root, Path(entry.path)),
                    "type": entry_type,
                }
                if entry_type == "file":
                    with suppress(OSError):
                        entry_record["size"] = entry.stat().st_size
                if entry_type == "symlink":
                    with suppress(OSError):
                        entry_record["size"] = entry.stat(follow_symlinks=False).st_size
                entries.append(entry_record)
        entries.sort(key=lambda item: str(item.get("name") or "").lower())
        return {
            "path": FilesystemModule._to_workspace_relative_path(workspace_root, target),
            "entries": entries,
            "truncated": remaining_count > 0,
            "remaining_count": remaining_count,
        }

    @staticmethod
    def _to_workspace_relative_path(workspace_root: Path, candidate: Path) -> str:
        try:
            relative = candidate.relative_to(workspace_root)
        except ValueError:
            return candidate.name
        rel_text = relative.as_posix()
        return rel_text if rel_text not in {"", "."} else "."

    @staticmethod
    def _read_text_file(target: Path, max_read_bytes: int) -> dict[str, Any]:
        if not target.exists():
            raise FileNotFoundError(f"path not found: {target}")
        if not target.is_file():
            raise ValueError(f"path is not a file: {target}")

        file_size = target.stat().st_size
        if file_size > max_read_bytes:
            raise ValueError(
                f"file exceeds fs.read_text limit ({file_size} bytes > {max_read_bytes} bytes)"
            )

        payload = target.read_bytes()
        if b"\x00" in payload:
            raise ValueError("binary content is not supported by fs.read_text")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("binary content is not supported by fs.read_text") from exc
        return {"text": text}

    @staticmethod
    def _write_text_file(target: Path, content: str) -> dict[str, Any]:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        return {"bytes_written": len(data)}
