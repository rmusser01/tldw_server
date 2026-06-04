"""
Workspace-bounded filesystem MCP module.

Exposes:
- fs.list
- fs.read_text
- fs.write_text
- fs.stat
- fs.glob
- fs.grep
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import stat as stat_module
from contextlib import suppress
from datetime import datetime, timezone
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
        }
        shared_path_metadata = {
            **shared_fs_metadata,
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
                **shared_path_metadata,
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
                **shared_path_metadata,
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
                **shared_path_metadata,
            },
        )
        write_text_tool["inputSchema"]["additionalProperties"] = False

        stat_tool = create_tool_definition(
            name="fs.stat",
            description="Return metadata for one path under the active trusted workspace root.",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative or absolute path"},
                    "follow_symlinks": {
                        "type": "boolean",
                        "default": False,
                        "description": "Follow a symlink after verifying the target remains in workspace scope.",
                    },
                },
                "required": ["path"],
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["filesystem.read"],
                **shared_path_metadata,
            },
        )
        stat_tool["inputSchema"]["additionalProperties"] = False

        glob_tool = create_tool_definition(
            name="fs.glob",
            description="Find workspace paths matching a portable pattern without invoking a shell.",
            parameters={
                "properties": {
                    "pattern": {"type": "string", "description": "Portable pattern using / separators"},
                    "base_path": {"type": "string", "description": "Workspace-relative base path"},
                    "include_hidden": {"type": "boolean", "default": False},
                    "include_files": {"type": "boolean", "default": True},
                    "include_directories": {"type": "boolean", "default": True},
                    "follow_symlinks": {"type": "boolean", "default": False},
                    "case_sensitive": {"type": "boolean", "default": True},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["pattern"],
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["filesystem.read"],
                **shared_fs_metadata,
                "path_argument_hints": ["base_path"],
            },
        )
        glob_tool["inputSchema"]["additionalProperties"] = False

        grep_tool = create_tool_definition(
            name="fs.grep",
            description="Search UTF-8 text files under a workspace path without invoking a shell.",
            parameters={
                "properties": {
                    "pattern": {"type": "string"},
                    "base_path": {"type": "string", "description": "Workspace-relative base path"},
                    "include": {"type": "array", "items": {"type": "string"}},
                    "exclude": {"type": "array", "items": {"type": "string"}},
                    "regex": {"type": "boolean", "default": False},
                    "case_sensitive": {"type": "boolean", "default": True},
                    "include_hidden": {"type": "boolean", "default": False},
                    "follow_symlinks": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "minimum": 1},
                    "max_file_bytes": {"type": "integer", "minimum": 1},
                },
                "required": ["pattern"],
            },
            metadata={
                "category": "retrieval",
                "readOnlyHint": True,
                "capabilities": ["filesystem.read"],
                **shared_fs_metadata,
                "path_argument_hints": ["base_path"],
            },
        )
        grep_tool["inputSchema"]["additionalProperties"] = False

        return [list_tool, read_text_tool, write_text_tool, stat_tool, glob_tool, grep_tool]

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

        if tool_name == "fs.stat":
            target = self._resolve_workspace_path_no_follow(workspace_root, str(args.get("path")))
            return await asyncio.to_thread(
                self._stat_path,
                workspace_root,
                target,
                bool(args.get("follow_symlinks", False)),
            )

        if tool_name == "fs.glob":
            base_path = str(args.get("base_path") or ".")
            base = self._resolve_workspace_path(workspace_root, base_path)
            pattern = self._normalize_portable_pattern(str(args.get("pattern")))
            self._reject_unsafe_pattern(pattern)
            limit = self._bounded_positive_int(args.get("limit"), self._setting_positive_int("glob_result_limit", 500))
            return await asyncio.to_thread(
                self._glob_paths,
                workspace_root,
                base,
                pattern,
                bool(args.get("include_hidden", False)),
                bool(args.get("include_files", True)),
                bool(args.get("include_directories", True)),
                bool(args.get("follow_symlinks", False)),
                bool(args.get("case_sensitive", True)),
                limit,
                self._setting_positive_int("glob_walk_entry_limit", 50_000),
            )

        if tool_name == "fs.grep":
            base_path = str(args.get("base_path") or ".")
            base = self._resolve_workspace_path(workspace_root, base_path)
            pattern = str(args.get("pattern") or "")
            regex = bool(args.get("regex", False))
            case_sensitive = bool(args.get("case_sensitive", True))
            regex_pattern = None
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    regex_pattern = re.compile(pattern, flags)
                except re.error as exc:
                    raise ValueError(f"invalid regex pattern: {exc}") from exc
            include = [
                self._normalize_portable_pattern(str(item))
                for item in (args.get("include") if args.get("include") is not None else ["*", "**/*"])
            ]
            exclude = [
                self._normalize_portable_pattern(str(item))
                for item in (args.get("exclude") if args.get("exclude") is not None else [])
            ]
            for include_pattern in include:
                self._reject_unsafe_pattern(include_pattern)
            for exclude_pattern in exclude:
                self._reject_unsafe_pattern(exclude_pattern)
            limit = self._bounded_positive_int(args.get("limit"), self._setting_positive_int("grep_result_limit", 200))
            max_file_bytes = self._bounded_positive_int(
                args.get("max_file_bytes"),
                self._setting_positive_int("grep_max_file_bytes", self._max_read_bytes()),
            )
            return await asyncio.to_thread(
                self._grep_files,
                workspace_root,
                base,
                pattern,
                regex_pattern,
                regex,
                case_sensitive,
                include,
                exclude,
                bool(args.get("include_hidden", False)),
                bool(args.get("follow_symlinks", False)),
                limit,
                max_file_bytes,
                self._setting_positive_int("grep_walk_entry_limit", 50_000),
            )

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

    def _setting_positive_int(self, name: str, default: int) -> int:
        raw_limit = self.config.settings.get(name, default)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = default
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

        if tool_name == "fs.stat":
            unknown = sorted({key for key in arguments.keys()} - {"path", "follow_symlinks"})
            if unknown:
                raise ValueError(f"unknown arguments: {', '.join(unknown)}")
            path = arguments.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("path is required")
            self._validate_bool_argument(arguments, "follow_symlinks")
            return

        if tool_name == "fs.glob":
            unknown = sorted(
                {key for key in arguments.keys()}
                - {
                    "pattern",
                    "base_path",
                    "include_hidden",
                    "include_files",
                    "include_directories",
                    "follow_symlinks",
                    "case_sensitive",
                    "limit",
                }
            )
            if unknown:
                raise ValueError(f"unknown arguments: {', '.join(unknown)}")
            pattern = arguments.get("pattern")
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("pattern is required")
            base_path = arguments.get("base_path")
            if base_path is not None and not isinstance(base_path, str):
                raise ValueError("base_path must be a string")
            for key in (
                "include_hidden",
                "include_files",
                "include_directories",
                "follow_symlinks",
                "case_sensitive",
            ):
                self._validate_bool_argument(arguments, key)
            self._validate_positive_int_argument(arguments, "limit")
            return

        if tool_name == "fs.grep":
            unknown = sorted(
                {key for key in arguments.keys()}
                - {
                    "pattern",
                    "base_path",
                    "include",
                    "exclude",
                    "regex",
                    "case_sensitive",
                    "include_hidden",
                    "follow_symlinks",
                    "limit",
                    "max_file_bytes",
                }
            )
            if unknown:
                raise ValueError(f"unknown arguments: {', '.join(unknown)}")
            pattern = arguments.get("pattern")
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("pattern is required")
            base_path = arguments.get("base_path")
            if base_path is not None and not isinstance(base_path, str):
                raise ValueError("base_path must be a string")
            include = arguments.get("include")
            if include is not None and (
                not isinstance(include, list) or not all(isinstance(item, str) for item in include)
            ):
                raise ValueError("include must be a list of strings")
            exclude = arguments.get("exclude")
            if exclude is not None and (
                not isinstance(exclude, list) or not all(isinstance(item, str) for item in exclude)
            ):
                raise ValueError("exclude must be a list of strings")
            for key in ("regex", "case_sensitive", "include_hidden", "follow_symlinks"):
                self._validate_bool_argument(arguments, key)
            self._validate_positive_int_argument(arguments, "limit")
            self._validate_positive_int_argument(arguments, "max_file_bytes")
            if arguments.get("regex") is True:
                max_pattern_length = self._setting_positive_int("grep_max_pattern_length", 512)
                if len(pattern) > max_pattern_length:
                    raise ValueError(
                        f"pattern exceeds grep regex length limit ({len(pattern)} > {max_pattern_length})"
                    )
            return

        raise ValueError(f"Unknown tool: {tool_name}")

    @staticmethod
    def _validate_bool_argument(arguments: dict[str, Any], key: str) -> None:
        value = arguments.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")

    @staticmethod
    def _validate_positive_int_argument(arguments: dict[str, Any], key: str) -> None:
        value = arguments.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{key} must be a positive integer")

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        """Sanitize filesystem inputs while allowing portable glob syntax."""

        if _depth > 20:
            raise ValueError("Input too deeply nested")

        if isinstance(input_data, str):
            return "".join(ch for ch in input_data if ch >= " " or ch == "\n")
        if isinstance(input_data, dict):
            return {k: self.sanitize_input(v, _depth + 1) for k, v in input_data.items()}
        if isinstance(input_data, list):
            return [self.sanitize_input(v, _depth + 1) for v in input_data]
        return input_data

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
    def _resolve_workspace_path_no_follow(workspace_root: Path, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        normalized = Path(os.path.abspath(os.fspath(candidate)))
        if normalized != workspace_root and workspace_root not in normalized.parents:
            raise PermissionError("path is outside workspace scope")
        return normalized

    @staticmethod
    def _resolved_path_within_workspace(workspace_root: Path, target: Path) -> Path:
        resolved = target.resolve(strict=False)
        if resolved != workspace_root and workspace_root not in resolved.parents:
            raise PermissionError("path is outside workspace scope")
        return resolved

    @staticmethod
    def _bounded_positive_int(value: Any, default: int) -> int:
        if value is None:
            return max(1, int(default))
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("limit must be a positive integer")
        return value

    @staticmethod
    def _normalize_portable_pattern(pattern: str) -> str:
        normalized = pattern.strip().replace("\\", "/")
        while "//" in normalized and not normalized.startswith("//"):
            normalized = normalized.replace("//", "/")
        return normalized

    @staticmethod
    def _reject_unsafe_pattern(pattern: str) -> None:
        if not pattern:
            raise ValueError("unsafe pattern: pattern is required")
        if pattern.startswith("/") or pattern.startswith("//"):
            raise ValueError("unsafe pattern: absolute patterns are not allowed")
        if len(pattern) >= 2 and pattern[1] == ":" and pattern[0].isalpha():
            raise ValueError("unsafe pattern: drive-qualified patterns are not allowed")
        if any(part == ".." for part in pattern.split("/")):
            raise ValueError("unsafe pattern: parent traversal is not allowed")

    @staticmethod
    def _portable_pattern_matches(path: str, pattern: str, *, case_sensitive: bool) -> bool:
        candidate = path if case_sensitive else path.lower()
        patterns = FilesystemModule._expand_double_star_zero_dir_patterns(
            pattern if case_sensitive else pattern.lower()
        )
        return any(fnmatch.fnmatchcase(candidate, candidate_pattern) for candidate_pattern in patterns)

    @staticmethod
    def _expand_double_star_zero_dir_patterns(pattern: str) -> set[str]:
        patterns = {pattern}
        queue = [pattern]
        while queue:
            current = queue.pop()
            start = current.find("**/")
            while start != -1:
                collapsed = f"{current[:start]}{current[start + 3:]}"
                if collapsed not in patterns:
                    patterns.add(collapsed)
                    queue.append(collapsed)
                start = current.find("**/", start + 1)
        return patterns

    @staticmethod
    def _is_hidden_relative_path(path: str) -> bool:
        return any(part.startswith(".") and part not in {"."} for part in path.split("/"))

    @staticmethod
    def _stat_path(workspace_root: Path, target: Path, follow_symlinks: bool) -> dict[str, Any]:
        if not target.exists() and not target.is_symlink():
            raise FileNotFoundError(f"path not found: {target}")

        if follow_symlinks:
            stat_target = FilesystemModule._resolved_path_within_workspace(workspace_root, target)
            stat_result = stat_target.stat()
            is_symlink = target.is_symlink()
            target_within_workspace = True
        else:
            stat_result = target.lstat()
            is_symlink = stat_module.S_ISLNK(stat_result.st_mode)
            target_within_workspace = None

        mode = stat_result.st_mode
        if is_symlink and not follow_symlinks:
            entry_type = "symlink"
        elif stat_module.S_ISDIR(mode):
            entry_type = "directory"
        elif stat_module.S_ISREG(mode):
            entry_type = "file"
        else:
            entry_type = "other"

        record: dict[str, Any] = {
            "path": FilesystemModule._to_workspace_relative_path(workspace_root, target),
            "name": target.name or ".",
            "type": entry_type,
            "size": stat_result.st_size,
            "modified_at": datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat(),
            "mode": stat_module.S_IMODE(mode),
            "is_symlink": is_symlink,
        }
        if target_within_workspace is not None:
            record["target_within_workspace"] = target_within_workspace
        return record

    @staticmethod
    def _glob_paths(
        workspace_root: Path,
        base: Path,
        pattern: str,
        include_hidden: bool,
        include_files: bool,
        include_directories: bool,
        follow_symlinks: bool,
        case_sensitive: bool,
        limit: int,
        walk_entry_limit: int,
    ) -> dict[str, Any]:
        if not base.exists():
            raise FileNotFoundError(f"path not found: {base}")
        if not base.is_dir():
            raise NotADirectoryError(f"path is not a directory: {base}")

        matches: list[dict[str, Any]] = []
        visited_entries = 0
        walk_truncated = False
        seen_dirs: set[Path] = {base.resolve(strict=False)}

        for root, dirnames, filenames in os.walk(base, topdown=True, followlinks=follow_symlinks):
            current_root = Path(root)
            dirnames.sort()
            filenames.sort()

            for dirname in list(dirnames):
                dir_path = current_root / dirname
                rel_path = FilesystemModule._to_workspace_relative_path(workspace_root, dir_path)
                if not include_hidden and FilesystemModule._is_hidden_relative_path(rel_path):
                    dirnames.remove(dirname)
                    continue
                if dir_path.is_symlink():
                    resolved_dir = dir_path.resolve(strict=False)
                    if follow_symlinks:
                        if resolved_dir != workspace_root and workspace_root not in resolved_dir.parents:
                            raise PermissionError("path is outside workspace scope")
                        if resolved_dir in seen_dirs:
                            dirnames.remove(dirname)
                            continue
                        seen_dirs.add(resolved_dir)
                    else:
                        dirnames.remove(dirname)

            candidates: list[tuple[Path, str]] = [(current_root / dirname, "directory") for dirname in dirnames]
            candidates.extend((current_root / filename, "file") for filename in filenames)

            for candidate, candidate_kind in candidates:
                visited_entries += 1
                if visited_entries > walk_entry_limit:
                    walk_truncated = True
                    dirnames.clear()
                    break

                rel_path = FilesystemModule._to_workspace_relative_path(workspace_root, candidate)
                if not include_hidden and FilesystemModule._is_hidden_relative_path(rel_path):
                    continue
                is_symlink = candidate.is_symlink()
                if is_symlink:
                    candidate_type = "symlink"
                else:
                    candidate_type = candidate_kind

                include_candidate = (
                    (candidate_kind == "file" and include_files)
                    or (candidate_kind == "directory" and include_directories)
                )
                if not include_candidate:
                    continue
                if not FilesystemModule._portable_pattern_matches(rel_path, pattern, case_sensitive=case_sensitive):
                    continue

                record: dict[str, Any] = {
                    "path": rel_path,
                    "type": candidate_type,
                }
                if candidate_kind == "file":
                    with suppress(OSError):
                        record["size"] = candidate.stat(follow_symlinks=False).st_size
                matches.append(record)

            if walk_truncated:
                break

        matches.sort(key=lambda item: str(item.get("path") or ""))
        limited_matches = matches[:limit]
        remaining_count = max(0, len(matches) - len(limited_matches))
        return {
            "base_path": FilesystemModule._to_workspace_relative_path(workspace_root, base),
            "pattern": pattern,
            "matches": limited_matches,
            "truncated": walk_truncated or remaining_count > 0,
            "remaining_count": remaining_count,
        }

    @staticmethod
    def _grep_files(
        workspace_root: Path,
        base: Path,
        pattern: str,
        regex_pattern: re.Pattern[str] | None,
        regex: bool,
        case_sensitive: bool,
        include: list[str],
        exclude: list[str],
        include_hidden: bool,
        follow_symlinks: bool,
        limit: int,
        max_file_bytes: int,
        walk_entry_limit: int,
    ) -> dict[str, Any]:
        if not base.exists():
            raise FileNotFoundError(f"path not found: {base}")
        if not base.is_dir():
            raise NotADirectoryError(f"path is not a directory: {base}")

        matches: list[dict[str, Any]] = []
        remaining_count = 0
        skipped = {
            "binary": 0,
            "decode_error": 0,
            "too_large": 0,
            "permission_error": 0,
            "unsupported_type": 0,
        }
        visited_entries = 0
        walk_truncated = False
        seen_dirs: set[Path] = {base.resolve(strict=False)}
        file_candidates: list[tuple[str, Path]] = []

        for root, dirnames, filenames in os.walk(base, topdown=True, followlinks=follow_symlinks):
            current_root = Path(root)
            dirnames.sort()
            filenames.sort()

            for dirname in list(dirnames):
                dir_path = current_root / dirname
                rel_path = FilesystemModule._to_workspace_relative_path(workspace_root, dir_path)
                if not include_hidden and FilesystemModule._is_hidden_relative_path(rel_path):
                    dirnames.remove(dirname)
                    continue
                if dir_path.is_symlink():
                    resolved_dir = dir_path.resolve(strict=False)
                    if follow_symlinks:
                        if resolved_dir != workspace_root and workspace_root not in resolved_dir.parents:
                            raise PermissionError("path is outside workspace scope")
                        if resolved_dir in seen_dirs:
                            dirnames.remove(dirname)
                            continue
                        seen_dirs.add(resolved_dir)
                    else:
                        dirnames.remove(dirname)

            for filename in filenames:
                visited_entries += 1
                if visited_entries > walk_entry_limit:
                    walk_truncated = True
                    dirnames.clear()
                    break

                candidate = current_root / filename
                rel_path = FilesystemModule._to_workspace_relative_path(workspace_root, candidate)
                if not include_hidden and FilesystemModule._is_hidden_relative_path(rel_path):
                    continue
                if candidate.is_symlink():
                    if not follow_symlinks:
                        skipped["unsupported_type"] += 1
                        continue
                    resolved_file = candidate.resolve(strict=False)
                    if resolved_file != workspace_root and workspace_root not in resolved_file.parents:
                        raise PermissionError("path is outside workspace scope")
                    read_target = resolved_file
                else:
                    read_target = candidate
                if not any(
                    FilesystemModule._portable_pattern_matches(rel_path, include_pattern, case_sensitive=True)
                    for include_pattern in include
                ):
                    continue
                if any(
                    FilesystemModule._portable_pattern_matches(rel_path, exclude_pattern, case_sensitive=True)
                    for exclude_pattern in exclude
                ):
                    continue
                if not read_target.is_file():
                    skipped["unsupported_type"] += 1
                    continue
                file_candidates.append((rel_path, read_target))

            if walk_truncated:
                break

        for rel_path, read_target in sorted(file_candidates, key=lambda item: item[0]):
            try:
                file_size = read_target.stat().st_size
            except PermissionError:
                skipped["permission_error"] += 1
                continue
            except OSError:
                skipped["unsupported_type"] += 1
                continue
            if file_size > max_file_bytes:
                skipped["too_large"] += 1
                continue
            try:
                payload = read_target.read_bytes()
            except PermissionError:
                skipped["permission_error"] += 1
                continue
            except OSError:
                skipped["unsupported_type"] += 1
                continue
            if b"\x00" in payload:
                skipped["binary"] += 1
                continue
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                skipped["decode_error"] += 1
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                match_text = FilesystemModule._line_match_text(
                    line,
                    pattern,
                    regex_pattern,
                    regex=regex,
                    case_sensitive=case_sensitive,
                )
                if match_text is None:
                    continue
                match_record = {
                    "path": rel_path,
                    "line_number": line_number,
                    "line": line,
                    "match_text": match_text,
                }
                if len(matches) < limit:
                    matches.append(match_record)
                else:
                    remaining_count += 1

        matches.sort(key=lambda item: (str(item.get("path") or ""), int(item.get("line_number") or 0)))
        return {
            "base_path": FilesystemModule._to_workspace_relative_path(workspace_root, base),
            "matches": matches,
            "truncated": walk_truncated or remaining_count > 0,
            "remaining_count": remaining_count,
            "skipped": skipped,
        }

    @staticmethod
    def _line_match_text(
        line: str,
        pattern: str,
        regex_pattern: re.Pattern[str] | None,
        *,
        regex: bool,
        case_sensitive: bool,
    ) -> str | None:
        if regex:
            if regex_pattern is None:
                return None
            match = regex_pattern.search(line)
            return match.group(0) if match else None

        haystack = line if case_sensitive else line.lower()
        needle = pattern if case_sensitive else pattern.lower()
        index = haystack.find(needle)
        if index < 0:
            return None
        return line[index : index + len(pattern)]

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
