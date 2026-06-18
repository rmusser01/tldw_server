"""MCP module exposing the phase-1 virtual CLI `run` tool."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from loguru import logger

from ...command_runtime.adapters import (
    AdapterContext,
    PhaseOneCommandAdapters,
    PreflightCommandError,
    run_help_text,
    visible_command_registry,
)
from ...command_runtime.executor import CommandBackend, CommandRuntimeExecutor
from ...command_runtime.models import CommandExecutionResult, CommandStepResult
from ...command_runtime.parser import parse_command
from ...command_runtime.presentation import present_command_execution_result
from ...command_runtime.registry import CommandDescriptor, CommandRegistry, build_default_registry
from ..base import BaseModule, ModuleConfig, create_tool_definition

RUN_PARENT_IDEMPOTENCY_KEY_METADATA_KEY = "run_parent_idempotency_key"

_RUN_WRITE_BACKEND_TOOLS = {"fs.write", "fs.write_text", "sandbox.run"}
_RUN_TOOL_NAME = "run"
_RUN_TOOL_ALIASES = ("bash", "shell", "powershell", "pwsh")
_RUN_TOOL_NAMES = frozenset((_RUN_TOOL_NAME, *_RUN_TOOL_ALIASES))
_RUN_ENV_FILE_MAX_BYTES = 64 * 1024


class RunEnvFileValidationError(ValueError):
    """Raised when run env-file input fails validation or safe loading."""


class _AdapterBackend(CommandBackend):
    def __init__(self, adapters: PhaseOneCommandAdapters) -> None:
        self._adapters = adapters

    async def execute(
        self,
        argv: list[str],
        stdin: Any,
        handler_context: Any | None = None,
    ) -> CommandStepResult:
        return await self._adapters.execute(argv, stdin, handler_context)


class RunCommandModule(BaseModule):
    """Expose a policy-aware virtual CLI through MCP tool `run`."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self._registry: CommandRegistry = build_default_registry()

    async def on_initialize(self) -> None:
        logger.info(f"Initializing run command module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down run command module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        run_tool = self._create_run_tool_definition(
            name=_RUN_TOOL_NAME,
            description="Execute a governed command in the MCP virtual CLI runtime.",
            metadata={
                "category": "utility",
                "notes": "Wrapper tool; nested prepared MCP calls carry path/process metadata",
            },
        )
        alias_tools = [
            self._create_run_tool_definition(
                name=alias_name,
                description=(
                    f"Governed shell-like facade for `run`; {alias_name} is not a raw host shell "
                    "and only executes profile-granted virtual CLI commands."
                ),
                metadata={
                    "category": "utility",
                    "canonical_tool": _RUN_TOOL_NAME,
                    "notes": "Compatibility alias; no raw host shell execution is performed.",
                },
            )
            for alias_name in _RUN_TOOL_ALIASES
        ]
        return [run_tool, *alias_tools]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        if tool_name not in _RUN_TOOL_NAMES:
            raise ValueError(f"Unknown tool: {tool_name}")
        args = self.sanitize_input(arguments or {})
        self.validate_tool_arguments(tool_name, args)

        command_text = str(args.get("command") or "").strip()
        timeout_seconds = self._timeout_seconds(args)
        cwd = self._cwd(args)
        retain_output_artifacts = self._retain_output_artifacts(args)
        sandbox_session_id = self._sandbox_session_id(args)
        env_file = self._env_file(args)
        visible = await self._visible_commands_for_context(context)
        if command_text in {"help", "--help"}:
            return present_command_execution_result(
                CommandExecutionResult(
                    stdout=run_help_text(visible),
                    stderr="",
                    exit_code=0,
                    duration_ms=0.0,
                )
            )

        start = time.perf_counter()
        unsupported_feature = self._unsupported_shell_feature_message(command_text)
        if unsupported_feature is not None:
            return present_command_execution_result(
                CommandExecutionResult(
                    stdout="",
                    stderr=unsupported_feature,
                    exit_code=2,
                    duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
                )
            )
        try:
            chain = parse_command(command_text)
        except ValueError as exc:
            return present_command_execution_result(
                CommandExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=2,
                    duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
                )
            )

        try:
            sandbox_env, env_file_scope = await self._sandbox_env_for_chain(
                env_file=env_file,
                cwd=cwd,
                chain=chain,
                visible=visible,
                context=context,
            )
        except RunEnvFileValidationError as exc:
            return present_command_execution_result(
                CommandExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=2,
                    duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
                )
            )

        protocol = await self._resolve_protocol()
        spill_parent_dir = await self._resolve_spill_dir(context)
        invocation_spill_dir = await self._create_invocation_spill_dir(spill_parent_dir, context)
        spill_threshold_bytes = self._setting_int("spill_threshold_bytes", default=65_536)
        preview_line_limit = self._setting_int("preview_line_limit", default=200)
        preview_byte_limit = self._setting_int("preview_byte_limit", default=51_200)
        adapter_context = AdapterContext(
            protocol=protocol,
            request_context=context,
            visible_commands=visible,
            parent_idempotency_key=self._scoped_parent_idempotency_key(
                self._parent_idempotency_key(context, arguments),
                cwd,
                sandbox_session_id,
                env_file_scope,
            ),
            cwd=cwd,
            sandbox_session_id=sandbox_session_id,
            sandbox_env=sandbox_env,
        )
        adapters = PhaseOneCommandAdapters(adapter_context)
        executor = CommandRuntimeExecutor(
            backend=_AdapterBackend(adapters),
            spill_dir=invocation_spill_dir,
            spill_threshold_bytes=spill_threshold_bytes,
            preview_bytes=preview_byte_limit,
        )

        async def _execute_chain() -> CommandExecutionResult:
            """Preflight and execute the parsed command chain."""

            if adapters.requires_whole_chain_preflight(chain):
                await adapters.preflight_chain(chain)
            return await executor.execute(chain)

        try:
            try:
                if timeout_seconds is None:
                    result = await _execute_chain()
                else:
                    result = await asyncio.wait_for(_execute_chain(), timeout=timeout_seconds)
            except PreflightCommandError as exc:
                result = CommandExecutionResult(
                    stdout="",
                    stderr=exc.result.stderr,
                    exit_code=exc.result.exit_code,
                    duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
                )
            except TimeoutError:
                timeout_text = self._format_seconds(timeout_seconds if timeout_seconds is not None else 0.0)
                result = CommandExecutionResult(
                    stdout="",
                    stderr=f"Command timed out after {timeout_text}s",
                    exit_code=124,
                    duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
                )
            except (OSError, ValueError) as exc:
                if self._is_passthrough_runtime_exception(exc):
                    raise
                result = CommandExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=2 if isinstance(exc, ValueError) else 1,
                    duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
                )
            return present_command_execution_result(
                result,
                spill_dir=invocation_spill_dir,
                byte_limit=preview_byte_limit,
                line_limit=preview_line_limit,
                include_artifact_handles=retain_output_artifacts,
            )
        finally:
            if not retain_output_artifacts:
                await self._cleanup_invocation_spill_dir(invocation_spill_dir)

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name not in _RUN_TOOL_NAMES:
            raise ValueError(f"Unknown tool: {tool_name}")
        unknown = sorted(
            set(arguments)
            - {
                "command",
                "cwd",
                "envFile",
                "env_file",
                "idempotencyKey",
                "idempotency_key",
                "retainOutputArtifacts",
                "retain_output_artifacts",
                "sandboxSessionId",
                "sandbox_session_id",
                "timeoutSeconds",
                "timeout_seconds",
                "workingDirectory",
            }
        )
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command is required")
        idempotency_key = arguments.get("idempotencyKey")
        legacy_idempotency_key = arguments.get("idempotency_key")
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            raise ValueError("idempotencyKey must be a string")
        if legacy_idempotency_key is not None and not isinstance(legacy_idempotency_key, str):
            raise ValueError("idempotency_key must be a string")
        if (
            isinstance(idempotency_key, str)
            and isinstance(legacy_idempotency_key, str)
            and idempotency_key.strip()
            and legacy_idempotency_key.strip()
            and idempotency_key.strip() != legacy_idempotency_key.strip()
        ):
            raise ValueError("idempotencyKey and idempotency_key must match when both are provided")
        self._validate_timeout_arguments(arguments)
        self._validate_cwd_arguments(arguments)
        self._validate_retain_output_artifact_arguments(arguments)
        self._validate_sandbox_session_arguments(arguments)
        self._validate_env_file_arguments(arguments)

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        """Sanitize input while allowing CLI flags like `--help` and shell-like tokens."""

        if _depth > 20:
            raise ValueError("Input too deeply nested")

        def _clean_string(value: str) -> str:
            cleaned = []
            for ch in value:
                if ch == "\n" or ch == "\t" or ch >= " ":
                    cleaned.append(ch)
            return "".join(cleaned)

        if isinstance(input_data, str):
            return _clean_string(input_data)
        if isinstance(input_data, dict):
            return {k: self.sanitize_input(v, _depth + 1) for k, v in input_data.items()}
        if isinstance(input_data, list):
            return [self.sanitize_input(v, _depth + 1) for v in input_data]
        return input_data

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        if tool_name not in _RUN_TOOL_NAMES:
            return super().is_write_tool_call(tool_name, arguments, tool_def=tool_def)

        command = str(arguments.get("command") or "").strip()
        if not command or command in {"help", "--help"}:
            return False
        try:
            chain = parse_command(command)
        except ValueError:
            return False

        for segment in chain.segments:
            for invocation in segment.commands:
                if not invocation.argv:
                    continue
                try:
                    descriptor = self._registry.get_command(invocation.argv[0])
                except KeyError:
                    continue
                if any(tool in _RUN_WRITE_BACKEND_TOOLS for tool in descriptor.backend_tools):
                    return True
        return False

    @staticmethod
    def _create_run_tool_definition(
        *,
        name: str,
        description: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a run-compatible tool definition with the shared command schema."""

        tool = create_tool_definition(
            name=name,
            description=description,
            parameters={
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command chain to execute (example: ls | grep py).",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "description": "Legacy alias for idempotencyKey.",
                    },
                    "idempotencyKey": {
                        "type": "string",
                        "description": "Optional parent idempotency key for nested governed steps.",
                    },
                    "env_file": {
                        "type": "string",
                        "description": "Legacy alias for envFile.",
                    },
                    "envFile": {
                        "type": "string",
                        "description": (
                            "Workspace-relative .env-style file to pass only to governed sandbox steps."
                        ),
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Workspace-relative current directory for relative command paths.",
                    },
                    "workingDirectory": {
                        "type": "string",
                        "description": "Alias for cwd.",
                    },
                    "retain_output_artifacts": {
                        "type": "boolean",
                        "description": "Legacy alias for retainOutputArtifacts.",
                    },
                    "retainOutputArtifacts": {
                        "type": "boolean",
                        "description": "Retain oversized output spill files and include redacted artifact handles.",
                    },
                    "sandbox_session_id": {
                        "type": "string",
                        "description": "Legacy alias for sandboxSessionId.",
                    },
                    "sandboxSessionId": {
                        "type": "string",
                        "description": "Sandbox session id for governed sandbox command steps.",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Legacy alias for timeoutSeconds.",
                    },
                    "timeoutSeconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Optional wall-clock timeout for the governed command chain.",
                    },
                },
                "required": ["command"],
            },
            metadata=metadata,
        )
        tool["inputSchema"]["additionalProperties"] = False
        return tool

    async def _resolve_protocol(self) -> Any:
        configured = self.config.settings.get("protocol")
        if configured is not None:
            return configured
        from ...protocol import MCPProtocol
        from ...server import get_mcp_server

        protocol = get_mcp_server().protocol
        if protocol is None:
            return MCPProtocol()
        return protocol

    async def _visible_commands_for_context(self, context: Any | None) -> Mapping[str, CommandDescriptor]:
        protocol = await self._resolve_protocol()
        context_for_listing = context
        if context_for_listing is None:
            from ...protocol import RequestContext

            context_for_listing = RequestContext(request_id="run-module")
        tools_payload = await protocol._handle_tools_list({}, context_for_listing)
        tools_payload = await self._filter_tools_payload_for_context(
            protocol,
            tools_payload,
            context_for_listing,
        )
        return visible_command_registry(tools_payload=tools_payload, registry=self._registry)

    async def _filter_tools_payload_for_context(
        self,
        protocol: Any,
        tools_payload: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        tools = tools_payload.get("tools")
        if not isinstance(tools, list):
            return tools_payload

        resolve_policy = getattr(protocol, "_resolve_effective_tool_policy", None)
        if callable(resolve_policy):
            effective_policy = await resolve_policy(context)
        else:
            effective_policy = None
        allowed_patterns = self._allowed_patterns_for_context(protocol, context)
        denied_patterns = self._policy_patterns(effective_policy, "denied_tools")
        policy_allowed_patterns = self._policy_patterns(effective_policy, "allowed_tools")
        resolution_error = str((effective_policy or {}).get("resolution_error") or "").strip()

        filtered_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            if allowed_patterns and not self._tool_name_matches_patterns(name, allowed_patterns):
                continue
            if resolution_error:
                continue
            if self._tool_name_matches_exact_patterns(name, denied_patterns):
                continue
            if policy_allowed_patterns and not self._tool_name_matches_patterns(name, policy_allowed_patterns):
                continue
            filtered_tools.append(tool)
        return {"tools": filtered_tools}

    def _setting_int(self, key: str, *, default: int) -> int:
        raw_value = self.config.settings.get(key, default)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return default

    async def _resolve_spill_dir(self, context: Any | None) -> Path | None:
        """Resolve the configured spill parent directory for the current workspace context."""

        raw_value = self.config.settings.get("spill_dir")
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return candidate
        workspace_root = await self._resolve_workspace_root(context)
        if workspace_root is not None:
            return workspace_root / candidate
        return Path.cwd() / candidate

    async def _create_invocation_spill_dir(self, spill_parent_dir: Path | None, context: Any | None) -> Path:
        """Create a private spill directory for one run invocation."""

        return await asyncio.to_thread(self._make_invocation_spill_dir, spill_parent_dir, context)

    @staticmethod
    def _make_invocation_spill_dir(spill_parent_dir: Path | None, context: Any | None) -> Path:
        """Create the concrete invocation spill directory under a validated parent."""

        prefix = RunCommandModule._safe_spill_prefix(context)
        if spill_parent_dir is None:
            return Path(tempfile.mkdtemp(prefix=prefix))

        RunCommandModule._ensure_spill_parent_dir(spill_parent_dir)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(spill_parent_dir)))

    @staticmethod
    def _ensure_spill_parent_dir(spill_parent_dir: Path) -> None:
        """Create and validate the configured spill parent directory."""

        try:
            spill_parent_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        except FileExistsError:
            pass
        if spill_parent_dir.is_symlink():
            raise PermissionError(f"Refusing to use symlink spill directory: {spill_parent_dir}")
        if not spill_parent_dir.is_dir():
            raise PermissionError(f"Refusing to use non-directory spill path: {spill_parent_dir}")

        try:
            info = spill_parent_dir.stat(follow_symlinks=False)
        except OSError as exc:
            raise PermissionError(f"Unable to inspect spill directory: {spill_parent_dir}") from exc

        getuid = getattr(os, "getuid", None)
        if callable(getuid) and info.st_uid != getuid():
            raise PermissionError(f"Refusing to use spill directory owned by another user: {spill_parent_dir}")
        if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError(f"Refusing to use spill directory with non-private permissions: {spill_parent_dir}")

    @staticmethod
    def _safe_spill_prefix(context: Any | None) -> str:
        """Build a filesystem-safe temp prefix from the request id."""

        raw_request_id = str(getattr(context, "request_id", "") or "run")
        safe_request_id = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "-"
            for char in raw_request_id
        ).strip("-._")
        if not safe_request_id:
            safe_request_id = "run"
        return f"mcp-run-{safe_request_id[:40]}-"

    async def _resolve_workspace_root(self, context: Any | None) -> Path | None:
        """Resolve the workspace root used for relative spill parent settings."""

        if context is None:
            return None
        resolver = self.config.settings.get("workspace_root_resolver")
        if resolver is None:
            from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
                McpHubWorkspaceRootResolver,
            )

            resolver = McpHubWorkspaceRootResolver()
            self.config.settings["workspace_root_resolver"] = resolver

        metadata = getattr(context, "metadata", None)
        metadata_map = dict(metadata) if isinstance(metadata, dict) else {}
        try:
            resolution = await resolver.resolve_for_context(
                session_id=self._first_nonempty(getattr(context, "session_id", None), metadata_map.get("session_id")),
                user_id=self._first_nonempty(getattr(context, "user_id", None), metadata_map.get("user_id")),
                workspace_id=self._first_nonempty(metadata_map.get("workspace_id")),
                workspace_trust_source=self._first_nonempty(
                    metadata_map.get("workspace_trust_source"),
                    metadata_map.get("selected_workspace_trust_source"),
                ),
                owner_scope_type=self._first_nonempty(
                    metadata_map.get("owner_scope_type"),
                    metadata_map.get("selected_workspace_scope_type"),
                ),
                owner_scope_id=metadata_map.get("owner_scope_id", metadata_map.get("selected_workspace_scope_id")),
            )
        except Exception as exc:
            logger.debug("Failed to resolve workspace root for run spill dir: {}", exc)
            return None

        workspace_root_raw = str(resolution.get("workspace_root") or "").strip()
        if not workspace_root_raw:
            return None
        return Path(workspace_root_raw).expanduser().resolve(strict=False)

    async def _sandbox_env_for_chain(
        self,
        *,
        env_file: str | None,
        cwd: str | None,
        chain: Any,
        visible: Mapping[str, CommandDescriptor],
        context: Any | None,
    ) -> tuple[dict[str, str] | None, str | None]:
        """Load env-file values for sandbox-backed command chains only."""

        if env_file is None:
            return None, None
        if not self._chain_includes_visible_sandbox(chain, visible):
            raise RunEnvFileValidationError("envFile is only supported for command chains that include sandbox")
        relative_path = self._env_file_relative_path(env_file, cwd)
        env_file_path = await self._resolve_env_file_path(relative_path, context)
        payload = await asyncio.to_thread(self._read_env_file_bytes, env_file_path)
        env = self._parse_env_file_bytes(payload)
        env_file_digest = hashlib.sha256(payload).hexdigest()
        scope = json.dumps(
            {"path": relative_path, "sha256": env_file_digest},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return env, scope

    @staticmethod
    def _chain_includes_visible_sandbox(
        chain: Any,
        visible: Mapping[str, CommandDescriptor],
    ) -> bool:
        """Return True when the parsed chain contains a sandbox step visible in policy."""

        if "sandbox" not in visible:
            return False
        for segment in getattr(chain, "segments", []) or []:
            for invocation in getattr(segment, "commands", []) or []:
                argv = list(getattr(invocation, "argv", []) or [])
                if argv and argv[0] == "sandbox":
                    return True
        return False

    @staticmethod
    def _env_file_relative_path(env_file: str, cwd: str | None) -> str:
        """Apply cwd to a normalized env-file path."""

        if not cwd:
            return env_file
        return f"{cwd}/{env_file}"

    async def _resolve_env_file_path(self, relative_path: str, context: Any | None) -> Path:
        """Resolve an env file under the active workspace root with symlink containment."""

        workspace_root = await self._resolve_workspace_root(context)
        if workspace_root is None:
            raise RunEnvFileValidationError("envFile requires a resolved workspace root")
        workspace_root = workspace_root.resolve(strict=False)
        lexical_path = workspace_root / relative_path
        if not self._path_is_relative_to(lexical_path, workspace_root):
            raise RunEnvFileValidationError("envFile must stay within the workspace root")
        try:
            resolved_path = await asyncio.to_thread(lexical_path.resolve, strict=True)
        except FileNotFoundError as exc:
            raise RunEnvFileValidationError("envFile was not found") from exc
        except (OSError, RuntimeError) as exc:
            raise RunEnvFileValidationError("envFile could not be resolved") from exc
        if not self._path_is_relative_to(resolved_path, workspace_root):
            raise RunEnvFileValidationError("envFile symlink target must stay within the workspace root")
        if not await asyncio.to_thread(resolved_path.is_file):
            raise RunEnvFileValidationError("envFile must reference a regular file")
        return resolved_path

    @staticmethod
    def _path_is_relative_to(path: Path, parent: Path) -> bool:
        """Compatibility wrapper for Path.relative_to containment checks."""

        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def _read_env_file_bytes(path: Path) -> bytes:
        """Read a bounded env file payload."""

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd: int | None = None
        try:
            fd = os.open(path, flags)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise RunEnvFileValidationError("envFile must reference a regular file")
            if info.st_size > _RUN_ENV_FILE_MAX_BYTES:
                raise RunEnvFileValidationError(f"envFile exceeds {_RUN_ENV_FILE_MAX_BYTES} bytes")
            handle = os.fdopen(fd, "rb", closefd=True)
            fd = None
            with handle:
                payload = handle.read(_RUN_ENV_FILE_MAX_BYTES + 1)
        except OSError as exc:
            raise RunEnvFileValidationError("envFile could not be read") from exc
        finally:
            if fd is not None:
                os.close(fd)
        if len(payload) > _RUN_ENV_FILE_MAX_BYTES:
            raise RunEnvFileValidationError(f"envFile exceeds {_RUN_ENV_FILE_MAX_BYTES} bytes")
        return payload

    @staticmethod
    def _parse_env_file_bytes(payload: bytes) -> dict[str, str]:
        """Parse a minimal .env file without expansion or host environment access."""

        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RunEnvFileValidationError("envFile must be UTF-8") from exc

        env: dict[str, str] = {}
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                raise RunEnvFileValidationError(f"envFile line {line_number} must be KEY=value")
            key, value = line.split("=", 1)
            key = key.strip()
            if not RunCommandModule._is_valid_env_key(key):
                raise RunEnvFileValidationError(f"envFile line {line_number} has invalid variable name")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            env[key] = value
        return env

    @staticmethod
    def _is_valid_env_key(key: str) -> bool:
        """Return whether a key is a portable environment variable name."""

        if not key or not (key[0] == "_" or RunCommandModule._is_ascii_alpha(key[0])):
            return False
        return all(ch == "_" or RunCommandModule._is_ascii_alnum(ch) for ch in key[1:])

    @staticmethod
    def _is_ascii_alpha(char: str) -> bool:
        """Return whether one character is in the ASCII alpha range."""

        return ("A" <= char <= "Z") or ("a" <= char <= "z")

    @staticmethod
    def _is_ascii_alnum(char: str) -> bool:
        """Return whether one character is in the ASCII alphanumeric range."""

        return RunCommandModule._is_ascii_alpha(char) or ("0" <= char <= "9")

    @staticmethod
    def _first_nonempty(*values: Any) -> str | None:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return None

    def _allowed_patterns_for_context(self, protocol: Any, context: Any) -> list[str]:
        extract_allowed = getattr(protocol, "_extract_allowed_tools", None)
        if callable(extract_allowed):
            try:
                extracted = extract_allowed(context)
            except Exception:
                extracted = None
            if isinstance(extracted, list):
                return [str(pattern).strip() for pattern in extracted if str(pattern).strip()]

        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return []
        allowed = metadata.get("allowed_tools")
        if isinstance(allowed, list):
            return [str(pattern).strip() for pattern in allowed if str(pattern).strip()]
        return []

    @staticmethod
    def _policy_patterns(policy: dict[str, Any] | None, key: str) -> list[str]:
        if not isinstance(policy, dict):
            return []
        return [str(pattern).strip() for pattern in (policy.get(key) or []) if str(pattern).strip()]

    @staticmethod
    def _tool_name_matches_patterns(tool_name: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            base_name = RunCommandModule._pattern_base_name(pattern)
            if base_name == tool_name:
                return True
        return False

    @staticmethod
    def _tool_name_matches_exact_patterns(tool_name: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            normalized = str(pattern or "").strip()
            if normalized and "(" not in normalized and normalized == tool_name:
                return True
        return False

    @staticmethod
    def _pattern_base_name(pattern: str) -> str | None:
        normalized = str(pattern or "").strip()
        if not normalized:
            return None
        if "(" not in normalized:
            return normalized
        if not normalized.endswith(")"):
            return None
        base_name, _ = normalized.split("(", 1)
        base_name = base_name.strip()
        return base_name or None

    @staticmethod
    def _parent_idempotency_key(context: Any | None, arguments: dict[str, Any]) -> str | None:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            meta_value = metadata.get(RUN_PARENT_IDEMPOTENCY_KEY_METADATA_KEY)
            if isinstance(meta_value, str) and meta_value.strip():
                return meta_value.strip()

        for key in ("idempotency_key", "idempotencyKey"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def _validate_timeout_arguments(cls, arguments: dict[str, Any]) -> None:
        """Validate timeout aliases before command execution starts."""

        timeout_seconds = arguments.get("timeout_seconds")
        timeout_seconds_camel = arguments.get("timeoutSeconds")
        if timeout_seconds is not None:
            cls._coerce_timeout_seconds(timeout_seconds, "timeout_seconds")
        if timeout_seconds_camel is not None:
            cls._coerce_timeout_seconds(timeout_seconds_camel, "timeoutSeconds")
        if timeout_seconds is not None and timeout_seconds_camel is not None:
            legacy_value = cls._coerce_timeout_seconds(timeout_seconds, "timeout_seconds")
            camel_value = cls._coerce_timeout_seconds(timeout_seconds_camel, "timeoutSeconds")
            if legacy_value != camel_value:
                raise ValueError("timeoutSeconds and timeout_seconds must match when both are provided")

    @classmethod
    def _timeout_seconds(cls, arguments: dict[str, Any]) -> float | None:
        """Return the normalized timeout value from either supported alias."""

        for key in ("timeoutSeconds", "timeout_seconds"):
            value = arguments.get(key)
            if value is not None:
                return cls._coerce_timeout_seconds(value, key)
        return None

    @staticmethod
    def _coerce_timeout_seconds(value: Any, key: str) -> float:
        """Coerce one timeout argument to a positive finite float."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a positive number")
        coerced = float(value)
        if not math.isfinite(coerced) or coerced <= 0:
            raise ValueError(f"{key} must be greater than 0")
        return coerced

    @staticmethod
    def _format_seconds(value: float) -> str:
        """Format a timeout value for CLI-style status output."""

        return f"{value:g}"

    @staticmethod
    def _scoped_parent_idempotency_key(
        parent_key: str | None,
        cwd: str | None,
        sandbox_session_id: str | None,
        env_file_scope: str | None = None,
    ) -> str | None:
        """Salt a parent idempotency key with unambiguous execution scope data."""

        if not parent_key:
            return parent_key
        scope_payload: dict[str, str] = {}
        if cwd:
            scope_payload["cwd"] = cwd
        if sandbox_session_id:
            scope_payload["sandbox_session_id"] = sandbox_session_id
        if env_file_scope:
            scope_payload["env_file"] = env_file_scope
        if not scope_payload:
            return parent_key
        serialized_scope = json.dumps(
            scope_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(serialized_scope.encode("utf-8")).hexdigest()[:16]
        return f"{parent_key}:scope:{digest}"

    @classmethod
    def _validate_sandbox_session_arguments(cls, arguments: dict[str, Any]) -> None:
        """Validate sandbox session aliases before command execution starts."""

        sandbox_session_id = arguments.get("sandbox_session_id")
        sandbox_session_id_camel = arguments.get("sandboxSessionId")
        if sandbox_session_id is not None:
            cls._normalize_sandbox_session_id(sandbox_session_id, "sandbox_session_id")
        if sandbox_session_id_camel is not None:
            cls._normalize_sandbox_session_id(sandbox_session_id_camel, "sandboxSessionId")
        if sandbox_session_id is not None and sandbox_session_id_camel is not None:
            legacy_value = cls._normalize_sandbox_session_id(sandbox_session_id, "sandbox_session_id")
            camel_value = cls._normalize_sandbox_session_id(sandbox_session_id_camel, "sandboxSessionId")
            if legacy_value != camel_value:
                raise ValueError("sandboxSessionId and sandbox_session_id must match when both are provided")

    @classmethod
    def _sandbox_session_id(cls, arguments: dict[str, Any]) -> str | None:
        """Return the normalized sandbox session id from either supported alias."""

        for key in ("sandboxSessionId", "sandbox_session_id"):
            value = arguments.get(key)
            if value is not None:
                return cls._normalize_sandbox_session_id(value, key)
        return None

    @staticmethod
    def _normalize_sandbox_session_id(value: Any, key: str) -> str:
        """Normalize a sandbox session id without accepting empty or non-string values."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return value.strip()

    @classmethod
    def _validate_env_file_arguments(cls, arguments: dict[str, Any]) -> None:
        """Validate env-file aliases before command execution starts."""

        env_file = arguments.get("env_file")
        env_file_camel = arguments.get("envFile")
        legacy_value = cls._normalize_env_file(env_file, "env_file") if env_file is not None else None
        camel_value = cls._normalize_env_file(env_file_camel, "envFile") if env_file_camel is not None else None
        if legacy_value is not None and camel_value is not None and legacy_value != camel_value:
            raise RunEnvFileValidationError("envFile and env_file must match when both are provided")

    @classmethod
    def _env_file(cls, arguments: dict[str, Any]) -> str | None:
        """Return the normalized env-file path from either supported alias."""

        for key in ("envFile", "env_file"):
            value = arguments.get(key)
            if value is not None:
                return cls._normalize_env_file(value, key)
        return None

    @classmethod
    def _normalize_env_file(cls, value: Any, key: str) -> str:
        """Normalize env-file paths as workspace-relative file references."""

        if not isinstance(value, str) or not value.strip():
            raise RunEnvFileValidationError(f"{key} must be a non-empty workspace-relative path")
        text = value.strip()
        if cls._is_anchored_path(text) or text.startswith("~"):
            raise RunEnvFileValidationError(f"{key} must be workspace-relative")
        parts: list[str] = []
        for raw_part in text.replace("\\", "/").split("/"):
            part = raw_part.strip()
            if not part or part == ".":
                continue
            if part == "..":
                raise RunEnvFileValidationError(f"{key} must not contain path traversal")
            parts.append(part)
        if not parts:
            raise RunEnvFileValidationError(f"{key} must reference a workspace-relative file")
        return "/".join(parts)

    @classmethod
    def _validate_cwd_arguments(cls, arguments: dict[str, Any]) -> None:
        """Validate cwd aliases before command execution starts."""

        cwd = arguments.get("cwd")
        working_directory = arguments.get("workingDirectory")
        if cwd is not None:
            cls._normalize_cwd(cwd, "cwd")
        if working_directory is not None:
            cls._normalize_cwd(working_directory, "workingDirectory")
        if cwd is not None and working_directory is not None:
            cwd_value = cls._normalize_cwd(cwd, "cwd")
            working_directory_value = cls._normalize_cwd(working_directory, "workingDirectory")
            if cwd_value != working_directory_value:
                raise ValueError("cwd and workingDirectory must match when both are provided")

    @classmethod
    def _cwd(cls, arguments: dict[str, Any]) -> str | None:
        """Return the normalized cwd from either supported alias."""

        for key in ("workingDirectory", "cwd"):
            value = arguments.get(key)
            if value is not None:
                normalized = cls._normalize_cwd(value, key)
                return None if normalized == "." else normalized
        return None

    @classmethod
    def _normalize_cwd(cls, value: Any, key: str) -> str:
        """Normalize the run-level cwd as a workspace-relative directory."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty workspace-relative path")
        text = value.strip()
        if cls._is_anchored_path(text) or text.startswith("~"):
            raise ValueError(f"{key} must be workspace-relative")
        parts: list[str] = []
        for raw_part in text.replace("\\", "/").split("/"):
            part = raw_part.strip()
            if not part or part == ".":
                continue
            if part == "..":
                raise ValueError(f"{key} must not contain path traversal")
            parts.append(part)
        return "/".join(parts) if parts else "."

    @staticmethod
    def _is_anchored_path(path: str) -> bool:
        """Return whether a path token is absolute or drive-anchored."""

        text = str(path or "").strip()
        if not text:
            return False
        windows_path = PureWindowsPath(text)
        return bool(PurePosixPath(text).is_absolute() or windows_path.is_absolute() or windows_path.drive)

    @classmethod
    def _validate_retain_output_artifact_arguments(cls, arguments: dict[str, Any]) -> None:
        """Validate retained-output artifact aliases before command execution starts."""

        retain_output_artifacts = arguments.get("retain_output_artifacts")
        retain_output_artifacts_camel = arguments.get("retainOutputArtifacts")
        if retain_output_artifacts is not None:
            cls._coerce_retain_output_artifacts(retain_output_artifacts, "retain_output_artifacts")
        if retain_output_artifacts_camel is not None:
            cls._coerce_retain_output_artifacts(retain_output_artifacts_camel, "retainOutputArtifacts")
        if retain_output_artifacts is not None and retain_output_artifacts_camel is not None:
            legacy_value = cls._coerce_retain_output_artifacts(
                retain_output_artifacts,
                "retain_output_artifacts",
            )
            camel_value = cls._coerce_retain_output_artifacts(
                retain_output_artifacts_camel,
                "retainOutputArtifacts",
            )
            if legacy_value != camel_value:
                raise ValueError(
                    "retainOutputArtifacts and retain_output_artifacts must match when both are provided"
                )

    @classmethod
    def _retain_output_artifacts(cls, arguments: dict[str, Any]) -> bool:
        """Return whether oversized output spill files should be retained."""

        for key in ("retainOutputArtifacts", "retain_output_artifacts"):
            value = arguments.get(key)
            if value is not None:
                return cls._coerce_retain_output_artifacts(value, key)
        return False

    @staticmethod
    def _coerce_retain_output_artifacts(value: Any, key: str) -> bool:
        """Coerce one retained-output flag without accepting truthy non-bools."""

        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        return value

    @staticmethod
    def _unsupported_shell_feature_message(command_text: str) -> str | None:
        """Detect raw-shell syntax that the governed facade intentionally rejects."""

        first_token = RunCommandModule._first_unquoted_token(command_text)
        if first_token and RunCommandModule._looks_like_env_assignment(first_token):
            return (
                "Unsupported shell feature: environment assignment prefixes are not supported by "
                "the governed shell facade"
            )

        quote: str | None = None
        escaped = False
        position = 0
        length = len(command_text)
        while position < length:
            char = command_text[position]

            if escaped:
                escaped = False
                position += 1
                continue

            if char == "\\":
                escaped = True
                position += 1
                continue

            if quote is not None:
                if char == quote:
                    quote = None
                    position += 1
                    continue
                if quote != "'" and char == "$":
                    return RunCommandModule._shell_expansion_message(command_text, position)
                if quote != "'" and char == "`":
                    return "Unsupported shell feature: command substitution is not supported by the governed shell facade"
                position += 1
                continue

            if char in {'"', "'"}:
                quote = char
                position += 1
                continue

            if char in {"<", ">"}:
                return "Unsupported shell feature: redirection is not supported by the governed shell facade"

            if char == "&":
                next_char = command_text[position + 1] if position + 1 < length else ""
                previous_char = command_text[position - 1] if position > 0 else ""
                if next_char != "&" and previous_char != "&":
                    return (
                        "Unsupported shell feature: background execution is not supported by "
                        "the governed shell facade"
                    )

            if char == "$":
                return RunCommandModule._shell_expansion_message(command_text, position)

            if char == "`":
                return "Unsupported shell feature: command substitution is not supported by the governed shell facade"

            position += 1

        return None

    @staticmethod
    def _shell_expansion_message(command_text: str, position: int) -> str:
        """Return the specific unsupported expansion diagnostic for a dollar token."""

        next_char = command_text[position + 1] if position + 1 < len(command_text) else ""
        if next_char in {"(", "`"}:
            return "Unsupported shell feature: command substitution is not supported by the governed shell facade"
        if next_char == "{":
            return "Unsupported shell feature: environment expansion is not supported by the governed shell facade"
        if next_char and (next_char == "_" or next_char.isalpha()):
            return "Unsupported shell feature: environment expansion is not supported by the governed shell facade"
        return "Unsupported shell feature: shell expansion is not supported by the governed shell facade"

    @staticmethod
    def _first_unquoted_token(command_text: str) -> str | None:
        """Extract the first shell-like token while respecting simple quoting."""

        token: list[str] = []
        quote: str | None = None
        escaped = False
        seen_token = False
        for char in command_text.lstrip():
            if escaped:
                token.append(char)
                escaped = False
                seen_token = True
                continue
            if char == "\\":
                escaped = True
                seen_token = True
                continue
            if quote is not None:
                if char == quote:
                    quote = None
                    continue
                token.append(char)
                seen_token = True
                continue
            if char in {'"', "'"}:
                quote = char
                seen_token = True
                continue
            if char.isspace():
                if seen_token:
                    break
                continue
            token.append(char)
            seen_token = True
        if not token:
            return None
        return "".join(token)

    @staticmethod
    def _looks_like_env_assignment(token: str) -> bool:
        """Return whether a token resembles a shell environment assignment prefix."""

        if "=" not in token:
            return False
        name, value = token.split("=", 1)
        if not name or value == "":
            return False
        if not (name[0] == "_" or name[0].isalpha()):
            return False
        return all(ch == "_" or ch.isalnum() for ch in name[1:])

    @staticmethod
    async def _cleanup_invocation_spill_dir(spill_dir: Path) -> None:
        """Delete the private spill directory for a completed run invocation."""

        await asyncio.to_thread(shutil.rmtree, spill_dir, ignore_errors=True)

    @staticmethod
    def _is_passthrough_runtime_exception(exc: BaseException) -> bool:
        """Return whether governance exceptions should propagate unchanged."""

        try:
            from ...protocol import ApprovalRequiredError, GovernanceDeniedError
        except ImportError:
            return False
        return isinstance(exc, (ApprovalRequiredError, GovernanceDeniedError))
