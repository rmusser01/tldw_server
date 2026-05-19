"""MCP module exposing the phase-1 virtual CLI `run` tool."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Mapping

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

_RUN_WRITE_BACKEND_TOOLS = {"fs.write_text", "sandbox.run"}


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
        run_tool = create_tool_definition(
            name="run",
            description="Execute a governed command in the MCP virtual CLI runtime.",
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
                },
                "required": ["command"],
            },
            metadata={
                "category": "utility",
                "notes": "Wrapper tool; nested prepared MCP calls carry path/process metadata",
            },
        )
        run_tool["inputSchema"]["additionalProperties"] = False
        return [run_tool]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        if tool_name != "run":
            raise ValueError(f"Unknown tool: {tool_name}")
        args = self.sanitize_input(arguments or {})
        self.validate_tool_arguments(tool_name, args)

        command_text = str(args.get("command") or "").strip()
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

        protocol = await self._resolve_protocol()
        spill_dir = await self._resolve_spill_dir(context)
        spill_threshold_bytes = self._setting_int("spill_threshold_bytes", default=65_536)
        preview_line_limit = self._setting_int("preview_line_limit", default=200)
        preview_byte_limit = self._setting_int("preview_byte_limit", default=51_200)
        adapter_context = AdapterContext(
            protocol=protocol,
            request_context=context,
            visible_commands=visible,
            parent_idempotency_key=self._parent_idempotency_key(context, arguments),
        )
        adapters = PhaseOneCommandAdapters(adapter_context)
        executor = CommandRuntimeExecutor(
            backend=_AdapterBackend(adapters),
            spill_dir=spill_dir,
            spill_threshold_bytes=spill_threshold_bytes,
            preview_bytes=preview_byte_limit,
        )
        try:
            if adapters.requires_whole_chain_preflight(chain):
                await adapters.preflight_chain(chain)
            result = await executor.execute(chain)
        except PreflightCommandError as exc:
            result = CommandExecutionResult(
                stdout="",
                stderr=exc.result.stderr,
                exit_code=exc.result.exit_code,
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
        try:
            return present_command_execution_result(
                result,
                spill_dir=spill_dir,
                byte_limit=preview_byte_limit,
                line_limit=preview_line_limit,
            )
        finally:
            await self._cleanup_spill_artifacts(result)

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name != "run":
            raise ValueError(f"Unknown tool: {tool_name}")
        unknown = sorted({key for key in arguments.keys()} - {"command", "idempotencyKey", "idempotency_key"})
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
        if tool_name != "run":
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

    async def _resolve_workspace_root(self, context: Any | None) -> Path | None:
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

    @staticmethod
    async def _cleanup_spill_artifacts(result: CommandExecutionResult) -> None:
        spill_paths: set[Path] = set()

        def _record(spill: Any) -> None:
            path_value = getattr(spill, "path", None)
            if isinstance(path_value, str) and path_value.strip():
                spill_paths.add(Path(path_value))

        _record(result.stdout_spill)
        _record(result.stderr_spill)
        for spill in result.stderr_spills:
            _record(spill)
        for step in result.steps:
            _record(step.stdout_spill)
            _record(step.stderr_spill)

        if not spill_paths:
            return

        await asyncio.to_thread(RunCommandModule._unlink_spills, spill_paths)

    @staticmethod
    def _unlink_spills(spill_paths: set[Path]) -> None:
        for spill_path in spill_paths:
            try:
                spill_path.unlink(missing_ok=True)
            except OSError:
                continue

    @staticmethod
    def _is_passthrough_runtime_exception(exc: BaseException) -> bool:
        try:
            from ...protocol import ApprovalRequiredError, GovernanceDeniedError
        except ImportError:
            return False
        return isinstance(exc, (ApprovalRequiredError, GovernanceDeniedError))
