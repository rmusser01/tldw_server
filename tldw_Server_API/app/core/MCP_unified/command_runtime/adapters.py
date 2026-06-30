"""Phase-1 command adapters for the MCP-backed virtual CLI runtime."""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .executor import HandlerInvocationContext
from .models import CommandChain, CommandSpillReference, CommandStepResult
from .registry import CommandDescriptor

_CWD_UNSET = object()


@dataclass(slots=True)
class AdapterContext:
    """Execution context shared by all command adapters."""

    protocol: Any
    request_context: Any
    visible_commands: Mapping[str, CommandDescriptor]
    parent_idempotency_key: str | None = None
    cwd: str | None = None
    sandbox_session_id: str | None = None
    sandbox_env: Mapping[str, str] | None = None
    shell_name: str | None = None


@dataclass(frozen=True, slots=True)
class _GovernedCallPlan:
    tool_name: str
    arguments: dict[str, Any]
    renderer: Callable[[Any], str]


@dataclass(frozen=True, slots=True)
class _UsageError:
    message: str
    exit_code: int = 2


@dataclass(slots=True)
class _PreparedStep:
    prepared: Any
    plan: _GovernedCallPlan


class PreflightCommandError(Exception):
    """Raised when a preflighted command cannot be parsed/validated."""

    def __init__(self, result: CommandStepResult):
        super().__init__(result.stderr if isinstance(result.stderr, str) else "preflight command error")
        self.result = result


def normalize_step_content(argv: list[str]) -> str:
    """Return a deterministic normalized representation for one command step."""

    return json.dumps([str(part) for part in argv], ensure_ascii=False, separators=(",", ":"))


def derive_step_idempotency_key(parent_key: str | None, argv: list[str], step_index: int) -> str | None:
    """Derive deterministic nested idempotency key from parent key + normalized step."""

    base = str(parent_key or "").strip()
    if not base:
        return None
    normalized = normalize_step_content(argv)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{base}:step:{step_index}:{digest}"


def run_help_text(visible_commands: Mapping[str, CommandDescriptor]) -> str:
    """Render policy-filtered help for the run command surface."""

    lines = ["Virtual CLI commands available in this context:"]
    for name in sorted(visible_commands.keys()):
        descriptor = visible_commands[name]
        lines.append(f"  {name:9} {descriptor.summary}")
    return "\n".join(lines)


def visible_command_registry(
    *,
    tools_payload: dict[str, Any],
    registry: Any,
) -> dict[str, CommandDescriptor]:
    """Filter phase-1 commands by currently executable backing tools."""

    allowed_tools: set[str] = set()
    for tool in tools_payload.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        can_execute = tool.get("canExecute")
        if can_execute is False:
            continue
        allowed_tools.add(name.strip())
    return registry.visible_commands(allowed_tools)


class PhaseOneCommandAdapters:
    """Adapter layer that separates pure transforms from governed MCP tool calls."""

    _PURE_GREP_MAX_OUTPUT_BYTES = 1_000_000

    def __init__(self, context: AdapterContext) -> None:
        self.context = context
        self._initial_cwd = context.cwd
        self._current_cwd = context.cwd
        self._preflighted: dict[tuple[str, int], _PreparedStep] = {}
        self._next_step_index = 0

    def validate_chain(self, chain: CommandChain) -> CommandStepResult | None:
        """Return a fail-closed result when stateful cwd commands are ambiguous."""

        error = self._validate_cd_flow(chain)
        if error is None:
            return None
        return CommandStepResult(stderr=error.message, exit_code=error.exit_code)

    def requires_whole_chain_preflight(self, chain: CommandChain) -> bool:
        """Return True when the chain includes any governed command."""

        for segment in chain.segments:
            for invocation in segment.commands:
                if not invocation.argv:
                    continue
                descriptor = self.context.visible_commands.get(invocation.argv[0])
                if descriptor is not None and not descriptor.pure_transform:
                    return True
        return False

    async def preflight_chain(self, chain: CommandChain) -> None:
        """Prepare all governed steps before execution starts."""

        flow_error = self.validate_chain(chain)
        if flow_error is not None:
            raise PreflightCommandError(flow_error)

        handled_exceptions = self._handled_governed_exceptions()
        step_index = 0
        current_cwd = self._initial_cwd
        for segment in chain.segments:
            for invocation in segment.commands:
                argv = list(invocation.argv)
                if not argv:
                    continue
                descriptor = self.context.visible_commands.get(argv[0])
                if descriptor is None:
                    raise PreflightCommandError(
                        self._unknown_command_result(argv[0])
                    )
                if descriptor.pure_transform:
                    if argv[0] == "cd":
                        cd_target = self._cd_target(argv, current_cwd)
                        if not isinstance(cd_target, _UsageError):
                            current_cwd = cd_target
                    step_index += 1
                    continue

                plan_or_error = self._governed_plan(argv, cwd=current_cwd)
                if isinstance(plan_or_error, _UsageError):
                    raise PreflightCommandError(
                        CommandStepResult(stderr=plan_or_error.message, exit_code=plan_or_error.exit_code)
                    )
                try:
                    prepared = await self.context.protocol.prepare_tool_call(
                        params={"name": plan_or_error.tool_name, "arguments": dict(plan_or_error.arguments)},
                        context=self.context.request_context,
                        idempotency_key=self._step_idempotency_key(argv, step_index, current_cwd),
                    )
                except handled_exceptions as exc:
                    if self._is_passthrough_governed_exception(exc):
                        raise
                    raise PreflightCommandError(self._governed_error_result(exc)) from exc
                prepared_key = (normalize_step_content(argv), step_index)
                self._preflighted[prepared_key] = _PreparedStep(prepared=prepared, plan=plan_or_error)
                step_index += 1

    async def execute(
        self,
        argv: list[str],
        stdin: Any,
        handler_context: Any | None = None,
    ) -> CommandStepResult:
        """Execute one command invocation for the runtime backend."""

        if not argv:
            return CommandStepResult(stderr="Missing command", exit_code=127)

        step_index = self._resolve_step_index(handler_context)
        descriptor = self.context.visible_commands.get(argv[0])
        if descriptor is None:
            return self._unknown_command_result(argv[0])

        if descriptor.pure_transform:
            return self._execute_pure_transform(argv, stdin, handler_context)

        return await self._execute_governed(argv, step_index)

    async def _execute_governed(self, argv: list[str], step_index: int) -> CommandStepResult:
        signature = normalize_step_content(argv)
        handled_exceptions = self._handled_governed_exceptions()
        try:
            prepared_key = (signature, step_index)
            prepared_step = self._preflighted.pop(prepared_key, None)
            if prepared_step is None:
                plan_or_error = self._governed_plan(argv, cwd=self._current_cwd)
                if isinstance(plan_or_error, _UsageError):
                    return CommandStepResult(stderr=plan_or_error.message, exit_code=plan_or_error.exit_code)
                prepared = await self.context.protocol.prepare_tool_call(
                    params={"name": plan_or_error.tool_name, "arguments": dict(plan_or_error.arguments)},
                    context=self.context.request_context,
                    idempotency_key=self._step_idempotency_key(argv, step_index, self._current_cwd),
                )
                prepared_step = _PreparedStep(prepared=prepared, plan=plan_or_error)

            payload = await self.context.protocol.execute_prepared_tool_call(prepared_step.prepared)
            rendered = prepared_step.plan.renderer(payload)
        except handled_exceptions as exc:
            if self._is_passthrough_governed_exception(exc):
                raise
            return self._governed_error_result(exc)

        return CommandStepResult(stdout=rendered, stderr="", exit_code=0)

    def _resolve_step_index(self, handler_context: Any | None) -> int:
        if isinstance(handler_context, HandlerInvocationContext):
            return int(handler_context.lexical_step_index)
        step_index = self._next_step_index
        self._next_step_index += 1
        return step_index

    def _execute_pure_transform(
        self,
        argv: list[str],
        stdin: Any,
        handler_context: Any | None = None,
    ) -> CommandStepResult:
        """Execute one pure in-memory transform or virtual cwd command."""

        command = argv[0]
        if command == "grep":
            return self._pure_grep(argv, stdin)
        if command == "pwd":
            return self._pure_pwd(argv)
        if command == "cd":
            return self._pure_cd(argv, handler_context)
        if command == "head":
            return self._pure_head(argv, stdin)
        if command == "tail":
            return self._pure_tail(argv, stdin)
        if command == "json":
            return self._pure_json(argv, stdin)
        return CommandStepResult(stderr=f"Unknown command: {command}", exit_code=127)

    def _governed_plan(self, argv: list[str], *, cwd: str | None = None) -> _GovernedCallPlan | _UsageError:
        """Build the governed backend call plan for one virtual CLI command."""

        command = argv[0]
        if command == "ls":
            if len(argv) > 2:
                return _UsageError("usage: ls [path]")
            path = self._path_with_cwd(argv[1] if len(argv) == 2 else ".", cwd)
            if isinstance(path, _UsageError):
                return path
            return _GovernedCallPlan(
                tool_name="fs.list",
                arguments={"path": path},
                renderer=self._render_ls,
            )
        if command == "cat":
            if len(argv) != 2:
                return _UsageError("usage: cat <path>")
            tool_name = self._visible_backend_tool("cat", ("fs.read", "fs.read_text"))
            if tool_name is None:
                return _UsageError(self._unknown_command_message(command), exit_code=127)
            path = self._path_with_cwd(argv[1], cwd)
            if isinstance(path, _UsageError):
                return path
            return _GovernedCallPlan(
                tool_name=tool_name,
                arguments={"path": path},
                renderer=self._render_cat,
            )
        if command == "write":
            if len(argv) < 3:
                return _UsageError("usage: write <path> <content>")
            path = self._path_with_cwd(argv[1], cwd)
            if isinstance(path, _UsageError):
                return path
            return _GovernedCallPlan(
                tool_name="fs.write",
                arguments={"path": path, "content": " ".join(argv[2:]), "mode": "create"},
                renderer=self._render_write,
            )
        if command == "write-create":
            if len(argv) < 3:
                return _UsageError("usage: write-create <path> <content>")
            path = self._path_with_cwd(argv[1], cwd)
            if isinstance(path, _UsageError):
                return path
            return _GovernedCallPlan(
                tool_name="fs.write",
                arguments={"path": path, "content": " ".join(argv[2:]), "mode": "create"},
                renderer=self._render_write,
            )
        if command == "stat":
            if len(argv) != 2:
                return _UsageError("usage: stat <path>")
            path = self._path_with_cwd(argv[1], cwd)
            if isinstance(path, _UsageError):
                return path
            return _GovernedCallPlan(
                tool_name="fs.stat",
                arguments={"path": path},
                renderer=self._render_json_payload,
            )
        if command in {"glob", "find"}:
            if len(argv) not in {2, 3}:
                return _UsageError(f"usage: {command} <pattern> [base_path]")
            arguments = {"pattern": argv[1]}
            base_path = self._path_with_cwd(argv[2], cwd) if len(argv) == 3 else cwd
            if isinstance(base_path, _UsageError):
                return base_path
            if base_path:
                arguments["base_path"] = base_path
            return _GovernedCallPlan(
                tool_name="fs.glob",
                arguments=arguments,
                renderer=self._render_json_payload,
            )
        if command in {"rg", "grep-files"}:
            if len(argv) not in {2, 3}:
                return _UsageError(f"usage: {command} <pattern> [base_path]")
            arguments = {"pattern": argv[1]}
            base_path = self._path_with_cwd(argv[2], cwd) if len(argv) == 3 else cwd
            if isinstance(base_path, _UsageError):
                return base_path
            if base_path:
                arguments["base_path"] = base_path
            return _GovernedCallPlan(
                tool_name="fs.grep",
                arguments=arguments,
                renderer=self._render_json_payload,
            )
        if command == "knowledge":
            return self._knowledge_plan(argv)
        if command == "media":
            return self._media_plan(argv)
        if command == "mcp":
            return self._mcp_plan(argv)
        if command == "sandbox":
            if len(argv) < 2:
                return _UsageError("usage: sandbox <command...>")
            arguments: dict[str, Any] = {"command": argv[1:]}
            if self.context.sandbox_env:
                arguments["env"] = dict(self.context.sandbox_env)
            if self.context.sandbox_session_id:
                arguments["session_id"] = self.context.sandbox_session_id
            else:
                arguments["base_image"] = "python:3.11"
            return _GovernedCallPlan(
                tool_name="sandbox.run",
                arguments=arguments,
                renderer=self._render_json_payload,
            )
        return _UsageError(self._unknown_command_message(command), exit_code=127)

    def _visible_backend_tool(self, command: str, preferred: tuple[str, ...]) -> str | None:
        descriptor = self.context.visible_commands.get(command)
        if descriptor is None:
            return None
        available = set(descriptor.backend_tools)
        for tool_name in preferred:
            if tool_name in available:
                return tool_name
        return None

    def _command_has_visible_backend(self, command: str, tool_name: str) -> bool:
        return self._visible_backend_tool(command, (tool_name,)) is not None

    @staticmethod
    def _unavailable_backend_error(tool_name: str) -> _UsageError:
        return _UsageError(f"{tool_name} unavailable in this context")

    def _path_with_cwd(self, path: str, cwd: str | None | object = _CWD_UNSET) -> str | _UsageError:
        """Apply the current cwd to relative path tokens without mutating token text."""

        current_cwd = self._current_cwd if cwd is _CWD_UNSET else cwd
        text = "." if path is None or path == "" else str(path)
        if not current_cwd or self._is_anchored_path(text):
            return text
        normalized = self._normalize_relative_path(text)
        if isinstance(normalized, _UsageError):
            return normalized
        if normalized == ".":
            return current_cwd
        return f"{current_cwd}/{normalized}"

    def _step_idempotency_key(self, argv: list[str], step_index: int, cwd: str | None) -> str | None:
        """Derive an idempotency key that includes dynamic cwd changes."""

        parent_key = self.context.parent_idempotency_key
        if parent_key and cwd != self._initial_cwd:
            scope_payload = json.dumps(
                {"cwd": cwd or "."},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            scope_digest = hashlib.sha256(scope_payload.encode("utf-8")).hexdigest()[:16]
            parent_key = f"{parent_key}:cwd:{scope_digest}"
        return derive_step_idempotency_key(parent_key, argv, step_index)

    @classmethod
    def _normalize_relative_path(cls, path: str) -> str | _UsageError:
        """Normalize relative separators while preserving literal path segment text."""

        text = "." if path is None or path == "" else str(path)
        if cls._is_anchored_path(text):
            return text
        if text.startswith("~"):
            return _UsageError("path must be workspace-relative when cwd is set")
        parts: list[str] = []
        for raw_part in text.replace("\\", "/").split("/"):
            if not raw_part or raw_part == ".":
                continue
            if raw_part == "..":
                return _UsageError("path traversal is not supported when cwd is set")
            parts.append(raw_part)
        return "/".join(parts) if parts else "."

    @staticmethod
    def _is_anchored_path(path: str) -> bool:
        """Return whether a token is already absolute or drive-anchored."""

        text = "" if path is None else str(path)
        if not text:
            return False
        windows_path = PureWindowsPath(text)
        return bool(PurePosixPath(text).is_absolute() or windows_path.is_absolute() or windows_path.drive)

    def _validate_cd_flow(self, chain: CommandChain) -> _UsageError | None:
        """Reject cd forms whose state cannot be safely modeled before execution."""

        message = "cd is only supported as a semicolon-separated command in v1"
        for segment_index, segment in enumerate(chain.segments):
            for invocation in segment.commands:
                argv = list(invocation.argv)
                if not argv or argv[0] != "cd":
                    continue
                if len(segment.commands) != 1:
                    return _UsageError(message)
                previous_link = chain.links[segment_index - 1] if segment_index > 0 else None
                next_link = chain.links[segment_index] if segment_index < len(chain.links) else None
                if previous_link in {"&&", "||"} or next_link in {"&&", "||"}:
                    return _UsageError(message)
        return None

    def _cd_target(self, argv: list[str], cwd: str | None) -> str | None | _UsageError:
        """Resolve a cd invocation to the next bounded virtual cwd."""

        if len(argv) > 2:
            return _UsageError("usage: cd [path]")
        if len(argv) == 1:
            return None
        return self._resolve_virtual_cwd(argv[1], cwd)

    @classmethod
    def _resolve_virtual_cwd(cls, path: str, cwd: str | None) -> str | None | _UsageError:
        """Resolve one virtual cd target without escaping the workspace root."""

        text = "." if path is None or path == "" else str(path)
        if cls._is_anchored_path(text) or text.startswith("~"):
            return _UsageError("cd path must be workspace-relative")

        parts = [] if not cwd else [part for part in str(cwd).replace("\\", "/").split("/") if part]
        for raw_part in text.replace("\\", "/").split("/"):
            if not raw_part or raw_part == ".":
                continue
            if raw_part == "..":
                if not parts:
                    return _UsageError("cd path must stay within the workspace root")
                parts.pop()
                continue
            parts.append(raw_part)
        return "/".join(parts) if parts else None

    def _knowledge_plan(self, argv: list[str]) -> _GovernedCallPlan | _UsageError:
        if len(argv) < 2:
            return _UsageError("usage: knowledge <search|get> ...")
        sub = argv[1]
        if sub == "search":
            if len(argv) < 3:
                return _UsageError("usage: knowledge search <query>")
            if not self._command_has_visible_backend("knowledge", "knowledge.search"):
                return self._unavailable_backend_error("knowledge.search")
            return _GovernedCallPlan(
                tool_name="knowledge.search",
                arguments={"query": " ".join(argv[2:])},
                renderer=self._render_json_payload,
            )
        if sub == "get":
            if len(argv) != 4:
                return _UsageError("usage: knowledge get <source> <id>")
            if not self._command_has_visible_backend("knowledge", "knowledge.get"):
                return self._unavailable_backend_error("knowledge.get")
            return _GovernedCallPlan(
                tool_name="knowledge.get",
                arguments={"source": argv[2], "id": self._coerce_scalar(argv[3])},
                renderer=self._render_json_payload,
            )
        return _UsageError("usage: knowledge <search|get> ...")

    def _media_plan(self, argv: list[str]) -> _GovernedCallPlan | _UsageError:
        if len(argv) < 2:
            return _UsageError("usage: media <search|get> ...")
        sub = argv[1]
        if sub == "search":
            if len(argv) < 3:
                return _UsageError("usage: media search <query>")
            if not self._command_has_visible_backend("media", "media.search"):
                return self._unavailable_backend_error("media.search")
            return _GovernedCallPlan(
                tool_name="media.search",
                arguments={"query": " ".join(argv[2:])},
                renderer=self._render_json_payload,
            )
        if sub == "get":
            if len(argv) != 3:
                return _UsageError("usage: media get <media_id>")
            if not self._command_has_visible_backend("media", "media.get"):
                return self._unavailable_backend_error("media.get")
            return _GovernedCallPlan(
                tool_name="media.get",
                arguments={"media_id": self._coerce_scalar(argv[2])},
                renderer=self._render_json_payload,
            )
        return _UsageError("usage: media <search|get> ...")

    def _mcp_plan(self, argv: list[str]) -> _GovernedCallPlan | _UsageError:
        if len(argv) != 2:
            return _UsageError("usage: mcp <tools|modules|catalogs>")
        sub = argv[1]
        if sub == "tools":
            if not self._command_has_visible_backend("mcp", "mcp.tools.list"):
                return self._unavailable_backend_error("mcp.tools.list")
            return _GovernedCallPlan(
                tool_name="mcp.tools.list",
                arguments={},
                renderer=self._render_json_payload,
            )
        if sub == "modules":
            if not self._command_has_visible_backend("mcp", "mcp.modules.list"):
                return self._unavailable_backend_error("mcp.modules.list")
            return _GovernedCallPlan(
                tool_name="mcp.modules.list",
                arguments={},
                renderer=self._render_json_payload,
            )
        if sub == "catalogs":
            if not self._command_has_visible_backend("mcp", "mcp.catalogs.list"):
                return self._unavailable_backend_error("mcp.catalogs.list")
            return _GovernedCallPlan(
                tool_name="mcp.catalogs.list",
                arguments={},
                renderer=self._render_json_payload,
            )
        return _UsageError("usage: mcp <tools|modules|catalogs>")

    def _pure_pwd(self, argv: list[str]) -> CommandStepResult:
        """Print the current virtual workspace directory."""

        if len(argv) != 1:
            return CommandStepResult(stderr="usage: pwd", exit_code=2)
        return CommandStepResult(stdout=f"{self._current_cwd or '.'}\n", stderr="", exit_code=0)

    def _pure_cd(self, argv: list[str], handler_context: Any | None) -> CommandStepResult:
        """Change the virtual cwd for later commands in this run invocation."""

        if isinstance(handler_context, HandlerInvocationContext) and handler_context.command_index != 0:
            return CommandStepResult(
                stderr="cd is only supported as a semicolon-separated command in v1",
                exit_code=2,
            )
        target = self._cd_target(argv, self._current_cwd)
        if isinstance(target, _UsageError):
            return CommandStepResult(stderr=target.message, exit_code=target.exit_code)
        self._current_cwd = target
        return CommandStepResult(stdout="", stderr="", exit_code=0)

    def _pure_grep(self, argv: list[str], stdin: Any) -> CommandStepResult:
        if len(argv) < 2:
            return CommandStepResult(stderr="usage: grep <pattern> [-i|--ignore-case]", exit_code=2)
        pattern = argv[1]
        flags = set(argv[2:])
        unsupported_flags = [flag for flag in flags if flag not in {"-i", "--ignore-case"}]
        if unsupported_flags:
            return CommandStepResult(stderr="usage: grep <pattern> [-i|--ignore-case]", exit_code=2)

        ignore_case = "-i" in flags or "--ignore-case" in flags
        needle = pattern.lower() if ignore_case else pattern
        matched: list[str] = []
        matched_byte_count = 0
        for line in self._iter_stdin_lines(stdin):
            haystack = line.lower() if ignore_case else line
            if needle in haystack:
                encoded = line.encode("utf-8")
                if matched_byte_count + len(encoded) > self._PURE_GREP_MAX_OUTPUT_BYTES:
                    return CommandStepResult(
                        stderr=(
                            "grep: matched output exceeds "
                            f"{self._PURE_GREP_MAX_OUTPUT_BYTES} bytes; refine the pattern or narrow the pipeline"
                        ),
                        exit_code=2,
                    )
                matched.append(line)
                matched_byte_count += len(encoded)
        output = "".join(matched)
        return CommandStepResult(stdout=output, stderr="", exit_code=0 if matched else 1)

    def _pure_head(self, argv: list[str], stdin: Any) -> CommandStepResult:
        count_or_error = self._line_count_or_error(argv, default=10, usage="usage: head [count]")
        if isinstance(count_or_error, _UsageError):
            return CommandStepResult(stderr=count_or_error.message, exit_code=count_or_error.exit_code)
        text = self._stdin_text(stdin)
        return CommandStepResult(stdout=self._slice_head(text, count_or_error), stderr="", exit_code=0)

    def _pure_tail(self, argv: list[str], stdin: Any) -> CommandStepResult:
        count_or_error = self._line_count_or_error(argv, default=10, usage="usage: tail [count]")
        if isinstance(count_or_error, _UsageError):
            return CommandStepResult(stderr=count_or_error.message, exit_code=count_or_error.exit_code)
        text = self._stdin_text(stdin)
        return CommandStepResult(stdout=self._slice_tail(text, count_or_error), stderr="", exit_code=0)

    def _pure_json(self, argv: list[str], stdin: Any) -> CommandStepResult:
        if len(argv) > 2:
            return CommandStepResult(stderr="usage: json [path]", exit_code=2)
        text = self._stdin_text(stdin)
        if not text.strip():
            return CommandStepResult(stderr="json: stdin is required", exit_code=1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return CommandStepResult(stderr=f"json: invalid input ({exc.msg})", exit_code=1)

        if len(argv) == 1:
            return CommandStepResult(stdout=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), stderr="", exit_code=0)

        path = argv[1]
        current: Any = payload
        for part in self._split_json_path(path):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return CommandStepResult(stderr=f"json: path not found: {path}", exit_code=1)

        if isinstance(current, str):
            return CommandStepResult(stdout=current, stderr="", exit_code=0)
        return CommandStepResult(stdout=json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True), stderr="", exit_code=0)

    @staticmethod
    def _split_json_path(path: str) -> list[str]:
        normalized = path.strip().lstrip(".")
        if not normalized:
            return []

        parts: list[str] = []
        current: list[str] = []
        escaped = False
        for char in normalized:
            if escaped:
                current.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == ".":
                if current:
                    parts.append("".join(current))
                    current = []
                continue
            current.append(char)
        if escaped:
            current.append("\\")
        if current:
            parts.append("".join(current))
        return parts

    @staticmethod
    def _slice_head(text: str, count: int) -> str:
        if count <= 0:
            return ""
        lines = text.splitlines(keepends=True)
        return "".join(lines[:count])

    @staticmethod
    def _slice_tail(text: str, count: int) -> str:
        if count <= 0:
            return ""
        lines = text.splitlines(keepends=True)
        return "".join(lines[-count:])

    @staticmethod
    def _line_count_or_error(argv: list[str], *, default: int, usage: str) -> int | _UsageError:
        if len(argv) == 1:
            return default
        if len(argv) != 2:
            return _UsageError(usage)
        try:
            return max(0, int(argv[1]))
        except ValueError:
            return _UsageError(usage)

    @staticmethod
    def _coerce_scalar(value: str) -> int | float | str:
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def _stdin_text(stdin: Any) -> str:
        if isinstance(stdin, CommandSpillReference):
            try:
                return stdin.read_text()
            except OSError:
                return ""
        if isinstance(stdin, bytes):
            try:
                return stdin.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        if stdin is None:
            return ""
        return str(stdin)

    @staticmethod
    def _iter_stdin_lines(stdin: Any) -> Iterator[str]:
        if isinstance(stdin, CommandSpillReference):
            try:
                with Path(stdin.path).open("r", encoding=stdin.encoding) as handle:
                    yield from handle
            except OSError:
                return
            return
        text = PhaseOneCommandAdapters._stdin_text(stdin)
        if not text:
            return
        yield from text.splitlines(keepends=True)

    @staticmethod
    def _extract_json_content(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        content = payload.get("content")
        if not isinstance(content, list):
            return payload
        for item in content:
            if isinstance(item, dict) and item.get("type") == "json":
                return item.get("json")
        if len(content) == 1 and isinstance(content[0], dict) and "text" in content[0]:
            return content[0].get("text")
        return payload

    def _render_ls(self, payload: Any) -> str:
        decoded = self._extract_json_content(payload)
        if not isinstance(decoded, dict):
            return str(decoded)
        entries = decoded.get("entries")
        if not isinstance(entries, list):
            return str(decoded)
        lines: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                lines.append(str(entry))
                continue
            name = str(entry.get("name") or entry.get("path") or "")
            if not name:
                continue
            if str(entry.get("type") or "").lower() == "directory":
                name = f"{name}/"
            lines.append(name)
        if decoded.get("truncated") is True:
            remaining = decoded.get("remaining_count")
            if isinstance(remaining, int) and remaining > 0:
                lines.append(f"... truncated ({remaining} more entries)")
            else:
                lines.append("... truncated")
        return "\n".join(lines)

    def _render_cat(self, payload: Any) -> str:
        decoded = self._extract_json_content(payload)
        if isinstance(decoded, dict):
            text = decoded.get("text")
            if text is None:
                text = decoded.get("content")
            rendered = str(text or "")
            if decoded.get("truncated") is True:
                separator = "" if not rendered or rendered.endswith("\n") else "\n"
                rendered = f"{rendered}{separator}{self._render_cat_truncation_marker(decoded)}"
            return rendered
        return str(decoded or "")

    def _render_write(self, payload: Any) -> str:
        decoded = self._extract_json_content(payload)
        if isinstance(decoded, dict):
            path = str(decoded.get("path") or "").strip()
            bytes_written = decoded.get("bytes_written")
            if path and isinstance(bytes_written, int):
                return f"wrote {bytes_written} bytes to {path}"
            if path:
                return f"wrote file: {path}"
        return str(decoded or "")

    @staticmethod
    def _render_cat_truncation_marker(decoded: dict[str, Any]) -> str:
        metadata: list[str] = []
        for key in ("bytes_returned", "bytes_total", "lines_returned"):
            value = decoded.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                metadata.append(f"{key}={value}")
        reason = decoded.get("truncation_reason")
        if isinstance(reason, str) and reason.strip():
            metadata.append(f"truncation_reason={reason.strip().replace(chr(10), ' ').replace(chr(13), ' ')}")
        details = f": {' '.join(metadata)}" if metadata else ""
        return f"[truncated{details}]"

    def _render_json_payload(self, payload: Any) -> str:
        decoded = self._extract_json_content(payload)
        if isinstance(decoded, str):
            return decoded
        try:
            return json.dumps(decoded, ensure_ascii=False, indent=2, sort_keys=True)
        except TypeError:
            return str(decoded)

    @staticmethod
    def _handled_governed_exceptions() -> tuple[type[BaseException], ...]:
        exceptions: list[type[BaseException]] = [OSError, ValueError]
        try:
            from ..protocol import InvalidParamsException
        except ImportError:
            InvalidParamsException = None
        if InvalidParamsException is not None:
            exceptions.append(InvalidParamsException)
        return tuple(exceptions)

    @classmethod
    def _governed_error_result(cls, exc: BaseException) -> CommandStepResult:
        exit_code = 2 if cls._is_usage_like_exception(exc) else 1
        message = str(exc).strip() or exc.__class__.__name__
        return CommandStepResult(stderr=message, exit_code=exit_code)

    @staticmethod
    def _is_usage_like_exception(exc: BaseException) -> bool:
        if isinstance(exc, ValueError):
            return True
        try:
            from ..protocol import InvalidParamsException
        except ImportError:
            InvalidParamsException = None
        return InvalidParamsException is not None and isinstance(exc, InvalidParamsException)

    @staticmethod
    def _is_passthrough_governed_exception(exc: BaseException) -> bool:
        try:
            from ..protocol import ApprovalRequiredError, GovernanceDeniedError
        except ImportError:
            return False
        return isinstance(exc, (ApprovalRequiredError, GovernanceDeniedError))

    def _unknown_command_result(self, command: str) -> CommandStepResult:
        return CommandStepResult(stderr=self._unknown_command_message(command), exit_code=127)

    def _unknown_command_message(self, command: str) -> str:
        available = sorted(self.context.visible_commands.keys())
        suggestions = difflib.get_close_matches(command, available, n=3, cutoff=0.3)
        if suggestions:
            return (
                f"Unknown command: {command}. "
                f"Did you mean: {', '.join(suggestions)}? "
                f"Available: {', '.join(available)}"
            )
        return f"Unknown command: {command}. Available: {', '.join(available)}"
