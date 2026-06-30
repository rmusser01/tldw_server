# app/core/Skills/skill_executor.py
#
# Execute skills with argument substitution and fork support
#
"""
Skill Executor
==============

Executes skills with:
- Argument substitution ($ARGUMENTS, ${0}, ${1}, etc.)
- Tool resolution against MCP registry
- Inline and fork execution modes
"""

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class SkillExecutionResult:
    """Result of skill execution."""
    skill_name: str
    rendered_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    model_override: Optional[str] = None
    execution_mode: str = "inline"  # "inline" or "fork"
    fork_output: Optional[str] = None
    dry_run: bool = False


@dataclass
class RequestContext:
    """Context for skill execution."""
    user_id: int
    default_model: Optional[str] = None
    default_provider: Optional[str] = None
    api_key: Optional[str] = None
    app_config: Optional[dict[str, Any]] = None
    client_id: Optional[str] = None
    conversation_id: Optional[str] = None
    available_tools: list[str] = field(default_factory=list)
    tool_definitions: Optional[list[dict[str, Any]]] = None
    tool_executor: Optional[Any] = None
    max_tool_calls: int = 5


class SkillExecutor:
    """Execute skills with argument substitution and tool resolution."""

    # Pattern for $ARGUMENTS[N] or ${N} (brace-delimited to avoid collision with dollar amounts)
    INDEXED_ARG_PATTERN = re.compile(r'\$ARGUMENTS\[(\d+)\]|\$\{(\d+)\}')

    def substitute_arguments(self, content: str, arguments: str) -> str:
        """
        Replace argument placeholders with actual values.

        Supports:
        - $ARGUMENTS - all arguments as a single string
        - $ARGUMENTS[N] - specific argument by index (0-based)
        - ${N} - shorthand for $ARGUMENTS[N] (brace-delimited to avoid collision with $100 etc.)

        Args:
            content: The skill content with placeholders
            arguments: The arguments string

        Returns:
            Content with arguments substituted
        """
        if not content:
            return content

        # Parse arguments
        try:
            args = shlex.split(arguments) if arguments else []
        except ValueError:
            # If shlex fails, fall back to simple split
            args = arguments.split() if arguments else []

        def replace_indexed(match: re.Match) -> str:
            # Get the index from either group
            idx_str = match.group(1) or match.group(2)
            idx = int(idx_str)
            if idx < len(args):
                return args[idx]
            return ""  # Empty string for out-of-range indices

        # Replace indexed arguments first
        result = self.INDEXED_ARG_PATTERN.sub(replace_indexed, content)

        # Replace $ARGUMENTS with the full arguments string
        result = result.replace("$ARGUMENTS", arguments or "")

        return result

    def resolve_allowed_tools(
        self,
        allowed_tools: Optional[list[str]],
        available_tools: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Resolve allowed-tools against available tools.

        Handles patterns like:
        - Simple names: "Read", "Grep", "Glob"
        - Tool with command restriction: "Bash(git *)"

        Args:
            allowed_tools: List of allowed tool specs from skill
            available_tools: List of available tools from MCP registry

        Returns:
            Resolved list of allowed tool names/patterns
        """
        if not allowed_tools:
            return []

        resolved = []
        for tool_spec in allowed_tools:
            tool_spec = tool_spec.strip()
            if not tool_spec:
                continue

            if "(" in tool_spec:
                # Tool with command restriction: "Bash(git *)"
                # Keep the full pattern for command validation later
                resolved.append(tool_spec)
            else:
                # Simple tool name - add if no available_tools check
                # or if it's in the available tools
                if available_tools is None or tool_spec in available_tools:
                    resolved.append(tool_spec)
                else:
                    logger.warning(f"Tool '{tool_spec}' not available, skipping")

        return resolved

    def _tool_name_from_def(self, tool: dict[str, Any]) -> str:
        """Extract tool name from MCP or OpenAI tool definition."""
        if not isinstance(tool, dict):
            return ""
        name = tool.get("name")
        if isinstance(name, str) and name:
            return name
        func = tool.get("function")
        if isinstance(func, dict):
            func_name = func.get("name")
            if isinstance(func_name, str):
                return func_name
        return ""

    def filter_tools_for_skill(
        self,
        all_tools: list[dict[str, Any]],
        allowed_tools: list[str],
    ) -> list[dict[str, Any]]:
        """
        Filter MCP tools based on skill's allowed-tools.

        Args:
            all_tools: List of all available MCP tool definitions
            allowed_tools: List of allowed tool patterns from skill

        Returns:
            Filtered list of tool definitions
        """
        if not allowed_tools:
            return []

        # Extract base tool names from patterns
        allowed_base_names = set()
        for tool_spec in allowed_tools:
            base_name = tool_spec.split("(")[0].strip() if "(" in tool_spec else tool_spec.strip()
            allowed_base_names.add(base_name)

        # Filter tools
        filtered = []
        for tool in all_tools:
            tool_name = self._tool_name_from_def(tool)
            if tool_name in allowed_base_names:
                filtered.append(tool)

        return filtered

    def _normalize_tool_definitions(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize MCP tool definitions into OpenAI function tool format."""
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                normalized.append(tool)
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = tool.get("description") or ""
            parameters = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters") or {}
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters if isinstance(parameters, dict) else {},
                    },
                }
            )
        return normalized

    def matches_tool_pattern(self, tool_name: str, command: str, pattern: str) -> bool:
        """
        Check if a tool invocation matches an allowed pattern.

        Examples:
        - "Bash(git *)" matches Bash tool with command "git status"
        - "Bash(npm run *)" matches Bash tool with command "npm run test"

        Args:
            tool_name: The tool being invoked (e.g., "Bash")
            command: The command/arguments for the tool
            pattern: The allowed pattern (e.g., "Bash(git *)")

        Returns:
            True if the invocation matches the pattern
        """
        if "(" not in pattern:
            # Simple tool name, just match the name
            return tool_name == pattern.strip()

        # Extract base name and command pattern
        match = re.match(r'^(\w+)\((.+)\)$', pattern.strip())
        if not match:
            return False

        base_name, cmd_pattern = match.groups()

        if tool_name != base_name:
            return False

        # Convert glob-style pattern to regex
        # * matches any characters (including spaces for command args)
        regex_pattern = re.escape(cmd_pattern)
        regex_pattern = regex_pattern.replace(r'\*', '.*')
        regex_pattern = f'^{regex_pattern}$'

        try:
            return bool(re.match(regex_pattern, command.strip()))
        except re.error:
            logger.warning(f"Invalid command pattern: {pattern}")
            return False

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from an OpenAI-style response payload."""
        if not isinstance(response, dict):
            return []
        choices = response.get("choices") or []
        if not choices:
            return []
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or message.get("toolCalls")
        if isinstance(tool_calls, list):
            return tool_calls
        return []

    def _extract_response_text(self, response: Any) -> str:
        """Extract assistant text from an OpenAI-style response payload."""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = response.get("content") or response.get("text")
            if isinstance(text, str):
                return text
        if isinstance(response, str):
            return response
        return ""

    async def execute(
        self,
        skill_data: dict[str, Any],
        arguments: str,
        context: Optional[RequestContext] = None,
        dry_run: bool = False,
    ) -> SkillExecutionResult:
        """
        Execute a skill.

        Args:
            skill_data: The skill data dict (from SkillsService.get_skill)
            arguments: Arguments to pass to the skill
            context: Request context for execution
            dry_run: If true, render prompt metadata without model/tool execution

        Returns:
            SkillExecutionResult with rendered prompt and metadata
        """
        skill_name = skill_data.get("name", "unknown")
        content = skill_data.get("content", "")
        allowed_tools = skill_data.get("allowed_tools") or []
        model = skill_data.get("model")
        execution_context = skill_data.get("context", "inline")

        # Substitute arguments
        rendered = self.substitute_arguments(content, arguments)

        # Resolve allowed tools
        available = context.available_tools if context else None
        resolved_tools = self.resolve_allowed_tools(allowed_tools, available)

        if dry_run:
            return SkillExecutionResult(
                skill_name=skill_name,
                rendered_prompt=rendered,
                allowed_tools=resolved_tools,
                model_override=model,
                execution_mode="fork" if execution_context == "fork" else "inline",
                fork_output=None,
                dry_run=True,
            )

        if execution_context == "fork":
            return await self._execute_forked(
                skill_name=skill_name,
                rendered_prompt=rendered,
                allowed_tools=resolved_tools,
                model=model,
                context=context,
            )
        else:
            return self._execute_inline(
                skill_name=skill_name,
                rendered_prompt=rendered,
                allowed_tools=resolved_tools,
                model=model,
            )

    def _execute_inline(
        self,
        skill_name: str,
        rendered_prompt: str,
        allowed_tools: list[str],
        model: Optional[str] = None,
    ) -> SkillExecutionResult:
        """
        Inline execution - inject prompt into current conversation.

        Returns the rendered prompt for inclusion in the main conversation.
        """
        return SkillExecutionResult(
            skill_name=skill_name,
            rendered_prompt=rendered_prompt,
            allowed_tools=allowed_tools,
            model_override=model,
            execution_mode="inline",
            dry_run=False,
        )

    async def _execute_forked(
        self,
        skill_name: str,
        rendered_prompt: str,
        allowed_tools: list[str],
        model: Optional[str] = None,
        context: Optional[RequestContext] = None,
    ) -> SkillExecutionResult:
        """
        Fork execution - run in isolated subagent context.

        Creates an isolated chat session for skill execution.
        """
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
        from tldw_Server_API.app.core.Tools.tool_executor import ToolExecutionError, ToolExecutor

        provider = context.default_provider if context else None
        if not provider:
            try:
                from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
                provider = DEFAULT_LLM_PROVIDER
            except Exception:
                provider = "openai"

        model_to_use = model or (context.default_model if context else None)

        tool_executor = context.tool_executor if context and context.tool_executor else None
        if tool_executor is None:
            try:
                tool_executor = ToolExecutor()
            except Exception as e:
                logger.warning(f"Tool executor unavailable for skill fork: {e}")
                tool_executor = None

        tool_defs: list[dict[str, Any]] = []
        if context and context.tool_definitions is not None:
            tool_defs = context.tool_definitions
        elif tool_executor is not None:
            try:
                listing = await tool_executor.list_tools(
                    user_id=str(context.user_id) if context else None,
                    client_id=context.client_id if context else None,
                )
                tool_defs = listing.get("tools", []) or []
            except Exception as e:
                logger.warning(f"Failed to list tools for skill fork: {e}")
                tool_defs = []

        tool_defs = [
            t for t in tool_defs
            if not isinstance(t, dict) or t.get("canExecute", True)
        ]
        tool_defs = self.filter_tools_for_skill(tool_defs, allowed_tools)
        llm_tools = self._normalize_tool_definitions(tool_defs)

        system_prompt = (
            f'You are executing the skill "{skill_name}".\n'
            "Follow the instructions below and return your findings.\n\n"
            f"{rendered_prompt}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Execute the skill instructions."}
        ]

        max_steps = int(context.max_tool_calls) if context else 5
        max_steps = max(1, max_steps)
        final_output = ""

        for step in range(max_steps + 1):
            response = await perform_chat_api_call_async(
                messages=messages,
                api_provider=provider,
                model=model_to_use,
                system_message=system_prompt,
                tools=llm_tools or None,
                api_key=context.api_key if context else None,
                app_config=context.app_config if context else None,
            )

            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                final_output = self._extract_response_text(response)
                break

            if not llm_tools:
                final_output = (
                    self._extract_response_text(response)
                    or "Tool calls were requested but no tools are allowed for this skill."
                )
                logger.warning(f"Tool calls requested but no tools are allowed for skill fork '{skill_name}'.")
                break

            if tool_executor is None:
                final_output = self._extract_response_text(response)
                logger.warning("Tool calls requested but no executor available for skill fork.")
                break

            for idx, tc in enumerate(tool_calls):
                tool_name = tc.get("function", {}).get("name")
                tool_args_str = tc.get("function", {}).get("arguments", "{}")
                if not tool_name:
                    continue
                try:
                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else {}
                except json.JSONDecodeError:
                    tool_args = {}

                tool_call_id = tc.get("id") or f"skill_tool_{step}_{idx}"
                try:
                    result = await tool_executor.execute(
                        user_id=str(context.user_id) if context else None,
                        client_id=context.client_id if context else None,
                        tool_name=tool_name,
                        arguments=tool_args,
                        allowed_tools=allowed_tools,
                    )
                    result_payload = json.dumps(result, default=str)
                except ToolExecutionError as e:
                    result_payload = f"Error: {e}"
                except Exception as e:
                    logger.warning(
                        f"Unexpected error executing tool '{tool_name}' in skill fork '{skill_name}': {e}",
                        exc_info=True,
                    )
                    result_payload = f"Error: {e}"

                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result_payload})

        return SkillExecutionResult(
            skill_name=skill_name,
            rendered_prompt=rendered_prompt,
            allowed_tools=[],
            model_override=None,
            execution_mode="fork",
            fork_output=final_output,
            dry_run=False,
        )


# Skill tool definition for LLM invocation
SKILL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "Skill",
        "description": (
            "Execute a skill with optional arguments. Use this to invoke available skills "
            "that provide specialized capabilities. Check the available-skills context for "
            "skill names and their purposes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The skill name to execute (e.g., 'summarize', 'code-review')"
                },
                "args": {
                    "type": "string",
                    "description": "Optional arguments for the skill (e.g., 'detailed' or 'issue-123')"
                }
            },
            "required": ["skill"]
        }
    }
}
