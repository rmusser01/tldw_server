"""LLMCaller abstraction for the MCP adapter's LLM-driven orchestration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: dict[str, int | float] | None = None


class LLMCaller(ABC):
    """Abstract interface for calling an LLM with tool definitions."""

    @abstractmethod
    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send messages + tool definitions to LLM, return response."""


def mcp_tools_to_openai_format(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI function-calling format.

    MCP tools have: name, description, inputSchema (JSON Schema)
    OpenAI tools have: type="function", function={name, description, parameters}
    """
    result = []
    for tool in mcp_tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object"}),
            },
        })
    return result
