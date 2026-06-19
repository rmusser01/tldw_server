"""Deterministic fixture runtimes for MCP smoke tests."""

from __future__ import annotations

from typing import Any

from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext


class SmokeFixtureGatewayRuntime:
    """Small safe gateway runtime used by in-process smoke tests."""

    name = "smoke-fixture-gateway"
    version = "0.0-test"

    def __init__(
        self,
        *,
        include_denied_tool: bool = False,
        denied_tool_name: str = "smoke.denied",
    ) -> None:
        self.denied_tool_name = denied_tool_name if include_denied_tool else None
        self.call_requests: list[tuple[str, dict[str, Any], GatewayRequestContext]] = []
        self.resource_reads: list[tuple[str, GatewayRequestContext]] = []
        self.prompt_gets: list[tuple[str, dict[str, Any], GatewayRequestContext]] = []

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return the deterministic smoke fixture tool catalog."""

        tools = [
            {
                "name": "echo.search",
                "description": "Echo a search query for smoke testing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "annotations": {"readOnlyHint": True},
            }
        ]
        if self.denied_tool_name is not None:
            tools.append(
                {
                    "name": self.denied_tool_name,
                    "description": "Always denied by the smoke fixture policy.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Execute the safe echo tool or raise the fixture policy denial."""

        self.call_requests.append((name, dict(arguments), context))
        if name == self.denied_tool_name:
            raise GatewayPolicyDenied(
                "Smoke fixture denied tool execution",
                reason_code="smoke_policy_denied",
                provenance={"tool": name},
            )
        if name != "echo.search":
            raise NotImplementedError(name)
        query = str(arguments.get("query", ""))
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"echo.search:{query}",
                }
            ],
            "structuredContent": {"query": query},
        }

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return one safe read-only smoke resource."""

        return [
            {
                "uri": "resource://smoke/doc",
                "name": "Smoke Fixture Document",
                "mimeType": "text/plain",
            }
        ]

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Read the safe smoke fixture resource."""

        self.resource_reads.append((uri, context))
        if uri != "resource://smoke/doc":
            raise ValueError("unknown smoke fixture resource")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": "Smoke fixture resource body.",
                }
            ]
        }

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return one safe prompt descriptor."""

        return [
            {
                "name": "smoke.review",
                "description": "Ask for a concise smoke-test review.",
                "arguments": [
                    {
                        "name": "topic",
                        "description": "Topic to review.",
                        "required": False,
                    }
                ],
            }
        ]

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Return the safe smoke review prompt."""

        self.prompt_gets.append((name, dict(arguments), context))
        if name != "smoke.review":
            raise NotImplementedError(name)
        topic = str(arguments.get("topic", "smoke"))
        return {
            "description": "Ask for a concise smoke-test review.",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Review {topic}",
                    },
                }
            ],
        }

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no standalone modules for the fixture runtime."""

        return []

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Return an empty module health map for the fixture runtime."""

        return {}


__all__ = ["SmokeFixtureGatewayRuntime"]
