"""Deterministic fixture runtimes for MCP smoke tests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext

_ARTIFACT_TOOL_NAMES = frozenset(
    {"artifact.read", "artifact.summarize", "artifact.write", "artifact.stat"}
)


class SmokeFixtureGatewayRuntime:
    """Small safe gateway runtime used by in-process smoke tests."""

    name = "smoke-fixture-gateway"
    version = "0.0-test"

    def __init__(
        self,
        *,
        include_denied_tool: bool = False,
        denied_tool_name: str = "smoke.denied",
        artifact_root: str | Path | None = None,
    ) -> None:
        self.denied_tool_name = denied_tool_name if include_denied_tool else None
        self.artifact_root = _coerce_artifact_root(artifact_root)
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
        if self.artifact_root is not None:
            tools.extend(_artifact_tool_descriptors())
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
        if name in _ARTIFACT_TOOL_NAMES:
            return self._call_artifact_tool(name, arguments)
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

    def _call_artifact_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch an artifact tool call against the configured artifact root."""
        if self.artifact_root is None:
            raise NotImplementedError(name)
        if name == "artifact.read":
            return self._artifact_read(arguments)
        if name == "artifact.summarize":
            return self._artifact_summarize(arguments)
        if name == "artifact.write":
            return self._artifact_write(arguments)
        if name == "artifact.stat":
            return self._artifact_stat(arguments)
        raise NotImplementedError(name)

    def _artifact_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Read a UTF-8 artifact and return MCP text plus structured metadata."""
        relative_path, target = self._artifact_target(arguments.get("path"))
        body = target.read_text(encoding="utf-8")
        digest = _sha256_text(body)
        return {
            "content": [
                {
                    "type": "text",
                    "text": body,
                }
            ],
            "structuredContent": {
                "path": relative_path,
                "bytes": len(body.encode("utf-8")),
                "sha256": digest,
                "text": body,
            },
        }

    def _artifact_summarize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return a deterministic summary for a UTF-8 artifact."""
        source_path = arguments.get("source_path", arguments.get("path"))
        relative_path, target = self._artifact_target(source_path)
        body = target.read_text(encoding="utf-8")
        summary = _summarize_artifact(body, relative_path)
        return {
            "content": [
                {
                    "type": "text",
                    "text": summary,
                }
            ],
            "structuredContent": {
                "source_path": relative_path,
                "summary_markdown": summary,
                "bytes": len(summary.encode("utf-8")),
                "sha256": _sha256_text(summary),
            },
        }

    def _artifact_write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Write a UTF-8 artifact and return path, size, and digest metadata."""
        relative_path, target = self._artifact_target(arguments.get("path"))
        content = arguments.get("content", arguments.get("text", ""))
        if not isinstance(content, str):
            raise ValueError("artifact_content_must_be_text")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"wrote {relative_path}",
                }
            ],
            "structuredContent": {
                "path": relative_path,
                "bytes": len(content.encode("utf-8")),
                "sha256": _sha256_text(content),
            },
        }

    def _artifact_stat(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return existence, size, and optional digest metadata for an artifact."""
        relative_path, target = self._artifact_target(arguments.get("path"))
        exists = target.exists()
        size = target.stat().st_size if exists else 0
        digest = _sha256_bytes(target.read_bytes()) if exists and target.is_file() else None
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{relative_path}: {'exists' if exists else 'missing'}",
                }
            ],
            "structuredContent": {
                "path": relative_path,
                "exists": exists,
                "bytes": size,
                "sha256": digest,
            },
        }

    def _artifact_target(self, path_value: object) -> tuple[str, Path]:
        """Resolve an artifact-relative path and reject root escapes."""
        if self.artifact_root is None:
            raise ValueError("artifact_root_not_configured")
        relative_path = _normalize_relative_artifact_path(path_value)
        target = (self.artifact_root / relative_path).resolve()
        try:
            target.relative_to(self.artifact_root)
        except ValueError as exc:
            raise ValueError("artifact_path_denied") from exc
        return relative_path, target


def _coerce_artifact_root(value: str | Path | None) -> Path | None:
    configured = value
    if configured is None:
        configured = os.environ.get("MCP_SMOKE_ARTIFACT_ROOT")
    if configured is None or str(configured).strip() == "":
        return None
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_tool_descriptors() -> list[dict[str, Any]]:
    path_schema = {"type": "string", "minLength": 1}
    return [
        {
            "name": "artifact.read",
            "description": "Read a smoke artifact from the configured artifact root.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": path_schema},
                "required": ["path"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "artifact.summarize",
            "description": "Create a deterministic summary from a smoke artifact.",
            "inputSchema": {
                "type": "object",
                "properties": {"source_path": path_schema},
                "required": ["source_path"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "artifact.write",
            "description": "Write a derived smoke artifact under the artifact root.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": path_schema,
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "name": "artifact.stat",
            "description": "Return metadata for a smoke artifact under the artifact root.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": path_schema},
                "required": ["path"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        },
    ]


def _normalize_relative_artifact_path(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("artifact_path_required")
    candidate = Path(path_value)
    if candidate.is_absolute():
        raise ValueError("artifact_path_denied")
    parts = candidate.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact_path_denied")
    return candidate.as_posix()


def _summarize_artifact(body: str, relative_path: str) -> str:
    words = [word.strip(".,:;!?()[]{}\"'").lower() for word in body.split()]
    unique_words = len({word for word in words if word})
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    if first_line.startswith("#"):
        first_line = first_line.lstrip("#").strip()
    title = first_line or "Untitled artifact"
    return (
        "# Smoke UAT Derived Artifact\n\n"
        f"- Source: {relative_path}\n"
        f"- Title: {title}\n"
        f"- Word count: {len(words)}\n"
        f"- Unique words: {unique_words}\n"
    )


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = ["SmokeFixtureGatewayRuntime"]
