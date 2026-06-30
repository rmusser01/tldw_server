from __future__ import annotations

from typing import Any

import pytest
from mcp_unified.federation.models import VirtualExternalTool

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.external_federation_module import (
    ExternalFederationModule,
)


class _WriteFlagOnlyManager:
    def __init__(self) -> None:
        self.lookups: list[str] = []

    def get_virtual_tool_write_flag(self, virtual_tool_name: str) -> bool | None:
        self.lookups.append(virtual_tool_name)
        if virtual_tool_name == "ext.docs.docs.update":
            return True
        if virtual_tool_name == "ext.docs.docs.search":
            return False
        return None

    def list_virtual_tools(self) -> list[Any]:
        raise AssertionError("is_write_tool_call should not copy the virtual tool catalog")


def test_external_federation_write_classification_uses_scalar_manager_lookup() -> None:
    module = ExternalFederationModule(ModuleConfig(name="external_federation"))
    manager = _WriteFlagOnlyManager()
    module._manager = manager  # noqa: SLF001 - focused module wiring test.

    assert module.is_write_tool_call("ext.docs.docs.update", {}) is True  # nosec B101
    assert module.is_write_tool_call("ext.docs.docs.search", {}) is False  # nosec B101
    assert manager.lookups == ["ext.docs.docs.update", "ext.docs.docs.search"]  # nosec B101


class _VirtualToolListManager:
    def list_virtual_tools(self) -> list[VirtualExternalTool]:
        return [
            VirtualExternalTool(
                virtual_name="ext.docs.docs.search",
                server_id="docs",
                upstream_tool_name="docs.search",
                description="Search external docs.",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                metadata={"source": "unit-test"},
                is_write=False,
            )
        ]


class _VirtualToolWithUpstreamEvalManager:
    def list_virtual_tools(self) -> list[VirtualExternalTool]:
        return [
            VirtualExternalTool(
                virtual_name="ext.docs.docs.search",
                server_id="docs",
                upstream_tool_name="docs.search",
                description="Search external docs.",
                input_schema={"type": "object"},
                metadata={
                    "eval": {
                        "tool_prompt_id": "mcp.upstream.untrusted.v1",
                        "tool_prompt_version": "upstream",
                        "task_families": ["raw"],
                        "expected_result_kind": "raw_payload",
                        "success_signals": ["raw_output"],
                    }
                },
                is_write=False,
            )
        ]


@pytest.mark.asyncio
async def test_external_federation_virtual_tools_include_eval_metadata() -> None:
    module = ExternalFederationModule(ModuleConfig(name="external_federation"))
    module._manager = _VirtualToolListManager()  # noqa: SLF001 - focused module wiring test.

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}
    metadata = by_name["ext.docs.docs.search"]["metadata"]
    eval_metadata = metadata["eval"]

    assert metadata["federated"] is True  # nosec B101
    assert metadata["server_id"] == "docs"  # nosec B101
    assert metadata["upstream_tool"] == "docs.search"  # nosec B101
    assert eval_metadata["tool_prompt_id"] == "mcp.ext.docs.docs.search.v1"  # nosec B101
    assert eval_metadata["task_families"] == ["external"]  # nosec B101
    assert eval_metadata["expected_result_kind"] == "external_result"  # nosec B101
    assert eval_metadata["prompt_variant"] == "external_federated"  # nosec B101


@pytest.mark.asyncio
async def test_external_federation_ignores_upstream_eval_metadata() -> None:
    module = ExternalFederationModule(ModuleConfig(name="external_federation"))
    module._manager = _VirtualToolWithUpstreamEvalManager()  # noqa: SLF001 - focused module wiring test.

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}
    eval_metadata = by_name["ext.docs.docs.search"]["metadata"]["eval"]

    assert eval_metadata["tool_prompt_id"] == "mcp.ext.docs.docs.search.v1"  # nosec B101
    assert eval_metadata["tool_prompt_version"] != "upstream"  # nosec B101
    assert eval_metadata["task_families"] == ["external"]  # nosec B101
    assert eval_metadata["prompt_variant"] == "external_federated"  # nosec B101
