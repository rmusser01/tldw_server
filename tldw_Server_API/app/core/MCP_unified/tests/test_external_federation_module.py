from __future__ import annotations

from typing import Any

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
