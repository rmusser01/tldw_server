"""
Lint/test gate: Write-capable tools must set metadata.category to 'ingestion' or 'management'.

This complements the validator override guard by ensuring module authors
explicitly categorize write tools for rate limiting and policy enforcement.
"""

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module import MediaModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.notes_module import NotesModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module import PromptsModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module import KnowledgeModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.characters_module import CharactersModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.chats_module import ChatsModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.template_module import TemplateModule


class _MinimalModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):  # noqa: ANN001
        return None


@pytest.mark.asyncio
async def test_write_tools_have_ingestion_or_management_category():
    modules = [
        MediaModule(ModuleConfig(name="media")),
        NotesModule(ModuleConfig(name="notes")),
        PromptsModule(ModuleConfig(name="prompts")),
        KnowledgeModule(ModuleConfig(name="knowledge")),
        CharactersModule(ModuleConfig(name="characters")),
        ChatsModule(ModuleConfig(name="chats")),
        TemplateModule(ModuleConfig(name="template")),
    ]

    violations = []
    for mod in modules:
        tools = await mod.get_tools()
        for tool in tools:
            # Use shared helper to determine write-capable status
            if mod.is_write_tool_def(tool):
                meta = (tool.get("metadata") or {}) if isinstance(tool, dict) else {}
                category = str(meta.get("category") or "").lower()
                if category not in {"ingestion", "management"}:
                    violations.append((mod.name, tool.get("name"), category))

    assert not violations, (
        "Write-capable tools must set metadata.category to 'ingestion' or 'management':\n" +
        "\n".join(f"module={m}, tool={t}, category='{c or 'missing'}'" for m, t, c in violations)
    )


def test_write_classification_conflicting_flags_prefers_write_capable():
    module = _MinimalModule(ModuleConfig(name="minimal"))

    assert module.is_write_tool_def(
        {
            "name": "read.report",
            "metadata": {
                "write_capable": False,
                "is_write": True,
                "mutates_state": False,
                "category": "read",
            },
        }
    ) is True


def test_write_classification_all_false_flags_skip_write_category_fallback():
    module = _MinimalModule(ModuleConfig(name="minimal"))

    assert module.is_write_tool_def(
        {
            "name": "read.report",
            "metadata": {
                "write_capable": False,
                "is_write": False,
                "mutates_state": False,
                "category": "management",
            },
        }
    ) is False
