from __future__ import annotations

from tldw_Server_API.app.core.MCP_unified.command_runtime.registry import build_default_registry


def test_registry_hides_commands_without_visible_backing_tools():
    registry = build_default_registry()
    visible = registry.visible_commands(allowed_tools={"fs.list", "mcp.tools.list"})

    assert "ls" in visible
    assert "grep" in visible
    assert "head" in visible
    assert "tail" in visible
    assert "json" in visible
    assert "mcp" in visible
    assert "cat" not in visible
    assert "write" not in visible
    assert "knowledge" not in visible
    assert "media" not in visible
    assert "sandbox" not in visible
    assert "stat" not in visible
    assert "glob" not in visible
    assert "find" not in visible
    assert "rg" not in visible
    assert "grep-files" not in visible


def test_registry_exposes_filesystem_aliases_only_when_backing_tools_are_visible() -> None:
    registry = build_default_registry()
    visible = registry.visible_commands(allowed_tools={"fs.stat", "fs.grep"})

    assert "stat" in visible
    assert "rg" in visible
    assert "grep-files" in visible
    assert "grep" in visible
    assert "glob" not in visible
    assert "find" not in visible


def test_registry_exposes_phase_one_mappings():
    registry = build_default_registry()

    assert registry.get_command("ls").backend_tools == ("fs.list",)
    assert registry.get_command("cat").backend_tools == ("fs.read", "fs.read_text")
    assert registry.get_command("write").backend_tools == ("fs.write_text",)
    assert registry.get_command("write-create").backend_tools == ("fs.write",)
    assert registry.get_command("knowledge").backend_tools == ("knowledge.search", "knowledge.get")
    assert registry.get_command("media").backend_tools == ("media.search", "media.get")
    assert registry.get_command("mcp").backend_tools == (
        "mcp.modules.list",
        "mcp.tools.list",
        "mcp.catalogs.list",
    )
    assert registry.get_command("sandbox").backend_tools == ("sandbox.run",)
    assert registry.get_command("grep").pure_transform is True
    assert registry.get_command("stat").backend_tools == ("fs.stat",)
    assert registry.get_command("glob").backend_tools == ("fs.glob",)
    assert registry.get_command("find").backend_tools == ("fs.glob",)
    assert registry.get_command("rg").backend_tools == ("fs.grep",)
    assert registry.get_command("grep-files").backend_tools == ("fs.grep",)


def test_registry_filters_visible_backend_tools_for_multi_backend_commands() -> None:
    registry = build_default_registry()

    visible = registry.visible_commands(allowed_tools={"fs.read"})
    assert "cat" in visible
    assert visible["cat"].backend_tools == ("fs.read",)

    visible = registry.visible_commands(allowed_tools={"fs.read_text"})
    assert "cat" in visible
    assert visible["cat"].backend_tools == ("fs.read_text",)

    visible = registry.visible_commands(allowed_tools={"fs.write"})
    assert "write-create" in visible
    assert "write" not in visible


def test_registry_filters_visible_backend_tools_for_mcp_catalogs() -> None:
    registry = build_default_registry()

    visible = registry.visible_commands(allowed_tools={"mcp.catalogs.list"})

    assert "mcp" in visible
    assert visible["mcp"].backend_tools == ("mcp.catalogs.list",)
