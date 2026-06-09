"""Policy-aware registry for virtual CLI commands."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final


@dataclass(frozen=True, slots=True)
class CommandDescriptor:
    """Describe a virtual CLI command and the tools it depends on."""

    name: str
    summary: str
    backend_tools: tuple[str, ...]
    pure_transform: bool = False


@dataclass(slots=True)
class CommandRegistry:
    """Registry of command descriptors with policy-aware visibility."""

    _commands: dict[str, CommandDescriptor] = field(default_factory=dict)

    def register(self, descriptor: CommandDescriptor) -> None:
        self._commands[descriptor.name] = descriptor

    def get_command(self, name: str) -> CommandDescriptor:
        try:
            return self._commands[name]
        except KeyError as exc:
            raise KeyError(f"Unknown command: {name}") from exc

    def visible_commands(self, allowed_tools: set[str]) -> dict[str, CommandDescriptor]:
        visible: dict[str, CommandDescriptor] = {}
        for name, descriptor in self._commands.items():
            if descriptor.pure_transform:
                visible[name] = descriptor
                continue
            visible_backend_tools = tuple(tool for tool in descriptor.backend_tools if tool in allowed_tools)
            if visible_backend_tools:
                visible[name] = replace(descriptor, backend_tools=visible_backend_tools)
        return visible


def build_default_registry() -> CommandRegistry:
    """Build the phase-1 command registry."""

    registry = CommandRegistry()
    for descriptor in _DEFAULT_COMMANDS:
        registry.register(descriptor)
    return registry


_DEFAULT_COMMANDS: Final[tuple[CommandDescriptor, ...]] = (
    CommandDescriptor(
        name="ls",
        summary="List files in the current workspace scope.",
        backend_tools=("fs.list",),
    ),
    CommandDescriptor(
        name="cat",
        summary="Read a UTF-8 text file from the current workspace scope.",
        backend_tools=("fs.read", "fs.read_text"),
    ),
    CommandDescriptor(
        name="write",
        summary="Write a UTF-8 text file in the current workspace scope.",
        backend_tools=("fs.write_text",),
    ),
    CommandDescriptor(
        name="write-create",
        summary="Create a UTF-8 text file in the current workspace scope.",
        backend_tools=("fs.write",),
    ),
    CommandDescriptor(
        name="grep",
        summary="Filter lines matching a pattern.",
        backend_tools=(),
        pure_transform=True,
    ),
    CommandDescriptor(
        name="stat",
        summary="Inspect workspace file metadata.",
        backend_tools=("fs.stat",),
    ),
    CommandDescriptor(
        name="glob",
        summary="Find workspace paths matching a portable pattern.",
        backend_tools=("fs.glob",),
    ),
    CommandDescriptor(
        name="find",
        summary="Alias for glob over workspace paths.",
        backend_tools=("fs.glob",),
    ),
    CommandDescriptor(
        name="rg",
        summary="Search workspace UTF-8 text files.",
        backend_tools=("fs.grep",),
    ),
    CommandDescriptor(
        name="grep-files",
        summary="Alias for governed workspace file search.",
        backend_tools=("fs.grep",),
    ),
    CommandDescriptor(
        name="head",
        summary="Keep the first lines of input.",
        backend_tools=(),
        pure_transform=True,
    ),
    CommandDescriptor(
        name="tail",
        summary="Keep the last lines of input.",
        backend_tools=(),
        pure_transform=True,
    ),
    CommandDescriptor(
        name="json",
        summary="Extract data from JSON input.",
        backend_tools=(),
        pure_transform=True,
    ),
    CommandDescriptor(
        name="knowledge",
        summary="Search or fetch knowledge records.",
        backend_tools=("knowledge.search", "knowledge.get"),
    ),
    CommandDescriptor(
        name="media",
        summary="Search or fetch media records.",
        backend_tools=("media.search", "media.get"),
    ),
    CommandDescriptor(
        name="mcp",
        summary="Inspect visible MCP modules and tools.",
        backend_tools=("mcp.modules.list", "mcp.tools.list", "mcp.catalogs.list"),
    ),
    CommandDescriptor(
        name="sandbox",
        summary="Execute a governed sandbox command.",
        backend_tools=("sandbox.run",),
    ),
)
