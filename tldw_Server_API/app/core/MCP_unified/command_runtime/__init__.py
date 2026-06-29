"""Virtual CLI command runtime for MCP Unified."""

from .executor import CommandBackend, CommandRuntimeExecutor
from .models import (
    CommandChain,
    CommandExecutionResult,
    CommandExecutionStep,
    CommandInvocation,
    CommandSpillReference,
    CommandStepResult,
    Pipeline,
)
from .parser import parse_command
from .presentation import present_command_execution_result
from .registry import CommandDescriptor, CommandRegistry, build_default_registry

__all__ = [
    "CommandBackend",
    "CommandChain",
    "CommandDescriptor",
    "CommandExecutionResult",
    "CommandExecutionStep",
    "CommandInvocation",
    "CommandRegistry",
    "CommandRuntimeExecutor",
    "Pipeline",
    "CommandSpillReference",
    "CommandStepResult",
    "build_default_registry",
    "parse_command",
    "present_command_execution_result",
]
