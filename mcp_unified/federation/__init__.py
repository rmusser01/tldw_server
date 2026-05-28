"""Standalone non-spawning external federation shell."""

from .manager import ExternalFederationManager
from .models import (
    ExternalToolCallResult,
    ExternalToolDefinition,
    FederatedToolResult,
    FederationPolicyDenied,
    VirtualExternalTool,
)
from .transports import ExternalFederationTransport, FakeExternalTransport

__all__ = [
    "ExternalFederationManager",
    "ExternalFederationTransport",
    "ExternalToolCallResult",
    "ExternalToolDefinition",
    "FakeExternalTransport",
    "FederatedToolResult",
    "FederationPolicyDenied",
    "VirtualExternalTool",
]
