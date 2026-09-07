"""Compatibility re-exports for model-completion interface contracts."""

from mcp_unified.interfaces.model_completion import (
    ManagedModelCompletionPort,
    ModelCompletionCapabilities,
    ModelCompletionFailure,
    ModelCompletionPort,
    ModelCompletionPortFactory,
    ModelCompletionPortSettings,
    ModelCompletionRequest,
    ModelCompletionResult,
    ModelFailureDomain,
    ModelInvocationIdentity,
)

__all__ = [
    "ManagedModelCompletionPort",
    "ModelCompletionCapabilities",
    "ModelCompletionFailure",
    "ModelCompletionPort",
    "ModelCompletionPortFactory",
    "ModelCompletionPortSettings",
    "ModelCompletionRequest",
    "ModelCompletionResult",
    "ModelFailureDomain",
    "ModelInvocationIdentity",
]
