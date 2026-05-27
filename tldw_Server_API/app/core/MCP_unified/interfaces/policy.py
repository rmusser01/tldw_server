"""Compatibility re-exports for MCP Unified policy interfaces."""

from mcp_unified.interfaces.policy import (
    ApprovalEvaluator,
    EffectivePolicyResolver,
    ExternalAccessEvaluator,
    PathScopeEnforcer,
)

__all__ = [
    "ApprovalEvaluator",
    "EffectivePolicyResolver",
    "ExternalAccessEvaluator",
    "PathScopeEnforcer",
]
