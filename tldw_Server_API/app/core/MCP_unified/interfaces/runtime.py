"""Compatibility re-exports for MCP Unified runtime interfaces."""

from mcp_unified.interfaces.runtime import (
    ApiKeyScopeNormalizer,
    AuthenticatedIdentity,
    CircuitBreakerFactory,
    DatabasePathResolver,
    LifecycleGuard,
    MCPRuntimeDependencies,
    MetricsCollector,
    ModuleConfigProvider,
    ModuleRegistry,
    PermissionSeeder,
    PolicyContextProvider,
    RateLimiter,
    RbacPolicy,
    RedisClientFactory,
    ServerAuthProvider,
    TelemetryProvider,
)

__all__ = [
    "AuthenticatedIdentity",
    "ApiKeyScopeNormalizer",
    "CircuitBreakerFactory",
    "DatabasePathResolver",
    "LifecycleGuard",
    "MCPRuntimeDependencies",
    "MetricsCollector",
    "ModuleConfigProvider",
    "ModuleRegistry",
    "PermissionSeeder",
    "PolicyContextProvider",
    "RateLimiter",
    "RbacPolicy",
    "RedisClientFactory",
    "ServerAuthProvider",
    "TelemetryProvider",
]
