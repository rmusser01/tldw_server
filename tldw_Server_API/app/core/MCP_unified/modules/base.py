"""
Base module interface for unified MCP with production features

Includes health checking, metrics, circuit breaker support, and proper error handling.
"""

import asyncio
import contextlib
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, TypeVar

from loguru import logger

from mcp_unified.interfaces.path_scope import PathScopeCandidate

from ..tool_observability import ensure_tool_definition_eval_metadata

T = TypeVar("T")


class HealthStatus(str, Enum):
    """Module health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModuleHealth:
    """Module health information"""
    status: HealthStatus
    message: str = ""
    # Set to None initially so the first health_check() performs real checks
    last_check: Optional[datetime] = None
    checks: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_operational(self) -> bool:
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


@dataclass
class ModuleMetrics:
    """Module performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0
    avg_latency_ms: float = 0

    def record_request(self, success: bool, latency_ms: float):
        """Record a request"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.last_request_time = datetime.utcnow()
        self.error_rate = self.failed_requests / max(1, self.total_requests)
        self.avg_latency_ms = self.total_latency_ms / max(1, self.total_requests)


@dataclass
class ModuleConfig:
    """Module configuration"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    department: str = "general"
    enabled: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    # Concurrency guard per module (0 disables guard)
    max_concurrent: int = 20
    # Circuit breaker backoff and caps for half-open failures
    circuit_breaker_backoff_factor: float = 2.0
    circuit_breaker_max_timeout: int = 300
    settings: dict[str, Any] = field(default_factory=dict)
    circuit_breaker_factory: Any | None = None


@dataclass(frozen=True, slots=True)
class ModuleCircuitBreakerConfig:
    """Host-neutral circuit breaker settings used by MCP modules."""

    failure_threshold: int
    recovery_timeout: float
    backoff_factor: float
    max_recovery_timeout: float
    half_open_max_calls: int = 1
    success_threshold: int = 1
    category: str = "mcp"
    service: str = ""


def _build_module_circuit_breaker_config(config: ModuleConfig) -> ModuleCircuitBreakerConfig:
    return ModuleCircuitBreakerConfig(
        failure_threshold=config.circuit_breaker_threshold,
        recovery_timeout=float(config.circuit_breaker_timeout),
        backoff_factor=config.circuit_breaker_backoff_factor,
        max_recovery_timeout=float(config.circuit_breaker_max_timeout),
        half_open_max_calls=1,
        success_threshold=1,
        category="mcp",
        service=config.name,
    )


class ModuleCircuitBreakerOpenError(Exception):
    """Raised when the module fallback circuit breaker rejects a call."""

    def __init__(
        self,
        message: str,
        *,
        breaker_name: str = "",
        recovery_timeout: float = 0.0,
        failure_count: int = 0,
        recovery_at: float | None = None,
    ) -> None:
        super().__init__(message)
        self.breaker_name = breaker_name
        self.recovery_timeout = recovery_timeout
        self.failure_count = failure_count
        self.recovery_at = recovery_at


class _DefaultModuleCircuitBreaker:
    """Small host-neutral async circuit breaker used when no host factory is injected."""

    def __init__(self, *, name: str, config: ModuleCircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self._state = "closed"
        self._opened_at: float | None = None
        self._half_open_in_flight = 0
        self._current_recovery_timeout = config.recovery_timeout

    def can_attempt(self) -> bool:
        if self._state == "half_open":
            return self._half_open_in_flight < self.config.half_open_max_calls
        if self._state != "open":
            return True
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at >= self._current_recovery_timeout:
            self._state = "half_open"
            self.success_count = 0
            self._half_open_in_flight = 0
            return True
        return False

    def record_failure(self) -> None:
        self.failure_count += 1
        self.success_count = 0
        if self._state == "half_open":
            self._open(with_backoff=True)
            return
        if self.failure_count >= self.config.failure_threshold:
            self._open(with_backoff=False)

    def record_success(self) -> None:
        if self._state == "half_open":
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close()
            return
        self.failure_count = 0

    async def call_async(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation if the fallback breaker state allows an attempt."""
        if not self.can_attempt():
            raise ModuleCircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open",
                breaker_name=self.name,
                recovery_timeout=self._current_recovery_timeout,
                failure_count=self.failure_count,
                recovery_at=(
                    self._opened_at + self._current_recovery_timeout
                    if self._opened_at is not None
                    else None
                ),
            )
        half_open_probe = self._state == "half_open"
        if half_open_probe:
            self._half_open_in_flight += 1
        try:
            result = await operation()
        except Exception:
            self.record_failure()
            raise
        finally:
            if half_open_probe:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
        self.record_success()
        return result

    def _open(self, *, with_backoff: bool) -> None:
        if with_backoff:
            self._current_recovery_timeout = min(
                self._current_recovery_timeout * self.config.backoff_factor,
                self.config.max_recovery_timeout,
            )
        self._state = "open"
        self._opened_at = time.time()
        self._half_open_in_flight = 0

    def _close(self) -> None:
        self._state = "closed"
        self._opened_at = None
        self._half_open_in_flight = 0
        self.failure_count = 0
        self.success_count = 0
        self._current_recovery_timeout = self.config.recovery_timeout


def _default_circuit_breaker_factory(*, name: str, config: Any) -> Any:
    return _DefaultModuleCircuitBreaker(name=name, config=config)


def _is_circuit_breaker_open_error(exc: BaseException) -> bool:
    """Return whether an exception is a local or host circuit-open rejection."""
    return isinstance(exc, ModuleCircuitBreakerOpenError) or (
        type(exc).__name__ == "CircuitBreakerOpenError"
        and hasattr(exc, "breaker_name")
        and hasattr(exc, "recovery_timeout")
    )


class BaseModule(ABC):
    """
    Enhanced base module interface with production features.

    All modules must inherit from this class and implement required methods.
    """

    def __init__(self, config: ModuleConfig):
        self.config = config
        self.name = config.name
        self.version = config.version
        self.department = config.department
        self.enabled = config.enabled

        # Health and metrics
        self._health = ModuleHealth(status=HealthStatus.UNKNOWN)
        self._metrics = ModuleMetrics()

        # Circuit breaker (unified)
        circuit_breaker_factory = (
            config.circuit_breaker_factory
            or _default_circuit_breaker_factory
        )
        self._circuit_breaker = circuit_breaker_factory(
            name=f"mcp_{config.name}",
            config=_build_module_circuit_breaker_config(config),
        )

        # Initialization state
        self._initialized = False
        self._initializing = False
        self._shutdown = False
        self._lifecycle_lock = asyncio.Lock()

        # Tools, resources, and prompts cache
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None
        self._capability_cache_generation = 0

        # Per-module concurrency guard
        self._semaphore = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent and config.max_concurrent > 0 else None

        logger.info(f"Module created: {self.name} v{self.version}")

    async def initialize(self) -> None:
        """
        Initialize the module with error handling and health check.

        This method should not be overridden. Override on_initialize instead.
        """
        async with self._lifecycle_lock:
            if self._initialized:
                logger.warning(f"Module {self.name} already initialized")
                return

            if self._initializing:
                logger.warning(f"Module {self.name} is already initializing")
                return

            self._initializing = True
            self._shutdown = False
            logger.info(f"Initializing module: {self.name}")

            try:
                # Call module-specific initialization
                await self.on_initialize()

                self.invalidate_capability_caches()

                # Perform initial health check
                health = await self.health_check()

                if not health.is_operational:
                    raise Exception(f"Module failed health check: {health.message}")

                self._initialized = True
                logger.info(f"Module initialized successfully: {self.name}")

            except Exception as e:
                logger.error(f"Module initialization failed: {self.name} - {str(e)}")
                self._health = ModuleHealth(
                    status=HealthStatus.UNHEALTHY,
                    message="Initialization failed"
                )
                raise
            finally:
                self._initializing = False

    async def shutdown(self) -> None:
        """
        Shutdown the module gracefully.

        This method should not be overridden. Override on_shutdown instead.
        """
        async with self._lifecycle_lock:
            if self._shutdown:
                logger.warning(f"Module {self.name} already shut down")
                return

            logger.info(f"Shutting down module: {self.name}")

            try:
                # Call module-specific shutdown
                await self.on_shutdown()

                self.invalidate_capability_caches()

                self._shutdown = True
                self._initialized = False
                self._health = ModuleHealth(
                    status=HealthStatus.UNKNOWN,
                    message="Module shut down"
                )

                logger.info(f"Module shut down successfully: {self.name}")

            except Exception as e:
                logger.error(f"Module shutdown failed: {self.name} - {str(e)}")
                # Continue shutdown even if there's an error

    def invalidate_capability_caches(self) -> None:
        """Clear cached capability lists after dynamic catalog changes."""
        self._capability_cache_generation += 1
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None

    async def health_check(self) -> ModuleHealth:
        """
        Perform health check with caching.

        This method should not be overridden. Override check_health instead.
        """
        try:
            # Check if we need a new health check
            if self._health.last_check:
                time_since_check = datetime.utcnow() - self._health.last_check
                if time_since_check < timedelta(seconds=10):  # Cache for 10 seconds
                    return self._health

            # Perform health check
            checks = await self.check_health()
            now = datetime.utcnow()

            # Determine overall status
            if all(checks.values()):
                status = HealthStatus.HEALTHY
                message = "All checks passed"
            elif any(checks.values()):
                status = HealthStatus.DEGRADED
                failed = [k for k, v in checks.items() if not v]
                message = f"Some checks failed: {', '.join(failed)}"
            else:
                status = HealthStatus.UNHEALTHY
                message = "All checks failed"

            self._health = ModuleHealth(
                status=status,
                message=message,
                checks=checks,
                last_check=now,
            )

        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {str(e)}")
            self._health = ModuleHealth(
                status=HealthStatus.UNHEALTHY,
                message="Health check error",
                last_check=datetime.utcnow(),
            )

        return self._health

    def is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        return not self._circuit_breaker.can_attempt()

    def record_circuit_breaker_failure(self):
        """Record a failure for circuit breaker."""
        self._circuit_breaker.record_failure()

    def record_circuit_breaker_success(self):
        """Record a success for circuit breaker."""
        self._circuit_breaker.record_success()

    async def execute_with_circuit_breaker(self, operation, *args, **kwargs):
        """Execute an operation with circuit breaker protection.

        Delegates to the unified breaker's ``call_async`` for correct
        half-open probe slot management and exception-type filtering.
        The semaphore concurrency guard and timeout wrapping are applied
        as an inner wrapper around the operation.
        """
        start_time = time.time()

        async def _guarded_operation():
            acquired = False
            try:
                if self._semaphore is not None:
                    await self._semaphore.acquire()
                    acquired = True
                return await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(f"Operation timeout in module {self.name}")
                raise Exception(f"Operation timeout after {self.config.timeout_seconds}s") from None
            finally:
                if acquired:
                    with contextlib.suppress(Exception):
                        self._semaphore.release()

        try:
            result = await self._circuit_breaker.call_async(_guarded_operation)
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record_request(True, latency_ms)
            return result

        except Exception as e:
            if _is_circuit_breaker_open_error(e):
                raise
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record_request(False, latency_ms)
            logger.error(f"Operation failed in module {self.name}: {str(e)}")
            raise

    async def get_tool_def(self, tool_name: str) -> Optional[dict[str, Any]]:
        """Return a single tool definition, using cached tool list if available."""
        tools = self._tools_cache
        if tools is None:
            generation = self._capability_cache_generation
            tools = await self.get_tools()
            if generation == self._capability_cache_generation:
                self._tools_cache = tools
        try:
            for tool in tools or []:
                if isinstance(tool, dict) and tool.get("name") == tool_name:
                    return tool
        except Exception as tool_lookup_error:
            logger.debug("MCP module tool cache lookup failed", exc_info=tool_lookup_error)
        return None

    def get_metrics(self) -> ModuleMetrics:
        """Get module metrics"""
        return self._metrics

    # Abstract methods that modules must implement

    @abstractmethod
    async def on_initialize(self) -> None:
        """Module-specific initialization logic"""
        pass

    @abstractmethod
    async def on_shutdown(self) -> None:
        """Module-specific shutdown logic"""
        pass

    @abstractmethod
    async def check_health(self) -> dict[str, bool]:
        """
        Module-specific health checks.

        Returns:
            Dictionary of check_name -> passed (True/False)
        """
        pass

    @abstractmethod
    async def get_tools(self) -> list[dict[str, Any]]:
        """
        Get list of tools provided by this module.

        Returns:
            List of tool definitions in MCP format
        """
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Optional[Any] = None) -> Any:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Optional RequestContext with user/session/db_paths

        Returns:
            Tool execution result
        """
        pass

    # Optional methods with default implementations

    async def has_tool(self, tool_name: str) -> bool:
        """Check if module provides a tool"""
        tools = self._tools_cache
        if tools is None:
            generation = self._capability_cache_generation
            tools = await self.get_tools()
            if generation == self._capability_cache_generation:
                self._tools_cache = tools
        return any(isinstance(tool, dict) and tool.get("name") == tool_name for tool in tools or [])

    async def get_resources(self) -> list[dict[str, Any]]:
        """Get list of resources (optional)"""
        return []

    async def has_resource(self, uri: str) -> bool:
        """Check if module provides a resource"""
        resources = self._resources_cache
        if resources is None:
            generation = self._capability_cache_generation
            resources = await self.get_resources()
            if generation == self._capability_cache_generation:
                self._resources_cache = resources
        return any(isinstance(resource, dict) and resource.get("uri") == uri for resource in resources or [])

    async def read_resource(self, uri: str, context: Optional[Any] = None) -> dict[str, Any]:
        """Read a resource"""
        raise NotImplementedError(f"Resource reading not implemented for {self.name}")

    async def get_prompts(self) -> list[dict[str, Any]]:
        """Get list of prompts (optional)"""
        return []

    async def has_prompt(self, name: str) -> bool:
        """Check if module provides a prompt"""
        prompts = self._prompts_cache
        if prompts is None:
            generation = self._capability_cache_generation
            prompts = await self.get_prompts()
            if generation == self._capability_cache_generation:
                self._prompts_cache = prompts
        return any(isinstance(prompt, dict) and prompt.get("name") == name for prompt in prompts or [])

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Get a prompt with arguments"""
        raise NotImplementedError(f"Prompt not implemented for {self.name}")

    # Validation helpers

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):  # noqa: B027
        """
        Validate tool arguments against schema.

        Override this to add custom validation.
        """
        # Basic validation - check required fields
        pass

    async def extract_path_scope_candidates(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Optional[Any] = None,
    ) -> list[PathScopeCandidate]:
        """Return module-derived path/action candidates for path-scope enforcement."""

        del arguments, context
        raise NotImplementedError(f"Path scope candidate extraction not implemented for {tool_name}")

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        """
        Sanitize user input to prevent injection attacks (deep, recursive).

        This implementation recursively validates dicts/lists and inspects strings
        for common injection/control patterns. Override to add module-specific
        allowlisting or transforms. A small maximum depth guard prevents abuse.
        """
        # Depth guard
        if _depth > 20:
            raise ValueError("Input too deeply nested")

        dangerous_patterns = [
            "';",
            '";',
            "--",
            "/*",
            "*/",
            "xp_",
            "sp_",
            "\\x00",
        ]

        def _check_str(s: str) -> str:
            ls = s.lower()
            for pattern in dangerous_patterns:
                if pattern in ls:
                    raise ValueError(f"Potentially dangerous input detected: {pattern}")
            # Strip NULs and control chars
            return "".join(ch for ch in s if ch >= " " or ch == "\n")

        if isinstance(input_data, str):
            return _check_str(input_data)

        if isinstance(input_data, dict):
            return {k: self.sanitize_input(v, _depth + 1) for k, v in input_data.items()}

        if isinstance(input_data, list):
            return [self.sanitize_input(v, _depth + 1) for v in input_data]

        # Pass-through for other primitives
        return input_data

    # Shared helpers for validators
    def is_write_tool_def(self, tool_def: dict[str, Any]) -> bool:
        """Heuristic and metadata-based check for write/management tools.

        Criteria:
        - metadata write flags such as write_capable/is_write/mutates_state
        - metadata.category in {ingestion, management, write, mutation, admin}
        - or name matches keywords (ingest|update|delete|create|import)
        """
        try:
            name = str(tool_def.get("name") or "").lower()
            meta = tool_def.get("metadata") or {}
            category = (meta.get("category") or "").lower()
            write_flags: list[bool] = []
            for flag_name in ("write_capable", "is_write", "mutates_state"):
                flag_value = meta.get(flag_name)
                if isinstance(flag_value, bool):
                    write_flags.append(flag_value)
            if any(write_flags):
                return True
            if write_flags:
                return False
            if category in {"ingestion", "management", "write", "mutation", "admin"}:
                return True
            import re as _re
            return bool(_re.search(r"(ingest|update|delete|create|import)", name))
        except Exception:
            return False

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        """Determine whether a specific tool invocation should be treated as write-capable.

        Default behavior:
        - Prefer tool-definition classification when ``tool_def`` is available
        - Fallback to legacy name-based heuristic otherwise
        """
        try:
            if tool_def is not None:
                return self.is_write_tool_def(tool_def)
        except Exception:
            logger.debug(
                "Falling back to tool-name write heuristic after tool definition classification failure",
            )
        try:
            import re as _re

            return bool(_re.search(r"(ingest|update|delete|create|import)", str(tool_name).lower()))
        except Exception:
            return False


# Helper functions for creating MCP-compliant definitions

def create_tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Create MCP-compliant tool definition"""
    tool_def = {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", [])
        }
    }

    if metadata:
        tool_def["metadata"] = metadata

    return ensure_tool_definition_eval_metadata(tool_def)


def create_resource_definition(
    uri: str,
    name: str,
    description: str,
    mime_type: str = "application/json",
    metadata: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Create MCP-compliant resource definition"""
    resource_def = {
        "uri": uri,
        "name": name,
        "description": description,
        "mimeType": mime_type
    }

    if metadata:
        resource_def["metadata"] = metadata

    return resource_def


def create_prompt_definition(
    name: str,
    description: str,
    arguments: Optional[list[dict[str, Any]]] = None,
    metadata: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Create MCP-compliant prompt definition"""
    prompt_def = {
        "name": name,
        "description": description,
        "arguments": arguments or []
    }

    if metadata:
        prompt_def["metadata"] = metadata

    return prompt_def
