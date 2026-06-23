"""Schema and loader helpers for external MCP federation config.

This module defines a conservative configuration model for external MCP servers.
It is intentionally strict about transport wiring and defaults to a safe posture
(read-oriented, explicit write enablement).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

try:
    from pydantic import field_validator  # v2
except (ImportError, AttributeError):  # pragma: no cover - v1 fallback
    from pydantic import validator as field_validator  # type: ignore

_DEFAULT_CONFIG_PATH = "mcp_external_servers.yaml"


class ExternalTransportType(str, Enum):
    """Supported external transport modes."""

    WEBSOCKET = "websocket"
    STDIO = "stdio"


class ExternalAuthMode(str, Enum):
    """Auth behavior for upstream external MCP calls."""

    NONE = "none"
    BEARER_ENV = "bearer_env"
    API_KEY_ENV = "api_key_env"


class ExternalTimeoutConfig(BaseModel):
    """Timeout controls for external server interactions."""

    connect_seconds: float = Field(default=10.0, ge=0.1)
    request_seconds: float = Field(default=30.0, ge=0.1)


class ExternalRetryConfig(BaseModel):
    """Retry controls for transient external server failures."""

    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)


class ExternalCircuitBreakerConfig(BaseModel):
    """Per-server circuit breaker defaults."""

    failure_threshold: int = Field(default=5, ge=1, le=100)
    recovery_timeout_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)


class ExternalWebSocketConfig(BaseModel):
    """Configuration for websocket-connected external MCP servers."""

    url: str = Field(..., min_length=1)
    subprotocols: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)


class ExternalStdioConfig(BaseModel):
    """Configuration for stdio-connected external MCP servers."""

    command: str = Field(..., min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = Field(default=None)


class ExternalAuthConfig(BaseModel):
    """Authentication indirection for external server access."""

    mode: ExternalAuthMode = Field(default=ExternalAuthMode.NONE)
    token_env: str | None = Field(default=None)
    api_key_env: str | None = Field(default=None)
    api_key_header: str = Field(default="X-API-KEY")

    def resolve_headers(self) -> dict[str, str]:
        """Resolve auth headers from environment variables.

        Missing env values are tolerated here; callers may decide to reject
        execution if upstream auth material is required.
        """

        if self.mode == ExternalAuthMode.NONE:
            return {}

        if self.mode == ExternalAuthMode.BEARER_ENV:
            token = os.getenv(self.token_env or "", "").strip()
            return {"Authorization": f"Bearer {token}"} if token else {}

        if self.mode == ExternalAuthMode.API_KEY_ENV:
            api_key = os.getenv(self.api_key_env or "", "").strip()
            return {self.api_key_header: api_key} if api_key else {}

        return {}


class ExternalToolPolicy(BaseModel):
    """Local policy enforcement before forwarding to external servers."""

    allow_tool_patterns: list[str] = Field(default_factory=list)
    deny_tool_patterns: list[str] = Field(default_factory=list)
    allow_writes: bool = Field(default=False)
    require_write_confirmation: bool = Field(default=True)

    def allows_tool(self, tool_name: str) -> bool:
        """Return True if a tool passes allowlist/denylist checks."""

        if any(fnmatch(tool_name, pat) for pat in self.deny_tool_patterns):
            return False

        if not self.allow_tool_patterns:
            return True

        return any(fnmatch(tool_name, pat) for pat in self.allow_tool_patterns)


class ExternalMCPServerConfig(BaseModel):
    """Single external MCP server definition."""

    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=256)
    enabled: bool = Field(default=True)
    transport: ExternalTransportType
    websocket: ExternalWebSocketConfig | None = None
    stdio: ExternalStdioConfig | None = None
    auth: ExternalAuthConfig = Field(default_factory=ExternalAuthConfig)
    policy: ExternalToolPolicy = Field(default_factory=ExternalToolPolicy)
    timeouts: ExternalTimeoutConfig = Field(default_factory=ExternalTimeoutConfig)
    retries: ExternalRetryConfig = Field(default_factory=ExternalRetryConfig)
    circuit_breaker: ExternalCircuitBreakerConfig = Field(
        default_factory=ExternalCircuitBreakerConfig
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        """Constrain IDs for safe namespacing in tool names."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("server id cannot be empty")

        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
        lowered = cleaned.lower()
        if any(ch not in allowed for ch in lowered):
            raise ValueError("server id must use [a-z0-9_-]")
        return lowered

    def validate_transport_requirements(self) -> None:
        """Validate cross-field transport requirements."""

        if self.transport == ExternalTransportType.WEBSOCKET and self.websocket is None:
            raise ValueError(
                f"External server '{self.id}' requires websocket config when transport=websocket"
            )
        if self.transport == ExternalTransportType.STDIO and self.stdio is None:
            raise ValueError(
                f"External server '{self.id}' requires stdio config when transport=stdio"
            )


class ExternalServerRegistryConfig(BaseModel):
    """Top-level external MCP server registry configuration."""

    servers: list[ExternalMCPServerConfig] = Field(default_factory=list)


@dataclass(slots=True)
class ExternalServerRegistryPartialLoadError(RuntimeError):
    """Raised when a registry loader can return only part of the runtime config."""

    message: str
    servers: list[ExternalMCPServerConfig]
    errors: dict[str, str]

    def __str__(self) -> str:
        return self.message


def _model_validate(model_cls: type[BaseModel], payload: Any) -> BaseModel:
    """Validate payload using either Pydantic v2 or v1 API."""

    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    return model_cls.parse_obj(payload)


def parse_external_server_registry(payload: dict[str, Any]) -> ExternalServerRegistryConfig:
    """Parse and validate registry config from an in-memory payload."""

    cfg = _model_validate(ExternalServerRegistryConfig, payload)

    seen_ids: set[str] = set()
    for server in cfg.servers:
        server.validate_transport_requirements()
        if server.id in seen_ids:
            raise ValueError(f"Duplicate external server id: {server.id}")
        seen_ids.add(server.id)

    return cfg


def load_external_server_registry(
    config_path: str | None = None,
    *,
    default_config_path: str | None = None,
) -> ExternalServerRegistryConfig:
    """Load and validate external server registry config from YAML or JSON file.

    If the file does not exist, returns an empty config (safe default). Hosts can
    pass ``default_config_path`` to preserve a deployment-specific legacy path.
    """

    env_path = os.getenv("MCP_EXTERNAL_SERVERS_CONFIG")
    path = config_path or env_path or default_config_path or _DEFAULT_CONFIG_PATH
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.info("External MCP config not found at {}; using empty registry", cfg_path)
        return ExternalServerRegistryConfig()
    if not cfg_path.is_file():
        logger.warning(
            "External MCP config path is not a file at {}; using empty registry",
            cfg_path,
        )
        return ExternalServerRegistryConfig()

    suffix = cfg_path.suffix.lower()
    raw: dict[str, Any]

    if suffix == ".json":
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "PyYAML is required to load external MCP YAML config"
            ) from exc
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if data is None:
            raw = {}
        elif isinstance(data, dict):
            raw = data
        else:
            raise ValueError(
                "External MCP config at "
                f"{cfg_path} must be a mapping root; got "
                f"{type(data).__name__}: {data!r}"
            )

    cfg = parse_external_server_registry(raw)
    logger.info(
        "Loaded {} external MCP server definitions from {}",
        len(cfg.servers),
        cfg_path,
    )
    return cfg


__all__ = [
    "ExternalAuthConfig",
    "ExternalAuthMode",
    "ExternalCircuitBreakerConfig",
    "ExternalMCPServerConfig",
    "ExternalRetryConfig",
    "ExternalServerRegistryConfig",
    "ExternalServerRegistryPartialLoadError",
    "ExternalStdioConfig",
    "ExternalTimeoutConfig",
    "ExternalToolPolicy",
    "ExternalTransportType",
    "ExternalWebSocketConfig",
    "load_external_server_registry",
    "parse_external_server_registry",
]
