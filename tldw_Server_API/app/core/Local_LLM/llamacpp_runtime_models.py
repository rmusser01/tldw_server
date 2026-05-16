"""Runtime models for managed llama.cpp profiles."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LlamaCppProfileMode(str, Enum):
    CHAT = "chat"
    VISION = "vision"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    SERVER_GENERIC = "server_generic"


class LlamaCppPortPolicy(str, Enum):
    EXPLICIT = "explicit"
    AUTOSELECT = "autoselect"


class LlamaCppRuntimeState(str, Enum):
    DEFINED = "defined"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PAUSED = "paused"


class LlamaCppProfile(BaseModel):
    """Durable managed llama.cpp runtime profile."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    enabled: bool = True
    mode: LlamaCppProfileMode = LlamaCppProfileMode.CHAT
    model_id: str | None = None
    model_path: str | None = None
    mmproj_model_id: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    port_policy: LlamaCppPortPolicy = LlamaCppPortPolicy.EXPLICIT
    server_args: dict[str, object] = Field(default_factory=dict)
    autostart: bool = False
    restart_policy: dict[str, object] = Field(default_factory=dict)
    provider_alias: str | None = None
    tags: list[str] = Field(default_factory=list)


class LlamaCppRuntime(BaseModel):
    """Observed runtime state for one managed llama.cpp profile."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    state: LlamaCppRuntimeState
    pid: int | None = None
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    endpoint: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    resolved_args: list[str] = Field(default_factory=list)
    started_at: str | None = None
    stopped_at: str | None = None
    last_health_at: str | None = None
    restart_count: int = Field(default=0, ge=0)
    next_restart_at: str | None = None
    exit_code: int | None = None
    last_error: str | None = None
    log_tail_available: bool = False
    log_file: str | None = None
    command: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    health: dict[str, object] = Field(default_factory=dict)
    message: str | None = None


class LlamaCppProfileStoreError(RuntimeError):
    """Base error for managed llama.cpp profile persistence."""


class LlamaCppProfileNotFoundError(LlamaCppProfileStoreError):
    """Raised when a requested managed llama.cpp profile does not exist."""


class LlamaCppProfileConflictError(LlamaCppProfileStoreError):
    """Raised when managed llama.cpp profiles have conflicting settings."""
