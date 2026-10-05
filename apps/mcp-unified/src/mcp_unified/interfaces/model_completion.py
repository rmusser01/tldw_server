"""Host-neutral contracts for bounded model completion."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, fields
from enum import Enum
from typing import Protocol

_FAILURE_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")


def _require_positive_int(value: object, *, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive non-boolean integer")


def _require_non_empty_text(value: object, *, field_name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


@dataclass(frozen=True, slots=True)
class ModelInvocationIdentity:
    """Minimized server-authenticated identity for one model invocation."""

    user_id: int
    active_team_id: int | None
    active_organization_id: int | None
    execution_id: str

    def __post_init__(self) -> None:
        _require_positive_int(self.user_id, field_name="user_id")
        if self.active_team_id is not None:
            _require_positive_int(self.active_team_id, field_name="active_team_id")
        if self.active_organization_id is not None:
            _require_positive_int(
                self.active_organization_id,
                field_name="active_organization_id",
            )
        if type(self.execution_id) is not str:
            raise ValueError("execution_id must be a canonical lowercase UUIDv4")
        try:
            parsed = uuid.UUID(self.execution_id)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                "execution_id must be a canonical lowercase UUIDv4"
            ) from None
        if parsed.version != 4 or str(parsed) != self.execution_id:
            raise ValueError("execution_id must be a canonical lowercase UUIDv4")


@dataclass(frozen=True, slots=True)
class ModelCompletionCapabilities:
    """Capabilities required from a production model-completion path."""

    native_async_cancellation: bool
    response_limit_before_decode: bool
    native_max_output_tokens: bool
    tool_suppression: bool
    automatic_retries_disabled: bool

    def __post_init__(self) -> None:
        if any(type(getattr(self, field.name)) is not bool for field in fields(self)):
            raise ValueError("capability flags must be booleans")


@dataclass(frozen=True, slots=True)
class ModelCompletionRequest:
    """Bounded prompts and output limits for one completion."""

    system_prompt: str
    user_prompt: str
    max_output_tokens: int
    max_output_chars: int
    max_output_bytes: int
    max_provider_response_bytes: int

    def __post_init__(self) -> None:
        if type(self.system_prompt) is not str:
            raise ValueError("system_prompt must be text")
        if type(self.user_prompt) is not str:
            raise ValueError("user_prompt must be text")
        for field_name in (
            "max_output_tokens",
            "max_output_chars",
            "max_output_bytes",
            "max_provider_response_bytes",
        ):
            _require_positive_int(getattr(self, field_name), field_name=field_name)


@dataclass(frozen=True, slots=True)
class ModelCompletionResult:
    """Normalized text returned by the model-completion boundary."""

    content: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.content, field_name="content")


@dataclass(frozen=True, slots=True)
class ModelCompletionPortSettings:
    """Operator-controlled values captured when a completion port is built."""

    provider: str
    model: str
    run_timeout_seconds: int
    cancellation_cleanup_seconds: int

    def __post_init__(self) -> None:
        _require_non_empty_text(self.provider, field_name="provider")
        _require_non_empty_text(self.model, field_name="model")
        _require_positive_int(
            self.run_timeout_seconds,
            field_name="run_timeout_seconds",
        )
        _require_positive_int(
            self.cancellation_cleanup_seconds,
            field_name="cancellation_cleanup_seconds",
        )


class ModelFailureDomain(str, Enum):
    """Trusted provenance used to isolate shared failures from local failures."""

    REQUEST = "request"
    CREDENTIAL_SCOPE = "credential_scope"
    SHARED_INFRASTRUCTURE = "shared_infrastructure"


class ModelCompletionFailure(RuntimeError):
    """Sanitized model-completion failure with stable trusted provenance."""

    __slots__ = ("code", "domain")

    def __init__(self, code: str, domain: ModelFailureDomain) -> None:
        if type(code) is not str or _FAILURE_CODE_PATTERN.fullmatch(code) is None:
            raise ValueError("code must be a stable failure code")
        if not isinstance(domain, ModelFailureDomain):
            raise TypeError("domain must be a ModelFailureDomain")
        self.code = code
        self.domain = domain
        super().__init__(code)


class ModelCompletionPort(Protocol):
    """Minimal completion capability consumed by an MCP module."""

    @property
    def capabilities(self) -> ModelCompletionCapabilities: ...

    async def complete(
        self,
        request: ModelCompletionRequest,
        identity: ModelInvocationIdentity,
    ) -> ModelCompletionResult: ...


class ManagedModelCompletionPort(ModelCompletionPort, Protocol):
    """Completion port with explicit health and lifecycle ownership."""

    def is_healthy(self) -> bool: ...

    async def shutdown(self) -> None: ...

    async def wait_for_shutdown_completion(self) -> None: ...


class ModelCompletionPortFactory(Protocol):
    """Build a managed completion port from frozen operator settings."""

    def __call__(
        self,
        settings: ModelCompletionPortSettings,
    ) -> ManagedModelCompletionPort: ...


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
