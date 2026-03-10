"""Domain models for managed vLLM instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class VLLMInstanceCreate:
    """Creation payload for a managed vLLM instance."""

    name: str
    execution_mode: str
    transport_config: dict[str, Any] = field(default_factory=dict)
    launch_spec: dict[str, Any] = field(default_factory=dict)
    routing_policy: dict[str, Any] = field(default_factory=dict)
    declared_capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VLLMInstanceRecord:
    """Persisted managed vLLM instance state."""

    instance_id: str
    name: str
    execution_mode: str
    transport_config: dict[str, Any]
    launch_spec: dict[str, Any]
    routing_policy: dict[str, Any]
    declared_capabilities: dict[str, Any]
    desired_state: str
    observed_state: str
    created_at: str
    updated_at: str
    probed_capabilities: dict[str, Any] = field(default_factory=dict)
    effective_capabilities: dict[str, Any] = field(default_factory=dict)
    last_known_base_url: str | None = None
    last_error: str | None = None
    executor_handle: dict[str, Any] = field(default_factory=dict)
