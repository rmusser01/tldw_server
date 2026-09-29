"""Base types and protocols for workflow adapters.

This module defines the base Pydantic models and type aliases used by adapters.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class AdapterContext(BaseModel):
    """Standard context passed to all adapters.

    This model defines the expected structure of the context dict passed to adapters.
    Adapters can access these fields for user/tenant info, previous step outputs, etc.
    """

    user_id: str | None = None
    tenant_id: str | None = None
    run_id: str | None = None
    step_run_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    prev: dict[str, Any] = Field(default_factory=dict)
    last: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    step_capability: dict[str, Any] = Field(default_factory=dict)
    workflow_metadata: dict[str, Any] = Field(default_factory=dict)
    workflow_mcp_policy: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")  # Allow is_cancelled, add_artifact, etc.


class BaseAdapterConfig(BaseModel):
    """Base config model for adapters.

    Adapters can extend this with specific fields. The base config allows
    extra fields to support forward compatibility.
    """

    timeout_seconds: int | None = Field(None, description="Step timeout in seconds")
    save_artifact: bool | None = Field(False, description="Whether to save output as artifact")

    model_config = ConfigDict(extra="allow")  # Allow additional fields


# Type aliases for adapter signatures
AdapterFunc = Callable[[dict[str, Any], dict[str, Any]], Any]
"""Type alias for adapter function signature: async (config, context) -> result"""

AdapterResult = dict[str, Any]
"""Type alias for adapter return value"""
