from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceCreate


class VLLMInstanceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Display name for the managed vLLM instance")
    execution_mode: Literal["local", "ssh"] = Field(..., description="Where the managed runtime executes")
    transport_config: dict[str, Any] = Field(default_factory=dict)
    launch_spec: dict[str, Any] = Field(default_factory=dict)
    routing_policy: dict[str, Any] = Field(default_factory=dict)
    declared_capabilities: dict[str, bool] = Field(default_factory=dict)

    def to_domain(self) -> VLLMInstanceCreate:
        return VLLMInstanceCreate(
            name=self.name.strip(),
            execution_mode=self.execution_mode,
            transport_config=dict(self.transport_config),
            launch_spec=dict(self.launch_spec),
            routing_policy=dict(self.routing_policy),
            declared_capabilities=dict(self.declared_capabilities),
        )


class VLLMInstanceUpdateRequest(BaseModel):
    name: str | None = None
    execution_mode: Literal["local", "ssh"] | None = None
    transport_config: dict[str, Any] | None = None
    launch_spec: dict[str, Any] | None = None
    routing_policy: dict[str, Any] | None = None
    declared_capabilities: dict[str, bool] | None = None

    def to_patch(self) -> dict[str, Any]:
        patch = self.model_dump(exclude_unset=True, exclude_none=True)
        if "name" in patch and isinstance(patch["name"], str):
            patch["name"] = patch["name"].strip()
        return patch


class VLLMDefaultRouteRequest(BaseModel):
    instance_id: str | None = Field(default=None, description="Set or clear the default managed instance")


class VLLMInstanceRecordResponse(BaseModel):
    instance_id: str
    name: str
    execution_mode: str
    transport_config: dict[str, Any] = Field(default_factory=dict)
    launch_spec: dict[str, Any] = Field(default_factory=dict)
    routing_policy: dict[str, Any] = Field(default_factory=dict)
    declared_capabilities: dict[str, Any] = Field(default_factory=dict)
    desired_state: str
    observed_state: str
    created_at: str
    updated_at: str
    probed_capabilities: dict[str, Any] = Field(default_factory=dict)
    effective_capabilities: dict[str, Any] = Field(default_factory=dict)
    last_known_base_url: str | None = None
    last_error: str | None = None
    executor_handle: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class VLLMInstanceEnvelope(BaseModel):
    backend: str = "vllm"
    instance: VLLMInstanceRecordResponse


class VLLMInstanceListResponse(BaseModel):
    backend: str = "vllm"
    default_instance_id: str | None = None
    instances: list[VLLMInstanceRecordResponse] = Field(default_factory=list)


class VLLMDefaultRouteResponse(BaseModel):
    backend: str = "vllm"
    default_instance_id: str | None = None


class VLLMDeleteResponse(BaseModel):
    backend: str = "vllm"
    deleted: bool
    instance_id: str


class VLLMInstanceJobResponse(BaseModel):
    backend: str = "vllm"
    instance_id: str
    requested_action: str
    job_id: int
    job_uuid: str | None = None
    status: str
