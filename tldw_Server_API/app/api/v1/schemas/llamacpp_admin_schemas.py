"""Schemas for the admin llama.cpp configuration and managed-server APIs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LlamaCppSavedConfig(BaseModel):
    """Saved [LlamaCpp] settings as persisted in config.txt."""

    enabled: bool = False
    executable_path: str | None = None
    models_dir: str | None = None
    default_host: str | None = None
    default_port: int | None = None
    default_threads: int | None = None
    default_n_gpu_layers: int | None = None
    default_ctx_size: int | None = None
    allow_unvalidated_args: bool | None = None
    allow_cli_secrets: bool | None = None
    port_autoselect: bool | None = None
    port_probe_max: int | None = None
    allowed_paths: list[str] = Field(default_factory=list)
    registered_model_paths: list[str] = Field(default_factory=list)
    log_output_file: str | None = None


class LlamaCppActiveConfig(BaseModel):
    """Runtime llama.cpp handler state observed from the active API process."""

    handler_configured: bool
    enabled: bool | None = None
    executable_path: str | None = None
    models_dir: str | None = None
    default_host: str | None = None
    default_port: int | None = None
    active_model: str | None = None
    active_host: str | None = None
    active_port: int | None = None
    active_pid: int | None = None


class LlamaCppConfigResponse(BaseModel):
    """Combined saved/runtime config state plus restart and warning signals."""

    saved_config: LlamaCppSavedConfig
    active_config: LlamaCppActiveConfig
    restart_required: bool
    restart_reasons: list[str] = Field(default_factory=list)
    env_overrides: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppConfigUpdateRequest(BaseModel):
    """Partial llama.cpp admin config update payload."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    executable_path: str | None = None
    models_dir: str | None = None
    default_host: str | None = None
    default_port: int | None = Field(default=None, ge=1, le=65535)
    default_threads: int | None = Field(default=None, ge=1)
    default_n_gpu_layers: int | None = None
    default_ctx_size: int | None = Field(default=None, ge=1)
    allow_unvalidated_args: bool | None = None
    allow_cli_secrets: bool | None = None
    port_autoselect: bool | None = None
    port_probe_max: int | None = Field(default=None, ge=0)
    allowed_paths: list[str] | None = None
    log_output_file: str | None = None

    @model_validator(mode="after")
    def reject_boolean_null_clears(self) -> "LlamaCppConfigUpdateRequest":
        boolean_fields = {
            "enabled",
            "allow_unvalidated_args",
            "allow_cli_secrets",
            "port_autoselect",
        }
        for field in boolean_fields & self.model_fields_set:
            if getattr(self, field) is None:
                raise ValueError(f"{field} must be true or false; null clears are not supported for boolean fields.")
        return self


class LlamaCppValidationRequest(BaseModel):
    """Request to stat or optionally probe a llama.cpp server binary."""

    model_config = ConfigDict(extra="forbid")

    binary_path: str = Field(..., min_length=1)
    timeout_seconds: float = Field(default=3.0, gt=0, le=15)
    run_probe: bool = False


class LlamaCppValidationResponse(BaseModel):
    """Binary validation result returned to the Admin UI."""

    valid: bool
    exists: bool
    executable: bool
    resolved_path: str | None = None
    version_output: str | None = None
    help_output: str | None = None
    warnings: list[str] = Field(default_factory=list)


class LlamaCppModelMetadata(BaseModel):
    """Best-effort metadata parsed from a GGUF model filename."""

    quantization: str | None = None
    parameter_hint: str | None = None
    context_hint: int | None = None


class LlamaCppInventoryItem(BaseModel):
    """A single model inventory entry from models_dir or registered paths."""

    model_id: str
    display_name: str
    basename: str
    source: str
    path: str
    size_bytes: int | None = None
    modified_at: str | None = None
    metadata: LlamaCppModelMetadata = Field(default_factory=LlamaCppModelMetadata)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppInventoryResponse(BaseModel):
    """Bounded llama.cpp GGUF model inventory response."""

    models: list[LlamaCppInventoryItem]
    warnings: list[str] = Field(default_factory=list)
    scan_limited: bool = False


class LlamaCppRegisterModelPathRequest(BaseModel):
    """Request to register an allowlisted local GGUF path for inventory."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1)


class LlamaCppStartByModelRequest(BaseModel):
    """Request to launch the managed llama.cpp server by inventory model ID."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., min_length=1)
    server_args: dict[str, object] = Field(default_factory=dict)


class LlamaCppStartByModelResponse(BaseModel):
    """Response returned after launching a managed llama.cpp model by ID."""

    model_config = ConfigDict(extra="allow")

    status: str
    backend: str
    model_id: str
    model: str | None = None
    path: str | None = None


class LlamaCppUseInChatResponse(BaseModel):
    """Result of wiring the active managed server into chat provider config."""

    provider: str
    endpoint: str
    updated: bool
    effective: bool
    warnings: list[str] = Field(default_factory=list)


class LlamaCppLogTailResponse(BaseModel):
    """Redacted bounded tail of the active managed llama.cpp log file."""

    lines: list[str] = Field(default_factory=list)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class LlamaCppGpuSnapshot(BaseModel):
    """Best-effort GPU memory snapshot for hardware readiness guidance."""

    index: int
    name: str | None = None
    memory_total_bytes: int | None = None
    memory_free_bytes: int | None = None
    memory_used_bytes: int | None = None


class LlamaCppHardwareSnapshotResponse(BaseModel):
    """CPU/RAM/GPU snapshot used by the llama.cpp Admin readiness panel."""

    ram_total_bytes: int | None = None
    ram_available_bytes: int | None = None
    cpu_count: int | None = None
    gpus: list[LlamaCppGpuSnapshot] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
