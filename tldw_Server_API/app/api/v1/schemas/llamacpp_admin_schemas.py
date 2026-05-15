from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LlamaCppSavedConfig(BaseModel):
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
    saved_config: LlamaCppSavedConfig
    active_config: LlamaCppActiveConfig
    restart_required: bool
    restart_reasons: list[str] = Field(default_factory=list)
    env_overrides: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppConfigUpdateRequest(BaseModel):
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


class LlamaCppValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binary_path: str = Field(..., min_length=1)
    timeout_seconds: float = Field(default=3.0, gt=0, le=15)


class LlamaCppValidationResponse(BaseModel):
    valid: bool
    exists: bool
    executable: bool
    resolved_path: str | None = None
    version_output: str | None = None
    help_output: str | None = None
    warnings: list[str] = Field(default_factory=list)
