"""Core-owned llama.cpp admin models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LlamaCppModelMetadata(BaseModel):
    """Best-effort metadata parsed from a GGUF model filename."""

    quantization: str | None = None
    parameter_hint: str | None = None
    context_hint: int | None = None


class LlamaCppAssetMetadata(BaseModel):
    """Best-effort metadata parsed from a local llama.cpp asset."""

    quantization: str | None = None
    parameter_hint: str | None = None
    context_hint: int | None = None
    family_hint: str | None = None


class LlamaCppAsset(BaseModel):
    """A local llama.cpp file or folder discovered by asset inventory."""

    asset_id: str
    kind: Literal["gguf", "mmproj", "folder", "unknown"]
    identity_basis: Literal["resolved_path", "manual"]
    path: str
    resolved_path: str | None = None
    display_name: str
    source: Literal["models_dir", "registered_path", "imported_folder"]
    size_bytes: int | None = None
    modified_at: str | None = None
    metadata: LlamaCppAssetMetadata = Field(default_factory=LlamaCppAssetMetadata)
    capabilities: list[str] = Field(default_factory=list)
    mmproj_asset_ids: list[str] = Field(default_factory=list)
    base_model_asset_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LlamaCppAssetsResponse(BaseModel):
    """Bounded local llama.cpp asset inventory response."""

    assets: list[LlamaCppAsset]
    warnings: list[str] = Field(default_factory=list)
    scan_limited: bool = False


class LlamaCppAssetImportPreviewResponse(BaseModel):
    """Non-mutating preview of an allowlisted local asset folder import."""

    folder: LlamaCppAsset
    assets: list[LlamaCppAsset] = Field(default_factory=list)
    asset_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    scan_limited: bool = False
    will_persist: bool = False


class LlamaCppAssetDownloadRequest(BaseModel):
    """Request to queue a safe llama.cpp asset download job."""

    model_config = ConfigDict(extra="forbid")

    url: str
    destination_dir: str | None = None
    filename: str | None = None
    expected_sha256: str | None = None
    expected_size_bytes: int | None = Field(default=None, ge=1)
    source_label: str | None = None
    overwrite: bool = False
    register_asset: bool = True


class LlamaCppAcquisitionJobResponse(BaseModel):
    """Normalized llama.cpp acquisition job state for admin clients."""

    job_id: str
    status: str
    operation: Literal["download"]
    queue: str
    source_label: str | None = None
    destination_path: str | None = None
    asset_id: str | None = None
    progress: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None


class LlamaCppAcquisitionJobListResponse(BaseModel):
    """Recent llama.cpp acquisition jobs."""

    jobs: list[LlamaCppAcquisitionJobResponse] = Field(default_factory=list)


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
