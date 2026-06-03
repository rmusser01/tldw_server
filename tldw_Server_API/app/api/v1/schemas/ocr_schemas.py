"""Schemas for OCR discovery and management endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, RootModel


class OCRBackendDiscoveryEntry(BaseModel):
    """Document the lightweight discovery metadata returned for an OCR backend."""

    available: Optional[bool] = None
    mode: Optional[str] = None
    runtime_family: Optional[str] = None
    configured_family: Optional[str] = None
    configured_mode: Optional[str] = None
    model: Optional[str] = None
    configured: Optional[bool] = None
    supports_structured_output: Optional[bool] = None
    supports_json: Optional[bool] = None
    configured_flags: Optional[str] = None
    auto_eligible: Optional[bool] = None
    auto_high_quality_eligible: Optional[bool] = None
    url_configured: Optional[bool] = None
    managed_configured: Optional[bool] = None
    managed_running: Optional[bool] = None
    allow_managed_start: Optional[bool] = None
    cli_configured: Optional[bool] = None
    backend_concurrency_cap: Optional[int] = None
    healthcheck_url_configured: Optional[bool] = None
    prompt: Optional[str] = None
    prompt_preset: Optional[str] = None
    text_format: Optional[str] = None
    table_format: Optional[str] = None
    remote_mode: Optional[str] = None
    sglang_reachable: Optional[bool] = None
    vllm_reachable: Optional[bool] = None
    pdf_only: Optional[bool] = None
    document_level: Optional[bool] = None
    opt_in_only: Optional[bool] = None
    supports_per_page_metrics: Optional[bool] = None
    timeout_sec: Optional[int] = None
    max_concurrency: Optional[int] = None
    tmp_root: Optional[str] = None
    debug_save_raw: Optional[bool] = None
    model_id: Optional[str] = None
    base_size: Optional[int] = None
    image_size: Optional[int] = None
    crop_mode: Optional[bool | str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_impl: Optional[str] = None
    native: Optional[dict[str, Any]] = None
    llamacpp: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class OCRBackendsResponse(RootModel[dict[str, OCRBackendDiscoveryEntry]]):
    """Map backend names to their discovery metadata."""
