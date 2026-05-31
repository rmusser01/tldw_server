"""Shared Pydantic schemas for the VN platform API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VNErrorDetail(BaseModel):
    """Stable error detail shape returned by VN platform endpoints."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
