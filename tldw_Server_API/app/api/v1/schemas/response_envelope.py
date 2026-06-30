"""Shared API response-envelope schemas.

These models define the canonical envelope shape for new or migrated API
responses. Existing endpoint payloads are intentionally unchanged until a route
opts into the envelope contract.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class ResponseEnvelope(BaseModel, Generic[T]):
    """Canonical response envelope for successful and failed API responses."""

    success: bool
    data: T | None = None
    error: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] | None = None


__all__ = ["ResponseEnvelope"]
