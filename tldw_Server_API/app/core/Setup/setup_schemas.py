"""Core-owned setup provider models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SetupProviderType(str, Enum):
    """Provider setup categories supported by the first-run wizard."""

    HOSTED_API_KEY = "hosted_api_key"
    LOCAL_ENDPOINT = "local_endpoint"


class SetupProviderCatalogEntry(BaseModel):
    """One provider option exposed to first-run setup."""

    provider_key: str
    label: str
    provider_type: SetupProviderType
    config_section: str
    api_key_field: str | None = None
    base_url_field: str | None = None
    model_field: str | None = None
    default_base_url: str | None = None
    supports_preflight: bool = False
    recommended_for_first_chat: bool = False


class SetupProviderCatalogResponse(BaseModel):
    """Catalog of provider options for the first-run setup UI."""

    providers: list[SetupProviderCatalogEntry] = Field(default_factory=list)


class SetupProviderValidationResponse(BaseModel):
    """Provider validation result safe to return to unauthenticated setup clients."""

    provider_key: str
    status: str
    failure_category: str | None = None
    message: str | None = None
    models: list[str] = Field(default_factory=list)
    validation_level: str | None = None
    can_gate_first_chat: bool = False
